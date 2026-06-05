#pragma once

// A small open-addressing hash map (linear probing, power-of-two capacity,
// backward-shift deletion -> no tombstones) tailored for the chunk cache's
// entries_ table. std::unordered_map is node-based: every find chases a pointer
// to a separately heap-allocated node, which cache-misses. This map stores the
// key+value inline in one contiguous array, so a lookup is a contiguous probe --
// far friendlier to the cache. It implements only the operations ChunkCache uses
// (find / emplace / erase / iterate / size / clear) with a std::unordered_map-
// compatible surface (it->first, it->second, structured-binding range-for,
// emplace returning {iterator,bool}).
//
// Value must be movable (entries are relocated on grow / on backward-shift
// delete). Key must be trivially comparable and have a reserved "empty" value
// that never appears as a real key (provided by KeyTraits::empty()).

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vc::render {

template <typename Key, typename Value, typename Hash, typename KeyTraits>
class OpenAddrMap {
    struct Slot {
        Key key;
        Value value;
    };

public:
    OpenAddrMap()
    {
        slots_.resize(kInitialCapacity);
        for (auto& s : slots_)
            s.key = KeyTraits::empty();
        mask_ = kInitialCapacity - 1;
    }

    // Forward iterator over occupied slots. Dereferences to a reference-pair-like
    // proxy so `it->first` / `it->second` and structured bindings work.
    template <bool Const>
    class Iter {
        using SlotPtr = std::conditional_t<Const, const Slot*, Slot*>;
        SlotPtr cur_ = nullptr;
        SlotPtr end_ = nullptr;

        void skipEmpty()
        {
            while (cur_ != end_ && cur_->key == KeyTraits::empty())
                ++cur_;
        }

    public:
        // A view onto the current slot: .first (const key), .second (value).
        struct Ref {
            SlotPtr s;
            const Key& first() const { return s->key; }
            auto& second() const { return s->value; }
        };
        struct Arrow {
            SlotPtr s;
            const Key* firstPtr() const { return &s->key; }
        };

        Iter() = default;
        Iter(SlotPtr cur, SlotPtr end) : cur_(cur), end_(end) { skipEmpty(); }

        // Proxy with .first/.second members for `it->first`, `it->second`.
        struct Proxy {
            SlotPtr s;
            const Key& first;
            std::conditional_t<Const, const Value&, Value&> second;
            Proxy(SlotPtr p) : s(p), first(p->key), second(p->value) {}
            const Proxy* operator->() const { return this; }
        };
        Proxy operator->() const { return Proxy(cur_); }
        // For range-for structured bindings `auto& [k, v]`.
        std::pair<const Key&, std::conditional_t<Const, const Value&, Value&>>
        operator*() const { return {cur_->key, cur_->value}; }

        Iter& operator++() { ++cur_; skipEmpty(); return *this; }
        bool operator==(const Iter& o) const { return cur_ == o.cur_; }
        bool operator!=(const Iter& o) const { return cur_ != o.cur_; }

        SlotPtr slot() const { return cur_; }
        friend class OpenAddrMap;
    };

    using iterator = Iter<false>;
    using const_iterator = Iter<true>;

    iterator begin() { return iterator(slots_.data(), slots_.data() + slots_.size()); }
    iterator end() { return iterator(slots_.data() + slots_.size(), slots_.data() + slots_.size()); }
    const_iterator begin() const { return const_iterator(slots_.data(), slots_.data() + slots_.size()); }
    const_iterator end() const { return const_iterator(slots_.data() + slots_.size(), slots_.data() + slots_.size()); }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    iterator find(const Key& key)
    {
        std::size_t i = Hash{}(key) & mask_;
        for (;;) {
            Slot& s = slots_[i];
            if (s.key == key)
                return iterator(&s, slots_.data() + slots_.size());
            if (s.key == KeyTraits::empty())
                return end();
            i = (i + 1) & mask_;
        }
    }
    const_iterator find(const Key& key) const
    {
        std::size_t i = Hash{}(key) & mask_;
        for (;;) {
            const Slot& s = slots_[i];
            if (s.key == key)
                return const_iterator(&s, slots_.data() + slots_.size());
            if (s.key == KeyTraits::empty())
                return end();
            i = (i + 1) & mask_;
        }
    }

    // emplace(key, value...) -> {iterator, inserted}. If the key already exists,
    // returns the existing slot and inserted=false (value args are not used).
    template <typename... Args>
    std::pair<iterator, bool> emplace(const Key& key, Args&&... args)
    {
        if ((size_ + 1) * 4 >= slots_.size() * 3)   // load factor > 0.75 -> grow
            grow();
        std::size_t i = Hash{}(key) & mask_;
        for (;;) {
            Slot& s = slots_[i];
            if (s.key == KeyTraits::empty()) {
                s.key = key;
                s.value = Value(std::forward<Args>(args)...);
                ++size_;
                return {iterator(&s, slots_.data() + slots_.size()), true};
            }
            if (s.key == key)
                return {iterator(&s, slots_.data() + slots_.size()), false};
            i = (i + 1) & mask_;
        }
    }

    // Erase by iterator using backward-shift deletion (keeps the probe sequence
    // contiguous without tombstones). Returns an iterator to the next element.
    iterator erase(iterator it)
    {
        Slot* base = slots_.data();
        std::size_t hole = static_cast<std::size_t>(it.slot() - base);
        std::size_t next = (hole + 1) & mask_;
        // Backward-shift: pull in following entries that probed past `hole`.
        while (slots_[next].key != KeyTraits::empty()) {
            const std::size_t ideal = Hash{}(slots_[next].key) & mask_;
            // Is `next` allowed to move back into `hole`? It can if `hole` lies in
            // the cyclic range [ideal, next].
            const bool movable =
                (hole <= next) ? (ideal <= hole || ideal > next)
                               : (ideal <= hole && ideal > next);
            if (movable) {
                slots_[hole] = std::move(slots_[next]);
                hole = next;
            }
            next = (next + 1) & mask_;
        }
        slots_[hole].key = KeyTraits::empty();
        slots_[hole].value = Value{};
        --size_;
        // Next occupied slot at/after the original hole position.
        return iterator(base + hole, base + slots_.size());
    }

    void clear()
    {
        for (auto& s : slots_) {
            if (!(s.key == KeyTraits::empty())) {
                s.key = KeyTraits::empty();
                s.value = Value{};
            }
        }
        size_ = 0;
    }

    void reserve(std::size_t n)
    {
        std::size_t want = kInitialCapacity;
        while (want * 3 < (n + 1) * 4)
            want <<= 1;
        if (want > slots_.size())
            rehash(want);
    }

private:
    static constexpr std::size_t kInitialCapacity = 64;

    void grow() { rehash(slots_.size() << 1); }

    void rehash(std::size_t newCap)
    {
        std::vector<Slot> old = std::move(slots_);
        slots_.clear();
        slots_.resize(newCap);
        for (auto& s : slots_)
            s.key = KeyTraits::empty();
        mask_ = newCap - 1;
        size_ = 0;
        for (auto& s : old) {
            if (!(s.key == KeyTraits::empty())) {
                std::size_t i = Hash{}(s.key) & mask_;
                while (!(slots_[i].key == KeyTraits::empty()))
                    i = (i + 1) & mask_;
                slots_[i].key = s.key;
                slots_[i].value = std::move(s.value);
                ++size_;
            }
        }
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
};

}  // namespace vc::render
