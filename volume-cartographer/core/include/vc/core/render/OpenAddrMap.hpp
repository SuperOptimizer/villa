#pragma once

// A small open-addressing hash map (linear probing, power-of-two capacity,
// reinsert-cluster deletion -> no tombstones) tailored for the chunk cache.
//
// The PROBE array holds 8-byte PACKED keys (KeyTraits::pack), not the 16-byte
// full keys: 8 packs per cache line vs 4 full keys, and the slot match is a
// single 64-bit compare. The full key array is parallel (for iteration / exact
// rehash) but the hot find() touches only the pack array. Values can be move-only
// (they hold a shared_ptr); they are relocated by move on grow / delete.
//
// KeyTraits must provide: Key empty(); uint64_t pack(const Key&). pack must be a
// bijection on real keys and pack(empty()) must be a value no real key produces.
//
// API mirrors the std::unordered_map operations the cache uses: find / emplace /
// erase / iterate (structured bindings) / size / clear / reserve.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vc::render {

template <typename Key, typename Value, typename Hash, typename KeyTraits>
class OpenAddrMap {
    static std::uint64_t emptyPack() { return KeyTraits::pack(KeyTraits::empty()); }

public:
    OpenAddrMap() { allocate(kInitialCapacity); }

    template <bool Const>
    class Iter {
        using Map = std::conditional_t<Const, const OpenAddrMap, OpenAddrMap>;
        Map* m_ = nullptr;
        std::size_t i_ = 0;

        void skipEmpty()
        {
            const std::uint64_t e = emptyPack();
            while (i_ < m_->cap_ && m_->packed_[i_] == e)
                ++i_;
        }

    public:
        Iter() = default;
        Iter(Map* m, std::size_t i) : m_(m), i_(i) { skipEmpty(); }

        struct Proxy {
            const Key& first;
            std::conditional_t<Const, const Value&, Value&> second;
            const Proxy* operator->() const { return this; }
        };
        Proxy operator->() const { return Proxy{m_->keys_[i_], m_->values_[i_]}; }
        std::pair<const Key&, std::conditional_t<Const, const Value&, Value&>>
        operator*() const { return {m_->keys_[i_], m_->values_[i_]}; }

        Iter& operator++() { ++i_; skipEmpty(); return *this; }
        bool operator==(const Iter& o) const { return i_ == o.i_; }
        bool operator!=(const Iter& o) const { return i_ != o.i_; }

        std::size_t index() const { return i_; }
        friend class OpenAddrMap;
    };

    using iterator = Iter<false>;
    using const_iterator = Iter<true>;

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, cap_); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, cap_); }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    iterator find(const Key& key)
    {
        const std::size_t idx = probe(KeyTraits::pack(key));
        return idx != kNpos ? iterator(this, idx) : end();
    }
    const_iterator find(const Key& key) const
    {
        const std::size_t idx = probe(KeyTraits::pack(key));
        return idx != kNpos ? const_iterator(this, idx) : end();
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(const Key& key, Args&&... args)
    {
        if ((size_ + 1) * 4 >= cap_ * 3)   // load factor > 0.75 -> grow
            grow();
        const std::uint64_t p = KeyTraits::pack(key);
        const std::uint64_t e = emptyPack();
        std::size_t i = std::size_t(p) & mask_;
        for (;;) {
            if (packed_[i] == e) {
                packed_[i] = p;
                keys_[i] = key;
                values_[i] = Value(std::forward<Args>(args)...);
                ++size_;
                return {iterator(this, i), true};
            }
            if (packed_[i] == p)
                return {iterator(this, i), false};
            i = (i + 1) & mask_;
        }
    }

    iterator erase(iterator it)
    {
        // Empty the slot, then RE-INSERT every following element in the cluster
        // (up to the next empty slot) at its proper position. Provably correct for
        // linear probing; avoids the subtle backward-shift wrap bugs.
        const std::uint64_t e = emptyPack();
        const std::size_t start = it.index();
        packed_[start] = e;
        keys_[start] = KeyTraits::empty();
        values_[start] = Value{};
        --size_;

        std::size_t i = (start + 1) & mask_;
        while (packed_[i] != e) {
            const std::uint64_t p = packed_[i];
            Key k = keys_[i];
            Value v = std::move(values_[i]);
            packed_[i] = e;
            keys_[i] = KeyTraits::empty();
            std::size_t j = std::size_t(p) & mask_;
            while (packed_[j] != e)
                j = (j + 1) & mask_;
            packed_[j] = p;
            keys_[j] = k;
            values_[j] = std::move(v);
            i = (i + 1) & mask_;
        }
        return iterator(this, start);
    }

    void clear()
    {
        const std::uint64_t e = emptyPack();
        for (std::size_t i = 0; i < cap_; ++i) {
            if (packed_[i] != e) {
                packed_[i] = e;
                keys_[i] = KeyTraits::empty();
                values_[i] = Value{};
            }
        }
        size_ = 0;
    }

    void reserve(std::size_t n)
    {
        std::size_t want = kInitialCapacity;
        while (want * 3 < (n + 1) * 4)
            want <<= 1;
        if (want > cap_)
            rehash(want);
    }

private:
    static constexpr std::size_t kInitialCapacity = 64;
    static constexpr std::size_t kNpos = ~std::size_t(0);

    void allocate(std::size_t cap)
    {
        cap_ = cap;
        mask_ = cap - 1;
        size_ = 0;
        packed_.assign(cap, emptyPack());
        keys_.assign(cap, KeyTraits::empty());
        values_.clear();
        values_.resize(cap);
    }

    // Probe the PACKED array only (8B/slot, cache-dense). Returns the slot index
    // holding `p`, or kNpos if the chain ends at an empty slot without a match.
    std::size_t probe(std::uint64_t p) const
    {
        const std::uint64_t e = emptyPack();
        std::size_t i = std::size_t(p) & mask_;
        for (;;) {
            const std::uint64_t s = packed_[i];
            if (s == p)
                return i;
            if (s == e)
                return kNpos;
            i = (i + 1) & mask_;
        }
    }

    void grow() { rehash(cap_ << 1); }

    void rehash(std::size_t newCap)
    {
        std::vector<std::uint64_t> oldPacked = std::move(packed_);
        std::vector<Key> oldKeys = std::move(keys_);
        std::vector<Value> oldValues = std::move(values_);
        const std::size_t oldCap = cap_;
        const std::uint64_t e = emptyPack();
        allocate(newCap);
        for (std::size_t j = 0; j < oldCap; ++j) {
            if (oldPacked[j] != e) {
                std::size_t i = std::size_t(oldPacked[j]) & mask_;
                while (packed_[i] != e)
                    i = (i + 1) & mask_;
                packed_[i] = oldPacked[j];
                keys_[i] = oldKeys[j];
                values_[i] = std::move(oldValues[j]);
                ++size_;
            }
        }
    }

    std::vector<std::uint64_t> packed_;  // 8B packed keys -- the hot probe array
    std::vector<Key> keys_;              // full keys, parallel (iteration/exact)
    std::vector<Value> values_;          // parallel to packed_/keys_
    std::size_t cap_ = 0;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
};

}  // namespace vc::render
