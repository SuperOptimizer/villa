#pragma once

// A small open-addressing hash map (linear probing, power-of-two capacity,
// backward-shift deletion -> no tombstones) tailored for the chunk cache.
//
// Storage is SPLIT: a dense array of keys (probed) parallel to an array of
// values. The find loop scans only the key array, so more keys fit per cache
// line and the probe touches far fewer cache lines than a node-based
// std::unordered_map (pointer-per-node) or an inline {key,value} slot array
// (value bloats the stride). Values can be move-only (e.g. hold a unique_ptr);
// they are relocated by move on grow / backward-shift delete.
//
// API surface mirrors the std::unordered_map operations the cache uses:
// find / emplace / erase / iterate (structured bindings) / size / clear /
// reserve, with it->first, it->second.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace vc::render {

template <typename Key, typename Value, typename Hash, typename KeyTraits>
class OpenAddrMap {
public:
    OpenAddrMap() { allocate(kInitialCapacity); }

    template <bool Const>
    class Iter {
        using Map = std::conditional_t<Const, const OpenAddrMap, OpenAddrMap>;
        Map* m_ = nullptr;
        std::size_t i_ = 0;

        void skipEmpty()
        {
            while (i_ < m_->cap_ && m_->keys_[i_] == KeyTraits::empty())
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
        const std::size_t idx = probe(key);
        return keys_[idx] == key ? iterator(this, idx) : end();
    }
    const_iterator find(const Key& key) const
    {
        const std::size_t idx = probe(key);
        return keys_[idx] == key ? const_iterator(this, idx) : end();
    }

    template <typename... Args>
    std::pair<iterator, bool> emplace(const Key& key, Args&&... args)
    {
        if ((size_ + 1) * 4 >= cap_ * 3)   // load factor > 0.75 -> grow
            grow();
        std::size_t i = Hash{}(key) & mask_;
        for (;;) {
            if (keys_[i] == KeyTraits::empty()) {
                keys_[i] = key;
                values_[i] = Value(std::forward<Args>(args)...);
                ++size_;
                return {iterator(this, i), true};
            }
            if (keys_[i] == key)
                return {iterator(this, i), false};
            i = (i + 1) & mask_;
        }
    }

    iterator erase(iterator it)
    {
        // Empty the slot, then RE-INSERT every following element in the same
        // cluster (up to the next empty slot). This is the provably-correct
        // deletion for linear probing: any element whose probe chain passed
        // through the hole is reinserted at its proper position. Simpler and
        // safer than hand-rolled backward-shift (a subtle wrap-around bug there
        // mis-associated keys with values -> chunks read with the wrong status).
        const std::size_t start = it.index();
        keys_[start] = KeyTraits::empty();
        values_[start] = Value{};
        --size_;

        std::size_t i = (start + 1) & mask_;
        while (!(keys_[i] == KeyTraits::empty())) {
            Key k = keys_[i];
            Value v = std::move(values_[i]);
            keys_[i] = KeyTraits::empty();
            // reinsert (size_ unchanged: it's already counted)
            std::size_t j = Hash{}(k) & mask_;
            while (!(keys_[j] == KeyTraits::empty()))
                j = (j + 1) & mask_;
            keys_[j] = k;
            values_[j] = std::move(v);
            i = (i + 1) & mask_;
        }
        return iterator(this, start);
    }

    void clear()
    {
        for (std::size_t i = 0; i < cap_; ++i) {
            if (!(keys_[i] == KeyTraits::empty())) {
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

    void allocate(std::size_t cap)
    {
        cap_ = cap;
        mask_ = cap - 1;
        size_ = 0;
        keys_.assign(cap, KeyTraits::empty());
        values_.clear();
        values_.resize(cap);
    }

    // Probe to the slot holding `key`, or the first empty slot in its chain.
    std::size_t probe(const Key& key) const
    {
        std::size_t i = Hash{}(key) & mask_;
        for (;;) {
            if (keys_[i] == key || keys_[i] == KeyTraits::empty())
                return i;
            i = (i + 1) & mask_;
        }
    }

    void grow() { rehash(cap_ << 1); }

    void rehash(std::size_t newCap)
    {
        std::vector<Key> oldKeys = std::move(keys_);
        std::vector<Value> oldValues = std::move(values_);
        const std::size_t oldCap = cap_;
        allocate(newCap);
        for (std::size_t j = 0; j < oldCap; ++j) {
            if (!(oldKeys[j] == KeyTraits::empty())) {
                std::size_t i = Hash{}(oldKeys[j]) & mask_;
                while (!(keys_[i] == KeyTraits::empty()))
                    i = (i + 1) & mask_;
                keys_[i] = oldKeys[j];
                values_[i] = std::move(oldValues[j]);
                ++size_;
            }
        }
    }

    std::vector<Key> keys_;       // dense, probed -- the hot array
    std::vector<Value> values_;   // parallel to keys_
    std::size_t cap_ = 0;
    std::size_t mask_ = 0;
    std::size_t size_ = 0;
};

}  // namespace vc::render
