#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace utils {

/// Statistics for an LruCache instance.
struct LruCacheStats {
    std::uint64_t hits{0};
    std::uint64_t misses{0};
    std::uint64_t evictions{0};
    std::uint64_t inserts{0};
};

/// Thread-safe LRU cache with configurable maximum entry count.
///
/// Uses a hash map for O(1) lookup and a doubly-linked list for O(1) LRU
/// ordering.  Concurrent access is protected by a shared_mutex for the map
/// (readers can overlap) and a pool of 64 per-key mutexes to serialize
/// competing writers on the same key without blocking unrelated keys.
///
/// @tparam Key    Key type (must be hashable via std::hash and equality-
///                comparable).
/// @tparam Value  Value type (must be copy- or move-constructible).
template <typename Key, typename Value>
class LruCache {
public:
    /// Construct a cache that holds at most @p max_entries items.
    /// A @p max_entries of 0 means unlimited (no eviction).
    explicit LruCache(std::size_t max_entries = 0)
        : max_entries_{max_entries} {}

    ~LruCache() = default;

    LruCache(const LruCache&) = delete;
    auto operator=(const LruCache&) -> LruCache& = delete;
    LruCache(LruCache&&) = delete;
    auto operator=(LruCache&&) -> LruCache& = delete;

    /// Insert or update a key-value pair.
    /// If the key already exists its value is replaced and it is promoted to
    /// most-recently-used.  If the cache is at capacity the least-recently-used
    /// entry is evicted first.
    auto put(const Key& key, Value value) -> void {
        std::lock_guard<std::mutex> per_key{lock_pool_[lock_index(key)]};

        {
            std::unique_lock wlock{map_mutex_};

            if (auto it = map_.find(key); it != map_.end()) {
                // Update existing entry — move to front of LRU list.
                it->second->value = std::move(value);
                order_.splice(order_.begin(), order_, it->second);
                return;
            }
        }

        // New key — evict if at capacity, then insert.
        inserts_.fetch_add(1, std::memory_order_relaxed);
        evict_if_needed();

        {
            std::unique_lock wlock{map_mutex_};

            // Double-check after eviction (another thread could have inserted).
            if (auto it = map_.find(key); it != map_.end()) {
                it->second->value = std::move(value);
                order_.splice(order_.begin(), order_, it->second);
                return;
            }

            order_.emplace_front(Entry{key, std::move(value)});
            map_.emplace(key, order_.begin());
        }
    }

    /// Retrieve a copy of the value associated with @p key.
    /// Returns @c std::nullopt on a cache miss.  On a hit the entry is promoted
    /// to most-recently-used.
    [[nodiscard]] auto get(const Key& key) -> std::optional<Value> {
        std::unique_lock wlock{map_mutex_};

        auto it = map_.find(key);
        if (it == map_.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }

        hits_.fetch_add(1, std::memory_order_relaxed);
        order_.splice(order_.begin(), order_, it->second);
        return it->second->value;
    }

    /// Check whether @p key is present without promoting it in the LRU order.
    [[nodiscard]] auto contains(const Key& key) const -> bool {
        std::shared_lock rlock{map_mutex_};
        return map_.contains(key);
    }

    /// Remove a specific key from the cache.
    /// Returns true if the key was found and removed, false otherwise.
    auto erase(const Key& key) -> bool {
        std::unique_lock wlock{map_mutex_};
        auto it = map_.find(key);
        if (it == map_.end()) {
            return false;
        }
        order_.erase(it->second);
        map_.erase(it);
        return true;
    }

    /// Remove all entries.
    auto clear() -> void {
        std::unique_lock wlock{map_mutex_};
        map_.clear();
        order_.clear();
    }

    /// Number of entries currently in the cache.
    [[nodiscard]] auto size() const -> std::size_t {
        std::shared_lock rlock{map_mutex_};
        return map_.size();
    }

    /// Returns true if the cache is empty.
    [[nodiscard]] auto empty() const -> bool {
        std::shared_lock rlock{map_mutex_};
        return map_.empty();
    }

    /// Maximum number of entries (0 = unlimited).
    [[nodiscard]] auto max_entries() const noexcept -> std::size_t {
        return max_entries_;
    }

    /// Change the capacity.  If the new capacity is smaller than the current
    /// size, excess entries are evicted (least-recently-used first).
    auto set_max_entries(std::size_t max_entries) -> void {
        max_entries_ = max_entries;
        shrink_to_fit();
    }

    /// Return a snapshot of the cache statistics.
    [[nodiscard]] auto stats() const noexcept -> LruCacheStats {
        return {
            hits_.load(std::memory_order_relaxed),
            misses_.load(std::memory_order_relaxed),
            evictions_.load(std::memory_order_relaxed),
            inserts_.load(std::memory_order_relaxed),
        };
    }

    /// Reset all statistics counters to zero.
    auto reset_stats() noexcept -> void {
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
        inserts_.store(0, std::memory_order_relaxed);
    }

private:
    struct Entry {
        Key key;
        Value value;
    };

    using ListIter = typename std::list<Entry>::iterator;

    std::size_t max_entries_;

    mutable std::shared_mutex map_mutex_;
    std::list<Entry> order_;  // front = most recent, back = least recent
    std::unordered_map<Key, ListIter> map_;

    // Per-key lock pool to serialize competing writers on the same key without
    // blocking unrelated keys (adapted from volume-cartographer's ChunkCache).
    static constexpr int kLockPoolSize = 64;
    std::mutex lock_pool_[kLockPoolSize];

    // Statistics (atomic for lock-free reads).
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> misses_{0};
    std::atomic<std::uint64_t> evictions_{0};
    std::atomic<std::uint64_t> inserts_{0};

    [[nodiscard]] auto lock_index(const Key& key) const -> std::size_t {
        return std::hash<Key>{}(key) % kLockPoolSize;
    }

    /// Evict the single least-recently-used entry if at capacity.
    /// Called with the per-key lock held but NOT the map lock.
    auto evict_if_needed() -> void {
        if (max_entries_ == 0) {
            return;
        }

        std::unique_lock wlock{map_mutex_};
        while (map_.size() >= max_entries_ && !order_.empty()) {
            auto& victim = order_.back();
            map_.erase(victim.key);
            order_.pop_back();
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// Shrink the cache to fit the current max_entries_ limit.
    auto shrink_to_fit() -> void {
        if (max_entries_ == 0) {
            return;
        }

        std::unique_lock wlock{map_mutex_};
        while (map_.size() > max_entries_ && !order_.empty()) {
            auto& victim = order_.back();
            map_.erase(victim.key);
            order_.pop_back();
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }
};

}  // namespace utils
