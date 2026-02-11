#include "vc/core/util/SliceCache.hpp"

#include <algorithm>
#include <vector>

template<typename T>
SliceCache<T>::SliceCache(size_t maxBytes) : _maxBytes(maxBytes)
{
}

template<typename T>
SliceCache<T>::~SliceCache()
{
    invalidateAll();
}

template<typename T>
auto SliceCache<T>::get(const Key& key) const -> TileData
{
    std::shared_lock<std::shared_mutex> rlock(_mapMutex);
    auto it = _map.find(key);
    if (it != _map.end()) {
        it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
        _hits.fetch_add(1, std::memory_order_relaxed);
        return it->second.data;
    }
    _misses.fetch_add(1, std::memory_order_relaxed);
    return {};
}

template<typename T>
auto SliceCache<T>::getBest(const Key& key, int maxLevel) const -> Result
{
    std::shared_lock<std::shared_mutex> rlock(_mapMutex);

    // Try exact level first
    auto it = _map.find(key);
    if (it != _map.end()) {
        it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
        _hits.fetch_add(1, std::memory_order_relaxed);
        return {it->second.data, key.level};
    }

    // Progressive fallback: try coarser levels
    Key probe = key;
    for (int lvl = key.level + 1; lvl <= maxLevel; ++lvl) {
        probe.level = lvl;
        it = _map.find(probe);
        if (it != _map.end()) {
            it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
            _fallbacks.fetch_add(1, std::memory_order_relaxed);
            return {it->second.data, lvl};
        }
    }

    _misses.fetch_add(1, std::memory_order_relaxed);
    return {{}, -1};
}

template<typename T>
void SliceCache<T>::put(const Key& key, TileData data)
{
    if (data.empty()) return;

    size_t tileBytes = data.total() * data.elemSize();

    {
        std::lock_guard<std::mutex> evictLock(_evictionMutex);
        if (_maxBytes > 0 &&
            _storedBytes.load(std::memory_order_relaxed) + tileBytes > _maxBytes) {
            evictIfNeeded();
        }
    }

    {
        std::unique_lock<std::shared_mutex> wlock(_mapMutex);
        auto [it, inserted] = _map.try_emplace(
            key,
            CacheEntry{data, tileBytes,
                       _generation.fetch_add(1, std::memory_order_relaxed),
                       key.epoch});
        if (inserted) {
            _storedBytes.fetch_add(tileBytes, std::memory_order_relaxed);
            _storedCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Update existing entry
            size_t oldBytes = it->second.bytes;
            it->second.data = data;
            it->second.bytes = tileBytes;
            it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
            it->second.epoch = key.epoch;
            if (tileBytes != oldBytes) {
                // Adjust stored bytes (may underflow briefly with relaxed ordering,
                // but corrects itself — same pattern as ChunkCache)
                _storedBytes.fetch_sub(oldBytes, std::memory_order_relaxed);
                _storedBytes.fetch_add(tileBytes, std::memory_order_relaxed);
            }
        }
    }
}

template<typename T>
void SliceCache<T>::invalidateAll()
{
    std::unique_lock<std::shared_mutex> wlock(_mapMutex);
    _map.clear();
    _storedBytes.store(0, std::memory_order_relaxed);
    _storedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);
}

template<typename T>
void SliceCache<T>::evictBefore(uint64_t minEpoch)
{
    std::unique_lock<std::shared_mutex> wlock(_mapMutex);

    size_t evictedBytes = 0;
    size_t evictedCount = 0;

    for (auto it = _map.begin(); it != _map.end(); ) {
        if (it->second.epoch < minEpoch) {
            evictedBytes += it->second.bytes;
            evictedCount++;
            it = _map.erase(it);
        } else {
            ++it;
        }
    }

    if (evictedCount > 0) {
        _storedBytes.fetch_sub(evictedBytes, std::memory_order_relaxed);
        _storedCount.fetch_sub(evictedCount, std::memory_order_relaxed);
        _evictions.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

template<typename T>
void SliceCache<T>::setMaxBytes(size_t maxBytes)
{
    _maxBytes = maxBytes;
}

template<typename T>
auto SliceCache<T>::stats() const -> Stats
{
    return {
        _hits.load(std::memory_order_relaxed),
        _misses.load(std::memory_order_relaxed),
        _fallbacks.load(std::memory_order_relaxed),
        _evictions.load(std::memory_order_relaxed),
        _storedBytes.load(std::memory_order_relaxed),
        _storedCount.load(std::memory_order_relaxed)
    };
}

template<typename T>
void SliceCache<T>::resetStats()
{
    _hits.store(0, std::memory_order_relaxed);
    _misses.store(0, std::memory_order_relaxed);
    _fallbacks.store(0, std::memory_order_relaxed);
    _evictions.store(0, std::memory_order_relaxed);
}

template<typename T>
void SliceCache<T>::evictIfNeeded()
{
    // Called with _evictionMutex held
    if (_maxBytes == 0) return;

    size_t currentBytes = _storedBytes.load(std::memory_order_relaxed);
    if (currentBytes <= _maxBytes) return;

    struct EvictCandidate {
        Key key;
        size_t bytes;
        uint64_t epoch;
        uint64_t lastAccess;
    };

    std::vector<EvictCandidate> candidates;

    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        candidates.reserve(_map.size());
        for (auto& [key, entry] : _map) {
            candidates.push_back({key, entry.bytes, entry.epoch, entry.lastAccess});
        }
    }

    if (candidates.empty()) return;

    // Two-tier sort: old epochs first, then LRU within epoch
    std::sort(candidates.begin(), candidates.end(),
              [](const EvictCandidate& a, const EvictCandidate& b) {
                  if (a.epoch != b.epoch) return a.epoch < b.epoch;
                  return a.lastAccess < b.lastAccess;
              });

    size_t target = _maxBytes * 15 / 16;
    size_t evictedBytes = 0;
    size_t evictedCount = 0;

    std::vector<Key> toRemove;
    for (auto& c : candidates) {
        if (currentBytes - evictedBytes <= target) break;
        toRemove.push_back(c.key);
        evictedBytes += c.bytes;
        evictedCount++;
    }

    if (!toRemove.empty()) {
        std::unique_lock<std::shared_mutex> wlock(_mapMutex);
        for (auto& key : toRemove) {
            _map.erase(key);
        }
        _storedBytes.fetch_sub(evictedBytes, std::memory_order_relaxed);
        _storedCount.fetch_sub(evictedCount, std::memory_order_relaxed);
        _evictions.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

// Explicit template instantiations
template class SliceCache<uint8_t>;
template class SliceCache<uint16_t>;
