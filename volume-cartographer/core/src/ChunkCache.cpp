#include "vc/core/util/ChunkCache.hpp"
#include <utils/zarr.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

// Helper to read and decompress a chunk from Zarr.
// Handles uint16->uint8 conversion when the dataset dtype is uint16
// but ChunkCache<uint8_t> is requested.
template<typename T>
static std::shared_ptr<ChunkArray3D<T>> readChunkFromDataset(
    vc::Zarr& ds, size_t iz, size_t iy, size_t ix)
{
    const auto& chunkShape = ds.chunks();
    const std::size_t chunkSize = ds.chunkSize();

    auto out = std::make_shared<ChunkArray3D<T>>(chunkShape[0], chunkShape[1], chunkShape[2]);

    if (ds.dtype() == vc::DType::UInt8) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            if (!ds.readChunk(iz, iy, ix, out->data()))
                return nullptr;
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (ds.dtype() == vc::DType::UInt16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            if (!ds.readChunk(iz, iy, ix, out->data()))
                return nullptr;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            // Read as uint16, then convert to uint8 by dividing by 257
            std::vector<uint16_t> tmp(chunkSize);
            if (!ds.readChunk(iz, iy, ix, tmp.data()))
                return nullptr;

            uint8_t* p8 = out->data();
            const uint16_t* p16 = tmp.data();
            for (size_t i = 0; i < chunkSize; i++)
                p8[i] = static_cast<uint8_t>(p16[i] / 257);
        }
    }

    return out;
}

template<typename T>
ChunkCache<T>::ChunkCache(size_t maxBytes) : _maxBytes(maxBytes)
{
}

template<typename T>
ChunkCache<T>::~ChunkCache()
{
    clear();
}

template<typename T>
void ChunkCache<T>::setMaxBytes(size_t maxBytes)
{
    _maxBytes = maxBytes;
}

template<typename T>
auto ChunkCache<T>::get(vc::Zarr* ds, int iz, int iy, int ix) -> ChunkPtr
{
    ChunkKey key{ds, iz, iy, ix};

    // Fast path: shared lock read
    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
            _hits.fetch_add(1, std::memory_order_relaxed);
            return it->second.chunk;
        }
    }

    _misses.fetch_add(1, std::memory_order_relaxed);

    // Slow path: load from disk (per-key lock to avoid duplicate reads)
    std::lock_guard<std::mutex> diskLock(_lockPool[lockIndex(key)]);

    // Re-check after acquiring disk lock
    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        auto it = _map.find(key);
        if (it != _map.end()) {
            it->second.lastAccess = _generation.fetch_add(1, std::memory_order_relaxed);
            return it->second.chunk;
        }
    }

    ChunkPtr newChunk = loadChunk(ds, iz, iy, ix);
    if (!newChunk) return nullptr;

    size_t chunkBytes = newChunk->size() * sizeof(T);
    _bytesRead.fetch_add(chunkBytes, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> evictLock(_evictionMutex);

        // Track re-reads: if we've loaded this chunk before, it was evicted
        // and is now being re-read -- wasted I/O
        auto [it, firstTime] = _everLoaded.insert(key);
        if (!firstTime) {
            _reReads.fetch_add(1, std::memory_order_relaxed);
            _reReadBytes.fetch_add(chunkBytes, std::memory_order_relaxed);
        }

        if (_maxBytes > 0 && _storedBytes.load(std::memory_order_relaxed) + chunkBytes > _maxBytes) {
            evictIfNeeded();
        }
    }

    {
        std::unique_lock<std::shared_mutex> wlock(_mapMutex);
        auto [it, inserted] = _map.try_emplace(key, CacheEntry{newChunk, chunkBytes, _generation.fetch_add(1, std::memory_order_relaxed)});
        if (inserted) {
            _storedBytes.fetch_add(chunkBytes, std::memory_order_relaxed);
            _cachedCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Another thread inserted while we were loading -- use theirs
            return it->second.chunk;
        }
    }

    return newChunk;
}

template<typename T>
auto ChunkCache<T>::getIfCached(vc::Zarr* ds, int iz, int iy, int ix) const -> ChunkPtr
{
    ChunkKey key{ds, iz, iy, ix};
    std::shared_lock<std::shared_mutex> rlock(_mapMutex);
    auto it = _map.find(key);
    if (it != _map.end()) {
        _hits.fetch_add(1, std::memory_order_relaxed);
        return it->second.chunk;
    }
    return nullptr;
}

template<typename T>
void ChunkCache<T>::prefetch(vc::Zarr* ds, int minIz, int minIy, int minIx, int maxIz, int maxIy, int maxIx)
{
    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int ix = minIx; ix <= maxIx; ix++) {
        for (int iy = minIy; iy <= maxIy; iy++) {
            for (int iz = minIz; iz <= maxIz; iz++) {
                get(ds, iz, iy, ix);
            }
        }
    }
}

template<typename T>
void ChunkCache<T>::clear()
{
    std::unique_lock<std::shared_mutex> wlock(_mapMutex);
    _map.clear();
    _storedBytes.store(0, std::memory_order_relaxed);
    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);
}

template<typename T>
void ChunkCache<T>::flush()
{
    clear();
}

template<typename T>
void ChunkCache<T>::evictIfNeeded()
{
    // Called with _evictionMutex held
    if (_maxBytes == 0) return;

    size_t currentBytes = _storedBytes.load(std::memory_order_relaxed);
    if (currentBytes <= _maxBytes) return;

    struct EvictCandidate {
        ChunkKey key;
        size_t bytes;
        uint64_t lastAccess;
    };

    std::vector<EvictCandidate> candidates;

    {
        std::shared_lock<std::shared_mutex> rlock(_mapMutex);
        candidates.reserve(_map.size());
        for (auto& [key, entry] : _map) {
            candidates.push_back({key, entry.bytes, entry.lastAccess});
        }
    }

    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const EvictCandidate& a, const EvictCandidate& b) {
                  return a.lastAccess < b.lastAccess;
              });

    size_t target = _maxBytes * 15 / 16;
    size_t evictedBytes = 0;
    size_t evictedCount = 0;

    std::vector<ChunkKey> toRemove;
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
        _cachedCount.fetch_sub(evictedCount, std::memory_order_relaxed);
        _evictions.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

template<typename T>
auto ChunkCache<T>::stats() const -> Stats
{
    return {
        _hits.load(std::memory_order_relaxed),
        _misses.load(std::memory_order_relaxed),
        _evictions.load(std::memory_order_relaxed),
        _bytesRead.load(std::memory_order_relaxed),
        _reReads.load(std::memory_order_relaxed),
        _reReadBytes.load(std::memory_order_relaxed)
    };
}

template<typename T>
void ChunkCache<T>::resetStats()
{
    _hits.store(0, std::memory_order_relaxed);
    _misses.store(0, std::memory_order_relaxed);
    _evictions.store(0, std::memory_order_relaxed);
    _bytesRead.store(0, std::memory_order_relaxed);
    _reReads.store(0, std::memory_order_relaxed);
    _reReadBytes.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> evictLock(_evictionMutex);
        _everLoaded.clear();
    }
}

template<typename T>
auto ChunkCache<T>::loadChunk(vc::Zarr* ds, int iz, int iy, int ix) -> ChunkPtr
{
    if (!ds) return nullptr;
    try {
        return readChunkFromDataset<T>(
            *ds,
            static_cast<size_t>(iz),
            static_cast<size_t>(iy),
            static_cast<size_t>(ix));
    } catch (const std::exception&) {
        return nullptr;
    }
}

// Explicit template instantiations
template class ChunkCache<uint8_t>;
template class ChunkCache<uint16_t>;
