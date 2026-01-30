#pragma once

#include "vc/core/zarr/Tensor3D.hpp"
#include "vc/core/types/IChunkSource.hpp"

#include <atomic>
#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

/**
 * @brief Thread-safe chunk cache with shared_ptr lifetime management
 *
 * @tparam T Data type of cached chunks (uint8_t or uint16_t)
 *
 * Chunks are stored as shared_ptr so eviction removes from the cache but
 * doesn't free memory until all readers are done. This prevents use-after-free
 * when one thread evicts a chunk another thread is reading.
 *
 * Uses a flat 3D array indexed by chunk coordinates (iz, iy, ix).
 * Dimensions follow ZYX ordering: dim 0 = Z, dim 1 = Y, dim 2 = X.
 */
template<typename T>
class ChunkCache
{
public:
    using ChunkPtr = std::shared_ptr<volcart::zarr::Tensor3D<T>>;

    explicit ChunkCache(size_t maxBytes = 0);
    ~ChunkCache();

    ChunkCache(const ChunkCache&) = delete;
    ChunkCache& operator=(const ChunkCache&) = delete;
    ChunkCache(ChunkCache&&) = delete;
    ChunkCache& operator=(ChunkCache&&) = delete;

    void init(IChunkSource* src);
    bool initialized() const { return _src != nullptr; }

    void setMaxBytes(size_t maxBytes);
    size_t cachedCount() const { return _cachedCount.load(std::memory_order_relaxed); }

    /**
     * @brief Get a chunk, loading from disk if needed.
     * Returns shared_ptr — caller holds the chunk alive even if evicted.
     */
    ChunkPtr get(int iz, int iy, int ix);

    /**
     * @brief Get raw pointer (for hot-path compatibility). Caller must ensure
     * chunk stays alive (e.g. by also holding a ChunkPtr from get()).
     */
    volcart::zarr::Tensor3D<T>* getRaw(int iz, int iy, int ix);

    /**
     * @brief Fast raw pointer lookup with no refcount overhead.
     * Returns nullptr on cache miss. Caller must ensure no concurrent eviction
     * of the returned chunk (safe during rendering where eviction only happens
     * inside get() calls).
     */
    const volcart::zarr::Tensor3D<T>* getRawFast(int iz, int iy, int ix) const;

    ChunkPtr getIfCached(int iz, int iy, int ix) const;

    void prefetch(int minIz, int minIy, int minIx, int maxIz, int maxIy, int maxIx);
    void clear();

    /** @brief Drop all cached chunks but keep the cache initialized for the same dataset */
    void flush();

    int chunkSizeZ() const { return _cz; }
    int chunkSizeY() const { return _cy; }
    int chunkSizeX() const { return _cx; }
    int datasetSizeZ() const { return _sz; }
    int datasetSizeY() const { return _sy; }
    int datasetSizeX() const { return _sx; }
    int chunksZ() const { return _chunksZ; }
    int chunksY() const { return _chunksY; }
    int chunksX() const { return _chunksX; }

    int chunkShiftZ() const { return _czShift; }
    int chunkShiftY() const { return _cyShift; }
    int chunkShiftX() const { return _cxShift; }
    int chunkMaskZ() const { return _czMask; }
    int chunkMaskY() const { return _cyMask; }
    int chunkMaskX() const { return _cxMask; }

    IChunkSource* source() const { return _src; }

private:
    IChunkSource* _src = nullptr;

    size_t _maxBytes = 0;
    size_t _maxChunks = 0;
    std::atomic<size_t> _cachedCount{0};
    std::atomic<uint64_t> _generation{0};

    int _cz = 0, _cy = 0, _cx = 0;
    int _sz = 0, _sy = 0, _sx = 0;
    int _chunksZ = 0, _chunksY = 0, _chunksX = 0;

    int _czShift = 0, _cyShift = 0, _cxShift = 0;
    int _czMask = 0, _cyMask = 0, _cxMask = 0;

    // Flat 3D array: _chunks[iz * _chunksY * _chunksX + iy * _chunksX + ix]
    // atomic shared_ptr for lock-free reads on the fast path
    size_t _totalSlots = 0;
    std::unique_ptr<std::atomic<ChunkPtr>[]> _chunks;
    std::unique_ptr<std::atomic<const volcart::zarr::Tensor3D<T>*>[]> _rawChunks;
    std::unique_ptr<std::atomic<uint64_t>[]> _lastAccess;

    std::mutex _evictionMutex;

    static constexpr int kLockPoolSize = 64;
    std::mutex _lockPool[kLockPoolSize];

    size_t idx(int iz, int iy, int ix) const {
        return static_cast<size_t>(iz) * _chunksY * _chunksX +
               static_cast<size_t>(iy) * _chunksX +
               static_cast<size_t>(ix);
    }

    ChunkPtr loadChunk(int iz, int iy, int ix);
    void evictIfNeeded();

    static int log2_pow2(int v) {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};

extern template class ChunkCache<uint8_t>;
extern template class ChunkCache<uint16_t>;
