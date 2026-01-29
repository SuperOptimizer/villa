#pragma once

#include "vc/core/zarr/Tensor3D.hpp"
#include "vc/core/zarr/ZarrDataset.hpp"

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
 * Uses a flat 3D array indexed by chunk coordinates (ix, iy, iz).
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

    void init(volcart::zarr::ZarrDataset* ds);
    bool initialized() const { return _ds != nullptr; }

    void setMaxBytes(size_t maxBytes);
    size_t cachedCount() const { return _cachedCount.load(std::memory_order_relaxed); }

    /**
     * @brief Get a chunk, loading from disk if needed.
     * Returns shared_ptr — caller holds the chunk alive even if evicted.
     */
    ChunkPtr get(int ix, int iy, int iz);

    /**
     * @brief Get raw pointer (for hot-path compatibility). Caller must ensure
     * chunk stays alive (e.g. by also holding a ChunkPtr from get()).
     */
    volcart::zarr::Tensor3D<T>* getRaw(int ix, int iy, int iz);

    ChunkPtr getIfCached(int ix, int iy, int iz) const;

    void prefetch(int minIx, int minIy, int minIz, int maxIx, int maxIy, int maxIz);
    void clear();

    /** @brief Drop all cached chunks but keep the cache initialized for the same dataset */
    void flush();

    int chunkSizeX() const { return _cw; }
    int chunkSizeY() const { return _ch; }
    int chunkSizeZ() const { return _cd; }
    int datasetSizeX() const { return _sx; }
    int datasetSizeY() const { return _sy; }
    int datasetSizeZ() const { return _sz; }
    int chunksX() const { return _chunksX; }
    int chunksY() const { return _chunksY; }
    int chunksZ() const { return _chunksZ; }

    int chunkShiftX() const { return _cwShift; }
    int chunkShiftY() const { return _chShift; }
    int chunkShiftZ() const { return _cdShift; }
    int chunkMaskX() const { return _cwMask; }
    int chunkMaskY() const { return _chMask; }
    int chunkMaskZ() const { return _cdMask; }

    volcart::zarr::ZarrDataset* dataset() const { return _ds; }

private:
    volcart::zarr::ZarrDataset* _ds = nullptr;

    size_t _maxBytes = 0;
    size_t _maxChunks = 0;
    std::atomic<size_t> _cachedCount{0};
    std::atomic<uint64_t> _generation{0};

    int _cw = 0, _ch = 0, _cd = 0;
    int _sx = 0, _sy = 0, _sz = 0;
    int _chunksX = 0, _chunksY = 0, _chunksZ = 0;

    int _cwShift = 0, _chShift = 0, _cdShift = 0;
    int _cwMask = 0, _chMask = 0, _cdMask = 0;

    // Flat 3D array: _chunks[ix * _chunksY * _chunksZ + iy * _chunksZ + iz]
    // atomic shared_ptr for lock-free reads on the fast path
    size_t _totalSlots = 0;
    std::unique_ptr<std::atomic<ChunkPtr>[]> _chunks;
    std::unique_ptr<std::atomic<uint64_t>[]> _lastAccess;

    std::mutex _evictionMutex;

    static constexpr int kLockPoolSize = 64;
    std::mutex _lockPool[kLockPoolSize];

    size_t idx(int ix, int iy, int iz) const {
        return static_cast<size_t>(ix) * _chunksY * _chunksZ +
               static_cast<size_t>(iy) * _chunksZ +
               static_cast<size_t>(iz);
    }

    ChunkPtr loadChunk(int ix, int iy, int iz);
    void evictIfNeeded();

    static int log2_pow2(int v) {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};

extern template class ChunkCache<uint8_t>;
extern template class ChunkCache<uint16_t>;
