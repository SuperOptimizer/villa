#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)

#include <atomic>
#include <memory>
#include <vector>
#include <cstdint>
#include <mutex>

// Forward declaration
namespace z5 { class Dataset; }

/**
 * @brief Fast lock-free chunk cache with direct 3D array indexing and NRU eviction
 *
 * @tparam T Data type of cached chunks (uint8_t or uint16_t)
 *
 * Uses a flat array of atomic pointers for O(1) direct-indexed access.
 * Reads are always lock-free. Writes use compare-exchange for thread safety.
 * NRU (Not Recently Used) eviction when cache exceeds max size.
 */
template<typename T>
class ChunkCache
{
public:
    /**
     * @brief Construct cache with optional size limit in bytes
     * @param maxBytes Maximum cache size in bytes (0 = unlimited)
     *
     * The actual max chunks is computed when init() is called based on the
     * dataset's chunk size.
     */
    explicit ChunkCache(size_t maxBytes = 0);
    ~ChunkCache();

    // Non-copyable, non-movable (due to atomics)
    ChunkCache(const ChunkCache&) = delete;
    ChunkCache& operator=(const ChunkCache&) = delete;
    ChunkCache(ChunkCache&&) = delete;
    ChunkCache& operator=(ChunkCache&&) = delete;

    /**
     * @brief Initialize cache for a dataset
     * @param ds Dataset to cache chunks for
     *
     * Clears any existing cache and allocates storage for all possible chunks.
     * Computes maxChunks from maxBytes based on actual chunk size.
     * Must be called before get().
     */
    void init(z5::Dataset* ds);

    /**
     * @brief Check if cache is initialized
     */
    bool initialized() const { return _ds != nullptr; }

    /**
     * @brief Set maximum cache size in bytes
     * @param maxBytes Maximum cache size in bytes (0 = unlimited)
     *
     * If already initialized, recomputes maxChunks based on current chunk size.
     */
    void setMaxBytes(size_t maxBytes);

    /**
     * @brief Get current cache size
     */
    size_t cachedCount() const { return _cachedCount.load(std::memory_order_relaxed); }

    /**
     * @brief Get a chunk, loading it if necessary (lock-free read path)
     * @param ix Chunk x index
     * @param iy Chunk y index
     * @param iz Chunk z index
     * @return Pointer to chunk data, or nullptr if out of bounds/doesn't exist
     *
     * Lock-free if chunk is already loaded. If not loaded, will load from
     * disk and cache atomically. Thread-safe even if multiple threads try
     * to load the same chunk simultaneously.
     */
    xt::xarray<T>* get(int ix, int iy, int iz);

    /**
     * @brief Get a chunk without loading (pure lookup)
     * @return Pointer to chunk if cached, nullptr otherwise
     */
    xt::xarray<T>* getIfCached(int ix, int iy, int iz) const;

    /**
     * @brief Prefetch chunks in a region (useful for predictable access patterns)
     * @param minIx,minIy,minIz Lower bounds (inclusive)
     * @param maxIx,maxIy,maxIz Upper bounds (inclusive)
     */
    void prefetch(int minIx, int minIy, int minIz, int maxIx, int maxIy, int maxIz);

    /**
     * @brief Clear all cached chunks
     */
    void clear();

    // Dimension accessors
    int chunkSizeX() const { return _cw; }
    int chunkSizeY() const { return _ch; }
    int chunkSizeZ() const { return _cd; }
    int datasetSizeX() const { return _sx; }
    int datasetSizeY() const { return _sy; }
    int datasetSizeZ() const { return _sz; }
    int chunksX() const { return _chunksX; }
    int chunksY() const { return _chunksY; }
    int chunksZ() const { return _chunksZ; }

    // Bit-shift accessors for optimized coordinate conversion
    int chunkShiftX() const { return _cwShift; }
    int chunkShiftY() const { return _chShift; }
    int chunkShiftZ() const { return _cdShift; }
    int chunkMaskX() const { return _cwMask; }
    int chunkMaskY() const { return _chMask; }
    int chunkMaskZ() const { return _cdMask; }

    /**
     * @brief Get the underlying dataset
     */
    z5::Dataset* dataset() const { return _ds; }

private:
    z5::Dataset* _ds = nullptr;

    // Cache size limits
    size_t _maxBytes = 0;   // User-specified limit in bytes (0 = unlimited)
    size_t _maxChunks = 0;  // Computed from _maxBytes and actual chunk size
    std::atomic<size_t> _cachedCount{0};
    std::atomic<uint64_t> _generation{0};  // Global generation counter for NRU

    // Chunk dimensions
    int _cw = 0, _ch = 0, _cd = 0;

    // Dataset dimensions
    int _sx = 0, _sy = 0, _sz = 0;

    // Number of chunks in each dimension
    int _chunksX = 0, _chunksY = 0, _chunksZ = 0;

    // Bit shift constants for power-of-2 chunk sizes
    int _cwShift = 0, _chShift = 0, _cdShift = 0;
    int _cwMask = 0, _chMask = 0, _cdMask = 0;

    // Cache entry with chunk pointer and NRU generation
    struct Entry {
        std::atomic<xt::xarray<T>*> chunk{nullptr};
        std::atomic<uint64_t> lastAccess{0};

        Entry() = default;
        ~Entry() = default;

        // Move constructor - atomics aren't movable, so load/store values
        Entry(Entry&& other) noexcept {
            chunk.store(other.chunk.exchange(nullptr, std::memory_order_relaxed), std::memory_order_relaxed);
            lastAccess.store(other.lastAccess.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }

        // Move assignment
        Entry& operator=(Entry&& other) noexcept {
            if (this != &other) {
                chunk.store(other.chunk.exchange(nullptr, std::memory_order_relaxed), std::memory_order_relaxed);
                lastAccess.store(other.lastAccess.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
            return *this;
        }

        // Delete copy operations
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
    };

    // Flat array of entries - direct indexed by (ix, iy, iz)
    std::vector<Entry> _entries;

    // Mutex for eviction (only used during eviction, not reads)
    std::mutex _evictionMutex;
    // Mutex for chunk loading (serializes blosc2 decompression which is not thread-safe)
    std::mutex _loadMutex;

    // Convert chunk indices to flat array index
    size_t idx(int ix, int iy, int iz) const {
        return static_cast<size_t>(iz) * _chunksY * _chunksX +
               static_cast<size_t>(iy) * _chunksX +
               static_cast<size_t>(ix);
    }

    // Load a chunk from disk
    xt::xarray<T>* loadChunk(int ix, int iy, int iz);

    // Evict old chunks if over limit (called with _evictionMutex held)
    void evictIfNeeded();

    // Compute log2 for power-of-2 values
    static int log2_pow2(int v) {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};

// Explicit template instantiations (defined in ChunkCache.cpp)
extern template class ChunkCache<uint8_t>;
extern template class ChunkCache<uint16_t>;
