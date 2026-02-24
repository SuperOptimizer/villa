#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declaration
namespace vc { class Zarr; }

/**
 * @brief Simple row-major 3D array for chunk data storage.
 *
 * Replaces xt::xarray<T> in the cache. Independent of ChunkedTensor.hpp's
 * Array3D to avoid circular dependencies.
 */
template<typename T>
class ChunkArray3D
{
public:
    ChunkArray3D() : d0_(0), d1_(0), d2_(0) {}

    ChunkArray3D(size_t d0, size_t d1, size_t d2)
        : d0_(d0), d1_(d1), d2_(d2), data_(d0 * d1 * d2) {}

    T& operator()(size_t z, size_t y, size_t x) {
        return data_[z * d1_ * d2_ + y * d2_ + x];
    }
    const T& operator()(size_t z, size_t y, size_t x) const {
        return data_[z * d1_ * d2_ + y * d2_ + x];
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }

    std::array<size_t, 3> shape() const { return {d0_, d1_, d2_}; }

private:
    size_t d0_, d1_, d2_;
    std::vector<T> data_;
};

/**
 * @brief Thread-safe chunk cache with shared_ptr lifetime management
 *
 * @tparam T Data type of cached chunks (uint8_t or uint16_t)
 *
 * Supports caching chunks from multiple VcDataset instances simultaneously.
 * Chunks are stored as shared_ptr so eviction removes from the cache but
 * doesn't free memory until all readers are done.
 *
 * Uses a hash map keyed by (dataset pointer, chunk z, chunk y, chunk x).
 * LRU eviction based on generation counter.
 */
template<typename T>
class ChunkCache
{
public:
    using ChunkPtr = std::shared_ptr<ChunkArray3D<T>>;

    explicit ChunkCache(size_t maxBytes = 0);
    ~ChunkCache();

    ChunkCache(const ChunkCache&) = delete;
    ChunkCache& operator=(const ChunkCache&) = delete;
    ChunkCache(ChunkCache&&) = delete;
    ChunkCache& operator=(ChunkCache&&) = delete;

    void setMaxBytes(size_t maxBytes);
    size_t cachedCount() const { return _cachedCount.load(std::memory_order_relaxed); }

    // Cache statistics
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t evictions = 0;
        uint64_t bytesRead = 0;
        uint64_t reReads = 0;        // chunks loaded again after eviction
        uint64_t reReadBytes = 0;    // bytes wasted on re-reads
    };
    Stats stats() const;
    void resetStats();

    /**
     * @brief Get a chunk, loading from disk if needed.
     * Returns shared_ptr -- caller holds the chunk alive even if evicted.
     */
    ChunkPtr get(vc::Zarr* ds, int iz, int iy, int ix);

    /**
     * @brief Check if chunk is cached without loading.
     */
    ChunkPtr getIfCached(vc::Zarr* ds, int iz, int iy, int ix) const;

    void prefetch(vc::Zarr* ds, int minIz, int minIy, int minIx, int maxIz, int maxIy, int maxIx);
    void clear();
    void flush();

private:
    struct ChunkKey {
        vc::Zarr* ds;
        int iz, iy, ix;
        bool operator==(const ChunkKey& o) const {
            return ds == o.ds && iz == o.iz && iy == o.iy && ix == o.ix;
        }
    };

    struct ChunkKeyHash {
        size_t operator()(const ChunkKey& k) const {
            size_t h = std::hash<const void*>()(k.ds);
            h ^= std::hash<int>()(k.iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct CacheEntry {
        ChunkPtr chunk;
        size_t bytes;
        uint64_t lastAccess;
    };

    size_t _maxBytes = 0;
    std::atomic<size_t> _storedBytes{0};
    std::atomic<size_t> _cachedCount{0};
    std::atomic<uint64_t> _generation{0};

    mutable std::shared_mutex _mapMutex;
    std::unordered_map<ChunkKey, CacheEntry, ChunkKeyHash> _map;

    static constexpr int kLockPoolSize = 64;
    std::mutex _lockPool[kLockPoolSize];
    std::mutex _evictionMutex;

    size_t lockIndex(const ChunkKey& k) const { return ChunkKeyHash()(k) % kLockPoolSize; }

    ChunkPtr loadChunk(vc::Zarr* ds, int iz, int iy, int ix);
    void evictIfNeeded();

    // Stats counters
    mutable std::atomic<uint64_t> _hits{0};
    std::atomic<uint64_t> _misses{0};
    std::atomic<uint64_t> _evictions{0};
    std::atomic<uint64_t> _bytesRead{0};
    std::atomic<uint64_t> _reReads{0};
    std::atomic<uint64_t> _reReadBytes{0};

    // Track which chunks have been evicted (persists across evictions)
    std::unordered_set<ChunkKey, ChunkKeyHash> _everLoaded;
    // Protected by _evictionMutex (reused since loads are already serialized)
};

extern template class ChunkCache<uint8_t>;
extern template class ChunkCache<uint16_t>;
