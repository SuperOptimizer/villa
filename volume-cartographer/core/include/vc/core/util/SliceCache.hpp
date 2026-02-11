#pragma once

#include <opencv2/core/mat.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

/**
 * @brief Thread-safe tile cache for pre-window/level sampled slice data
 *
 * @tparam T Data type of cached tiles (uint8_t or uint16_t)
 *
 * Sits between the render thread pool and display. Stores raw sampled tile
 * data so that window/level adjustments are instant without re-sampling from
 * volume chunks.
 *
 * Uses epoch-based spatial invalidation: the controller bumps the epoch on
 * spatial parameter changes (slice position, orientation, etc.), and stale
 * tiles are bulk-purged via evictBefore().
 *
 * Progressive fallback: when an exact pyramid level isn't cached, getBest()
 * returns the nearest coarser level that is available, enabling smooth
 * progressive rendering.
 *
 * Thread safety follows ChunkCache's pattern: shared_mutex for map access,
 * separate eviction mutex, atomic stat counters.
 */
template<typename T>
class SliceCache
{
public:
    using TileData = cv::Mat_<T>;

    struct Key {
        const void* volumeId;  // Volume pointer identity
        int col, row;          // Tile position in viewport grid
        int level;             // Pyramid level (0=finest)
        uint64_t epoch;        // Bumped by controller on spatial param changes

        bool operator==(const Key&) const = default;
    };

    struct Result {
        TileData data;
        int actualLevel;  // -1 on total miss
    };

    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t fallbacks = 0;
        uint64_t evictions = 0;
        size_t storedBytes = 0;
        size_t storedCount = 0;
    };

    explicit SliceCache(size_t maxBytes = 0);
    ~SliceCache();

    SliceCache(const SliceCache&) = delete;
    SliceCache& operator=(const SliceCache&) = delete;
    SliceCache(SliceCache&&) = delete;
    SliceCache& operator=(SliceCache&&) = delete;

    /** @brief Exact-match lookup. Returns empty Mat on miss. */
    TileData get(const Key& key) const;

    /** @brief Progressive fallback: tries key.level, then coarser toward maxLevel. */
    Result getBest(const Key& key, int maxLevel) const;

    /** @brief Insert or update a tile. */
    void put(const Key& key, TileData data);

    /** @brief Clear everything. */
    void invalidateAll();

    /** @brief Bulk-purge all entries with epoch < minEpoch. */
    void evictBefore(uint64_t minEpoch);

    void setMaxBytes(size_t maxBytes);
    Stats stats() const;
    void resetStats();

private:
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = std::hash<const void*>()(k.volumeId);
            h ^= std::hash<int>()(k.col) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.row) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.level) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint64_t>()(k.epoch) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct CacheEntry {
        TileData data;
        size_t bytes;
        mutable uint64_t lastAccess;  // Written under shared lock (benign race)
        uint64_t epoch;
    };

    size_t _maxBytes = 0;
    std::atomic<size_t> _storedBytes{0};
    std::atomic<size_t> _storedCount{0};
    mutable std::atomic<uint64_t> _generation{0};

    mutable std::shared_mutex _mapMutex;
    std::unordered_map<Key, CacheEntry, KeyHash> _map;
    std::mutex _evictionMutex;

    // Stats counters
    mutable std::atomic<uint64_t> _hits{0};
    mutable std::atomic<uint64_t> _misses{0};
    mutable std::atomic<uint64_t> _fallbacks{0};
    std::atomic<uint64_t> _evictions{0};

    void evictIfNeeded();
};

extern template class SliceCache<uint8_t>;
extern template class SliceCache<uint16_t>;
