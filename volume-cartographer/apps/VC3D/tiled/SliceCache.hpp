#pragma once

#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include <QPixmap>
#include <opencv2/core.hpp>

#include "TileScene.hpp"
#include "TiledViewerCamera.hpp"

// Cache key for a rendered tile.
// World tile content is position-independent for both surface types
// (surfacePtr cancels out in gen() with world-aligned offsets).
// Quantizes floating-point values to prevent float comparison issues.
struct SliceCacheKey {
    WorldTileKey worldTile;

    // Quantized scale (1/32 log2 stops)
    int16_t scaleQ = 0;

    // Z offset quantized to 0.25 units
    int16_t zOffQ = 0;

    // Pyramid level
    int8_t dsScaleIdx = 0;

    // Hash of rendering parameters (window/level, colormap, composite settings)
    uint64_t paramsHash = 0;

    bool operator==(const SliceCacheKey& o) const
    {
        return worldTile == o.worldTile && scaleQ == o.scaleQ && zOffQ == o.zOffQ &&
               dsScaleIdx == o.dsScaleIdx && paramsHash == o.paramsHash;
    }

    // Build a cache key from a world tile key and camera state + render params.
    static SliceCacheKey make(const WorldTileKey& worldTile, const TiledViewerCamera& camera,
                              uint64_t paramsHash);
};

// Hash for SliceCacheKey
struct SliceCacheKeyHash {
    size_t operator()(const SliceCacheKey& k) const;
};

// Result of a level-aware cache lookup.
struct SliceCacheLookup {
    QPixmap pixmap;
    int8_t level = -1;  // actual level found, -1 = miss
};

// Thread-safe LRU cache for rendered tile pixmaps.
// Default capacity: 2048 entries (~384MB at 256x256 BGR).
class SliceCache
{
public:
    explicit SliceCache(size_t maxEntries = 2048);

    // Look up a tile. Returns empty optional on miss.
    std::optional<QPixmap> get(const SliceCacheKey& key);

    // Find best available: checks requested level, then coarser.
    // Returns {pixmap, level} where level is the actual pyramid level found,
    // or level = -1 on complete miss.
    SliceCacheLookup getBest(const SliceCacheKey& key, int maxCoarserLevels = tiled_config::MAX_COARSER_LEVELS);

    // Store a rendered tile.
    void put(const SliceCacheKey& key, const QPixmap& pixmap);

    // Clear all entries.
    void clear();

    // Remove entries matching a predicate.
    void invalidate(const std::function<bool(const SliceCacheKey&)>& predicate);

    // Stats
    size_t size() const;
    size_t hits() const { return _hits; }
    size_t misses() const { return _misses; }

private:
    struct Entry {
        SliceCacheKey key;
        QPixmap pixmap;
    };

    using ListIt = std::list<Entry>::iterator;

    mutable std::mutex _mutex;
    std::list<Entry> _lruList;   // front = most recently used
    std::unordered_map<SliceCacheKey, ListIt, SliceCacheKeyHash> _map;
    size_t _maxEntries;
    size_t _hits = 0;
    size_t _misses = 0;
};
