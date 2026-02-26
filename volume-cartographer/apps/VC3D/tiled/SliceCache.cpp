#include "SliceCache.hpp"
#include "FnvHash.hpp"
#include <cmath>

// ============================================================================
// SliceCacheKey
// ============================================================================

SliceCacheKey SliceCacheKey::make(const WorldTileKey& worldTile, const TiledViewerCamera& camera,
                                  uint64_t paramsHash)
{
    SliceCacheKey k;
    k.worldTile = worldTile;

    // Quantize scale to 1/32 log2 stops
    k.scaleQ = static_cast<int16_t>(std::round(std::log2(camera.scale) * 32.0f));

    // Quantize zOff to 0.25 units
    k.zOffQ = static_cast<int16_t>(std::round(camera.zOff * 4.0f));

    k.dsScaleIdx = static_cast<int8_t>(camera.dsScaleIdx);
    k.paramsHash = paramsHash;

    return k;
}

size_t SliceCacheKeyHash::operator()(const SliceCacheKey& k) const
{
    uint64_t h = fnv::BASIS;
    fnv::mix(h, static_cast<uint64_t>(k.worldTile.worldCol));
    fnv::mix(h, static_cast<uint64_t>(k.worldTile.worldRow));
    fnv::mix(h, static_cast<uint64_t>(k.scaleQ));
    fnv::mix(h, static_cast<uint64_t>(k.zOffQ));
    fnv::mix(h, static_cast<uint64_t>(k.dsScaleIdx));
    fnv::mix(h, k.paramsHash);
    return static_cast<size_t>(h);
}

// ============================================================================
// SliceCache
// ============================================================================

SliceCache::SliceCache(size_t maxEntries)
    : _maxEntries(maxEntries)
{
}

SliceCacheLookup SliceCache::getBest(const SliceCacheKey& key, int maxCoarserLevels)
{
    std::lock_guard<std::mutex> lock(_mutex);

    // Try exact match first
    auto it = _map.find(key);
    if (it != _map.end()) {
        _lruList.splice(_lruList.begin(), _lruList, it->second);
        ++_hits;
        return {it->second->pixmap, key.dsScaleIdx};
    }

    // Try coarser levels
    for (int delta = 1; delta <= maxCoarserLevels; delta++) {
        SliceCacheKey coarser = key;
        coarser.dsScaleIdx = key.dsScaleIdx + delta;
        // Recompute scaleQ for the coarser level — the camera scale is the
        // same, only the pyramid level changes, so scaleQ stays the same.
        // dsScaleIdx is the only difference in the key.

        auto cit = _map.find(coarser);
        if (cit != _map.end()) {
            _lruList.splice(_lruList.begin(), _lruList, cit->second);
            ++_hits;
            return {cit->second->pixmap, coarser.dsScaleIdx};
        }
    }

    ++_misses;
    return {QPixmap(), -1};
}

void SliceCache::put(const SliceCacheKey& key, const QPixmap& pixmap)
{
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _map.find(key);
    if (it != _map.end()) {
        // Update existing entry and move to front
        it->second->pixmap = pixmap;
        _lruList.splice(_lruList.begin(), _lruList, it->second);
        return;
    }

    // Evict oldest entries if at capacity
    while (_lruList.size() >= _maxEntries) {
        auto& back = _lruList.back();
        _map.erase(back.key);
        _lruList.pop_back();
    }

    // Insert new entry at front
    _lruList.push_front({key, pixmap});
    _map[key] = _lruList.begin();
}

void SliceCache::clear()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _map.clear();
    _lruList.clear();
}

size_t SliceCache::size() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    return _lruList.size();
}
