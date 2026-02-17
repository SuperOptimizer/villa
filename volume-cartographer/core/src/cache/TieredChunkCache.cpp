#include "vc/core/cache/TieredChunkCache.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace vc::cache {

static FILE* debugLog()
{
    static FILE* f = [] {
        FILE* fp = std::fopen("/tmp/tiled_debug.log", "w");
        if (fp) std::setvbuf(fp, nullptr, _IOLBF, 0);  // line-buffered
        return fp;
    }();
    return f;
}

TieredChunkCache::TieredChunkCache(
    Config config,
    std::unique_ptr<ChunkSource> source,
    DecompressFn decompress,
    std::shared_ptr<DiskStore> diskStore)
    : config_(std::move(config))
    , diskStore_(std::move(diskStore))
    , source_(std::move(source))
    , decompress_(std::move(decompress))
    , ioPool_(config_.ioThreads)
{
    // Wire up the IO pool: fetch = check cold first, then ice
    ioPool_.setFetchFunc([this](const ChunkKey& key) -> std::vector<uint8_t> {
        if (auto* log = debugLog())
            std::fprintf(log, "FETCH start lvl=%d (%d,%d,%d)\n", key.level, key.iz, key.iy, key.ix);

        // Try cold (disk cache) first
        if (diskStore_) {
            auto diskData = diskStore_->get(config_.volumeId, key);
            if (diskData && !diskData->empty()) {
                statColdHits_.fetch_add(1, std::memory_order_relaxed);
                if (auto* log = debugLog())
                    std::fprintf(log, "FETCH cold-hit lvl=%d (%d,%d,%d) bytes=%zu\n",
                                 key.level, key.iz, key.iy, key.ix, diskData->size());
                return std::move(*diskData);
            }
        }

        // Fetch from ice (remote/filesystem source)
        if (!source_) return {};

        auto data = source_->fetch(key);
        if (data.empty()) {
            if (auto* log = debugLog())
                std::fprintf(log, "FETCH ice-empty lvl=%d (%d,%d,%d)\n",
                             key.level, key.iz, key.iy, key.ix);
            return {};
        }

        statIceFetches_.fetch_add(1, std::memory_order_relaxed);
        if (auto* log = debugLog())
            std::fprintf(log, "FETCH ice-ok lvl=%d (%d,%d,%d) bytes=%zu\n",
                         key.level, key.iz, key.iy, key.ix, data.size());

        // Store to cold (disk cache) for persistence
        if (diskStore_) {
            diskStore_->put(config_.volumeId, key, data);
        }

        return data;
    });

    // Wire up IO completion: compressed bytes → warm → hot
    ioPool_.setCompletionCallback(
        [this](const ChunkKey& key, std::vector<uint8_t>&& compressed) {
            if (compressed.empty()) {
                if (auto* log = debugLog())
                    std::fprintf(log, "COMPLETE empty lvl=%d (%d,%d,%d)\n",
                                 key.level, key.iz, key.iy, key.ix);
                return;
            }

            // Estimate decompressed size from chunk shape metadata
            size_t decompSize = 0;
            if (source_) {
                auto cs = source_->chunkShape(key.level);
                decompSize =
                    static_cast<size_t>(cs[0]) * cs[1] * cs[2];  // * elementSize
            }

            // Store in warm tier
            warmPut(key, std::move(compressed), decompSize);

            // Decompress and store in hot tier
            auto warmEntry = warmGet(key);
            if (warmEntry && decompress_) {
                auto data = decompress_(warmEntry->data, key);
                if (data) {
                    // Check if this key should be pinned
                    bool shouldPin = false;
                    {
                        std::lock_guard plock(pinnedKeysMutex_);
                        shouldPin = pendingPinKeys_.count(key) > 0;
                    }
                    size_t decompBytes = data->totalBytes();
                    hotPut(key, std::move(data), shouldPin);
                    if (auto* log = debugLog())
                        std::fprintf(log, "COMPLETE hot-put lvl=%d (%d,%d,%d) decompBytes=%zu\n",
                                     key.level, key.iz, key.iy, key.ix, decompBytes);
                } else {
                    if (auto* log = debugLog())
                        std::fprintf(log, "COMPLETE decompress-fail lvl=%d (%d,%d,%d)\n",
                                     key.level, key.iz, key.iy, key.ix);
                }
            }

            // Notify caller (e.g., to trigger UI refresh)
            if (chunkReadyCb_) {
                chunkReadyCb_(key);
            }
        });
}

TieredChunkCache::~TieredChunkCache()
{
    ioPool_.stop();
}

// =============================================================================
// Non-blocking reads
// =============================================================================

ChunkDataPtr TieredChunkCache::get(const ChunkKey& key)
{
    // Check hot tier
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    // Check warm tier — decompress and promote to hot
    auto warm = warmGet(key);
    if (warm) {
        statWarmHits_.fetch_add(1, std::memory_order_relaxed);
        return promoteFromWarm(key, std::move(*warm));
    }

    statMisses_.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
}

std::pair<ChunkDataPtr, int> TieredChunkCache::getBestAvailable(
    const ChunkKey& key)
{
    int maxLevel = source_ ? source_->numLevels() - 1 : 0;

    // Try the requested level first, then progressively coarser
    for (int lvl = key.level; lvl <= maxLevel; lvl++) {
        ChunkKey coarsened =
            (lvl == key.level) ? key : key.coarsen(lvl);

        auto data = get(coarsened);
        if (data) return {data, lvl};
    }

    // Nothing available at any level
    return {nullptr, -1};
}

// =============================================================================
// Blocking reads
// =============================================================================

ChunkDataPtr TieredChunkCache::getBlocking(const ChunkKey& key)
{
    // Check hot
    auto hot = hotGet(key);
    if (hot) {
        statHotHits_.fetch_add(1, std::memory_order_relaxed);
        return hot;
    }

    // Check warm
    {
        auto warm = warmGet(key);
        if (warm) {
            statWarmHits_.fetch_add(1, std::memory_order_relaxed);
            return promoteFromWarm(key, std::move(*warm));
        }
    }

    // Per-key lock to prevent duplicate loads
    std::lock_guard diskLock(lockPool_[lockIndex(key)]);

    // Re-check hot after acquiring lock (another thread may have loaded)
    hot = hotGet(key);
    if (hot) return hot;

    // Full promotion chain: cold → warm → hot, or ice → cold → warm → hot
    return loadFull(key);
}

// =============================================================================
// Async prefetch
// =============================================================================

void TieredChunkCache::prefetch(const ChunkKey& key)
{
    // Already in hot or warm? No-op.
    if (hotGet(key)) return;
    if (warmGet(key).has_value()) return;

    ioPool_.submit(key);
}

void TieredChunkCache::prefetchRegion(
    int level, int iz0, int iy0, int ix0, int iz1, int iy1, int ix1)
{
    std::vector<ChunkKey> keys;
    int totalChecked = 0;
    for (int iz = iz0; iz <= iz1; iz++) {
        for (int iy = iy0; iy <= iy1; iy++) {
            for (int ix = ix0; ix <= ix1; ix++) {
                totalChecked++;
                ChunkKey key{level, iz, iy, ix};
                if (!hotGet(key) && !warmGet(key).has_value()) {
                    keys.push_back(key);
                }
            }
        }
    }
    static std::atomic<int> prefetchLogCount{0};
    int n = prefetchLogCount.fetch_add(1, std::memory_order_relaxed);
    if (n < 5) {
        std::fprintf(stderr, "[TILED] prefetchRegion: level=%d range=(%d-%d,%d-%d,%d-%d) checked=%d toSubmit=%zu\n",
                     level, iz0, iz1, iy0, iy1, ix0, ix1, totalChecked, keys.size());
    }
    if (!keys.empty()) {
        ioPool_.submit(keys);
    }
}

void TieredChunkCache::cancelPendingPrefetch()
{
    ioPool_.cancelPending();
}

// =============================================================================
// Pin level
// =============================================================================

void TieredChunkCache::pinLevel(
    int level,
    const std::array<int, 3>& gridDims,
    bool blocking)
{
    std::vector<ChunkKey> keys;
    for (int iz = 0; iz < gridDims[0]; iz++) {
        for (int iy = 0; iy < gridDims[1]; iy++) {
            for (int ix = 0; ix < gridDims[2]; ix++) {
                keys.push_back({level, iz, iy, ix});
            }
        }
    }

    if (blocking) {
        // Load all chunks synchronously and pin them
        int pinned = 0, failed = 0;
        for (auto& key : keys) {
            auto data = getBlocking(key);
            if (data) {
                // Re-insert as pinned
                hotPut(key, std::move(data), /*pinned=*/true);
                pinned++;
            } else {
                failed++;
            }
        }
        std::fprintf(stderr, "[TILED] pinLevel: level=%d total=%zu pinned=%d failed=%d\n",
                     level, keys.size(), pinned, failed);
    } else {
        // Register keys as pending pin, then submit for background loading
        {
            std::lock_guard lock(pinnedKeysMutex_);
            for (auto& key : keys) {
                pendingPinKeys_.insert(key);
            }
        }
        for (auto& key : keys) {
            if (!hotGet(key)) {
                ioPool_.submit(key);
            } else {
                // Already in hot — just mark as pinned
                hotPut(key, hotGet(key), /*pinned=*/true);
            }
        }
    }
}

// =============================================================================
// Cache management
// =============================================================================

void TieredChunkCache::clearMemory()
{
    {
        std::unique_lock lock(hotMutex_);
        hot_.clear();
        hotBytes_.store(0, std::memory_order_relaxed);
    }
    {
        std::lock_guard lock(warmMutex_);
        warm_.clear();
        warmBytes_ = 0;
    }
}

void TieredChunkCache::clearAll()
{
    ioPool_.cancelPending();
    clearMemory();
    if (diskStore_) {
        diskStore_->clearVolume(config_.volumeId);
    }
}

int TieredChunkCache::numLevels() const
{
    return source_ ? source_->numLevels() : 0;
}

std::array<int, 3> TieredChunkCache::chunkShape(int level) const
{
    return source_ ? source_->chunkShape(level) : std::array<int, 3>{0, 0, 0};
}

std::array<int, 3> TieredChunkCache::levelShape(int level) const
{
    return source_ ? source_->levelShape(level) : std::array<int, 3>{0, 0, 0};
}

void TieredChunkCache::setChunkReadyCallback(ChunkReadyCallback cb)
{
    chunkReadyCb_ = std::move(cb);
}

// =============================================================================
// Stats
// =============================================================================

auto TieredChunkCache::stats() const -> Stats
{
    Stats s;
    s.hotHits = statHotHits_.load(std::memory_order_relaxed);
    s.warmHits = statWarmHits_.load(std::memory_order_relaxed);
    s.coldHits = statColdHits_.load(std::memory_order_relaxed);
    s.iceFetches = statIceFetches_.load(std::memory_order_relaxed);
    s.misses = statMisses_.load(std::memory_order_relaxed);
    s.hotEvictions = statHotEvictions_.load(std::memory_order_relaxed);
    s.warmEvictions = statWarmEvictions_.load(std::memory_order_relaxed);
    s.hotBytes = hotBytes_.load(std::memory_order_relaxed);
    {
        std::lock_guard lock(warmMutex_);
        s.warmBytes = warmBytes_;
    }
    return s;
}

void TieredChunkCache::resetStats()
{
    statHotHits_.store(0, std::memory_order_relaxed);
    statWarmHits_.store(0, std::memory_order_relaxed);
    statColdHits_.store(0, std::memory_order_relaxed);
    statIceFetches_.store(0, std::memory_order_relaxed);
    statMisses_.store(0, std::memory_order_relaxed);
    statHotEvictions_.store(0, std::memory_order_relaxed);
    statWarmEvictions_.store(0, std::memory_order_relaxed);
}

// =============================================================================
// Hot tier
// =============================================================================

ChunkDataPtr TieredChunkCache::hotGet(const ChunkKey& key)
{
    std::shared_lock lock(hotMutex_);
    auto it = hot_.find(key);
    if (it == hot_.end()) return nullptr;
    it->second.generation = hotGen_.fetch_add(1, std::memory_order_relaxed);
    return it->second.data;
}

void TieredChunkCache::hotPut(
    const ChunkKey& key, ChunkDataPtr data, bool pinned)
{
    size_t bytes = data ? data->totalBytes() : 0;

    {
        std::unique_lock lock(hotMutex_);
        auto [it, inserted] = hot_.try_emplace(
            key,
            HotEntry{data, bytes,
                     hotGen_.fetch_add(1, std::memory_order_relaxed), pinned});
        if (!inserted) {
            // Update existing entry
            size_t oldBytes = it->second.bytes;
            it->second.data = data;
            it->second.bytes = bytes;
            it->second.generation =
                hotGen_.fetch_add(1, std::memory_order_relaxed);
            it->second.pinned = it->second.pinned || pinned;
            hotBytes_.fetch_sub(oldBytes, std::memory_order_relaxed);
        }
    }

    hotBytes_.fetch_add(bytes, std::memory_order_relaxed);
    hotEvictIfNeeded();
}

void TieredChunkCache::hotEvictIfNeeded()
{
    if (config_.hotMaxBytes == 0) return;
    size_t current = hotBytes_.load(std::memory_order_relaxed);
    if (current <= config_.hotMaxBytes) return;

    std::lock_guard evictLock(hotEvictMutex_);

    // Re-check after acquiring eviction lock
    current = hotBytes_.load(std::memory_order_relaxed);
    if (current <= config_.hotMaxBytes) return;

    struct Candidate {
        ChunkKey key;
        size_t bytes;
        uint64_t generation;
    };

    std::vector<Candidate> candidates;
    {
        std::shared_lock lock(hotMutex_);
        candidates.reserve(hot_.size());
        for (auto& [k, e] : hot_) {
            if (e.pinned) continue;  // never evict pinned entries
            candidates.push_back({k, e.bytes, e.generation});
        }
    }

    if (candidates.empty()) return;

    // Sort by generation (oldest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.generation < b.generation;
              });

    size_t target = config_.hotMaxBytes * 15 / 16;  // evict to 15/16 capacity
    size_t evictedBytes = 0;
    size_t evictedCount = 0;

    std::vector<ChunkKey> toRemove;
    for (auto& c : candidates) {
        if (current - evictedBytes <= target) break;
        toRemove.push_back(c.key);
        evictedBytes += c.bytes;
        evictedCount++;
    }

    if (!toRemove.empty()) {
        std::unique_lock lock(hotMutex_);
        for (auto& k : toRemove) {
            hot_.erase(k);
        }
        hotBytes_.fetch_sub(evictedBytes, std::memory_order_relaxed);
        statHotEvictions_.fetch_add(evictedCount, std::memory_order_relaxed);
    }
}

// =============================================================================
// Warm tier
// =============================================================================

std::optional<CompressedChunk> TieredChunkCache::warmGet(const ChunkKey& key)
{
    std::lock_guard lock(warmMutex_);
    auto it = warm_.find(key);
    if (it == warm_.end()) return std::nullopt;
    it->second.generation = warmGen_++;
    return it->second.chunk;  // copy while lock is held
}

void TieredChunkCache::warmPut(
    const ChunkKey& key,
    std::vector<uint8_t> compressed,
    size_t decompressedSize)
{
    size_t bytes = compressed.size();

    {
        std::lock_guard lock(warmMutex_);
        auto it = warm_.find(key);
        if (it != warm_.end()) {
            // Update existing entry
            warmBytes_ -= it->second.chunk.data.size();
            it->second.chunk.data = std::move(compressed);
            it->second.chunk.decompressedSize = decompressedSize;
            it->second.generation = warmGen_++;
        } else {
            warm_.emplace(
                key,
                WarmEntry{
                    CompressedChunk{std::move(compressed), decompressedSize},
                    warmGen_++});
        }
        warmBytes_ += bytes;
    }

    warmEvictIfNeeded();
}

void TieredChunkCache::warmEvictIfNeeded()
{
    if (config_.warmMaxBytes == 0) return;

    std::lock_guard lock(warmMutex_);
    if (warmBytes_ <= config_.warmMaxBytes) return;

    struct Candidate {
        ChunkKey key;
        size_t bytes;
        uint64_t generation;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(warm_.size());
    for (auto& [k, e] : warm_) {
        candidates.push_back({k, e.chunk.data.size(), e.generation});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.generation < b.generation;
              });

    size_t target = config_.warmMaxBytes * 15 / 16;
    for (auto& c : candidates) {
        if (warmBytes_ <= target) break;
        auto it = warm_.find(c.key);
        if (it != warm_.end()) {
            warmBytes_ -= it->second.chunk.data.size();
            warm_.erase(it);
            statWarmEvictions_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// =============================================================================
// Promotion helpers
// =============================================================================

ChunkDataPtr TieredChunkCache::promoteFromWarm(
    const ChunkKey& key, CompressedChunk warm)
{
    if (!decompress_) return nullptr;

    auto data = decompress_(warm.data, key);
    if (!data) return nullptr;

    hotPut(key, data);
    return data;
}

ChunkDataPtr TieredChunkCache::promoteFromCold(const ChunkKey& key)
{
    if (!diskStore_) return nullptr;

    auto compressed = diskStore_->get(config_.volumeId, key);
    if (!compressed || compressed->empty()) return nullptr;

    statColdHits_.fetch_add(1, std::memory_order_relaxed);

    // Estimate decompressed size
    size_t decompSize = 0;
    if (source_) {
        auto cs = source_->chunkShape(key.level);
        decompSize = static_cast<size_t>(cs[0]) * cs[1] * cs[2];
    }

    // Store in warm
    warmPut(key, std::move(*compressed), decompSize);

    // Decompress and promote to hot
    auto warmEntry = warmGet(key);
    if (!warmEntry) return nullptr;

    return promoteFromWarm(key, std::move(*warmEntry));
}

ChunkDataPtr TieredChunkCache::promoteFromIce(const ChunkKey& key)
{
    if (!source_) return nullptr;

    auto compressed = source_->fetch(key);
    if (compressed.empty()) return nullptr;

    statIceFetches_.fetch_add(1, std::memory_order_relaxed);

    // Store to cold (disk cache)
    if (diskStore_) {
        diskStore_->put(config_.volumeId, key, compressed);
    }

    // Estimate decompressed size
    size_t decompSize = 0;
    auto cs = source_->chunkShape(key.level);
    decompSize = static_cast<size_t>(cs[0]) * cs[1] * cs[2];

    // Store in warm
    warmPut(key, std::move(compressed), decompSize);

    // Decompress and promote to hot
    auto warmEntry = warmGet(key);
    if (!warmEntry) return nullptr;

    return promoteFromWarm(key, std::move(*warmEntry));
}

ChunkDataPtr TieredChunkCache::loadFull(const ChunkKey& key)
{
    // Try cold (disk cache)
    auto data = promoteFromCold(key);
    if (data) return data;

    // Try ice (remote/filesystem)
    return promoteFromIce(key);
}

void TieredChunkCache::onIOComplete(
    const ChunkKey& key, std::vector<uint8_t>&& compressed)
{
    if (compressed.empty()) return;

    size_t decompSize = 0;
    if (source_) {
        auto cs = source_->chunkShape(key.level);
        decompSize = static_cast<size_t>(cs[0]) * cs[1] * cs[2];
    }

    warmPut(key, std::move(compressed), decompSize);

    auto warmEntry = warmGet(key);
    if (warmEntry && decompress_) {
        auto data = decompress_(warmEntry->data, key);
        if (data) {
            hotPut(key, std::move(data));
        }
    }

    if (chunkReadyCb_) {
        chunkReadyCb_(key);
    }
}

}  // namespace vc::cache
