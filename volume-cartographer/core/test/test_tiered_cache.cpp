#include "test.hpp"

#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/DiskStore.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace vc::cache;

// ---- helpers ----------------------------------------------------------------

static fs::path makeTempDir(const std::string& prefix)
{
    auto p = fs::temp_directory_path() / (prefix + "_" + std::to_string(::getpid()));
    fs::create_directories(p);
    return p;
}

struct TempDirGuard {
    fs::path path;
    explicit TempDirGuard(const std::string& prefix) : path(makeTempDir(prefix)) {}
    ~TempDirGuard() { std::error_code ec; fs::remove_all(path, ec); }
};

// Wait for a condition with timeout
template <typename Pred>
bool waitFor(Pred pred, int timeoutMs = 5000)
{
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeoutMs);
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// A mock ChunkSource that serves in-memory data
class MockChunkSource : public ChunkSource {
public:
    struct LevelInfo {
        std::array<int, 3> shape;
        std::array<int, 3> chunkShape;
    };

    explicit MockChunkSource(std::vector<LevelInfo> levels)
        : levels_(std::move(levels)) {}

    // Pre-populate a chunk
    void setChunk(const ChunkKey& key, std::vector<uint8_t> data)
    {
        std::lock_guard lock(mu_);
        chunks_[key] = std::move(data);
    }

    std::vector<uint8_t> fetch(const ChunkKey& key) override
    {
        fetchCount_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(mu_);
        auto it = chunks_.find(key);
        if (it == chunks_.end()) return {};
        return it->second;
    }

    int numLevels() const override { return static_cast<int>(levels_.size()); }
    std::array<int, 3> chunkShape(int level) const override
    {
        if (level < 0 || level >= numLevels()) return {0, 0, 0};
        return levels_[level].chunkShape;
    }
    std::array<int, 3> levelShape(int level) const override
    {
        if (level < 0 || level >= numLevels()) return {0, 0, 0};
        return levels_[level].shape;
    }

    int fetchCount() const { return fetchCount_.load(std::memory_order_relaxed); }

private:
    std::vector<LevelInfo> levels_;
    std::mutex mu_;
    std::unordered_map<ChunkKey, std::vector<uint8_t>, ChunkKeyHash> chunks_;
    std::atomic<int> fetchCount_{0};
};

// A simple identity "decompressor" that wraps raw bytes into ChunkData
static DecompressFn identityDecompress(std::array<int, 3> chunkShape, int elemSize = 1)
{
    return [chunkShape, elemSize](const std::vector<uint8_t>& compressed,
                                  const ChunkKey&) -> ChunkDataPtr {
        auto cd = std::make_shared<ChunkData>();
        cd->bytes = compressed;
        cd->shape = chunkShape;
        cd->elementSize = elemSize;
        return cd;
    };
}

// Make a chunk data vector of known content
static std::vector<uint8_t> makeChunkBytes(int size, uint8_t fill)
{
    return std::vector<uint8_t>(size, fill);
}

// Create a TieredChunkCache with a MockChunkSource
struct CacheSetup {
    TempDirGuard tmpDir;
    MockChunkSource* source;  // raw ptr, owned by cache
    std::shared_ptr<DiskStore> diskStore;
    std::unique_ptr<TieredChunkCache> cache;

    CacheSetup(
        const std::string& testName,
        int numLevels = 3,
        std::array<int, 3> chunkShape = {2, 4, 4},
        size_t hotMax = 1 << 20,   // 1 MB
        size_t warmMax = 1 << 20,  // 1 MB
        int ioThreads = 2,
        bool useDisk = false)
        : tmpDir(testName)
    {
        std::vector<MockChunkSource::LevelInfo> levels;
        for (int i = 0; i < numLevels; i++) {
            int scale = 1 << i;
            levels.push_back({
                {64 / scale, 128 / scale, 128 / scale},
                chunkShape
            });
        }

        auto src = std::make_unique<MockChunkSource>(std::move(levels));
        source = src.get();

        if (useDisk) {
            DiskStore::Config dc;
            dc.root = tmpDir.path / "disk_cache";
            dc.persistent = false;
            diskStore = std::make_shared<DiskStore>(std::move(dc));
        }

        TieredChunkCache::Config cfg;
        cfg.hotMaxBytes = hotMax;
        cfg.warmMaxBytes = warmMax;
        cfg.volumeId = "test_vol";
        cfg.ioThreads = ioThreads;

        cache = std::make_unique<TieredChunkCache>(
            std::move(cfg),
            std::move(src),
            identityDecompress(chunkShape),
            diskStore);
    }
};

// =============================================================================
// Tests
// =============================================================================

TEST(TieredChunkCache, GetMissReturnsNull)
{
    CacheSetup s("test_get_miss");
    auto result = s.cache->get({0, 0, 0, 0});
    EXPECT_TRUE(result == nullptr);
}

TEST(TieredChunkCache, GetBlockingLoadsFromSource)
{
    CacheSetup s("test_blocking");
    auto data = makeChunkBytes(32, 0xAB);
    s.source->setChunk({0, 0, 0, 0}, data);

    auto result = s.cache->getBlocking({0, 0, 0, 0});
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(result->bytes.size(), 32u);
    EXPECT_EQ(result->bytes[0], 0xAB);
}

TEST(TieredChunkCache, GetBlockingPromotesToHot)
{
    CacheSetup s("test_promotes_hot");
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0xCC));

    // First call: loads from source
    auto r1 = s.cache->getBlocking({0, 0, 0, 0});
    ASSERT_TRUE(r1 != nullptr);

    // Second call: should hit hot tier (no additional fetch)
    int fetchBefore = s.source->fetchCount();
    auto r2 = s.cache->get({0, 0, 0, 0});
    ASSERT_TRUE(r2 != nullptr);
    EXPECT_EQ(s.source->fetchCount(), fetchBefore);

    auto stats = s.cache->stats();
    EXPECT_GT(stats.hotHits, 0u);
}

TEST(TieredChunkCache, GetBlockingMissingChunkReturnsNull)
{
    CacheSetup s("test_blocking_miss");
    // Don't set any chunk data in source
    auto result = s.cache->getBlocking({0, 5, 5, 5});
    EXPECT_TRUE(result == nullptr);
}

TEST(TieredChunkCache, NegativeCaching)
{
    CacheSetup s("test_negative");

    // First call: source returns empty, cache records negative
    auto r1 = s.cache->getBlocking({0, 9, 9, 9});
    EXPECT_TRUE(r1 == nullptr);
    EXPECT_TRUE(s.cache->isNegativeCached({0, 9, 9, 9}));

    // Second call: should not fetch from source again
    int fetchBefore = s.source->fetchCount();
    auto r2 = s.cache->getBlocking({0, 9, 9, 9});
    EXPECT_TRUE(r2 == nullptr);
    EXPECT_EQ(s.source->fetchCount(), fetchBefore);
}

TEST(TieredChunkCache, PrefetchAsync)
{
    CacheSetup s("test_prefetch");
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0xDD));

    s.cache->prefetch({0, 0, 0, 0});

    // Wait for async completion
    bool ready = waitFor([&] {
        return s.cache->get({0, 0, 0, 0}) != nullptr;
    });
    EXPECT_TRUE(ready);

    auto result = s.cache->get({0, 0, 0, 0});
    ASSERT_TRUE(result != nullptr);
    EXPECT_EQ(result->bytes[0], 0xDD);
}

TEST(TieredChunkCache, PrefetchRegion)
{
    CacheSetup s("test_prefetch_region");

    // Populate a 2x2x2 region
    for (int iz = 0; iz < 2; iz++)
        for (int iy = 0; iy < 2; iy++)
            for (int ix = 0; ix < 2; ix++)
                s.source->setChunk({0, iz, iy, ix},
                    makeChunkBytes(32, static_cast<uint8_t>(iz * 4 + iy * 2 + ix)));

    s.cache->prefetchRegion(0, 0, 0, 0, 1, 1, 1);

    // Wait for all 8 chunks
    bool ready = waitFor([&] {
        return s.cache->areAllCachedInRegion(0, 0, 0, 0, 1, 1, 1);
    });
    EXPECT_TRUE(ready);

    // Verify each chunk has correct data
    for (int iz = 0; iz < 2; iz++) {
        for (int iy = 0; iy < 2; iy++) {
            for (int ix = 0; ix < 2; ix++) {
                auto d = s.cache->get({0, iz, iy, ix});
                ASSERT_TRUE(d != nullptr);
                EXPECT_EQ(d->bytes[0], static_cast<uint8_t>(iz * 4 + iy * 2 + ix));
            }
        }
    }
}

TEST(TieredChunkCache, GetBestAvailableFallsBack)
{
    CacheSetup s("test_best_avail");

    // Only populate coarsest level (level 2)
    s.source->setChunk({2, 0, 0, 0}, makeChunkBytes(32, 0xEE));
    (void)s.cache->getBlocking({2, 0, 0, 0});  // force into hot

    // Request fine level (0) — should fall back to level 2
    auto [data, actualLevel] = s.cache->getBestAvailable({0, 0, 0, 0});
    ASSERT_TRUE(data != nullptr);
    EXPECT_EQ(actualLevel, 2);
    EXPECT_EQ(data->bytes[0], 0xEE);
}

TEST(TieredChunkCache, GetBestAvailablePrefersFiner)
{
    CacheSetup s("test_best_prefers_fine");

    // Populate both level 0 and level 2
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0x00));
    s.source->setChunk({2, 0, 0, 0}, makeChunkBytes(32, 0x22));
    (void)s.cache->getBlocking({0, 0, 0, 0});
    (void)s.cache->getBlocking({2, 0, 0, 0});

    // Should prefer level 0
    auto [data, actualLevel] = s.cache->getBestAvailable({0, 0, 0, 0});
    ASSERT_TRUE(data != nullptr);
    EXPECT_EQ(actualLevel, 0);
    EXPECT_EQ(data->bytes[0], 0x00);
}

TEST(TieredChunkCache, HotEviction)
{
    // Hot tier max = 256 bytes, each chunk = 32 bytes, so ~8 chunks fit
    CacheSetup s("test_hot_evict", 1, {2, 4, 4}, 256, 1 << 20);

    // Load 16 chunks to exceed hot budget
    for (int i = 0; i < 16; i++) {
        s.source->setChunk({0, 0, 0, i}, makeChunkBytes(32, static_cast<uint8_t>(i)));
        (void)s.cache->getBlocking({0, 0, 0, i});
    }

    auto stats = s.cache->stats();
    EXPECT_GT(stats.hotEvictions, 0u);
    // Hot bytes should be at or below max
    EXPECT_LE(stats.hotBytes, 256u);
}

TEST(TieredChunkCache, PinnedNotEvicted)
{
    // Tiny hot tier: 128 bytes
    CacheSetup s("test_pinned", 3, {2, 4, 4}, 128, 1 << 20);

    // Pin level 2 (1x1x1 grid = 1 chunk)
    s.source->setChunk({2, 0, 0, 0}, makeChunkBytes(32, 0xFF));
    s.cache->pinLevel(2, {1, 1, 1}, true);

    // Verify pinned chunk is in hot
    auto pinned = s.cache->get({2, 0, 0, 0});
    ASSERT_TRUE(pinned != nullptr);
    EXPECT_EQ(pinned->bytes[0], 0xFF);

    // Now load many other chunks to trigger eviction
    for (int i = 0; i < 20; i++) {
        s.source->setChunk({0, 0, 0, i}, makeChunkBytes(32, static_cast<uint8_t>(i)));
        (void)s.cache->getBlocking({0, 0, 0, i});
    }

    // Pinned chunk should still be in hot
    auto stillPinned = s.cache->get({2, 0, 0, 0});
    ASSERT_TRUE(stillPinned != nullptr);
    EXPECT_EQ(stillPinned->bytes[0], 0xFF);
}

TEST(TieredChunkCache, ClearMemory)
{
    CacheSetup s("test_clear_mem");
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0xAA));
    (void)s.cache->getBlocking({0, 0, 0, 0});

    s.cache->clearMemory();

    auto result = s.cache->get({0, 0, 0, 0});
    EXPECT_TRUE(result == nullptr);

    auto stats = s.cache->stats();
    EXPECT_EQ(stats.hotBytes, 0u);
    EXPECT_EQ(stats.warmBytes, 0u);
}

TEST(TieredChunkCache, ClearAll)
{
    CacheSetup s("test_clear_all", 3, {2, 4, 4}, 1 << 20, 1 << 20, 2, true);
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0xBB));
    (void)s.cache->getBlocking({0, 0, 0, 0});

    // Also create a negative cache entry
    (void)s.cache->getBlocking({0, 9, 9, 9});
    EXPECT_TRUE(s.cache->isNegativeCached({0, 9, 9, 9}));

    s.cache->clearAll();

    EXPECT_TRUE(s.cache->get({0, 0, 0, 0}) == nullptr);
    EXPECT_FALSE(s.cache->isNegativeCached({0, 9, 9, 9}));
}

TEST(TieredChunkCache, ChunkReadyCallback)
{
    CacheSetup s("test_callback");
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0x11));

    std::atomic<int> callbackCount{0};
    ChunkKey lastKey{};

    auto id = s.cache->addChunkReadyListener([&](const ChunkKey& key) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
        lastKey = key;
    });

    s.cache->prefetch({0, 0, 0, 0});

    bool called = waitFor([&] {
        return callbackCount.load(std::memory_order_relaxed) > 0;
    });
    EXPECT_TRUE(called);
    EXPECT_EQ(lastKey.level, 0);

    s.cache->removeChunkReadyListener(id);
}

TEST(TieredChunkCache, ChunkReadyDebounce)
{
    CacheSetup s("test_debounce");

    // Set up multiple chunks
    for (int i = 0; i < 5; i++) {
        s.source->setChunk({0, 0, 0, i}, makeChunkBytes(32, static_cast<uint8_t>(i)));
    }

    std::atomic<int> callbackCount{0};
    (void)s.cache->addChunkReadyListener([&](const ChunkKey&) {
        callbackCount.fetch_add(1, std::memory_order_relaxed);
    });

    // Prefetch all 5 — debounce should limit callbacks
    for (int i = 0; i < 5; i++) {
        s.cache->prefetch({0, 0, 0, i});
    }

    // Wait for all chunks to arrive
    bool allReady = waitFor([&] {
        return s.cache->areAllCachedInRegion(0, 0, 0, 0, 0, 0, 4);
    });
    EXPECT_TRUE(allReady);

    // With debouncing, callback count should be 1 (not 5)
    EXPECT_EQ(callbackCount.load(std::memory_order_relaxed), 1);

    // Clear flag and trigger more
    s.cache->clearChunkArrivedFlag();
    s.cache->clearMemory();
    s.cache->prefetch({0, 0, 0, 0});

    bool refetched = waitFor([&] {
        return s.cache->get({0, 0, 0, 0}) != nullptr;
    });
    EXPECT_TRUE(refetched);
    EXPECT_EQ(callbackCount.load(std::memory_order_relaxed), 2);
}

TEST(TieredChunkCache, NumLevelsAndShapes)
{
    CacheSetup s("test_shapes");
    EXPECT_EQ(s.cache->numLevels(), 3);

    auto shape0 = s.cache->levelShape(0);
    EXPECT_EQ(shape0[0], 64);
    EXPECT_EQ(shape0[1], 128);
    EXPECT_EQ(shape0[2], 128);

    auto shape1 = s.cache->levelShape(1);
    EXPECT_EQ(shape1[0], 32);

    auto chunk = s.cache->chunkShape(0);
    EXPECT_EQ(chunk[0], 2);
    EXPECT_EQ(chunk[1], 4);
    EXPECT_EQ(chunk[2], 4);
}

TEST(TieredChunkCache, DataBounds)
{
    CacheSetup s("test_bounds");

    auto b = s.cache->dataBounds();
    EXPECT_FALSE(b.valid);

    s.cache->setDataBounds(10, 200, 20, 300, 5, 100);
    b = s.cache->dataBounds();
    EXPECT_TRUE(b.valid);
    EXPECT_EQ(b.minX, 10);
    EXPECT_EQ(b.maxX, 200);
    EXPECT_EQ(b.minZ, 5);
    EXPECT_EQ(b.maxZ, 100);
}

TEST(TieredChunkCache, StatsTracking)
{
    CacheSetup s("test_stats");
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0x42));

    // Miss
    (void)s.cache->get({0, 0, 0, 0});
    auto st = s.cache->stats();
    EXPECT_EQ(st.misses, 1u);

    // Blocking load (ice fetch)
    (void)s.cache->getBlocking({0, 0, 0, 0});
    st = s.cache->stats();
    EXPECT_EQ(st.iceFetches, 1u);

    // Hot hit
    (void)s.cache->get({0, 0, 0, 0});
    st = s.cache->stats();
    EXPECT_GT(st.hotHits, 0u);
}

TEST(TieredChunkCache, DiskStoreIntegration)
{
    CacheSetup s("test_disk", 3, {2, 4, 4}, 1 << 20, 1 << 20, 2, true);
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0x77));

    // Load via blocking (promotes through all tiers)
    auto r = s.cache->getBlocking({0, 0, 0, 0});
    ASSERT_TRUE(r != nullptr);
    EXPECT_EQ(r->bytes[0], 0x77);

    // Clear memory tiers — data should still be on disk
    s.cache->clearMemory();
    EXPECT_TRUE(s.cache->get({0, 0, 0, 0}) == nullptr);

    // Blocking load should find it on disk (cold hit, not ice)
    int fetchBefore = s.source->fetchCount();
    auto r2 = s.cache->getBlocking({0, 0, 0, 0});
    ASSERT_TRUE(r2 != nullptr);
    EXPECT_EQ(r2->bytes[0], 0x77);
    // Should NOT have fetched from source again (cold hit)
    EXPECT_EQ(s.source->fetchCount(), fetchBefore);

    auto st = s.cache->stats();
    EXPECT_GT(st.coldHits, 0u);
}

TEST(TieredChunkCache, AreAllCachedInRegion)
{
    CacheSetup s("test_all_cached");

    // Empty cache — not all cached
    EXPECT_FALSE(s.cache->areAllCachedInRegion(0, 0, 0, 0, 0, 0, 1));

    // Load one of two chunks
    s.source->setChunk({0, 0, 0, 0}, makeChunkBytes(32, 0x01));
    (void)s.cache->getBlocking({0, 0, 0, 0});

    // Still not all (missing 0,0,1)
    EXPECT_FALSE(s.cache->areAllCachedInRegion(0, 0, 0, 0, 0, 0, 1));

    // The missing chunk is negative-cached (source returns empty)
    (void)s.cache->getBlocking({0, 0, 0, 1});  // will negative-cache
    // Now all are "cached" (one real, one negative)
    EXPECT_TRUE(s.cache->areAllCachedInRegion(0, 0, 0, 0, 0, 0, 1));
}

TEST(TieredChunkCache, CancelPendingPrefetch)
{
    CacheSetup s("test_cancel", 1, {2, 4, 4}, 1 << 20, 1 << 20, 1);

    // Set up many chunks
    for (int i = 0; i < 100; i++) {
        s.source->setChunk({0, 0, 0, i}, makeChunkBytes(32, static_cast<uint8_t>(i)));
    }

    // Prefetch many
    for (int i = 0; i < 100; i++) {
        s.cache->prefetch({0, 0, 0, i});
    }

    // Cancel immediately — some may have started
    s.cache->cancelPendingPrefetch();

    // Not all should be loaded (some were cancelled)
    // This is a best-effort check since timing is unpredictable
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    int loaded = 0;
    for (int i = 0; i < 100; i++) {
        if (s.cache->get({0, 0, 0, i})) loaded++;
    }
    // At least some should NOT have been loaded (cancelled)
    // But some may have completed before cancel
    EXPECT_LT(loaded, 100);
}

TEST(TieredChunkCache, WarmEviction)
{
    // Warm max = 256 bytes, each chunk compressed = 32 bytes, ~8 fit
    CacheSetup s("test_warm_evict", 1, {2, 4, 4}, 1 << 20, 256);

    for (int i = 0; i < 16; i++) {
        s.source->setChunk({0, 0, 0, i}, makeChunkBytes(32, static_cast<uint8_t>(i)));
        (void)s.cache->getBlocking({0, 0, 0, i});
    }

    auto stats = s.cache->stats();
    EXPECT_GT(stats.warmEvictions, 0u);
    EXPECT_LE(stats.warmBytes, 256u);
}
