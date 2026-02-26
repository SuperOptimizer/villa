#include "test.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

#include "vc/core/cache/DiskStore.hpp"
#include "vc/core/cache/IOPool.hpp"

namespace fs = std::filesystem;

// ---- DiskStore --------------------------------------------------------------

static fs::path makeTempDir(const std::string& name)
{
    auto p = fs::temp_directory_path() / ("vc_test_" + name);
    fs::remove_all(p);
    return p;
}

TEST(DiskStore, PutGetRoundTrip)
{
    auto dir = makeTempDir("diskstore_roundtrip");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF, 0x42};
    vc::cache::ChunkKey key{0, 1, 2, 3};

    store.put("vol1", key, data);
    auto result = store.get("vol1", key);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ((*result)[i], data[i]);
    }
}

TEST(DiskStore, GetMissReturnsNullopt)
{
    auto dir = makeTempDir("diskstore_miss");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    auto result = store.get("vol1", {0, 99, 99, 99});
    EXPECT_FALSE(result.has_value());
}

TEST(DiskStore, TotalBytesTracked)
{
    auto dir = makeTempDir("diskstore_totalbytes");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    EXPECT_EQ(store.totalBytes(), 0u);

    std::vector<uint8_t> data(1000, 0xAA);
    store.put("vol1", {0, 0, 0, 0}, data);
    EXPECT_EQ(store.totalBytes(), 1000u);

    store.put("vol1", {0, 0, 0, 1}, data);
    EXPECT_EQ(store.totalBytes(), 2000u);
}

TEST(DiskStore, RemoveUpdatesTotal)
{
    auto dir = makeTempDir("diskstore_remove");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    std::vector<uint8_t> data(500, 0xBB);
    store.put("vol1", {0, 0, 0, 0}, data);
    EXPECT_EQ(store.totalBytes(), 500u);

    store.remove("vol1", {0, 0, 0, 0});
    EXPECT_EQ(store.totalBytes(), 0u);

    // Verify file is actually gone
    auto result = store.get("vol1", {0, 0, 0, 0});
    EXPECT_FALSE(result.has_value());
}

TEST(DiskStore, RemoveNonexistentIsNoOp)
{
    auto dir = makeTempDir("diskstore_remove_noop");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    // Should not crash or change total
    store.remove("vol1", {0, 0, 0, 0});
    EXPECT_EQ(store.totalBytes(), 0u);
}

TEST(DiskStore, MultipleVolumes)
{
    auto dir = makeTempDir("diskstore_multivol");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    std::vector<uint8_t> data1 = {1, 2, 3};
    std::vector<uint8_t> data2 = {4, 5, 6, 7};

    store.put("volA", {0, 0, 0, 0}, data1);
    store.put("volB", {0, 0, 0, 0}, data2);

    auto r1 = store.get("volA", {0, 0, 0, 0});
    auto r2 = store.get("volB", {0, 0, 0, 0});
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r1->size(), 3u);
    EXPECT_EQ(r2->size(), 4u);

    // Cross-volume miss
    auto r3 = store.get("volA", {1, 0, 0, 0});
    EXPECT_FALSE(r3.has_value());
}

TEST(DiskStore, ClearVolume)
{
    auto dir = makeTempDir("diskstore_clearvol");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    std::vector<uint8_t> data(100, 0xCC);
    store.put("volA", {0, 0, 0, 0}, data);
    store.put("volB", {0, 0, 0, 0}, data);
    EXPECT_EQ(store.totalBytes(), 200u);

    store.clearVolume("volA");
    EXPECT_EQ(store.totalBytes(), 100u);

    EXPECT_FALSE(store.get("volA", {0, 0, 0, 0}).has_value());
    EXPECT_TRUE(store.get("volB", {0, 0, 0, 0}).has_value());
}

TEST(DiskStore, ClearAll)
{
    auto dir = makeTempDir("diskstore_clearall");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    std::vector<uint8_t> data(100, 0xDD);
    store.put("vol1", {0, 0, 0, 0}, data);
    store.put("vol2", {0, 0, 0, 0}, data);
    EXPECT_GT(store.totalBytes(), 0u);

    store.clearAll();
    EXPECT_EQ(store.totalBytes(), 0u);
}

TEST(DiskStore, EvictToSize)
{
    auto dir = makeTempDir("diskstore_evict");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    // Write 5 chunks of 100 bytes each. We just need to verify that
    // eviction brings total under target — ordering doesn't matter.
    for (int i = 0; i < 5; ++i) {
        std::vector<uint8_t> data(100, static_cast<uint8_t>(i));
        store.put("vol1", {0, 0, 0, i}, data);
    }
    EXPECT_EQ(store.totalBytes(), 500u);

    // Evict to 250 bytes — should remove enough to fit
    store.evictToSize(250);
    EXPECT_LE(store.totalBytes(), 250u);
}

TEST(DiskStore, PersistenceAcrossInstances)
{
    auto dir = makeTempDir("diskstore_persist");

    std::vector<uint8_t> data = {10, 20, 30};
    vc::cache::ChunkKey key{0, 0, 0, 0};

    // Write with first instance (persistent = true)
    {
        vc::cache::DiskStore store({.root = dir, .persistent = true});
        store.put("vol1", key, data);
    }

    // Read with second instance — data should still be there
    {
        vc::cache::DiskStore store({.root = dir, .persistent = true});
        auto result = store.get("vol1", key);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->size(), 3u);
        EXPECT_EQ((*result)[0], 10);
        // totalBytes should reflect existing files
        EXPECT_GT(store.totalBytes(), 0u);
    }

    // Cleanup
    fs::remove_all(dir);
}

TEST(DiskStore, NonPersistentCleansUpOnDestruction)
{
    auto dir = makeTempDir("diskstore_nonpersist");

    {
        vc::cache::DiskStore store({.root = dir, .persistent = false});
        store.put("vol1", {0, 0, 0, 0}, std::vector<uint8_t>(100, 0));
        EXPECT_TRUE(fs::exists(dir));
    }

    // After destruction, directory should be gone
    EXPECT_FALSE(fs::exists(dir));
}

TEST(DiskStore, DirectMode)
{
    auto dir = makeTempDir("diskstore_direct");
    vc::cache::DiskStore store({.root = dir, .persistent = false, .directMode = true});

    std::vector<uint8_t> data = {1, 2, 3};
    store.put("ignored_vol_id", {0, 5, 6, 7}, data);

    auto result = store.get("ignored_vol_id", {0, 5, 6, 7});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 3u);
}

TEST(DiskStore, MultipleLevels)
{
    auto dir = makeTempDir("diskstore_levels");
    vc::cache::DiskStore store({.root = dir, .persistent = false});

    std::vector<uint8_t> data0 = {0};
    std::vector<uint8_t> data1 = {1, 1};
    std::vector<uint8_t> data2 = {2, 2, 2};

    store.put("vol", {0, 0, 0, 0}, data0);
    store.put("vol", {1, 0, 0, 0}, data1);
    store.put("vol", {2, 0, 0, 0}, data2);

    auto r0 = store.get("vol", {0, 0, 0, 0});
    auto r1 = store.get("vol", {1, 0, 0, 0});
    auto r2 = store.get("vol", {2, 0, 0, 0});
    ASSERT_TRUE(r0.has_value());
    ASSERT_TRUE(r1.has_value());
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(r0->size(), 1u);
    EXPECT_EQ(r1->size(), 2u);
    EXPECT_EQ(r2->size(), 3u);
}

// ---- IOPool -----------------------------------------------------------------

// Helper: spin-wait on an atomic with 1ms polls, up to maxMs total.
static void waitFor(const std::atomic<int>& val, int target, int maxMs = 500)
{
    for (int i = 0; i < maxMs && val.load() < target; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

TEST(IOPool, SubmitAndComplete)
{
    vc::cache::IOPool pool(2);
    std::atomic<int> completed{0};

    pool.setFetchFunc([](const vc::cache::ChunkKey&) {
        return std::vector<uint8_t>{1, 2, 3};
    });
    pool.setCompletionCallback([&](const vc::cache::ChunkKey& key,
                                   std::vector<uint8_t>&& data) {
        EXPECT_EQ(data.size(), 3u);
        completed.fetch_add(1);
    });

    pool.submit({0, 0, 0, 0});
    waitFor(completed, 1);
    EXPECT_EQ(completed.load(), 1);
    pool.stop();
}

TEST(IOPool, Deduplication)
{
    // Dedup is tested at the queue level: submit same key 3 times before
    // the worker can even pick it up. Only 1 fetch should happen.
    vc::cache::IOPool pool(1);
    std::atomic<int> fetchCount{0};
    std::atomic<int> started{0};
    std::atomic<bool> gate{false};

    pool.setFetchFunc([&](const vc::cache::ChunkKey&) {
        started.fetch_add(1);
        while (!gate.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        fetchCount.fetch_add(1);
        return std::vector<uint8_t>{1};
    });
    pool.setCompletionCallback([](const vc::cache::ChunkKey&, std::vector<uint8_t>&&) {});

    vc::cache::ChunkKey key{0, 0, 0, 0};
    pool.submit(key);

    // Wait for first fetch to start (it's now blocking on gate)
    waitFor(started, 1);

    // These should be deduped since key is in-flight
    pool.submit(key);
    pool.submit(key);

    gate.store(true);
    waitFor(fetchCount, 1);
    // Give time for any erroneous extra fetches
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(fetchCount.load(), 1);
    pool.stop();
}

TEST(IOPool, BatchSubmit)
{
    vc::cache::IOPool pool(2);
    std::atomic<int> completed{0};

    pool.setFetchFunc([](const vc::cache::ChunkKey&) {
        return std::vector<uint8_t>{42};
    });
    pool.setCompletionCallback([&](const vc::cache::ChunkKey&, std::vector<uint8_t>&&) {
        completed.fetch_add(1);
    });

    std::vector<vc::cache::ChunkKey> keys = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 1, 0}
    };
    pool.submit(keys);
    waitFor(completed, 4);
    EXPECT_EQ(completed.load(), 4);
    pool.stop();
}

TEST(IOPool, CancelPending)
{
    vc::cache::IOPool pool(1);
    std::atomic<int> fetchCount{0};
    std::atomic<int> started{0};
    std::atomic<bool> gate{false};

    pool.setFetchFunc([&](const vc::cache::ChunkKey&) {
        started.fetch_add(1);
        while (!gate.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        fetchCount.fetch_add(1);
        return std::vector<uint8_t>{};
    });
    pool.setCompletionCallback([](const vc::cache::ChunkKey&, std::vector<uint8_t>&&) {});

    // Submit many tasks — 1 thread, so most will queue
    for (int i = 0; i < 10; ++i)
        pool.submit({0, 0, 0, i});

    // Wait for one to start fetching
    waitFor(started, 1);

    // Cancel the queued ones
    pool.cancelPending();

    // Let the in-flight one finish
    gate.store(true);
    waitFor(fetchCount, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    pool.stop();
    EXPECT_EQ(fetchCount.load(), 1);
}

TEST(IOPool, PriorityCoarseFirst)
{
    vc::cache::IOPool pool(1);
    std::vector<int> completionOrder;
    std::mutex orderMtx;
    std::atomic<int> started{0};
    std::atomic<bool> gate{false};

    pool.setFetchFunc([&](const vc::cache::ChunkKey& key) {
        if (started.fetch_add(1) == 0) {
            // Block first task so others queue up
            while (!gate.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return std::vector<uint8_t>{};
    });
    pool.setCompletionCallback([&](const vc::cache::ChunkKey& key, std::vector<uint8_t>&&) {
        std::lock_guard lock(orderMtx);
        completionOrder.push_back(key.level);
    });

    pool.submit({0, 0, 0, 0});  // blocker
    waitFor(started, 1);

    // Queue tasks at levels 1, 2, 3 while worker is blocked
    pool.submit({1, 0, 0, 0});
    pool.submit({2, 0, 0, 0});
    pool.submit({3, 0, 0, 0});

    gate.store(true);

    // Wait for all 4
    for (int i = 0; i < 500; ++i) {
        {
            std::lock_guard lock(orderMtx);
            if (completionOrder.size() >= 4) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    pool.stop();

    std::lock_guard lock(orderMtx);
    ASSERT_EQ(completionOrder.size(), 4u);
    EXPECT_EQ(completionOrder[0], 0);  // was in-flight
    EXPECT_EQ(completionOrder[1], 3);  // coarsest queued
    EXPECT_EQ(completionOrder[2], 2);
    EXPECT_EQ(completionOrder[3], 1);  // finest queued
}

TEST(IOPool, FetchExceptionHandled)
{
    vc::cache::IOPool pool(1);
    std::atomic<int> completions{0};

    pool.setFetchFunc([](const vc::cache::ChunkKey& key) -> std::vector<uint8_t> {
        if (key.ix == 0) throw std::runtime_error("test error");
        return {42};
    });
    pool.setCompletionCallback([&](const vc::cache::ChunkKey&, std::vector<uint8_t>&&) {
        completions.fetch_add(1);
    });

    pool.submit({0, 0, 0, 0});  // will throw
    pool.submit({0, 0, 0, 1});  // will succeed

    waitFor(completions, 1);
    pool.stop();
    // Exception in first task shouldn't prevent second from completing
    EXPECT_GE(completions.load(), 1);
}

TEST(IOPool, StopWithPendingTasks)
{
    vc::cache::IOPool pool(1);
    std::atomic<int> started{0};

    pool.setFetchFunc([&](const vc::cache::ChunkKey&) {
        started.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return std::vector<uint8_t>{};
    });
    pool.setCompletionCallback([](const vc::cache::ChunkKey&, std::vector<uint8_t>&&) {});

    for (int i = 0; i < 100; ++i)
        pool.submit({0, 0, 0, i});

    // Wait for at least one to start, then stop
    waitFor(started, 1);
    pool.stop();
    // If we get here, stop() didn't deadlock
    EXPECT_TRUE(true);
}

TEST(IOPool, ResubmitAfterCompletion)
{
    vc::cache::IOPool pool(1);
    std::atomic<int> fetchCount{0};

    pool.setFetchFunc([&](const vc::cache::ChunkKey&) {
        fetchCount.fetch_add(1);
        return std::vector<uint8_t>{1};
    });
    pool.setCompletionCallback([](const vc::cache::ChunkKey&, std::vector<uint8_t>&&) {});

    pool.submit({0, 0, 0, 0});
    waitFor(fetchCount, 1);
    EXPECT_EQ(fetchCount.load(), 1);

    // Re-submit same key — should fetch again (not deduped against completed)
    pool.submit({0, 0, 0, 0});
    waitFor(fetchCount, 2);
    EXPECT_EQ(fetchCount.load(), 2);
    pool.stop();
}
