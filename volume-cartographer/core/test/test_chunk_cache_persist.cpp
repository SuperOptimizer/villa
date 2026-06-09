// More ChunkCache coverage: listener invocation order, prefetch w/ no wait,
// concurrent request coalescing.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using vc::render::ChunkCache;
using vc::render::ChunkDtype;
using vc::render::ChunkFetchResult;
using vc::render::ChunkFetchStatus;
using vc::render::ChunkKey;
using vc::render::ChunkResult;
using vc::render::ChunkStatus;
using vc::render::IChunkFetcher;

namespace {

class CountingFetcher : public IChunkFetcher {
public:
    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ++fetchCalls;
        std::lock_guard<std::mutex> lk(m_);
        auto it = canned_.find(key);
        if (it != canned_.end()) return it->second;
        ChunkFetchResult r;
        r.status = ChunkFetchStatus::Missing;
        return r;
    }
    void setCanned(const ChunkKey& k, ChunkFetchResult r)
    {
        std::lock_guard<std::mutex> lk(m_);
        canned_[k] = std::move(r);
    }
    std::atomic<int> fetchCalls{0};
private:
    std::mutex m_;
    std::unordered_map<ChunkKey, ChunkFetchResult, vc::render::ChunkKeyHash> canned_;
};

std::shared_ptr<ChunkCache> makeCache(std::shared_ptr<CountingFetcher> f)
{
    std::vector<ChunkCache::LevelInfo> levels = {{{8, 8, 8}, {4, 4, 4}, {}}};
    ChunkCache::Options opts;
    opts.maxConcurrentReads = 4;
    opts.detectAllFillChunks = true;
    return std::make_shared<ChunkCache>(
        std::move(levels),
        std::vector<std::shared_ptr<IChunkFetcher>>{f},
        0.0, ChunkDtype::UInt8, opts);
}

ChunkResult waitForResolved(ChunkCache& c, int level, int iz, int iy, int ix,
                            std::chrono::milliseconds timeout = std::chrono::seconds{2})
{
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        auto r = c.tryGetChunk(level, iz, iy, ix);
        if (r.status != ChunkStatus::MissQueued) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return c.tryGetChunk(level, iz, iy, ix);
}

} // namespace

TEST_CASE("prefetchChunks(wait=false): non-blocking; later tryGetChunk picks it up")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{99});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    std::vector<ChunkKey> keys = {{0, 0, 0, 0}};
    c->prefetchChunks(keys, /*wait=*/false, /*priorityOffset=*/0);
    // Don't assert immediate state — just wait for resolved.
    auto r = waitForResolved(*c, 0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);
}

TEST_CASE("prefetchChunks with negative priority offset still resolves")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{200});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);
    std::vector<ChunkKey> keys = {{0, 0, 0, 0}};
    c->prefetchChunks(keys, /*wait=*/true, /*priorityOffset=*/-5);
    auto r = c->tryGetChunk(0, 0, 0, 0);
    CHECK(r.status == ChunkStatus::Data);
}

TEST_CASE("multiple listeners are all notified")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{1});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);

    std::atomic<int> a{0}, b{0};
    auto idA = c->addChunkReadyListener([&]() { ++a; });
    auto idB = c->addChunkReadyListener([&]() { ++b; });
    (void)waitForResolved(*c, 0, 0, 0, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(a.load() >= 1);
    CHECK(b.load() >= 1);
    c->removeChunkReadyListener(idA);
    c->removeChunkReadyListener(idB);
}

TEST_CASE("Many concurrent tryGetChunk calls converge on the same Entry")
{
    auto f = std::make_shared<CountingFetcher>();
    ChunkFetchResult fr;
    fr.status = ChunkFetchStatus::Found;
    fr.bytes = std::vector<std::byte>(64, std::byte{50});
    f->setCanned({0, 0, 0, 0}, fr);
    auto c = makeCache(f);

    std::vector<std::thread> threads;
    std::atomic<int> success{0};
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 20; ++j) {
                auto r = c->tryGetChunk(0, 0, 0, 0);
                if (r.status == ChunkStatus::Data) ++success;
            }
        });
    }
    for (auto& t : threads) t.join();
    // The fetcher should have been called at most a small number of times
    // (cache coalesces in-flight requests).
    CHECK(f->fetchCalls.load() <= 4);
}
