#include "VolumePrefetcher.hpp"

#include <algorithm>

#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/util/Logging.hpp"

void VolumePrefetcher::start(std::shared_ptr<Volume> volume, std::vector<int> levels,
                            int threads)
{
    stop();
    if (!volume)
        return;
    const int nThreads = std::max(1, threads);

    std::shared_ptr<vc::render::ChunkCache> cache;
    try {
        cache = volume->chunkedCacheShared();
    } catch (const std::exception& e) {
        Logger()->warn("prefetch: no chunk cache for {}: {}", volume->id(), e.what());
        return;
    }
    if (!cache || cache->chunkShape(0)[0] != 16) {
        Logger()->info("prefetch: mca cache not engaged for {}; skipping", volume->id());
        return;
    }

    const int numLevels = cache->numLevels();
    if (levels.empty())
        for (int l = numLevels - 1; l >= 0; --l)
            levels.push_back(l);

    // region grids per level (in the configured order) + cumulative offsets:
    // a worker maps its global index to (level, rz, ry, rx) arithmetically, so
    // even an all-levels run over a huge volume needs no materialized key list.
    constexpr int kRegion = 256, kBlk = 16;
    struct LevelGrid { int level, rz, ry, rx; std::size_t begin; };
    auto grids = std::make_shared<std::vector<LevelGrid>>();
    std::size_t total = 0;
    for (const int level : levels) {
        if (level < 0 || level >= numLevels) {
            Logger()->warn("prefetch: level {} out of range (volume has {})", level, numLevels);
            continue;
        }
        const auto shape = cache->shape(level);
        const LevelGrid g{level, (shape[0] + kRegion - 1) / kRegion,
                          (shape[1] + kRegion - 1) / kRegion,
                          (shape[2] + kRegion - 1) / kRegion, total};
        total += std::size_t(g.rz) * g.ry * g.rx;
        grids->push_back(g);
    }
    if (total == 0)
        return;

    Logger()->info("prefetch: {} regions across {} level(s) for {} ({} threads)",
                   total, grids->size(), volume->id(), nThreads);

    stopFlag_ = std::make_shared<std::atomic<bool>>(false);
    auto next = std::make_shared<std::atomic<std::size_t>>(0);
    auto stopFlag = stopFlag_;

    threads_.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        threads_.emplace_back([cache, grids, total, next, stopFlag, volume] {
            for (;;) {
                if (stopFlag->load(std::memory_order_relaxed))
                    return;
                const std::size_t i = next->fetch_add(1, std::memory_order_relaxed);
                if (i >= total) {
                    if (i == total)
                        Logger()->info("prefetch: complete for {}", volume->id());
                    return;
                }
                // find the level grid containing i (few levels; linear scan is fine)
                const LevelGrid* g = &grids->back();
                for (const auto& cand : *grids)
                    if (i >= cand.begin)
                        g = &cand;
                    else
                        break;
                std::size_t rest = i - g->begin;
                const int x = static_cast<int>(rest % g->rx);
                rest /= g->rx;
                const int y = static_cast<int>(rest % g->ry);
                const int z = static_cast<int>(rest / g->ry);
                cache->warmChunkBlocking(g->level, z * kBlk, y * kBlk, x * kBlk);
            }
        });
    }
}

void VolumePrefetcher::stop()
{
    if (stopFlag_)
        stopFlag_->store(true, std::memory_order_relaxed);
    threads_.clear();   // jthread joins on destruction
    stopFlag_.reset();
}
