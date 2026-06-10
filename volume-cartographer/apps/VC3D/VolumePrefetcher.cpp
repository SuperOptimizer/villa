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

    // Work unit = one SHARD's worth of regions (a zarr shard is the source's
    // natural batch; a sequential sweep warms it in one GET). Probe each level's
    // batch edge from the first region; tile the level into batch-sized cells and
    // claim cells atomically. A worker warms its cell's batch once, then warms
    // every region inside it from the warm buffer.
    constexpr int kRegion = 256, kBlk = 16;
    struct LevelGrid { int level, rz, ry, rx, batchRegions; std::size_t begin; };
    auto grids = std::make_shared<std::vector<LevelGrid>>();
    std::size_t totalCells = 0;
    for (const int level : levels) {
        if (level < 0 || level >= numLevels) {
            Logger()->warn("prefetch: level {} out of range (volume has {})", level, numLevels);
            continue;
        }
        const auto shape = cache->shape(level);
        const int rz = (shape[0] + kRegion - 1) / kRegion;
        const int ry = (shape[1] + kRegion - 1) / kRegion;
        const int rx = (shape[2] + kRegion - 1) / kRegion;
        // shard edge in regions (block units / kBlk). One work cell = one shard.
        const auto fb = cache->shardBatch(level, 0, 0, 0);
        const int batchRegions = std::max(1, fb.edgeChunks / kBlk);
        const int cz = (rz + batchRegions - 1) / batchRegions;
        const int cy = (ry + batchRegions - 1) / batchRegions;
        const int cx = (rx + batchRegions - 1) / batchRegions;
        grids->push_back({level, rz, ry, rx, batchRegions, totalCells});
        totalCells += std::size_t(cz) * cy * cx;
    }
    if (totalCells == 0)
        return;

    Logger()->info("prefetch: {} shard-cells across {} level(s) for {} ({} threads)",
                   totalCells, grids->size(), volume->id(), nThreads);

    stopFlag_ = std::make_shared<std::atomic<bool>>(false);
    auto next = std::make_shared<std::atomic<std::size_t>>(0);
    auto stopFlag = stopFlag_;

    threads_.reserve(nThreads);
    for (int t = 0; t < nThreads; ++t) {
        threads_.emplace_back([cache, grids, totalCells, next, stopFlag, volume] {
            for (;;) {
                if (stopFlag->load(std::memory_order_relaxed))
                    return;
                const std::size_t i = next->fetch_add(1, std::memory_order_relaxed);
                if (i >= totalCells) {
                    if (i == totalCells)
                        Logger()->info("prefetch: complete for {}", volume->id());
                    return;
                }
                const LevelGrid* g = &grids->back();
                for (const auto& cand : *grids)
                    if (i >= cand.begin) g = &cand; else break;

                const int b = g->batchRegions;
                const int ccx = (g->rx + b - 1) / b;
                const int ccy = (g->ry + b - 1) / b;
                std::size_t rest = i - g->begin;
                const int cx = static_cast<int>(rest % ccx); rest /= ccx;
                const int cy = static_cast<int>(rest % ccy);
                const int cz = static_cast<int>(rest / ccy);
                // region origin of this shard cell -> one parallel shard download
                // + encode of all its regions, on this thread.
                const int rz0 = cz * b, ry0 = cy * b, rx0 = cx * b;
                cache->prefetchShardBlocking(g->level, rz0 * kBlk, ry0 * kBlk, rx0 * kBlk);
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
