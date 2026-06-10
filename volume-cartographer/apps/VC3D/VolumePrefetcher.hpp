#pragma once

// VolumePrefetcher — background population of the current volume's mca cache,
// the GUI counterpart of vc_cache_prefetch. Walks every 256^3 region of the
// configured LOD levels in order on a small DEDICATED thread team, pulling each
// through ChunkCache::warmChunkBlocking — the shared interactive fetch pool is
// never occupied, so navigation latency is unaffected.

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

class Volume;

class VolumePrefetcher final {
public:
    VolumePrefetcher() = default;
    ~VolumePrefetcher() { stop(); }
    VolumePrefetcher(const VolumePrefetcher&) = delete;
    VolumePrefetcher& operator=(const VolumePrefetcher&) = delete;

    // Start prefetching `volume` at `levels` (in order; empty = all levels,
    // coarsest first) on `threads` dedicated workers (clamped to >=1). Stops any
    // previous run first. No-op if the volume has no persistent mca cache.
    void start(std::shared_ptr<Volume> volume, std::vector<int> levels, int threads);

    // Signal and join the worker team.
    void stop();

    bool running() const { return !threads_.empty(); }

private:
    std::shared_ptr<std::atomic<bool>> stopFlag_;
    std::vector<std::jthread> threads_;
};
