#include "RenderPool.hpp"

#include <QDebug>
#include <QThreadPool>

#include <omp.h>

#include "vc/core/util/Surface.hpp"
#include "vc/core/types/Volume.hpp"

// ============================================================================
// TileRenderTask
// ============================================================================

TileRenderTask::TileRenderTask(TileRenderParams params,
                               std::shared_ptr<Surface> surface,
                               Volume* volume,
                               RenderPool* pool)
    : _params(std::move(params))
    , _surface(std::move(surface))
    , _volume(volume)
    , _pool(pool)
{
    setAutoDelete(true);
}

void TileRenderTask::run()
{
    // Skip if stale before rendering
    if (_params.epoch < _pool->currentEpoch()) {
        _pool->_pendingCount.fetch_sub(1, std::memory_order_relaxed);
        return;
    }

    // Disable OMP parallelism for tile rendering — tiles are already parallelized
    // at the task level by the RenderPool.  Without this, each tile render spawns
    // ~N_cores OMP threads that busy-wait (spin) after the parallel region ends,
    // causing 100% CPU usage at idle.
    omp_set_num_threads(1);

    TileRenderResult result = TileRenderer::renderTile(_params, _surface, _volume);

    // Skip if stale after rendering
    if (_params.epoch < _pool->currentEpoch()) {
        _pool->_pendingCount.fetch_sub(1, std::memory_order_relaxed);
        return;
    }

    _pool->pushResult(std::move(result));
}

// ============================================================================
// RenderPool
// ============================================================================

RenderPool::RenderPool(int numThreads, QObject* parent)
    : QObject(parent)
{
    _pool = new QThreadPool(this);
    _pool->setMaxThreadCount(numThreads);
}

RenderPool::~RenderPool()
{
    cancelAll();
}

void RenderPool::submit(const TileRenderParams& params,
                        const std::shared_ptr<Surface>& surface,
                        Volume* volume)
{
    _pendingCount.fetch_add(1, std::memory_order_relaxed);
    auto* task = new TileRenderTask(params, surface, volume, this);
    _pool->start(task);
}

std::vector<TileRenderResult> RenderPool::drainCompleted(int maxResults, uint64_t minEpoch)
{
    std::vector<TileRenderResult> results;
    std::lock_guard<std::mutex> lock(_resultsMutex);

    results.reserve(std::min(static_cast<int>(_completedResults.size()), maxResults));

    auto it = _completedResults.begin();
    while (it != _completedResults.end() && static_cast<int>(results.size()) < maxResults) {
        if (it->epoch >= minEpoch) {
            results.push_back(std::move(*it));
        }
        // Stale results (epoch < minEpoch) are silently discarded
        it = _completedResults.erase(it);
    }

    return results;
}

void RenderPool::cancelAll()
{
    _pool->clear();  // Remove pending tasks from the queue
    _pool->waitForDone();  // Wait for in-flight tasks to complete

    std::lock_guard<std::mutex> lock(_resultsMutex);
    _completedResults.clear();
    _pendingCount.store(0, std::memory_order_relaxed);
}

void RenderPool::setCurrentEpoch(uint64_t epoch)
{
    _currentEpoch.store(epoch, std::memory_order_relaxed);
}

uint64_t RenderPool::currentEpoch() const
{
    return _currentEpoch.load(std::memory_order_relaxed);
}

int RenderPool::pendingCount() const
{
    return _pendingCount.load(std::memory_order_relaxed);
}

void RenderPool::pushResult(TileRenderResult result)
{
    {
        std::lock_guard<std::mutex> lock(_resultsMutex);
        _completedResults.push_back(std::move(result));
    }
    _pendingCount.fetch_sub(1, std::memory_order_relaxed);
    emit tileReady();
}
