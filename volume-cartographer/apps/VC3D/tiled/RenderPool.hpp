#pragma once

#include <QObject>
#include <QRunnable>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "TileRenderer.hpp"

class QThreadPool;
class Surface;
class Volume;

// A single render task submitted to the pool.
// Inherits QRunnable so QThreadPool can execute it.
class TileRenderTask : public QRunnable
{
public:
    TileRenderTask(TileRenderParams params,
                   std::shared_ptr<Surface> surface,
                   std::shared_ptr<Volume> volume,
                   class RenderPool* pool);

    void run() override;

private:
    TileRenderParams _params;
    std::shared_ptr<Surface> _surface;
    std::shared_ptr<Volume> _volume;
    RenderPool* _pool;
};

// Background tile rendering pool.
// Workers render tiles off the main thread. Completed results are
// collected by the main thread via drainCompleted().
class RenderPool : public QObject
{
    Q_OBJECT

public:
    explicit RenderPool(int numThreads = 2, QObject* parent = nullptr);
    ~RenderPool() override;

    // Submit a tile for background rendering.
    void submit(const TileRenderParams& params,
                const std::shared_ptr<Surface>& surface,
                const std::shared_ptr<Volume>& volume);

    // Take up to maxResults completed results (main thread).
    // Results with epoch < minEpoch are discarded.
    std::vector<TileRenderResult> drainCompleted(int maxResults, uint64_t minEpoch);

    // Cancel all pending work and clear results.
    void cancelAll();

    // Epoch management: workers check this to skip stale renders.
    void setCurrentEpoch(uint64_t epoch);
    uint64_t currentEpoch() const;

    // Number of pending + in-flight tasks
    int pendingCount() const;

signals:
    // Emitted (from worker thread) when a result is ready.
    // Connect with Qt::QueuedConnection to receive on main thread.
    void tileReady();

private:
    friend class TileRenderTask;

    // Called by TileRenderTask::run() from a worker thread.
    void pushResult(TileRenderResult result);

    QThreadPool* _pool;
    std::mutex _resultsMutex;
    std::vector<TileRenderResult> _completedResults;
    std::atomic<int> _pendingCount{0};
    std::atomic<uint64_t> _currentEpoch{0};
};
