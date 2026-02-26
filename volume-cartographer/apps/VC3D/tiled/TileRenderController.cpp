#include "TileRenderController.hpp"

#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QImage>
#include <algorithm>
#include <vector>

#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

TileRenderController::TileRenderController(TileScene* tileScene, QObject* parent)
    : QObject(parent)
    , _tileScene(tileScene)
    , _renderPool(std::max(2, QThread::idealThreadCount()), this)
{
    // Tick timer (~30 Hz) handles periodic work; started on-demand, auto-stops
    // when idle to avoid burning CPU.
    _tickTimer = new QTimer(this);
    _tickTimer->setInterval(33);
    connect(_tickTimer, &QTimer::timeout, this, &TileRenderController::tick);
    // Not started here — ensureTickRunning() starts it when work arrives.

    // Also drain immediately when a tile completes (makes draining more responsive)
    connect(&_renderPool, &RenderPool::tileReady, this, &TileRenderController::drainResults,
            Qt::QueuedConnection);
}

TileRenderController::~TileRenderController()
{
    _tickTimer->stop();
    _renderPool.cancelAll();
}

void TileRenderController::ensureTickRunning()
{
    if (!_tickTimer->isActive())
        _tickTimer->start();
}

void TileRenderController::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    ensureTickRunning();
    bool epochChanged = (camera.epoch != _currentEpoch);
    _currentEpoch = camera.epoch;
    _renderPool.setCurrentEpoch(_currentEpoch);
    _desiredLevel = camera.dsScaleIdx;

    if (epochChanged && volume) {
        volume->cancelPendingPrefetch();
        volume->tieredCache()->setIOEpoch(camera.epoch);
    }

    // Store last render state for refinement retries
    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;
    _lastViewportRect = viewportRect;

    // Get visible tiles (+ buffer for smooth scrolling)
    auto visibleKeys = _tileScene->visibleTiles(viewportRect, tiled_config::VISIBLE_BUFFER_TILES);

    for (const auto& wk : visibleKeys) {
        SliceCacheKey cacheKey = SliceCacheKey::make(wk, camera, _paramsHash);

        // Check slice cache — apply best available immediately
        auto lookup = _cache.getBest(cacheKey);
        if (lookup.level >= 0) {
            _tileScene->setTileWorld(wk, lookup.pixmap, _currentEpoch, lookup.level);
            if (lookup.level == camera.dsScaleIdx) {
                continue;  // exact hit, no need to re-render
            }
        }

        // Submit to background pool (tiered cache handles progressive resolution)
        TileRenderParams params = buildParams(wk);
        _renderPool.submit(params, surface, volume);
    }
}

void TileRenderController::onParamsChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    _cache.clear();
    onCameraChanged(camera, surface, volume, buildParams, viewportRect);
}

void TileRenderController::scheduleRender(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    const std::shared_ptr<Volume>& volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    _pendingCamera = camera;
    _pendingSurface = surface;
    _pendingVolume = volume;
    _pendingBuildParams = buildParams;
    _pendingViewportRect = viewportRect;
    _pendingDirty = true;
    ensureTickRunning();
}

void TileRenderController::cancelAll()
{
    _renderPool.cancelAll();
}

void TileRenderController::drainResults()
{
    // Take up to DRAIN_BATCH_SIZE results per drain cycle
    auto results = _renderPool.drainCompleted(tiled_config::DRAIN_BATCH_SIZE, _currentEpoch);

    bool anyUpdated = false;

    for (auto& result : results) {
        if (result.image.isNull()) {
            // Still mark the tile's epoch so progressive refinement knows
            // it belongs to the current generation and is stale (level=-1).
            _tileScene->setTileWorld(result.worldKey, QPixmap(),
                                     result.epoch, -1);
            continue;
        }

        // QImage was already built on the worker thread (BGR→RGB + copy)
        QPixmap pixmap = QPixmap::fromImage(
            result.image, _skipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);

        // Always cache with world tile key (even if tile is no longer visible)
        TiledViewerCamera snapCamera;
        snapCamera.scale = result.scale;
        snapCamera.zOff = result.zOff;
        snapCamera.dsScaleIdx = result.actualLevel;

        SliceCacheKey cacheKey = SliceCacheKey::make(
            result.worldKey, snapCamera, _paramsHash);
        _cache.put(cacheKey, pixmap);

        // Apply to scene directly via world key
        if (_tileScene->setTileWorld(result.worldKey, pixmap, result.epoch,
                                     static_cast<int8_t>(result.actualLevel))) {
            anyUpdated = true;
        }
    }

    // Ensure the scene repaints after progressive refinement updates.
    // QGraphicsPixmapItem::setPixmap() should trigger this automatically,
    // but explicitly requesting an update guarantees the view refreshes
    // even when updates arrive while the view is idle.
    if (anyUpdated) {
        emit sceneNeedsUpdate();
    }

    // Process pending viewport change (coalesced)
    if (_pendingDirty) {
        _pendingDirty = false;
        onCameraChanged(_pendingCamera, _pendingSurface,
                        _pendingVolume, _pendingBuildParams, _pendingViewportRect);
        // Release references after dispatch
        _pendingSurface.reset();
        _pendingVolume.reset();
        _pendingBuildParams = nullptr;
    }
}

void TileRenderController::tick()
{
    bool moreWork = false;

    // 1. Drain completed render results
    drainResults();

    // 2. Check if chunks arrived → directly re-submit stale tiles using _last* state.
    //    Do NOT use the _pendingDirty mechanism — that path clears _pending* after
    //    dispatch, so a second chunk arrival would pass null state to onCameraChanged
    //    and corrupt _last*.
    bool chunksJustArrived = _chunkArrived;
    if (_chunkArrived) {
        _chunkArrived = false;
    }

    // 3. Progressive refinement: re-submit stale tiles only when new chunks
    //    have arrived (meaning finer data may now be available).
    //    Without the chunksJustArrived guard, idle pool + stale tiles = infinite loop.
    if (_progressiveEnabled && chunksJustArrived) {
        if (_lastSurface && _lastVolume && _lastBuildParams) {
            auto stale = _tileScene->staleTilesInRect(_desiredLevel, _currentEpoch, _lastViewportRect, tiled_config::VISIBLE_BUFFER_TILES);
            if (!stale.empty()) {
                // Sort by distance to viewport center so the user sees
                // center-of-screen tiles refine first.
                const auto& b = _tileScene->bounds();
                float cx = static_cast<float>(_lastViewportRect.center().x());
                float cy = static_cast<float>(_lastViewportRect.center().y());
                // Convert viewport center to fractional world-tile coords
                float wcx = cx / TileScene::TILE_PX - 0.5f;
                float wcy = cy / TileScene::TILE_PX - 0.5f;

                std::sort(stale.begin(), stale.end(),
                    [wcx, wcy](const WorldTileKey& a, const WorldTileKey& b) {
                        float da = (a.worldCol - wcx) * (a.worldCol - wcx)
                                 + (a.worldRow - wcy) * (a.worldRow - wcy);
                        float db = (b.worldCol - wcx) * (b.worldCol - wcx)
                                 + (b.worldRow - wcy) * (b.worldRow - wcy);
                        return da < db;
                    });

                int maxRefine = std::min(static_cast<int>(stale.size()),
                                         tiled_config::DRAIN_BATCH_SIZE);
                for (int i = 0; i < maxRefine; i++) {
                    TileRenderParams params = _lastBuildParams(stale[i]);
                    _renderPool.submit(params, _lastSurface, _lastVolume);
                }
                moreWork = true;
            }
        }
    }

    // 4. Zoom settle countdown
    if (_zoomSettlePending) {
        if (_zoomSettleTicksLeft == 0) {
            _zoomSettlePending = false;
            if (_zoomSettleCallback) _zoomSettleCallback();
        } else {
            _zoomSettleTicksLeft--;
            moreWork = true;
        }
    }

    // 5. Overlay update
    if (_overlaysDirty) {
        _overlaysDirty = false;
        if (_overlayCallback) _overlayCallback();
    }

    // Still have in-flight renders? Keep ticking to drain them.
    if (_renderPool.pendingCount() > 0 || _pendingDirty)
        moreWork = true;

    // Auto-stop when idle to avoid burning CPU
    if (!moreWork)
        _tickTimer->stop();
}

void TileRenderController::markOverlaysDirty()
{
    _overlaysDirty = true;
    ensureTickRunning();
}

void TileRenderController::markChunkArrived()
{
    _chunkArrived = true;
    ensureTickRunning();
}

void TileRenderController::startZoomSettle()
{
    _zoomSettlePending = true;
    ensureTickRunning();
    _zoomSettleTicksLeft = tiled_config::ZOOM_SETTLE_TICKS;  // ~200ms at 33ms/tick
}

void TileRenderController::cancelZoomSettle()
{
    _zoomSettlePending = false;
    _zoomSettleTicksLeft = 0;
}

void TileRenderController::setOverlayCallback(std::function<void()> cb)
{
    _overlayCallback = std::move(cb);
}

void TileRenderController::setZoomSettleCallback(std::function<void()> cb)
{
    _zoomSettleCallback = std::move(cb);
}
