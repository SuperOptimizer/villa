#include "TileRenderController.hpp"

#include <QDebug>
#include <QTimer>
#include <QImage>
#include <vector>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

TileRenderController::TileRenderController(TileScene* tileScene, QObject* parent)
    : QObject(parent)
    , _tileScene(tileScene)
    , _renderPool(2, this)
{
    // Single unified tick timer (~30 Hz) handles all periodic work
    _tickTimer = new QTimer(this);
    _tickTimer->setInterval(33);
    connect(_tickTimer, &QTimer::timeout, this, &TileRenderController::tick);
    _tickTimer->start();

    // Also drain immediately when a tile completes (makes draining more responsive)
    connect(&_renderPool, &RenderPool::tileReady, this, &TileRenderController::drainResults,
            Qt::QueuedConnection);
}

TileRenderController::~TileRenderController()
{
    _tickTimer->stop();
    _renderPool.cancelAll();
}

void TileRenderController::onCameraChanged(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    Volume* volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    bool epochChanged = (camera.epoch != _currentEpoch);
    _currentEpoch = camera.epoch;
    _renderPool.setCurrentEpoch(_currentEpoch);
    _desiredLevel = camera.dsScaleIdx;

    if (epochChanged && volume) {
        volume->cancelPendingPrefetch();
    }

    // Store last render state for refinement retries
    _lastCamera = camera;
    _lastSurface = surface;
    _lastVolume = volume;
    _lastBuildParams = buildParams;
    _lastViewportRect = viewportRect;

    // Get visible tiles (+ 1 tile buffer for smooth scrolling)
    auto visibleKeys = _tileScene->visibleTiles(viewportRect, 1);

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
    Volume* volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    _cache.clear();
    onCameraChanged(camera, surface, volume, buildParams, viewportRect);
}

void TileRenderController::scheduleRender(
    const TiledViewerCamera& camera,
    const std::shared_ptr<Surface>& surface,
    Volume* volume,
    const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
    const QRectF& viewportRect)
{
    _pendingCamera = camera;
    _pendingSurface = surface;
    _pendingVolume = volume;
    _pendingBuildParams = buildParams;
    _pendingViewportRect = viewportRect;
    _pendingDirty = true;
}

void TileRenderController::cancelAll()
{
    _renderPool.cancelAll();
}

void TileRenderController::drainResults()
{
    // Take up to 32 results per drain cycle
    auto results = _renderPool.drainCompleted(32, _currentEpoch);


    for (auto& result : results) {
        if (result.image.empty()) {
            continue;
        }

        // Convert cv::Mat to QPixmap
        QImage qimg = Mat2QImage(result.image);
        QPixmap pixmap = QPixmap::fromImage(
            qimg, _skipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);

        // Always cache with world tile key (even if tile is no longer visible)
        TiledViewerCamera snapCamera;
        snapCamera.scale = result.scale;
        snapCamera.zOff = result.zOff;
        snapCamera.dsScaleIdx = result.actualLevel;

        SliceCacheKey cacheKey = SliceCacheKey::make(
            result.worldKey, snapCamera, _paramsHash);
        _cache.put(cacheKey, pixmap);

        // Apply to scene directly via world key
        _tileScene->setTileWorld(result.worldKey, pixmap, result.epoch,
                                 static_cast<int8_t>(result.actualLevel));
    }

    // Process pending viewport change (coalesced)
    if (_pendingDirty) {
        _pendingDirty = false;
        onCameraChanged(_pendingCamera, _pendingSurface,
                        _pendingVolume, _pendingBuildParams, _pendingViewportRect);
        // Release references after dispatch
        _pendingSurface.reset();
        _pendingVolume = nullptr;
        _pendingBuildParams = nullptr;
    }
}

void TileRenderController::tick()
{
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

    // 3. Progressive refinement: re-submit stale tiles when pool is idle
    //    or when new chunks just arrived (some tiles may now have all chunks cached).
    if (_progressiveEnabled && _lastSurface && _lastVolume && _lastBuildParams
        && (chunksJustArrived || _renderPool.pendingCount() == 0)) {
        auto stale = _tileScene->staleTilesInRect(_desiredLevel, _currentEpoch, _lastViewportRect, 1);
        if (!stale.empty()) {
            for (const auto& wk : stale) {
                TileRenderParams params = _lastBuildParams(wk);
                _renderPool.submit(params, _lastSurface, _lastVolume);
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
        }
    }

    // 5. Overlay update
    if (_overlaysDirty) {
        _overlaysDirty = false;
        if (_overlayCallback) _overlayCallback();
    }
}

void TileRenderController::markOverlaysDirty()
{
    _overlaysDirty = true;
}

void TileRenderController::markChunkArrived()
{
    _chunkArrived = true;
}

void TileRenderController::startZoomSettle()
{
    _zoomSettlePending = true;
    _zoomSettleTicksLeft = 6;  // ~200ms at 33ms/tick
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
