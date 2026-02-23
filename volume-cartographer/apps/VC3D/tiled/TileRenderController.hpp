#pragma once

#include <QObject>
#include <QPixmap>
#include <QRectF>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "SliceCache.hpp"
#include "RenderPool.hpp"
#include "TileScene.hpp"
#include "TiledViewerCamera.hpp"
#include "TileRenderer.hpp"

class QTimer;
class Surface;
class Volume;

// Orchestrates tile rendering: checks cache, submits misses to background
// pool, drains completed results, and updates the tile scene.
//
// All public methods must be called from the main thread.
class TileRenderController : public QObject
{
    Q_OBJECT

public:
    explicit TileRenderController(TileScene* tileScene, QObject* parent = nullptr);
    ~TileRenderController() override;

    // Called when camera state changes (pan, zoom, slice offset).
    // For each visible tile:
    //   - Cache hit -> apply pixmap immediately
    //   - Cache miss -> submit to background pool
    void onCameraChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         Volume* volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         const QRectF& viewportRect);

    // Called when rendering parameters change (window/level, colormap, etc.)
    // Clears the slice cache and re-renders everything.
    void onParamsChanged(const TiledViewerCamera& camera,
                         const std::shared_ptr<Surface>& surface,
                         Volume* volume,
                         const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                         const QRectF& viewportRect);

    // Deferred render: stores state and processes on next drain tick.
    // Rapid calls coalesce into a single onCameraChanged().
    void scheduleRender(const TiledViewerCamera& camera,
                        const std::shared_ptr<Surface>& surface,
                        Volume* volume,
                        const std::function<TileRenderParams(const WorldTileKey&)>& buildParams,
                        const QRectF& viewportRect);

    // Cancel all in-flight renders and clear results.
    void cancelAll();

    // Access the slice cache (for stats, manual invalidation, etc.)
    SliceCache& sliceCache() { return _cache; }

    // Hash rendering parameters for cache key generation.
    // Should be recomputed whenever window/level, colormap, composite settings change.
    void setParamsHash(uint64_t hash) { _paramsHash = hash; }
    uint64_t paramsHash() const { return _paramsHash; }

    // Whether to skip Qt image format conversion (performance setting)
    void setSkipImageFormatConv(bool skip) { _skipImageFormatConv = skip; }

    // Progressive rendering: show coarse previews while full-res loads
    void setProgressiveEnabled(bool enabled) { _progressiveEnabled = enabled; }
    bool progressiveEnabled() const { return _progressiveEnabled; }

    // --- Dirty flags (set by viewer, processed each tick) ---
    void markOverlaysDirty();
    void markChunkArrived();
    void startZoomSettle();
    void cancelZoomSettle();
    void setOverlayCallback(std::function<void()> cb);
    void setZoomSettleCallback(std::function<void()> cb);

signals:
    // Emitted when drainResults() actually updated tile pixmaps.
    // Connect to viewport()->update() to guarantee repaints during
    // progressive refinement (chunk arrival while the view is idle).
    void sceneNeedsUpdate();

private slots:
    // Drain completed results from the render pool and update tile scene.
    void drainResults();

    // Unified tick: runs at ~30 Hz, handles all periodic work.
    void tick();

    // Start the tick timer if not already running.
    void ensureTickRunning();

private:
    TileScene* _tileScene;
    SliceCache _cache;
    RenderPool _renderPool;
    QTimer* _tickTimer;

    uint64_t _currentEpoch = 0;
    uint64_t _paramsHash = 0;
    QRectF _lastViewportRect;
    bool _skipImageFormatConv = false;
    bool _progressiveEnabled = true;

    // Desired pyramid level for current render pass
    int _desiredLevel = 0;

    // Last render state for refinement re-submission
    TiledViewerCamera _lastCamera;
    std::shared_ptr<Surface> _lastSurface;
    Volume* _lastVolume = nullptr;
    std::function<TileRenderParams(const WorldTileKey&)> _lastBuildParams;

    // Pending state for coalescing rapid viewport changes
    bool _pendingDirty = false;
    TiledViewerCamera _pendingCamera;
    std::shared_ptr<Surface> _pendingSurface;
    Volume* _pendingVolume = nullptr;
    std::function<TileRenderParams(const WorldTileKey&)> _pendingBuildParams;
    QRectF _pendingViewportRect;

    // Dirty flags set by the viewer, processed each tick
    bool _overlaysDirty = false;
    bool _chunkArrived = false;
    bool _zoomSettlePending = false;
    uint64_t _zoomSettleTicksLeft = 0;  // countdown in ticks (~6 ticks = 200ms)

    // Callbacks to notify the viewer
    std::function<void()> _overlayCallback;
    std::function<void()> _zoomSettleCallback;
};
