#include "CTiledVolumeViewer.hpp"

#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "VolumeViewerCmaps.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include <limits>

#include <QSettings>
#include <QVBoxLayout>
#include <QLabel>
#include <QPainter>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QMdiSubWindow>
#include <QWindowStateChangeEvent>
#include <QScrollBar>
#include <QGuiApplication>

#include <algorithm>
#include <cmath>

#include <QDebug>
#include "vc/core/cache/TieredChunkCache.hpp"

constexpr auto COLOR_CURSOR = Qt::cyan;
#define COLOR_FOCUS QColor(50, 255, 215)
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

// ============================================================================
// Construction / destruction
// ============================================================================

CTiledVolumeViewer::CTiledVolumeViewer(CSurfaceCollection* col,
                                       ViewerManager* manager,
                                       QWidget* parent)
    : QWidget(parent)
    , _surfCol(col)
    , _viewerManager(manager)
{
    _compositeSettings.params.method = "max";
    _compositeSettings.params.alphaMin = 170 / 255.0f;
    _compositeSettings.params.alphaMax = 220 / 255.0f;
    _compositeSettings.params.alphaOpacity = 230 / 255.0f;
    _compositeSettings.params.alphaCutoff = 9950 / 10000.0f;

    // Create graphics view with scrollbars disabled
    fGraphicsView = new CVolumeViewerView(this);
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    fGraphicsView->setRenderHint(QPainter::Antialiasing);
    fGraphicsView->setScrollPanDisabled(true);

    // Connect signals from view
    connect(fGraphicsView, &CVolumeViewerView::sendScrolled, this, &CTiledVolumeViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked, this, &CTiledVolumeViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom, this, &CTiledVolumeViewer::onZoom);
    connect(fGraphicsView, &CVolumeViewerView::sendResized, this, &CTiledVolumeViewer::onResized);
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove, this, &CTiledVolumeViewer::onCursorMove);
    connect(fGraphicsView, &CVolumeViewerView::sendPanRelease, this, &CTiledVolumeViewer::onPanRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendPanStart, this, &CTiledVolumeViewer::onPanStart);
    connect(fGraphicsView, &CVolumeViewerView::sendMousePress, this, &CTiledVolumeViewer::onMousePress);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseMove, this, &CTiledVolumeViewer::onMouseMove);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseRelease, this, &CTiledVolumeViewer::onMouseRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendKeyRelease, this, &CTiledVolumeViewer::onKeyRelease);

    // Create fixed-size scene
    _scene = new QGraphicsScene(this);

    // Create tile scene manager
    _tileScene = new TileScene(_scene);

    // Create render controller (async tile rendering)
    _renderController = new TileRenderController(_tileScene, this);

    // Set the scene on the view
    fGraphicsView->setScene(_scene);

    // Read settings
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _skipImageFormatConv = settings.value(perf::SKIP_IMAGE_FORMAT_CONV, perf::SKIP_IMAGE_FORMAT_CONV_DEFAULT).toBool();
    _camera.downscaleOverride = settings.value(perf::DOWNSCALE_OVERRIDE, perf::DOWNSCALE_OVERRIDE_DEFAULT).toInt();
    _useFastInterpolation = settings.value(perf::FAST_INTERPOLATION, perf::FAST_INTERPOLATION_DEFAULT).toBool();
    _renderController->setSkipImageFormatConv(_skipImageFormatConv);

    auto* layout = new QVBoxLayout;
    layout->addWidget(fGraphicsView);
    setLayout(layout);

    // Wire callbacks into the unified tick on the render controller
    _renderController->setOverlayCallback([this]() { updateAllOverlays(); });
    _renderController->setZoomSettleCallback([this]() {
        fGraphicsView->resetTransform();
        _renderScale = _camera.scale;
        _renderSurfacePtr = _camera.surfacePtr;
        rebuildContentGrid();
        centerViewport();
        _camera.invalidate();
        submitRender();
    });

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    _lbl->setMinimumWidth(300);
    _lbl->move(10, 5);
}

CTiledVolumeViewer::~CTiledVolumeViewer()
{
    delete _tileScene;
    delete fGraphicsView;
    delete _scene;
}

// ============================================================================
// Data setup
// ============================================================================


void CTiledVolumeViewer::setPointCollection(VCCollection* pc)
{
    _pointCollection = pc;
    emit overlaysUpdated();
}

void CTiledVolumeViewer::setSurface(const std::string& name)
{
    _surfName = name;
    _surfWeak.reset();
    markActiveSegmentationDirty();
    onSurfaceChanged(name, _surfCol->surface(name));
}

void CTiledVolumeViewer::setIntersects(const std::set<std::string>& set) { _intersectTgts = set; }

Surface* CTiledVolumeViewer::currentSurface() const
{
    if (!_surfCol) {
        auto shared = _surfWeak.lock();
        return shared ? shared.get() : nullptr;
    }
    return _surfCol->surfaceRaw(_surfName);
}

// ============================================================================
// Volume / surface change handlers
// ============================================================================

void CTiledVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    // Invalidate all caches and cancel in-flight work from the old volume
    _renderController->cancelAll();
    _renderController->sliceCache().clear();
    _tileScene->resetMetadata();

    _volume = vol;

    // Reset pin progress tracking
    _pinTotal = 0;
    _pinReceived = 0;
    _pinLevel = -1;

    // Set up tiered chunk cache for progressive rendering
    if (_volume->numScales() >= 1) {
        // Wire chunk-ready callback BEFORE pin to ensure no callbacks are missed
        auto* ctrl = _renderController;
        auto* viewer = this;
        int coarsestLevel = static_cast<int>(_volume->numScales()) - 1;
        _volume->tieredCache()->setChunkReadyCallback(
            [ctrl, viewer, coarsestLevel](const vc::cache::ChunkKey& key) {
                QMetaObject::invokeMethod(ctrl, [ctrl]() { ctrl->markChunkArrived(); },
                                          Qt::QueuedConnection);
                // Track coarsest-level pin progress for remote volumes
                if (key.level == coarsestLevel && viewer->_pinTotal > 0) {
                    QMetaObject::invokeMethod(viewer, [viewer]() {
                        viewer->_pinReceived++;
                        viewer->updateStatusLabel();
                    }, Qt::QueuedConnection);
                }
            });

        // Pin coarsest pyramid level in hot tier.
        // For local volumes this blocks (fast), ensuring sampleBestEffort
        // always has fallback data. For remote volumes, pin non-blocking
        // to avoid freezing the UI during HTTP downloads.
        const bool isRemote = _volume->isRemote();
        if (isRemote) {
            // Set up progress tracking before pin so we can show download progress
            auto* tc = _volume->tieredCache();
            auto shape = tc->levelShape(coarsestLevel);
            auto chunks = tc->chunkShape(coarsestLevel);
            if (chunks[0] > 0 && chunks[1] > 0 && chunks[2] > 0) {
                _pinTotal = static_cast<int>(
                    ((shape[0] + chunks[0] - 1) / chunks[0]) *
                    ((shape[1] + chunks[1] - 1) / chunks[1]) *
                    ((shape[2] + chunks[2] - 1) / chunks[2]));
                _pinReceived = 0;
                _pinLevel = coarsestLevel;
            }
        }
        _volume->pinCoarsestLevel(!isRemote);
    }

    _camera.recalcPyramidLevel(_volume->numScales());
    updateContentMinScale();

    // Initialize zoom debounce tracking
    _renderScale = _camera.scale;
    _renderSurfacePtr = _camera.surfacePtr;

    updateStatusLabel();

    rebuildContentGrid();
    centerViewport();

    submitRender();

    // Update scalebar
    double vs = _volume->voxelSize() / _camera.dsScale;
    fGraphicsView->setVoxelSize(vs, vs);
}

void CTiledVolumeViewer::onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf,
                                           bool isEditUpdate)
{
    if (name == "segmentation" || name == _surfName) {
        markActiveSegmentationDirty();
    }

    if (_surfName == name) {
        _surfWeak = surf;
        if (!surf) {
            clearAllOverlayGroups();
            _tileScene->sceneCleared();
            _scene->clear();
            _intersectItems.clear();
            _sliceVisItems.clear();
            _paths.clear();
            emit overlaysUpdated();
            _cursor = nullptr;
            _centerMarker = nullptr;
            // Grid will be rebuilt when new surface is set
        } else {
            invalidateVis();
            if (!isEditUpdate) {
                _camera.zOff = 0.0f;
            }

            // Clear rendered tile cache and visible tiles — the cache keys
            // have no surface identifier so stale entries from the previous
            // surface would be served as false hits.
            _renderController->cancelAll();
            _renderController->sliceCache().clear();
            _tileScene->clearAll();

            updateContentMinScale();
            rebuildContentGrid();
            centerViewport();
            if (name == "segmentation" && _resetViewOnSurfaceChange) {
                fitSurfaceInView();
            }
        }
    }

    if (name == _surfName) {
        _camera.invalidate();
        submitRender();
    }

    _renderController->markOverlaysDirty();
}

void CTiledVolumeViewer::onVolumeClosing()
{
    if (_surfName == "segmentation") {
        onSurfaceChanged(_surfName, nullptr);
    } else if (_surfName == "xy plane" || _surfName == "xz plane" || _surfName == "yz plane") {
        clearAllOverlayGroups();
        _tileScene->sceneCleared();
        _scene->clear();
        _intersectItems.clear();
        _sliceVisItems.clear();
        _paths.clear();
        emit overlaysUpdated();
        _cursor = nullptr;
        _centerMarker = nullptr;
        _contentBounds = ContentBounds{};
    } else {
        onSurfaceChanged(_surfName, nullptr);
    }
}

void CTiledVolumeViewer::onSurfaceWillBeDeleted(std::string /*name*/, std::shared_ptr<Surface> surf)
{
    auto current = _surfWeak.lock();
    if (current && current == surf) {
        _surfWeak.reset();
    }
}

// ============================================================================
// Zoom limits
// ============================================================================

void CTiledVolumeViewer::updateContentMinScale()
{
    if (!fGraphicsView) return;

    QSize vpSize = fGraphicsView->viewport()->size();
    float vpW = vpSize.width();
    float vpH = vpSize.height();
    if (vpW <= 0 || vpH <= 0) return;

    float contentW = 0, contentH = 0;

    if (_volume && (_surfName == "xy plane" || _surfName == "xz plane" || _surfName == "yz plane")) {
        auto [w, h, d] = _volume->shape();
        if (_surfName == "xy plane") {
            contentW = static_cast<float>(w);
            contentH = static_cast<float>(h);
        } else if (_surfName == "xz plane") {
            contentW = static_cast<float>(w);
            contentH = static_cast<float>(d);
        } else {
            contentW = static_cast<float>(h);
            contentH = static_cast<float>(d);
        }
    } else {
        auto surf = _surfWeak.lock();
        if (auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get())) {
            const cv::Mat_<cv::Vec3f>& pts = quadSurf->rawPoints();
            cv::Vec2f sc = quadSurf->scale();
            contentW = pts.cols / sc[0];
            contentH = pts.rows / sc[1];
        }
    }

    if (contentW <= 0 || contentH <= 0) {
        _contentMinScale = TiledViewerCamera::MIN_SCALE;
        return;
    }

    // Scale at which content just fills viewport (fit, not cover)
    float fitScale = std::min(vpW / contentW, vpH / contentH);
    _contentMinScale = std::max(fitScale, TiledViewerCamera::MIN_SCALE);
}

void CTiledVolumeViewer::rebuildContentGrid()
{
    if (!fGraphicsView) return;

    float contentMinX = 0, contentMinY = 0, contentMaxX = 0, contentMaxY = 0;

    auto surf = _surfWeak.lock();
    if (_volume && surf) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            // Project volume bounding box corners onto the plane
            auto [w, h, d] = _volume->shape();
            float corners[][3] = {
                {0,0,0}, {(float)w,0,0}, {0,(float)h,0}, {(float)w,(float)h,0},
                {0,0,(float)d}, {(float)w,0,(float)d}, {0,(float)h,(float)d}, {(float)w,(float)h,(float)d}
            };
            contentMinX = contentMinY = std::numeric_limits<float>::max();
            contentMaxX = contentMaxY = std::numeric_limits<float>::lowest();
            for (auto& c : corners) {
                cv::Vec3f proj = plane->project(cv::Vec3f(c[0], c[1], c[2]), 1.0, 1.0);
                contentMinX = std::min(contentMinX, proj[0]);
                contentMinY = std::min(contentMinY, proj[1]);
                contentMaxX = std::max(contentMaxX, proj[0]);
                contentMaxY = std::max(contentMaxY, proj[1]);
            }
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            const cv::Mat_<cv::Vec3f>& pts = quad->rawPoints();
            cv::Vec2f sc = quad->scale();
            float halfW = (pts.cols * 0.5f) / sc[0];
            float halfH = (pts.rows * 0.5f) / sc[1];
            contentMinX = -halfW;
            contentMinY = -halfH;
            contentMaxX = halfW;
            contentMaxY = halfH;
        }
    }

    // Compute content bounds
    ContentBounds bounds;
    bounds.scale = _camera.scale;
    bounds.worldTileSize = static_cast<float>(TileScene::TILE_PX) / _camera.scale;

    if (bounds.worldTileSize > 0 && contentMaxX > contentMinX && contentMaxY > contentMinY) {
        bounds.firstWorldCol = static_cast<int>(std::floor(contentMinX / bounds.worldTileSize));
        bounds.firstWorldRow = static_cast<int>(std::floor(contentMinY / bounds.worldTileSize));
        int lastCol = static_cast<int>(std::floor(contentMaxX / bounds.worldTileSize));
        int lastRow = static_cast<int>(std::floor(contentMaxY / bounds.worldTileSize));
        bounds.totalCols = lastCol - bounds.firstWorldCol + 1;
        bounds.totalRows = lastRow - bounds.firstWorldRow + 1;
    }

    _contentBounds = bounds;

    QSize vpSize = fGraphicsView->viewport()->size();
    _tileScene->rebuildGrid(bounds, vpSize.width(), vpSize.height());
}

void CTiledVolumeViewer::centerViewport()
{
    if (!fGraphicsView || !_tileScene) return;
    QPointF scenePos = _tileScene->surfaceToScene(_camera.surfacePtr[0], _camera.surfacePtr[1]);
    fGraphicsView->centerOn(scenePos);
}

QRectF CTiledVolumeViewer::viewportSceneRect() const
{
    if (!fGraphicsView) return QRectF();
    return fGraphicsView->mapToScene(fGraphicsView->viewport()->rect()).boundingRect();
}

// ============================================================================
// Navigation
// ============================================================================

void CTiledVolumeViewer::panBy(int dx, int dy)
{
    const float invScale = 1.0f / _camera.scale;
    _camera.surfacePtr[0] -= dx * invScale;
    _camera.surfacePtr[1] -= dy * invScale;
    centerViewport();
    submitRender();
}

void CTiledVolumeViewer::zoomAt(float factor, const QPointF& scenePos)
{
    // Convert continuous factor into discrete zoom-stop steps
    int steps = (factor > 1.0f) ? 1 : (factor < 1.0f) ? -1 : 0;
    if (steps == 0) return;
    zoomStepsAt(steps, scenePos);
}

void CTiledVolumeViewer::zoomStepsAt(int steps, const QPointF& scenePos)
{
    const float newScale = std::max(
        TiledViewerCamera::stepScale(_camera.scale, steps), _contentMinScale);
    if (std::abs(newScale - _camera.scale) < 0.001f) {
        return;
    }

    // Convert scene position to viewport coordinates for zoom-at-point math
    QPointF vpPos = fGraphicsView->mapFromScene(scenePos);
    QSize vpSize = fGraphicsView->viewport()->size();
    const float vpCenterX = vpSize.width() * 0.5f;
    const float vpCenterY = vpSize.height() * 0.5f;
    const float dx = static_cast<float>(vpPos.x()) - vpCenterX;
    const float dy = static_cast<float>(vpPos.y()) - vpCenterY;

    _camera.surfacePtr[0] += dx * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.surfacePtr[1] += dy * (1.0f / _camera.scale - 1.0f / newScale);
    _camera.scale = newScale;

    int oldDsIdx = _camera.dsScaleIdx;
    if (_volume) {
        float oldDs = _camera.dsScale;
        _camera.recalcPyramidLevel(_volume->numScales());
        if (std::abs(_camera.dsScale - oldDs) > 1e-6f) {
            double vs = _volume->voxelSize() / _camera.dsScale;
            fGraphicsView->setVoxelSize(vs, vs);
        }
    }

    // Every stop change is significant — always do a full re-render
    _renderController->cancelZoomSettle();
    fGraphicsView->resetTransform();
    _renderScale = _camera.scale;
    _renderSurfacePtr = _camera.surfacePtr;
    rebuildContentGrid();
    centerViewport();
    _camera.invalidate();
    submitRender();
}

void CTiledVolumeViewer::setSliceOffset(float dz)
{
    _camera.zOff += dz;
    _camera.invalidate();
    submitRender();
}

void CTiledVolumeViewer::onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    if (_segmentationEditActive && (modifiers & Qt::ControlModifier)) {
        cv::Vec3f world = sceneToVolume(scene_point);
        emit sendSegmentationRadiusWheel(steps, scene_point, world);
        return;
    }

    if (modifiers & Qt::ShiftModifier) {
        if (steps == 0) return;

        PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
        int stepSize = _viewerManager ? _viewerManager->sliceStepSize() : 1;
        int adjustedSteps = steps * stepSize;

        if (_surfName != "segmentation" && plane && _surfCol) {
            POI* focus = _surfCol->poi("focus");
            if (!focus) {
                focus = new POI;
                focus->p = plane->origin();
                focus->n = plane->normal(cv::Vec3f(0, 0, 0), {});
            }

            cv::Vec3f normal = plane->normal(cv::Vec3f(0, 0, 0), {});
            const double length = cv::norm(normal);
            if (length > 0.0) normal *= static_cast<float>(1.0 / length);

            cv::Vec3f newPosition = focus->p + normal * static_cast<float>(adjustedSteps);

            if (_volume) {
                auto [w, h, d] = _volume->shape();
                newPosition[0] = std::clamp(newPosition[0], 0.0f, static_cast<float>(w - 1));
                newPosition[1] = std::clamp(newPosition[1], 0.0f, static_cast<float>(h - 1));
                newPosition[2] = std::clamp(newPosition[2], 0.0f, static_cast<float>(d - 1));
            }

            focus->p = newPosition;
            if (length > 0.0) focus->n = normal;
            focus->surfaceId = _surfName;
            _surfCol->setPOI("focus", focus);
        } else {
            setSliceOffset(static_cast<float>(adjustedSteps));
        }
    } else {
        // One zoom stop per scroll tick, regardless of scroll delta magnitude
        int zoomDir = (steps > 0) ? 1 : (steps < 0) ? -1 : 0;
        if (zoomDir != 0) {
            zoomStepsAt(zoomDir, scene_point);
            emit overlaysUpdated();
        }
    }

    updateStatusLabel();
    _renderController->markOverlaysDirty();
}

void CTiledVolumeViewer::adjustZoomByFactor(float factor)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    int steps = (factor > 1.0f) ? 1 : (factor < 1.0f) ? -1 : 0;
    if (steps == 0) return;

    // Zoom centered on viewport center
    QSize vpSize = fGraphicsView->viewport()->size();
    QPointF vpCenter(vpSize.width() * 0.5, vpSize.height() * 0.5);
    QPointF sceneCenter = fGraphicsView->mapToScene(vpCenter.toPoint());
    zoomStepsAt(steps, sceneCenter);

    emit overlaysUpdated();
    updateStatusLabel();
    _renderController->markOverlaysDirty();
}

void CTiledVolumeViewer::adjustSurfaceOffset(float dn)
{
    setSliceOffset(dn);
    emit overlaysUpdated();
    updateStatusLabel();
    _renderController->markOverlaysDirty();
}

void CTiledVolumeViewer::resetSurfaceOffsets()
{
    _camera.zOff = 0.0f;
    _camera.invalidate();
    submitRender();
    emit overlaysUpdated();
    updateStatusLabel();
    _renderController->markOverlaysDirty();
}

// ============================================================================
// Pan handling via CVolumeViewerView events
// ============================================================================

void CTiledVolumeViewer::onPanStart(Qt::MouseButton /*buttons*/, Qt::KeyboardModifiers /*modifiers*/)
{
    _isPanning = true;
    // The view handles pan tracking with _last_pan_position internally,
    // but since we disabled scrollbars, its scroll-based panning won't work.
    // We need to intercept the mouse move deltas instead.
    // Store current mouse pos for delta computation
    _lastPanPos = QCursor::pos();
}

void CTiledVolumeViewer::onPanRelease(Qt::MouseButton /*buttons*/, Qt::KeyboardModifiers /*modifiers*/)
{
    _isPanning = false;
    _renderController->markOverlaysDirty();
}

void CTiledVolumeViewer::onScrolled()
{
    // In tiled mode, scrollbar-based scrolling is disabled.
    // Pan is handled via panBy() from mouse move deltas.
}

void CTiledVolumeViewer::onResized()
{
    updateContentMinScale();
    rebuildContentGrid();
    centerViewport();
    _camera.invalidate();
    submitRender();
}

// ============================================================================
// Rendering
// ============================================================================

void CTiledVolumeViewer::renderVisible(bool force)
{
    if (isWindowMinimized()) {
        _dirtyWhileMinimized = true;
        return;
    }
    if (force) {
        _camera.invalidate();
    }
    submitRender();
}

void CTiledVolumeViewer::renderAllTiles()
{
    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) return;
    if (_tileScene->cols() == 0 || _tileScene->rows() == 0) return;

    _tileScene->forEachTile([&](const TileKey& key) {
        WorldTileKey wk = _contentBounds.worldKeyAt(key.col, key.row);
        TileRenderParams params = buildRenderParams(wk);
        TileRenderResult result = TileRenderer::renderTile(params, surf, _volume.get());

        if (!result.image.empty()) {
            QImage qimg = Mat2QImage(result.image);
            QPixmap pixmap = QPixmap::fromImage(
                qimg, _skipImageFormatConv ? Qt::NoFormatConversion : Qt::AutoColor);
            _tileScene->setTileWorld(wk, pixmap, _camera.epoch, static_cast<int8_t>(result.actualLevel));
        }
    });

    // Update center marker
    if (!_centerMarker) {
        _centerMarker = _scene->addEllipse({-10, -10, 20, 20},
            QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _centerMarker->setZValue(11);
    }
    QPointF centerScene = _tileScene->surfaceToScene(_camera.surfacePtr[0], _camera.surfacePtr[1]);
    _centerMarker->setPos(centerScene);
}

void CTiledVolumeViewer::submitRender()
{
    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_volume->zarrDataset()) {
        return;
    }

    if (_tileScene->cols() == 0 || _tileScene->rows() == 0) {
        return;
    }

    QRectF vpRect = viewportSceneRect();

    // Update center marker
    if (!_centerMarker) {
        _centerMarker = _scene->addEllipse({-10, -10, 20, 20},
            QPen(COLOR_FOCUS, 3, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin));
        _centerMarker->setZValue(11);
    }
    QPointF centerScene = _tileScene->surfaceToScene(_camera.surfacePtr[0], _camera.surfacePtr[1]);
    _centerMarker->setPos(centerScene);

    // Submit to async render controller
    auto buildParams = [this](const WorldTileKey& wk) { return buildRenderParams(wk); };
    _renderController->scheduleRender(_camera, surf, _volume.get(), buildParams, vpRect);

    // Viewport-aware prefetch
    if (_volume->tieredCache()) {
        auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
        if (plane) {
            const float invScale = 1.0f / _camera.scale;
            const float margin = TileScene::TILE_PX * invScale;

            // Get viewport extent in surface params
            cv::Vec2f vpTopLeft = _tileScene->sceneToSurface(QPointF(vpRect.left(), vpRect.top()));
            cv::Vec2f vpBotRight = _tileScene->sceneToSurface(QPointF(vpRect.right(), vpRect.bottom()));

            const float uMin = vpTopLeft[0] - margin * invScale;
            const float uMax = vpBotRight[0] + margin * invScale;
            const float vMin = vpTopLeft[1] - margin * invScale;
            const float vMax = vpBotRight[1] + margin * invScale;

            const cv::Vec3f o = plane->origin();
            const cv::Vec3f bx = plane->basisX();
            const cv::Vec3f by = plane->basisY();
            const cv::Vec3f n = plane->normal(cv::Vec3f(0, 0, 0));

            cv::Vec3f corners[4] = {
                o + bx * uMin + by * vMin + n * _camera.zOff,
                o + bx * uMax + by * vMin + n * _camera.zOff,
                o + bx * uMin + by * vMax + n * _camera.zOff,
                o + bx * uMax + by * vMax + n * _camera.zOff,
            };

            cv::Vec3f lo(std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max());
            cv::Vec3f hi(std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::lowest());
            for (const auto& c : corners) {
                for (int i = 0; i < 3; i++) {
                    lo[i] = std::min(lo[i], c[i]);
                    hi[i] = std::max(hi[i], c[i]);
                }
            }
            _volume->prefetchWorldBBox(lo, hi, _camera.dsScaleIdx);
        }
    }
}

void CTiledVolumeViewer::updateParamsHash()
{
    // Simple hash combining rendering parameters that affect output
    size_t h = 14695981039346656037ULL;
    auto mix = [&h](size_t val) {
        h ^= val;
        h *= 1099511628211ULL;
    };

    mix(std::hash<float>{}(_baseWindowLow));
    mix(std::hash<float>{}(_baseWindowHigh));
    mix(std::hash<bool>{}(_stretchValues));
    mix(std::hash<std::string>{}(_baseColormapId));
    mix(std::hash<bool>{}(_useFastInterpolation));
    mix(std::hash<bool>{}(_compositeSettings.enabled));
    mix(std::hash<int>{}(_compositeSettings.layersFront));
    mix(std::hash<int>{}(_compositeSettings.layersBehind));
    mix(std::hash<bool>{}(_compositeSettings.planeEnabled));
    mix(std::hash<std::string>{}(_compositeSettings.params.method));
    mix(std::hash<uint8_t>{}(_compositeSettings.params.isoCutoff));

    _renderController->setParamsHash(static_cast<uint64_t>(h));
}

TileRenderParams CTiledVolumeViewer::buildRenderParams(const WorldTileKey& wk) const
{
    TileRenderParams params;
    params.worldKey = wk;
    params.epoch = _camera.epoch;

    auto surf = _surfWeak.lock();
    params.isPlaneSurface = surf && dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;

    // Surface parameter ROI from world tile coordinates
    params.surfaceROI.x = wk.worldCol * _contentBounds.worldTileSize;
    params.surfaceROI.y = wk.worldRow * _contentBounds.worldTileSize;
    params.surfaceROI.width = _contentBounds.worldTileSize;
    params.surfaceROI.height = _contentBounds.worldTileSize;

    params.tileW = TileScene::TILE_PX;
    params.tileH = TileScene::TILE_PX;

    params.scale = _camera.scale;
    params.dsScale = _camera.dsScale;
    params.dsScaleIdx = _camera.dsScaleIdx;
    params.zOff = _camera.zOff;

    params.windowLow = _baseWindowLow;
    params.windowHigh = _baseWindowHigh;
    params.stretchValues = _stretchValues;
    params.colormapId = _baseColormapId;
    params.useFastInterpolation = _useFastInterpolation;
    params.compositeSettings = _compositeSettings;
    params.isoCutoff = _compositeSettings.params.isoCutoff;

    return params;
}

// ============================================================================
// Coordinate transforms
// ============================================================================

QPointF CTiledVolumeViewer::volumeToScene(const cv::Vec3f& vol_point)
{
    auto surf = _surfWeak.lock();
    if (!surf) return QPointF();

    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf.get());
    QuadSurface* quad = dynamic_cast<QuadSurface*>(surf.get());

    if (plane) {
        cv::Vec3f surfPos = plane->project(vol_point, 1.0, 1.0);
        return _tileScene->surfaceToScene(surfPos[0], surfPos[1]);
    } else if (quad) {
        cv::Vec3f ptr(0, 0, 0);
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        surf->pointTo(ptr, vol_point, 4.0, 100, patchIndex);
        cv::Vec3f loc = surf->loc(ptr);
        return _tileScene->surfaceToScene(loc[0], loc[1]);
    }

    return QPointF();
}

cv::Vec3f CTiledVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    cv::Vec3f p, n;
    if (sceneToVolumeHelper(p, n, scenePoint)) {
        return p;
    }
    return {0.0f, 0.0f, 0.0f};
}

bool CTiledVolumeViewer::sceneToVolumeHelper(cv::Vec3f& p, cv::Vec3f& n,
                                              const QPointF& scenePos) const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        p = cv::Vec3f(0, 0, 0);
        n = cv::Vec3f(0, 0, 1);
        return false;
    }

    try {
        cv::Vec2f surfParam = _tileScene->sceneToSurface(scenePos);
        cv::Vec3f surfLoc = {surfParam[0], surfParam[1], 0};
        cv::Vec3f ptr(0, 0, 0);
        n = surf->normal(ptr, surfLoc);
        p = surf->coord(ptr, surfLoc);
    } catch (const cv::Exception&) {
        return false;
    }
    return true;
}

// ============================================================================
// Mouse handlers
// ============================================================================

void CTiledVolumeViewer::onCursorMove(QPointF scene_loc)
{
    auto surf = _surfWeak.lock();
    if (!surf || !_surfCol) return;

    // Handle panning: if middle/right button is down, pan instead
    if (_isPanning) {
        QPoint currentPos = QCursor::pos();
        QPoint delta = _lastPanPos - currentPos;
        _lastPanPos = currentPos;
        if (delta.x() != 0 || delta.y() != 0) {
            panBy(-delta.x(), -delta.y());
        }
        return;
    }

    cv::Vec3f p, n;
    if (!sceneToVolumeHelper(p, n, scene_loc)) {
        if (_cursor) _cursor->hide();
    } else {
        if (_cursor) {
            _cursor->show();
            _cursor->setPos(scene_loc);
        }

        POI* cursor = _surfCol->poi("cursor");
        if (!cursor) cursor = new POI;
        cursor->p = p;
        cursor->n = n;
        cursor->surfaceId = _surfName;
        _surfCol->setPOI("cursor", cursor);
    }

    // Point highlight logic
    if (_pointCollection && _draggedPointId == 0) {
        uint64_t oldHighlighted = _highlightedPointId;
        _highlightedPointId = 0;

        const float threshold = 10.0f;
        float minDistSq = threshold * threshold;

        const auto& collections = _pointCollection->getAllCollections();
        for (const auto& [colId, col] : collections) {
            for (const auto& [ptId, pt] : col.points) {
                QPointF ptScene = volumeToScene(pt.p);
                QPointF diff = scene_loc - ptScene;
                float distSq = QPointF::dotProduct(diff, diff);
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    _highlightedPointId = pt.id;
                }
            }
        }

        if (oldHighlighted != _highlightedPointId) {
            emit overlaysUpdated();
        }
    }
}

void CTiledVolumeViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons,
                                          Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf) return;

    if (_draggedPointId != 0) return;

    cv::Vec3f p, n;
    if (!sceneToVolumeHelper(p, n, scene_loc)) return;

    if (buttons == Qt::LeftButton) {
        bool isShift = modifiers.testFlag(Qt::ShiftModifier);
        if (isShift && !_segmentationEditActive) {
            if (_selectedCollectionId != 0) {
                const auto& collections = _pointCollection->getAllCollections();
                auto it = collections.find(_selectedCollectionId);
                if (it != collections.end()) {
                    _pointCollection->addPoint(it->second.name, p);
                }
            } else {
                std::string newName = _pointCollection->generateNewCollectionName("col");
                auto newPoint = _pointCollection->addPoint(newName, p);
                _selectedCollectionId = newPoint.collectionId;
                emit sendCollectionSelected(_selectedCollectionId);
            }
        } else if (_highlightedPointId != 0) {
            emit pointClicked(_highlightedPointId);
        }
    }

    const auto& segmentation = activeSegmentationHandle();
    if (dynamic_cast<PlaneSurface*>(surf.get())) {
        sendVolumeClicked(p, n, surf.get(), buttons, modifiers);
    } else if (segmentation.viewerIsSegmentationView && segmentation.surface) {
        sendVolumeClicked(p, n, segmentation.surface, buttons, modifiers);
    }
}

void CTiledVolumeViewer::onMousePress(QPointF scene_loc, Qt::MouseButton button,
                                       Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!_pointCollection || !surf) return;

    if (button == Qt::LeftButton) {
        if (_highlightedPointId != 0 && !modifiers.testFlag(Qt::ControlModifier)) {
            emit pointClicked(_highlightedPointId);
            _draggedPointId = _highlightedPointId;
        }
    } else if (button == Qt::RightButton) {
        if (_highlightedPointId != 0) {
            _pointCollection->removePoint(_highlightedPointId);
        }
    }

    cv::Vec3f p, n;
    if (sceneToVolumeHelper(p, n, scene_loc)) {
        _lastScenePos = scene_loc;
        sendMousePressVolume(p, n, button, modifiers);
    }
}

void CTiledVolumeViewer::onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons,
                                      Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    onCursorMove(scene_loc);

    if ((buttons & Qt::LeftButton) && _draggedPointId != 0) {
        cv::Vec3f p, n;
        if (sceneToVolumeHelper(p, n, scene_loc)) {
            if (auto pointOpt = _pointCollection->getPoint(_draggedPointId)) {
                ColPoint updated = *pointOpt;
                updated.p = p;
                _pointCollection->updatePoint(updated);
            }
        }
    } else {
        if (!surf) return;
        cv::Vec3f p, n;
        if (!sceneToVolumeHelper(p, n, scene_loc)) return;
        _lastScenePos = scene_loc;
        emit sendMouseMoveVolume(p, buttons, modifiers);
    }
}

void CTiledVolumeViewer::onMouseRelease(QPointF scene_loc, Qt::MouseButton button,
                                         Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (button == Qt::LeftButton && _draggedPointId != 0) {
        _draggedPointId = 0;
        onCursorMove(scene_loc);
    }

    cv::Vec3f p, n;
    if (sceneToVolumeHelper(p, n, scene_loc)) {
        const auto& segmentation = activeSegmentationHandle();
        if (dynamic_cast<PlaneSurface*>(surf.get())) {
            emit sendMouseReleaseVolume(p, button, modifiers);
        } else if (segmentation.viewerIsSegmentationView) {
            emit sendMouseReleaseVolume(p, button, modifiers);
        }
    }
}

void CTiledVolumeViewer::onKeyRelease(int key, Qt::KeyboardModifiers /*modifiers*/)
{
    constexpr int PAN_PX = 64;
    switch (key) {
    case Qt::Key_Left:  panBy( PAN_PX, 0); break;
    case Qt::Key_Right: panBy(-PAN_PX, 0); break;
    case Qt::Key_Up:    panBy(0,  PAN_PX); break;
    case Qt::Key_Down:  panBy(0, -PAN_PX); break;
    default: break;
    }
}

// ============================================================================
// POI handling
// ============================================================================

void CTiledVolumeViewer::onPOIChanged(std::string name, POI* poi)
{
    auto surf = _surfWeak.lock();
    if (!poi || !surf) return;

    if (name == "focus") {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            if (poi->p == plane->origin()) return;
            plane->setOrigin(poi->p);
            emit overlaysUpdated();
            _surfCol->setSurface(_surfName, surf);
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            cv::Vec3f ptr(0, 0, 0);
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            float dist = quad->pointTo(ptr, poi->p, 4.0, 100, patchIndex);
            if (dist < 4.0) {
                // Center camera on the focus point
                cv::Vec3f loc = quad->loc(ptr);
                _camera.surfacePtr[0] = loc[0];
                _camera.surfacePtr[1] = loc[1];
                _camera.invalidate();
                submitRender();
            }
        }
    } else if (name == "cursor") {
        Surface* currentSurf = this->currentSurface();
        if (!currentSurf) return;

        if (_surfName == "segmentation" && !_mirrorCursorToSegmentation) {
            if (poi->surfaceId.empty() || poi->surfaceId != _surfName) return;
        }

        // Position cursor in canvas coords
        QPointF scenePos = volumeToScene(poi->p);
        if (!_cursor) {
            // Create cursor item
            QPen pen(QBrush(COLOR_CURSOR), 2);
            QGraphicsLineItem* parent = new QGraphicsLineItem(-10, 0, -5, 0);
            parent->setZValue(10);
            parent->setPen(pen);
            auto* l1 = new QGraphicsLineItem(10, 0, 5, 0, parent); l1->setPen(pen);
            auto* l2 = new QGraphicsLineItem(0, -10, 0, -5, parent); l2->setPen(pen);
            auto* l3 = new QGraphicsLineItem(0, 10, 0, 5, parent); l3->setPen(pen);
            _cursor = parent;
            _scene->addItem(_cursor);
        }

        // Simple distance-based opacity
        PlaneSurface* slicePlane = dynamic_cast<PlaneSurface*>(currentSurf);
        if (slicePlane) {
            float dist = slicePlane->pointDist(poi->p);
            if (dist < 20.0f / _camera.scale) {
                _cursor->setPos(scenePos);
                _cursor->setOpacity(1.0 - dist * _camera.scale / 20.0);
            } else {
                _cursor->setOpacity(0.0);
            }
        } else {
            _cursor->setPos(scenePos);
            _cursor->setOpacity(1.0);
        }
    }
}

// ============================================================================
// Status / display
// ============================================================================

void CTiledVolumeViewer::updateStatusLabel()
{
    QString status = QString("%1x").arg(_camera.scale, 0, 'f', 2);

    auto surf = _surfWeak.lock();
    if (surf) {
        if (dynamic_cast<PlaneSurface*>(surf.get())) {
            cv::Vec3f center(0, 0, 0);
            status += QString(" ctr(%1,%2,%3)")
                .arg(center[0], 0, 'f', 0)
                .arg(center[1], 0, 'f', 0)
                .arg(center[2], 0, 'f', 0);
        }
    }

    status += QString(" z=%1").arg(_camera.zOff, 0, 'f', 1);

    if (_compositeSettings.enabled) {
        QString method = QString::fromStdString(_compositeSettings.params.method);
        method[0] = method[0].toUpper();
        status += QString(" | %1(%2)").arg(method).arg(
            _compositeSettings.layersFront + _compositeSettings.layersBehind);
    }

    // Cache stats
    if (_volume && _volume->tieredCache()) {
        auto s = _volume->tieredCache()->stats();
        uint64_t total = s.hotHits + s.warmHits + s.coldHits + s.iceFetches + s.misses;
        if (total > 0) {
            int hitPct = static_cast<int>(100 * (s.hotHits + s.warmHits) / total);
            status += QString(" | cache %1%").arg(hitPct);
        }
    } else {
        auto& sc = _renderController->sliceCache();
        size_t total = sc.hits() + sc.misses();
        if (total > 0) {
            int hitPct = static_cast<int>(100 * sc.hits() / total);
            status += QString(" | tile$ %1%").arg(hitPct);
        }
    }

    // Remote download stats
    if (_volume && _volume->tieredCache()) {
        auto s = _volume->tieredCache()->stats();
        if (s.iceFetches > 0 || s.ioPending > 0) {
            status += QString(" | dl %1").arg(s.iceFetches);
            if (s.ioPending > 0)
                status += QString(" q%1").arg(s.ioPending);
        }
    } else if (_pinTotal > 0 && _pinReceived < _pinTotal) {
        status += QString(" | downloading %1/%2").arg(_pinReceived).arg(_pinTotal);
    }

    status += " [tiled]";

    _lbl->setText(status);
}

void CTiledVolumeViewer::fitSurfaceInView()
{
    auto surf = _surfWeak.lock();
    if (!surf || !fGraphicsView) return;

    auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());
    if (!quadSurf) return;

    // Auto-crop: find bounding box of valid (non-sentinel) points
    const cv::Mat_<cv::Vec3f>& pts = quadSurf->rawPoints();
    int colMin = pts.cols, colMax = -1, rowMin = pts.rows, rowMax = -1;
    for (int j = 0; j < pts.rows; j++)
        for (int i = 0; i < pts.cols; i++)
            if (pts(j, i)[0] != -1 && std::isfinite(pts(j, i)[0])) {
                colMin = std::min(colMin, i);
                colMax = std::max(colMax, i);
                rowMin = std::min(rowMin, j);
                rowMax = std::max(rowMax, j);
            }

    if (colMax < colMin || rowMax < rowMin) {
        _camera.scale = 1.0f;
        _camera.surfacePtr = cv::Vec3f(0, 0, 0);
        if (_volume) _camera.recalcPyramidLevel(_volume->numScales());
        updateStatusLabel();
        return;
    }

    // Valid region in grid pixels
    float validW = static_cast<float>(colMax - colMin + 1);
    float validH = static_cast<float>(rowMax - rowMin + 1);

    // Convert to surface parameter space (what the viewport uses)
    cv::Vec2f sc = quadSurf->scale();
    float validSurfW = validW / sc[0];
    float validSurfH = validH / sc[1];

    QSize vpSize = fGraphicsView->viewport()->size();
    float vpW = vpSize.width();
    float vpH = vpSize.height();
    if (vpW <= 0 || vpH <= 0) return;

    float fitFactor = 0.8f;
    float reqScaleX = (vpW * fitFactor) / validSurfW;
    float reqScaleY = (vpH * fitFactor) / validSurfH;
    _camera.scale = TiledViewerCamera::roundScale(std::min(reqScaleX, reqScaleY));

    // Center on the valid region.
    // surfacePtr is in grid units; (0,0,0) maps to the grid center (cols/2, rows/2).
    // To center on grid position (gx, gy): surfacePtr = (gx - cols/2, gy - rows/2, 0)
    float gridCenterX = (colMin + colMax) * 0.5f;
    float gridCenterY = (rowMin + rowMax) * 0.5f;
    _camera.surfacePtr[0] = gridCenterX - pts.cols * 0.5f;
    _camera.surfacePtr[1] = gridCenterY - pts.rows * 0.5f;
    _camera.surfacePtr[2] = 0;

    if (_volume) _camera.recalcPyramidLevel(_volume->numScales());
    _camera.invalidate();
    updateStatusLabel();
    rebuildContentGrid();
    centerViewport();
    submitRender();
}

bool CTiledVolumeViewer::isWindowMinimized() const
{
    auto* subWindow = qobject_cast<QMdiSubWindow*>(parentWidget());
    return subWindow && subWindow->isMinimized();
}

bool CTiledVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::WindowStateChange) {
        auto* subWindow = qobject_cast<QMdiSubWindow*>(watched);
        if (subWindow && !subWindow->isMinimized()) {
            auto* stateEvent = static_cast<QWindowStateChangeEvent*>(event);
            if (stateEvent->oldState() & Qt::WindowMinimized) {
                if (_dirtyWhileMinimized) {
                    _dirtyWhileMinimized = false;
                    renderVisible(true);
                    updateAllOverlays();
                }
            }
        }
    }
    return QWidget::eventFilter(watched, event);
}

// ============================================================================
// Settings pass-through
// ============================================================================

void CTiledVolumeViewer::setCompositeRenderSettings(const CompositeRenderSettings& settings)
{
    if (_compositeSettings == settings) return;
    _compositeSettings = settings;
    updateParamsHash();
    // Params changed: clear cache and re-render
    auto surf = _surfWeak.lock();
    if (_volume && surf) {
        _camera.invalidate();
        _renderController->onParamsChanged(_camera, surf, _volume.get(),
            [this](const WorldTileKey& wk) { return buildRenderParams(wk); },
            viewportSceneRect());
    }
    updateStatusLabel();
}

void CTiledVolumeViewer::setVolumeWindow(float low, float high)
{
    constexpr float kMax = 255.0f;
    float cLow = std::clamp(low, 0.0f, kMax);
    float cHigh = std::clamp(high, 0.0f, kMax);
    if (cHigh <= cLow) cHigh = std::min(kMax, cLow + 1.0f);
    if (std::abs(cLow - _baseWindowLow) < 1e-6f && std::abs(cHigh - _baseWindowHigh) < 1e-6f) return;
    _baseWindowLow = cLow;
    _baseWindowHigh = cHigh;
    updateParamsHash();
    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setBaseColormap(const std::string& id)
{
    if (_baseColormapId == id) return;
    _baseColormapId = id;
    updateParamsHash();
    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setStretchValues(bool enabled)
{
    if (_stretchValues == enabled) return;
    _stretchValues = enabled;
    updateParamsHash();
    if (_volume) renderVisible(true);
}

void CTiledVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> vol) { _overlayVolume = std::move(vol); }
void CTiledVolumeViewer::setOverlayOpacity(float opacity) { _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f); }
void CTiledVolumeViewer::setOverlayColormap(const std::string& id) { _overlayColormapId = id; }
void CTiledVolumeViewer::setOverlayThreshold(float threshold) { setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh); }
void CTiledVolumeViewer::setOverlayWindow(float low, float high) { _overlayWindowLow = low; _overlayWindowHigh = high; }

void CTiledVolumeViewer::setResetViewOnSurfaceChange(bool reset) { _resetViewOnSurfaceChange = reset; }
void CTiledVolumeViewer::setSegmentationEditActive(bool active) { _segmentationEditActive = active; }

void CTiledVolumeViewer::setIntersectionOpacity(float opacity) { _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f); }
void CTiledVolumeViewer::setIntersectionThickness(float thickness) { _intersectionThickness = thickness; }
void CTiledVolumeViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids) { _highlightedSurfaceIds = {ids.begin(), ids.end()}; }
void CTiledVolumeViewer::setSurfacePatchSamplingStride(int stride) { _surfacePatchSamplingStride = std::max(1, stride); }

void CTiledVolumeViewer::setSurfaceOverlayEnabled(bool enabled) { _surfaceOverlayEnabled = enabled; if (_volume) renderVisible(true); }
void CTiledVolumeViewer::setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) { _surfaceOverlays = overlays; if (_volume && _surfaceOverlayEnabled) renderVisible(true); }
void CTiledVolumeViewer::setSurfaceOverlapThreshold(float threshold) { _surfaceOverlapThreshold = std::max(0.1f, threshold); if (_volume && _surfaceOverlayEnabled) renderVisible(true); }

// ============================================================================
// Overlay group management
// ============================================================================

void CTiledVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _overlayGroups[key] = items;
    // Items are already added to the scene by applyPrimitives().
    // Do NOT re-add here (matches CVolumeViewer behavior).
}

void CTiledVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlayGroups.find(key);
    if (it == _overlayGroups.end()) return;
    for (auto* item : it->second) {
        if (!item) continue;
        if (item->scene()) {
            item->scene()->removeItem(item);
        }
        delete item;
    }
    _overlayGroups.erase(it);
}

void CTiledVolumeViewer::clearAllOverlayGroups()
{
    for (auto& [key, items] : _overlayGroups) {
        for (auto* item : items) {
            if (!item) continue;
            if (item->scene()) {
                item->scene()->removeItem(item);
            }
            delete item;
        }
    }
    _overlayGroups.clear();
}

void CTiledVolumeViewer::updateAllOverlays()
{
    emit overlaysUpdated();
}

// ============================================================================
// Stubs for functionality ported in later phases
// ============================================================================

void CTiledVolumeViewer::renderIntersections() { /* Phase 4 */ }
void CTiledVolumeViewer::invalidateVis()
{
    for (auto* item : _sliceVisItems) {
        _scene->removeItem(item);
        delete item;
    }
    _sliceVisItems.clear();
}
void CTiledVolumeViewer::invalidateIntersect(const std::string& /*name*/) { /* Phase 4 */ }

void CTiledVolumeViewer::onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths)
{
    _paths.clear();
    _paths.reserve(paths.size());
    for (const auto& path : paths) _paths.push_back(path);
    emit overlaysUpdated();
}

void CTiledVolumeViewer::onCollectionSelected(uint64_t id)
{
    _selectedCollectionId = id;
    emit overlaysUpdated();
}

void CTiledVolumeViewer::onPointSelected(uint64_t pointId)
{
    if (_selectedPointId == pointId) return;
    _selectedPointId = pointId;
    emit overlaysUpdated();
}

void CTiledVolumeViewer::onDrawingModeActive(bool active, float brushSize, bool isSquare)
{
    _drawingModeActive = active;
    _brushSize = brushSize;
    _brushIsSquare = isSquare;
    if (_cursor) {
        _scene->removeItem(_cursor);
        delete _cursor;
        _cursor = nullptr;
    }
    POI* cursor = _surfCol->poi("cursor");
    if (cursor) onPOIChanged("cursor", cursor);
}

void CTiledVolumeViewer::markActiveSegmentationDirty()
{
    _activeSegHandleDirty = true;
    _activeSegHandle.reset();
}

const CTiledVolumeViewer::ActiveSegmentationHandle& CTiledVolumeViewer::activeSegmentationHandle() const
{
    if (!_activeSegHandleDirty) return _activeSegHandle;

    ActiveSegmentationHandle handle;
    handle.slotName = "segmentation";
    handle.viewerIsSegmentationView = (_surfName == "segmentation");
    handle.accentColor =
        (_surfName == "seg yz"   ? QColor(COLOR_SEG_YZ)
         : _surfName == "seg xz" ? QColor(COLOR_SEG_XZ)
                                  : QColor(COLOR_SEG_XY));
    if (_surfCol) {
        auto surfHolder = _surfCol->surface(handle.slotName);
        handle.surface = dynamic_cast<QuadSurface*>(surfHolder.get());
    }
    if (!handle.surface) handle.slotName.clear();

    _activeSegHandle = handle;
    _activeSegHandleDirty = false;
    return _activeSegHandle;
}

// BBox stubs
void CTiledVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSceneRect) {
        _activeBBoxSceneRect.reset();
        emit overlaysUpdated();
    }
}

QuadSurface* CTiledVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& /*sceneRect*/)
{
    // TODO: port from CVolumeViewer (Phase 4)
    return nullptr;
}

auto CTiledVolumeViewer::selections() const -> std::vector<std::pair<QRectF, QColor>>
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& s : _selections) {
        QPointF topLeft = _tileScene->surfaceToScene(s.surfRect.left(), s.surfRect.top());
        QPointF botRight = _tileScene->surfaceToScene(s.surfRect.right(), s.surfRect.bottom());
        QRectF sceneRect(topLeft, botRight);
        out.emplace_back(sceneRect.normalized(), s.color);
    }
    return out;
}

void CTiledVolumeViewer::clearSelections()
{
    _selections.clear();
    emit overlaysUpdated();
}
