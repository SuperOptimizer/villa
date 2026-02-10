// CSliceViewer.cpp -- widget setup, event handling, surface management
//
// See CSliceViewerTiles.cpp for tile rendering, worker thread, tile management.

#include "CSliceViewer.hpp"
#include "ViewerManager.hpp"
#include "VCSettings.hpp"
#include "VolumeViewerCmaps.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"

#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsPixmapItem>
#include <QGraphicsEllipseItem>
#include <QGuiApplication>
#include <QLabel>
#include <QMdiSubWindow>
#include <QPainter>
#include <QSettings>
#include <QTimer>
#include <QVBoxLayout>
#include <QWindowStateChangeEvent>

#include <algorithm>
#include <cmath>
#include <iostream>

constexpr double ZOOM_FACTOR = 1.05;
constexpr auto COLOR_CURSOR = Qt::cyan;
constexpr float MIN_ZOOM = 0.03125f;
constexpr float MAX_ZOOM = 4.0f;

#define COLOR_FOCUS QColor(50, 255, 215)
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red
#define COLOR_SEG_XY QColor(255, 140, 0)

static float round_scale(float scale)
{
    if (std::abs(scale - std::round(std::log2(scale))) < 0.02f)
        scale = std::pow(2.0f, std::round(std::log2(scale)));
    if (scale < MIN_ZOOM) scale = MIN_ZOOM;
    if (scale > MAX_ZOOM) scale = MAX_ZOOM;
    return scale;
}

static QPointF visible_center(QGraphicsView *view)
{
    QRectF bbox = view->mapToScene(view->viewport()->geometry()).boundingRect();
    return bbox.topLeft() + QPointF(bbox.width(), bbox.height()) * 0.5;
}

// ---- scene <-> volume coordinate conversion ----
static bool scene2vol(cv::Vec3f &p, cv::Vec3f &n,
                      Surface *surf, const std::string & /*surfName*/,
                      CSurfaceCollection * /*col*/,
                      const QPointF &scene_loc, const cv::Vec2f & /*visCenter*/,
                      float scale)
{
    if (!surf) { p = {}; n = {0,0,1}; return false; }
    try {
        cv::Vec3f surf_loc = {static_cast<float>(scene_loc.x() / scale),
                              static_cast<float>(scene_loc.y() / scale), 0};
        auto ptr = surf->pointer();
        n = surf->normal(ptr, surf_loc);
        p = surf->coord(ptr, surf_loc);
    } catch (...) { return false; }
    return true;
}

static QGraphicsItem *cursorItem(bool drawingMode, float brushSize, bool isSquare)
{
    if (drawingMode) {
        auto *group = new QGraphicsItemGroup();
        group->setZValue(10);
        QPen pen(QBrush(COLOR_CURSOR), 1.5);
        pen.setStyle(Qt::DashLine);
        if (isSquare) {
            float h = brushSize / 2.0f;
            auto *r = new QGraphicsRectItem(-h, -h, brushSize, brushSize);
            r->setPen(pen); r->setBrush(Qt::NoBrush);
            group->addToGroup(r);
        } else {
            auto *c = new QGraphicsEllipseItem(-brushSize/2, -brushSize/2, brushSize, brushSize);
            c->setPen(pen); c->setBrush(Qt::NoBrush);
            group->addToGroup(c);
        }
        QPen cp(QBrush(COLOR_CURSOR), 1);
        auto *l = new QGraphicsLineItem(-2,0,2,0); l->setPen(cp); group->addToGroup(l);
        l = new QGraphicsLineItem(0,-2,0,2); l->setPen(cp); group->addToGroup(l);
        return group;
    }
    QPen pen(QBrush(COLOR_CURSOR), 2);
    auto *parent = new QGraphicsLineItem(-10,0,-5,0);
    parent->setZValue(10); parent->setPen(pen);
    auto *l = new QGraphicsLineItem(10,0,5,0, parent); l->setPen(pen);
    l = new QGraphicsLineItem(0,-10,0,-5, parent); l->setPen(pen);
    l = new QGraphicsLineItem(0,10,0,5, parent); l->setPen(pen);
    return parent;
}

// =========================================================================
// Constructor / destructor
// =========================================================================

CSliceViewer::CSliceViewer(CSurfaceCollection *col, ViewerManager* manager, QWidget* parent)
    : QWidget(parent)
    , fGraphicsView(nullptr)
    , _surf_col(col)
    , _viewerManager(manager)
{
    fGraphicsView = new CVolumeViewerView(this);
    fGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fGraphicsView->setTransformationAnchor(QGraphicsView::NoAnchor);
    fGraphicsView->setRenderHint(QPainter::Antialiasing);

    connect(fGraphicsView, &CVolumeViewerView::sendScrolled,       this, &CSliceViewer::onScrolled);
    connect(fGraphicsView, &CVolumeViewerView::sendVolumeClicked,  this, &CSliceViewer::onVolumeClicked);
    connect(fGraphicsView, &CVolumeViewerView::sendZoom,           this, &CSliceViewer::onZoom);
    connect(fGraphicsView, &CVolumeViewerView::sendResized,        this, &CSliceViewer::onResized);
    connect(fGraphicsView, &CVolumeViewerView::sendCursorMove,     this, &CSliceViewer::onCursorMove);
    connect(fGraphicsView, &CVolumeViewerView::sendPanRelease,     this, &CSliceViewer::onPanRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendPanStart,       this, &CSliceViewer::onPanStart);
    connect(fGraphicsView, &CVolumeViewerView::sendMousePress,     this, &CSliceViewer::onMousePress);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseMove,      this, &CSliceViewer::onMouseMove);
    connect(fGraphicsView, &CVolumeViewerView::sendMouseRelease,   this, &CSliceViewer::onMouseRelease);
    connect(fGraphicsView, &CVolumeViewerView::sendKeyRelease,     this, &CSliceViewer::onKeyRelease);

    fScene = new QGraphicsScene({-2500, -2500, 5000, 5000}, this);
    fGraphicsView->setScene(fScene);

    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    fSkipImageFormatConv = settings.value(perf::SKIP_IMAGE_FORMAT_CONV, perf::SKIP_IMAGE_FORMAT_CONV_DEFAULT).toBool();
    _downscale_override = settings.value(perf::DOWNSCALE_OVERRIDE, perf::DOWNSCALE_OVERRIDE_DEFAULT).toInt();
    _useFastInterpolation = settings.value(perf::FAST_INTERPOLATION, perf::FAST_INTERPOLATION_DEFAULT).toBool();

    auto *layout = new QVBoxLayout;
    layout->addWidget(fGraphicsView);
    setLayout(layout);

    _overlayUpdateTimer = new QTimer(this);
    _overlayUpdateTimer->setSingleShot(true);
    _overlayUpdateTimer->setInterval(50);
    connect(_overlayUpdateTimer, &QTimer::timeout, this, &CSliceViewer::updateAllOverlays);

    _lbl = new QLabel(this);
    _lbl->setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    _lbl->setMinimumWidth(300);
    _lbl->move(10, 5);

    // Timer to consume rendered tiles on the main thread (16 ms ~= 60 Hz)
    _resultPollTimer = new QTimer(this);
    _resultPollTimer->setInterval(16);
    connect(_resultPollTimer, &QTimer::timeout, this, &CSliceViewer::consumeReadyTiles);
    _resultPollTimer->start();

    // Start background render thread
    _renderThread = std::thread(&CSliceViewer::renderWorkerLoop, this);
}

CSliceViewer::~CSliceViewer()
{
    // Signal worker to stop
    {
        std::lock_guard<std::mutex> lk(_renderQueueMutex);
        _shutdown.store(true, std::memory_order_release);
    }
    _renderQueueCV.notify_all();

    if (_renderThread.joinable())
        _renderThread.join();

    delete fGraphicsView;
    delete fScene;
}

// =========================================================================
// Window helpers
// =========================================================================

bool CSliceViewer::isWindowMinimized() const
{
    auto* sub = qobject_cast<QMdiSubWindow*>(parentWidget());
    return sub && sub->isMinimized();
}

bool CSliceViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::WindowStateChange) {
        auto* sub = qobject_cast<QMdiSubWindow*>(watched);
        if (sub && !sub->isMinimized()) {
            auto* se = static_cast<QWindowStateChangeEvent*>(event);
            if (se->oldState() & Qt::WindowMinimized) {
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

// =========================================================================
// Volume / surface management
// =========================================================================

void CSliceViewer::setCache(ChunkCache<uint8_t> *c) { cache = c; }

void CSliceViewer::setPointCollection(VCCollection* pc)
{
    _point_collection = pc;
    emit overlaysUpdated();
}

Surface* CSliceViewer::currentSurface() const
{
    if (!_surf_col) {
        auto s = _surf_weak.lock();
        return s ? s.get() : nullptr;
    }
    return _surf_col->surfaceRaw(_surf_name);
}

void CSliceViewer::setSurface(const std::string &name)
{
    _surf_name = name;
    _surf_weak.reset();
    markActiveSegmentationDirty();
    onSurfaceChanged(name, _surf_col->surface(name));
}

void CSliceViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    volume = vol;
    int max_size = 100000;
    fGraphicsView->setSceneRect(-max_size/2, -max_size/2, max_size, max_size);

    if (volume->numScales() >= 2) {
        _max_scale = 0.5f;
        _min_scale = std::pow(2.0f, 1.0f - static_cast<float>(volume->numScales()));
    } else {
        _max_scale = 1.0f;
        _min_scale = 1.0f;
    }

    recalcScales();
    updateStatusLabel();
    renderVisible(true);

    double vs = volume->voxelSize() / _ds_scale;
    fGraphicsView->setVoxelSize(vs, vs);
}

void CSliceViewer::onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate)
{
    if (name == "segmentation" || name == _surf_name)
        markActiveSegmentationDirty();

    if (_surf_name == name) {
        _surf_weak = surf;
        if (!surf) {
            clearAllOverlayGroups();
            invalidateAllTiles();
            fScene->clear();
            _tiles.clear();
            _intersect_items.clear();
            _cachedIntersectionLines.clear();
            slice_vis_items.clear();
            _paths.clear();
            emit overlaysUpdated();
            _cursor = nullptr;
            _center_marker = nullptr;
        } else {
            invalidateVis();
            if (!isEditUpdate)
                _z_off = 0.0f;
            if (name == "segmentation" && _resetViewOnSurfaceChange)
                fitSurfaceInView();
        }
    }

    if (name == _surf_name) {
        invalidateAllTiles();
        renderVisible(true);
    }

    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CSliceViewer::onVolumeClosing()
{
    if (_surf_name == "segmentation") {
        onSurfaceChanged(_surf_name, nullptr);
    } else if (_surf_name == "xy plane" || _surf_name == "xz plane" || _surf_name == "yz plane") {
        clearAllOverlayGroups();
        invalidateAllTiles();
        fScene->clear();
        _tiles.clear();
        _intersect_items.clear();
        _cachedIntersectionLines.clear();
        slice_vis_items.clear();
        _paths.clear();
        emit overlaysUpdated();
        _cursor = nullptr;
        _center_marker = nullptr;
    } else {
        onSurfaceChanged(_surf_name, nullptr);
    }
}

void CSliceViewer::onSurfaceWillBeDeleted(std::string /*name*/, std::shared_ptr<Surface> surf)
{
    auto current = _surf_weak.lock();
    if (current && current == surf)
        _surf_weak.reset();

    auto quad = std::dynamic_pointer_cast<QuadSurface>(surf);
    for (auto it = _cachedIntersectSurfaces.begin(); it != _cachedIntersectSurfaces.end();) {
        if (it->second == quad) it = _cachedIntersectSurfaces.erase(it);
        else ++it;
    }
    if (quad)
        _trianglesBySurface.erase(quad);
}

// =========================================================================
// Scale management
// =========================================================================

void CSliceViewer::recalcScales()
{
    float old_ds = _ds_scale;
    _min_scale = std::pow(2.0f, 1.0f - static_cast<float>(volume->numScales()));

    if      (_scale >= _max_scale) _ds_sd_idx = 0;
    else if (_scale <  _min_scale) _ds_sd_idx = static_cast<int>(volume->numScales()) - 1;
    else    _ds_sd_idx = static_cast<int>(std::round(-std::log2(_scale)));

    if (_downscale_override > 0) {
        _ds_sd_idx += _downscale_override;
        _ds_sd_idx = std::min(_ds_sd_idx, static_cast<int>(volume->numScales()) - 1);
    }
    _ds_scale = std::pow(2.0f, -_ds_sd_idx);

    if (volume && std::abs(_ds_scale - old_ds) > 1e-6f) {
        double vs = volume->voxelSize() / _ds_scale;
        fGraphicsView->setVoxelSize(vs, vs);
    }
}

// =========================================================================
// Zoom / scroll / resize
// =========================================================================

void CSliceViewer::onZoom(int steps, QPointF scene_loc, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surf_weak.lock();
    if (!surf) return;

    if (_segmentationEditActive && (modifiers & Qt::ControlModifier)) {
        cv::Vec3f world = sceneToVolume(scene_loc);
        emit sendSegmentationRadiusWheel(steps, scene_loc, world);
        return;
    }

    bool handled = false;

    if (modifiers & Qt::ShiftModifier) {
        if (steps == 0) return;

        auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
        int stepSize = _viewerManager ? _viewerManager->sliceStepSize() : 1;
        int adj = steps * stepSize;

        if (_surf_name != "segmentation" && plane && _surf_col) {
            POI* focus = _surf_col->poi("focus");
            if (!focus) {
                focus = new POI;
                focus->p = plane->origin();
                focus->n = plane->normal(plane->pointer(), {});
            }
            cv::Vec3f n = plane->normal(plane->pointer(), {});
            float len = static_cast<float>(cv::norm(n));
            if (len > 0) n *= 1.0f / len;
            cv::Vec3f newPos = focus->p + n * static_cast<float>(adj);
            if (volume) {
                auto [w, h, d] = volume->shape();
                newPos[0] = std::clamp(newPos[0], 0.0f, static_cast<float>(w - 1));
                newPos[1] = std::clamp(newPos[1], 0.0f, static_cast<float>(h - 1));
                newPos[2] = std::clamp(newPos[2], 0.0f, static_cast<float>(d - 1));
            }
            focus->p = newPos;
            if (len > 0) focus->n = n;
            focus->surfaceId = _surf_name;
            _surf_col->setPOI("focus", focus);
            handled = true;
        } else {
            _z_off += adj;
            if (volume && plane) {
                float ez = plane->origin()[2] + _z_off;
                ez = std::clamp(ez, 0.0f, static_cast<float>(volume->numSlices() - 1));
                _z_off = ez - plane->origin()[2];
            }
            invalidateAllTiles();
            renderVisible(true);
            handled = true;
        }
    }

    if (!handled) {
        float zoom = static_cast<float>(std::pow(ZOOM_FACTOR, steps));
        _scale *= zoom;
        _scale = round_scale(_scale);
        if (_scale > MIN_ZOOM && _scale < MAX_ZOOM) {
            recalcScales();
            fGraphicsView->translate(scene_loc.x() * (1 - zoom),
                                     scene_loc.y() * (1 - zoom));
            int max_size = 100000;
            fGraphicsView->setSceneRect(-max_size/2, -max_size/2, max_size, max_size);
        }
        invalidateAllTiles();
        renderVisible();
        emit overlaysUpdated();
    }

    updateStatusLabel();
    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CSliceViewer::adjustZoomByFactor(float factor)
{
    auto surf = _surf_weak.lock();
    if (!surf) return;

    float newScale = round_scale(_scale * factor);
    if (newScale > MIN_ZOOM && newScale < MAX_ZOOM && std::abs(newScale - _scale) > 0.001f) {
        float zoom = newScale / _scale;
        _scale = newScale;
        recalcScales();
        QPointF center = visible_center(fGraphicsView);
        fGraphicsView->translate(center.x() * (1 - zoom), center.y() * (1 - zoom));
        int max_size = 100000;
        fGraphicsView->setSceneRect(-max_size/2, -max_size/2, max_size, max_size);
    }
    invalidateAllTiles();
    renderVisible();
    emit overlaysUpdated();
    updateStatusLabel();
    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CSliceViewer::onScrolled()
{
    // Scrolling does NOT invalidate existing tiles -- just need new edge tiles
    renderVisible(false);
}

void CSliceViewer::onResized()
{
    renderVisible(false);
}

void CSliceViewer::onPanStart(Qt::MouseButton, Qt::KeyboardModifiers)
{
    // Nothing to do -- tiles persist
}

void CSliceViewer::onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers)
{
    renderVisible(false);
    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

// =========================================================================
// renderVisible -- the main entry point from the GUI
// =========================================================================

void CSliceViewer::renderVisible(bool force)
{
    if (isWindowMinimized()) {
        _dirtyWhileMinimized = true;
        return;
    }

    auto surf = _surf_weak.lock();
    if (surf && _surf_col) {
        auto cur = _surf_col->surface(_surf_name);
        if (!cur) { _surf_weak.reset(); surf.reset(); }
    }
    if (!volume || !volume->zarrDataset() || !surf)
        return;

    QRectF bbox = fGraphicsView->mapToScene(fGraphicsView->viewport()->geometry()).boundingRect();

    // Remove tiles that have scrolled far out of view
    removeOffscreenTiles(bbox);

    // Enqueue tiles covering the visible area
    // First pass: coarser level (fast fallback)
    int coarseIdx = std::min(_ds_sd_idx + 2, static_cast<int>(volume->numScales()) - 1);
    if (coarseIdx != _ds_sd_idx) {
        float coarseScale = std::pow(2.0f, -coarseIdx);
        enqueueTilesForViewport(bbox, coarseIdx, coarseScale);
    }
    // Second pass: native level
    enqueueTilesForViewport(bbox, _ds_sd_idx, _ds_scale);

    (void)force;
}

// =========================================================================
// Coordinate conversion
// =========================================================================

QPointF CSliceViewer::volumeToScene(const cv::Vec3f& vol_point)
{
    auto surf = _surf_weak.lock();
    if (!surf) return {};

    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (plane) {
        cv::Vec3f sp = plane->project(vol_point, 1.0f, _scale);
        return {sp[0], sp[1]};
    }
    if (quad) {
        auto ptr = quad->pointer();
        auto* idx = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        surf->pointTo(ptr, vol_point, 4.0, 100, idx);
        cv::Vec3f sp = surf->loc(ptr) * _scale;
        return {sp[0], sp[1]};
    }
    return {};
}

cv::Vec3f CSliceViewer::sceneToVolume(const QPointF& scenePoint) const
{
    auto surf = _surf_weak.lock();
    cv::Vec3f p, n;
    if (scene2vol(p, n, surf.get(), _surf_name,
                  const_cast<CSurfaceCollection*>(_surf_col),
                  scenePoint, _vis_center, _scale))
        return p;
    return {0,0,0};
}

// =========================================================================
// Cursor / POI
// =========================================================================

void CSliceViewer::onCursorMove(QPointF scene_loc)
{
    auto surf = _surf_weak.lock();
    if (!surf || !_surf_col) return;

    cv::Vec3f p, n;
    if (!scene2vol(p, n, surf.get(), _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        if (_cursor) _cursor->hide();
    } else {
        if (_cursor) {
            _cursor->show();
            auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
            if (plane) {
                cv::Vec3f sp = plane->project(p, 1.0f, _scale);
                _cursor->setPos(sp[0], sp[1]);
            } else {
                _cursor->setPos(scene_loc);
            }
        }
        POI *cur = _surf_col->poi("cursor");
        if (!cur) cur = new POI;
        cur->p = p;
        cur->n = n;
        cur->surfaceId = _surf_name;
        _surf_col->setPOI("cursor", cur);
    }
}

void CSliceViewer::onPOIChanged(std::string name, POI *poi)
{
    auto surf = _surf_weak.lock();
    if (!poi || !surf) return;

    if (name == "focus") {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            fGraphicsView->centerOn(0, 0);
            if (poi->p == plane->origin()) return;
            plane->setOrigin(poi->p);
            emit overlaysUpdated();
            _surf_col->setSurface(_surf_name, surf);
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            auto ptr = quad->pointer();
            auto* pIdx = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            float dist = quad->pointTo(ptr, poi->p, 4.0, 100, pIdx);
            if (dist < 4.0) {
                cv::Vec3f sp = quad->loc(ptr) * _scale;
                if (_center_marker) { _center_marker->setPos(sp[0], sp[1]); _center_marker->show(); }
                fGraphicsView->centerOn(sp[0], sp[1]);
                invalidateAllTiles();
                renderVisible(true);
            } else {
                if (_center_marker) _center_marker->hide();
            }
        }
    } else if (name == "cursor") {
        Surface* cs = currentSurface();
        if (!cs) return;

        if (_surf_name == "segmentation" && !_mirrorCursorToSegmentation) {
            if (poi->surfaceId.empty() || poi->surfaceId != _surf_name) return;
        }

        auto* slice_plane = dynamic_cast<PlaneSurface*>(cs);
        const auto& seg = activeSegmentationHandle();
        QuadSurface* crop = seg.surface;

        cv::Vec3f sp;
        float dist = -1;
        if (slice_plane) {
            dist = slice_plane->pointDist(poi->p);
            sp = slice_plane->project(poi->p, 1.0f, _scale);
        } else if (seg.viewerIsSegmentationView && crop) {
            auto ptr = crop->pointer();
            auto* pIdx = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            dist = crop->pointTo(ptr, poi->p, 2.0f, 1000, pIdx);
            sp = crop->loc(ptr) * _scale;
        }

        if (!_cursor) {
            _cursor = cursorItem(_drawingModeActive, _brushSize, _brushIsSquare);
            fScene->addItem(_cursor);
        }
        if (dist != -1) {
            if (dist < 20.0f / _scale) {
                _cursor->setPos(sp[0], sp[1]);
                _cursor->setOpacity(1.0 - dist * _scale / 20.0);
            } else {
                _cursor->setOpacity(0.0);
            }
        }
    }
}

// =========================================================================
// Mouse events
// =========================================================================

void CSliceViewer::onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surf_weak.lock();
    if (!surf) return;
    if (_dragged_point_id != 0) return;

    cv::Vec3f p, n;
    if (!scene2vol(p, n, surf.get(), _surf_name, _surf_col, scene_loc, _vis_center, _scale))
        return;

    const auto& seg = activeSegmentationHandle();
    if (dynamic_cast<PlaneSurface*>(surf.get()))
        sendVolumeClicked(p, n, surf.get(), buttons, modifiers);
    else if (seg.viewerIsSegmentationView && seg.surface)
        sendVolumeClicked(p, n, seg.surface, buttons, modifiers);
}

void CSliceViewer::onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surf_weak.lock();
    if (!_point_collection || !surf) return;

    if (button == Qt::LeftButton && _highlighted_point_id != 0 && !modifiers.testFlag(Qt::ControlModifier)) {
        emit pointClicked(_highlighted_point_id);
        _dragged_point_id = _highlighted_point_id;
    } else if (button == Qt::RightButton && _highlighted_point_id != 0) {
        _point_collection->removePoint(_highlighted_point_id);
    }

    cv::Vec3f p, n;
    if (scene2vol(p, n, surf.get(), _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
        _lastScenePos = scene_loc;
        sendMousePressVolume(p, n, button, modifiers);
    }
}

void CSliceViewer::onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surf_weak.lock();
    onCursorMove(scene_loc);

    if ((buttons & Qt::LeftButton) && _dragged_point_id != 0) {
        cv::Vec3f p, n;
        if (scene2vol(p, n, surf.get(), _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
            if (auto pt = _point_collection->getPoint(_dragged_point_id)) {
                ColPoint up = *pt;
                up.p = p;
                _point_collection->updatePoint(up);
            }
        }
    } else {
        if (!surf) return;
        cv::Vec3f p, n;
        if (scene2vol(p, n, surf.get(), _surf_name, _surf_col, scene_loc, _vis_center, _scale)) {
            _lastScenePos = scene_loc;
            emit sendMouseMoveVolume(p, buttons, modifiers);
        }
    }
}

void CSliceViewer::onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surf_weak.lock();
    if (button == Qt::LeftButton && _dragged_point_id != 0) {
        _dragged_point_id = 0;
        onCursorMove(scene_loc);
    }

    cv::Vec3f p, n;
    if (scene2vol(p, n, surf.get(), _surf_name, _surf_col, scene_loc, _vis_center, _scale))
        emit sendMouseReleaseVolume(p, button, modifiers);
}

void CSliceViewer::onKeyRelease(int, Qt::KeyboardModifiers) {}

void CSliceViewer::onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths)
{
    _paths.clear();
    _paths.reserve(paths.size());
    for (const auto& p : paths) _paths.push_back(p);
    emit overlaysUpdated();
}

void CSliceViewer::onPointSelected(uint64_t pointId)
{
    if (_selected_point_id == pointId) return;
    _selected_point_id = pointId;
    emit overlaysUpdated();
}

void CSliceViewer::onCollectionSelected(uint64_t id)
{
    _selected_collection_id = id;
    emit overlaysUpdated();
}

void CSliceViewer::onDrawingModeActive(bool active, float brushSize, bool isSquare)
{
    _drawingModeActive = active;
    _brushSize = brushSize;
    _brushIsSquare = isSquare;
    if (_cursor) { fScene->removeItem(_cursor); delete _cursor; _cursor = nullptr; }
    POI *c = _surf_col->poi("cursor");
    if (c) onPOIChanged("cursor", c);
}

// =========================================================================
// Status label
// =========================================================================

void CSliceViewer::updateStatusLabel()
{
    QString status = QString("%1x").arg(_scale, 0, 'f', 2);

    auto surf = _surf_weak.lock();
    if (surf) {
        if (dynamic_cast<PlaneSurface*>(surf.get())) {
            cv::Vec3f c = surf->pointer();
            status += QString(" ctr(%1,%2,%3)").arg(c[0],0,'f',0).arg(c[1],0,'f',0).arg(c[2],0,'f',0);
        }
    }
    status += QString(" z=%1").arg(_z_off, 0, 'f', 1);

    if (_composite_enabled) {
        QString m = QString::fromStdString(_composite_method);
        m[0] = m[0].toUpper();
        status += QString(" | %1(%2)").arg(m).arg(_composite_layers_front + _composite_layers_behind);
    }
    _lbl->setText(status);
}

// =========================================================================
// Overlay / intersection stubs
// =========================================================================

void CSliceViewer::invalidateVis()
{
    for (auto* item : slice_vis_items) { fScene->removeItem(item); delete item; }
    slice_vis_items.clear();
}

void CSliceViewer::updateAllOverlays() { emit overlaysUpdated(); }

void CSliceViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _overlay_groups[key] = items;
    for (auto* i : items) fScene->addItem(i);
}

void CSliceViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlay_groups.find(key);
    if (it == _overlay_groups.end()) return;
    for (auto* i : it->second) { fScene->removeItem(i); delete i; }
    _overlay_groups.erase(it);
}

void CSliceViewer::clearAllOverlayGroups()
{
    for (auto& [k, v] : _overlay_groups)
        for (auto* i : v) { fScene->removeItem(i); delete i; }
    _overlay_groups.clear();
}

void CSliceViewer::setIntersects(const std::set<std::string> &set) { _intersect_tgts = set; }
void CSliceViewer::renderIntersections() { /* TODO */ }
void CSliceViewer::invalidateIntersect(const std::string &) { /* TODO */ }

void CSliceViewer::setIntersectionOpacity(float o) { _intersectionOpacity = o; }
void CSliceViewer::setIntersectionThickness(float t) { _intersectionThickness = t; }
void CSliceViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    _highlightedSurfaceIds = std::unordered_set<std::string>(ids.begin(), ids.end());
}
void CSliceViewer::setSurfacePatchSamplingStride(int s) { _surfacePatchSamplingStride = std::max(1, s); }

void CSliceViewer::setSegmentationEditActive(bool active) { _segmentationEditActive = active; }

void CSliceViewer::adjustSurfaceOffset(float dn)
{
    _z_off += dn;
    invalidateAllTiles();
    renderVisible(true);
    emit overlaysUpdated();
    updateStatusLabel();
    _overlayUpdateTimer->stop();
    _overlayUpdateTimer->start();
}

void CSliceViewer::resetSurfaceOffsets()
{
    _z_off = 0.0f;
    invalidateAllTiles();
    renderVisible(true);
    emit overlaysUpdated();
    updateStatusLabel();
}

void CSliceViewer::fitSurfaceInView()
{
    auto surf = _surf_weak.lock();
    if (!surf || !fGraphicsView) return;

    Rect3D bbox;
    bool haveBounds = false;
    if (auto* q = dynamic_cast<QuadSurface*>(surf.get())) {
        bbox = q->bbox();
        haveBounds = true;
    }
    if (!haveBounds) {
        _scale = 1.0f;
        recalcScales();
        fGraphicsView->resetTransform();
        fGraphicsView->centerOn(0, 0);
        updateStatusLabel();
        return;
    }

    float bw = bbox.high[0] - bbox.low[0];
    float bh = bbox.high[1] - bbox.low[1];
    if (bw <= 0 || bh <= 0) return;

    QSize vs = fGraphicsView->viewport()->size();
    if (vs.width() <= 0 || vs.height() <= 0) return;

    float fit = 0.8f;
    float sx = vs.width() * fit / bw;
    float sy = vs.height() * fit / bh;
    _scale = round_scale(std::min(sx, sy));
    recalcScales();

    fGraphicsView->resetTransform();
    fGraphicsView->centerOn(0, 0);
    updateStatusLabel();
    invalidateAllTiles();
}

// =========================================================================
// Active segmentation handle
// =========================================================================

void CSliceViewer::markActiveSegmentationDirty()
{
    _activeSegHandleDirty = true;
    _activeSegHandle.reset();
}

const CSliceViewer::ActiveSegmentationHandle& CSliceViewer::activeSegmentationHandle() const
{
    if (!_activeSegHandleDirty) return _activeSegHandle;

    ActiveSegmentationHandle h;
    h.slotName = "segmentation";
    h.viewerIsSegmentationView = (_surf_name == "segmentation");
    h.accentColor = (_surf_name == "seg yz" ? COLOR_SEG_YZ
                     : _surf_name == "seg xz" ? COLOR_SEG_XZ : COLOR_SEG_XY);
    if (_surf_col) {
        auto s = _surf_col->surface(h.slotName);
        h.surface = dynamic_cast<QuadSurface*>(s.get());
    }
    if (!h.surface) h.slotName.clear();

    _activeSegHandle = h;
    _activeSegHandleDirty = false;
    return _activeSegHandle;
}

// =========================================================================
// Settings
// =========================================================================

void CSliceViewer::setCompositeEnabled(bool e)       { if (_composite_enabled!=e) { _composite_enabled=e; invalidateAllTiles(); renderVisible(true); updateStatusLabel(); } }
void CSliceViewer::setCompositeLayersInFront(int l)   { if (l>=0&&l<=100&&l!=_composite_layers_front) { _composite_layers_front=l; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeLayersBehind(int l)    { if (l>=0&&l<=100&&l!=_composite_layers_behind) { _composite_layers_behind=l; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeMethod(const std::string& m) {
    static const std::unordered_set<std::string> ok = {"max","mean","min","alpha","beerLambert"};
    if (m!=_composite_method && ok.count(m)) { _composite_method=m; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); updateStatusLabel(); } }
}
void CSliceViewer::setCompositeAlphaMin(int v)        { if (v>=0&&v<=255&&v!=_composite_alpha_min) { _composite_alpha_min=v; if (_composite_enabled&&_composite_method=="alpha") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeAlphaMax(int v)        { if (v>=0&&v<=255&&v!=_composite_alpha_max) { _composite_alpha_max=v; if (_composite_enabled&&_composite_method=="alpha") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeAlphaThreshold(int v)  { if (v>=0&&v<=10000&&v!=_composite_alpha_threshold) { _composite_alpha_threshold=v; if (_composite_enabled&&_composite_method=="alpha") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeMaterial(int v)        { if (v>=0&&v<=255&&v!=_composite_material) { _composite_material=v; if (_composite_enabled&&_composite_method=="alpha") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeReverseDirection(bool r){ if (r!=_composite_reverse_direction) { _composite_reverse_direction=r; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeBLExtinction(float v)  { if (v!=_composite_bl_extinction) { _composite_bl_extinction=v; if (_composite_enabled&&_composite_method=="beerLambert") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeBLEmission(float v)    { if (v!=_composite_bl_emission) { _composite_bl_emission=v; if (_composite_enabled&&_composite_method=="beerLambert") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setCompositeBLAmbient(float v)     { if (v!=_composite_bl_ambient) { _composite_bl_ambient=v; if (_composite_enabled&&_composite_method=="beerLambert") { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setLightingEnabled(bool e)         { if (e!=_lighting_enabled) { _lighting_enabled=e; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setLightAzimuth(float d)           { if (d!=_light_azimuth) { _light_azimuth=d; if (_composite_enabled&&_lighting_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setLightElevation(float d)         { if (d!=_light_elevation) { _light_elevation=d; if (_composite_enabled&&_lighting_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setLightDiffuse(float v)           { if (v!=_light_diffuse) { _light_diffuse=v; if (_composite_enabled&&_lighting_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setLightAmbient(float v)           { if (v!=_light_ambient) { _light_ambient=v; if (_composite_enabled&&_lighting_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setUseVolumeGradients(bool e)      { if (e!=_use_volume_gradients) { _use_volume_gradients=e; if (_composite_enabled&&_lighting_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setIsoCutoff(int v)                { v=std::clamp(v,0,255); if (v!=_iso_cutoff) { _iso_cutoff=v; invalidateAllTiles(); renderVisible(true); } }
void CSliceViewer::setResetViewOnSurfaceChange(bool r){ _resetViewOnSurfaceChange=r; }

void CSliceViewer::setPlaneCompositeEnabled(bool e)   { if (_plane_composite_enabled!=e) { _plane_composite_enabled=e; invalidateAllTiles(); renderVisible(true); } }
void CSliceViewer::setPlaneCompositeLayers(int f, int b) {
    f=std::max(0,f); b=std::max(0,b);
    if (_plane_composite_layers_front!=f || _plane_composite_layers_behind!=b) {
        _plane_composite_layers_front=f; _plane_composite_layers_behind=b;
        if (_plane_composite_enabled) { invalidateAllTiles(); renderVisible(true); }
    }
}

void CSliceViewer::setPostStretchValues(bool e)       { if (e!=_postStretchValues) { _postStretchValues=e; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setPostRemoveSmallComponents(bool e){ if (e!=_postRemoveSmallComponents) { _postRemoveSmallComponents=e; if (_composite_enabled) { invalidateAllTiles(); renderVisible(true); } } }
void CSliceViewer::setPostMinComponentSize(int s)     { s=std::clamp(s,1,100000); if (s!=_postMinComponentSize) { _postMinComponentSize=s; if (_composite_enabled&&_postRemoveSmallComponents) { invalidateAllTiles(); renderVisible(true); } } }

void CSliceViewer::setVolumeWindow(float low, float high)
{
    float cl = std::clamp(low, 0.f, 255.f);
    float ch = std::clamp(high, 0.f, 255.f);
    if (ch <= cl) ch = std::min(255.f, cl + 1.f);
    if (std::abs(cl - _baseWindowLow) < 1e-6f && std::abs(ch - _baseWindowHigh) < 1e-6f) return;
    _baseWindowLow = cl;
    _baseWindowHigh = ch;
    if (volume) { invalidateAllTiles(); renderVisible(true); }
}

void CSliceViewer::setBaseColormap(const std::string& id)
{
    if (_baseColormapId == id) return;
    _baseColormapId = id;
    if (volume) { invalidateAllTiles(); renderVisible(true); }
}

void CSliceViewer::setStretchValues(bool e)
{
    if (_stretchValues == e) return;
    _stretchValues = e;
    if (volume) { invalidateAllTiles(); renderVisible(true); }
}

const std::vector<CSliceViewer::OverlayColormapEntry>& CSliceViewer::overlayColormapEntries()
{
    static std::vector<OverlayColormapEntry> entries;
    static bool initialized = false;
    if (!initialized) {
        const auto& shared = volume_viewer_cmaps::entries();
        entries.reserve(shared.size());
        for (const auto& e : shared)
            entries.push_back({e.label, e.id});
        initialized = true;
    }
    return entries;
}

cv::Mat_<uint8_t> CSliceViewer::renderCompositeForSurface(std::shared_ptr<QuadSurface>, cv::Size)
{
    // TODO: implement if needed
    return {};
}

// =========================================================================
// Composite params helper
// =========================================================================

CompositeParams CSliceViewer::currentCompositeParams() const
{
    CompositeParams p;
    p.method = _composite_method;
    p.alphaMin = _composite_alpha_min / 255.0f;
    p.alphaMax = _composite_alpha_max / 255.0f;
    p.alphaOpacity = _composite_material / 255.0f;
    p.alphaCutoff = _composite_alpha_threshold / 10000.0f;
    p.blExtinction = _composite_bl_extinction;
    p.blEmission = _composite_bl_emission;
    p.blAmbient = _composite_bl_ambient;
    p.lightingEnabled = _lighting_enabled;
    p.lightAzimuth = _light_azimuth;
    p.lightElevation = _light_elevation;
    p.lightDiffuse = _light_diffuse;
    p.lightAmbient = _light_ambient;
    p.isoCutoff = static_cast<uint8_t>(_iso_cutoff);
    return p;
}
