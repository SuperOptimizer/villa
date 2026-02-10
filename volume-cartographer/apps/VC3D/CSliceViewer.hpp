#pragma once

#include <QWidget>
#include <QPointF>
#include <QRectF>
#include <QColor>
#include <QString>
#include <QList>
#include <QImage>
#include <QEvent>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>

#include "overlays/ViewerOverlayControllerBase.hpp"
#include "vc/ui/VCCollection.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/Slicing.hpp"

class QGraphicsScene;
class QGraphicsItem;
class QGraphicsPixmapItem;
class QLabel;
class QTimer;
class ViewerManager;

// ---------------------------------------------------------------------------
// CSliceViewer -- tile-based, non-blocking volume slice renderer
//
// Drop-in replacement for CVolumeViewer.  Instead of rendering a single large
// pixmap on the main thread, it:
//   1. Breaks the viewport into independent 256x256 tiles
//   2. Renders tiles on a background thread (with OpenMP inside each tile)
//   3. Delivers finished tiles to the main thread via a poll timer
//   4. Supports multi-resolution scale-space: coarse tiles render first,
//      fine tiles overlay later
//   5. Tiles persist when panning -- only new edge tiles need rendering
//   6. IO (chunk cache loads) happens naturally on the render thread but
//      does NOT block the main thread
// ---------------------------------------------------------------------------
class CSliceViewer : public QWidget
{
    Q_OBJECT

public:
    CSliceViewer(CSurfaceCollection *col, ViewerManager* manager, QWidget* parent = nullptr);
    ~CSliceViewer() override;

    static constexpr int TILE_SIZE = 256;

    // ---- Core interface (same as CVolumeViewer) ----
    void setCache(ChunkCache<uint8_t> *cache);
    void setPointCollection(VCCollection* point_collection);
    void setSurface(const std::string &name);
    void renderVisible(bool force = false);
    void invalidateVis();

    std::string surfName() const { return _surf_name; }
    void recalcScales();

    // Composite view methods
    void setCompositeEnabled(bool enabled);
    void setCompositeLayersInFront(int layers);
    void setCompositeLayersBehind(int layers);
    void setCompositeMethod(const std::string& method);
    void setCompositeAlphaMin(int value);
    void setCompositeAlphaMax(int value);
    void setCompositeAlphaThreshold(int value);
    void setCompositeMaterial(int value);
    void setCompositeReverseDirection(bool reverse);
    void setCompositeBLExtinction(float value);
    void setCompositeBLEmission(float value);
    void setCompositeBLAmbient(float value);
    void setLightingEnabled(bool enabled);
    void setLightAzimuth(float degrees);
    void setLightElevation(float degrees);
    void setLightDiffuse(float value);
    void setLightAmbient(float value);
    void setUseVolumeGradients(bool enabled);
    void setIsoCutoff(int value);
    void setResetViewOnSurfaceChange(bool reset);

    // Plane composite
    void setPlaneCompositeEnabled(bool enabled);
    void setPlaneCompositeLayers(int front, int behind);
    bool isPlaneCompositeEnabled() const { return _plane_composite_enabled; }
    int planeCompositeLayersFront() const { return _plane_composite_layers_front; }
    int planeCompositeLayersBehind() const { return _plane_composite_layers_behind; }

    // Postprocessing
    void setPostStretchValues(bool enabled);
    bool postStretchValues() const { return _postStretchValues; }
    void setPostRemoveSmallComponents(bool enabled);
    bool postRemoveSmallComponents() const { return _postRemoveSmallComponents; }
    void setPostMinComponentSize(int size);
    int postMinComponentSize() const { return _postMinComponentSize; }

    // Query state
    bool isCompositeEnabled() const { return _composite_enabled; }
    std::shared_ptr<Volume> currentVolume() const { return volume; }
    ChunkCache<uint8_t>* chunkCachePtr() const { return cache; }
    int datasetScaleIndex() const { return _ds_sd_idx; }
    float datasetScaleFactor() const { return _ds_scale; }
    VCCollection* pointCollection() const { return _point_collection; }
    uint64_t highlightedPointId() const { return _highlighted_point_id; }
    uint64_t selectedPointId() const { return _selected_point_id; }
    uint64_t selectedCollectionId() const { return _selected_collection_id; }
    bool isPointDragActive() const { return _dragged_point_id != 0; }
    const std::vector<ViewerOverlayControllerBase::PathPrimitive>& drawingPaths() const { return _paths; }

    void setShowDirectionHints(bool on) { _showDirectionHints = on; }
    bool isShowDirectionHints() const { return _showDirectionHints; }
    void setShowSurfaceNormals(bool on) { _showSurfaceNormals = on; }
    bool isShowSurfaceNormals() const { return _showSurfaceNormals; }
    void setNormalArrowLengthScale(float scale) { _normalArrowLengthScale = scale; }
    float normalArrowLengthScale() const { return _normalArrowLengthScale; }
    void setNormalMaxArrows(int maxArrows) { _normalMaxArrows = maxArrows; }
    int normalMaxArrows() const { return _normalMaxArrows; }

    void adjustSurfaceOffset(float dn);
    void resetSurfaceOffsets();
    float normalOffset() const { return _z_off; }

    void updateStatusLabel();
    void setSegmentationEditActive(bool active);
    void fitSurfaceInView();
    void updateAllOverlays();

    bool isWindowMinimized() const;
    bool eventFilter(QObject* watched, QEvent* event) override;

    void setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items);
    void clearOverlayGroup(const std::string& key);
    void clearAllOverlayGroups();

    float getCurrentScale() const { return _scale; }
    cv::Vec3f sceneToVolume(const QPointF& scenePoint) const;
    QPointF volumePointToScene(const cv::Vec3f& vol_point) { return volumeToScene(vol_point); }
    QPointF lastScenePosition() const { return _lastScenePos; }
    float dsScale() const { return _ds_scale; }
    Surface* currentSurface() const;

    void setIntersects(const std::set<std::string> &set);
    void renderIntersections();
    void invalidateIntersect(const std::string &name = "");

    void setBBoxMode(bool /*enabled*/) {}
    bool isBBoxMode() const { return false; }
    QuadSurface* makeBBoxFilteredSurfaceFromSceneRect(const QRectF& /*sceneRect*/) { return nullptr; }
    auto selections() const -> std::vector<std::pair<QRectF, QColor>> { return {}; }
    std::optional<QRectF> activeBBoxSceneRect() const { return std::nullopt; }
    void clearSelections() {}

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }
    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    void setSurfacePatchSamplingStride(int stride);
    int surfacePatchSamplingStride() const { return _surfacePatchSamplingStride; }

    void setOverlayVolume(std::shared_ptr<Volume> vol) { _overlayVolume = vol; }
    std::shared_ptr<Volume> overlayVolume() const { return _overlayVolume; }
    void setOverlayOpacity(float opacity) { _overlayOpacity = opacity; }
    float overlayOpacity() const { return _overlayOpacity; }
    void setOverlayColormap(const std::string& id) { _overlayColormapId = id; }
    const std::string& overlayColormap() const { return _overlayColormapId; }
    void setOverlayThreshold(float threshold) { _overlayWindowLow = threshold; }
    float overlayThreshold() const { return _overlayWindowLow; }
    void setOverlayWindow(float low, float high) { _overlayWindowLow = low; _overlayWindowHigh = high; }
    float overlayWindowLow() const { return _overlayWindowLow; }
    float overlayWindowHigh() const { return _overlayWindowHigh; }

    void setSegmentationCursorMirroring(bool enabled) { _mirrorCursorToSegmentation = enabled; }
    bool segmentationCursorMirroringEnabled() const { return _mirrorCursorToSegmentation; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _baseWindowLow; }
    float volumeWindowHigh() const { return _baseWindowHigh; }

    struct ActiveSegmentationHandle {
        QuadSurface* surface{nullptr};
        std::string slotName;
        QColor accentColor;
        bool viewerIsSegmentationView{false};
        bool valid() const { return surface != nullptr; }
        explicit operator bool() const { return valid(); }
        void reset() { surface = nullptr; slotName.clear(); accentColor = QColor(); viewerIsSegmentationView = false; }
    };
    const ActiveSegmentationHandle& activeSegmentationHandle() const;

    void setBaseColormap(const std::string& colormapId);
    const std::string& baseColormap() const { return _baseColormapId; }
    void setStretchValues(bool enabled);
    bool stretchValues() const { return _stretchValues; }

    void setSurfaceOverlayEnabled(bool enabled) { _surfaceOverlayEnabled = enabled; }
    bool surfaceOverlayEnabled() const { return _surfaceOverlayEnabled; }
    void setSurfaceOverlays(const std::map<std::string, cv::Vec3b>& overlays) { _surfaceOverlays = overlays; }
    const std::map<std::string, cv::Vec3b>& surfaceOverlays() const { return _surfaceOverlays; }
    void setSurfaceOverlapThreshold(float t) { _surfaceOverlapThreshold = t; }
    float surfaceOverlapThreshold() const { return _surfaceOverlapThreshold; }

    struct OverlayColormapEntry {
        QString label;
        std::string id;
    };
    static const std::vector<OverlayColormapEntry>& overlayColormapEntries();

    cv::Mat_<uint8_t> renderCompositeForSurface(std::shared_ptr<QuadSurface> surface, cv::Size outputSize);

    CVolumeViewerView* fGraphicsView;

public slots:
    void OnVolumeChanged(std::shared_ptr<Volume> vol);
    void onVolumeClicked(QPointF scene_loc, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanRelease(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onPanStart(Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void onCollectionSelected(uint64_t collectionId);
    void onSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void onPOIChanged(std::string name, POI *poi);
    void onScrolled();
    void onResized();
    void onZoom(int steps, QPointF scene_point, Qt::KeyboardModifiers modifiers);
    void adjustZoomByFactor(float factor);
    void onCursorMove(QPointF);
    void onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths);
    void onPointSelected(uint64_t pointId);
    void onMousePress(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onMouseMove(QPointF scene_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void onMouseRelease(QPointF scene_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void onVolumeClosing();
    void onSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);
    void onKeyRelease(int key, Qt::KeyboardModifiers modifiers);
    void onDrawingModeActive(bool active, float brushSize = 3.0f, bool isSquare = false);

signals:
    void sendVolumeClicked(cv::Vec3f vol_loc, cv::Vec3f normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers);
    void sendZSliceChanged(int z_value);
    void sendMousePressVolume(cv::Vec3f vol_loc, cv::Vec3f normal, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendMouseMoveVolume(cv::Vec3f vol_loc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers);
    void sendMouseReleaseVolume(cv::Vec3f vol_loc, Qt::MouseButton button, Qt::KeyboardModifiers modifiers);
    void sendCollectionSelected(uint64_t collectionId);
    void pointSelected(uint64_t pointId);
    void pointClicked(uint64_t pointId);
    void overlaysUpdated();
    void sendSegmentationRadiusWheel(int steps, QPointF scenePoint, cv::Vec3f worldPos);

protected:
    QPointF volumeToScene(const cv::Vec3f& vol_point);

    // =====================================================================
    // Tile system
    // =====================================================================
    struct TileKey {
        int tx, ty;
        bool operator==(const TileKey& o) const { return tx == o.tx && ty == o.ty; }
    };
    struct TileKeyHash {
        size_t operator()(const TileKey& k) const {
            return std::hash<int64_t>()((static_cast<int64_t>(k.tx) << 32) | static_cast<uint32_t>(k.ty));
        }
    };

    // A tile living in the QGraphicsScene
    struct SceneTile {
        QGraphicsPixmapItem* item = nullptr;   // owned by the scene
        int renderedDsIdx = -1;                // pyramid level used
        uint64_t generation = 0;               // view generation when last rendered
    };

    std::unordered_map<TileKey, SceneTile, TileKeyHash> _tiles;
    std::atomic<uint64_t> _viewGeneration{0};

    // ---- Render request sent to the worker thread ----
    struct TileRenderRequest {
        TileKey key;
        cv::Rect sceneRect;     // scene-space rectangle for this tile
        float scale;            // _scale at time of request
        int dsIdx;              // pyramid level to sample
        float dsScale;          // pow(2, -dsIdx)
        float z_off;
        uint64_t generation;    // for staleness check
        // Surface parameters (pre-computed on main thread)
        bool isPlane;
        cv::Vec3f planeOrigin;
        cv::Vec3f planeNormal;
        cv::Vec3f planeVx;
        cv::Vec3f planeVy;
        // For QuadSurface
        cv::Vec3f quadPtr;      // pre-computed surface pointer for tile center
        std::shared_ptr<Surface> surfRef; // prevent surface destruction during render
        // Composite
        bool useComposite;
        bool usePlaneComposite;
        CompositeParams compositeParams;
        int compositeLayersFront;
        int compositeLayersBehind;
        bool compositeReverse;
        // Window/level
        float windowLow;
        float windowHigh;
        bool stretchValues;
        std::string colormapId;
        int isoCutoff;
        bool useFastInterpolation;
    };

    // ---- Result delivered back to the main thread ----
    struct TileResult {
        TileKey key;
        QImage image;
        int dsIdx;
        uint64_t generation;
    };

    // Render worker thread
    std::thread _renderThread;
    std::mutex _renderQueueMutex;
    std::condition_variable _renderQueueCV;
    std::deque<TileRenderRequest> _renderQueue;
    std::atomic<bool> _shutdown{false};

    // Results (consumed on main thread via timer)
    std::mutex _resultMutex;
    std::vector<TileResult> _readyTiles;
    QTimer* _resultPollTimer = nullptr;

    void renderWorkerLoop();
    QImage renderTile(const TileRenderRequest& req);
    void updateVisibleTiles();
    void consumeReadyTiles();
    void removeOffscreenTiles(const QRectF& visibleBBox);
    void invalidateAllTiles();
    void enqueueTilesForViewport(const QRectF& bbox, int dsIdx, float dsScale);
    TileRenderRequest makeTileRequest(const TileKey& key, int dsIdx, float dsScale);
    CompositeParams currentCompositeParams() const;

    // =====================================================================
    // Data  (mirrors CVolumeViewer)
    // =====================================================================
    QGraphicsScene* fScene = nullptr;
    bool fSkipImageFormatConv = false;

    std::shared_ptr<Volume> volume;
    std::weak_ptr<Surface> _surf_weak;
    cv::Vec3f _ptr = cv::Vec3f(0,0,0);
    cv::Vec2f _vis_center = {0,0};
    std::string _surf_name;

    ChunkCache<uint8_t> *cache = nullptr;
    float _scale = 0.5f;
    float _scene_scale = 1.0f;
    float _ds_scale = 0.5f;
    int _ds_sd_idx = 1;
    float _max_scale = 1.0f;
    float _min_scale = 1.0f;

    QLabel *_lbl = nullptr;
    float _z_off = 0.0f;
    QPointF _lastScenePos;

    // Composite
    bool _composite_enabled = false;
    int _composite_layers_front = 8;
    int _composite_layers_behind = 0;
    std::string _composite_method = "max";
    int _composite_alpha_min = 170;
    int _composite_alpha_max = 220;
    int _composite_alpha_threshold = 9950;
    int _composite_material = 230;
    bool _composite_reverse_direction = false;
    float _composite_bl_extinction = 1.5f;
    float _composite_bl_emission = 1.5f;
    float _composite_bl_ambient = 0.1f;
    bool _lighting_enabled = false;
    float _light_azimuth = 45.0f;
    float _light_elevation = 45.0f;
    float _light_diffuse = 0.7f;
    float _light_ambient = 0.3f;
    bool _use_volume_gradients = false;
    int _iso_cutoff = 0;

    // Plane composite
    bool _plane_composite_enabled = false;
    int _plane_composite_layers_front = 4;
    int _plane_composite_layers_behind = 4;

    QGraphicsItem *_center_marker = nullptr;
    QGraphicsItem *_cursor = nullptr;
    std::vector<QGraphicsItem*> slice_vis_items;

    std::set<std::string> _intersect_tgts = {"visible_segmentation"};
    std::unordered_map<std::string, SurfacePatchIndex::SurfacePtr> _cachedIntersectSurfaces;
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _intersect_items;
    std::unordered_map<std::string, std::vector<IntersectionLine>> _cachedIntersectionLines;
    float _cachedIntersectionScale = 0.0f;
    std::vector<SurfacePatchIndex::TriangleCandidate> _triangleCandidates;
    std::unordered_map<SurfacePatchIndex::SurfacePtr, std::vector<size_t>> _trianglesBySurface;

    CSurfaceCollection *_surf_col = nullptr;
    ViewerManager* _viewerManager = nullptr;
    VCCollection* _point_collection = nullptr;

    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::unordered_set<std::string> _highlightedSurfaceIds;
    std::unordered_map<std::string, size_t> _surfaceColorAssignments;
    size_t _nextColorIndex{0};

    uint64_t _highlighted_point_id = 0;
    uint64_t _selected_point_id = 0;
    uint64_t _dragged_point_id = 0;
    uint64_t _selected_collection_id = 0;

    std::vector<ViewerOverlayControllerBase::PathPrimitive> _paths;
    std::unordered_map<std::string, std::vector<QGraphicsItem*>> _overlay_groups;

    bool _drawingModeActive = false;
    float _brushSize = 3.0f;
    bool _brushIsSquare = false;
    bool _resetViewOnSurfaceChange = true;
    bool _showDirectionHints = true;
    bool _showSurfaceNormals = false;
    float _normalArrowLengthScale = 1.0f;
    int _normalMaxArrows = 32;
    bool _segmentationEditActive = false;

    int _downscale_override = 0;
    QTimer* _overlayUpdateTimer = nullptr;
    bool _useFastInterpolation = false;

    std::shared_ptr<Volume> _overlayVolume;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    float _baseWindowLow{0.0f};
    float _baseWindowHigh{255.0f};
    bool _mirrorCursorToSegmentation{false};

    int _surfacePatchSamplingStride{1};

    void markActiveSegmentationDirty();
    mutable ActiveSegmentationHandle _activeSegHandle;
    mutable bool _activeSegHandleDirty{true};

    std::string _baseColormapId;
    bool _stretchValues{false};
    bool _surfaceOverlayEnabled{false};
    std::map<std::string, cv::Vec3b> _surfaceOverlays;
    float _surfaceOverlapThreshold{5.0f};

    bool _postStretchValues{false};
    bool _postRemoveSmallComponents{false};
    int _postMinComponentSize{50};
    bool _dirtyWhileMinimized = false;
};
