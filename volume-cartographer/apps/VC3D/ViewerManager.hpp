#pragma once

#include <QObject>
#include <QString>
#include <QFutureWatcher>

class QTimer;

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "vc/core/util/SurfacePatchIndex.hpp"

class QMdiArea;
class CChunkedVolumeViewer;
class CState;
class QWidget;
class VCCollection;
class SegmentationOverlayController;
class PointsOverlayController;
class RawPointsOverlayController;
class PathsOverlayController;
class BBoxOverlayController;
class VectorOverlayController;
class VolumeOverlayController;
class SegmentationModule;
class ViewerOverlayControllerBase;
class VolumeViewerBase;
class Volume;
class Surface;
class QuadSurface;

class ViewerManager : public QObject
{
    Q_OBJECT

public:
    enum class ViewerRole {
        Standard,
        Annotation,
    };

    ViewerManager(CState* state,
                  VCCollection* points,
                  QObject* parent = nullptr);

    VolumeViewerBase* createViewer(const std::string& surfaceName,
                                   const QString& title,
                                   QMdiArea* mdiArea,
                                   ViewerRole role = ViewerRole::Standard);
    VolumeViewerBase* createViewerInWidget(const std::string& surfaceName,
                                           QWidget* parent,
                                           ViewerRole role = ViewerRole::Standard);
    void unregisterViewer(VolumeViewerBase* viewer);

    const std::vector<VolumeViewerBase*>& baseViewers() const { return _baseViewers; }

    void setSegmentationOverlay(SegmentationOverlayController* overlay);
    SegmentationOverlayController* segmentationOverlay() const { return _segmentationOverlay; }
    void setSegmentationEditActive(bool active);
    void setSegmentationModule(SegmentationModule* module);
    void setPointsOverlay(PointsOverlayController* overlay);
    void setRawPointsOverlay(RawPointsOverlayController* overlay);
    RawPointsOverlayController* rawPointsOverlay() const { return _rawPointsOverlay; }
    void setPathsOverlay(PathsOverlayController* overlay);
    void setBBoxOverlay(BBoxOverlayController* overlay);
    void setVectorOverlay(VectorOverlayController* overlay);
    void setVolumeOverlay(VolumeOverlayController* overlay);

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }

    void setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId);
    std::shared_ptr<Volume> overlayVolume() const { return _overlayVolume; }
    const std::string& overlayVolumeId() const { return _overlayVolumeId; }

    void setOverlayOpacity(float opacity);
    float overlayOpacity() const { return _overlayOpacity; }

    void setOverlayColormap(const std::string& colormapId);
    const std::string& overlayColormap() const { return _overlayColormapId; }
    void setOverlayThreshold(float threshold);
    float overlayThreshold() const { return _overlayWindowLow; }

    void setOverlayWindow(float low, float high);
    float overlayWindowLow() const { return _overlayWindowLow; }
    float overlayWindowHigh() const { return _overlayWindowHigh; }

    void setVolumeWindow(float low, float high);
    float volumeWindowLow() const { return _volumeWindowLow; }
    float volumeWindowHigh() const { return _volumeWindowHigh; }

    void setSurfacePatchSamplingStride(int stride, bool userInitiated = true);
    int surfacePatchSamplingStride() const { return _surfacePatchSamplingStride; }
    void setIntersectionMaxSurfaces(int limit);
    int intersectionMaxSurfaces() const { return _intersectionMaxSurfaces; }
    void primeSurfacePatchIndicesAsync();
    void resetStrideUserOverride() {}

    bool resetDefaultFor(VolumeViewerBase* viewer) const;
    void setResetDefaultFor(VolumeViewerBase* viewer, bool value);

    void setSegmentationCursorMirroring(bool enabled);
    bool segmentationCursorMirroring() const { return _mirrorCursorToSegmentation; }
    void broadcastLinkedCursor(VolumeViewerBase* source,
                               const std::optional<cv::Vec3f>& point);

    void setSliceStepSize(int size);
    int sliceStepSize() const { return _sliceStepSize; }

    void forEachBaseViewer(const std::function<void(VolumeViewerBase*)>& fn) const;

    // ---- global render clock (one heartbeat for the whole app) --------------
    // Any change that should produce a frame (camera moved, surface changed,
    // resized, a chunk landed) calls requestGlobalRender(): it sets one
    // idempotent dirty flag, coalescing N events into a single render per tick.
    // onGlobalTick() (the one QTimer) then, per tick: thaw -> freeze each
    // distinct chunk cache ONCE -> drive every dirty viewer's render (worker) ->
    // the worker stages its framebuffer + the main thread flips it. Replaces the
    // per-viewer render/status/intersection/resize debounce timers.
    void requestGlobalRender();

    void setIntersectionThickness(float thickness);
    float intersectionThickness() const { return _intersectionThickness; }
    void setHighlightedSurfaceIds(const std::vector<std::string>& ids);
    SurfacePatchIndex* surfacePatchIndex();
    SurfacePatchIndex* surfacePatchIndexIfReady();
    void refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface);
    void refreshSurfacePatchIndex(const SurfacePatchIndex::SurfacePtr& surface, const cv::Rect& changedRegion);

    // Stop maintaining the SurfacePatchIndex. Any subsequent
    // surfaceWillBeDeleted signals will be ignored instead of triggering
    // an O(N) rtree removal. Intended to be called from the close path so
    // that tearing down thousands of cells doesn't block app exit.
    void beginShutdown() noexcept { _shuttingDown = true; }

signals:
    void baseViewerCreated(VolumeViewerBase* viewer);
    void baseViewerClosing(VolumeViewerBase* viewer);
    void overlayWindowChanged(float low, float high);
    void volumeWindowChanged(float low, float high);
    void overlayVolumeAvailabilityChanged(bool hasOverlay);
    void samplingStrideChanged(int stride);
    void sliceStepSizeChanged(int size);

private slots:
    void onGlobalTick();
    void handleSurfacePatchIndexPrimeFinished();
    void handleSurfacePatchIndexTaskFinished();
    void handleSurfaceChanged(std::string name, std::shared_ptr<Surface> surf, bool isEditUpdate = false);
    void handleSurfaceWillBeDeleted(std::string name, std::shared_ptr<Surface> surf);

private:
    enum class SurfacePatchIndexTaskType {
        Update,
        Remove,
    };

    struct SurfacePatchIndexTask {
        SurfacePatchIndexTaskType type{SurfacePatchIndexTaskType::Update};
        std::string id;
        SurfacePatchIndex::SurfacePtr surface;
    };

    struct SurfacePatchIndexTaskResult {
        SurfacePatchIndexTaskType type{SurfacePatchIndexTaskType::Update};
        std::string id;
        SurfacePatchIndex::SurfacePtr surface;
        bool success{false};
    };

    void registerOverlay(ViewerOverlayControllerBase* overlay);
    VolumeViewerBase* initializeChunkedViewer(CChunkedVolumeViewer* chunkedViewer,
                                              const std::string& surfaceName,
                                              ViewerRole role);
    bool updateSurfacePatchIndexForSurface(const SurfacePatchIndex::SurfacePtr& quad, bool isEditUpdate);
    void queueSurfacePatchIndexTask(SurfacePatchIndexTask task);
    void startNextSurfacePatchIndexTask();

    CState* _state;
    VCCollection* _points;
    SegmentationOverlayController* _segmentationOverlay{nullptr};
    PointsOverlayController* _pointsOverlay{nullptr};
    RawPointsOverlayController* _rawPointsOverlay{nullptr};
    PathsOverlayController* _pathsOverlay{nullptr};
    BBoxOverlayController* _bboxOverlay{nullptr};
    VectorOverlayController* _vectorOverlay{nullptr};
    // All overlay controllers that should be attached/detached from viewers.
    // Populated by the set*Overlay() methods. Does NOT include VolumeOverlayController
    // (which is not a ViewerOverlayControllerBase subclass).
    std::vector<ViewerOverlayControllerBase*> _allOverlays;
    bool _segmentationEditActive{false};
    SegmentationModule* _segmentationModule{nullptr};
    std::vector<VolumeViewerBase*> _baseViewers;
    std::unordered_map<VolumeViewerBase*, bool> _resetDefaults;

    // Global render clock. _globalClock fires every 33ms; onGlobalTick renders
    // only when _globalRenderPending was set since the last tick.
    QTimer* _globalClock{nullptr};
    bool _globalRenderPending{false};
    float _intersectionOpacity{1.0f};
    float _intersectionThickness{0.0f};
    std::shared_ptr<Volume> _overlayVolume;
    std::string _overlayVolumeId;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    float _volumeWindowLow{0.0f};
    float _volumeWindowHigh{255.0f};
    bool _mirrorCursorToSegmentation{false};
    int _sliceStepSize{1};
    int _surfacePatchSamplingStride{1};
    std::atomic<bool> _shuttingDown{false};
    int _intersectionMaxSurfaces{0};  // 0 = unlimited

    VolumeOverlayController* _volumeOverlay{nullptr};
    SurfacePatchIndex _surfacePatchIndex;
    bool _surfacePatchIndexNeedsRebuild{true};
    // Use string IDs for surface tracking to avoid dangling pointers in async operations
    std::unordered_set<std::string> _indexedSurfaceIds;
    std::vector<std::string> _pendingSurfacePatchIndexSurfaceIds;
    std::vector<SurfacePatchIndexTask> _pendingSurfacePatchIndexTasks;
    std::vector<SurfacePatchIndexTask> _surfacesQueuedDuringRebuild;
    QFutureWatcher<std::shared_ptr<SurfacePatchIndex>>* _surfacePatchIndexWatcher{nullptr};
    QFutureWatcher<SurfacePatchIndexTaskResult>* _surfacePatchIndexTaskWatcher{nullptr};
    // Dedicated pool for the patch-index rtree rebuild. QtConcurrent::run defaults
    // to the GLOBAL pool, which the volume render bands (mc_render_points_par) also
    // use -- so an rtree rebuild stole render workers and inflated frame latency.
    // Isolating it here keeps the render pool for rendering.
    QThreadPool* _surfacePatchIndexPool{nullptr};

    // Surfaces currently pinned in the LRU as "highlighted/visible".
    // We track them so we can unpin the right set when highlights change.
    std::vector<std::shared_ptr<QuadSurface>> _pinnedHighlightSurfaces;

    void rebuildSurfacePatchIndexIfNeeded();

public:
    // Async-intersection mutual exclusion. A viewer's plane-intersection query reads
    // _surfacePatchIndex on a worker thread; while any such read is in flight the
    // index must not be mutated (rebuild swap / updateSurface / clear) or the worker
    // tears. Mutation sites call indexMutationGuarded(): if a read is in flight it
    // marks the index dirty and returns true (caller defers); the deferred work
    // re-runs once reads drain. Begin/end bracket each worker read (main thread).
    void beginIndexRead() { ++_indexReadsInFlight; }
    void endIndexRead();
    bool indexReadInFlight() const { return _indexReadsInFlight > 0; }
private:
    int _indexReadsInFlight{0};
    // A rebuilt index that finished while a read was in flight: held here and
    // swapped in by endIndexRead() once reads drain (the swap would tear an
    // in-flight worker read otherwise).
    std::shared_ptr<SurfacePatchIndex> _deferredIndexSwap;
    std::vector<std::string> _deferredIndexSwapIds;

};
