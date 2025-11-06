#pragma once

#include <QObject>
#include <QString>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include "vc/core/util/Surface.hpp"

class QMdiArea;
class QGraphicsItem;
class CVolumeViewer;
class CSurfaceCollection;
class VCCollection;
class SegmentationOverlayController;
class PointsOverlayController;
class PathsOverlayController;
class BBoxOverlayController;
class VectorOverlayController;
class VolumeOverlayController;
class ChunkCache;
class SegmentationModule;
class Volume;

class ViewerManager : public QObject
{
    Q_OBJECT

public:

    ViewerManager(CSurfaceCollection* surfaces,
                  VCCollection* points,
                  ChunkCache* cache,
                  QObject* parent = nullptr);

    CVolumeViewer* createViewer(const std::string& surfaceName,
                                const QString& title,
                                QMdiArea* mdiArea);

    const std::vector<CVolumeViewer*>& viewers() const { return _viewers; }

    void setSegmentationOverlay(SegmentationOverlayController* overlay);
    void setSegmentationEditActive(bool active);
    void setSegmentationModule(SegmentationModule* module);
    void setPointsOverlay(PointsOverlayController* overlay);
    void setPathsOverlay(PathsOverlayController* overlay);
    void setBBoxOverlay(BBoxOverlayController* overlay);
    void setVectorOverlay(VectorOverlayController* overlay);
    void setVolumeOverlay(VolumeOverlayController* overlay);

    void setIntersectionOpacity(float opacity);
    float intersectionOpacity() const { return _intersectionOpacity; }

    void setMaxIntersections(int maxIntersections);
    int maxIntersections() const { return _maxIntersections; }

    void setIntersectionLineWidth(int lineWidth);
    int intersectionLineWidth() const { return _intersectionLineWidth; }

    void setHighlightedSegments(const std::vector<std::string>& segments);
    const std::vector<std::string>& highlightedSegments() const { return _highlightedSegments; }

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

    bool resetDefaultFor(CVolumeViewer* viewer) const;
    void setResetDefaultFor(CVolumeViewer* viewer, bool value);

    void forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const;

    // Cached candidate selection for intersection rendering
    struct CandidateInfo {
        std::string key;
        float distance;
    };
    std::vector<CandidateInfo> getCachedCandidates(
        const cv::Vec3f& referenceCenter,
        const std::set<std::string>& intersectTargets,
        const std::unordered_map<std::string, std::vector<QGraphicsItem*>>& alreadyRendered,
        bool useHighlightedSegments);
    void invalidateCandidateCache();

    // Global spatial index for fast segment culling
    void buildGlobalSpatialIndex();
    void invalidateGlobalSpatialIndex();
    std::vector<std::string> querySegmentsNearPlane(const Rect3D& planeBounds) const;

signals:
    void viewerCreated(CVolumeViewer* viewer);
    void overlayWindowChanged(float low, float high);
    void volumeWindowChanged(float low, float high);
    void overlayVolumeAvailabilityChanged(bool hasOverlay);

private:
    CSurfaceCollection* _surfaces;
    VCCollection* _points;
    ChunkCache* _chunkCache;
    SegmentationOverlayController* _segmentationOverlay{nullptr};
    PointsOverlayController* _pointsOverlay{nullptr};
    PathsOverlayController* _pathsOverlay{nullptr};
    BBoxOverlayController* _bboxOverlay{nullptr};
    VectorOverlayController* _vectorOverlay{nullptr};
    bool _segmentationEditActive{false};
    SegmentationModule* _segmentationModule{nullptr};
    std::vector<CVolumeViewer*> _viewers;
    std::unordered_map<CVolumeViewer*, bool> _resetDefaults;
    float _intersectionOpacity{1.0f};
    int _maxIntersections{250};
    int _intersectionLineWidth{2};
    std::vector<std::string> _highlightedSegments;
    std::shared_ptr<Volume> _overlayVolume;
    std::string _overlayVolumeId;
    float _overlayOpacity{0.5f};
    std::string _overlayColormapId;
    float _overlayWindowLow{0.0f};
    float _overlayWindowHigh{255.0f};
    float _volumeWindowLow{0.0f};
    float _volumeWindowHigh{255.0f};

    VolumeOverlayController* _volumeOverlay{nullptr};

    // Candidate cache for intersection rendering
    cv::Vec3f _cachedReferenceCenter{0, 0, 0};
    std::vector<CandidateInfo> _cachedCandidates;
    bool _candidateCacheValid{false};

    // Global spatial index covering entire volume (cells of 500 voxels)
    MultiSpatialIndex _globalSpatialIndex{500.0f};
};
