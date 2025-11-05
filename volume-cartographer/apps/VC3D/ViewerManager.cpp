#include "ViewerManager.hpp"

#include "VCSettings.hpp"
#include "CVolumeViewer.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "CSurfaceCollection.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/Surface.hpp"

#include <QMdiArea>
#include <QMdiSubWindow>
#include <QSettings>
#include <algorithm>
#include <chrono>
#include <cmath>

ViewerManager::ViewerManager(CSurfaceCollection* surfaces,
                             VCCollection* points,
                             ChunkCache* cache,
                             QObject* parent)
    : QObject(parent)
    , _surfaces(surfaces)
    , _points(points)
    , _chunkCache(cache)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const int savedOpacityPercent = settings.value("viewer/intersection_opacity", 100).toInt();
    const float normalized = static_cast<float>(savedOpacityPercent) / 100.0f;
    _intersectionOpacity = std::clamp(normalized, 0.0f, 1.0f);

    const float storedBaseLow = settings.value("viewer/base_window_low", 0.0f).toFloat();
    const float storedBaseHigh = settings.value("viewer/base_window_high", 255.0f).toFloat();
    _volumeWindowLow = std::clamp(storedBaseLow, 0.0f, 255.0f);
    const float minHigh = std::min(_volumeWindowLow + 1.0f, 255.0f);
    _volumeWindowHigh = std::clamp(storedBaseHigh, minHigh, 255.0f);
}

CVolumeViewer* ViewerManager::createViewer(const std::string& surfaceName,
                                           const QString& title,
                                           QMdiArea* mdiArea)
{
    if (!mdiArea || !_surfaces) {
        return nullptr;
    }

    auto* viewer = new CVolumeViewer(_surfaces, mdiArea);
    auto* win = mdiArea->addSubWindow(viewer);
    win->setWindowTitle(title);
    win->setWindowFlags(Qt::WindowTitleHint | Qt::WindowMinMaxButtonsHint);

    viewer->setCache(_chunkCache);
    viewer->setPointCollection(_points);
    viewer->setViewerManager(this);

    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged, viewer, &CVolumeViewer::onSurfaceChanged);
        connect(_surfaces, &CSurfaceCollection::sendPOIChanged, viewer, &CVolumeViewer::onPOIChanged);
        connect(_surfaces, &CSurfaceCollection::sendIntersectionChanged, viewer, &CVolumeViewer::onIntersectionChanged);
    }

    // Restore persisted viewer preferences
    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool showHints = settings.value("viewer/show_direction_hints", true).toBool();
        viewer->setShowDirectionHints(showHints);
    }

    {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool resetView = settings.value("viewer/reset_view_on_surface_change", true).toBool();
        viewer->setResetViewOnSurfaceChange(resetView);
        _resetDefaults[viewer] = resetView;
    }

    viewer->setSurface(surfaceName);
    viewer->setSegmentationEditActive(_segmentationEditActive);

    if (_segmentationOverlay) {
        _segmentationOverlay->attachViewer(viewer);
    }

    if (_pointsOverlay) {
        _pointsOverlay->attachViewer(viewer);
    }

    if (_pathsOverlay) {
        _pathsOverlay->attachViewer(viewer);
    }

    if (_bboxOverlay) {
        _bboxOverlay->attachViewer(viewer);
    }

    if (_vectorOverlay) {
        _vectorOverlay->attachViewer(viewer);
    }

    viewer->setIntersectionOpacity(_intersectionOpacity);
    viewer->setMaxIntersections(_maxIntersections);
    viewer->setIntersectionLineWidth(_intersectionLineWidth);
    viewer->setHighlightedSegments(_highlightedSegments);
    viewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
    viewer->setOverlayVolume(_overlayVolume);
    viewer->setOverlayOpacity(_overlayOpacity);
    viewer->setOverlayColormap(_overlayColormapId);
    viewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);

    _viewers.push_back(viewer);
    if (_segmentationModule) {
        _segmentationModule->attachViewer(viewer);
    }
    emit viewerCreated(viewer);
    return viewer;
}

void ViewerManager::setSegmentationOverlay(SegmentationOverlayController* overlay)
{
    _segmentationOverlay = overlay;
    if (!_segmentationOverlay) {
        return;
    }
    _segmentationOverlay->bindToViewerManager(this);
}

void ViewerManager::setSegmentationEditActive(bool active)
{
    _segmentationEditActive = active;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setSegmentationEditActive(active);
        }
    }
}

void ViewerManager::setSegmentationModule(SegmentationModule* module)
{
    _segmentationModule = module;
    if (!_segmentationModule) {
        return;
    }

    for (auto* viewer : _viewers) {
        _segmentationModule->attachViewer(viewer);
    }
}

void ViewerManager::setPointsOverlay(PointsOverlayController* overlay)
{
    _pointsOverlay = overlay;
    if (!_pointsOverlay) {
        return;
    }
    _pointsOverlay->bindToViewerManager(this);
}

void ViewerManager::setPathsOverlay(PathsOverlayController* overlay)
{
    _pathsOverlay = overlay;
    if (!_pathsOverlay) {
        return;
    }
    _pathsOverlay->bindToViewerManager(this);
}

void ViewerManager::setBBoxOverlay(BBoxOverlayController* overlay)
{
    _bboxOverlay = overlay;
    if (!_bboxOverlay) {
        return;
    }
    _bboxOverlay->bindToViewerManager(this);
}

void ViewerManager::setVectorOverlay(VectorOverlayController* overlay)
{
    _vectorOverlay = overlay;
    if (!_vectorOverlay) {
        return;
    }
    _vectorOverlay->bindToViewerManager(this);
}

void ViewerManager::setVolumeOverlay(VolumeOverlayController* overlay)
{
    _volumeOverlay = overlay;
    if (_volumeOverlay) {
        _volumeOverlay->syncWindowFromManager(_overlayWindowLow, _overlayWindowHigh);
    }
}

void ViewerManager::setIntersectionOpacity(float opacity)
{
    _intersectionOpacity = std::clamp(opacity, 0.0f, 1.0f);

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("viewer/intersection_opacity",
                      static_cast<int>(std::lround(_intersectionOpacity * 100.0f)));

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setIntersectionOpacity(_intersectionOpacity);
        }
    }
}

void ViewerManager::setMaxIntersections(int maxIntersections)
{
    _maxIntersections = std::clamp(maxIntersections, 1, 500);

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setMaxIntersections(_maxIntersections);
        }
    }
}

void ViewerManager::setIntersectionLineWidth(int lineWidth)
{
    _intersectionLineWidth = std::clamp(lineWidth, 1, 10);

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setIntersectionLineWidth(_intersectionLineWidth);
        }
    }
}

void ViewerManager::setHighlightedSegments(const std::vector<std::string>& segments)
{
    _highlightedSegments = segments;

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setHighlightedSegments(_highlightedSegments);
        }
    }
}

void ViewerManager::setOverlayVolume(std::shared_ptr<Volume> volume, const std::string& volumeId)
{
    _overlayVolume = std::move(volume);
    _overlayVolumeId = volumeId;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayVolume(_overlayVolume);
        }
    }

    emit overlayVolumeAvailabilityChanged(static_cast<bool>(_overlayVolume));
}

void ViewerManager::setOverlayOpacity(float opacity)
{
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayOpacity(_overlayOpacity);
        }
    }
}

void ViewerManager::setOverlayColormap(const std::string& colormapId)
{
    _overlayColormapId = colormapId;
    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayColormap(_overlayColormapId);
        }
    }
}

void ViewerManager::setOverlayThreshold(float threshold)
{
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void ViewerManager::setOverlayWindow(float low, float high)
{
    constexpr float kMaxOverlayValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxOverlayValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxOverlayValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxOverlayValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _overlayWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _overlayWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _overlayWindowLow = clampedLow;
    _overlayWindowHigh = clampedHigh;

    if (_volumeOverlay) {
        _volumeOverlay->syncWindowFromManager(_overlayWindowLow, _overlayWindowHigh);
    }

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setOverlayWindow(_overlayWindowLow, _overlayWindowHigh);
        }
    }

    emit overlayWindowChanged(_overlayWindowLow, _overlayWindowHigh);
}

void ViewerManager::setVolumeWindow(float low, float high)
{
    constexpr float kMaxValue = 255.0f;
    const float clampedLow = std::clamp(low, 0.0f, kMaxValue);
    float clampedHigh = std::clamp(high, 0.0f, kMaxValue);
    if (clampedHigh <= clampedLow) {
        clampedHigh = std::min(kMaxValue, clampedLow + 1.0f);
    }

    const bool unchanged = std::abs(clampedLow - _volumeWindowLow) < 1e-6f &&
                           std::abs(clampedHigh - _volumeWindowHigh) < 1e-6f;
    if (unchanged) {
        return;
    }

    _volumeWindowLow = clampedLow;
    _volumeWindowHigh = clampedHigh;

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("viewer/base_window_low", _volumeWindowLow);
    settings.setValue("viewer/base_window_high", _volumeWindowHigh);

    for (auto* viewer : _viewers) {
        if (viewer) {
            viewer->setVolumeWindow(_volumeWindowLow, _volumeWindowHigh);
        }
    }

    emit volumeWindowChanged(_volumeWindowLow, _volumeWindowHigh);
}

bool ViewerManager::resetDefaultFor(CVolumeViewer* viewer) const
{
    auto it = _resetDefaults.find(viewer);
    return it != _resetDefaults.end() ? it->second : true;
}

void ViewerManager::setResetDefaultFor(CVolumeViewer* viewer, bool value)
{
    if (!viewer) {
        return;
    }
    _resetDefaults[viewer] = value;
}

void ViewerManager::forEachViewer(const std::function<void(CVolumeViewer*)>& fn) const
{
    if (!fn) {
        return;
    }
    for (auto* viewer : _viewers) {
        fn(viewer);
    }
}

std::vector<ViewerManager::CandidateInfo> ViewerManager::getCachedCandidates(
    const cv::Vec3f& referenceCenter,
    const std::set<std::string>& intersectTargets,
    const std::unordered_map<std::string, std::vector<QGraphicsItem*>>& alreadyRendered,
    bool useHighlightedSegments)
{
    // If using highlighted segments, build a different candidate list
    if (useHighlightedSegments && !_highlightedSegments.empty()) {
        std::vector<CandidateInfo> result;

        // Always include the current segment (segmentation)
        if (intersectTargets.find("segmentation") != intersectTargets.end() &&
            alreadyRendered.find("segmentation") == alreadyRendered.end()) {
            result.push_back({"segmentation", 0.0f});
        }

        // Add highlighted segments
        for (const auto& segName : _highlightedSegments) {
            if (intersectTargets.find(segName) != intersectTargets.end() &&
                alreadyRendered.find(segName) == alreadyRendered.end()) {
                auto* seg = dynamic_cast<QuadSurface*>(_surfaces->surface(segName));
                if (seg) {
                    result.push_back({segName, 1.0f});  // Distance doesn't matter here
                }
            }
        }

        return result;
    }

    // Check if cache is valid (same reference center)
    constexpr float epsilon = 1e-6f;
    const bool sameReference = _candidateCacheValid &&
                               std::abs(referenceCenter[0] - _cachedReferenceCenter[0]) < epsilon &&
                               std::abs(referenceCenter[1] - _cachedReferenceCenter[1]) < epsilon &&
                               std::abs(referenceCenter[2] - _cachedReferenceCenter[2]) < epsilon;

    if (!sameReference) {
        // Cache miss - rebuild candidate list
        _cachedCandidates.clear();

        if (!_surfaces) {
            _candidateCacheValid = false;
            return {};
        }

        // Build list of all candidates with their distances
        for (const auto& key : intersectTargets) {
            auto* seg = dynamic_cast<QuadSurface*>(_surfaces->surface(key));
            if (!seg) continue;

            // For the segmentation itself, use distance 0
            if (key == "segmentation") {
                _cachedCandidates.push_back({key, 0.0f});
                continue;
            }

            Rect3D seg_bbox = seg->bbox();

            // Calculate centroid of segment bbox
            cv::Vec3f seg_center = {
                (seg_bbox.low[0] + seg_bbox.high[0]) / 2.0f,
                (seg_bbox.low[1] + seg_bbox.high[1]) / 2.0f,
                (seg_bbox.low[2] + seg_bbox.high[2]) / 2.0f
            };

            // Calculate distance from reference to this segment center
            float dist = cv::norm(referenceCenter - seg_center);
            _cachedCandidates.push_back({key, dist});
        }

        // Sort by distance
        std::sort(_cachedCandidates.begin(), _cachedCandidates.end(),
                  [](const CandidateInfo& a, const CandidateInfo& b) {
                      return a.distance < b.distance;
                  });

        _cachedReferenceCenter = referenceCenter;
        _candidateCacheValid = true;
    }

    // Filter out already rendered segments
    std::vector<CandidateInfo> result;
    result.reserve(_cachedCandidates.size());
    for (const auto& candidate : _cachedCandidates) {
        if (alreadyRendered.find(candidate.key) == alreadyRendered.end()) {
            result.push_back(candidate);
        }
    }

    return result;
}

void ViewerManager::invalidateCandidateCache()
{
    _candidateCacheValid = false;
    _cachedCandidates.clear();
}

void ViewerManager::buildGlobalSpatialIndex()
{
    if (!_surfaces) {
        qDebug() << "[GLOBAL SPATIAL INDEX] No surfaces collection";
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Clear existing index
    _globalSpatialIndex.clear();

    auto surfaceNames = _surfaces->surfaceNames();
    qDebug() << "[GLOBAL SPATIAL INDEX] Building index for" << surfaceNames.size() << "surfaces";

    int successCount = 0;
    int skippedCount = 0;
    for (const auto& name : surfaceNames) {
        auto* surface = _surfaces->surface(name);
        if (!surface) {
            skippedCount++;
            continue;
        }

        // Try to get QuadSurface - either directly or from DeltaSurface base
        QuadSurface* quadSurf = dynamic_cast<QuadSurface*>(surface);
        if (!quadSurf) {
            auto* deltaSurf = dynamic_cast<DeltaSurface*>(surface);
            if (deltaSurf && deltaSurf->getBase()) {
                quadSurf = dynamic_cast<QuadSurface*>(deltaSurf->getBase());
            }
        }

        if (!quadSurf) {
            skippedCount++;
            continue;
        }

        // Insert segment into spatial index with its bounding box
        Rect3D bbox = quadSurf->bbox();
        _globalSpatialIndex.insert(name, bbox);
        successCount++;

        // Log progress every 100 segments
        if (successCount % 100 == 0) {
            qDebug() << "[GLOBAL SPATIAL INDEX] Indexed" << successCount << "segments so far...";
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    qDebug() << "[GLOBAL SPATIAL INDEX] Indexed" << successCount << "segments in" << duration.count() << "ms"
             << "using" << _globalSpatialIndex.cellCount() << "cells";
}

void ViewerManager::invalidateGlobalSpatialIndex()
{
    _globalSpatialIndex.clear();
}

std::vector<std::string> ViewerManager::querySegmentsNearPlane(const Rect3D& planeBounds) const
{
    // Query spatial index for segments in this region (fast, no linear search!)
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::string> result = _globalSpatialIndex.query(planeBounds);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    qDebug() << "[GLOBAL SPATIAL INDEX] Query returned" << result.size() << "segments in" << duration.count() << "μs";

    return result;
}
