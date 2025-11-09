#include "IntersectionOverlayController.hpp"

#include "CVolumeViewer.hpp"
#include "CSurfaceCollection.hpp"
#include "ViewerManager.hpp"
#include "vc/core/util/Logging.hpp"

#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QPainterPath>

#include <algorithm>
#include <chrono>
#include <functional>

// Special colors for current segment
#define COLOR_SEG_XY QColor(255, 140, 0)  // Orange
#define COLOR_SEG_YZ Qt::yellow
#define COLOR_SEG_XZ Qt::red

IntersectionOverlayController::IntersectionOverlayController(ViewerManager* manager, QObject* parent)
    : ViewerOverlayControllerBase("intersection", parent)
    , _spatialIndex(100.0f)  // 100 voxel cell size
{
    if (manager) {
        bindToViewerManager(manager);
    }
}

void IntersectionOverlayController::setCurrentSegment(const std::string& segmentId)
{
    if (_currentSegmentId == segmentId) {
        return;
    }

    _currentSegmentId = segmentId;

    // Invalidate all caches since current segment changed (affects colors)
    for (auto& [viewer, cache] : _viewerCaches) {
        cache.invalidate();
    }

    refreshAll();
}

void IntersectionOverlayController::setIntersectionTargets(const std::set<std::string>& targets)
{
    if (_intersectionTargets == targets) {
        return;
    }

    _intersectionTargets = targets;

    // Rebuild spatial index with new targets
    rebuildIndex();

    // Invalidate all caches
    for (auto& [viewer, cache] : _viewerCaches) {
        cache.invalidate();
    }

    refreshAll();
}

void IntersectionOverlayController::segmentChanged(const std::string& segmentId)
{
    // Invalidate this segment in all viewer caches
    for (auto& [viewer, cache] : _viewerCaches) {
        cache.invalidateSegment(segmentId);
    }

    // Mark index as dirty (will rebuild on next render)
    for (auto& [viewer, cache] : _viewerCaches) {
        cache.indexDirty = true;
    }

    refreshAll();
}

void IntersectionOverlayController::setLineWidth(float width)
{
    _lineWidth = std::clamp(width, 1.0f, 10.0f);
    refreshAll();
}

void IntersectionOverlayController::setOpacity(float opacity)
{
    _opacity = std::clamp(opacity, 0.0f, 1.0f);
    refreshAll();
}

void IntersectionOverlayController::setHighlightedSegments(const QSet<QString>& segments)
{
    if (_highlightedSegments == segments) {
        return;
    }

    _highlightedSegments = segments;

    // Invalidate all caches since filtering changed
    for (auto& [viewer, cache] : _viewerCaches) {
        cache.invalidate();
    }

    refreshAll();
}

void IntersectionOverlayController::rebuildIndex()
{
    auto t_start = std::chrono::high_resolution_clock::now();

    // Clear existing index
    _spatialIndex = MultiSurfaceIndex(100.0f);
    _segmentToIndexMap.clear();
    _indexToSegmentMap.clear();

    if (!_surfaceCollection) {
        Logger()->warn("IntersectionOverlayController: No surface collection set");
        return;
    }

    // Build index from intersection targets (limit to 100 segments)
    int idx = 0;
    for (const auto& segmentId : _intersectionTargets) {
        if (idx >= 100) {
            Logger()->warn("IntersectionOverlayController: Reached limit of 100 segments");
            break;
        }

        Surface* baseSurf = _surfaceCollection->surface(segmentId);
        QuadSurface* surf = dynamic_cast<QuadSurface*>(baseSurf);

        if (surf) {
            _spatialIndex.addPatch(idx, surf);
            _segmentToIndexMap[segmentId] = idx;
            _indexToSegmentMap[idx] = segmentId;
            idx++;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    Logger()->info("IntersectionOverlayController: Rebuilt index with {} segments in {:.2f}ms",
                   idx, duration_ms);
}

bool IntersectionOverlayController::CachedLine::isValid(
    PlaneSurface* plane,
    float currentScale,
    const QRectF& currentViewport) const
{
    if (!plane) return false;

    // Check if plane origin/normal changed
    cv::Vec3f ptr = plane->pointer();
    if (cv::norm(plane->origin() - planeOrigin) > 0.1f) return false;

    cv::Vec3f currentNormal = plane->normal(ptr, {0,0,0});
    if (cv::norm(currentNormal - planeNormal) > 0.01f) return false;

    // Check if scale changed significantly (zoom)
    if (std::abs(currentScale - scale) / scale > 0.05f) return false;

    // Viewport pan is OK (we can just translate the lines)
    // But if viewport moved too far, might be outside cached region
    // For now, accept any viewport (could optimize later)

    return true;
}

QColor IntersectionOverlayController::getSegmentColor(
    const std::string& segmentId,
    const std::string& viewerName) const
{
    // Current segment gets special colors
    if (segmentId == _currentSegmentId) {
        if (viewerName == "xy plane") return COLOR_SEG_XY;
        if (viewerName == "seg yz") return COLOR_SEG_YZ;
        if (viewerName == "seg xz") return COLOR_SEG_XZ;
    }

    // Other segments get deterministic hash-based colors
    return hashColor(segmentId);
}

QColor IntersectionOverlayController::hashColor(const std::string& segmentId) const
{
    std::hash<std::string> hasher;
    size_t hash = hasher(segmentId);

    // Use different parts of hash for R, G, B
    unsigned char r = 100 + ((hash >> 0) % 156);   // 100-255
    unsigned char g = 100 + ((hash >> 8) % 156);   // 100-255
    unsigned char b = 100 + ((hash >> 16) % 156);  // 100-255

    // Make one channel dominant for better visibility
    int primary = (hash >> 24) % 3;
    if (primary == 0) r = 200 + (hash % 56);        // 200-255
    else if (primary == 1) g = 200 + (hash % 56);
    else b = 200 + (hash % 56);

    return QColor(r, g, b);
}

std::vector<std::string> IntersectionOverlayController::findVisibleSegments(
    PlaneSurface* plane,
    const QRectF& viewport)
{
    // Convert viewport to 3D bounding box in volume space
    cv::Vec3f corner = plane->coord(cv::Vec3f(0,0,0), {static_cast<float>(viewport.x()),
                                                        static_cast<float>(viewport.y()), 0.0f});

    Rect3D viewBbox = {corner, corner};
    viewBbox = expand_rect(viewBbox, plane->coord(cv::Vec3f(0,0,0),
        {static_cast<float>(viewport.right()), static_cast<float>(viewport.y()), 0}));
    viewBbox = expand_rect(viewBbox, plane->coord(cv::Vec3f(0,0,0),
        {static_cast<float>(viewport.x()), static_cast<float>(viewport.bottom()), 0}));
    viewBbox = expand_rect(viewBbox, plane->coord(cv::Vec3f(0,0,0),
        {static_cast<float>(viewport.right()), static_cast<float>(viewport.bottom()), 0}));

    // Add proportional margin to viewBbox for hysteresis (prevents popping)
    // Use 50% of viewport size on each side for strong anti-popping
    cv::Vec3f bboxSize = viewBbox.high - viewBbox.low;
    cv::Vec3f margin = bboxSize * 0.5f;
    Rect3D expandedViewBbox = viewBbox;
    expandedViewBbox.low = viewBbox.low - margin;
    expandedViewBbox.high = viewBbox.high + margin;

    // Calculate center and radius for spatial query
    cv::Vec3f bboxCenter = {
        (expandedViewBbox.low[0] + expandedViewBbox.high[0]) / 2.0f,
        (expandedViewBbox.low[1] + expandedViewBbox.high[1]) / 2.0f,
        (expandedViewBbox.low[2] + expandedViewBbox.high[2]) / 2.0f
    };
    // Radius is half the diagonal of expanded bbox
    float searchRadius = cv::norm(expandedViewBbox.high - expandedViewBbox.low) / 2.0f;

    // Query spatial index for candidates
    std::vector<int> candidateIndices = _spatialIndex.getCandidatePatches(bboxCenter, searchRadius);

    // Convert indices to segment IDs and filter by bbox intersection
    std::vector<std::string> visibleSegments;
    for (int idx : candidateIndices) {
        auto it = _indexToSegmentMap.find(idx);
        if (it == _indexToSegmentMap.end()) continue;

        const std::string& segmentId = it->second;

        // Get segment surface
        Surface* baseSurf = _surfaceCollection->surface(segmentId);
        QuadSurface* surf = dynamic_cast<QuadSurface*>(baseSurf);

        if (surf && intersect(expandedViewBbox, surf->bbox())) {
            visibleSegments.push_back(segmentId);
        }
    }

    // Sort by distance from current segment center (if we have a current segment)
    if (!_currentSegmentId.empty() && _surfaceCollection) {
        Surface* currentBaseSurf = _surfaceCollection->surface(_currentSegmentId);
        QuadSurface* currentSurf = dynamic_cast<QuadSurface*>(currentBaseSurf);

        if (currentSurf) {
            Rect3D currentBbox = currentSurf->bbox();
            cv::Vec3f currentCenter = {
                (currentBbox.low[0] + currentBbox.high[0]) / 2.0f,
                (currentBbox.low[1] + currentBbox.high[1]) / 2.0f,
                (currentBbox.low[2] + currentBbox.high[2]) / 2.0f
            };

            std::sort(visibleSegments.begin(), visibleSegments.end(),
                [&](const std::string& a, const std::string& b) {
                    QuadSurface* surfA = dynamic_cast<QuadSurface*>(_surfaceCollection->surface(a));
                    QuadSurface* surfB = dynamic_cast<QuadSurface*>(_surfaceCollection->surface(b));
                    if (!surfA || !surfB) return false;

                    Rect3D bboxA = surfA->bbox();
                    Rect3D bboxB = surfB->bbox();

                    cv::Vec3f centerA = {
                        (bboxA.low[0] + bboxA.high[0]) / 2.0f,
                        (bboxA.low[1] + bboxA.high[1]) / 2.0f,
                        (bboxA.low[2] + bboxA.high[2]) / 2.0f
                    };
                    cv::Vec3f centerB = {
                        (bboxB.low[0] + bboxB.high[0]) / 2.0f,
                        (bboxB.low[1] + bboxB.high[1]) / 2.0f,
                        (bboxB.low[2] + bboxB.high[2]) / 2.0f
                    };

                    float distA = cv::norm(centerA - currentCenter);
                    float distB = cv::norm(centerB - currentCenter);
                    // Use UUID as tiebreaker for stable sorting when distances are equal
                    if (distA != distB) {
                        return distA < distB;
                    }
                    return a < b;
                });
        }
    }

    // Limit to 100 closest segments
    if (visibleSegments.size() > 100) {
        visibleSegments.resize(100);
    }

    // Filter by highlighted segments if specified
    if (!_highlightedSegments.isEmpty()) {
        std::vector<std::string> filteredSegments;
        for (const auto& segmentId : visibleSegments) {
            // Always include current segment
            if (segmentId == _currentSegmentId) {
                filteredSegments.push_back(segmentId);
                continue;
            }
            // Include if in highlighted list
            if (_highlightedSegments.contains(QString::fromStdString(segmentId))) {
                filteredSegments.push_back(segmentId);
            }
        }
        return filteredSegments;
    }

    return visibleSegments;
}

void IntersectionOverlayController::renderSegmentIntersection(
    CVolumeViewer* viewer,
    const std::string& segmentId,
    PlaneSurface* plane,
    const QRectF& viewport,
    OverlayBuilder& builder)
{
    // ALWAYS redraw - caching disabled for now

    // Need to compute intersection
    Surface* baseSurf = _surfaceCollection->surface(segmentId);
    QuadSurface* surf = dynamic_cast<QuadSurface*>(baseSurf);
    if (!surf) {
        Logger()->warn("Segment '{}' not found or not a QuadSurface", segmentId);
        return;
    }

    // Get the raw points
    const cv::Mat_<cv::Vec3f>& rawPoints = surf->rawPoints();

    // Convert viewport to plane ROI
    float viewerScale = viewer->scale();
    cv::Rect planeRoi = {
        static_cast<int>(viewport.x() / viewerScale),
        static_cast<int>(viewport.y() / viewerScale),
        static_cast<int>(viewport.width() / viewerScale),
        static_cast<int>(viewport.height() / viewerScale)
    };

    // Compute intersection segments
    std::vector<std::vector<cv::Vec3f>> intersectionSegments3D;
    std::vector<std::vector<cv::Vec2f>> intersectionSegments2D;

    // Get POI hint if available - use it as a starting point for search
    cv::Vec3f poiHint{0, 0, 0};
    bool havePOI = false;
    if (_surfaceCollection) {
        POI* poi = _surfaceCollection->poi("focus");
        if (poi) {
            poiHint = poi->p;
            havePOI = true;
        }
    }

    // Zoom-based optimizations:
    // - Low zoom (< 1.0): zoomed out, need less detail
    // - High zoom (> 1.0): zoomed in, need more detail
    int minTries;
    float stepSizeMultiplier;

    if (viewerScale < 0.5f) {
        // Very zoomed out (0.03x - 0.5x): dense grid sampling provides excellent coverage
        minTries = (segmentId == _currentSegmentId) ? 20 : 12;
        stepSizeMultiplier = 4.0f;
    } else if (viewerScale < 1.0f) {
        // Zoomed out (0.5x - 1.0x): moderate tries with dense grid
        minTries = (segmentId == _currentSegmentId) ? 25 : 15;
        stepSizeMultiplier = 3.0f;
    } else if (viewerScale < 2.0f) {
        // Normal zoom (1.0x - 2.0x): more tries for complete coverage
        minTries = (segmentId == _currentSegmentId) ? 30 : 20;
        stepSizeMultiplier = 2.5f;
    } else {
        // Zoomed in (2.0x - 4.0x): maximum tries for finest detail
        minTries = (segmentId == _currentSegmentId) ? 40 : 25;
        stepSizeMultiplier = 2.0f;
    }

    // Double the step size to trace half as many points (2x faster with minimal visual difference)
    find_intersect_segments(intersectionSegments3D, intersectionSegments2D,
                           rawPoints, plane, planeRoi, (stepSizeMultiplier * 2.0f) / viewerScale, minTries,
                           havePOI ? &poiHint : nullptr);

    if (intersectionSegments3D.empty()) {
        Logger()->warn("No intersection segments found for '{}' on plane '{}' (scale={:.2f}, minTries={})",
                      segmentId, viewer->surfName(), viewerScale, minTries);
        return;
    }

    Logger()->info("Found {} intersection segments for '{}' on plane '{}' (scale={:.2f}, minTries={})",
                  intersectionSegments3D.size(), segmentId, viewer->surfName(), viewerScale, minTries);

    // Convert 3D segments to screen coordinates and render
    QColor color = getSegmentColor(segmentId, viewer->surfName());
    std::vector<QPointF> allPoints;

    // Determine point decimation factor based on zoom
    int decimation = 1;  // Keep all points by default
    if (viewerScale < 0.5f) {
        decimation = 4;  // Keep every 4th point when very zoomed out
    } else if (viewerScale < 1.0f) {
        decimation = 2;  // Keep every 2nd point when zoomed out
    }

    for (const auto& segment : intersectionSegments3D) {
        std::vector<QPointF> segmentPoints;

        for (size_t i = 0; i < segment.size(); i += decimation) {
            cv::Vec3f screenPos = plane->project(segment[i], 1.0, viewerScale);
            segmentPoints.push_back(QPointF(screenPos[0], screenPos[1]));
        }

        // Always include the last point to ensure segment endpoints are preserved
        if (segment.size() > 1 && (segment.size() - 1) % decimation != 0) {
            cv::Vec3f screenPos = plane->project(segment.back(), 1.0, viewerScale);
            segmentPoints.push_back(QPointF(screenPos[0], screenPos[1]));
        }

        // Render each segment as a continuous line
        // The curve tracer already handles segment splitting based on valid data
        OverlayStyle style;
        style.penColor = color;
        style.penWidth = (segmentId == _currentSegmentId) ? _lineWidth * 1.5f : _lineWidth;
        style.z = (segmentId == _currentSegmentId) ? 100 : 5;

        if (!segmentPoints.empty()) {
            builder.addLineStrip(segmentPoints, false, style);
            allPoints.insert(allPoints.end(), segmentPoints.begin(), segmentPoints.end());
        }
    }

    // No caching - always redraw
}

void IntersectionOverlayController::collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder)
{
    if (!viewer || !_surfaceCollection) {
        Logger()->warn("IntersectionOverlayController: No viewer or surface collection");
        return;
    }

    // Get viewer's surface
    Surface* surf = viewer->surf();
    if (!surf) {
        Logger()->warn("IntersectionOverlayController: Viewer has no surface");
        return;
    }

    // Handle plane surfaces (xy, xz, yz viewers)
    if (PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surf)) {
        // Get viewport
        QRectF viewport = viewer->currentImageArea();
        float scale = viewer->scale();

        // Find all visible segments (sorted by distance, limited to 100)
        std::vector<std::string> visibleSegments = findVisibleSegments(plane, viewport);

        // Render non-current segments first (so current segment draws on top)
        for (const auto& segmentId : visibleSegments) {
            if (segmentId != _currentSegmentId) {
                renderSegmentIntersection(viewer, segmentId, plane, viewport, builder);
            }
        }

        // Render current segment last (on top with higher z-order)
        if (!_currentSegmentId.empty()) {
            renderSegmentIntersection(viewer, _currentSegmentId, plane, viewport, builder);
        }
    }
    // Handle flattened view (segmentation viewer)
    else if (QuadSurface* quadSurf = dynamic_cast<QuadSurface*>(surf)) {
        Logger()->debug("IntersectionOverlayController: Flattened view - skipping for now");
    }
    else {
        Logger()->debug("IntersectionOverlayController: Unknown surface type");
    }
}

void IntersectionOverlayController::renderFlattenedViewIntersections(
    CVolumeViewer* viewer,
    QuadSurface* flattenedSurf)
{
    // TODO: Implement rendering of xz (yellow) and yz (red) plane intersections
    // on the flattened segmentation view
    // This is more complex and will be implemented in a follow-up
    // For now, just log that we're here
    Logger()->debug("IntersectionOverlayController: Flattened view rendering not yet implemented");
}
