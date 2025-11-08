#pragma once

#include "ViewerOverlayControllerBase.hpp"
#include "vc/core/util/Surface.hpp"

#include <QColor>
#include <QString>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class CSurfaceCollection;
class QuadSurface;
class PlaneSurface;

/**
 * Overlay controller for rendering segment intersection lines on viewers.
 *
 * Renders:
 * - Current segment (special colors): orange (xy), yellow (yz), red (xz)
 * - Other segments: deterministic hash-based colors
 * - Cached lines with smart invalidation
 */
class IntersectionOverlayController : public ViewerOverlayControllerBase
{
    Q_OBJECT

public:
    explicit IntersectionOverlayController(ViewerManager* manager, QObject* parent = nullptr);
    ~IntersectionOverlayController() override = default;

    // Set the surface collection
    void setSurfaceCollection(CSurfaceCollection* collection) { _surfaceCollection = collection; }

    // Set which segment is the "current" one (gets special coloring)
    void setCurrentSegment(const std::string& segmentId);

    // Set which segments to render intersections for
    void setIntersectionTargets(const std::set<std::string>& targets);

    // Notify that a segment's geometry has changed
    void segmentChanged(const std::string& segmentId);

    // Set rendering parameters
    void setLineWidth(float width);
    void setOpacity(float opacity);

    // Rebuild the spatial index (call when segments added/removed/modified)
    void rebuildIndex();

protected:
    // Override from base class
    void collectPrimitives(CVolumeViewer* viewer, OverlayBuilder& builder) override;

private:
    struct CachedLine {
        std::vector<QPointF> points;
        QColor color;

        // Cache validity metadata
        cv::Vec3f planeOrigin;
        cv::Vec3f planeNormal;
        float scale;
        QRectF viewport;

        bool isValid(PlaneSurface* plane, float currentScale, const QRectF& currentViewport) const;
    };

    struct ViewerCache {
        std::unordered_map<std::string, CachedLine> lines;  // segmentId -> cached line
        bool indexDirty = true;

        void invalidate() {
            lines.clear();
            indexDirty = true;
        }

        void invalidateSegment(const std::string& segmentId) {
            lines.erase(segmentId);
        }
    };

    // Per-viewer caching
    std::unordered_map<CVolumeViewer*, ViewerCache> _viewerCaches;

    // Spatial index for fast segment lookup
    MultiSurfaceIndex _spatialIndex;
    std::unordered_map<std::string, int> _segmentToIndexMap;  // segmentId -> spatial index
    std::unordered_map<int, std::string> _indexToSegmentMap;  // spatial index -> segmentId

    // Current state
    std::string _currentSegmentId;
    std::set<std::string> _intersectionTargets;
    CSurfaceCollection* _surfaceCollection = nullptr;

    // Rendering parameters
    float _lineWidth = 5.0f;
    float _opacity = 1.0f;

    // Helper methods
    QColor getSegmentColor(const std::string& segmentId, const std::string& viewerName) const;
    std::vector<std::string> findVisibleSegments(PlaneSurface* plane, const QRectF& viewport);
    void renderSegmentIntersection(CVolumeViewer* viewer, const std::string& segmentId,
                                   PlaneSurface* plane, const QRectF& viewport, OverlayBuilder& builder);
    void renderFlattenedViewIntersections(CVolumeViewer* viewer, QuadSurface* flattenedSurf);

    QColor hashColor(const std::string& segmentId) const;
};
