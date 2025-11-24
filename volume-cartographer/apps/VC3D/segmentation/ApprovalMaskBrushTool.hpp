#pragma once

#include <opencv2/core.hpp>
#include <QElapsedTimer>
#include <unordered_set>
#include <vector>

class QuadSurface;
class SegmentationModule;
class SegmentationWidget;
class SegmentationEditManager;

/**
 * Brush tool for painting approval/unapproved regions on segmentation surfaces.
 *
 * This tool allows users to interactively mark portions of a surface as approved
 * or unapproved by brushing in any viewer. The approval mask is stored as a channel
 * on the QuadSurface and persists through segment growth operations.
 */
class ApprovalMaskBrushTool
{
public:
    enum class PaintMode {
        Approve,    // Paint approval (value 255)
        Unapprove   // Paint unapproved (value 0)
    };

    ApprovalMaskBrushTool(SegmentationModule& module,
                          SegmentationEditManager* editManager,
                          SegmentationWidget* widget);

    void setDependencies(SegmentationWidget* widget);
    void setSurface(QuadSurface* surface);
    void setPaintMode(PaintMode mode) { _paintMode = mode; }
    [[nodiscard]] PaintMode paintMode() const { return _paintMode; }

    void setActive(bool active);
    [[nodiscard]] bool brushActive() const { return _brushActive; }
    [[nodiscard]] bool strokeActive() const { return _strokeActive; }
    [[nodiscard]] bool hasPendingStrokes() const { return !_pendingStrokes.empty(); }

    void startStroke(const cv::Vec3f& worldPos);
    void extendStroke(const cv::Vec3f& worldPos, bool forceSample);
    void finishStroke();
    bool applyPending(float dragRadiusSteps);
    void clear();

    [[nodiscard]] const std::vector<cv::Vec3f>& overlayPoints() const { return _overlayPoints; }
    [[nodiscard]] const std::vector<std::vector<cv::Vec3f>>& overlayStrokeSegments() const { return _overlayStrokeSegments; }
    [[nodiscard]] const std::vector<cv::Vec3f>& currentStrokePoints() const { return _currentStroke; }

    void cancel() { clear(); }
    [[nodiscard]] bool isActive() const { return brushActive() || strokeActive(); }

private:
    // Convert world position to grid indices on current surface
    std::optional<std::pair<int, int>> worldToGridIndex(const cv::Vec3f& worldPos) const;

    // Paint accumulated points into QImage (for real-time painting)
    void paintAccumulatedPointsToImage();

    SegmentationModule& _module;
    SegmentationEditManager* _editManager{nullptr};
    SegmentationWidget* _widget{nullptr};
    QuadSurface* _surface{nullptr};

    PaintMode _paintMode{PaintMode::Approve};
    bool _brushActive{false};
    bool _strokeActive{false};
    std::vector<cv::Vec3f> _currentStroke;
    std::vector<std::vector<cv::Vec3f>> _pendingStrokes;
    std::vector<cv::Vec3f> _overlayPoints;  // Current active stroke overlay
    std::vector<std::vector<cv::Vec3f>> _overlayStrokeSegments;  // Completed stroke segments for overlay
    cv::Vec3f _lastSample{0.0f, 0.0f, 0.0f};
    bool _hasLastSample{false};
    cv::Vec3f _lastOverlaySample{0.0f, 0.0f, 0.0f};
    bool _hasLastOverlaySample{false};

    // Throttling for overlay refresh during painting
    QElapsedTimer _lastRefreshTimer;
    qint64 _lastRefreshTime{0};
    bool _pendingRefresh{false};

    // Accumulated grid positions for real-time painting
    std::vector<std::pair<int, int>> _accumulatedGridPositions;
    std::unordered_set<uint64_t> _accumulatedGridPosSet;  // For deduplication
};
