#include "ApprovalMaskBrushTool.hpp"

#include "SegmentationEditManager.hpp"
#include "SegmentationModule.hpp"
#include "SegmentationWidget.hpp"
#include "../overlays/SegmentationOverlayController.hpp"

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <limits>

#include "vc/core/util/QuadSurface.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcApprovalMask)
Q_LOGGING_CATEGORY(lcApprovalMask, "vc.segmentation.approvalmask")

namespace
{
constexpr float kBrushSampleSpacing = 2.0f;       // For accurate stroke data
constexpr float kOverlayPointSpacing = 20.0f;     // For visual overlay (much sparser)

// Check if a point is invalid (NaN, infinity, or the -1,-1,-1 marker)
bool isInvalidPoint(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
}

// Gaussian falloff function
float gaussianFalloff(float distance, float sigma)
{
    if (sigma <= 0.0f) {
        return distance <= 0.0f ? 1.0f : 0.0f;
    }
    return std::exp(-(distance * distance) / (2.0f * sigma * sigma));
}
}

ApprovalMaskBrushTool::ApprovalMaskBrushTool(SegmentationModule& module,
                                             SegmentationEditManager* editManager,
                                             SegmentationWidget* widget)
    : _module(module)
    , _editManager(editManager)
    , _widget(widget)
{
}

void ApprovalMaskBrushTool::setDependencies(SegmentationWidget* widget)
{
    _widget = widget;
}

void ApprovalMaskBrushTool::setSurface(QuadSurface* surface)
{
    _surface = surface;
    qCInfo(lcApprovalMask) << "Surface set on approval tool:" << (surface ? "valid" : "null");
    if (surface) {
        qCInfo(lcApprovalMask) << "  Surface ID:" << QString::fromStdString(surface->id);
        const auto* points = surface->rawPointsPtr();
        if (points) {
            qCInfo(lcApprovalMask) << "  Surface size:" << points->cols << "x" << points->rows;
        }
    }
}

void ApprovalMaskBrushTool::setActive(bool active)
{
    if (_brushActive == active) {
        return;
    }

    _brushActive = active;
    qCInfo(lcApprovalMask) << "Approval brush active:" << active;
    if (!_brushActive) {
        _hasLastSample = false;
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::startStroke(const cv::Vec3f& worldPos)
{
    qCInfo(lcApprovalMask) << "Starting approval stroke at:" << worldPos[0] << worldPos[1] << worldPos[2];
    qCInfo(lcApprovalMask) << "  Surface:" << (_surface ? "valid" : "NULL");
    _strokeActive = true;
    _currentStroke.clear();
    _currentStroke.push_back(worldPos);

    // Clear overlay points to start fresh - prevents connecting to previous strokes
    _overlayPoints.clear();
    _overlayPoints.push_back(worldPos);

    _lastSample = worldPos;
    _hasLastSample = true;
    _lastOverlaySample = worldPos;
    _hasLastOverlaySample = true;

    // Initialize throttling timer
    _lastRefreshTimer.start();
    _lastRefreshTime = 0;
    _pendingRefresh = false;

    // Clear accumulated grid positions for real-time painting
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    // Add the starting point for painting
    auto gridIdx = worldToGridIndex(worldPos);
    if (gridIdx) {
        const uint64_t hash = (static_cast<uint64_t>(gridIdx->first) << 32) | static_cast<uint64_t>(gridIdx->second);
        _accumulatedGridPosSet.insert(hash);
        _accumulatedGridPositions.push_back(*gridIdx);
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::extendStroke(const cv::Vec3f& worldPos, bool forceSample)
{
    if (!_strokeActive) {
        return;
    }

    // Check if position is within valid surface bounds
    auto gridIdx = worldToGridIndex(worldPos);
    if (!gridIdx) {
        // Outside valid surface area - break the current stroke segment
        // but keep stroke active so we can start a new segment when back in bounds
        if (!_currentStroke.empty()) {
            qCInfo(lcApprovalMask) << "Stroke went out of bounds, saving current segment with" << _currentStroke.size() << "points";
            _pendingStrokes.push_back(_currentStroke);
            _currentStroke.clear();
        }

        // Save current overlay segment and clear for next segment
        if (!_overlayPoints.empty()) {
            _overlayStrokeSegments.push_back(_overlayPoints);
            _overlayPoints.clear();
        }

        // Reset last sample tracking so we don't interpolate across the gap
        _hasLastSample = false;
        _hasLastOverlaySample = false;
        return;
    }

    // Back in bounds - if we were out of bounds, this starts a new stroke segment
    if (_currentStroke.empty() && !_hasLastSample) {
        qCInfo(lcApprovalMask) << "Back in bounds, starting new stroke segment";
    }

    const float spacing = kBrushSampleSpacing;
    const float spacingSq = spacing * spacing;

    // Sample stroke data at high resolution (every 2.0 units)
    if (_hasLastSample) {
        const cv::Vec3f delta = worldPos - _lastSample;
        const float distanceSq = delta.dot(delta);
        if (!forceSample && distanceSq < spacingSq) {
            return;
        }

        const float distance = std::sqrt(distanceSq);
        if (distance > spacing) {
            const cv::Vec3f direction = delta / distance;
            float travelled = spacing;
            while (travelled < distance) {
                const cv::Vec3f intermediate = _lastSample + direction * travelled;
                _currentStroke.push_back(intermediate);
                travelled += spacing;
            }
        }
    }

    _currentStroke.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;

    // Accumulate grid position for real-time painting (reuse gridIdx from above)
    // We know gridIdx is valid here because we would have returned early if it was nullopt
    const uint64_t hash = (static_cast<uint64_t>(gridIdx->first) << 32) | static_cast<uint64_t>(gridIdx->second);
    if (_accumulatedGridPosSet.insert(hash).second) {
        _accumulatedGridPositions.push_back(*gridIdx);
    }

    // Paint accumulated points periodically (every 20 points or forceSample)
    constexpr size_t kPaintBatchSize = 20;
    if (forceSample || _accumulatedGridPositions.size() >= kPaintBatchSize) {
        paintAccumulatedPointsToImage();
    }

    // Sample overlay points at much lower resolution (every 20.0 units) for performance
    const float overlaySpacing = kOverlayPointSpacing;
    const float overlaySpacingSq = overlaySpacing * overlaySpacing;

    bool overlayNeedsRefresh = false;
    if (_hasLastOverlaySample) {
        const cv::Vec3f overlayDelta = worldPos - _lastOverlaySample;
        const float overlayDistSq = overlayDelta.dot(overlayDelta);
        if (forceSample || overlayDistSq >= overlaySpacingSq) {
            _overlayPoints.push_back(worldPos);
            _lastOverlaySample = worldPos;
            overlayNeedsRefresh = true;
        }
    } else {
        _overlayPoints.push_back(worldPos);
        _lastOverlaySample = worldPos;
        _hasLastOverlaySample = true;
        overlayNeedsRefresh = true;
    }

    // Only refresh overlay when we actually add a new overlay point (every 20 units)
    // AND throttle to max 20 FPS (50ms minimum interval) to avoid excessive redraws
    if (overlayNeedsRefresh) {
        const qint64 currentTime = _lastRefreshTimer.elapsed();
        const qint64 timeSinceLastRefresh = currentTime - _lastRefreshTime;
        constexpr qint64 kMinRefreshIntervalMs = 50;  // 20 FPS max

        if (timeSinceLastRefresh >= kMinRefreshIntervalMs) {
            QElapsedTimer refreshTimer;
            refreshTimer.start();

            qCInfo(lcApprovalMask) << "[PERF] extendStroke: overlay points count:" << _overlayPoints.size()
                                   << "current stroke:" << _currentStroke.size()
                                   << "pending strokes:" << _pendingStrokes.size()
                                   << "| time since last:" << timeSinceLastRefresh << "ms";

            _module.refreshOverlay();
            _lastRefreshTime = currentTime;
            _pendingRefresh = false;

            qCInfo(lcApprovalMask) << "[PERF] extendStroke: refreshOverlay took:" << refreshTimer.elapsed() << "ms";
        } else {
            // Refresh was skipped due to throttling, mark as pending
            _pendingRefresh = true;
            qCInfo(lcApprovalMask) << "[PERF] extendStroke: THROTTLED (only" << timeSinceLastRefresh << "ms since last refresh)";
        }
    }
}

void ApprovalMaskBrushTool::finishStroke()
{
    if (!_strokeActive) {
        return;
    }

    // Paint any remaining accumulated points
    if (!_accumulatedGridPositions.empty()) {
        paintAccumulatedPointsToImage();
    }

    _strokeActive = false;
    if (!_currentStroke.empty()) {
        _pendingStrokes.push_back(_currentStroke);
    }
    _currentStroke.clear();

    // Save current overlay segment to keep it visible
    if (!_overlayPoints.empty()) {
        _overlayStrokeSegments.push_back(_overlayPoints);
        _overlayPoints.clear();
    }

    _hasLastSample = false;
    _hasLastOverlaySample = false;

    // Refresh on finish to show final state (even if throttled during drawing)
    if (_pendingRefresh) {
        _pendingRefresh = false;
    }
    _module.refreshOverlay();
}

bool ApprovalMaskBrushTool::applyPending(float /*dragRadiusSteps*/)
{
    QElapsedTimer totalTimer;
    totalTimer.start();

    if (!_surface) {
        qCWarning(lcApprovalMask) << "Cannot apply: no surface";
        return false;
    }

    if (_strokeActive) {
        finishStroke();
    }

    // Since we're painting in real-time, just save the QImage to disk
    auto overlay = _module.overlay();
    if (!overlay) {
        qCWarning(lcApprovalMask) << "Cannot apply: no overlay controller";
        return false;
    }

    // Save the approval mask QImage to disk
    overlay->saveApprovalMaskToSurface(_surface);

    qCInfo(lcApprovalMask) << "Saved approval mask to disk in" << totalTimer.elapsed() << "ms";

    // Clear pending strokes and overlay segments (but keep the painted QImage)
    _strokeActive = false;
    _currentStroke.clear();
    _pendingStrokes.clear();
    _overlayPoints.clear();
    _overlayStrokeSegments.clear();
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    _module.refreshOverlay();

    Q_EMIT _module.statusMessageRequested(
        QCoreApplication::translate("ApprovalMaskBrushTool", "Applied approval mask to surface."),
        2000);

    return true;
}

void ApprovalMaskBrushTool::clear()
{
    _strokeActive = false;
    _currentStroke.clear();
    _pendingStrokes.clear();
    _overlayPoints.clear();
    _overlayStrokeSegments.clear();
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();
    _hasLastSample = false;
    _hasLastOverlaySample = false;

    // Reload approval mask from disk to discard pending changes
    auto overlay = _module.overlay();
    if (overlay && _surface) {
        overlay->loadApprovalMaskImage(_surface);
        qCInfo(lcApprovalMask) << "Reloaded approval mask from disk (discarded pending changes)";
    }

    _module.refreshOverlay();
}

void ApprovalMaskBrushTool::paintAccumulatedPointsToImage()
{
    if (_accumulatedGridPositions.empty()) {
        return;
    }

    auto overlay = _module.overlay();
    if (!overlay) {
        qCWarning(lcApprovalMask) << "Cannot paint: no overlay controller";
        _accumulatedGridPositions.clear();
        _accumulatedGridPosSet.clear();
        return;
    }

    const uint8_t paintValue = (_paintMode == PaintMode::Approve) ? 255 : 0;

    // Get the current brush radius from the module
    const float brushRadius = _module.approvalMaskBrushRadius();
    const float clampedRadius = std::clamp(brushRadius, 0.5f, 10.0f);

    // Paint the accumulated points into the QImage
    overlay->paintApprovalMaskDirect(_accumulatedGridPositions, clampedRadius, paintValue);

    qCInfo(lcApprovalMask) << "[REALTIME] Painted" << _accumulatedGridPositions.size()
                           << "grid positions into QImage";

    // Clear for next batch
    _accumulatedGridPositions.clear();
    _accumulatedGridPosSet.clear();

    // Trigger overlay refresh to show the updated image
    _module.refreshOverlay();
}

std::optional<std::pair<int, int>> ApprovalMaskBrushTool::worldToGridIndex(const cv::Vec3f& worldPos) const
{
    // For approval mask, we must use the base surface directly, NOT the editManager's worldToGridIndex
    // which may use preview points from an active session with different dimensions/positions.
    if (!_surface) {
        qCWarning(lcApprovalMask) << "No surface available for worldToGridIndex";
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = _surface->rawPointsPtr();
    if (!points || points->empty()) {
        qCWarning(lcApprovalMask) << "No points on surface for worldToGridIndex";
        return std::nullopt;
    }

    static bool loggedOnce = false;
    if (!loggedOnce) {
        qCInfo(lcApprovalMask) << "  [worldToGridIndex] Surface grid size:" << points->cols << "x" << points->rows;
        qCInfo(lcApprovalMask) << "  [worldToGridIndex] Surface scale:" << _surface->scale()[0] << _surface->scale()[1] << _surface->scale()[2];
        // Log a few sample grid points
        if (points->rows > 0 && points->cols > 0) {
            qCInfo(lcApprovalMask) << "  [worldToGridIndex] Grid point (0,0):" << (*points)(0, 0)[0] << (*points)(0, 0)[1] << (*points)(0, 0)[2];
            qCInfo(lcApprovalMask) << "  [worldToGridIndex] Grid point (0," << (points->cols-1) << "):" << (*points)(0, points->cols-1)[0] << (*points)(0, points->cols-1)[1] << (*points)(0, points->cols-1)[2];
            int midRow = points->rows / 2;
            int midCol = points->cols / 2;
            qCInfo(lcApprovalMask) << "  [worldToGridIndex] Grid point (" << midRow << "," << midCol << "):" << (*points)(midRow, midCol)[0] << (*points)(midRow, midCol)[1] << (*points)(midRow, midCol)[2];
        }
        loggedOnce = true;
    }

    // Use surface's pointTo to find closest point
    cv::Vec3f ptr = _surface->pointer();
    _surface->pointTo(ptr, worldPos, std::numeric_limits<float>::max(), 400);  // High iteration count for accuracy
    cv::Vec3f raw = _surface->loc_raw(ptr);

    // Convert to approximate grid indices
    int approxCol = static_cast<int>(std::round(raw[0]));
    int approxRow = static_cast<int>(std::round(raw[1]));

    // Clamp to valid range for initial search
    approxRow = std::clamp(approxRow, 0, points->rows - 1);
    approxCol = std::clamp(approxCol, 0, points->cols - 1);

    // Search for the closest VALID point in a radius around the approximate position
    constexpr int kSearchRadius = 12;
    int bestRow = -1;
    int bestCol = -1;
    float bestDistSq = std::numeric_limits<float>::max();

    for (int radius = 0; radius <= kSearchRadius; ++radius) {
        const int rowStart = std::max(0, approxRow - radius);
        const int rowEnd = std::min(points->rows - 1, approxRow + radius);
        const int colStart = std::max(0, approxCol - radius);
        const int colEnd = std::min(points->cols - 1, approxCol + radius);

        for (int r = rowStart; r <= rowEnd; ++r) {
            for (int c = colStart; c <= colEnd; ++c) {
                const cv::Vec3f& candidate = (*points)(r, c);

                // Skip invalid points (e.g., -1, -1, -1 or NaN)
                if (isInvalidPoint(candidate)) {
                    continue;
                }

                // Track the closest valid point in grid space
                const int dr = r - approxRow;
                const int dc = c - approxCol;
                const float gridDistSq = static_cast<float>(dr * dr + dc * dc);

                if (gridDistSq < bestDistSq) {
                    bestDistSq = gridDistSq;
                    bestRow = r;
                    bestCol = c;
                }
            }
        }

        // Early exit if we found a valid point - don't keep searching
        if (bestRow != -1) {
            return std::make_pair(bestRow, bestCol);
        }
    }

    // No valid point found in search area - user is over a hole or out of bounds
    return std::nullopt;
}
