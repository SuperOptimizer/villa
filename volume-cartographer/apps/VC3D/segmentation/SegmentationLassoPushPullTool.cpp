#include "SegmentationLassoPushPullTool.hpp"

#include "SegmentationModule.hpp"
#include "SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "CSurfaceCollection.hpp"

#include "vc/core/util/QuadSurface.hpp"

#include <QTimer>
#include <QLoggingCategory>

#include <algorithm>
#include <cmath>
#include <limits>

Q_LOGGING_CATEGORY(lcSegLassoPushPull, "vc.segmentation.lassopushpull")

namespace
{

bool isFiniteVec3(const cv::Vec3f& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool isValidNormal(const cv::Vec3f& normal)
{
    if (!isFiniteVec3(normal)) {
        return false;
    }
    const float norm = static_cast<float>(cv::norm(normal));
    return norm > 1e-4f;
}

cv::Vec3f normalizeVec(const cv::Vec3f& value)
{
    const float norm = static_cast<float>(cv::norm(value));
    if (!std::isfinite(norm) || norm <= 1e-6f) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    return value / norm;
}

}  // namespace

SegmentationLassoPushPullTool::SegmentationLassoPushPullTool(
    SegmentationModule& module,
    SegmentationEditManager* editManager,
    SegmentationOverlayController* overlay,
    CSurfaceCollection* surfaces)
    : _module(module)
    , _editManager(editManager)
    , _overlay(overlay)
    , _surfaces(surfaces)
{
}

void SegmentationLassoPushPullTool::setDependencies(
    SegmentationEditManager* editManager,
    SegmentationOverlayController* overlay,
    CSurfaceCollection* surfaces)
{
    _editManager = editManager;
    _overlay = overlay;
    _surfaces = surfaces;
}

void SegmentationLassoPushPullTool::startLasso(const cv::Vec3f& worldPos)
{
    // Cancel any existing operation
    if (_state != State::Idle) {
        cancel();
    }

    _currentStroke.clear();
    _currentStroke.push_back(worldPos);
    _lastSample = worldPos;
    _hasLastSample = true;
    _state = State::Drawing;

    qCInfo(lcSegLassoPushPull) << "Started lasso at" << worldPos[0] << worldPos[1] << worldPos[2];
}

void SegmentationLassoPushPullTool::extendLasso(const cv::Vec3f& worldPos, bool forceSample)
{
    if (_state != State::Drawing) {
        return;
    }

    // Check if we should add a new sample based on distance
    bool shouldAdd = forceSample;
    if (!shouldAdd && _hasLastSample) {
        const cv::Vec3f delta = worldPos - _lastSample;
        const float dist = static_cast<float>(cv::norm(delta));
        shouldAdd = dist >= kSampleSpacing;
    }

    if (shouldAdd) {
        _currentStroke.push_back(worldPos);
        _lastSample = worldPos;
        _hasLastSample = true;
    }
}

bool SegmentationLassoPushPullTool::finishLasso()
{
    if (_state != State::Drawing) {
        return false;
    }

    qCInfo(lcSegLassoPushPull) << "Finishing lasso with" << _currentStroke.size() << "points";

    // Need at least 3 points for a valid polygon
    if (_currentStroke.size() < 3) {
        qCWarning(lcSegLassoPushPull) << "Lasso too small (< 3 points), canceling";
        cancel();
        return false;
    }

    // Build selection from the lasso polygon
    if (!buildSelectionFromLasso()) {
        qCWarning(lcSegLassoPushPull) << "Failed to build selection from lasso";
        cancel();
        return false;
    }

    _state = State::SelectionActive;
    qCInfo(lcSegLassoPushPull) << "Lasso selection active with" << _selection.samples.size() << "vertices";
    return true;
}

bool SegmentationLassoPushPullTool::buildSelectionFromLasso()
{
    if (!_editManager || !_editManager->hasSession()) {
        return false;
    }

    // Convert world points to grid coordinates
    _selection.polygonWorld = _currentStroke;
    _selection.polygonGrid.clear();
    _selection.polygonGrid.reserve(_currentStroke.size());

    for (const auto& worldPt : _currentStroke) {
        auto gridIdx = _editManager->worldToGridIndex(worldPt);
        if (gridIdx) {
            _selection.polygonGrid.push_back(
                cv::Vec2f(static_cast<float>(gridIdx->second),   // col
                          static_cast<float>(gridIdx->first)));  // row
        }
    }

    if (_selection.polygonGrid.size() < 3) {
        qCWarning(lcSegLassoPushPull) << "Not enough valid grid points for polygon";
        return false;
    }

    // Compute bounding box of polygon in grid coords
    float minRow = std::numeric_limits<float>::max();
    float maxRow = std::numeric_limits<float>::lowest();
    float minCol = std::numeric_limits<float>::max();
    float maxCol = std::numeric_limits<float>::lowest();

    for (const auto& pt : _selection.polygonGrid) {
        minCol = std::min(minCol, pt[0]);
        maxCol = std::max(maxCol, pt[0]);
        minRow = std::min(minRow, pt[1]);
        maxRow = std::max(maxRow, pt[1]);
    }

    // Collect all grid vertices inside the polygon
    _selection.samples.clear();
    const auto& previewPoints = _editManager->previewPoints();

    const int rowStart = std::max(0, static_cast<int>(std::floor(minRow)));
    const int rowEnd = std::min(previewPoints.rows - 1, static_cast<int>(std::ceil(maxRow)));
    const int colStart = std::max(0, static_cast<int>(std::floor(minCol)));
    const int colEnd = std::min(previewPoints.cols - 1, static_cast<int>(std::ceil(maxCol)));

    qCInfo(lcSegLassoPushPull) << "Searching grid region: rows" << rowStart << "-" << rowEnd
                               << "cols" << colStart << "-" << colEnd;

    for (int row = rowStart; row <= rowEnd; ++row) {
        for (int col = colStart; col <= colEnd; ++col) {
            if (!pointInPolygon(static_cast<float>(row), static_cast<float>(col))) {
                continue;
            }

            const cv::Vec3f& worldPos = previewPoints(row, col);
            if (!isFiniteVec3(worldPos)) {
                continue;
            }

            SegmentationEditManager::DragSample sample;
            sample.row = row;
            sample.col = col;
            sample.baseWorld = worldPos;
            sample.distanceWorldSq = 0.0f;  // Will be computed after centroid is known
            _selection.samples.push_back(sample);
        }
    }

    if (_selection.samples.empty()) {
        qCWarning(lcSegLassoPushPull) << "No valid vertices inside lasso";
        return false;
    }

    computeCentroidAndNormal();
    return true;
}

void SegmentationLassoPushPullTool::computeCentroidAndNormal()
{
    if (_selection.samples.empty()) {
        return;
    }

    // Compute centroid
    cv::Vec3f sum(0, 0, 0);
    for (const auto& sample : _selection.samples) {
        sum += sample.baseWorld;
    }
    _selection.centroidWorld = sum * (1.0f / static_cast<float>(_selection.samples.size()));

    // Find nearest grid point to centroid
    float minDist = std::numeric_limits<float>::max();
    for (const auto& sample : _selection.samples) {
        const cv::Vec3f diff = sample.baseWorld - _selection.centroidWorld;
        const float dist = diff.dot(diff);
        if (dist < minDist) {
            minDist = dist;
            _selection.centroidGrid = {sample.row, sample.col};
        }
    }

    // Compute distances from centroid for Gaussian falloff
    for (auto& sample : _selection.samples) {
        const cv::Vec3f diff = sample.baseWorld - _selection.centroidWorld;
        sample.distanceWorldSq = diff.dot(diff);
    }

    // Compute average normal using grid normals
    auto baseSurface = _editManager->baseSurface();
    if (baseSurface) {
        cv::Vec3f normalSum(0, 0, 0);
        int validNormals = 0;
        for (const auto& sample : _selection.samples) {
            cv::Vec3f normal = baseSurface->gridNormal(sample.row, sample.col);
            if (isValidNormal(normal)) {
                normalSum += normalizeVec(normal);
                ++validNormals;
            }
        }
        if (validNormals > 0) {
            _selection.averageNormal = normalSum / static_cast<float>(validNormals);
            const float norm = static_cast<float>(cv::norm(_selection.averageNormal));
            if (norm > 1e-4f) {
                _selection.averageNormal /= norm;
            } else {
                _selection.averageNormal = cv::Vec3f(0, 0, 1);
            }
        }
    }

    qCInfo(lcSegLassoPushPull) << "Centroid:" << _selection.centroidWorld[0]
                               << _selection.centroidWorld[1] << _selection.centroidWorld[2]
                               << "Normal:" << _selection.averageNormal[0]
                               << _selection.averageNormal[1] << _selection.averageNormal[2];
}

bool SegmentationLassoPushPullTool::pointInPolygon(float gridRow, float gridCol) const
{
    const auto& polygon = _selection.polygonGrid;
    if (polygon.size() < 3) {
        return false;
    }

    // Ray casting algorithm
    bool inside = false;
    const size_t n = polygon.size();

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        // polygon[i] is (col, row), so:
        const float yi = polygon[i][1];  // row
        const float xi = polygon[i][0];  // col
        const float yj = polygon[j][1];
        const float xj = polygon[j][0];

        if (((yi > gridRow) != (yj > gridRow)) &&
            (gridCol < (xj - xi) * (gridRow - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }

    return inside;
}

void SegmentationLassoPushPullTool::clearSelection()
{
    _state = State::Idle;
    _currentStroke.clear();
    _selection = LassoSelection{};
    _hasLastSample = false;
    _pushPullActive = false;
    _pushPullDirection = 0;
    _undoCaptured = false;

    if (_timer && _timer->isActive()) {
        _timer->stop();
    }
}

void SegmentationLassoPushPullTool::cancel()
{
    const bool wasActive = _state != State::Idle || _pushPullActive;

    if (_pushPullActive) {
        stopAllPushPull();
    }

    clearSelection();

    if (wasActive) {
        _module.refreshOverlay();
    }
}

bool SegmentationLassoPushPullTool::isActive() const
{
    return _state != State::Idle || _pushPullActive;
}

void SegmentationLassoPushPullTool::setStepMultiplier(float multiplier)
{
    _stepMultiplier = std::clamp(multiplier, 0.05f, 10.0f);
}

bool SegmentationLassoPushPullTool::startPushPull(int direction)
{
    if (direction == 0) {
        return false;
    }

    if (_state != State::SelectionActive) {
        qCWarning(lcSegLassoPushPull) << "Cannot start push/pull: no active selection";
        return false;
    }

    if (_selection.samples.empty()) {
        qCWarning(lcSegLassoPushPull) << "Cannot start push/pull: selection is empty";
        return false;
    }

    if (!_editManager || !_editManager->hasSession()) {
        qCWarning(lcSegLassoPushPull) << "Cannot start push/pull: no editing session";
        return false;
    }

    ensureTimer();

    if (_pushPullActive && _pushPullDirection == direction) {
        // Already running in this direction
        if (_timer && !_timer->isActive()) {
            _timer->start();
        }
        return true;
    }

    _pushPullActive = true;
    _pushPullDirection = direction;
    _undoCaptured = false;

    if (_timer) {
        _timer->setInterval(kPushPullIntervalMs);
        if (!_timer->isActive()) {
            _timer->start();
        }
    }

    qCInfo(lcSegLassoPushPull) << "Started lasso push/pull, direction:" << direction;
    return true;
}

void SegmentationLassoPushPullTool::stopPushPull(int direction)
{
    if (!_pushPullActive) {
        return;
    }
    if (direction != 0 && direction != _pushPullDirection) {
        return;
    }
    stopAllPushPull();
}

void SegmentationLassoPushPullTool::stopAllPushPull()
{
    const bool wasActive = _pushPullActive;
    _pushPullActive = false;
    _pushPullDirection = 0;

    if (_timer && _timer->isActive()) {
        _timer->stop();
    }

    if (wasActive && _editManager && _editManager->hasSession() && _surfaces) {
        // Capture undo delta
        _module.captureUndoDelta();

        // Auto-approve edited regions using our selection samples
        if (_overlay && _overlay->hasApprovalMaskData() && !_selection.samples.empty()) {
            std::vector<std::pair<int, int>> gridPositions;
            gridPositions.reserve(_selection.samples.size());
            for (const auto& sample : _selection.samples) {
                gridPositions.emplace_back(sample.row, sample.col);
            }
            constexpr uint8_t kApproved = 255;
            constexpr float kRadius = 1.0f;
            constexpr bool kIsAutoApproval = true;
            const QColor brushColor = _module.approvalBrushColor();
            _overlay->paintApprovalMaskDirect(gridPositions, kRadius, kApproved, brushColor,
                                              false, 0.0f, 0.0f, kIsAutoApproval);
            _overlay->scheduleDebouncedSave(_editManager->baseSurface().get());
            qCInfo(lcSegLassoPushPull) << "Auto-approved" << gridPositions.size()
                                       << "lasso push/pull edited vertices";
        }

        _editManager->applyPreview();
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
        _module.emitPendingChanges();
    }

    _undoCaptured = false;
    qCInfo(lcSegLassoPushPull) << "Stopped lasso push/pull";
}

bool SegmentationLassoPushPullTool::applyPushPullStep()
{
    return applyStepInternal();
}

bool SegmentationLassoPushPullTool::applyStepInternal()
{
    if (!_pushPullActive || _state != State::SelectionActive) {
        return false;
    }

    if (!_editManager || !_editManager->hasSession()) {
        qCWarning(lcSegLassoPushPull) << "Push/pull step aborted: no editing session";
        return false;
    }

    if (_selection.samples.empty()) {
        qCWarning(lcSegLassoPushPull) << "Push/pull step aborted: selection empty";
        return false;
    }

    // Compute step size
    const float stepWorld = _module.gridStepWorld() * _stepMultiplier;
    if (stepWorld <= 0.0f) {
        qCWarning(lcSegLassoPushPull) << "Push/pull step aborted: invalid step size";
        return false;
    }

    const cv::Vec3f delta = _selection.averageNormal *
                            (static_cast<float>(_pushPullDirection) * stepWorld);

    // Compute Gaussian sigma based on max distance in selection
    float maxDistSq = 0.0f;
    for (const auto& sample : _selection.samples) {
        maxDistSq = std::max(maxDistSq, sample.distanceWorldSq);
    }

    // Sigma = half the max radius for nice falloff
    const float sigmaWorld = std::sqrt(maxDistSq) * 0.5f;
    const float invTwoSigmaSq = (sigmaWorld > 1e-4f)
                                    ? 1.0f / (2.0f * sigmaWorld * sigmaWorld)
                                    : 0.0f;

    // Apply Gaussian-weighted displacement
    auto& previewPoints = _editManager->previewPointsMutable();

    for (auto& sample : _selection.samples) {
        float weight = 1.0f;
        if (sample.distanceWorldSq > 0.0f && invTwoSigmaSq > 0.0f) {
            weight = std::exp(-sample.distanceWorldSq * invTwoSigmaSq);
        }

        const cv::Vec3f newWorld = sample.baseWorld + delta * weight;
        previewPoints(sample.row, sample.col) = newWorld;

        // Update base position for next tick
        sample.baseWorld = newWorld;
    }

    // Recompute centroid for next step (it moves with the surface)
    cv::Vec3f sum(0, 0, 0);
    for (const auto& sample : _selection.samples) {
        sum += sample.baseWorld;
    }
    _selection.centroidWorld = sum * (1.0f / static_cast<float>(_selection.samples.size()));

    // Recompute distances from new centroid
    for (auto& sample : _selection.samples) {
        const cv::Vec3f diff = sample.baseWorld - _selection.centroidWorld;
        sample.distanceWorldSq = diff.dot(diff);
    }

    // Update visuals
    if (_surfaces) {
        _surfaces->setSurface("segmentation", _editManager->previewSurface(), false, true);
    }

    _module.refreshOverlay();
    _module.markAutosaveNeeded();
    return true;
}

void SegmentationLassoPushPullTool::ensureTimer()
{
    if (_timer) {
        return;
    }

    _timer = new QTimer(&_module);
    _timer->setInterval(kPushPullIntervalMs);
    QObject::connect(_timer, &QTimer::timeout, &_module, [this]() {
        if (!applyStepInternal()) {
            stopAllPushPull();
        }
    });
}

void SegmentationLassoPushPullTool::updateSampleBasePositions()
{
    if (!_editManager || !_editManager->hasSession()) {
        return;
    }

    const auto& previewPoints = _editManager->previewPoints();
    for (auto& sample : _selection.samples) {
        if (sample.row >= 0 && sample.row < previewPoints.rows &&
            sample.col >= 0 && sample.col < previewPoints.cols) {
            sample.baseWorld = previewPoints(sample.row, sample.col);
        }
    }
}
