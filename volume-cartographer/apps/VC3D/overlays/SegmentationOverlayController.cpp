#include "SegmentationOverlayController.hpp"

#include "../CSurfaceCollection.hpp"
#include "../CVolumeViewer.hpp"

#include <QColor>
#include <QDebug>

#include <algorithm>
#include <cmath>
#include <chrono>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"

namespace
{
constexpr const char* kOverlayGroupKey = "segmentation_vertex_edit";
const QColor kActiveFill = QColor(255, 215, 0, 220);
const QColor kActiveBorder = QColor(255, 180, 0, 255);
const QColor kNeighborFill = QColor(0, 220, 255, 160);
const QColor kNeighborBorder = QColor(255, 255, 255, 210);
const QColor kGrowthFill = QColor(140, 255, 160, 210);
const QColor kGrowthBorder = QColor(80, 190, 90, 230);
const QColor kHoverFill = QColor(255, 255, 255, 200);
const QColor kHoverBorder = QColor(255, 255, 255, 240);
constexpr qreal kMarkerRadius = 5.0;
constexpr qreal kMarkerPenWidth = 2.0;
constexpr qreal kActivePenWidth = 2.5;
constexpr qreal kMaskZ = 60.0;
constexpr qreal kMarkerZ = 95.0;
constexpr qreal kRadiusCircleZ = 80.0;
}

bool SegmentationOverlayController::State::operator==(const State& rhs) const
{
    const auto equalMarker = [](const VertexMarker& lhs, const VertexMarker& rhs) {
        return lhs.row == rhs.row &&
               lhs.col == rhs.col &&
               std::fabs(lhs.world[0] - rhs.world[0]) < 1e-4f &&
               std::fabs(lhs.world[1] - rhs.world[1]) < 1e-4f &&
               std::fabs(lhs.world[2] - rhs.world[2]) < 1e-4f &&
               lhs.isActive == rhs.isActive &&
               lhs.isGrowth == rhs.isGrowth;
    };

    const auto optionalEqual = [&](const std::optional<VertexMarker>& lhsOpt,
                                   const std::optional<VertexMarker>& rhsOpt) {
        if (!lhsOpt && !rhsOpt) {
            return true;
        }
        if (static_cast<bool>(lhsOpt) != static_cast<bool>(rhsOpt)) {
            return false;
        }
        return equalMarker(*lhsOpt, *rhsOpt);
    };

    const auto vectorEqual = [&](const std::vector<VertexMarker>& lhsVec,
                                 const std::vector<VertexMarker>& rhsVec) {
        if (lhsVec.size() != rhsVec.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhsVec.size(); ++i) {
            if (!equalMarker(lhsVec[i], rhsVec[i])) {
                return false;
            }
        }
        return true;
    };

    const auto maskEqual = [&](const std::vector<cv::Vec3f>& lhsVec,
                               const std::vector<cv::Vec3f>& rhsVec) {
        if (lhsVec.size() != rhsVec.size()) {
            return false;
        }
        for (std::size_t i = 0; i < lhsVec.size(); ++i) {
            const cv::Vec3f delta = lhsVec[i] - rhsVec[i];
            if (cv::norm(delta) >= 1e-4f) {
                return false;
            }
        }
        return true;
    };

    const auto floatEqual = [](float lhs, float rhs) {
        return std::fabs(lhs - rhs) < 1e-4f;
    };

    return optionalEqual(activeMarker, rhs.activeMarker) &&
           vectorEqual(neighbours, rhs.neighbours) &&
           maskEqual(maskPoints, rhs.maskPoints) &&
           maskVisible == rhs.maskVisible &&
           brushActive == rhs.brushActive &&
           brushStrokeActive == rhs.brushStrokeActive &&
           lineStrokeActive == rhs.lineStrokeActive &&
           hasLineStroke == rhs.hasLineStroke &&
           pushPullActive == rhs.pushPullActive &&
           falloff == rhs.falloff &&
           floatEqual(gaussianRadiusSteps, rhs.gaussianRadiusSteps) &&
           floatEqual(gaussianSigmaSteps, rhs.gaussianSigmaSteps) &&
           floatEqual(displayRadiusSteps, rhs.displayRadiusSteps) &&
           floatEqual(gridStepWorld, rhs.gridStepWorld) &&
           approvalMaskMode == rhs.approvalMaskMode &&
           approvalStrokeActive == rhs.approvalStrokeActive &&
           approvalStrokeSegments == rhs.approvalStrokeSegments &&
           maskEqual(approvalCurrentStroke, rhs.approvalCurrentStroke) &&
           floatEqual(approvalBrushRadius, rhs.approvalBrushRadius) &&
           paintingApproval == rhs.paintingApproval &&
           surface == rhs.surface;
}

SegmentationOverlayController::SegmentationOverlayController(CSurfaceCollection* surfaces, QObject* parent)
    : ViewerOverlayControllerBase(kOverlayGroupKey, parent)
    , _surfaces(surfaces)
{
    if (_surfaces) {
        connect(_surfaces, &CSurfaceCollection::sendSurfaceChanged,
                this, &SegmentationOverlayController::onSurfaceChanged);
    }
}

void SegmentationOverlayController::setEditingEnabled(bool enabled)
{
    if (_editingEnabled == enabled) {
        return;
    }
    _editingEnabled = enabled;
    refreshAll();
}

void SegmentationOverlayController::setEditManager(SegmentationEditManager* manager)
{
    _editManager = manager;
}

void SegmentationOverlayController::applyState(const State& state)
{
    State sanitized = state;
    sanitized.gaussianRadiusSteps = std::max(sanitized.gaussianRadiusSteps, 0.0f);
    sanitized.gaussianSigmaSteps = std::max(sanitized.gaussianSigmaSteps, 0.0f);
    sanitized.displayRadiusSteps = std::max(sanitized.displayRadiusSteps, 0.0f);
    sanitized.gridStepWorld = std::max(sanitized.gridStepWorld, 1e-4f);

    _currentState = std::move(sanitized);
    refreshAll();
}

void SegmentationOverlayController::loadApprovalMaskImage(QuadSurface* surface)
{
    if (!surface) {
        _savedApprovalMaskImage = QImage();
        _pendingApprovalMaskImage = QImage();
        return;
    }

    cv::Mat approvalMask = surface->channel("approval", SURF_CHANNEL_NORESIZE);
    if (approvalMask.empty()) {
        // Create new empty mask
        const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
        if (!points || points->empty()) {
            _savedApprovalMaskImage = QImage();
            _pendingApprovalMaskImage = QImage();
            return;
        }
        approvalMask = cv::Mat_<uint8_t>(points->size(), static_cast<uint8_t>(0));
    }

    // Convert saved mask to ARGB32_Premultiplied format with DARK GREEN, semi-transparent
    QImage savedImage(approvalMask.cols, approvalMask.rows, QImage::Format_ARGB32_Premultiplied);
    for (int row = 0; row < approvalMask.rows; ++row) {
        const uint8_t* maskRow = approvalMask.ptr<uint8_t>(row);
        QRgb* imageRow = reinterpret_cast<QRgb*>(savedImage.scanLine(row));
        for (int col = 0; col < approvalMask.cols; ++col) {
            uint8_t val = maskRow[col];
            if (val > 0) {
                // DARK GREEN: RGB(0, 100, 0) with 60% opacity (alpha = 153)
                // Premultiply: R=0, G=60, B=0, A=153
                imageRow[col] = qRgba(0, 60, 0, 153);
            } else {
                imageRow[col] = qRgba(0, 0, 0, 0);  // Fully transparent
            }
        }
    }

    _savedApprovalMaskImage = savedImage;

    // Initialize empty pending mask with same dimensions
    _pendingApprovalMaskImage = QImage(approvalMask.cols, approvalMask.rows, QImage::Format_ARGB32_Premultiplied);
    _pendingApprovalMaskImage.fill(qRgba(0, 0, 0, 0));  // Fully transparent
}

void SegmentationOverlayController::paintApprovalMaskDirect(
    const std::vector<std::pair<int, int>>& gridPositions,
    float radiusSteps,
    uint8_t paintValue)
{
    if (_pendingApprovalMaskImage.isNull()) {
        return;
    }

    const int radius = static_cast<int>(std::ceil(radiusSteps));
    const float sigma = radiusSteps / 2.0f;

    // Gaussian falloff function
    auto gaussianFalloff = [](float distance, float sigma) -> float {
        if (sigma <= 0.0f) {
            return distance <= 0.0f ? 1.0f : 0.0f;
        }
        return std::exp(-(distance * distance) / (2.0f * sigma * sigma));
    };

    // Paint directly into the PENDING QImage
    for (const auto& [centerRow, centerCol] : gridPositions) {
        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                const int row = centerRow + dr;
                const int col = centerCol + dc;

                if (row < 0 || row >= _pendingApprovalMaskImage.height() ||
                    col < 0 || col >= _pendingApprovalMaskImage.width()) {
                    continue;
                }

                const float distance = std::sqrt(static_cast<float>(dr * dr + dc * dc));
                if (distance > radiusSteps) {
                    continue;
                }

                const float falloff = gaussianFalloff(distance, sigma);
                QRgb pixel = _pendingApprovalMaskImage.pixel(col, row);

                // Extract current green value
                float currentGreen = static_cast<float>(qGreen(pixel));
                float targetGreen = static_cast<float>(paintValue);
                float blendedGreen = currentGreen * (1.0f - falloff) + targetGreen * falloff;
                uint8_t newVal = static_cast<uint8_t>(std::clamp(blendedGreen, 0.0f, 255.0f));

                // Update pixel with LIGHT GREEN, semi-transparent
                // LIGHT GREEN: RGB(0, 200, 0) with 50% opacity (alpha = 128)
                // Premultiply: R=0, G=100, B=0, A=128
                if (newVal > 0) {
                    _pendingApprovalMaskImage.setPixel(col, row, qRgba(0, 100, 0, 128));
                } else {
                    _pendingApprovalMaskImage.setPixel(col, row, qRgba(0, 0, 0, 0));  // Fully transparent
                }
            }
        }
    }
}

void SegmentationOverlayController::saveApprovalMaskToSurface(QuadSurface* surface)
{
    if (!surface || (_savedApprovalMaskImage.isNull() && _pendingApprovalMaskImage.isNull())) {
        return;
    }

    // Merge pending into saved, then convert to cv::Mat
    int width = !_savedApprovalMaskImage.isNull() ? _savedApprovalMaskImage.width() : _pendingApprovalMaskImage.width();
    int height = !_savedApprovalMaskImage.isNull() ? _savedApprovalMaskImage.height() : _pendingApprovalMaskImage.height();

    cv::Mat_<uint8_t> approvalMask(height, width, static_cast<uint8_t>(0));

    for (int row = 0; row < height; ++row) {
        uint8_t* maskRow = approvalMask.ptr<uint8_t>(row);
        for (int col = 0; col < width; ++col) {
            uint8_t savedVal = 0;
            uint8_t pendingVal = 0;

            // Get saved value
            if (!_savedApprovalMaskImage.isNull() && row < _savedApprovalMaskImage.height() && col < _savedApprovalMaskImage.width()) {
                const QRgb* savedRow = reinterpret_cast<const QRgb*>(_savedApprovalMaskImage.constScanLine(row));
                savedVal = qGreen(savedRow[col]);
            }

            // Get pending value
            if (!_pendingApprovalMaskImage.isNull() && row < _pendingApprovalMaskImage.height() && col < _pendingApprovalMaskImage.width()) {
                const QRgb* pendingRow = reinterpret_cast<const QRgb*>(_pendingApprovalMaskImage.constScanLine(row));
                pendingVal = qGreen(pendingRow[col]);
            }

            // Merge: take max value (approved = 255)
            maskRow[col] = std::max(savedVal, pendingVal);
        }
    }

    // Save to surface
    surface->setChannel("approval", approvalMask);
    surface->saveOverwrite();

    // Merge pending into saved and clear pending
    for (int row = 0; row < height; ++row) {
        QRgb* savedRow = reinterpret_cast<QRgb*>(_savedApprovalMaskImage.scanLine(row));
        for (int col = 0; col < width; ++col) {
            uint8_t mergedVal = approvalMask(row, col);
            if (mergedVal > 0) {
                // Update saved image with dark green
                savedRow[col] = qRgba(0, 60, 0, 153);
            } else {
                savedRow[col] = qRgba(0, 0, 0, 0);
            }
        }
    }

    // Clear pending
    _pendingApprovalMaskImage.fill(qRgba(0, 0, 0, 0));
}

bool SegmentationOverlayController::isOverlayEnabledFor(CVolumeViewer* viewer) const
{
    Q_UNUSED(viewer);
    return _editingEnabled;
}

void SegmentationOverlayController::collectPrimitives(CVolumeViewer* viewer,
                                                      ViewerOverlayControllerBase::OverlayBuilder& builder)
{
    if (!viewer || !_editingEnabled || !_currentState) {
        return;
    }

    const State& state = *_currentState;

    // Render approval mask overlays (dark green for saved, light green for pending)
    if (state.approvalMaskMode && state.surface) {
        buildApprovalMaskOverlay(state, viewer, builder);
    }

    if (shouldShowMask(state)) {
        builder.addPath(buildMaskPrimitive(state));
    }

    buildVertexMarkers(state, viewer, builder);
    buildRadiusOverlay(state, viewer, builder);
}

ViewerOverlayControllerBase::PathPrimitive SegmentationOverlayController::buildMaskPrimitive(const State& state) const
{
    ViewerOverlayControllerBase::PathPrimitive maskPath;
    maskPath.points = state.maskPoints;
    maskPath.renderMode = state.hasLineStroke ? ViewerOverlayControllerBase::PathRenderMode::LineStrip
                                              : ViewerOverlayControllerBase::PathRenderMode::Points;
    maskPath.brushShape = ViewerOverlayControllerBase::PathBrushShape::Circle;

    const float brushPixelRadius = std::clamp(state.displayRadiusSteps * 1.5f, 3.0f, 18.0f);
    const bool drawingOverlay = state.brushActive || state.brushStrokeActive || state.lineStrokeActive || state.hasLineStroke;
    const float brushOpacity = drawingOverlay ? 0.6f : 0.45f;

    maskPath.pointRadius = state.hasLineStroke ? std::max(brushPixelRadius * 0.35f, 2.0f) : brushPixelRadius;
    maskPath.lineWidth = state.hasLineStroke ? 3.0f : std::max(brushPixelRadius * 0.5f, 2.0f);
    maskPath.color = state.hasLineStroke ? QColor(80, 170, 255) : QColor(255, 140, 0);
    maskPath.opacity = state.hasLineStroke ? 0.85f : brushOpacity;
    maskPath.z = kMaskZ;

    return maskPath;
}

bool SegmentationOverlayController::shouldShowMask(const State& state) const
{
    return state.maskVisible && !state.maskPoints.empty();
}

void SegmentationOverlayController::onSurfaceChanged(std::string name, Surface* surface)
{
    Q_UNUSED(surface);
    if (name == "segmentation") {
        refreshAll();
    }
}

void SegmentationOverlayController::buildRadiusOverlay(const State& state,
                                                       CVolumeViewer* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!state.activeMarker || !state.activeMarker->isActive) {
        return;
    }

    const float radiusWorld = state.gaussianRadiusSteps * state.gridStepWorld;
    if (radiusWorld <= 0.0f) {
        return;
    }

    const cv::Vec3f world = state.activeMarker->world;
    const QPointF sceneCenter = viewer->volumePointToScene(world);

    cv::Vec3f offsetWorld = world;
    offsetWorld[0] += radiusWorld;
    const QPointF sceneEdge = viewer->volumePointToScene(offsetWorld);
    const qreal radiusPixels = std::hypot(sceneEdge.x() - sceneCenter.x(),
                                          sceneEdge.y() - sceneCenter.y());

    if (radiusPixels <= 1.0e-3) {
        return;
    }

    ViewerOverlayControllerBase::OverlayStyle style;
    style.penColor = QColor(255, 255, 255, 120);
    style.penWidth = 1.5f;
    style.brushColor = QColor(Qt::transparent);
    style.z = kRadiusCircleZ;

    builder.addCircle(sceneCenter, radiusPixels, false, style);
}

void SegmentationOverlayController::buildVertexMarkers(const State& state,
                                                       CVolumeViewer* viewer,
                                                       ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    const auto buildStyle = [](const VertexMarker& marker) {
        ViewerOverlayControllerBase::OverlayStyle style;
        style.penWidth = marker.isActive ? kActivePenWidth : kMarkerPenWidth;
        style.z = kMarkerZ;

        if (marker.isActive) {
            style.penColor = kActiveBorder;
            style.brushColor = kActiveFill;
        } else if (marker.isGrowth) {
            style.penColor = kGrowthBorder;
            style.brushColor = kGrowthFill;
        } else {
            style.penColor = kNeighborBorder;
            style.brushColor = kNeighborFill;
        }

        return style;
    };

    const auto appendMarker = [&](VertexMarker marker) {
        const QPointF scene = viewer->volumePointToScene(marker.world);
        const auto style = buildStyle(marker);
        builder.addCircle(scene, kMarkerRadius, true, style);
    };

    for (const auto& marker : state.neighbours) {
        appendMarker(marker);
    }

    if (state.activeMarker) {
        auto active = *state.activeMarker;
        active.isActive = true;
        appendMarker(active);
    }
}

void SegmentationOverlayController::buildApprovalMaskOverlay(const State& state,
                                                              CVolumeViewer* viewer,
                                                              ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    if (!state.surface) {
        return;
    }

    const cv::Mat_<cv::Vec3f>* points = state.surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    // Get surface grid dimensions
    const int gridRows = points->rows;
    const int gridCols = points->cols;

    // Check if we have valid images
    const bool hasSaved = !_savedApprovalMaskImage.isNull() &&
                         _savedApprovalMaskImage.width() == gridCols &&
                         _savedApprovalMaskImage.height() == gridRows;
    const bool hasPending = !_pendingApprovalMaskImage.isNull() &&
                           _pendingApprovalMaskImage.width() == gridCols &&
                           _pendingApprovalMaskImage.height() == gridRows;

    if (!hasSaved && !hasPending) {
        return;
    }

    // Collect world points for saved (dark green) approval mask
    std::vector<cv::Vec3f> savedPoints;
    if (hasSaved) {
        savedPoints.reserve(gridRows * gridCols / 10);  // Rough estimate
        for (int row = 0; row < gridRows; ++row) {
            for (int col = 0; col < gridCols; ++col) {
                // Check if this pixel is painted (non-transparent)
                QRgb pixel = _savedApprovalMaskImage.pixel(col, row);
                if (qAlpha(pixel) > 0) {
                    const cv::Vec3f& worldPos = (*points)(row, col);
                    // Skip invalid points
                    if (std::isfinite(worldPos[0]) && std::isfinite(worldPos[1]) && std::isfinite(worldPos[2]) &&
                        !(worldPos[0] == -1.0f && worldPos[1] == -1.0f && worldPos[2] == -1.0f)) {
                        savedPoints.push_back(worldPos);
                    }
                }
            }
        }
    }

    // Collect world points for pending (light green) approval mask
    std::vector<cv::Vec3f> pendingPoints;
    if (hasPending) {
        pendingPoints.reserve(gridRows * gridCols / 10);  // Rough estimate
        for (int row = 0; row < gridRows; ++row) {
            for (int col = 0; col < gridCols; ++col) {
                // Check if this pixel is painted (non-transparent)
                QRgb pixel = _pendingApprovalMaskImage.pixel(col, row);
                if (qAlpha(pixel) > 0) {
                    const cv::Vec3f& worldPos = (*points)(row, col);
                    // Skip invalid points
                    if (std::isfinite(worldPos[0]) && std::isfinite(worldPos[1]) && std::isfinite(worldPos[2]) &&
                        !(worldPos[0] == -1.0f && worldPos[1] == -1.0f && worldPos[2] == -1.0f)) {
                        pendingPoints.push_back(worldPos);
                    }
                }
            }
        }
    }

    // Render saved approval mask as PathPrimitive (dark green)
    if (!savedPoints.empty()) {
        ViewerOverlayControllerBase::PathPrimitive savedPath;
        savedPath.points = savedPoints;
        savedPath.renderMode = ViewerOverlayControllerBase::PathRenderMode::Points;
        savedPath.brushShape = ViewerOverlayControllerBase::PathBrushShape::Square;
        savedPath.pointRadius = 1.5f;  // Small squares
        savedPath.color = QColor(0, 100, 0);  // Dark green
        savedPath.opacity = 0.6f;
        savedPath.z = 10.0;
        builder.addPath(savedPath);
    }

    // Render pending approval mask as PathPrimitive (light green)
    if (!pendingPoints.empty()) {
        ViewerOverlayControllerBase::PathPrimitive pendingPath;
        pendingPath.points = pendingPoints;
        pendingPath.renderMode = ViewerOverlayControllerBase::PathRenderMode::Points;
        pendingPath.brushShape = ViewerOverlayControllerBase::PathBrushShape::Square;
        pendingPath.pointRadius = 1.5f;  // Small squares
        pendingPath.color = QColor(0, 200, 0);  // Light green
        pendingPath.opacity = 0.5f;
        pendingPath.z = 15.0;
        builder.addPath(pendingPath);
    }
}
