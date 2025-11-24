#include "SegmentationOverlayController.hpp"

#include "../CSurfaceCollection.hpp"
#include "../CVolumeViewer.hpp"

#include <QColor>
#include <QDebug>
#include <QElapsedTimer>
#include <QPainter>

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
constexpr qreal kApprovalMaskZ = 50.0;  // Below mask path (60) but above most other overlays
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

    // Invalidate all viewer caches
    ++_savedImageVersion;
    ++_pendingImageVersion;
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

    // Invalidate pending version since we modified the pending image
    ++_pendingImageVersion;
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

    // Invalidate both versions
    ++_savedImageVersion;
    ++_pendingImageVersion;
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

void SegmentationOverlayController::rebuildViewerCache(CVolumeViewer* viewer, QuadSurface* surface) const
{
    QElapsedTimer totalTimer;
    totalTimer.start();

    qDebug() << "[ApprovalCache] ===== REBUILD CACHE START =====";

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        qDebug() << "[ApprovalCache] No points, returning";
        return;
    }

    const int gridRows = points->rows;
    const int gridCols = points->cols;
    qDebug() << "[ApprovalCache] Grid size:" << gridCols << "x" << gridRows;

    const bool hasSaved = !_savedApprovalMaskImage.isNull() &&
                         _savedApprovalMaskImage.width() == gridCols &&
                         _savedApprovalMaskImage.height() == gridRows;
    const bool hasPending = !_pendingApprovalMaskImage.isNull() &&
                           _pendingApprovalMaskImage.width() == gridCols &&
                           _pendingApprovalMaskImage.height() == gridRows;

    qDebug() << "[ApprovalCache] hasSaved:" << hasSaved << "hasPending:" << hasPending;

    if (!hasSaved && !hasPending) {
        qDebug() << "[ApprovalCache] No masks to render, returning";
        return;
    }

    // Simple approach: composite saved and pending images using QPainter (fast!)
    QElapsedTimer mergeTimer;
    mergeTimer.start();

    QImage compositeImage(gridCols, gridRows, QImage::Format_ARGB32_Premultiplied);
    compositeImage.fill(Qt::transparent);

    QPainter painter(&compositeImage);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);

    // Draw saved (dark green) first
    if (hasSaved) {
        painter.drawImage(0, 0, _savedApprovalMaskImage);
    }

    // Draw pending (light green) on top
    if (hasPending) {
        painter.drawImage(0, 0, _pendingApprovalMaskImage);
    }

    painter.end();

    qDebug() << "[ApprovalCache] Merge took:" << mergeTimer.elapsed() << "ms";

    // Calculate scene-space bounds to determine scale
    QElapsedTimer boundsTimer;
    boundsTimer.start();

    QRectF sceneBounds;
    bool foundAny = false;

    // Sample corners and edges to find scene bounds
    std::vector<std::pair<int, int>> samplePoints;
    for (int row = 0; row < gridRows; row += std::max(1, gridRows / 20)) {
        for (int col = 0; col < gridCols; col += std::max(1, gridCols / 20)) {
            samplePoints.push_back({row, col});
        }
    }

    for (const auto& [row, col] : samplePoints) {
        const cv::Vec3f& worldPos = (*points)(row, col);
        if (!std::isfinite(worldPos[0]) || !std::isfinite(worldPos[1]) || !std::isfinite(worldPos[2]) ||
            (worldPos[0] == -1.0f && worldPos[1] == -1.0f && worldPos[2] == -1.0f)) {
            continue;
        }

        QPointF scenePos = viewer->volumePointToScene(worldPos);
        if (!foundAny) {
            sceneBounds = QRectF(scenePos, QSizeF(1, 1));
            foundAny = true;
        } else {
            if (scenePos.x() < sceneBounds.left()) sceneBounds.setLeft(scenePos.x());
            if (scenePos.x() > sceneBounds.right()) sceneBounds.setRight(scenePos.x());
            if (scenePos.y() < sceneBounds.top()) sceneBounds.setTop(scenePos.y());
            if (scenePos.y() > sceneBounds.bottom()) sceneBounds.setBottom(scenePos.y());
        }
    }

    qDebug() << "[ApprovalCache] Bounds calc took:" << boundsTimer.elapsed() << "ms";

    if (!foundAny) {
        qDebug() << "[ApprovalCache] No valid scene bounds found, returning";
        return;
    }

    // Calculate scale to map from grid-space to scene-space
    const qreal sceneWidth = sceneBounds.width();
    const qreal sceneHeight = sceneBounds.height();
    const qreal scaleX = sceneWidth / static_cast<qreal>(gridCols);
    const qreal scaleY = sceneHeight / static_cast<qreal>(gridRows);
    const qreal scale = std::max(scaleX, scaleY);

    qDebug() << "[ApprovalCache] Scene bounds:" << sceneBounds;
    qDebug() << "[ApprovalCache] Grid size:" << gridCols << "x" << gridRows;
    qDebug() << "[ApprovalCache] Calculated scale:" << scale << "(scaleX=" << scaleX << "scaleY=" << scaleY << ")";

    // Store in cache
    ViewerImageCache& cache = _viewerCaches[viewer];
    cache.compositeImage = compositeImage;
    cache.topLeft = sceneBounds.topLeft();
    cache.scale = scale;
    cache.surface = surface;
    cache.savedImageVersion = _savedImageVersion;
    cache.pendingImageVersion = _pendingImageVersion;

    qDebug() << "[ApprovalCache] Storing: topLeft=" << cache.topLeft << "imageSize=" << compositeImage.size() << "scale=" << cache.scale;
    qDebug() << "[ApprovalCache] ===== REBUILD CACHE COMPLETE: TOTAL" << totalTimer.elapsed() << "ms =====";
}

void SegmentationOverlayController::buildApprovalMaskOverlay(const State& state,
                                                              CVolumeViewer* viewer,
                                                              ViewerOverlayControllerBase::OverlayBuilder& builder) const
{
    QElapsedTimer buildTimer;
    buildTimer.start();

    if (!state.surface) {
        return;
    }

    // Check if we need to rebuild the COMPOSITE IMAGE (only when masks change, not when view changes)
    auto it = _viewerCaches.find(viewer);
    bool needsRebuild = (it == _viewerCaches.end()) ||
                       (it->second.surface != state.surface) ||
                       (it->second.savedImageVersion != _savedImageVersion) ||
                       (it->second.pendingImageVersion != _pendingImageVersion);

    if (needsRebuild) {
        qDebug() << "[ApprovalOverlay] Cache miss, rebuilding composite image...";
        rebuildViewerCache(viewer, state.surface);
        it = _viewerCaches.find(viewer);
    } else {
        qDebug() << "[ApprovalOverlay] Cache HIT, using cached composite image";
    }

    // If cache still doesn't exist or is empty, nothing to render
    if (it == _viewerCaches.end() || it->second.compositeImage.isNull()) {
        qDebug() << "[ApprovalOverlay] No cache available, returning";
        return;
    }

    const ViewerImageCache& cache = it->second;

    // Find valid reference points and calculate the current grid-to-scene scale dynamically
    const cv::Mat_<cv::Vec3f>* points = state.surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    const int gridRows = points->rows;
    const int gridCols = points->cols;

    // Find two adjacent valid points to calculate the current scale
    QPointF refScenePos;
    int refRow = -1;
    int refCol = -1;
    bool foundRef = false;

    // Start from center and spiral outward
    for (int offset = 0; offset < std::max(gridRows, gridCols) / 2 && !foundRef; offset += 10) {
        int row = gridRows / 2 + offset;
        int col = gridCols / 2 + offset;

        if (row >= 0 && row < gridRows && col >= 0 && col < gridCols) {
            const cv::Vec3f& worldPos = (*points)(row, col);
            if (std::isfinite(worldPos[0]) && std::isfinite(worldPos[1]) && std::isfinite(worldPos[2]) &&
                !(worldPos[0] == -1.0f && worldPos[1] == -1.0f && worldPos[2] == -1.0f)) {
                refScenePos = viewer->volumePointToScene(worldPos);
                refRow = row;
                refCol = col;
                foundRef = true;
            }
        }
    }

    if (!foundRef) {
        qDebug() << "[ApprovalOverlay] No valid reference point found";
        return;
    }

    // Find an adjacent valid point to calculate scale
    qreal gridToSceneScale = 1.0;
    bool foundAdjacent = false;

    // Try right neighbor first
    if (refCol + 1 < gridCols) {
        const cv::Vec3f& adjWorldPos = (*points)(refRow, refCol + 1);
        if (std::isfinite(adjWorldPos[0]) && std::isfinite(adjWorldPos[1]) && std::isfinite(adjWorldPos[2]) &&
            !(adjWorldPos[0] == -1.0f && adjWorldPos[1] == -1.0f && adjWorldPos[2] == -1.0f)) {
            QPointF adjScenePos = viewer->volumePointToScene(adjWorldPos);
            gridToSceneScale = std::hypot(adjScenePos.x() - refScenePos.x(), adjScenePos.y() - refScenePos.y());
            foundAdjacent = true;
        }
    }

    // If no right neighbor, try down neighbor
    if (!foundAdjacent && refRow + 1 < gridRows) {
        const cv::Vec3f& adjWorldPos = (*points)(refRow + 1, refCol);
        if (std::isfinite(adjWorldPos[0]) && std::isfinite(adjWorldPos[1]) && std::isfinite(adjWorldPos[2]) &&
            !(adjWorldPos[0] == -1.0f && adjWorldPos[1] == -1.0f && adjWorldPos[2] == -1.0f)) {
            QPointF adjScenePos = viewer->volumePointToScene(adjWorldPos);
            gridToSceneScale = std::hypot(adjScenePos.x() - refScenePos.x(), adjScenePos.y() - refScenePos.y());
            foundAdjacent = true;
        }
    }

    if (!foundAdjacent) {
        qDebug() << "[ApprovalOverlay] Could not find adjacent point to calculate scale";
        return;
    }

    // Calculate the scene position of grid origin (0,0) from the reference point
    // In grid-space: each pixel is 1x1
    // In scene-space: each pixel is gridToSceneScale x gridToSceneScale
    QPointF gridOriginScene = refScenePos;
    gridOriginScene.rx() -= refCol * gridToSceneScale;
    gridOriginScene.ry() -= refRow * gridToSceneScale;

    // Count non-transparent pixels for debugging
    int nonTransparentCount = 0;
    for (int y = 0; y < cache.compositeImage.height() && y < 10; ++y) {
        for (int x = 0; x < cache.compositeImage.width() && x < 10; ++x) {
            if (qAlpha(cache.compositeImage.pixel(x, y)) > 0) {
                nonTransparentCount++;
            }
        }
    }

    qDebug() << "[ApprovalOverlay] Rendering: gridOrigin=" << gridOriginScene << "gridToSceneScale=" << gridToSceneScale << "imageSize=" << cache.compositeImage.size();
    qDebug() << "[ApprovalOverlay]   refPoint: grid(" << refCol << "," << refRow << ") -> scene" << refScenePos;
    qDebug() << "[ApprovalOverlay]   Image format:" << cache.compositeImage.format() << "nonTransparent(sampled):" << nonTransparentCount;
    qDebug() << "[ApprovalOverlay]   Adding image with opacity=1.0 z=" << kApprovalMaskZ;

    // Render the cached composite image with the correct grid-to-scene scale
    builder.addImage(cache.compositeImage, gridOriginScene, gridToSceneScale, 1.0, kApprovalMaskZ);

    qDebug() << "[ApprovalOverlay] buildApprovalMaskOverlay complete in" << buildTimer.elapsed() << "ms";
}
