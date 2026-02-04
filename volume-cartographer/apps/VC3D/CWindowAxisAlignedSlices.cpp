/**
 * @file CWindowAxisAlignedSlices.cpp
 * @brief Axis-aligned slice handling extracted from CWindow
 *
 * This file contains methods for managing axis-aligned slice orientations,
 * mouse interaction for slice rotation, and slice plane orientation updates.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"
#include "CSurfaceCollection.hpp"

#include <cmath>

#include <QLoggingCategory>
#include <QTimer>

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "ViewerManager.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"

Q_DECLARE_LOGGING_CATEGORY(lcAxisSlices)

namespace
{

constexpr float kAxisRotationDegreesPerScenePixel = 0.25f;
constexpr float kEpsilon = 1e-6f;
constexpr float kDegToRad = static_cast<float>(CV_PI / 180.0);
constexpr int kAxisAlignedRotationApplyDelayMs = 25;

cv::Vec3f rotateAroundZ(const cv::Vec3f& v, float radians)
{
    const float c = std::cos(radians);
    const float s = std::sin(radians);
    return {
        v[0] * c - v[1] * s,
        v[0] * s + v[1] * c,
        v[2]
    };
}

cv::Vec3f projectVectorOntoPlane(const cv::Vec3f& v, const cv::Vec3f& normal)
{
    const float dot = v.dot(normal);
    return v - normal * dot;
}

cv::Vec3f normalizeOrZero(const cv::Vec3f& v)
{
    const float magnitude = cv::norm(v);
    if (magnitude <= kEpsilon) {
        return cv::Vec3f(0.0f, 0.0f, 0.0f);
    }
    return v * (1.0f / magnitude);
}

cv::Vec3f crossProduct(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return cv::Vec3f(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]);
}

float signedAngleBetween(const cv::Vec3f& from, const cv::Vec3f& to, const cv::Vec3f& axis)
{
    cv::Vec3f fromNorm = normalizeOrZero(from);
    cv::Vec3f toNorm = normalizeOrZero(to);
    if (cv::norm(fromNorm) <= kEpsilon || cv::norm(toNorm) <= kEpsilon) {
        return 0.0f;
    }

    float dot = fromNorm.dot(toNorm);
    dot = std::clamp(dot, -1.0f, 1.0f);
    cv::Vec3f cross = crossProduct(fromNorm, toNorm);
    float angle = std::atan2(cv::norm(cross), dot);
    float sign = cross.dot(axis) >= 0.0f ? 1.0f : -1.0f;
    return angle * sign;
}

}  // namespace

float CWindow::normalizeDegrees(float degrees)
{
    while (degrees > 180.0f) {
        degrees -= 360.0f;
    }
    while (degrees <= -180.0f) {
        degrees += 360.0f;
    }
    return degrees;
}

float CWindow::currentAxisAlignedRotationDegrees(const std::string& surfaceName) const
{
    if (surfaceName == "seg xz") {
        return _axisAlignedSegXZRotationDeg;
    }
    if (surfaceName == "seg yz") {
        return _axisAlignedSegYZRotationDeg;
    }
    return 0.0f;
}

void CWindow::setAxisAlignedRotationDegrees(const std::string& surfaceName, float degrees)
{
    const float normalized = normalizeDegrees(degrees);
    if (surfaceName == "seg xz") {
        _axisAlignedSegXZRotationDeg = normalized;
    } else if (surfaceName == "seg yz") {
        _axisAlignedSegYZRotationDeg = normalized;
    }
}

void CWindow::scheduleAxisAlignedOrientationUpdate()
{
    if (!_useAxisAlignedSlices) {
        applySlicePlaneOrientation();
        return;
    }
    _axisAlignedOrientationDirty = true;
    if (!_axisAlignedRotationTimer) {
        applySlicePlaneOrientation();
        return;
    }
    if (!_axisAlignedRotationTimer->isActive()) {
        _axisAlignedRotationTimer->start(kAxisAlignedRotationApplyDelayMs);
    }
}

void CWindow::flushAxisAlignedOrientationUpdate()
{
    if (!_axisAlignedOrientationDirty) {
        return;
    }
    cancelAxisAlignedOrientationTimer();
    applySlicePlaneOrientation();
}

void CWindow::processAxisAlignedOrientationUpdate()
{
    if (!_axisAlignedOrientationDirty) {
        return;
    }
    _axisAlignedOrientationDirty = false;
    applySlicePlaneOrientation();
}

void CWindow::cancelAxisAlignedOrientationTimer()
{
    if (_axisAlignedRotationTimer && _axisAlignedRotationTimer->isActive()) {
        _axisAlignedRotationTimer->stop();
    }
    _axisAlignedOrientationDirty = false;
}

void CWindow::updateAxisAlignedSliceInteraction()
{
    if (!_viewerManager) {
        return;
    }

    _viewerManager->forEachViewer([this](CVolumeViewer* viewer) {
        if (!viewer || !viewer->fGraphicsView) {
            return;
        }
        const std::string& name = viewer->surfName();
        if (name == "seg xz" || name == "seg yz") {
            viewer->fGraphicsView->setMiddleButtonPanEnabled(!_useAxisAlignedSlices);
            qCDebug(lcAxisSlices) << "Middle-button pan set" << QString::fromStdString(name)
                                 << "enabled" << viewer->fGraphicsView->middleButtonPanEnabled();
        }
    });
}

void CWindow::onAxisAlignedSliceMousePress(CVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButton button, Qt::KeyboardModifiers)
{
    if (!_useAxisAlignedSlices || button != Qt::MiddleButton || !viewer) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName != "seg xz" && surfaceName != "seg yz") {
        return;
    }

    AxisAlignedSliceDragState& state = _axisAlignedSliceDrags[viewer];
    state.active = true;
    state.startScenePos = viewer->volumePointToScene(volLoc);
    state.startRotationDegrees = currentAxisAlignedRotationDegrees(surfaceName);

}

void CWindow::onAxisAlignedSliceMouseMove(CVolumeViewer* viewer, const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers)
{
    if (!_useAxisAlignedSlices || !viewer || !(buttons & Qt::MiddleButton)) {
        return;
    }

    const std::string surfaceName = viewer->surfName();
    if (surfaceName != "seg xz" && surfaceName != "seg yz") {
        return;
    }

    auto it = _axisAlignedSliceDrags.find(viewer);
    if (it == _axisAlignedSliceDrags.end() || !it->second.active) {
        return;
    }

    AxisAlignedSliceDragState& state = it->second;
    QPointF currentScenePos = viewer->volumePointToScene(volLoc);
    const float dragPixels = static_cast<float>(currentScenePos.y() - state.startScenePos.y());
    const float candidate = normalizeDegrees(state.startRotationDegrees - dragPixels * kAxisRotationDegreesPerScenePixel);
    const float currentRotation = currentAxisAlignedRotationDegrees(surfaceName);

    if (std::abs(candidate - currentRotation) < 0.01f) {
        return;
    }

    setAxisAlignedRotationDegrees(surfaceName, candidate);
    scheduleAxisAlignedOrientationUpdate();

}

void CWindow::onAxisAlignedSliceMouseRelease(CVolumeViewer* viewer, Qt::MouseButton button, Qt::KeyboardModifiers)
{
    if (button != Qt::MiddleButton) {
        return;
    }

    auto it = _axisAlignedSliceDrags.find(viewer);
    if (it != _axisAlignedSliceDrags.end()) {
        it->second.active = false;
    }
    flushAxisAlignedOrientationUpdate();
}

void CWindow::applySlicePlaneOrientation(Surface* sourceOverride)
{
    if (!_surf_col) {
        return;
    }

    cancelAxisAlignedOrientationTimer();

    POI *focus = _surf_col->poi("focus");
    cv::Vec3f origin = focus ? focus->p : cv::Vec3f(0, 0, 0);

    // Helper to configure a plane with optional yaw rotation
    const auto configurePlane = [&](const std::string& planeName,
                                    const cv::Vec3f& baseNormal,
                                    float yawDeg = 0.0f) {
        auto planeShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface(planeName));
        if (!planeShared) {
            planeShared = std::make_shared<PlaneSurface>();
        }

        planeShared->setOrigin(origin);
        planeShared->setInPlaneRotation(0.0f);

        // Apply yaw rotation if set
        cv::Vec3f rotatedNormal;
        if (std::abs(yawDeg) > 0.001f) {
            const float radians = yawDeg * kDegToRad;
            rotatedNormal = rotateAroundZ(baseNormal, radians);
        } else {
            rotatedNormal = baseNormal;
        }

        planeShared->setNormal(rotatedNormal);

        // Adjust in-plane rotation so "up" is aligned with volume Z when possible
        const cv::Vec3f upAxis(0.0f, 0.0f, 1.0f);
        const cv::Vec3f projectedUp = projectVectorOntoPlane(upAxis, rotatedNormal);
        const cv::Vec3f desiredUp = normalizeOrZero(projectedUp);

        if (cv::norm(desiredUp) > kEpsilon) {
            const cv::Vec3f currentUp = planeShared->basisY();
            const float delta = signedAngleBetween(currentUp, desiredUp, rotatedNormal);
            if (std::abs(delta) > kEpsilon) {
                planeShared->setInPlaneRotation(delta);
            }
        } else {
            planeShared->setInPlaneRotation(0.0f);
        }

        _surf_col->setSurface(planeName, planeShared);
        return planeShared;
    };

    // Always update the XY plane
    auto xyPlane = configurePlane("xy plane", cv::Vec3f(0.0f, 0.0f, 1.0f));

    if (_useAxisAlignedSlices) {
        auto segXZShared = configurePlane("seg xz", cv::Vec3f(0.0f, 1.0f, 0.0f), _axisAlignedSegXZRotationDeg);
        auto segYZShared = configurePlane("seg yz", cv::Vec3f(1.0f, 0.0f, 0.0f), _axisAlignedSegYZRotationDeg);

        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    } else {
        QuadSurface* segment = nullptr;
        std::shared_ptr<Surface> segmentHolder;  // Keep surface alive during this scope
        if (sourceOverride) {
            segment = dynamic_cast<QuadSurface*>(sourceOverride);
        } else {
            segmentHolder = _surf_col->surface("segmentation");
            segment = dynamic_cast<QuadSurface*>(segmentHolder.get());
        }
        if (!segment) {
            return;
        }

        auto segXZShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface("seg xz"));
        auto segYZShared = std::dynamic_pointer_cast<PlaneSurface>(_surf_col->surface("seg yz"));

        if (!segXZShared) {
            segXZShared = std::make_shared<PlaneSurface>();
        }
        if (!segYZShared) {
            segYZShared = std::make_shared<PlaneSurface>();
        }

        segXZShared->setOrigin(origin);
        segYZShared->setOrigin(origin);

        auto ptr = segment->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        segment->pointTo(ptr, origin, 1.0f, 1000, patchIndex);

        cv::Vec3f xDir = segment->coord(ptr, {1, 0, 0});
        cv::Vec3f yDir = segment->coord(ptr, {0, 1, 0});
        segXZShared->setNormal(xDir - origin);
        segYZShared->setNormal(yDir - origin);
        segXZShared->setInPlaneRotation(0.0f);
        segYZShared->setInPlaneRotation(0.0f);

        _surf_col->setSurface("seg xz", segXZShared);
        _surf_col->setSurface("seg yz", segYZShared);
        if (_planeSlicingOverlay) {
            _planeSlicingOverlay->refreshAll();
        }
        return;
    }
}
