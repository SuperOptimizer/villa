/**
 * @file CWindowUserInteraction.cpp
 * @brief User interaction handlers extracted from CWindow
 *
 * This file contains methods for handling user interactions such as
 * zoom, point clicks, focus changes, and coordinate copying.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include "CPointCollectionWidget.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "SurfacePanelController.hpp"
#include "ViewerManager.hpp"

#include <QApplication>
#include <QClipboard>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QStatusBar>

#include "vc/core/util/QuadSurface.hpp"

void CWindow::onZoomIn()
{
    // Get the active sub-window
    QMdiSubWindow* activeWindow = mdiArea->activeSubWindow();
    if (!activeWindow) return;

    // Get the viewer from the active window
    CVolumeViewer* viewer = qobject_cast<CVolumeViewer*>(activeWindow->widget());
    if (!viewer) return;

    // Get the center of the current view as the zoom point
    QPointF center = viewer->fGraphicsView->mapToScene(
        viewer->fGraphicsView->viewport()->rect().center());

    // Trigger zoom in (positive steps)
    viewer->onZoom(1, center, Qt::NoModifier);
}

void CWindow::onFocusPOIChanged(std::string name, POI* poi)
{
    if (name == "focus" && poi) {
        lblLocFocus->setText(QString("%1, %2, %3")
            .arg(static_cast<int>(poi->p[0]))
            .arg(static_cast<int>(poi->p[1]))
            .arg(static_cast<int>(poi->p[2])));

        if (_surfacePanel) {
            _surfacePanel->refreshFiltersOnly();
        }

        applySlicePlaneOrientation();
    }
}

void CWindow::onPointDoubleClicked(uint64_t pointId)
{
    auto point_opt = _point_collection->getPoint(pointId);
    if (point_opt) {
        POI *poi = _surf_col->poi("focus");
        if (!poi) {
            poi = new POI;
        }
        poi->p = point_opt->p;

        // Find the closest normal on the segmentation surface
        auto seg_surface = _surf_col->surface("segmentation");
        if (auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get())) {
            auto ptr = quad_surface->pointer();
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            quad_surface->pointTo(ptr, point_opt->p, 4.0, 100, patchIndex);
            poi->n = quad_surface->normal(ptr, quad_surface->loc(ptr));
        } else {
            poi->n = cv::Vec3f(0, 0, 1); // Default normal if no surface
        }

        _surf_col->setPOI("focus", poi);
    }
}

void CWindow::onConvertPointToAnchor(uint64_t pointId, uint64_t collectionId)
{
    auto point_opt = _point_collection->getPoint(pointId);
    if (!point_opt) {
        statusBar()->showMessage(tr("Point not found"), 2000);
        return;
    }

    // Get the segmentation surface to project the point onto
    auto seg_surface = _surf_col->surface("segmentation");
    auto* quad_surface = dynamic_cast<QuadSurface*>(seg_surface.get());
    if (!quad_surface) {
        statusBar()->showMessage(tr("No active segmentation surface for anchor conversion"), 3000);
        return;
    }

    // Find the 2D grid location of this point on the surface
    auto ptr = quad_surface->pointer();
    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    float dist = quad_surface->pointTo(ptr, point_opt->p, 4.0, 1000, patchIndex);

    if (dist > 10.0) {
        statusBar()->showMessage(tr("Point is too far from surface (distance: %1)").arg(dist), 3000);
        return;
    }

    // Get the raw grid location (internal coordinates)
    cv::Vec3f loc_3d = quad_surface->loc_raw(ptr);
    cv::Vec2f anchor2d(loc_3d[0], loc_3d[1]);

    // Set the anchor2d on the collection
    _point_collection->setCollectionAnchor2d(collectionId, anchor2d);

    // Remove the point (it's now represented by the anchor)
    _point_collection->removePoint(pointId);

    statusBar()->showMessage(tr("Converted point to anchor at grid position (%1, %2)").arg(anchor2d[0]).arg(anchor2d[1]), 3000);
}

void CWindow::onZoomOut()
{
    // Get the active sub-window
    QMdiSubWindow* activeWindow = mdiArea->activeSubWindow();
    if (!activeWindow) return;

    // Get the viewer from the active window
    CVolumeViewer* viewer = qobject_cast<CVolumeViewer*>(activeWindow->widget());
    if (!viewer) return;

    // Get the center of the current view as the zoom point
    QPointF center = viewer->fGraphicsView->mapToScene(
        viewer->fGraphicsView->viewport()->rect().center());

    // Trigger zoom out (negative steps)
    viewer->onZoom(-1, center, Qt::NoModifier);
}

void CWindow::onCopyCoordinates()
{
    QString coords = lblLocFocus->text().trimmed();
    if (!coords.isEmpty()) {
        QApplication::clipboard()->setText(coords);
        statusBar()->showMessage(tr("Coordinates copied to clipboard: %1").arg(coords), 2000);
    }
}
