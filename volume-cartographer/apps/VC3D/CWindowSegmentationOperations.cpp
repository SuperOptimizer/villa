/**
 * @file CWindowSegmentationOperations.cpp
 * @brief Segmentation operation handlers extracted from CWindow
 *
 * This file contains methods for handling segmentation operations,
 * including editing mode, growth, and segment management.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include "CommandLineToolRunner.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationGrower.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "SurfacePanelController.hpp"

#include <QMessageBox>
#include <QSignalBlocker>
#include <QStatusBar>

#include <filesystem>

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

void CWindow::onSegmentationEditingModeChanged(bool enabled)
{
    if (!_segmentationModule) {
        return;
    }

    const bool already = _segmentationModule->editingEnabled();
    if (already != enabled) {
        // Update widget to reflect actual module state to avoid drift.
        if (_segmentationWidget && _segmentationWidget->isEditingEnabled() != already) {
            _segmentationWidget->setEditingEnabled(already);
        }
        enabled = already;
    }

    std::optional<std::string> recentlyEditedId;
    if (!enabled) {
        if (auto* activeSurface = _segmentationModule->activeBaseSurface()) {
            recentlyEditedId = activeSurface->id;
        }
    }

    // Set flag BEFORE beginEditingSession so the surface change doesn't reset view
    if (_viewerManager) {
        _viewerManager->forEachViewer([this, enabled](CVolumeViewer* viewer) {
            if (!viewer) {
                return;
            }
            if (viewer->surfName() == "segmentation") {
                bool defaultReset = _viewerManager->resetDefaultFor(viewer);
                if (enabled) {
                    viewer->setResetViewOnSurfaceChange(false);
                } else {
                    viewer->setResetViewOnSurfaceChange(defaultReset);
                }
            }
        });
    }

    if (enabled) {
        auto activeSurfaceShared = std::dynamic_pointer_cast<QuadSurface>(_surf_col->surface("segmentation"));

        if (!_segmentationModule->beginEditingSession(activeSurfaceShared)) {
            statusBar()->showMessage(tr("Unable to start segmentation editing"), 3000);
            if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
                QSignalBlocker blocker(_segmentationWidget);
                _segmentationWidget->setEditingEnabled(false);
            }
            _segmentationModule->setEditingEnabled(false);
            return;
        }

        if (_viewerManager) {
            _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->clearOverlayGroup("segmentation_radius_indicator");
                }
            });
        }
    } else {
        _segmentationModule->endEditingSession();

#ifdef __linux__
        if (recentlyEditedId && !recentlyEditedId->empty()) {
            markSegmentRecentlyEdited(*recentlyEditedId);
        }
#else
        (void)recentlyEditedId;
#endif
    }

    const QString message = enabled
        ? tr("Segmentation editing enabled")
        : tr("Segmentation editing disabled");
    statusBar()->showMessage(message, 2000);
}

void CWindow::onSegmentationStopToolsRequested()
{
    if (!initializeCommandLineRunner()) {
        return;
    }
    if (_cmdRunner) {
        _cmdRunner->cancel();
        statusBar()->showMessage(tr("Cancelling running tools..."), 3000);
    }
}

void CWindow::onGrowSegmentationSurface(SegmentationGrowthMethod method,
                                        SegmentationGrowthDirection direction,
                                        int steps,
                                        bool inpaintOnly)
{
    if (!_segmentationGrower) {
        statusBar()->showMessage(tr("Segmentation growth is unavailable."), 4000);
        return;
    }

    SegmentationGrower::Context context{
        _segmentationModule.get(),
        _segmentationWidget,
        _surf_col,
        _viewerManager.get(),
        chunk_cache
    };
    _segmentationGrower->updateContext(context);

    SegmentationGrower::VolumeContext volumeContext{
        fVpkg,
        currentVolume,
        currentVolumeId,
        _segmentationGrowthVolumeId.empty() ? currentVolumeId : _segmentationGrowthVolumeId,
        _normalGridPath,
        _segmentationWidget ? _segmentationWidget->normal3dZarrPath() : QString()
    };

    if (!_segmentationGrower->start(volumeContext, method, direction, steps, inpaintOnly)) {
        return;
    }
}

void CWindow::onMoveSegmentToPaths(const QString& segmentId)
{
    if (!fVpkg) {
        statusBar()->showMessage(tr("No volume package loaded"), 3000);
        return;
    }

    // Verify we're in traces directory
    if (fVpkg->getSegmentationDirectory() != "traces") {
        statusBar()->showMessage(tr("Can only move segments from traces directory"), 3000);
        return;
    }

    // Get the segment
    auto seg = fVpkg->segmentation(segmentId.toStdString());
    if (!seg) {
        statusBar()->showMessage(tr("Segment not found: %1").arg(segmentId), 3000);
        return;
    }

    // Build paths
    std::filesystem::path volpkgPath(fVpkg->getVolpkgDirectory());
    std::filesystem::path currentPath = seg->path();
    std::filesystem::path newPath = volpkgPath / "paths" / currentPath.filename();

    // Check if destination exists
    if (std::filesystem::exists(newPath)) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this,
            tr("Destination Exists"),
            tr("Segment '%1' already exists in paths/.\nDo you want to replace it?").arg(segmentId),
            QMessageBox::Yes | QMessageBox::No,
            QMessageBox::No
        );

        if (reply != QMessageBox::Yes) {
            return;
        }

        // Remove the existing one
        try {
            std::filesystem::remove_all(newPath);
        } catch (const std::exception& e) {
            QMessageBox::critical(this, tr("Error"),
                tr("Failed to remove existing segment: %1").arg(e.what()));
            return;
        }
    }

    // Confirm the move
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        tr("Move to Paths"),
        tr("Move segment '%1' from traces/ to paths/?\n\n"
           "Note: The segment will be closed if currently open.").arg(segmentId),
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::Yes
    );

    if (reply != QMessageBox::Yes) {
        return;
    }

    // === CRITICAL: Clean up the segment before moving ===
    std::string idStd = segmentId.toStdString();

    // Check if this is the currently selected segment
    bool wasSelected = (_surfID == idStd);

    // Clear from surface collection (including "segmentation" if it matches)
    if (_surf_col) {
        auto currentSurface = _surf_col->surface(idStd);
        auto segmentationSurface = _surf_col->surface("segmentation");

        // If this surface is currently shown as "segmentation", clear it
        if (currentSurface && segmentationSurface && currentSurface == segmentationSurface) {
            _surf_col->setSurface("segmentation", nullptr, false, false);
        }

        // Clear the surface from the collection
        _surf_col->setSurface(idStd, nullptr, false, false);
    }

    // Unload the surface from VolumePkg
    fVpkg->unloadSurface(idStd);

    // Clear selection if this was selected
    if (wasSelected) {
        clearSurfaceSelection();

        // Clear tree selection
        if (treeWidgetSurfaces) {
            treeWidgetSurfaces->clearSelection();
        }
    }

    // Perform the move
    try {
        std::filesystem::rename(currentPath, newPath);

        // Remove from VolumePkg's internal tracking for traces
        fVpkg->removeSingleSegmentation(idStd);

        // The inotify system will pick up the IN_MOVED_TO in paths/
        // and handle adding it there if the user switches to that directory

        if (_surfacePanel) {
            _surfacePanel->removeSingleSegmentation(idStd);
        }

        statusBar()->showMessage(
            tr("Moved %1 from traces/ to paths/. Switch to paths directory to see it.").arg(segmentId), 5000);

    } catch (const std::exception& e) {
        // If move failed, we might want to reload the segment
        // but it's probably safer to leave it unloaded
        QMessageBox::critical(this, tr("Error"),
            tr("Failed to move segment: %1\n\n"
               "The segment has been unloaded from the viewer.").arg(e.what()));
    }
}
