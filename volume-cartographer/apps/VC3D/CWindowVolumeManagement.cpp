/**
 * @file CWindowVolumeManagement.cpp
 * @brief Volume package open/close operations extracted from CWindow
 *
 * This file contains methods for opening and closing volume packages,
 * including initialization, cleanup, and related UI updates.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include "CSurfaceCollection.hpp"
#include "MenuActionController.hpp"
#include "SeedingWidget.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationWidget.hpp"

#include <QComboBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QSignalBlocker>
#include <QTimer>
#include <QTreeWidget>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/ui/VCCollection.hpp"

// Open volume package
void CWindow::OpenVolume(const QString& path)
{
    QString aVpkgPath = path;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    if (aVpkgPath.isEmpty()) {
        aVpkgPath = QFileDialog::getExistingDirectory(
            this, tr("Open Directory"), settings.value(vc3d::settings::volpkg::DEFAULT_PATH).toString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks | QFileDialog::ReadOnly | QFileDialog::DontUseNativeDialog);
        // Dialog box cancelled
        if (aVpkgPath.length() == 0) {
            Logger()->info("Open .volpkg canceled");
            return;
        }
    }

    // Checks the folder path for .volpkg extension
    auto const extension = aVpkgPath.toStdString().substr(
        aVpkgPath.toStdString().length() - 7, aVpkgPath.toStdString().length());
    if (extension != ".volpkg") {
        QMessageBox::warning(
            this, tr("ERROR"),
            "The selected file is not of the correct type: \".volpkg\"");
        Logger()->error(
            "Selected file is not .volpkg: {}", aVpkgPath.toStdString());
        fVpkg = nullptr;  // Is needed for User Experience, clears screen.
        updateNormalGridAvailability();
        return;
    }

    // Open volume package
    if (!InitializeVolumePkg(aVpkgPath.toStdString() + "/")) {
        return;
    }

    // Check version number
    if (fVpkg->version() < VOLPKG_MIN_VERSION) {
        const auto msg = "Volume package is version " +
                         std::to_string(fVpkg->version()) +
                         " but this program requires version " +
                         std::to_string(VOLPKG_MIN_VERSION) + "+.";
        Logger()->error(msg);
        QMessageBox::warning(this, tr("ERROR"), QString(msg.c_str()));
        fVpkg = nullptr;
        updateNormalGridAvailability();
        return;
    }

    fVpkgPath = aVpkgPath;
    if (_segmentationWidget) {
        _segmentationWidget->setVolumePackagePath(aVpkgPath);
    }
    setVolume(fVpkg->volume());
    {
        const QSignalBlocker blocker{volSelect};
        volSelect->clear();
    }
    QVector<QPair<QString, QString>> volumeEntries;
    QString bestGrowthVolumeId = QString::fromStdString(currentVolumeId);
    bool preferredVolumeFound = false;
    for (const auto& id : fVpkg->volumeIDs()) {
        auto vol = fVpkg->volume(id);
        const QString idStr = QString::fromStdString(id);
        const QString nameStr = QString::fromStdString(vol->name());
        const QString label = nameStr.isEmpty() ? idStr : QStringLiteral("%1 (%2)").arg(nameStr, idStr);
        volSelect->addItem(label, QVariant(idStr));
        volumeEntries.append({idStr, label});

        const QString loweredName = nameStr.toLower();
        const QString loweredId = idStr.toLower();
        const bool matchesPreferred = loweredName.contains(QStringLiteral("surface")) ||
                                      loweredName.contains(QStringLiteral("surf")) ||
                                      loweredId.contains(QStringLiteral("surface")) ||
                                      loweredId.contains(QStringLiteral("surf"));

        if (!preferredVolumeFound && matchesPreferred) {
            bestGrowthVolumeId = idStr;
            preferredVolumeFound = true;
        }
    }

    if (bestGrowthVolumeId.isEmpty() && !volumeEntries.isEmpty()) {
        bestGrowthVolumeId = volumeEntries.front().first;
    }
    _segmentationGrowthVolumeId = bestGrowthVolumeId.toStdString();

    if (_segmentationWidget) {
        _segmentationWidget->setAvailableVolumes(volumeEntries, bestGrowthVolumeId);
        // Set initial volume zarr path for neural tracing
        if (!bestGrowthVolumeId.isEmpty()) {
            try {
                auto vol = fVpkg->volume(bestGrowthVolumeId.toStdString());
                if (vol) {
                    _segmentationWidget->setVolumeZarrPath(QString::fromStdString(vol->path().string()));
                }
            } catch (...) {
                // Ignore errors - zarr path will be empty
            }
        }
    }

    if (_volumeOverlay) {
        _volumeOverlay->setVolumePkg(fVpkg, aVpkgPath);
    }

    // Populate the segmentation directory dropdown
    {
        const QSignalBlocker blocker{cmbSegmentationDir};
        cmbSegmentationDir->clear();

        auto availableDirs = fVpkg->getAvailableSegmentationDirectories();
        for (const auto& dirName : availableDirs) {
            cmbSegmentationDir->addItem(QString::fromStdString(dirName));
        }

        // Select the current directory (default is "paths")
        int currentIndex = cmbSegmentationDir->findText(QString::fromStdString(fVpkg->getSegmentationDirectory()));
        if (currentIndex >= 0) {
            cmbSegmentationDir->setCurrentIndex(currentIndex);
        }
    }

    if (_surfacePanel) {
        _surfacePanel->setVolumePkg(fVpkg);
        // Reset stride user override so tiered defaults apply to new volume
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        _surfacePanel->loadSurfaces(false);
    }
    if (_menuController) {
        _menuController->updateRecentVolpkgList(aVpkgPath);
    }

    // Set volume package in Seeding widget
   if (_seedingWidget) {
       _seedingWidget->setVolumePkg(fVpkg);
   }

   if (_surfacePanel) {
       _surfacePanel->refreshPointSetFilterOptions();
   }

#ifdef __linux__
    startWatchingWithInotify();
#endif
}

void CWindow::CloseVolume(void)
{
#ifdef __linux__
    stopWatchingWithInotify();
    if (_inotifyProcessTimer) {
        _inotifyProcessTimer->stop();
    }

    // Clear any pending inotify events
    _pendingInotifyEvents.clear();
    _pendingSegmentUpdates.clear();
    _pendingMoves.clear();
#endif

    // Notify viewers to clear their surface pointers before we delete them
    emit sendVolumeClosing();

    // Tear down active segmentation editing before surfaces disappear to avoid
    // dangling pointers inside the edit manager when the underlying surfaces
    // are unloaded (reloading with editing enabled previously triggered a
    // use-after-free crash).
    if (_segmentationModule) {
        if (_segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationModule->hasActiveSession()) {
            _segmentationModule->endEditingSession();
        }
    }

    // Clear surface collection first
    _surf_col->setSurface("segmentation", nullptr, true);

    // Clear all surfaces from the surface collection
    if (fVpkg) {
        for (const auto& id : fVpkg->getLoadedSurfaceIDs()) {
            _surf_col->setSurface(id, nullptr, true);
        }
        // Tell VolumePkg to unload all surfaces
        fVpkg->unloadAllSurfaces();
    }

    // Clear the volume package
    fVpkg = nullptr;
    currentVolume = nullptr;
    _focusHistory.clear();
    _focusHistoryIndex = -1;
    _navigatingFocusHistory = false;
    _segmentationGrowthVolumeId.clear();
    updateNormalGridAvailability();
    if (_segmentationWidget) {
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    if (_surfacePanel) {
        _surfacePanel->clear();
        _surfacePanel->setVolumePkg(nullptr);
        _surfacePanel->resetTagUi();
    }

    // Update UI
    UpdateView();
    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clear();
    }

    // Clear points
    _point_collection->clearAll();

    if (_volumeOverlay) {
        _volumeOverlay->clearVolumePkg();
    }
}

// Handle open request
auto CWindow::can_change_volume_() -> bool
{
    bool canChange = fVpkg != nullptr && fVpkg->numberOfVolumes() > 1;
    return canChange;
}
