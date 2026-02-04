/**
 * @file SurfacePanelControllerContextMenu.cpp
 * @brief Context menu handling extracted from SurfacePanelController
 *
 * This file contains methods for showing and handling context menu actions.
 * Extracted from SurfacePanelController.cpp to improve parallel compilation.
 */

#include "SurfacePanelController.hpp"

#include "CSurfaceCollection.hpp"
#include "SurfaceTreeWidget.hpp"

#include <QAction>
#include <QMenu>
#include <QMessageBox>
#include <QStyle>
#include <QTreeWidget>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "vc/core/types/VolumePkg.hpp"

void SurfacePanelController::showContextMenu(const QPoint& pos)
{
    if (!_ui.treeWidget) {
        return;
    }

    QTreeWidgetItem* item = _ui.treeWidget->itemAt(pos);
    if (!item) {
        return;
    }

    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
    QStringList selectedSegmentIds;
    selectedSegmentIds.reserve(selectedItems.size());
    for (auto* selectedItem : selectedItems) {
        selectedSegmentIds << selectedItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    }

    const QString segmentId = selectedSegmentIds.isEmpty() ?
        item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() :
        selectedSegmentIds.front();

    QMenu contextMenu(tr("Context Menu"), _ui.treeWidget);

    std::string currentDir = _volumePkg->getSegmentationDirectory();
    if (currentDir == "traces") {
        QAction* moveToPathsAction = contextMenu.addAction(tr("Move to Paths"));
        moveToPathsAction->setIcon(_ui.treeWidget->style()->standardIcon(QStyle::SP_FileDialogDetailedView));
        connect(moveToPathsAction, &QAction::triggered, this, [this, segmentId]() {
            emit moveToPathsRequested(segmentId);
        });
        contextMenu.addSeparator();
    }

    QAction* copyPathAction = contextMenu.addAction(tr("Copy Segment Path"));
    connect(copyPathAction, &QAction::triggered, this, [this, segmentId]() {
        emit copySegmentPathRequested(segmentId);
    });

    contextMenu.addSeparator();

    QMenu* seedMenu = contextMenu.addMenu(tr("Run Seed"));
    QAction* seedWithSeedAction = seedMenu->addAction(tr("Seed from Focus Point"));
    connect(seedWithSeedAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSeedsRequested(segmentId, false, false);
    });
    QAction* seedWithRandomAction = seedMenu->addAction(tr("Random Seed"));
    connect(seedWithRandomAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSeedsRequested(segmentId, false, true);
    });
    QAction* seedWithExpandAction = seedMenu->addAction(tr("Expand Seed"));
    connect(seedWithExpandAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSeedsRequested(segmentId, true, false);
    });

    QAction* growSegmentAction = contextMenu.addAction(tr("Run Trace"));
    connect(growSegmentAction, &QAction::triggered, this, [this, segmentId]() {
        emit growSegmentRequested(segmentId);
    });

    QAction* addOverlapAction = contextMenu.addAction(tr("Add overlap"));
    connect(addOverlapAction, &QAction::triggered, this, [this, segmentId]() {
        emit addOverlapRequested(segmentId);
    });

    if (_volumePkg) {
        QAction* copyOutAction = contextMenu.addAction(tr("Copy Out"));
        connect(copyOutAction, &QAction::triggered, this, [this, segmentId]() {
            emit neighborCopyRequested(segmentId, true);
        });
        QAction* copyInAction = contextMenu.addAction(tr("Copy In"));
        connect(copyInAction, &QAction::triggered, this, [this, segmentId]() {
            emit neighborCopyRequested(segmentId, false);
        });

        // Reload from Backup submenu
        std::filesystem::path backupsDir =
            std::filesystem::path(_volumePkg->getVolpkgDirectory()) / "backups" / segmentId.toStdString();
        if (std::filesystem::exists(backupsDir) && std::filesystem::is_directory(backupsDir)) {
            std::vector<int> availableBackups;
            for (const auto& entry : std::filesystem::directory_iterator(backupsDir)) {
                if (entry.is_directory()) {
                    try {
                        int idx = std::stoi(entry.path().filename().string());
                        if (idx >= 0 && idx <= 9) {
                            availableBackups.push_back(idx);
                        }
                    } catch (...) {
                        // Not a numeric directory, skip
                    }
                }
            }
            if (!availableBackups.empty()) {
                std::sort(availableBackups.begin(), availableBackups.end());
                QMenu* backupMenu = contextMenu.addMenu(tr("Reload from Backup"));
                for (int idx : availableBackups) {
                    std::filesystem::path backupPath = backupsDir / std::to_string(idx);
                    QString label = tr("Backup %1").arg(idx);

                    // Try to get timestamp from meta.json
                    std::filesystem::path metaPath = backupPath / "meta.json";
                    if (std::filesystem::exists(metaPath)) {
                        try {
                            std::ifstream f(metaPath);
                            nlohmann::json meta = nlohmann::json::parse(f);
                            if (meta.contains("backup_timestamp")) {
                                label = tr("Backup %1 - %2").arg(idx).arg(
                                    QString::fromStdString(meta["backup_timestamp"].get<std::string>()));
                            }
                        } catch (...) {
                            // Couldn't read meta, use simple label
                        }
                    }

                    QAction* backupAction = backupMenu->addAction(label);
                    connect(backupAction, &QAction::triggered, this, [this, segmentId, idx]() {
                        emit reloadFromBackupRequested(segmentId, idx);
                    });
                }
            }
        }
    }

    contextMenu.addSeparator();

    QAction* renderAction = contextMenu.addAction(tr("Render segment"));
    connect(renderAction, &QAction::triggered, this, [this, segmentId]() {
        emit renderSegmentRequested(segmentId);
    });

    QAction* convertToObjAction = contextMenu.addAction(tr("Convert to OBJ"));
    connect(convertToObjAction, &QAction::triggered, this, [this, segmentId]() {
        emit convertToObjRequested(segmentId);
    });
    QAction* cropBoundsAction = contextMenu.addAction(tr("Crop bounds to valid region"));
    connect(cropBoundsAction, &QAction::triggered, this, [this, segmentId]() {
        emit cropBoundsRequested(segmentId);
    });

    QMenu* flipMenu = contextMenu.addMenu(tr("Flip Surface"));
    QAction* flipUAction = flipMenu->addAction(tr("Flip over U axis (reverse V)"));
    connect(flipUAction, &QAction::triggered, this, [this, segmentId]() {
        emit flipURequested(segmentId);
    });
    QAction* flipVAction = flipMenu->addAction(tr("Flip over V axis (reverse U)"));
    connect(flipVAction, &QAction::triggered, this, [this, segmentId]() {
        emit flipVRequested(segmentId);
    });

    QAction* refineAlphaCompAction = contextMenu.addAction(tr("Refine (Alpha-comp)"));
    connect(refineAlphaCompAction, &QAction::triggered, this, [this, segmentId]() {
        emit alphaCompRefineRequested(segmentId);
    });

    QAction* slimFlattenAction = contextMenu.addAction(tr("SLIM-flatten"));
    connect(slimFlattenAction, &QAction::triggered, this, [this, segmentId]() {
        emit slimFlattenRequested(segmentId);
    });

    QAction* abfFlattenAction = contextMenu.addAction(tr("ABF++ flatten"));
    connect(abfFlattenAction, &QAction::triggered, this, [this, segmentId]() {
        emit abfFlattenRequested(segmentId);
    });

    QAction* awsUploadAction = contextMenu.addAction(tr("Upload artifacts to AWS"));
    connect(awsUploadAction, &QAction::triggered, this, [this, segmentId]() {
        emit awsUploadRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* exportChunksAction = contextMenu.addAction(tr("Export width-chunks (40k px)"));
    connect(exportChunksAction, &QAction::triggered, this, [this, segmentId]() {
        emit exportTifxyzChunksRequested(segmentId);
    });

    contextMenu.addSeparator();

    QAction* inpaintTeleaAction = contextMenu.addAction(tr("Inpaint (Telea) && Rebuild Segment"));
    connect(inpaintTeleaAction, &QAction::triggered, this, [this]() {
        emit teleaInpaintRequested();
    });

    QStringList recalcTargets = selectedSegmentIds;
    if (recalcTargets.isEmpty()) {
        recalcTargets << segmentId;
    }

    contextMenu.addSeparator();

    QAction* recalcAreaAction = contextMenu.addAction(tr("Recalculate Area from Mask"));
    connect(recalcAreaAction, &QAction::triggered, this, [this, recalcTargets]() {
        emit recalcAreaRequested(recalcTargets);
    });

    QStringList deletionTargets = selectedSegmentIds;
    if (deletionTargets.isEmpty()) {
        deletionTargets << segmentId;
    }

    QString deleteText = deletionTargets.size() > 1 ?
        tr("Delete %1 Segments").arg(deletionTargets.size()) :
        tr("Delete Segment");
    QAction* deleteAction = contextMenu.addAction(deleteText);
    deleteAction->setIcon(_ui.treeWidget->style()->standardIcon(QStyle::SP_TrashIcon));
    connect(deleteAction, &QAction::triggered, this, [this, deletionTargets]() {
        handleDeleteSegments(deletionTargets);
    });

    contextMenu.addSeparator();

    const std::string segmentIdStd = segmentId.toStdString();
    QAction* highlightAction = contextMenu.addAction(tr("Highlight in slice views"));
    highlightAction->setCheckable(true);
    highlightAction->setChecked(_highlightedSurfaceIds.count(segmentIdStd) > 0);
    connect(highlightAction, &QAction::toggled, this, [this, segmentIdStd](bool checked) {
        applyHighlightSelection(segmentIdStd, checked);
    });

    contextMenu.exec(_ui.treeWidget->mapToGlobal(pos));
}

void SurfacePanelController::handleDeleteSegments(const QStringList& segmentIds)
{
    if (segmentIds.isEmpty() || !_volumePkg) {
        return;
    }

    QString message;
    if (segmentIds.size() == 1) {
        message = tr("Are you sure you want to delete segment '%1'?\n\nThis action cannot be undone.")
                      .arg(segmentIds.first());
    } else {
        message = tr("Are you sure you want to delete %1 segments?\n\nThis action cannot be undone.")
                      .arg(segmentIds.size());
    }

    QWidget* parentWidget = _ui.treeWidget ? static_cast<QWidget*>(_ui.treeWidget) : nullptr;
    QMessageBox::StandardButton reply = QMessageBox::question(
        parentWidget,
        tr("Confirm Deletion"),
        message,
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No);

    if (reply != QMessageBox::Yes) {
        return;
    }

    int successCount = 0;
    QStringList failedSegments;
    bool anyChanges = false;

    for (const auto& id : segmentIds) {
        const std::string idStd = id.toStdString();
        try {
            // Must clean up CSurfaceCollection before destroying the Surface
            // to avoid dangling pointers in signal handlers.
            // Suppress signals during batch deletion to prevent handlers from
            // iterating over surfaces while we're in the middle of deleting them.
            removeSingleSegmentation(idStd, true);
            _volumePkg->removeSegmentation(idStd);
            ++successCount;
            anyChanges = true;
        } catch (const std::filesystem::filesystem_error& e) {
            if (e.code() == std::errc::permission_denied) {
                failedSegments << id + tr(" (permission denied)");
            } else {
                failedSegments << id + tr(" (filesystem error)");
            }
            std::cerr << "Failed to delete segment " << idStd << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            failedSegments << id;
            std::cerr << "Failed to delete segment " << idStd << ": " << e.what() << std::endl;
        }
    }

    // After all deletions are done, emit a single signal to trigger surface index rebuild
    if (anyChanges && _surfaces) {
        _surfaces->emitSurfacesChanged();
    }

    if (anyChanges) {
        try {
            _volumePkg->refreshSegmentations();
        } catch (const std::exception& e) {
            std::cerr << "Error refreshing segmentations after deletion: " << e.what() << std::endl;
        }
        applyFilters();
        if (_filtersUpdated) {
            _filtersUpdated();
        }
        emit surfacesLoaded();
    }

    if (successCount == segmentIds.size()) {
        emit statusMessageRequested(tr("Successfully deleted %1 segment(s)").arg(successCount), 5000);
    } else if (successCount > 0) {
        QMessageBox::warning(parentWidget,
                             tr("Partial Success"),
                             tr("Deleted %1 segment(s), but failed to delete: %2\n\n"
                                "Note: Permission errors may require manual deletion or running with elevated privileges.")
                                 .arg(successCount)
                                 .arg(failedSegments.join(", ")));
    } else {
        QMessageBox::critical(parentWidget,
                              tr("Deletion Failed"),
                              tr("Failed to delete any segments.\n\n"
                                 "Failed segments: %1\n\n"
                                 "This may be due to insufficient permissions. "
                                 "Try running the application with elevated privileges or manually delete the folders.")
                                  .arg(failedSegments.join(", ")));
    }
}
