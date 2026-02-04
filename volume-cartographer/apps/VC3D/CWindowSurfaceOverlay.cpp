/**
 * @file CWindowSurfaceOverlay.cpp
 * @brief Surface overlay dropdown and selection handling extracted from CWindow
 *
 * This file contains methods for managing the surface overlay multi-select dropdown
 * and associated color assignment logic.
 * Extracted from CWindow.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"

#include <QComboBox>
#include <QListView>
#include <QSignalBlocker>
#include <QStandardItem>
#include <QStandardItemModel>

#include "ViewerManager.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

void CWindow::updateSurfaceOverlayDropdown()
{
    if (!ui.surfaceOverlaySelect) {
        return;
    }

    // Disconnect previous model's signals if any
    if (_surfaceOverlayModel) {
        disconnect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
                   this, &CWindow::onSurfaceOverlaySelectionChanged);
    }

    // Create new model
    _surfaceOverlayModel = new QStandardItemModel(ui.surfaceOverlaySelect);
    ui.surfaceOverlaySelect->setModel(_surfaceOverlayModel);

    // Use a QListView to properly show checkboxes
    auto* listView = new QListView(ui.surfaceOverlaySelect);
    ui.surfaceOverlaySelect->setView(listView);

    // Get current segmentation directory for filtering
    std::string currentDir;
    if (fVpkg) {
        currentDir = fVpkg->getSegmentationDirectory();
    }

    // Add "All" item at the top
    auto* allItem = new QStandardItem(tr("All"));
    allItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    allItem->setData(Qt::Unchecked, Qt::CheckStateRole);
    allItem->setData(QStringLiteral("__all__"), Qt::UserRole);
    _surfaceOverlayModel->appendRow(allItem);

    if (_surf_col) {
        const auto names = _surf_col->surfaceNames();
        for (const auto& name : names) {
            // Only add QuadSurfaces (actual segmentations), skip PlaneSurfaces
            auto surf = _surf_col->surface(name);
            auto* quadSurf = dynamic_cast<QuadSurface*>(surf.get());
            if (!quadSurf) {
                continue;
            }

            // Filter by current segmentation directory
            if (!currentDir.empty() && !surf->path.empty()) {
                std::string surfDir = surf->path.parent_path().filename().string();
                if (surfDir != currentDir) {
                    continue;
                }
            }

            auto* item = new QStandardItem(QString::fromStdString(name));
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
            item->setData(Qt::Unchecked, Qt::CheckStateRole);
            item->setData(QString::fromStdString(name), Qt::UserRole);

            // Assign persistent color if not already assigned
            if (_surfaceOverlayColorAssignments.find(name) == _surfaceOverlayColorAssignments.end()) {
                _surfaceOverlayColorAssignments[name] = _nextSurfaceOverlayColorIndex++;
            }
            size_t colorIdx = _surfaceOverlayColorAssignments[name];

            // Create color swatch icon (16x16 colored square)
            QPixmap swatch(16, 16);
            swatch.fill(getOverlayColor(colorIdx));
            item->setIcon(QIcon(swatch));

            _surfaceOverlayModel->appendRow(item);
        }
    }

    // Connect model's dataChanged signal for checkbox state changes
    connect(_surfaceOverlayModel, &QStandardItemModel::dataChanged,
            this, &CWindow::onSurfaceOverlaySelectionChanged);
}

void CWindow::onSurfaceOverlaySelectionChanged(const QModelIndex& topLeft,
                                                const QModelIndex& /*bottomRight*/,
                                                const QVector<int>& roles)
{
    if (!roles.contains(Qt::CheckStateRole) || !_surfaceOverlayModel || !_viewerManager) {
        return;
    }

    // Check if "All" was toggled (row 0)
    QStandardItem* changedItem = _surfaceOverlayModel->itemFromIndex(topLeft);
    if (changedItem && changedItem->data(Qt::UserRole).toString() == QStringLiteral("__all__")) {
        bool allChecked = changedItem->checkState() == Qt::Checked;

        // Block signals while updating all items
        {
            QSignalBlocker blocker(_surfaceOverlayModel);
            for (int row = 1; row < _surfaceOverlayModel->rowCount(); ++row) {
                QStandardItem* item = _surfaceOverlayModel->item(row);
                if (item) {
                    item->setCheckState(allChecked ? Qt::Checked : Qt::Unchecked);
                }
            }
        }
    }

    // Build map of selected surfaces with colors
    std::map<std::string, cv::Vec3b> selectedSurfaces;
    int checkedCount = 0;
    int totalSurfaces = 0;

    for (int row = 1; row < _surfaceOverlayModel->rowCount(); ++row) {
        QStandardItem* item = _surfaceOverlayModel->item(row);
        if (!item) continue;

        totalSurfaces++;
        if (item->checkState() == Qt::Checked) {
            checkedCount++;
            std::string name = item->data(Qt::UserRole).toString().toStdString();
            size_t colorIdx = _surfaceOverlayColorAssignments[name];
            selectedSurfaces[name] = getOverlayColorBGR(colorIdx);
        }
    }

    // Update "All" checkbox state (partial/full/none) without triggering recursion
    {
        QSignalBlocker blocker(_surfaceOverlayModel);
        QStandardItem* allItem = _surfaceOverlayModel->item(0);
        if (allItem) {
            if (checkedCount == 0) {
                allItem->setCheckState(Qt::Unchecked);
            } else if (checkedCount == totalSurfaces && totalSurfaces > 0) {
                allItem->setCheckState(Qt::Checked);
            } else {
                allItem->setCheckState(Qt::PartiallyChecked);
            }
        }
    }

    // Propagate to all viewers
    _viewerManager->forEachViewer([&selectedSurfaces](CVolumeViewer* viewer) {
        viewer->setSurfaceOverlays(selectedSurfaces);
    });
}

QColor CWindow::getOverlayColor(size_t index) const
{
    static const std::vector<QColor> palette = {
        QColor(80, 180, 255),   // sky blue
        QColor(180, 80, 220),   // violet
        QColor(80, 220, 200),   // aqua/teal
        QColor(220, 80, 180),   // magenta
        QColor(80, 130, 255),   // medium blue
        QColor(160, 80, 255),   // purple
        QColor(80, 255, 220),   // cyan
        QColor(255, 80, 200),   // hot pink
        QColor(120, 220, 80),   // lime green
        QColor(80, 180, 120),   // spring green
        QColor(150, 200, 255),  // light sky blue
        QColor(200, 150, 230),  // light violet
    };
    return palette[index % palette.size()];
}

cv::Vec3b CWindow::getOverlayColorBGR(size_t index) const
{
    QColor c = getOverlayColor(index);
    return cv::Vec3b(c.blue(), c.green(), c.red());
}
