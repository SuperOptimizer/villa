#include "SurfacePanelController.hpp"

#include "SurfaceTreeWidget.hpp"
#include "ViewerManager.hpp"
#include "CSurfaceCollection.hpp"
#include "CVolumeViewer.hpp"
#include "elements/DropdownChecklistButton.hpp"
#include "VCSettings.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/ui/VCCollection.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QAction>
#include <QMenu>
#include <QMessageBox>
#include <QModelIndex>
#include <QPushButton>
#include <QSettings>
#include <QSignalBlocker>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QStyle>
#include <QWidget>
#include <QString>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QVector>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <optional>
#include <unordered_set>
#include <set>
#include <filesystem>

namespace {

void sync_tag(std::optional<SurfaceTagEntry>& field, bool checked, const std::string& username = {})
{
    if (checked && !field) {
        SurfaceTagEntry entry;
        if (!username.empty()) {
            entry.user = username;
        }
        entry.date = get_surface_time_str();
        field = std::move(entry);
    }

    if (!checked && field) {
        field.reset();
    }
}

} // namespace

SurfacePanelController::SurfacePanelController(const UiRefs& ui,
                                               CSurfaceCollection* surfaces,
                                               ViewerManager* viewerManager,
                                               std::function<CVolumeViewer*()> segmentationViewerProvider,
                                               std::function<void()> filtersUpdated,
                                               QObject* parent)
    : QObject(parent)
    , _ui(ui)
    , _surfaces(surfaces)
    , _viewerManager(viewerManager)
    , _segmentationViewerProvider(std::move(segmentationViewerProvider))
    , _filtersUpdated(std::move(filtersUpdated))
{
    if (_ui.reloadButton) {
        connect(_ui.reloadButton, &QPushButton::clicked, this, &SurfacePanelController::loadSurfacesIncremental);
    }

    if (_ui.treeWidget) {
        _ui.treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(_ui.treeWidget, &QTreeWidget::itemSelectionChanged,
                this, &SurfacePanelController::handleTreeSelectionChanged);
        connect(_ui.treeWidget, &QWidget::customContextMenuRequested,
                this, &SurfacePanelController::showContextMenu);
    }
}

void SurfacePanelController::setVolumePkg(const std::shared_ptr<VolumePkg>& pkg)
{
    _volumePkg = pkg;
}

void SurfacePanelController::clear()
{
    if (_ui.treeWidget) {
        const QSignalBlocker blocker{_ui.treeWidget};
        _ui.treeWidget->clear();
    }
}

void SurfacePanelController::loadSurfaces(bool reload)
{
    if (!_volumePkg) {
        return;
    }

    if (reload) {
        // Wait for any pending index rebuild before deleting surfaces
        if (_viewerManager) {
            _viewerManager->waitForPendingIndexRebuild();
        }
        // Clear all surfaces from collection BEFORE unloading to prevent dangling pointers
        if (_surfaces) {
            auto names = _surfaces->surfaceNames();
            for (const auto& name : names) {
                _surfaces->setSurface(name, nullptr, true, false);
            }
        }
        _volumePkg->unloadAllSurfaces();
    }

    auto segIds = _volumePkg->segmentationIDs();
    _volumePkg->loadSurfacesBatch(segIds);

    if (_surfaces) {
        for (const auto& id : segIds) {
            auto surf = _volumePkg->getSurface(id);
            if (surf) {
                _surfaces->setSurface(id, surf, true, false);
            }
        }
    }

    populateSurfaceTree();
    applyFilters();
    logSurfaceLoadSummary();
    if (_filtersUpdated) {
        _filtersUpdated();
    }
    if (_viewerManager) {
        _viewerManager->primeSurfacePatchIndicesAsync();
    }
    emit surfacesLoaded();
}

void SurfacePanelController::loadSurfacesIncremental()
{
    if (!_volumePkg) {
        return;
    }

    std::cout << "Starting incremental surface load..." << "\n";
    _volumePkg->refreshSegmentations();
    auto changes = detectSurfaceChanges();

    // Suppress signals during batch removal to avoid dangling pointer crashes
    if (_ui.treeWidget) {
        const QSignalBlocker blocker{_ui.treeWidget};
        // Perform UI mutations without emitting per-item signals.
        for (const auto& id : changes.toRemove) {
            removeSingleSegmentation(id, true);
        }
        for (const auto& id : changes.toAdd) {
            addSingleSegmentation(id);
        }
    } else {
        for (const auto& id : changes.toRemove) {
            removeSingleSegmentation(id, true);
        }
        for (const auto& id : changes.toAdd) {
            addSingleSegmentation(id);
        }
    }
    // Emit a single signal after batch removal
    if (!changes.toRemove.empty() && _surfaces) {
        _surfaces->emitSurfacesChanged();
    }

    if (!changes.toReload.empty()) {
        // Wait for any pending index rebuild before deleting surfaces for reload
        if (_viewerManager) {
            _viewerManager->waitForPendingIndexRebuild();
        }

        std::vector<std::string> reloadedIds;
        reloadedIds.reserve(changes.toReload.size());

        for (const auto& id : changes.toReload) {
            std::cout << "Queueing for reload: " << id << "\n";
            auto currentSurface = _surfaces ? _surfaces->surface(id) : nullptr;
            auto activeSegSurface = _surfaces ? _surfaces->surface("segmentation") : nullptr;
            const bool wasActiveSeg = (currentSurface != nullptr && activeSegSurface.get() == currentSurface.get());

            if (_surfaces) {
                _surfaces->setSurface(id, nullptr, true, false);
                if (wasActiveSeg) {
                    _surfaces->setSurface("segmentation", nullptr, false, false);
                }
            }

            _volumePkg->unloadSurface(id);
            reloadedIds.push_back(id);
        }

        _volumePkg->loadSurfacesBatch(reloadedIds);

        for (const auto& id : reloadedIds) {
            auto reloadedSurface = _volumePkg->getSurface(id);
            if (!reloadedSurface) {
                continue;
            }

            if (_surfaces) {
                _surfaces->setSurface(id, reloadedSurface, true, false);
                auto activeSegSurface = _surfaces ? _surfaces->surface("segmentation") : nullptr;
                if (activeSegSurface == nullptr) {
                    _surfaces->setSurface("segmentation", reloadedSurface, false, false);
                }
            }

            refreshSurfaceMetrics(id);
            if (_currentSurfaceId == id) {
                syncSelectionUi(id, reloadedSurface.get());
            }
        }
    }

    std::cout << "Incremental delta: add=" << changes.toAdd.size()
              << " remove=" << changes.toRemove.size()
              << " reload=" << changes.toReload.size() << "\n";

    applyFilters();
    logSurfaceLoadSummary();
    if (_filtersUpdated) {
        _filtersUpdated();
    }
    if (_viewerManager) {
        _viewerManager->primeSurfacePatchIndicesAsync();
    }
    emit surfacesLoaded();
    std::cout << "Incremental surface load completed." << "\n";
}

SurfacePanelController::SurfaceChanges SurfacePanelController::detectSurfaceChanges() const
{
    SurfaceChanges changes;
    if (!_volumePkg) {
        return changes;
    }

    // Build the set of segmentation IDs currently present on disk.
    std::unordered_set<std::string> diskIds;
    for (const auto& id : _volumePkg->segmentationIDs()) {
        diskIds.insert(id);
    }

    // Build the set of IDs that the UI currently knows about (tree contents).
    std::unordered_set<std::string> uiIds;
    if (_ui.treeWidget) {
        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            const auto qid = (*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (!qid.isEmpty()) {
                uiIds.insert(qid.toStdString());
            }
            ++it;
        }
    } else {
        // Fallback: if no UI is present, best-effort use currently loaded surfaces
        // (legacy behavior), but note this may be over-inclusive.
        for (const auto& id : _volumePkg->getLoadedSurfaceIDs()) {
            uiIds.insert(id);
        }
    }

    // toAdd: present on disk but not yet in the UI tree
    changes.toAdd.reserve(diskIds.size());
    for (const auto& id : diskIds) {
        if (!uiIds.contains(id)) {
            changes.toAdd.push_back(id);
        }
    }

    // toRemove: present in the UI tree but no longer on disk
    changes.toRemove.reserve(uiIds.size());
    for (const auto& uiId : uiIds) {
        if (!diskIds.contains(uiId)) {
            changes.toRemove.push_back(uiId);
        }
    }

    std::unordered_set<std::string> addedIds(
        changes.toAdd.begin(), changes.toAdd.end());
    if (_volumePkg) {
        for (const auto& uiId : uiIds) {
            if (!diskIds.contains(uiId)) {
                continue;
            }
            if (addedIds.find(uiId) != addedIds.end()) {
                continue;
            }
            // Only check timestamps for surfaces that are actually loaded in memory.
            // If not loaded, we'll get fresh data when we eventually load it.
            if (!_volumePkg->isSurfaceLoaded(uiId)) {
                continue;
            }
            auto surf = _volumePkg->getSurface(uiId);
            if (!surf) {
                continue;
            }
            const auto storedTs = surf->maskTimestamp();
            const auto currentTs = QuadSurface::readMaskTimestamp(surf->path);
            if (storedTs != currentTs) {
                changes.toReload.push_back(uiId);
            }
        }
    }

    std::cout << "detectSurfaceChanges: disk=" << diskIds.size()
              << " ui=" << uiIds.size()
              << " add=" << changes.toAdd.size()
              << " remove=" << changes.toRemove.size()
              << " reload=" << changes.toReload.size() << "\n";
    return changes;
}

void SurfacePanelController::populateSurfaceTree()
{
    if (!_ui.treeWidget || !_volumePkg) {
        return;
    }

    const QSignalBlocker blocker{_ui.treeWidget};
    _ui.treeWidget->clear();

    for (const auto& id : _volumePkg->segmentationIDs()) {
        auto surf = _volumePkg->getSurface(id);
        if (!surf) {
            continue;
        }

        auto* item = new SurfaceTreeWidgetItem(_ui.treeWidget);
        item->setText(SURFACE_ID_COLUMN, QString::fromStdString(id));
        item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(id));
        item->setText(2, QString::number(surf->meta.area_cm2, 'f', 3));
        item->setText(3, QString::number(surf->meta.avg_cost, 'f', 3));
        item->setText(4, QString::number(surf->overlappingIds().size()));
        item->setText(5, QString::fromStdString(surf->meta.date_last_modified));
        updateTreeItemIcon(item);
    }

    _ui.treeWidget->resizeColumnToContents(0);
    _ui.treeWidget->resizeColumnToContents(1);
    _ui.treeWidget->resizeColumnToContents(2);
    _ui.treeWidget->resizeColumnToContents(3);
}

void SurfacePanelController::refreshSurfaceMetrics(const std::string& surfaceId)
{
    if (!_ui.treeWidget) {
        return;
    }

    SurfaceTreeWidgetItem* targetItem = nullptr;
    const QString idQString = QString::fromStdString(surfaceId);
    QTreeWidgetItemIterator iterator(_ui.treeWidget);
    while (*iterator) {
        if ((*iterator)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() == idQString) {
            targetItem = static_cast<SurfaceTreeWidgetItem*>(*iterator);
            break;
        }
        ++iterator;
    }

    auto surf = _volumePkg ? _volumePkg->getSurface(surfaceId) : nullptr;
    double areaCm2 = -1.0;
    double avgCost = -1.0;
    int overlapCount = 0;
    QString timestamp;

    if (surf) {
        areaCm2 = surf->meta.area_cm2;
        avgCost = surf->meta.avg_cost;
        overlapCount = static_cast<int>(surf->overlappingIds().size());
        timestamp = QString::fromStdString(surf->meta.date_last_modified);
    }

    if (targetItem) {
        const QString areaText = areaCm2 >= 0.0 ? QString::number(areaCm2, 'f', 3) : QStringLiteral("-");
        const QString costText = avgCost >= 0.0 ? QString::number(avgCost, 'f', 3) : QStringLiteral("-");
        targetItem->setText(2, areaText);
        targetItem->setText(3, costText);
        targetItem->setText(4, QString::number(overlapCount));
        targetItem->setText(TIMESTAMP_COLUMN, timestamp);
        updateTreeItemIcon(targetItem);
    }
}

void SurfacePanelController::updateTreeItemIcon(SurfaceTreeWidgetItem* item)
{
    if (!item || !_volumePkg) {
        return;
    }

    const auto id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
    auto surf = _volumePkg->getSurface(id);
    if (!surf) {
        return;
    }

    item->updateItemIcon(surf->meta.tags.approved.has_value(), surf->meta.tags.defective.has_value());
}

void SurfacePanelController::addSingleSegmentation(const std::string& segId)
{
    if (!_volumePkg) {
        return;
    }

    std::cout << "Adding segmentation: " << segId << "\n";
    try {
        auto surf = _volumePkg->loadSurface(segId);
        if (!surf) {
            return;
        }
        if (_surfaces) {
            _surfaces->setSurface(segId, surf, true, false);
        }
        if (_ui.treeWidget) {
            auto* item = new SurfaceTreeWidgetItem(_ui.treeWidget);
            item->setText(SURFACE_ID_COLUMN, QString::fromStdString(segId));
            item->setData(SURFACE_ID_COLUMN, Qt::UserRole, QString::fromStdString(segId));
            item->setText(2, QString::number(surf->meta.area_cm2, 'f', 3));
            item->setText(3, QString::number(surf->meta.avg_cost, 'f', 3));
            item->setText(4, QString::number(surf->overlappingIds().size()));
            item->setText(5, QString::fromStdString(surf->meta.date_last_modified));
            updateTreeItemIcon(item);
        }
    } catch (const std::exception& e) {
        std::cout << "Failed to add segmentation " << segId << ": " << e.what() << "\n";
    }
}

void SurfacePanelController::removeSingleSegmentation(const std::string& segId, bool suppressSignals)
{
    std::cout << "Removing segmentation: " << segId << "\n";

    // Wait for any pending index rebuild to finish before deleting surfaces
    // to avoid use-after-free in the background rebuild thread
    if (_viewerManager) {
        _viewerManager->waitForPendingIndexRebuild();
    }

    std::shared_ptr<Surface> removedSurface;
    std::shared_ptr<Surface> activeSegSurface;

    if (_surfaces) {
        removedSurface = _surfaces->surface(segId);
        activeSegSurface = _surfaces->surface("segmentation");
    }

    if (_surfaces) {
        if (removedSurface && activeSegSurface.get() == removedSurface.get()) {
            _surfaces->setSurface("segmentation", nullptr, suppressSignals);
        }
        _surfaces->setSurface(segId, nullptr, suppressSignals);
    }

    if (_volumePkg) {
        _volumePkg->unloadSurface(segId);
    }

    if (_ui.treeWidget) {
        // When suppressing signals, also block tree widget signals to prevent
        // handleTreeSelectionChanged from running during batch deletion.
        // This avoids accessing surfaces that may have been deleted.
        std::optional<QSignalBlocker> blocker;
        if (suppressSignals) {
            blocker.emplace(_ui.treeWidget);
        }

        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == segId) {
                const bool wasSelected = (*it)->isSelected();
                delete *it;
                if (wasSelected && !suppressSignals) {
                    emit surfaceSelectionCleared();
                }
                break;
            }
            ++it;
        }
    }
}

void SurfacePanelController::handleTreeSelectionChanged()
{
    if (!_ui.treeWidget) {
        return;
    }

    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();

    if (_selectionLocked) {
        QStringList currentIds;
        currentIds.reserve(selectedItems.size());
        for (auto* item : selectedItems) {
            if (!item) {
                continue;
            }
            const QString id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (!id.isEmpty()) {
                currentIds.append(id);
            }
        }

        QStringList normalizedCurrent = currentIds;
        QStringList normalizedLocked = _lockedSelectionIds;
        std::sort(normalizedCurrent.begin(), normalizedCurrent.end());
        std::sort(normalizedLocked.begin(), normalizedLocked.end());

        if (normalizedCurrent != normalizedLocked) {
            const QSignalBlocker blocker{_ui.treeWidget};
            _ui.treeWidget->clearSelection();
            for (const QString& id : _lockedSelectionIds) {
                if (id.isEmpty()) {
                    continue;
                }
                QTreeWidgetItemIterator it(_ui.treeWidget);
                while (*it) {
                    if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString() == id) {
                        (*it)->setSelected(true);
                        break;
                    }
                    ++it;
                }
            }
            if (!_selectionLockNotified) {
                _selectionLockNotified = true;
                constexpr int kLockNoticeMs = 3000;
                emit statusMessageRequested(tr("Surface selection is locked while growth runs."), kLockNoticeMs);
            }
        }
        return;
    }

    if (selectedItems.isEmpty()) {
        _currentSurfaceId.clear();
        resetTagUi();
        if (_segmentationViewerProvider) {
            if (auto* viewer = _segmentationViewerProvider()) {
                viewer->setWindowTitle(tr("Surface"));
            }
        }
        emit surfaceSelectionCleared();
        return;
    }

    auto* firstSelected = selectedItems.first();
    const QString idQString = firstSelected->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    const std::string id = idQString.toStdString();

    std::shared_ptr<QuadSurface> surface;
    bool surfaceJustLoaded = false;
    if (_volumePkg) {
        surface = _volumePkg->getSurface(id);
        surfaceJustLoaded = (surface != nullptr);
    }

    if (surface && _surfaces) {
        // Keep the named entry in sync so intersection viewers can retain this mesh
        if (surfaceJustLoaded || !_surfaces->surface(id)) {
            _surfaces->setSurface(id, surface, true, false);
        }
        _surfaces->setSurface("segmentation", surface, false, false);
    }

    syncSelectionUi(id, surface.get());

    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            viewer->setWindowTitle(surface ? tr("Surface %1").arg(idQString)
                                           : tr("Surface"));
        }
    }

    emit surfaceActivated(idQString, surface.get());

    if (surfaceJustLoaded) {
        applyFilters();
    }
}

// Note: showContextMenu and handleDeleteSegments are in SurfacePanelControllerContextMenu.cpp

void SurfacePanelController::configureFilters(const FilterUiRefs& filters, VCCollection* pointCollection)
{
    _filters = filters;
    _pointCollection = pointCollection;

    if (_filters.dropdown) {
        _filters.dropdown->clearOptions();
        _filters.dropdown->setText(tr("Filters"));
        if (auto* menu = _filters.dropdown->menu()) {
            menu->setObjectName(QStringLiteral("menuFilters"));
        }
    }

    _filters.focusPoints = nullptr;
    _filters.unreviewed = nullptr;
    _filters.revisit = nullptr;
    _filters.hideUnapproved = nullptr;
    _filters.noExpansion = nullptr;
    _filters.noDefective = nullptr;
    _filters.partialReview = nullptr;
    _filters.inspectOnly = nullptr;
    _filters.currentOnly = nullptr;

    const auto addFilterOption = [&](QCheckBox*& target, const QString& text, const QString& objectName) {
        if (_filters.dropdown) {
            target = _filters.dropdown->addOption(text, objectName);
            return;
        }

        if (!target) {
            target = new QCheckBox(text);
            if (!objectName.isEmpty()) {
                target->setObjectName(objectName);
            }
        } else {
            target->setText(text);
        }
        target->hide();
    };

    const auto addSeparator = [&]() {
        if (_filters.dropdown) {
            _filters.dropdown->addSeparator();
        }
    };

    addFilterOption(_filters.focusPoints, tr("Focus Point"), QStringLiteral("chkFilterFocusPoints"));
    addSeparator();
    addFilterOption(_filters.unreviewed, tr("Unreviewed"), QStringLiteral("chkFilterUnreviewed"));
    addFilterOption(_filters.revisit, tr("Revisit"), QStringLiteral("chkFilterRevisit"));
    addFilterOption(_filters.hideUnapproved, tr("Hide Unapproved"), QStringLiteral("chkFilterHideUnapproved"));
    addSeparator();
    addFilterOption(_filters.noExpansion, tr("Hide Expansion"), QStringLiteral("chkFilterNoExpansion"));
    addFilterOption(_filters.noDefective, tr("Hide Defective"), QStringLiteral("chkFilterNoDefective"));
    addFilterOption(_filters.partialReview, tr("Hide Partial Review"), QStringLiteral("chkFilterPartialReview"));
    addFilterOption(_filters.inspectOnly, tr("Inspect Only"), QStringLiteral("chkFilterInspectOnly"));
    addSeparator();
    addFilterOption(_filters.currentOnly, tr("Current Segment Only"), QStringLiteral("chkFilterCurrentOnly"));

    connectFilterSignals();
    rebuildPointSetFilterModel();
    applyFilters();
    updateFilterSummary();
}

void SurfacePanelController::configureTags(const TagUiRefs& tags)
{
    _tags = tags;
    connectTagSignals();
    resetTagUi();
}

void SurfacePanelController::refreshPointSetFilterOptions()
{
    rebuildPointSetFilterModel();
    applyFilters();
}

void SurfacePanelController::applyFilters()
{
    if (_configuringFilters) {
        return;
    }
    applyFiltersInternal();
    updateFilterSummary();
}

void SurfacePanelController::syncSelectionUi(const std::string& surfaceId, QuadSurface* surface)
{
    _currentSurfaceId = surfaceId;
    updateTagCheckboxStatesForSurface(surface);
    if (isCurrentOnlyFilterEnabled()) {
        applyFilters();
    }
}

void SurfacePanelController::resetTagUi()
{
    _currentSurfaceId.clear();

    auto resetBox = [](QCheckBox* box) {
        if (!box) {
            return;
        }
        const QSignalBlocker blocker{box};
        box->setCheckState(Qt::Unchecked);
        box->setEnabled(false);
    };

    resetBox(_tags.approved);
    resetBox(_tags.defective);
    resetBox(_tags.reviewed);
    resetBox(_tags.revisit);
    resetBox(_tags.inspect);
}

bool SurfacePanelController::isCurrentOnlyFilterEnabled() const
{
    return _filters.currentOnly && _filters.currentOnly->isChecked();
}

bool SurfacePanelController::toggleTag(Tag tag)
{
    QCheckBox* target = nullptr;
    switch (tag) {
        case Tag::Approved: target = _tags.approved; break;
        case Tag::Defective: target = _tags.defective; break;
        case Tag::Reviewed: target = _tags.reviewed; break;
        case Tag::Revisit: target = _tags.revisit; break;
        case Tag::Inspect: target = _tags.inspect; break;
    }

    if (!target || !target->isEnabled()) {
        return false;
    }

    target->setCheckState(target->checkState() == Qt::Checked ? Qt::Unchecked : Qt::Checked);
    return true;
}

void SurfacePanelController::reloadSurfacesFromDisk()
{
    loadSurfacesIncremental();
}

void SurfacePanelController::refreshFiltersOnly()
{
    applyFilters();
}

void SurfacePanelController::setSelectionLocked(bool locked)
{
    if (_selectionLocked == locked) {
        return;
    }

    _selectionLocked = locked;
    _lockedSelectionIds.clear();
    _selectionLockNotified = false;

    if (_ui.reloadButton) {
        _ui.reloadButton->setDisabled(locked);
    }

    if (!_ui.treeWidget) {
        return;
    }

    _ui.treeWidget->setDisabled(locked);

    if (locked) {
        const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
        _lockedSelectionIds.reserve(selectedItems.size());
        for (auto* item : selectedItems) {
            if (!item) {
                continue;
            }
            const QString id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            if (!id.isEmpty()) {
                _lockedSelectionIds.append(id);
            }
        }
    }
}

void SurfacePanelController::connectFilterSignals()
{
    auto connectToggle = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this]() { applyFilters(); });
    };

    connectToggle(_filters.focusPoints);
    connectToggle(_filters.unreviewed);
    connectToggle(_filters.revisit);
    connectToggle(_filters.noExpansion);
    connectToggle(_filters.noDefective);
    connectToggle(_filters.partialReview);
    connectToggle(_filters.hideUnapproved);
    connectToggle(_filters.inspectOnly);
    connectToggle(_filters.currentOnly);

    if (_filters.pointSetMode) {
        connect(_filters.pointSetMode, &QComboBox::currentIndexChanged, this, [this]() { applyFilters(); });
    }

    if (_filters.pointSetAll) {
        connect(_filters.pointSetAll, &QPushButton::clicked, this, [this]() {
            auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
            if (!model) {
                return;
            }
            QSignalBlocker blocker(model);
            for (int row = 0; row < model->rowCount(); ++row) {
                model->setData(model->index(row, 0), Qt::Checked, Qt::CheckStateRole);
            }
            applyFilters();
        });
    }

    if (_filters.pointSetNone) {
        connect(_filters.pointSetNone, &QPushButton::clicked, this, [this]() {
            auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
            if (!model) {
                return;
            }
            QSignalBlocker blocker(model);
            for (int row = 0; row < model->rowCount(); ++row) {
                model->setData(model->index(row, 0), Qt::Unchecked, Qt::CheckStateRole);
            }
            applyFilters();
        });
    }

    if (_pointCollection) {
        connect(_pointCollection, &VCCollection::collectionsAdded, this, [this](const std::vector<uint64_t>&) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::collectionRemoved, this, [this](uint64_t) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::collectionChanged, this, [this](uint64_t) {
            rebuildPointSetFilterModel();
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointAdded, this, [this](const ColPoint&) {
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointChanged, this, [this](const ColPoint&) {
            applyFilters();
        });
        connect(_pointCollection, &VCCollection::pointRemoved, this, [this](uint64_t) {
            applyFilters();
        });
    }

    if (_filters.surfaceIdFilter) {
        connect(_filters.surfaceIdFilter, &QLineEdit::textChanged, this, [this]() { applyFilters(); });
    }
}

void SurfacePanelController::connectTagSignals()
{
    auto connectBox = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
#if QT_VERSION < QT_VERSION_CHECK(6, 8, 0)
        connect(box, &QCheckBox::stateChanged, this, [this](int) { onTagCheckboxToggled(); });
#else
        connect(box, &QCheckBox::checkStateChanged, this, [this](Qt::CheckState) { onTagCheckboxToggled(); });
#endif
    };

    connectBox(_tags.approved);
    connectBox(_tags.defective);
    connectBox(_tags.reviewed);
    connectBox(_tags.revisit);
    connectBox(_tags.inspect);
}

void SurfacePanelController::rebuildPointSetFilterModel()
{
    if (!_filters.pointSet) {
        return;
    }

    _configuringFilters = true;

    auto* model = new QStandardItemModel(_filters.pointSet);
    if (_pointSetModelConnection) {
        disconnect(_pointSetModelConnection);
        _pointSetModelConnection = QMetaObject::Connection{};
    }
    if (auto* existingModel = _filters.pointSet->model()) {
        existingModel->deleteLater();
    }
    _filters.pointSet->setModel(model);

    if (_pointCollection) {
        for (const auto& pair : _pointCollection->getAllCollections()) {
            auto* item = new QStandardItem(QString::fromStdString(pair.second.name));
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
            item->setData(Qt::Unchecked, Qt::CheckStateRole);
            model->appendRow(item);
        }
    }

    _pointSetModelConnection = connect(model, &QStandardItemModel::dataChanged,
        this,
        [this](const QModelIndex&, const QModelIndex&, const QVector<int>& roles) {
            if (roles.contains(Qt::CheckStateRole)) {
                applyFilters();
            }
        });

    _configuringFilters = false;
    updateFilterSummary();
}

void SurfacePanelController::updateFilterSummary()
{
    if (!_filters.dropdown) {
        return;
    }

    int activeFilters = 0;
    const auto countIfChecked = [&activeFilters](QCheckBox* box) {
        if (box && box->isChecked()) {
            ++activeFilters;
        }
    };

    countIfChecked(_filters.focusPoints);
    countIfChecked(_filters.unreviewed);
    countIfChecked(_filters.revisit);
    countIfChecked(_filters.hideUnapproved);
    countIfChecked(_filters.noExpansion);
    countIfChecked(_filters.noDefective);
    countIfChecked(_filters.partialReview);
    countIfChecked(_filters.inspectOnly);
    countIfChecked(_filters.currentOnly);

    QString label = tr("Filters");
    if (activeFilters > 0) {
        label += tr(" (%1)").arg(activeFilters);
    }
    _filters.dropdown->setText(label);
}

void SurfacePanelController::onTagCheckboxToggled()
{
    if (!_ui.treeWidget) {
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const std::string username = settings.value(vc3d::settings::viewer::USERNAME, vc3d::settings::viewer::USERNAME_DEFAULT).toString().toStdString();

    const auto selectedItems = _ui.treeWidget->selectedItems();
    for (auto* item : selectedItems) {
        if (!item) {
            continue;
        }

        const std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();
        auto surface = _volumePkg ? _volumePkg->getSurface(id) : nullptr;

        if (!surface) {
            continue;
        }

        const bool wasApproved = surface->meta.tags.approved.has_value();
        const bool wasReviewed = surface->meta.tags.reviewed.has_value();
        const bool isNowReviewed = _tags.reviewed && _tags.reviewed->checkState() == Qt::Checked;
        const bool reviewedJustAdded = !wasReviewed && isNowReviewed;

        sync_tag(surface->meta.tags.approved, _tags.approved && _tags.approved->checkState() == Qt::Checked, username);
        sync_tag(surface->meta.tags.defective, _tags.defective && _tags.defective->checkState() == Qt::Checked, username);
        sync_tag(surface->meta.tags.reviewed, _tags.reviewed && _tags.reviewed->checkState() == Qt::Checked, username);
        sync_tag(surface->meta.tags.revisit, _tags.revisit && _tags.revisit->checkState() == Qt::Checked, username);
        sync_tag(surface->meta.tags.inspect, _tags.inspect && _tags.inspect->checkState() == Qt::Checked, username);

        if (wasApproved != surface->meta.tags.approved.has_value()) {
            surface->meta.date_last_modified = get_surface_time_str();
        }

        surface->save_meta();

        if (reviewedJustAdded && _volumePkg) {
            auto surf = _volumePkg->getSurface(id);
            if (surf) {
                for (const auto& overlapId : surf->overlappingIds()) {
                    auto overlapSurf = _volumePkg->getSurface(overlapId);
                    if (!overlapSurf) {
                        continue;
                    }

                    if (overlapSurf->meta.tags.reviewed.has_value()) {
                        continue;
                    }

                    SurfaceTagEntry prEntry;
                    if (!username.empty()) {
                        prEntry.user = username;
                    }
                    prEntry.source = id;
                    prEntry.date = get_surface_time_str();
                    overlapSurf->meta.tags.partial_review = std::move(prEntry);
                    overlapSurf->save_meta();
                }
            }
        }

        if (auto* treeItem = dynamic_cast<SurfaceTreeWidgetItem*>(item)) {
            updateTreeItemIcon(treeItem);
        }
    }

    applyFilters();
}

void SurfacePanelController::applyFiltersInternal()
{
    if (!_ui.treeWidget || !_volumePkg) {
        emit filtersApplied(0);
        return;
    }

    auto isChecked = [](QCheckBox* box) {
        return box && box->isChecked();
    };

    const QString surfaceIdFilterText = _filters.surfaceIdFilter ? _filters.surfaceIdFilter->text().trimmed() : QString{};
    const bool hasSurfaceIdFilter = !surfaceIdFilterText.isEmpty();

    bool hasActiveFilters = isChecked(_filters.focusPoints) ||
                            isChecked(_filters.unreviewed) ||
                            isChecked(_filters.revisit) ||
                            isChecked(_filters.noExpansion) ||
                            isChecked(_filters.noDefective) ||
                            isChecked(_filters.partialReview) ||
                            isChecked(_filters.currentOnly) ||
                            isChecked(_filters.hideUnapproved) ||
                            isChecked(_filters.inspectOnly) ||
                            hasSurfaceIdFilter;

    auto* model = qobject_cast<QStandardItemModel*>(_filters.pointSet ? _filters.pointSet->model() : nullptr);
    if (!hasActiveFilters && model) {
        for (int row = 0; row < model->rowCount(); ++row) {
            if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                hasActiveFilters = true;
                break;
            }
        }
    }

    auto collectVisibleSurfaces = [&](std::set<std::string>& out) {
        if (!_ui.treeWidget) {
            return;
        }
        QTreeWidgetItemIterator visIt(_ui.treeWidget);
        while (*visIt) {
            auto* item = *visIt;
            const auto idStr = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
            std::string id = idStr.toStdString();
            if (!id.empty() && !item->isHidden()) {
                auto meta = _volumePkg->getSurface(id);
                if (!meta) {
                    meta = _volumePkg->loadSurface(id);
                }
                if (meta) {
                    out.insert(id);
                    if (_surfaces && !_surfaces->surface(id)) {
                        _surfaces->setSurface(id, meta, true, false);
                    }
                }
            }
            ++visIt;
        }
    };

    if (!hasActiveFilters) {
        QTreeWidgetItemIterator it(_ui.treeWidget);
        while (*it) {
            (*it)->setHidden(false);
            ++it;
        }

        std::set<std::string> intersects = {"segmentation"};
        collectVisibleSurfaces(intersects);

        if (_viewerManager) {
            _viewerManager->forEachViewer([&intersects](CVolumeViewer* viewer) {
                if (viewer && viewer->surfName() != "segmentation") {
                    viewer->setIntersects(intersects);
                }
            });
        }

        emit filtersApplied(0);
        return;
    }

    std::set<std::string> intersects = {"segmentation"};
    POI* poi = _surfaces ? _surfaces->poi("focus") : nullptr;
    int filterCounter = 0;
    const bool currentOnly = isChecked(_filters.currentOnly);
    const bool restrictToCurrent = currentOnly && !_currentSurfaceId.empty();

    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        auto* item = *it;
        std::string id = item->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString();

        bool show = true;
        auto surf = _volumePkg->getSurface(id);
        if (!surf) {
            surf = _volumePkg->loadSurface(id);
        }
        if (surf && _surfaces && !_surfaces->surface(id)) {
            _surfaces->setSurface(id, surf, true, false);
        }

        if (restrictToCurrent && !id.empty()) {
            show = show && (id == _currentSurfaceId);
        }

        if (hasSurfaceIdFilter && !id.empty()) {
            show = show && QString::fromStdString(id).contains(surfaceIdFilterText, Qt::CaseInsensitive);
        }

        if (surf) {
            if (isChecked(_filters.focusPoints) && poi) {
                show = show && contains(*surf, poi->p);
            }

            if (model) {
                bool anyChecked = false;
                bool anyMatches = false;
                bool allMatch = true;
                for (int row = 0; row < model->rowCount(); ++row) {
                    if (model->data(model->index(row, 0), Qt::CheckStateRole) == Qt::Checked) {
                        anyChecked = true;
                        const auto collectionName = model->data(model->index(row, 0), Qt::DisplayRole).toString().toStdString();
                        std::vector<cv::Vec3f> points;
                        if (_pointCollection) {
                            auto collection = _pointCollection->getPoints(collectionName);
                            points.reserve(collection.size());
                            for (const auto& p : collection) {
                                points.push_back(p.p);
                            }
                        }
                        if (allMatch && !contains(*surf, points)) {
                            allMatch = false;
                        }
                        if (!anyMatches && contains_any(*surf, points)) {
                            anyMatches = true;
                        }
                    }
                }

                if (anyChecked) {
                    if (_filters.pointSetMode && _filters.pointSetMode->currentIndex() == 0) {
                        show = show && anyMatches;
                    } else {
                        show = show && allMatch;
                    }
                }
            }

            if (isChecked(_filters.unreviewed)) {
                show = show && !surf->meta.tags.reviewed.has_value();
            }

            if (isChecked(_filters.revisit)) {
                show = show && surf->meta.tags.revisit.has_value();
            }

            if (isChecked(_filters.noExpansion)) {
                auto it = surf->meta.extras.find("vc_gsfs_mode");
                if (it != surf->meta.extras.end()) {
                    show = show && (it->second != "\"expansion\"");
                }
            }

            if (isChecked(_filters.noDefective)) {
                show = show && !surf->meta.tags.defective.has_value();
            }

            if (isChecked(_filters.partialReview)) {
                show = show && !surf->meta.tags.partial_review.has_value();
            }

            if (isChecked(_filters.hideUnapproved)) {
                show = show && surf->meta.tags.approved.has_value();
            }

            if (isChecked(_filters.inspectOnly)) {
                show = show && surf->meta.tags.inspect.has_value();
            }
        }

        item->setHidden(!show);

        if (!show) {
            filterCounter++;
        }

        ++it;
    }

    intersects.clear();
    intersects.insert("segmentation");
    bool insertedCurrent = false;
    if (restrictToCurrent && _volumePkg->getSurface(_currentSurfaceId)) {
        intersects.insert(_currentSurfaceId);
        insertedCurrent = true;
    }
    if (!restrictToCurrent || !insertedCurrent) {
        collectVisibleSurfaces(intersects);
    }

    if (_viewerManager) {
        _viewerManager->forEachViewer([&intersects](CVolumeViewer* viewer) {
            if (viewer && viewer->surfName() != "segmentation") {
                viewer->setIntersects(intersects);
            }
        });
    }

    emit filtersApplied(filterCounter);
}

void SurfacePanelController::updateTagCheckboxStatesForSurface(QuadSurface* surface)
{
    auto resetState = [](QCheckBox* box) {
        if (!box) {
            return;
        }
        const QSignalBlocker blocker{box};
        box->setCheckState(Qt::Unchecked);
    };

    resetState(_tags.approved);
    resetState(_tags.defective);
    resetState(_tags.reviewed);
    resetState(_tags.revisit);
    resetState(_tags.inspect);

    if (!surface) {
        setTagCheckboxEnabled(false, false, false, false, false);
        return;
    }

    setTagCheckboxEnabled(true, true, true, true, true);

    auto applyTag = [](QCheckBox* box, const std::optional<SurfaceTagEntry>& field) {
        if (!box) {
            return;
        }
        const QSignalBlocker blocker{box};
        if (field.has_value()) {
            box->setCheckState(Qt::Checked);
        }
    };

    applyTag(_tags.approved, surface->meta.tags.approved);
    applyTag(_tags.defective, surface->meta.tags.defective);
    applyTag(_tags.reviewed, surface->meta.tags.reviewed);
    applyTag(_tags.revisit, surface->meta.tags.revisit);
    applyTag(_tags.inspect, surface->meta.tags.inspect);
}

void SurfacePanelController::setTagCheckboxEnabled(bool enabledApproved,
                                                   bool enabledDefective,
                                                   bool enabledReviewed,
                                                   bool enabledRevisit,
                                                   bool enabledInspect)
{
    if (_tags.approved) {
        _tags.approved->setEnabled(enabledApproved);
    }
    if (_tags.defective) {
        _tags.defective->setEnabled(enabledDefective);
    }
    if (_tags.reviewed) {
        _tags.reviewed->setEnabled(enabledReviewed);
    }
    if (_tags.revisit) {
        _tags.revisit->setEnabled(enabledRevisit);
    }
    if (_tags.inspect) {
        _tags.inspect->setEnabled(enabledInspect);
    }
}

void SurfacePanelController::logSurfaceLoadSummary() const
{
    if (!_volumePkg) {
        std::cout << "[SurfacePanel] No volume package set; skipping surface load summary." << "\n";
        return;
    }

    const auto segIds = _volumePkg->segmentationIDs();
    if (segIds.empty()) {
        std::cout << "[SurfacePanel] No segmentation IDs available." << "\n";
        return;
    }

    size_t loadedCount = 0;
    std::vector<std::string> missing;
    missing.reserve(segIds.size());

    for (const auto& id : segIds) {
        bool hasSurface = false;
        if (_surfaces) {
            if (_surfaces->surface(id)) {
                hasSurface = true;
            }
        } else {
            hasSurface = static_cast<bool>(_volumePkg->getSurface(id));
        }

        if (hasSurface) {
            ++loadedCount;
        } else {
            missing.push_back(id);
        }
    }

    std::cout << "[SurfacePanel] Loaded " << loadedCount << " / " << segIds.size()
              << " surfaces into memory." << "\n";
    if (!missing.empty()) {
        const size_t previewCount = std::min<size_t>(missing.size(), 10);
        std::cout << "[SurfacePanel] Missing (" << missing.size() << ") IDs: ";
        for (size_t i = 0; i < previewCount; ++i) {
            std::cout << missing[i];
            if (i + 1 < previewCount) {
                std::cout << ", ";
            }
        }
        if (missing.size() > previewCount) {
            std::cout << ", ...";
        }
        std::cout << "\n";
    }
}

void SurfacePanelController::applyHighlightSelection(const std::string& id, bool enabled)
{
    if (id.empty()) {
        return;
    }

    if (enabled) {
        _highlightedSurfaceIds.insert(id);
    } else {
        _highlightedSurfaceIds.erase(id);
    }

    if (_viewerManager) {
        std::vector<std::string> ids(_highlightedSurfaceIds.begin(), _highlightedSurfaceIds.end());
        _viewerManager->setHighlightedSurfaceIds(ids);
    }
}

bool SurfacePanelController::cycleToNextVisibleSegment()
{
    return cycleVisibleSegment(1);
}

bool SurfacePanelController::cycleToPreviousVisibleSegment()
{
    return cycleVisibleSegment(-1);
}

bool SurfacePanelController::cycleVisibleSegment(int direction)
{
    if (!_ui.treeWidget) {
        return false;
    }

    // Collect all visible (non-hidden) items in tree order
    std::vector<QTreeWidgetItem*> visibleItems;
    QTreeWidgetItemIterator it(_ui.treeWidget);
    while (*it) {
        if (!(*it)->isHidden()) {
            visibleItems.push_back(*it);
        }
        ++it;
    }

    if (visibleItems.empty()) {
        return false;
    }

    // Find current selection index
    int currentIndex = -1;
    const QList<QTreeWidgetItem*> selectedItems = _ui.treeWidget->selectedItems();
    if (!selectedItems.isEmpty()) {
        QTreeWidgetItem* currentItem = selectedItems.first();
        for (size_t i = 0; i < visibleItems.size(); ++i) {
            if (visibleItems[i] == currentItem) {
                currentIndex = static_cast<int>(i);
                break;
            }
        }
    }

    // Calculate next index with wraparound
    int nextIndex;
    if (currentIndex < 0) {
        nextIndex = (direction > 0) ? 0 : static_cast<int>(visibleItems.size()) - 1;
    } else {
        nextIndex = currentIndex + direction;
        if (nextIndex < 0) {
            nextIndex = static_cast<int>(visibleItems.size()) - 1;
        } else if (nextIndex >= static_cast<int>(visibleItems.size())) {
            nextIndex = 0;
        }
    }

    QTreeWidgetItem* nextItem = visibleItems[nextIndex];

    // Block signals to prevent normal handleTreeSelectionChanged
    const QSignalBlocker blocker{_ui.treeWidget};
    _ui.treeWidget->clearSelection();
    nextItem->setSelected(true);
    _ui.treeWidget->scrollToItem(nextItem);

    // Get surface and update state
    const QString idQString = nextItem->data(SURFACE_ID_COLUMN, Qt::UserRole).toString();
    const std::string id = idQString.toStdString();

    std::shared_ptr<QuadSurface> surface;
    if (_volumePkg) {
        surface = _volumePkg->getSurface(id);
    }

    if (surface && _surfaces) {
        if (!_surfaces->surface(id)) {
            _surfaces->setSurface(id, surface, true, false);
        }
        _surfaces->setSurface("segmentation", surface, false, false);
    }

    _currentSurfaceId = id;
    syncSelectionUi(id, surface.get());

    if (_segmentationViewerProvider) {
        if (auto* viewer = _segmentationViewerProvider()) {
            viewer->setWindowTitle(surface ? tr("Surface %1").arg(idQString) : tr("Surface"));
        }
    }

    emit surfaceActivatedPreserveEditing(idQString, surface.get());
    return true;
}
