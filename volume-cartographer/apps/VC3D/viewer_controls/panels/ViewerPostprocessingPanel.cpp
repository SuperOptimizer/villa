#include "viewer_controls/panels/ViewerPostprocessingPanel.hpp"

#include "ViewerManager.hpp"
#include "VolumeViewerCmaps.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLayout>
#include <QScrollArea>
#include <QSpinBox>
#include <QString>
#include <QVBoxLayout>

#include <algorithm>

namespace
{

void reparentItemWidgets(QLayoutItem* item, QWidget* newParent)
{
    if (!item || !newParent) {
        return;
    }
    if (auto* widget = item->widget()) {
        widget->setParent(newParent);
        return;
    }
    if (auto* layout = item->layout()) {
        for (int i = 0; i < layout->count(); ++i) {
            reparentItemWidgets(layout->itemAt(i), newParent);
        }
    }
}

void moveLayoutItems(QLayout* from, QLayout* to, QWidget* newParent)
{
    if (!from || !to) {
        return;
    }
    to->setContentsMargins(from->contentsMargins());
    to->setSpacing(from->spacing());
    while (auto* item = from->takeAt(0)) {
        reparentItemWidgets(item, newParent);
        if (auto* layout = item->layout()) {
            layout->setParent(to);
        }
        to->addItem(item);
    }
}

} // namespace

ViewerPostprocessingPanel::ViewerPostprocessingPanel(const UiRefs& uiRefs,
                                                     ViewerManager* viewerManager,
                                                     QWidget* parent)
    : QWidget(parent)
    , _uiRefs(uiRefs)
    , _viewerManager(viewerManager)
{
    if (_uiRefs.scrollArea && _uiRefs.scrollArea->widget() == _uiRefs.contents) {
        _uiRefs.scrollArea->takeWidget();
    }

    auto* layout = new QVBoxLayout(this);
    moveLayoutItems(_uiRefs.contents ? _uiRefs.contents->layout() : nullptr, layout, this);

    setupColormapSelector();
    setupControls();
}

void ViewerPostprocessingPanel::setupColormapSelector()
{
    if (!_uiRefs.baseColormap) {
        return;
    }

    const auto& entries = volume_viewer_cmaps::entries(volume_viewer_cmaps::EntryScope::SharedOnly);
    _uiRefs.baseColormap->clear();
    _uiRefs.baseColormap->addItem(tr("None (Grayscale)"), QString());
    for (const auto& entry : entries) {
        _uiRefs.baseColormap->addItem(entry.label, QString::fromStdString(entry.id));
    }
    _uiRefs.baseColormap->setCurrentIndex(0);

    connect(_uiRefs.baseColormap, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0 || !_viewerManager || !_uiRefs.baseColormap) {
            return;
        }
        const QString id = _uiRefs.baseColormap->currentData().toString();
        _viewerManager->forEachBaseViewer([&id](VolumeViewerBase* viewer) {
            viewer->setBaseColormap(id.toStdString());
        });
    });
}

void ViewerPostprocessingPanel::setupControls()
{
    if (_uiRefs.stretchValues) {
        connect(_uiRefs.stretchValues, &QCheckBox::toggled, this, [this](bool checked) {
            if (auto* viewer = segmentationBaseViewer()) {
                auto s = viewer->compositeRenderSettings();
                s.postStretchValues = checked;
                viewer->setCompositeRenderSettings(s);
            }
        });
    }
}

VolumeViewerBase* ViewerPostprocessingPanel::segmentationBaseViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return viewer;
        }
    }
    return nullptr;
}
