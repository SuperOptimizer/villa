#include "viewer_controls/panels/ViewerCompositePanel.hpp"

#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cstdint>
#include <string>

namespace
{

std::string compositeMethodForModeIndex(int index)
{
    switch (index) {
        case 0:  return "max";
        case 1:  return "mean";
        case 2:  return "min";
        case 3:  return "alpha";
        default: return "mean";
    }
}

int compositeModeIndexForMethod(const std::string& method)
{
    if (method == "max") return 0;
    if (method == "mean") return 1;
    if (method == "min") return 2;
    if (method == "alpha") return 3;
    return 1;
}

bool isPlaneViewer(const std::string& name)
{
    return name == "seg xz" || name == "seg yz" || name == "xy plane";
}

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

void setWidgetVisible(QWidget* widget, bool visible)
{
    if (widget) {
        widget->setVisible(visible);
    }
}

} // namespace

ViewerCompositePanel::ViewerCompositePanel(const UiRefs& uiRefs,
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

    if (_uiRefs.compositeMode) {
        QSignalBlocker blocker(_uiRefs.compositeMode);
        _uiRefs.compositeMode->setCurrentIndex(compositeModeIndexForMethod("max"));
    }

    setupControls();
    initializeExistingViewers();

    if (_viewerManager) {
        connect(_viewerManager, &ViewerManager::baseViewerCreated,
                this, &ViewerCompositePanel::applyInitialSettingsToViewer);
    }
}

void ViewerCompositePanel::toggleSegmentationComposite()
{
    applyToSegmentationViewer([this](VolumeViewerBase* viewer) {
        auto s = viewer->compositeRenderSettings();
        s.enabled = !s.enabled;
        viewer->setCompositeRenderSettings(s);
        setSegmentationCompositeChecked(s.enabled);
    });
}

void ViewerCompositePanel::setSegmentationCompositeChecked(bool checked)
{
    if (!_uiRefs.compositeEnabled) {
        return;
    }
    QSignalBlocker blocker(_uiRefs.compositeEnabled);
    _uiRefs.compositeEnabled->setChecked(checked);
}

void ViewerCompositePanel::setupControls()
{
    if (_uiRefs.compositeEnabled) {
        connect(_uiRefs.compositeEnabled, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.enabled = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    if (_uiRefs.compositeMode) {
        connect(_uiRefs.compositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
            const std::string method = compositeMethodForModeIndex(index);
            applyToAllViewers([&method](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.method = method;
                viewer->setCompositeRenderSettings(s);
            });
            updateCompositeParamsVisibility();
        });
    }

    if (_uiRefs.layersInFront) {
        connect(_uiRefs.layersInFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.layersFront = value;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.layersBehind) {
        connect(_uiRefs.layersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.layersBehind = value;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaMin) {
        connect(_uiRefs.alphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaMin = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaMax) {
        connect(_uiRefs.alphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaMax = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.alphaThreshold) {
        connect(_uiRefs.alphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaCutoff = value / 10000.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.material) {
        connect(_uiRefs.material, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            applyToSegmentationViewer([value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.alphaOpacity = value / 255.0f;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.reverseDirection) {
        connect(_uiRefs.reverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.reverseDirection = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    if (_uiRefs.preNormalizeLayers) {
        connect(_uiRefs.preNormalizeLayers, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.preNormalizeLayers = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.preHistEqLayers) {
        connect(_uiRefs.preHistEqLayers, &QCheckBox::toggled, this, [this](bool checked) {
            applyToSegmentationViewer([checked](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.params.preHistEqLayers = checked;
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    auto applyParam = [this](auto&& mutate) {
        applyToSegmentationViewer([&mutate](VolumeViewerBase* viewer) {
            auto s = viewer->compositeRenderSettings();
            mutate(s.params);
            viewer->setCompositeRenderSettings(s);
        });
    };
    if (_uiRefs.preTfEnabled) {
        connect(_uiRefs.preTfEnabled, &QCheckBox::toggled, this, [this, applyParam](bool v) {
            applyParam([v](CompositeParams& p) { p.preTfEnabled = v; });
            updateCompositeParamsVisibility();
        });
    }
    if (_uiRefs.preTfX1) {
        connect(_uiRefs.preTfX1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfX1 = uint8_t(v); }); });
    }
    if (_uiRefs.preTfY1) {
        connect(_uiRefs.preTfY1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfY1 = uint8_t(v); }); });
    }
    if (_uiRefs.preTfX2) {
        connect(_uiRefs.preTfX2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfX2 = uint8_t(v); }); });
    }
    if (_uiRefs.preTfY2) {
        connect(_uiRefs.preTfY2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.preTfY2 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfEnabled) {
        connect(_uiRefs.postTfEnabled, &QCheckBox::toggled, this, [this, applyParam](bool v) {
            applyParam([v](CompositeParams& p) { p.postTfEnabled = v; });
            updateCompositeParamsVisibility();
        });
    }
    if (_uiRefs.postTfX1) {
        connect(_uiRefs.postTfX1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfX1 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfY1) {
        connect(_uiRefs.postTfY1, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfY1 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfX2) {
        connect(_uiRefs.postTfX2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfX2 = uint8_t(v); }); });
    }
    if (_uiRefs.postTfY2) {
        connect(_uiRefs.postTfY2, QOverload<int>::of(&QSpinBox::valueChanged), this,
                [applyParam](int v) { applyParam([v](CompositeParams& p) { p.postTfY2 = uint8_t(v); }); });
    }
    if (_uiRefs.methodScale) {
        connect(_uiRefs.methodScale, &QSlider::valueChanged, this, [this](int value) {
            if (_uiRefs.methodScaleValue) {
                _uiRefs.methodScaleValue->setText(QString::number(value / 10.0f, 'f', 1));
            }
        });
    }
    if (_uiRefs.methodParam) {
        connect(_uiRefs.methodParam, &QSlider::valueChanged, this, [](int) {});
    }

    if (_uiRefs.planeCompositeXY) {
        connect(_uiRefs.planeCompositeXY, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                if (viewer->surfName() == "xy plane") {
                    auto s = viewer->compositeRenderSettings();
                    s.planeEnabled = checked;
                    viewer->setCompositeRenderSettings(s);
                }
            });
        });
    }
    if (_uiRefs.planeCompositeXZ) {
        connect(_uiRefs.planeCompositeXZ, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                if (viewer->surfName() == "seg xz") {
                    auto s = viewer->compositeRenderSettings();
                    s.planeEnabled = checked;
                    viewer->setCompositeRenderSettings(s);
                }
            });
        });
    }
    if (_uiRefs.planeCompositeYZ) {
        connect(_uiRefs.planeCompositeYZ, &QCheckBox::toggled, this, [this](bool checked) {
            applyToAllViewers([checked](VolumeViewerBase* viewer) {
                if (viewer->surfName() == "seg yz") {
                    auto s = viewer->compositeRenderSettings();
                    s.planeEnabled = checked;
                    viewer->setCompositeRenderSettings(s);
                }
            });
        });
    }
    if (_uiRefs.planeLayersFront) {
        connect(_uiRefs.planeLayersFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            const int behind = _uiRefs.planeLayersBehind ? _uiRefs.planeLayersBehind->value() : 0;
            applyToPlaneViewers([value, behind](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.planeLayersFront = std::max(0, value);
                s.planeLayersBehind = std::max(0, behind);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }
    if (_uiRefs.planeLayersBehind) {
        connect(_uiRefs.planeLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
            const int front = _uiRefs.planeLayersFront ? _uiRefs.planeLayersFront->value() : 0;
            applyToPlaneViewers([front, value](VolumeViewerBase* viewer) {
                auto s = viewer->compositeRenderSettings();
                s.planeLayersFront = std::max(0, front);
                s.planeLayersBehind = std::max(0, value);
                viewer->setCompositeRenderSettings(s);
            });
        });
    }

    updateCompositeParamsVisibility();
}

void ViewerCompositePanel::initializeExistingViewers()
{
    if (!_viewerManager) {
        return;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        applyInitialSettingsToViewer(viewer);
    }
}

void ViewerCompositePanel::applyInitialSettingsToViewer(VolumeViewerBase* viewer)
{
    if (!viewer) {
        return;
    }
    auto s = viewer->compositeRenderSettings();
    s.params.method = compositeMethodForModeIndex(_uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0);
    viewer->setCompositeRenderSettings(s);
    if (viewer->surfName() == "segmentation") {
        setSegmentationCompositeChecked(s.enabled);
    }
}

void ViewerCompositePanel::updateCompositeParamsVisibility()
{
    const int methodIndex = _uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0;
    const bool preTfOn = _uiRefs.preTfEnabled && _uiRefs.preTfEnabled->isChecked();
    const bool postTfOn = _uiRefs.postTfEnabled && _uiRefs.postTfEnabled->isChecked();

    const bool isAlpha = methodIndex == 3;

    setWidgetVisible(_uiRefs.alphaMinLabel, isAlpha);
    setWidgetVisible(_uiRefs.alphaMin, isAlpha);
    setWidgetVisible(_uiRefs.alphaMaxLabel, isAlpha);
    setWidgetVisible(_uiRefs.alphaMax, isAlpha);
    setWidgetVisible(_uiRefs.alphaThresholdLabel, isAlpha);
    setWidgetVisible(_uiRefs.alphaThreshold, isAlpha);
    setWidgetVisible(_uiRefs.materialLabel, isAlpha);
    setWidgetVisible(_uiRefs.material, isAlpha);

    setWidgetVisible(_uiRefs.preTfX1, preTfOn);
    setWidgetVisible(_uiRefs.preTfY1, preTfOn);
    setWidgetVisible(_uiRefs.preTfX2, preTfOn);
    setWidgetVisible(_uiRefs.preTfY2, preTfOn);
    setWidgetVisible(_uiRefs.preTfKnot2Label, preTfOn);
    setWidgetVisible(_uiRefs.postTfX1, postTfOn);
    setWidgetVisible(_uiRefs.postTfY1, postTfOn);
    setWidgetVisible(_uiRefs.postTfX2, postTfOn);
    setWidgetVisible(_uiRefs.postTfY2, postTfOn);
    setWidgetVisible(_uiRefs.postTfKnot2Label, postTfOn);

    setWidgetVisible(_uiRefs.methodScaleLabel, false);
    setWidgetVisible(_uiRefs.methodScale, false);
    setWidgetVisible(_uiRefs.methodScaleValue, false);
    setWidgetVisible(_uiRefs.methodParamLabel, false);
    setWidgetVisible(_uiRefs.methodParam, false);
    setWidgetVisible(_uiRefs.methodParamValue, false);
}

void ViewerCompositePanel::applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply)
{
    if (!_viewerManager || !apply) {
        return;
    }
    for (auto* viewer : _viewerManager->baseViewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            apply(viewer);
            return;
        }
    }
}

void ViewerCompositePanel::applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply)
{
    if (!_viewerManager || !apply) {
        return;
    }
    _viewerManager->forEachBaseViewer([&apply](VolumeViewerBase* viewer) {
        if (viewer) {
            apply(viewer);
        }
    });
}

void ViewerCompositePanel::applyToPlaneViewers(const std::function<void(VolumeViewerBase*)>& apply)
{
    applyToAllViewers([&apply](VolumeViewerBase* viewer) {
        if (isPlaneViewer(viewer->surfName())) {
            apply(viewer);
        }
    });
}
