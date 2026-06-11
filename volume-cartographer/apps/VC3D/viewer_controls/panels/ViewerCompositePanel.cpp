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

// mc reduction modes, in dropdown order (see cmbCompositeMode in VCMain.ui).
std::string compositeMethodForModeIndex(int index)
{
    switch (index) {
        case 0: return "max";
        case 1: return "mean";
        case 2: return "min";
        case 3: return "alpha";
        case 4: return "stddev";
        case 5: return "shaded";
        case 6: return "percentile";
        case 7: return "depth";
        default: return "mean";
    }
}

int compositeModeIndexForMethod(const std::string& method)
{
    if (method == "max") return 0;
    if (method == "mean") return 1;
    if (method == "min") return 2;
    if (method == "alpha") return 3;
    if (method == "stddev") return 4;
    if (method == "shaded") return 5;
    if (method == "percentile") return 6;
    if (method == "depth") return 7;
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

void setWidgetEnabled(QWidget* widget, bool enabled)
{
    if (widget) {
        widget->setEnabled(enabled);
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
    // mc reduction modes: max/mean/min/alpha/stddev/shaded/percentile/depth.
    // Only ALPHA (index 3) exposes the alpha threshold + material/opacity knobs;
    // every other knob in this panel was for the deleted C++ composite passes and
    // stays hidden.
    const int methodIndex = _uiRefs.compositeMode ? _uiRefs.compositeMode->currentIndex() : 0;
    const bool isAlpha = methodIndex == 3;

    setWidgetVisible(_uiRefs.alphaMinLabel, isAlpha);
    setWidgetVisible(_uiRefs.alphaMin, isAlpha);
    setWidgetVisible(_uiRefs.alphaMaxLabel, false);
    setWidgetVisible(_uiRefs.alphaMax, false);
    setWidgetVisible(_uiRefs.alphaThresholdLabel, false);
    setWidgetVisible(_uiRefs.alphaThreshold, false);
    setWidgetVisible(_uiRefs.materialLabel, isAlpha);
    setWidgetVisible(_uiRefs.material, isAlpha);

    setWidgetVisible(_uiRefs.blExtinctionLabel, false);
    setWidgetVisible(_uiRefs.blExtinction, false);
    setWidgetVisible(_uiRefs.blEmissionLabel, false);
    setWidgetVisible(_uiRefs.blEmission, false);
    setWidgetVisible(_uiRefs.blAmbientLabel, false);
    setWidgetVisible(_uiRefs.blAmbient, false);
    setWidgetVisible(_uiRefs.shadowStepsLabel, false);
    setWidgetVisible(_uiRefs.shadowSteps, false);
    setWidgetVisible(_uiRefs.dvrAmbientLabel, false);
    setWidgetVisible(_uiRefs.dvrAmbient, false);
    setWidgetVisible(_uiRefs.pbrRoughnessLabel, false);
    setWidgetVisible(_uiRefs.pbrRoughness, false);
    setWidgetVisible(_uiRefs.pbrMetallicLabel, false);
    setWidgetVisible(_uiRefs.pbrMetallic, false);

    setWidgetVisible(_uiRefs.lightingEnabled, false);
    setWidgetVisible(_uiRefs.lightAzimuthLabel, false);
    setWidgetVisible(_uiRefs.lightAzimuth, false);
    setWidgetVisible(_uiRefs.lightElevationLabel, false);
    setWidgetVisible(_uiRefs.lightElevation, false);
    setWidgetVisible(_uiRefs.lightDiffuseLabel, false);
    setWidgetVisible(_uiRefs.lightDiffuse, false);
    setWidgetVisible(_uiRefs.lightAmbientLabel, false);
    setWidgetVisible(_uiRefs.lightAmbient, false);
    setWidgetVisible(_uiRefs.useVolumeGradients, false);

    setWidgetVisible(_uiRefs.preTfX1, false);
    setWidgetVisible(_uiRefs.preTfY1, false);
    setWidgetVisible(_uiRefs.preTfX2, false);
    setWidgetVisible(_uiRefs.preTfY2, false);
    setWidgetVisible(_uiRefs.preTfKnot2Label, false);
    setWidgetVisible(_uiRefs.postTfX1, false);
    setWidgetVisible(_uiRefs.postTfY1, false);
    setWidgetVisible(_uiRefs.postTfX2, false);
    setWidgetVisible(_uiRefs.postTfY2, false);
    setWidgetVisible(_uiRefs.postTfKnot2Label, false);

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
