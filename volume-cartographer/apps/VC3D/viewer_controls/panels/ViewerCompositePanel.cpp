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
#include <QFormLayout>
#include <QSpinBox>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numbers>
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
        case 8: return "ink";
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
    if (method == "ink") return 8;
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
        if (_uiRefs.compositeMode->count() <= 8)
            _uiRefs.compositeMode->addItem("ink");
        _uiRefs.compositeMode->setCurrentIndex(compositeModeIndexForMethod("max"));
    }

    setupControls();
    setupShadingControls();
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
            if (method == "ink") {
                // ink lives in the top tens of microns: a 32-voxel slab (~77um
                // at 2.4um) beats a deep one that averages in sheet interior /
                // the neighboring wrap. Set via the spinboxes so it's visible.
                if (_uiRefs.layersInFront)
                    _uiRefs.layersInFront->setValue(8);
                if (_uiRefs.layersBehind)
                    _uiRefs.layersBehind->setValue(24);
            }
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

void ViewerCompositePanel::setupShadingControls()
{
    auto* vlayout = qobject_cast<QVBoxLayout*>(layout());
    if (!vlayout)
        return;

    // generic "mutate CompositeParams on every viewer" hook
    auto setParam = [this](std::function<void(CompositeParams&)> fn) {
        applyToAllViewers([&fn](VolumeViewerBase* viewer) {
            auto s = viewer->compositeRenderSettings();
            fn(s.params);
            viewer->setCompositeRenderSettings(s);
        });
    };

    // lighting quartet from the .ui (hidden until shaded/ink is selected):
    // azimuth/elevation define the light direction in volume coords; the
    // "enable" checkbox toggles explicit (raking) light vs headlight.
    if (_uiRefs.lightAzimuth) {
        _uiRefs.lightAzimuth->setRange(-180, 180);
        connect(_uiRefs.lightAzimuth, QOverload<int>::of(&QSpinBox::valueChanged),
                this, [this](int) { applyLightDirection(); });
    }
    if (_uiRefs.lightElevation) {
        _uiRefs.lightElevation->setRange(0, 90);
        _uiRefs.lightElevation->setValue(30);
        connect(_uiRefs.lightElevation, QOverload<int>::of(&QSpinBox::valueChanged),
                this, [this](int) { applyLightDirection(); });
    }
    if (_uiRefs.lightingEnabled) {
        _uiRefs.lightingEnabled->setText(tr("Raking light (off = headlight)"));
        connect(_uiRefs.lightingEnabled, &QCheckBox::toggled,
                this, [this](bool) { applyLightDirection(); });
    }
    if (_uiRefs.lightDiffuse) {
        _uiRefs.lightDiffuse->setRange(0.0, 2.0);
        _uiRefs.lightDiffuse->setSingleStep(0.05);
        _uiRefs.lightDiffuse->setValue(0.75);
        connect(_uiRefs.lightDiffuse, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, [setParam](double v) {
                    setParam([v](CompositeParams& p) { p.diffuse = float(v); });
                });
    }
    if (_uiRefs.lightAmbient) {
        _uiRefs.lightAmbient->setRange(0.0, 1.0);
        _uiRefs.lightAmbient->setSingleStep(0.05);
        _uiRefs.lightAmbient->setValue(0.25);
        connect(_uiRefs.lightAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this, [setParam](double v) {
                    setParam([v](CompositeParams& p) { p.ambient = float(v); });
                });
    }

    // the remaining mc shading params, built programmatically
    _shadeForm = new QFormLayout();
    vlayout->addLayout(_shadeForm);
    auto dspin = [this](double lo, double hi, double step, double val) {
        auto* sp = new QDoubleSpinBox(this);
        sp->setRange(lo, hi);
        sp->setSingleStep(step);
        sp->setValue(val);
        return sp;
    };

    _specular = dspin(0.0, 1.0, 0.05, 0.20);
    _shadeForm->addRow(tr("Specular"), _specular);
    connect(_specular, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [setParam](double v) { setParam([v](CompositeParams& p) { p.specular = float(v); }); });

    _shadow = dspin(0.0, 1.0, 0.1, 0.0);
    _shadeForm->addRow(tr("Shadow"), _shadow);
    connect(_shadow, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [setParam](double v) { setParam([v](CompositeParams& p) { p.shadow = float(v); }); });

    _sss = dspin(0.0, 2.0, 0.1, 0.5);
    _shadeForm->addRow(tr("Subsurface"), _sss);
    connect(_sss, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [setParam](double v) { setParam([v](CompositeParams& p) { p.sss = float(v); }); });

    _curvature = dspin(-2.0, 2.0, 0.1, 0.0);
    _shadeForm->addRow(tr("Curvature"), _curvature);
    connect(_curvature, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [setParam](double v) { setParam([v](CompositeParams& p) { p.curvature = float(v); }); });

    _shadeRowCount = _shadeForm->rowCount();   // rows beyond here are ink-only

    _transmission = dspin(0.0, 1.0, 0.05, 0.35);
    _shadeForm->addRow(tr("Transmission"), _transmission);
    connect(_transmission, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [setParam](double v) { setParam([v](CompositeParams& p) { p.transmission = float(v); }); });

    auto* inkLock = new QSpinBox(this);
    inkLock->setRange(0, 64);
    inkLock->setValue(14);
    inkLock->setToolTip(tr("Sheet lock: composite only this many voxels from the "
                           "found sheet surface (slab = search range; 0 = off)"));
    _shadeForm->addRow(tr("Lock depth (vox)"), inkLock);
    connect(inkLock, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [setParam](int v) { setParam([v](CompositeParams& p) { p.inkLockVox = float(v); }); });

    _inkGain = dspin(0.0, 3.0, 0.1, 1.0);
    _shadeForm->addRow(tr("Ink contrast"), _inkGain);
    connect(_inkGain, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [setParam](double v) { setParam([v](CompositeParams& p) { p.inkGain = float(v); }); });

    _inkScaleVox = new QSpinBox(this);
    _inkScaleVox->setRange(8, 512);
    _inkScaleVox->setSingleStep(8);
    _inkScaleVox->setValue(56);
    _inkScaleVox->setToolTip(tr("Stroke scale in LOD-0 voxels (56 = ~0.13mm at 2.4um)"));
    _shadeForm->addRow(tr("Ink scale (vox)"), _inkScaleVox);
    connect(_inkScaleVox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, [setParam](int v) { setParam([v](CompositeParams& p) { p.inkScaleVox = float(v); }); });

    updateCompositeParamsVisibility();   // hide the new rows until shaded/ink
}

void ViewerCompositePanel::applyLightDirection()
{
    const bool raking = _uiRefs.lightingEnabled && _uiRefs.lightingEnabled->isChecked();
    float lz = 0.f, ly = 0.f, lx = 0.f;            // (0,0,0) = mc headlight
    if (raking) {
        const float az = float(_uiRefs.lightAzimuth ? _uiRefs.lightAzimuth->value() : 0) *
                         std::numbers::pi_v<float> / 180.0f;
        const float el = float(_uiRefs.lightElevation ? _uiRefs.lightElevation->value() : 30) *
                         std::numbers::pi_v<float> / 180.0f;
        lz = std::sin(el);
        ly = std::cos(el) * std::sin(az);
        lx = std::cos(el) * std::cos(az);
    }
    applyToAllViewers([lz, ly, lx](VolumeViewerBase* viewer) {
        auto s = viewer->compositeRenderSettings();
        s.params.lightZ = lz;
        s.params.lightY = ly;
        s.params.lightX = lx;
        viewer->setCompositeRenderSettings(s);
    });
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

    // shaded(5) / ink(8): mc's gradient-lit composite family
    const bool isShadedFamily = methodIndex == 5 || methodIndex == 8;
    const bool isInk = methodIndex == 8;
    setWidgetVisible(_uiRefs.lightingEnabled, isShadedFamily);
    setWidgetVisible(_uiRefs.lightAzimuthLabel, isShadedFamily);
    setWidgetVisible(_uiRefs.lightAzimuth, isShadedFamily);
    setWidgetVisible(_uiRefs.lightElevationLabel, isShadedFamily);
    setWidgetVisible(_uiRefs.lightElevation, isShadedFamily);
    setWidgetVisible(_uiRefs.lightDiffuseLabel, isShadedFamily);
    setWidgetVisible(_uiRefs.lightDiffuse, isShadedFamily);
    setWidgetVisible(_uiRefs.lightAmbientLabel, isShadedFamily);
    setWidgetVisible(_uiRefs.lightAmbient, isShadedFamily);
    setWidgetVisible(_uiRefs.useVolumeGradients, false);
    if (_shadeForm) {
        for (int r = 0; r < _shadeForm->rowCount(); ++r)
            _shadeForm->setRowVisible(r, isShadedFamily && (r < _shadeRowCount || isInk));
    }


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
