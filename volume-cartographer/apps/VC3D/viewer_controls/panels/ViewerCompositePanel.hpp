#pragma once

#include <QWidget>

#include <functional>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFormLayout;
class QLabel;
class QScrollArea;
class QSlider;
class QSpinBox;
class ViewerManager;
class VolumeViewerBase;

class ViewerCompositePanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QScrollArea* scrollArea{nullptr};
        QWidget* contents{nullptr};

        QCheckBox* compositeEnabled{nullptr};
        QComboBox* compositeMode{nullptr};
        QSpinBox* layersInFront{nullptr};
        QSpinBox* layersBehind{nullptr};

        QLabel* alphaMinLabel{nullptr};
        QSpinBox* alphaMin{nullptr};
        QLabel* alphaMaxLabel{nullptr};
        QSpinBox* alphaMax{nullptr};
        QLabel* alphaThresholdLabel{nullptr};
        QSpinBox* alphaThreshold{nullptr};
        QLabel* materialLabel{nullptr};
        QSpinBox* material{nullptr};
        QCheckBox* reverseDirection{nullptr};

        QLabel* methodScaleLabel{nullptr};
        QSlider* methodScale{nullptr};
        QLabel* methodScaleValue{nullptr};
        QLabel* methodParamLabel{nullptr};
        QSlider* methodParam{nullptr};
        QLabel* methodParamValue{nullptr};

        QLabel* blExtinctionLabel{nullptr};
        QDoubleSpinBox* blExtinction{nullptr};
        QLabel* blEmissionLabel{nullptr};
        QDoubleSpinBox* blEmission{nullptr};
        QLabel* blAmbientLabel{nullptr};
        QDoubleSpinBox* blAmbient{nullptr};

        QCheckBox* lightingEnabled{nullptr};
        QLabel* lightAzimuthLabel{nullptr};
        QSpinBox* lightAzimuth{nullptr};
        QLabel* lightElevationLabel{nullptr};
        QSpinBox* lightElevation{nullptr};
        QLabel* lightDiffuseLabel{nullptr};
        QDoubleSpinBox* lightDiffuse{nullptr};
        QLabel* lightAmbientLabel{nullptr};
        QDoubleSpinBox* lightAmbient{nullptr};
        QCheckBox* useVolumeGradients{nullptr};
        QLabel* shadowStepsLabel{nullptr};
        QSpinBox* shadowSteps{nullptr};


        QLabel* dvrAmbientLabel{nullptr};
        QDoubleSpinBox* dvrAmbient{nullptr};
        QLabel* pbrRoughnessLabel{nullptr};
        QDoubleSpinBox* pbrRoughness{nullptr};
        QLabel* pbrMetallicLabel{nullptr};
        QDoubleSpinBox* pbrMetallic{nullptr};

        QCheckBox* planeCompositeXY{nullptr};
        QCheckBox* planeCompositeXZ{nullptr};
        QCheckBox* planeCompositeYZ{nullptr};
        QSpinBox* planeLayersFront{nullptr};
        QSpinBox* planeLayersBehind{nullptr};
    };

    explicit ViewerCompositePanel(const UiRefs& uiRefs,
                                  ViewerManager* viewerManager,
                                  QWidget* parent = nullptr);

    void toggleSegmentationComposite();
    void setSegmentationCompositeChecked(bool checked);

private:
    void setupControls();
    void initializeExistingViewers();
    void applyInitialSettingsToViewer(VolumeViewerBase* viewer);
    void updateCompositeParamsVisibility();
    void applyToSegmentationViewer(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToAllViewers(const std::function<void(VolumeViewerBase*)>& apply);
    void applyToPlaneViewers(const std::function<void(VolumeViewerBase*)>& apply);

    void setupShadingControls();
    void applyLightDirection();

    UiRefs _uiRefs;
    // programmatic shaded/ink knobs (the lighting az/el/diffuse/ambient widgets
    // come from the .ui; these rows cover the rest of mc's shading params)
    QFormLayout* _shadeForm{nullptr};
    QDoubleSpinBox* _specular{nullptr};
    QDoubleSpinBox* _shadow{nullptr};
    QDoubleSpinBox* _sss{nullptr};
    QDoubleSpinBox* _curvature{nullptr};
    QDoubleSpinBox* _transmission{nullptr};
    QDoubleSpinBox* _inkGain{nullptr};
    QSpinBox* _inkScaleVox{nullptr};
    int _shadeRowCount{0};   // rows below this index are ink-only
    ViewerManager* _viewerManager{nullptr};
};
