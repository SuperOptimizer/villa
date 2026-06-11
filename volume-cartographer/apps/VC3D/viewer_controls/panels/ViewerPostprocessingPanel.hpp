#pragma once

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QScrollArea;
class QSpinBox;
class ViewerManager;
class VolumeViewerBase;

class ViewerPostprocessingPanel : public QWidget
{
    Q_OBJECT

public:
    struct UiRefs {
        QScrollArea* scrollArea{nullptr};
        QWidget* contents{nullptr};

        QComboBox* baseColormap{nullptr};
    };

    explicit ViewerPostprocessingPanel(const UiRefs& uiRefs,
                                       ViewerManager* viewerManager,
                                       QWidget* parent = nullptr);

private:
    void setupColormapSelector();

    UiRefs _uiRefs;
    ViewerManager* _viewerManager{nullptr};
};
