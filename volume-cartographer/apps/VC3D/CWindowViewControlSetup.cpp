/**
 * @file CWindowViewControlSetup.cpp
 * @brief View and composite control setup extracted from CWindow::CreateWidgets()
 *
 * This file contains the setupViewControls() and setupCompositeControls() methods
 * that configure UI control widgets and their signal/slot connections.
 * Extracted from CreateWidgets() to improve parallel compilation.
 */

#include "CWindow.hpp"

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QMessageBox>
#include <QRegularExpression>
#include <QRegularExpressionValidator>
#include <QSettings>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStatusBar>
#include <QTimer>

#include "CVolumeViewer.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/ui/VCCollection.hpp"
#include "WindowRangeWidget.hpp"

void CWindow::setupViewControls()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    // TODO CHANGE VOLUME LOADING; FIRST CHECK FOR OTHER VOLUMES IN THE STRUCTS
    volSelect = ui.volSelect;

    if (_volumeOverlay) {
        VolumeOverlayController::UiRefs overlayUi{
            .volumeSelect = ui.overlayVolumeSelect,
            .colormapSelect = ui.overlayColormapSelect,
            .opacitySpin = ui.overlayOpacitySpin,
            .thresholdSpin = ui.overlayThresholdSpin,
        };
        _volumeOverlay->setUi(overlayUi);
    }

    // Setup base colormap selector
    {
        const auto& entries = CVolumeViewer::overlayColormapEntries();
        ui.baseColormapSelect->clear();
        ui.baseColormapSelect->addItem(tr("None (Grayscale)"), QString());
        for (const auto& entry : entries) {
            ui.baseColormapSelect->addItem(entry.label, QString::fromStdString(entry.id));
        }
        ui.baseColormapSelect->setCurrentIndex(0);
    }

    connect(ui.baseColormapSelect, qOverload<int>(&QComboBox::currentIndexChanged), [this](int index) {
        if (index < 0 || !_viewerManager) return;
        const QString id = ui.baseColormapSelect->currentData().toString();
        _viewerManager->forEachViewer([&id](CVolumeViewer* viewer) {
            viewer->setBaseColormap(id.toStdString());
        });
    });

    // Setup surface overlay controls
    connect(ui.chkSurfaceOverlay, &QCheckBox::toggled, [this](bool checked) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([checked](CVolumeViewer* viewer) {
            viewer->setSurfaceOverlayEnabled(checked);
        });
        ui.surfaceOverlaySelect->setEnabled(checked);
        ui.spinOverlapThreshold->setEnabled(checked);
    });

    connect(ui.spinOverlapThreshold, qOverload<double>(&QDoubleSpinBox::valueChanged), [this](double value) {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([value](CVolumeViewer* viewer) {
            viewer->setSurfaceOverlapThreshold(static_cast<float>(value));
        });
    });

    // Initially disable surface overlay controls
    ui.surfaceOverlaySelect->setEnabled(false);
    ui.spinOverlapThreshold->setEnabled(false);

    // Initialize surface overlay dropdown (will be populated when surfaces load)
    updateSurfaceOverlayDropdown();

    connect(
        volSelect, &QComboBox::currentIndexChanged, [this](const int& index) {
            std::shared_ptr<Volume> newVolume;
            try {
                newVolume = fVpkg->volume(volSelect->currentData().toString().toStdString());
            } catch (const std::out_of_range& e) {
                QMessageBox::warning(this, "Error", "Could not load volume.");
                return;
            }
            setVolume(newVolume);
        });

    auto* filterDropdown = ui.btnFilterDropdown;
    auto* cmbPointSetFilter = ui.cmbPointSetFilter;
    auto* btnPointSetFilterAll = ui.btnPointSetFilterAll;
    auto* btnPointSetFilterNone = ui.btnPointSetFilterNone;
    auto* cmbPointSetFilterMode = new QComboBox();
    cmbPointSetFilterMode->addItem("Any (OR)");
    cmbPointSetFilterMode->addItem("All (AND)");
    ui.pointSetFilterLayout->insertWidget(1, cmbPointSetFilterMode);

    SurfacePanelController::FilterUiRefs filterUi;
    filterUi.dropdown = filterDropdown;
    filterUi.pointSet = cmbPointSetFilter;
    filterUi.pointSetAll = btnPointSetFilterAll;
    filterUi.pointSetNone = btnPointSetFilterNone;
    filterUi.pointSetMode = cmbPointSetFilterMode;
    filterUi.surfaceIdFilter = ui.lineEditSurfaceFilter;
    _surfacePanel->configureFilters(filterUi, _point_collection);

    SurfacePanelController::TagUiRefs tagUi{
        .approved = ui.chkApproved,
        .defective = ui.chkDefective,
        .reviewed = ui.chkReviewed,
        .revisit = ui.chkRevisit,
        .inspect = ui.chkInspect,
    };
    _surfacePanel->configureTags(tagUi);

    cmbSegmentationDir = ui.cmbSegmentationDir;
    connect(cmbSegmentationDir, &QComboBox::currentIndexChanged, this, &CWindow::onSegmentationDirChanged);

    // Location input element (single QLineEdit for comma-separated values)
    lblLocFocus = ui.sliceFocus;

    // Set up validator for location input (accepts digits, commas, and spaces)
    QRegularExpressionValidator* validator = new QRegularExpressionValidator(
        QRegularExpression("^\\s*\\d+\\s*,\\s*\\d+\\s*,\\s*\\d+\\s*$"), this);
    lblLocFocus->setValidator(validator);
    connect(lblLocFocus, &QLineEdit::editingFinished, this, &CWindow::onManualLocationChanged);

    QPushButton* btnCopyCoords = ui.btnCopyCoords;
    connect(btnCopyCoords, &QPushButton::clicked, this, &CWindow::onCopyCoordinates);

    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        bool showOverlays = settings.value(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS,
                                           vc3d::settings::viewer::SHOW_AXIS_OVERLAYS_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisOverlays);
        chkAxisOverlays->setChecked(showOverlays);
        connect(chkAxisOverlays, &QCheckBox::toggled, this, &CWindow::onAxisOverlayVisibilityToggled);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        int storedOpacity = settings.value(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY,
                                           spinAxisOverlayOpacity->value()).toInt();
        storedOpacity = std::clamp(storedOpacity, spinAxisOverlayOpacity->minimum(), spinAxisOverlayOpacity->maximum());
        QSignalBlocker blocker(spinAxisOverlayOpacity);
        spinAxisOverlayOpacity->setValue(storedOpacity);
        connect(spinAxisOverlayOpacity, qOverload<int>(&QSpinBox::valueChanged), this, &CWindow::onAxisOverlayOpacityChanged);
    }

    if (auto* spinSliceStep = ui.spinSliceStepSize) {
        int savedStep = settings.value(vc3d::settings::viewer::SLICE_STEP_SIZE,
                                       vc3d::settings::viewer::SLICE_STEP_SIZE_DEFAULT).toInt();
        savedStep = std::clamp(savedStep, spinSliceStep->minimum(), spinSliceStep->maximum());
        QSignalBlocker blocker(spinSliceStep);
        spinSliceStep->setValue(savedStep);
        if (_viewerManager) {
            _viewerManager->setSliceStepSize(savedStep);
        }
        connect(spinSliceStep, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
            if (_viewerManager) {
                _viewerManager->setSliceStepSize(value);
            }
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(vc3d::settings::viewer::SLICE_STEP_SIZE, value);
            if (_sliceStepLabel) {
                _sliceStepLabel->setText(tr("Step: %1").arg(value));
            }
        });
    }

    // Surface normals visualization controls
    if (auto* chkShowNormals = ui.chkShowSurfaceNormals) {
        bool showNormals = settings.value(vc3d::settings::viewer::SHOW_SURFACE_NORMALS,
                                          vc3d::settings::viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        QSignalBlocker blocker(chkShowNormals);
        chkShowNormals->setChecked(showNormals);

        // Enable/disable the arrow length and max arrows controls based on checkbox state
        if (auto* lblArrowLength = ui.labelNormalArrowLength) {
            lblArrowLength->setEnabled(showNormals);
        }
        if (auto* sliderArrowLength = ui.sliderNormalArrowLength) {
            sliderArrowLength->setEnabled(showNormals);
        }
        if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
            lblArrowLengthValue->setEnabled(showNormals);
        }
        if (auto* lblMaxArrows = ui.labelNormalMaxArrows) {
            lblMaxArrows->setEnabled(showNormals);
        }
        if (auto* sliderMaxArrows = ui.sliderNormalMaxArrows) {
            sliderMaxArrows->setEnabled(showNormals);
        }
        if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
            lblMaxArrowsValue->setEnabled(showNormals);
        }

        connect(chkShowNormals, &QCheckBox::toggled, this, [this](bool checked) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::SHOW_SURFACE_NORMALS, checked ? "1" : "0");
            if (_viewerManager) {
                _viewerManager->forEachViewer([checked](CVolumeViewer* viewer) {
                    if (viewer) {
                        viewer->setShowSurfaceNormals(checked);
                    }
                });
            }
            // Enable/disable arrow length and max arrows controls
            if (auto* lblArrowLength = ui.labelNormalArrowLength) {
                lblArrowLength->setEnabled(checked);
            }
            if (auto* sliderArrowLength = ui.sliderNormalArrowLength) {
                sliderArrowLength->setEnabled(checked);
            }
            if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
                lblArrowLengthValue->setEnabled(checked);
            }
            if (auto* lblMaxArrows = ui.labelNormalMaxArrows) {
                lblMaxArrows->setEnabled(checked);
            }
            if (auto* sliderMaxArrows = ui.sliderNormalMaxArrows) {
                sliderMaxArrows->setEnabled(checked);
            }
            if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
                lblMaxArrowsValue->setEnabled(checked);
            }
            statusBar()->showMessage(checked ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
        });
    }

    if (auto* sliderArrowLength = ui.sliderNormalArrowLength) {
        int savedScale = settings.value(vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE,
                                        vc3d::settings::viewer::NORMAL_ARROW_LENGTH_SCALE_DEFAULT).toInt();
        savedScale = std::clamp(savedScale, sliderArrowLength->minimum(), sliderArrowLength->maximum());
        QSignalBlocker blocker(sliderArrowLength);
        sliderArrowLength->setValue(savedScale);

        if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
            lblArrowLengthValue->setText(tr("%1%").arg(savedScale));
        }

        float scaleFloat = static_cast<float>(savedScale) / 100.0f;
        if (_viewerManager) {
            _viewerManager->forEachViewer([scaleFloat](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setNormalArrowLengthScale(scaleFloat);
                }
            });
        }

        connect(sliderArrowLength, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_ARROW_LENGTH_SCALE, value);

            if (auto* lblArrowLengthValue = ui.labelNormalArrowLengthValue) {
                lblArrowLengthValue->setText(tr("%1%").arg(value));
            }

            float scaleFloat = static_cast<float>(value) / 100.0f;
            if (_viewerManager) {
                _viewerManager->forEachViewer([scaleFloat](CVolumeViewer* viewer) {
                    if (viewer) {
                        viewer->setNormalArrowLengthScale(scaleFloat);
                    }
                });
            }
        });
    }

    if (auto* sliderMaxArrows = ui.sliderNormalMaxArrows) {
        int savedMaxArrows = settings.value(vc3d::settings::viewer::NORMAL_MAX_ARROWS,
                                            vc3d::settings::viewer::NORMAL_MAX_ARROWS_DEFAULT).toInt();
        savedMaxArrows = std::clamp(savedMaxArrows, sliderMaxArrows->minimum(), sliderMaxArrows->maximum());
        QSignalBlocker blocker(sliderMaxArrows);
        sliderMaxArrows->setValue(savedMaxArrows);

        if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
            lblMaxArrowsValue->setText(QString::number(savedMaxArrows));
        }

        if (_viewerManager) {
            _viewerManager->forEachViewer([savedMaxArrows](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setNormalMaxArrows(savedMaxArrows);
                }
            });
        }

        connect(sliderMaxArrows, &QSlider::valueChanged, this, [this](int value) {
            using namespace vc3d::settings;
            QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
            s.setValue(viewer::NORMAL_MAX_ARROWS, value);

            if (auto* lblMaxArrowsValue = ui.labelNormalMaxArrowsValue) {
                lblMaxArrowsValue->setText(QString::number(value));
            }

            if (_viewerManager) {
                _viewerManager->forEachViewer([value](CVolumeViewer* viewer) {
                    if (viewer) {
                        viewer->setNormalMaxArrows(value);
                    }
                });
            }
        });
    }

    if (auto* btnResetRot = ui.btnResetAxisRotations) {
        connect(btnResetRot, &QPushButton::clicked, this, &CWindow::onResetAxisAlignedRotations);
    }

    // Zoom buttons
    btnZoomIn = ui.btnZoomIn;
    btnZoomOut = ui.btnZoomOut;

    connect(btnZoomIn, &QPushButton::clicked, this, &CWindow::onZoomIn);
    connect(btnZoomOut, &QPushButton::clicked, this, &CWindow::onZoomOut);

    if (auto* volumeContainer = ui.volumeWindowContainer) {
        auto* layout = new QHBoxLayout(volumeContainer);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        _volumeWindowWidget = new WindowRangeWidget(volumeContainer);
        _volumeWindowWidget->setRange(0, 255);
        _volumeWindowWidget->setMinimumSeparation(1);
        _volumeWindowWidget->setControlsEnabled(false);
        layout->addWidget(_volumeWindowWidget);

        connect(_volumeWindowWidget, &WindowRangeWidget::windowValuesChanged,
                this, [this](int low, int high) {
                    if (_viewerManager) {
                        _viewerManager->setVolumeWindow(static_cast<float>(low),
                                                        static_cast<float>(high));
                    }
                });

        if (_viewerManager) {
            connect(_viewerManager.get(), &ViewerManager::volumeWindowChanged,
                    this, [this](float low, float high) {
                        if (!_volumeWindowWidget) {
                            return;
                        }
                        const int lowInt = static_cast<int>(std::lround(low));
                        const int highInt = static_cast<int>(std::lround(high));
                        _volumeWindowWidget->setWindowValues(lowInt, highInt);
                    });

            _volumeWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->volumeWindowLow())),
                static_cast<int>(std::lround(_viewerManager->volumeWindowHigh())));
        }

        const bool viewEnabled = !ui.grpVolManager || ui.grpVolManager->isEnabled();
        _volumeWindowWidget->setControlsEnabled(viewEnabled);
    }

    if (auto* container = ui.overlayWindowContainer) {
        auto* layout = new QHBoxLayout(container);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(6);

        _overlayWindowWidget = new WindowRangeWidget(container);
        _overlayWindowWidget->setRange(0, 255);
        _overlayWindowWidget->setMinimumSeparation(1);
        _overlayWindowWidget->setControlsEnabled(false);
        layout->addWidget(_overlayWindowWidget);

        connect(_overlayWindowWidget, &WindowRangeWidget::windowValuesChanged,
                this, [this](int low, int high) {
                    if (_viewerManager) {
                        _viewerManager->setOverlayWindow(static_cast<float>(low),
                                                         static_cast<float>(high));
                    }
                });

        if (_viewerManager) {
            connect(_viewerManager.get(), &ViewerManager::overlayWindowChanged,
                    this, [this](float low, float high) {
                        if (!_overlayWindowWidget) {
                            return;
                        }
                        const int lowInt = static_cast<int>(std::lround(low));
                        const int highInt = static_cast<int>(std::lround(high));
                        _overlayWindowWidget->setWindowValues(lowInt, highInt);
                    });

            _overlayWindowWidget->setWindowValues(
                static_cast<int>(std::lround(_viewerManager->overlayWindowLow())),
                static_cast<int>(std::lround(_viewerManager->overlayWindowHigh())));
        }
    }

    if (_viewerManager && _overlayWindowWidget) {
        connect(_viewerManager.get(), &ViewerManager::overlayVolumeAvailabilityChanged,
                this, [this](bool hasOverlay) {
                    if (!_overlayWindowWidget) {
                        return;
                    }
                    const bool viewEnabled = !ui.grpVolManager || ui.grpVolManager->isEnabled();
                    _overlayWindowWidget->setControlsEnabled(hasOverlay && viewEnabled);
                });
    }

    if (_overlayWindowWidget) {
        const bool hasOverlay = _volumeOverlay && _volumeOverlay->hasOverlaySelection();
        const bool viewEnabled = !ui.grpVolManager || ui.grpVolManager->isEnabled();
        _overlayWindowWidget->setControlsEnabled(hasOverlay && viewEnabled);
    }

    auto* spinIntersectionOpacity = ui.spinIntersectionOpacity;
    const int savedIntersectionOpacity = settings.value(vc3d::settings::viewer::INTERSECTION_OPACITY,
                                                        spinIntersectionOpacity->value()).toInt();
    const int boundedIntersectionOpacity = std::clamp(savedIntersectionOpacity,
                                                      spinIntersectionOpacity->minimum(),
                                                      spinIntersectionOpacity->maximum());
    spinIntersectionOpacity->setValue(boundedIntersectionOpacity);

    connect(spinIntersectionOpacity, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) {
            return;
        }
        const float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
        _viewerManager->setIntersectionOpacity(normalized);
    });
    if (_viewerManager) {
        _viewerManager->setIntersectionOpacity(spinIntersectionOpacity->value() / 100.0f);
    }

    if (auto* spinIntersectionThickness = ui.doubleSpinIntersectionThickness) {
        const double savedThickness = settings.value(vc3d::settings::viewer::INTERSECTION_THICKNESS,
                                                     spinIntersectionThickness->value()).toDouble();
        const double boundedThickness = std::clamp(savedThickness,
                                                   static_cast<double>(spinIntersectionThickness->minimum()),
                                                   static_cast<double>(spinIntersectionThickness->maximum()));
        spinIntersectionThickness->setValue(boundedThickness);
        connect(spinIntersectionThickness,
                QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                this,
                [this](double value) {
                    if (!_viewerManager) {
                        return;
                    }
                    _viewerManager->setIntersectionThickness(static_cast<float>(value));
                });
        if (_viewerManager) {
            _viewerManager->setIntersectionThickness(static_cast<float>(spinIntersectionThickness->value()));
        }
    }

    auto* comboIntersectionSampling = ui.comboIntersectionSampling;
    if (comboIntersectionSampling) {
        struct SamplingOption {
            const char* label;
            int stride;
        };
        const SamplingOption options[] = {
            {"Full (1x)", 1},
            {"2x", 2},
            {"4x", 4},
            {"8x", 8},
        };
        comboIntersectionSampling->clear();
        for (const auto& opt : options) {
            comboIntersectionSampling->addItem(tr(opt.label), opt.stride);
        }

        const int savedStride = settings.value(vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE,
                                              vc3d::settings::viewer::INTERSECTION_SAMPLING_STRIDE_DEFAULT).toInt();
        int selectedIndex = comboIntersectionSampling->findData(savedStride);
        if (selectedIndex < 0) {
            selectedIndex = comboIntersectionSampling->findData(1);
        }
        if (selectedIndex >= 0) {
            comboIntersectionSampling->setCurrentIndex(selectedIndex);
        }

        connect(comboIntersectionSampling,
                QOverload<int>::of(&QComboBox::currentIndexChanged),
                this,
                [this, comboIntersectionSampling](int) {
                    if (!_viewerManager) {
                        return;
                    }
                    const int stride = std::max(1, comboIntersectionSampling->currentData().toInt());
                    _viewerManager->setSurfacePatchSamplingStride(stride);
                });

        // Update combobox when stride changes programmatically (e.g., tiered defaults)
        if (_viewerManager) {
            connect(_viewerManager.get(),
                    &ViewerManager::samplingStrideChanged,
                    this,
                    [comboIntersectionSampling](int stride) {
                        const int index = comboIntersectionSampling->findData(stride);
                        if (index >= 0 && index != comboIntersectionSampling->currentIndex()) {
                            QSignalBlocker blocker(comboIntersectionSampling);
                            comboIntersectionSampling->setCurrentIndex(index);
                        }
                    });
        }
    }

    chkAxisAlignedSlices = ui.chkAxisAlignedSlices;
    if (chkAxisAlignedSlices) {
        bool useAxisAligned = settings.value(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES,
                                             vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES_DEFAULT).toBool();
        QSignalBlocker blocker(chkAxisAlignedSlices);
        chkAxisAlignedSlices->setChecked(useAxisAligned);
        connect(chkAxisAlignedSlices, &QCheckBox::toggled, this, &CWindow::onAxisAlignedSlicesToggled);
    }

    spNorm[0] = ui.dspNX;
    spNorm[1] = ui.dspNY;
    spNorm[2] = ui.dspNZ;

    for (int i = 0; i < 3; i++) {
        spNorm[i]->setRange(-10, 10);
    }

    connect(spNorm[0], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[1], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);
    connect(spNorm[2], &QDoubleSpinBox::valueChanged, this, &CWindow::onManualPlaneChanged);

    connect(ui.btnEditMask, &QPushButton::pressed, this, &CWindow::onEditMaskPressed);
    connect(ui.btnAppendMask, &QPushButton::pressed, this, &CWindow::onAppendMaskPressed);
}

void CWindow::setupCompositeControls()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    // Connect composite view controls
    connect(ui.chkCompositeEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeEnabled(checked);
        }
    });

    connect(ui.cmbCompositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        // Find the segmentation viewer and update its composite method
        std::string method = "max";
        switch (index) {
            case 0: method = "max"; break;
            case 1: method = "mean"; break;
            case 2: method = "min"; break;
            case 3: method = "alpha"; break;
            case 4: method = "beerLambert"; break;
        }

        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeMethod(method);
        }
    });

    if (chkAxisAlignedSlices) {
        onAxisAlignedSlicesToggled(chkAxisAlignedSlices->isChecked());
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        onAxisOverlayOpacityChanged(spinAxisOverlayOpacity->value());
    }
    if (auto* chkAxisOverlays = ui.chkAxisOverlays) {
        onAxisOverlayVisibilityToggled(chkAxisOverlays->isChecked());
    }

    // Connect Layers In Front controls
    connect(ui.spinLayersInFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeLayersInFront(value);
        }
    });

    // Connect Layers Behind controls
    connect(ui.spinLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeLayersBehind(value);
        }
    });

    // Connect Alpha Min controls
    connect(ui.spinAlphaMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeAlphaMin(value);
        }
    });

    // Connect Alpha Max controls
    connect(ui.spinAlphaMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeAlphaMax(value);
        }
    });

    // Connect Alpha Threshold controls
    connect(ui.spinAlphaThreshold, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) {
            return;
        }
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeAlphaThreshold(value);
                break;
            }
        }
    });

    // Connect Material controls
    connect(ui.spinMaterial, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (!_viewerManager) {
            return;
        }
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeMaterial(value);
                break;
            }
        }
    });

    // Connect Reverse Direction control
    connect(ui.chkReverseDirection, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) {
            return;
        }
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "segmentation") {
                viewer->setCompositeReverseDirection(checked);
                break;
            }
        }
    });

    // Connect Beer-Lambert Extinction control
    connect(ui.spinBLExtinction, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeBLExtinction(static_cast<float>(value));
        }
    });

    // Connect Beer-Lambert Emission control
    connect(ui.spinBLEmission, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeBLEmission(static_cast<float>(value));
        }
    });

    // Connect Beer-Lambert Ambient control
    connect(ui.spinBLAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setCompositeBLAmbient(static_cast<float>(value));
        }
    });

    // Connect Lighting Enable control
    connect(ui.chkLightingEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setLightingEnabled(checked);
        }
    });

    // Connect Light Azimuth control
    connect(ui.spinLightAzimuth, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setLightAzimuth(static_cast<float>(value));
        }
    });

    // Connect Light Elevation control
    connect(ui.spinLightElevation, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setLightElevation(static_cast<float>(value));
        }
    });

    // Connect Light Diffuse control
    connect(ui.spinLightDiffuse, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setLightDiffuse(static_cast<float>(value));
        }
    });

    // Connect Light Ambient control
    connect(ui.spinLightAmbient, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setLightAmbient(static_cast<float>(value));
        }
    });

    // Connect Volume Gradients checkbox
    connect(ui.chkUseVolumeGradients, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setUseVolumeGradients(checked);
        }
    });

    // Connect ISO Cutoff slider - applies to all viewers (segmentation, XY, XZ, YZ)
    connect(ui.sliderIsoCutoff, &QSlider::valueChanged, this, [this](int value) {
        ui.lblIsoCutoffValue->setText(QString::number(value));
        if (!_viewerManager) {
            return;
        }
        _viewerManager->forEachViewer([value](CVolumeViewer* viewer) {
            viewer->setIsoCutoff(value);
        });
    });

    // Connect Method Scale slider (for methods with scale parameters)
    connect(ui.sliderMethodScale, &QSlider::valueChanged, this, [this](int value) {
        // Convert slider value (1-100) to scale (0.1-10.0)
        float scale = value / 10.0f;
        ui.lblMethodScaleValue->setText(QString::number(scale, 'f', 1));

        if (!_viewerManager) {
            return;
        }

        // Currently no methods use the scale parameter
        (void)scale;
    });

    // Connect Method Param slider (for methods with threshold/percentile parameters)
    connect(ui.sliderMethodParam, &QSlider::valueChanged, this, [this](int value) {
        // Currently no methods use this parameter
        (void)value;
    });

    // Helper lambda to update visibility of method-specific parameters
    auto updateCompositeParamsVisibility = [this](int methodIndex) {
        // Alpha parameters (row 1, 2 - AlphaMin/Max, AlphaThreshold/Material)
        bool showAlphaParams = (methodIndex == 3); // Alpha method
        ui.lblAlphaMin->setVisible(showAlphaParams);
        ui.spinAlphaMin->setVisible(showAlphaParams);
        ui.lblAlphaMax->setVisible(showAlphaParams);
        ui.spinAlphaMax->setVisible(showAlphaParams);
        ui.lblAlphaThreshold->setVisible(showAlphaParams);
        ui.spinAlphaThreshold->setVisible(showAlphaParams);
        ui.lblMaterial->setVisible(showAlphaParams);
        ui.spinMaterial->setVisible(showAlphaParams);

        // Beer-Lambert parameters (row 7, 8 - Extinction/Emission, Ambient)
        bool showBLParams = (methodIndex == 4); // Beer-Lambert method
        ui.lblBLExtinction->setVisible(showBLParams);
        ui.spinBLExtinction->setVisible(showBLParams);
        ui.lblBLEmission->setVisible(showBLParams);
        ui.spinBLEmission->setVisible(showBLParams);
        ui.lblBLAmbient->setVisible(showBLParams);
        ui.spinBLAmbient->setVisible(showBLParams);

        // Lighting parameters (rows 9-12) - always shown, works with all methods
        ui.chkLightingEnabled->setVisible(true);
        ui.lblLightAzimuth->setVisible(true);
        ui.spinLightAzimuth->setVisible(true);
        ui.lblLightElevation->setVisible(true);
        ui.spinLightElevation->setVisible(true);
        ui.lblLightDiffuse->setVisible(true);
        ui.spinLightDiffuse->setVisible(true);
        ui.lblLightAmbient->setVisible(true);
        ui.spinLightAmbient->setVisible(true);
        ui.chkUseVolumeGradients->setVisible(true);

        // No methods currently use scale or param sliders
        ui.lblMethodScale->setVisible(false);
        ui.sliderMethodScale->setVisible(false);
        ui.lblMethodScaleValue->setVisible(false);
        ui.lblMethodParam->setVisible(false);
        ui.sliderMethodParam->setVisible(false);
        ui.lblMethodParamValue->setVisible(false);
    };

    // Update the cmbCompositeMode connection to also update visibility
    connect(ui.cmbCompositeMode, QOverload<int>::of(&QComboBox::currentIndexChanged), this, updateCompositeParamsVisibility);

    // Initialize visibility based on current selection
    updateCompositeParamsVisibility(ui.cmbCompositeMode->currentIndex());

    // Connect Plane Composite controls (separate enable for XY/XZ/YZ, shared layer counts)
    connect(ui.chkPlaneCompositeXY, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) return;
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "xy plane") {
                viewer->setPlaneCompositeEnabled(checked);
            }
        }
    });

    connect(ui.chkPlaneCompositeXZ, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) return;
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "seg xz") {
                viewer->setPlaneCompositeEnabled(checked);
            }
        }
    });

    connect(ui.chkPlaneCompositeYZ, &QCheckBox::toggled, this, [this](bool checked) {
        if (!_viewerManager) return;
        for (auto* viewer : _viewerManager->viewers()) {
            if (viewer->surfName() == "seg yz") {
                viewer->setPlaneCompositeEnabled(checked);
            }
        }
    });

    auto isPlaneViewer = [](const std::string& name) {
        return name == "seg xz" || name == "seg yz" || name == "xy plane";
    };

    connect(ui.spinPlaneLayersFront, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, isPlaneViewer](int value) {
        if (!_viewerManager) return;
        int behind = ui.spinPlaneLayersBehind->value();
        for (auto* viewer : _viewerManager->viewers()) {
            if (isPlaneViewer(viewer->surfName())) {
                viewer->setPlaneCompositeLayers(value, behind);
            }
        }
    });

    connect(ui.spinPlaneLayersBehind, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, isPlaneViewer](int value) {
        if (!_viewerManager) return;
        int front = ui.spinPlaneLayersFront->value();
        for (auto* viewer : _viewerManager->viewers()) {
            if (isPlaneViewer(viewer->surfName())) {
                viewer->setPlaneCompositeLayers(front, value);
            }
        }
    });

    // Connect Postprocessing controls
    connect(ui.chkStretchValuesPost, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setPostStretchValues(checked);
        }
    });

    connect(ui.chkRemoveSmallComponents, &QCheckBox::toggled, this, [this](bool checked) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setPostRemoveSmallComponents(checked);
        }
        // Enable/disable the min component size spinbox based on checkbox state
        ui.spinMinComponentSize->setEnabled(checked);
        ui.lblMinComponentSize->setEnabled(checked);
    });

    connect(ui.spinMinComponentSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (auto* viewer = segmentationViewer()) {
            viewer->setPostMinComponentSize(value);
        }
    });

    // Initialize min component size controls based on checkbox state
    ui.spinMinComponentSize->setEnabled(ui.chkRemoveSmallComponents->isChecked());
    ui.lblMinComponentSize->setEnabled(ui.chkRemoveSmallComponents->isChecked());

    bool resetViewOnSurfaceChange = settings.value(vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE,
                                                   vc3d::settings::viewer::RESET_VIEW_ON_SURFACE_CHANGE_DEFAULT).toBool();
    if (_viewerManager) {
        for (auto* viewer : _viewerManager->viewers()) {
            viewer->setResetViewOnSurfaceChange(resetViewOnSurfaceChange);
            _viewerManager->setResetDefaultFor(viewer, resetViewOnSurfaceChange);
        }
    }
}
