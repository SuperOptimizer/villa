// Signal connections for SegmentationWidget, extracted from buildUi() for parallel compilation.

#include "SegmentationWidget.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"
#include "NeuralTraceServiceManager.hpp"

#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>

namespace
{

float displayOpacityToNormalized(double displayValue)
{
    constexpr float kAlphaOpacityScale = 255.0f;
    return static_cast<float>(displayValue / kAlphaOpacityScale);
}

}  // namespace

void SegmentationWidget::connectSignals()
{
    connect(_chkEditing, &QCheckBox::toggled, this, [this](bool enabled) {
        updateEditingState(enabled, true);
    });
    connect(_chkShowHoverMarker, &QCheckBox::toggled, this, [this](bool enabled) {
        setShowHoverMarker(enabled);
    });

    // Approval mask signal connections
    connect(_chkShowApprovalMask, &QCheckBox::toggled, this, [this](bool enabled) {
        setShowApprovalMask(enabled);
        // If show is being unchecked and edit modes are active, turn them off
        if (!enabled) {
            if (_editApprovedMask) {
                setEditApprovedMask(false);
            }
            if (_editUnapprovedMask) {
                setEditUnapprovedMask(false);
            }
        }
    });

    connect(_chkEditApprovedMask, &QCheckBox::toggled, this, [this](bool enabled) {
        setEditApprovedMask(enabled);
    });

    connect(_chkEditUnapprovedMask, &QCheckBox::toggled, this, [this](bool enabled) {
        setEditUnapprovedMask(enabled);
    });

    connect(_spinApprovalBrushRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setApprovalBrushRadius(static_cast<float>(value));
    });

    connect(_spinApprovalBrushDepth, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setApprovalBrushDepth(static_cast<float>(value));
    });

    connect(_sliderApprovalMaskOpacity, &QSlider::valueChanged, this, [this](int value) {
        setApprovalMaskOpacity(value);
    });

    connect(_btnApprovalColor, &QPushButton::clicked, this, [this]() {
        QColor newColor = QColorDialog::getColor(_approvalBrushColor, this, tr("Choose Approval Mask Color"));
        if (newColor.isValid()) {
            setApprovalBrushColor(newColor);
        }
    });

    connect(_btnUndoApprovalStroke, &QPushButton::clicked, this, &SegmentationWidget::approvalStrokesUndoRequested);

    // Cell reoptimization signal connections
    connect(_chkCellReoptMode, &QCheckBox::toggled, this, [this](bool enabled) {
        setCellReoptMode(enabled);
    });

    connect(_spinCellReoptMaxSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_cellReoptMaxSteps != value) {
            _cellReoptMaxSteps = value;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_max_steps"), value);
                emit cellReoptMaxStepsChanged(value);
            }
        }
    });

    connect(_spinCellReoptMaxPoints, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_cellReoptMaxPoints != value) {
            _cellReoptMaxPoints = value;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_max_points"), value);
                emit cellReoptMaxPointsChanged(value);
            }
        }
    });

    connect(_spinCellReoptMinSpacing, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float floatVal = static_cast<float>(value);
        if (_cellReoptMinSpacing != floatVal) {
            _cellReoptMinSpacing = floatVal;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_min_spacing"), value);
                emit cellReoptMinSpacingChanged(floatVal);
            }
        }
    });

    connect(_spinCellReoptPerimeterOffset, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        float floatVal = static_cast<float>(value);
        if (_cellReoptPerimeterOffset != floatVal) {
            _cellReoptPerimeterOffset = floatVal;
            if (!_restoringSettings) {
                writeSetting(QStringLiteral("cell_reopt_perimeter_offset"), value);
                emit cellReoptPerimeterOffsetChanged(floatVal);
            }
        }
    });

    connect(_btnCellReoptRun, &QPushButton::clicked, this, [this]() {
        uint64_t collectionId = 0;
        if (_comboCellReoptCollection && _comboCellReoptCollection->currentIndex() >= 0) {
            collectionId = _comboCellReoptCollection->currentData().toULongLong();
        }
        emit cellReoptGrowthRequested(collectionId);
    });

    auto connectDirectionCheckbox = [this](QCheckBox* box) {
        if (!box) {
            return;
        }
        connect(box, &QCheckBox::toggled, this, [this, box](bool) {
            updateGrowthDirectionMaskFromUi(box);
        });
    };
    connectDirectionCheckbox(_chkGrowthDirUp);
    connectDirectionCheckbox(_chkGrowthDirDown);
    connectDirectionCheckbox(_chkGrowthDirLeft);
    connectDirectionCheckbox(_chkGrowthDirRight);

    connect(_chkGrowthKeybindsEnabled, &QCheckBox::toggled, this, [this](bool checked) {
        _growthKeybindsEnabled = checked;
        writeSetting(QStringLiteral("growth_keybinds_enabled"), _growthKeybindsEnabled);
    });

    connect(_spinGrowthSteps, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) { applyGrowthSteps(value, true, true); });

    connect(_comboGrowthMethod, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const auto method = static_cast<SegmentationGrowthMethod>(
                    _comboGrowthMethod->itemData(index).toInt());
                setGrowthMethod(method);
            });

    connect(_spinExtrapolationPoints, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this](int value) {
                _extrapolationPointCount = std::clamp(value, 3, 20);
                writeSetting(QStringLiteral("extrapolation_point_count"), _extrapolationPointCount);
            });

    connect(_comboExtrapolationType, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                _extrapolationType = extrapolationTypeFromInt(
                    _comboExtrapolationType->itemData(index).toInt());
                writeSetting(QStringLiteral("extrapolation_type"), static_cast<int>(_extrapolationType));
                // Show SDT params only when Extrapolation method AND Linear+Fit type
                if (_sdtParamsContainer) {
                    _sdtParamsContainer->setVisible(
                        _growthMethod == SegmentationGrowthMethod::Extrapolation &&
                        _extrapolationType == ExtrapolationType::LinearFit);
                }
                // Show skeleton params only when Extrapolation method AND SkeletonPath type
                if (_skeletonParamsContainer) {
                    _skeletonParamsContainer->setVisible(
                        _growthMethod == SegmentationGrowthMethod::Extrapolation &&
                        _extrapolationType == ExtrapolationType::SkeletonPath);
                }
                // Hide fit points label and spinbox for SkeletonPath (it doesn't use polynomial fitting)
                bool showFitPoints = _extrapolationType != ExtrapolationType::SkeletonPath;
                if (_lblExtrapolationPoints) {
                    _lblExtrapolationPoints->setVisible(showFitPoints);
                }
                if (_spinExtrapolationPoints) {
                    _spinExtrapolationPoints->setVisible(showFitPoints);
                }
            });

    // SDT/Newton refinement parameter connections
    connect(_spinSDTMaxSteps, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _sdtMaxSteps = std::clamp(value, 1, 10);
        writeSetting(QStringLiteral("sdt_max_steps"), _sdtMaxSteps);
    });
    connect(_spinSDTStepSize, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _sdtStepSize = std::clamp(static_cast<float>(value), 0.1f, 2.0f);
        writeSetting(QStringLiteral("sdt_step_size"), static_cast<double>(_sdtStepSize));
    });
    connect(_spinSDTConvergence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _sdtConvergence = std::clamp(static_cast<float>(value), 0.1f, 2.0f);
        writeSetting(QStringLiteral("sdt_convergence"), static_cast<double>(_sdtConvergence));
    });
    connect(_spinSDTChunkSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _sdtChunkSize = std::clamp(value, 32, 256);
        writeSetting(QStringLiteral("sdt_chunk_size"), _sdtChunkSize);
    });

    // Skeleton path parameter connections
    connect(_comboSkeletonConnectivity, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _skeletonConnectivity = _comboSkeletonConnectivity->itemData(index).toInt();
        writeSetting(QStringLiteral("skeleton_connectivity"), _skeletonConnectivity);
    });
    connect(_comboSkeletonSliceOrientation, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _skeletonSliceOrientation = _comboSkeletonSliceOrientation->itemData(index).toInt();
        writeSetting(QStringLiteral("skeleton_slice_orientation"), _skeletonSliceOrientation);
    });
    connect(_spinSkeletonChunkSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _skeletonChunkSize = std::clamp(value, 32, 256);
        writeSetting(QStringLiteral("skeleton_chunk_size"), _skeletonChunkSize);
    });
    connect(_spinSkeletonSearchRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _skeletonSearchRadius = std::clamp(value, 1, 100);
        writeSetting(QStringLiteral("skeleton_search_radius"), _skeletonSearchRadius);
    });

    const auto triggerConfiguredGrowth = [this]() {
        const auto allowed = allowedGrowthDirections();
        auto direction = SegmentationGrowthDirection::All;
        if (allowed.size() == 1) {
            direction = allowed.front();
        }
        triggerGrowthRequest(direction, _growthSteps, false);
    };

    connect(_btnGrow, &QPushButton::clicked, this, triggerConfiguredGrowth);
    connect(_btnInpaint, &QPushButton::clicked, this, [this]() {
        triggerGrowthRequest(SegmentationGrowthDirection::All, 0, true);
    });

    connect(_comboVolumes, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            return;
        }
        const QString volumeId = _comboVolumes->itemData(index).toString();
        if (volumeId.isEmpty() || volumeId == _activeVolumeId) {
            return;
        }
        _activeVolumeId = volumeId;
        emit volumeSelectionChanged(volumeId);
    });

    connect(_spinDragRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDragRadius(static_cast<float>(value));
        emit dragRadiusChanged(_dragRadiusSteps);
    });

    connect(_spinDragSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setDragSigma(static_cast<float>(value));
        emit dragSigmaChanged(_dragSigmaSteps);
    });

    connect(_spinLineRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setLineRadius(static_cast<float>(value));
        emit lineRadiusChanged(_lineRadiusSteps);
    });

    connect(_spinLineSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setLineSigma(static_cast<float>(value));
        emit lineSigmaChanged(_lineSigmaSteps);
    });

    connect(_spinPushPullRadius, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullRadius(static_cast<float>(value));
        emit pushPullRadiusChanged(_pushPullRadiusSteps);
    });

    connect(_spinPushPullSigma, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullSigma(static_cast<float>(value));
        emit pushPullSigmaChanged(_pushPullSigmaSteps);
    });

    connect(_spinPushPullStep, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setPushPullStep(static_cast<float>(value));
        emit pushPullStepChanged(_pushPullStep);
    });

    auto onAlphaValueChanged = [this](auto updater) {
        AlphaPushPullConfig config = _alphaPushPullConfig;
        updater(config);
        applyAlphaPushPullConfig(config, true);
    };

    connect(_spinAlphaStart, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.start = static_cast<float>(value);
        });
    });
    connect(_spinAlphaStop, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.stop = static_cast<float>(value);
        });
    });
    connect(_spinAlphaStep, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.step = static_cast<float>(value);
        });
    });
    connect(_spinAlphaLow, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.low = displayOpacityToNormalized(value);
        });
    });
    connect(_spinAlphaHigh, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.high = displayOpacityToNormalized(value);
        });
    });
    connect(_spinAlphaBorder, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.borderOffset = static_cast<float>(value);
        });
    });
    connect(_spinAlphaBlurRadius, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, onAlphaValueChanged](int value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.blurRadius = value;
        });
    });
    connect(_spinAlphaPerVertexLimit, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, onAlphaValueChanged](double value) {
        onAlphaValueChanged([value](AlphaPushPullConfig& cfg) {
            cfg.perVertexLimit = static_cast<float>(value);
        });
    });
    connect(_chkAlphaPerVertex, &QCheckBox::toggled, this, [this, onAlphaValueChanged](bool checked) {
        onAlphaValueChanged([checked](AlphaPushPullConfig& cfg) {
            cfg.perVertex = checked;
        });
    });

    connect(_spinSmoothStrength, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        setSmoothingStrength(static_cast<float>(value));
        emit smoothingStrengthChanged(_smoothStrength);
    });

    connect(_spinSmoothIterations, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        setSmoothingIterations(value);
        emit smoothingIterationsChanged(_smoothIterations);
    });

    connect(_directionFieldPathEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _directionFieldPath = text.trimmed();
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_directionFieldBrowseButton, &QToolButton::clicked, this, [this]() {
        const QString initial = _directionFieldPath.isEmpty() ? QDir::homePath() : _directionFieldPath;
        const QString dir = QFileDialog::getExistingDirectory(this, tr("Select direction field"), initial);
        if (dir.isEmpty()) {
            return;
        }
        _directionFieldPath = dir;
        _directionFieldPathEdit->setText(dir);
    });

    connect(_comboDirectionFieldOrientation, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldOrientation = segmentationDirectionFieldOrientationFromInt(
            _comboDirectionFieldOrientation->itemData(index).toInt());
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_comboDirectionFieldScale, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _directionFieldScale = _comboDirectionFieldScale->itemData(index).toInt();
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_spinDirectionFieldWeight, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
        _directionFieldWeight = value;
        if (!_updatingDirectionFieldForm) {
            applyDirectionFieldDraftToSelection(_directionFieldList ? _directionFieldList->currentRow() : -1);
        }
    });

    connect(_directionFieldAddButton, &QPushButton::clicked, this, [this]() {
        auto config = buildDirectionFieldDraft();
        if (!config.isValid()) {
            return;
        }
        _directionFields.push_back(std::move(config));
        refreshDirectionFieldList();
        persistDirectionFields();
        clearDirectionFieldForm();
    });

    connect(_directionFieldRemoveButton, &QPushButton::clicked, this, [this]() {
        const int row = _directionFieldList ? _directionFieldList->currentRow() : -1;
        if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
            return;
        }
        _directionFields.erase(_directionFields.begin() + row);
        refreshDirectionFieldList();
        persistDirectionFields();
    });

    connect(_directionFieldClearButton, &QPushButton::clicked, this, [this]() {
        clearDirectionFieldForm();
    });

    connect(_directionFieldList, &QListWidget::currentRowChanged, this, [this](int row) {
        updateDirectionFieldFormFromSelection(row);
        if (_directionFieldRemoveButton) {
            _directionFieldRemoveButton->setEnabled(_editingEnabled && row >= 0);
        }
    });

    connect(_comboCorrections, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (index < 0) {
            emit correctionsCollectionSelected(0);
            return;
        }
        const QVariant data = _comboCorrections->itemData(index);
        emit correctionsCollectionSelected(data.toULongLong());
    });

    connect(_btnCorrectionsNew, &QPushButton::clicked, this, [this]() {
        emit correctionsCreateRequested();
    });

    connect(_editCustomParams, &QPlainTextEdit::textChanged, this, [this]() {
        handleCustomParamsEdited();
    });

    connect(_comboCustomParamsProfile, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (_restoringSettings) {
            return;
        }
        if (!_comboCustomParamsProfile || idx < 0) {
            return;
        }
        const QString profile = _comboCustomParamsProfile->itemData(idx).toString();
        applyCustomParamsProfile(profile, /*persist=*/true, /*fromUi=*/true);
    });

    connect(_comboNormal3d, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        if (_restoringSettings) {
            return;
        }
        if (!_comboNormal3d || idx < 0) {
            return;
        }
        const QString path = _comboNormal3d->itemData(idx).toString();
        if (path.isEmpty() || path == _normal3dSelectedPath) {
            return;
        }
        _normal3dSelectedPath = path;
        writeSetting(QStringLiteral("normal3d_selected_path"), _normal3dSelectedPath);
        updateNormal3dUi();
    });

    connect(_chkCorrectionsAnnotate, &QCheckBox::toggled, this, [this](bool enabled) {
        emit correctionsAnnotateToggled(enabled);
    });

    connect(_chkCorrectionsUseZRange, &QCheckBox::toggled, this, [this](bool enabled) {
        _correctionsZRangeEnabled = enabled;
        writeSetting(QStringLiteral("corrections_z_range_enabled"), _correctionsZRangeEnabled);
        updateGrowthUiState();
        emit correctionsZRangeChanged(enabled, _correctionsZMin, _correctionsZMax);
    });

    connect(_spinCorrectionsZMin, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMin == value) {
            return;
        }
        _correctionsZMin = value;
        writeSetting(QStringLiteral("corrections_z_min"), _correctionsZMin);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_spinCorrectionsZMax, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        if (_correctionsZMax == value) {
            return;
        }
        _correctionsZMax = value;
        writeSetting(QStringLiteral("corrections_z_max"), _correctionsZMax);
        if (_correctionsZRangeEnabled) {
            emit correctionsZRangeChanged(true, _correctionsZMin, _correctionsZMax);
        }
    });

    connect(_btnApply, &QPushButton::clicked, this, &SegmentationWidget::applyRequested);
    connect(_btnReset, &QPushButton::clicked, this, &SegmentationWidget::resetRequested);
    connect(_btnStop, &QPushButton::clicked, this, &SegmentationWidget::stopToolsRequested);

    // Neural tracer connections
    connect(_chkNeuralTracerEnabled, &QCheckBox::toggled, this, [this](bool enabled) {
        setNeuralTracerEnabled(enabled);
    });

    connect(_neuralCheckpointEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _neuralCheckpointPath = text.trimmed();
        writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);
    });

    connect(_neuralCheckpointBrowse, &QToolButton::clicked, this, [this]() {
        const QString initial = _neuralCheckpointPath.isEmpty() ? QDir::homePath() : _neuralCheckpointPath;
        const QString file = QFileDialog::getOpenFileName(this, tr("Select neural tracer checkpoint"),
                                                          initial, tr("PyTorch Checkpoint (*.pt *.pth);;All Files (*)"));
        if (!file.isEmpty()) {
            _neuralCheckpointPath = file;
            _neuralCheckpointEdit->setText(file);
        }
    });

    connect(_neuralPythonEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        _neuralPythonPath = text.trimmed();
        writeSetting(QStringLiteral("neural_python_path"), _neuralPythonPath);
    });

    connect(_neuralPythonBrowse, &QToolButton::clicked, this, [this]() {
        const QString initial = _neuralPythonPath.isEmpty() ? QDir::homePath() : QFileInfo(_neuralPythonPath).absolutePath();
        const QString file = QFileDialog::getOpenFileName(this, tr("Select Python executable"),
                                                          initial, tr("All Files (*)"));
        if (!file.isEmpty()) {
            _neuralPythonPath = file;
            _neuralPythonEdit->setText(file);
        }
    });

    connect(_comboNeuralVolumeScale, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
        _neuralVolumeScale = _comboNeuralVolumeScale->itemData(index).toInt();
        writeSetting(QStringLiteral("neural_volume_scale"), _neuralVolumeScale);
    });

    connect(_spinNeuralBatchSize, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int value) {
        _neuralBatchSize = value;
        writeSetting(QStringLiteral("neural_batch_size"), _neuralBatchSize);
    });

    // Connect to service manager signals
    auto& serviceManager = NeuralTraceServiceManager::instance();
    connect(&serviceManager, &NeuralTraceServiceManager::statusMessage, this, [this](const QString& message) {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(message);
            _lblNeuralTracerStatus->setVisible(true);
            _lblNeuralTracerStatus->setStyleSheet(QString());
        }
        emit neuralTracerStatusMessage(message);
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceStarted, this, [this]() {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(tr("Service running"));
            _lblNeuralTracerStatus->setStyleSheet(QStringLiteral("color: #27ae60;"));
        }
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceStopped, this, [this]() {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(tr("Service stopped"));
            _lblNeuralTracerStatus->setStyleSheet(QString());
        }
    });
    connect(&serviceManager, &NeuralTraceServiceManager::serviceError, this, [this](const QString& error) {
        if (_lblNeuralTracerStatus) {
            _lblNeuralTracerStatus->setText(tr("Error: %1").arg(error));
            _lblNeuralTracerStatus->setStyleSheet(QStringLiteral("color: #c0392b;"));
            _lblNeuralTracerStatus->setVisible(true);
        }
    });
}
