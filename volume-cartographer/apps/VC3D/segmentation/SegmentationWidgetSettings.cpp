/**
 * @file SegmentationWidgetSettings.cpp
 * @brief Settings persistence methods extracted from SegmentationWidget
 *
 * This file contains methods for restoring and saving widget settings.
 * Extracted from SegmentationWidget.cpp to improve parallel compilation.
 */

#include "SegmentationWidget.hpp"

#include "../VCSettings.hpp"
#include "../elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>

#include <algorithm>

// Duplicated from SegmentationWidget.cpp (local constants and helpers)
namespace {
constexpr int kGrowDirUpBit    = 1 << 0;
constexpr int kGrowDirDownBit  = 1 << 1;
constexpr int kGrowDirLeftBit  = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;

QString settingsGroup()
{
    return QStringLiteral("segmentation_edit");
}
}

void SegmentationWidget::restoreSettings()
{
    using namespace vc3d::settings;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());

    _restoringSettings = true;

    if (settings.contains(segmentation::DRAG_RADIUS_STEPS)) {
        _dragRadiusSteps = settings.value(segmentation::DRAG_RADIUS_STEPS, _dragRadiusSteps).toFloat();
    } else {
        _dragRadiusSteps = settings.value(segmentation::RADIUS_STEPS, _dragRadiusSteps).toFloat();
    }

    if (settings.contains(segmentation::DRAG_SIGMA_STEPS)) {
        _dragSigmaSteps = settings.value(segmentation::DRAG_SIGMA_STEPS, _dragSigmaSteps).toFloat();
    } else {
        _dragSigmaSteps = settings.value(segmentation::SIGMA_STEPS, _dragSigmaSteps).toFloat();
    }

    _lineRadiusSteps = settings.value(segmentation::LINE_RADIUS_STEPS, _dragRadiusSteps).toFloat();
    _lineSigmaSteps = settings.value(segmentation::LINE_SIGMA_STEPS, _dragSigmaSteps).toFloat();

    _pushPullRadiusSteps = settings.value(segmentation::PUSH_PULL_RADIUS_STEPS, _dragRadiusSteps).toFloat();
    _pushPullSigmaSteps = settings.value(segmentation::PUSH_PULL_SIGMA_STEPS, _dragSigmaSteps).toFloat();
    _showHoverMarker = settings.value(segmentation::SHOW_HOVER_MARKER, _showHoverMarker).toBool();

    _dragRadiusSteps = std::clamp(_dragRadiusSteps, 0.25f, 128.0f);
    _dragSigmaSteps = std::clamp(_dragSigmaSteps, 0.05f, 64.0f);
    _lineRadiusSteps = std::clamp(_lineRadiusSteps, 0.25f, 128.0f);
    _lineSigmaSteps = std::clamp(_lineSigmaSteps, 0.05f, 64.0f);
    _pushPullRadiusSteps = std::clamp(_pushPullRadiusSteps, 0.25f, 128.0f);
    _pushPullSigmaSteps = std::clamp(_pushPullSigmaSteps, 0.05f, 64.0f);

    _pushPullStep = settings.value(segmentation::PUSH_PULL_STEP, _pushPullStep).toFloat();
    _pushPullStep = std::clamp(_pushPullStep, 0.05f, 40.0f);

    AlphaPushPullConfig storedAlpha = _alphaPushPullConfig;
    storedAlpha.start = settings.value(segmentation::PUSH_PULL_ALPHA_START, storedAlpha.start).toFloat();
    storedAlpha.stop = settings.value(segmentation::PUSH_PULL_ALPHA_STOP, storedAlpha.stop).toFloat();
    storedAlpha.step = settings.value(segmentation::PUSH_PULL_ALPHA_STEP, storedAlpha.step).toFloat();
    storedAlpha.low = settings.value(segmentation::PUSH_PULL_ALPHA_LOW, storedAlpha.low).toFloat();
    storedAlpha.high = settings.value(segmentation::PUSH_PULL_ALPHA_HIGH, storedAlpha.high).toFloat();
    storedAlpha.borderOffset = settings.value(segmentation::PUSH_PULL_ALPHA_BORDER, storedAlpha.borderOffset).toFloat();
    storedAlpha.blurRadius = settings.value(segmentation::PUSH_PULL_ALPHA_RADIUS, storedAlpha.blurRadius).toInt();
    storedAlpha.perVertexLimit = settings.value(segmentation::PUSH_PULL_ALPHA_LIMIT, storedAlpha.perVertexLimit).toFloat();
    storedAlpha.perVertex = settings.value(segmentation::PUSH_PULL_ALPHA_PER_VERTEX, storedAlpha.perVertex).toBool();
    applyAlphaPushPullConfig(storedAlpha, false, false);
    _smoothStrength = settings.value(segmentation::SMOOTH_STRENGTH, _smoothStrength).toFloat();
    _smoothIterations = settings.value(segmentation::SMOOTH_ITERATIONS, _smoothIterations).toInt();
    _smoothStrength = std::clamp(_smoothStrength, 0.0f, 1.0f);
    _smoothIterations = std::clamp(_smoothIterations, 1, 25);
    _growthMethod = segmentationGrowthMethodFromInt(
        settings.value(segmentation::GROWTH_METHOD, static_cast<int>(_growthMethod)).toInt());
    _extrapolationPointCount = settings.value(QStringLiteral("extrapolation_point_count"), _extrapolationPointCount).toInt();
    _extrapolationPointCount = std::clamp(_extrapolationPointCount, 3, 20);
    _extrapolationType = extrapolationTypeFromInt(
        settings.value(QStringLiteral("extrapolation_type"), static_cast<int>(_extrapolationType)).toInt());

    // Restore SDT/Newton refinement parameters
    _sdtMaxSteps = std::clamp(settings.value(QStringLiteral("sdt_max_steps"), _sdtMaxSteps).toInt(), 1, 10);
    _sdtStepSize = std::clamp(settings.value(QStringLiteral("sdt_step_size"), static_cast<double>(_sdtStepSize)).toFloat(), 0.1f, 2.0f);
    _sdtConvergence = std::clamp(settings.value(QStringLiteral("sdt_convergence"), static_cast<double>(_sdtConvergence)).toFloat(), 0.1f, 2.0f);
    _sdtChunkSize = std::clamp(settings.value(QStringLiteral("sdt_chunk_size"), _sdtChunkSize).toInt(), 32, 256);

    // Restore skeleton path parameters
    int storedConnectivity = settings.value(QStringLiteral("skeleton_connectivity"), _skeletonConnectivity).toInt();
    if (storedConnectivity == 6 || storedConnectivity == 18 || storedConnectivity == 26) {
        _skeletonConnectivity = storedConnectivity;
    }
    _skeletonSliceOrientation = std::clamp(settings.value(QStringLiteral("skeleton_slice_orientation"), _skeletonSliceOrientation).toInt(), 0, 1);
    _skeletonChunkSize = std::clamp(settings.value(QStringLiteral("skeleton_chunk_size"), _skeletonChunkSize).toInt(), 32, 256);
    _skeletonSearchRadius = std::clamp(settings.value(QStringLiteral("skeleton_search_radius"), _skeletonSearchRadius).toInt(), 1, 100);

    int storedGrowthSteps = settings.value(segmentation::GROWTH_STEPS, _growthSteps).toInt();
    storedGrowthSteps = std::clamp(storedGrowthSteps, 0, 1024);
    _tracerGrowthSteps = settings
                             .value(QStringLiteral("growth_steps_tracer"),
                                    std::max(1, storedGrowthSteps))
                             .toInt();
    _tracerGrowthSteps = std::clamp(_tracerGrowthSteps, 1, 1024);
    applyGrowthSteps(storedGrowthSteps, false, false);
    _growthDirectionMask = normalizeGrowthDirectionMask(
        settings.value(segmentation::GROWTH_DIRECTION_MASK, kGrowDirAllMask).toInt());
    _growthKeybindsEnabled = settings.value(segmentation::GROWTH_KEYBINDS_ENABLED,
                                            segmentation::GROWTH_KEYBINDS_ENABLED_DEFAULT).toBool();

    QVariantList serialized = settings.value(segmentation::DIRECTION_FIELDS, QVariantList{}).toList();
    _directionFields.clear();
    for (const QVariant& entry : serialized) {
        const QVariantMap map = entry.toMap();
        SegmentationDirectionFieldConfig config;
        config.path = map.value(QStringLiteral("path")).toString();
        config.orientation = segmentationDirectionFieldOrientationFromInt(
            map.value(QStringLiteral("orientation"), 0).toInt());
        config.scale = map.value(QStringLiteral("scale"), 0).toInt();
        config.weight = map.value(QStringLiteral("weight"), 1.0).toDouble();
        if (config.isValid()) {
            _directionFields.push_back(std::move(config));
        }
    }

    _correctionsEnabled = settings.value(segmentation::CORRECTIONS_ENABLED, segmentation::CORRECTIONS_ENABLED_DEFAULT).toBool();
    _correctionsZRangeEnabled = settings.value(segmentation::CORRECTIONS_Z_RANGE_ENABLED, segmentation::CORRECTIONS_Z_RANGE_ENABLED_DEFAULT).toBool();
    _correctionsZMin = settings.value(segmentation::CORRECTIONS_Z_MIN, segmentation::CORRECTIONS_Z_MIN_DEFAULT).toInt();
   _correctionsZMax = settings.value(segmentation::CORRECTIONS_Z_MAX, _correctionsZMin).toInt();
    if (_correctionsZMax < _correctionsZMin) {
        _correctionsZMax = _correctionsZMin;
    }

    _customParamsText = settings.value(segmentation::CUSTOM_PARAMS_TEXT, QString()).toString();
    _customParamsProfile = settings.value(QStringLiteral("custom_params_profile"), _customParamsProfile).toString();
    if (_customParamsProfile != QStringLiteral("custom") &&
        _customParamsProfile != QStringLiteral("default") &&
        _customParamsProfile != QStringLiteral("robust")) {
        _customParamsProfile = QStringLiteral("custom");
    }
    validateCustomParamsText();

    _normal3dSelectedPath = settings.value(QStringLiteral("normal3d_selected_path"), QString()).toString();

    // Apply profile behavior (auto-fill + read-only) after restoring.
    if (_customParamsProfile != QStringLiteral("custom")) {
        applyCustomParamsProfile(_customParamsProfile, /*persist=*/false, /*fromUi=*/false);
    }

    _approvalBrushRadius = settings.value(segmentation::APPROVAL_BRUSH_RADIUS, _approvalBrushRadius).toFloat();
    _approvalBrushRadius = std::clamp(_approvalBrushRadius, 1.0f, 1000.0f);
    _approvalBrushDepth = settings.value(segmentation::APPROVAL_BRUSH_DEPTH, _approvalBrushDepth).toFloat();
    _approvalBrushDepth = std::clamp(_approvalBrushDepth, 1.0f, 500.0f);
    // Don't restore approval mask show/edit states - user must explicitly enable each session

    _approvalMaskOpacity = settings.value(segmentation::APPROVAL_MASK_OPACITY, _approvalMaskOpacity).toInt();
    _approvalMaskOpacity = std::clamp(_approvalMaskOpacity, 0, 100);
    const QString colorName = settings.value(segmentation::APPROVAL_BRUSH_COLOR, _approvalBrushColor.name()).toString();
    if (QColor::isValidColorName(colorName)) {
        _approvalBrushColor = QColor::fromString(colorName);
    }
    _showApprovalMask = settings.value(segmentation::SHOW_APPROVAL_MASK, _showApprovalMask).toBool();
    // Don't restore edit states - user must explicitly enable editing each session

    // Neural tracer settings
    _neuralTracerEnabled = settings.value(QStringLiteral("neural_tracer_enabled"), false).toBool();
    _neuralCheckpointPath = settings.value(QStringLiteral("neural_checkpoint_path"), QString()).toString();
    _neuralPythonPath = settings.value(QStringLiteral("neural_python_path"), QString()).toString();
    _neuralVolumeScale = settings.value(QStringLiteral("neural_volume_scale"), 0).toInt();
    _neuralVolumeScale = std::clamp(_neuralVolumeScale, 0, 5);
    _neuralBatchSize = settings.value(QStringLiteral("neural_batch_size"), 4).toInt();
    _neuralBatchSize = std::clamp(_neuralBatchSize, 1, 64);

    // Cell reoptimization settings
    _cellReoptMaxSteps = settings.value(QStringLiteral("cell_reopt_max_steps"), _cellReoptMaxSteps).toInt();
    _cellReoptMaxSteps = std::clamp(_cellReoptMaxSteps, 10, 10000);
    _cellReoptMaxPoints = settings.value(QStringLiteral("cell_reopt_max_points"), _cellReoptMaxPoints).toInt();
    _cellReoptMaxPoints = std::clamp(_cellReoptMaxPoints, 3, 200);
    _cellReoptMinSpacing = settings.value(QStringLiteral("cell_reopt_min_spacing"), static_cast<double>(_cellReoptMinSpacing)).toFloat();
    _cellReoptMinSpacing = std::clamp(_cellReoptMinSpacing, 1.0f, 50.0f);
    _cellReoptPerimeterOffset = settings.value(QStringLiteral("cell_reopt_perimeter_offset"), static_cast<double>(_cellReoptPerimeterOffset)).toFloat();
    _cellReoptPerimeterOffset = std::clamp(_cellReoptPerimeterOffset, -50.0f, 50.0f);
    // Don't restore cell reopt mode - user must explicitly enable each session

    const bool editingExpanded = settings.value(segmentation::GROUP_EDITING_EXPANDED, segmentation::GROUP_EDITING_EXPANDED_DEFAULT).toBool();
    const bool dragExpanded = settings.value(segmentation::GROUP_DRAG_EXPANDED, segmentation::GROUP_DRAG_EXPANDED_DEFAULT).toBool();
    const bool lineExpanded = settings.value(segmentation::GROUP_LINE_EXPANDED, segmentation::GROUP_LINE_EXPANDED_DEFAULT).toBool();
    const bool pushPullExpanded = settings.value(segmentation::GROUP_PUSH_PULL_EXPANDED, segmentation::GROUP_PUSH_PULL_EXPANDED_DEFAULT).toBool();
    const bool directionExpanded = settings.value(segmentation::GROUP_DIRECTION_FIELD_EXPANDED, segmentation::GROUP_DIRECTION_FIELD_EXPANDED_DEFAULT).toBool();

    if (_groupEditing) {
        _groupEditing->setExpanded(editingExpanded);
    }
    if (_groupDrag) {
        _groupDrag->setExpanded(dragExpanded);
    }
    if (_groupLine) {
        _groupLine->setExpanded(lineExpanded);
    }
    if (_groupPushPull) {
        _groupPushPull->setExpanded(pushPullExpanded);
    }
    if (_groupDirectionField) {
        _groupDirectionField->setExpanded(directionExpanded);
    }

    const bool neuralExpanded = settings.value(QStringLiteral("group_neural_tracer_expanded"), false).toBool();
    if (_groupNeuralTracer) {
        _groupNeuralTracer->setExpanded(neuralExpanded);
    }

    // Sync neural tracer UI
    if (_chkNeuralTracerEnabled) {
        const QSignalBlocker blocker(_chkNeuralTracerEnabled);
        _chkNeuralTracerEnabled->setChecked(_neuralTracerEnabled);
    }
    if (_neuralCheckpointEdit) {
        const QSignalBlocker blocker(_neuralCheckpointEdit);
        _neuralCheckpointEdit->setText(_neuralCheckpointPath);
    }
    if (_neuralPythonEdit) {
        const QSignalBlocker blocker(_neuralPythonEdit);
        _neuralPythonEdit->setText(_neuralPythonPath);
    }
    if (_comboNeuralVolumeScale) {
        const QSignalBlocker blocker(_comboNeuralVolumeScale);
        int idx = _comboNeuralVolumeScale->findData(_neuralVolumeScale);
        if (idx >= 0) {
            _comboNeuralVolumeScale->setCurrentIndex(idx);
        }
    }
    if (_spinNeuralBatchSize) {
        const QSignalBlocker blocker(_spinNeuralBatchSize);
        _spinNeuralBatchSize->setValue(_neuralBatchSize);
    }

    settings.endGroup();
    _restoringSettings = false;
}

void SegmentationWidget::writeSetting(const QString& key, const QVariant& value)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.beginGroup(settingsGroup());
    settings.setValue(key, value);
    settings.endGroup();
}
