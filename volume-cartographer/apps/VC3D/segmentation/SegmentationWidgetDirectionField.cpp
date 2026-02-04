/**
 * @file SegmentationWidgetDirectionField.cpp
 * @brief Direction field and growth direction management extracted from SegmentationWidget
 *
 * This file contains methods for managing direction field configurations,
 * growth direction masks, and related UI state updates.
 * Extracted from SegmentationWidget.cpp to improve parallel compilation.
 */

#include "SegmentationWidget.hpp"

#include "elements/CollapsibleSettingsGroup.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSpinBox>

#include <algorithm>
#include <cmath>

namespace
{

constexpr int kGrowDirUpBit = 1 << 0;
constexpr int kGrowDirDownBit = 1 << 1;
constexpr int kGrowDirLeftBit = 1 << 2;
constexpr int kGrowDirRightBit = 1 << 3;
constexpr int kGrowDirAllMask = kGrowDirUpBit | kGrowDirDownBit | kGrowDirLeftBit | kGrowDirRightBit;
constexpr int kCompactDirectionFieldRowLimit = 3;

}  // namespace

SegmentationDirectionFieldConfig SegmentationWidget::buildDirectionFieldDraft() const
{
    SegmentationDirectionFieldConfig config;
    config.path = _directionFieldPath.trimmed();
    config.orientation = _directionFieldOrientation;
    config.scale = std::clamp(_directionFieldScale, 0, 5);
    config.weight = std::clamp(_directionFieldWeight, 0.0, 10.0);
    return config;
}

void SegmentationWidget::refreshDirectionFieldList()
{
    if (!_directionFieldList) {
        return;
    }
    const QSignalBlocker blocker(_directionFieldList);
    const int previousRow = _directionFieldList->currentRow();
    _directionFieldList->clear();

    for (const auto& config : _directionFields) {
        QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
        const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
        const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                     .arg(config.path,
                                          orientationLabel,
                                          QString::number(std::clamp(config.scale, 0, 5)),
                                          weightText);
        auto* item = new QListWidgetItem(itemText, _directionFieldList);
        item->setToolTip(config.path);
    }

    if (!_directionFields.empty()) {
        const int clampedRow = std::clamp(previousRow, 0, static_cast<int>(_directionFields.size()) - 1);
        _directionFieldList->setCurrentRow(clampedRow);
    }
    if (_directionFieldRemoveButton) {
        _directionFieldRemoveButton->setEnabled(_editingEnabled && !_directionFields.empty() && _directionFieldList->currentRow() >= 0);
    }

    updateDirectionFieldFormFromSelection(_directionFieldList->currentRow());
    updateDirectionFieldListGeometry();
}

void SegmentationWidget::updateDirectionFieldFormFromSelection(int row)
{
    const bool previousUpdating = _updatingDirectionFieldForm;
    _updatingDirectionFieldForm = true;

    if (row >= 0 && row < static_cast<int>(_directionFields.size())) {
        const auto& config = _directionFields[static_cast<std::size_t>(row)];
        _directionFieldPath = config.path;
        _directionFieldOrientation = config.orientation;
        _directionFieldScale = config.scale;
        _directionFieldWeight = config.weight;
    }

    if (_directionFieldPathEdit) {
        const QSignalBlocker blocker(_directionFieldPathEdit);
        _directionFieldPathEdit->setText(_directionFieldPath);
    }
    if (_comboDirectionFieldOrientation) {
        const QSignalBlocker blocker(_comboDirectionFieldOrientation);
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(_directionFieldOrientation));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        const QSignalBlocker blocker(_comboDirectionFieldScale);
        int idx = _comboDirectionFieldScale->findData(_directionFieldScale);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        const QSignalBlocker blocker(_spinDirectionFieldWeight);
        _spinDirectionFieldWeight->setValue(_directionFieldWeight);
    }

    _updatingDirectionFieldForm = previousUpdating;
}

void SegmentationWidget::applyDirectionFieldDraftToSelection(int row)
{
    if (row < 0 || row >= static_cast<int>(_directionFields.size())) {
        return;
    }

    auto config = buildDirectionFieldDraft();
    if (!config.isValid()) {
        return;
    }

    auto& target = _directionFields[static_cast<std::size_t>(row)];
    if (target.path == config.path &&
        target.orientation == config.orientation &&
        target.scale == config.scale &&
        std::abs(target.weight - config.weight) < 1e-4) {
        return;
    }

    target = std::move(config);
    updateDirectionFieldListItem(row);
    persistDirectionFields();
}

void SegmentationWidget::updateDirectionFieldListItem(int row)
{
    if (!_directionFieldList) {
        return;
    }
    if (row < 0 || row >= _directionFieldList->count()) {
        return;
    }
    if (row >= static_cast<int>(_directionFields.size())) {
        return;
    }

    const auto& config = _directionFields[static_cast<std::size_t>(row)];
    QString orientationLabel = segmentationDirectionFieldOrientationKey(config.orientation);
    const QString weightText = QString::number(std::clamp(config.weight, 0.0, 10.0), 'f', 2);
    const QString itemText = tr("%1 — %2 (scale %3, weight %4)")
                                 .arg(config.path,
                                      orientationLabel,
                                      QString::number(std::clamp(config.scale, 0, 5)),
                                      weightText);

    if (auto* item = _directionFieldList->item(row)) {
        item->setText(itemText);
        item->setToolTip(config.path);
    }
}

void SegmentationWidget::updateDirectionFieldListGeometry()
{
    if (!_directionFieldList) {
        return;
    }

    auto policy = _directionFieldList->sizePolicy();
    const int itemCount = _directionFieldList->count();

    if (itemCount <= kCompactDirectionFieldRowLimit) {
        const int sampleRowHeight = _directionFieldList->sizeHintForRow(0);
        const int rowHeight = sampleRowHeight > 0 ? sampleRowHeight : _directionFieldList->fontMetrics().height() + 8;
        const int visibleRows = std::max(1, itemCount);
        const int frameHeight = 2 * _directionFieldList->frameWidth();
        const auto* hScroll = _directionFieldList->horizontalScrollBar();
        const int scrollHeight = (hScroll && hScroll->isVisible()) ? hScroll->sizeHint().height() : 0;
        const int targetHeight = rowHeight * visibleRows + frameHeight + scrollHeight;

        policy.setVerticalPolicy(QSizePolicy::Fixed);
        policy.setVerticalStretch(0);
        _directionFieldList->setSizePolicy(policy);
        _directionFieldList->setMinimumHeight(targetHeight);
        _directionFieldList->setMaximumHeight(targetHeight);
    } else {
        policy.setVerticalPolicy(QSizePolicy::Expanding);
        policy.setVerticalStretch(1);
        _directionFieldList->setSizePolicy(policy);
        _directionFieldList->setMinimumHeight(0);
        _directionFieldList->setMaximumHeight(QWIDGETSIZE_MAX);
    }

    _directionFieldList->updateGeometry();
}

void SegmentationWidget::clearDirectionFieldForm()
{
    // Clear the list selection
    if (_directionFieldList) {
        _directionFieldList->setCurrentRow(-1);
    }

    // Reset member variables to defaults
    _directionFieldPath.clear();
    _directionFieldOrientation = SegmentationDirectionFieldOrientation::Normal;
    _directionFieldScale = 0;
    _directionFieldWeight = 1.0;

    // Update the form fields to reflect the cleared state
    const bool previousUpdating = _updatingDirectionFieldForm;
    _updatingDirectionFieldForm = true;

    if (_directionFieldPathEdit) {
        _directionFieldPathEdit->clear();
    }
    if (_comboDirectionFieldOrientation) {
        int idx = _comboDirectionFieldOrientation->findData(static_cast<int>(SegmentationDirectionFieldOrientation::Normal));
        if (idx >= 0) {
            _comboDirectionFieldOrientation->setCurrentIndex(idx);
        }
    }
    if (_comboDirectionFieldScale) {
        int idx = _comboDirectionFieldScale->findData(0);
        if (idx >= 0) {
            _comboDirectionFieldScale->setCurrentIndex(idx);
        }
    }
    if (_spinDirectionFieldWeight) {
        _spinDirectionFieldWeight->setValue(1.0);
    }

    _updatingDirectionFieldForm = previousUpdating;

    // Update button states
    if (_directionFieldRemoveButton) {
        _directionFieldRemoveButton->setEnabled(false);
    }
}

void SegmentationWidget::persistDirectionFields()
{
    QVariantList serialized;
    serialized.reserve(static_cast<int>(_directionFields.size()));
    for (const auto& config : _directionFields) {
        QVariantMap map;
        map.insert(QStringLiteral("path"), config.path);
        map.insert(QStringLiteral("orientation"), static_cast<int>(config.orientation));
        map.insert(QStringLiteral("scale"), std::clamp(config.scale, 0, 5));
        map.insert(QStringLiteral("weight"), std::clamp(config.weight, 0.0, 10.0));
        serialized.push_back(map);
    }
    writeSetting(QStringLiteral("direction_fields"), serialized);
}

void SegmentationWidget::setGrowthDirectionMask(int mask)
{
    mask = normalizeGrowthDirectionMask(mask);
    if (_growthDirectionMask == mask) {
        return;
    }
    _growthDirectionMask = mask;
    writeSetting(QStringLiteral("growth_direction_mask"), _growthDirectionMask);
    applyGrowthDirectionMaskToUi();
}

void SegmentationWidget::updateGrowthDirectionMaskFromUi(QCheckBox* changedCheckbox)
{
    int mask = 0;
    if (_chkGrowthDirUp && _chkGrowthDirUp->isChecked()) {
        mask |= kGrowDirUpBit;
    }
    if (_chkGrowthDirDown && _chkGrowthDirDown->isChecked()) {
        mask |= kGrowDirDownBit;
    }
    if (_chkGrowthDirLeft && _chkGrowthDirLeft->isChecked()) {
        mask |= kGrowDirLeftBit;
    }
    if (_chkGrowthDirRight && _chkGrowthDirRight->isChecked()) {
        mask |= kGrowDirRightBit;
    }

    if (mask == 0) {
        if (changedCheckbox) {
            const QSignalBlocker blocker(changedCheckbox);
            changedCheckbox->setChecked(true);
        }
        mask = kGrowDirAllMask;
    }

    setGrowthDirectionMask(mask);
}

void SegmentationWidget::applyGrowthDirectionMaskToUi()
{
    if (_chkGrowthDirUp) {
        const QSignalBlocker blocker(_chkGrowthDirUp);
        _chkGrowthDirUp->setChecked((_growthDirectionMask & kGrowDirUpBit) != 0);
    }
    if (_chkGrowthDirDown) {
        const QSignalBlocker blocker(_chkGrowthDirDown);
        _chkGrowthDirDown->setChecked((_growthDirectionMask & kGrowDirDownBit) != 0);
    }
    if (_chkGrowthDirLeft) {
        const QSignalBlocker blocker(_chkGrowthDirLeft);
        _chkGrowthDirLeft->setChecked((_growthDirectionMask & kGrowDirLeftBit) != 0);
    }
    if (_chkGrowthDirRight) {
        const QSignalBlocker blocker(_chkGrowthDirRight);
        _chkGrowthDirRight->setChecked((_growthDirectionMask & kGrowDirRightBit) != 0);
    }
}

void SegmentationWidget::updateGrowthUiState()
{
    const bool enableGrowth = _editingEnabled && !_growthInProgress;
    if (_spinGrowthSteps) {
        _spinGrowthSteps->setEnabled(enableGrowth);
    }
    if (_btnGrow) {
        _btnGrow->setEnabled(enableGrowth);
    }
    if (_btnInpaint) {
        _btnInpaint->setEnabled(enableGrowth);
    }
    const bool enableDirCheckbox = enableGrowth;
    if (_chkGrowthDirUp) {
        _chkGrowthDirUp->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirDown) {
        _chkGrowthDirDown->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirLeft) {
        _chkGrowthDirLeft->setEnabled(enableDirCheckbox);
    }
    if (_chkGrowthDirRight) {
        _chkGrowthDirRight->setEnabled(enableDirCheckbox);
    }
    if (_directionFieldAddButton) {
        _directionFieldAddButton->setEnabled(_editingEnabled);
    }
    if (_directionFieldRemoveButton) {
        const bool hasSelection = _directionFieldList && _directionFieldList->currentRow() >= 0;
        _directionFieldRemoveButton->setEnabled(_editingEnabled && hasSelection);
    }
    if (_directionFieldList) {
        _directionFieldList->setEnabled(_editingEnabled);
    }

    const bool allowZRange = _editingEnabled && !_growthInProgress;
    if (_chkCorrectionsUseZRange) {
        _chkCorrectionsUseZRange->setEnabled(allowZRange);
    }
    if (_spinCorrectionsZMin) {
        _spinCorrectionsZMin->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (_spinCorrectionsZMax) {
        _spinCorrectionsZMax->setEnabled(allowZRange && _correctionsZRangeEnabled);
    }
    if (_groupCustomParams) {
        _groupCustomParams->setEnabled(_editingEnabled);
    }

    const bool allowCorrections = _editingEnabled && _correctionsEnabled && !_growthInProgress;
    if (_groupCorrections) {
        _groupCorrections->setEnabled(allowCorrections);
    }
    if (_comboCorrections) {
        const QSignalBlocker blocker(_comboCorrections);
        _comboCorrections->setEnabled(allowCorrections && _comboCorrections->count() > 0);
    }
    if (_btnCorrectionsNew) {
        _btnCorrectionsNew->setEnabled(_editingEnabled && !_growthInProgress);
    }
    if (_chkCorrectionsAnnotate) {
        _chkCorrectionsAnnotate->setEnabled(allowCorrections);
    }
}

void SegmentationWidget::triggerGrowthRequest(SegmentationGrowthDirection direction,
                                              int steps,
                                              bool inpaintOnly)
{
    if (!_editingEnabled || _growthInProgress) {
        return;
    }

    const SegmentationGrowthMethod method = inpaintOnly
        ? SegmentationGrowthMethod::Tracer
        : _growthMethod;

    const bool allowZeroSteps = inpaintOnly || method == SegmentationGrowthMethod::Corrections;
    const int minSteps = allowZeroSteps ? 0 : 1;
    const int clampedSteps = std::clamp(steps, minSteps, 1024);
    const int finalSteps = clampedSteps;

    emit growSurfaceRequested(method, direction, finalSteps, inpaintOnly);
}

int SegmentationWidget::normalizeGrowthDirectionMask(int mask)
{
    mask &= kGrowDirAllMask;
    if (mask == 0) {
        // If no directions are selected, enable all directions by default.
        // This ensures that growth is not unintentionally disabled.
        mask = kGrowDirAllMask;
    }
    return mask;
}
