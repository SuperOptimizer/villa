/**
 * @file SegmentationWidgetApprovalMask.cpp
 * @brief Approval mask and cell reoptimization settings extracted from SegmentationWidget
 *
 * This file contains methods for managing approval mask display and editing,
 * brush settings, and cell reoptimization mode.
 * Extracted from SegmentationWidget.cpp to improve parallel compilation.
 */

#include "SegmentationWidget.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>

#include <algorithm>
#include <cmath>

void SegmentationWidget::setShowHoverMarker(bool enabled)
{
    if (_showHoverMarker == enabled) {
        return;
    }
    _showHoverMarker = enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("show_hover_marker"), _showHoverMarker);
        emit hoverMarkerToggled(_showHoverMarker);
    }
    if (_chkShowHoverMarker) {
        const QSignalBlocker blocker(_chkShowHoverMarker);
        _chkShowHoverMarker->setChecked(_showHoverMarker);
    }
}

void SegmentationWidget::setShowApprovalMask(bool enabled)
{
    if (_showApprovalMask == enabled) {
        return;
    }
    _showApprovalMask = enabled;
    qInfo() << "SegmentationWidget: Show approval mask changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("show_approval_mask"), _showApprovalMask);
        qInfo() << "  Emitting showApprovalMaskChanged signal";
        emit showApprovalMaskChanged(_showApprovalMask);
    }
    if (_chkShowApprovalMask) {
        const QSignalBlocker blocker(_chkShowApprovalMask);
        _chkShowApprovalMask->setChecked(_showApprovalMask);
    }
    syncUiState();
}

void SegmentationWidget::setEditApprovedMask(bool enabled)
{
    if (_editApprovedMask == enabled) {
        return;
    }
    _editApprovedMask = enabled;
    qInfo() << "SegmentationWidget: Edit approved mask changed to:" << enabled;

    // Mutual exclusion: if enabling approved, disable unapproved
    if (enabled && _editUnapprovedMask) {
        setEditUnapprovedMask(false);
    }

    if (!_restoringSettings) {
        writeSetting(QStringLiteral("edit_approved_mask"), _editApprovedMask);
        qInfo() << "  Emitting editApprovedMaskChanged signal";
        emit editApprovedMaskChanged(_editApprovedMask);
    }
    if (_chkEditApprovedMask) {
        const QSignalBlocker blocker(_chkEditApprovedMask);
        _chkEditApprovedMask->setChecked(_editApprovedMask);
    }
    syncUiState();
}

void SegmentationWidget::setEditUnapprovedMask(bool enabled)
{
    if (_editUnapprovedMask == enabled) {
        return;
    }
    _editUnapprovedMask = enabled;
    qInfo() << "SegmentationWidget: Edit unapproved mask changed to:" << enabled;

    // Mutual exclusion: if enabling unapproved, disable approved
    if (enabled && _editApprovedMask) {
        setEditApprovedMask(false);
    }

    if (!_restoringSettings) {
        writeSetting(QStringLiteral("edit_unapproved_mask"), _editUnapprovedMask);
        qInfo() << "  Emitting editUnapprovedMaskChanged signal";
        emit editUnapprovedMaskChanged(_editUnapprovedMask);
    }
    if (_chkEditUnapprovedMask) {
        const QSignalBlocker blocker(_chkEditUnapprovedMask);
        _chkEditUnapprovedMask->setChecked(_editUnapprovedMask);
    }
    syncUiState();
}

void SegmentationWidget::setApprovalBrushRadius(float radius)
{
    const float sanitized = std::clamp(radius, 1.0f, 1000.0f);
    if (std::abs(_approvalBrushRadius - sanitized) < 1e-4f) {
        return;
    }
    _approvalBrushRadius = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_radius"), _approvalBrushRadius);
        emit approvalBrushRadiusChanged(_approvalBrushRadius);
    }
    if (_spinApprovalBrushRadius) {
        const QSignalBlocker blocker(_spinApprovalBrushRadius);
        _spinApprovalBrushRadius->setValue(static_cast<double>(_approvalBrushRadius));
    }
}

void SegmentationWidget::setApprovalBrushDepth(float depth)
{
    const float sanitized = std::clamp(depth, 1.0f, 500.0f);
    if (std::abs(_approvalBrushDepth - sanitized) < 1e-4f) {
        return;
    }
    _approvalBrushDepth = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_depth"), _approvalBrushDepth);
        emit approvalBrushDepthChanged(_approvalBrushDepth);
    }
    if (_spinApprovalBrushDepth) {
        const QSignalBlocker blocker(_spinApprovalBrushDepth);
        _spinApprovalBrushDepth->setValue(static_cast<double>(_approvalBrushDepth));
    }
}

void SegmentationWidget::setApprovalMaskOpacity(int opacity)
{
    const int sanitized = std::clamp(opacity, 0, 100);
    if (_approvalMaskOpacity == sanitized) {
        return;
    }
    _approvalMaskOpacity = sanitized;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_mask_opacity"), _approvalMaskOpacity);
        emit approvalMaskOpacityChanged(_approvalMaskOpacity);
    }
    if (_sliderApprovalMaskOpacity) {
        const QSignalBlocker blocker(_sliderApprovalMaskOpacity);
        _sliderApprovalMaskOpacity->setValue(_approvalMaskOpacity);
    }
    if (_lblApprovalMaskOpacity) {
        _lblApprovalMaskOpacity->setText(QString::number(_approvalMaskOpacity) + QStringLiteral("%"));
    }
}

void SegmentationWidget::setApprovalBrushColor(const QColor& color)
{
    if (!color.isValid() || _approvalBrushColor == color) {
        return;
    }
    _approvalBrushColor = color;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("approval_brush_color"), _approvalBrushColor.name());
        emit approvalBrushColorChanged(_approvalBrushColor);
    }
    if (_btnApprovalColor) {
        _btnApprovalColor->setStyleSheet(
            QStringLiteral("background-color: %1; border: 1px solid #888;").arg(_approvalBrushColor.name()));
    }
}

void SegmentationWidget::setCellReoptMode(bool enabled)
{
    if (_cellReoptMode == enabled) {
        return;
    }
    _cellReoptMode = enabled;
    qInfo() << "SegmentationWidget: Cell reoptimization mode changed to:" << enabled;
    if (!_restoringSettings) {
        writeSetting(QStringLiteral("cell_reopt_mode"), _cellReoptMode);
        emit cellReoptModeChanged(_cellReoptMode);
    }
    if (_chkCellReoptMode) {
        const QSignalBlocker blocker(_chkCellReoptMode);
        _chkCellReoptMode->setChecked(_cellReoptMode);
    }
    syncUiState();
}

void SegmentationWidget::setCellReoptCollections(const QVector<QPair<uint64_t, QString>>& collections)
{
    if (!_comboCellReoptCollection) {
        return;
    }

    // Remember current selection
    uint64_t currentId = 0;
    if (_comboCellReoptCollection->currentIndex() >= 0) {
        currentId = _comboCellReoptCollection->currentData().toULongLong();
    }

    const QSignalBlocker blocker(_comboCellReoptCollection);
    _comboCellReoptCollection->clear();

    int indexToSelect = -1;
    for (int i = 0; i < collections.size(); ++i) {
        const auto& [id, name] = collections[i];
        _comboCellReoptCollection->addItem(name, QVariant::fromValue(id));
        if (id == currentId) {
            indexToSelect = i;
        }
    }

    // Restore selection if possible, otherwise select first item
    if (indexToSelect >= 0) {
        _comboCellReoptCollection->setCurrentIndex(indexToSelect);
    } else if (_comboCellReoptCollection->count() > 0) {
        _comboCellReoptCollection->setCurrentIndex(0);
    }

    // Update run button state - need a collection selected to run
    syncUiState();
}
