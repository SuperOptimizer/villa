/**
 * @file SegmentationWidgetNeuralTracer.cpp
 * @brief Neural tracer and Normal3D settings extracted from SegmentationWidget
 *
 * This file contains methods for managing neural tracer configuration,
 * Normal3D zarr volume selection, and related UI updates.
 * Extracted from SegmentationWidget.cpp to improve parallel compilation.
 */

#include "SegmentationWidget.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QSignalBlocker>
#include <QSpinBox>

#include <algorithm>

void SegmentationWidget::updateNormal3dUi()
{
    if (!_lblNormal3d) {
        return;
    }

    const int count = _normal3dCandidates.size();
    const bool hasAny = count > 0;

    // Keep selection valid.
    if (hasAny) {
        if (_normal3dSelectedPath.isEmpty() || !_normal3dCandidates.contains(_normal3dSelectedPath)) {
            _normal3dSelectedPath = _normal3dCandidates.front();
        }
    } else {
        _normal3dSelectedPath.clear();
    }

    const bool showCombo = count > 1;
    if (_comboNormal3d) {
        _comboNormal3d->setVisible(showCombo);
        _comboNormal3d->setEnabled(_editingEnabled && hasAny);
        if (showCombo) {
            const QSignalBlocker blocker(_comboNormal3d);
            _comboNormal3d->clear();
            for (const QString& p : _normal3dCandidates) {
                _comboNormal3d->addItem(p, p);
            }
            const int idx = _comboNormal3d->findData(_normal3dSelectedPath);
            if (idx >= 0) {
                _comboNormal3d->setCurrentIndex(idx);
            }
        }
    }

    const QString icon = hasAny
        ? QStringLiteral("<span style=\"color:#2e7d32; font-size:16px;\">&#10003;</span>")
        : QStringLiteral("<span style=\"color:#c62828; font-size:16px;\">&#10007;</span>");

    QString message;
    if (!hasAny) {
        message = tr("Normal3D volume not found.");
    } else if (count == 1) {
        message = tr("Normal3D volume found.");
    } else {
        message = tr("Normal3D volumes found (%1). Select one:").arg(count);
    }

    QString tooltip = message;
    if (!_normal3dHint.isEmpty()) {
        tooltip.append(QStringLiteral("\n"));
        tooltip.append(_normal3dHint);
    }
    if (!_volumePackagePath.isEmpty()) {
        tooltip.append(QStringLiteral("\n"));
        tooltip.append(tr("Volume package: %1").arg(_volumePackagePath));
    }

    _lblNormal3d->setText(icon + QStringLiteral("&nbsp;") + message);
    _lblNormal3d->setToolTip(tooltip);
    _lblNormal3d->setAccessibleDescription(message);

    if (_editNormal3dPath) {
        const bool show = hasAny && !showCombo;
        _editNormal3dPath->setVisible(show);
        _editNormal3dPath->setText(_normal3dSelectedPath);
        _editNormal3dPath->setToolTip(_normal3dSelectedPath);
    }
}

void SegmentationWidget::setNormal3dZarrCandidates(const QStringList& candidates, const QString& hint)
{
    _normal3dCandidates = candidates;
    _normal3dHint = hint;
    syncUiState();
}

void SegmentationWidget::setNeuralTracerEnabled(bool enabled)
{
    if (_neuralTracerEnabled == enabled) {
        return;
    }
    _neuralTracerEnabled = enabled;
    writeSetting(QStringLiteral("neural_tracer_enabled"), _neuralTracerEnabled);

    if (_chkNeuralTracerEnabled) {
        const QSignalBlocker blocker(_chkNeuralTracerEnabled);
        _chkNeuralTracerEnabled->setChecked(enabled);
    }

    emit neuralTracerEnabledChanged(enabled);
}

void SegmentationWidget::setNeuralCheckpointPath(const QString& path)
{
    if (_neuralCheckpointPath == path) {
        return;
    }
    _neuralCheckpointPath = path;
    writeSetting(QStringLiteral("neural_checkpoint_path"), _neuralCheckpointPath);

    if (_neuralCheckpointEdit) {
        const QSignalBlocker blocker(_neuralCheckpointEdit);
        _neuralCheckpointEdit->setText(path);
    }
}

void SegmentationWidget::setNeuralPythonPath(const QString& path)
{
    if (_neuralPythonPath == path) {
        return;
    }
    _neuralPythonPath = path;
    writeSetting(QStringLiteral("neural_python_path"), _neuralPythonPath);

    if (_neuralPythonEdit) {
        const QSignalBlocker blocker(_neuralPythonEdit);
        _neuralPythonEdit->setText(path);
    }
}

void SegmentationWidget::setNeuralVolumeScale(int scale)
{
    scale = std::clamp(scale, 0, 5);
    if (_neuralVolumeScale == scale) {
        return;
    }
    _neuralVolumeScale = scale;
    writeSetting(QStringLiteral("neural_volume_scale"), _neuralVolumeScale);

    if (_comboNeuralVolumeScale) {
        const QSignalBlocker blocker(_comboNeuralVolumeScale);
        int idx = _comboNeuralVolumeScale->findData(scale);
        if (idx >= 0) {
            _comboNeuralVolumeScale->setCurrentIndex(idx);
        }
    }
}

void SegmentationWidget::setNeuralBatchSize(int size)
{
    size = std::clamp(size, 1, 64);
    if (_neuralBatchSize == size) {
        return;
    }
    _neuralBatchSize = size;
    writeSetting(QStringLiteral("neural_batch_size"), _neuralBatchSize);

    if (_spinNeuralBatchSize) {
        const QSignalBlocker blocker(_spinNeuralBatchSize);
        _spinNeuralBatchSize->setValue(size);
    }
}

void SegmentationWidget::setVolumeZarrPath(const QString& path)
{
    _volumeZarrPath = path;
}
