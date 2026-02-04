/**
 * @file SegmentationWidgetToolSettings.cpp
 * @brief Tool settings management extracted from SegmentationWidget
 *
 * This file contains methods for managing tool-specific settings like
 * drag/line/push-pull radius/sigma, alpha push-pull configuration,
 * and smoothing parameters.
 * Extracted from SegmentationWidget.cpp to improve parallel compilation.
 */

#include "SegmentationWidget.hpp"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QWidget>

#include <algorithm>
#include <cmath>

namespace
{

constexpr float kFloatEpsilon = 1e-4f;
constexpr float kAlphaOpacityScale = 255.0f;

bool nearlyEqual(float lhs, float rhs)
{
    return std::fabs(lhs - rhs) < kFloatEpsilon;
}

float displayOpacityToNormalized(double displayValue)
{
    return static_cast<float>(displayValue / kAlphaOpacityScale);
}

double normalizedOpacityToDisplay(float normalizedValue)
{
    return static_cast<double>(normalizedValue * kAlphaOpacityScale);
}

AlphaPushPullConfig sanitizeAlphaConfig(const AlphaPushPullConfig& config)
{
    AlphaPushPullConfig sanitized = config;

    sanitized.start = std::clamp(sanitized.start, -128.0f, 128.0f);
    sanitized.stop = std::clamp(sanitized.stop, -128.0f, 128.0f);
    if (sanitized.start > sanitized.stop) {
        std::swap(sanitized.start, sanitized.stop);
    }

    const float minStep = 0.05f;
    const float maxStep = 20.0f;
    const float magnitude = std::clamp(std::fabs(sanitized.step), minStep, maxStep);
    sanitized.step = (sanitized.step < 0.0f) ? -magnitude : magnitude;

    sanitized.low = std::clamp(sanitized.low, 0.0f, 1.0f);
    sanitized.high = std::clamp(sanitized.high, 0.0f, 1.0f);
    if (sanitized.high <= sanitized.low + 0.01f) {
        sanitized.high = std::min(1.0f, sanitized.low + 0.05f);
    }

    sanitized.borderOffset = std::clamp(sanitized.borderOffset, -20.0f, 20.0f);
    sanitized.blurRadius = std::clamp(sanitized.blurRadius, 0, 15);
    sanitized.perVertexLimit = std::clamp(sanitized.perVertexLimit, 0.0f, 128.0f);

    return sanitized;
}

}  // namespace

void SegmentationWidget::setDragRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _dragRadiusSteps) < 1e-4f) {
        return;
    }
    _dragRadiusSteps = clamped;
    writeSetting(QStringLiteral("drag_radius_steps"), _dragRadiusSteps);
    if (_spinDragRadius) {
        const QSignalBlocker blocker(_spinDragRadius);
        _spinDragRadius->setValue(static_cast<double>(_dragRadiusSteps));
    }
}

void SegmentationWidget::setDragSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _dragSigmaSteps) < 1e-4f) {
        return;
    }
    _dragSigmaSteps = clamped;
    writeSetting(QStringLiteral("drag_sigma_steps"), _dragSigmaSteps);
    if (_spinDragSigma) {
        const QSignalBlocker blocker(_spinDragSigma);
        _spinDragSigma->setValue(static_cast<double>(_dragSigmaSteps));
    }
}

void SegmentationWidget::setLineRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _lineRadiusSteps) < 1e-4f) {
        return;
    }
    _lineRadiusSteps = clamped;
    writeSetting(QStringLiteral("line_radius_steps"), _lineRadiusSteps);
    if (_spinLineRadius) {
        const QSignalBlocker blocker(_spinLineRadius);
        _spinLineRadius->setValue(static_cast<double>(_lineRadiusSteps));
    }
}

void SegmentationWidget::setLineSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _lineSigmaSteps) < 1e-4f) {
        return;
    }
    _lineSigmaSteps = clamped;
    writeSetting(QStringLiteral("line_sigma_steps"), _lineSigmaSteps);
    if (_spinLineSigma) {
        const QSignalBlocker blocker(_spinLineSigma);
        _spinLineSigma->setValue(static_cast<double>(_lineSigmaSteps));
    }
}

void SegmentationWidget::setPushPullRadius(float value)
{
    const float clamped = std::clamp(value, 0.25f, 128.0f);
    if (std::fabs(clamped - _pushPullRadiusSteps) < 1e-4f) {
        return;
    }
    _pushPullRadiusSteps = clamped;
    writeSetting(QStringLiteral("push_pull_radius_steps"), _pushPullRadiusSteps);
    if (_spinPushPullRadius) {
        const QSignalBlocker blocker(_spinPushPullRadius);
        _spinPushPullRadius->setValue(static_cast<double>(_pushPullRadiusSteps));
    }
}

void SegmentationWidget::setPushPullSigma(float value)
{
    const float clamped = std::clamp(value, 0.05f, 64.0f);
    if (std::fabs(clamped - _pushPullSigmaSteps) < 1e-4f) {
        return;
    }
    _pushPullSigmaSteps = clamped;
    writeSetting(QStringLiteral("push_pull_sigma_steps"), _pushPullSigmaSteps);
    if (_spinPushPullSigma) {
        const QSignalBlocker blocker(_spinPushPullSigma);
        _spinPushPullSigma->setValue(static_cast<double>(_pushPullSigmaSteps));
    }
}

void SegmentationWidget::setPushPullStep(float value)
{
    const float clamped = std::clamp(value, 0.05f, 40.0f);
    if (std::fabs(clamped - _pushPullStep) < 1e-4f) {
        return;
    }
    _pushPullStep = clamped;
    writeSetting(QStringLiteral("push_pull_step"), _pushPullStep);
    if (_spinPushPullStep) {
        const QSignalBlocker blocker(_spinPushPullStep);
        _spinPushPullStep->setValue(static_cast<double>(_pushPullStep));
    }
}

AlphaPushPullConfig SegmentationWidget::alphaPushPullConfig() const
{
    return _alphaPushPullConfig;
}

void SegmentationWidget::setAlphaPushPullConfig(const AlphaPushPullConfig& config)
{
    applyAlphaPushPullConfig(config, false);
}

void SegmentationWidget::applyAlphaPushPullConfig(const AlphaPushPullConfig& config,
                                                  bool emitSignal,
                                                  bool persist)
{
    AlphaPushPullConfig sanitized = sanitizeAlphaConfig(config);

    const bool changed = !nearlyEqual(sanitized.start, _alphaPushPullConfig.start) ||
                         !nearlyEqual(sanitized.stop, _alphaPushPullConfig.stop) ||
                         !nearlyEqual(sanitized.step, _alphaPushPullConfig.step) ||
                         !nearlyEqual(sanitized.low, _alphaPushPullConfig.low) ||
                         !nearlyEqual(sanitized.high, _alphaPushPullConfig.high) ||
                         !nearlyEqual(sanitized.borderOffset, _alphaPushPullConfig.borderOffset) ||
                         sanitized.blurRadius != _alphaPushPullConfig.blurRadius ||
                         !nearlyEqual(sanitized.perVertexLimit, _alphaPushPullConfig.perVertexLimit) ||
                         sanitized.perVertex != _alphaPushPullConfig.perVertex;

    if (changed) {
        _alphaPushPullConfig = sanitized;
        if (persist) {
            writeSetting(QStringLiteral("push_pull_alpha_start"), _alphaPushPullConfig.start);
            writeSetting(QStringLiteral("push_pull_alpha_stop"), _alphaPushPullConfig.stop);
            writeSetting(QStringLiteral("push_pull_alpha_step"), _alphaPushPullConfig.step);
            writeSetting(QStringLiteral("push_pull_alpha_low"), _alphaPushPullConfig.low);
            writeSetting(QStringLiteral("push_pull_alpha_high"), _alphaPushPullConfig.high);
            writeSetting(QStringLiteral("push_pull_alpha_border"), _alphaPushPullConfig.borderOffset);
            writeSetting(QStringLiteral("push_pull_alpha_radius"), _alphaPushPullConfig.blurRadius);
            writeSetting(QStringLiteral("push_pull_alpha_limit"), _alphaPushPullConfig.perVertexLimit);
            writeSetting(QStringLiteral("push_pull_alpha_per_vertex"), _alphaPushPullConfig.perVertex);
        }
    }

    const bool editingActive = _editingEnabled && !_growthInProgress;

    if (_spinAlphaStart) {
        const QSignalBlocker blocker(_spinAlphaStart);
        _spinAlphaStart->setValue(static_cast<double>(_alphaPushPullConfig.start));
        _spinAlphaStart->setEnabled(editingActive);
    }
    if (_spinAlphaStop) {
        const QSignalBlocker blocker(_spinAlphaStop);
        _spinAlphaStop->setValue(static_cast<double>(_alphaPushPullConfig.stop));
        _spinAlphaStop->setEnabled(editingActive);
    }
    if (_spinAlphaStep) {
        const QSignalBlocker blocker(_spinAlphaStep);
        _spinAlphaStep->setValue(static_cast<double>(_alphaPushPullConfig.step));
        _spinAlphaStep->setEnabled(editingActive);
    }
    if (_spinAlphaLow) {
        const QSignalBlocker blocker(_spinAlphaLow);
        _spinAlphaLow->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.low));
        _spinAlphaLow->setEnabled(editingActive);
    }
    if (_spinAlphaHigh) {
        const QSignalBlocker blocker(_spinAlphaHigh);
        _spinAlphaHigh->setValue(normalizedOpacityToDisplay(_alphaPushPullConfig.high));
        _spinAlphaHigh->setEnabled(editingActive);
    }
    if (_spinAlphaBorder) {
        const QSignalBlocker blocker(_spinAlphaBorder);
        _spinAlphaBorder->setValue(static_cast<double>(_alphaPushPullConfig.borderOffset));
        _spinAlphaBorder->setEnabled(editingActive);
    }
    if (_spinAlphaBlurRadius) {
        const QSignalBlocker blocker(_spinAlphaBlurRadius);
        _spinAlphaBlurRadius->setValue(_alphaPushPullConfig.blurRadius);
        _spinAlphaBlurRadius->setEnabled(editingActive);
    }
    if (_spinAlphaPerVertexLimit) {
        const QSignalBlocker blocker(_spinAlphaPerVertexLimit);
        _spinAlphaPerVertexLimit->setValue(static_cast<double>(_alphaPushPullConfig.perVertexLimit));
        _spinAlphaPerVertexLimit->setEnabled(editingActive);
    }
    if (_chkAlphaPerVertex) {
        const QSignalBlocker blocker(_chkAlphaPerVertex);
        _chkAlphaPerVertex->setChecked(_alphaPushPullConfig.perVertex);
        _chkAlphaPerVertex->setEnabled(editingActive);
    }
    if (_alphaPushPullPanel) {
        _alphaPushPullPanel->setEnabled(editingActive);
    }

    if (emitSignal && changed) {
        emit alphaPushPullConfigChanged();
    }
}

void SegmentationWidget::setSmoothingStrength(float value)
{
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    if (std::fabs(clamped - _smoothStrength) < 1e-4f) {
        return;
    }
    _smoothStrength = clamped;
    writeSetting(QStringLiteral("smooth_strength"), _smoothStrength);
    if (_spinSmoothStrength) {
        const QSignalBlocker blocker(_spinSmoothStrength);
        _spinSmoothStrength->setValue(static_cast<double>(_smoothStrength));
    }
}

void SegmentationWidget::setSmoothingIterations(int value)
{
    const int clamped = std::clamp(value, 1, 25);
    if (_smoothIterations == clamped) {
        return;
    }
    _smoothIterations = clamped;
    writeSetting(QStringLiteral("smooth_iterations"), _smoothIterations);
    if (_spinSmoothIterations) {
        const QSignalBlocker blocker(_spinSmoothIterations);
        _spinSmoothIterations->setValue(_smoothIterations);
    }
}
