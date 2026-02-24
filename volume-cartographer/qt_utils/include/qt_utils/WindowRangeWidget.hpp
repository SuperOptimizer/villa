#pragma once

#include <QWidget>

class QSpinBox;

namespace qt_utils
{

class RangeSlider;

/// A composite widget that pairs a RangeSlider with two QSpinBox controls
/// (low and high), keeping them synchronized. Useful for windowing controls
/// (e.g. brightness/contrast, histogram range selection).
class WindowRangeWidget : public QWidget
{
    Q_OBJECT

public:
    explicit WindowRangeWidget(QWidget* parent = nullptr);

    /// Set the allowed range for both slider and spinboxes.
    void setRange(int minimum, int maximum);

    /// Set the current low/high values.
    void setWindowValues(int low, int high);

    /// Enable or disable all child controls.
    void setControlsEnabled(bool enabled);

    /// Set the minimum required separation between low and high values.
    void setMinimumSeparation(int separation);

    /// Return the current low value.
    [[nodiscard]] auto lowValue() const -> int;

    /// Return the current high value.
    [[nodiscard]] auto highValue() const -> int;

signals:
    /// Emitted when either the low or high value changes (from any source).
    void windowValuesChanged(int low, int high);

    /// Emitted when only the low value changes.
    void lowChanged(int value);

    /// Emitted when only the high value changes.
    void highChanged(int value);

private:
    void syncControls(int low, int high, bool emitSignal);

    RangeSlider* slider_{nullptr};
    QSpinBox* lowSpin_{nullptr};
    QSpinBox* highSpin_{nullptr};
    int minimum_{0};
    int maximum_{255};
    int minimumSeparation_{1};
    bool suppressSignals_{false};
};

}  // namespace qt_utils
