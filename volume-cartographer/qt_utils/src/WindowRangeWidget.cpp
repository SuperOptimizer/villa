#include "qt_utils/WindowRangeWidget.hpp"
#include "qt_utils/RangeSlider.hpp"

#include <QHBoxLayout>
#include <QSignalBlocker>
#include <QSpinBox>

#include <algorithm>

namespace qt_utils
{

WindowRangeWidget::WindowRangeWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    // Range slider
    slider_ = new RangeSlider(Qt::Horizontal, this);
    {
        QSignalBlocker blocker(slider_);
        slider_->setRange(minimum_, maximum_);
    }
    slider_->setMinimumSeparation(minimumSeparation_);
    slider_->setValues(minimum_, maximum_);

    connect(slider_, &RangeSlider::valuesChanged, this, [this](int low, int high) {
        if (suppressSignals_) {
            return;
        }
        syncControls(low, high, true);
    });

    // Low spinbox
    lowSpin_ = new QSpinBox(this);
    lowSpin_->setRange(minimum_, maximum_);
    lowSpin_->setValue(minimum_);
    lowSpin_->setSuffix(QStringLiteral(" L"));

    connect(lowSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        if (suppressSignals_) {
            return;
        }
        syncControls(value, slider_->highValue(), true);
    });

    // High spinbox
    highSpin_ = new QSpinBox(this);
    highSpin_->setRange(minimum_, maximum_);
    highSpin_->setValue(maximum_);
    highSpin_->setSuffix(QStringLiteral(" H"));

    connect(highSpin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        if (suppressSignals_) {
            return;
        }
        syncControls(slider_->lowValue(), value, true);
    });

    layout->addWidget(slider_, /*stretch=*/1);
    layout->addWidget(lowSpin_);
    layout->addWidget(highSpin_);
}

void WindowRangeWidget::setRange(int minimum, int maximum)
{
    if (minimum > maximum) {
        std::swap(minimum, maximum);
    }

    if (minimum_ == minimum && maximum_ == maximum) {
        return;
    }

    minimum_ = minimum;
    maximum_ = maximum;

    {
        QSignalBlocker blocker(slider_);
        slider_->setRange(minimum_, maximum_);
    }
    lowSpin_->setRange(minimum_, maximum_);
    highSpin_->setRange(minimum_, maximum_);

    syncControls(slider_->lowValue(), slider_->highValue(), false);
}

void WindowRangeWidget::setWindowValues(int low, int high)
{
    syncControls(low, high, false);
}

void WindowRangeWidget::setControlsEnabled(bool enabled)
{
    setEnabled(enabled);
    if (slider_) {
        slider_->setEnabled(enabled);
    }
    if (lowSpin_) {
        lowSpin_->setEnabled(enabled);
    }
    if (highSpin_) {
        highSpin_->setEnabled(enabled);
    }
}

void WindowRangeWidget::setMinimumSeparation(int separation)
{
    minimumSeparation_ = std::max(0, separation);
    if (slider_) {
        QSignalBlocker blocker(slider_);
        slider_->setMinimumSeparation(minimumSeparation_);
    }
    syncControls(slider_->lowValue(), slider_->highValue(), false);
}

auto WindowRangeWidget::lowValue() const -> int
{
    return lowSpin_ ? lowSpin_->value() : minimum_;
}

auto WindowRangeWidget::highValue() const -> int
{
    return highSpin_ ? highSpin_->value() : maximum_;
}

void WindowRangeWidget::syncControls(int low, int high, bool emitSignal)
{
    if (!slider_ || !lowSpin_ || !highSpin_) {
        return;
    }

    const int minGap = std::max(minimumSeparation_, 0);
    int clampedLow = std::clamp(low, minimum_, maximum_ - minGap);
    int clampedHigh = std::clamp(high, minimum_ + minGap, maximum_);
    if (clampedHigh - clampedLow < minGap) {
        if (emitSignal && clampedLow != low) {
            clampedHigh = std::min(maximum_, clampedLow + minGap);
        } else {
            clampedLow = std::max(minimum_, clampedHigh - minGap);
        }
    }

    const int prevLow = lowSpin_->value();
    const int prevHigh = highSpin_->value();

    suppressSignals_ = true;
    {
        QSignalBlocker sliderBlocker(slider_);
        slider_->setValues(clampedLow, clampedHigh);
    }
    {
        QSignalBlocker lowBlocker(lowSpin_);
        lowSpin_->setValue(clampedLow);
    }
    {
        QSignalBlocker highBlocker(highSpin_);
        highSpin_->setValue(clampedHigh);
    }
    suppressSignals_ = false;

    if (emitSignal) {
        emit windowValuesChanged(clampedLow, clampedHigh);
        if (clampedLow != prevLow) {
            emit lowChanged(clampedLow);
        }
        if (clampedHigh != prevHigh) {
            emit highChanged(clampedHigh);
        }
    }
}

}  // namespace qt_utils
