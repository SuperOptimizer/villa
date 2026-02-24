#include "qt_utils/RangeSlider.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>

#include <algorithm>
#include <cmath>

namespace qt_utils
{

namespace
{
constexpr int HandleRadius = 6;
constexpr int TrackThickness = 4;
constexpr int Padding = HandleRadius;
constexpr int HandleBorderWidth = 2;
}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

enum class DragTarget { None, Low, High };

struct RangeSlider::Impl {
    Qt::Orientation orientation{Qt::Horizontal};
    int minimum{0};
    int maximum{100};
    int lowValue{0};
    int highValue{100};
    int minimumSeparation{1};
    DragTarget dragTarget{DragTarget::None};
    int dragOffset{0};
    QColor handleColor;

    // True when the orientation axis is the primary axis (horizontal = x).
    [[nodiscard]] auto isHorizontal() const -> bool
    {
        return orientation == Qt::Horizontal;
    }

    // The length of the slider along the primary axis.
    [[nodiscard]] auto sliderLength(const QWidget* w) const -> int
    {
        return isHorizontal() ? w->width() : w->height();
    }

    // Track rectangle in widget coordinates.
    [[nodiscard]] auto trackRect(const QWidget* w) const -> QRect
    {
        if (isHorizontal()) {
            const int left = Padding;
            const int right = w->width() - Padding;
            const int cy = w->height() / 2;
            return {left, cy - TrackThickness / 2, right - left, TrackThickness};
        }
        const int top = Padding;
        const int bottom = w->height() - Padding;
        const int cx = w->width() / 2;
        return {cx - TrackThickness / 2, top, TrackThickness, bottom - top};
    }

    // Map a value to a pixel position along the primary axis.
    [[nodiscard]] auto positionFromValue(const QWidget* w, int value) const
        -> int
    {
        const QRect track = trackRect(w);
        const int trackLen =
            isHorizontal() ? track.width() : track.height();
        if (trackLen <= 0) {
            return isHorizontal() ? track.left() : track.top();
        }
        const double ratio =
            static_cast<double>(value - minimum) / (maximum - minimum);
        const int origin = isHorizontal() ? track.left() : track.top();
        return origin + static_cast<int>(std::round(ratio * trackLen));
    }

    // Map a pixel position along the primary axis to a value.
    [[nodiscard]] auto valueFromPosition(const QWidget* w, int position) const
        -> int
    {
        const QRect track = trackRect(w);
        const int trackLen =
            isHorizontal() ? track.width() : track.height();
        if (trackLen <= 0) {
            return minimum;
        }
        const int origin = isHorizontal() ? track.left() : track.top();
        const int clamped = std::clamp(
            position, origin, origin + trackLen);
        const double ratio =
            static_cast<double>(clamped - origin) / trackLen;
        const int value =
            static_cast<int>(std::round(ratio * (maximum - minimum))) +
            minimum;
        return std::clamp(value, minimum, maximum);
    }

    // Rectangle for a handle centered on the position for `value`.
    [[nodiscard]] auto handleRect(const QWidget* w, int value) const -> QRect
    {
        const int pos = positionFromValue(w, value);
        if (isHorizontal()) {
            const int cy = w->height() / 2;
            return {pos - HandleRadius, cy - HandleRadius,
                    HandleRadius * 2, HandleRadius * 2};
        }
        const int cx = w->width() / 2;
        return {cx - HandleRadius, pos - HandleRadius,
                HandleRadius * 2, HandleRadius * 2};
    }

    // Clamp and set low/high values, enforcing separation. Returns true if
    // anything changed.
    auto updateValues(RangeSlider* slider, int low, int high, bool emitSignals)
        -> bool
    {
        if (minimum >= maximum) {
            return false;
        }

        const int span = std::max(minimumSeparation, 0);
        int lo = std::clamp(low, minimum, maximum);
        int hi = std::clamp(high, minimum, maximum);

        // Enforce minimum separation by adjusting the non-dragged handle.
        if (hi - lo < span) {
            if (dragTarget == DragTarget::High) {
                lo = std::clamp(hi - span, minimum, maximum - span);
            } else {
                hi = std::clamp(lo + span, minimum + span, maximum);
            }
        }

        lo = std::clamp(lo, minimum, maximum - span);
        hi = std::clamp(hi, minimum + span, maximum);

        if (lo == lowValue && hi == highValue) {
            return false;
        }

        lowValue = lo;
        highValue = hi;

        if (lowValue > highValue) {
            std::swap(lowValue, highValue);
        }

        slider->update();

        if (emitSignals) {
            emit slider->lowValueChanged(lowValue);
            emit slider->highValueChanged(highValue);
            emit slider->valuesChanged(lowValue, highValue);
        }
        return true;
    }

    // Extract the primary-axis coordinate from a point.
    [[nodiscard]] auto primaryCoord(const QPoint& pt) const -> int
    {
        return isHorizontal() ? pt.x() : pt.y();
    }

    // Extract the primary-axis coordinate of a rect's center.
    [[nodiscard]] auto primaryCenter(const QRect& r) const -> int
    {
        return isHorizontal() ? r.center().x() : r.center().y();
    }
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

RangeSlider::RangeSlider(Qt::Orientation orientation, QWidget* parent)
    : QWidget(parent)
    , impl_(std::make_unique<Impl>())
{
    impl_->orientation = orientation;
    impl_->handleColor = palette().color(QPalette::Highlight);

    setFocusPolicy(Qt::StrongFocus);
    if (orientation == Qt::Horizontal) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    } else {
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    }
    setCursor(Qt::PointingHandCursor);
}

RangeSlider::~RangeSlider() = default;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

auto RangeSlider::orientation() const -> Qt::Orientation
{
    return impl_->orientation;
}

auto RangeSlider::minimum() const -> int { return impl_->minimum; }
auto RangeSlider::maximum() const -> int { return impl_->maximum; }
auto RangeSlider::lowValue() const -> int { return impl_->lowValue; }
auto RangeSlider::highValue() const -> int { return impl_->highValue; }

auto RangeSlider::minimumSeparation() const -> int
{
    return impl_->minimumSeparation;
}

auto RangeSlider::handleColor() const -> QColor
{
    return impl_->handleColor;
}

// ---------------------------------------------------------------------------
// Mutators
// ---------------------------------------------------------------------------

void RangeSlider::setOrientation(Qt::Orientation orientation)
{
    if (impl_->orientation == orientation) {
        return;
    }
    impl_->orientation = orientation;
    if (orientation == Qt::Horizontal) {
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    } else {
        setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    }
    updateGeometry();
    update();
}

void RangeSlider::setMinimum(int minimum)
{
    setRange(minimum, impl_->maximum);
}

void RangeSlider::setMaximum(int maximum)
{
    setRange(impl_->minimum, maximum);
}

void RangeSlider::setRange(int minimum, int maximum)
{
    if (minimum > maximum) {
        std::swap(minimum, maximum);
    }
    if (impl_->minimum == minimum && impl_->maximum == maximum) {
        return;
    }

    impl_->minimum = minimum;
    impl_->maximum = maximum;

    const int span = std::max(impl_->minimumSeparation, 1);
    const int lo =
        std::clamp(impl_->lowValue, minimum, maximum - span);
    const int hi =
        std::clamp(impl_->highValue, minimum + span, maximum);
    impl_->updateValues(this, lo, hi, true);
    update();
}

void RangeSlider::setLowValue(int value)
{
    impl_->updateValues(this, value, impl_->highValue, true);
}

void RangeSlider::setHighValue(int value)
{
    impl_->updateValues(this, impl_->lowValue, value, true);
}

void RangeSlider::setValues(int low, int high)
{
    impl_->updateValues(this, low, high, true);
}

void RangeSlider::setMinimumSeparation(int separation)
{
    impl_->minimumSeparation = std::max(0, separation);
    impl_->updateValues(
        this, impl_->lowValue, impl_->highValue, true);
}

void RangeSlider::setHandleColor(const QColor& color)
{
    if (impl_->handleColor == color) {
        return;
    }
    impl_->handleColor = color;
    update();
}

// ---------------------------------------------------------------------------
// Size hints
// ---------------------------------------------------------------------------

auto RangeSlider::sizeHint() const -> QSize
{
    if (impl_->isHorizontal()) {
        return {200, 2 * HandleRadius + 12};
    }
    return {2 * HandleRadius + 12, 200};
}

auto RangeSlider::minimumSizeHint() const -> QSize
{
    const int cross = 2 * HandleRadius + 4;
    const int along = 2 * (Padding + HandleRadius);
    if (impl_->isHorizontal()) {
        return {along, cross};
    }
    return {cross, along};
}

// ---------------------------------------------------------------------------
// Paint
// ---------------------------------------------------------------------------

void RangeSlider::paintEvent(QPaintEvent* /*event*/)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const auto& d = *impl_;
    const QRect track = d.trackRect(this);
    const double trackRound = TrackThickness / 2.0;

    // --- Track background ---
    p.setPen(Qt::NoPen);
    p.setBrush(palette().color(QPalette::Mid));
    p.drawRoundedRect(track, trackRound, trackRound);

    // --- Selected (highlighted) range ---
    const int lowPos = d.positionFromValue(this, d.lowValue);
    const int highPos = d.positionFromValue(this, d.highValue);

    QRect selected(track);
    if (d.isHorizontal()) {
        selected.setLeft(lowPos);
        selected.setRight(highPos);
    } else {
        selected.setTop(lowPos);
        selected.setBottom(highPos);
    }

    p.setBrush(palette().color(QPalette::Highlight));
    p.drawRoundedRect(selected, trackRound, trackRound);

    // --- Handles ---
    QPen handlePen(d.handleColor);
    handlePen.setWidth(HandleBorderWidth);
    p.setPen(handlePen);
    p.setBrush(palette().color(QPalette::Base));

    p.drawEllipse(d.handleRect(this, d.lowValue));
    p.drawEllipse(d.handleRect(this, d.highValue));

    // --- Focus indicator ---
    if (hasFocus()) {
        const QRect focusHandle =
            (d.dragTarget == DragTarget::High)
                ? d.handleRect(this, d.highValue)
                : d.handleRect(this, d.lowValue);
        QPen focusPen(palette().color(QPalette::Highlight));
        focusPen.setWidth(1);
        focusPen.setStyle(Qt::DotLine);
        p.setPen(focusPen);
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(focusHandle.adjusted(-2, -2, 2, 2));
    }
}

// ---------------------------------------------------------------------------
// Mouse handling
// ---------------------------------------------------------------------------

void RangeSlider::mousePressEvent(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton) {
        QWidget::mousePressEvent(event);
        return;
    }

    auto& d = *impl_;
    const QPoint pos = event->position().toPoint();
    const QRect lowHandle = d.handleRect(this, d.lowValue);
    const QRect highHandle = d.handleRect(this, d.highValue);

    if (lowHandle.contains(pos)) {
        d.dragTarget = DragTarget::Low;
        d.dragOffset = d.primaryCoord(pos) - d.primaryCenter(lowHandle);
    } else if (highHandle.contains(pos)) {
        d.dragTarget = DragTarget::High;
        d.dragOffset = d.primaryCoord(pos) - d.primaryCenter(highHandle);
    } else {
        // Click on track -- snap the nearer handle.
        const int lowDist =
            std::abs(d.primaryCoord(pos) - d.primaryCenter(lowHandle));
        const int highDist =
            std::abs(d.primaryCoord(pos) - d.primaryCenter(highHandle));
        const int clickValue =
            d.valueFromPosition(this, d.primaryCoord(pos));
        d.dragOffset = 0;

        if (lowDist <= highDist) {
            d.dragTarget = DragTarget::Low;
            d.updateValues(this, clickValue, d.highValue, true);
        } else {
            d.dragTarget = DragTarget::High;
            d.updateValues(this, d.lowValue, clickValue, true);
        }
    }

    setFocus(Qt::MouseFocusReason);
    update();
}

void RangeSlider::mouseMoveEvent(QMouseEvent* event)
{
    auto& d = *impl_;
    if (d.dragTarget == DragTarget::None) {
        QWidget::mouseMoveEvent(event);
        return;
    }

    const QPoint pos = event->position().toPoint();
    const int value =
        d.valueFromPosition(this, d.primaryCoord(pos) - d.dragOffset);

    if (d.dragTarget == DragTarget::Low) {
        d.updateValues(this, value, d.highValue, true);
    } else {
        d.updateValues(this, d.lowValue, value, true);
    }
}

void RangeSlider::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton &&
        impl_->dragTarget != DragTarget::None) {
        impl_->dragTarget = DragTarget::None;
        impl_->dragOffset = 0;
        update();
    }
    QWidget::mouseReleaseEvent(event);
}

// ---------------------------------------------------------------------------
// Keyboard navigation
// ---------------------------------------------------------------------------

void RangeSlider::keyPressEvent(QKeyEvent* event)
{
    auto& d = *impl_;
    int step = 0;

    switch (event->key()) {
        case Qt::Key_Left:
        case Qt::Key_Down:
            step = -1;
            break;
        case Qt::Key_Right:
        case Qt::Key_Up:
            step = 1;
            break;
        case Qt::Key_PageDown:
            step = -10;
            break;
        case Qt::Key_PageUp:
            step = 10;
            break;
        case Qt::Key_Home:
            d.updateValues(
                this, d.minimum, d.minimum + (d.highValue - d.lowValue), true);
            return;
        case Qt::Key_End:
            d.updateValues(
                this, d.maximum - (d.highValue - d.lowValue), d.maximum, true);
            return;
        default:
            QWidget::keyPressEvent(event);
            return;
    }

    // Shift multiplies step by 5.
    if (event->modifiers() & Qt::ShiftModifier) {
        step *= 5;
    }

    // Ctrl moves only the high handle; default moves both together.
    if (event->modifiers() & Qt::ControlModifier) {
        d.updateValues(this, d.lowValue, d.highValue + step, true);
    } else if (event->modifiers() & Qt::AltModifier) {
        d.updateValues(this, d.lowValue + step, d.highValue, true);
    } else {
        d.updateValues(
            this, d.lowValue + step, d.highValue + step, true);
    }
}

}  // namespace qt_utils
