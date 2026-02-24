#pragma once

#include <QColor>
#include <QWidget>

namespace qt_utils
{

/// A dual-handle range slider widget that allows selecting a sub-range within
/// a minimum/maximum interval. Supports horizontal and vertical orientations,
/// keyboard navigation, minimum separation enforcement, and custom handle colors.
class RangeSlider : public QWidget
{
    Q_OBJECT

    Q_PROPERTY(int minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(int maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(int lowValue READ lowValue WRITE setLowValue NOTIFY lowValueChanged)
    Q_PROPERTY(int highValue READ highValue WRITE setHighValue NOTIFY highValueChanged)
    Q_PROPERTY(int minimumSeparation READ minimumSeparation WRITE setMinimumSeparation)
    Q_PROPERTY(Qt::Orientation orientation READ orientation WRITE setOrientation)
    Q_PROPERTY(QColor handleColor READ handleColor WRITE setHandleColor)

public:
    explicit RangeSlider(
        Qt::Orientation orientation = Qt::Horizontal,
        QWidget* parent = nullptr);
    ~RangeSlider() override;

    // --- Accessors ---

    [[nodiscard]] auto orientation() const -> Qt::Orientation;
    [[nodiscard]] auto minimum() const -> int;
    [[nodiscard]] auto maximum() const -> int;
    [[nodiscard]] auto lowValue() const -> int;
    [[nodiscard]] auto highValue() const -> int;
    [[nodiscard]] auto minimumSeparation() const -> int;
    [[nodiscard]] auto handleColor() const -> QColor;

    // --- Mutators ---

    void setOrientation(Qt::Orientation orientation);
    void setMinimum(int minimum);
    void setMaximum(int maximum);
    void setRange(int minimum, int maximum);
    void setLowValue(int value);
    void setHighValue(int value);
    void setValues(int low, int high);
    void setMinimumSeparation(int separation);
    void setHandleColor(const QColor& color);

    // --- Size hints ---

    [[nodiscard]] auto sizeHint() const -> QSize override;
    [[nodiscard]] auto minimumSizeHint() const -> QSize override;

signals:
    void lowValueChanged(int value);
    void highValueChanged(int value);
    void valuesChanged(int low, int high);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qt_utils
