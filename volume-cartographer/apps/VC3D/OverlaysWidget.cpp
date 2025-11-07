#include "OverlaysWidget.hpp"

#include <QSettings>
#include "VCSettings.hpp"

OverlaysWidget::OverlaysWidget(QWidget* parent)
    : QWidget(parent)
    , intersectionLineWidth(2.0f)
{
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(10, 10, 10, 10);

    // Intersection Line Width control
    auto* lineWidthLayout = new QVBoxLayout();
    
    intersectionLineWidthLabel = new QLabel("Intersection Line Width: 2.0", this);
    lineWidthLayout->addWidget(intersectionLineWidthLabel);
    
    intersectionLineWidthSlider = new QSlider(Qt::Horizontal, this);
    intersectionLineWidthSlider->setMinimum(10);  // 1.0 * 10
    intersectionLineWidthSlider->setMaximum(100); // 10.0 * 10
    intersectionLineWidthSlider->setValue(20);    // 2.0 * 10
    intersectionLineWidthSlider->setTickPosition(QSlider::TicksBelow);
    intersectionLineWidthSlider->setTickInterval(10);
    
    connect(intersectionLineWidthSlider, &QSlider::valueChanged,
            this, &OverlaysWidget::onIntersectionLineWidthSliderChanged);
    
    lineWidthLayout->addWidget(intersectionLineWidthSlider);
    
    mainLayout->addLayout(lineWidthLayout);
    mainLayout->addStretch();

    // Load saved value
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const float savedWidth = settings.value("viewer/intersection_line_width", 2.0f).toFloat();
    setIntersectionLineWidth(savedWidth);
}

void OverlaysWidget::setIntersectionLineWidth(float width)
{
    intersectionLineWidth = std::clamp(width, 1.0f, 10.0f);
    
    // Update slider without triggering signal
    intersectionLineWidthSlider->blockSignals(true);
    intersectionLineWidthSlider->setValue(static_cast<int>(intersectionLineWidth * 10.0f));
    intersectionLineWidthSlider->blockSignals(false);
    
    // Update label
    intersectionLineWidthLabel->setText(
        QString("Intersection Line Width: %1").arg(intersectionLineWidth, 0, 'f', 1));
}

void OverlaysWidget::onIntersectionLineWidthSliderChanged(int value)
{
    intersectionLineWidth = static_cast<float>(value) / 10.0f;
    intersectionLineWidthLabel->setText(
        QString("Intersection Line Width: %1").arg(intersectionLineWidth, 0, 'f', 1));
    
    emit intersectionLineWidthChanged(intersectionLineWidth);
}
