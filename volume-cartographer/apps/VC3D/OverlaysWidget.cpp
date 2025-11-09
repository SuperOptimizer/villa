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

    // Highlighted Segments control
    auto* highlightedSegmentsLayout = new QVBoxLayout();

    highlightedSegmentsLabel = new QLabel("Highlighted Segments (one per line, empty = all):", this);
    highlightedSegmentsLayout->addWidget(highlightedSegmentsLabel);

    highlightedSegmentsEdit = new QPlainTextEdit(this);
    highlightedSegmentsEdit->setPlaceholderText("Enter segment names, one per line...");
    highlightedSegmentsEdit->setMaximumHeight(150);

    connect(highlightedSegmentsEdit, &QPlainTextEdit::textChanged,
            this, &OverlaysWidget::onHighlightedSegmentsTextChanged);

    highlightedSegmentsLayout->addWidget(highlightedSegmentsEdit);

    mainLayout->addLayout(highlightedSegmentsLayout);
    mainLayout->addStretch();

    // Load saved values
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const float savedWidth = settings.value("viewer/intersection_line_width", 2.0f).toFloat();
    setIntersectionLineWidth(savedWidth);

    const QString savedSegments = settings.value("viewer/highlighted_segments", "").toString();
    highlightedSegmentsEdit->setPlainText(savedSegments);
    onHighlightedSegmentsTextChanged(); // Initialize the set
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

    // Save to settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("viewer/intersection_line_width", intersectionLineWidth);

    emit intersectionLineWidthChanged(intersectionLineWidth);
}

void OverlaysWidget::setHighlightedSegments(const QSet<QString>& segments)
{
    highlightedSegments = segments;

    // Update text edit without triggering signal
    highlightedSegmentsEdit->blockSignals(true);
    QStringList segmentList = segments.values();
    highlightedSegmentsEdit->setPlainText(segmentList.join('\n'));
    highlightedSegmentsEdit->blockSignals(false);
}

void OverlaysWidget::onHighlightedSegmentsTextChanged()
{
    QString text = highlightedSegmentsEdit->toPlainText();
    QStringList lines = text.split('\n', Qt::SkipEmptyParts);

    highlightedSegments.clear();
    for (const QString& line : lines) {
        QString trimmed = line.trimmed();
        if (!trimmed.isEmpty()) {
            highlightedSegments.insert(trimmed);
        }
    }

    // Save to settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue("viewer/highlighted_segments", text);

    emit highlightedSegmentsChanged(highlightedSegments);
}
