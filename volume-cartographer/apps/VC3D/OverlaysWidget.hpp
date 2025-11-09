#pragma once

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPlainTextEdit>
#include <QSet>
#include <QString>

class OverlaysWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OverlaysWidget(QWidget* parent = nullptr);

    float getIntersectionLineWidth() const { return intersectionLineWidth; }
    void setIntersectionLineWidth(float width);

    QSet<QString> getHighlightedSegments() const { return highlightedSegments; }
    void setHighlightedSegments(const QSet<QString>& segments);

signals:
    void intersectionLineWidthChanged(float width);
    void highlightedSegmentsChanged(const QSet<QString>& segments);

private slots:
    void onIntersectionLineWidthSliderChanged(int value);
    void onHighlightedSegmentsTextChanged();

private:
    float intersectionLineWidth;
    QSlider* intersectionLineWidthSlider;
    QLabel* intersectionLineWidthLabel;

    QSet<QString> highlightedSegments;
    QPlainTextEdit* highlightedSegmentsEdit;
    QLabel* highlightedSegmentsLabel;
};
