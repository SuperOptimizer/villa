#pragma once

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>

class OverlaysWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OverlaysWidget(QWidget* parent = nullptr);

    float getIntersectionLineWidth() const { return intersectionLineWidth; }
    void setIntersectionLineWidth(float width);

signals:
    void intersectionLineWidthChanged(float width);

private slots:
    void onIntersectionLineWidthSliderChanged(int value);

private:
    float intersectionLineWidth;
    QSlider* intersectionLineWidthSlider;
    QLabel* intersectionLineWidthLabel;
};
