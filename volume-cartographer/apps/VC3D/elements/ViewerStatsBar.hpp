#pragma once

#include <QLabel>
#include <QStringList>

class ViewerStatsBar : public QLabel
{
    Q_OBJECT

public:
    explicit ViewerStatsBar(QWidget* parent = nullptr);

    // One row, or two stacked rows (the second hidden when empty).
    void setItems(const QStringList& items);
    void setItems(const QStringList& row1, const QStringList& row2);
};
