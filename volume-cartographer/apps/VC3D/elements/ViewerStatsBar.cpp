#include "elements/ViewerStatsBar.hpp"

ViewerStatsBar::ViewerStatsBar(QWidget* parent)
    : QLabel(parent)
{
    setStyleSheet("QLabel { color : #00FF00; background-color: rgba(0,0,0,128); padding: 2px 4px; }");
    setMinimumWidth(520);
}

namespace {
QString joinVisible(const QStringList& items)
{
    QStringList visible;
    for (const auto& item : items) {
        if (!item.isEmpty())
            visible.push_back(item);
    }
    return visible.join("  ");
}
}

void ViewerStatsBar::setItems(const QStringList& items)
{
    setText(joinVisible(items));
    adjustSize();
}

void ViewerStatsBar::setItems(const QStringList& row1, const QStringList& row2)
{
    const QString top = joinVisible(row1);
    const QString bottom = joinVisible(row2);
    setText(bottom.isEmpty() ? top : top + "\n" + bottom);
    adjustSize();
}
