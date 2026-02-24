#include "qt_utils/CollapsibleSettingsGroup.hpp"

#include <algorithm>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

namespace qt_utils {

CollapsibleSettingsGroup::CollapsibleSettingsGroup(
    const QString& title, QWidget* parent)
    : QWidget(parent)
{
    // --- outer layout ---
    auto* outerLayout = new QVBoxLayout(this);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->setSpacing(0);

    // --- toggle button ---
    toggleButton_ = new QToolButton(this);
    toggleButton_->setText(title);
    toggleButton_->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    toggleButton_->setArrowType(Qt::DownArrow);
    toggleButton_->setCheckable(true);
    toggleButton_->setChecked(true);
    toggleButton_->setAutoRaise(true);
    toggleButton_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    outerLayout->addWidget(toggleButton_);

    // --- content area ---
    contentWidget_ = new QFrame(this);
    contentWidget_->setFrameShape(QFrame::StyledPanel);
    contentWidget_->setFrameShadow(QFrame::Raised);

    contentLayout_ = new QVBoxLayout(contentWidget_);
    contentLayout_->setContentsMargins(12, 8, 12, 12);
    contentLayout_->setSpacing(8);

    outerLayout->addWidget(contentWidget_);

    ensureGridLayout();

    connect(toggleButton_, &QToolButton::toggled, this, [this](bool checked) {
        setExpanded(checked);
    });

    updateIndicator();
}

// ---------------------------------------------------------------------------
// Expand / collapse
// ---------------------------------------------------------------------------

void CollapsibleSettingsGroup::setExpanded(bool expanded)
{
    if (expanded_ == expanded) {
        return;
    }
    expanded_ = expanded;
    contentWidget_->setVisible(expanded);
    toggleButton_->setChecked(expanded);
    updateIndicator();
    emit toggled(expanded);
}

auto CollapsibleSettingsGroup::isExpanded() const -> bool
{
    return expanded_;
}

// ---------------------------------------------------------------------------
// Grid dimensions
// ---------------------------------------------------------------------------

void CollapsibleSettingsGroup::setColumns(int columns)
{
    const int normalized = std::max(1, columns);
    if (preferredColumns_ == normalized && preferredRows_ == 0) {
        return;
    }
    preferredColumns_ = normalized;
    preferredRows_ = 0;
    rebuildGrid();
}

void CollapsibleSettingsGroup::setRows(int rows)
{
    const int normalized = std::max(1, rows);
    if (preferredRows_ == normalized && preferredColumns_ == 0) {
        return;
    }
    preferredRows_ = normalized;
    preferredColumns_ = 0;
    rebuildGrid();
}

void CollapsibleSettingsGroup::setGrid(int rows, int columns)
{
    int newRows = std::max(0, rows);
    int newColumns = std::max(0, columns);
    if (newRows == 0 && newColumns == 0) {
        newColumns = 1;
    }
    if (preferredRows_ == newRows && preferredColumns_ == newColumns) {
        return;
    }
    preferredRows_ = newRows;
    preferredColumns_ = newColumns;
    rebuildGrid();
}

// ---------------------------------------------------------------------------
// Adding controls
// ---------------------------------------------------------------------------

auto CollapsibleSettingsGroup::addLabeledWidget(
    const QString& labelText,
    QWidget* widget,
    const QString& tooltip) -> QWidget*
{
    if (!widget) {
        return nullptr;
    }

    QLabel* label = nullptr;
    const QString trimmed = labelText.trimmed();
    if (!trimmed.isEmpty()) {
        label = new QLabel(trimmed, contentWidget_);
        label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    }

    if (!tooltip.isEmpty()) {
        if (label) {
            label->setToolTip(tooltip);
        }
        widget->setToolTip(tooltip);
    }

    entries_.push_back({label, widget});
    rebuildGrid();
    return widget;
}

auto CollapsibleSettingsGroup::addRow(
    const QString& label,
    const std::function<void(QHBoxLayout*)>& builder,
    const QString& tooltip) -> QWidget*
{
    if (!builder) {
        return nullptr;
    }

    auto* container = new QWidget(contentWidget_);
    auto* rowLayout = new QHBoxLayout(container);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->setSpacing(8);
    builder(rowLayout);
    return addLabeledWidget(label, container, tooltip);
}

auto CollapsibleSettingsGroup::addSpinBox(
    const QString& label,
    int minimum,
    int maximum,
    int step,
    const QString& tooltip) -> QSpinBox*
{
    auto* spin = new QSpinBox(contentWidget_);
    spin->setRange(minimum, maximum);
    spin->setSingleStep(step);
    addLabeledWidget(label, spin, tooltip);
    return spin;
}

auto CollapsibleSettingsGroup::addDoubleSpinBox(
    const QString& label,
    double minimum,
    double maximum,
    double step,
    int decimals,
    const QString& tooltip) -> QDoubleSpinBox*
{
    auto* spin = new QDoubleSpinBox(contentWidget_);
    spin->setDecimals(decimals);
    spin->setRange(minimum, maximum);
    spin->setSingleStep(step);
    addLabeledWidget(label, spin, tooltip);
    return spin;
}

auto CollapsibleSettingsGroup::addCheckBox(
    const QString& text, const QString& tooltip) -> QCheckBox*
{
    auto* checkbox = new QCheckBox(text, contentWidget_);
    addLabeledWidget(QString(), checkbox, tooltip);
    return checkbox;
}

auto CollapsibleSettingsGroup::addComboBox(
    const QString& label,
    const QStringList& items,
    const QString& tooltip) -> QComboBox*
{
    auto* combo = new QComboBox(contentWidget_);
    combo->addItems(items);
    addLabeledWidget(label, combo, tooltip);
    return combo;
}

void CollapsibleSettingsGroup::addFullWidthWidget(
    QWidget* widget, const QString& tooltip)
{
    addLabeledWidget(QString(), widget, tooltip);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

auto CollapsibleSettingsGroup::contentWidget() const -> QWidget*
{
    return contentWidget_;
}

auto CollapsibleSettingsGroup::contentLayout() const -> QVBoxLayout*
{
    return contentLayout_;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void CollapsibleSettingsGroup::updateIndicator()
{
    toggleButton_->setArrowType(expanded_ ? Qt::DownArrow : Qt::RightArrow);
}

void CollapsibleSettingsGroup::ensureGridLayout()
{
    if (gridLayout_) {
        return;
    }
    gridLayout_ = new QGridLayout();
    gridLayout_->setContentsMargins(0, 0, 0, 0);
    gridLayout_->setHorizontalSpacing(12);
    gridLayout_->setVerticalSpacing(8);
    contentLayout_->addLayout(gridLayout_);
}

void CollapsibleSettingsGroup::rebuildGrid()
{
    ensureGridLayout();

    // Remove all items from the grid (widgets are not deleted, just detached).
    while (gridLayout_->count() > 0) {
        delete gridLayout_->takeAt(0);
    }

    const auto totalEntries = static_cast<int>(entries_.size());
    if (totalEntries == 0) {
        return;
    }

    const int entriesPerRow = std::max(1, slotsPerRow());

    // Configure column stretches: labels fixed, widgets expand.
    for (int slot = 0; slot < entriesPerRow; ++slot) {
        gridLayout_->setColumnStretch(slot * 2, 0);
        gridLayout_->setColumnStretch(slot * 2 + 1, 1);
    }

    int row = 0;
    int columnSlot = 0;
    for (const auto& entry : entries_) {
        if (columnSlot >= entriesPerRow) {
            columnSlot = 0;
            ++row;
        }

        const int baseColumn = columnSlot * 2;
        if (entry.label) {
            gridLayout_->addWidget(entry.label, row, baseColumn);
        }

        if (entry.widget) {
            const int widgetColumn = entry.label ? baseColumn + 1 : baseColumn;
            const int span = entry.label ? 1 : 2;
            gridLayout_->addWidget(entry.widget, row, widgetColumn, 1, span);
        }

        ++columnSlot;
    }
}

auto CollapsibleSettingsGroup::slotsPerRow() const -> int
{
    if (preferredColumns_ > 0) {
        return preferredColumns_;
    }
    if (preferredRows_ > 0 && !entries_.empty()) {
        const auto total = static_cast<int>(entries_.size());
        return std::max(1, (total + preferredRows_ - 1) / preferredRows_);
    }
    return 1;
}

}  // namespace qt_utils
