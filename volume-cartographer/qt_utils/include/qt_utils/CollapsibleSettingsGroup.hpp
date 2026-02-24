#pragma once

#include <QWidget>

#include <functional>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QGridLayout;
class QHBoxLayout;
class QLabel;
class QSpinBox;
class QToolButton;
class QVBoxLayout;

namespace qt_utils {

/// A collapsible/expandable settings panel widget.
///
/// Provides a toggle button that shows/hides a content area containing a grid
/// of labeled controls. Helper methods add common control types (spinboxes,
/// checkboxes, combo boxes) with optional tooltips.
class CollapsibleSettingsGroup : public QWidget
{
    Q_OBJECT

public:
    explicit CollapsibleSettingsGroup(
        const QString& title, QWidget* parent = nullptr);

    /// Expand or collapse the content area.
    void setExpanded(bool expanded);

    /// Whether the content area is currently visible.
    [[nodiscard]] auto isExpanded() const -> bool;

    /// Set the number of label+widget columns in the grid.
    /// Clears any row-based layout preference.
    void setColumns(int columns);

    /// Set the desired number of rows; columns are computed automatically.
    /// Clears any column-based layout preference.
    void setRows(int rows);

    /// Set both preferred rows and columns explicitly.
    /// A zero value for either dimension means "compute from the other."
    /// If both are zero, defaults to a single column.
    void setGrid(int rows, int columns);

    /// Add a widget with an optional label and tooltip to the grid.
    /// Returns @p widget for chaining.
    auto addLabeledWidget(
        const QString& label,
        QWidget* widget,
        const QString& tooltip = {}) -> QWidget*;

    /// Build a custom horizontal row using a builder callback.
    /// The builder receives an QHBoxLayout to populate.
    auto addRow(
        const QString& label,
        const std::function<void(QHBoxLayout*)>& builder,
        const QString& tooltip = {}) -> QWidget*;

    /// Add an integer spin box with the given range and step.
    auto addSpinBox(
        const QString& label,
        int minimum,
        int maximum,
        int step = 1,
        const QString& tooltip = {}) -> QSpinBox*;

    /// Add a double spin box with the given range, step, and decimal count.
    auto addDoubleSpinBox(
        const QString& label,
        double minimum,
        double maximum,
        double step = 0.1,
        int decimals = 2,
        const QString& tooltip = {}) -> QDoubleSpinBox*;

    /// Add a checkbox (placed without a separate label column).
    auto addCheckBox(
        const QString& text, const QString& tooltip = {}) -> QCheckBox*;

    /// Add a combo box with the given items and an optional label.
    auto addComboBox(
        const QString& label,
        const QStringList& items = {},
        const QString& tooltip = {}) -> QComboBox*;

    /// Add a widget that spans the full row width.
    void addFullWidthWidget(QWidget* widget, const QString& tooltip = {});

    /// Access the content frame (for advanced customisation).
    [[nodiscard]] auto contentWidget() const -> QWidget*;

    /// Access the content area's vertical layout.
    [[nodiscard]] auto contentLayout() const -> QVBoxLayout*;

signals:
    /// Emitted when the expanded/collapsed state changes.
    void toggled(bool expanded);

private:
    void updateIndicator();
    void ensureGridLayout();
    void rebuildGrid();
    [[nodiscard]] auto slotsPerRow() const -> int;

    struct Entry {
        QLabel* label{nullptr};
        QWidget* widget{nullptr};
    };

    QToolButton* toggleButton_{nullptr};
    QFrame* contentWidget_{nullptr};
    QVBoxLayout* contentLayout_{nullptr};
    QGridLayout* gridLayout_{nullptr};
    int preferredColumns_{1};
    int preferredRows_{0};
    std::vector<Entry> entries_;
    bool expanded_{true};
};

}  // namespace qt_utils
