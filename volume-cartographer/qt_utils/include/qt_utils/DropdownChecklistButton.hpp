#pragma once

#include <QList>
#include <QString>
#include <QToolButton>

class QCheckBox;

namespace qt_utils
{

/// A QToolButton that shows a dropdown menu containing checkable options.
///
/// Each option is a QCheckBox embedded in a QWidgetAction. The button itself
/// displays text (e.g. a label or count) and pops up the menu on click.
class DropdownChecklistButton : public QToolButton
{
    Q_OBJECT

public:
    explicit DropdownChecklistButton(QWidget* parent = nullptr);

    /// Add a checkable option to the dropdown menu.
    /// @param label       Display text for the checkbox.
    /// @param objectName  Optional QObject name for lookup.
    /// @param checked     Initial checked state.
    /// @return Pointer to the created QCheckBox (owned by the menu).
    auto addOption(
        const QString& label,
        const QString& objectName = {},
        bool checked = false) -> QCheckBox*;

    /// Add a visual separator to the dropdown menu.
    void addSeparator();

    /// Remove all options from the dropdown menu.
    void clearOptions();

    /// Return all option checkboxes.
    [[nodiscard]] auto options() const -> QList<QCheckBox*>;

    /// Return the number of currently checked options.
    [[nodiscard]] auto checkedCount() const -> int;

    /// Query whether a specific option (by label text) is checked.
    [[nodiscard]] auto isChecked(const QString& label) const -> bool;

signals:
    /// Emitted when any option's checked state changes.
    void optionToggled(const QString& label, bool checked);

private:
    QList<QCheckBox*> options_;
};

}  // namespace qt_utils
