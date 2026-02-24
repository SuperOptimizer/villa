#include "qt_utils/DropdownChecklistButton.hpp"

#include <QCheckBox>
#include <QMenu>
#include <QWidgetAction>

namespace qt_utils
{

DropdownChecklistButton::DropdownChecklistButton(QWidget* parent)
    : QToolButton(parent)
{
    setToolButtonStyle(Qt::ToolButtonTextOnly);
    setPopupMode(QToolButton::InstantPopup);
    auto* dropdownMenu = new QMenu(this);
    setMenu(dropdownMenu);
}

auto DropdownChecklistButton::addOption(
    const QString& label,
    const QString& objectName,
    bool checked) -> QCheckBox*
{
    auto* dropdownMenu = menu();
    if (!dropdownMenu) {
        dropdownMenu = new QMenu(this);
        setMenu(dropdownMenu);
    }

    auto* checkBox = new QCheckBox(label, dropdownMenu);
    if (!objectName.isEmpty()) {
        checkBox->setObjectName(objectName);
    }
    checkBox->setChecked(checked);

    auto* action = new QWidgetAction(dropdownMenu);
    action->setDefaultWidget(checkBox);
    dropdownMenu->addAction(action);

    connect(checkBox, &QCheckBox::toggled, this, [this, checkBox](bool state) {
        emit optionToggled(checkBox->text(), state);
    });

    options_.append(checkBox);
    return checkBox;
}

void DropdownChecklistButton::addSeparator()
{
    if (auto* dropdownMenu = menu()) {
        dropdownMenu->addSeparator();
    }
}

void DropdownChecklistButton::clearOptions()
{
    if (auto* dropdownMenu = menu()) {
        dropdownMenu->clear();
    }
    options_.clear();
}

auto DropdownChecklistButton::options() const -> QList<QCheckBox*>
{
    return options_;
}

auto DropdownChecklistButton::checkedCount() const -> int
{
    int count = 0;
    for (const auto* option : options_) {
        if (option && option->isChecked()) {
            ++count;
        }
    }
    return count;
}

auto DropdownChecklistButton::isChecked(const QString& label) const -> bool
{
    for (const auto* option : options_) {
        if (option && option->text() == label) {
            return option->isChecked();
        }
    }
    return false;
}

}  // namespace qt_utils
