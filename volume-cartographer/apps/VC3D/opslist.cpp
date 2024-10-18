#include "opslist.hpp"
#include "ui_opslist.h"

#include "OpChain.hpp"

#include <iostream>

OpsList::OpsList(QWidget* parent) : QWidget(parent), ui(new Ui::OpsList)
{
    ui->setupUi(this);

    _tree = this->findChild<QTreeWidget*>("treeWidget");
    connect(_tree, &QTreeWidget::currentItemChanged, this, &OpsList::onSelChanged);
}

OpsList::~OpsList() { delete ui; }

void OpsList::onOpChainSelected(OpChain *ops)
{
    std::cout << "opchain selected" << ops << std::endl;

    _tree->clear();

    QTreeWidgetItem *item = new QTreeWidgetItem(_tree);
    item->setText(0, QString(op_name(ops)));
    item->setData(0, Qt::UserRole, QVariant::fromValue((void*)ops));

    for (auto& op : ops->ops()) {
        QTreeWidgetItem *item = new QTreeWidgetItem(_tree);
        item->setText(0, QString(op_name(op)));
        item->setData(0, Qt::UserRole, QVariant::fromValue((void*)op));
    }
}

void OpsList::onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous)
{
    if (!current)
        sendOpSelected(nullptr);
    else
        sendOpSelected((Surface*)get<void*>(current->data(0, Qt::UserRole)));
}
