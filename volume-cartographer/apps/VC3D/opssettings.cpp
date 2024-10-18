#include "opssettings.hpp"
#include "ui_opssettings.h"

#include "OpChain.hpp"

#include <iostream>

OpsSettings::OpsSettings(QWidget* parent)
    : QWidget(parent), ui(new Ui::OpsSettings)
{
    ui->setupUi(this);
    _box = this->findChild<QGroupBox*>("groupBox");
}

OpsSettings::~OpsSettings() { delete ui; }


void OpsSettings::onOpSelected(Surface *surf)
{
    std::cout << "op was selected: " <<  op_name(surf) << std::endl;
    _box->setTitle(QString(op_name(surf)));
}
