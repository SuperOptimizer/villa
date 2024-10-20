#include "opssettings.hpp"
#include "ui_opssettings.h"

#include "OpChain.hpp"

#include <iostream>

OpsSettings::OpsSettings(QWidget* parent)
    : QWidget(parent), ui(new Ui::OpsSettings)
{
    ui->setupUi(this);
    _box = this->findChild<QGroupBox*>("groupBox");
    _enable = this->findChild<QCheckBox*>("chkEnableLayer");

    connect(_enable, &QCheckBox::checkStateChanged, this, &OpsSettings::onEnabledChanged);
}

OpsSettings::~OpsSettings() { delete ui; }

void OpsSettings::onEnabledChanged()
{
    _chain->setEnabled((DeltaQuadSurface*)_op, _enable->isChecked());
    sendOpChainChanged(_chain);
}

void OpsSettings::onOpSelected(Surface *op, OpChain *chain)
{
    _op = op;
    _chain = chain;

    _box->setTitle(QString(op_name(op)));

    if (!dynamic_cast<DeltaQuadSurface*>(_op))
        _enable->setEnabled(false);
    else {
        _enable->setEnabled(true);
        QSignalBlocker blocker(_enable);
        _enable->setChecked(_chain->enabled((DeltaQuadSurface*)_op));
    }
}
