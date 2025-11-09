#include "OpsSettings.hpp"
#include "ui_OpsSettings.h"

OpsSettings::OpsSettings(QWidget* parent)
    : QWidget(parent), ui(new Ui::OpsSettings)
{
    ui->setupUi(this);
    _box = ui->groupBox;

    if (_box)
        _box->setVisible(false);
}

OpsSettings::~OpsSettings() { delete ui; }

void OpsSettings::onOpSelected(Surface *op)
{
    _op = op;

    if(!_op) {
        if (_box)
            _box->setVisible(false);
        return;
    }
}
