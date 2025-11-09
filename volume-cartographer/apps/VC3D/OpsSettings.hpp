#pragma once

#include <QWidget>
#include <ui_OpsSettings.h>

#include "vc/core/util/Surface.hpp"


class OpsSettings : public QWidget
{
    Q_OBJECT

public:
    explicit OpsSettings(QWidget* parent = nullptr);
    ~OpsSettings();

public slots:
    void onOpSelected(Surface *op);

private:
    Ui::OpsSettings* ui;
    QGroupBox *_box = nullptr;

    Surface *_op = nullptr;
};
