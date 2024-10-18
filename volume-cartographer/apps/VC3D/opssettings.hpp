#ifndef OPSSETTINGS_HPP
#define OPSSETTINGS_HPP

#include <QWidget>

class Surface;
class QGroupBox;

namespace Ui
{
class OpsSettings;
}

class OpsSettings : public QWidget
{
    Q_OBJECT

public:
    explicit OpsSettings(QWidget* parent = nullptr);
    ~OpsSettings();

public slots:
    void onOpSelected(Surface *surf);

private:
    Ui::OpsSettings* ui;
    QGroupBox *_box;
};

#endif  // OPSSETTINGS_HPP
