#ifndef OPSLIST_HPP
#define OPSLIST_HPP

#include <QWidget>

class OpChain;
class Surface;
class QTreeWidget;
class QTreeWidgetItem;

namespace Ui
{
class OpsList;
}

class OpsList : public QWidget
{
    Q_OBJECT

public:
    explicit OpsList(QWidget* parent = nullptr);
    ~OpsList();


private slots:
    void onSelChanged(QTreeWidgetItem *current, QTreeWidgetItem *previous);

public slots:
    void onOpChainSelected(OpChain *ops);

signals:
    void sendOpSelected(Surface *surf);

private:
    Ui::OpsList* ui;
    QTreeWidget *_tree;
};

#endif  // OPSLIST_HPP
