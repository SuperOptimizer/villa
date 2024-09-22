// CVolumeViewer.cpp
// Chao Du 2015 April
#include <QGraphicsView>

namespace ChaoVis
{

class CVolumeViewerView : public QGraphicsView
{
    Q_OBJECT
    
public:
    CVolumeViewerView(QWidget* parent = 0) : QGraphicsView(parent) {};
    void mouseReleaseEvent(QMouseEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);
    void scrollContentsBy(int dx, int dy);
    
signals:
    void sendScrolled();
    void sendZoom(int steps);
    void sendVolumeClicked(QPointF, Qt::MouseButton, Qt::KeyboardModifiers);
    
protected:
    bool _regular_pan = false;
    QPointF _last_pan_position;
};

}