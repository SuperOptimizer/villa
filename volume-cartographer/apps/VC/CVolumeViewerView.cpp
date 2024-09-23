// CVolumeViewer.cpp
// Chao Du 2015 April
#include "CVolumeViewerView.hpp"

#include <QGraphicsView>
#include <QMouseEvent>
#include <QScrollBar>

using namespace ChaoVis;

void CVolumeViewerView::scrollContentsBy(int dx, int dy)
{
    sendScrolled();
    QGraphicsView::scrollContentsBy(dx,dy);
}

void CVolumeViewerView::wheelEvent(QWheelEvent *event)
{
    int num_degrees = event->angleDelta().y() / 8;
    
    sendZoom(num_degrees/15);
    
    event->accept();
}

void CVolumeViewerView::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton)
    {
        _regular_pan = false;
        setCursor(Qt::ArrowCursor);
        event->accept();
        return;
    }
    else
    {
        QPointF global_loc = viewport()->mapFromGlobal(event->globalPosition());
        QPointF scene_loc = mapToScene({int(global_loc.x()),int(global_loc.y())});
        
        sendVolumeClicked(scene_loc, event->button(), event->modifiers());
        
        event->accept();
        return;
    }
    event->ignore();
}

void CVolumeViewerView::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::MiddleButton)
    {
        _regular_pan = true;
        _last_pan_position = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    event->ignore();
}

void CVolumeViewerView::mouseMoveEvent(QMouseEvent *event)
{
    if (_regular_pan)
    {
        QPointF scroll = _last_pan_position - event->position();
        
        int x = horizontalScrollBar()->value() + scroll.x();
        horizontalScrollBar()->setValue(x);
        int y = verticalScrollBar()->value() + scroll.y();
        verticalScrollBar()->setValue(y);
        
        _last_pan_position = event->position();
        event->accept();
        return;
    }
    event->ignore();
}