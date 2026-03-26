#include "vc4d/ui/volume_viewer.hpp"

#include <QMouseEvent>
#include <QWheelEvent>

namespace vc4d {

VolumeViewer::VolumeViewer(QWidget* parent)
    : QOpenGLWidget(parent) {}

VolumeViewer::~VolumeViewer() = default;

void VolumeViewer::set_volume(Volume* vol) {
    volume_ = vol;
    slice_ = 0;
    scale_level_ = 0;
    zoom_ = 1.f;
    pan_x_ = pan_y_ = 0.f;
    update();
}

void VolumeViewer::set_slice(int z) {
    if (!volume_) return;
    slice_ = std::clamp(z, 0, volume_->depth() - 1);
    emit slice_changed(slice_);
    update();
}

void VolumeViewer::set_scale_level(int level) {
    if (!volume_) return;
    scale_level_ = std::clamp(level, 0, static_cast<int>(volume_->num_scales()) - 1);
    update();
}

void VolumeViewer::initializeGL() {
    // Initialize OpenGL state for 2D slice rendering
}

void VolumeViewer::resizeGL(int /*w*/, int /*h*/) {
    // Update viewport
}

void VolumeViewer::paintGL() {
    // Render current slice as a textured quad
    // TODO: Sample volume data into texture, upload to GPU, draw
}

void VolumeViewer::wheelEvent(QWheelEvent* event) {
    auto delta = event->angleDelta().y();
    if (event->modifiers() & Qt::ControlModifier) {
        // Zoom
        float factor = delta > 0 ? 1.1f : 0.9f;
        zoom_ *= factor;
        zoom_ = std::clamp(zoom_, 0.01f, 100.f);
        emit zoom_changed(zoom_);
    } else {
        // Scroll slices
        int step = delta > 0 ? -1 : 1;
        set_slice(slice_ + step);
    }
    update();
}

void VolumeViewer::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::MiddleButton || event->button() == Qt::LeftButton) {
        dragging_ = true;
        drag_start_x_ = static_cast<float>(event->position().x()) - pan_x_;
        drag_start_y_ = static_cast<float>(event->position().y()) - pan_y_;
    }
}

void VolumeViewer::mouseMoveEvent(QMouseEvent* event) {
    if (dragging_) {
        pan_x_ = static_cast<float>(event->position().x()) - drag_start_x_;
        pan_y_ = static_cast<float>(event->position().y()) - drag_start_y_;
        update();
    }
}

void VolumeViewer::mouseReleaseEvent(QMouseEvent* /*event*/) {
    dragging_ = false;
}

} // namespace vc4d
