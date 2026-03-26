#pragma once
/// vc4d::VolumeViewer — GPU-accelerated volume slice viewer.
///
/// Uses QOpenGLWidget (Qt 6) for hardware-accelerated rendering of volume
/// slices.  Replaces vc3d's CTiledVolumeViewer which was built on
/// QGraphicsView + QGraphicsScene (software-rendered tiles).
///
/// The Qt 6 OpenGL path is simpler and faster for our use case (textured
/// quads displaying volume slice data).

#include "vc4d/core/volume.hpp"

#include <QOpenGLWidget>

namespace vc4d {

class VolumeViewer : public QOpenGLWidget {
    Q_OBJECT

public:
    explicit VolumeViewer(QWidget* parent = nullptr);
    ~VolumeViewer() override;

    void set_volume(Volume* vol);
    void set_slice(int z);
    void set_scale_level(int level);

    [[nodiscard]] int current_slice() const { return slice_; }
    [[nodiscard]] float zoom() const { return zoom_; }

signals:
    void slice_changed(int z);
    void zoom_changed(float zoom);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    Volume* volume_{};
    int slice_{};
    int scale_level_{};
    float zoom_{1.f};
    float pan_x_{}, pan_y_{};

    // Mouse interaction state
    bool dragging_{};
    float drag_start_x_{}, drag_start_y_{};
};

} // namespace vc4d
