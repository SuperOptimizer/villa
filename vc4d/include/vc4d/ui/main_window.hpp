#pragma once
/// vc4d::MainWindow — Top-level application window.
///
/// Key differences from vc3d's CWindow:
///   • Not a god object — CWindow was 2000+ lines managing viewers, panels,
///     overlays, menus, keybinds, and state all in one class.
///   • Delegates to focused components: VolumeViewer, SurfacePanel, etc.
///   • Uses Qt 6 dock widget system properly.

#include "app_state.hpp"
#include "volume_viewer.hpp"

#include <QMainWindow>
#include <memory>

namespace vc4d {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void open_package();
    void on_package_opened();
    void on_volume_changed(Volume* vol);

private:
    void setup_menus();
    void setup_dock_widgets();
    void setup_connections();

    std::unique_ptr<AppState> state_;
    VolumeViewer* viewer_{};  // owned by Qt parent
};

} // namespace vc4d
