#include "vc4d/ui/main_window.hpp"

#include <QDockWidget>
#include <QFileDialog>
#include <QMenuBar>
#include <QStatusBar>

namespace vc4d {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , state_(std::make_unique<AppState>(this))
{
    setWindowTitle("VC4D");
    resize(1600, 1000);

    viewer_ = new VolumeViewer(this);
    setCentralWidget(viewer_);

    setup_menus();
    setup_dock_widgets();
    setup_connections();

    statusBar()->showMessage("Ready");
}

MainWindow::~MainWindow() = default;

void MainWindow::setup_menus() {
    auto* file_menu = menuBar()->addMenu("&File");

    auto* open_action = file_menu->addAction("&Open Package...");
    open_action->setShortcut(QKeySequence::Open);
    connect(open_action, &QAction::triggered, this, &MainWindow::open_package);

    file_menu->addSeparator();

    auto* quit_action = file_menu->addAction("&Quit");
    quit_action->setShortcut(QKeySequence::Quit);
    connect(quit_action, &QAction::triggered, this, &QWidget::close);

    auto* view_menu = menuBar()->addMenu("&View");
    (void)view_menu;  // Dock widget toggles will be added here
}

void MainWindow::setup_dock_widgets() {
    // Surface panel dock (left side)
    auto* surface_dock = new QDockWidget("Surfaces", this);
    surface_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    // TODO: Add SurfacePanel widget
    addDockWidget(Qt::LeftDockWidgetArea, surface_dock);

    // Properties dock (right side)
    auto* props_dock = new QDockWidget("Properties", this);
    props_dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    // TODO: Add PropertiesPanel widget
    addDockWidget(Qt::RightDockWidgetArea, props_dock);
}

void MainWindow::setup_connections() {
    connect(state_.get(), &AppState::package_opened, this, &MainWindow::on_package_opened);
    connect(state_.get(), &AppState::volume_changed, this, &MainWindow::on_volume_changed);
}

void MainWindow::open_package() {
    auto dir = QFileDialog::getExistingDirectory(this, "Open Volume Package");
    if (dir.isEmpty()) return;

    TieredCache::Config cache_config;
    cache_config.hot_budget_bytes = 8ULL << 30;
    cache_config.warm_budget_bytes = 2ULL << 30;

    state_->open(dir.toStdString(), cache_config);
}

void MainWindow::on_package_opened() {
    setWindowTitle(QString("VC4D - %1").arg(QString::fromStdString(state_->pkg()->name())));
    statusBar()->showMessage("Package loaded");

    // Select first volume if available
    auto vol_ids = state_->pkg()->volume_ids();
    if (!vol_ids.empty())
        state_->select_volume(vol_ids.front());
}

void MainWindow::on_volume_changed(Volume* vol) {
    viewer_->set_volume(vol);
    if (vol)
        statusBar()->showMessage(QString("Volume: %1 (%2 x %3 x %4)")
            .arg(QString::fromStdString(vol->name()))
            .arg(vol->width()).arg(vol->height()).arg(vol->depth()));
}

} // namespace vc4d
