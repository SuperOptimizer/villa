#include "CWindow.hpp"

#include <cstdlib>

#include "WindowRangeWidget.hpp"
#include "VCSettings.hpp"
#include <QKeySequence>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QSettings>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QApplication>
#include <QGuiApplication>
#include <QStyleHints>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>
#include <QDateTime>
#include <QFileDialog>
#include <QTextStream>
#include <QFileInfo>
#include <QDir>
#include <QProgressDialog>
#include <QMessageBox>
#include <QThread>
#include <QtConcurrent/QtConcurrent>
#include <QComboBox>
#include <QFutureWatcher>
#include <QRegularExpressionValidator>
#include <QDockWidget>
#include <QLabel>
#include <QSizePolicy>
#include <QProcess>
#include <QTemporaryDir>
#include <QToolBar>
#include <QFileInfo>
#include <QTimer>
#include <QSize>
#include <QVector>
#include <QLoggingCategory>
#include <QDebug>
#include <QScrollArea>
#include <QSignalBlocker>
#include <nlohmann/json.hpp>
#include <QGraphicsSimpleTextItem>
#include <QPointer>
#include <QPen>
#include <QListView>
#include <QFont>
#include <QPainter>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cmath>
#include <limits>
#include <optional>
#include <cctype>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <vector>
#include <initializer_list>
#include <omp.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QStringList>

#include "CVolumeViewer.hpp"
#include "CVolumeViewerView.hpp"
#include "vc/ui/UDataManipulateUtils.hpp"
#include "vc/ui/VCCollection.hpp"
#include "SettingsDialog.hpp"
#include "CSurfaceCollection.hpp"
#include "CPointCollectionWidget.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SeedingWidget.hpp"
#include "DrawingWidget.hpp"
#include "segmentation/SegmentationWidget.hpp"
#include "CommandLineToolRunner.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "segmentation/SegmentationGrowth.hpp"
#include "segmentation/SegmentationGrower.hpp"
#include "SurfacePanelController.hpp"
#include "MenuActionController.hpp"
#include "ViewerManager.hpp"
#include "segmentation/SegmentationEditManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "overlays/PointsOverlayController.hpp"
#include "overlays/RawPointsOverlayController.hpp"
#include "overlays/PathsOverlayController.hpp"
#include "overlays/BBoxOverlayController.hpp"
#include "overlays/VectorOverlayController.hpp"
#include "overlays/PlaneSlicingOverlayController.hpp"
#include "overlays/VolumeOverlayController.hpp"
#include "vc/core/Version.hpp"

#include "vc/core/util/Logging.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Render.hpp"





Q_LOGGING_CATEGORY(lcSegGrowth, "vc.segmentation.growth");
Q_LOGGING_CATEGORY(lcAxisSlices, "vc.axis_aligned");

using qga = QGuiApplication;
using PathBrushShape = ViewerOverlayControllerBase::PathBrushShape;

// Note: Area calculation helpers and recalcAreaForSegments are in CWindowAreaCalculation.cpp

namespace
{

void ensureDockWidgetFeatures(QDockWidget* dock)
{
    if (!dock) {
        return;
    }

    auto features = dock->features();
    features |= QDockWidget::DockWidgetMovable;
    features |= QDockWidget::DockWidgetFloatable;
    features |= QDockWidget::DockWidgetClosable;
    dock->setFeatures(features);
}

QString normalGridDirectoryForVolumePkg(const std::shared_ptr<VolumePkg>& pkg,
                                        QString* checkedPath)
{
    if (checkedPath) {
        *checkedPath = QString();
    }

    if (!pkg) {
        qCInfo(lcSegGrowth) << "Normal grid lookup skipped (no volume package loaded)";
        return QString();
    }

    std::filesystem::path rootPath(pkg->getVolpkgDirectory());
    if (rootPath.empty()) {
        qCInfo(lcSegGrowth) << "Normal grid lookup skipped (volume package path empty)";
        return QString();
    }

    const std::filesystem::path candidate = rootPath / "normal_grids";
    const QString candidateStr = QString::fromStdString(candidate.string());
    if (checkedPath) {
        *checkedPath = candidateStr;
    }

    if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
        qCInfo(lcSegGrowth) << "Normal grid lookup at" << candidateStr << ": found";
        return candidateStr;
    }

    qCInfo(lcSegGrowth) << "Normal grid lookup at" << candidateStr << ": missing";
    return QString();
}

QStringList normal3dZarrCandidatesForVolumePkg(const std::shared_ptr<VolumePkg>& pkg,
                                               QString* hint)
{
    if (hint) {
        *hint = QString();
    }
    if (!pkg) {
        if (hint) {
            *hint = QObject::tr("Normal3D lookup skipped (no volume package loaded)");
        }
        return {};
    }
    std::filesystem::path rootPath(pkg->getVolpkgDirectory());
    if (rootPath.empty()) {
        if (hint) {
            *hint = QObject::tr("Normal3D lookup skipped (volume package path empty)");
        }
        return {};
    }
    const std::filesystem::path base = rootPath / "normal3d";
    const QString baseStr = QString::fromStdString(base.string());
    if (hint) {
        *hint = QObject::tr("Checked: %1").arg(baseStr);
    }
    std::error_code ec;
    if (!std::filesystem::exists(base, ec) || !std::filesystem::is_directory(base, ec)) {
        return {};
    }

    QStringList candidates;
    for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_directory(ec) || ec) {
            continue;
        }
        const std::filesystem::path& p = entry.path();
        // Heuristic: treat as zarr if it contains x/0, y/0, z/0.
        if (std::filesystem::is_directory(p / "x" / "0") &&
            std::filesystem::is_directory(p / "y" / "0") &&
            std::filesystem::is_directory(p / "z" / "0")) {
            candidates.push_back(QString::fromStdString(p.string()));
        }
    }

    candidates.sort();
    return candidates;
}

} // namespace

// Dark mode detection - works on all Qt 6.x versions
static bool isDarkMode() {
#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
    if (QGuiApplication::styleHints()->colorScheme() == Qt::ColorScheme::Dark)
        return true;
#endif
    // Fallback: check system palette brightness
    const auto windowColor = QGuiApplication::palette().color(QPalette::Window);
    return windowColor.lightness() < 128;
}

// Apply a consistent dark palette application-wide
static void applyDarkPalette() {
    QPalette p;
    p.setColor(QPalette::Window, QColor(53, 53, 53));
    p.setColor(QPalette::WindowText, Qt::white);
    p.setColor(QPalette::Base, QColor(42, 42, 42));
    p.setColor(QPalette::AlternateBase, QColor(66, 66, 66));
    p.setColor(QPalette::ToolTipBase, QColor(53, 53, 53));
    p.setColor(QPalette::ToolTipText, Qt::white);
    p.setColor(QPalette::Text, Qt::white);
    p.setColor(QPalette::Button, QColor(53, 53, 53));
    p.setColor(QPalette::ButtonText, Qt::white);
    p.setColor(QPalette::BrightText, Qt::red);
    p.setColor(QPalette::Link, QColor(42, 130, 218));
    p.setColor(QPalette::Highlight, QColor(42, 130, 218));
    p.setColor(QPalette::HighlightedText, Qt::black);
    p.setColor(QPalette::Disabled, QPalette::Text, QColor(127, 127, 127));
    p.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(127, 127, 127));
    QApplication::setPalette(p);
}

// Constructor
CWindow::CWindow() :
    fVpkg(nullptr),
    _cmdRunner(nullptr),
    _seedingWidget(nullptr),
    _drawingWidget(nullptr),
    _point_collection_widget(nullptr)
#ifdef __linux__
    ,_inotifyFd(-1),
    _inotifyNotifier(nullptr)
#endif
{
#ifdef __linux__
    // Initialize periodic timer for inotify events
    _inotifyProcessTimer = new QTimer(this);
    connect(_inotifyProcessTimer, &QTimer::timeout, this, &CWindow::processPendingInotifyEvents);
#endif

    // Initialize timer for debounced window state saving (500ms delay)
    _windowStateSaveTimer = new QTimer(this);
    _windowStateSaveTimer->setSingleShot(true);
    _windowStateSaveTimer->setInterval(500);
    connect(_windowStateSaveTimer, &QTimer::timeout, this, &CWindow::saveWindowState);

    _point_collection = new VCCollection(this);
    const QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    _mirrorCursorToSegmentation = settings.value(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION,
                                                  vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION_DEFAULT).toBool();
    setWindowIcon(QPixmap(":/images/logo.png"));
    ui.setupUi(this);
    const QString baseTitle = windowTitle();
    const QString repoShortHash = QString::fromStdString(ProjectInfo::RepositoryShortHash()).trimmed();
    if (!repoShortHash.isEmpty() && !repoShortHash.startsWith('@')
        && repoShortHash.compare("Untracked", Qt::CaseInsensitive) != 0) {
        setWindowTitle(QString("%1 %2").arg(baseTitle, repoShortHash));
    }
    // setAttribute(Qt::WA_DeleteOnClose);

    chunk_cache = new ChunkCache<uint8_t>(CHUNK_CACHE_SIZE_GB*1024ULL*1024ULL*1024ULL);
    std::cout << "chunk cache size is " << CHUNK_CACHE_SIZE_GB << " gigabytes " << "\n";

    _surf_col = new CSurfaceCollection();

    //_surf_col->setSurface("manual plane", new PlaneSurface({2000,2000,2000},{1,1,1}));
    _surf_col->setSurface("xy plane", std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,0,1}));
    _surf_col->setSurface("xz plane", std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{0,1,0}));
    _surf_col->setSurface("yz plane", std::make_shared<PlaneSurface>(cv::Vec3f{2000,2000,2000}, cv::Vec3f{1,0,0}));

    connect(_surf_col, &CSurfaceCollection::sendPOIChanged, this, &CWindow::onFocusPOIChanged);
    connect(_surf_col, &CSurfaceCollection::sendSurfaceWillBeDeleted, this, &CWindow::onSurfaceWillBeDeleted);

    _viewerManager = std::make_unique<ViewerManager>(_surf_col, _point_collection, chunk_cache, this);
    _viewerManager->setSegmentationCursorMirroring(_mirrorCursorToSegmentation);
    connect(_viewerManager.get(), &ViewerManager::viewerCreated, this, [this](CVolumeViewer* viewer) {
        configureViewerConnections(viewer);
    });

    // Slice step size label in status bar
    _sliceStepLabel = new QLabel(this);
    _sliceStepLabel->setContentsMargins(4, 0, 4, 0);
    int initialStepSize = _viewerManager->sliceStepSize();
    _sliceStepLabel->setText(tr("Step: %1").arg(initialStepSize));
    _sliceStepLabel->setToolTip(tr("Slice step size: use Shift+G / Shift+H to adjust"));
    statusBar()->addPermanentWidget(_sliceStepLabel);

    _pointsOverlay = std::make_unique<PointsOverlayController>(_point_collection, this);
    _viewerManager->setPointsOverlay(_pointsOverlay.get());

    _rawPointsOverlay = std::make_unique<RawPointsOverlayController>(_surf_col, this);
    _viewerManager->setRawPointsOverlay(_rawPointsOverlay.get());

    _pathsOverlay = std::make_unique<PathsOverlayController>(this);
    _viewerManager->setPathsOverlay(_pathsOverlay.get());

    _bboxOverlay = std::make_unique<BBoxOverlayController>(this);
    _viewerManager->setBBoxOverlay(_bboxOverlay.get());

    _vectorOverlay = std::make_unique<VectorOverlayController>(_surf_col, this);
    _viewerManager->setVectorOverlay(_vectorOverlay.get());

    _planeSlicingOverlay = std::make_unique<PlaneSlicingOverlayController>(_surf_col, this);
    _planeSlicingOverlay->bindToViewerManager(_viewerManager.get());
    _planeSlicingOverlay->setRotationSetter([this](const std::string& planeName, float degrees) {
        setAxisAlignedRotationDegrees(planeName, degrees);
        scheduleAxisAlignedOrientationUpdate();
    });
    _planeSlicingOverlay->setRotationFinishedCallback([this]() {
        flushAxisAlignedOrientationUpdate();
    });
    _planeSlicingOverlay->setAxisAlignedEnabled(_useAxisAlignedSlices);

    _axisAlignedRotationTimer = new QTimer(this);
    _axisAlignedRotationTimer->setSingleShot(true);
    _axisAlignedRotationTimer->setInterval(25);  // ms, matches kAxisAlignedRotationApplyDelayMs in CWindowAxisAlignedSlices.cpp
    connect(_axisAlignedRotationTimer, &QTimer::timeout,
            this, &CWindow::processAxisAlignedOrientationUpdate);

    _volumeOverlay = std::make_unique<VolumeOverlayController>(_viewerManager.get(), this);
    connect(_volumeOverlay.get(), &VolumeOverlayController::requestStatusMessage, this,
            [this](const QString& message, int timeout) {
                if (statusBar()) {
                    statusBar()->showMessage(message, timeout);
                }
            });
    _viewerManager->setVolumeOverlay(_volumeOverlay.get());

    // create UI widgets
    CreateWidgets();

    // create menus/actions controller
    _menuController = std::make_unique<MenuActionController>(this);
    _menuController->populateMenus(menuBar());

    if (isDarkMode()) {
        applyDarkPalette();
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(55, 80, 170), stop:0.8 rgb(225, 90, 80), stop:1 rgb(225, 150, 0)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(235, 180, 30); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(55, 55, 55); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(55, 55, 55); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(60, 60, 75); }"
            "QTabBar::tab { background: rgb(60, 60, 75); }"
            "QWidget#tabSegment { background: rgb(55, 55, 55); }";
        setStyleSheet(style);
    } else {
        const auto style = "QMenuBar { background: qlineargradient( x0:0 y0:0, x1:1 y1:0, stop:0 rgb(85, 110, 200), stop:0.8 rgb(255, 120, 110), stop:1 rgb(255, 180, 30)); }"
            "QMenuBar::item { background: transparent; }"
            "QMenuBar::item:selected { background: rgb(255, 200, 50); }"
            "QWidget#dockWidgetVolumesContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetSegmentationContent { background: rgb(245, 245, 255); }"
            "QWidget#dockWidgetAnnotationsContent { background: rgb(245, 245, 255); }"
            "QDockWidget::title { padding-top: 6px; background: rgb(205, 210, 240); }"
            "QTabBar::tab { background: rgb(205, 210, 240); }"
            "QWidget#tabSegment { background: rgb(245, 245, 255); }"
            "QRadioButton:disabled { color: gray; }";
        setStyleSheet(style);
    }

    // Restore geometry / sizes
    const QSettings geometry(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QByteArray savedGeometry = geometry.value(vc3d::settings::window::GEOMETRY).toByteArray();
    if (!savedGeometry.isEmpty()) {
        restoreGeometry(savedGeometry);
    }
    const QByteArray savedState = geometry.value(vc3d::settings::window::STATE).toByteArray();
    if (!savedState.isEmpty()) {
        restoreState(savedState);
    } else {
        // No saved state - set sensible default sizes for dock widgets
        // The Volume Package dock (left side) should have a reasonable width and height
        resizeDocks({ui.dockWidgetVolumes}, {300}, Qt::Horizontal);
        resizeDocks({ui.dockWidgetVolumes}, {400}, Qt::Vertical);
    }

    for (QDockWidget* dock : { ui.dockWidgetSegmentation,
                               ui.dockWidgetDistanceTransform,
                               ui.dockWidgetDrawing,
                               ui.dockWidgetComposite,
                               ui.dockWidgetPostprocessing,
                               ui.dockWidgetVolumes,
                               ui.dockWidgetView,
                               ui.dockWidgetOverlay,
                               ui.dockWidgetRenderSettings  }) {
        ensureDockWidgetFeatures(dock);
        // Connect dock widget signals to trigger state saving
        connect(dock, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
        connect(dock, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);
    }
    ensureDockWidgetFeatures(_point_collection_widget);
    connect(_point_collection_widget, &QDockWidget::topLevelChanged, this, &CWindow::scheduleWindowStateSave);
    connect(_point_collection_widget, &QDockWidget::dockLocationChanged, this, &CWindow::scheduleWindowStateSave);

    const QSize minWindowSize(960, 640);
    setMinimumSize(minWindowSize);
    if (width() < minWindowSize.width() || height() < minWindowSize.height()) {
        resize(std::max(width(), minWindowSize.width()),
               std::max(height(), minWindowSize.height()));
    }

    // If enabled, auto open the last used volpkg
    if (settings.value(vc3d::settings::volpkg::AUTO_OPEN, vc3d::settings::volpkg::AUTO_OPEN_DEFAULT).toInt() != 0) {

        QStringList files = settings.value(vc3d::settings::volpkg::RECENT).toStringList();

        if (!files.empty() && !files.at(0).isEmpty()) {
            if (_menuController) {
                _menuController->openVolpkgAt(files[0]);
            }
        }
    }

    // Create application-wide keyboard shortcuts
    fDrawingModeShortcut = new QShortcut(QKeySequence("Ctrl+Shift+D"), this);
    fDrawingModeShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDrawingModeShortcut, &QShortcut::activated, [this]() {
        if (_drawingWidget) {
            _drawingWidget->toggleDrawingMode();
        }
    });

    fCompositeViewShortcut = new QShortcut(QKeySequence("C"), this);
    fCompositeViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCompositeViewShortcut, &QShortcut::activated, [this]() {
        if (!_viewerManager) {
            return;
        }
        _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
            if (viewer && viewer->surfName() == "segmentation") {
                viewer->setCompositeEnabled(!viewer->isCompositeEnabled());
            }
        });
    });

    // Toggle direction hints overlay (Ctrl+T)
    fDirectionHintsShortcut = new QShortcut(QKeySequence("Ctrl+T"), this);
    fDirectionHintsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fDirectionHintsShortcut, &QShortcut::activated, [this]() {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_DIRECTION_HINTS, viewer::SHOW_DIRECTION_HINTS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_DIRECTION_HINTS, next ? "1" : "0");
        if (_viewerManager) {
            _viewerManager->forEachViewer([next](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setShowDirectionHints(next);
                }
            });
        }
    });

    // Toggle surface normals visualization (Ctrl+N)
    fSurfaceNormalsShortcut = new QShortcut(QKeySequence("Ctrl+N"), this);
    fSurfaceNormalsShortcut->setContext(Qt::ApplicationShortcut);
    connect(fSurfaceNormalsShortcut, &QShortcut::activated, [this]() {
        using namespace vc3d::settings;
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        bool current = settings.value(viewer::SHOW_SURFACE_NORMALS, viewer::SHOW_SURFACE_NORMALS_DEFAULT).toBool();
        bool next = !current;
        settings.setValue(viewer::SHOW_SURFACE_NORMALS, next ? "1" : "0");
        if (_viewerManager) {
            _viewerManager->forEachViewer([next](CVolumeViewer* viewer) {
                if (viewer) {
                    viewer->setShowSurfaceNormals(next);
                }
            });
        }
        statusBar()->showMessage(next ? tr("Surface normals: ON") : tr("Surface normals: OFF"), 2000);
    });

    fAxisAlignedSlicesShortcut = new QShortcut(QKeySequence("Ctrl+J"), this);
    fAxisAlignedSlicesShortcut->setContext(Qt::ApplicationShortcut);
    connect(fAxisAlignedSlicesShortcut, &QShortcut::activated, [this]() {
        if (chkAxisAlignedSlices) {
            chkAxisAlignedSlices->toggle();
        }
    });

    // Raw points overlay shortcut (P key)
    auto* rawPointsShortcut = new QShortcut(QKeySequence("P"), this);
    rawPointsShortcut->setContext(Qt::ApplicationShortcut);
    connect(rawPointsShortcut, &QShortcut::activated, [this]() {
        if (_rawPointsOverlay) {
            bool newEnabled = !_rawPointsOverlay->isEnabled();
            _rawPointsOverlay->setEnabled(newEnabled);
            statusBar()->showMessage(
                newEnabled ? tr("Raw points overlay enabled") : tr("Raw points overlay disabled"),
                2000);
        }
    });

    // Zoom shortcuts (Shift+= for zoom in, Shift+- for zoom out)
    // Use 15% steps for smooth, proportional zooming - only affects active viewer
    constexpr float ZOOM_FACTOR = 1.15f;
    fZoomInShortcut = new QShortcut(QKeySequence("Shift+="), this);
    fZoomInShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomInShortcut, &QShortcut::activated, [this]() {
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CVolumeViewer*>(subWindow->widget())) {
                viewer->adjustZoomByFactor(ZOOM_FACTOR);
            }
        }
    });

    fZoomOutShortcut = new QShortcut(QKeySequence("Shift+-"), this);
    fZoomOutShortcut->setContext(Qt::ApplicationShortcut);
    connect(fZoomOutShortcut, &QShortcut::activated, [this]() {
        if (!mdiArea) return;
        if (auto* subWindow = mdiArea->activeSubWindow()) {
            if (auto* viewer = qobject_cast<CVolumeViewer*>(subWindow->widget())) {
                viewer->adjustZoomByFactor(1.0f / ZOOM_FACTOR);
            }
        }
    });

    // Reset view shortcut (m to fit surface in view and reset all offsets)
    fResetViewShortcut = new QShortcut(QKeySequence("m"), this);
    fResetViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fResetViewShortcut, &QShortcut::activated, [this]() {
        if (!_viewerManager) {
            return;
        }
        _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
            if (viewer) {
                viewer->resetSurfaceOffsets();
                viewer->fitSurfaceInView();
                viewer->renderVisible(true);
            }
        });
    });

    // Z offset: Ctrl+. = +Z (further/deeper), Ctrl+, = -Z (closer)
    fWorldOffsetZPosShortcut = new QShortcut(QKeySequence("Ctrl+."), this);
    fWorldOffsetZPosShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZPosShortcut, &QShortcut::activated, [this]() {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
            if (viewer) viewer->adjustSurfaceOffset(1.0f);
        });
    });

    fWorldOffsetZNegShortcut = new QShortcut(QKeySequence("Ctrl+,"), this);
    fWorldOffsetZNegShortcut->setContext(Qt::ApplicationShortcut);
    connect(fWorldOffsetZNegShortcut, &QShortcut::activated, [this]() {
        if (!_viewerManager) return;
        _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
            if (viewer) viewer->adjustSurfaceOffset(-1.0f);
        });
    });

    // Segment cycling shortcuts (] for next, [ for previous)
    fCycleNextSegmentShortcut = new QShortcut(QKeySequence("]"), this);
    fCycleNextSegmentShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCycleNextSegmentShortcut, &QShortcut::activated, [this]() {
        if (!_surfacePanel) {
            return;
        }

        const bool preserveEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        bool previousIgnore = false;
        if (preserveEditing && _segmentationModule) {
            previousIgnore = _segmentationModule->ignoreSegSurfaceChange();
            _segmentationModule->setIgnoreSegSurfaceChange(true);
        }

        _surfacePanel->cycleToNextVisibleSegment();

        if (preserveEditing && _segmentationModule) {
            _segmentationModule->setIgnoreSegSurfaceChange(previousIgnore);
        }
    });

    fCyclePrevSegmentShortcut = new QShortcut(QKeySequence("["), this);
    fCyclePrevSegmentShortcut->setContext(Qt::ApplicationShortcut);
    connect(fCyclePrevSegmentShortcut, &QShortcut::activated, [this]() {
        if (!_surfacePanel) {
            return;
        }

        const bool preserveEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        bool previousIgnore = false;
        if (preserveEditing && _segmentationModule) {
            previousIgnore = _segmentationModule->ignoreSegSurfaceChange();
            _segmentationModule->setIgnoreSegSurfaceChange(true);
        }

        _surfacePanel->cycleToPreviousVisibleSegment();

        if (preserveEditing && _segmentationModule) {
            _segmentationModule->setIgnoreSegSurfaceChange(previousIgnore);
        }
    });

    // Focused view toggle (Shift+Ctrl+F) - hides dock widgets, keeps all viewers
    fFocusedViewShortcut = new QShortcut(QKeySequence("Shift+Ctrl+F"), this);
    fFocusedViewShortcut->setContext(Qt::ApplicationShortcut);
    connect(fFocusedViewShortcut, &QShortcut::activated, this, &CWindow::toggleFocusedView);

    connect(_surfacePanel.get(), &SurfacePanelController::moveToPathsRequested, this, &CWindow::onMoveSegmentToPaths);
}

// Destructor
CWindow::~CWindow()
{
#ifdef __linux__
    if (_inotifyProcessTimer) {
        _inotifyProcessTimer->stop();
        delete _inotifyProcessTimer;
    }

    stopWatchingWithInotify();
#endif
    setStatusBar(nullptr);

    CloseVolume();
    delete chunk_cache;
    delete _surf_col;
    delete _point_collection;
}

CVolumeViewer *CWindow::newConnectedCVolumeViewer(const std::string& surfaceName, const QString& title, QMdiArea *mdiArea)
{
    if (!_viewerManager) {
        return nullptr;
    }

    CVolumeViewer* viewer = _viewerManager->createViewer(surfaceName, title, mdiArea);
    if (!viewer) {
        return nullptr;
    }

    return viewer;
}

void CWindow::configureViewerConnections(CVolumeViewer* viewer)
{
    if (!viewer) {
        return;
    }

    connect(this, &CWindow::sendVolumeChanged, viewer, &CVolumeViewer::OnVolumeChanged, Qt::UniqueConnection);
    connect(this, &CWindow::sendVolumeClosing, viewer, &CVolumeViewer::onVolumeClosing, Qt::UniqueConnection);
    connect(viewer, &CVolumeViewer::sendVolumeClicked, this, &CWindow::onVolumeClicked, Qt::UniqueConnection);

    if (viewer->fGraphicsView) {
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMousePress,
                viewer, &CVolumeViewer::onMousePress, Qt::UniqueConnection);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseMove,
                viewer, &CVolumeViewer::onMouseMove, Qt::UniqueConnection);
        connect(viewer->fGraphicsView, &CVolumeViewerView::sendMouseRelease,
                viewer, &CVolumeViewer::onMouseRelease, Qt::UniqueConnection);
    }

    if (_drawingWidget && !viewer->property("vc_drawing_bound").toBool()) {
        connect(_drawingWidget, &DrawingWidget::sendPathsChanged,
                viewer, &CVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                _drawingWidget, &DrawingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                _drawingWidget, &DrawingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                _drawingWidget, &DrawingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendZSliceChanged,
                _drawingWidget, &DrawingWidget::updateCurrentZSlice, Qt::UniqueConnection);
    connect(_drawingWidget, &DrawingWidget::sendDrawingModeActive,
            this, [this, viewer](bool active) {
                viewer->onDrawingModeActive(active,
                    _drawingWidget->getBrushSize(),
                    _drawingWidget->getBrushShape() == PathBrushShape::Square);
            });
        viewer->setProperty("vc_drawing_bound", true);
    }

    if (_seedingWidget && !viewer->property("vc_seeding_bound").toBool()) {
        connect(_seedingWidget, &SeedingWidget::sendPathsChanged,
                viewer, &CVolumeViewer::onPathsChanged, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                _seedingWidget, &SeedingWidget::onMousePress, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                _seedingWidget, &SeedingWidget::onMouseMove, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                _seedingWidget, &SeedingWidget::onMouseRelease, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendZSliceChanged,
                _seedingWidget, &SeedingWidget::updateCurrentZSlice, Qt::UniqueConnection);
        viewer->setProperty("vc_seeding_bound", true);
    }

    if (_segmentationModule) {
        _segmentationModule->attachViewer(viewer);
    }

    if (_point_collection_widget && !viewer->property("vc_points_bound").toBool()) {
        connect(_point_collection_widget, &CPointCollectionWidget::collectionSelected,
                viewer, &CVolumeViewer::onCollectionSelected, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::sendCollectionSelected,
                _point_collection_widget, &CPointCollectionWidget::selectCollection, Qt::UniqueConnection);
        connect(_point_collection_widget, &CPointCollectionWidget::pointSelected,
                viewer, &CVolumeViewer::onPointSelected, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::pointSelected,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        connect(viewer, &CVolumeViewer::pointClicked,
                _point_collection_widget, &CPointCollectionWidget::selectPoint, Qt::UniqueConnection);
        viewer->setProperty("vc_points_bound", true);
    }

    const std::string& surfName = viewer->surfName();
    if ((surfName == "seg xz" || surfName == "seg yz") && !viewer->property("vc_axisaligned_bound").toBool()) {
        if (viewer->fGraphicsView) {
            viewer->fGraphicsView->setMiddleButtonPanEnabled(!_useAxisAlignedSlices);
        }

        connect(viewer, &CVolumeViewer::sendMousePressVolume,
                this, [this, viewer](const cv::Vec3f& volLoc, const cv::Vec3f& /*normal*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    onAxisAlignedSliceMousePress(viewer, volLoc, button, modifiers);
                });

        connect(viewer, &CVolumeViewer::sendMouseMoveVolume,
                this, [this, viewer](const cv::Vec3f& volLoc, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers) {
                    onAxisAlignedSliceMouseMove(viewer, volLoc, buttons, modifiers);
                });

        connect(viewer, &CVolumeViewer::sendMouseReleaseVolume,
                this, [this, viewer](const cv::Vec3f& /*volLoc*/, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                    onAxisAlignedSliceMouseRelease(viewer, button, modifiers);
                });

        viewer->setProperty("vc_axisaligned_bound", true);
    }
}

CVolumeViewer* CWindow::segmentationViewer() const
{
    if (!_viewerManager) {
        return nullptr;
    }
    for (auto* viewer : _viewerManager->viewers()) {
        if (viewer && viewer->surfName() == "segmentation") {
            return viewer;
        }
    }
    return nullptr;
}

void CWindow::clearSurfaceSelection()
{
    _surf_weak.reset();
    _surfID.clear();

    if (_surfacePanel) {
        _surfacePanel->resetTagUi();
    }

    if (auto* viewer = segmentationViewer()) {
        viewer->setWindowTitle(tr("Surface"));
    }

    if (treeWidgetSurfaces) {
        treeWidgetSurfaces->clearSelection();
    }
}

void CWindow::setVolume(const std::shared_ptr<Volume>& newvol)
{
    const bool hadVolume = static_cast<bool>(currentVolume);
    auto previousVolume = currentVolume;
    POI* existingFocusPoi = _surf_col ? _surf_col->poi("focus") : nullptr;
    currentVolume = newvol;

    if (previousVolume != newvol) {
        _focusHistory.clear();
        _focusHistoryIndex = -1;
        _navigatingFocusHistory = false;
    }

    // Find the volume ID for the current volume
    currentVolumeId.clear();
    if (fVpkg && currentVolume) {
        for (const auto& id : fVpkg->volumeIDs()) {
            if (fVpkg->volume(id) == currentVolume) {
                currentVolumeId = id;
                break;
            }
        }
    }

    const bool growthVolumeValid = fVpkg && !_segmentationGrowthVolumeId.empty() &&
                                   fVpkg->hasVolume(_segmentationGrowthVolumeId);
    if (!growthVolumeValid) {
        _segmentationGrowthVolumeId = currentVolumeId;
        if (_segmentationWidget) {
            _segmentationWidget->setActiveVolume(QString::fromStdString(currentVolumeId));
        }
    }

    updateNormalGridAvailability();

    sendVolumeChanged(currentVolume, currentVolumeId);

    if (currentVolume && _surf_col) {
        auto [w, h, d] = currentVolume->shape();

        POI* poi = existingFocusPoi;
        const bool createdPoi = (poi == nullptr);
        if (!poi) {
            poi = new POI;
            poi->n = cv::Vec3f(0, 0, 1);
        }

        const auto clampCoord = [](float value, int maxDim) {
            if (maxDim <= 0) {
                return 0.0f;
            }
            const float maxValue = static_cast<float>(maxDim - 1);
            return std::clamp(value, 0.0f, maxValue);
        };

        if (createdPoi || !hadVolume) {
            poi->p = cv::Vec3f(w / 2.0f, h / 2.0f, d / 2.0f);
        } else {
            poi->p[0] = clampCoord(poi->p[0], w);
            poi->p[1] = clampCoord(poi->p[1], h);
            poi->p[2] = clampCoord(poi->p[2], d);
        }

        _surf_col->setPOI("focus", poi);
    }

    onManualPlaneChanged();
    applySlicePlaneOrientation(_surf_col ? _surf_col->surface("segmentation").get() : nullptr);
}

void CWindow::updateNormalGridAvailability()
{
    QString checkedPath;
    const QString path = normalGridDirectoryForVolumePkg(fVpkg, &checkedPath);
    const bool available = !path.isEmpty();

    _normalGridAvailable = available;
    _normalGridPath = path;

    if (_segmentationWidget) {
        _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
        _segmentationWidget->setNormalGridPath(_normalGridPath);
        QString hint;
        if (_normalGridAvailable) {
        } else if (!checkedPath.isEmpty()) {
            hint = tr("Checked: %1").arg(checkedPath);
        } else {
            hint = tr("No volume package loaded.");
        }
        _segmentationWidget->setNormalGridPathHint(hint);

        QString normal3dHint;
        const QStringList normal3d = normal3dZarrCandidatesForVolumePkg(fVpkg, &normal3dHint);
        _segmentationWidget->setNormal3dZarrCandidates(normal3d, normal3dHint);
    }
}

void CWindow::toggleVolumeOverlayVisibility()
{
    if (_volumeOverlay) {
        _volumeOverlay->toggleVisibility();
    }
}

void CWindow::toggleFocusedView()
{
    if (_focusedViewActive) {
        for (const auto& [dock, state] : _savedDockStates) {
            if (dock) {
                dock->setVisible(state.visible);
            }
        }
        for (const auto& [dock, state] : _savedDockStates) {
            if (dock && state.wasRaised) {
                dock->raise();
            }
        }
        _savedDockStates.clear();
        _focusedViewActive = false;
        statusBar()->showMessage(tr("Restored full view"), 2000);
    } else {
        _savedDockStates.clear();
        const QList<QDockWidget*> docks = findChildren<QDockWidget*>();
        for (QDockWidget* dock : docks) {
            bool wasRaised = false;
            if (dock->isVisible() && !dock->isFloating()) {
                if (QWidget* content = dock->widget()) {
                    wasRaised = !content->visibleRegion().isEmpty();
                }
            }
            _savedDockStates[dock] = {dock->isVisible(), dock->isFloating(), wasRaised};
            dock->hide();
        }
        _focusedViewActive = true;
        statusBar()->showMessage(tr("Focused view (Shift+Ctrl+F to restore)"), 2000);
    }
}

// Note: Focus history methods (recordFocusHistory, stepFocusHistory, centerFocusAt,
// centerFocusOnCursor) are in CWindowFocusHistory.cpp

void CWindow::setSegmentationCursorMirroring(bool enabled)
{
    if (_mirrorCursorToSegmentation == enabled) {
        return;
    }

    _mirrorCursorToSegmentation = enabled;
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::MIRROR_CURSOR_TO_SEGMENTATION, enabled ? "1" : "0");

    if (_viewerManager) {
        _viewerManager->setSegmentationCursorMirroring(enabled);
    }

    if (statusBar()) {
        statusBar()->showMessage(enabled ? tr("Mirroring cursor to Surface view enabled")
                                         : tr("Mirroring cursor to Surface view disabled"),
                                  2000);
    }
}

// Create widgets
void CWindow::CreateWidgets(void)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);

    // add volume viewer
    auto aWidgetLayout = new QVBoxLayout;
    ui.tabSegment->setLayout(aWidgetLayout);

    mdiArea = new QMdiArea(ui.tabSegment);
    aWidgetLayout->addWidget(mdiArea);

    // Ensure the viewer's graphics view gets focus when subwindow is activated
    connect(mdiArea, &QMdiArea::subWindowActivated, [](QMdiSubWindow* subWindow) {
        if (subWindow) {
            if (auto* viewer = qobject_cast<CVolumeViewer*>(subWindow->widget())) {
                viewer->fGraphicsView->setFocus();
            }
        }
    });

    // newConnectedCVolumeViewer("manual plane", tr("Manual Plane"), mdiArea);
    newConnectedCVolumeViewer("seg xz", tr("Segmentation XZ"), mdiArea)->setIntersects({"segmentation"});
    newConnectedCVolumeViewer("seg yz", tr("Segmentation YZ"), mdiArea)->setIntersects({"segmentation"});
    newConnectedCVolumeViewer("xy plane", tr("XY / Slices"), mdiArea)->setIntersects({"segmentation"});
    newConnectedCVolumeViewer("segmentation", tr("Surface"), mdiArea)->setIntersects({"seg xz","seg yz"});
    mdiArea->tileSubWindows();

    treeWidgetSurfaces = ui.treeWidgetSurfaces;
    treeWidgetSurfaces->setSelectionMode(QAbstractItemView::ExtendedSelection);
    btnReloadSurfaces = ui.btnReloadSurfaces;

    SurfacePanelController::UiRefs surfaceUi{
        .treeWidget = treeWidgetSurfaces,
        .reloadButton = btnReloadSurfaces,
    };
    _surfacePanel = std::make_unique<SurfacePanelController>(
        surfaceUi,
        _surf_col,
        _viewerManager.get(),
        [this]() { return segmentationViewer(); },
        std::function<void()>{},
        this);
    if (_segmentationGrower) {
        _segmentationGrower->setSurfacePanel(_surfacePanel.get());
    }
    connect(_surfacePanel.get(), &SurfacePanelController::surfacesLoaded, this, [this]() {
        emit sendSurfacesLoaded();
        // Update surface overlay dropdown when surfaces are loaded
        updateSurfaceOverlayDropdown();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceSelectionCleared, this, [this]() {
        clearSurfaceSelection();
    });
    connect(_surfacePanel.get(), &SurfacePanelController::filtersApplied, this, [this](int filterCount) {
        UpdateVolpkgLabel(filterCount);
    });
    connect(_surfacePanel.get(), &SurfacePanelController::copySegmentPathRequested,
            this, [this](const QString& segmentId) {
                if (!fVpkg) {
                    return;
                }
                auto surf = fVpkg->getSurface(segmentId.toStdString());
                if (!surf) {
                    return;
                }
                const QString path = QString::fromStdString(surf->path.string());
                QApplication::clipboard()->setText(path);
                statusBar()->showMessage(tr("Copied segment path to clipboard: %1").arg(path), 3000);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::renderSegmentRequested,
            this, [this](const QString& segmentId) {
                onRenderSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::growSegmentRequested,
            this, [this](const QString& segmentId) {
                onGrowSegmentFromSegment(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::addOverlapRequested,
            this, [this](const QString& segmentId) {
                onAddOverlap(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::neighborCopyRequested,
            this, [this](const QString& segmentId, bool copyOut) {
                onNeighborCopyRequested(segmentId, copyOut);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::reloadFromBackupRequested,
            this, [this](const QString& segmentId, int backupIndex) {
                onReloadFromBackup(segmentId, backupIndex);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::convertToObjRequested,
            this, [this](const QString& segmentId) {
                onConvertToObj(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::cropBoundsRequested,
            this, [this](const QString& segmentId) {
                onCropSurfaceToValidRegion(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::flipURequested,
            this, [this](const QString& segmentId) {
                onFlipSurface(segmentId.toStdString(), true);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::flipVRequested,
            this, [this](const QString& segmentId) {
                onFlipSurface(segmentId.toStdString(), false);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::alphaCompRefineRequested,
            this, [this](const QString& segmentId) {
                onAlphaCompRefine(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::slimFlattenRequested,
            this, [this](const QString& segmentId) {
                onSlimFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::abfFlattenRequested,
            this, [this](const QString& segmentId) {
                onABFFlatten(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::awsUploadRequested,
            this, [this](const QString& segmentId) {
                onAWSUpload(segmentId.toStdString());
            });
    connect(_surfacePanel.get(), &SurfacePanelController::exportTifxyzChunksRequested,
        this, [this](const QString& segmentId) {
            onExportWidthChunks(segmentId.toStdString());
        });

    connect(_surfacePanel.get(), &SurfacePanelController::growSeedsRequested,
            this, [this](const QString& segmentId, bool isExpand, bool isRandomSeed) {
                onGrowSeeds(segmentId.toStdString(), isExpand, isRandomSeed);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::teleaInpaintRequested,
            this, [this]() {
                if (_menuController) {
                    _menuController->triggerTeleaInpaint();
                }
            });
    connect(_surfacePanel.get(), &SurfacePanelController::recalcAreaRequested,
            this, [this](const QStringList& segmentIds) {
                if (segmentIds.isEmpty()) {
                    return;
                }
                std::vector<std::string> ids;
                ids.reserve(segmentIds.size());
                for (const auto& id : segmentIds) {
                    ids.push_back(id.toStdString());
                }
                recalcAreaForSegments(ids);
            });
    connect(_surfacePanel.get(), &SurfacePanelController::statusMessageRequested,
            this, [this](const QString& message, int timeoutMs) {
                statusBar()->showMessage(message, timeoutMs);
            });

    // i recognize that having both a seeding widget and a drawing widget that both handle mouse events and paths is redundant,
    // but i can't find an easy way yet to merge them and maintain the path iteration that the seeding widget currently uses
    // so for now we have both. i suppose i could probably add a 'mode' , but for now i will just hate this section :(

    const auto attachScrollAreaToDock = [](QDockWidget* dock, QWidget* content, const QString& objectName) {
        if (!dock || !content) {
            return;
        }

        auto* container = new QWidget(dock);
        container->setObjectName(objectName);
        auto* layout = new QVBoxLayout(container);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        layout->addWidget(content);
        layout->addStretch(1);

        auto* scrollArea = new QScrollArea(dock);
        scrollArea->setWidgetResizable(true);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setWidget(container);

        dock->setWidget(scrollArea);
    };


    // Create Segmentation widget
    _segmentationWidget = new SegmentationWidget();
    _segmentationWidget->setNormalGridAvailable(_normalGridAvailable);
    _segmentationWidget->setNormalGridPath(_normalGridPath);
    const QString initialHint = _normalGridAvailable
        ? tr("Normal grids directory found.")
        : tr("No volume package loaded.");
    _segmentationWidget->setNormalGridPathHint(initialHint);
    attachScrollAreaToDock(ui.dockWidgetSegmentation, _segmentationWidget, QStringLiteral("dockWidgetSegmentationContent"));

    _segmentationEdit = std::make_unique<SegmentationEditManager>(this);
    _segmentationEdit->setViewerManager(_viewerManager.get());
    _segmentationOverlay = std::make_unique<SegmentationOverlayController>(_surf_col, this);
    _segmentationOverlay->setEditManager(_segmentationEdit.get());
    _segmentationOverlay->setViewerManager(_viewerManager.get());

    _segmentationModule = std::make_unique<SegmentationModule>(
        _segmentationWidget,
        _segmentationEdit.get(),
        _segmentationOverlay.get(),
        _viewerManager.get(),
        _surf_col,
        _point_collection,
        _segmentationWidget->isEditingEnabled(),
        this);

    if (_segmentationModule && _planeSlicingOverlay) {
        QPointer<PlaneSlicingOverlayController> overlayPtr(_planeSlicingOverlay.get());
        _segmentationModule->setRotationHandleHitTester(
            [overlayPtr](CVolumeViewer* viewer, const cv::Vec3f& worldPos) {
                if (!overlayPtr) {
                    return false;
                }
                return overlayPtr->isVolumePointNearRotationHandle(viewer, worldPos, 1.5);
            });
    }

    if (_viewerManager) {
        _viewerManager->setSegmentationOverlay(_segmentationOverlay.get());
    }

    connect(_segmentationModule.get(), &SegmentationModule::editingEnabledChanged,
            this, &CWindow::onSegmentationEditingModeChanged);
    connect(_segmentationModule.get(), &SegmentationModule::statusMessageRequested,
            this, &CWindow::onShowStatusMessage);
    connect(_segmentationModule.get(), &SegmentationModule::stopToolsRequested,
            this, &CWindow::onSegmentationStopToolsRequested);
    connect(_segmentationModule.get(), &SegmentationModule::growthInProgressChanged,
            this, &CWindow::onSegmentationGrowthStatusChanged);
    connect(_segmentationModule.get(), &SegmentationModule::focusPoiRequested,
            this, [this](const cv::Vec3f& position, QuadSurface* base) {
                Q_UNUSED(position);
                applySlicePlaneOrientation(base);
            });
    connect(_segmentationModule.get(), &SegmentationModule::growSurfaceRequested,
            this, &CWindow::onGrowSegmentationSurface);
#ifdef __linux__
    connect(_segmentationModule.get(), &SegmentationModule::approvalMaskSaved,
            this, &CWindow::markSegmentRecentlyEdited);
#endif

    SegmentationGrower::Context growerContext{
        _segmentationModule.get(),
        _segmentationWidget,
        _surf_col,
        _viewerManager.get(),
        chunk_cache
    };
    SegmentationGrower::UiCallbacks growerCallbacks{
        [this](const QString& text, int timeout) {
            if (statusBar()) {
                statusBar()->showMessage(text, timeout);
            }
        },
        [this](QuadSurface* surface) {
            applySlicePlaneOrientation(surface);
        }
    };
    _segmentationGrower = std::make_unique<SegmentationGrower>(growerContext, growerCallbacks, this);

    connect(_segmentationWidget, &SegmentationWidget::volumeSelectionChanged, this, [this](const QString& volumeId) {
        if (!fVpkg) {
            statusBar()->showMessage(tr("No volume package loaded."), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!_segmentationGrowthVolumeId.empty()
                                                                   ? _segmentationGrowthVolumeId
                                                                   : currentVolumeId);
                _segmentationWidget->setActiveVolume(fallbackId);
            }
            return;
        }

        const std::string requestedId = volumeId.toStdString();
        try {
            auto vol = fVpkg->volume(requestedId);
            _segmentationGrowthVolumeId = requestedId;
            // Set volume zarr path for neural tracing
            if (_segmentationWidget && vol) {
                _segmentationWidget->setVolumeZarrPath(QString::fromStdString(vol->path().string()));
            }
            statusBar()->showMessage(tr("Using volume '%1' for surface growth.").arg(volumeId), 2500);
        } catch (const std::out_of_range&) {
            statusBar()->showMessage(tr("Volume '%1' not found in this package.").arg(volumeId), 4000);
            if (_segmentationWidget) {
                const QString fallbackId = QString::fromStdString(!currentVolumeId.empty()
                                                                   ? currentVolumeId
                                                                   : std::string{});
                _segmentationWidget->setActiveVolume(fallbackId);
                _segmentationGrowthVolumeId = currentVolumeId;
            }
        }
    });

    // Create Drawing widget
    _drawingWidget = new DrawingWidget();
    attachScrollAreaToDock(ui.dockWidgetDrawing, _drawingWidget, QStringLiteral("dockWidgetDrawingContent"));

    connect(this, &CWindow::sendVolumeChanged, _drawingWidget,
            static_cast<void (DrawingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&DrawingWidget::onVolumeChanged));
    connect(_drawingWidget, &DrawingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(this, &CWindow::sendSurfacesLoaded, _drawingWidget, &DrawingWidget::onSurfacesLoaded);

    _drawingWidget->setCache(chunk_cache);

    // Create Seeding widget
    _seedingWidget = new SeedingWidget(_point_collection, _surf_col);
    attachScrollAreaToDock(ui.dockWidgetDistanceTransform, _seedingWidget, QStringLiteral("dockWidgetDistanceTransformContent"));

    connect(this, &CWindow::sendVolumeChanged, _seedingWidget,
            static_cast<void (SeedingWidget::*)(std::shared_ptr<Volume>, const std::string&)>(&SeedingWidget::onVolumeChanged));
    connect(_seedingWidget, &SeedingWidget::sendStatusMessageAvailable, this, &CWindow::onShowStatusMessage);
    connect(this, &CWindow::sendSurfacesLoaded, _seedingWidget, &SeedingWidget::onSurfacesLoaded);

    _seedingWidget->setCache(chunk_cache);

    // Create and add the point collection widget
    _point_collection_widget = new CPointCollectionWidget(_point_collection, this);
    _point_collection_widget->setObjectName("pointCollectionDock");
    addDockWidget(Qt::RightDockWidgetArea, _point_collection_widget);

    // Selection dock (removed per request; selection actions remain in the menu)
    if (_viewerManager) {
        _viewerManager->forEachViewer([this](CVolumeViewer* viewer) {
            configureViewerConnections(viewer);
        });
    }
    connect(_point_collection_widget, &CPointCollectionWidget::pointDoubleClicked, this, &CWindow::onPointDoubleClicked);
    connect(_point_collection_widget, &CPointCollectionWidget::convertPointToAnchorRequested, this, &CWindow::onConvertPointToAnchor);

    // Tab the docks - keep Segmentation, Seeding, Point Collections, and Drawing together
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDistanceTransform);
    tabifyDockWidget(ui.dockWidgetSegmentation, _point_collection_widget);
    tabifyDockWidget(ui.dockWidgetSegmentation, ui.dockWidgetDrawing);

    // Make Drawing dock the active tab by default
    ui.dockWidgetDrawing->raise();

    // Keep the view-related docks on the left and grouped together as tabs
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetView);
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetOverlay);
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetRenderSettings);
    addDockWidget(Qt::LeftDockWidgetArea, ui.dockWidgetComposite);

    auto ensureTabified = [this](QDockWidget* primary, QDockWidget* candidate) {
        const auto currentTabs = tabifiedDockWidgets(primary);
        const bool alreadyTabified = std::find(currentTabs.cbegin(), currentTabs.cend(), candidate) != currentTabs.cend();
        if (!alreadyTabified) {
            tabifyDockWidget(primary, candidate);
        }
    };

    ensureTabified(ui.dockWidgetView, ui.dockWidgetOverlay);
    ensureTabified(ui.dockWidgetView, ui.dockWidgetRenderSettings);
    ensureTabified(ui.dockWidgetView, ui.dockWidgetComposite);
    ensureTabified(ui.dockWidgetView, ui.dockWidgetPostprocessing);

    const auto tabOrder = tabifiedDockWidgets(ui.dockWidgetView);
    for (QDockWidget* dock : tabOrder) {
        tabifyDockWidget(ui.dockWidgetView, dock);
    }

    ui.dockWidgetView->show();
    ui.dockWidgetView->raise();
    QTimer::singleShot(0, this, [this]() {
        if (ui.dockWidgetView) {
            ui.dockWidgetView->raise();
        }
    });

    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivated,
            this, &CWindow::onSurfaceActivated);
    connect(_surfacePanel.get(), &SurfacePanelController::surfaceActivatedPreserveEditing,
            this, &CWindow::onSurfaceActivatedPreserveEditing);

    // View controls and composite controls setup extracted to CWindowViewControlSetup.cpp
    setupViewControls();
    setupCompositeControls();
}

// Create menus
// Create actions
void CWindow::keyPressEvent(QKeyEvent* event)
{
    if (event->key() == Qt::Key_Space && event->modifiers() == Qt::NoModifier) {
        toggleVolumeOverlayVisibility();
        event->accept();
        return;
    }

    if (event->key() == Qt::Key_R && event->modifiers() == Qt::NoModifier) {
        if (centerFocusOnCursor()) {
            event->accept();
            return;
        }
    }

    if (event->key() == Qt::Key_F) {
        if (event->modifiers() == Qt::NoModifier) {
            stepFocusHistory(-1);
            event->accept();
            return;
        } else if (event->modifiers() == Qt::ControlModifier) {
            stepFocusHistory(1);
            event->accept();
            return;
        }
    }

    // Shift+G decreases slice step size, Shift+H increases it
    if (event->modifiers() == Qt::ShiftModifier && _viewerManager) {
        if (event->key() == Qt::Key_G) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::max(1, currentStep - 1);
            _viewerManager->setSliceStepSize(newStep);
            onSliceStepSizeChanged(newStep);
            event->accept();
            return;
        } else if (event->key() == Qt::Key_H) {
            int currentStep = _viewerManager->sliceStepSize();
            int newStep = std::min(100, currentStep + 1);
            _viewerManager->setSliceStepSize(newStep);
            onSliceStepSizeChanged(newStep);
            event->accept();
            return;
        }
    }

    if (_segmentationModule && _segmentationModule->handleKeyPress(event)) {
        return;
    }

    QMainWindow::keyPressEvent(event);
}

void CWindow::keyReleaseEvent(QKeyEvent* event)
{
    if (_segmentationModule && _segmentationModule->handleKeyRelease(event)) {
        return;
    }

    QMainWindow::keyReleaseEvent(event);
}

void CWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    scheduleWindowStateSave();
}

void CWindow::scheduleWindowStateSave()
{
    // Restart the timer - this debounces rapid changes
    if (_windowStateSaveTimer) {
        _windowStateSaveTimer->start();
    }
}

void CWindow::saveWindowState()
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::window::GEOMETRY, saveGeometry());
    settings.setValue(vc3d::settings::window::STATE, saveState());
    settings.sync();
}

// Asks User to Save Data Prior to VC.app Exit
void CWindow::closeEvent(QCloseEvent* event)
{
    saveWindowState();
    std::quick_exit(0);
}

void CWindow::setWidgetsEnabled(bool state)
{
    ui.grpVolManager->setEnabled(state);
    if (_volumeWindowWidget) {
        _volumeWindowWidget->setControlsEnabled(state);
    }
    if (_overlayWindowWidget) {
        const bool hasOverlay = _volumeOverlay && _volumeOverlay->hasOverlaySelection();
        _overlayWindowWidget->setControlsEnabled(state && hasOverlay);
    }
}

auto CWindow::InitializeVolumePkg(const std::string& nVpkgPath) -> bool
{
    fVpkg = nullptr;
    updateNormalGridAvailability();
    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        _segmentationModule->setEditingEnabled(false);
    }
    if (_segmentationWidget) {
        if (!_segmentationModule || _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }
        _segmentationWidget->setAvailableVolumes({}, QString());
        _segmentationWidget->setVolumePackagePath(QString());
    }

    try {
        fVpkg = VolumePkg::New(nVpkgPath);
    } catch (const std::exception& e) {
        Logger()->error("Failed to initialize volpkg: {}", e.what());
    }

    if (fVpkg == nullptr) {
        Logger()->error("Cannot open .volpkg: {}", nVpkgPath);
        QMessageBox::warning(
            this, "Error",
            "Volume package failed to load. Package might be corrupt. Check the console log for a detailed error message.");
        return false;
    }
    return true;
}

// Update the widgets
void CWindow::UpdateView(void)
{
    if (fVpkg == nullptr) {
        setWidgetsEnabled(false);  // Disable Widgets for User
        ui.lblVpkgName->setText("[ No Volume Package Loaded ]");
        return;
    }

    setWidgetsEnabled(true);  // Enable Widgets for User

    // show volume package name
    UpdateVolpkgLabel(0);

    volSelect->setEnabled(can_change_volume_());

    update();
}

void CWindow::UpdateVolpkgLabel(int filterCounter)
{
    if (!fVpkg) {
        return;
    }
    QString label = tr("%1").arg(QString::fromStdString(fVpkg->name()));
    ui.lblVpkgName->setText(label);
}

void CWindow::onShowStatusMessage(const QString& text, int timeout)
{
    statusBar()->showMessage(text, timeout);
}

void CWindow::onSegmentationGrowthStatusChanged(bool running)
{
    if (!statusBar()) {
        return;
    }

    if (_surfacePanel) {
        _surfacePanel->setSelectionLocked(running);
    }

    if (running) {
        if (!_segmentationGrowthWarning) {
            _segmentationGrowthWarning = new QLabel(statusBar());
            _segmentationGrowthWarning->setObjectName(QStringLiteral("segmentationGrowthWarning"));
            _segmentationGrowthWarning->setStyleSheet(QStringLiteral("color: #c62828; font-weight: 600;"));
            _segmentationGrowthWarning->setContentsMargins(8, 0, 8, 0);
            _segmentationGrowthWarning->setAlignment(Qt::AlignCenter);
            _segmentationGrowthWarning->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
            _segmentationGrowthWarning->setMinimumWidth(260);
            _segmentationGrowthWarning->hide();
            statusBar()->addPermanentWidget(_segmentationGrowthWarning, 1);
        }
        _segmentationGrowthStatusText = tr("Surface growth in progress - surface selection locked");
        _segmentationGrowthWarning->setText(_segmentationGrowthStatusText);
        _segmentationGrowthWarning->setVisible(true);
        statusBar()->showMessage(_segmentationGrowthStatusText, 0);
    } else if (_segmentationGrowthWarning) {
        _segmentationGrowthWarning->clear();
        _segmentationGrowthWarning->setVisible(false);
        if (statusBar()->currentMessage() == _segmentationGrowthStatusText) {
            statusBar()->clearMessage();
        }
        _segmentationGrowthStatusText.clear();
    }
}

void CWindow::onSliceStepSizeChanged(int newSize)
{
    // Update status bar label
    if (_sliceStepLabel) {
        _sliceStepLabel->setText(tr("Step: %1").arg(newSize));
    }

    // Save to settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SLICE_STEP_SIZE, newSize);

    // Update View dock widget spinbox
    if (auto* spinSliceStep = ui.spinSliceStepSize) {
        QSignalBlocker blocker(spinSliceStep);
        spinSliceStep->setValue(newSize);
    }
}

std::filesystem::path seg_path_name(const std::filesystem::path &path)
{
    std::string name;
    bool store = false;
    for(auto elm : path) {
        if (store)
            name += "/"+elm.string();
        else if (elm == "paths")
            store = true;
    }
    name.erase(0,1);
    return name;
}

// Note: OpenVolume, CloseVolume, and can_change_volume_ are in CWindowVolumeManagement.cpp

// Handle request to step impact range down
void CWindow::onLocChanged(void)
{
    // std::cout << "loc changed!" << "\n";

}

void CWindow::onVolumeClicked(const cv::Vec3f& vol_loc, const cv::Vec3f& normal, Surface *surf, Qt::MouseButton buttons, Qt::KeyboardModifiers modifiers)
{
    if (modifiers & Qt::ShiftModifier) {
        return;
    }
    else if (modifiers & Qt::ControlModifier) {
        std::cout << "clicked on vol loc " << vol_loc << "\n";
        // Get the surface ID from the surface collection
        std::string surfId;
        if (_surf_col && surf) {
            surfId = _surf_col->findSurfaceId(surf);
        }
        centerFocusAt(vol_loc, normal, surfId, true);
    }
    else {
    }
}

void CWindow::onManualPlaneChanged(void)
{
    cv::Vec3f normal;

    for(int i=0;i<3;i++) {
        normal[i] = spNorm[i]->value();
    }

    auto planeShared = _surf_col->surface("manual plane");
    PlaneSurface *plane = dynamic_cast<PlaneSurface*>(planeShared.get());

    if (!plane)
        return;

    plane->setNormal(normal);
    _surf_col->setSurface("manual plane", planeShared);
}

void CWindow::onSurfaceActivated(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _surfID;
    _surfID = surfaceId.toStdString();

    // Look up the shared_ptr by ID
    if (fVpkg && !_surfID.empty()) {
        _surf_weak = fVpkg->getSurface(_surfID);
    } else {
        _surf_weak.reset();
    }

    auto surf = _surf_weak.lock();

    if (_surfID != previousSurfId) {
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            _segmentationModule->setEditingEnabled(false);
        } else if (_segmentationWidget && _segmentationWidget->isEditingEnabled()) {
            _segmentationWidget->setEditingEnabled(false);
        }

        // Handle approval mask when switching segments
        if (_segmentationModule) {
            _segmentationModule->onActiveSegmentChanged(surf.get());
        }
    }

    if (surf) {
        applySlicePlaneOrientation(surf.get());
    } else {
        applySlicePlaneOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::onSurfaceActivatedPreserveEditing(const QString& surfaceId, QuadSurface* surface)
{
    const std::string previousSurfId = _surfID;
    _surfID = surfaceId.toStdString();

    if (fVpkg && !_surfID.empty()) {
        _surf_weak = fVpkg->getSurface(_surfID);
    } else {
        _surf_weak.reset();
    }

    auto surf = _surf_weak.lock();

    if (_surfID != previousSurfId && _segmentationModule) {
        _segmentationModule->onActiveSegmentChanged(surf.get());

        const bool wantsEditing = _segmentationWidget && _segmentationWidget->isEditingEnabled();
        if (wantsEditing) {
            if (!_segmentationModule->editingEnabled()) {
                _segmentationModule->setEditingEnabled(true);
            } else if (_surf_col) {
                auto targetSurface = surf;
                if (!targetSurface) {
                    targetSurface = std::dynamic_pointer_cast<QuadSurface>(_surf_col->surface("segmentation"));
                }

                if (targetSurface) {
                    _segmentationModule->endEditingSession();
                    if (_segmentationModule->beginEditingSession(targetSurface) && _viewerManager) {
                        _viewerManager->forEachViewer([](CVolumeViewer* viewer) {
                            if (viewer) {
                                viewer->clearOverlayGroup("segmentation_radius_indicator");
                            }
                        });
                    }
                }
            }
        }
    }

    if (surf) {
        applySlicePlaneOrientation(surf.get());
    } else {
        applySlicePlaneOrientation();
    }

    if (_surfacePanel && _surfacePanel->isCurrentOnlyFilterEnabled()) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::onSurfaceWillBeDeleted(const std::string& name, const std::shared_ptr<Surface>& surf)
{
    // Called BEFORE surface deletion - clear all references to prevent use-after-free

    // Clear if this is our current active surface
    auto currentSurf = _surf_weak.lock();
    if (currentSurf && currentSurf == surf) {
        _surf_weak.reset();
        _surfID.clear();
    }

    // Focus history uses string IDs now, so no cleanup needed for surface pointers
    // (the ID remains valid for lookup - will just return nullptr if surface is gone)
}

// Note: onEditMaskPressed and onAppendMaskPressed are in CWindowMaskOperations.cpp

QString CWindow::getCurrentVolumePath() const
{
    if (currentVolume == nullptr) {
        return QString();
    }
    return QString::fromStdString(currentVolume->path().string());
}

void CWindow::onSegmentationDirChanged(int index)
{
    if (!fVpkg || index < 0 || !cmbSegmentationDir) {
        return;
    }

    std::string newDir = cmbSegmentationDir->itemText(index).toStdString();

    // Only reload if the directory actually changed
    if (newDir != fVpkg->getSegmentationDirectory()) {
        // Clear the current segmentation surface first to ensure viewers update
        _surf_col->setSurface("segmentation", nullptr, true);

        // Clear current surface selection
        _surf_weak.reset();
        _surfID.clear();
        treeWidgetSurfaces->clearSelection();

        if (_surfacePanel) {
            _surfacePanel->resetTagUi();
        }

        // Set the new directory in the VolumePkg
        fVpkg->setSegmentationDirectory(newDir);

        // Reset stride user override so tiered defaults apply to new directory
        if (_viewerManager) {
            _viewerManager->resetStrideUserOverride();
        }
        if (_surfacePanel) {
            _surfacePanel->loadSurfaces(false);
        }

        // Update the status bar to show the change
        statusBar()->showMessage(tr("Switched to %1 directory").arg(QString::fromStdString(newDir)), 3000);
    }
}


void CWindow::onManualLocationChanged()
{
    // Check if we have a valid volume loaded
    if (!currentVolume) {
        return;
    }

    // Parse the comma-separated values
    QString text = lblLocFocus->text().trimmed();
    QStringList parts = text.split(',');

    // Validate we have exactly 3 parts
    if (parts.size() != 3) {
        // Invalid input - restore the previous values
        POI* poi = _surf_col->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Parse each coordinate
    bool ok[3];
    int x = parts[0].trimmed().toInt(&ok[0]);
    int y = parts[1].trimmed().toInt(&ok[1]);
    int z = parts[2].trimmed().toInt(&ok[2]);

    // Validate the input
    if (!ok[0] || !ok[1] || !ok[2]) {
        // Invalid input - restore the previous values
        POI* poi = _surf_col->poi("focus");
        if (poi) {
            lblLocFocus->setText(QString("%1, %2, %3")
                .arg(static_cast<int>(poi->p[0]))
                .arg(static_cast<int>(poi->p[1]))
                .arg(static_cast<int>(poi->p[2])));
        }
        return;
    }

    // Clamp values to volume bounds
    auto [w, h, d] = currentVolume->shape();

    x = std::max(0, std::min(x, w - 1));
    y = std::max(0, std::min(y, h - 1));
    z = std::max(0, std::min(z, d - 1));

    // Update the line edit with clamped values
    lblLocFocus->setText(QString("%1, %2, %3").arg(x).arg(y).arg(z));

    // Update the focus POI
    POI* poi = _surf_col->poi("focus");
    if (!poi) {
        poi = new POI;
    }

    poi->p = cv::Vec3f(x, y, z);
    poi->n = cv::Vec3f(0, 0, 1); // Default normal for XY plane

    _surf_col->setPOI("focus", poi);

    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
}

// Note: User interaction handlers are in CWindowUserInteraction.cpp
// (onZoomIn, onZoomOut, onCopyCoordinates, onFocusPOIChanged, onPointDoubleClicked, onConvertPointToAnchor)

void CWindow::onResetAxisAlignedRotations()
{
    _axisAlignedSegXZRotationDeg = 0.0f;
    _axisAlignedSegYZRotationDeg = 0.0f;
    _axisAlignedSliceDrags.clear();
    applySlicePlaneOrientation();
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->refreshAll();
    }
    statusBar()->showMessage(tr("All plane rotations reset"), 2000);
}

void CWindow::onAxisOverlayVisibilityToggled(bool enabled)
{
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && _useAxisAlignedSlices);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        spinAxisOverlayOpacity->setEnabled(_useAxisAlignedSlices && enabled);
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::SHOW_AXIS_OVERLAYS, enabled ? "1" : "0");
}

void CWindow::onAxisOverlayOpacityChanged(int value)
{
    float normalized = std::clamp(static_cast<float>(value) / 100.0f, 0.0f, 1.0f);
    if (_planeSlicingOverlay) {
        _planeSlicingOverlay->setAxisAlignedOverlayOpacity(normalized);
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::AXIS_OVERLAY_OPACITY, value);
}

// Note: recalcAreaForSegments is in CWindowAreaCalculation.cpp

void CWindow::onAxisAlignedSlicesToggled(bool enabled)
{
    _useAxisAlignedSlices = enabled;
    if (enabled) {
        _axisAlignedSegXZRotationDeg = 0.0f;
        _axisAlignedSegYZRotationDeg = 0.0f;
    }
    _axisAlignedSliceDrags.clear();
    qCDebug(lcAxisSlices) << "Axis-aligned slices" << (enabled ? "enabled" : "disabled");
    if (_planeSlicingOverlay) {
        bool overlaysVisible = !ui.chkAxisOverlays || ui.chkAxisOverlays->isChecked();
        _planeSlicingOverlay->setAxisAlignedEnabled(enabled && overlaysVisible);
    }
    if (auto* spinAxisOverlayOpacity = ui.spinAxisOverlayOpacity) {
        spinAxisOverlayOpacity->setEnabled(enabled && (!ui.chkAxisOverlays || ui.chkAxisOverlays->isChecked()));
    }
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    settings.setValue(vc3d::settings::viewer::USE_AXIS_ALIGNED_SLICES, enabled ? "1" : "0");
    updateAxisAlignedSliceInteraction();
    applySlicePlaneOrientation();
}

// Note: Segmentation operation handlers are in CWindowSegmentationOperations.cpp
// (onSegmentationEditingModeChanged, onSegmentationStopToolsRequested, onGrowSegmentationSurface, onMoveSegmentToPaths)

// Note: Axis-aligned slice methods are in CWindowAxisAlignedSlices.cpp

// Note: inotify file watcher code is in CWindowFileWatcher.cpp

// Note: Surface overlay methods are in CWindowSurfaceOverlay.cpp
