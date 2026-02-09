#include <qapplication.h>
#include <QCommandLineParser>

#include "CWindow.hpp"
#include "vc/core/Version.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

#include <opencv2/core.hpp>
#include <iostream>
#include <thread>
#include <omp.h>
#include <dlfcn.h>

// Set env vars before the OMP runtime initializes (runs before main).
// OMP_WAIT_POLICY=passive: threads sleep instead of spin-waiting.
// Eliminates ~79% idle CPU overhead from OMP sched_yield spinning.
// KMP_BLOCKTIME=0: libomp-specific equivalent (immediate sleep).
// OPENBLAS_NUM_THREADS=1: no need for BLAS parallelism in GUI.
__attribute__((constructor))
static void setOmpEnvEarly()
{
    setenv("OMP_WAIT_POLICY", "passive", /*overwrite=*/0);
    setenv("KMP_BLOCKTIME", "0", /*overwrite=*/0);
    setenv("OPENBLAS_NUM_THREADS", "1", /*overwrite=*/0);
}

auto main(int argc, char* argv[]) -> int
{
    cv::setNumThreads(std::thread::hardware_concurrency());
    // kmp_set_blocktime(0): libomp-specific, make threads sleep immediately.
    // Use dlsym so we don't get a link error when building with GCC/libgomp.
    if (auto fn = reinterpret_cast<void(*)(int)>(dlsym(RTLD_DEFAULT, "kmp_set_blocktime")))
        fn(0);

    // Workaround for Qt dock widget issues on Wayland (QTBUG-87332)
    // Floating dock widgets become unmovable after initial drag on Wayland.
    // Force XCB (X11/XWayland) platform to restore full functionality.
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM")) {
        const char* waylandDisplay = qgetenv("WAYLAND_DISPLAY").constData();
        if (waylandDisplay && *waylandDisplay) {
            qputenv("QT_QPA_PLATFORM", "xcb");
        }
    }

    QApplication app(argc, argv);
    QApplication::setOrganizationName("Vesuvius Challenge");
    QApplication::setApplicationName("VC3D");
    QApplication::setWindowIcon(QIcon(":/images/logo.png"));
    QApplication::setApplicationVersion(QString::fromStdString(ProjectInfo::VersionString()));
    std::cout << "VC3D commit: " << ProjectInfo::RepositoryHash() << std::endl;

    QCommandLineParser parser;
    parser.setApplicationDescription("VC3D - Volume Cartographer 3D Viewer");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption skipShapeCheckOption(
        "skip-shape-check",
        "Skip validation of zarr shape against meta.json dimensions");
    parser.addOption(skipShapeCheckOption);

    QCommandLineOption loadFirstOption(
        "load-first",
        "Load segmentations from the specified directory first and defer others (e.g. paths or traces).",
        "dir");
    parser.addOption(loadFirstOption);

    QCommandLineOption cacheSizeOption(
        "cache-size",
        QString("Set the chunk cache size in gigabytes (default: %1 GB).")
            .arg(CHUNK_CACHE_SIZE_GB),
        "GB",
        QString::number(CHUNK_CACHE_SIZE_GB));
    parser.addOption(cacheSizeOption);

    parser.process(app);

    if (parser.isSet(skipShapeCheckOption)) {
        Volume::skipShapeCheck = true;
    }
    if (parser.isSet(loadFirstOption)) {
        QString loadFirstDir = parser.value(loadFirstOption).trimmed().toLower();
        if (!loadFirstDir.isEmpty()) {
            VolumePkg::setLoadFirstSegmentationDirectory(loadFirstDir.toStdString());
        }
    }

    size_t cacheSizeGB = CHUNK_CACHE_SIZE_GB;
    if (parser.isSet(cacheSizeOption)) {
        bool ok = false;
        const qulonglong parsed = parser.value(cacheSizeOption).toULongLong(&ok);
        if (!ok || parsed == 0) {
            std::cerr << "Error: Invalid cache size. Must be a positive integer (GB)." << std::endl;
            return 1;
        }
        if (parsed > 256) {
            std::cerr << "Warning: Cache size " << parsed
                      << " GB is very large. Ensure sufficient system memory." << std::endl;
        }
        cacheSizeGB = static_cast<size_t>(parsed);
    }

    CWindow aWin(cacheSizeGB);
    aWin.show();
    return QApplication::exec();
}
