// main.cpp
// Chao Du 2014 Dec

#include <QScreen>
#include <qapplication.h>

#include "CWindow.hpp"
#include "vc/core/Version.hpp"

#include <opencv2/core.hpp>
#include <thread>

using namespace ChaoVis;

namespace vc = volcart;

auto main(int argc, char* argv[]) -> int
{
    cv::setNumThreads(std::thread::hardware_concurrency());
    
    QApplication app(argc, argv);
    QApplication::setOrganizationName("EduceLab");
    QApplication::setApplicationName("VC");
    QApplication::setApplicationVersion(
        QString::fromStdString(vc::ProjectInfo::VersionString()));

    qRegisterMetaType<CWindow::Segmenter>("Segmenter");
    qRegisterMetaType<CWindow::Segmenter::PointSet>(
        "CWindow::Segmenter::PointSet");

    CWindow aWin;
    aWin.show();
    return QApplication::exec();
}
