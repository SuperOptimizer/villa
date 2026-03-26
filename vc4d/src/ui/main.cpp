#include "vc4d/ui/main_window.hpp"

#include <QApplication>

auto main(int argc, char* argv[]) -> int
{
    QApplication app(argc, argv);
    QApplication::setOrganizationName("Vesuvius Challenge");
    QApplication::setApplicationName("VC4D");
    QApplication::setApplicationVersion("4.0.0");

    vc4d::MainWindow window;
    window.show();

    return QApplication::exec();
}
