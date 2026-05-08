#include "CWindow.hpp"
#include <iostream>
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "SegmentationCommandHandler.hpp"

#include <QDialogButtonBox>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QMessageBox>
#include <QStandardPaths>
#include <QtConcurrent>
#include <QProgressDialog>

#include "CommandLineToolRunner.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/flattening/ABFFlattening.hpp"
#include "ToolDialogs.hpp"
#include "LasagnaServiceManager.hpp"
#include "elements/VolumeSelector.hpp"
#include "elements/JsonProfilePresets.hpp"
#include "utils/Json.hpp"

// --------- locate generic vc_* executables -----------------------------------
static bool isExecutableFile(const QString& path)
{
    QFileInfo fi(path);
    return fi.exists() && fi.isFile() && fi.isExecutable();
}

static QStringList applicationRelativeExecutablePaths(const QString& name)
{
#ifdef _WIN32
    const QString executableName = name.endsWith(QStringLiteral(".exe"), Qt::CaseInsensitive)
        ? name
        : name + QStringLiteral(".exe");
#else
    const QString executableName = name;
#endif

    const QString appDir = QCoreApplication::applicationDirPath();
    QStringList candidates{
        QDir(appDir).filePath(executableName),
        QDir(appDir).filePath(QStringLiteral("../") + executableName),
        QDir(appDir).filePath(QStringLiteral("../bin/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../../bin/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../libexec/") + executableName),
        QDir(appDir).filePath(QStringLiteral("../Resources/bin/") + executableName),
    };
    candidates.removeDuplicates();
    return candidates;
}

static QString findVcTool(const char* name)
{
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString key1 = QStringLiteral("tools/%1_path").arg(name);
    const QString key2 = QStringLiteral("tools/%1").arg(name);
    const QString iniPath =
        settings.value(key1, settings.value(key2)).toString().trimmed();
    if (!iniPath.isEmpty() && isExecutableFile(iniPath)) {
        return QFileInfo(iniPath).absoluteFilePath();
    }

    for (const QString& candidate : applicationRelativeExecutablePaths(QString::fromLatin1(name))) {
        if (isExecutableFile(candidate)) {
            return QFileInfo(candidate).absoluteFilePath();
        }
    }

#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    const QString onPath = QStandardPaths::findExecutable(QString::fromLatin1(name));
    if (!onPath.isEmpty()) return onPath;
#else
    const QStringList pathDirs =
        QProcessEnvironment::systemEnvironment().value("PATH")
            .split(QDir::listSeparator(), Qt::SkipEmptyParts);
    for (const QString& dir : pathDirs) {
        const QString candidate = QDir(dir).filePath(QString::fromLatin1(name));
        QFileInfo fi(candidate);
        if (fi.exists() && fi.isFile() && fi.isExecutable())
            return fi.absoluteFilePath();
    }
#endif
    return {};
}


// ====================== CWindow member functions ==============================

void CWindow::onVisLasagnaObj(const std::string& segmentId)
{
    if (!_state || !_state->vpkg()) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot export vis: No volume package loaded"));
        return;
    }

    auto surf = _state->vpkg()->getSurface(segmentId);
    if (!surf) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot export vis: Invalid segment"));
        return;
    }

    auto& mgr = LasagnaServiceManager::instance();
    if (!mgr.isRunning()) {
        QMessageBox::warning(this, tr("Error"), tr("Lasagna service is not running"));
        return;
    }

    // Find model.pt in the surface directory
    std::filesystem::path surfPath = surf->path;
    std::filesystem::path modelPath = surfPath / "model.pt";
    if (!std::filesystem::exists(modelPath)) {
        QMessageBox::warning(this, tr("Error"),
            tr("No model.pt found in %1").arg(QString::fromStdString(surfPath.string())));
        return;
    }

    // Default output directory next to the surface
    QString defaultOutput = QString::fromStdString((surfPath / "vis_obj").string());

    VisLasagnaObjDialog dlg(this, defaultOutput);
    if (dlg.exec() != QDialog::Accepted) {
        statusBar()->showMessage(tr("Vis as OBJ cancelled"), 3000);
        return;
    }

    // Build request JSON
    QJsonObject request;

    // Always read model file and send as base64 (server uses a tempdir)
    QFile modelFile(QString::fromStdString(modelPath.string()));
    if (!modelFile.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Error"),
            tr("Cannot read model file: %1").arg(QString::fromStdString(modelPath.string())));
        return;
    }
    QByteArray modelBytes = modelFile.readAll();
    modelFile.close();
    request[QStringLiteral("model_data")] = QString::fromLatin1(modelBytes.toBase64());

    // data_input is extracted from the checkpoint's _fit_config_ on the Python side.
    // No need to send it explicitly.

    // output_dir is where the client unpacks the results (not sent to server)
    request[QStringLiteral("output_dir")] = dlg.outputDir();

    QJsonArray slicesArr;
    for (const auto& s : dlg.slices()) slicesArr.append(s);
    request[QStringLiteral("slices")] = slicesArr;

    QJsonArray chansArr;
    for (const auto& c : dlg.channels()) chansArr.append(c);
    request[QStringLiteral("channels")] = chansArr;

    QJsonArray lossArr;
    for (const auto& l : dlg.losses()) lossArr.append(l);
    request[QStringLiteral("losses")] = lossArr;

    request[QStringLiteral("include_mesh")] = dlg.includeMesh();
    request[QStringLiteral("include_connections")] = dlg.includeConnections();

    // Connect result signals (one-shot)
    auto connFinish = std::make_shared<QMetaObject::Connection>();
    auto connError = std::make_shared<QMetaObject::Connection>();
    *connFinish = connect(&mgr, &LasagnaServiceManager::visExportFinished,
            this, [this, connFinish, connError](const QString& outputDir) {
                disconnect(*connFinish);
                disconnect(*connError);
                statusBar()->showMessage(
                    tr("Vis OBJ export finished: %1").arg(outputDir), 5000);
            });
    *connError = connect(&mgr, &LasagnaServiceManager::visExportError,
            this, [this, connFinish, connError](const QString& message) {
                disconnect(*connFinish);
                disconnect(*connError);
                QMessageBox::warning(this, tr("Export Error"), message);
            });

    mgr.exportLasagnaVis(request);
    statusBar()->showMessage(
        tr("Exporting vis OBJ for %1...").arg(QString::fromStdString(segmentId)), 3000);
}

bool CWindow::initializeCommandLineRunner()
{
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);

        if (_segmentationCommandHandler) {
            _segmentationCommandHandler->setCmdRunner(_cmdRunner);
        }

        connect(_cmdRunner, &CommandLineToolRunner::toolStarted,
                [this](CommandLineToolRunner::Tool /*tool*/, const QString& message) {
                    statusBar()->showMessage(message, 0);
                });
        connect(_cmdRunner, &CommandLineToolRunner::toolFinished,
                [this](CommandLineToolRunner::Tool tool, bool success, const QString& message,
                       const QString& outputPath, bool copyToClipboard) {
                    Q_UNUSED(outputPath);
                    const bool neighborJobActive = _segmentationCommandHandler &&
                        _segmentationCommandHandler->neighborCopyJob().has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;
                    const bool resumeLocalActive = _segmentationCommandHandler &&
                        !_segmentationCommandHandler->neighborCopyJob() &&
                        _segmentationCommandHandler->resumeLocalJob().has_value() &&
                        tool == CommandLineToolRunner::Tool::NeighborCopy;

                    bool suppressDialogs = neighborJobActive && success &&
                                           _segmentationCommandHandler->neighborCopyJob()->stage ==
                                               SegmentationCommandHandler::NeighborCopyJob::Stage::FirstPass;

                    if (!suppressDialogs) {
                        if (success) {
                            QString displayMsg = message;
                            if (copyToClipboard) displayMsg += tr(" - Path copied to clipboard");
                            statusBar()->showMessage(displayMsg, 5000);
                            QMessageBox::information(this, tr("Operation Complete"), displayMsg);
                        } else {
                            statusBar()->showMessage(tr("Operation failed"), 5000);
                            QMessageBox::critical(this, tr("Error"), message);
                        }
                    } else {
                        statusBar()->showMessage(tr("Neighbor copy pass 1 complete"), 2000);
                    }

                    if (neighborJobActive) {
                        _segmentationCommandHandler->handleNeighborCopyToolFinished(success);
                    }
                    if (resumeLocalActive) {
                        if (success && _surfacePanel) {
                            _surfacePanel->reloadSurfacesFromDisk();
                        }
                        _cmdRunner->setOmpThreads(-1);
                        _segmentationCommandHandler->resumeLocalJob().reset();
                    }
                });
    }
    return true;
}

#include "CWindowContextMenu.moc"
