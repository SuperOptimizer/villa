#include "CWindow.hpp"
#include "SurfacePanelController.hpp"
#include "VCSettings.hpp"
#include "SegmentationCommandHandler.hpp"

#include <QSettings>
#include <QMessageBox>

#include "CommandLineToolRunner.hpp"

// ====================== CWindow member functions ==============================

bool CWindow::initializeCommandLineRunner()
{
    if (!_cmdRunner) {
        _cmdRunner = new CommandLineToolRunner(statusBar(), this, this);

        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        int parallelProcesses = settings.value(vc3d::settings::perf::PARALLEL_PROCESSES,
                                               vc3d::settings::perf::PARALLEL_PROCESSES_DEFAULT).toInt();
        int iterationCount = settings.value(vc3d::settings::perf::ITERATION_COUNT,
                                            vc3d::settings::perf::ITERATION_COUNT_DEFAULT).toInt();

        _cmdRunner->setParallelProcesses(parallelProcesses);
        _cmdRunner->setIterationCount(iterationCount);

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
