/**
 * @file SeedingWidgetNeuralTrace.cpp
 * @brief Neural trace methods extracted from SeedingWidget
 *
 * This file contains methods for neural tracing functionality.
 * Extracted from SeedingWidget.cpp to improve parallel compilation.
 */

#include "SeedingWidget.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>

#include <filesystem>
#include <iostream>

#include "CSurfaceCollection.hpp"

#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"

QString SeedingWidget::findPythonExecutable()
{
    QStringList candidates;

    // Check for explicit PYTHON_EXECUTABLE override first
    QString envPython = qEnvironmentVariable("PYTHON_EXECUTABLE");
    if (!envPython.isEmpty()) {
        candidates.append(envPython);
    }

    // Check for active conda environment (CONDA_PREFIX is set when env is active)
    QString condaPrefix = qEnvironmentVariable("CONDA_PREFIX");
    if (!condaPrefix.isEmpty()) {
        candidates.append(QDir(condaPrefix).filePath("bin/python"));
        candidates.append(QDir(condaPrefix).filePath("bin/python3"));
    }

    // Check for miniconda in home directory
    QString home = QDir::homePath();
    candidates.append(QDir(home).filePath("miniconda3/bin/python"));
    candidates.append(QDir(home).filePath("miniconda3/bin/python3"));
    candidates.append(QDir(home).filePath("anaconda3/bin/python"));
    candidates.append(QDir(home).filePath("anaconda3/bin/python3"));

    // System Python as fallback
    candidates.append("python3");
    candidates.append("python");
    candidates.append("/usr/bin/python3");
    candidates.append("/usr/local/bin/python3");

    for (const QString& candidate : candidates) {
        QProcess test;
        test.start(candidate, {"--version"});
        if (test.waitForFinished(1000) && test.exitCode() == 0) {
            return candidate;
        }
    }

    return "python3"; // Default fallback
}

QString SeedingWidget::findNeuralTracePyPath()
{
    QString appDir = QCoreApplication::applicationDirPath();
    QStringList searchPaths = {
        // Development paths
        QDir(appDir).filePath("../../vesuvius/src/vesuvius/neural_tracing/trace.py"),
        QDir(appDir).filePath("../../../vesuvius/src/vesuvius/neural_tracing/trace.py"),
        // Installed paths
        QDir(appDir).filePath("../share/vesuvius/neural_tracing/trace.py"),
        // Environment variable
        qEnvironmentVariable("NEURAL_TRACE_PY_PATH"),
    };

    for (const QString& path : searchPaths) {
        if (!path.isEmpty() && QFile::exists(path)) {
            return QFileInfo(path).absoluteFilePath();
        }
    }

    return QString();
}

void SeedingWidget::onNeuralCheckpointBrowseClicked()
{
    QString startDir = _neuralCheckpointPath.isEmpty()
        ? QDir::homePath()
        : QFileInfo(_neuralCheckpointPath).absolutePath();

    QString path = QFileDialog::getOpenFileName(
        this,
        tr("Select Neural Tracer Checkpoint"),
        startDir,
        tr("Checkpoint Files (*.pt *.pth *.ckpt);;All Files (*)")
    );

    if (!path.isEmpty()) {
        _neuralCheckpointEdit->setText(path);
    }
}

void SeedingWidget::onNeuralTraceClicked()
{
    // Validate prerequisites
    if (!currentVolume) {
        QMessageBox::warning(this, "Error", "No volume selected.");
        return;
    }

    POI* focusPoi = _surface_collection ? _surface_collection->poi("focus") : nullptr;
    if (!focusPoi) {
        QMessageBox::warning(this, "Error", "No focus point set. Please set a focus point first.");
        return;
    }

    if (_neuralCheckpointPath.isEmpty() || !QFile::exists(_neuralCheckpointPath)) {
        QMessageBox::warning(this, "Error", "Please select a valid checkpoint file.");
        return;
    }

    QString tracePyPath = findNeuralTracePyPath();
    if (tracePyPath.isEmpty()) {
        QMessageBox::warning(this, "Error",
            "Could not find trace.py. Set NEURAL_TRACE_PY_PATH environment variable.");
        return;
    }

    // Get volume zarr path
    std::filesystem::path volumePath = currentVolume->path();
    QString volumeZarr = QString::fromStdString(volumePath.string());

    // Get output path (paths directory in the volume package)
    std::filesystem::path pathsDir;
    if (fVpkg && fVpkg->hasSegmentations() && !fVpkg->segmentationIDs().empty()) {
        auto segID = fVpkg->segmentationIDs()[0];
        auto seg = fVpkg->segmentation(segID);
        pathsDir = seg->path().parent_path();
    } else if (fVpkg && fVpkg->hasVolumes()) {
        auto vol = fVpkg->volume();
        std::filesystem::path vpkgPath = vol->path().parent_path().parent_path();
        pathsDir = vpkgPath / "paths";
    } else {
        QMessageBox::warning(this, "Error", "Could not determine output directory.");
        return;
    }

    QString outPath = QString::fromStdString(pathsDir.string());

    // Get focus point coordinates (trace.py expects XYZ)
    const cv::Vec3f& p = focusPoi->p;
    int startX = static_cast<int>(p[0]);
    int startY = static_cast<int>(p[1]);
    int startZ = static_cast<int>(p[2]);

    // Find Python executable - use custom path if specified, otherwise auto-detect
    QString python = _neuralPythonPath.isEmpty() ? findPythonExecutable() : _neuralPythonPath;

    // Build arguments
    QStringList args = {
        tracePyPath,
        "--checkpoint_path", _neuralCheckpointPath,
        "--out_path", outPath,
        "--start_xyz", QString::number(startX), QString::number(startY), QString::number(startZ),
        "--volume_zarr", volumeZarr,
        "--volume_scale", QString::number(_neuralVolumeScale),
        "--steps_per_crop", QString::number(_neuralStepsPerCrop),
        "--max_size", QString::number(_neuralMaxSize),
        "--save_partial"
    };

    // Update UI
    infoLabel->setText("Running neural trace...");
    _btnNeuralTrace->setEnabled(false);
    jobsRunning = true;
    cancelButton->setVisible(true);

    // Create process
    QProcess* process = new QProcess(this);
    process->setProcessChannelMode(QProcess::MergedChannels);
    process->setWorkingDirectory(outPath);

    // Connect finished signal
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
        [this, process](int exitCode, QProcess::ExitStatus exitStatus) {
            jobsRunning = false;
            cancelButton->setVisible(false);
            runningProcesses.removeOne(process);
            process->deleteLater();

            if (exitStatus == QProcess::CrashExit) {
                infoLabel->setText("Neural trace crashed.");
                emit sendStatusMessageAvailable("Neural trace process crashed", 5000);
            } else if (exitCode != 0) {
                infoLabel->setText(QString("Neural trace failed (exit code %1)").arg(exitCode));
                emit sendStatusMessageAvailable(QString("Neural trace failed with exit code %1").arg(exitCode), 5000);
            } else {
                infoLabel->setText("Neural trace completed successfully.");
                emit sendStatusMessageAvailable("Neural trace completed successfully", 5000);
            }

            updateButtonStates();
        });

    // Connect output signal for logging
    connect(process, &QProcess::readyReadStandardOutput, [process]() {
        QString output = QString::fromUtf8(process->readAllStandardOutput());
        std::cout << "[neural-trace] " << output.toStdString();
    });

    // Log command and start
    std::cout << "Starting neural trace: " << python.toStdString();
    for (const QString& arg : args) {
        std::cout << " " << arg.toStdString();
    }
    std::cout << "\n";

    process->start(python, args);
    runningProcesses.append(QPointer<QProcess>(process));

    if (!process->waitForStarted(5000)) {
        QMessageBox::warning(this, "Error", "Failed to start neural trace process.");
        jobsRunning = false;
        cancelButton->setVisible(false);
        runningProcesses.removeOne(process);
        process->deleteLater();
        updateButtonStates();
        return;
    }

    emit sendStatusMessageAvailable(QString("Neural trace started from (%1, %2, %3)").arg(startX).arg(startY).arg(startZ), 5000);
}
