/**
 * @file CWindowExportOperations.cpp
 * @brief Export and upload operations extracted from CWindow
 *
 * This file contains methods for exporting segments and uploading to AWS.
 * Extracted from CWindowContextMenu.cpp to improve parallel compilation.
 */

#include "CWindow.hpp"

#include "CommandLineToolRunner.hpp"
#include "ToolDialogs.hpp"
#include "VCSettings.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QInputDialog>
#include <QMessageBox>
#include <QProcess>
#include <QProgressDialog>
#include <QSettings>
#include <QStatusBar>

#include <cmath>
#include <filesystem>

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

void CWindow::onAWSUpload(const std::string& segmentId)
{
    auto surf = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surf) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot upload to AWS: No volume or invalid segment selected"));
        return;
    }
    if (_cmdRunner && _cmdRunner->isRunning()) {
        QMessageBox::warning(this, tr("Warning"), tr("A command line tool is already running."));
        return;
    }

    const std::filesystem::path segDirFs = surf->path;
    const QString  segDir   = QString::fromStdString(segDirFs.string());
    const QString  objPath  = QDir(segDir).filePath(QString::fromStdString(segmentId) + ".obj");
    const QString  flatObj  = QDir(segDir).filePath(QString::fromStdString(segmentId) + "_flatboi.obj");
    QString        outTifxyz= segDir + "_flatboi";

    if (!QFileInfo::exists(segDir)) {
        QMessageBox::warning(this, tr("Error"), tr("Cannot upload to AWS: Segment directory not found"));
        return;
    }

    QStringList scrollOptions;
    scrollOptions << "PHerc0172" << "PHerc0343P" << "PHerc0500P2";

    bool ok;
    QString selectedScroll = QInputDialog::getItem(
        this,
        tr("Select Scroll for Upload"),
        tr("Select the target scroll directory:"),
        scrollOptions,
        0, false, &ok
    );

    if (!ok || selectedScroll.isEmpty()) {
        statusBar()->showMessage(tr("AWS upload cancelled by user"), 3000);
        return;
    }

    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    QString defaultProfile = settings.value(vc3d::settings::aws::DEFAULT_PROFILE,
                                            vc3d::settings::aws::DEFAULT_PROFILE_DEFAULT).toString();

    QString awsProfile = QInputDialog::getText(
        this, tr("AWS Profile"),
        tr("Enter AWS profile name (leave empty for default credentials):"),
        QLineEdit::Normal, defaultProfile, &ok
    );

    if (!ok) {
        statusBar()->showMessage(tr("AWS upload cancelled by user"), 3000);
        return;
    }

    if (!awsProfile.isEmpty()) settings.setValue(vc3d::settings::aws::DEFAULT_PROFILE, awsProfile);

    QStringList uploadedFiles;
    QStringList failedFiles;

    auto uploadFileWithProgress = [&](const QString& localPath, const QString& s3Path, const QString& description, bool isDirectory = false) {
        if (!QFileInfo::exists(localPath)) return;
        if (isDirectory && !QFileInfo(localPath).isDir()) return;

        QStringList awsArgs;
        awsArgs << "s3" << "cp" << localPath << s3Path;
        if (isDirectory) awsArgs << "--recursive";
        if (!awsProfile.isEmpty()) { awsArgs << "--profile" << awsProfile; }

        statusBar()->showMessage(tr("Uploading %1...").arg(description), 0);

        QProcess p;
        p.setWorkingDirectory(segDir);
        p.setProcessChannelMode(QProcess::MergedChannels);
        p.start("aws", awsArgs);
        if (!p.waitForStarted()) { failedFiles << QString("%1: Failed to start AWS CLI").arg(description); return; }

        while (p.state() != QProcess::NotRunning) {
            if (p.waitForReadyRead(100)) {
                QString output = QString::fromLocal8Bit(p.readAllStandardOutput());
                if (!output.isEmpty()) {
                    const QStringList lines = output.split('\n', Qt::SkipEmptyParts);
                    for (const QString& line : lines) {
                        if (line.contains("Completed") || line.contains("upload:")) {
                            statusBar()->showMessage(QString("Uploading %1: %2").arg(description, line.trimmed()), 0);
                        }
                    }
                }
            }
            QCoreApplication::processEvents();
        }

        p.waitForFinished(-1);
        if (p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0) {
            uploadedFiles << description;
        } else {
            QString error = QString::fromLocal8Bit(p.readAllStandardError());
            if (error.isEmpty()) error = QString::fromLocal8Bit(p.readAllStandardOutput());
            failedFiles << QString("%1: %2").arg(description, error);
        }
    };

    auto uploadSegmentContents = [&](const QString& targetDir, const QString& segmentSuffix) {
        QString segmentName = QString::fromStdString(segmentId) + segmentSuffix;

        QString meshPath = QString("s3://vesuvius-challenge/%1/segments/meshes/%2/")
            .arg(selectedScroll).arg(segmentName);

        QString objFile = QDir(targetDir).filePath(segmentName + ".obj");
        uploadFileWithProgress(objFile, meshPath, QString("%1.obj").arg(segmentName));

        QString flatboiObjFile = QDir(targetDir).filePath(segmentName + "_flatboi.obj");
        uploadFileWithProgress(flatboiObjFile, meshPath, QString("%1_flatboi.obj").arg(segmentName));

        QString xTif = QDir(targetDir).filePath("x.tif");
        QString yTif = QDir(targetDir).filePath("y.tif");
        QString zTif = QDir(targetDir).filePath("z.tif");
        QString metaJson = QDir(targetDir).filePath("meta.json");

        if (QFileInfo::exists(xTif) && QFileInfo::exists(yTif) &&
            QFileInfo::exists(zTif) && QFileInfo::exists(metaJson)) {
            uploadFileWithProgress(xTif, meshPath, QString("%1/x.tif").arg(segmentName));
            uploadFileWithProgress(yTif, meshPath, QString("%1/y.tif").arg(segmentName));
            uploadFileWithProgress(zTif, meshPath, QString("%1/z.tif").arg(segmentName));
            uploadFileWithProgress(metaJson, meshPath, QString("%1/meta.json").arg(segmentName));
        }

        QString overlappingJson = QDir(targetDir).filePath("overlapping.json");
        uploadFileWithProgress(overlappingJson, meshPath, QString("%1/overlapping.json").arg(segmentName));

        QString layersDir = QDir(targetDir).filePath("layers");
        if (QFileInfo::exists(layersDir) && QFileInfo(layersDir).isDir()) {
            QString surfaceVolPath = QString("s3://vesuvius-challenge/%1/segments/surface-volumes/%2/layers/")
                .arg(selectedScroll).arg(segmentName);
            uploadFileWithProgress(layersDir, surfaceVolPath, QString("%1/layers").arg(segmentName), true);
        }
    };

    QProgressDialog progressDlg(tr("Uploading to AWS S3..."), tr("Cancel"), 0, 0, this);
    progressDlg.setWindowModality(Qt::WindowModal);
    progressDlg.setAutoClose(false);
    progressDlg.show();

    uploadSegmentContents(segDir, "");
    if (progressDlg.wasCanceled()) { statusBar()->showMessage(tr("AWS upload cancelled"), 3000); return; }
    if (QFileInfo::exists(outTifxyz) && QFileInfo(outTifxyz).isDir()) {
        uploadSegmentContents(outTifxyz, "_flatboi");
    }

    progressDlg.close();

    if (!uploadedFiles.isEmpty() && failedFiles.isEmpty()) {
        QMessageBox::information(this, tr("Upload Complete"),
            tr("Successfully uploaded to S3:\n\n%1").arg(uploadedFiles.join("\n")));
        statusBar()->showMessage(tr("AWS upload complete"), 5000);
    } else if (!uploadedFiles.isEmpty() && !failedFiles.isEmpty()) {
        QMessageBox::warning(this, tr("Partial Upload"),
            tr("Uploaded:\n%1\n\nFailed:\n%2").arg(uploadedFiles.join("\n"), failedFiles.join("\n")));
        statusBar()->showMessage(tr("AWS upload partially complete"), 5000);
    } else if (uploadedFiles.isEmpty() && !failedFiles.isEmpty()) {
        QMessageBox::critical(this, tr("Upload Failed"),
            tr("All uploads failed:\n\n%1\n\nPlease check:\n"
               "- AWS CLI is installed\n"
               "- AWS credentials are configured\n"
               "- You have internet connection\n"
               "- You have permissions for the S3 bucket").arg(failedFiles.join("\n")));
        statusBar()->showMessage(tr("AWS upload failed"), 5000);
    } else {
        QMessageBox::information(this, tr("No Files to Upload"),
            tr("No files found to upload for segment: %1").arg(QString::fromStdString(segmentId)));
        statusBar()->showMessage(tr("No files to upload"), 3000);
    }
}

void CWindow::onExportWidthChunks(const std::string& segmentId)
{
    auto surf = fVpkg ? fVpkg->getSurface(segmentId) : nullptr;
    if (currentVolume == nullptr || !surf) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Cannot export: No volume or invalid segment selected"));
        return;
    }

    // Pull points and get dimensions early so we can show them in the dialog
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    const int W = points.cols;
    const int H = points.rows;
    const cv::Vec2f sc = surf->scale();
    const double sx = (std::isfinite(sc[0]) && sc[0] > 0.0f) ? double(sc[0]) : 1.0; // guard

    if (W <= 0 || H <= 0) {
        QMessageBox::warning(this, tr("Error"),
                             tr("Surface has invalid dimensions (%1 x %2)").arg(W).arg(H));
        return;
    }

    // Show dialog to get export parameters
    ExportChunksDialog dlg(this, W, sx);
    if (dlg.exec() != QDialog::Accepted) {
        return;
    }

    const int chunkWidthReal = dlg.chunkWidth();
    const int overlapReal = dlg.overlapPerSide();
    const bool overwrite = dlg.overwrite();

    // Determine export root directory: <volpkg>/export (not inside paths)
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    const QString configuredRoot = settings.value(vc3d::settings::export_::DIR,
                                                   vc3d::settings::export_::DIR_DEFAULT).toString().trimmed();
    const QString segDir  = QString::fromStdString(surf->path.string());
    const QString segName = QString::fromStdString(segmentId);

    QString volpkgRoot = fVpkg ? QString::fromStdString(fVpkg->getVolpkgDirectory()) : QString();
    if (volpkgRoot.isEmpty()) {
        QDir d(QFileInfo(segDir).absoluteDir());   // start at parent of the segment folder
        while (!d.isRoot() && !d.dirName().endsWith(".volpkg")) d.cdUp();
        volpkgRoot = d.dirName().endsWith(".volpkg") ? d.absolutePath()
                                                    : QFileInfo(segDir).absolutePath();
    }
    const QString exportRoot = configuredRoot.isEmpty()
        ? QDir(volpkgRoot).filePath("export")
        : configuredRoot;

    QDir outRoot(exportRoot);
    if (!outRoot.exists() && !outRoot.mkpath(".")) {
        QMessageBox::critical(this, tr("Error"),
                              tr("Cannot create export directory:\n%1").arg(exportRoot));
        return;
    }

    // Convert real pixels to grid columns
    // Example: 40k real px with scale 0.05 → 2,000 columns per chunk
    const int chunkCols = std::max(1, int(std::llround(double(chunkWidthReal) * sx)));
    const int overlapCols = int(std::llround(double(overlapReal) * sx));

    // Calculate number of chunks: step through by chunkCols (the core width)
    const int nChunks = (W + chunkCols - 1) / chunkCols; // ceil-div purely in grid space

    if (nChunks <= 0) {
        QMessageBox::information(this, tr("Export"), tr("Nothing to export."));
        return;
    }

    // Progress dialog
    QProgressDialog prog(tr("Exporting width-chunks…"), tr("Cancel"), 0, nChunks, this);
    prog.setWindowModality(Qt::WindowModal);
    prog.setAutoClose(false);
    prog.setAutoReset(true);
    prog.setMinimumDuration(0);

    // Helper to generate a unique directory name if overwrite is false and target exists
    auto uniqueName = [&](const QString& base)->QString {
        if (!QFileInfo(outRoot.filePath(base)).exists()) return base;
        int k = 1;
        while (QFileInfo(outRoot.filePath(QString("%1_%2").arg(base).arg(k))).exists()) ++k;
        return QString("%1_%2").arg(base).arg(k);
    };

    // Zero-pad for nicer sorting
    auto padded = [nChunks](int idx)->QString {
        const int digits = (nChunks < 10) ? 1 : (nChunks < 100) ? 2 : (nChunks < 1000) ? 3 : 4;
        return QString("%1").arg(idx, digits, 10, QChar('0'));
    };

    // Export loop
    int exported = 0;
    QStringList results;
    QStringList failures;

    for (int c = 0; c < nChunks; ++c) {
        if (prog.wasCanceled()) break;
        prog.setLabelText(tr("Exporting chunk %1 / %2…").arg(c+1).arg(nChunks));
        prog.setValue(c);
        QCoreApplication::processEvents();

        // Core region for chunk c starts at c * chunkCols
        const int coreStart = c * chunkCols;

        // Calculate actual region with overlap:
        // - Left overlap: only if not the first chunk
        // - Right overlap: only if not the last chunk
        const int leftOverlap = (c == 0) ? 0 : overlapCols;
        const int rightOverlap = (c == nChunks - 1) ? 0 : overlapCols;

        // x0 = start of region (core start minus left overlap, clamped to 0)
        const int x0 = std::max(0, coreStart - leftOverlap);
        // x1 = end of region (core end plus right overlap, clamped to W)
        const int coreEnd = std::min(coreStart + chunkCols, W);
        const int x1 = std::min(coreEnd + rightOverlap, W);
        const int dx = x1 - x0;

        if (dx <= 0) continue;

        // ROI [all rows, x0:x1)
        cv::Mat_<cv::Vec3f> roi(points, cv::Range::all(), cv::Range(x0, x1));
        cv::Mat_<cv::Vec3f> roiCopy = roi.clone();  // ensure contiguous, independent buffer

        // Create a temp surface for this chunk; scale is preserved.
        QuadSurface chunkSurf(roiCopy, surf->scale());

        // Build target dir under exportRoot, name "<segName>_<indexPadded>"
        const QString baseName = QString("%1_%2").arg(segName, padded(c));
        QString outDirName = baseName;
        bool forceOverwrite = false;
        if (QFileInfo(outRoot.filePath(outDirName)).exists()) {
            if (overwrite) {
                forceOverwrite = true;
            } else {
                outDirName = uniqueName(baseName);
            }
        }
        const QString outAbs = outRoot.filePath(outDirName);
        const std::string outPath = outAbs.toStdString();
        const std::string uuid    = outDirName.toStdString();  // uuid ~ folder name

        try {
            chunkSurf.save(outPath, uuid, forceOverwrite);
            ++exported;
            results << outAbs;
        } catch (const std::exception& e) {
            failures << QString("%1 — %2").arg(outAbs, e.what());
        }

        QCoreApplication::processEvents();
    }
    prog.setValue(nChunks);

    // Summarize
    if (exported > 0 && failures.isEmpty()) {
        QMessageBox::information(this, tr("Export complete"),
                                 tr("Exported %1 chunk(s) to:\n%2")
                                 .arg(exported)
                                 .arg(QDir::toNativeSeparators(exportRoot)));
        statusBar()->showMessage(tr("Exported %1 chunk(s) → %2")
                                 .arg(exported)
                                 .arg(QDir::toNativeSeparators(exportRoot)),
                                 5000);
    } else if (exported > 0 && !failures.isEmpty()) {
        QMessageBox::warning(this, tr("Partial export"),
                             tr("Exported %1 chunk(s), but failed:\n\n%2")
                             .arg(exported)
                             .arg(failures.join('\n')));
        statusBar()->showMessage(tr("Export partially complete"), 5000);
    } else if (!failures.isEmpty()) {
        QMessageBox::critical(this, tr("Export failed"),
                              tr("All chunks failed:\n\n%1").arg(failures.join('\n')));
        statusBar()->showMessage(tr("Export failed"), 5000);
    } else {
        statusBar()->showMessage(tr("Export cancelled"), 3000);
    }
}
