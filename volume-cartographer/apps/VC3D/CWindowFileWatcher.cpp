/**
 * CWindowFileWatcher.cpp - Linux inotify-based file watching for CWindow
 *
 * This file contains all inotify-related functionality for detecting
 * external changes to segmentation files. Only compiled on Linux.
 */

#include "CWindow.hpp"
#include "CSurfaceCollection.hpp"

#ifdef __linux__

#include <sys/inotify.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <map>

#include <QSettings>
#include <QTimer>
#include <QTreeWidgetItemIterator>
#include <QSignalBlocker>
#include <QApplication>
#include <QStatusBar>

#include "VCSettings.hpp"
#include "SurfaceTreeWidget.hpp"
#include "SurfacePanelController.hpp"
#include "segmentation/SegmentationModule.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

void CWindow::startWatchingWithInotify()
{
    if (!fVpkg) {
        return;
    }

    // Check if file watching is enabled in settings
    QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
    if (!settings.value(vc3d::settings::perf::ENABLE_FILE_WATCHING, vc3d::settings::perf::ENABLE_FILE_WATCHING_DEFAULT).toBool()) {
        Logger()->info("File watching is disabled in settings");
        return;
    }

    // Stop any existing watches
    stopWatchingWithInotify();

    // Initialize inotify
    _inotifyFd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (_inotifyFd < 0) {
        Logger()->error("Failed to initialize inotify: {}", strerror(errno));
        return;
    }

    // Watch both paths and traces directories
    auto availableDirs = fVpkg->getAvailableSegmentationDirectories();
    for (const auto& dirName : availableDirs) {
        std::filesystem::path dirPath = std::filesystem::path(fVpkg->getVolpkgDirectory()) / dirName;

        if (!std::filesystem::exists(dirPath)) {
            Logger()->debug("Directory {} does not exist, skipping watch", dirPath.string());
            continue;
        }

        // Watch for directory create, delete, and move events
        int wd = inotify_add_watch(_inotifyFd, dirPath.c_str(),
                                  IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_ONLYDIR);

        if (wd < 0) {
            Logger()->error("Failed to add inotify watch for {}: {}", dirPath.string(), strerror(errno));
            continue;
        }

        _watchDescriptors[wd] = dirName;
        Logger()->info("Started inotify watch for {} directory (wd={})", dirName, wd);
    }

    // Set up Qt socket notifier to integrate with event loop
    _inotifyNotifier = new QSocketNotifier(_inotifyFd, QSocketNotifier::Read, this);
    connect(_inotifyNotifier, &QSocketNotifier::activated, this, &CWindow::onInotifyEvent);;
}

void CWindow::stopWatchingWithInotify()
{
    if (_inotifyProcessTimer) {if (_inotifyProcessTimer) {
        _inotifyProcessTimer->stop();
    }
        _inotifyProcessTimer->stop();
    }

    if (_inotifyNotifier) {
        delete _inotifyNotifier;
        _inotifyNotifier = nullptr;
    }

    if (_inotifyFd >= 0) {
        // Remove all watches
        for (const auto& [wd, dirName] : _watchDescriptors) {
            inotify_rm_watch(_inotifyFd, wd);
        }
        _watchDescriptors.clear();

        ::close(_inotifyFd);
        _inotifyFd = -1;
    }

    _pendingMoves.clear();
}

void CWindow::onInotifyEvent()
{
    char buffer[4096] __attribute__((aligned(__alignof__(struct inotify_event))));
    ssize_t length = read(_inotifyFd, buffer, sizeof(buffer));

    if (length < 0) {
        if (errno != EAGAIN) {
            std::cerr << "Error reading inotify events: " << strerror(errno) << "\n";
        }
        return;
    }

    ssize_t i = 0;
    while (i < length) {
        struct inotify_event* event = reinterpret_cast<struct inotify_event*>(&buffer[i]);

        if (event->len > 0) {
            std::string fileName(event->name);

            // Skip hidden files and temporary files
            if (fileName.empty() || fileName[0] == '.' || fileName.find("~") != std::string::npos) {
                i += sizeof(struct inotify_event) + event->len;
                continue;
            }

            // Find the directory name for this watch descriptor
            auto it = _watchDescriptors.find(event->wd);
            if (it != _watchDescriptors.end()) {
                std::string dirName = it->second;

                // Handle different event types
                if (event->mask & IN_CREATE) {
                    // New segment created
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Addition;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingInotifyEvents.push_back(evt);

                } else if (event->mask & IN_DELETE) {
                    // Segment deleted
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Removal;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingInotifyEvents.push_back(evt);

                } else if (event->mask & IN_MOVED_FROM) {
                    // First part of move/rename - store with cookie
                    // Store both the filename and directory for orphaned move cleanup
                    _pendingMoves[event->cookie] = dirName + "/" + fileName;

                } else if (event->mask & IN_MOVED_TO) {
                    // Second part of move/rename
                    auto moveIt = _pendingMoves.find(event->cookie);
                    if (moveIt != _pendingMoves.end()) {
                        // This is a rename within watched directories
                        // Extract the old filename from the stored path
                        std::string oldPath = moveIt->second;
                        size_t lastSlash = oldPath.rfind('/');
                        std::string oldName = (lastSlash != std::string::npos) ?
                                               oldPath.substr(lastSlash + 1) : oldPath;
                        _pendingMoves.erase(moveIt);

                        InotifyEvent evt;
                        evt.type = InotifyEvent::Rename;
                        evt.dirName = dirName;
                        evt.segmentId = oldName;  // old segment ID
                        evt.newId = fileName;      // new segment ID
                        _pendingInotifyEvents.push_back(evt);

                    } else {
                        // File moved from outside watched directory - treat as new addition
                        InotifyEvent evt;
                        evt.type = InotifyEvent::Addition;
                        evt.dirName = dirName;
                        evt.segmentId = fileName;
                        _pendingInotifyEvents.push_back(evt);
                    }

                } else if (event->mask & (IN_MODIFY | IN_CLOSE_WRITE)) {
                    // Segment modified or closed after writing
                    // Use set to avoid duplicate updates for the same segment
                    _pendingSegmentUpdates.insert({dirName, fileName});
                }

                // Handle overflow
                if (event->mask & IN_Q_OVERFLOW) {
                    std::cerr << "Inotify queue overflow - some events may have been lost" << "\n";
                    // Could trigger a full reload here if needed
                }
            }
        }

        i += sizeof(struct inotify_event) + event->len;
    }

    // Clean up old pending moves that never got their MOVED_TO pair
    if (!_pendingMoves.empty()) {
        static auto lastCleanup = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();

        // Clean up orphaned moves every 5 seconds
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastCleanup).count() > 5) {
            for (auto it = _pendingMoves.begin(); it != _pendingMoves.end(); ) {
                // Extract directory and filename from stored path
                std::string fullPath = it->second;
                size_t lastSlash = fullPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    std::string dirName = fullPath.substr(0, lastSlash);
                    std::string fileName = fullPath.substr(lastSlash + 1);

                    // Treat orphaned MOVED_FROM as deletions
                    InotifyEvent evt;
                    evt.type = InotifyEvent::Removal;
                    evt.dirName = dirName;
                    evt.segmentId = fileName;
                    _pendingInotifyEvents.push_back(evt);
                }

                it = _pendingMoves.erase(it);
            }
            lastCleanup = now;
        }
    }

    scheduleInotifyProcessing();
}

void CWindow::processInotifySegmentUpdate(const std::string& dirName, const std::string& segmentName)
{
    if (!fVpkg) return;

    Logger()->info("Processing update of {} in {}", segmentName, dirName);

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());
    if (!isCurrentDir) {
        Logger()->debug("Update in non-current directory {}, skipping UI update", dirName);
        return;
    }

    std::string segmentId = segmentName; // UUID = directory name

    // Check if the segment exists
    auto seg = fVpkg->segmentation(segmentId);
    if (!seg) {
        Logger()->warn("Segment {} not found for update, treating as addition", segmentId);
        processInotifySegmentAddition(dirName, segmentName);
        return;
    }

    // Skip reloads triggered right after editing sessions end
    if (shouldSkipInotifyForSegment(segmentId, "reload")) {
        return;
    }

    bool wasSelected = (_surfID == segmentId);

    // Reload the segmentation
    if (fVpkg->reloadSingleSegmentation(segmentId)) {
        try {
            auto reloadedSurf = fVpkg->loadSurface(segmentId);
            if (reloadedSurf) {
                if (_surf_col) {
                    _surf_col->setSurface(segmentId, reloadedSurf, false, false);
                }

                if (_surfacePanel) {
                    _surfacePanel->refreshSurfaceMetrics(segmentId);
                }

                statusBar()->showMessage(tr("Updated: %1").arg(QString::fromStdString(segmentName)), 2000);

                if (wasSelected) {
                    _surfID = segmentId;
                    _surf_weak = reloadedSurf;

                    if (_surf_col) {
                        _surf_col->setSurface("segmentation", reloadedSurf, false, false);
                    }

                    if (_surfacePanel) {
                        _surfacePanel->syncSelectionUi(segmentId, reloadedSurf.get());
                    }
                }
            }
        } catch (const std::exception& e) {
            Logger()->error("Failed to reload segment {}: {}", segmentId, e.what());
        }
    }
}

void CWindow::processInotifySegmentRename(const std::string& dirName,
                                          const std::string& oldDirName,
                                          const std::string& newDirName)
{
    if (!fVpkg) return;

    Logger()->info("Processing rename in {}: {} -> {}", dirName, oldDirName, newDirName);

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());

    // The old UUID would have been the old directory name
    std::string oldId = oldDirName;
    std::string newId = newDirName;

    // Check if the old segment exists
    if (!fVpkg->segmentation(oldId)) {
        Logger()->warn("Old segment {} not found, treating as new addition", oldId);
        processInotifySegmentAddition(dirName, newDirName);
        return;
    }

    // Remove the old entry
    bool wasSelected = isCurrentDir && (_surfID == oldId);
    fVpkg->removeSingleSegmentation(oldId);

    if (isCurrentDir && _surfacePanel) {
        _surfacePanel->removeSingleSegmentation(oldId);
    }

    // Add with new name (which will read the meta.json and update the UUID)
    std::string previousDir;
    if (!isCurrentDir) {
        previousDir = fVpkg->getSegmentationDirectory();
        fVpkg->setSegmentationDirectory(dirName);
    }

    if (fVpkg->addSingleSegmentation(newDirName)) {
        // The UUID in meta.json will be updated when the segment is saved/loaded
        try {
            auto loadedSurf = fVpkg->loadSurface(newId);

            if (loadedSurf && isCurrentDir) {
                _surf_col->setSurface(newId, loadedSurf, true);
                if (_surfacePanel) {
                    _surfacePanel->addSingleSegmentation(newId);
                }

                statusBar()->showMessage(tr("Renamed: %1 → %2")
                                       .arg(QString::fromStdString(oldDirName),
                                            QString::fromStdString(newDirName)), 3000);

                // Reselect if it was selected
                if (wasSelected) {
                    _surfID = newId;
                    auto surf = _surf_col->surface(newId);
                    if (surf && _surfacePanel) {
                        _surfacePanel->syncSelectionUi(newId, dynamic_cast<QuadSurface*>(surf.get()));
                    }
                }

                if (_surfacePanel) {
                    //_surfacePanel->refreshFiltersOnly();
                }
            }
        } catch (const std::exception& e) {
            Logger()->error("Failed to load renamed segment {}: {}", newId, e.what());
        }
    }

    if (!isCurrentDir && !previousDir.empty()) {
        fVpkg->setSegmentationDirectory(previousDir);
    }
}

void CWindow::processInotifySegmentAddition(const std::string& dirName, const std::string& segmentName)
{
    if (!fVpkg) return;

    Logger()->info("Processing addition of {} to {}", segmentName, dirName);

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());

    // The UUID will be the directory name (or will be updated to match)
    std::string segmentId = segmentName;

    // Skip addition if editing just stopped recently to avoid thrashing the active surface
    if (shouldSkipInotifyForSegment(segmentId, "addition")) {
        return;
    }

    // Switch directory if needed
    std::string previousDir;
    if (!isCurrentDir) {
        previousDir = fVpkg->getSegmentationDirectory();
        fVpkg->setSegmentationDirectory(dirName);
    }

    // Add the segment
    if (fVpkg->addSingleSegmentation(segmentName)) {
        if (isCurrentDir) {
            try {
                auto loadedSurf = fVpkg->loadSurface(segmentId);
                if (loadedSurf) {
                    _surf_col->setSurface(segmentId, loadedSurf, true);
                    if (_surfacePanel) {
                        _surfacePanel->addSingleSegmentation(segmentId);
                    }
                    statusBar()->showMessage(tr("Added: %1").arg(QString::fromStdString(segmentName)), 2000);
                    if (_surfacePanel) {
                        //_surfacePanel->refreshFiltersOnly();
                    }
                }
            } catch (const std::exception& e) {
                Logger()->error("Failed to load segment {}: {}", segmentId, e.what());
            }
        }
    }

    if (!isCurrentDir && !previousDir.empty()) {
        fVpkg->setSegmentationDirectory(previousDir);
    }
}

void CWindow::processInotifySegmentRemoval(const std::string& dirName, const std::string& segmentName)
{
    if (!fVpkg) return;

    std::string segmentId = segmentName;

    Logger()->info("Processing removal of {} from {}", segmentId, dirName);

    // First check if this segment even exists and belongs to this directory
    auto seg = fVpkg->segmentation(segmentId);
    if (!seg) {
        Logger()->debug("Segment {} not found, ignoring removal event from {}", segmentId, dirName);
        return;
    }

    // Verify the segment is actually in the directory that reported the removal
    if (seg->path().parent_path().filename() != dirName) {
        Logger()->warn("Removal event for {} from {}, but segment is actually in {}",
                      segmentId, dirName, seg->path().parent_path().filename().string());
        return;
    }

    bool isCurrentDir = (dirName == fVpkg->getSegmentationDirectory());

    // Skip removal if editing just ended or is still running
    if (shouldSkipInotifyForSegment(segmentId, "removal")) {
        return;
    }

    // Remove from VolumePkg
    if (fVpkg->removeSingleSegmentation(segmentId)) {
        if (isCurrentDir && _surfacePanel) {
            _surfacePanel->removeSingleSegmentation(segmentId);
            statusBar()->showMessage(tr("Removed: %1").arg(QString::fromStdString(segmentName)), 2000);
            //_surfacePanel->refreshFiltersOnly();
        }
    }
}

void CWindow::processPendingInotifyEvents()
{
    if (_pendingInotifyEvents.empty() && _pendingSegmentUpdates.empty()) {
        return;
    }

    // Store current selection to restore later
    std::string previousSelection = _surfID;
    auto previousSurface = _surf_weak.lock();

    // Track if the previously selected segment gets removed
    bool previousSelectionRemoved = false;

    // Sort events to process removals first, then renames, then additions
    std::vector<InotifyEvent> removals, renames, additions, updates;
    for (const auto& evt : _pendingInotifyEvents) {
        switch (evt.type) {
            case InotifyEvent::Removal:
                removals.push_back(evt);
                break;
            case InotifyEvent::Rename:
                renames.push_back(evt);
                break;
            case InotifyEvent::Addition:
                additions.push_back(evt);
                break;
            case InotifyEvent::Update:
                updates.push_back(evt);
                break;
        }
    }

    if (!removals.empty() && !additions.empty()) {
        using EventKey = std::pair<std::string, std::string>;
        auto makeKey = [](const InotifyEvent& evt) -> EventKey {
            return {evt.dirName, evt.segmentId};
        };

        std::map<EventKey, int> additionCounts;
        for (const auto& addition : additions) {
            additionCounts[makeKey(addition)]++;
        }

        std::map<EventKey, int> pairedCounts;
        std::vector<InotifyEvent> filteredRemovals;
        filteredRemovals.reserve(removals.size());

        for (const auto& removal : removals) {
            EventKey key = makeKey(removal);
            auto availableIt = additionCounts.find(key);
            const int available = (availableIt != additionCounts.end()) ? availableIt->second : 0;
            auto existingPairIt = pairedCounts.find(key);
            const int alreadyPaired = (existingPairIt != pairedCounts.end()) ? existingPairIt->second : 0;

            if (available > alreadyPaired) {
                pairedCounts[key] = alreadyPaired + 1;

                InotifyEvent updateEvt;
                updateEvt.type = InotifyEvent::Update;
                updateEvt.dirName = removal.dirName;
                updateEvt.segmentId = removal.segmentId;
                updates.push_back(updateEvt);
            } else {
                filteredRemovals.push_back(removal);
            }
        }

        std::vector<InotifyEvent> filteredAdditions;
        filteredAdditions.reserve(additions.size());
        for (const auto& addition : additions) {
            EventKey key = makeKey(addition);
            auto pairIt = pairedCounts.find(key);
            if (pairIt != pairedCounts.end() && pairIt->second > 0) {
                pairIt->second--;
                continue;
            }
            filteredAdditions.push_back(addition);
        }

        removals.swap(filteredRemovals);
        additions.swap(filteredAdditions);
    }

    for (const auto& evt : removals) {
        if (evt.segmentId == previousSelection) {
            previousSelectionRemoved = true;
            break;
        }
    }

    // Process in order: removals, renames, additions, updates
    for (const auto& evt : removals) {
        processInotifySegmentRemoval(evt.dirName, evt.segmentId);
    }

    for (const auto& evt : renames) {
        processInotifySegmentRename(evt.dirName, evt.segmentId, evt.newId);
    }

    for (const auto& evt : additions) {
        processInotifySegmentAddition(evt.dirName, evt.segmentId);
    }

    for (const auto& evt : updates) {
        processInotifySegmentUpdate(evt.dirName, evt.segmentId);
    }

    // Process unique segment updates
    for (const auto& [dirName, segmentId] : _pendingSegmentUpdates) {
        processInotifySegmentUpdate(dirName, segmentId);
    }

    // Clear the queues
    _pendingInotifyEvents.clear();
    _pendingSegmentUpdates.clear();

    // Restore selection if it still exists (might have been renamed or re-added)
    if (!previousSelection.empty() && previousSelectionRemoved) {
        // If editing is active for this segment, skip re-emitting the segmentation surface change
        // because autosave-triggered inotify events were intentionally skipped.
        bool skipRestoreForActiveEdit = false;
        if (_segmentationModule && _segmentationModule->editingEnabled()) {
            auto* activeBaseSurface = _segmentationModule->activeBaseSurface();
            if (activeBaseSurface && activeBaseSurface->id == previousSelection) {
                Logger()->info("Skipping selection restore of {} - currently being edited", previousSelection);
                skipRestoreForActiveEdit = true;
            }
        }

        if (!skipRestoreForActiveEdit) {
            // Check if the segment was re-added in this batch
            bool wasReAdded = false;
            for (const auto& evt : additions) {
                if (evt.segmentId == previousSelection) {
                    wasReAdded = true;
                    break;
                }
            }

            if (wasReAdded) {
                // The segment was removed and re-added - restore selection
                auto seg = fVpkg ? fVpkg->segmentation(previousSelection) : nullptr;
                if (seg) {
                    _surfID = previousSelection;
                    auto surf = fVpkg->getSurface(previousSelection);
                    if (surf) {
                        _surf_weak = surf;

                        if (_surf_col) {
                            _surf_col->setSurface("segmentation", surf, false, false);
                        }

                        if (treeWidgetSurfaces) {
                            QTreeWidgetItemIterator it(treeWidgetSurfaces);
                            while (*it) {
                                if ((*it)->data(SURFACE_ID_COLUMN, Qt::UserRole).toString().toStdString() == previousSelection) {
                                    const QSignalBlocker blocker{treeWidgetSurfaces};
                                    treeWidgetSurfaces->clearSelection();
                                    (*it)->setSelected(true);
                                    treeWidgetSurfaces->scrollToItem(*it);
                                    break;
                                }
                                ++it;
                            }
                        }

                        if (_surfacePanel) {
                            _surfacePanel->syncSelectionUi(previousSelection, surf.get());
                        }

                        if (auto* viewer = segmentationViewer()) {
                            viewer->setWindowTitle(tr("Surface %1").arg(QString::fromStdString(previousSelection)));
                        }
                    }
                }
            }
        }
    } else if (!previousSelection.empty()) {
        // Original logic for non-removed segments
        auto seg = fVpkg ? fVpkg->segmentation(previousSelection) : nullptr;
        if (seg) {
            _surfID = previousSelection;
            _surf_weak = previousSurface;

            if (_surfacePanel) {
                auto surface = _surf_col->surface(previousSelection);
                if (surface) {
                    _surfacePanel->syncSelectionUi(previousSelection, dynamic_cast<QuadSurface*>(surface.get()));
                }
            }
        }
    }

    // Refresh filters once at the end instead of multiple times
    if (_surfacePanel) {
        _surfacePanel->refreshFiltersOnly();
    }
}

void CWindow::scheduleInotifyProcessing()
{
    if (!_inotifyProcessTimer) {
        return;
    }

    // Stop any existing timer
    _inotifyProcessTimer->stop();

    // Use single-shot timer with short delay
    _inotifyProcessTimer->setSingleShot(true);
    _inotifyProcessTimer->setInterval(INOTIFY_THROTTLE_MS);
    _inotifyProcessTimer->start();
}

bool CWindow::shouldSkipInotifyForSegment(const std::string& segmentId, const char* eventCategory)
{
    if (segmentId.empty()) {
        return false;
    }

    const char* category = eventCategory ? eventCategory : "inotify event";

    if (_segmentationModule && _segmentationModule->editingEnabled()) {
        if (auto* activeBaseSurface = _segmentationModule->activeBaseSurface()) {
            if (activeBaseSurface->id == segmentId) {
                Logger()->info("Skipping {} for {} - currently being edited", category, segmentId);
                return true;
            }
        }
    }

    // Also skip if approval mask editing is active for this segment
    if (_segmentationModule && _segmentationModule->isEditingApprovalMask()) {
        // Get the segment being used for approval mask (the active segmentation surface)
        if (_surf_col) {
            auto segSurface = _surf_col->surface("segmentation");
            if (segSurface && segSurface->id == segmentId) {
                Logger()->info("Skipping {} for {} - approval mask being edited", category, segmentId);
                return true;
            }
        }
    }

    pruneExpiredRecentlyEdited();

    if (_recentlyEditedSegments.empty()) {
        return false;
    }

    auto it = _recentlyEditedSegments.find(segmentId);
    if (it == _recentlyEditedSegments.end()) {
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = now - it->second;
    const auto grace = std::chrono::seconds(RECENT_EDIT_GRACE_SECONDS);

    if (elapsed < grace) {
        const double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
        const double remainingSeconds = std::max(
            0.0, static_cast<double>(RECENT_EDIT_GRACE_SECONDS) - elapsedSeconds);
        Logger()->info("Skipping {} for {} - edits ended {:.1f}s ago ({}s grace remaining)",
                       category,
                       segmentId,
                       elapsedSeconds,
                       static_cast<int>(std::ceil(remainingSeconds)));
        return true;
    }

    _recentlyEditedSegments.erase(it);
    return false;
}

void CWindow::markSegmentRecentlyEdited(const std::string& segmentId)
{
    if (segmentId.empty()) {
        return;
    }

    pruneExpiredRecentlyEdited();
    _recentlyEditedSegments[segmentId] = std::chrono::steady_clock::now();
}

void CWindow::pruneExpiredRecentlyEdited()
{
    if (_recentlyEditedSegments.empty()) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto grace = std::chrono::seconds(RECENT_EDIT_GRACE_SECONDS);

    for (auto it = _recentlyEditedSegments.begin(); it != _recentlyEditedSegments.end(); ) {
        if (now - it->second >= grace) {
            it = _recentlyEditedSegments.erase(it);
        } else {
            ++it;
        }
    }
}

void CWindow::setFileWatchingEnabled(bool enabled)
{
    if (enabled) {
        startWatchingWithInotify();
    } else {
        stopWatchingWithInotify();
    }
}

#else // !__linux__

// Stub implementation for non-Linux platforms
void CWindow::setFileWatchingEnabled(bool /*enabled*/)
{
    // File watching is only supported on Linux
}

#endif // __linux__
