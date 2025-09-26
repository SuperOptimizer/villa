#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_map>

#include <filesystem>
#include "vc/core/types/Metadata.hpp"
#include "vc/core/types/Segmentation.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkgVersion.hpp"

class VolumePkg
{
public:
    explicit VolumePkg(const std::filesystem::path& fileLocation);
    ~VolumePkg();
    static std::shared_ptr<VolumePkg> New(const std::filesystem::path& fileLocation);

    [[nodiscard]] std::string name() const;
    [[nodiscard]] int version() const;
    [[nodiscard]] bool hasVolumes() const;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    std::shared_ptr<Volume> volume();
    std::shared_ptr<Volume> volume(const std::string& id);
    [[nodiscard]] bool hasSegmentations() const;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;

    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);
    void setSegmentationDirectory(const std::string& dirName);
    [[nodiscard]] std::string getSegmentationDirectory() const;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;
    [[nodiscard]] std::string getVolpkgDirectory() const;

    void refreshSegmentations();

    // File watching control
    void enableFileWatching(bool enable = true);
    [[nodiscard]] bool isFileWatchingEnabled() const { return watcherRunning_; }

    // Surface management
    [[nodiscard]] bool isSurfaceLoaded(const std::string& id) const;
    std::shared_ptr<SurfaceMeta> loadSurface(const std::string& id);
    std::shared_ptr<SurfaceMeta> getSurface(const std::string& id);
    bool unloadSurface(const std::string& id);
    [[nodiscard]] std::vector<std::string> getLoadedSurfaceIDs() const;
    void unloadAllSurfaces();
    void loadSurfacesBatch(const std::vector<std::string>& ids);

private:
    Metadata config_;
    std::filesystem::path rootDir_;
    std::map<std::string, std::shared_ptr<Volume>> volumes_;
    std::map<std::string, std::shared_ptr<Segmentation>> segmentations_;
    std::string currentSegmentationDir_ = "paths";
    std::map<std::string, std::string> segmentationDirectories_;

    // File watching members
    int inotifyFd_ = -1;
    std::thread watchThread_;
    std::atomic<bool> watcherRunning_{false};
    std::atomic<bool> shouldStopWatcher_{false};
    std::unordered_map<int, std::filesystem::path> watchDescriptors_;
    std::mutex watchMutex_;

    void loadSegmentationsFromDirectory(const std::string& dirName);

    // File watching methods
    void startWatcher();
    void stopWatcher();
    void watchLoop();
    void addWatch(const std::filesystem::path& path);
    void addWatchesRecursive(const std::filesystem::path& path);
};