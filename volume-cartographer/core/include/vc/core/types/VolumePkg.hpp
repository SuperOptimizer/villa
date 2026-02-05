#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

// Forward declarations — full types only needed in VolumePkg.cpp
class Volume;
class Segmentation;
class QuadSurface;

class VolumePkg final
{
public:
    explicit VolumePkg(const std::filesystem::path& fileLocation);
    ~VolumePkg();
    static std::shared_ptr<VolumePkg> New(const std::filesystem::path& fileLocation);

    [[nodiscard]] std::string name() const;
    [[nodiscard]] int version() const;
    [[nodiscard]] bool hasVolumes() const noexcept;
    [[nodiscard]] bool hasVolume(const std::string& id) const;
    [[nodiscard]] std::size_t numberOfVolumes() const noexcept;
    [[nodiscard]] std::vector<std::string> volumeIDs() const;
    std::shared_ptr<Volume> volume();
    std::shared_ptr<Volume> volume(const std::string& id);
    [[nodiscard]] bool hasSegmentations() const noexcept;
    [[nodiscard]] std::vector<std::string> segmentationIDs() const;

    std::shared_ptr<Segmentation> segmentation(const std::string& id);
    void removeSegmentation(const std::string& id);
    void setSegmentationDirectory(const std::string& dirName);
    [[nodiscard]] std::string getSegmentationDirectory() const noexcept;
    [[nodiscard]] std::vector<std::string> getAvailableSegmentationDirectories() const;
    [[nodiscard]] std::string getVolpkgDirectory() const noexcept;

    void refreshSegmentations();

    // Surface management - returns QuadSurface directly (no SurfaceMeta wrapper)
    [[nodiscard]] bool isSurfaceLoaded(const std::string& id) const;
    std::shared_ptr<QuadSurface> loadSurface(const std::string& id);
    std::shared_ptr<QuadSurface> getSurface(const std::string& id);
    bool unloadSurface(const std::string& id);
    [[nodiscard]] std::vector<std::string> getLoadedSurfaceIDs() const;
    void unloadAllSurfaces();
    void loadSurfacesBatch(const std::vector<std::string>& ids);
    bool addSingleSegmentation(const std::string& id);
    bool removeSingleSegmentation(const std::string& id);
    bool reloadSingleSegmentation(const std::string& id);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
