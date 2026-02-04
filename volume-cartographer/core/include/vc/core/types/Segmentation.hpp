#pragma once

#include <nlohmann/json_fwd.hpp>
#include <filesystem>
#include <memory>
#include <string>

// Forward declarations
class QuadSurface;

class Segmentation final
{
public:
    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, const std::string& uuid, const std::string& name);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }
    void saveMetadata();
    void ensureScrollSource(const std::string& scrollName, const std::string& volumeUuid);

    // Surface management - returns QuadSurface directly (no SurfaceMeta wrapper)
    [[nodiscard]] bool isSurfaceLoaded() const noexcept { return surface_ != nullptr; }
    [[nodiscard]] bool canLoadSurface() const;
    std::shared_ptr<QuadSurface> loadSurface();
    [[nodiscard]] std::shared_ptr<QuadSurface> getSurface() const;
    void unloadSurface();

    static bool checkDir(const std::filesystem::path& path);

private:
    std::filesystem::path path_;
    std::unique_ptr<nlohmann::json> metadata_;
    std::shared_ptr<QuadSurface> surface_;

    void loadMetadata();
};