#pragma once

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
    ~Segmentation();
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const noexcept;
    void saveMetadata();
    void ensureScrollSource(const std::string& scrollName, const std::string& volumeUuid);

    // Surface management - returns QuadSurface directly (no SurfaceMeta wrapper)
    [[nodiscard]] bool isSurfaceLoaded() const noexcept;
    [[nodiscard]] bool canLoadSurface() const;
    std::shared_ptr<QuadSurface> loadSurface();
    [[nodiscard]] std::shared_ptr<QuadSurface> getSurface() const;
    void unloadSurface();

    static bool checkDir(const std::filesystem::path& path);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
