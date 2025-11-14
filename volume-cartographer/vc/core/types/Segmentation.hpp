#pragma once

#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include "vc/core/util/SurfaceMeta.hpp"

class Segmentation
{
public:
    explicit Segmentation(std::filesystem::path path);
    Segmentation(std::filesystem::path path, std::string uuid, std::string name);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path);
    static std::shared_ptr<Segmentation> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const { return metadata_.at("uuid").get<std::string>(); }
    [[nodiscard]] std::string name() const { return metadata_.at("name").get<std::string>(); }
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    [[nodiscard]] nlohmann::json& metadata() { return metadata_; }

    // Surface management
    [[nodiscard]] bool isSurfaceLoaded() const;
    [[nodiscard]] bool canLoadSurface() const;
    std::shared_ptr<SurfaceMeta> loadSurface();
    [[nodiscard]] std::shared_ptr<SurfaceMeta> getSurface() const;
    void unloadSurface();

private:
    std::filesystem::path path_;
    nlohmann::json metadata_;
    std::shared_ptr<SurfaceMeta> surface_;
};
