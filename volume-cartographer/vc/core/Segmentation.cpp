#include "vc/core/types/Segmentation.hpp"
#include "vc/core/util/Logging.hpp"

#include <fstream>

Segmentation::Segmentation(std::filesystem::path path)
    : path_(std::move(path))
{
    // Load metadata from meta.json
    auto metaPath = path_ / "meta.json";
    if (!std::filesystem::exists(metaPath)) {
        throw std::runtime_error("could not find json file '" + metaPath.string() + "'");
    }
    std::ifstream jsonFile(metaPath);
    if (!jsonFile) {
        throw std::runtime_error("could not open json file '" + metaPath.string() + "'");
    }
    jsonFile >> metadata_;
    if (jsonFile.bad()) {
        throw std::runtime_error("could not read json file '" + metaPath.string() + "'");
    }

    if (metadata_.at("type").get<std::string>() != "seg") {
        throw std::runtime_error("File not of type: seg");
    }
}

Segmentation::Segmentation(std::filesystem::path path, std::string uuid, std::string name)
    : path_(std::move(path))
{
    metadata_["uuid"] = std::move(uuid);
    metadata_["name"] = std::move(name);
    metadata_["type"] = "seg";
    metadata_["volume"] = std::string{};

    // Save metadata
    auto metaPath = path_ / "meta.json";
    std::ofstream jsonFile(metaPath, std::ofstream::out);
    jsonFile << metadata_ << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

std::shared_ptr<Segmentation> Segmentation::New(const std::filesystem::path& path)
{
    return std::make_shared<Segmentation>(path);
}

std::shared_ptr<Segmentation> Segmentation::New(const std::filesystem::path& path, const std::string& uuid, const std::string& name)
{
    return std::make_shared<Segmentation>(path, uuid, name);
}

bool Segmentation::isSurfaceLoaded() const
{
    return surface_ != nullptr;
}

bool Segmentation::canLoadSurface() const
{
    return metadata_.contains("format") &&
           metadata_.at("format").get<std::string>() == "tifxyz";
}

std::shared_ptr<SurfaceMeta> Segmentation::loadSurface()
{
    if (surface_) {
        return surface_;
    }

    if (!canLoadSurface()) {
        return nullptr;
    }

    try {
        surface_ = std::make_shared<SurfaceMeta>(path_);
        surface_->surface();
        surface_->readOverlapping();
        return surface_;
    } catch (const std::exception& e) {
        Logger()->error("Failed to load surface for {}: {}", id(), e.what());
        surface_ = nullptr;
        return nullptr;
    }
}

std::shared_ptr<SurfaceMeta> Segmentation::getSurface() const
{
    return surface_;
}

void Segmentation::unloadSurface()
{
    surface_ = nullptr;
}
