#include "vc/core/types/Segmentation.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/QuadSurface.hpp"

static const std::filesystem::path METADATA_FILE = "meta.json";

struct Segmentation::Impl {
    std::filesystem::path path_;
    std::unique_ptr<nlohmann::json> metadata_;
    std::shared_ptr<QuadSurface> surface_;

    Impl(std::filesystem::path p)
        : path_(std::move(p)), metadata_(std::make_unique<nlohmann::json>()) {}

    void loadMetadata()
    {
        auto metaPath = path_ / METADATA_FILE;
        *metadata_ = vc::json::load_json_file(metaPath);
        vc::json::require_type(*metadata_, "type", "seg", metaPath.string());
        vc::json::require_fields(*metadata_, {"uuid"}, metaPath.string());
    }
};

Segmentation::Segmentation(std::filesystem::path path)
    : pImpl_(std::make_unique<Impl>(std::move(path)))
{
    pImpl_->loadMetadata();
}

Segmentation::Segmentation(std::filesystem::path path, const std::string& uuid, const std::string& name)
    : pImpl_(std::make_unique<Impl>(std::move(path)))
{
    (*pImpl_->metadata_)["uuid"] = uuid;
    (*pImpl_->metadata_)["name"] = name;
    (*pImpl_->metadata_)["type"] = "seg";
    (*pImpl_->metadata_)["volume"] = std::string{};
    saveMetadata();
}

Segmentation::~Segmentation() = default;

std::string Segmentation::id() const
{
    return (*pImpl_->metadata_)["uuid"].get<std::string>();
}

std::string Segmentation::name() const
{
    return (*pImpl_->metadata_)["name"].get<std::string>();
}

void Segmentation::setName(const std::string& n)
{
    (*pImpl_->metadata_)["name"] = n;
}

std::filesystem::path Segmentation::path() const noexcept
{
    return pImpl_->path_;
}

bool Segmentation::isSurfaceLoaded() const noexcept
{
    return pImpl_->surface_ != nullptr;
}

void Segmentation::saveMetadata()
{
    auto metaPath = pImpl_->path_ / METADATA_FILE;
    std::ofstream jsonFile(metaPath.string(), std::ofstream::out);
    jsonFile << *pImpl_->metadata_ << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

void Segmentation::ensureScrollSource(const std::string& scrollName, const std::string& volumeUuid)
{
    bool changed = false;
    if (!pImpl_->metadata_->contains("scroll_source") || (*pImpl_->metadata_)["scroll_source"].get<std::string>().empty()) {
        (*pImpl_->metadata_)["scroll_source"] = scrollName;
        changed = true;
    }
    if (!pImpl_->metadata_->contains("volume") || (*pImpl_->metadata_)["volume"].get<std::string>().empty()) {
        (*pImpl_->metadata_)["volume"] = volumeUuid;
        changed = true;
    }
    if (changed) {
        saveMetadata();
    }
}

bool Segmentation::checkDir(const std::filesystem::path& path)
{
    return std::filesystem::is_directory(path) && std::filesystem::exists(path / METADATA_FILE);
}

std::shared_ptr<Segmentation> Segmentation::New(const std::filesystem::path& path)
{
    return std::make_shared<Segmentation>(path);
}

std::shared_ptr<Segmentation> Segmentation::New(const std::filesystem::path& path, const std::string& uuid, const std::string& name)
{
    return std::make_shared<Segmentation>(path, uuid, name);
}

bool Segmentation::canLoadSurface() const
{
    return pImpl_->metadata_->contains("format") &&
           (*pImpl_->metadata_)["format"].get<std::string>() == "tifxyz";
}

std::shared_ptr<QuadSurface> Segmentation::loadSurface()
{
    if (pImpl_->surface_) {
        return pImpl_->surface_;
    }

    if (!canLoadSurface()) {
        return nullptr;
    }

    try {
        // Load the surface directly (no SurfaceMeta wrapper)
        pImpl_->surface_ = load_quad_from_tifxyz(pImpl_->path_.string());

        // Load overlapping info and cache mask timestamp
        pImpl_->surface_->readOverlappingJson();
        pImpl_->surface_->refreshMaskTimestamp();

        return pImpl_->surface_;
    } catch (const std::exception& e) {
        Logger()->error("Failed to load surface for {}: {}", id(), e.what());
        pImpl_->surface_ = nullptr;
        return nullptr;
    }
}

std::shared_ptr<QuadSurface> Segmentation::getSurface() const
{
    return pImpl_->surface_;
}

void Segmentation::unloadSurface()
{
    pImpl_->surface_ = nullptr;
}
