#include "vc/core/util/Surface.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace {

Rect3D rect_from_json(const nlohmann::json &json)
{
    return {{json[0][0],json[0][1],json[0][2]},{json[1][0],json[1][1],json[1][2]}};
}

} // namespace

// Helper functions from Surface.cpp
extern std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);
extern QuadSurface *load_quad_from_tifxyz(const std::string &path, int flags);

SurfaceMeta::SurfaceMeta(const std::filesystem::path &path_, const nlohmann::json &json) : path(path_)
{
    if (json.contains("bbox"))
        bbox = rect_from_json(json["bbox"]);
    meta = new nlohmann::json;
    *meta = json;
    cacheMaskTimestamp();
}

SurfaceMeta::SurfaceMeta(const std::filesystem::path &path_) : path(path_)
{
    std::ifstream meta_f(path_/"meta.json");
    if (!meta_f.is_open() || !meta_f.good()) {
        throw std::runtime_error("Cannot open meta.json file at: " + path_.string());
    }

    meta = new nlohmann::json;
    try {
        *meta = nlohmann::json::parse(meta_f);
    } catch (const nlohmann::json::parse_error& e) {
        delete meta;
        meta = nullptr;
        throw std::runtime_error("Invalid JSON in meta.json at: " + path_.string() + " - " + e.what());
    }

    if (meta->contains("bbox"))
        bbox = rect_from_json((*meta)["bbox"]);

    cacheMaskTimestamp();
}

SurfaceMeta::~SurfaceMeta()
{
    if (_surf && _ownsSurface) {
        delete _surf;
    }

    if (meta) {
        delete meta;
    }
}

void SurfaceMeta::readOverlapping()
{
    if (std::filesystem::exists(path / "overlapping")) {
        throw std::runtime_error(
            "Found overlapping directory at: " + (path / "overlapping").string() +
            "\nPlease run overlapping_to_json.py on " +  path.parent_path().string() + " to convert it to JSON format"
        );
    }
    overlapping_str = read_overlapping_json(path);
}

std::optional<std::filesystem::file_time_type> SurfaceMeta::readMaskTimestamp(const std::filesystem::path& dir)
{
    const std::filesystem::path maskPath = dir / "mask.tif";
    std::error_code ec;
    if (!std::filesystem::exists(maskPath, ec) || ec) {
        return std::nullopt;
    }
    auto ts = std::filesystem::last_write_time(maskPath, ec);
    if (ec) {
        return std::nullopt;
    }
    return ts;
}

void SurfaceMeta::cacheMaskTimestamp()
{
    maskTimestamp_ = readMaskTimestamp(path);
}

QuadSurface *SurfaceMeta::surface()
{
    if (!_surf) {
        _surf = load_quad_from_tifxyz(path);
        _ownsSurface = true;
        cacheMaskTimestamp();
    }
    return _surf;
}

void SurfaceMeta::setSurface(QuadSurface *surf, bool takeOwnership)
{
    if (_surf && _ownsSurface && _surf != surf) {
        delete _surf;
    }

    _surf = surf;
    _ownsSurface = takeOwnership && (surf != nullptr);
    cacheMaskTimestamp();
}

std::string SurfaceMeta::name()
{
    return path.filename();
}
