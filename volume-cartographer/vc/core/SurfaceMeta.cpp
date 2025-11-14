#include "vc/core/util/SurfaceMeta.hpp"

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include <fstream>
#include <iomanip>
#include <system_error>
#include <algorithm>
#include <cmath>

// Geometry utility functions sdist, tdist, tdist_sum are now in vc::utils namespace (GeometryUtils.cpp)
// Use them directly in this file
using vc::utils::sdist;
using vc::utils::tdist;
using vc::utils::tdist_sum;

// rect_from_json helper function
static Rect3D rect_from_json(const nlohmann::json& json)
{
    return {{json[0][0], json[0][1], json[0][2]}, {json[1][0], json[1][1], json[1][2]}};
}

// SurfaceMeta constructors and destructor

SurfaceMeta::SurfaceMeta(const std::filesystem::path& path_, const nlohmann::json& json)
    : path(path_)
{
    if (json.contains("bbox"))
        bbox = rect_from_json(json["bbox"]);
    meta = new nlohmann::json;
    *meta = json;
    cacheMaskTimestamp();
}

SurfaceMeta::SurfaceMeta(const std::filesystem::path& path_) : path(path_)
{
    std::ifstream meta_f(path_ / "meta.json");
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

// SurfaceMeta member functions

void SurfaceMeta::readOverlapping()
{
    if (std::filesystem::exists(path / "overlapping")) {
        throw std::runtime_error(
            "Found overlapping directory at: " + (path / "overlapping").string() +
            "\nPlease run overlapping_to_json.py on " + path.parent_path().string() +
            " to convert it to JSON format");
    }
    overlapping_str = read_overlapping_json(path);
}

std::optional<std::filesystem::file_time_type> SurfaceMeta::readMaskTimestamp(
    const std::filesystem::path& dir)
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

void SurfaceMeta::cacheMaskTimestamp() { maskTimestamp_ = readMaskTimestamp(path); }

QuadSurface* SurfaceMeta::surface()
{
    if (!_surf) {
        _surf = load_quad_from_tifxyz(path);
        _ownsSurface = true;
        cacheMaskTimestamp();
    }
    return _surf;
}

void SurfaceMeta::setSurface(QuadSurface* surf, bool takeOwnership)
{
    if (_surf && _ownsSurface && _surf != surf) {
        delete _surf;
    }

    _surf = surf;
    _ownsSurface = takeOwnership && (surf != nullptr);
    cacheMaskTimestamp();
}

std::string SurfaceMeta::name() { return path.filename(); }

// Utility functions for overlapping JSON

void write_overlapping_json(const std::filesystem::path& seg_path,
                             const std::set<std::string>& overlapping_names)
{
    nlohmann::json overlap_json;
    overlap_json["overlapping"] =
        std::vector<std::string>(overlapping_names.begin(), overlapping_names.end());

    std::ofstream o(seg_path / "overlapping.json");
    o << std::setw(4) << overlap_json << std::endl;
}

std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path)
{
    std::set<std::string> overlapping;
    std::filesystem::path json_path = seg_path / "overlapping.json";

    if (std::filesystem::exists(json_path)) {
        std::ifstream i(json_path);
        nlohmann::json overlap_json;
        i >> overlap_json;

        if (overlap_json.contains("overlapping")) {
            for (const auto& name : overlap_json["overlapping"]) {
                overlapping.insert(name.get<std::string>());
            }
        }
    }

    return overlapping;
}

// Utility functions for overlap and contains checks

bool overlap(SurfaceMeta& a, SurfaceMeta& b, int max_iters)
{
    if (!intersect(a.bbox, b.bbox))
        return false;

    cv::Mat_<cv::Vec3f> points = a.surface()->rawPoints();
    for (int r = 0; r < std::max(10, max_iters / 10); r++) {
        cv::Vec2f p = {rand() % points.cols, rand() % points.rows};
        const cv::Vec3f& loc = points(p[1], p[0]);
        if (loc[0] == -1)
            continue;

        cv::Vec3f ptr{0, 0, 0};
        if (b.surface()->pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
            return true;
        }
    }
    return false;
}

bool contains(SurfaceMeta& a, const cv::Vec3f& loc, int max_iters)
{
    if (!intersect(a.bbox, {loc, loc}))
        return false;

    cv::Vec3f ptr{0, 0, 0};
    if (a.surface()->pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
        return true;
    }
    return false;
}

bool contains(SurfaceMeta& a, const std::vector<cv::Vec3f>& locs)
{
    for (auto& p : locs)
        if (!contains(a, p))
            return false;

    return true;
}

bool contains_any(SurfaceMeta& a, const std::vector<cv::Vec3f>& locs)
{
    for (auto& p : locs)
        if (contains(a, p))
            return true;

    return false;
}

// min_loc function

float min_loc(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f& loc, cv::Vec3f& out,
              const std::vector<cv::Vec3f>& tgts, const std::vector<float>& tds,
              PlaneSurface* plane, float init_step, float min_step)
{
    if (!loc_valid(points, {loc[1], loc[0]})) {
        out = {-1, -1, -1};
        return -1;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    if (plane) {
        float d = plane->pointDist(val);
        best += d * d;
    }
    float res;

    std::vector<cv::Vec2f> search = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    float step = init_step;

    while (changed) {
        changed = false;

        for (auto& off : search) {
            cv::Vec2f cand = loc + off * step;

            if (!loc_valid(points, {cand[1], cand[0]})) {
                continue;
            }

            val = at_int(points, cand);
            res = tdist_sum(val, tgts, tds);
            if (plane) {
                float d = plane->pointDist(val);
                res += d * d;
            }
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step < min_step)
            break;
    }

    return best;
}
