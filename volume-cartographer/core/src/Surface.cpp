#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <unordered_map>
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <tiffio.h>


void normalizeMaskChannel(cv::Mat& mask)
{
    if (mask.empty()) {
        return;
    }

    cv::Mat singleChannel;
    if (mask.channels() == 1) {
        singleChannel = mask;
    } else if (mask.channels() == 3) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGR2GRAY);
    } else if (mask.channels() == 4) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGRA2GRAY);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(mask, channels);
        if (!channels.empty()) {
            singleChannel = channels[0];
        } else {
            singleChannel = mask;
        }
    }

    if (singleChannel.depth() != CV_8U) {
        cv::Mat converted;
        singleChannel.convertTo(converted, CV_8U);
        singleChannel = converted;
    }

    mask = singleChannel;
}


void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names) {
    nlohmann::json overlap_json;
    overlap_json["overlapping"] = std::vector<std::string>(overlapping_names.begin(), overlapping_names.end());

    std::ofstream o(seg_path / "overlapping.json");
    o << std::setw(4) << overlap_json << std::endl;
}

std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path) {
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



//NOTE we have 3 coordiante systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}

Surface::~Surface()
{
    if (meta) {
        delete meta;
    }
}


//given origin and normal, return the normalized vector v which describes a point : origin + v which lies in the plane and maximizes v.x at the cost of v.y,v.z
cv::Vec3f vx_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    //impossible
    if (n[1] == 0 && n[2] == 0)
        return {0,0,0};

    //also trivial
    if (n[0] == 0)
        return {1,0,0};

    cv::Vec3f v = {1,0,0};

    if (n[1] == 0) {
        v[1] = 0;
        //either n1 or n2 must be != 0, see first edge case
        v[2] = -n[0]/n[2];
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    if (n[2] == 0) {
        //either n1 or n2 must be != 0, see first edge case
        v[1] = -n[0]/n[1];
        v[2] = 0;
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    v[1] = -n[0]/(n[1]+n[2]);
    v[2] = v[1];
    cv::normalize(v, v, 1,0, cv::NORM_L2);

    return v;
}

cv::Vec3f vy_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    cv::Vec3f v = vx_from_orig_norm({o[1],o[0],o[2]}, {n[1],n[0],n[2]});
    return {v[1],v[0],v[2]};
}

void vxy_from_normal(cv::Vec3f orig, cv::Vec3f normal, cv::Vec3f &vx, cv::Vec3f &vy)
{
    vx = vx_from_orig_norm(orig, normal);
    vy = vy_from_orig_norm(orig, normal);

    //TODO will there be a jump around the midpoint?
    if (abs(vx[0]) >= abs(vy[1]))
        vy = cv::Mat(normal).cross(cv::Mat(vx));
    else
        vx = cv::Mat(normal).cross(cv::Mat(vy));

    //FIXME probably not the right way to normalize the direction?
    if (vx[0] < 0)
        vx *= -1;
    if (vy[1] < 0)
        vy *= -1;
}

cv::Vec3f rotateAroundAxis(const cv::Vec3f& vector, const cv::Vec3f& axis, float angle)
{
    if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) {
        return vector;
    }

    cv::Vec3d axis_d(axis[0], axis[1], axis[2]);
    double axis_norm = cv::norm(axis_d);
    if (axis_norm == 0.0) {
        return vector;
    }
    axis_d /= axis_norm;

    cv::Mat R;
    cv::Mat rot_vec = (cv::Mat_<double>(3, 1)
        << axis_d[0] * static_cast<double>(angle),
           axis_d[1] * static_cast<double>(angle),
           axis_d[2] * static_cast<double>(angle));
    cv::Rodrigues(rot_vec, R);

    cv::Mat v = (cv::Mat_<double>(3, 1) << vector[0], vector[1], vector[2]);
    cv::Mat res = R * v;
    return cv::Vec3f(static_cast<float>(res.at<double>(0)),
                     static_cast<float>(res.at<double>(1)),
                     static_cast<float>(res.at<double>(2)));
}

float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    // return d.dot(d);
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrt(d.dot(d));

    return abs(l-t_dist);
}

float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds)
{
    float sum = 0;
    for(int i=0;i<tgts.size();i++) {
        float d = tdist(v, tgts[i], tds[i]);
        sum += d*d;
    }

    return sum;
}

//search location in points where we minimize error to multiple objectives using iterated local search
//tgts,tds -> distance to some POIs
//plane -> stay on plane
float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, float init_step, float min_step)
{
    if (!loc_valid(points, {loc[1],loc[0]})) {
        out = {-1,-1,-1};
        return -1;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    if (plane) {
        float d = plane->pointDist(val);
        best += d*d;
    }
    float res;

    // Use 4-direction search for speed (2x faster than 8-direction)
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;

    // Loosen convergence criteria - 2x min_step is sufficient
    float convergence_threshold = min_step * 2.0f;

    // Early exit if we're close enough (helps with intersection finding)
    const float good_enough_threshold = 0.1f;

    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;

            if (!loc_valid(points, {cand[1],cand[0]})) {
                continue;
            }

            val = at_int(points, cand);
            res = tdist_sum(val, tgts, tds);
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;

                // Early exit if we found a very good match
                if (best < good_enough_threshold * good_enough_threshold) {
                    return best;
                }
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step < convergence_threshold)
            break;
    }

    return best;
}


float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    return pointTo_(loc, points, tgt, th, max_iters, scale);
}

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    return pointTo_(loc, points, tgt, th, max_iters, scale);
}



Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p)
{
    Rect3D res = a;
    for(int d=0;d<3;d++) {
        res.low[d] = std::min(res.low[d], p[d]);
        res.high[d] = std::max(res.high[d], p[d]);
    }

    return res;
}


bool intersect(const Rect3D &a, const Rect3D &b)
{
    for(int d=0;d<3;d++) {
        if (a.high[d] < b.low[d])
            return false;
        if (a.low[d] > b.high[d])
            return false;
    }

    return true;
}

Rect3D rect_from_json(const nlohmann::json &json)
{
    return {{json[0][0],json[0][1],json[0][2]},{json[1][0],json[1][1],json[1][2]}};
}


uint64_t MultiSurfaceIndex::hash(int x, int y, int z) const {
    // Ensure non-negative values for hashing
    uint32_t ux = static_cast<uint32_t>(x + 1000000);
    uint32_t uy = static_cast<uint32_t>(y + 1000000);
    uint32_t uz = static_cast<uint32_t>(z + 1000000);
    return (static_cast<uint64_t>(ux) << 40) |
           (static_cast<uint64_t>(uy) << 20) |
           static_cast<uint64_t>(uz);
}

MultiSurfaceIndex::MultiSurfaceIndex(float cell_sz) : cell_size(cell_sz) {}

void MultiSurfaceIndex::addPatch(int idx, QuadSurface* patch) {
    Rect3D bbox = patch->bbox();
    patch_bboxes.push_back(bbox);

    // Expand bbox slightly to handle edge cases
    int x0 = std::floor((bbox.low[0] - cell_size) / cell_size);
    int y0 = std::floor((bbox.low[1] - cell_size) / cell_size);
    int z0 = std::floor((bbox.low[2] - cell_size) / cell_size);
    int x1 = std::ceil((bbox.high[0] + cell_size) / cell_size);
    int y1 = std::ceil((bbox.high[1] + cell_size) / cell_size);
    int z1 = std::ceil((bbox.high[2] + cell_size) / cell_size);

    for (int z = z0; z <= z1; z++) {
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                grid[hash(x, y, z)].patch_indices.push_back(idx);
            }
        }
    }
}

std::vector<int> MultiSurfaceIndex::getCandidatePatches(const cv::Vec3f& point, float tolerance) const {
    // Get the cell containing this point
    int x = std::floor(point[0] / cell_size);
    int y = std::floor(point[1] / cell_size);
    int z = std::floor(point[2] / cell_size);

    // If tolerance is specified, check neighboring cells too
    std::set<int> unique_patches;

    if (tolerance > 0) {
        int cell_radius = std::ceil(tolerance / cell_size);
        for (int dz = -cell_radius; dz <= cell_radius; dz++) {
            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    auto it = grid.find(hash(x + dx, y + dy, z + dz));
                    if (it != grid.end()) {
                        for (int idx : it->second.patch_indices) {
                            unique_patches.insert(idx);
                        }
                    }
                }
            }
        }
    } else {
        auto it = grid.find(hash(x, y, z));
        if (it != grid.end()) {
            for (int idx : it->second.patch_indices) {
                unique_patches.insert(idx);
            }
        }
    }

    // Filter by bounding box for extra safety
    std::vector<int> result;
    for (int idx : unique_patches) {
        const Rect3D& bbox = patch_bboxes[idx];
        if (point[0] >= bbox.low[0] - tolerance &&
            point[0] <= bbox.high[0] + tolerance &&
            point[1] >= bbox.low[1] - tolerance &&
            point[1] <= bbox.high[1] + tolerance &&
            point[2] >= bbox.low[2] - tolerance &&
            point[2] <= bbox.high[2] + tolerance) {
            result.push_back(idx);
        }
    }

    return result;
}

size_t MultiSurfaceIndex::getCellCount() const { return grid.size(); }
size_t MultiSurfaceIndex::getPatchCount() const { return patch_bboxes.size(); }
