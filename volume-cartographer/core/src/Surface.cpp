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
#include <set>
#include <iostream>

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

    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;



    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;

            if (!loc_valid(points, {cand[1],cand[0]})) {
                // out = {-1,-1,-1};
                // loc = {-1,-1};
                // return -1;
                continue;
            }

            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds);
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                // std::cout << res << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << val << step << cand << "\n";
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step < min_step)
            break;
    }

    // std::cout << "best" << best << out << "\n" <<  std::endl;
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


MultiSurfaceIndex::MultiSurfaceIndex(float cell_sz)
    : cell_size(cell_sz), grid_size_x(0), grid_size_y(0), grid_size_z(0), grid_origin(0,0,0) {}

MultiSurfaceIndex::MultiSurfaceIndex(float cell_sz, const cv::Vec3f& min_bound, const cv::Vec3f& max_bound)
    : cell_size(cell_sz), grid_origin(min_bound)
{
    // Calculate grid dimensions to encompass the bounding box
    grid_size_x = std::ceil((max_bound[0] - min_bound[0]) / cell_size) + 1;
    grid_size_y = std::ceil((max_bound[1] - min_bound[1]) / cell_size) + 1;
    grid_size_z = std::ceil((max_bound[2] - min_bound[2]) / cell_size) + 1;

    // Allocate 3D array
    grid.resize(grid_size_x);
    for (int x = 0; x < grid_size_x; x++) {
        grid[x].resize(grid_size_y);
        for (int y = 0; y < grid_size_y; y++) {
            grid[x][y].resize(grid_size_z);
        }
    }
}

void MultiSurfaceIndex::addPatch(int idx, QuadSurface* patch) {
    Rect3D bbox = patch->bbox();

    // First patch AND grid not pre-allocated: compute bounds and allocate grid
    if (patch_bboxes.empty() && grid.empty()) {
        grid_origin = bbox.low;
        cv::Vec3f grid_max = bbox.high;

        grid_size_x = std::ceil((grid_max[0] - grid_origin[0]) / cell_size) + 10;  // +10 for padding
        grid_size_y = std::ceil((grid_max[1] - grid_origin[1]) / cell_size) + 10;
        grid_size_z = std::ceil((grid_max[2] - grid_origin[2]) / cell_size) + 10;

        // Allocate 3D array
        grid.resize(grid_size_x);
        for (int x = 0; x < grid_size_x; x++) {
            grid[x].resize(grid_size_y);
            for (int y = 0; y < grid_size_y; y++) {
                grid[x][y].resize(grid_size_z);
            }
        }
    }
    // If grid is pre-allocated (from constructor with bounds), just use it as-is
    // Segments can extend beyond volume bounds, which is fine - we just won't index those points

    patch_bboxes.push_back(bbox);

    // Add points to grid
    auto points = patch->rawPoints();
    int points_processed = 0;
    int points_out_of_bounds = 0;
    std::vector<cv::Vec3f> out_of_bounds_samples;  // Store first 10 for logging

    for (const auto& pt : points) {
        // Skip invalid points (NaN or negative sentinel values)
        if (std::isnan(pt[0]) || std::isnan(pt[1]) || std::isnan(pt[2]) ||
            pt[0] < 0 || pt[1] < 0 || pt[2] < 0) {
            continue;
        }

        // Calculate grid cell for this point
        int x = worldToGridX(pt[0]);
        int y = worldToGridY(pt[1]);
        int z = worldToGridZ(pt[2]);

        // Check if the point's primary cell is out of bounds
        if (!validGrid(x, y, z)) {
            points_out_of_bounds++;
            if (out_of_bounds_samples.size() < 10) {
                out_of_bounds_samples.push_back(pt);
            }
            continue;  // Skip this point entirely if its center cell is out of bounds
        }

        // Add this cell plus immediate neighbors (3x3x3 = 1 cell dilation)
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int gx = x + dx;
                    int gy = y + dy;
                    int gz = z + dz;

                    if (validGrid(gx, gy, gz)) {
                        auto& cell = grid[gx][gy][gz];
                        // Check if idx already in this cell
                        if (std::find(cell.patch_indices.begin(), cell.patch_indices.end(), idx) == cell.patch_indices.end()) {
                            cell.patch_indices.push_back(idx);
                        }
                    }
                }
            }
        }

        points_processed++;
    }

    if (points_out_of_bounds > 0) {
        std::cout << "  Patch " << idx << ": " << points_out_of_bounds << " points outside grid bounds (total: "
                  << (points_processed + points_out_of_bounds) << ")" << std::endl;
        std::cout << "    Grid bounds: [0, 0, 0] to ["
                  << (grid_origin[0] + grid_size_x * cell_size) << ", "
                  << (grid_origin[1] + grid_size_y * cell_size) << ", "
                  << (grid_origin[2] + grid_size_z * cell_size) << "]" << std::endl;
        std::cout << "    First " << out_of_bounds_samples.size() << " out-of-bounds points:" << std::endl;
        for (const auto& pt : out_of_bounds_samples) {
            std::cout << "      [" << pt[0] << ", " << pt[1] << ", " << pt[2] << "]" << std::endl;
        }
    }
}

std::vector<int> MultiSurfaceIndex::getCandidatePatches(const cv::Vec3f& point, float tolerance) const {
    int x = worldToGridX(point[0]);
    int y = worldToGridY(point[1]);
    int z = worldToGridZ(point[2]);

    std::set<int> unique_patches;

    if (tolerance > 0) {
        int cell_radius = std::ceil(tolerance / cell_size);
        for (int dz = -cell_radius; dz <= cell_radius; dz++) {
            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    int gx = x + dx;
                    int gy = y + dy;
                    int gz = z + dz;
                    if (validGrid(gx, gy, gz)) {
                        for (int idx : grid[gx][gy][gz].patch_indices) {
                            unique_patches.insert(idx);
                        }
                    }
                }
            }
        }
    } else {
        if (validGrid(x, y, z)) {
            for (int idx : grid[x][y][z].patch_indices) {
                unique_patches.insert(idx);
            }
        }
    }

    return std::vector<int>(unique_patches.begin(), unique_patches.end());
}

void MultiSurfaceIndex::removePatch(int idx) {
    // Remove this patch index from all cells in the grid
    for (int x = 0; x < grid_size_x; x++) {
        for (int y = 0; y < grid_size_y; y++) {
            for (int z = 0; z < grid_size_z; z++) {
                auto& indices = grid[x][y][z].patch_indices;
                indices.erase(std::remove(indices.begin(), indices.end(), idx), indices.end());
            }
        }
    }

    // Clear the bbox (mark as invalid with zero-size bbox)
    if (idx < patch_bboxes.size()) {
        patch_bboxes[idx] = {{0,0,0}, {0,0,0}};
    }
}

void MultiSurfaceIndex::updatePatch(int idx, QuadSurface* patch) {
    removePatch(idx);
    // Note: addPatch expects patch_bboxes to be empty or properly sized
    // So we need to handle this carefully - for now just remove and re-add
    // This is inefficient but correct
    addPatch(idx, patch);
}

std::vector<int> MultiSurfaceIndex::getCandidatePatchesByRegion(const cv::Vec3f& min_bound, const cv::Vec3f& max_bound) const {
    int x0 = worldToGridX(min_bound[0]);
    int y0 = worldToGridY(min_bound[1]);
    int z0 = worldToGridZ(min_bound[2]);
    int x1 = worldToGridX(max_bound[0]);
    int y1 = worldToGridY(max_bound[1]);
    int z1 = worldToGridZ(max_bound[2]);

    // Clamp to valid range
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    z0 = std::max(0, z0);
    x1 = std::min(grid_size_x - 1, x1);
    y1 = std::min(grid_size_y - 1, y1);
    z1 = std::min(grid_size_z - 1, z1);

    std::set<int> unique_patches;
    for (int z = z0; z <= z1; z++) {
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                for (int idx : grid[x][y][z].patch_indices) {
                    unique_patches.insert(idx);
                }
            }
        }
    }

    return std::vector<int>(unique_patches.begin(), unique_patches.end());
}

std::vector<int> MultiSurfaceIndex::getCandidatePatchesInYZPlane(float y_min, float y_max, float z_min, float z_max) const {
    int y0 = worldToGridY(y_min);
    int z0 = worldToGridZ(z_min);
    int y1 = worldToGridY(y_max);
    int z1 = worldToGridZ(z_max);

    // Clamp to valid range
    y0 = std::max(0, y0);
    z0 = std::max(0, z0);
    y1 = std::min(grid_size_y - 1, y1);
    z1 = std::min(grid_size_z - 1, z1);

    std::set<int> unique_patches;
    // Iterate through ALL X (entire plane)
    for (int x = 0; x < grid_size_x; x++) {
        for (int z = z0; z <= z1; z++) {
            for (int y = y0; y <= y1; y++) {
                for (int idx : grid[x][y][z].patch_indices) {
                    unique_patches.insert(idx);
                }
            }
        }
    }

    return std::vector<int>(unique_patches.begin(), unique_patches.end());
}

std::vector<int> MultiSurfaceIndex::getCandidatePatchesInXZPlane(float x_min, float x_max, float z_min, float z_max) const {
    int x0 = worldToGridX(x_min);
    int z0 = worldToGridZ(z_min);
    int x1 = worldToGridX(x_max);
    int z1 = worldToGridZ(z_max);

    // Clamp to valid range
    x0 = std::max(0, x0);
    z0 = std::max(0, z0);
    x1 = std::min(grid_size_x - 1, x1);
    z1 = std::min(grid_size_z - 1, z1);

    std::set<int> unique_patches;
    // Iterate through ALL Y (entire plane)
    for (int y = 0; y < grid_size_y; y++) {
        for (int z = z0; z <= z1; z++) {
            for (int x = x0; x <= x1; x++) {
                for (int idx : grid[x][y][z].patch_indices) {
                    unique_patches.insert(idx);
                }
            }
        }
    }

    return std::vector<int>(unique_patches.begin(), unique_patches.end());
}

std::vector<int> MultiSurfaceIndex::getCandidatePatchesInXYPlane(float x_min, float x_max, float y_min, float y_max) const {
    int x0 = worldToGridX(x_min);
    int y0 = worldToGridY(y_min);
    int x1 = worldToGridX(x_max);
    int y1 = worldToGridY(y_max);

    // Clamp to valid range
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(grid_size_x - 1, x1);
    y1 = std::min(grid_size_y - 1, y1);

    std::set<int> unique_patches;
    // Iterate through ALL Z (entire plane)
    for (int z = 0; z < grid_size_z; z++) {
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                for (int idx : grid[x][y][z].patch_indices) {
                    unique_patches.insert(idx);
                }
            }
        }
    }

    return std::vector<int>(unique_patches.begin(), unique_patches.end());
}

size_t MultiSurfaceIndex::getCellCount() const {
    size_t count = 0;
    for (int x = 0; x < grid_size_x; x++) {
        for (int y = 0; y < grid_size_y; y++) {
            for (int z = 0; z < grid_size_z; z++) {
                if (!grid[x][y][z].patch_indices.empty()) {
                    count++;
                }
            }
        }
    }
    return count;
}

size_t MultiSurfaceIndex::getPatchCount() const { return patch_bboxes.size(); }

std::vector<MultiSurfaceIndex::GridCellBounds> MultiSurfaceIndex::getGridCellBoundsInRegion(
    const cv::Vec3f& min_bound, const cv::Vec3f& max_bound) const {

    int x0 = worldToGridX(min_bound[0]);
    int y0 = worldToGridY(min_bound[1]);
    int z0 = worldToGridZ(min_bound[2]);
    int x1 = worldToGridX(max_bound[0]);
    int y1 = worldToGridY(max_bound[1]);
    int z1 = worldToGridZ(max_bound[2]);

    // Clamp to valid range
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    z0 = std::max(0, z0);
    x1 = std::min(grid_size_x - 1, x1);
    y1 = std::min(grid_size_y - 1, y1);
    z1 = std::min(grid_size_z - 1, z1);

    std::vector<GridCellBounds> bounds;
    bounds.reserve((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1));

    for (int z = z0; z <= z1; z++) {
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                // Only add cells that have content (not empty)
                if (!grid[x][y][z].patch_indices.empty()) {
                    GridCellBounds cell;
                    cell.min = cv::Vec3f(
                        grid_origin[0] + x * cell_size,
                        grid_origin[1] + y * cell_size,
                        grid_origin[2] + z * cell_size
                    );
                    cell.max = cv::Vec3f(
                        cell.min[0] + cell_size,
                        cell.min[1] + cell_size,
                        cell.min[2] + cell_size
                    );
                    bounds.push_back(cell);
                }
            }
        }
    }

    return bounds;
}
