#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include <chrono>
#include <memory>



//TODO check if this actually works?!
static void set_block(cv::Mat_<uint8_t> &block, const cv::Vec3f &last_loc, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x1 = (loc[0]-roi.x)/step;
    int y1 = (loc[1]-roi.y)/step;
    int x2 = (last_loc[0]-roi.x)/step;
    int y2 = (last_loc[1]-roi.y)/step;

    if (x1 < 0 || y1 < 0 || x1 >= block.cols || y1 >= block.rows)
        return;
    if (x2 < 0 || y2 < 0 || x2 >= block.cols || y2 >= block.rows)
        return;

    if (x1 == x2 && y1 == y2)
        block(y1, x1) = 1;
    else
        cv::line(block, {x1,y1},{x2,y2}, 3);
}

static uint8_t get_block(const cv::Mat_<uint8_t> &block, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x = (loc[0]-roi.x)/step;
    int y = (loc[1]-roi.y)/step;

    if (x < 0 || y < 0 || x >= block.cols || y >= block.rows)
        return 1;

    return block(y, x);
}


PlaneSurface::PlaneSurface(const cv::Vec3f& origin_, const cv::Vec3f& normal) : _origin(origin_)
{
    cv::normalize(normal, _normal);
    update();
};

void PlaneSurface::setNormal(const cv::Vec3f& normal)
{
    cv::normalize(normal, _normal);
    update();
}

void PlaneSurface::setOrigin(const cv::Vec3f& origin)
{
    _origin = origin;
    update();
}

cv::Vec3f PlaneSurface::origin()
{
    return _origin;
}

float PlaneSurface::pointDist(const cv::Vec3f& wp) const {
    float plane_off = _origin.dot(_normal);
    float scalarp = wp.dot(_normal) - plane_off /*- _z_off*/;

    return abs(scalarp);
}

void PlaneSurface::setInPlaneRotation(float radians)
{
    _inPlaneRotation = radians;
    update();
}

void PlaneSurface::setAxisAlignedRotationKey(int key)
{
    _axisAlignedRotationKey = key;
}


void PlaneSurface::update()
{
    cv::Vec3f vx, vy;

    vxy_from_normal(_origin,_normal,vx,vy);

    if (std::abs(_inPlaneRotation) > std::numeric_limits<float>::epsilon()) {
        vx = rotateAroundAxis(vx, _normal, _inPlaneRotation);
        vy = rotateAroundAxis(vy, _normal, _inPlaneRotation);
    }

    cv::normalize(vx, vx, 1, 0, cv::NORM_L2);
    cv::normalize(vy, vy, 1, 0, cv::NORM_L2);

    _vx = vx;
    _vy = vy;

    std::vector <cv::Vec3f> src = {_origin,_origin+_normal,_origin+_vx,_origin+_vy};
    std::vector <cv::Vec3f> tgt = {{0,0,0},{0,0,1},{1,0,0},{0,1,0}};
    cv::Mat transf;
    cv::Mat inliers;

    cv::estimateAffine3D(src, tgt, transf, inliers, 0.1, 0.99);

    _M = transf({0,0,3,3});
    _T = transf({3,0,1,3});
}

cv::Vec3f PlaneSurface::project(const cv::Vec3f& wp, float render_scale, float coord_scale) const {
    cv::Vec3d res = _M*cv::Vec3d(wp)+_T;
    res *= render_scale*coord_scale;

    return {res(0), res(1), res(2)};
}

float PlaneSurface::scalarp(const cv::Vec3f& point) const
{
    return point.dot(_normal) - _origin.dot(_normal);
}



void PlaneSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    bool create_normals = normals || offset[2] || ptr[2];
    cv::Vec3f total_offset = internal_loc(offset/scale, ptr, {1,1});

    int w = size.width;
    int h = size.height;

    cv::Mat_<cv::Vec3f> _coords_header;
    cv::Mat_<cv::Vec3f> _normals_header;

    if (!coords)
        coords = &_coords_header;
    if (!normals)
        normals = &_normals_header;

    coords->create(size);

    if (create_normals)
        normals->create(size);

    const cv::Vec3f vx = _vx;
    const cv::Vec3f vy = _vy;

    float m = 1/scale;
    cv::Vec3f use_origin = _origin + _normal*total_offset[2];

#pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            (*coords)(j,i) = vx*(i*m+total_offset[0]) + vy*(j*m+total_offset[1]) + use_origin;
        }
}

cv::Vec3f PlaneSurface::pointer()
{
    return cv::Vec3f(0, 0, 0);
}

void PlaneSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ptr += offset;
}

cv::Vec3f PlaneSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return ptr + offset;
}

cv::Vec3f PlaneSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> coords;
    gen(&coords, nullptr, {1,1}, ptr, 1.0, offset);
    return coords(0,0);
}

cv::Vec3f PlaneSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _normal;
}


//search location in points where we minimize error to multiple objectives using iterated local search
//tgts,tds -> distance to some POIs
//plane -> stay on plane
float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, const PlaneSurface *plane, float init_step, float min_step, float early_exit_threshold, float epsilon, bool use_8way)
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

    // Choose search pattern: 8-way or 4-way
    std::vector<cv::Vec2f> search = use_8way ?
        std::vector<cv::Vec2f>{{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}} :
        std::vector<cv::Vec2f>{{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;

    int max_iterations = 100;  // Prevent infinite loops
    int iterations = 0;

    while (changed && iterations < max_iterations) {
        iterations++;
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
            float res = tdist_sum(val, tgts, tds);
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                float improvement = best - res;
                // Early termination if improvement is negligible
                if (epsilon > 0.0f && improvement < epsilon) {
                    best = res;
                    loc = cand;
                    out = val;
                    break;  // Exit the search loop
                }
                // std::cout << res << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << val << step << cand << "\n";
        }

        // Early exit: if we're already close enough to the plane, we're done
        if (plane && best < early_exit_threshold) {
            break;
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


void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries, const cv::Vec3f* poi_hint)
{
    auto t_total_start = std::chrono::high_resolution_clock::now();

    //start with random points and search for a plane intersection
    float block_step = 0.5*step;

    cv::Mat_<uint8_t> block(cv::Size(plane_roi.width/block_step, plane_roi.height/block_step), 0);

    cv::Rect grid_bounds(1,1,points.cols-2,points.rows-2);

    std::vector<std::vector<cv::Vec3f>> seg_vol_raw;
    std::vector<std::vector<cv::Vec2f>> seg_grid_raw;

    int max_iterations = std::max(min_tries, std::max(points.cols,points.rows)/100);

    int total_search_attempts = 0;
    int successful_traces = 0;
    int consecutive_failures = 0;  // Track iterations without finding a curve
    const int max_consecutive_failures = 3;  // Stop after 3 iterations with no curves found

    // Build grid candidate list ONCE (not inside the loop!)
    // Filter to only candidates visible in the viewport
    auto t_grid_start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Vec2f> grid_candidates;
    // Use dense sampling: surfaces are typically ~20x downsampled from volume,
    // so sampling every 2-4 surface pixels gives good coverage (~40-80 voxels in volume)
    // Balance between coverage and performance
    int grid_step = std::max(2, std::max(points.cols, points.rows) / 150);
    int total_candidates = 0;

    // Expand plane_roi proportionally to avoid clipping at edges and prevent pop-in/pop-out
    // Use 200% margin (3x viewport size) for strong anti-popping - grid points can be very sparse
    int margin_x = plane_roi.width * 2.0;
    int margin_y = plane_roi.height * 2.0;
    cv::Rect expanded_roi(plane_roi.x - margin_x, plane_roi.y - margin_y,
                          plane_roi.width + 2*margin_x, plane_roi.height + 2*margin_y);

    for(int y = 1; y < points.rows - 1; y += grid_step) {
        for(int x = 1; x < points.cols - 1; x += grid_step) {
            total_candidates++;
            cv::Vec3f pt = at_int(points, {static_cast<float>(x), static_cast<float>(y)});
            cv::Vec3f plane_loc = plane->project(pt);
            if (expanded_roi.contains(cv::Point(plane_loc[0], plane_loc[1]))) {
                grid_candidates.push_back({static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }

    auto t_grid_end = std::chrono::high_resolution_clock::now();
    auto grid_build_ms = std::chrono::duration<double, std::milli>(t_grid_end - t_grid_start).count();

    // Sort candidates deterministically by distance from viewport center
    // This ensures consistent results across renders
    auto t_sort_start = std::chrono::high_resolution_clock::now();
    cv::Vec2f viewport_center_plane = {
        static_cast<float>(plane_roi.x + plane_roi.width / 2),
        static_cast<float>(plane_roi.y + plane_roi.height / 2)
    };

    // Cache plane projections to avoid recomputing in sort comparisons
    // Each comparison would do 4 expensive calls (2x at_int + 2x project)
    struct CandidateWithDist {
        cv::Vec2f grid_loc;
        float dist_sq;
    };
    std::vector<CandidateWithDist> candidates_with_dist;
    candidates_with_dist.reserve(grid_candidates.size());

    for (const auto& grid_loc : grid_candidates) {
        cv::Vec3f pt = at_int(points, grid_loc);
        cv::Vec3f plane_loc = plane->project(pt);
        float dist_sq = (plane_loc[0] - viewport_center_plane[0]) * (plane_loc[0] - viewport_center_plane[0]) +
                       (plane_loc[1] - viewport_center_plane[1]) * (plane_loc[1] - viewport_center_plane[1]);
        candidates_with_dist.push_back({grid_loc, dist_sq});
    }

    std::sort(candidates_with_dist.begin(), candidates_with_dist.end(),
        [](const CandidateWithDist& a, const CandidateWithDist& b) {
            // Sort by distance, then by grid position for determinism
            if (std::abs(a.dist_sq - b.dist_sq) < 0.01f) {
                if (std::abs(a.grid_loc[0] - b.grid_loc[0]) < 0.01f) {
                    return a.grid_loc[1] < b.grid_loc[1];
                }
                return a.grid_loc[0] < b.grid_loc[0];
            }
            return a.dist_sq < b.dist_sq;
        });

    // Extract sorted grid locations
    grid_candidates.clear();
    grid_candidates.reserve(candidates_with_dist.size());
    for (const auto& cwd : candidates_with_dist) {
        grid_candidates.push_back(cwd.grid_loc);
    }

    auto t_sort_end = std::chrono::high_resolution_clock::now();
    auto sort_ms = std::chrono::duration<double, std::milli>(t_sort_end - t_sort_start).count();

    Logger()->info("find_intersect: grid_step={}, candidates={}, grid_build={:.2f}ms, sort={:.2f}ms, max_iterations={}",
                   grid_step, grid_candidates.size(), grid_build_ms, sort_ms, max_iterations);

    auto t_loop_start = std::chrono::high_resolution_clock::now();
    for(int r=0;r<max_iterations;r++) {
        std::vector<cv::Vec3f> seg;
        std::vector<cv::Vec2f> seg_loc;
        std::vector<cv::Vec3f> seg2;
        std::vector<cv::Vec2f> seg_loc2;
        cv::Vec2f loc;
        cv::Vec2f loc2;
        cv::Vec3f point;
        cv::Vec3f point2;
        cv::Vec3f point3;
        cv::Vec3f plane_loc;
        cv::Vec3f last_plane_loc;
        float dist = -1;


        //initial points
        int search_attempts = 0;
        int max_search_attempts = 1000;

        // Use the pre-built grid candidates in deterministic order
        // Note: POI hint is not used to reorder candidates to maintain deterministic rendering
        const std::vector<cv::Vec2f>& candidate_locs = grid_candidates;

        // Search through candidates
        for(int i=0;i<std::min(max_search_attempts, static_cast<int>(candidate_locs.size()));i++) {
            search_attempts++;
            total_search_attempts++;

            loc = candidate_locs[i];
            point = at_int(points, loc);

            plane_loc = plane->project(point);
            if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1])))
                continue;

                // Relaxed threshold for pixel-scale rendering: 1.0 squared
                // Use 4-way search and epsilon=0.01 for faster convergence in intersection rendering
                dist = min_loc(points, loc, point, {}, {}, plane, std::min(points.cols,points.rows)*0.1, 0.01, 1.0, 0.01, false);

                plane_loc = plane->project(point);
                if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1])))
                    dist = -1;

                if (get_block(block, plane_loc, plane_roi, block_step))
                    dist = -1;

            if (dist >= 0 && dist <= 1 || !loc_valid_xy(points, loc))
                break;
        }


        if (dist < 0 || dist > 1) {
            // Failed to find a starting point in this iteration
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        // Found a valid starting point
        seg.push_back(point);
        seg_loc.push_back(loc);

        //point2
        loc2 = loc;
        //search point at distance of 1 to init point
        // Use optimized parameters for intersection rendering
        dist = min_loc(points, loc2, point2, {point}, {1}, plane, 0.01, 0.0001, 1.0, 0.01, false);

        if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc)) {
            consecutive_failures++;
            if (consecutive_failures >= max_consecutive_failures) {
                break;
            }
            continue;
        }

        seg.push_back(point2);
        seg_loc.push_back(loc2);

        last_plane_loc = plane->project(point);
        plane_loc = plane->project(point2);
        set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
        last_plane_loc = plane_loc;

        //go one direction
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3)))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                // Use optimized parameters for intersection rendering
                // Moderately loose thresholds: balanced speed/quality
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.05, 0.002, 1.5, 0.05, false);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.05, 0.002, 1.5, 0.05, false);

                // If large step failed, try a half-step to avoid discontinuities
                // Use higher threshold (4.0) to allow larger steps without breaking
                if ((dist < 0 || dist > 4.0) && n > 0) {
                    // Try intermediate point (half-step)
                    loc3 = loc2 + (loc2 - loc) * 0.5f;
                    if (!grid_bounds.contains(cv::Point(loc3)))
                        break;

                    point3 = at_int(points, loc3);
                    // Moderately loose thresholds for half-step too
                    dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.05, 0.002, 1.5, 0.05, false);
                    dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.05, 0.002, 1.5, 0.05, false);
                }

                // Allow higher error tolerance (4.0) for larger step sizes
                if (dist < 0 || dist > 4.0 || !loc_valid_xy(points, loc3))
                    break;

            seg.push_back(point3);
            seg_loc.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);

            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        //now the other direction
        loc2 = seg_loc[0];
        loc = seg_loc[1];
        point2 = seg[0];
        point = seg[1];

        last_plane_loc = plane->project(point2);

        //FIXME repeat by not copying code ...
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3)))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                // Use optimized parameters for intersection rendering
                // Moderately loose thresholds: balanced speed/quality
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.05, 0.002, 1.5, 0.05, false);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.05, 0.002, 1.5, 0.05, false);

                // If large step failed, try a half-step to avoid discontinuities
                // Use higher threshold (4.0) to allow larger steps without breaking
                if ((dist < 0 || dist > 4.0) && n > 0) {
                    // Try intermediate point (half-step)
                    loc3 = loc2 + (loc2 - loc) * 0.5f;
                    if (!grid_bounds.contains(cv::Point(loc3)))
                        break;

                    point3 = at_int(points, loc3);
                    // Moderately loose thresholds for half-step too
                    dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.05, 0.002, 1.5, 0.05, false);
                    dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.05, 0.002, 1.5, 0.05, false);
                }

                // Allow higher error tolerance (4.0) for larger step sizes
                if (dist < 0 || dist > 4.0 || !loc_valid_xy(points, loc3))
                    break;

            seg2.push_back(point3);
            seg_loc2.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);

            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        std::reverse(seg2.begin(), seg2.end());
        std::reverse(seg_loc2.begin(), seg_loc2.end());

        seg2.insert(seg2.end(), seg.begin(), seg.end());
        seg_loc2.insert(seg_loc2.end(), seg_loc.begin(), seg_loc.end());


        seg_vol_raw.push_back(seg2);
        seg_grid_raw.push_back(seg_loc2);

        successful_traces++;
        consecutive_failures = 0;  // Reset failure counter on successful trace

        // Early exit: if we've found enough curves, stop searching
        if (successful_traces >= min_tries) {
            break;
        }
    }

    // Each traced curve is already continuous - the tracer stops at invalid grid points
    // Don't split based on 3D distance since curved surfaces can have large 3D distances
    // between consecutive grid points while still being validly connected
    for(int s=0;s<seg_vol_raw.size();s++) {
        if (seg_vol_raw[s].size() >= 2) {
            seg_vol.push_back(seg_vol_raw[s]);
            seg_grid.push_back(seg_grid_raw[s]);
        }
    }

    auto t_loop_end = std::chrono::high_resolution_clock::now();
    auto loop_ms = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();
    auto t_total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    Logger()->info("find_intersect: loop={:.2f}ms, total={:.2f}ms, successful_traces={}, output_segments={}",
                   loop_ms, total_ms, successful_traces, seg_vol.size());

}