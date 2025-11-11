#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <algorithm>
#include <vector>


// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <tiffio.h>


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


void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries, const cv::Mat_<uint8_t> *sample_mask, const cv::Vec3f *viewport_min, const cv::Vec3f *viewport_max)
{
    //start with random points and search for a plane intersection

    // Adaptive min_tries based on grid size - don't waste time on small grids
    int grid_area = points.cols * points.rows;
    int adaptive_tries = std::min(min_tries, std::max(50, grid_area / 100));
    min_tries = adaptive_tries;

    float block_step = 0.5*step;

    cv::Mat_<uint8_t> block(cv::Size(plane_roi.width/block_step, plane_roi.height/block_step), 0);

    cv::Rect grid_bounds(1,1,points.cols-2,points.rows-2);

    // Grid coverage tracking - mark areas we've already searched
    int coverage_scale = 10;
    cv::Mat_<uint8_t> searched_grid(std::max(1, points.rows/coverage_scale),
                                     std::max(1, points.cols/coverage_scale),
                                     static_cast<uint8_t>(0));

    std::vector<std::vector<cv::Vec3f>> seg_vol_raw;
    std::vector<std::vector<cv::Vec2f>> seg_grid_raw;

    int consecutive_failures = 0;
    int cumulative_failures = 0;
    int max_consecutive_failures = 50;
    int max_cumulative_failures = 100;  // Increased - with smart sampling we can afford more attempts

    // Smart sampling strategies
    std::vector<cv::Vec2f> systematic_sample_points;
    std::vector<cv::Vec2f> seed_based_samples;  // Points near successful segments

    // Determine grid spacing based on whether we have a mask
    int systematic_grid_spacing = std::max(3, std::min(points.cols, points.rows) / 40);

    // Generate systematic grid sampling points
    // IMPORTANT: Only sample within grid_bounds to ensure at_int() can safely
    // access (y+1, x+1) for bilinear interpolation without going out of bounds
    for (int y = grid_bounds.y + systematic_grid_spacing/2;
         y < grid_bounds.y + grid_bounds.height;
         y += systematic_grid_spacing) {
        for (int x = grid_bounds.x + systematic_grid_spacing/2;
             x < grid_bounds.x + grid_bounds.width;
             x += systematic_grid_spacing) {

            // If viewport bounds provided, only sample points within viewport
            if (viewport_min && viewport_max) {
                const cv::Vec3f& pt = at_int(points, {(float)x, (float)y});
                if (pt[0] >= (*viewport_min)[0] && pt[0] <= (*viewport_max)[0] &&
                    pt[1] >= (*viewport_min)[1] && pt[1] <= (*viewport_max)[1] &&
                    pt[2] >= (*viewport_min)[2] && pt[2] <= (*viewport_max)[2]) {
                    systematic_sample_points.push_back({(float)x, (float)y});
                }
            } else {
                // No viewport constraint, add all systematic points
                systematic_sample_points.push_back({(float)x, (float)y});
            }
        }
    }

    std::random_shuffle(systematic_sample_points.begin(), systematic_sample_points.end());

    int systematic_samples_used = 0;
    int seed_samples_used = 0;
    int random_samples_used = 0;

    for(int r = 0; r < min_tries; r++) {
        // Early exit if we've had too many consecutive failures or too many total failures
        if (consecutive_failures >= max_consecutive_failures || cumulative_failures >= max_cumulative_failures) {
            break;
        }

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
        int inner_consecutive_failures = 0;
        int inner_roi_misses = 0;
        int inner_already_visited = 0;
        int max_inner_consecutive_failures = 50;  // Early exit much sooner

        for(int i=0; i < min_tries; i++) {

            // Smart sampling strategy with prioritization:
            // 1. Systematic grid points (cover the space methodically)
            // 2. Seed-based points (near successful segments)
            // 3. Random fallback
            if (systematic_samples_used < systematic_sample_points.size()) {
                // Priority 1: Use systematic grid sampling
                loc = systematic_sample_points[systematic_samples_used];
                systematic_samples_used++;
            } else if (seed_samples_used < seed_based_samples.size()) {
                // Priority 2: Use seed-based sampling near successful segments
                loc = seed_based_samples[seed_samples_used];
                seed_samples_used++;
            } else {
                // Priority 3: Fall back to random sampling within safe grid bounds
                // If viewport bounds provided, only sample points within viewport
                bool found_valid_random = false;
                int random_attempts = 0;
                const int max_random_attempts = 50;  // Limit attempts to avoid infinite loop

                while (!found_valid_random && random_attempts < max_random_attempts) {
                    loc = {grid_bounds.x + (std::rand() % grid_bounds.width),
                           grid_bounds.y + (std::rand() % grid_bounds.height)};

                    if (viewport_min && viewport_max) {
                        const cv::Vec3f& pt = at_int(points, loc);
                        if (pt[0] >= (*viewport_min)[0] && pt[0] <= (*viewport_max)[0] &&
                            pt[1] >= (*viewport_min)[1] && pt[1] <= (*viewport_max)[1] &&
                            pt[2] >= (*viewport_min)[2] && pt[2] <= (*viewport_max)[2]) {
                            found_valid_random = true;
                        }
                    } else {
                        // No viewport constraint, accept any point
                        found_valid_random = true;
                    }
                    random_attempts++;
                }

                if (!found_valid_random) {
                    // Couldn't find a valid random point in viewport after max_attempts
                    // This means we've exhausted the search space
                    inner_consecutive_failures++;
                    if (inner_consecutive_failures >= max_inner_consecutive_failures) {
                        break;
                    }
                    continue;
                }

                random_samples_used++;
            }

            // Check grid coverage - skip already heavily searched areas
            int grid_x = loc[0] / coverage_scale;
            int grid_y = loc[1] / coverage_scale;
            if (grid_x >= 0 && grid_x < searched_grid.cols &&
                grid_y >= 0 && grid_y < searched_grid.rows) {
                if (searched_grid(grid_y, grid_x) > 5) {
                    // This area has been searched many times, skip it
                    inner_consecutive_failures++;
                    if (inner_consecutive_failures >= max_inner_consecutive_failures) {
                        break;
                    }
                    continue;
                }
                searched_grid(grid_y, grid_x)++;
            }

            point = at_int(points, loc);

            plane_loc = plane->project(point);
            if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1]))) {
                inner_roi_misses++;
                // Don't count ROI misses as consecutive failures - they're not intersection failures,
                // just samples outside the region of interest
                if (inner_roi_misses >= max_inner_consecutive_failures) {
                    break;
                }
                continue;
            }

            dist = min_loc(points, loc, point, {}, {}, plane, std::min(points.cols,points.rows)*0.1, 0.1);

            plane_loc = plane->project(point);
                if (!plane_roi.contains(cv::Point(plane_loc[0],plane_loc[1]))) {
                    dist = -1;
                    inner_roi_misses++;
                    // Don't count as consecutive failure, just ROI miss
                    if (inner_roi_misses >= max_inner_consecutive_failures) {
                        break;
                    }
                    continue;
                }

                if (get_block(block, plane_loc, plane_roi, block_step)) {
                    dist = -1;
                    inner_already_visited++;
                    inner_consecutive_failures++;
                    if (inner_consecutive_failures >= max_inner_consecutive_failures) {
                        break;
                    }
                    continue;
                }

            if (dist >= 0 && dist <= 4 || !loc_valid_xy(points, loc)) {
                inner_consecutive_failures = 0;  // Reset on success
                break;
            }

            inner_consecutive_failures++;
            if (inner_consecutive_failures >= max_inner_consecutive_failures) {
                break;
            }
        }

        bool inner_roi_only_exit = (inner_roi_misses >= max_inner_consecutive_failures);

        if (dist < 0 || dist > 4) {
            // Only count as a real failure if we actually tried min_loc and it failed
            // Don't count exits due to all ROI misses as failures
            bool is_real_failure = !inner_roi_only_exit;

            if (is_real_failure) {
                consecutive_failures++;
                cumulative_failures++;
            }
            continue;
        }

        seg.push_back(point);
        seg_loc.push_back(loc);

        //point2
        loc2 = loc;
        //search point at distance of 1 to init point
        dist = min_loc(points, loc2, point2, {point}, {1}, plane, 0.1, 0.01);

        if (dist < 0 || dist > 4 || !loc_valid_xy(points, loc)) {
            consecutive_failures++;
            cumulative_failures++;
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

            if (!grid_bounds.contains(cv::Point(loc3))) {
                break;
            }

            point3 = at_int(points, loc3);

            //search point close to prediction + dist 1 to last point
            dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.1, 0.001);
            float dist1 = dist;

            //then refine
            dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.1, 0.001);

            if (dist < 0 || dist > 4 || !loc_valid_xy(points, loc)) {
                break;
            }

            seg.push_back(point3);
            seg_loc.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            // Note: Removed block checking during trace - it was causing premature termination
            // The iteration limit (100) and dist checks are sufficient to prevent infinite loops
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

            if (!grid_bounds.contains(cv::Point(loc3))) {
                break;
            }

            point3 = at_int(points, loc3);

            //search point close to prediction + dist 1 to last point
            dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.1, 0.001);
            float dist1 = dist;

            //then refine
            dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.1, 0.001);

            if (dist < 0 || dist > 4 || !loc_valid_xy(points, loc)) {
                break;
            }

            seg2.push_back(point3);
            seg_loc2.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            // Note: Removed block checking during trace - it was causing premature termination
            // The iteration limit (100) and dist checks are sufficient to prevent infinite loops
        }

        std::reverse(seg2.begin(), seg2.end());
        std::reverse(seg_loc2.begin(), seg_loc2.end());

        seg2.insert(seg2.end(), seg.begin(), seg.end());
        seg_loc2.insert(seg_loc2.end(), seg_loc.begin(), seg_loc.end());


        seg_vol_raw.push_back(seg2);
        seg_grid_raw.push_back(seg_loc2);
        consecutive_failures = 0;  // Reset on success

        // Generate seed-based samples around this successful segment
        // Sample every few points along the segment and add nearby search points
        int seed_spacing = std::max(5, (int)seg_loc2.size() / 10);  // Sample ~10 points per segment
        int seed_radius = std::max(3, systematic_grid_spacing / 2);  // Search within half-grid spacing
        for (int s = 0; s < seg_loc2.size(); s += seed_spacing) {
            // Add sample points in a small radius around this segment point
            for (int dx = -seed_radius; dx <= seed_radius; dx += seed_radius/2) {
                for (int dy = -seed_radius; dy <= seed_radius; dy += seed_radius/2) {
                    if (dx == 0 && dy == 0) continue;  // Skip the segment point itself
                    cv::Vec2f seed_point = {seg_loc2[s][0] + dx, seg_loc2[s][1] + dy};
                    // Check that seed point is within safe grid bounds for at_int()
                    if (grid_bounds.contains(cv::Point(seed_point[0], seed_point[1]))) {
                        seed_based_samples.push_back(seed_point);
                    }
                }
            }
        }
    }

    //keep segments as traced - only split when tracing actually failed (not based on distance)
    for(int s=0;s<seg_vol_raw.size();s++) {
        if (seg_vol_raw[s].size() >= 2) {
            seg_vol.push_back(seg_vol_raw[s]);
            seg_grid.push_back(seg_grid_raw[s]);
        }
    }
}
