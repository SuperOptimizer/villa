#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>


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


void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Removed verbose logging - keeping only summary

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
    int max_consecutive_failures = 100;
    int max_cumulative_failures = 500;  // Increased - with smart sampling we can afford more attempts

    // Smart sampling strategies
    std::vector<cv::Vec2f> systematic_sample_points;
    std::vector<cv::Vec2f> seed_based_samples;  // Points near successful segments
    // Use denser grid spacing to ensure we catch intersections even with small ROI overlap
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
            systematic_sample_points.push_back({(float)x, (float)y});
        }
    }
    std::random_shuffle(systematic_sample_points.begin(), systematic_sample_points.end());

    int systematic_samples_used = 0;
    int seed_samples_used = 0;
    int random_samples_used = 0;

    // Statistics tracking
    int initial_point_search_attempts = 0;
    int successful_segments = 0;
    double total_minloc_time_ms = 0;
    int minloc_call_count = 0;
    int inner_early_exits = 0;
    int total_inner_roi_misses = 0;
    int total_inner_already_visited = 0;

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
        cv::Vec2f loc3;
        cv::Vec3f point;
        cv::Vec3f point2;
        cv::Vec3f point3;
        cv::Vec3f plane_loc;
        cv::Vec3f last_plane_loc;
        float dist = -1;


        //initial points
        auto init_search_start = std::chrono::high_resolution_clock::now();
        int init_attempts = 0;
        int inner_consecutive_failures = 0;
        int inner_roi_misses = 0;
        int inner_already_visited = 0;
        int max_inner_consecutive_failures = 100;  // Early exit much sooner

        for(int i=0; i < min_tries; i++) {
            init_attempts++;
            initial_point_search_attempts++;

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
                loc = {grid_bounds.x + (std::rand() % grid_bounds.width),
                       grid_bounds.y + (std::rand() % grid_bounds.height)};
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

                auto minloc_start = std::chrono::high_resolution_clock::now();
                dist = min_loc(points, loc, point, {}, {}, plane, std::min(points.cols,points.rows)*0.1, 0.01);
                auto minloc_end = std::chrono::high_resolution_clock::now();
                double minloc_time = std::chrono::duration<double, std::milli>(minloc_end - minloc_start).count();
                total_minloc_time_ms += minloc_time;
                minloc_call_count++;

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

            if (dist >= 0 && dist <= 1 || !loc_valid_xy(points, loc)) {
                inner_consecutive_failures = 0;  // Reset on success
                break;
            }

            inner_consecutive_failures++;
            if (inner_consecutive_failures >= max_inner_consecutive_failures) {
                break;
            }
        }
        auto init_search_end = std::chrono::high_resolution_clock::now();
        double init_search_time = std::chrono::duration<double, std::milli>(init_search_end - init_search_start).count();

        bool inner_early_exit = inner_consecutive_failures >= max_inner_consecutive_failures;
        bool inner_roi_only_exit = (inner_roi_misses >= max_inner_consecutive_failures);

        if (dist < 0 || dist > 1) {
            // Only count as a real failure if we actually tried min_loc and it failed
            // Don't count exits due to all ROI misses as failures
            bool is_real_failure = !inner_roi_only_exit;

            if (inner_early_exit) {
                inner_early_exits++;
            }
            total_inner_roi_misses += inner_roi_misses;
            total_inner_already_visited += inner_already_visited;
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
        auto minloc_start = std::chrono::high_resolution_clock::now();
        dist = min_loc(points, loc2, point2, {point}, {1}, plane, 0.01, 0.0001);
        auto minloc_end = std::chrono::high_resolution_clock::now();
        double minloc_time = std::chrono::duration<double, std::milli>(minloc_end - minloc_start).count();
        total_minloc_time_ms += minloc_time;
        minloc_call_count++;

        if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc)) {
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
        auto forward_trace_start = std::chrono::high_resolution_clock::now();
        int forward_points = 0;
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3))) {
                break;
            }

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                minloc_start = std::chrono::high_resolution_clock::now();
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.01, 0.0001);
                minloc_end = std::chrono::high_resolution_clock::now();
                minloc_time = std::chrono::duration<double, std::milli>(minloc_end - minloc_start).count();
                total_minloc_time_ms += minloc_time;
                minloc_call_count++;
                float dist1 = dist;

                //then refine
                minloc_start = std::chrono::high_resolution_clock::now();
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.01, 0.0001);
                minloc_end = std::chrono::high_resolution_clock::now();
                minloc_time = std::chrono::duration<double, std::milli>(minloc_end - minloc_start).count();
                total_minloc_time_ms += minloc_time;
                minloc_call_count++;

                if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc)) {
                    break;
                }

            seg.push_back(point3);
            seg_loc.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;
            forward_points++;

            // Note: Removed block checking during trace - it was causing premature termination
            // The iteration limit (100) and dist checks are sufficient to prevent infinite loops
        }
        auto forward_trace_end = std::chrono::high_resolution_clock::now();
        double forward_trace_time = std::chrono::duration<double, std::milli>(forward_trace_end - forward_trace_start).count();

        //now the other direction
        loc2 = seg_loc[0];
        loc = seg_loc[1];
        point2 = seg[0];
        point = seg[1];

        last_plane_loc = plane->project(point2);

        //FIXME repeat by not copying code ...
        auto backward_trace_start = std::chrono::high_resolution_clock::now();
        int backward_points = 0;
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains(cv::Point(loc3))) {
                break;
            }

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                minloc_start = std::chrono::high_resolution_clock::now();
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.01, 0.0001);
                minloc_end = std::chrono::high_resolution_clock::now();
                minloc_time = std::chrono::duration<double, std::milli>(minloc_end - minloc_start).count();
                total_minloc_time_ms += minloc_time;
                minloc_call_count++;
                float dist1 = dist;

                //then refine
                minloc_start = std::chrono::high_resolution_clock::now();
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.01, 0.0001);
                minloc_end = std::chrono::high_resolution_clock::now();
                minloc_time = std::chrono::duration<double, std::milli>(minloc_end - minloc_start).count();
                total_minloc_time_ms += minloc_time;
                minloc_call_count++;

                if (dist < 0 || dist > 1 || !loc_valid_xy(points, loc)) {
                    break;
                }

            seg2.push_back(point3);
            seg_loc2.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;
            backward_points++;

            // Note: Removed block checking during trace - it was causing premature termination
            // The iteration limit (100) and dist checks are sufficient to prevent infinite loops
        }
        auto backward_trace_end = std::chrono::high_resolution_clock::now();
        double backward_trace_time = std::chrono::duration<double, std::milli>(backward_trace_end - backward_trace_start).count();

        std::reverse(seg2.begin(), seg2.end());
        std::reverse(seg_loc2.begin(), seg_loc2.end());

        seg2.insert(seg2.end(), seg.begin(), seg.end());
        seg_loc2.insert(seg_loc2.end(), seg_loc.begin(), seg_loc.end());


        seg_vol_raw.push_back(seg2);
        seg_grid_raw.push_back(seg_loc2);
        consecutive_failures = 0;  // Reset on success
        successful_segments++;

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

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "\n=== find_intersect_segments SUMMARY ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "ms" << std::endl;
    std::cout << "Successful segments: " << successful_segments << std::endl;
    std::cout << "Output segments (after filtering): " << seg_vol.size() << std::endl;
    std::cout << "Consecutive failures at exit: " << consecutive_failures << std::endl;
    std::cout << "Cumulative failures: " << cumulative_failures << std::endl;
    std::cout << "Initial point search attempts: " << initial_point_search_attempts << std::endl;
    std::cout << "\nOPTIMIZATION EFFECTIVENESS:" << std::endl;
    std::cout << "  Inner loop early exits: " << inner_early_exits << std::endl;
    std::cout << "  Total ROI misses avoided: " << total_inner_roi_misses << std::endl;
    std::cout << "  Already visited checks: " << total_inner_already_visited << std::endl;
    double time_saved_estimate = inner_early_exits * 0.15;  // Rough estimate of 0.15ms per avoided full search
    std::cout << "  Estimated time saved: ~" << std::fixed << std::setprecision(1)
              << time_saved_estimate << "ms" << std::endl;
    std::cout << "\nSMART SAMPLING STRATEGY USAGE:" << std::endl;
    std::cout << "  Systematic grid samples: " << systematic_samples_used
              << "/" << systematic_sample_points.size() << std::endl;
    std::cout << "  Seed-based samples: " << seed_samples_used
              << "/" << seed_based_samples.size() << std::endl;
    std::cout << "  Random samples: " << random_samples_used << std::endl;
    int total_samples = systematic_samples_used + seed_samples_used + random_samples_used;
    if (total_samples > 0) {
        std::cout << "  Distribution: "
                  << (systematic_samples_used * 100.0 / total_samples) << "% systematic, "
                  << (seed_samples_used * 100.0 / total_samples) << "% seed-based, "
                  << (random_samples_used * 100.0 / total_samples) << "% random" << std::endl;
    }
    std::cout << "\nmin_loc PERFORMANCE:" << std::endl;
    std::cout << "  Calls: " << minloc_call_count << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_minloc_time_ms << "ms"
              << " (" << std::fixed << std::setprecision(1) << (total_minloc_time_ms / total_time * 100) << "% of total)" << std::endl;
    std::cout << "  Avg time: " << std::fixed << std::setprecision(3)
              << (minloc_call_count > 0 ? total_minloc_time_ms / minloc_call_count : 0) << "ms" << std::endl;

    // Calculate total points in all segments
    int total_points = 0;
    for (const auto& seg : seg_vol) {
        total_points += seg.size();
    }
    std::cout << "\nRESULTS:" << std::endl;
    std::cout << "  Total points in all segments: " << total_points << std::endl;
    if (total_points > 0) {
        std::cout << "  Avg points per segment: " << std::fixed << std::setprecision(1)
                  << (double)total_points / seg_vol.size() << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}
