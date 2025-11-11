#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>

struct NearPlanePoint {
    cv::Vec3f point_3d;
    cv::Vec2f grid_pos;
    int index;
};

void find_intersect_segments(
    std::vector<std::vector<cv::Vec3f>>& seg_vol,
    std::vector<std::vector<cv::Vec2f>>& seg_grid,
    const cv::Mat_<cv::Vec3f>& points,
    PlaneSurface* plane,
    const cv::Rect& plane_roi)
{
    if (points.rows < 2 || points.cols < 2) {
        return;
    }

    std::vector<NearPlanePoint> near_points;
    cv::Mat_<int> point_map(points.size(), -1);

    // Simple, generous tolerance
    float tolerance = 5.0f;

    for (int y = 0; y < points.rows; y++) {
        for (int x = 0; x < points.cols; x++) {
            cv::Vec3f p = points(y, x);
            if (p[0] == -1) continue;

            float dist = std::abs(plane->pointDist(p));

            if (dist <= tolerance) {
                cv::Vec3f plane_loc = plane->project(p);
                if (!plane_roi.contains(cv::Point(plane_loc[0], plane_loc[1]))) {
                    continue;
                }

                NearPlanePoint npp;
                npp.point_3d = p;
                npp.grid_pos = cv::Vec2f(x, y);
                npp.index = near_points.size();
                point_map(y, x) = near_points.size();
                near_points.push_back(npp);
            }
        }
    }

    std::cout << "Found " << near_points.size() << " points" << std::endl;

    if (near_points.empty()) {
        return;
    }

    // VERY AGGRESSIVE connectivity to bridge gaps
    std::vector<std::vector<int>> adjacency(near_points.size());

    // Huge search radius and distance to connect fragments
    int grid_search_radius = 15;  // Search VERY far
    float max_3d_distance = 500.0f;  // Allow huge gaps

    for (size_t i = 0; i < near_points.size(); i++) {
        int gx = (int)near_points[i].grid_pos[0];
        int gy = (int)near_points[i].grid_pos[1];

        for (int dy = -grid_search_radius; dy <= grid_search_radius; dy++) {
            for (int dx = -grid_search_radius; dx <= grid_search_radius; dx++) {
                if (dx == 0 && dy == 0) continue;

                int nx = gx + dx;
                int ny = gy + dy;

                if (nx < 0 || nx >= points.cols || ny < 0 || ny >= points.rows) {
                    continue;
                }

                int neighbor_idx = point_map(ny, nx);
                if (neighbor_idx >= 0 && neighbor_idx != i) {
                    float dist_3d = cv::norm(near_points[i].point_3d - near_points[neighbor_idx].point_3d);

                    if (dist_3d <= max_3d_distance) {
                        adjacency[i].push_back(neighbor_idx);
                    }
                }
            }
        }
    }

    // BFS
    std::vector<bool> used(near_points.size(), false);
    int total_segments = 0;

    for (size_t i = 0; i < near_points.size(); i++) {
        if (used[i]) continue;

        std::vector<int> segment_indices;
        std::queue<int> to_visit;
        to_visit.push(i);
        used[i] = true;

        while (!to_visit.empty()) {
            int current_idx = to_visit.front();
            to_visit.pop();
            segment_indices.push_back(current_idx);

            for (int neighbor_idx : adjacency[current_idx]) {
                if (!used[neighbor_idx]) {
                    used[neighbor_idx] = true;
                    to_visit.push(neighbor_idx);
                }
            }
        }

        if (segment_indices.size() < 2) {
            continue;
        }

        // Order points
        std::vector<bool> visited(segment_indices.size(), false);
        std::vector<cv::Vec3f> sorted_3d;
        std::vector<cv::Vec2f> sorted_grid;

        // Find endpoint
        int start_idx = 0;
        int min_degree = INT_MAX;

        for (size_t j = 0; j < segment_indices.size(); j++) {
            int idx = segment_indices[j];
            int degree = 0;

            for (int neighbor : adjacency[idx]) {
                if (std::find(segment_indices.begin(), segment_indices.end(), neighbor) != segment_indices.end()) {
                    degree++;
                }
            }

            if (degree < min_degree) {
                min_degree = degree;
                start_idx = j;
            }
        }

        visited[start_idx] = true;
        int current_pt_idx = segment_indices[start_idx];
        sorted_3d.push_back(near_points[current_pt_idx].point_3d);
        sorted_grid.push_back(near_points[current_pt_idx].grid_pos);

        for (size_t j = 1; j < segment_indices.size(); j++) {
            cv::Vec3f last = sorted_3d.back();
            float min_dist = FLT_MAX;
            int best = -1;
            int best_seg_idx = -1;

            // Prefer connected neighbors
            for (int neighbor : adjacency[current_pt_idx]) {
                auto it = std::find(segment_indices.begin(), segment_indices.end(), neighbor);
                if (it != segment_indices.end()) {
                    int seg_idx = it - segment_indices.begin();
                    if (!visited[seg_idx]) {
                        float d = cv::norm(near_points[neighbor].point_3d - last);
                        if (d < min_dist) {
                            min_dist = d;
                            best = neighbor;
                            best_seg_idx = seg_idx;
                        }
                    }
                }
            }

            // Fall back to nearest
            if (best < 0) {
                for (size_t k = 0; k < segment_indices.size(); k++) {
                    if (visited[k]) continue;
                    int idx = segment_indices[k];
                    float d = cv::norm(near_points[idx].point_3d - last);
                    if (d < min_dist) {
                        min_dist = d;
                        best = idx;
                        best_seg_idx = k;
                    }
                }
            }

            if (best >= 0) {
                visited[best_seg_idx] = true;
                current_pt_idx = best;
                sorted_3d.push_back(near_points[best].point_3d);
                sorted_grid.push_back(near_points[best].grid_pos);
            }
        }

        seg_vol.push_back(sorted_3d);
        seg_grid.push_back(sorted_grid);
        total_segments++;
    }

    std::cout << "Created " << total_segments << " segments" << std::endl;
}