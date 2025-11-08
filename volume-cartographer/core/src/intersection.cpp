#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

// Intersection detection code

void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step, int min_tries)
{
    seg_vol.clear();
    seg_grid.clear();

    if (!plane || points.empty()) {
        return;
    }

    // Get plane parameters
    cv::Vec3f planeOrigin = plane->origin();
    cv::Vec3f ptr = plane->pointer();
    cv::Vec3f planeNormal = plane->normal(ptr, {0,0,0});

    // Normalize the normal vector (should already be normalized, but be safe)
    float normalLen = cv::norm(planeNormal);
    if (normalLen < 0.001f) {
        return;
    }
    planeNormal = planeNormal / normalLen;

    // Collect points that are close to the plane
    std::vector<cv::Vec3f> currentSegment;

    for (int row = 0; row < points.rows; ++row) {
        for (int col = 0; col < points.cols; ++col) {
            const cv::Vec3f& pt = points(row, col);

            // Skip invalid points (all zeros)
            if (pt[0] == 0 && pt[1] == 0 && pt[2] == 0) {
                if (!currentSegment.empty()) {
                    if (currentSegment.size() >= 2) {
                        seg_vol.push_back(currentSegment);
                    }
                    currentSegment.clear();
                }
                continue;
            }

            // Calculate distance from point to plane
            cv::Vec3f toPoint = pt - planeOrigin;
            float distance = std::abs(toPoint.dot(planeNormal));

            // If point is close enough to the plane, add it to current segment
            if (distance < step) {
                currentSegment.push_back(pt);
            } else {
                // Point is far from plane, finish current segment
                if (!currentSegment.empty()) {
                    if (currentSegment.size() >= 2) {
                        seg_vol.push_back(currentSegment);
                    }
                    currentSegment.clear();
                }
            }
        }
    }

    // Add final segment if any
    if (currentSegment.size() >= 2) {
        seg_vol.push_back(currentSegment);
    }
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

Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p)
{
    Rect3D res = a;
    for(int d=0;d<3;d++) {
        res.low[d] = std::min(res.low[d], p[d]);
        res.high[d] = std::max(res.high[d], p[d]);
    }

    return res;
}
