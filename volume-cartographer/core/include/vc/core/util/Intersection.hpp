#pragma once

#include <vector>
#include <opencv2/core.hpp>

class PlaneSurface;

// Find intersection segments between a quad surface and a plane
void find_intersect_segments(
    std::vector<std::vector<cv::Vec3f>>& seg_vol,
    std::vector<std::vector<cv::Vec2f>>& seg_grid,
    const cv::Mat_<cv::Vec3f>& points,
    PlaneSurface* plane,
    const cv::Rect& plane_roi,
    float step,
    int min_tries = 10);

// Search location in points where we minimize error to multiple objectives using iterated local search
// tgts,tds -> distance to some POIs
// plane -> stay on plane
float min_loc(
    const cv::Mat_<cv::Vec3f>& points,
    cv::Vec2f& loc,
    cv::Vec3f& out,
    const std::vector<cv::Vec3f>& tgts,
    const std::vector<float>& tds,
    PlaneSurface* plane,
    float init_step = 16.0,
    float min_step = 0.125);
