#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>

struct POI
{
    cv::Vec3f p = {0,0,0};
    std::string surfaceId;  // ID of the source surface (for lookup, not ownership)
    cv::Vec3f n = {0,0,0};
};

struct IntersectionLine
{
    std::array<cv::Vec3f, 2> world{};         // 3D points in volume space
    std::array<cv::Vec3f, 2> surfaceParams{}; // QuadSurface ptr-space samples aligned with `world`
};

struct Intersection
{
    std::vector<IntersectionLine> lines;
};
