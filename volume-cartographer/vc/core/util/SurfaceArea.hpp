#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"

namespace vc::surface
{
struct MaskAreaOptions
{
    bool insideIsNonZero = true;
};

struct MaskAreaResult
{
    double area_vx2 = 0.0;
    double area_cm2 = 0.0;
    double median_step_u = 0.0;
    double median_step_v = 0.0;
    std::size_t contributing_quads = 0;
    std::size_t inside_pixels = 0;
};

namespace detail
{
template <typename Vec>
bool isFiniteVec(const Vec& v);

template <typename Vec>
bool isSentinelInvalid(const Vec& v);

template <typename Vec>
double triangleArea(const Vec& a, const Vec& b, const Vec& c);

template <typename Vec>
double quadArea(const Vec& p00,
                const Vec& p10,
                const Vec& p01,
                const Vec& p11);

template <typename Vec>
double surfaceArea(const cv::Mat_<Vec>& points);

double median(std::vector<double>& values);

struct MaskConversion
{
    cv::Mat mask;
    std::size_t insidePixelCount = 0;
};

MaskConversion toBinaryMask(const cv::Mat& input, bool insideIsNonZero);
} // namespace detail

double triangleAreaVox2(const cv::Vec3d& a, const cv::Vec3d& b, const cv::Vec3d& c);

double triangleAreaVox2(const cv::Vec3f& a, const cv::Vec3f& b, const cv::Vec3f& c);

double quadAreaVox2(const cv::Vec3d& p00,
                    const cv::Vec3d& p10,
                    const cv::Vec3d& p01,
                    const cv::Vec3d& p11);

double quadAreaVox2(const cv::Vec3f& p00,
                    const cv::Vec3f& p10,
                    const cv::Vec3f& p01,
                    const cv::Vec3f& p11);

double computeSurfaceAreaVox2(const cv::Mat_<cv::Vec3f>& points);

double computeSurfaceAreaVox2(const cv::Mat_<cv::Vec3d>& points);

double computeSurfaceAreaVox2(const QuadSurface& surface);

bool computeMaskAreaFromGrid(const cv::Mat_<cv::Vec3f>& coords,
                              const cv::Mat& maskInput,
                              double voxelSize,
                              const MaskAreaOptions& options,
                              MaskAreaResult& result,
                              std::string* error = nullptr);

bool computeMaskArea(QuadSurface& surface,
                     const cv::Mat& mask,
                     double voxelSize,
                     const MaskAreaOptions& options,
                     MaskAreaResult& result,
                     std::string* error = nullptr);

} // namespace vc::surface
