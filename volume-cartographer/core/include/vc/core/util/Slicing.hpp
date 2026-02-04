#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xlayout.hpp>

#include <vc/core/types/InterpolationMethod.hpp>

// Forward declarations
namespace z5 {
class Dataset;
}  // namespace z5
namespace cv {
template <typename _Tp> class Mat_;
}  // namespace cv
struct CompositeParams;
template <typename T> class ChunkCache;

// Read interpolated 3D data from a z5 dataset with specified interpolation method
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache, InterpolationMethod method);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache, InterpolationMethod method);

// Legacy overloads for backward compatibility (bool nearest_neighbor maps to Nearest/Trilinear)
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache, bool nearest_neighbor=false);

// Read a 3D area from a z5 dataset
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint8_t> *cache);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint16_t> *cache);

// Fast composite rendering - nearest neighbor only, uses ChunkCache directly
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache
);

// Fast composite rendering with constant normal (optimized for plane surfaces)
void readCompositeFastConstantNormal(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache
);

// Bulk multi-slice read with trilinear interpolation.
// Samples basePoints + offsets[i] * stepDirs for each offset, returning one Mat per offset.
// Does a single prefetch pass covering all slices, then samples in parallel.
// basePoints/stepDirs use (X,Y,Z) in Vec3f[0],[1],[2] (same convention as readInterpolated3D coords).
void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Centralized z5 subarray reading to consolidate template instantiations.
// Uses xt::xarray with default (row-major) layout.
// offset is ZYX order (same as z5 dataset shape).
// These functions exist to prevent template bloat from having readSubarray
// instantiated in multiple translation units.
#include <xtensor/containers/xarray.hpp>

void readSubarray3D(xt::xarray<uint8_t>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset);
void readSubarray3D(xt::xarray<uint16_t>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset);
void readSubarray3D(xt::xarray<float>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset);

// Compute volume gradients at native surface resolution (the raw point grid)
// Returns normalized gradient vectors at each raw grid point
// dsScale converts from world coordinates to dataset coordinates
cv::Mat_<cv::Vec3f> computeVolumeGradientsNative(
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& rawPoints,
    float dsScale);
