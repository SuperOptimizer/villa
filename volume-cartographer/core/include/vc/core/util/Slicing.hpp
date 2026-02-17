#pragma once

#include <xtensor/containers/xarray.hpp>
#include <opencv2/core.hpp>
#include <string>

#include <vc/core/cache/TieredChunkCache.hpp>
#include <vc/core/util/Compositing.hpp>
#include <vc/core/types/Sampling.hpp>

// Forward declaration
namespace vc { class VcDataset; }

// Read interpolated 3D data from a zarr dataset via TieredChunkCache
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);

// Overloads accepting vc::Sampling enum (supports Nearest, Trilinear, Tricubic)
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);

// Read a 3D area from a zarr dataset via TieredChunkCache
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level);

// Fast composite rendering - nearest neighbor only
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params
);

// Bulk multi-slice read with trilinear interpolation.
void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Single-threaded per-tile multi-slice sampler (called from within OMP thread).
void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Compute volume gradients at native surface resolution (the raw point grid).
// Returns normalized gradient vectors at each raw grid point.
// dsScale converts from world coordinates to dataset coordinates.
// Uses VcDataset directly for batch reading (not cached).
cv::Mat_<cv::Vec3f> computeVolumeGradientsNative(
    vc::VcDataset* ds,
    const cv::Mat_<cv::Vec3f>& rawPoints,
    float dsScale);
