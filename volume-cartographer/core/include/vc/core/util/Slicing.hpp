#pragma once

#include <utils/tensor.hpp>
#include <opencv2/core.hpp>
#include <string>

#include <vc/core/util/Compositing.hpp>
#include <vc/core/types/Sampling.hpp>

// Forward declarations
namespace vc { class Zarr; }
namespace vc::cache { class TieredChunkCache; }
template<typename T> class ChunkCache;

// ============================================================================
// TieredChunkCache API (z5-independent)
// ============================================================================

void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);

// Overloads accepting vc::Sampling enum (supports Nearest, Trilinear, Tricubic)
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);

void readArea3D(utils::Tensor& out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level);

void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    vc::Sampling method = vc::Sampling::Nearest
);

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

cv::Mat_<cv::Vec3f> computeVolumeGradientsNative(
    vc::Zarr* ds,
    const cv::Mat_<cv::Vec3f>& rawPoints,
    float dsScale);

// ============================================================================
// Zarr + ChunkCache API (z5-independent, used by vc_render_tifxyz)
// ============================================================================

void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::Zarr* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache
);

void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::Zarr* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::Zarr* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::Zarr* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::Zarr* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);
