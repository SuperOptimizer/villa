#pragma once

#include <xtensor/containers/xarray.hpp>
#include <opencv2/core.hpp>
#include <string>

#include <vc/core/util/ChunkCache.hpp>
#include <vc/core/util/Compositing.hpp>

// Forward declaration
namespace z5 { class Dataset; }

// Read interpolated 3D data from a z5 dataset
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
