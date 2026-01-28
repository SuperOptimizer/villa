#pragma once

#include "vc/core/zarr/Tensor3D.hpp"
#include "vc/core/zarr/ZarrDataset.hpp"
#include <opencv2/core.hpp>
#include <string>

#include <vc/core/util/ChunkCache.hpp>
#include <vc/core/util/Compositing.hpp>

// Read interpolated 3D data from a zarr dataset
void readInterpolated3D(cv::Mat_<uint8_t> &out, volcart::zarr::ZarrDataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, volcart::zarr::ZarrDataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache, bool nearest_neighbor=false);

// Read a 3D area from a zarr dataset
void readArea3D(volcart::zarr::Tensor3D<uint8_t> &out, const cv::Vec3i& offset, volcart::zarr::ZarrDataset *ds, ChunkCache<uint8_t> *cache);
void readArea3D(volcart::zarr::Tensor3D<uint16_t> &out, const cv::Vec3i& offset, volcart::zarr::ZarrDataset *ds, ChunkCache<uint16_t> *cache);

// Fast composite rendering - nearest neighbor only, uses ChunkCache directly
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    volcart::zarr::ZarrDataset* ds,
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
    volcart::zarr::ZarrDataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache
);
