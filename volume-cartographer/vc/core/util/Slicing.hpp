#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include <opencv2/core.hpp>

#include <vc/core/util/ChunkCache.hpp>
#include <vc/core/util/GeometryUtils.hpp>

#include <z5/dataset.hxx>

//NOTE depending on request this might load a lot (the whole array) into RAM
// void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache<uint8_t> *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache = nullptr, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache = nullptr, bool nearest_neighbor=false);

template <typename T>
void readArea3D(xt::xtensor<T,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache<T> *cache);
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points);
cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);

// Import geometry utility functions from vc::utils namespace
// These functions are now implemented in GeometryUtils.cpp with explicit instantiations
using vc::utils::at_int;
using vc::utils::loc_valid;
using vc::utils::loc_valid_xy;
using vc::utils::internal_loc;
using vc::utils::nominal_loc;
using vc::utils::grid_normal;
