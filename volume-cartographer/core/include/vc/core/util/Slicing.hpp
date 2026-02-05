#pragma once

// Full Slicing header — includes SlicingLite.hpp (cv::Mat-only functions)
// plus xtensor-dependent declarations (readArea3D, readSubarray3D, writeSubarray3D).

#include "vc/core/util/SlicingLite.hpp"

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/containers/xarray.hpp>

// Forward declarations (z5, ChunkCache already declared in SlicingLite.hpp)

// Read a 3D area from a z5 dataset
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint8_t> *cache);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint16_t> *cache);

// Centralized z5 subarray reading to consolidate template instantiations.
// Uses xt::xarray with default (row-major) layout.
// offset is ZYX order (same as z5 dataset shape).
// These functions exist to prevent template bloat from having readSubarray
// instantiated in multiple translation units.
void readSubarray3D(xt::xarray<uint8_t>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset);
void readSubarray3D(xt::xarray<uint16_t>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset);
void readSubarray3D(xt::xarray<float>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset);

// Centralized z5 subarray writing (same pattern — prevents writeSubarray
// template bloat across multiple translation units).
void writeSubarray3D(z5::Dataset& ds, const xt::xarray<uint8_t>& data, const std::vector<std::size_t>& offset);
void writeSubarray3D(z5::Dataset& ds, const xt::xarray<uint16_t>& data, const std::vector<std::size_t>& offset);
void writeSubarray3D(z5::Dataset& ds, const xt::xarray<float>& data, const std::vector<std::size_t>& offset);
