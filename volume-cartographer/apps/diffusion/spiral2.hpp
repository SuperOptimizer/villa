#pragma once

#include "spiral_common.hpp"
#include <vc/core/PointCollections.hpp>

int spiral2_main(
    const cv::Mat& slice_mat,
    const PointCollections& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    const po::variables_map& vm
);
