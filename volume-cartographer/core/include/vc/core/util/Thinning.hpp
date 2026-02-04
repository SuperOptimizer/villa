#pragma once

#include <vector>

#include <opencv2/core/types.hpp>

namespace cv {
class Mat;
}  // namespace cv

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<std::vector<cv::Point>>* traces = nullptr);
