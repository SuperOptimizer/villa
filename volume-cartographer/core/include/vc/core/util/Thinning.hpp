#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

namespace cv {
class Mat;
}  // namespace cv

void customThinning(const cv::Mat& inputImage, cv::Mat& outputImage, std::vector<std::vector<cv::Point>>* traces = nullptr);
