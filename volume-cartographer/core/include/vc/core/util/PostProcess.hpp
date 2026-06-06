#pragma once

#include <opencv2/core/mat.hpp>
#include <cstdint>

namespace vc {

// Grayscale post-processing parameters (no colormap — Qt-free).
struct PostProcessParams {
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    bool stretchValues = false;
    bool postStretchValues = false;
    bool removeSmallComponents = false;
    int minComponentSize = 50;

    constexpr PostProcessParams() noexcept = default;
};

// Apply grayscale post-processing pipeline in canonical order:
//   1. Composite post-stretch (if enabled)
//   2. Composite component removal (if enabled)
//   3. Window/level or value stretch
// Input/output: single-channel uint8 grayscale.
void applyPostProcess(cv::Mat_<uint8_t>& img, const PostProcessParams& params);

}  // namespace vc
