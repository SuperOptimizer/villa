#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <cstdint>

// Consolidated post-processing parameters.
// Replaces scattered postprocessGrayscale/postprocessComposite logic.
struct PostProcessParams {
    // Window/level: map [windowLow, windowHigh] -> [0, 255]
    float windowLow = 0.0f;
    float windowHigh = 255.0f;

    // Stretch: auto-stretch to full range (overrides window/level)
    bool stretchValues = false;

    // Composite-specific post-processing
    bool postStretchValues = false;
    bool removeSmallComponents = false;
    int minComponentSize = 50;

    // ISO cutoff: zero out values below threshold (0 = disabled)
    uint8_t isoCutoff = 0;

    // Colormap (empty = grayscale->BGR)
    std::string colormapId;
};

// Apply all post-processing steps in canonical order:
//   1. ISO cutoff
//   2. Composite post-stretch (if enabled)
//   3. Composite component removal (if enabled)
//   4. Window/level or value stretch
//   5. Colormap application
// Input: single-channel uint8 grayscale. Output: BGR cv::Mat.
cv::Mat applyPostProcess(const cv::Mat_<uint8_t>& gray,
                         const PostProcessParams& params);
