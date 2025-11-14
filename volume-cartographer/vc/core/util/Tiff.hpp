#pragma once

#include <filesystem>
#include <opencv2/core.hpp>

namespace vc {
namespace tiff {

// Options for writing TIFF files
struct WriteOptions {
    enum class Compression { NONE, LZW, DEFLATE };
    enum class Predictor  { NONE, HORIZONTAL, FLOATINGPOINT };

    bool forceBigTiff = false;
    int  tileSize = 1024;                 // square tiles
    Compression compression = Compression::LZW;
    Predictor  predictor   = Predictor::FLOATINGPOINT; // for float32
};

// Write a 32-bit float single-channel image as tiled BigTIFF with LZW compression
void writeFloatBigTiff(const std::filesystem::path& outPath,
                       const cv::Mat& img,
                       uint32_t tileW = 1024,
                       uint32_t tileH = 1024);

// Write a single-channel image (8U/16U/32F) as tiled BigTIFF
void writeSingleChannelBigTiff(const std::filesystem::path& outPath,
                               const cv::Mat& img,
                               const WriteOptions& opts);

// Normalize mask channel to single 8-bit channel (modifies mask in place)
void normalizeMaskChannel(cv::Mat& mask);

} // namespace tiff
} // namespace vc
