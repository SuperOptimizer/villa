#pragma once

#include <filesystem>
#include <opencv2/core.hpp>

// Options for writing TIFF from QuadSurface
struct TiffWriteOptions {
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
                      const TiffWriteOptions& opts = TiffWriteOptions{true, 1024,
                                                                        TiffWriteOptions::Compression::LZW,
                                                                        TiffWriteOptions::Predictor::FLOATINGPOINT});

// Write a single-channel image (8U, 16U, or 32F) as tiled BigTIFF
void writeSingleChannelBigTiff(const std::filesystem::path& outPath,
                              const cv::Mat& img,
                              const TiffWriteOptions& opts = TiffWriteOptions{true, 1024,
                                                                                TiffWriteOptions::Compression::LZW,
                                                                                TiffWriteOptions::Predictor::FLOATINGPOINT});
