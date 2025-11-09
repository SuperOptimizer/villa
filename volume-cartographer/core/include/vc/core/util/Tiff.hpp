#include <tiffio.h>


void writeSingleChannelBigTiff(const std::filesystem::path& outPath,
const cv::Mat& img,
uint32_t tileW = 1024,
uint32_t tileH = 1024);

void writeFloatBigTiff(const std::filesystem::path& outPath,
                              const cv::Mat& img,
                              uint32_t tileW = 1024,
                              uint32_t tileH = 1024);