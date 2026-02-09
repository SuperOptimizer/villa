#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace tiff {

// Compression (same numeric values as libtiff for source compat)
enum Compression : uint16_t {
    None     = 1,
    LZW      = 5,
    PackBits = 32773,
};

// Sample format
enum SampleFormat : uint16_t {
    UInt   = 1,
    Int    = 2,
    Float  = 3,
};

// Photometric
enum Photometric : uint16_t {
    MinIsBlack = 1,
    RGB        = 2,
};

// Read any supported TIFF -> cv::Mat (single or multi-channel)
cv::Mat imread(const std::filesystem::path& path);

// Write cv::Mat -> tiled TIFF (single or multi-channel)
void imwrite(const std::filesystem::path& path, const cv::Mat& img,
             uint16_t compression = LZW);

// Read all IFDs (layers) from a multi-layer TIFF
std::vector<cv::Mat> imreadmulti(const std::filesystem::path& path);

// Write multiple layers as chained IFDs in a single TIFF
void imwritemulti(const std::filesystem::path& path,
                  const std::vector<cv::Mat>& layers);

// Merge partial .partN.tif files into final TIFFs.
// Scans outputPath (or its parent directory) for .partN.tif files,
// merges them tile-by-tile, and removes the part files.
// Returns true on success, false if no part files found or on error.
bool mergeTiffParts(const std::string& outputPath, int numParts);

} // namespace tiff

// Compat aliases so existing callers don't need changes
inline constexpr uint16_t COMPRESSION_NONE     = tiff::None;
inline constexpr uint16_t COMPRESSION_LZW      = tiff::LZW;
inline constexpr uint16_t COMPRESSION_PACKBITS  = tiff::PackBits;
inline constexpr uint16_t COMPRESSION_DEFLATE   = 32946;    // not implemented, kept for parse compat
inline constexpr uint16_t COMPRESSION_ADOBE_DEFLATE = 8;     // ditto

// Write single-channel image (8U, 16U, 32F) as tiled TIFF
// cvType: output type (-1 = same as input). If different, values are scaled:
//         8U<->16U: scale by 257, 8U<->32F: scale by 1/255, 16U<->32F: scale by 1/65535
// compression: compression constant (e.g. COMPRESSION_LZW, COMPRESSION_PACKBITS)
// padValue: value for padding partial tiles (default -1.0f, used for float; int types use 0)
void writeTiff(const std::filesystem::path& outPath,
               const cv::Mat& img,
               int cvType = -1,
               uint32_t tileW = 1024,
               uint32_t tileH = 1024,
               float padValue = -1.0f,
               uint16_t compression = COMPRESSION_LZW);

// TIFF reader for tiled/scanline TIFF files
class TiffReader {
public:
    explicit TiffReader(const std::filesystem::path& path);
    ~TiffReader();

    TiffReader(const TiffReader&) = delete;
    TiffReader& operator=(const TiffReader&) = delete;

    // Metadata
    uint32_t width() const { return _width; }
    uint32_t height() const { return _height; }
    uint32_t tileWidth() const { return _tileW; }
    uint32_t tileHeight() const { return _tileH; }
    uint16_t bitsPerSample() const { return _bps; }
    uint16_t samplesPerPixel() const { return _spp; }
    uint16_t sampleFormat() const { return _sf; }
    uint16_t compression() const { return _compression; }
    bool isTiled() const { return _tileW > 0 && _tileH > 0; }
    int cvType() const;

    // Read entire image as cv::Mat
    cv::Mat readAll();

    // Read single decoded tile into caller-provided buffer
    void readTile(uint32_t tileX, uint32_t tileY, void* buf, size_t bufSize);

    // Tile grid dimensions
    uint32_t tilesAcross() const;
    uint32_t tilesDown() const;

    // Decoded tile size in bytes
    size_t tileBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    uint32_t _width = 0, _height = 0;
    uint32_t _tileW = 0, _tileH = 0;
    uint16_t _bps = 0, _spp = 1, _sf = tiff::UInt;
    uint16_t _compression = tiff::None;
};

// Class for incremental tiled TIFF writing
class TiffWriter {
public:
    // Open a new TIFF file for tiled writing
    // cvType: CV_8UC1, CV_8UC3, CV_16UC1, or CV_32FC1
    // padValue: value for padding partial tiles (used for float; int types use 0)
    TiffWriter(const std::filesystem::path& path,
               uint32_t width, uint32_t height,
               int cvType,
               uint32_t tileW = 1024,
               uint32_t tileH = 1024,
               float padValue = -1.0f,
               uint16_t compression = COMPRESSION_LZW);

    ~TiffWriter();

    // Non-copyable
    TiffWriter(const TiffWriter&) = delete;
    TiffWriter& operator=(const TiffWriter&) = delete;

    // Movable
    TiffWriter(TiffWriter&& other) noexcept;
    TiffWriter& operator=(TiffWriter&& other) noexcept;

    // Write a tile at the given position (should be tile-aligned)
    void writeTile(uint32_t x0, uint32_t y0, const cv::Mat& tile);

    // Write pre-encoded (actually decoded) tile data directly
    void writeRawTile(uint32_t tileX, uint32_t tileY, const void* data, size_t len);

    // Explicitly close the file (also called by destructor)
    void close();

    // Check if file is open
    bool isOpen() const;

    // Accessors
    uint32_t width() const { return _width; }
    uint32_t height() const { return _height; }
    uint32_t tileWidth() const { return _tileW; }
    uint32_t tileHeight() const { return _tileH; }
    int cvType() const { return _cvType; }

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    uint32_t _width = 0;
    uint32_t _height = 0;
    uint32_t _tileW = 0;
    uint32_t _tileH = 0;
    int _cvType = 0;
    int _elemSize = 0;
    float _padValue = -1.0f;
    uint16_t _compression = tiff::LZW;
    std::vector<uint8_t> _tileBuf;
    std::filesystem::path _path;
};
