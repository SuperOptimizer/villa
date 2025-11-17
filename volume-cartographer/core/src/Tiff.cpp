#include "vc/core/util/Tiff.hpp"

#include <tiffio.h>
#include <opencv2/core.hpp>

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>

void writeFloatBigTiff(const std::filesystem::path& outPath,
                      const cv::Mat& img,
                      const TiffWriteOptions& opts)
{
    if (img.empty())
        throw std::runtime_error("empty image for " + outPath.string());
    if (img.type() != CV_32FC1) {
        throw std::runtime_error("expected CV_32FC1 for " + outPath.string());
    }

    TIFF* tf = TIFFOpen(outPath.string().c_str(), opts.forceBigTiff ? "w8" : "w");
    if (!tf)
        throw std::runtime_error("Failed to open TIFF for writing: " + outPath.string());

    const uint32_t W = static_cast<uint32_t>(img.cols);
    const uint32_t H = static_cast<uint32_t>(img.rows);
    const uint32_t tileW = static_cast<uint32_t>(opts.tileSize);
    const uint32_t tileH = static_cast<uint32_t>(opts.tileSize);

    // Core tags
    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH,      W);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH,     H);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE,   32);
    TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT,    SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
    TIFFSetField(tf, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);

    // Compression
    switch (opts.compression) {
        case TiffWriteOptions::Compression::NONE:
            TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
            break;
        case TiffWriteOptions::Compression::LZW:
            TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
            break;
        case TiffWriteOptions::Compression::DEFLATE:
            TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
            break;
    }

    // Predictor
    switch (opts.predictor) {
        case TiffWriteOptions::Predictor::NONE:
            break;
        case TiffWriteOptions::Predictor::HORIZONTAL:
            TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
            break;
        case TiffWriteOptions::Predictor::FLOATINGPOINT:
#ifdef PREDICTOR_FLOATINGPOINT
            TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
#else
            TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
#endif
            break;
    }

    // Tiling
    TIFFSetField(tf, TIFFTAG_TILEWIDTH,  tileW);
    TIFFSetField(tf, TIFFTAG_TILELENGTH, tileH);

    // Write tiles
    const tmsize_t tileBytes = static_cast<tmsize_t>(tileW) *
                               static_cast<tmsize_t>(tileH) *
                               static_cast<tmsize_t>(sizeof(float));
    std::vector<float> tileBuf(static_cast<size_t>(tileW) * tileH, -1.0f); // pad invalid with -1

    for (uint32_t y0 = 0; y0 < H; y0 += tileH) {
        const uint32_t dy = std::min(tileH, H - y0);
        for (uint32_t x0 = 0; x0 < W; x0 += tileW) {
            const uint32_t dx = std::min(tileW, W - x0);

            // Fill tile buffer (pad right/bottom with -1.0f)
            for (uint32_t ty = 0; ty < tileH; ++ty) {
                float* dst = tileBuf.data() + ty * tileW;
                if (ty < dy) {
                    const float* src = img.ptr<float>(static_cast<int>(y0 + ty)) + x0;
                    if (dx > 0) std::memcpy(dst, src, sizeof(float) * dx);
                    if (dx < tileW) std::fill(dst + dx, dst + tileW, -1.0f);
                } else {
                    std::fill(dst, dst + tileW, -1.0f);
                }
            }

            const ttile_t tileIndex = TIFFComputeTile(tf, x0, y0, 0, 0);
            if (TIFFWriteEncodedTile(tf, tileIndex, tileBuf.data(), tileBytes) < 0) {
                TIFFClose(tf);
                throw std::runtime_error("TIFFWriteEncodedTile failed at tile (" +
                                          std::to_string(x0) + "," + std::to_string(y0) +
                                          ") in " + outPath.string());
            }
        }
    }

    if (!TIFFWriteDirectory(tf)) {
        TIFFClose(tf);
        throw std::runtime_error("TIFFWriteDirectory failed for " + outPath.string());
    }
    TIFFClose(tf);
}

void writeSingleChannelBigTiff(const std::filesystem::path& outPath,
                              const cv::Mat& img,
                              const TiffWriteOptions& opts)
{
    if (img.empty())
        throw std::runtime_error("empty image for " + outPath.string());
    if (img.channels() != 1)
        throw std::runtime_error("expected single-channel image for " + outPath.string());

    TIFF* tf = TIFFOpen(outPath.string().c_str(), opts.forceBigTiff ? "w8" : "w");
    if (!tf)
        throw std::runtime_error("Failed to open TIFF: " + outPath.string());

    const uint32_t W = static_cast<uint32_t>(img.cols);
    const uint32_t H = static_cast<uint32_t>(img.rows);
    const uint32_t tileW = static_cast<uint32_t>(opts.tileSize);
    const uint32_t tileH = static_cast<uint32_t>(opts.tileSize);

    TIFFSetField(tf, TIFFTAG_IMAGEWIDTH,      W);
    TIFFSetField(tf, TIFFTAG_IMAGELENGTH,     H);
    TIFFSetField(tf, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tf, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
    TIFFSetField(tf, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tf, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
    TIFFSetField(tf, TIFFTAG_TILEWIDTH,       tileW);
    TIFFSetField(tf, TIFFTAG_TILELENGTH,      tileH);

    // Compression
    switch (opts.compression) {
        case TiffWriteOptions::Compression::NONE:
            TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
            break;
        case TiffWriteOptions::Compression::LZW:
            TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
            break;
        case TiffWriteOptions::Compression::DEFLATE:
            TIFFSetField(tf, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
            break;
    }

    int bits = 0, samplefmt = 0, elem = 0;
    switch (img.type()) {
        case CV_8UC1:
            bits = 8;
            samplefmt = SAMPLEFORMAT_UINT;
            elem = 1;
            break;
        case CV_16UC1:
            bits = 16;
            samplefmt = SAMPLEFORMAT_UINT;
            elem = 2;
            break;
        case CV_32FC1:
            bits = 32;
            samplefmt = SAMPLEFORMAT_IEEEFP;
            elem = 4;
            break;
        default:
            TIFFClose(tf);
            throw std::runtime_error("unsupported channel type for " + outPath.string());
    }

    TIFFSetField(tf, TIFFTAG_BITSPERSAMPLE, bits);
    TIFFSetField(tf, TIFFTAG_SAMPLEFORMAT,  samplefmt);

    // Predictor
    bool useFloatPredictor = (img.type() == CV_32FC1) &&
                            (opts.predictor == TiffWriteOptions::Predictor::FLOATINGPOINT);
    if (useFloatPredictor) {
#ifdef PREDICTOR_FLOATINGPOINT
        TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_FLOATINGPOINT);
#else
        TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
#endif
    } else if (opts.predictor == TiffWriteOptions::Predictor::HORIZONTAL) {
        TIFFSetField(tf, TIFFTAG_PREDICTOR, PREDICTOR_HORIZONTAL);
    }

    const tmsize_t tileBytes = static_cast<tmsize_t>(tileW) *
                               static_cast<tmsize_t>(tileH) *
                               static_cast<tmsize_t>(elem);
    std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes), 0);

    for (uint32_t y0 = 0; y0 < H; y0 += tileH) {
        const uint32_t dy = std::min(tileH, H - y0);
        for (uint32_t x0 = 0; x0 < W; x0 += tileW) {
            const uint32_t dx = std::min(tileW, W - x0);

            // Fill tile (pad with zeros)
            std::fill(tileBuf.begin(), tileBuf.end(), 0);
            for (uint32_t ty = 0; ty < dy; ++ty) {
                const uint8_t* src = img.ptr<uint8_t>(static_cast<int>(y0 + ty)) + x0 * elem;
                std::memcpy(tileBuf.data() + (static_cast<size_t>(ty) * tileW * elem),
                           src,
                           static_cast<size_t>(dx) * elem);
            }

            const ttile_t tileIndex = TIFFComputeTile(tf, x0, y0, 0, 0);
            if (TIFFWriteEncodedTile(tf, tileIndex, tileBuf.data(), tileBytes) < 0) {
                TIFFClose(tf);
                throw std::runtime_error("TIFFWriteEncodedTile failed in channel " + outPath.string());
            }
        }
    }

    if (!TIFFWriteDirectory(tf)) {
        TIFFClose(tf);
        throw std::runtime_error("TIFFWriteDirectory failed for channel " + outPath.string());
    }
    TIFFClose(tf);
}
