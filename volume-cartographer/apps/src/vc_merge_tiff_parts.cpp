// Merge TIFF part files produced by vc_render_tifxyz --num-parts into
// contiguous images, then delete the part directories.
//
// Usage:
//   vc_merge_tiff_parts --output <dir> --num-parts <N>
//
// Expects part directories named <dir>_part_000, <dir>_part_001, etc.
// Each part dir contains numbered TIFF slices (00.tif, 01.tif, ...).
// The merged images are written to <dir>/ and the part dirs are removed.

#include "vc/core/util/Tiff.hpp"

#include <boost/program_options.hpp>
#include <tiffio.h>
#include <opencv2/core.hpp>

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstring>

namespace po = boost::program_options;
namespace fs = std::filesystem;

// Read a tiled TIFF into a cv::Mat
static cv::Mat readTiledTiff(const fs::path& path) {
    TIFF* tif = TIFFOpen(path.string().c_str(), "r");
    if (!tif) {
        std::cerr << "Error: cannot open " << path << "\n";
        return {};
    }

    uint32_t w, h, tileW, tileH;
    uint16_t bps, spp;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileW);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);

    int cvType;
    if (bps == 8) cvType = CV_8UC1;
    else if (bps == 16) cvType = CV_16UC1;
    else if (bps == 32) cvType = CV_32FC1;
    else {
        std::cerr << "Error: unsupported bps=" << bps << " in " << path << "\n";
        TIFFClose(tif);
        return {};
    }

    cv::Mat img = cv::Mat::zeros(h, w, cvType);
    int elemSize = (int)CV_ELEM_SIZE(cvType);

    std::vector<uint8_t> buf(TIFFTileSize(tif));

    for (uint32_t y = 0; y < h; y += tileH) {
        for (uint32_t x = 0; x < w; x += tileW) {
            TIFFReadTile(tif, buf.data(), x, y, 0, 0);
            uint32_t copyW = std::min(tileW, w - x);
            uint32_t copyH = std::min(tileH, h - y);
            for (uint32_t row = 0; row < copyH; ++row) {
                std::memcpy(img.ptr(y + row) + x * elemSize,
                            buf.data() + row * tileW * elemSize,
                            copyW * elemSize);
            }
        }
    }

    TIFFClose(tif);
    return img;
}

// List sorted TIFF files in a directory
static std::vector<fs::path> listTiffs(const fs::path& dir) {
    std::vector<fs::path> result;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return result;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".tif" || e.path().extension() == ".tiff")
            result.push_back(e.path());
    }
    std::sort(result.begin(), result.end());
    return result;
}

int main(int argc, char* argv[]) {
    po::options_description opts("vc_merge_tiff_parts options");
    opts.add_options()
        ("help,h", "Show help")
        ("output,o", po::value<std::string>()->required(), "Output directory (same as --output used for rendering)")
        ("num-parts", po::value<int>()->required(), "Number of parts to merge")
        ("tile-width", po::value<int>()->default_value(64), "Output TIFF tile width")
        ("tile-height", po::value<int>()->default_value(64), "Output TIFF tile height");

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(opts).run(), vm);
        if (vm.count("help")) {
            std::cout << opts << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\nUse --help for usage\n";
        return 1;
    }

    fs::path outDir = vm["output"].as<std::string>();
    int numParts = vm["num-parts"].as<int>();
    int tileW = vm["tile-width"].as<int>();
    int tileH = vm["tile-height"].as<int>();

    if (numParts < 2) {
        std::cerr << "Error: --num-parts must be >= 2\n";
        return 1;
    }

    // Build part directory paths
    std::vector<fs::path> partDirs(numParts);
    for (int i = 0; i < numParts; ++i) {
        std::ostringstream suffix;
        suffix << "_part_" << std::setw(3) << std::setfill('0') << i;
        partDirs[i] = outDir.string() + suffix.str();
        if (!fs::exists(partDirs[i])) {
            std::cerr << "Error: part directory not found: " << partDirs[i] << "\n";
            return 1;
        }
    }

    // Get slice list from first part
    auto sliceFiles = listTiffs(partDirs[0]);
    if (sliceFiles.empty()) {
        std::cerr << "Error: no TIFF files found in " << partDirs[0] << "\n";
        return 1;
    }

    size_t numSlices = sliceFiles.size();
    std::cout << "Merging " << numParts << " parts, " << numSlices << " slices each\n";

    // Verify all parts have same number of slices
    for (int p = 0; p < numParts; ++p) {
        auto files = listTiffs(partDirs[p]);
        if (files.size() != numSlices) {
            std::cerr << "Error: part " << p << " has " << files.size()
                      << " slices, expected " << numSlices << "\n";
            return 1;
        }
    }

    // Get dimensions: width from first part, total height = sum of part heights
    // Read first slice of first part to get width and type
    cv::Mat sample = readTiledTiff(sliceFiles[0]);
    if (sample.empty()) return 1;

    int fullWidth = sample.cols;
    int cvType = sample.type();

    // Get heights of each part
    std::vector<int> partHeights(numParts);
    std::vector<std::vector<fs::path>> allPartFiles(numParts);
    allPartFiles[0] = sliceFiles;
    partHeights[0] = sample.rows;

    for (int p = 1; p < numParts; ++p) {
        allPartFiles[p] = listTiffs(partDirs[p]);
        cv::Mat s = readTiledTiff(allPartFiles[p][0]);
        if (s.empty()) return 1;
        partHeights[p] = s.rows;
        if (s.cols != fullWidth) {
            std::cerr << "Error: part " << p << " width " << s.cols
                      << " != expected " << fullWidth << "\n";
            return 1;
        }
    }

    int fullHeight = 0;
    for (int h : partHeights) fullHeight += h;

    std::cout << "Output: " << fullWidth << " x " << fullHeight
              << " (" << (cvType == CV_16UC1 ? "uint16" : cvType == CV_8UC1 ? "uint8" : "float32")
              << ")\n";

    fs::create_directories(outDir);

    // Merge each slice
    for (size_t z = 0; z < numSlices; ++z) {
        // Determine output filename from part 0's filename
        fs::path outFile = outDir / allPartFiles[0][z].filename();

        TiffWriter writer(outFile, fullWidth, fullHeight, cvType, tileW, tileH, 0.0f);

        int yOffset = 0;
        for (int p = 0; p < numParts; ++p) {
            cv::Mat partImg = readTiledTiff(allPartFiles[p][z]);
            if (partImg.empty()) {
                std::cerr << "Error reading " << allPartFiles[p][z] << "\n";
                return 1;
            }

            // Write tiles from this part
            for (int ty = 0; ty < partImg.rows; ty += tileH) {
                for (int tx = 0; tx < partImg.cols; tx += tileW) {
                    int tw = std::min(tileW, partImg.cols - tx);
                    int th = std::min(tileH, partImg.rows - ty);
                    cv::Mat tile = partImg(cv::Rect(tx, ty, tw, th));
                    writer.writeTile(tx, yOffset + ty, tile);
                }
            }
            yOffset += partImg.rows;
        }
        writer.close();

        if ((z + 1) % 1 == 0 || z + 1 == numSlices) {
            std::cout << "\r[merge] " << (z + 1) << "/" << numSlices << std::flush;
        }
    }
    std::cout << "\n";

    // Delete part directories
    for (int p = 0; p < numParts; ++p) {
        std::cout << "Removing " << partDirs[p] << "\n";
        fs::remove_all(partDirs[p]);
    }

    std::cout << "Done. Merged output in " << outDir << "\n";
    return 0;
}
