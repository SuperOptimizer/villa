#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <z5/dataset.hxx>
#include <z5/filesystem/handle.hxx>

// Map a tile index through rotation + flip (pure integer tile coordinate transform).
// Used by both zarr and tif writers.
inline void mapTileIndex(int tx, int ty, int tilesX, int tilesY,
                         int quadRot, int flipType,
                         int& outTx, int& outTy, int& outTilesX, int& outTilesY)
{
    bool swap = (quadRot % 2) == 1;
    int rTX = swap ? tilesY : tilesX, rTY = swap ? tilesX : tilesY;
    int rx = tx, ry = ty;
    switch (quadRot) {
        case 1: rx = ty;              ry = tilesX - 1 - tx; break;
        case 2: rx = tilesX - 1 - tx; ry = tilesY - 1 - ty; break;
        case 3: rx = tilesY - 1 - ty; ry = tx;              break;
        default: break;
    }
    int fx = rx, fy = ry;
    if (flipType == 0)      fy = rTY - 1 - ry;
    else if (flipType == 1) fx = rTX - 1 - rx;
    else if (flipType == 2) { fx = rTX - 1 - rx; fy = rTY - 1 - ry; }
    outTx = fx; outTy = fy; outTilesX = rTX; outTilesY = rTY;
}

// Write one band's slices as zarr chunks using writeChunk (fast, avoids subarray overhead).
// chunks0 = {chunkZ, chunkY, chunkX} from the L0 dataset's chunk shape.
template <typename T>
void writeZarrBand(z5::Dataset* dsOut, const std::vector<cv::Mat>& slices,
                   uint32_t bandIdx, const std::vector<size_t>& chunks0,
                   size_t tilesXSrc, size_t tilesYSrc,
                   int rotQuad, int flipAxis);

// Build one pyramid level (2x mean downsample) via readSubarray/writeSubarray + OMP.
template <typename T>
void buildPyramidLevel(z5::filesystem::handle::File& outFile, int level,
                       size_t CH, size_t CW);

// Write OME-Zarr .zattrs multiscales JSON.
void writeZarrAttrs(z5::filesystem::handle::File& outFile,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW);

