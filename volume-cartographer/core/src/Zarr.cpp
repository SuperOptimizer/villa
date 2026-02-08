#include "vc/core/util/Zarr.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <omp.h>
#include <z5/factory.hxx>
#include <z5/multiarray/xtensor_access.hxx>
#include <xtensor/containers/xarray.hpp>

using json = nlohmann::json;

// ============================================================
// writeZarrBand
// ============================================================

template <typename T>
void writeZarrBand(z5::Dataset* dsOut, const std::vector<cv::Mat>& slices,
                   uint32_t bandIdx, const std::vector<size_t>& chunks0,
                   size_t tilesXSrc, size_t tilesYSrc,
                   int rotQuad, int flipAxis)
{
    int outH = slices[0].rows, outW = slices[0].cols;
    size_t numZ = slices.size();
    size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];

    for (size_t tx = 0; tx < tilesXSrc; tx++) {
        size_t x0s = tx * chunkX;
        size_t dxc = std::min(chunkX, size_t(outW) - x0s);

        int dstTx = int(tx), dstTy = int(bandIdx), dTX, dTY;
        if (rotQuad >= 0 || flipAxis >= 0)
            mapTileIndex(int(tx), int(bandIdx), int(tilesXSrc), int(tilesYSrc),
                         std::max(rotQuad, 0), flipAxis, dstTx, dstTy, dTX, dTY);

        std::vector<T> chunkBuf(chunkZ * chunkY * chunkX, T(0));
        size_t dy_actual = std::min(chunkY, size_t(outH));
        size_t dx_actual = std::min(chunkX, dxc);
        for (size_t zi = 0; zi < numZ; zi++) {
            size_t sliceOff = zi * chunkY * chunkX;
            for (size_t yy = 0; yy < dy_actual; yy++) {
                const T* row = slices[zi].ptr<T>(int(yy));
                std::memcpy(&chunkBuf[sliceOff + yy * chunkX], &row[x0s], dx_actual * sizeof(T));
            }
        }
        z5::types::ShapeType chunkId = {0, size_t(dstTy), size_t(dstTx)};
        dsOut->writeChunk(chunkId, chunkBuf.data());
    }
}

template void writeZarrBand<uint8_t>(z5::Dataset*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);
template void writeZarrBand<uint16_t>(z5::Dataset*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);

// ============================================================
// buildPyramidLevel
// ============================================================

template <typename T>
void buildPyramidLevel(z5::filesystem::handle::File& outFile, int level,
                       size_t CH, size_t CW)
{
    auto src = z5::openDataset(outFile, std::to_string(level - 1));
    const auto& ss = src->shape();
    std::vector<size_t> ds = {(ss[0]+1)/2, (ss[1]+1)/2, (ss[2]+1)/2};
    std::vector<size_t> dc = {ds[0], std::min(CH, ds[1]), std::min(CW, ds[2])};
    json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};

    std::string dtype = std::is_same_v<T, uint16_t> ? "uint16" : "uint8";
    std::unique_ptr<z5::Dataset> dst;
    try { dst = z5::createDataset(outFile, std::to_string(level), dtype, ds, dc, std::string("blosc"), compOpts); }
    catch (const std::invalid_argument&) { dst = z5::openDataset(outFile, std::to_string(level)); }

    size_t totalTiles = ((ds[1]+CH-1)/CH) * ((ds[2]+CW-1)/CW);
    std::atomic<size_t> done{0};

    for (size_t z = 0; z < ds[0]; z += ds[0]) {
        size_t lz = std::min(ds[0], ds[0] - z);
        #pragma omp parallel for schedule(dynamic, 2)
        for (long long y = 0; y < (long long)ds[1]; y += CH)
            for (long long x = 0; x < (long long)ds[2]; x += CW) {
                size_t ly = std::min(CH, size_t(ds[1] - y)), lx = std::min(CW, size_t(ds[2] - x));
                size_t sz = std::min<size_t>(2*lz, ss[0] - 2*z);
                size_t sy = std::min<size_t>(2*ly, ss[1] - 2*y);
                size_t sx = std::min<size_t>(2*lx, ss[2] - 2*x);

                xt::xarray<T> sc = xt::empty<T>({sz, sy, sx});
                z5::types::ShapeType so = {2*z, size_t(2*y), size_t(2*x)};
                z5::multiarray::readSubarray<T>(src, sc, so.begin());

                xt::xarray<T> dca = xt::empty<T>({lz, ly, lx});
                for (size_t zz = 0; zz < lz; zz++)
                    for (size_t yy = 0; yy < ly; yy++)
                        for (size_t xx = 0; xx < lx; xx++) {
                            uint32_t sum = 0; int cnt = 0;
                            for (int d0 = 0; d0 < 2 && 2*zz+d0 < sz; d0++)
                                for (int d1 = 0; d1 < 2 && 2*yy+d1 < sy; d1++)
                                    for (int d2 = 0; d2 < 2 && 2*xx+d2 < sx; d2++) {
                                        sum += sc(2*zz+d0, 2*yy+d1, 2*xx+d2); cnt++;
                                    }
                            dca(zz, yy, xx) = T((sum + cnt/2) / std::max(1, cnt));
                        }

                z5::types::ShapeType doff = {z, size_t(y), size_t(x)};
                z5::multiarray::writeSubarray<T>(dst, dca, doff.begin());

                size_t d = ++done;
                #pragma omp critical(pp)
                { std::cout << "\r[pyramid L" << level << "] " << d << "/" << totalTiles
                            << " (" << int(100.0*d/totalTiles) << "%)" << std::flush; }
            }
    }
    std::cout << std::endl;
}

template void buildPyramidLevel<uint8_t>(z5::filesystem::handle::File&, int, size_t, size_t);
template void buildPyramidLevel<uint16_t>(z5::filesystem::handle::File&, int, size_t, size_t);

// ============================================================
// writeZarrAttrs
// ============================================================

void writeZarrAttrs(z5::filesystem::handle::File& outFile,
                    const std::filesystem::path& volPath, int groupIdx,
                    size_t baseZ, double sliceStep, double accumStep,
                    const std::string& accumTypeStr, size_t accumSamples,
                    const cv::Size& canvasSize, size_t CZ, size_t CH, size_t CW)
{
    json attrs;
    attrs["source_zarr"] = volPath.string();
    attrs["source_group"] = groupIdx;
    attrs["num_slices"] = baseZ;
    attrs["slice_step"] = sliceStep;
    if (accumSamples > 0) {
        attrs["accum_step"] = accumStep;
        attrs["accum_type"] = accumTypeStr;
        attrs["accum_samples"] = int(accumSamples);
    }
    attrs["canvas_size"] = {canvasSize.width, canvasSize.height};
    attrs["chunk_size"] = {int(CZ), int(CH), int(CW)};
    attrs["note_axes_order"] = "ZYX (slice, row, col)";

    json ms;
    ms["version"] = "0.4"; ms["name"] = "render";
    ms["axes"] = json::array({
        json{{"name","z"},{"type","space"}},
        json{{"name","y"},{"type","space"}},
        json{{"name","x"},{"type","space"}}
    });
    ms["datasets"] = json::array();
    for (int l = 0; l <= 5; l++) {
        double s = std::pow(2.0, l);
        ms["datasets"].push_back({
            {"path", std::to_string(l)},
            {"coordinateTransformations", json::array({
                json{{"type","scale"},{"scale",json::array({s,s,s})}},
                json{{"type","translation"},{"translation",json::array({0.0,0.0,0.0})}}
            })}
        });
    }
    ms["metadata"] = json{{"downsampling_method","mean"}};
    attrs["multiscales"] = json::array({ms});
    z5::filesystem::writeAttributes(outFile, attrs);
}

