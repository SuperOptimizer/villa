#include "vc/core/util/Zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <omp.h>
#include <z5/factory.hxx>

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
    const auto& sc_shape = src->defaultChunkShape();
    size_t scZ = sc_shape[0], scY = sc_shape[1], scX = sc_shape[2];
    size_t srcChunksY = (ss[1] + scY - 1) / scY;
    size_t srcChunksX = (ss[2] + scX - 1) / scX;

    std::vector<size_t> ds = {(ss[0]+1)/2, (ss[1]+1)/2, (ss[2]+1)/2};
    std::vector<size_t> dc = {ds[0], std::min(CH, ds[1]), std::min(CW, ds[2])};
    json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};

    std::string dtype = std::is_same_v<T, uint16_t> ? "uint16" : "uint8";
    std::unique_ptr<z5::Dataset> dst;
    try { dst = z5::createDataset(outFile, std::to_string(level), dtype, ds, dc, std::string("blosc"), compOpts); }
    catch (const std::invalid_argument&) { dst = z5::openDataset(outFile, std::to_string(level)); }

    size_t dstChunksY = (ds[1] + dc[1] - 1) / dc[1];
    size_t dstChunksX = (ds[2] + dc[2] - 1) / dc[2];
    size_t totalTiles = dstChunksY * dstChunksX;
    std::atomic<size_t> done{0};

    size_t srcChunkElems = scZ * scY * scX;
    size_t dstChunkElems = dc[0] * dc[1] * dc[2];

    #pragma omp parallel for schedule(dynamic, 2)
    for (long long ti = 0; ti < (long long)totalTiles; ti++) {
        size_t dty = size_t(ti) / dstChunksX;
        size_t dtx = size_t(ti) % dstChunksX;

        // Dst chunk covers dst voxels [dy0..dy1) x [dx0..dx1) in Y,X
        size_t dy0 = dty * dc[1], dy1 = std::min(dy0 + dc[1], ds[1]);
        size_t dx0 = dtx * dc[2], dx1 = std::min(dx0 + dc[2], ds[2]);
        size_t dly = dy1 - dy0, dlx = dx1 - dx0, dlz = ds[0];

        // Source region is 2x in Y,X
        size_t sy0 = 2 * dy0, sx0 = 2 * dx0;
        size_t sly = std::min(2 * dly, ss[1] - sy0);
        size_t slx = std::min(2 * dlx, ss[2] - sx0);
        size_t slz = std::min(2 * dlz, ss[0]);

        // Read up to 2x2 source chunks (Z is always chunk 0) into a
        // contiguous source buffer with strides [slz, sly, slx]
        std::vector<T> srcBuf(slz * sly * slx, T(0));

        // Source chunk indices that overlap our region
        size_t scy0 = sy0 / scY, scy1 = (sy0 + sly + scY - 1) / scY;
        size_t scx0 = sx0 / scX, scx1 = (sx0 + slx + scX - 1) / scX;

        std::vector<T> chunkTmp(srcChunkElems);
        for (size_t cy = scy0; cy < scy1 && cy < srcChunksY; cy++) {
            for (size_t cx = scx0; cx < scx1 && cx < srcChunksX; cx++) {
                z5::types::ShapeType cid = {0, cy, cx};
                if (!src->chunkExists(cid)) continue;
                src->readChunk(cid, chunkTmp.data());

                // Copy relevant portion into srcBuf
                size_t cyStart = cy * scY, cxStart = cx * scX;
                size_t copyY0 = std::max(cyStart, sy0);
                size_t copyY1 = std::min(cyStart + scY, sy0 + sly);
                size_t copyX0 = std::max(cxStart, sx0);
                size_t copyX1 = std::min(cxStart + scX, sx0 + slx);
                size_t copyW = copyX1 - copyX0;

                for (size_t z = 0; z < slz; z++) {
                    for (size_t y = copyY0; y < copyY1; y++) {
                        size_t srcOff = z * scY * scX + (y - cyStart) * scX + (copyX0 - cxStart);
                        size_t dstOff = z * sly * slx + (y - sy0) * slx + (copyX0 - sx0);
                        std::memcpy(&srcBuf[dstOff], &chunkTmp[srcOff], copyW * sizeof(T));
                    }
                }
            }
        }

        // 2x downsample into dst chunk buffer
        std::vector<T> dstBuf(dstChunkElems, T(0));
        for (size_t zz = 0; zz < dlz; zz++)
            for (size_t yy = 0; yy < dly; yy++)
                for (size_t xx = 0; xx < dlx; xx++) {
                    uint32_t sum = 0; int cnt = 0;
                    for (int d0 = 0; d0 < 2 && 2*zz+d0 < slz; d0++)
                        for (int d1 = 0; d1 < 2 && 2*yy+d1 < sly; d1++)
                            for (int d2 = 0; d2 < 2 && 2*xx+d2 < slx; d2++) {
                                sum += srcBuf[(2*zz+d0)*sly*slx + (2*yy+d1)*slx + (2*xx+d2)];
                                cnt++;
                            }
                    dstBuf[zz*dc[1]*dc[2] + yy*dc[2] + xx] = T((sum + cnt/2) / std::max(1, cnt));
                }

        z5::types::ShapeType dstId = {0, dty, dtx};
        dst->writeChunk(dstId, dstBuf.data());

        size_t d = ++done;
        #pragma omp critical(pp)
        { std::cout << "\r[pyramid L" << level << "] " << d << "/" << totalTiles
                    << " (" << int(100.0*d/totalTiles) << "%)" << std::flush; }
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

