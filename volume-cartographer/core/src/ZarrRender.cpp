#include "vc/core/util/ZarrRender.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ============================================================
// writeZarrBand
// ============================================================

template <typename T>
void writeZarrBand(vc::Zarr* dsOut, const std::vector<cv::Mat>& slices,
                   uint32_t bandIdx, const std::vector<size_t>& chunks0,
                   size_t tilesXSrc, size_t tilesYSrc,
                   int rotQuad, int flipAxis)
{
    int outH = slices[0].rows, outW = slices[0].cols;
    size_t numZ = slices.size();
    size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];
    size_t chunkElements = chunkZ * chunkY * chunkX;
    size_t nbytes = chunkElements * dsOut->itemsize();

    for (size_t tx = 0; tx < tilesXSrc; tx++) {
        size_t x0s = tx * chunkX;
        size_t dxc = std::min(chunkX, size_t(outW) - x0s);

        int dstTx = int(tx), dstTy = int(bandIdx), dTX, dTY;
        if (rotQuad >= 0 || flipAxis >= 0)
            mapTileIndex(int(tx), int(bandIdx), int(tilesXSrc), int(tilesYSrc),
                         std::max(rotQuad, 0), flipAxis, dstTx, dstTy, dTX, dTY);

        std::vector<T> chunkBuf(chunkElements, T(0));
        size_t dy_actual = std::min(chunkY, size_t(outH));
        size_t dx_actual = std::min(chunkX, dxc);
        for (size_t zi = 0; zi < numZ; zi++) {
            size_t sliceOff = zi * chunkY * chunkX;
            for (size_t yy = 0; yy < dy_actual; yy++) {
                const T* row = slices[zi].ptr<T>(int(yy));
                std::memcpy(&chunkBuf[sliceOff + yy * chunkX], &row[x0s], dx_actual * sizeof(T));
            }
        }
        dsOut->writeChunk(0, size_t(dstTy), size_t(dstTx), chunkBuf.data(), nbytes);
    }
}

template void writeZarrBand<uint8_t>(vc::Zarr*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);
template void writeZarrBand<uint16_t>(vc::Zarr*, const std::vector<cv::Mat>&,
    uint32_t, const std::vector<size_t>&, size_t, size_t, int, int);

// ============================================================
// writeZarrAttrs
// ============================================================

void writeZarrAttrs(const std::filesystem::path& outFile,
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
        const double sz = 1.0;
        ms["datasets"].push_back({
            {"path", std::to_string(l)},
            {"coordinateTransformations", json::array({
                json{{"type","scale"},{"scale",json::array({sz,s,s})}},
                json{{"type","translation"},{"translation",json::array({0.0,0.0,0.0})}}
            })}
        });
    }
    ms["metadata"] = json{{"downsampling_method","mean"}};
    attrs["multiscales"] = json::array({ms});
    vc::Zarr::writeAttributes(outFile, attrs);
}
