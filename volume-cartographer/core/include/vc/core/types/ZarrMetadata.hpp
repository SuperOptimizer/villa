#pragma once

// Metadata-only multiscale-zarr discovery. Volume::zarrOpen()/NewFromUrl() need
// per-level shape/chunkShape/storageChunkShape/dtype/fillValue/transform to set up
// the volume; the actual reads go through McVolumeArray on the render path. This
// reads exactly that metadata off utils::zarr, building no fetcher.

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/RemoteAuth.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace vc::render {

struct ZarrPyramidMeta {
    std::vector<int> levelNumbers;
    std::vector<IChunkedArray::LevelTransform> transforms;
    std::vector<std::array<int, 3>> shapes;
    std::vector<std::array<int, 3>> chunkShapes;
    std::vector<std::array<int, 3>> storageChunkShapes;
    double fillValue = 0.0;
    ChunkDtype dtype = ChunkDtype::UInt8;
};

// Probe numeric level subdirs ("0","1",...) under `root`, falling back to a
// single array at `root`. Throws if no zarr array is found.
ZarrPyramidMeta openLocalZarrMeta(const std::filesystem::path& root);

// Discover remote multiscale levels via .zattrs (multiscales/datasets) or by
// probing "0","1",... keys. Throws on transport/decode failure.
ZarrPyramidMeta openHttpZarrMeta(const std::string& url, const vc::HttpAuth& auth);

} // namespace vc::render
