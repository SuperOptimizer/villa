#pragma once

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/RemoteAuth.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace vc::render {

struct OpenedChunkedZarr {
    std::vector<int> levelNumbers;
    std::vector<IChunkedArray::LevelTransform> transforms;
    std::vector<std::array<int, 3>> shapes;
    std::vector<std::array<int, 3>> chunkShapes;
    std::vector<std::array<int, 3>> storageChunkShapes;
    std::vector<std::shared_ptr<IChunkFetcher>> fetchers;
    double fillValue = 0.0;
    ChunkDtype dtype = ChunkDtype::UInt8;
};

OpenedChunkedZarr openLocalZarrPyramid(const std::filesystem::path& root);
OpenedChunkedZarr openHttpZarrPyramid(const std::string& url);
OpenedChunkedZarr openHttpZarrPyramid(
    const std::string& url,
    const vc::HttpAuth& auth,
    int baseScaleLevel = 0);

std::unique_ptr<ChunkCache> createChunkCache(
    OpenedChunkedZarr opened,
    std::size_t decodedByteCapacity,
    std::size_t maxConcurrentReads = 16);

// Wrap an opened pyramid with a persistent matter-compressor (.mca) cache: each level's
// source fetcher is decorated to fetch native chunks, re-encode 256^3 regions into the
// shared .mca, and serve decoded 16^3 blocks out of its mc_cache (`cacheBytes` resident
// budget). On success, `opened.fetchers` is replaced, `levelInfoOut` holds the
// 16^3-granular LevelInfo, and the shared archive is returned. Returns null (no change)
// if the volume isn't uint8 or the archive can't be opened. The .mca persists across runs.
std::shared_ptr<MatterArchive> applyMatterCache(OpenedChunkedZarr& opened,
                                                const std::filesystem::path& mcaPath,
                                                float quality,
                                                std::size_t cacheBytes,
                                                std::vector<ChunkCache::LevelInfo>& levelInfoOut);

} // namespace vc::render
