#pragma once

// McVolumeArray — an IChunkedArray backed directly by matter-compressor's
// mc_volume. mc_volume owns the whole remote/cache/prefetch stack (streaming,
// c3d transcode into a local .mca, the resident mc_cache, region single-flight,
// async transcode workers). This adapter is a thin pass-through: no entry table,
// no LRU, no fetchers — tryGetChunk forwards to mc_volume_try_block (present ->
// copy the 16^3 block; absent -> kick async + report MissQueued). It replaces
// the ChunkCache + MatterArchive + ZarrChunkFetcher machinery on the render path.

#include "vc/core/render/IChunkedArray.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct mc_volume;

namespace vc::render {

class McVolumeArray final : public IChunkedArray {
public:
    // Open the remote volume at `url` (an NGFF zarr group root). `cacheDir` holds
    // the local volume.mca; `cacheBytes` is the resident mc_cache budget; `quality`
    // the local re-encode q. Returns nullptr on failure. Static so callers can fall
    // back when the URL/format is unsupported.
    static std::shared_ptr<McVolumeArray> open(const std::string& url,
                                               const std::string& cacheDir,
                                               std::size_t cacheBytes,
                                               float quality);
    ~McVolumeArray() override;

    McVolumeArray(const McVolumeArray&) = delete;
    McVolumeArray& operator=(const McVolumeArray&) = delete;

    int numLevels() const override { return numLevels_; }
    std::array<int, 3> shape(int level) const override;
    std::array<int, 3> chunkShape(int /*level*/) const override { return {16, 16, 16}; }
    ChunkDtype dtype() const override { return ChunkDtype::UInt8; }
    double fillValue() const override { return 0.0; }
    LevelTransform levelTransform(int level) const override;

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override;
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) override;

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) override;
    void removeChunkReadyListener(ChunkReadyCallbackId id) override;

    Stats stats() const override;
    void prefetchShardBlocking(int level, int iz, int iy, int ix) override;

private:
    explicit McVolumeArray(mc_volume* v, int numLevels,
                           std::array<int, 3> shape0);
    void onReady();   // mc_volume worker finished a region

    mc_volume* vol_ = nullptr;
    int numLevels_ = 0;
    std::array<int, 3> shape0_{};

    std::mutex listenerMu_;
    std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> listeners_;
    std::atomic<ChunkReadyCallbackId> nextListenerId_{1};
};

} // namespace vc::render
