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
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct mc_volume;

// matter-compressor colormap helpers (defined in matter_compressor.h's extern "C"
// block). Re-declared here so GUI TUs can map mc_render's u8 output -> ARGB without
// pulling the full merged mc header onto their include path.
extern "C" {
int  mc_colormap_id(const char* name);
void mc_colormap_lut(uint32_t lut[256], float win_low, float win_high, int cmap_id);
}

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

    // Open an ALREADY-BUILT .mca at `url` (remote s3://.../https://... or a local
    // file path) and stream it into a local .mca in `cacheDir` on demand: each
    // fetched 256^3 chunk's compressed blob is appended VERBATIM (no decode, no
    // re-encode). Same render/cache machinery as open(); only the chunks the view
    // touches are pulled. Returns nullptr on failure.
    static std::shared_ptr<McVolumeArray> openStreaming(const std::string& url,
                                                        const std::string& cacheDir,
                                                        std::size_t cacheBytes);

    // Probe a built .mca's header only (no volume, no threads, no local archive):
    // LOD0 shape (z,y,x) + lod count. Returns false on failure.
    static bool probeStreaming(const std::string& url,
                               std::array<int, 3>& shapeZYX, int& numLevels);
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
    void setDecodedByteCapacity(std::size_t bytes) override;
    void freeze() override;   // mc_volume_freeze: immutable, lock-free reads
    void thaw() override;     // mc_volume_thaw: write phase, advance pin epoch

    // Extra parameters for mc's SHADED/PERCENTILE composite modes. Defaults match
    // mc's zero-field defaults (headlight relief). Pass null for slice/min/mean/
    // max/alpha. light is direction TOWARD the light in (z,y,x); (0,0,0)=headlight.
    struct ShadeParams {
        float lightZ = 0.f, lightY = 0.f, lightX = 0.f;
        float ambient = 0.f, diffuse = 0.f, specular = 0.f, shininess = 0.f;
        float absorption = 0.f, shadow = 0.f, sss = 0.f, curvature = 0.f;
        float percentile = 0.f;   // MC_COMP_PERCENTILE rank in (0,1] (0 -> 0.9)
    };

    // Render a W*H image directly via matter-compressor's mc_render, bypassing
    // any per-chunk C++ sampler. `ptsXYZ` is W*H*3 floats in VC's (x,y,z) order
    // (e.g. from Surface::gen()); `normalsXYZ` is W*H*3 (x,y,z) unit normals or
    // null (required when compositing). `comp` selects the reduction along the
    // normal: 0=none(slice) 1=min 2=mean 3=max 4=alpha 5=stddev 6=shaded
    // 7=percentile 8=depth. `t0..t1` step `dt` is the composite slab in voxels.
    // `voxPerPixel` picks the LOD. `shade` carries SHADED/PERCENTILE knobs (null
    // ok). `out` is W*H bytes.
    void render(const float* ptsXYZ, const float* normalsXYZ, int w, int h,
                int comp, float t0, float t1, float dt,
                float alphaMin, float alphaOpacity,
                float voxPerPixel, std::uint8_t* out,
                const ShadeParams* shade = nullptr);

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

    // download-rate estimate. net_bytes arrives in bursts (one s3_get_batch of
    // ~48 chunks every few seconds), so a poll-to-poll delta flickers to 0
    // between bursts. Instead: average bytes over a sliding ~2s window, and hold
    // the last nonzero rate for a few seconds of idle before declaring 0, so the
    // status bar shows a steady rate while downloads are ongoing. rateMu_ guards
    // (stats() is const + polled off the worker).
    mutable std::mutex rateMu_;
    mutable std::uint64_t windowStartBytes_ = 0;
    mutable std::chrono::steady_clock::time_point windowStartTime_{};
    mutable std::chrono::steady_clock::time_point lastProgressTime_{};
    mutable double lastRateBytesPerSec_ = 0.0;
};

} // namespace vc::render
