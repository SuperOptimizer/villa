#include "vc/core/render/McVolumeArray.hpp"

#include "matter_compressor.h"   // single merged TU: volume + codec + sample/render

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace vc::render {

namespace {
constexpr int kBlk = 16;
constexpr int kBlockBytes = kBlk * kBlk * kBlk;   // 4096
}

McVolumeArray::McVolumeArray(mc_volume* v, int numLevels, std::array<int, 3> shape0)
    : vol_(v), numLevels_(numLevels), shape0_(shape0)
{
    mc_volume_set_ready_cb(vol_, [](void* ud) {
        static_cast<McVolumeArray*>(ud)->onReady();
    }, this);
}

McVolumeArray::~McVolumeArray()
{
    if (vol_) {
        mc_volume_set_ready_cb(vol_, nullptr, nullptr);
        mc_volume_free(vol_);
    }
}

std::shared_ptr<McVolumeArray> McVolumeArray::open(const std::string& url,
                                                   const std::string& cacheDir,
                                                   std::size_t cacheBytes,
                                                   float quality)
{
    mc_volume* v = mc_volume_open(url.c_str(), cacheDir.c_str(), cacheBytes, quality);
    if (!v)
        return nullptr;
    const int n = mc_volume_nlods(v);
    int nz = 0, ny = 0, nx = 0;
    mc_volume_shape(v, 0, &nz, &ny, &nx);
    // private ctor -> wrap in shared_ptr by hand.
    return std::shared_ptr<McVolumeArray>(new McVolumeArray(v, n, {nz, ny, nx}));
}

std::array<int, 3> McVolumeArray::shape(int level) const
{
    int nz = 0, ny = 0, nx = 0;
    if (level >= 0 && level < numLevels_)
        mc_volume_shape(vol_, level, &nz, &ny, &nx);
    return {nz, ny, nx};
}

IChunkedArray::LevelTransform McVolumeArray::levelTransform(int level) const
{
    // NGFF power-of-two pyramid: each level halves resolution. Derive the exact
    // scale from the shape ratio so non-exact halving (odd edges) stays correct.
    LevelTransform t;
    if (level <= 0 || level >= numLevels_)
        return t;
    const auto s0 = shape0_;
    int nz = 0, ny = 0, nx = 0;
    mc_volume_shape(vol_, level, &nz, &ny, &nx);
    const std::array<int, 3> sl{nz, ny, nx};
    for (int d = 0; d < 3; ++d)
        t.scaleFromLevel0[d] = sl[d] > 0 ? static_cast<double>(s0[d]) / sl[d]
                                         : static_cast<double>(1u << level);
    return t;
}

ChunkResult McVolumeArray::tryGetChunk(int level, int iz, int iy, int ix)
{
    if (level < 0 || level >= numLevels_)
        return {ChunkStatus::Missing, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, {}, {}};
    auto bytes = std::make_shared<std::vector<std::byte>>(kBlockBytes);
    const int got = mc_volume_try_block(vol_, level, iz, iy, ix,
                                        reinterpret_cast<std::uint8_t*>(bytes->data()));
    if (got == 1)
        return {ChunkStatus::Data, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, std::move(bytes), {}};
    // absent: mc_volume kicked an async transcode; renderer falls to a coarser LOD.
    return {ChunkStatus::MissQueued, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, {}, {}};
}

ChunkResult McVolumeArray::getChunkBlocking(int level, int iz, int iy, int ix)
{
    if (level < 0 || level >= numLevels_)
        return {ChunkStatus::Missing, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, {}, {}};
    auto bytes = std::make_shared<std::vector<std::byte>>(kBlockBytes);
    const int got = mc_volume_get_block(vol_, level, iz, iy, ix,
                                        reinterpret_cast<std::uint8_t*>(bytes->data()));
    if (got == 1)
        return {ChunkStatus::Data, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, std::move(bytes), {}};
    if (got == 0)   // air -> all-fill (zeros)
        return {ChunkStatus::AllFill, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, {}, {}};
    return {ChunkStatus::Error, ChunkDtype::UInt8, {kBlk, kBlk, kBlk}, {}, "mc_volume decode error"};
}

void McVolumeArray::prefetchChunks(const std::vector<ChunkKey>& keys, bool /*wait*/, int /*priorityOffset*/)
{
    // mc_volume's async workers already dedup; kick each region once. Snap to the
    // 256^3 region corner (16 blocks) so 4096 block-keys -> one transcode request.
    for (const auto& k : keys) {
        std::uint8_t scratch[kBlockBytes];
        mc_volume_try_block(vol_, k.level, k.iz, k.iy, k.ix, scratch);
    }
}

void McVolumeArray::setDecodedByteCapacity(std::size_t bytes)
{
    mc_volume_set_cache_bytes(vol_, bytes);
}

void McVolumeArray::prefetchShardBlocking(int level, int iz, int iy, int ix)
{
    // iz/iy/ix are 16^3-block coords; mc_volume wants the source inner-chunk coords
    // (256^3) = block/16. Prefetch the whole source shard enclosing it.
    mc_volume_prefetch_shard(vol_, level, iz / kBlk, iy / kBlk, ix / kBlk);
}

McVolumeArray::ChunkReadyCallbackId McVolumeArray::addChunkReadyListener(ChunkReadyCallback cb)
{
    const ChunkReadyCallbackId id = nextListenerId_.fetch_add(1);
    std::lock_guard lock(listenerMu_);
    listeners_.emplace(id, std::move(cb));
    return id;
}

void McVolumeArray::removeChunkReadyListener(ChunkReadyCallbackId id)
{
    std::lock_guard lock(listenerMu_);
    listeners_.erase(id);
}

void McVolumeArray::onReady()
{
    std::lock_guard lock(listenerMu_);
    for (const auto& [id, cb] : listeners_)
        cb();
}

IChunkedArray::Stats McVolumeArray::stats() const
{
    mc_volume_stats s{};
    mc_volume_get_stats(vol_, &s);
    Stats out;
    // RAM cache: decoded 16^3 blocks resident vs the budget (4096 B each).
    constexpr std::size_t kBlockBytesU = kBlockBytes;
    out.decodedBytes = static_cast<std::size_t>(s.cache_used_blocks) * kBlockBytesU;
    out.decodedByteCapacity = static_cast<std::size_t>(s.cache_cap_blocks) * kBlockBytesU;
    out.persistentCacheBytes = static_cast<std::size_t>(s.disk_bytes);
    out.remoteFetchesInFlight = static_cast<std::size_t>(s.regions_inflight);

    // download rate: bytes over a sliding ~2s window (smooths the bursty
    // s3_get_batch arrivals), held through short idle gaps so the readout stays
    // steady while downloads continue and only drops to 0 after ~3s of no bytes.
    {
        constexpr double kWindowSec = 2.0;
        constexpr double kIdleHoldSec = 3.0;
        std::lock_guard<std::mutex> lock(rateMu_);
        const auto now = std::chrono::steady_clock::now();
        if (windowStartTime_.time_since_epoch().count() == 0) {       // first poll
            windowStartBytes_ = s.net_bytes;
            windowStartTime_ = now;
            lastProgressTime_ = now;
        } else {
            if (s.net_bytes > windowStartBytes_)
                lastProgressTime_ = now;                               // bytes moved
            const double winDt =
                std::chrono::duration<double>(now - windowStartTime_).count();
            if (winDt >= kWindowSec) {
                const std::uint64_t db = s.net_bytes >= windowStartBytes_
                                         ? s.net_bytes - windowStartBytes_ : 0;
                lastRateBytesPerSec_ = db / winDt;                     // window avg
                windowStartBytes_ = s.net_bytes;                       // slide window
                windowStartTime_ = now;
            }
            // declare idle only after a sustained gap with no new bytes.
            const double idleDt =
                std::chrono::duration<double>(now - lastProgressTime_).count();
            if (idleDt >= kIdleHoldSec)
                lastRateBytesPerSec_ = 0.0;
        }
        out.remoteDownloadBytesPerSecond = lastRateBytesPerSec_;
    }
    return out;
}

void McVolumeArray::render(const float* ptsXYZ, const float* normalsXYZ,
                           int w, int h, int comp, float t0, float t1, float dt,
                           float alphaMin, float alphaOpacity,
                           float voxPerPixel, std::uint8_t* out)
{
    const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);

    // mc_volume sample sources for every level (non-blocking: absent regions
    // sample as 0 and kick an async transcode — the interactive render path).
    mc_sample_lods lods = mc_volume_sample_lods(vol_, 0);
    const int L = mc_render_pick_lod(&lods, voxPerPixel > 0.f ? voxPerPixel : 1.f);
    // Power-of-two pyramid: each level is 2^L smaller. The remote level SHAPE is
    // padded up to whole 4096-chunks (e.g. z 77824 -> 8192 at L4 instead of the
    // true 4864) so the shape ratio is NOT the scale; the downsample factor is
    // exactly 2^L. c_L = (c_0 + 0.5)/2^L - 0.5 (half-voxel-center correct).
    const float s = static_cast<float>(1 << L);

    // VC points are (x,y,z); mc wants (z,y,x). Invalid (<0 / NaN) points pass
    // through unchanged so mc_render emits 0 there. Reuse thread-local scratch:
    // render() runs per-frame on a render worker, so a fresh 13MB vector every
    // call churns the allocator + first-touches pages (~3% page-fault traffic).
    // The buffers grow to the largest frame and then never reallocate.
    thread_local std::vector<float> pts, nrm;
    pts.resize(n * 3);
    auto remap = [&](float c) { return (c + 0.5f) / s - 0.5f; };
    for (std::size_t i = 0; i < n; ++i) {
        const float x = ptsXYZ[i * 3 + 0], y = ptsXYZ[i * 3 + 1], z = ptsXYZ[i * 3 + 2];
        const bool bad = !(x >= 0.f) || !(y >= 0.f) || !(z >= 0.f) ||
                         std::isnan(x) || std::isnan(y) || std::isnan(z);
        pts[i * 3 + 0] = bad ? -1.f : remap(z);   // z
        pts[i * 3 + 1] = bad ? -1.f : remap(y);   // y
        pts[i * 3 + 2] = bad ? -1.f : remap(x);   // x
    }
    const float* nrmPtr = nullptr;
    if (normalsXYZ && comp != 0 /*MC_COMP_NONE*/) {
        nrm.resize(n * 3);
        for (std::size_t i = 0; i < n; ++i) {     // normals are directions: just swap, no remap
            nrm[i * 3 + 0] = normalsXYZ[i * 3 + 2];   // z
            nrm[i * 3 + 1] = normalsXYZ[i * 3 + 1];   // y
            nrm[i * 3 + 2] = normalsXYZ[i * 3 + 0];   // x
        }
        nrmPtr = nrm.data();
    }

    mc_render_params p{};
    p.filter = MC_FILTER_TRILINEAR;
    p.comp = static_cast<mc_comp>(comp);
    // composite range is in LOD-0 voxels; scale to the sampled level's pitch.
    p.t0 = t0 / s; p.t1 = t1 / s; p.dt = (dt > 0.f ? dt : 1.f) / s;
    p.alpha_min = alphaMin;
    p.alpha_opacity = alphaOpacity > 0.f ? alphaOpacity : 1.f;

    mc_render_points_par(&lods.lods[L], pts.data(), nrmPtr, w, h, &p, out, 0);

    static const bool dbg = getenv("MCV_LOG") != nullptr;
    if (dbg && w > 4 && h > 4) {
        // measure how much the INPUT coord moves per-pixel along a center row
        // vs down a center column. Streaks => one of these deltas ~ 0.
        const int cy = h / 2, cx = w / 2;
        auto P = [&](int yy, int xx, int k){ return ptsXYZ[(std::size_t(yy)*w+xx)*3+k]; };
        float drow[3], dcol[3];
        for (int k = 0; k < 3; ++k) {
            drow[k] = P(cy, cx+1, k) - P(cy, cx, k);
            dcol[k] = P(cy+1, cx, k) - P(cy, cx, k);
        }
        std::size_t nz = 0; for (std::size_t i = 0; i < n; ++i) if (out[i]) ++nz;
        const bool quad = (nrmPtr != nullptr) || (P(0,0,2) != P(cy,cx,2) && false);
        fprintf(stderr, "[mcv-render] %dx%d L=%d nz=%zu vpp=%.2f | "
                "center_xyz(%.0f,%.0f,%.0f) d/col_xyz(%.3f,%.3f,%.3f) "
                "d/row_xyz(%.3f,%.3f,%.3f) nrm=%d\n",
                w, h, L, nz, voxPerPixel,
                P(cy,cx,0), P(cy,cx,1), P(cy,cx,2),
                dcol[0],dcol[1],dcol[2], drow[0],drow[1],drow[2], nrmPtr?1:0);
        (void)quad;
    }
}

} // namespace vc::render
