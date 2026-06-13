#include "vc/core/render/McVolumeArray.hpp"

#include "matter_compressor.h"   // single merged TU: volume + codec + sample/render

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vc::render {

namespace {
constexpr int kBlk = 16;
constexpr int kBlockBytes = kBlk * kBlk * kBlk;   // 4096

// NaN test that survives -ffast-math (std::isnan and x!=x are deleted there).
inline bool nanBits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x7FFFFFFFu) > 0x7F800000u;
}
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

bool McVolumeArray::probeStreaming(const std::string& url,
                                   std::array<int, 3>& shapeZYX, int& numLevels)
{
    int nx = 0, ny = 0, nz = 0, nl = 0;
    if (mc_mca_probe(url.c_str(), &nx, &ny, &nz, &nl, nullptr) != 0)
        return false;
    shapeZYX = {nz, ny, nx};
    numLevels = nl;
    return true;
}

std::shared_ptr<McVolumeArray> McVolumeArray::openStreaming(const std::string& url,
                                                            const std::string& cacheDir,
                                                            std::size_t cacheBytes)
{
    mc_volume* v = mc_volume_open_streaming(url.c_str(), cacheDir.c_str(), cacheBytes);
    if (!v)
        return nullptr;
    const int n = mc_volume_nlods(v);
    int nz = 0, ny = 0, nx = 0;
    mc_volume_shape(v, 0, &nz, &ny, &nx);
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
    // Predictive prefetch: request each region's DOWNLOAD if absent -- cheap,
    // non-blocking, deduped (LIFO stack). NOT a decode (the old try_block path
    // allocated a 4KB scratch + decoded synchronously per key). ChunkKey carries
    // 16^3-block coords; the 256^3 region is block/16.
    for (const auto& k : keys)
        mc_volume_request_region(vol_, k.level, k.iz / kBlk, k.iy / kBlk, k.ix / kBlk);
}

void McVolumeArray::setDecodedByteCapacity(std::size_t bytes)
{
    mc_volume_set_cache_bytes(vol_, bytes);
}

void McVolumeArray::freeze() { mc_volume_freeze(vol_); }
void McVolumeArray::thaw()   { mc_volume_thaw(vol_); }

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

std::uint64_t McVolumeArray::dataGeneration() const
{
    return mc_volume_render_gen(vol_);
}

std::uint64_t McVolumeArray::dataGenerationFor(const std::vector<ChunkKey>& keys) const
{
    if (keys.empty())
        return mc_volume_render_gen(vol_);   // no prediction -> volume-global
    // Max change gen over the predicted 256^3 regions (keys carry 16^3-block
    // coords; region = block/16). One hash + atomic load per region.
    std::uint64_t g = 0;
    for (const auto& k : keys) {
        const std::uint64_t rg = mc_volume_region_gen(
            vol_, k.level, k.iz / kBlk, k.iy / kBlk, k.ix / kBlk);
        if (rg > g)
            g = rg;
    }
    return g;
}

std::uint64_t McVolumeArray::dataGenerationForBox(int level, int rz0, int rz1,
                                                  int ry0, int ry1,
                                                  int rx0, int rx1) const
{
    // The box at `level` plus its covering ancestors at every coarser level
    // (region coords halve per level). Boxes are tiny (a 64px tile spans <=
    // a few regions), so this is a few dozen hash+load probes.
    std::uint64_t g = 0;
    for (int l = level; l < numLevels_; ++l) {
        const int sh = l - level;
        for (int z = rz0 >> sh; z <= (rz1 >> sh); ++z)
            for (int y = ry0 >> sh; y <= (ry1 >> sh); ++y)
                for (int x = rx0 >> sh; x <= (rx1 >> sh); ++x) {
                    const std::uint64_t rg = mc_volume_region_gen(vol_, l, z, y, x);
                    if (rg > g)
                        g = rg;
                }
    }
    return g;
}

int McVolumeArray::pickLevel(float voxPerPixel) const
{
    mc_sample_lods lods = mc_volume_sample_lods(vol_, 0);
    return mc_render_pick_lod(&lods, voxPerPixel > 0.f ? voxPerPixel : 1.f);
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
    out.workPending = static_cast<std::size_t>(s.work_pending);
    out.downloadQueued = static_cast<std::size_t>(s.regions_queued);
    out.downloading = static_cast<std::size_t>(s.regions_downloading);
    out.decodeQueued = static_cast<std::size_t>(s.regions_decode_queued);
    out.encoding = static_cast<std::size_t>(s.regions_encoding);
    out.decodeStagingBytes = static_cast<std::size_t>(s.staging_bytes);

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
                           float voxPerPixel, std::uint8_t* out,
                           const ShadeParams* shade)
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
    const bool composite = (comp != 0 /*MC_COMP_NONE*/);

    // Slice path uses LOD fallback: pass NATIVE L0 coords and let mc_render_points_
    // par_lod walk -- it samples the picked level L where that level's block is
    // RAM-resident, else the FINEST resident coarser level (cheap residency probe,
    // never decodes the fine block on the render thread). A not-yet-cached block is
    // recorded as a miss (THAW fills it) and shown coarse meanwhile -> the image
    // sharpens over a few frames instead of going black or stalling on decode.
    // mc wants (z,y,x); invalid (<0/NaN) -> -1 so mc emits 0. (The composite path
    // still pre-remaps to single level L below.)
    static const bool prof = getenv("MCV_PROF") != nullptr;
    const auto tA = std::chrono::steady_clock::now();
    thread_local std::vector<float> pts, nrm;
    pts.resize(n * 3);
    auto remap = [&](float c) { return (c + 0.5f) / s - 0.5f; };

    mc_render_params p{};
    p.filter = MC_FILTER_TRILINEAR;
    p.comp = static_cast<mc_comp>(comp);

    if (!composite) {
        // L0 coords, no remap (the lod sampler downscales per level internally).
        for (std::size_t i = 0; i < n; ++i) {
            const float x = ptsXYZ[i * 3 + 0], y = ptsXYZ[i * 3 + 1], z = ptsXYZ[i * 3 + 2];
            const bool bad = nanBits(x) || nanBits(y) || nanBits(z) ||
                             x < 0.f || y < 0.f || z < 0.f;
            pts[i * 3 + 0] = bad ? -1.f : z;   // z
            pts[i * 3 + 1] = bad ? -1.f : y;   // y
            pts[i * 3 + 2] = bad ? -1.f : x;   // x
        }
        const auto tB = std::chrono::steady_clock::now();
        mc_render_points_par_lod(&lods, L, pts.data(), w, h, &p, out, 0);
        if (prof) {
            const auto tC = std::chrono::steady_clock::now();
            using ms = std::chrono::duration<double, std::milli>;
            fprintf(stderr, "[mcv-prof] %dx%d L=%d remap=%.1fms sample=%.1fms (lod)\n",
                    w, h, L, ms(tB - tA).count(), ms(tC - tB).count());
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const float x = ptsXYZ[i * 3 + 0], y = ptsXYZ[i * 3 + 1], z = ptsXYZ[i * 3 + 2];
            const bool bad = nanBits(x) || nanBits(y) || nanBits(z) ||
                             x < 0.f || y < 0.f || z < 0.f;
            pts[i * 3 + 0] = bad ? -1.f : remap(z);   // z
            pts[i * 3 + 1] = bad ? -1.f : remap(y);   // y
            pts[i * 3 + 2] = bad ? -1.f : remap(x);   // x
        }
        nrm.resize(n * 3);
        for (std::size_t i = 0; i < n; ++i) {     // normals are directions: swap, no remap
            nrm[i * 3 + 0] = normalsXYZ ? normalsXYZ[i * 3 + 2] : 0.f;   // z
            nrm[i * 3 + 1] = normalsXYZ ? normalsXYZ[i * 3 + 1] : 0.f;   // y
            nrm[i * 3 + 2] = normalsXYZ ? normalsXYZ[i * 3 + 0] : 0.f;   // x
        }
        // composite range is in LOD-0 voxels; scale to the sampled level's pitch.
        p.t0 = t0 / s; p.t1 = t1 / s; p.dt = (dt > 0.f ? dt : 1.f) / s;
        p.alpha_min = alphaMin;
        p.alpha_opacity = alphaOpacity > 0.f ? alphaOpacity : 1.f;
        // SHADED / PERCENTILE knobs. mc treats all-zero fields as its defaults, so
        // a null/zero ShadeParams renders a sane headlight relief.
        if (shade) {
            p.light[0] = shade->lightZ; p.light[1] = shade->lightY; p.light[2] = shade->lightX;
            p.ambient = shade->ambient; p.diffuse = shade->diffuse;
            p.specular = shade->specular; p.shininess = shade->shininess;
            p.absorption = shade->absorption; p.shadow = shade->shadow;
            p.sss = shade->sss; p.curvature = shade->curvature;
            p.percentile = shade->percentile;
            p.transmission = shade->transmission;
            p.ink_lock = shade->inkLock;
            p.light_surface_rel = 1;   // raking = degrees above the LOCAL sheet
        }
        mc_render_points_par(&lods.lods[L], pts.data(), nrm.data(), w, h, &p, out, 0);
    }

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
        fprintf(stderr, "[mcv-render] %dx%d L=%d nz=%zu vpp=%.2f | "
                "center_xyz(%.0f,%.0f,%.0f) d/col_xyz(%.3f,%.3f,%.3f) "
                "d/row_xyz(%.3f,%.3f,%.3f) comp=%d\n",
                w, h, L, nz, voxPerPixel,
                P(cy,cx,0), P(cy,cx,1), P(cy,cx,2),
                dcol[0],dcol[1],dcol[2], drow[0],drow[1],drow[2], composite?1:0);
    }
}

} // namespace vc::render
