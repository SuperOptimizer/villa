#include "vc/core/render/McVolumeArray.hpp"

#include "mc_volume.h"

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
    // mc_cache residency: mc_volume reports hits/misses but not resident bytes
    // directly; surface disk + in-flight, leave RAM gauge to mc_cache budget.
    out.persistentCacheBytes = static_cast<std::size_t>(s.disk_bytes);
    out.remoteFetchesInFlight = static_cast<std::size_t>(s.regions_inflight);
    return out;
}

} // namespace vc::render
