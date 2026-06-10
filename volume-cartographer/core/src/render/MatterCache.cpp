#include "vc/core/render/MatterCache.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <chrono>
#include <future>

#include <filesystem>
#include <stdexcept>

#include "vc/core/util/Logging.hpp"

extern "C" {
#include "matter_compressor.h"
}

namespace vc::render {

struct MatterArchive::Impl {
    mc_archive* a = nullptr;
    mc_cache* cache = nullptr;
};

MatterArchive::MatterArchive(std::string path, std::array<int, 3> shape0, float quality,
                             std::size_t cacheBytes)
    : impl_(std::make_unique<Impl>())
    , path_(std::move(path))
    , shape0_(shape0)
    , quality_(quality)
{
    // shape0 is (z,y,x); mc takes (nx,ny,nz) and pads each axis to 256 internally.
    auto open = [&] {
        return mc_archive_open_dims(path_.c_str(), shape0_[2], shape0_[1], shape0_[0],
                                    quality);
    };
    impl_->a = open();
    if (!impl_->a && std::filesystem::exists(path_)) {
        // stale archive (older format version / different dims): it's a rebuildable
        // cache, so delete and recreate.
        Logger()->warn("MatterArchive: {} is stale/incompatible; recreating", path_);
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        impl_->a = open();
    }
    if (!impl_->a)
        throw std::runtime_error("MatterArchive: mc_archive_open failed for " + path_);
    impl_->cache = mc_cache_new_archive(cacheBytes, impl_->a);
    if (!impl_->cache) {
        mc_archive_close(impl_->a);
        throw std::runtime_error("MatterArchive: mc_cache_new_archive failed for " + path_);
    }
}

MatterArchive::~MatterArchive()
{
    if (impl_ && impl_->cache)
        mc_cache_free(impl_->cache);
    if (impl_ && impl_->a)
        mc_archive_close(impl_->a);
}

bool MatterArchive::appendChunkRaw(int lod, int cz, int cy, int cx, const std::uint8_t* vox256)
{
    if (!vox256) return false;
    return mc_archive_append_chunk_raw(impl_->a, lod, cz, cy, cx, vox256) == 0;
}

bool MatterArchive::hasChunk(int lod, int cz, int cy, int cx) const
{
    return mc_archive_chunk_coverage(impl_->a, lod, cz, cy, cx) == MC_PRESENT;
}

void MatterArchive::decodeBlock(int lod, int cz, int cy, int cx, int bz, int by, int bx,
                                std::uint8_t* dst4096) const
{
    // mc_cache keys on GLOBAL block coords.
    mc_cache_get_copy(impl_->cache, lod,
                      cz * kBlocksPerAxis + bz,
                      cy * kBlocksPerAxis + by,
                      cx * kBlocksPerAxis + bx,
                      dst4096);
}

MatterArchive::CacheStats MatterArchive::cacheStats() const
{
    mc_cache_stats s{};
    mc_cache_get_stats(impl_->cache, &s);
    CacheStats out;
    out.hits = s.hits;
    out.misses = s.misses;
    out.evictions = s.evictions;
    out.usedBytes = s.used * std::size_t{4096};
    out.capacityBytes = s.slots * std::size_t{4096};
    return out;
}




MatterCacheFetcher::MatterCacheFetcher(std::shared_ptr<IChunkFetcher> source,
                                       std::shared_ptr<MatterArchive> archive,
                                       int lod,
                                       int sourceChunkEdge,
                                       std::array<int, 3> levelShape)
    : source_(std::move(source))
    , archive_(std::move(archive))
    , lod_(lod)
    , srcEdge_(sourceChunkEdge)
    , levelShape_(levelShape)
{
}

static std::uint64_t regionKey(int cz, int cy, int cx)
{
    return (static_cast<std::uint64_t>(cz & 0x1FFFFF) << 42) |
           (static_cast<std::uint64_t>(cy & 0x1FFFFF) << 21) |
           (static_cast<std::uint64_t>(cx & 0x1FFFFF));
}

bool MatterCacheFetcher::ensureRegion(int regCz, int regCy, int regCx,
                                      std::size_t& downloadedBytes)
{
    downloadedBytes = 0;
    const std::uint64_t rk = regionKey(regCz, regCy, regCx);

    // Single-flight claim: wait out any in-flight assembly of this region; if it's
    // already resolved, return immediately; otherwise claim it (InFlight) and assemble.
    {
        std::unique_lock<std::mutex> lk(regMu_);
        for (;;) {
            auto it = regions_.find(rk);
            if (it == regions_.end())
                break;   // not claimed -> we'll claim it below
            if (it->second == RegionState::InFlight) {
                regCv_.wait(lk);   // another thread is assembling it; wait for publish
                continue;
            }
            return it->second == RegionState::Present;
        }
        // Persisted from a previous run? (cheap index probe under the lock is fine.)
        if (archive_->hasChunk(lod_, regCz, regCy, regCx)) {
            regions_[rk] = RegionState::Present;
            return true;
        }
        regions_[rk] = RegionState::InFlight;   // claim
    }

    // ---- we own this region: assemble + encode OUTSIDE the lock ----
    const auto t0 = std::chrono::steady_clock::now();
    // Assemble the 256^3 region from the source's native chunks (srcEdge_ each).
    const int subPerAxis = kMca / srcEdge_;   // 1 (c3d 256) or 2 (zarr 128)
    std::vector<std::uint8_t> region(static_cast<std::size_t>(kMca) * kMca * kMca, 0);

    // region voxel origin
    const int vz0 = regCz * kMca, vy0 = regCy * kMca, vx0 = regCx * kMca;
    std::atomic<bool> anyData{false};
    std::atomic<std::size_t> fetchedBytes{0};

    // fetch one source chunk + copy it into its (sz,sy,sx) sub-offset of the region.
    // Sub-blocks are disjoint byte ranges, so these run concurrently.
    auto fetchSub = [&](int sz, int sy, int sx) {
        // source native-chunk index covering this sub-block.
        const int scz = (vz0 + sz * srcEdge_) / srcEdge_;
        const int scy = (vy0 + sy * srcEdge_) / srcEdge_;
        const int scx = (vx0 + sx * srcEdge_) / srcEdge_;
        // skip sub-chunks entirely past the volume edge.
        if (scz * srcEdge_ >= levelShape_[0] ||
            scy * srcEdge_ >= levelShape_[1] ||
            scx * srcEdge_ >= levelShape_[2])
            return;

        ChunkKey sk{lod_, scz, scy, scx};
        ChunkFetchResult sub;
        try {
            sub = source_->fetch(sk);
        } catch (const std::exception& e) {
            Logger()->warn("MatterCacheFetcher: source fetch failed l{} ({},{},{}): {}",
                           lod_, scz, scy, scx, e.what());
            return;
        }
        if (sub.status != ChunkFetchStatus::Found || sub.bytes.empty())
            return;   // missing/air sub-chunk -> stays zero in the region
        fetchedBytes += sub.bytes.size();

        // copy the source chunk (srcEdge_^3, z,y,x raster, u8) into the region at
        // its (sz,sy,sx) sub-offset. Clamp to the actual fetched size at edges.
        const auto* src = reinterpret_cast<const std::uint8_t*>(sub.bytes.data());
        const std::size_t gotVox = sub.bytes.size();
        const int oz = sz * srcEdge_, oy = sy * srcEdge_, ox = sx * srcEdge_;
        for (int z = 0; z < srcEdge_; ++z) {
            for (int y = 0; y < srcEdge_; ++y) {
                const std::size_t srcRow =
                    (static_cast<std::size_t>(z) * srcEdge_ + y) * srcEdge_;
                if (srcRow + srcEdge_ > gotVox) break;   // partial-edge chunk
                const std::size_t dstRow =
                    ((static_cast<std::size_t>(oz + z) * kMca + (oy + y)) * kMca + ox);
                std::memcpy(region.data() + dstRow, src + srcRow, srcEdge_);
            }
        }
        anyData = true;
    };

    if (subPerAxis == 1) {
        fetchSub(0, 0, 0);
    } else {
        std::vector<std::future<void>> subs;
        subs.reserve(static_cast<std::size_t>(subPerAxis) * subPerAxis * subPerAxis);
        for (int sz = 0; sz < subPerAxis; ++sz)
            for (int sy = 0; sy < subPerAxis; ++sy)
                for (int sx = 0; sx < subPerAxis; ++sx)
                    subs.push_back(std::async(std::launch::async, fetchSub, sz, sy, sx));
        for (auto& f : subs)
            f.get();
    }

    const auto t1 = std::chrono::steady_clock::now();
    downloadedBytes = fetchedBytes.load();
    if (anyData)
        archive_->appendChunkRaw(lod_, regCz, regCy, regCx, region.data());
    const auto t2 = std::chrono::steady_clock::now();

    using ms = std::chrono::duration<double, std::milli>;
    Logger()->info("mca region l{} ({},{},{}): fetch+assemble={:.0f}ms encode={:.0f}ms data={}",
                   lod_, regCz, regCy, regCx,
                   ms(t1 - t0).count(), ms(t2 - t1).count(), anyData.load());

    // publish the result + wake everyone waiting on this region.
    {
        std::lock_guard<std::mutex> lk(regMu_);
        regions_[rk] = anyData ? RegionState::Present : RegionState::Absent;
    }
    regCv_.notify_all();
    return anyData;
}

ChunkFetchResult MatterCacheFetcher::fetch(const ChunkKey& key)
{
    ChunkFetchResult result;
    // key is in 16^3-chunk coords. enclosing 256^3 region:
    const int blkPerRegion = kMca / kBlk;   // 16
    const int regCz = key.iz / blkPerRegion;
    const int regCy = key.iy / blkPerRegion;
    const int regCx = key.ix / blkPerRegion;
    const int bz = key.iz % blkPerRegion;
    const int by = key.iy % blkPerRegion;
    const int bx = key.ix % blkPerRegion;

    std::size_t downloadedBytes = 0;
    const bool present = ensureRegion(regCz, regCy, regCx, downloadedBytes);
    result.downloadedBytes = downloadedBytes;
    if (!present) {
        // region had no source data -> this block is air/missing.
        result.status = ChunkFetchStatus::Missing;
        return result;
    }

    result.bytes.resize(static_cast<std::size_t>(kBlk) * kBlk * kBlk);
    archive_->decodeBlock(lod_, regCz, regCy, regCx, bz, by, bx,
                          reinterpret_cast<std::uint8_t*>(result.bytes.data()));
    result.status = ChunkFetchStatus::Found;
    return result;
}

}  // namespace vc::render
