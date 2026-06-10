#include "vc/core/render/MatterCacheFetcher.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <chrono>
#include <future>

#include "vc/core/util/Logging.hpp"

namespace vc::render {

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
