#include "vc/core/render/MatterCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstring>
#include <chrono>
#include <future>
#include <set>
#include <unordered_map>

#include <filesystem>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "vc/core/util/Logging.hpp"

extern "C" {
#include "libs3.h"
#include "matter_compressor.h"
}

namespace vc::render {

// shared all-zero 256^3 region buffer (air): appended to mark a region ZERO.
static const std::vector<std::byte>& kZeroRegion()
{
    static const std::vector<std::byte> z(
        static_cast<std::size_t>(MatterArchive::kChunk) * MatterArchive::kChunk *
            MatterArchive::kChunk,
        std::byte{0});
    return z;
}

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
    createdFresh_ = !std::filesystem::exists(path_);
    impl_->a = open();
    if (!impl_->a && std::filesystem::exists(path_)) {
        // stale archive (older format version / different dims): it's a rebuildable
        // cache, so delete and recreate.
        Logger()->warn("MatterArchive: {} is stale/incompatible; recreating", path_);
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        createdFresh_ = true;
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

std::uint64_t MatterArchive::diskBytes() const
{
    return impl_->a ? mc_archive_data_len(impl_->a) : 0;
}

bool MatterArchive::appendChunkCompressed(int lod, int cz, int cy, int cx,
                                          const std::uint8_t* blob, std::size_t len)
{
    if (!blob || !len) return false;
    return mc_archive_append_chunk_compressed(impl_->a, lod, cz, cy, cx, blob, len) == 0;
}

bool MatterArchive::setPriors(const std::uint16_t* plo, const std::uint16_t* phi)
{
    if (!plo || !phi) return false;
    return mc_archive_set_priors(impl_->a,
                                 reinterpret_cast<const std::uint16_t(*)[32]>(plo),
                                 reinterpret_cast<const std::uint16_t(*)[32]>(phi)) == 0;
}

bool MatterArchive::hasChunk(int lod, int cz, int cy, int cx) const
{
    // PRESENT (has data) OR ZERO (fetched, all-air) both count as cached — the
    // region was visited and never needs re-fetching. Only ABSENT means "fetch".
    return mc_archive_chunk_coverage(impl_->a, lod, cz, cy, cx) != MC_ABSENT;
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

    // Single-flight: `inFlight_` holds ONLY the regions currently being assembled,
    // so it is self-bounding (no resolved-state cache that grows forever). The
    // archive's hasChunk is the source of truth for "already done".
    {
        std::unique_lock<std::mutex> lk(regMu_);
        for (;;) {
            if (archive_->hasChunk(lod_, regCz, regCy, regCx))
                return true;   // already present (this run or a previous one)
            if (inFlight_.count(rk)) {
                regCv_.wait(lk);   // another thread is assembling it; recheck on wake
                continue;
            }
            break;   // not present, not in flight -> we claim it
        }
        inFlight_.insert(rk);
    }

    // ---- we own this region: assemble + encode OUTSIDE the lock ----
    const auto t0 = std::chrono::steady_clock::now();
    // Assemble the 256^3 region from the source's native chunks (srcEdge_ each).
    const int subPerAxis = kMca / srcEdge_;   // 1 (c3d 256) or 2 (zarr 128)
    std::vector<std::uint8_t> region(static_cast<std::size_t>(kMca) * kMca * kMca, 0);

    // region voxel origin
    const int vz0 = regCz * kMca, vy0 = regCy * kMca, vx0 = regCx * kMca;
    std::atomic<bool> anyData{false};
    std::atomic<bool> failed{false};
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
            failed = true;   // transient: do NOT bake zeros into the archive
            return;
        }
        if (sub.status == ChunkFetchStatus::HttpError ||
            sub.status == ChunkFetchStatus::IoError ||
            sub.status == ChunkFetchStatus::DecodeError) {
            failed = true;   // transient: do NOT bake zeros into the archive
            return;
        }
        if (sub.status != ChunkFetchStatus::Found || sub.bytes.empty())
            return;   // genuinely missing/air sub-chunk -> stays zero in the region
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

    // any sub-fetch error means the assembled region may have zero-filled holes:
    // appending it would PERSIST the corruption. Unclaim the region instead so a
    // later fetch retries once the source recovers.
    // unclaim helper: drop the in-flight mark and wake waiters (who re-check
    // hasChunk). Used on both success and transient failure.
    auto unclaim = [&] {
        std::lock_guard<std::mutex> lk(regMu_);
        inFlight_.erase(rk);
        regCv_.notify_all();
    };

    if (failed) {
        downloadedBytes = fetchedBytes.load();
        unclaim();
        Logger()->warn("mca region l{} ({},{},{}): sub-fetch failed; will retry",
                       lod_, regCz, regCy, regCx);
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    downloadedBytes = fetchedBytes.load();
    // Append the assembled region — data OR all-air. An all-air append records the
    // ZERO sentinel, so hasChunk returns true next time and it is never re-fetched.
    if (!archive_->appendChunkRaw(lod_, regCz, regCy, regCx, region.data())) {
        unclaim();   // archive write failed (disk full?) -> let a later fetch retry
        Logger()->warn("mca region l{} ({},{},{}): archive append failed; will retry",
                       lod_, regCz, regCy, regCx);
        return false;
    }
    const auto t2 = std::chrono::steady_clock::now();

    using ms = std::chrono::duration<double, std::milli>;
    Logger()->info("mca region l{} ({},{},{}): fetch+assemble={:.0f}ms encode={:.0f}ms data={}",
                   lod_, regCz, regCy, regCx,
                   ms(t1 - t0).count(), ms(t2 - t1).count(), anyData.load());

    unclaim();
    // the region is now in the archive (data, or the ZERO sentinel for air);
    // either way it is present and decodes correctly.
    return true;
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

FetchBatch MatterCacheFetcher::shardBatch(const ChunkKey& key) const
{
    // key is in 16^3-block coords; translate to the source chunk, ask the source
    // for its shard extent, translate back to block units.
    const int blkPerRegion = kMca / kBlk;
    const int regCz = key.iz / blkPerRegion;
    const int regCy = key.iy / blkPerRegion;
    const int regCx = key.ix / blkPerRegion;
    const int srcCz = (regCz * kMca) / srcEdge_;
    const int srcCy = (regCy * kMca) / srcEdge_;
    const int srcCx = (regCx * kMca) / srcEdge_;
    const FetchBatch sb = source_->shardBatch(ChunkKey{lod_, srcCz, srcCy, srcCx});
    const int blkPerSrc = srcEdge_ / kBlk;
    return FetchBatch{{sb.originChunk[0] * blkPerSrc, sb.originChunk[1] * blkPerSrc,
                       sb.originChunk[2] * blkPerSrc},
                      sb.edgeChunks * blkPerSrc};
}

bool MatterCacheFetcher::prefetchShard(const ChunkKey& key, const ShardSink&)
{
    // c3d: a 256^3 mca region IS one source chunk; the source shard holds many of
    // them. Download the whole shard once, collect its inner chunks, then encode
    // them into the .mca in PARALLEL — one thread per inner chunk (each c3d decode
    // + mc encode is single-threaded; appendChunkRaw is lock-free across threads).
    const int blkPerRegion = kMca / kBlk;
    const int srcCz = (key.iz / blkPerRegion * kMca) / srcEdge_;
    const int srcCy = (key.iy / blkPerRegion * kMca) / srcEdge_;
    const int srcCx = (key.ix / blkPerRegion * kMca) / srcEdge_;

    // region extent this shard covers (clamped to the volume edge).
    const FetchBatch sb = source_->shardBatch(ChunkKey{lod_, srcCz, srcCy, srcCx});
    const int subPerRegion = std::max(1, kMca / srcEdge_);
    const int regZ0 = (sb.originChunk[0] * srcEdge_) / kMca;
    const int regY0 = (sb.originChunk[1] * srcEdge_) / kMca;
    const int regX0 = (sb.originChunk[2] * srcEdge_) / kMca;
    const int regN = std::max(1, sb.edgeChunks / subPerRegion);
    const int rzMax = (levelShape_[0] + kMca - 1) / kMca;
    const int ryMax = (levelShape_[1] + kMca - 1) / kMca;
    const int rxMax = (levelShape_[2] + kMca - 1) / kMca;
    const int rz1 = std::min(regZ0 + regN, rzMax);
    const int ry1 = std::min(regY0 + regN, ryMax);
    const int rx1 = std::min(regX0 + regN, rxMax);

    // Skip the (expensive) shard download entirely if every region it covers is
    // already in the .mca — e.g. re-running prefetch over an already-warmed cache.
    {
        bool allPresent = true;
        for (int z = regZ0; z < rz1 && allPresent; ++z)
            for (int y = regY0; y < ry1 && allPresent; ++y)
                for (int x = regX0; x < rx1; ++x)
                    if (!archive_->hasChunk(lod_, z, y, x)) { allPresent = false; break; }
        if (allPresent)
            return true;
    }

    // Cheap index probe: if the source shard is ALL AIR, mark every region ZERO
    // and skip the multi-hundred-MB download. Common for air-padded coarse levels.
    if (auto air = source_->shardAllAir(ChunkKey{lod_, srcCz, srcCy, srcCx}); air && *air) {

        for (int z = regZ0; z < rz1; ++z)
            for (int y = regY0; y < ry1; ++y)
                for (int x = regX0; x < rx1; ++x)
                    if (!archive_->hasChunk(lod_, z, y, x))
                        archive_->appendChunkRaw(
                            lod_, z, y, x,
                            reinterpret_cast<const std::uint8_t*>(kZeroRegion().data()));
        return true;
    }

    std::atomic<bool> ok{true};
    if (srcEdge_ != kMca) {
        // 128^3 source: regions need 2x2x2 coalescing — use the normal path per
        // region (still benefits from the source's one-GET shard download).
        source_->prefetchShard(
            ChunkKey{lod_, srcCz, srcCy, srcCx},
            [&](const ChunkKey& sk, std::vector<std::byte>&&) {
                const int regCz = (sk.iz * srcEdge_) / kMca;
                const int regCy = (sk.iy * srcEdge_) / kMca;
                const int regCx = (sk.ix * srcEdge_) / kMca;
                std::size_t dl = 0;
                ensureRegion(regCz, regCy, regCx, dl);
            });
        return ok.load();
    }

    // collect the shard's PRESENT inner chunks (1:1 with mca regions), and track
    // which region positions were delivered so the rest can be marked all-air.
    struct Region { int cz, cy, cx; std::vector<std::byte> vox; };
    std::vector<Region> regions;
    std::set<std::array<int, 3>> present;
    {
        std::mutex mu;
        source_->prefetchShard(
            ChunkKey{lod_, srcCz, srcCy, srcCx},
            [&](const ChunkKey& sk, std::vector<std::byte>&& bytes) {
                const int regCz = (sk.iz * srcEdge_) / kMca;
                const int regCy = (sk.iy * srcEdge_) / kMca;
                const int regCx = (sk.ix * srcEdge_) / kMca;
                std::lock_guard<std::mutex> lk(mu);
                present.insert({regCz, regCy, regCx});
                if (archive_->hasChunk(lod_, regCz, regCy, regCx))
                    return;
                if (bytes.size() != static_cast<std::size_t>(kMca) * kMca * kMca)
                    return;
                regions.push_back({regCz, regCy, regCx, std::move(bytes)});
            });

        // Every region the shard covers but did NOT deliver is all-air: record it
        // as ZERO so a re-run skips it instead of re-downloading the shard. (Air
        // padding at coarse levels is common.) Append a zero buffer once.

        for (int z = regZ0; z < rz1; ++z)
            for (int y = regY0; y < ry1; ++y)
                for (int x = regX0; x < rx1; ++x) {
                    if (present.count({z, y, x}) || archive_->hasChunk(lod_, z, y, x))
                        continue;
                    archive_->appendChunkRaw(
                        lod_, z, y, x,
                        reinterpret_cast<const std::uint8_t*>(kZeroRegion().data()));
                }
    }
    if (regions.empty())
        return ok.load();

    std::atomic<std::size_t> next{0};
    // bound the inner encode team: the caller already runs several shard drivers,
    // so a full hardware_concurrency() team per shard would oversubscribe badly.
    const unsigned nThreads = std::min<unsigned>(
        std::min<unsigned>(8u, std::max(1u, std::thread::hardware_concurrency())),
        regions.size());
    std::vector<std::thread> team;
    for (unsigned t = 0; t < nThreads; ++t)
        team.emplace_back([&] {
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= regions.size())
                    return;
                const Region& r = regions[i];
                if (!archive_->appendChunkRaw(
                        lod_, r.cz, r.cy, r.cx,
                        reinterpret_cast<const std::uint8_t*>(r.vox.data())))
                    ok.store(false);
            }
        });
    for (auto& th : team)
        th.join();
    return ok.load();
}


// ============================================================================
// MatterRemoteSource
// ============================================================================

struct MatterRemoteSource::Impl {
    std::string url;
    s3_client* client = nullptr;
    mc_reader* reader = nullptr;            // streaming: resolve offsets/lengths only
    std::uint64_t totalLen = 0;
    std::mutex readerMu;                    // mc_reader is single-threaded
    std::atomic<std::uint64_t> downloaded{0};

    // mc_read_fn: small index reads (header, node tables, blob headers) served by
    // direct ranged GETs; the reader FIFO-caches node tables so these amortize.
    static int readThunk(void* ud, std::uint64_t off, std::uint32_t len, std::uint8_t* dst)
    {
        auto* impl = static_cast<Impl*>(ud);
        s3_response r{};
        const s3_status rc = s3_get_range(impl->client, impl->url.c_str(), off, len, &r);
        const bool ok = rc == S3_OK && (r.status == 206 || r.status == 200) &&
                        r.body && r.body_len == len;
        if (ok) {
            std::memcpy(dst, r.body, len);
            impl->downloaded.fetch_add(len, std::memory_order_relaxed);
        }
        s3_response_free(&r);
        return ok ? 0 : -1;
    }

    static s3_status credProvider(void*, s3_credentials* out)
    {
        return s3_credentials_load(nullptr, out);
    }
};

MatterRemoteSource::MatterRemoteSource(std::string url)
    : impl_(std::make_unique<Impl>())
{
    impl_->url = std::move(url);

    // libs3 resolves credentials per request (cache-served; survives STS rotation
    // over long sessions); no creds found -> anonymous (public buckets).
    s3_config cfg{};
    cfg.max_retries = 5;
    std::string region;
    s3_credentials probe{};
    if (s3_credentials_load(nullptr, &probe) == S3_OK) {
        cfg.cred_provider = &Impl::credProvider;
        if (probe.region && probe.region[0]) region = probe.region;
        cfg.region = region.empty() ? nullptr : region.c_str();
        s3_credentials_free(&probe);
    }
    impl_->client = s3_client_new(&cfg);
    if (!impl_->client)
        throw std::runtime_error("mca remote: s3_client_new failed");

    s3_response r{};
    if (s3_head(impl_->client, impl_->url.c_str(), &r) != S3_OK || r.status != 200) {
        const long st = r.status;
        s3_response_free(&r);
        throw std::runtime_error("mca remote: HEAD failed for " + impl_->url +
                                 " (status " + std::to_string(st) + ")");
    }
    impl_->totalLen = r.content_length;
    s3_response_free(&r);

    impl_->reader = mc_open_streaming(&Impl::readThunk, impl_.get(), impl_->totalLen);
    if (!impl_->reader)
        throw std::runtime_error("mca remote: not a matter-compressor archive: " + impl_->url);

    int nx = 0, ny = 0, nz = 0;
    mc_reader_dims(impl_->reader, &nx, &ny, &nz);
    nlods_ = mc_reader_nlods(impl_->reader);
    quality_ = mc_reader_quality(impl_->reader);
    if (nx <= 0 || ny <= 0 || nz <= 0 || nlods_ <= 0)
        throw std::runtime_error("mca remote: bad archive header in " + impl_->url);
    shape0_ = {nz, ny, nx};
}

MatterRemoteSource::~MatterRemoteSource()
{
    if (impl_ && impl_->reader)
        mc_close(impl_->reader);
    if (impl_ && impl_->client)
        s3_client_free(impl_->client);
}

std::pair<const std::uint16_t*, const std::uint16_t*> MatterRemoteSource::priors() const
{
    const std::uint16_t *plo = nullptr, *phi = nullptr;
    if (!mc_reader_priors(impl_->reader, &plo, &phi))
        return {nullptr, nullptr};
    return {plo, phi};
}

std::uint64_t MatterRemoteSource::downloadedBytes() const
{
    return impl_->downloaded.load(std::memory_order_relaxed);
}

std::vector<std::uint8_t> MatterRemoteSource::fetchChunkBlob(int lod, int cz, int cy, int cx)
{
    std::uint64_t off = 0, len = 0;
    {
        std::lock_guard<std::mutex> lk(impl_->readerMu);
        off = mc_chunk_offset(impl_->reader, lod, cz, cy, cx);
        if (!off)
            return {};   // absent chunk (air)
        len = mc_reader_chunk_blob_len(impl_->reader, off);
    }
    if (!len)
        throw std::runtime_error("mca remote: bad blob length for chunk in " + impl_->url);

    // one ranged GET for the whole compressed blob, outside the reader lock.
    std::vector<std::uint8_t> blob(len);
    for (std::uint64_t p = 0; p < len;) {
        const auto n = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(64ull * 1024 * 1024, len - p));
        s3_response r{};
        const s3_status rc = s3_get_range(impl_->client, impl_->url.c_str(), off + p, n, &r);
        const bool ok = rc == S3_OK && (r.status == 206 || r.status == 200) &&
                        r.body && r.body_len == n;
        if (ok)
            std::memcpy(blob.data() + p, r.body, n);
        s3_response_free(&r);
        if (!ok)
            throw std::runtime_error("mca remote: blob fetch failed for " + impl_->url);
        p += n;
    }
    impl_->downloaded.fetch_add(len, std::memory_order_relaxed);
    return blob;
}

// ============================================================================
// MatterStreamFetcher
// ============================================================================

MatterStreamFetcher::MatterStreamFetcher(std::shared_ptr<MatterRemoteSource> remote,
                                         std::shared_ptr<MatterArchive> archive,
                                         int lod)
    : remote_(std::move(remote))
    , archive_(std::move(archive))
    , lod_(lod)
{
}

bool MatterStreamFetcher::ensureChunk(int cz, int cy, int cx, std::size_t& downloadedBytes)
{
    downloadedBytes = 0;
    const std::uint64_t rk = (static_cast<std::uint64_t>(cz & 0x1FFFFF) << 42) |
                             (static_cast<std::uint64_t>(cy & 0x1FFFFF) << 21) |
                             (static_cast<std::uint64_t>(cx & 0x1FFFFF));

    // self-bounding single-flight: inFlight_ holds only chunks being mirrored now;
    // hasChunk (present OR ZERO sentinel for air) is the source of truth.
    {
        std::unique_lock<std::mutex> lk(regMu_);
        for (;;) {
            if (archive_->hasChunk(lod_, cz, cy, cx))
                return true;   // mirrored already (data or air-marked this/prior run)
            if (inFlight_.count(rk)) {
                regCv_.wait(lk);
                continue;
            }
            break;
        }
        inFlight_.insert(rk);
    }

    bool present = false;
    std::string error;
    try {
        const auto blob = remote_->fetchChunkBlob(lod_, cz, cy, cx);
        downloadedBytes = blob.size();
        if (!blob.empty()) {
            if (!archive_->appendChunkCompressed(lod_, cz, cy, cx, blob.data(), blob.size()))
                error = "appendChunkCompressed failed";
            else
                present = true;
        } else {
            // remote chunk is air: mark it ZERO so we never re-probe the remote.
            archive_->appendChunkRaw(
                lod_, cz, cy, cx,
                reinterpret_cast<const std::uint8_t*>(kZeroRegion().data()));
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!error.empty())
        Logger()->warn("mca stream l{} ({},{},{}): {}", lod_, cz, cy, cx, error);

    {
        std::lock_guard<std::mutex> lk(regMu_);
        inFlight_.erase(rk);   // transient failure leaves hasChunk false -> retries
    }
    regCv_.notify_all();
    return present;
}

ChunkFetchResult MatterStreamFetcher::fetch(const ChunkKey& key)
{
    ChunkFetchResult result;
    const int blkPerRegion = MatterArchive::kBlocksPerAxis;
    const int cz = key.iz / blkPerRegion, bz = key.iz % blkPerRegion;
    const int cy = key.iy / blkPerRegion, by = key.iy % blkPerRegion;
    const int cx = key.ix / blkPerRegion, bx = key.ix % blkPerRegion;

    std::size_t downloadedBytes = 0;
    const bool present = ensureChunk(cz, cy, cx, downloadedBytes);
    result.downloadedBytes = downloadedBytes;
    if (!present) {
        result.status = ChunkFetchStatus::Missing;
        return result;
    }
    result.bytes.resize(static_cast<std::size_t>(MatterArchive::kBlock) *
                        MatterArchive::kBlock * MatterArchive::kBlock);
    archive_->decodeBlock(lod_, cz, cy, cx, bz, by, bx,
                          reinterpret_cast<std::uint8_t*>(result.bytes.data()));
    result.status = ChunkFetchStatus::Found;
    return result;
}

// ============================================================================
// openHttpMcaArchive
// ============================================================================

std::shared_ptr<MatterArchive> openHttpMcaArchive(const std::string& url,
                                                  const std::filesystem::path& cacheDir,
                                                  std::size_t cacheBytes,
                                                  OpenedChunkedZarr& opened)
{
    auto remote = std::make_shared<MatterRemoteSource>(url);

    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    const auto mcaPath = cacheDir / "volume.mca";
    auto archive = std::make_shared<MatterArchive>(mcaPath.string(), remote->shape0(),
                                                   remote->quality(), cacheBytes);
    if (archive->createdFresh()) {
        // mirror the per-volume codec priors so local decode matches the remote.
        // createdFresh also covers a stale archive deleted+recreated by the ctor.
        auto [plo, phi] = remote->priors();
        if (plo && phi)
            archive->setPriors(plo, phi);
    }

    const auto s0 = remote->shape0();
    const int nlods = remote->nlods();
    opened = {};
    for (int lod = 0; lod < nlods; ++lod) {
        std::array<int, 3> shape{};
        for (int i = 0; i < 3; ++i)
            shape[i] = std::max(1, (s0[i] + (1 << lod) - 1) >> lod);
        const double inv = 1.0 / static_cast<double>(1u << lod);
        IChunkedArray::LevelTransform t;
        t.scaleFromLevel0 = {inv, inv, inv};
        opened.levelNumbers.push_back(lod);
        opened.shapes.push_back(shape);
        opened.chunkShapes.push_back({MatterArchive::kBlock, MatterArchive::kBlock,
                                      MatterArchive::kBlock});
        opened.storageChunkShapes.push_back({MatterArchive::kChunk, MatterArchive::kChunk,
                                             MatterArchive::kChunk});
        opened.transforms.push_back(t);
        opened.fetchers.push_back(std::make_shared<MatterStreamFetcher>(remote, archive, lod));
    }
    opened.dtype = ChunkDtype::UInt8;
    opened.fillValue = 0.0;
    Logger()->info("mca stream: {} -> {} (shape0={}x{}x{} lods={} q={})", url,
                   mcaPath.string(), s0[0], s0[1], s0[2], nlods, remote->quality());
    return archive;
}

}  // namespace vc::render
