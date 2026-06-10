#include "vc/core/render/MatterCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <chrono>
#include <future>
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
    if (failed) {
        downloadedBytes = fetchedBytes.load();
        {
            std::lock_guard<std::mutex> lk(regMu_);
            regions_.erase(rk);
        }
        regCv_.notify_all();
        Logger()->warn("mca region l{} ({},{},{}): sub-fetch failed; will retry",
                       lod_, regCz, regCy, regCx);
        return false;
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

    {
        std::unique_lock<std::mutex> lk(regMu_);
        for (;;) {
            auto it = regions_.find(rk);
            if (it == regions_.end())
                break;
            if (it->second == RegionState::InFlight) {
                regCv_.wait(lk);
                continue;
            }
            return it->second == RegionState::Present;
        }
        if (archive_->hasChunk(lod_, cz, cy, cx)) {   // mirrored in a previous run
            regions_[rk] = RegionState::Present;
            return true;
        }
        regions_[rk] = RegionState::InFlight;
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
        }
    } catch (const std::exception& e) {
        error = e.what();
    }
    if (!error.empty())
        Logger()->warn("mca stream l{} ({},{},{}): {}", lod_, cz, cy, cx, error);

    {
        std::lock_guard<std::mutex> lk(regMu_);
        if (error.empty())
            regions_[rk] = present ? RegionState::Present : RegionState::Absent;
        else
            regions_.erase(rk);   // transient failure: let a later fetch retry
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
    const bool fresh = !std::filesystem::exists(mcaPath);
    auto archive = std::make_shared<MatterArchive>(mcaPath.string(), remote->shape0(),
                                                   remote->quality(), cacheBytes);
    if (fresh) {
        // mirror the per-volume codec priors so local decode matches the remote.
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
