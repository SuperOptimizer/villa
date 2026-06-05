#include "vc/core/render/VcaHttpChunkFetcher.hpp"

#include <vc.h>     // libvc (vendored at libs/vc); extern "C" guarded internally
#include <libs3.h>  // vendored S3 client (libs/libs3): range GET + cred resolution

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::render {

namespace {

// Generous block size: atoms are tiny (~hundreds of bytes) and variable; fetching
// a 4 MiB window around any wanted offset turns N per-atom GETs into 1 and
// pre-warms spatially-nearby atoms (XY-Z export tiling clusters them). Over-fetch
// is free: the cache is persistent, so any block pulled is reused forever.
constexpr std::uint64_t kBlockSize = 1ull * 1024 * 1024;   // residency granularity
constexpr std::uint64_t kMaxGet    = 64ull * 1024 * 1024;  // cap a single range GET

// Persistent fixed-block range cache over [0,total_len). Backs libvc's streaming
// read callback: read(off,len) -> ensure covering blocks resident (range-GET +
// pwrite missing ones, coalesced; concurrent same-block fetches collapse to one),
// then copy out. Cache = a sparse local file + a per-block residency bitmap that
// survives across sessions (no re-download of revisited regions).
class VcaHttpBlockCache {
public:
    VcaHttpBlockCache(std::string url,
                      const std::filesystem::path& cacheDir, const std::string& id)
        : url_(std::move(url))
    {
        // libs3 resolves credentials itself (IMDS instance role / SSO / env /
        // ~/.aws) via a cache-served per-request provider — robust for long
        // sessions on rotating STS creds. Empty -> anonymous (public buckets).
        s3_config cfg; std::memset(&cfg, 0, sizeof cfg);
        cfg.max_retries = 5;
        s3_credentials probe; std::memset(&probe, 0, sizeof probe);
        if (s3_credentials_load(nullptr, &probe) == S3_OK) {
            cfg.cred_provider = &credProvider;
            if (probe.region && probe.region[0]) region_ = probe.region;
            cfg.region = region_.empty() ? nullptr : region_.c_str();
            s3_credentials_free(&probe);
        }
        client_ = s3_client_new(&cfg);
        if (!client_) throw std::runtime_error("vca-http: s3_client_new failed");

        // Total length from a HEAD (the archive's true size).
        s3_response r; std::memset(&r, 0, sizeof r);
        if (s3_head(client_, url_.c_str(), &r) != S3_OK || r.status != 200) {
            long st = r.status; s3_response_free(&r); s3_client_free(client_);
            throw std::runtime_error("vca-http: HEAD failed for " + url_ +
                                     " (status " + std::to_string(st) + ")");
        }
        totalLen_ = r.content_length;
        s3_response_free(&r);
        nblocks_ = (totalLen_ + kBlockSize - 1) / kBlockSize;

        std::error_code ec;
        std::filesystem::create_directories(cacheDir, ec);
        dataPath_ = cacheDir / (id + ".vcacache");
        bitsPath_ = cacheDir / (id + ".vcacache.bits");

        // Open (or create) the sparse backing file, sized to totalLen_.
        fd_ = ::open(dataPath_.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0) throw std::runtime_error("vca-http: cannot open cache " + dataPath_.string());
        struct stat st{};
        if (::fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("vca-http: fstat cache"); }
        if (static_cast<std::uint64_t>(st.st_size) != totalLen_) {
            // Size mismatch (new or stale/archive-changed) -> reset.
            if (::ftruncate(fd_, static_cast<off_t>(totalLen_)) != 0) {
                ::close(fd_); throw std::runtime_error("vca-http: ftruncate cache");
            }
            resident_.assign(nblocks_, 0);
            saveBits();
        } else {
            loadBits();   // reuse what we fetched in prior sessions
        }
        inflight_.assign(nblocks_, 0);
    }

    ~VcaHttpBlockCache()
    {
        saveBits();
        if (fd_ >= 0) ::close(fd_);
        if (client_) s3_client_free(client_);
    }

    VcaHttpBlockCache(const VcaHttpBlockCache&) = delete;
    VcaHttpBlockCache& operator=(const VcaHttpBlockCache&) = delete;

    std::uint64_t totalLen() const { return totalLen_; }

    // Register the v2 region blobs as fetch spans: a read anywhere in a region's
    // blob fetches the WHOLE blob in one GET (v2 regions are contiguous). Called
    // once after vc_open_streaming, with spans sorted ascending by offset. The
    // index head [0, firstBlobOff) is span-fetched too (fixed-block fallback
    // covers it before this is set). Idempotent / set-once.
    void registerSpans(std::vector<std::pair<std::uint64_t,std::uint64_t>> spans)
    {
        std::sort(spans.begin(), spans.end());
        std::lock_guard<std::mutex> lk(mu_);
        spans_ = std::move(spans);
    }

    // The libvc vc_read_fn: fill dst[0..len) from archive bytes [off,off+len).
    vc_status read(std::uint64_t off, std::uint32_t len, std::uint8_t* dst)
    {
        if (off + len > totalLen_) return VC_ERR_FORMAT;
        // Ensure the fetch span(s) covering [off,off+len) are resident, then serve.
        std::uint64_t cur = off, end = off + len;
        while (cur < end) {
            std::uint64_t soff, slen;
            spanFor(cur, &soff, &slen);            // the blob/region (or fallback block)
            if (!ensureSpan(soff, slen)) return VC_ERR_IO;
            cur = soff + slen;
        }
        if (preadAll(dst, len, off) != static_cast<ssize_t>(len)) return VC_ERR_IO;
        return VC_OK;
    }

private:
    // The fetch span covering byte `off`: the v2 region blob (or index head) it
    // belongs to once spans are registered, else a fixed kBlockSize block aligned
    // to `off` (used while the directory itself is still being read at open).
    void spanFor(std::uint64_t off, std::uint64_t* soff, std::uint64_t* slen)
    {
        // spans_ is set once at open and read-only after; a brief lock is cheap.
        std::vector<std::pair<std::uint64_t,std::uint64_t>>* sp;
        { std::lock_guard<std::mutex> lk(mu_); sp = spans_.empty()? nullptr : &spans_; }
        if (sp) {
            // binary search: last span with span.first <= off
            std::size_t lo=0, hi=sp->size(), best=SIZE_MAX;
            while (lo<hi){ std::size_t m=(lo+hi)/2; if((*sp)[m].first<=off){best=m;lo=m+1;} else hi=m; }
            if (best!=SIZE_MAX){ auto&s=(*sp)[best]; if(off < s.first+s.second){ *soff=s.first; *slen=s.second; return; } }
            // gap between spans (shouldn't happen for valid offsets) -> single block
        }
        std::uint64_t b = off / kBlockSize;
        *soff = b * kBlockSize;
        *slen = std::min<std::uint64_t>(kBlockSize, totalLen_ - *soff);
    }

    // Ensure the byte span [soff,soff+slen) is resident: if any covering fixed
    // block is missing, GET the whole span (split into <=kMaxGet chunks) and mark
    // its blocks resident. Concurrent callers for the same span coalesce on the
    // span's first block's inflight flag + CV. Slow GET/pwrite run off-lock.
    bool ensureSpan(std::uint64_t soff, std::uint64_t slen)
    {
        if (slen == 0) return true;
        std::uint64_t fb = soff / kBlockSize, lb = (soff+slen-1) / kBlockSize;
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            int allres = 1; for (std::uint64_t b=fb;b<=lb;++b) if(!resident_[b]){allres=0;break;}
            if (allres) return true;
            if (!inflight_[fb]) { inflight_[fb] = 1; break; }   // we own this span's fetch
            cv_.wait(lk);                                       // another thread is fetching; recheck
        }
        lk.unlock();

        // GET [soff,soff+slen) off-lock, in <=kMaxGet pieces (cap a huge blob).
        bool ok = true;
        for (std::uint64_t p = soff; p < soff+slen && ok; ) {
            std::uint32_t n = (std::uint32_t)std::min<std::uint64_t>(kMaxGet, soff+slen - p);
            s3_response resp; std::memset(&resp, 0, sizeof resp);
            s3_status rc = s3_get_range(client_, url_.c_str(), p, n, &resp);
            ok = rc == S3_OK && (resp.status == 206 || resp.status == 200) &&
                 resp.body && resp.body_len == n &&
                 pwriteAll(reinterpret_cast<const std::byte*>(resp.body), n, p) == (ssize_t)n;
            s3_response_free(&resp);
            p += n;
        }

        lk.lock();
        inflight_[fb] = 0;
        if (ok) for (std::uint64_t b=fb;b<=lb;++b) resident_[b] = 1;
        cv_.notify_all();
        return ok;
    }

    ssize_t pwriteAll(const std::byte* p, std::uint32_t n, std::uint64_t off)
    {
        std::uint32_t done = 0;
        while (done < n) {
            ssize_t w = ::pwrite(fd_, reinterpret_cast<const char*>(p) + done, n - done,
                                 static_cast<off_t>(off + done));
            if (w <= 0) return -1;
            done += static_cast<std::uint32_t>(w);
        }
        return n;
    }
    ssize_t preadAll(std::uint8_t* p, std::uint32_t n, std::uint64_t off)
    {
        std::uint32_t done = 0;
        while (done < n) {
            ssize_t r = ::pread(fd_, p + done, n - done, static_cast<off_t>(off + done));
            if (r <= 0) return -1;
            done += static_cast<std::uint32_t>(r);
        }
        return n;
    }

    void loadBits()
    {
        resident_.assign(nblocks_, 0);
        int bf = ::open(bitsPath_.c_str(), O_RDONLY);
        if (bf < 0) return;   // no sidecar -> treat all as not-resident (safe; re-fetch)
        std::vector<std::uint8_t> packed((nblocks_ + 7) / 8);
        ssize_t r = ::read(bf, packed.data(), packed.size());
        ::close(bf);
        if (r == static_cast<ssize_t>(packed.size())) {
            for (std::uint64_t b = 0; b < nblocks_; ++b)
                resident_[b] = (packed[b / 8] >> (b % 8)) & 1;
        }
    }
    void saveBits()
    {
        if (resident_.empty()) return;
        std::vector<std::uint8_t> packed((nblocks_ + 7) / 8, 0);
        for (std::uint64_t b = 0; b < nblocks_; ++b)
            if (resident_[b]) packed[b / 8] |= static_cast<std::uint8_t>(1u << (b % 8));
        int bf = ::open(bitsPath_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (bf < 0) return;
        ssize_t w = ::write(bf, packed.data(), packed.size()); (void)w;
        ::close(bf);
    }

    // Per-request credential resolver: cache-served (IMDS/SSO/env), refreshed
    // before expiry -> safe for long viewer sessions on rotating STS creds.
    static s3_status credProvider(void* ud, s3_credentials* out) {
        (void)ud; return s3_credentials_load(nullptr, out);
    }

    std::string url_;
    std::string region_;
    s3_client* client_ = nullptr;
    std::uint64_t totalLen_ = 0;
    std::uint64_t nblocks_ = 0;
    int fd_ = -1;
    std::filesystem::path dataPath_, bitsPath_;
    std::vector<std::uint8_t> resident_;   // 1 = block fetched
    std::vector<std::uint8_t> inflight_;   // 1 = a thread is fetching the span at this block
    std::vector<std::pair<std::uint64_t,std::uint64_t>> spans_;  // v2 region blobs (sorted), set once
    std::mutex mu_;                        // guards resident_/inflight_/spans_ + cv_
    std::condition_variable cv_;
};

// C thunk: libvc calls this with the cache as userdata.
vc_status blockCacheRead(void* ud, std::uint64_t off, std::uint32_t len, std::uint8_t* dst)
{
    return static_cast<VcaHttpBlockCache*>(ud)->read(off, len, dst);
}

// Owns the block cache + the libvc streaming handle. Shared by all per-LOD
// fetchers. vc_close before the cache (handle borrows the read callback).
class VcaHttpArchive {
public:
    VcaHttpArchive(const std::string& url,
                   const std::filesystem::path& cacheDir, const std::string& id)
        : cache_(std::make_unique<VcaHttpBlockCache>(url, cacheDir, id))
    {
        archive_ = vc_open_streaming(&blockCacheRead, cache_.get(), cache_->totalLen());
        if (!archive_)
            throw std::runtime_error("vca-http: vc_open_streaming rejected " + url);
        // Streaming requires the v2 region-contiguous container (each region is one
        // contiguous blob -> one range-GET). v1 archives scatter payloads and would
        // thrash the cache; reject them — repack v1 -> v2 first (vc_repack).
        if (vc_archive_version(archive_) != 2) {
            vc_close(archive_); archive_ = nullptr;
            throw std::runtime_error(
                "vca-http: " + url + " is not a v2 archive; streaming requires v2 "
                "(run vc_repack to upgrade v1 -> v2)");
        }
        registerRegionSpans();
    }

    ~VcaHttpArchive()
    {
        if (archive_) vc_close(archive_);
    }

    VcaHttpArchive(const VcaHttpArchive&) = delete;
    VcaHttpArchive& operator=(const VcaHttpArchive&) = delete;

    vc_archive* handle() const { return archive_; }

private:
    // Enumerate every present v2 region blob (across all LODs) as a fetch span, so
    // a read into a region pulls its whole contiguous blob in one GET. Reading the
    // directory here goes through the cache (fixed-block fallback). Also adds the
    // index head [0, firstBlobOff) as a span.
    void registerRegionSpans()
    {
        std::vector<std::pair<std::uint64_t,std::uint64_t>> spans;
        std::uint64_t firstBlob = cache_->totalLen();
        for (int lod = 0; lod < VC_NLOD; ++lod) {
            vc_dims d{};
            if (vc_lod_dims(archive_, lod, &d) != VC_OK) break;
            // region grid at this LOD (1024-voxel regions)
            auto rdim = [](int n){ return (unsigned)(((n + 31)/32 + 31)/32); };
            unsigned nrz = rdim(d.nz), nry = rdim(d.ny), nrx = rdim(d.nx);
            for (unsigned rz = 0; rz < nrz; ++rz)
            for (unsigned ry = 0; ry < nry; ++ry)
            for (unsigned rx = 0; rx < nrx; ++rx) {
                std::uint64_t off=0, len=0;
                if (vc_region_blob_range(archive_, lod, rz, ry, rx, &off, &len) == VC_PRESENT
                    && len > 0) {
                    spans.emplace_back(off, len);
                    if (off < firstBlob) firstBlob = off;
                }
            }
        }
        // index head (header + directory) up to the first blob.
        if (firstBlob > 0 && firstBlob <= cache_->totalLen())
            spans.emplace_back(0, firstBlob);
        cache_->registerSpans(std::move(spans));
    }

    std::unique_ptr<VcaHttpBlockCache> cache_;
    vc_archive* archive_ = nullptr;
};

// One fetcher per LOD. Identical to VcaChunkFetcher but the archive streams from
// S3 underneath (coverage + decode pull byte ranges via the block cache).
class VcaHttpChunkFetcher final : public IChunkFetcher {
public:
    VcaHttpChunkFetcher(std::shared_ptr<VcaHttpArchive> archive, int level)
        : archive_(std::move(archive)), level_(level)
    {
    }

    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ChunkFetchResult result;
        vc_archive* a = archive_->handle();
        const std::uint32_t az = static_cast<std::uint32_t>(key.iz);
        const std::uint32_t ay = static_cast<std::uint32_t>(key.iy);
        const std::uint32_t ax = static_cast<std::uint32_t>(key.ix);

        const vc_cover cover = vc_atom_coverage(a, level_, az, ay, ax);
        if (cover != VC_PRESENT) {
            // ABSENT can't happen on a complete remote archive; KNOWN_ZERO ->
            // all-fill. Either way no content to serve.
            result.status = ChunkFetchStatus::Missing;
            return result;
        }
        auto bytes = std::vector<std::byte>(VC_ATOM3);
        const vc_status s = vc_decode_atom(
            a, level_, static_cast<int>(ax), static_cast<int>(ay),
            static_cast<int>(az), reinterpret_cast<std::uint8_t*>(bytes.data()));
        if (s != VC_OK) {
            result.status = ChunkFetchStatus::DecodeError;
            result.message = "vc_decode_atom status " + std::to_string(s);
            return result;
        }
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::move(bytes);
        return result;
    }

private:
    std::shared_ptr<VcaHttpArchive> archive_;
    int level_;
};

// Derive a filesystem-safe cache id from the URL.
std::string cacheIdFromUrl(const std::string& url)
{
    std::string id;
    id.reserve(url.size());
    for (char c : url) id += (std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    if (id.size() > 200) id = id.substr(id.size() - 200);   // keep tail (the filename)
    return id;
}

} // namespace

OpenedChunkedZarr openHttpVcaArchive(const std::string& httpsUrl,
                                     const vc::HttpAuth& auth,
                                     const std::filesystem::path& cacheDir)
{
    // libs3 resolves credentials itself (IMDS/SSO/env); `auth` is accepted for
    // signature compatibility with the zarr opener but not needed here.
    (void)auth;
    auto archive = std::make_shared<VcaHttpArchive>(
        httpsUrl, cacheDir, cacheIdFromUrl(httpsUrl));
    vc_archive* a = archive->handle();

    OpenedChunkedZarr opened;
    opened.dtype = ChunkDtype::UInt8;
    opened.fillValue = 0.0;

    for (int lod = 0; lod < VC_NLOD; ++lod) {
        vc_dims d{};
        if (vc_lod_dims(a, lod, &d) != VC_OK) break;
        opened.levelNumbers.push_back(lod);
        opened.shapes.push_back({static_cast<int>(d.nz),
                                 static_cast<int>(d.ny),
                                 static_cast<int>(d.nx)});
        const std::array<int, 3> chunk32{static_cast<int>(VC_ATOM),
                                         static_cast<int>(VC_ATOM),
                                         static_cast<int>(VC_ATOM)};
        opened.chunkShapes.push_back(chunk32);
        opened.storageChunkShapes.push_back(chunk32);
        IChunkedArray::LevelTransform t;
        const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << lod);
        t.scaleFromLevel0 = {invScale, invScale, invScale};
        t.offsetFromLevel0 = {0.0, 0.0, 0.0};
        opened.transforms.push_back(t);
        opened.fetchers.push_back(
            std::make_shared<VcaHttpChunkFetcher>(archive, lod));
    }

    if (opened.fetchers.empty())
        throw std::runtime_error("vca-http: no LODs in " + httpsUrl);
    return opened;
}

} // namespace vc::render
