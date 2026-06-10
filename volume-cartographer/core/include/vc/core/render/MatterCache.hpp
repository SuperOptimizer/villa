#pragma once

// MatterArchive — RAII C++ wrapper around the matter-compressor (.mca) appendable,
// persistent, crash-safe archive plus its in-RAM decoded-block cache (mc_cache).
// Used as VC3D's render cache: remote chunks are fetched at the volume's native
// chunking, re-encoded into ONE .mca per volume (all chunks at all LODs), and served
// back as decoded 16^3 blocks through mc_cache (sharded CLOCK/NRU eviction).
//
// Storage/encode unit: 256^3 chunk (matter-compressor's contiguous on-disk unit).
// Decode/serve unit:    16^3 block (mc's native decode granularity).
//
// Thread-safety: appends, decodes and cache reads are all safe from many threads.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

#include "vc/core/render/ChunkFetch.hpp"

namespace vc::render {

class MatterArchive {
public:
    static constexpr int kChunk = 256;   // MC_CHUNK
    static constexpr int kBlock = 16;     // MC_BLK
    static constexpr int kBlocksPerAxis = kChunk / kBlock;   // 16

    // Open (or create) a persistent appendable archive at `path` for a volume of
    // LOD0 extent (nz,ny,nx) voxels (each axis padded to a 256 multiple internally),
    // encoded at `quality`, with an mc_cache resident-block budget of `cacheBytes`.
    // A stale archive (older format version / different dims) is deleted and
    // recreated. Throws std::runtime_error on failure.
    MatterArchive(std::string path, std::array<int, 3> shape0, float quality,
                  std::size_t cacheBytes);

    // Append an already-compressed chunk blob verbatim (no re-encode) — the
    // .mca -> .mca streaming fast path. Thread-safe. Returns true on success.
    bool appendChunkCompressed(int lod, int cz, int cy, int cx,
                               const std::uint8_t* blob, std::size_t len);

    // Install per-volume codec priors (from a remote archive being mirrored).
    bool setPriors(const std::uint16_t* plo, const std::uint16_t* phi);
    ~MatterArchive();

    MatterArchive(const MatterArchive&) = delete;
    MatterArchive& operator=(const MatterArchive&) = delete;

    // Append one 256^3 chunk of raw u8 voxels at chunk coords (cz,cy,cx) of `lod`.
    // `vox` is 256^3 in (z,y,x) raster order, x fastest. Re-encodes via mc. Thread-safe.
    // Returns true on success. An all-air chunk is a successful no-op.
    bool appendChunkRaw(int lod, int cz, int cy, int cx, const std::uint8_t* vox256);

    // Is a chunk present in the archive (without decoding)?
    bool hasChunk(int lod, int cz, int cy, int cx) const;

    // Get one 16^3 block (bz,by,bx in [0,16)) of chunk (cz,cy,cx) at `lod` into
    // `dst` (16^3 = 4096 bytes), via mc_cache (decode on miss). Thread-safe.
    void decodeBlock(int lod, int cz, int cy, int cx, int bz, int by, int bx,
                     std::uint8_t* dst4096) const;

    struct CacheStats {
        std::uint64_t hits = 0, misses = 0, evictions = 0;
        std::size_t usedBytes = 0, capacityBytes = 0;
    };
    CacheStats cacheStats() const;

    std::array<int, 3> shape0() const { return shape0_; }
    float quality() const { return quality_; }
    const std::string& path() const { return path_; }
    // true when the ctor created the file (including stale-delete + recreate).
    bool createdFresh() const { return createdFresh_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string path_;
    std::array<int, 3> shape0_{};
    float quality_ = 0.f;
    bool createdFresh_ = false;
};


// MatterCacheFetcher — an IChunkFetcher decorator that re-expresses a remote volume
// through a matter-compressor (.mca) cache. "Fetch native, serve mca-native":
//
//   - The volume is reported to the ChunkCache at mca's native 16^3 chunk granularity
//     (the resident cache resides 4KB blocks).
//   - On fetch(16^3 key): find the enclosing 256^3 mca region. If it's not yet in the
//     .mca, fetch the SOURCE's native chunks covering that region (256^3 c3d = 1:1, or
//     128^3 zarr-v2 = an eager 2x2x2 coalesce), assemble a 256^3 u8 buffer, encode it
//     into the .mca once. Then decode the requested 16^3 block out of the .mca.
//
// One .mca file holds every chunk at every LOD for a volume and persists across runs.
// Each level gets its own MatterCacheFetcher sharing the volume's single MatterArchive.




class MatterCacheFetcher final : public IChunkFetcher {
public:
    // `source` is the underlying (zarr/c3d) fetcher for THIS level; `sourceChunk` is its
    // native chunk edge (128 or 256, assumed cubic); `archive` is the shared per-volume
    // .mca; `lod` is this level's index in the archive; `levelShape` is the level's voxel
    // extent (to bound region assembly at the volume edge).
    MatterCacheFetcher(std::shared_ptr<IChunkFetcher> source,
                       std::shared_ptr<MatterArchive> archive,
                       int lod,
                       int sourceChunkEdge,
                       std::array<int, 3> levelShape);

    // fetch one 16^3 block (key is in 16^3-chunk coords). Returns the decoded 4096 bytes.
    ChunkFetchResult fetch(const ChunkKey& key) override;

    // Shard extent enclosing this 16^3-block key, in 16^3-block units.
    FetchBatch shardBatch(const ChunkKey& key) const override;

    // Download the source shard enclosing this region (one parallel GET) and
    // encode each covered 256^3 region into the .mca. The prefetch fast path.
    bool prefetchShard(const ChunkKey& key, const ShardSink& sink) override;

    static constexpr int kMca = MatterArchive::kChunk;   // 256
    static constexpr int kBlk = MatterArchive::kBlock;    // 16

private:
    // ensure the 256^3 region (regCz,regCy,regCx) is present in the .mca, encoding it
    // from the source on a miss. Returns false if the source had no data for the
    // region. `downloadedBytes` receives the bytes THIS call pulled from the source
    // (0 when the region was already present or another thread assembled it).
    bool ensureRegion(int regCz, int regCy, int regCx, std::size_t& downloadedBytes);

    std::shared_ptr<IChunkFetcher> source_;
    std::shared_ptr<MatterArchive> archive_;
    int lod_ = 0;
    int srcEdge_ = 256;                 // source chunk edge (128 or 256)
    std::array<int, 3> levelShape_{};   // voxel extent of this level

    // Per-256^3-region single-flight: the FIRST thread to touch a region assembles +
    // encodes it; every other thread requesting any 16^3 block of that same region waits
    // for that one assembly instead of redundantly re-fetching + re-encoding it.
    enum class RegionState { InFlight, Present, Absent };
    std::mutex regMu_;
    std::condition_variable regCv_;
    std::unordered_map<std::uint64_t, RegionState> regions_;   // region key -> state
};

// MatterRemoteSource — index/blob access to a REMOTE .mca over HTTPS/S3 (libs3):
// a streaming mc_reader resolves chunk offsets/blob lengths through small ranged
// GETs (node tables FIFO-cached by the reader); whole compressed chunk blobs are
// then fetched with one ranged GET each. Resolution is mutex-serialized (cheap);
// blob GETs run outside the lock.
class MatterRemoteSource {
public:
    explicit MatterRemoteSource(std::string url);
    ~MatterRemoteSource();
    MatterRemoteSource(const MatterRemoteSource&) = delete;
    MatterRemoteSource& operator=(const MatterRemoteSource&) = delete;

    std::array<int, 3> shape0() const { return shape0_; }   // (z,y,x) LOD0 voxels
    int nlods() const { return nlods_; }
    float quality() const { return quality_; }
    // raw per-volume priors (u16[8][32] each), or {nullptr,nullptr} if none.
    std::pair<const std::uint16_t*, const std::uint16_t*> priors() const;

    // Fetch the compressed blob of chunk (lod,cz,cy,cx). Empty result = chunk
    // absent in the remote (air). Throws std::runtime_error on transport errors.
    std::vector<std::uint8_t> fetchChunkBlob(int lod, int cz, int cy, int cx);

    std::uint64_t downloadedBytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::array<int, 3> shape0_{};
    int nlods_ = 0;
    float quality_ = 0.f;
};

// MatterStreamFetcher — IChunkFetcher for a remote .mca: on a region miss, the
// chunk's COMPRESSED blob is fetched from the remote and appended verbatim into
// the local mirror archive (no decode/re-encode); blocks then serve from the
// local archive's mc_cache exactly like every other mca volume.
class MatterStreamFetcher final : public IChunkFetcher {
public:
    MatterStreamFetcher(std::shared_ptr<MatterRemoteSource> remote,
                        std::shared_ptr<MatterArchive> archive,
                        int lod);
    ChunkFetchResult fetch(const ChunkKey& key) override;

private:
    // ensure chunk (cz,cy,cx) of lod_ is in the local mirror. Same single-flight
    // contract as MatterCacheFetcher::ensureRegion.
    bool ensureChunk(int cz, int cy, int cx, std::size_t& downloadedBytes);

    std::shared_ptr<MatterRemoteSource> remote_;
    std::shared_ptr<MatterArchive> archive_;
    int lod_ = 0;

    enum class RegionState { InFlight, Present, Absent };
    std::mutex regMu_;
    std::condition_variable regCv_;
    std::unordered_map<std::uint64_t, RegionState> regions_;
};

}  // namespace vc::render
