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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace vc::render {

class MatterArchive {
public:
    static constexpr int kChunk = 256;   // MC_CHUNK
    static constexpr int kBlock = 16;     // MC_BLK
    static constexpr int kBlocksPerAxis = kChunk / kBlock;   // 16

    // Open (or create) a persistent appendable archive at `path` for a volume whose
    // LOD0 edge is `dim0` voxels (chunk-aligned up to a multiple of 256), encoded at
    // `quality`, with an mc_cache resident-block budget of `cacheBytes`. A stale
    // archive (older format version / different dim) is deleted and recreated.
    // Throws std::runtime_error on failure.
    MatterArchive(std::string path, int dim0, float quality, std::size_t cacheBytes);
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

    int   dim0() const { return dim0_; }
    float quality() const { return quality_; }
    const std::string& path() const { return path_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string path_;
    int   dim0_ = 0;
    float quality_ = 0.f;
};

}  // namespace vc::render
