#pragma once

// MatterArchive — RAII C++ wrapper around the matter-compressor (.mca) appendable,
// persistent, crash-safe archive. Used as VC3D's on-disk render cache: remote chunks
// are fetched at the volume's native chunking, re-encoded into ONE .mca per volume
// (all chunks at all LODs), and served back as decoded voxels.
//
// Storage/encode unit: 256^3 chunk (matter-compressor's contiguous on-disk unit).
// Decode/serve unit:    16^3 block (mc's native decode granularity) -> this is what
//                       the resident chunk cache should key on.
//
// Thread-safety: the underlying mc writer is lock-free for concurrent appends, BUT mc's
// codec quality is process-global state, so this wrapper serializes encode/decode under
// a single quality. Appends from multiple threads are safe; this class guards the
// quality set + writer/reader handle lifecycle with a mutex.

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vc::render {

class MatterArchive {
public:
    static constexpr int kChunk = 256;   // MC_CHUNK
    static constexpr int kBlock = 16;     // MC_BLK
    static constexpr int kBlocksPerAxis = kChunk / kBlock;   // 16

    // Open (or create) a persistent appendable archive at `path` for a volume whose
    // LOD0 edge is `dim0` voxels (will be chunk-aligned up to a multiple of 256),
    // encoded at `quality`. Throws std::runtime_error on failure.
    MatterArchive(std::string path, int dim0, float quality);
    ~MatterArchive();

    MatterArchive(const MatterArchive&) = delete;
    MatterArchive& operator=(const MatterArchive&) = delete;

    // Append one 256^3 chunk of raw u8 voxels at chunk coords (cz,cy,cx) of `lod`.
    // `vox` is 256^3 in (z,y,x) raster order, x fastest. Re-encodes via mc. Thread-safe.
    // Returns true on success. An all-air chunk is a successful no-op.
    bool appendChunkRaw(int lod, int cz, int cy, int cx, const std::uint8_t* vox256);

    // Is a chunk present in the archive (without decoding)?
    bool hasChunk(int lod, int cz, int cy, int cx) const;

    // Decode one 16^3 block (bz,by,bx in [0,16)) of chunk (cz,cy,cx) at `lod` into
    // `dst` (16^3 = 4096 bytes). Missing/air -> zeroed. Thread-safe.
    void decodeBlock(int lod, int cz, int cy, int cx, int bz, int by, int bx,
                     std::uint8_t* dst4096) const;

    // Decode a whole 256^3 chunk into `dst` (256^3 = 16.7M bytes). Missing -> zeroed.
    void decodeChunk(int lod, int cz, int cy, int cx, std::uint8_t* dst) const;

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
