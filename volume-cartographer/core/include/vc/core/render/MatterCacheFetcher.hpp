#pragma once

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

#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/MatterArchive.hpp"

namespace vc::render {

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

    static constexpr int kMca = MatterArchive::kChunk;   // 256
    static constexpr int kBlk = MatterArchive::kBlock;    // 16

private:
    // ensure the 256^3 region (regCz,regCy,regCx) is present in the .mca, encoding it from
    // the source on a miss. Returns false if the source had no data for the region.
    bool ensureRegion(int regCz, int regCy, int regCx);

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

}  // namespace vc::render
