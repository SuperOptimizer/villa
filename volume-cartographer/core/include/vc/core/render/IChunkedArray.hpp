#pragma once

#include "vc/core/render/ChunkFetch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vc::render {

enum class ChunkStatus {
    MissQueued,
    Missing,
    AllFill,
    Data,
    Error
};

enum class ChunkDtype {
    UInt8,
    UInt16
};

struct ChunkResult {
    ChunkStatus status = ChunkStatus::MissQueued;
    ChunkDtype dtype = ChunkDtype::UInt8;
    std::array<int, 3> shape{};
    std::shared_ptr<const std::vector<std::byte>> bytes;
    std::string error;
};

// Options for Volume::createChunkCache. The render path (McVolumeArray) consumes
// only decodedByteCapacity (the resident decoded-block budget, in bytes).
struct DecodedCacheOptions {
    std::size_t decodedByteCapacity = 512ULL << 20;
};

class IChunkedArray {
public:
    using ChunkReadyCallbackId = std::uint64_t;

    struct LevelTransform {
        std::array<double, 3> scaleFromLevel0{1.0, 1.0, 1.0};
        std::array<double, 3> offsetFromLevel0{0.0, 0.0, 0.0};
    };

    using ChunkReadyCallback = std::function<void()>;

    virtual ~IChunkedArray() = default;
    virtual int numLevels() const = 0;
    virtual std::array<int, 3> shape(int level) const = 0;
    virtual std::array<int, 3> chunkShape(int level) const = 0;
    virtual ChunkDtype dtype() const = 0;
    virtual double fillValue() const = 0;
    virtual LevelTransform levelTransform(int level) const = 0;

    // Interactive viewers must use tryGetChunk() only. A miss queues I/O and
    // returns immediately; chunk-ready listeners are responsible for scheduling
    // a later repaint on the UI thread.
    virtual ChunkResult tryGetChunk(int level, int iz, int iy, int ix) = 0;

    // Blocking access is for CLI, batch, optimization, and prefetch callers.
    // Viewer rendering paths must not call this on the Qt/main thread.
    virtual ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) = 0;
    virtual void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) = 0;

    virtual ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) = 0;
    virtual void removeChunkReadyListener(ChunkReadyCallbackId id) = 0;

    // Status/metrics shared by the GUI status bar (resident RAM, on-disk cache,
    // in-flight + download rate). Backends fill what they track; 0 otherwise.
    struct Stats {
        std::size_t decodedBytes = 0;
        std::size_t decodedByteCapacity = 0;
        std::size_t persistentCacheBytes = 0;
        std::size_t remoteFetchesInFlight = 0;   // real pipeline depth (status bar)
        std::size_t workPending = 0;             // + undrained fill misses (render gate)
        double remoteDownloadBytesPerSecond = 0.0;
        // Per-stage pipeline depths (256^3 regions; sum == remoteFetchesInFlight):
        // download queue / on the wire / waiting for a decode worker / actively in
        // decode->re-encode->append, plus the compressed RAM the decode backlog
        // holds. The archive append is synchronous (no queue).
        std::size_t downloadQueued = 0;
        std::size_t downloading = 0;
        std::size_t decodeQueued = 0;
        std::size_t encoding = 0;
        std::size_t decodeStagingBytes = 0;
    };
    virtual Stats stats() const = 0;

    // Monotonic data generation: changes whenever a render could produce different
    // pixels (cache fill / chunk arrival). 0 = unknown (callers must not skip).
    // Same nonzero gen + unchanged camera across ticks => provably identical frame.
    virtual std::uint64_t dataGeneration() const { return 0; }

    // Viewport-local variant: the max data generation over the given working set
    // (16^3-block keys, e.g. from the predictive prefetch). Lets a viewer skip a
    // streaming re-render when data landed only OUTSIDE its viewport. Default
    // falls back to the volume-global generation.
    virtual std::uint64_t dataGenerationFor(const std::vector<ChunkKey>& /*keys*/) const
    {
        return dataGeneration();
    }

    // Max data generation over an inclusive 256^3-REGION box at `level`, plus the
    // covering ancestor regions at every coarser level (a tile rendered coarse
    // sharpens when its fine region lands; a tile waiting on coarse data changes
    // when the ancestor lands). Used for per-tile partial re-rendering. Default =
    // volume-global gen (any change dirties every tile -- always safe).
    virtual std::uint64_t dataGenerationForBox(int /*level*/, int /*rz0*/, int /*rz1*/,
                                               int /*ry0*/, int /*ry1*/,
                                               int /*rx0*/, int /*rx1*/) const
    {
        return dataGeneration();
    }

    // The LOD the renderer will pick for `voxPerPixel` (region-box gens are keyed
    // by level). Default 0.
    virtual int pickLevel(float /*voxPerPixel*/) const { return 0; }

    // Background prefetch (never the interactive path). shardBatch reports the
    // source shard enclosing a 16^3-block key (geometry only); prefetchShardBlocking
    // pulls + transcodes the whole enclosing shard. Default no-ops for backends
    // without a shard concept.
    virtual FetchBatch shardBatch(int level, int iz, int iy, int ix) const
    {
        return FetchBatch{{iz, iy, ix}, 1};
    }
    virtual void prefetchShardBlocking(int /*level*/, int /*iz*/, int /*iy*/, int /*ix*/) {}

    // Hint: a new interactive view request begins (raise fetch priority for the
    // new frame's chunks). No-op for backends with their own scheduling.
    virtual void beginViewRequest() {}

    // Resize the decoded (RAM) cache budget at runtime, in bytes. No-op for
    // backends with a fixed cache. Backends may discard resident data.
    virtual void setDecodedByteCapacity(std::size_t /*bytes*/) {}

    // ---- render game-loop (one global clock; see ViewerManager) -------------
    // freeze() brackets a frame: the cache becomes immutable for the duration so
    // render reads are consistent (and, for mc backends, LOCK-FREE). thaw()
    // reopens the write phase between frames — newly transcoded regions become
    // visible and the pin epoch advances. The global tick does: thaw -> (regions
    // land / predictive prefetch) -> freeze -> render all dirty viewers -> flip.
    // No-ops for backends without a phase model (they're always consistent).
    virtual void freeze() {}
    virtual void thaw() {}
};

} // namespace vc::render
