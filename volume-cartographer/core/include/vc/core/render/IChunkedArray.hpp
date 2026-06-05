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

    // tick/settle read path: a pure, side-effect-free resident lookup used by the
    // render while the array is "frozen". Unlike tryGetChunk it queues NO I/O and
    // mutates no shared state (lock-free in the ChunkCache override). A miss
    // returns MissQueued WITHOUT scheduling a fetch -- the caller records the miss
    // and the owner issues the fetch at the next tick. The default implementation
    // forwards to tryGetChunk for arrays that don't implement the tick model.
    virtual ChunkResult readResident(int level, int iz, int iy, int ix) const
    {
        return const_cast<IChunkedArray*>(this)->tryGetChunk(level, iz, iy, ix);
    }

    // Raw resident read for the innermost render loop. Returns status + a RAW
    // pointer to the decoded bytes for Data (no shared_ptr copy -> no atomic
    // refcount per chunk). The pointer is valid only for the current frozen
    // frame: the tick/settle discipline guarantees no eviction runs while a frame
    // reads, so the buffer cannot be freed underneath. Default forwards to
    // readResident for arrays that don't implement the raw path.
    struct ResidentView {
        ChunkStatus status = ChunkStatus::MissQueued;
        const std::vector<std::byte>* bytes = nullptr;  // valid iff status==Data
    };
    virtual ResidentView readResidentRaw(int level, int iz, int iy, int ix) const
    {
        ChunkResult r = readResident(level, iz, iy, ix);
        return ResidentView{r.status, r.bytes ? r.bytes.get() : nullptr};
    }

    // A pin holds the current resident map alive for the lifetime of a sampling
    // pass so the raw byte pointers from lookup() stay valid even if the owner
    // swaps in a new map concurrently (multiple viewers share a cache). A render
    // worker creates ONE pin and does all its raw lookups through it. The default
    // pin just forwards per-call to readResidentRaw (no extra lifetime guarantee,
    // adequate for arrays that don't swap maps).
    class IResidentPin {
    public:
        virtual ~IResidentPin() = default;
        virtual ResidentView lookup(int level, int iz, int iy, int ix) const = 0;
    };
    struct ForwardingPin : IResidentPin {
        const IChunkedArray* arr;
        ResidentView lookup(int level, int iz, int iy, int ix) const override
        {
            return arr->readResidentRaw(level, iz, iy, ix);
        }
    };
    virtual std::unique_ptr<IResidentPin> makeResidentPin() const
    {
        auto p = std::make_unique<ForwardingPin>();
        p->arr = this;
        return p;
    }

    // Blocking access is for CLI, batch, optimization, and prefetch callers.
    // Viewer rendering paths must not call this on the Qt/main thread.
    virtual ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) = 0;
    virtual void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) = 0;

    virtual ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) = 0;
    virtual void removeChunkReadyListener(ChunkReadyCallbackId id) = 0;
};

} // namespace vc::render
