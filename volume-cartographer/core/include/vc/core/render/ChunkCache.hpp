#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/render/OpenAddrMap.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace vc::render {

class ChunkCache final : public IChunkedArray {
public:
    struct LevelInfo {
        std::array<int, 3> shape{};
        std::array<int, 3> chunkShape{};
        LevelTransform transform{};
        // Precomputed chunk-grid extent = ceil(shape/chunkShape) per axis. Filled
        // by the ctor so isValidKey is 3 compares, not 3 integer divisions per
        // call (it runs once per chunk-change in the render read path).
        std::array<int, 3> chunkGrid{};
    };

    struct Options {
        std::size_t decodedByteCapacity = 512ULL * 1024ULL * 1024ULL;
        // Bound resolved non-data entries (all-fill/missing/error). These
        // entries are small individually, but sparse remote volumes can touch
        // unbounded empty chunk grids during exploration.
        std::size_t metadataEntryCapacity = 1ULL << 20;
        // Number of process-wide chunk I/O workers used by this cache. The
        // pool is shared by caches with the same worker count and is not
        // destroyed when a viewer is closed.
        std::size_t maxConcurrentReads = 16;
        bool detectAllFillChunks = true;
        std::optional<std::filesystem::path> persistentCachePath;
    };

    struct Stats {
        std::size_t decodedBytes = 0;
        std::size_t decodedByteCapacity = 0;
        std::size_t persistentCacheBytes = 0;
        std::size_t remoteFetchesInFlight = 0;
        double remoteDownloadBytesPerSecond = 0.0;
    };

    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype);
    ChunkCache(std::vector<LevelInfo> levels,
               std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
               double fillValue,
               ChunkDtype dtype,
               Options options);
    ~ChunkCache() override;

    int numLevels() const override;
    std::array<int, 3> shape(int level) const override;
    std::array<int, 3> chunkShape(int level) const override;
    ChunkDtype dtype() const override;
    double fillValue() const override;
    LevelTransform levelTransform(int level) const override;

    ChunkResult tryGetChunk(int level, int iz, int iy, int ix) override;
    ChunkResult getChunkBlocking(int level, int iz, int iy, int ix) override;
    void prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset = 0) override;

    ChunkReadyCallbackId addChunkReadyListener(ChunkReadyCallback cb) override;
    void removeChunkReadyListener(ChunkReadyCallbackId id) override;

    Stats stats() const;
    void invalidate();
    void beginViewRequest();

    // --- tick/settle phase control --------------------------------------
    // The cache has two phases. During SETTLE (a frame rendering) the cache is
    // frozen: read-only, sampled lock-free via readResident(), and fetch results
    // that arrive are parked in a staging queue instead of mutating entries_.
    // During the TICK (between frames, single-threaded, no render running) the
    // cache is mutable: tick() folds the staging queue into entries_, issues
    // fetches for the chunks the last frame missed, evicts to budget, and
    // refreshes the cached stats. This is the "don't mutate at the wrong time"
    // discipline -- mutation only happens inside tick().
    //
    // freeze()/thaw() bracket a render. tick() does the between-frame work and
    // leaves the cache frozen again for the next render. requestChunks() records
    // the keys a frame wanted but missed, so the next tick fetches them.
    void freeze();
    void thaw();
    void tick();
    void requestChunks(const std::vector<ChunkKey>& keys);

    // Lock-free resident read used by the sampler during SETTLE. Returns the
    // resident status/bytes for a chunk WITHOUT mutating (no fetch queue, no LRU
    // touch, no lock). Marks the entry referenced (NRU) for tick-time eviction.
    // Safe only while frozen() (nothing writes entries_). A miss returns
    // MissQueued but queues nothing -- the caller records it via requestChunks().
    ChunkResult readResident(int level, int iz, int iy, int ix) const override;
    ResidentView readResidentRaw(int level, int iz, int iy, int ix) const override;
    bool frozen() const;

private:
    enum class EntryStatus {
        InFlight,
        Missing,
        AllFill,
        Data,
        Error
    };

    struct Entry {
        EntryStatus status = EntryStatus::InFlight;
        std::shared_ptr<const std::vector<std::byte>> bytes;
        std::string error;
        std::size_t decodedBytes = 0;
        bool persisted = false;
        bool inLru = false;
        int basePriority = 0;
        std::int64_t priority = 0;
        std::uint64_t fetchSerial = 0;
        std::list<ChunkKey>::iterator lruIt;
        // NRU "referenced" bit: set (lock-free, monotonic) by readResident when a
        // frame reads this chunk; cleared by the tick's clock sweep. Replaces the
        // per-access LRU list move. Plain bool (so Entry stays movable for the
        // map); the const lock-free reader sets it via std::atomic_ref. A benign
        // race across viewers (set-to-1 only) at worst keeps a chunk one extra
        // cycle. Reads/writes outside readResident happen under mutex_.
        bool referenced = false;
    };

    // A fetch result parked in staging while the cache is frozen. Folded into
    // entries_ at the next tick (mutation only happens then).
    struct StagedFetch {
        ChunkKey key;
        ChunkFetchResult fetch;
        std::uint64_t generation = 0;
        std::uint64_t fetchSerial = 0;
        bool loadedFromPersistentCache = false;
    };

    struct ChunkKeyEmpty {
        static ChunkKey empty()
        {
            constexpr int k = -2147483647 - 1;  // INT_MIN; real keys have level>=0
            return ChunkKey{k, k, k, k};
        }
    };

    // The IMMUTABLE render-visible resident set. The render reads a
    // shared_ptr<const ResidentMap> lock-free; it never mutates. Holds only what
    // a sample needs: status (Data/AllFill) + the decoded bytes. The tick rebuilds
    // a fresh ResidentMap from the working entries_ and atomically swaps it in --
    // the single coalesced update per cycle. Because the map is const-by-type,
    // mutation-during-render is impossible by construction, not just discipline.
    // The cache OWNS the decoded bytes outright -- no shared_ptr. Survivors are
    // MOVED from the old resident map into the new one each tick (single owner,
    // no refcount, no buffer copy); evicted entries are simply not carried over
    // and their buffers free deterministically when the old map is destroyed.
    // Safe because the render worker that read the old map has already finished
    // before the tick rebuilds (finishRenderOnMainThread runs post-worker).
    struct ResidentEntry {
        ChunkStatus status = ChunkStatus::Missing;          // Data | AllFill
        std::unique_ptr<std::vector<std::byte>> bytes;      // valid iff Data; sole owner
    };
    using ResidentMap = OpenAddrMap<ChunkKey, ResidentEntry, ChunkKeyHash, ChunkKeyEmpty>;

    struct State {
        State(std::vector<LevelInfo> levels,
              std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
              double fillValue,
              ChunkDtype dtype,
              Options options)
            : levels_(std::move(levels))
            , fetchers_(std::move(fetchers))
            , fillValue_(fillValue)
            , dtype_(dtype)
            , options_(std::move(options))
        {}

        std::vector<LevelInfo> levels_;
        std::vector<std::shared_ptr<IChunkFetcher>> fetchers_;
        double fillValue_ = 0.0;
        ChunkDtype dtype_ = ChunkDtype::UInt8;
        Options options_;

        mutable std::mutex mutex_;
        std::condition_variable cv_;

        // The IMMUTABLE render-visible resident map. The render reads it lock-free
        // via a raw const* (see residentView()); it is replaced wholesale by the
        // tick (build next, swap). Owned outright -- no shared_ptr. residentSwapMutex_
        // guards only the pointer swap, held for the length of one pointer store.
        std::unique_ptr<const ResidentMap> resident_ = std::make_unique<const ResidentMap>();
        mutable std::mutex residentSwapMutex_;

        // entries_ is the tick-only WORKING store: it tracks InFlight fetches,
        // fetchSerial, eviction metadata, etc. The render never touches it. The
        // tick projects its resolved Data/AllFill entries into a new resident_.
        OpenAddrMap<ChunkKey, Entry, ChunkKeyHash, ChunkKeyEmpty> entries_;
        std::list<ChunkKey> lru_;
        std::size_t decodedBytes_ = 0;
        std::uint64_t generation_ = 0;
        std::int64_t viewEpoch_ = 1;
        std::uint64_t nextFetchSerial_ = 1;
        ChunkReadyCallbackId nextCallbackId_ = 1;
        std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> callbacks_;
        std::size_t remoteFetchesInFlight_ = 0;
        std::deque<std::pair<std::chrono::steady_clock::time_point, std::size_t>> remoteDownloadHistory_;
        std::chrono::steady_clock::time_point lastPersistentCacheSizeScan_{};
        std::size_t cachedPersistentCacheBytes_ = 0;

        // --- tick/settle phase ------------------------------------------
        // frozen_: true during SETTLE (render). While true, entries_ must not be
        // mutated; fetch completions park in stagedFetches_ instead. assertMutable
        // (in the .cpp) checks this on every mutation path so a discipline
        // violation fails loudly in debug instead of racing.
        std::atomic<bool> frozen_{false};
        std::mutex stagingMutex_;                  // guards stagedFetches_ only
        std::vector<StagedFetch> stagedFetches_;   // fetch results awaiting a tick
        std::mutex requestMutex_;                  // guards requestedThisCycle_
        std::vector<ChunkKey> requestedThisCycle_; // chunks a frame missed -> fetch at tick
    };

    static ChunkResult resultFromEntryLocked(State& state, const ChunkKey& key, Entry& entry);
    static void queueFetchLocked(const std::shared_ptr<State>& state,
                                 const ChunkKey& key,
                                 std::uint64_t generation,
                                 int priorityOffset);
    static void fetchAndStore(const std::shared_ptr<State>& state,
                              ChunkKey key,
                              std::uint64_t generation,
                              std::uint64_t fetchSerial);
    static void storeFetchResultLocked(const std::shared_ptr<State>& state,
                                       const ChunkKey& key,
                                       ChunkFetchResult fetch,
                                       bool loadedFromPersistentCache);
    static std::optional<std::vector<std::byte>> readPersistent(const State& state, const ChunkKey& key);
    static bool readPersistentEmpty(const State& state, const ChunkKey& key);
    static bool queuePersistentWrite(const std::shared_ptr<State>& state,
                                     const ChunkKey& key,
                                     std::shared_ptr<const std::vector<std::byte>> bytes);
    static bool queuePersistentEmptyWrite(const std::shared_ptr<State>& state,
                                          const ChunkKey& key);
    static void writePersistent(const State& state, const ChunkKey& key, const std::vector<std::byte>& bytes);
    static void writePersistentEmpty(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentPath(const State& state, const ChunkKey& key);
    static std::filesystem::path persistentEmptyPath(const State& state, const ChunkKey& key);
    static std::size_t persistentCacheBytes(const std::optional<std::filesystem::path>& path);
    static void pruneDownloadHistoryLocked(State& state, std::chrono::steady_clock::time_point now);
    static void touchLocked(State& state, const ChunkKey& key, Entry& entry);
    static void enforceCapacityLocked(const std::shared_ptr<State>& state);
    // NRU (clock) eviction run at the tick: evict Data entries whose referenced
    // bit is clear until under budget, clearing the bit on survivors (second
    // chance). Replaces per-access LRU touches. Mutates entries_/decodedBytes_,
    // so it must run inside tick() (cache not frozen).
    static void nruEvictLocked(const std::shared_ptr<State>& state);
    // Build a fresh immutable resident map from the resolved working entries and
    // swap it in -- the single coalesced render-visible update per tick.
    static void rebuildResidentLocked(const std::shared_ptr<State>& state);
    static bool isValidKey(const State& state, const ChunkKey& key);
    static bool isAllFill(const State& state, const std::vector<std::byte>& bytes);
    static std::size_t dtypeSize(ChunkDtype dtype);
    static std::size_t expectedChunkBytes(const State& state, const ChunkKey& key);
    static void notifyListeners(const std::shared_ptr<State>& state);
    static void drainStagingLocked(const std::shared_ptr<State>& state);
    static void waitForResolvedLocked(const std::shared_ptr<State>& state,
                                      std::unique_lock<std::mutex>& lock,
                                      const ChunkKey& key);

    std::shared_ptr<State> state_;
};

} // namespace vc::render
