#pragma once

#include "vc/core/render/IChunkedArray.hpp"

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

class MatterArchive;

class ChunkCache final : public IChunkedArray {
public:
    struct LevelInfo {
        std::array<int, 3> shape{};
        std::array<int, 3> chunkShape{};
        LevelTransform transform{};
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
        // The per-volume .mca archive when the mca cache is on. Decoded blocks then
        // live in the archive's mc_cache (the only resident cache) — entries here are
        // status-only and resolved Data blocks are pulled back through the fetcher.
        // Null = legacy mode (non-uint8 volumes): entries retain decoded bytes.
        std::shared_ptr<MatterArchive> archive;
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

    // Pull one chunk through its fetcher ON THE CALLING THREAD, bypassing the
    // shared worker pool entirely — for dedicated prefetch threads that must not
    // occupy interactive fetch slots. Persists via the fetcher (mca); does not
    // touch the entry table.
    void warmChunkBlocking(int level, int iz, int iy, int ix);

    // The shard extent (in chunk units) enclosing this chunk — geometry only.
    FetchBatch shardBatch(int level, int iz, int iy, int ix) const;

    // Download the source shard enclosing this chunk (one parallel GET) and
    // persist every covered region into the cache, ON THE CALLING THREAD. The
    // shard-at-a-time prefetch primitive; bypasses the interactive pool.
    void prefetchShardBlocking(int level, int iz, int iy, int ix);

    Stats stats() const;
    void invalidate();
    void beginViewRequest();

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
        bool inLru = false;
        int basePriority = 0;
        std::int64_t priority = 0;
        std::uint64_t fetchSerial = 0;
        std::list<ChunkKey>::iterator lruIt;
    };

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
        std::unordered_map<ChunkKey, Entry, ChunkKeyHash> entries_;
        std::list<ChunkKey> lru_;
        std::size_t decodedBytes_ = 0;
        std::uint64_t generation_ = 0;
        std::int64_t viewEpoch_ = 1;
        std::uint64_t nextFetchSerial_ = 1;
        ChunkReadyCallbackId nextCallbackId_ = 1;
        std::unordered_map<ChunkReadyCallbackId, ChunkReadyCallback> callbacks_;
        std::size_t remoteFetchesInFlight_ = 0;
        std::deque<std::pair<std::chrono::steady_clock::time_point, std::size_t>> remoteDownloadHistory_;
        std::chrono::steady_clock::time_point lastListenerNotify_{};
    };

    static ChunkResult resultFromEntryLocked(State& state, const ChunkKey& key, Entry& entry);
    static ChunkKey entryKey(const State& state, const ChunkKey& key);
    static ChunkResult refetchDataUnlocked(State& state, const ChunkKey& key);
    // Fetch submission is split: queueFetchLocked records the claim under the state
    // mutex; submitFetches dispatches to the worker pool OUTSIDE it (the pool has its
    // own lock — submitting under the state mutex serializes workers against callers).
    struct PendingFetch {
        std::int64_t priority = 0;
        ChunkKey key;
        std::uint64_t generation = 0;
        std::uint64_t fetchSerial = 0;
    };
    static void queueFetchLocked(const std::shared_ptr<State>& state,
                                 const ChunkKey& key,
                                 std::uint64_t generation,
                                 int priorityOffset,
                                 std::vector<PendingFetch>& out);
    static void submitFetches(const std::shared_ptr<State>& state,
                              const std::vector<PendingFetch>& pending);
    static void fetchAndStore(const std::shared_ptr<State>& state,
                              ChunkKey key,
                              std::uint64_t generation,
                              std::uint64_t fetchSerial);
    static void storeFetchResultLocked(const std::shared_ptr<State>& state,
                                       const ChunkKey& key,
                                       ChunkFetchResult fetch);
    static void pruneDownloadHistoryLocked(State& state, std::chrono::steady_clock::time_point now);
    static void touchLocked(State& state, const ChunkKey& key, Entry& entry);
    static void enforceCapacityLocked(const std::shared_ptr<State>& state);
    static bool isValidKey(const State& state, const ChunkKey& key);
    static bool isAllFill(const State& state, const std::vector<std::byte>& bytes);
    static std::size_t dtypeSize(ChunkDtype dtype);
    static std::size_t expectedChunkBytes(const State& state, const ChunkKey& key);
    static void notifyListeners(const std::shared_ptr<State>& state);
    static void waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key);

    std::shared_ptr<State> state_;
};

} // namespace vc::render
