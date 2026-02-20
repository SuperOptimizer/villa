#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <optional>
#include <unordered_set>
#include <vector>

#include "ChunkData.hpp"
#include "ChunkKey.hpp"
#include "ChunkSource.hpp"
#include "DiskStore.hpp"
#include "IOPool.hpp"

namespace vc::cache {

// Multi-tiered chunk cache with four storage levels:
//
//   HOT   — decompressed in RAM, ready to sample (ChunkDataPtr)
//   WARM  — compressed in RAM, fast to decompress (CompressedChunk)
//   COLD  — on local disk, persistent between runs (DiskStore)
//   ICE   — remote source, S3/HTTP/filesystem (ChunkSource)
//
// Promotion path:  ice → cold → warm → hot
// Eviction:        hot entries removed (warm still has compressed copy);
//                  warm entries removed (cold still has on-disk copy).
//
// The coarsest pyramid level is pinned in the hot tier and never evicted.
// This guarantees getBestAvailable() always returns data.
//
// Thread safety:
//   - get() / getBestAvailable(): safe from any thread
//   - getBlocking(): safe from any thread (may block)
//   - prefetch(): safe from any thread (non-blocking)
//   - Hot tier: shared_mutex (many readers, occasional writer)
//   - Warm tier: mutex (less concurrent access)
//   - Per-key lock pool prevents duplicate loads
class TieredChunkCache {
public:
    struct Config {
        size_t hotMaxBytes = 10ULL << 30;    // 10 GB
        size_t warmMaxBytes = 2ULL << 30;    // 2 GB
        std::string volumeId;                // for disk store keying
        int ioThreads = 32;
    };

    // source: where to fetch chunks (filesystem, HTTP, etc.)
    // decompress: converts compressed bytes → ChunkData
    // diskStore: shared disk cache (may be nullptr to disable cold tier)
    TieredChunkCache(
        Config config,
        std::unique_ptr<ChunkSource> source,
        DecompressFn decompress,
        std::shared_ptr<DiskStore> diskStore = nullptr);

    ~TieredChunkCache();

    TieredChunkCache(const TieredChunkCache&) = delete;
    TieredChunkCache& operator=(const TieredChunkCache&) = delete;

    // --- Non-blocking reads ---

    // Returns immediately. Returns nullptr on miss (hot and warm both miss).
    // Does NOT trigger a fetch. Use prefetch() to schedule background loading.
    ChunkDataPtr get(const ChunkKey& key);

    // Returns the best available data, searching from the requested level
    // up to the coarsest. Returns {data, actualLevel}.
    // The coarsest level is always pinned hot, so this never returns nullptr
    // (after pinLevel() has been called).
    std::pair<ChunkDataPtr, int> getBestAvailable(const ChunkKey& key);

    // --- Blocking reads ---

    // Blocks until the chunk is available (loads from cold/ice if needed).
    ChunkDataPtr getBlocking(const ChunkKey& key);

    // --- Async prefetch ---

    // Schedule background fetch from cold/ice tier. No-op if already cached.
    void prefetch(const ChunkKey& key);

    // Prefetch all chunks needed for a region at a given level.
    void prefetchRegion(int level, int iz0, int iy0, int ix0,
                        int iz1, int iy1, int ix1);

    // Cancel all pending (not in-flight) prefetch tasks.
    void cancelPendingPrefetch();

    // --- Cache management ---

    // Pin all chunks at a pyramid level: load them into hot and mark non-evictable.
    // gridDims: number of chunks along {z, y, x} at this level.
    // blocking: if true, waits for all chunks to load.
    void pinLevel(int level, const std::array<int, 3>& gridDims,
                  bool blocking = true);

    // Clear all tiers (hot, warm). Does not touch cold/ice.
    void clearMemory();

    // Clear everything including disk cache for this volume.
    void clearAll();

    // Number of pyramid levels in the source.
    int numLevels() const;

    // Chunk shape at a given level, in {z, y, x} order.
    std::array<int, 3> chunkShape(int level) const;

    // Full dataset shape at a given level, in {z, y, x} order.
    std::array<int, 3> levelShape(int level) const;

    // Check if a chunk is negative-cached (known to not exist on source).
    // In zarr format, missing chunks contain the fill value (zeros),
    // so callers should treat negative-cached chunks as available.
    bool isNegativeCached(const ChunkKey& key) const;

    // --- Notifications ---

    // Called from IOPool worker thread when a chunk becomes available.
    // Caller should bounce to main thread (e.g., via QMetaObject::invokeMethod).
    using ChunkReadyCallback = std::function<void(const ChunkKey&)>;
    void setChunkReadyCallback(ChunkReadyCallback cb);

    // --- Stats ---

    struct Stats {
        uint64_t hotHits = 0;
        uint64_t warmHits = 0;
        uint64_t coldHits = 0;
        uint64_t iceFetches = 0;
        uint64_t misses = 0;       // non-blocking misses (all tiers empty)
        uint64_t hotEvictions = 0;
        uint64_t warmEvictions = 0;
        size_t hotBytes = 0;
        size_t warmBytes = 0;
        size_t ioPending = 0;   // pending + in-flight IO tasks
    };

    Stats stats() const;
    void resetStats();

private:
    // --- Hot tier ---
    struct HotEntry {
        ChunkDataPtr data;
        size_t bytes;
        uint64_t generation;
        bool pinned;
    };

    ChunkDataPtr hotGet(const ChunkKey& key);
    void hotPut(const ChunkKey& key, ChunkDataPtr data, bool pinned = false);
    void hotEvictIfNeeded();

    mutable std::shared_mutex hotMutex_;
    std::unordered_map<ChunkKey, HotEntry, ChunkKeyHash> hot_;
    std::atomic<size_t> hotBytes_{0};
    std::atomic<uint64_t> hotGen_{0};
    std::mutex hotEvictMutex_;

    // --- Warm tier ---
    struct WarmEntry {
        CompressedChunk chunk;
        uint64_t generation;
    };

    std::optional<CompressedChunk> warmGet(const ChunkKey& key);
    void warmPut(const ChunkKey& key, std::vector<uint8_t> compressed,
                 size_t decompressedSize);
    void warmEvictIfNeeded();

    mutable std::mutex warmMutex_;
    std::unordered_map<ChunkKey, WarmEntry, ChunkKeyHash> warm_;
    size_t warmBytes_ = 0;
    uint64_t warmGen_ = 0;

    // --- Config (must be declared before ioPool_ for initialization order) ---
    Config config_;

    // --- Cold tier ---
    std::shared_ptr<DiskStore> diskStore_;

    // --- Ice tier ---
    std::unique_ptr<ChunkSource> source_;

    // --- Decompression ---
    DecompressFn decompress_;

    // --- I/O pool ---
    IOPool ioPool_;

    // --- Negative cache: chunks confirmed not to exist at the source ---
    // Prevents repeated S3 round-trips for out-of-bounds / sparse chunks.
    // Persisted to disk alongside the cold tier so it survives restarts.
    mutable std::shared_mutex negativeMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> negativeCache_;
    void loadNegativeCache();
    void saveNegativeCache() const;

    // --- Keys that should be pinned when they arrive in hot tier ---
    std::mutex pinnedKeysMutex_;
    std::unordered_set<ChunkKey, ChunkKeyHash> pendingPinKeys_;

    // --- Per-key lock pool (prevents duplicate loads) ---
    static constexpr int kLockPoolSize = 64;
    std::mutex lockPool_[kLockPoolSize];
    size_t lockIndex(const ChunkKey& key) const
    {
        return ChunkKeyHash()(key) % kLockPoolSize;
    }

    // --- Promotion helpers ---

    // Load from warm → hot. Returns the decompressed data.
    ChunkDataPtr promoteFromWarm(const ChunkKey& key, CompressedChunk warm);

    // Load from cold → warm → hot. Returns the decompressed data.
    ChunkDataPtr promoteFromCold(const ChunkKey& key);

    // Fetch from ice → cold → warm → hot. Returns the decompressed data.
    ChunkDataPtr promoteFromIce(const ChunkKey& key);

    // Full promotion chain (checks each tier in order).
    ChunkDataPtr loadFull(const ChunkKey& key);

    // IO completion handler (called from worker thread).
    void onIOComplete(const ChunkKey& key, std::vector<uint8_t>&& compressed);

    ChunkReadyCallback chunkReadyCb_;

    // --- Stats ---
    mutable std::atomic<uint64_t> statHotHits_{0};
    mutable std::atomic<uint64_t> statWarmHits_{0};
    std::atomic<uint64_t> statColdHits_{0};
    std::atomic<uint64_t> statIceFetches_{0};
    mutable std::atomic<uint64_t> statMisses_{0};
    std::atomic<uint64_t> statHotEvictions_{0};
    std::atomic<uint64_t> statWarmEvictions_{0};
};

}  // namespace vc::cache
