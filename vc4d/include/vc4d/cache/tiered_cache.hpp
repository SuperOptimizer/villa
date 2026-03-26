#pragma once
/// vc4d::TieredCache — Multi-tier chunk cache.
///
/// Same conceptual design as vc3d's TieredChunkCache but cleaner:
///   • No raw mutex soup — uses a simple shared_mutex for readers/writers.
///   • No per-key lock pool — Qt's task system handles dedup naturally.
///   • LRU eviction built into each tier instead of separate utility.
///   • Tiers: Hot (decompressed RAM) → Warm (compressed RAM) →
///            Cold (disk) → Remote (HTTP).
///
/// Each tier is a simple concept: get(key) -> optional<bytes>,
/// put(key, bytes), evict_to_budget().

#include "vc4d/core/zarr.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace vc4d {

// ---------------------------------------------------------------------------
// ChunkKey — identifies a chunk globally (dataset + coordinate)
// ---------------------------------------------------------------------------
struct ChunkKey {
    int scale_level;
    ChunkCoord coord;

    bool operator==(const ChunkKey&) const = default;
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const {
        size_t h = ChunkCoordHash{}(k.coord);
        h ^= std::hash<int>{}(k.scale_level) * 2654435761;
        return h;
    }
};

// ---------------------------------------------------------------------------
// TieredCache
// ---------------------------------------------------------------------------
class TieredCache {
public:
    struct Config {
        size_t hot_budget_bytes  = 8ULL << 30;   // 8 GB decompressed
        size_t warm_budget_bytes = 2ULL << 30;   // 2 GB compressed
        size_t cold_budget_bytes = 100ULL << 30;  // 100 GB on disk
        std::filesystem::path disk_cache_dir;     // empty = no disk tier
    };

    explicit TieredCache(Config config);
    ~TieredCache();

    // Get a decompressed chunk. Returns empty span if not available.
    // If the chunk is in a lower tier, it's promoted to hot.
    [[nodiscard]] std::span<const uint8_t> get(ChunkKey key);

    // Put a decompressed chunk into the hot tier.
    void put(ChunkKey key, std::vector<uint8_t> data);

    // Put compressed bytes into the warm tier (for prefetch paths).
    void put_compressed(ChunkKey key, std::vector<uint8_t> compressed);

    // Check if a chunk is available (any tier).
    [[nodiscard]] bool contains(ChunkKey key) const;

    // Pin a chunk so it's never evicted from hot tier.
    void pin(ChunkKey key);

    // Budget management.
    void set_hot_budget(size_t bytes);
    [[nodiscard]] size_t hot_usage() const;
    [[nodiscard]] size_t warm_usage() const;

    // Evict all data.
    void clear();

private:
    void evict_hot_to_budget();
    void evict_warm_to_budget();

    struct Entry {
        std::vector<uint8_t> data;
        bool pinned = false;
        uint64_t last_access = 0;
    };

    Config config_;
    mutable std::shared_mutex mutex_;

    std::unordered_map<ChunkKey, Entry, ChunkKeyHash> hot_;
    std::unordered_map<ChunkKey, std::vector<uint8_t>, ChunkKeyHash> warm_;
    size_t hot_bytes_{};
    size_t warm_bytes_{};
    uint64_t access_counter_{};
};

} // namespace vc4d
