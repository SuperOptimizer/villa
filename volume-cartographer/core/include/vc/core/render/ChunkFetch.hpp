#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vc::render {

// 8-byte packed chunk key: 4 bitfields in one u64 -- lod (3 bits, 8 LODs), then
// z/y/x chunk coords (20 bits each; volumes are capped at 2^20 voxels/dim). 63
// bits used; the high bit (bit 63) is free and marks the "empty" sentinel (a
// value no real key produces). At 8 bytes the hash-map probe array packs 8 keys
// per cache line (vs 4 for the old 4-int 16-byte struct). The fields keep the
// .level / .iz / .iy / .ix names so the ~165 call sites are unchanged. Equality
// reads the whole struct as one u64 (single 64-bit compare).
struct ChunkKey {
    std::uint64_t ix    : 20 = 0;
    std::uint64_t iy    : 20 = 0;
    std::uint64_t iz    : 20 = 0;
    std::uint64_t level : 3  = 0;
    std::uint64_t empty : 1  = 0;   // sentinel flag; 0 for all real keys

    ChunkKey() = default;
    ChunkKey(int level_, int iz_, int iy_, int ix_)
        : ix(std::uint32_t(ix_)), iy(std::uint32_t(iy_)),
          iz(std::uint32_t(iz_)), level(std::uint32_t(level_)) {}

    std::uint64_t word() const
    {
        std::uint64_t w;
        __builtin_memcpy(&w, this, sizeof(w));
        return w;
    }
    friend bool operator==(const ChunkKey& a, const ChunkKey& b) { return a.word() == b.word(); }
    friend bool operator!=(const ChunkKey& a, const ChunkKey& b) { return a.word() != b.word(); }
};

struct ChunkKeyHash {
    std::size_t operator()(const ChunkKey& key) const noexcept
    {
        // The key is already a packed 64-bit word (lod|z|y|x); just run a
        // splitmix64 finalizer over it. This is the dominant cost of entries_/
        // resident find -- one multiply-xorshift, spreads bits well.
        std::uint64_t h = key.word();
        h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
        h ^= h >> 27; h *= 0x94d049bb133111ebULL;
        h ^= h >> 31;
        return std::size_t(h);
    }
};

enum class ChunkFetchStatus {
    Found,
    Missing,
    HttpError,
    IoError,
    DecodeError
};

struct ChunkFetchResult {
    ChunkFetchStatus status = ChunkFetchStatus::Missing;
    std::vector<std::byte> bytes;
    std::vector<std::byte> persistentBytes;
    bool hasPersistentBytes = false;
    int httpStatus = 0;
    std::string message;
};

class IChunkFetcher {
public:
    virtual ~IChunkFetcher() = default;
    virtual ChunkFetchResult fetch(const ChunkKey& key) = 0;

    virtual std::string persistentCacheExtension(const ChunkKey&) const
    {
        return ".bin";
    }

    virtual ChunkFetchResult decodePersistentBytes(
        const ChunkKey&,
        std::vector<std::byte> bytes) const
    {
        ChunkFetchResult result;
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::move(bytes);
        return result;
    }

    // On-disk bytes this fetcher's persistent cache occupies, if it can report
    // that cheaply (O(1)) from its own bookkeeping. Returns nullopt when the
    // fetcher has no such knowledge, in which case ChunkCache falls back to
    // walking the persistent-cache directory. The single-file vca streaming
    // cache overrides this (resident-block count * block size) so the HUD size
    // label never triggers a recursive_directory_iterator / stat of a 503 GB
    // sparse cache file on the per-frame stats() path.
    virtual std::optional<std::size_t> persistentCacheBytes() const
    {
        return std::nullopt;
    }
};

} // namespace vc::render
