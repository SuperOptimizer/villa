#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace vc::render {

struct ChunkKey {
    int level = 0;
    int iz = 0;
    int iy = 0;
    int ix = 0;

    friend bool operator==(const ChunkKey&, const ChunkKey&) = default;
};

struct ChunkKeyHash {
    std::size_t operator()(const ChunkKey& key) const noexcept
    {
        // Hot path: probed on every chunk lookup in the render read path. Pack the
        // four small fields (level < 64, coords < 2^21 for any real volume) into a
        // single 64-bit word and apply one finalizer mix, instead of four
        // sequential hash-combine rounds. This is the dominant cost of entries_
        // .find; a single multiply-xorshift is far cheaper and spreads bits well.
        std::uint64_t h = (std::uint64_t(std::uint32_t(key.level)) << 60)
                        ^ (std::uint64_t(std::uint32_t(key.iz)) << 40)
                        ^ (std::uint64_t(std::uint32_t(key.iy)) << 20)
                        ^  std::uint64_t(std::uint32_t(key.ix));
        // splitmix64 finalizer
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
