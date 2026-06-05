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
        std::size_t seed = 0;
        auto combine = [&seed](int value) {
            seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        combine(key.level);
        combine(key.iz);
        combine(key.iy);
        combine(key.ix);
        return seed;
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
