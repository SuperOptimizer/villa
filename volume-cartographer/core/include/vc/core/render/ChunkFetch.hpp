#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
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
        // Keys pack losslessly into 64 bits (volumes are at most 2^20 voxels per
        // axis and <=8 LODs): 1 spare | 3 level | 20 iz | 20 iy | 20 ix. One
        // mix (splitmix64 finalizer) replaces four chained hash_combine rounds.
        std::uint64_t k = (static_cast<std::uint64_t>(key.level & 7) << 60) |
                          (static_cast<std::uint64_t>(key.iz & 0xFFFFF) << 40) |
                          (static_cast<std::uint64_t>(key.iy & 0xFFFFF) << 20) |
                          static_cast<std::uint64_t>(key.ix & 0xFFFFF);
        k ^= k >> 33;
        k *= 0xFF51AFD7ED558CCDULL;
        k ^= k >> 33;
        k *= 0xC4CEB9FE1A85EC53ULL;
        k ^= k >> 33;
        return static_cast<std::size_t>(k);
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
    int httpStatus = 0;
    std::string message;
    // Bytes actually pulled from the underlying source by THIS call (0 when served
    // from a local cache, e.g. an already-present mca region). Caching fetchers set
    // it so download stats don't count cache hits as network traffic.
    std::size_t downloadedBytes = 0;
};

class IChunkFetcher {
public:
    virtual ~IChunkFetcher() = default;
    virtual ChunkFetchResult fetch(const ChunkKey& key) = 0;
};

} // namespace vc::render
