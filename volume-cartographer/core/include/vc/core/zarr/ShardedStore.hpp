#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vc/core/zarr/BloscCodec.hpp"
#include "vc/core/zarr/Dtype.hpp"

namespace volcart::zarr
{

class ShardedStore
{
public:
    ShardedStore(
        const std::filesystem::path& basePath,
        const std::vector<std::size_t>& dataShape,
        const std::vector<std::size_t>& chunkShape,
        const std::vector<std::size_t>& shardShape,
        BloscCodec* codec,
        Dtype dtype);

    bool readChunk(const std::vector<std::size_t>& chunkId, void* buffer);

    void writeChunk(
        const std::vector<std::size_t>& chunkId,
        const void* buffer,
        std::size_t rawSize);

    void flush();

    bool chunkExists(const std::vector<std::size_t>& chunkId);

private:
    void chunkToShard(
        const std::vector<std::size_t>& chunkId,
        std::vector<std::size_t>& shardId,
        std::size_t& localIndex) const;

    std::filesystem::path shardPath(const std::vector<std::size_t>& shardId) const;

    struct ShardIndex {
        std::uint64_t offset;
        std::uint64_t nbytes;
    };
    static constexpr std::uint64_t kMissingSentinel = 0xFFFFFFFFFFFFFFFF;

    std::vector<ShardIndex> readShardIndex(
        const std::filesystem::path& path) const;

    struct ShardWriteBuffer {
        std::vector<std::vector<std::uint8_t>> compressedChunks;
        std::vector<bool> written;
        std::size_t writtenCount = 0;
    };

    void flushShard(const std::string& key, ShardWriteBuffer& buf);

    std::filesystem::path basePath_;
    std::vector<std::size_t> dataShape_;
    std::vector<std::size_t> chunkShape_;
    std::vector<std::size_t> shardShape_;
    std::vector<std::size_t> chunksPerShard_;
    std::size_t totalChunksPerShard_;
    BloscCodec* codec_;
    Dtype dtype_;
    std::size_t chunkBytes_;

    std::unordered_map<std::string, ShardWriteBuffer> writeBuffers_;
    std::mutex writeMutex_;
};

}  // namespace volcart::zarr
