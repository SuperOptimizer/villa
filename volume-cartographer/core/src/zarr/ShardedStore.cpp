#include "vc/core/zarr/ShardedStore.hpp"

#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace volcart::zarr
{

ShardedStore::ShardedStore(
    const std::filesystem::path& basePath,
    const std::vector<std::size_t>& dataShape,
    const std::vector<std::size_t>& chunkShape,
    const std::vector<std::size_t>& shardShape,
    BloscCodec* codec,
    Dtype dtype)
    : basePath_(basePath)
    , dataShape_(dataShape)
    , chunkShape_(chunkShape)
    , shardShape_(shardShape)
    , codec_(codec)
    , dtype_(dtype)
{
    std::size_t ndim = chunkShape_.size();
    chunksPerShard_.resize(ndim);
    totalChunksPerShard_ = 1;
    for (std::size_t i = 0; i < ndim; ++i) {
        chunksPerShard_[i] = shardShape_[i] / chunkShape_[i];
        totalChunksPerShard_ *= chunksPerShard_[i];
    }

    chunkBytes_ = dtypeSize(dtype_);
    for (auto c : chunkShape_) {
        chunkBytes_ *= c;
    }
}

void ShardedStore::chunkToShard(
    const std::vector<std::size_t>& chunkId,
    std::vector<std::size_t>& shardId,
    std::size_t& localIndex) const
{
    std::size_t ndim = chunkId.size();
    shardId.resize(ndim);

    std::vector<std::size_t> localChunk(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        shardId[i] = chunkId[i] / chunksPerShard_[i];
        localChunk[i] = chunkId[i] % chunksPerShard_[i];
    }

    // C-order linearization
    localIndex = 0;
    for (std::size_t i = 0; i < ndim; ++i) {
        localIndex = localIndex * chunksPerShard_[i] + localChunk[i];
    }
}

std::filesystem::path ShardedStore::shardPath(
    const std::vector<std::size_t>& shardId) const
{
    std::string name;
    for (std::size_t i = 0; i < shardId.size(); ++i) {
        if (i > 0) {
            name += '/';
        }
        name += std::to_string(shardId[i]);
    }
    return basePath_ / "c" / name;
}

std::vector<ShardedStore::ShardIndex> ShardedStore::readShardIndex(
    const std::filesystem::path& path) const
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return {};
    }

    auto fileSize = static_cast<std::size_t>(f.tellg());
    std::size_t indexBytes = totalChunksPerShard_ * 16;
    if (fileSize < indexBytes) {
        return {};
    }

    f.seekg(static_cast<std::streamoff>(fileSize - indexBytes));

    std::vector<ShardIndex> index(totalChunksPerShard_);
    f.read(
        reinterpret_cast<char*>(index.data()),
        static_cast<std::streamsize>(indexBytes));

    return index;
}

bool ShardedStore::readChunk(
    const std::vector<std::size_t>& chunkId, void* buffer)
{
    std::vector<std::size_t> shardId;
    std::size_t localIndex;
    chunkToShard(chunkId, shardId, localIndex);

    auto sp = shardPath(shardId);
    auto index = readShardIndex(sp);
    if (index.empty()) {
        return false;
    }

    auto& entry = index[localIndex];
    if (entry.offset == kMissingSentinel && entry.nbytes == kMissingSentinel) {
        return false;
    }

    std::ifstream f(sp, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    f.seekg(static_cast<std::streamoff>(entry.offset));
    std::vector<std::uint8_t> compressed(static_cast<std::size_t>(entry.nbytes));
    f.read(reinterpret_cast<char*>(compressed.data()),
           static_cast<std::streamsize>(entry.nbytes));

    if (codec_) {
        codec_->decompress(compressed.data(), compressed.size(), buffer, chunkBytes_);
    } else {
        std::memcpy(buffer, compressed.data(), compressed.size());
    }

    return true;
}

void ShardedStore::writeChunk(
    const std::vector<std::size_t>& chunkId,
    const void* buffer,
    std::size_t rawSize)
{
    std::vector<std::size_t> shardId;
    std::size_t localIndex;
    chunkToShard(chunkId, shardId, localIndex);

    // Compress
    std::vector<std::uint8_t> compressed;
    if (codec_) {
        compressed = codec_->compress(buffer, rawSize, dtypeSize(dtype_));
    } else {
        compressed.resize(rawSize);
        std::memcpy(compressed.data(), buffer, rawSize);
    }

    std::string key = shardPath(shardId).string();

    std::lock_guard<std::mutex> lock(writeMutex_);

    auto it = writeBuffers_.find(key);
    if (it == writeBuffers_.end()) {
        ShardWriteBuffer buf;
        buf.compressedChunks.resize(totalChunksPerShard_);
        buf.written.resize(totalChunksPerShard_, false);
        it = writeBuffers_.emplace(key, std::move(buf)).first;
    }

    auto& buf = it->second;
    buf.compressedChunks[localIndex] = std::move(compressed);
    buf.written[localIndex] = true;
    buf.writtenCount++;

    if (buf.writtenCount == totalChunksPerShard_) {
        flushShard(key, buf);
        writeBuffers_.erase(it);
    }
}

void ShardedStore::flushShard(const std::string& key, ShardWriteBuffer& buf)
{
    // Build index and concatenate data
    std::vector<ShardIndex> index(totalChunksPerShard_);
    std::vector<std::uint8_t> data;

    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < totalChunksPerShard_; ++i) {
        if (buf.written[i]) {
            index[i].offset = offset;
            index[i].nbytes = buf.compressedChunks[i].size();
            data.insert(
                data.end(),
                buf.compressedChunks[i].begin(),
                buf.compressedChunks[i].end());
            offset += index[i].nbytes;
        } else {
            index[i].offset = kMissingSentinel;
            index[i].nbytes = kMissingSentinel;
        }
    }

    // Append index
    std::size_t indexBytes = totalChunksPerShard_ * 16;
    data.resize(data.size() + indexBytes);
    std::memcpy(
        data.data() + data.size() - indexBytes,
        index.data(),
        indexBytes);

    // Write file
    std::filesystem::path filePath(key);
    std::filesystem::create_directories(filePath.parent_path());

    auto tmpPath = filePath;
    tmpPath += ".tmp";
    {
        std::ofstream f(tmpPath, std::ios::binary);
        f.write(
            reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    }
    std::filesystem::rename(tmpPath, filePath);
}

void ShardedStore::flush()
{
    std::lock_guard<std::mutex> lock(writeMutex_);
    for (auto& [key, buf] : writeBuffers_) {
        flushShard(key, buf);
    }
    writeBuffers_.clear();
}

bool ShardedStore::chunkExists(const std::vector<std::size_t>& chunkId)
{
    std::vector<std::size_t> shardId;
    std::size_t localIndex;
    chunkToShard(chunkId, shardId, localIndex);

    auto sp = shardPath(shardId);
    auto index = readShardIndex(sp);
    if (index.empty()) {
        return false;
    }

    auto& entry = index[localIndex];
    return !(entry.offset == kMissingSentinel && entry.nbytes == kMissingSentinel);
}

}  // namespace volcart::zarr
