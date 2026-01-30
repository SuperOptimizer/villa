#include "vc/core/csvs/CsvsDataset.hpp"

#include <cstring>
#include <stdexcept>

namespace volcart::csvs {

void CsvsDataset::VolDeleter::operator()(csvs_volume* v) const
{
    if (v) {
        csvs_close(v);
        delete v;
    }
}

CsvsDataset::CsvsDataset(const std::filesystem::path& path)
{
    auto* v = new csvs_volume{};
    if (csvs_open(v, path.c_str()) != 0) {
        delete v;
        throw std::runtime_error("Failed to open CSVS volume: " + path.string());
    }
    vol_.reset(v);
    shape_ = std::array<size_t,3>{v->shape[0], v->shape[1], v->shape[2]};
    chunkSize_ = v->chunk_size;
    shardSize_ = v->shard_size;
    dtype_ = zarr::Dtype::UInt8;
}

CsvsDataset::CsvsDataset(const std::filesystem::path& path,
                           const std::array<size_t, 3>& shape,
                           uint32_t chunkSize,
                           uint32_t shardSize,
                           zarr::Dtype dtype,
                           int codecLevel)
{
    auto* v = new csvs_volume{};
    size_t s[3] = {shape[0], shape[1], shape[2]};
    if (dtype != zarr::Dtype::UInt8) {
        delete v;
        throw std::runtime_error("CSVS only supports UInt8 dtype");
    }
    if (csvs_create(v, path.c_str(), s, chunkSize, shardSize,
                    CSVS_CODEC_ZSTD, codecLevel) != 0) {
        delete v;
        throw std::runtime_error("Failed to create CSVS volume: " + path.string());
    }
    vol_.reset(v);
    shape_ = shape;
    chunkSize_ = chunkSize;
    shardSize_ = shardSize;
    dtype_ = dtype;
}

CsvsDataset::~CsvsDataset() = default;

bool CsvsDataset::readChunk(size_t cz, size_t cy, size_t cx, void* buffer) const
{
    // csvs shard cache is not thread-safe (mmap eviction races), serialize reads
    std::lock_guard<std::mutex> lock(readMutex_);
    return csvs_read_chunk(vol_.get(), cz, cy, cx, buffer) == 0;
}

int CsvsDataset::writeChunk(size_t cz, size_t cy, size_t cx,
                              const void* buffer, size_t rawSize)
{
    return csvs_write_chunk(vol_.get(), cz, cy, cx, buffer, rawSize);
}

int CsvsDataset::writeShard(size_t sz, size_t sy, size_t sx,
                             const void* chunks, const uint8_t* mask,
                             size_t rawChunkSize)
{
    return csvs_write_shard(vol_.get(), sz, sy, sx, chunks, mask, rawChunkSize);
}

}  // namespace volcart::csvs
