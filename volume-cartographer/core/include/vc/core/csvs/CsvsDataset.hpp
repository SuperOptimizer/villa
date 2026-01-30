#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "vc/core/types/IChunkSource.hpp"
#include "vc/core/zarr/Dtype.hpp"
#include "../../../../../../csvs/csvs.h"

namespace volcart::csvs {

/// C++ RAII wrapper around the C csvs library
class CsvsDataset : public IChunkSource {
public:
    /// Open existing CSVS volume
    explicit CsvsDataset(const std::filesystem::path& path);

    /// Create new CSVS volume
    CsvsDataset(const std::filesystem::path& path,
                const std::array<size_t, 3>& shape,
                uint32_t chunkSize,
                uint32_t shardSize,
                zarr::Dtype dtype,
                int codecLevel = 1);

    ~CsvsDataset();

    CsvsDataset(const CsvsDataset&) = delete;
    CsvsDataset& operator=(const CsvsDataset&) = delete;

    // Accessors
    const std::array<size_t, 3>& shape() const noexcept { return shape_; }
    uint32_t chunkSize() const noexcept { return chunkSize_; }
    uint32_t shardSize() const noexcept { return shardSize_; }
    zarr::Dtype dtype() const noexcept { return dtype_; }

    // Chunk I/O (chunk grid coordinates)
    bool readChunk(size_t cz, size_t cy, size_t cx, void* buffer) const;
    int writeChunk(size_t cz, size_t cy, size_t cx,
                   const void* buffer, size_t rawSize);

    /// Write an entire shard at once (all chunks contiguous, C order)
    int writeShard(size_t sz, size_t sy, size_t sx,
                   const void* chunks, const uint8_t* mask,
                   size_t rawChunkSize);

    // IChunkSource interface
    std::array<size_t, 3> volShape() const override { return shape_; }
    std::array<size_t, 3> volChunkShape() const override {
        return {chunkSize_, chunkSize_, chunkSize_};
    }
    zarr::Dtype volDtype() const override { return dtype_; }
    size_t volChunkElements() const override {
        return size_t(chunkSize_) * chunkSize_ * chunkSize_;
    }
    bool volReadChunk(size_t cz, size_t cy, size_t cx, void* buf) const override {
        return readChunk(cz, cy, cx, buf);
    }

    /// Access underlying C struct (for advanced use)
    const csvs_volume* handle() const { return vol_.get(); }

private:
    struct VolDeleter { void operator()(csvs_volume* v) const; };
    std::unique_ptr<csvs_volume, VolDeleter> vol_;
    std::array<size_t, 3> shape_;
    uint32_t chunkSize_ = 0;
    uint32_t shardSize_ = 0;
    zarr::Dtype dtype_ = zarr::Dtype::Unknown;
    mutable std::mutex readMutex_;
};

}  // namespace volcart::csvs
