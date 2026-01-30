#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/zarr/BloscCodec.hpp"
#include "vc/core/zarr/Dtype.hpp"
#include "vc/core/zarr/Tensor3D.hpp"
#include "vc/core/types/IChunkSource.hpp"

namespace volcart::zarr
{

class ShardedStore;

enum class ZarrVersion { V1 = 1, V2 = 2, V3 = 3 };

class ZarrDataset : public IChunkSource
{
public:
    /** @brief Open existing zarr dataset for reading (auto-detects version) */
    explicit ZarrDataset(const std::filesystem::path& path);

    /** @brief Create new zarr dataset for writing */
    ZarrDataset(
        const std::filesystem::path& path,
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& chunks,
        Dtype dtype,
        const std::string& compressor = "blosc",
        const nlohmann::json& compressorOpts = nlohmann::json::object(),
        ZarrVersion version = ZarrVersion::V2,
        const std::vector<std::size_t>& shardShape = {});

    ~ZarrDataset();

    // Non-copyable, movable
    ZarrDataset(const ZarrDataset&) = delete;
    ZarrDataset& operator=(const ZarrDataset&) = delete;
    ZarrDataset(ZarrDataset&&) noexcept;
    ZarrDataset& operator=(ZarrDataset&&) noexcept;

    // --- Metadata accessors ---

    std::string path() const { return path_.string(); }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    Dtype getDtype() const noexcept { return dtype_; }
    const std::vector<std::size_t>& chunkShape() const noexcept
    {
        return chunks_;
    }
    std::size_t defaultChunkSize() const noexcept;
    bool chunkExists(const std::vector<std::size_t>& chunkId) const;
    void getChunkShape(
        const std::vector<std::size_t>& chunkId,
        std::vector<std::size_t>& shapeOut) const;
    char dimensionSeparator() const noexcept { return dimSeparator_; }
    std::size_t ndim() const noexcept { return shape_.size(); }

    ZarrVersion zarrVersion() const noexcept { return version_; }
    const std::vector<std::size_t>& shardShape() const noexcept
    {
        return shardShape_;
    }
    bool isSharded() const noexcept { return shardStore_ != nullptr; }

    void flush();

    // --- IChunkSource interface ---
    std::array<std::size_t, 3> volShape() const override {
        return {shape_[0], shape_[1], shape_[2]};
    }
    std::array<std::size_t, 3> volChunkShape() const override {
        return {chunks_[0], chunks_[1], chunks_[2]};
    }
    Dtype volDtype() const override { return dtype_; }
    std::size_t volChunkElements() const override { return defaultChunkSize(); }
    bool volReadChunk(std::size_t cz, std::size_t cy, std::size_t cx, void* buf) const override {
        return readChunk({cz, cy, cx}, buf);
    }

    // --- Low-level chunk I/O ---

    bool readChunk(const std::vector<std::size_t>& chunkId, void* buffer) const;

    void writeChunk(
        const std::vector<std::size_t>& chunkId,
        const void* buffer,
        std::size_t size);

    // --- High-level I/O ---

    template <typename T>
    void readSubarray(
        Tensor3D<T>& out,
        const std::vector<std::size_t>& offset,
        const std::vector<std::size_t>& shape) const;

    template <typename T>
    void writeSubarray(
        const Tensor3D<T>& data,
        const std::vector<std::size_t>& offset);

private:
    std::filesystem::path path_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> chunks_;
    Dtype dtype_ = Dtype::Unknown;
    char dimSeparator_ = '/';
    std::unique_ptr<BloscCodec> codec_;
    nlohmann::json fillValue_;

    ZarrVersion version_ = ZarrVersion::V2;
    std::vector<std::size_t> shardShape_;
    std::unique_ptr<ShardedStore> shardStore_;
    std::string chunkPrefix_;  // "" for v1/v2, "c" for v3

    std::filesystem::path chunkPath(
        const std::vector<std::size_t>& chunkId) const;

    void loadMetadata();
    void loadMetadataV3();

    void writeMetadata() const;
    void writeMetadataV3() const;

    void ensureChunkDir(const std::vector<std::size_t>& chunkId) const;
};

// Explicit instantiation declarations
extern template void ZarrDataset::readSubarray(
    Tensor3D<std::uint8_t>&,
    const std::vector<std::size_t>&,
    const std::vector<std::size_t>&) const;
extern template void ZarrDataset::readSubarray(
    Tensor3D<std::uint16_t>&,
    const std::vector<std::size_t>&,
    const std::vector<std::size_t>&) const;
extern template void ZarrDataset::readSubarray(
    Tensor3D<float>&,
    const std::vector<std::size_t>&,
    const std::vector<std::size_t>&) const;

extern template void ZarrDataset::writeSubarray(
    const Tensor3D<std::uint8_t>&,
    const std::vector<std::size_t>&);
extern template void ZarrDataset::writeSubarray(
    const Tensor3D<std::uint16_t>&,
    const std::vector<std::size_t>&);
extern template void ZarrDataset::writeSubarray(
    const Tensor3D<float>&,
    const std::vector<std::size_t>&);

}  // namespace volcart::zarr
