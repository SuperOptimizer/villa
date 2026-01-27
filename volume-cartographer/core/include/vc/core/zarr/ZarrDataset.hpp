#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/zarr/BloscCodec.hpp"
#include "vc/core/zarr/Tensor3D.hpp"

namespace volcart::zarr
{

/**
 * @brief Data type enumeration for zarr arrays
 */
enum class Dtype {
    UInt8,
    UInt16,
    Float32,
    Unknown
};

/**
 * @brief Convert Dtype to string representation
 */
std::string dtypeToString(Dtype dtype);

/**
 * @brief Parse dtype from zarr string representation
 */
Dtype dtypeFromString(const std::string& s);

/**
 * @brief Get size in bytes for a dtype
 */
std::size_t dtypeSize(Dtype dtype);

/**
 * @brief Zarr v2 dataset reader/writer.
 *
 * This class provides read/write access to zarr v2 datasets stored on disk.
 * It replaces z5::Dataset with a minimal implementation that supports only
 * the features used in volume-cartographer:
 * - Reading and writing chunks
 * - Reading and writing subarrays (multi-chunk regions)
 * - Blosc compression with zstd codec
 * - uint8, uint16, and float32 data types
 * - 3D arrays with configurable chunk sizes
 *
 * File layout (zarr v2 spec):
 *   dataset_path/
 *     .zarray          - JSON metadata
 *     0/0/0            - chunk files (with dimension separator)
 *     0/0/1
 *     ...
 */
class ZarrDataset
{
public:
    /**
     * @brief Open existing zarr dataset for reading
     * @param path Path to zarr dataset directory (containing .zarray)
     * @throws std::runtime_error if dataset doesn't exist or is invalid
     */
    explicit ZarrDataset(const std::filesystem::path& path);

    /**
     * @brief Create new zarr dataset for writing
     * @param path Path to zarr dataset directory (will be created)
     * @param shape Dataset shape (e.g., {z, y, x})
     * @param chunks Chunk shape (e.g., {64, 64, 64})
     * @param dtype Data type
     * @param compressor Compressor name ("blosc" or "" for none)
     * @param compressorOpts Compressor options (for blosc: cname, clevel, shuffle)
     * @throws std::runtime_error if dataset creation fails
     */
    ZarrDataset(
        const std::filesystem::path& path,
        const std::vector<std::size_t>& shape,
        const std::vector<std::size_t>& chunks,
        Dtype dtype,
        const std::string& compressor = "blosc",
        const nlohmann::json& compressorOpts = nlohmann::json::object());

    ~ZarrDataset() = default;

    // Non-copyable, movable
    ZarrDataset(const ZarrDataset&) = delete;
    ZarrDataset& operator=(const ZarrDataset&) = delete;
    ZarrDataset(ZarrDataset&&) noexcept = default;
    ZarrDataset& operator=(ZarrDataset&&) noexcept = default;

    // --- Metadata accessors (matching z5 API for easier migration) ---

    /** @brief Get dataset path */
    std::string path() const { return path_.string(); }

    /** @brief Get dataset shape */
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }

    /** @brief Get data type */
    Dtype getDtype() const noexcept { return dtype_; }

    /** @brief Get chunk shape */
    const std::vector<std::size_t>& chunkShape() const noexcept
    {
        return chunks_;
    }

    /**
     * @brief Get default chunk size (product of chunk dimensions)
     * @return Number of elements in a full chunk
     */
    std::size_t defaultChunkSize() const noexcept;

    /**
     * @brief Check if a chunk exists on disk
     * @param chunkId Chunk coordinates (e.g., {0, 1, 2})
     * @return True if chunk file exists
     */
    bool chunkExists(const std::vector<std::size_t>& chunkId) const;

    /**
     * @brief Get shape of a specific chunk (may be smaller at boundaries)
     * @param chunkId Chunk coordinates
     * @param shapeOut Output vector to receive chunk shape
     */
    void getChunkShape(
        const std::vector<std::size_t>& chunkId,
        std::vector<std::size_t>& shapeOut) const;

    /** @brief Get dimension separator character ('/' or '.') */
    char dimensionSeparator() const noexcept { return dimSeparator_; }

    /** @brief Get number of dimensions */
    std::size_t ndim() const noexcept { return shape_.size(); }

    // --- Low-level chunk I/O ---

    /**
     * @brief Read a chunk from disk
     * @param chunkId Chunk coordinates
     * @param buffer Output buffer (must be sized for full chunk)
     * @return True if chunk was read, false if chunk doesn't exist
     */
    bool readChunk(const std::vector<std::size_t>& chunkId, void* buffer) const;

    /**
     * @brief Write a chunk to disk
     * @param chunkId Chunk coordinates
     * @param buffer Input buffer
     * @param size Size of data in bytes
     */
    void writeChunk(
        const std::vector<std::size_t>& chunkId,
        const void* buffer,
        std::size_t size);

    // --- High-level I/O ---

    /**
     * @brief Read a subarray into a tensor
     * @tparam T Data type (must match dataset dtype)
     * @param out Output tensor (will be resized if needed)
     * @param offset Start offset for each dimension
     * @param shape Shape of region to read
     */
    template <typename T>
    void readSubarray(
        Tensor3D<T>& out,
        const std::vector<std::size_t>& offset,
        const std::vector<std::size_t>& shape) const;

    /**
     * @brief Write a tensor to a subarray region
     * @tparam T Data type (must match dataset dtype)
     * @param data Input tensor
     * @param offset Start offset for each dimension
     */
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

    /** @brief Build chunk file path from chunk ID */
    std::filesystem::path chunkPath(
        const std::vector<std::size_t>& chunkId) const;

    /** @brief Load metadata from .zarray file */
    void loadMetadata();

    /** @brief Write metadata to .zarray file */
    void writeMetadata() const;

    /** @brief Ensure directory for chunk exists */
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
