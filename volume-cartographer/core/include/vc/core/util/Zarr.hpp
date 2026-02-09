#pragma once

/**
 * @file Zarr.hpp
 * @brief Native Zarr I/O implementation (v1 + v2 + v3).
 *
 * Full spec support via codec pipeline abstraction. Supports:
 * - Zarr v1 (meta file metadata)
 * - Zarr v2 (.zarray / .zgroup metadata)
 * - Zarr v3 (zarr.json metadata with node_type)
 * - Codec pipeline: blosc, gzip, zstd, crc32c, bytes, transpose, sharding
 * - Only uint8, uint16, float32 dtypes
 * - N-dimensional arrays (3D convenience wrappers preserved)
 * - Filesystem store only (no cloud backends)
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/util/ZarrCodecs.hpp"

namespace vc::zarr {

// ============================================================================
// Type aliases (ShapeType also declared in ZarrCodecs.hpp)
// ============================================================================

// ============================================================================
// Array3D — minimal row-major 3D array (replaces xt::xarray / xt::xtensor)
// ============================================================================

template <typename T>
struct Array3D {
    std::vector<T> storage;
    std::size_t sz = 0, sy = 0, sx = 0;  // ZYX dimensions

    Array3D() = default;
    Array3D(std::size_t z, std::size_t y, std::size_t x)
        : storage(z * y * x), sz(z), sy(y), sx(x) {}
    Array3D(std::size_t z, std::size_t y, std::size_t x, T fill)
        : storage(z * y * x, fill), sz(z), sy(y), sx(x) {}

    T& operator()(std::size_t z, std::size_t y, std::size_t x)
        { return storage[z + y * sz + x * sz * sy]; }  // column-major
    const T& operator()(std::size_t z, std::size_t y, std::size_t x) const
        { return storage[z + y * sz + x * sz * sy]; }

    T* data() { return storage.data(); }
    const T* data() const { return storage.data(); }
    std::size_t size() const { return storage.size(); }
    std::array<std::size_t, 3> shape() const { return {sz, sy, sx}; }
};

// ============================================================================
// Enums
// ============================================================================
enum class Dtype { uint8, uint16, float32 };
enum class ZarrVersion { v1, v2, v3 };

// ============================================================================
// Fill value parsing
// ============================================================================

/// Parse fill value from JSON, handling null, number, and special strings
/// ("NaN", "Infinity", "-Infinity", hex float encoding).
double parseFillValue(const nlohmann::json& j, Dtype dtype);

// ============================================================================
// Dataset — concrete native implementation
// ============================================================================

/**
 * @brief A zarr dataset (array) with native chunk I/O.
 *
 * Supports Zarr v1, v2, and v3 metadata formats.
 * Uses CodecPipeline for compression/decompression.
 * Movable, non-copyable.
 */
class Dataset {
public:
    Dataset();
    ~Dataset();

    Dataset(Dataset&& o) noexcept;
    Dataset& operator=(Dataset&& o) noexcept;
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    /// Open an existing dataset from its directory path.
    /// Auto-detects v1 (meta) vs v2 (.zarray) vs v3 (zarr.json).
    static Dataset open(const std::filesystem::path& dsPath);

    /// Create a new dataset (Zarr v2 format).
    static Dataset create(
        const std::filesystem::path& dsPath,
        const std::string& dtype,
        const ShapeType& shape,
        const ShapeType& chunks,
        const std::string& compressor,
        const nlohmann::json& compressorOpts,
        double fillValue = 0,
        const std::string& zarrDelimiter = ".");

    /// Create a new dataset (Zarr v3 format).
    static Dataset createV3(
        const std::filesystem::path& dsPath,
        const std::string& dtype,
        const ShapeType& shape,
        const ShapeType& chunks,
        const std::string& compressor,
        const nlohmann::json& compressorOpts,
        double fillValue = 0,
        const std::string& separator = "/");

    /// Create a new sharded dataset (Zarr v3 format).
    /// shardShape = shard dimensions (stored as chunk_grid in metadata).
    /// innerChunkShape = inner chunk dimensions within each shard.
    static Dataset createV3Sharded(
        const std::filesystem::path& dsPath,
        const std::string& dtype,
        const ShapeType& shape,
        const ShapeType& shardShape,
        const ShapeType& innerChunkShape,
        const nlohmann::json& compressorOpts = {},
        double fillValue = 0,
        const std::string& separator = "/");

    /// Create a new dataset (Zarr v1 format).
    static Dataset createV1(
        const std::filesystem::path& dsPath,
        const std::string& dtype,
        const ShapeType& shape,
        const ShapeType& chunks,
        const std::string& compression = "zlib",
        const nlohmann::json& compressionOpts = 5,
        double fillValue = 0,
        const std::string& order = "C");

    // -- Properties --
    explicit operator bool() const;
    const ShapeType& shape() const;
    const ShapeType& chunkShape() const;
    std::size_t chunkSize() const;  ///< product of chunkShape
    Dtype dtype() const;
    std::size_t dtypeSize() const;  ///< 1, 2, or 4
    bool isUint8() const;
    bool isUint16() const;
    bool isFloat32() const;
    const std::filesystem::path& path() const;
    ZarrVersion version() const;
    double fillValue() const;

    /// Returns true if this dataset uses sharding.
    bool isSharded() const;

    /// If sharded, returns the inner chunk shape.
    ShapeType innerChunkShape() const;

    /// Access the codec pipeline (read-only).
    const CodecPipeline& codecs() const;

    // -- N-D Shard/Chunk I/O --
    /// Check if a shard/chunk file exists on disk.
    bool chunkExists(const ShapeType& chunkIdx) const;

    /// Read and decompress a full shard/chunk.
    /// For non-sharded: reads one chunk. For sharded: reads one full shard.
    /// Output buffer must be chunkSize()*dtypeSize() bytes.
    bool readChunk(const ShapeType& chunkIdx, void* out) const;

    /// Write a full shard/chunk to disk (encodes through full codec pipeline).
    /// For non-sharded: writes one chunk. For sharded: writes one full shard.
    /// data must be chunkSize()*dtypeSize() bytes.
    void writeChunk(const ShapeType& chunkIdx, const void* data);

    // -- 3D convenience overloads (delegate to N-D) --
    bool chunkExists(std::size_t iz, std::size_t iy, std::size_t ix) const;
    bool readChunk(std::size_t iz, std::size_t iy, std::size_t ix, void* out) const;
    void writeChunk(std::size_t iz, std::size_t iy, std::size_t ix, const void* data);

    // -- Sharded inner chunk access --
    /// Read a single inner chunk from a sharded dataset.
    /// shardIdx = which shard, innerIdx = which inner chunk within that shard.
    bool readInnerChunk(const ShapeType& shardIdx,
                        const ShapeType& innerIdx, void* out) const;

    /// Write a single inner chunk to a sharded dataset.
    /// Global chunk coordinates: chunkIdx[d] indexes the inner chunk grid
    /// (i.e. chunkIdx = global_voxel_offset / innerChunkShape).
    /// Internally maps to (shardIdx, innerIdx), compresses, and buffers.
    /// Call flush() to write buffered shards to disk.
    /// data must be innerChunkSize * dtypeSize() bytes.
    void writeInnerChunk(const ShapeType& chunkIdx, const void* data);

    /// Flush buffered shard writes to disk.
    /// Only relevant for sharded datasets; no-op otherwise.
    void flush();

    /// Set the maximum bytes to buffer for pending shard writes.
    /// When exceeded, the least-recently-touched shards are flushed to disk.
    /// Default: 0 (unlimited — only auto-flush on shard completion + final flush).
    void setShardBufferLimit(std::size_t maxBytes);

private:
    std::filesystem::path path_;
    ShapeType shape_;
    ShapeType chunkShape_;
    Dtype dtype_ = Dtype::uint8;
    double fillValue_ = 0;
    std::string delimiter_ = ".";
    ZarrVersion version_ = ZarrVersion::v2;
    std::string chunkKeyEncoding_ = "v2";  // "default" (v3) or "v2"
    CodecPipeline codecs_;
    bool valid_ = false;

    // Shard write buffer (only active for sharded datasets)
    struct ShardBuffer;
    std::unique_ptr<ShardBuffer> shardBuf_;
    void initShardBuffer();
    void writeShardRaw(const ShapeType& shardIdx,
                       const std::vector<uint8_t>& assembled);
    void evictOldestShards();

    // N-D chunk path
    std::filesystem::path chunkPath(const ShapeType& chunkIdx) const;
    // 3D convenience
    std::filesystem::path chunkPath(std::size_t iz, std::size_t iy, std::size_t ix) const;

    void readV1Metadata(const std::filesystem::path& dsPath);
    void readV2Metadata(const std::filesystem::path& dsPath);
    void readV3Metadata(const std::filesystem::path& dsPath);
    void writeV1Metadata() const;
    void writeV2Metadata() const;
    void writeV3Metadata() const;

    /// Fill output buffer with fillValue.
    void fillWithFillValue(void* out, std::size_t totalElems) const;
};

// ============================================================================
// File — lightweight zarr group handle
// ============================================================================

/**
 * @brief Represents a zarr store root (a directory).
 * Movable, non-copyable.
 */
class File {
public:
    explicit File(const std::filesystem::path& path);
    ~File();

    File(File&& o) noexcept;
    File& operator=(File&& o) noexcept;
    File(const File&) = delete;
    File& operator=(const File&) = delete;

    /// List sub-keys (groups/datasets) under this file
    void keys(std::vector<std::string>& out) const;

    /// Get the path this file was opened with
    const std::filesystem::path& path() const;

private:
    std::filesystem::path path_;
};

// ============================================================================
// File / Group operations
// ============================================================================

/// Create a new zarr file (directory). If overwrite=true, removes existing.
void createFile(const std::filesystem::path& path, bool overwrite = false);

/// Create a group within a zarr file.
void createGroup(File& file, const std::string& name);

// ============================================================================
// Dataset operations
// ============================================================================

/// Create a new dataset within a zarr file (v2 format).
Dataset createDataset(
    File& file,
    const std::string& name,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& compressor,
    const nlohmann::json& compressorOpts,
    double fillValue = 0,
    const std::string& zarrDelimiter = ".");

/// Create a dataset within a sub-group of a zarr file.
Dataset createDatasetInGroup(
    File& file,
    const std::string& groupName,
    const std::string& name,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& compressor,
    const nlohmann::json& compressorOpts,
    double fillValue = 0,
    const std::string& zarrDelimiter = ".");

/// Open an existing dataset by name within a zarr file.
Dataset openDataset(File& file, const std::string& name);

/// Open a dataset with automatic format/delimiter detection.
Dataset openDatasetAutoSep(
    const std::filesystem::path& groupPath,
    const std::string& name);

// ============================================================================
// Attribute I/O
// ============================================================================

/// Read attributes from a zarr group path into a JSON object.
void readAttributes(const std::filesystem::path& groupPath, nlohmann::json& out);

/// Write attributes to a zarr file.
void writeAttributes(File& file, const nlohmann::json& attrs);

/// Write attributes to a group within a zarr file (by path).
void writeGroupAttributes(
    const std::filesystem::path& groupPath,
    const std::string& groupName,
    const nlohmann::json& attrs);

/// Write attributes to the root group of a zarr file.
void writeGroupAttributes(File& file, const nlohmann::json& attrs);

// ============================================================================
// Subarray I/O — 3D (templated, explicit instantiations in Zarr.cpp)
// ============================================================================

template <typename T>
void readSubarray(
    const Dataset& ds,
    Array3D<T>& out,
    const ShapeType& offset);

template <typename T>
void writeSubarray(
    const Dataset& ds,
    const Array3D<T>& data,
    const ShapeType& offset);

/// Raw-pointer overload: writes data with given shape at offset.
template <typename T>
void writeSubarray(
    const Dataset& ds,
    const T* data,
    const ShapeType& dataShape,
    const ShapeType& offset);

// ============================================================================
// N-D Subarray I/O — raw buffer overloads
// ============================================================================

/// Read an N-D subarray into a raw buffer (C-order layout).
template <typename T>
void readSubarrayND(
    const Dataset& ds,
    T* out,
    const ShapeType& outShape,
    const ShapeType& offset);

/// Write an N-D subarray from a raw buffer (C-order layout).
template <typename T>
void writeSubarrayND(
    const Dataset& ds,
    const T* data,
    const ShapeType& dataShape,
    const ShapeType& offset);

// ============================================================================
// Consolidated metadata
// ============================================================================

/// Write consolidated metadata for a zarr store.
/// v2: writes .zmetadata at the store root.
/// v3: writes consolidated_metadata in root zarr.json.
void writeConsolidatedMetadata(const std::filesystem::path& storePath);

/// Read consolidated metadata from a zarr store.
/// Returns the full metadata JSON.
nlohmann::json readConsolidatedMetadata(const std::filesystem::path& storePath);

// ============================================================================
// Pyramid helpers
// ============================================================================

/// Write OME-Zarr multiscale attributes (levels 0..maxLevel).
void writeZarrMultiscaleAttrs(
    File& file,
    int maxLevel,
    const nlohmann::json& extraAttrs = {});

/// Build one pyramid level by 2x2x2 mean downsampling.
void buildPyramidLevel(
    File& file,
    int targetLevel,
    const std::string& dtype,
    std::size_t chunkH,
    std::size_t chunkW);

// ============================================================================
// Extern template declarations
// ============================================================================

extern template void readSubarray<uint8_t>(const Dataset&, Array3D<uint8_t>&, const ShapeType&);
extern template void readSubarray<uint16_t>(const Dataset&, Array3D<uint16_t>&, const ShapeType&);
extern template void readSubarray<float>(const Dataset&, Array3D<float>&, const ShapeType&);

extern template void writeSubarray<uint8_t>(const Dataset&, const Array3D<uint8_t>&, const ShapeType&);
extern template void writeSubarray<uint16_t>(const Dataset&, const Array3D<uint16_t>&, const ShapeType&);
extern template void writeSubarray<float>(const Dataset&, const Array3D<float>&, const ShapeType&);

extern template void writeSubarray<uint8_t>(const Dataset&, const uint8_t*, const ShapeType&, const ShapeType&);
extern template void writeSubarray<uint16_t>(const Dataset&, const uint16_t*, const ShapeType&, const ShapeType&);
extern template void writeSubarray<float>(const Dataset&, const float*, const ShapeType&, const ShapeType&);

extern template void readSubarrayND<uint8_t>(const Dataset&, uint8_t*, const ShapeType&, const ShapeType&);
extern template void readSubarrayND<uint16_t>(const Dataset&, uint16_t*, const ShapeType&, const ShapeType&);
extern template void readSubarrayND<float>(const Dataset&, float*, const ShapeType&, const ShapeType&);

extern template void writeSubarrayND<uint8_t>(const Dataset&, const uint8_t*, const ShapeType&, const ShapeType&);
extern template void writeSubarrayND<uint16_t>(const Dataset&, const uint16_t*, const ShapeType&, const ShapeType&);
extern template void writeSubarrayND<float>(const Dataset&, const float*, const ShapeType&, const ShapeType&);

}  // namespace vc::zarr
