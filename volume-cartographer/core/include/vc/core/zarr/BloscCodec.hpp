#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace volcart::zarr
{

/**
 * @brief Blosc compression/decompression codec for zarr chunks.
 *
 * Wraps the blosc library to provide compression and decompression
 * of raw data buffers. Supports various compressors (zstd, lz4, etc.)
 * and shuffle modes.
 *
 * Configuration follows zarr v2 blosc compressor format:
 * {
 *   "id": "blosc",
 *   "cname": "zstd",
 *   "clevel": 1,
 *   "shuffle": 1,
 *   "blocksize": 0
 * }
 */
class BloscCodec
{
public:
    /** @brief Default constructor with sensible defaults (zstd, level 1) */
    BloscCodec();

    /**
     * @brief Construct from zarr compressor configuration
     * @param config JSON object with blosc configuration
     *
     * Expected keys:
     * - "cname": Compressor name ("zstd", "lz4", "lz4hc", "blosclz", "snappy")
     * - "clevel": Compression level (1-9, higher = more compression)
     * - "shuffle": Shuffle mode (0=none, 1=byte, 2=bit)
     * - "blocksize": Block size (0 = auto)
     */
    explicit BloscCodec(const nlohmann::json& config);

    /**
     * @brief Compress data
     * @param src Pointer to source data
     * @param size Size of source data in bytes
     * @param typesize Size of each element (used for shuffle)
     * @return Compressed data as vector
     */
    std::vector<std::uint8_t>
    compress(const void* src, std::size_t size, std::size_t typesize) const;

    /**
     * @brief Decompress data
     * @param src Pointer to compressed data
     * @param srcSize Size of compressed data
     * @param dst Pointer to destination buffer
     * @param dstCapacity Capacity of destination buffer
     * @return Number of bytes written to dst
     * @throws std::runtime_error if decompression fails
     */
    std::size_t decompress(
        const void* src,
        std::size_t srcSize,
        void* dst,
        std::size_t dstCapacity) const;

    /**
     * @brief Check if data appears to be blosc-compressed
     * @param data Pointer to data
     * @param size Size of data
     * @return True if data has blosc header
     */
    static bool isBlosc(const void* data, std::size_t size);

    /**
     * @brief Get decompressed size from blosc header
     * @param data Pointer to compressed data
     * @return Decompressed size, or 0 if not valid blosc data
     */
    static std::size_t decompressedSize(const void* data);

    /**
     * @brief Get compressor name
     * @return Name of compression codec (e.g., "zstd")
     */
    const std::string& compressorName() const noexcept { return cname_; }

    /**
     * @brief Get compression level
     * @return Compression level (1-9)
     */
    int compressionLevel() const noexcept { return clevel_; }

    /**
     * @brief Get shuffle mode
     * @return Shuffle mode (0=none, 1=byte, 2=bit)
     */
    int shuffleMode() const noexcept { return shuffle_; }

    /**
     * @brief Convert to zarr compressor JSON config
     * @return JSON object representing this codec
     */
    nlohmann::json toJson() const;

private:
    std::string cname_ = "zstd";
    int clevel_ = 1;
    int shuffle_ = 1;  // BLOSC_SHUFFLE
    std::size_t blocksize_ = 0;  // auto

    /** @brief Initialize blosc library (thread-safe) */
    static void initBlosc();
};

}  // namespace volcart::zarr
