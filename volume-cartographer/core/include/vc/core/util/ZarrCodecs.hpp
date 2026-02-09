#pragma once

/**
 * @file ZarrCodecs.hpp
 * @brief Zarr v1/v2/v3 codec pipeline — base classes and all codec declarations.
 *
 * Implements the v3 codec pipeline model:
 *   raw array → [array→array codecs] → [array→bytes codec] → [bytes→bytes codecs] → stored bytes
 *
 * For v1/v2, legacy compressor/filters fields are translated into an equivalent pipeline.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc::zarr {

using ShapeType = std::vector<std::size_t>;

// ============================================================================
// Codec kind enum
// ============================================================================

enum class CodecKind { ArrayToArray, ArrayToBytes, BytesToBytes };

// ============================================================================
// Base classes
// ============================================================================

struct Codec {
    virtual ~Codec() = default;
    virtual CodecKind kind() const = 0;
    virtual std::string name() const = 0;
    virtual nlohmann::json toJson() const = 0;
};

/// array→array: transforms shape/layout (e.g., transpose)
struct ArrayToArrayCodec : Codec {
    CodecKind kind() const override { return CodecKind::ArrayToArray; }
    virtual void encode(std::vector<uint8_t>& data, ShapeType& shape,
                        std::size_t elemSize) const = 0;
    virtual void decode(std::vector<uint8_t>& data, ShapeType& shape,
                        std::size_t elemSize) const = 0;
};

/// array→bytes: serialization (exactly one required in a pipeline)
struct ArrayToBytesCodec : Codec {
    CodecKind kind() const override { return CodecKind::ArrayToBytes; }
    virtual void encode(std::vector<uint8_t>& data, const ShapeType& shape,
                        std::size_t elemSize) const = 0;
    virtual void decode(std::vector<uint8_t>& data, const ShapeType& shape,
                        std::size_t elemSize) const = 0;
};

/// bytes→bytes: compression / checksum
struct BytesToBytesCodec : Codec {
    CodecKind kind() const override { return CodecKind::BytesToBytes; }
    virtual std::vector<uint8_t> encode(const uint8_t* data,
                                        std::size_t len) const = 0;
    virtual std::vector<uint8_t> decode(const uint8_t* data, std::size_t len,
                                        std::size_t expectedLen) const = 0;
};

// ============================================================================
// Concrete codecs
// ============================================================================

/// BytesCodec — array→bytes serialization with optional endian swap.
/// On little-endian x86 with little-endian data this is identity.
struct BytesCodec : ArrayToBytesCodec {
    std::string endian = "little";  // "little" or "big"

    std::string name() const override { return "bytes"; }
    nlohmann::json toJson() const override;
    void encode(std::vector<uint8_t>& data, const ShapeType& shape,
                std::size_t elemSize) const override;
    void decode(std::vector<uint8_t>& data, const ShapeType& shape,
                std::size_t elemSize) const override;

    static std::unique_ptr<BytesCodec> fromJson(const nlohmann::json& j);
};

/// TransposeCodec — array→array dimension permutation.
struct TransposeCodec : ArrayToArrayCodec {
    std::vector<int> order;  // permutation array

    std::string name() const override { return "transpose"; }
    nlohmann::json toJson() const override;
    void encode(std::vector<uint8_t>& data, ShapeType& shape,
                std::size_t elemSize) const override;
    void decode(std::vector<uint8_t>& data, ShapeType& shape,
                std::size_t elemSize) const override;

    static std::unique_ptr<TransposeCodec> fromJson(const nlohmann::json& j);
};

/// BloscCodec — bytes→bytes blosc compression.
struct BloscCodec : BytesToBytesCodec {
    std::string cname = "zstd";
    int clevel = 1;
    int shuffle = 0;   // 0=noshuffle, 1=shuffle, 2=bitshuffle
    int typesize = 1;
    int blocksize = 0;

    std::string name() const override { return "blosc"; }
    nlohmann::json toJson() const override;
    std::vector<uint8_t> encode(const uint8_t* data,
                                std::size_t len) const override;
    std::vector<uint8_t> decode(const uint8_t* data, std::size_t len,
                                std::size_t expectedLen) const override;

    static std::unique_ptr<BloscCodec> fromJson(const nlohmann::json& j);
};

/// GzipCodec — bytes→bytes gzip/zlib compression.
struct GzipCodec : BytesToBytesCodec {
    int level = 6;

    std::string name() const override { return "gzip"; }
    nlohmann::json toJson() const override;
    std::vector<uint8_t> encode(const uint8_t* data,
                                std::size_t len) const override;
    std::vector<uint8_t> decode(const uint8_t* data, std::size_t len,
                                std::size_t expectedLen) const override;

    static std::unique_ptr<GzipCodec> fromJson(const nlohmann::json& j);
};

/// ZstdCodec — bytes→bytes Zstandard compression.
struct ZstdCodec : BytesToBytesCodec {
    int level = 3;
    bool checksum = false;

    std::string name() const override { return "zstd"; }
    nlohmann::json toJson() const override;
    std::vector<uint8_t> encode(const uint8_t* data,
                                std::size_t len) const override;
    std::vector<uint8_t> decode(const uint8_t* data, std::size_t len,
                                std::size_t expectedLen) const override;

    static std::unique_ptr<ZstdCodec> fromJson(const nlohmann::json& j);
};

/// Crc32cCodec — bytes→bytes CRC32C checksum (appends/verifies 4-byte CRC).
struct Crc32cCodec : BytesToBytesCodec {
    std::string name() const override { return "crc32c"; }
    nlohmann::json toJson() const override;
    std::vector<uint8_t> encode(const uint8_t* data,
                                std::size_t len) const override;
    std::vector<uint8_t> decode(const uint8_t* data, std::size_t len,
                                std::size_t expectedLen) const override;

    static std::unique_ptr<Crc32cCodec> fromJson(const nlohmann::json& j);
};

// ============================================================================
// CodecPipeline (declared before ShardingIndexedCodec which uses it)
// ============================================================================

struct CodecPipeline {
    std::vector<std::unique_ptr<ArrayToArrayCodec>> arrayToArray;
    std::unique_ptr<ArrayToBytesCodec> arrayToBytes;  // exactly 1
    std::vector<std::unique_ptr<BytesToBytesCodec>> bytesToBytes;

    CodecPipeline();
    ~CodecPipeline();
    CodecPipeline(CodecPipeline&&) noexcept;
    CodecPipeline& operator=(CodecPipeline&&) noexcept;

    // Non-copyable
    CodecPipeline(const CodecPipeline&) = delete;
    CodecPipeline& operator=(const CodecPipeline&) = delete;

    /// Encode raw array data through the full pipeline.
    std::vector<uint8_t> encode(const void* data, std::size_t dataLen,
                                ShapeType shape,
                                std::size_t elemSize) const;

    /// Decode compressed data through the full pipeline (reverse order).
    void decode(const uint8_t* compressed, std::size_t compLen, void* out,
                std::size_t outLen, ShapeType shape,
                std::size_t elemSize) const;

    /// Serialize pipeline to v3 JSON codecs array.
    nlohmann::json toJson() const;

    /// Build pipeline from v3 "codecs" JSON array.
    static CodecPipeline fromV3Json(const nlohmann::json& codecs);

    /// Build pipeline from v2 compressor + filters + order.
    static CodecPipeline fromV2(const nlohmann::json& compressor,
                                const nlohmann::json& filters,
                                const std::string& order,
                                std::size_t elemSize);

    /// Build pipeline from v1 compression string + opts.
    static CodecPipeline fromV1(const std::string& compression,
                                const nlohmann::json& compressionOpts,
                                const std::string& order,
                                std::size_t elemSize);

    /// Default pipeline: BytesCodec + BloscCodec(zstd, clevel=1).
    static CodecPipeline defaultPipeline(std::size_t elemSize);

    /// Returns true if the pipeline contains a ShardingIndexedCodec.
    bool isSharded() const;

    /// If sharded, returns the inner chunk shape.
    ShapeType innerChunkShape() const;

    /// If sharded, returns a pointer to the sharding codec.
    const struct ShardingIndexedCodec* shardingCodec() const;
};

// ============================================================================
// ShardingIndexedCodec (needs CodecPipeline to be complete)
// ============================================================================

/// ShardingIndexedCodec — array→bytes sharding.
/// Bundles multiple inner chunks into one shard file with an index.
struct ShardingIndexedCodec : ArrayToBytesCodec {
    ShapeType innerChunkShape;
    std::unique_ptr<CodecPipeline> innerCodecs;  // pipeline for each inner chunk
    std::unique_ptr<CodecPipeline> indexCodecs;   // pipeline for shard index
    bool indexAtEnd = true;  // "end" (default) or "start"

    ShardingIndexedCodec();
    ~ShardingIndexedCodec();
    ShardingIndexedCodec(ShardingIndexedCodec&&) noexcept;
    ShardingIndexedCodec& operator=(ShardingIndexedCodec&&) noexcept;

    std::string name() const override { return "sharding_indexed"; }
    nlohmann::json toJson() const override;
    void encode(std::vector<uint8_t>& data, const ShapeType& shape,
                std::size_t elemSize) const override;
    void decode(std::vector<uint8_t>& data, const ShapeType& shape,
                std::size_t elemSize) const override;

    /// Partial read: read one inner chunk from a shard file on disk.
    /// Returns false if inner chunk is empty (fill sentinel).
    bool readInnerChunk(int fd, off_t fileSize, const ShapeType& innerIdx,
                        const ShapeType& shardGridShape, std::size_t elemSize,
                        void* out, std::size_t outBytes) const;

    /// Compress one inner chunk through the inner codec pipeline.
    std::vector<uint8_t> encodeInnerChunk(const void* data,
                                           std::size_t len,
                                           std::size_t elemSize) const;

    /// Assemble a complete shard from pre-compressed inner chunks.
    /// compressedChunks[i] is empty for missing/fill-value inner chunks.
    std::vector<uint8_t> assembleShard(
        const std::vector<std::vector<uint8_t>>& compressedChunks,
        std::size_t numInner) const;

    static std::unique_ptr<ShardingIndexedCodec> fromJson(
        const nlohmann::json& j);
};

}  // namespace vc::zarr
