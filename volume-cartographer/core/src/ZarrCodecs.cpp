#include "vc/core/util/ZarrCodecs.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <blosc.h>
#include <zlib.h>
#include <zstd.h>

// POSIX I/O for partial shard reads
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// SSE4.2 CRC32C intrinsic
#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

namespace vc::zarr {

// ============================================================================
// Helpers
// ============================================================================

static std::size_t shapeProduct(const ShapeType& shape)
{
    if (shape.empty()) return 0;
    std::size_t p = 1;
    for (auto d : shape) p *= d;
    return p;
}

/// Compute CRC32C over a buffer.
static uint32_t computeCrc32c(const uint8_t* data, std::size_t len)
{
#if defined(__SSE4_2__)
    // Use hardware CRC32C via SSE4.2
    uint32_t crc = 0xFFFFFFFF;
    // Process 8 bytes at a time
    const uint8_t* p = data;
    std::size_t remaining = len;
    while (remaining >= 8) {
        uint64_t val;
        std::memcpy(&val, p, 8);
        crc = static_cast<uint32_t>(
            _mm_crc32_u64(static_cast<uint64_t>(crc), val));
        p += 8;
        remaining -= 8;
    }
    // Remaining bytes
    while (remaining > 0) {
        crc = _mm_crc32_u8(crc, *p);
        ++p;
        --remaining;
    }
    return crc ^ 0xFFFFFFFF;
#else
    // Software fallback — CRC32C lookup table
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j)
                c = (c >> 1) ^ (0x82F63B78 & (-(c & 1)));
            t[i] = c;
        }
        return t;
    }();
    uint32_t crc = 0xFFFFFFFF;
    for (std::size_t i = 0; i < len; ++i)
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
    return crc ^ 0xFFFFFFFF;
#endif
}

/// Byte-swap elements in-place.
static void byteSwapInPlace(uint8_t* data, std::size_t totalBytes,
                             std::size_t elemSize)
{
    if (elemSize <= 1) return;
    for (std::size_t i = 0; i < totalBytes; i += elemSize) {
        std::reverse(data + i, data + i + elemSize);
    }
}

/// Check if system is little-endian.
static bool isLittleEndian()
{
    uint16_t val = 1;
    return *reinterpret_cast<const uint8_t*>(&val) == 1;
}

// ============================================================================
// BytesCodec
// ============================================================================

nlohmann::json BytesCodec::toJson() const
{
    return {{"name", "bytes"},
            {"configuration", {{"endian", endian}}}};
}

void BytesCodec::encode(std::vector<uint8_t>& data, const ShapeType& /*shape*/,
                         std::size_t elemSize) const
{
    // If endian doesn't match native, swap
    bool needSwap = (endian == "big" && isLittleEndian()) ||
                    (endian == "little" && !isLittleEndian());
    if (needSwap && elemSize > 1) {
        byteSwapInPlace(data.data(), data.size(), elemSize);
    }
}

void BytesCodec::decode(std::vector<uint8_t>& data, const ShapeType& /*shape*/,
                         std::size_t elemSize) const
{
    bool needSwap = (endian == "big" && isLittleEndian()) ||
                    (endian == "little" && !isLittleEndian());
    if (needSwap && elemSize > 1) {
        byteSwapInPlace(data.data(), data.size(), elemSize);
    }
}

std::unique_ptr<BytesCodec> BytesCodec::fromJson(const nlohmann::json& j)
{
    auto c = std::make_unique<BytesCodec>();
    if (j.contains("configuration") && j["configuration"].contains("endian")) {
        c->endian = j["configuration"]["endian"].get<std::string>();
    }
    return c;
}

// ============================================================================
// TransposeCodec
// ============================================================================

nlohmann::json TransposeCodec::toJson() const
{
    nlohmann::json orderArr = nlohmann::json::array();
    for (int o : order) orderArr.push_back(o);
    return {{"name", "transpose"},
            {"configuration", {{"order", orderArr}}}};
}

void TransposeCodec::encode(std::vector<uint8_t>& data, ShapeType& shape,
                             std::size_t elemSize) const
{
    if (order.empty() || shape.empty()) return;
    const std::size_t ndim = shape.size();
    if (order.size() != ndim) {
        throw std::runtime_error("TransposeCodec: order size != ndim");
    }

    const std::size_t totalElems = shapeProduct(shape);
    std::vector<uint8_t> tmp(data.size());

    // Compute strides for source (row-major)
    std::vector<std::size_t> srcStrides(ndim);
    srcStrides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        srcStrides[i] = srcStrides[i + 1] * shape[i + 1];
    }

    // New shape and strides after transpose
    ShapeType newShape(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        newShape[i] = shape[order[i]];
    }
    std::vector<std::size_t> dstStrides(ndim);
    dstStrides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
        dstStrides[i] = dstStrides[i + 1] * newShape[i + 1];
    }

    // Transpose element by element
    std::vector<std::size_t> coords(ndim, 0);
    for (std::size_t flat = 0; flat < totalElems; ++flat) {
        // Compute multi-index in source
        std::size_t rem = flat;
        for (std::size_t d = 0; d < ndim; ++d) {
            coords[d] = rem / srcStrides[d];
            rem %= srcStrides[d];
        }
        // Compute destination index
        std::size_t dstFlat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            dstFlat += coords[order[d]] * dstStrides[d];
        }
        std::memcpy(tmp.data() + dstFlat * elemSize,
                     data.data() + flat * elemSize, elemSize);
    }

    data = std::move(tmp);
    shape = newShape;
}

void TransposeCodec::decode(std::vector<uint8_t>& data, ShapeType& shape,
                             std::size_t elemSize) const
{
    if (order.empty() || shape.empty()) return;
    // Inverse permutation
    const std::size_t ndim = order.size();
    std::vector<int> inv(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        inv[order[i]] = static_cast<int>(i);
    }

    // Create a temporary TransposeCodec with inverse order and apply encode
    TransposeCodec invCodec;
    invCodec.order = inv;
    invCodec.encode(data, shape, elemSize);
}

std::unique_ptr<TransposeCodec> TransposeCodec::fromJson(
    const nlohmann::json& j)
{
    auto c = std::make_unique<TransposeCodec>();
    if (j.contains("configuration") && j["configuration"].contains("order")) {
        c->order = j["configuration"]["order"].get<std::vector<int>>();
    }
    return c;
}

// ============================================================================
// BloscCodec
// ============================================================================

nlohmann::json BloscCodec::toJson() const
{
    std::string shuffleStr;
    switch (shuffle) {
    case 0: shuffleStr = "noshuffle"; break;
    case 1: shuffleStr = "shuffle"; break;
    case 2: shuffleStr = "bitshuffle"; break;
    default: shuffleStr = "noshuffle"; break;
    }
    return {{"name", "blosc"},
            {"configuration",
             {{"cname", cname},
              {"clevel", clevel},
              {"shuffle", shuffleStr},
              {"typesize", typesize},
              {"blocksize", blocksize}}}};
}

std::vector<uint8_t> BloscCodec::encode(const uint8_t* data,
                                         std::size_t len) const
{
    std::size_t maxOut = len + BLOSC_MAX_OVERHEAD;
    std::vector<uint8_t> out(maxOut);

    int compSize = blosc_compress_ctx(
        clevel, shuffle, static_cast<std::size_t>(typesize), len, data,
        out.data(), maxOut, cname.c_str(), static_cast<std::size_t>(blocksize),
        1 /* numinternalthreads */);

    if (compSize <= 0) {
        throw std::runtime_error("BloscCodec: compression failed");
    }

    out.resize(static_cast<std::size_t>(compSize));
    return out;
}

std::vector<uint8_t> BloscCodec::decode(const uint8_t* data, std::size_t len,
                                         std::size_t expectedLen) const
{
    std::vector<uint8_t> out(expectedLen);

    int decompSize = blosc_decompress_ctx(data, out.data(), expectedLen,
                                           1 /* numinternalthreads */);

    if (decompSize < 0) {
        throw std::runtime_error("BloscCodec: decompression failed");
    }

    return out;
}

std::unique_ptr<BloscCodec> BloscCodec::fromJson(const nlohmann::json& j)
{
    auto c = std::make_unique<BloscCodec>();
    if (!j.contains("configuration")) return c;
    auto& cfg = j["configuration"];

    c->cname = cfg.value("cname", std::string("zstd"));
    c->clevel = cfg.value("clevel", 1);
    c->blocksize = cfg.value("blocksize", 0);
    c->typesize = cfg.value("typesize", 1);

    if (cfg.contains("shuffle")) {
        if (cfg["shuffle"].is_string()) {
            std::string sh = cfg["shuffle"].get<std::string>();
            if (sh == "noshuffle")
                c->shuffle = 0;
            else if (sh == "shuffle")
                c->shuffle = 1;
            else if (sh == "bitshuffle")
                c->shuffle = 2;
            else
                c->shuffle = 0;
        } else {
            c->shuffle = cfg["shuffle"].get<int>();
        }
    }

    return c;
}

// ============================================================================
// GzipCodec
// ============================================================================

nlohmann::json GzipCodec::toJson() const
{
    return {{"name", "gzip"}, {"configuration", {{"level", level}}}};
}

std::vector<uint8_t> GzipCodec::encode(const uint8_t* data,
                                        std::size_t len) const
{
    // Use deflateInit2 with windowBits=31 for gzip format (RFC 1952)
    z_stream strm{};
    int ret = deflateInit2(&strm, level, Z_DEFLATED, 15 + 16, 8,
                           Z_DEFAULT_STRATEGY);
    if (ret != Z_OK) {
        throw std::runtime_error("GzipCodec: deflateInit2 failed");
    }

    strm.next_in = const_cast<Bytef*>(data);
    strm.avail_in = static_cast<uInt>(len);

    std::vector<uint8_t> out(deflateBound(&strm, static_cast<uLong>(len)));
    strm.next_out = out.data();
    strm.avail_out = static_cast<uInt>(out.size());

    ret = deflate(&strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        deflateEnd(&strm);
        throw std::runtime_error("GzipCodec: deflate failed with code " +
                                 std::to_string(ret));
    }

    out.resize(strm.total_out);
    deflateEnd(&strm);
    return out;
}

std::vector<uint8_t> GzipCodec::decode(const uint8_t* data, std::size_t len,
                                        std::size_t expectedLen) const
{
    // Use inflateInit2 with windowBits=15+32 to auto-detect gzip or zlib
    z_stream strm{};
    int ret = inflateInit2(&strm, 15 + 32);
    if (ret != Z_OK) {
        throw std::runtime_error("GzipCodec: inflateInit2 failed");
    }

    strm.next_in = const_cast<Bytef*>(data);
    strm.avail_in = static_cast<uInt>(len);

    std::vector<uint8_t> out(expectedLen);
    strm.next_out = out.data();
    strm.avail_out = static_cast<uInt>(out.size());

    ret = inflate(&strm, Z_FINISH);
    if (ret != Z_STREAM_END) {
        inflateEnd(&strm);
        throw std::runtime_error("GzipCodec: inflate failed with code " +
                                 std::to_string(ret));
    }

    out.resize(strm.total_out);
    inflateEnd(&strm);
    return out;
}

std::unique_ptr<GzipCodec> GzipCodec::fromJson(const nlohmann::json& j)
{
    auto c = std::make_unique<GzipCodec>();
    if (j.contains("configuration")) {
        c->level = j["configuration"].value("level", 6);
    }
    return c;
}

// ============================================================================
// ZstdCodec
// ============================================================================

nlohmann::json ZstdCodec::toJson() const
{
    return {{"name", "zstd"},
            {"configuration", {{"level", level}, {"checksum", checksum}}}};
}

std::vector<uint8_t> ZstdCodec::encode(const uint8_t* data,
                                        std::size_t len) const
{
    std::size_t maxOut = ZSTD_compressBound(len);
    std::vector<uint8_t> out(maxOut);

    std::size_t compSize = ZSTD_compress(out.data(), maxOut, data, len, level);
    if (ZSTD_isError(compSize)) {
        throw std::runtime_error(
            std::string("ZstdCodec: compression failed: ") +
            ZSTD_getErrorName(compSize));
    }

    out.resize(compSize);
    return out;
}

std::vector<uint8_t> ZstdCodec::decode(const uint8_t* data, std::size_t len,
                                        std::size_t expectedLen) const
{
    std::vector<uint8_t> out(expectedLen);

    std::size_t decompSize =
        ZSTD_decompress(out.data(), expectedLen, data, len);
    if (ZSTD_isError(decompSize)) {
        throw std::runtime_error(
            std::string("ZstdCodec: decompression failed: ") +
            ZSTD_getErrorName(decompSize));
    }

    out.resize(decompSize);
    return out;
}

std::unique_ptr<ZstdCodec> ZstdCodec::fromJson(const nlohmann::json& j)
{
    auto c = std::make_unique<ZstdCodec>();
    if (j.contains("configuration")) {
        c->level = j["configuration"].value("level", 3);
        c->checksum = j["configuration"].value("checksum", false);
    }
    return c;
}

// ============================================================================
// Crc32cCodec
// ============================================================================

nlohmann::json Crc32cCodec::toJson() const
{
    return {{"name", "crc32c"}};
}

std::vector<uint8_t> Crc32cCodec::encode(const uint8_t* data,
                                          std::size_t len) const
{
    // Append 4-byte CRC32C checksum
    std::vector<uint8_t> out(len + 4);
    std::memcpy(out.data(), data, len);

    uint32_t crc = computeCrc32c(data, len);
    // Store as little-endian
    out[len + 0] = static_cast<uint8_t>(crc & 0xFF);
    out[len + 1] = static_cast<uint8_t>((crc >> 8) & 0xFF);
    out[len + 2] = static_cast<uint8_t>((crc >> 16) & 0xFF);
    out[len + 3] = static_cast<uint8_t>((crc >> 24) & 0xFF);
    return out;
}

std::vector<uint8_t> Crc32cCodec::decode(const uint8_t* data, std::size_t len,
                                          std::size_t /*expectedLen*/) const
{
    if (len < 4) {
        throw std::runtime_error("Crc32cCodec: data too short for checksum");
    }

    std::size_t dataLen = len - 4;
    uint32_t stored = static_cast<uint32_t>(data[dataLen]) |
                      (static_cast<uint32_t>(data[dataLen + 1]) << 8) |
                      (static_cast<uint32_t>(data[dataLen + 2]) << 16) |
                      (static_cast<uint32_t>(data[dataLen + 3]) << 24);

    uint32_t computed = computeCrc32c(data, dataLen);
    if (stored != computed) {
        throw std::runtime_error("Crc32cCodec: checksum mismatch");
    }

    return std::vector<uint8_t>(data, data + dataLen);
}

std::unique_ptr<Crc32cCodec> Crc32cCodec::fromJson(const nlohmann::json&)
{
    return std::make_unique<Crc32cCodec>();
}

// ============================================================================
// ShardingIndexedCodec
// ============================================================================

ShardingIndexedCodec::ShardingIndexedCodec() = default;
ShardingIndexedCodec::~ShardingIndexedCodec() = default;
ShardingIndexedCodec::ShardingIndexedCodec(ShardingIndexedCodec&&) noexcept =
    default;
ShardingIndexedCodec& ShardingIndexedCodec::operator=(
    ShardingIndexedCodec&&) noexcept = default;

nlohmann::json ShardingIndexedCodec::toJson() const
{
    nlohmann::json cfg;
    cfg["chunk_shape"] = innerChunkShape;
    cfg["codecs"] = innerCodecs ? innerCodecs->toJson() : nlohmann::json::array();
    cfg["index_codecs"] =
        indexCodecs ? indexCodecs->toJson() : nlohmann::json::array();
    cfg["index_location"] = indexAtEnd ? "end" : "start";

    return {{"name", "sharding_indexed"}, {"configuration", cfg}};
}

void ShardingIndexedCodec::encode(std::vector<uint8_t>& data,
                                   const ShapeType& shape,
                                   std::size_t elemSize) const
{
    if (!innerCodecs) {
        throw std::runtime_error(
            "ShardingIndexedCodec: innerCodecs not configured");
    }

    // Compute shard grid (how many inner chunks per shard dimension)
    const std::size_t ndim = shape.size();
    ShapeType shardGrid(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        shardGrid[d] = (shape[d] + innerChunkShape[d] - 1) / innerChunkShape[d];
    }

    const std::size_t numInner = shapeProduct(shardGrid);
    const std::size_t innerElems = shapeProduct(innerChunkShape);
    const std::size_t innerBytes = innerElems * elemSize;

    // Index: pairs of (offset, nbytes) as uint64
    std::vector<uint64_t> index(numInner * 2);
    std::vector<uint8_t> shardData;

    // Iterate over inner chunks in C-order
    std::vector<std::size_t> innerCoords(ndim, 0);
    for (std::size_t ichunk = 0; ichunk < numInner; ++ichunk) {
        // Extract inner chunk from the shard data
        std::vector<uint8_t> innerBuf(innerBytes);

        // Compute source strides in elements (C-order)
        std::vector<std::size_t> srcStrides(ndim);
        srcStrides[ndim - 1] = 1;
        for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
            srcStrides[d] = srcStrides[d + 1] * shape[d + 1];
        }

        // Copy row-by-row (innermost dimension contiguous)
        std::size_t innerRow = innerChunkShape[ndim - 1] * elemSize;
        std::vector<std::size_t> coords(ndim, 0);

        // Total rows in inner chunk = product of all dims except last
        std::size_t numRows = innerElems / innerChunkShape[ndim - 1];
        for (std::size_t row = 0; row < numRows; ++row) {
            // Compute multi-index for this row
            std::size_t rem = row;
            for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
                std::size_t dimSize = innerChunkShape[d];
                coords[d] = rem % dimSize;
                rem /= dimSize;
            }

            // Source offset in the full shard buffer (in bytes)
            std::size_t srcOff = 0;
            for (std::size_t d = 0; d < ndim - 1; ++d) {
                std::size_t globalCoord =
                    innerCoords[d] * innerChunkShape[d] + coords[d];
                if (globalCoord >= shape[d]) goto skip_row;
                srcOff += globalCoord * srcStrides[d];
            }
            {
                std::size_t lastGlobal =
                    innerCoords[ndim - 1] * innerChunkShape[ndim - 1];
                if (lastGlobal >= shape[ndim - 1]) goto skip_row;
                srcOff += lastGlobal;
                srcOff *= elemSize;

                std::size_t copyLen = std::min(
                    innerChunkShape[ndim - 1],
                    shape[ndim - 1] - lastGlobal) * elemSize;
                std::size_t dstOff = row * innerRow;
                std::memcpy(innerBuf.data() + dstOff, data.data() + srcOff,
                            copyLen);
            }
        skip_row:;
        }

        // Check if all zeros (empty sentinel) before encoding
        bool allZero = true;
        for (std::size_t i = 0; i < innerBuf.size() && allZero; ++i) {
            if (innerBuf[i] != 0) allZero = false;
        }

        if (allZero) {
            index[ichunk * 2] = 0xFFFFFFFFFFFFFFFF;
            index[ichunk * 2 + 1] = 0xFFFFFFFFFFFFFFFF;
        } else {
            auto encoded =
                innerCodecs->encode(innerBuf.data(), innerBytes,
                                    innerChunkShape, elemSize);
            index[ichunk * 2] = shardData.size();
            index[ichunk * 2 + 1] = encoded.size();
            shardData.insert(shardData.end(), encoded.begin(), encoded.end());
        }

        // Advance coords
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++innerCoords[d] < shardGrid[d]) break;
            innerCoords[d] = 0;
        }
    }

    // Encode the index
    std::size_t indexBytes = numInner * 2 * sizeof(uint64_t);
    std::vector<uint8_t> indexBuf(indexBytes);
    std::memcpy(indexBuf.data(), index.data(), indexBytes);

    // Encode index through index pipeline if available
    if (indexCodecs) {
        // Shape for index: [numInner, 2] but we treat as raw bytes
        ShapeType indexShape = {numInner, 2};
        auto encodedIndex =
            indexCodecs->encode(indexBuf.data(), indexBytes, indexShape,
                                sizeof(uint64_t));
        indexBuf = std::move(encodedIndex);
    }

    // Assemble shard: data + index (or index + data if indexAtEnd=false)
    data.clear();
    if (indexAtEnd) {
        data.insert(data.end(), shardData.begin(), shardData.end());
        data.insert(data.end(), indexBuf.begin(), indexBuf.end());
    } else {
        data.insert(data.end(), indexBuf.begin(), indexBuf.end());
        data.insert(data.end(), shardData.begin(), shardData.end());
    }
}

void ShardingIndexedCodec::decode(std::vector<uint8_t>& data,
                                   const ShapeType& shape,
                                   std::size_t elemSize) const
{
    if (!innerCodecs) {
        throw std::runtime_error(
            "ShardingIndexedCodec: innerCodecs not configured");
    }

    const std::size_t ndim = shape.size();
    ShapeType shardGrid(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        shardGrid[d] = (shape[d] + innerChunkShape[d] - 1) / innerChunkShape[d];
    }

    const std::size_t numInner = shapeProduct(shardGrid);
    const std::size_t innerElems = shapeProduct(innerChunkShape);
    const std::size_t innerBytes = innerElems * elemSize;

    // Read index
    std::size_t indexBytes = numInner * 2 * sizeof(uint64_t);
    std::vector<uint8_t> indexBuf;

    if (indexAtEnd) {
        if (data.size() < indexBytes) {
            throw std::runtime_error("ShardingIndexedCodec: shard too small");
        }
        indexBuf.assign(data.end() - indexBytes, data.end());
    } else {
        if (data.size() < indexBytes) {
            throw std::runtime_error("ShardingIndexedCodec: shard too small");
        }
        indexBuf.assign(data.begin(), data.begin() + indexBytes);
    }

    // Decode index through index codec pipeline
    if (indexCodecs) {
        ShapeType indexShape = {numInner, 2};
        std::vector<uint8_t> decoded(indexBytes);
        indexCodecs->decode(indexBuf.data(), indexBuf.size(), decoded.data(),
                             indexBytes, indexShape, sizeof(uint64_t));
        indexBuf = std::move(decoded);
    }

    // Parse index entries
    std::vector<uint64_t> index(numInner * 2);
    if (indexBuf.size() >= indexBytes) {
        std::memcpy(index.data(), indexBuf.data(), indexBytes);
    }

    // Allocate output
    std::size_t totalBytes = shapeProduct(shape) * elemSize;
    std::vector<uint8_t> output(totalBytes, 0);

    // Decode each inner chunk
    std::vector<std::size_t> innerCoords(ndim, 0);
    for (std::size_t ichunk = 0; ichunk < numInner; ++ichunk) {
        uint64_t offset = index[ichunk * 2];
        uint64_t nbytes = index[ichunk * 2 + 1];

        if (offset == 0xFFFFFFFFFFFFFFFF && nbytes == 0xFFFFFFFFFFFFFFFF) {
            // Empty inner chunk — already zero-filled
        } else {
            // Adjust offset for index-at-start
            std::size_t dataOffset = static_cast<std::size_t>(offset);
            if (!indexAtEnd) {
                dataOffset += indexBytes;
            }

            // Decode inner chunk
            std::vector<uint8_t> innerBuf(innerBytes);
            innerCodecs->decode(data.data() + dataOffset,
                                 static_cast<std::size_t>(nbytes),
                                 innerBuf.data(), innerBytes,
                                 innerChunkShape, elemSize);

            // Copy into output at correct position
            // outStrides in elements (C-order)
            std::vector<std::size_t> outStrides(ndim);
            outStrides[ndim - 1] = 1;
            for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
                outStrides[d] = outStrides[d + 1] * shape[d + 1];
            }

            std::size_t innerRow = innerChunkShape[ndim - 1] * elemSize;
            std::size_t numRows = innerElems / innerChunkShape[ndim - 1];
            std::vector<std::size_t> coords(ndim, 0);

            for (std::size_t row = 0; row < numRows; ++row) {
                std::size_t rem = row;
                for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
                    coords[d] = rem % innerChunkShape[d];
                    rem /= innerChunkShape[d];
                }

                std::size_t dstOff = 0;
                bool outOfBounds = false;
                for (std::size_t d = 0; d < ndim - 1; ++d) {
                    std::size_t g =
                        innerCoords[d] * innerChunkShape[d] + coords[d];
                    if (g >= shape[d]) { outOfBounds = true; break; }
                    dstOff += g * outStrides[d];
                }
                if (outOfBounds) continue;

                std::size_t lastGlobal =
                    innerCoords[ndim - 1] * innerChunkShape[ndim - 1];
                if (lastGlobal >= shape[ndim - 1]) continue;
                dstOff += lastGlobal;
                dstOff *= elemSize;

                std::size_t copyLen = std::min(
                    innerChunkShape[ndim - 1],
                    shape[ndim - 1] - lastGlobal) * elemSize;
                std::size_t srcOff = row * innerRow;
                std::memcpy(output.data() + dstOff, innerBuf.data() + srcOff,
                            copyLen);
            }
        }

        // Advance coords
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++innerCoords[d] < shardGrid[d]) break;
            innerCoords[d] = 0;
        }
    }

    data = std::move(output);
}

bool ShardingIndexedCodec::readInnerChunk(
    int fd, off_t fileSize, const ShapeType& innerIdx,
    const ShapeType& shardGridShape, std::size_t elemSize, void* out,
    std::size_t outBytes) const
{
    const std::size_t numInner = shapeProduct(shardGridShape);
    const std::size_t indexBytes = numInner * 2 * sizeof(uint64_t);

    // Read the index from the shard file
    std::vector<uint8_t> indexBuf(indexBytes);
    off_t indexOffset;
    if (indexAtEnd) {
        indexOffset = fileSize - static_cast<off_t>(indexBytes);
    } else {
        indexOffset = 0;
    }

    if (::pread(fd, indexBuf.data(), indexBytes,
                indexOffset) != static_cast<ssize_t>(indexBytes)) {
        throw std::runtime_error(
            "ShardingIndexedCodec: failed to read shard index");
    }

    // Decode index if needed
    if (indexCodecs) {
        ShapeType indexShape = {numInner, 2};
        std::vector<uint8_t> decoded(indexBytes);
        indexCodecs->decode(indexBuf.data(), indexBuf.size(), decoded.data(),
                             indexBytes, indexShape, sizeof(uint64_t));
        indexBuf = std::move(decoded);
    }

    // Compute flat index for the requested inner chunk
    std::size_t flatIdx = 0;
    std::size_t stride = 1;
    for (int d = static_cast<int>(shardGridShape.size()) - 1; d >= 0; --d) {
        flatIdx += innerIdx[d] * stride;
        stride *= shardGridShape[d];
    }

    // Read (offset, nbytes) from index
    uint64_t chunkOffset, chunkNbytes;
    std::memcpy(&chunkOffset, indexBuf.data() + flatIdx * 16, 8);
    std::memcpy(&chunkNbytes, indexBuf.data() + flatIdx * 16 + 8, 8);

    if (chunkOffset == 0xFFFFFFFFFFFFFFFF &&
        chunkNbytes == 0xFFFFFFFFFFFFFFFF) {
        // Empty sentinel — fill with zeros
        std::memset(out, 0, outBytes);
        return false;
    }

    // Compute actual file offset
    off_t dataOffset = static_cast<off_t>(chunkOffset);
    if (!indexAtEnd) {
        dataOffset += static_cast<off_t>(indexBytes);
    }

    // Read compressed inner chunk
    std::vector<uint8_t> compressed(static_cast<std::size_t>(chunkNbytes));
    if (::pread(fd, compressed.data(), compressed.size(),
                dataOffset) != static_cast<ssize_t>(compressed.size())) {
        throw std::runtime_error(
            "ShardingIndexedCodec: failed to read inner chunk data");
    }

    // Decode through inner pipeline
    if (innerCodecs) {
        innerCodecs->decode(compressed.data(), compressed.size(), out, outBytes,
                             innerChunkShape, elemSize);
    } else {
        if (compressed.size() != outBytes) {
            throw std::runtime_error(
                "ShardingIndexedCodec: uncompressed size mismatch");
        }
        std::memcpy(out, compressed.data(), outBytes);
    }

    return true;
}

std::vector<uint8_t> ShardingIndexedCodec::encodeInnerChunk(
    const void* data, std::size_t len, std::size_t elemSize) const
{
    if (!innerCodecs) {
        throw std::runtime_error(
            "ShardingIndexedCodec::encodeInnerChunk: innerCodecs not configured");
    }
    return innerCodecs->encode(data, len, innerChunkShape, elemSize);
}

std::vector<uint8_t> ShardingIndexedCodec::assembleShard(
    const std::vector<std::vector<uint8_t>>& compressedChunks,
    std::size_t numInner) const
{
    // Build index and concatenate compressed data
    std::vector<uint64_t> index(numInner * 2);
    std::vector<uint8_t> shardData;

    for (std::size_t i = 0; i < numInner; ++i) {
        if (i < compressedChunks.size() && !compressedChunks[i].empty()) {
            index[i * 2] = shardData.size();
            index[i * 2 + 1] = compressedChunks[i].size();
            shardData.insert(shardData.end(),
                             compressedChunks[i].begin(),
                             compressedChunks[i].end());
        } else {
            // Empty sentinel
            index[i * 2] = 0xFFFFFFFFFFFFFFFF;
            index[i * 2 + 1] = 0xFFFFFFFFFFFFFFFF;
        }
    }

    // Encode the index
    std::size_t indexBytes = numInner * 2 * sizeof(uint64_t);
    std::vector<uint8_t> indexBuf(indexBytes);
    std::memcpy(indexBuf.data(), index.data(), indexBytes);

    if (indexCodecs) {
        ShapeType indexShape = {numInner, 2};
        auto encodedIndex = indexCodecs->encode(
            indexBuf.data(), indexBytes, indexShape, sizeof(uint64_t));
        indexBuf = std::move(encodedIndex);
    }

    // Assemble: data + index (or index + data)
    std::vector<uint8_t> result;
    result.reserve(shardData.size() + indexBuf.size());
    if (indexAtEnd) {
        result.insert(result.end(), shardData.begin(), shardData.end());
        result.insert(result.end(), indexBuf.begin(), indexBuf.end());
    } else {
        result.insert(result.end(), indexBuf.begin(), indexBuf.end());
        result.insert(result.end(), shardData.begin(), shardData.end());
    }
    return result;
}

std::unique_ptr<ShardingIndexedCodec> ShardingIndexedCodec::fromJson(
    const nlohmann::json& j)
{
    auto c = std::make_unique<ShardingIndexedCodec>();
    if (!j.contains("configuration")) return c;
    auto& cfg = j["configuration"];

    if (cfg.contains("chunk_shape")) {
        c->innerChunkShape = cfg["chunk_shape"].get<ShapeType>();
    }

    if (cfg.contains("codecs")) {
        c->innerCodecs =
            std::make_unique<CodecPipeline>(CodecPipeline::fromV3Json(cfg["codecs"]));
    }

    if (cfg.contains("index_codecs")) {
        c->indexCodecs = std::make_unique<CodecPipeline>(
            CodecPipeline::fromV3Json(cfg["index_codecs"]));
    }

    if (cfg.contains("index_location")) {
        std::string loc = cfg["index_location"].get<std::string>();
        c->indexAtEnd = (loc != "start");
    }

    return c;
}

// ============================================================================
// CodecPipeline
// ============================================================================

CodecPipeline::CodecPipeline() = default;
CodecPipeline::~CodecPipeline() = default;
CodecPipeline::CodecPipeline(CodecPipeline&&) noexcept = default;
CodecPipeline& CodecPipeline::operator=(CodecPipeline&&) noexcept = default;

std::vector<uint8_t> CodecPipeline::encode(const void* data,
                                            std::size_t dataLen,
                                            ShapeType shape,
                                            std::size_t elemSize) const
{
    // Start with a copy of the raw data
    std::vector<uint8_t> buf(dataLen);
    std::memcpy(buf.data(), data, dataLen);

    // 1. Array→array codecs (in order)
    for (auto& codec : arrayToArray) {
        codec->encode(buf, shape, elemSize);
    }

    // 2. Array→bytes codec (exactly one)
    if (arrayToBytes) {
        arrayToBytes->encode(buf, shape, elemSize);
    }

    // 3. Bytes→bytes codecs (in order)
    for (auto& codec : bytesToBytes) {
        auto encoded = codec->encode(buf.data(), buf.size());
        buf = std::move(encoded);
    }

    return buf;
}

void CodecPipeline::decode(const uint8_t* compressed, std::size_t compLen,
                            void* out, std::size_t outLen, ShapeType shape,
                            std::size_t elemSize) const
{
    // Start with the compressed data
    std::vector<uint8_t> buf(compressed, compressed + compLen);

    // Tracking expected size after bytes→bytes decode
    std::size_t expectedAfterB2B = outLen;

    // 3. Bytes→bytes codecs (in reverse order)
    for (auto it = bytesToBytes.rbegin(); it != bytesToBytes.rend(); ++it) {
        auto decoded = (*it)->decode(buf.data(), buf.size(), expectedAfterB2B);
        buf = std::move(decoded);
    }

    // 2. Array→bytes codec (reverse = decode)
    if (arrayToBytes) {
        arrayToBytes->decode(buf, shape, elemSize);
    }

    // 1. Array→array codecs (in reverse order)
    for (auto it = arrayToArray.rbegin(); it != arrayToArray.rend(); ++it) {
        (*it)->decode(buf, shape, elemSize);
    }

    // Copy to output
    if (out && outLen > 0) {
        std::size_t copyLen = std::min(outLen, buf.size());
        std::memcpy(out, buf.data(), copyLen);
    }
}

nlohmann::json CodecPipeline::toJson() const
{
    auto arr = nlohmann::json::array();

    for (auto& c : arrayToArray) {
        arr.push_back(c->toJson());
    }
    if (arrayToBytes) {
        arr.push_back(arrayToBytes->toJson());
    }
    for (auto& c : bytesToBytes) {
        arr.push_back(c->toJson());
    }

    return arr;
}

static std::unique_ptr<Codec> parseOneCodec(const nlohmann::json& j)
{
    std::string codecName = j.value("name", "");

    // Also check "id" for v2 filter compatibility
    if (codecName.empty()) {
        codecName = j.value("id", "");
    }

    if (codecName == "bytes" || codecName == "endian")
        return BytesCodec::fromJson(j);
    if (codecName == "transpose")
        return TransposeCodec::fromJson(j);
    if (codecName == "blosc")
        return BloscCodec::fromJson(j);
    if (codecName == "gzip" || codecName == "zlib")
        return GzipCodec::fromJson(j);
    if (codecName == "zstd")
        return ZstdCodec::fromJson(j);
    if (codecName == "crc32c")
        return Crc32cCodec::fromJson(j);
    if (codecName == "sharding_indexed")
        return ShardingIndexedCodec::fromJson(j);

    throw std::runtime_error("Unknown codec: " + codecName);
}

CodecPipeline CodecPipeline::fromV3Json(const nlohmann::json& codecs)
{
    CodecPipeline pipeline;

    for (auto& j : codecs) {
        auto codec = parseOneCodec(j);
        switch (codec->kind()) {
        case CodecKind::ArrayToArray:
            pipeline.arrayToArray.push_back(
                std::unique_ptr<ArrayToArrayCodec>(
                    static_cast<ArrayToArrayCodec*>(codec.release())));
            break;
        case CodecKind::ArrayToBytes:
            pipeline.arrayToBytes.reset(
                static_cast<ArrayToBytesCodec*>(codec.release()));
            break;
        case CodecKind::BytesToBytes:
            pipeline.bytesToBytes.push_back(
                std::unique_ptr<BytesToBytesCodec>(
                    static_cast<BytesToBytesCodec*>(codec.release())));
            break;
        }
    }

    // Ensure we have an array→bytes codec
    if (!pipeline.arrayToBytes) {
        pipeline.arrayToBytes = std::make_unique<BytesCodec>();
    }

    return pipeline;
}

CodecPipeline CodecPipeline::fromV2(const nlohmann::json& compressor,
                                     const nlohmann::json& filters,
                                     const std::string& /*order*/,
                                     std::size_t elemSize)
{
    CodecPipeline pipeline;

    // Array→bytes: always BytesCodec (C-order, little-endian on x86)
    pipeline.arrayToBytes = std::make_unique<BytesCodec>();

    // Filters: v2 filters are not supported. Throw if any are present
    // so we don't silently produce corrupt data.
    if (filters.is_array() && !filters.empty()) {
        std::string names;
        for (auto& f : filters) {
            if (!names.empty()) names += ", ";
            names += f.value("id", "unknown");
        }
        throw std::runtime_error(
            "Unsupported zarr v2 filters: " + names +
            ". Only compressor-based pipelines are supported.");
    }

    // Compressor → bytes→bytes codec
    if (!compressor.is_null()) {
        std::string id = compressor.value("id", "");
        if (id == "blosc") {
            auto blosc = std::make_unique<BloscCodec>();
            blosc->cname = compressor.value("cname", std::string("zstd"));
            blosc->clevel = compressor.value("clevel", 1);
            blosc->shuffle = compressor.value("shuffle", 0);
            blosc->blocksize = compressor.value("blocksize", 0);
            blosc->typesize = static_cast<int>(elemSize);
            pipeline.bytesToBytes.push_back(std::move(blosc));
        } else if (id == "zlib" || id == "gzip") {
            auto gz = std::make_unique<GzipCodec>();
            gz->level = compressor.value("level", 6);
            pipeline.bytesToBytes.push_back(std::move(gz));
        } else if (id == "zstd") {
            auto zstd = std::make_unique<ZstdCodec>();
            zstd->level = compressor.value("level", 3);
            pipeline.bytesToBytes.push_back(std::move(zstd));
        }
        // null/unknown compressor = no compression
    }

    return pipeline;
}

CodecPipeline CodecPipeline::fromV1(const std::string& compression,
                                     const nlohmann::json& compressionOpts,
                                     const std::string& /*order*/,
                                     std::size_t elemSize)
{
    CodecPipeline pipeline;
    pipeline.arrayToBytes = std::make_unique<BytesCodec>();

    if (compression == "none" || compression.empty()) {
        // No compression
    } else if (compression == "zlib") {
        auto gz = std::make_unique<GzipCodec>();
        if (compressionOpts.is_number()) {
            gz->level = compressionOpts.get<int>();
        }
        pipeline.bytesToBytes.push_back(std::move(gz));
    } else if (compression == "blosc") {
        auto blosc = std::make_unique<BloscCodec>();
        blosc->typesize = static_cast<int>(elemSize);
        if (compressionOpts.is_object()) {
            blosc->cname =
                compressionOpts.value("cname", std::string("lz4"));
            blosc->clevel = compressionOpts.value("clevel", 5);
            blosc->shuffle = compressionOpts.value("shuffle", 1);
            blosc->blocksize = compressionOpts.value("blocksize", 0);
        }
        pipeline.bytesToBytes.push_back(std::move(blosc));
    } else if (compression == "lz4") {
        // v1 lz4 = blosc with lz4
        auto blosc = std::make_unique<BloscCodec>();
        blosc->cname = "lz4";
        blosc->typesize = static_cast<int>(elemSize);
        if (compressionOpts.is_number()) {
            blosc->clevel = compressionOpts.get<int>();
        }
        pipeline.bytesToBytes.push_back(std::move(blosc));
    } else if (compression == "zstd") {
        auto zstd = std::make_unique<ZstdCodec>();
        if (compressionOpts.is_number()) {
            zstd->level = compressionOpts.get<int>();
        }
        pipeline.bytesToBytes.push_back(std::move(zstd));
    } else if (compression == "bz2") {
        // bz2 not supported — throw
        throw std::runtime_error(
            "v1 compression 'bz2' is not supported");
    } else {
        throw std::runtime_error(
            "Unknown v1 compression: " + compression);
    }

    return pipeline;
}

CodecPipeline CodecPipeline::defaultPipeline(std::size_t elemSize)
{
    CodecPipeline pipeline;
    pipeline.arrayToBytes = std::make_unique<BytesCodec>();

    auto blosc = std::make_unique<BloscCodec>();
    blosc->cname = "zstd";
    blosc->clevel = 1;
    blosc->shuffle = 0;
    blosc->typesize = static_cast<int>(elemSize);
    blosc->blocksize = 0;
    pipeline.bytesToBytes.push_back(std::move(blosc));

    return pipeline;
}

bool CodecPipeline::isSharded() const
{
    return dynamic_cast<const ShardingIndexedCodec*>(arrayToBytes.get()) !=
           nullptr;
}

ShapeType CodecPipeline::innerChunkShape() const
{
    auto* sc = dynamic_cast<const ShardingIndexedCodec*>(arrayToBytes.get());
    if (sc) return sc->innerChunkShape;
    return {};
}

const ShardingIndexedCodec* CodecPipeline::shardingCodec() const
{
    return dynamic_cast<const ShardingIndexedCodec*>(arrayToBytes.get());
}

}  // namespace vc::zarr
