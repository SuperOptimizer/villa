#include "utils/zarr.hpp"
#include "utils/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <variant>

#include <zlib.h>
#include <zstd.h>
#include <lz4.h>
#include <bzlib.h>
#include <lzma.h>
#include <blosc.h>

#ifdef UTILS_HAS_VIDEO_CODECS
#include <fcntl.h>
#include <unistd.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#pragma GCC diagnostic pop
#endif

namespace utils {
namespace {

// ── CRC32C (software, lookup table) ────────────────────────────────────────

static constexpr auto make_crc32c_table() -> std::array<uint32_t, 256> {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (crc & 1 ? 0x82F63B78u : 0u);
        t[i] = crc;
    }
    return t;
}

static constexpr auto crc32c_table = make_crc32c_table();

auto compute_crc32c(std::span<const uint8_t> data) -> uint32_t {
    uint32_t crc = 0xFFFFFFFF;
    for (auto b : data)
        crc = crc32c_table[(crc ^ b) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

// ── DType <-> string conversions ────────────────────────────────────────────

auto numpy_dtype_to_dtype(std::string_view s) -> std::expected<DType, std::string> {
    if (s == "<f4" || s == "float32")  return DType::Float32;
    if (s == "<f8" || s == "float64")  return DType::Float64;
    if (s == "<i1" || s == "|i1" || s == "int8")   return DType::Int8;
    if (s == "<i2" || s == "int16")    return DType::Int16;
    if (s == "<i4" || s == "int32")    return DType::Int32;
    if (s == "<i8" || s == "int64")    return DType::Int64;
    if (s == "<u1" || s == "|u1" || s == "uint8")  return DType::UInt8;
    if (s == "<u2" || s == "uint16")   return DType::UInt16;
    if (s == "<u4" || s == "uint32")   return DType::UInt32;
    if (s == "<u8" || s == "uint64")   return DType::UInt64;
    return std::unexpected("unknown dtype: " + std::string(s));
}

auto dtype_to_numpy_str(DType dt) -> std::string {
    switch (dt) {
    case DType::Float32: return "<f4";
    case DType::Float64: return "<f8";
    case DType::Int8:    return "|i1";
    case DType::Int16:   return "<i2";
    case DType::Int32:   return "<i4";
    case DType::Int64:   return "<i8";
    case DType::UInt8:   return "|u1";
    case DType::UInt16:  return "<u2";
    case DType::UInt32:  return "<u4";
    case DType::UInt64:  return "<u8";
    }
    return "<f4";
}

auto dtype_to_v3_str(DType dt) -> std::string {
    return dtype_name(dt);
}

// ── Path normalization helper ───────────────────────────────────────────────

auto normalize_path(std::string_view p) -> std::string {
    while (!p.empty() && p.front() == '/') p.remove_prefix(1);
    while (!p.empty() && p.back() == '/') p.remove_suffix(1);
    return std::string(p);
}

auto join_path(std::string_view base, std::string_view child) -> std::string {
    auto b = normalize_path(base);
    auto c = normalize_path(child);
    if (b.empty()) return c;
    if (c.empty()) return b;
    return b + "/" + c;
}

// ── Codec base and implementations ──────────────────────────────────────────

struct Codec {
    virtual ~Codec() = default;
    virtual auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> = 0;
    virtual auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> = 0;
};

// -- Blosc shuffle helpers --

static auto shuffle_int_to_str(int s) -> std::string {
    switch (s) {
    case 2:  return "bitshuffle";
    case 1:  return "shuffle";
    default: return "noshuffle";
    }
}

static auto shuffle_str_to_int(std::string_view s) -> int {
    if (s == "bitshuffle") return 2;
    if (s == "shuffle") return 1;
    return 0;
}

// -- BloscCodec --

struct BloscCodec : Codec {
    std::string cname;
    int clevel;
    int shuffle;
    int typesize;
    int blocksize;

    BloscCodec(std::string cn, int cl, int sh, int ts, int bs)
        : cname(std::move(cn)), clevel(cl), shuffle(sh), typesize(ts), blocksize(bs) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        int ts = typesize > 0 ? typesize : 1;
        size_t dest_size = src.size() + BLOSC_MAX_OVERHEAD;
        std::vector<uint8_t> dest(dest_size);
        int csize = blosc_compress_ctx(clevel, shuffle, static_cast<size_t>(ts),
                                       src.size(), src.data(), dest.data(), dest_size,
                                       cname.c_str(), static_cast<size_t>(blocksize), 1);
        if (csize <= 0) return std::unexpected("blosc: compression failed");
        dest.resize(static_cast<size_t>(csize));
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        size_t nbytes, cbytes, blocksize_out;
        blosc_cbuffer_sizes(src.data(), &nbytes, &cbytes, &blocksize_out);
        if (nbytes == 0) return std::unexpected("blosc: invalid compressed buffer");
        std::vector<uint8_t> dest(nbytes);
        int dsize = blosc_decompress_ctx(src.data(), dest.data(), dest.size(), 1);
        if (dsize < 0) return std::unexpected("blosc: decompression failed");
        dest.resize(static_cast<size_t>(dsize));
        return dest;
    }
};

// -- GzipCodec --

struct GzipCodec : Codec {
    int level;
    GzipCodec(int lvl) : level(lvl) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        z_stream strm{};
        if (deflateInit2(&strm, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK)
            return std::unexpected("gzip: deflateInit2 failed");
        size_t bound = deflateBound(&strm, static_cast<uLong>(src.size()));
        std::vector<uint8_t> dest(bound);
        strm.next_in = const_cast<Bytef*>(src.data());
        strm.avail_in = static_cast<uInt>(src.size());
        strm.next_out = dest.data();
        strm.avail_out = static_cast<uInt>(dest.size());
        int ret = deflate(&strm, Z_FINISH);
        if (ret != Z_STREAM_END) { deflateEnd(&strm); return std::unexpected("gzip: deflate failed"); }
        dest.resize(strm.total_out);
        deflateEnd(&strm);
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        z_stream strm{};
        if (inflateInit2(&strm, 15 + 32) != Z_OK)
            return std::unexpected("gzip: inflateInit2 failed");
        std::vector<uint8_t> dest;
        dest.resize(src.size() * 4);
        strm.next_in = const_cast<Bytef*>(src.data());
        strm.avail_in = static_cast<uInt>(src.size());
        size_t total = 0;
        int ret;
        do {
            if (total >= dest.size()) dest.resize(dest.size() * 2);
            strm.next_out = dest.data() + total;
            strm.avail_out = static_cast<uInt>(dest.size() - total);
            ret = inflate(&strm, Z_NO_FLUSH);
            if (ret != Z_OK && ret != Z_STREAM_END) {
                inflateEnd(&strm);
                return std::unexpected("gzip: inflate failed: " + std::to_string(ret));
            }
            total = strm.total_out;
        } while (ret != Z_STREAM_END);
        inflateEnd(&strm);
        dest.resize(total);
        return dest;
    }
};

// -- ZstdCodec --

struct ZstdCodec : Codec {
    int level;
    ZstdCodec(int lvl) : level(lvl) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        size_t bound = ZSTD_compressBound(src.size());
        std::vector<uint8_t> dest(bound);
        size_t csize = ZSTD_compress(dest.data(), dest.size(), src.data(), src.size(), level);
        if (ZSTD_isError(csize))
            return std::unexpected("zstd: compression failed: " + std::string(ZSTD_getErrorName(csize)));
        dest.resize(csize);
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        auto content_size = ZSTD_getFrameContentSize(src.data(), src.size());
        if (content_size == ZSTD_CONTENTSIZE_ERROR)
            return std::unexpected("zstd: not a valid frame");
        if (content_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            // Streaming decompress with growing buffer
            std::vector<uint8_t> dest(src.size() * 4);
            ZSTD_DCtx* dctx = ZSTD_createDCtx();
            if (!dctx) return std::unexpected("zstd: failed to create context");
            ZSTD_inBuffer in{src.data(), src.size(), 0};
            size_t total = 0;
            while (in.pos < in.size) {
                if (total >= dest.size()) dest.resize(dest.size() * 2);
                ZSTD_outBuffer out{dest.data() + total, dest.size() - total, 0};
                size_t ret = ZSTD_decompressStream(dctx, &out, &in);
                if (ZSTD_isError(ret)) {
                    ZSTD_freeDCtx(dctx);
                    return std::unexpected("zstd: decompress stream failed");
                }
                total += out.pos;
            }
            ZSTD_freeDCtx(dctx);
            dest.resize(total);
            return dest;
        }
        std::vector<uint8_t> dest(content_size);
        size_t dsize = ZSTD_decompress(dest.data(), dest.size(), src.data(), src.size());
        if (ZSTD_isError(dsize))
            return std::unexpected("zstd: decompression failed: " + std::string(ZSTD_getErrorName(dsize)));
        dest.resize(dsize);
        return dest;
    }
};

// -- Lz4Codec (numcodecs format: 4-byte LE original size prefix + LZ4 block) --

struct Lz4Codec : Codec {
    int acceleration;
    Lz4Codec(int accel) : acceleration(accel) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        int max_dst = LZ4_compressBound(static_cast<int>(src.size()));
        std::vector<uint8_t> dest(4 + static_cast<size_t>(max_dst));
        // Write 4-byte LE original size
        uint32_t orig_size = static_cast<uint32_t>(src.size());
        std::memcpy(dest.data(), &orig_size, 4);
        int csize = LZ4_compress_default(reinterpret_cast<const char*>(src.data()),
                                         reinterpret_cast<char*>(dest.data() + 4),
                                         static_cast<int>(src.size()), max_dst);
        if (csize <= 0) return std::unexpected("lz4: compression failed");
        dest.resize(4 + static_cast<size_t>(csize));
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        if (src.size() < 4) return std::unexpected("lz4: input too small");
        uint32_t orig_size;
        std::memcpy(&orig_size, src.data(), 4);
        std::vector<uint8_t> dest(orig_size);
        int dsize = LZ4_decompress_safe(reinterpret_cast<const char*>(src.data() + 4),
                                        reinterpret_cast<char*>(dest.data()),
                                        static_cast<int>(src.size() - 4),
                                        static_cast<int>(orig_size));
        if (dsize < 0) return std::unexpected("lz4: decompression failed");
        dest.resize(static_cast<size_t>(dsize));
        return dest;
    }
};

// -- Bz2Codec --

struct Bz2Codec : Codec {
    int level;
    Bz2Codec(int lvl) : level(lvl) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        unsigned int dest_len = static_cast<unsigned int>(src.size() + src.size() / 100 + 600);
        std::vector<uint8_t> dest(dest_len);
        int ret = BZ2_bzBuffToBuffCompress(reinterpret_cast<char*>(dest.data()), &dest_len,
                                           const_cast<char*>(reinterpret_cast<const char*>(src.data())),
                                           static_cast<unsigned int>(src.size()), level, 0, 30);
        if (ret != BZ_OK) return std::unexpected("bz2: compression failed: " + std::to_string(ret));
        dest.resize(dest_len);
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        unsigned int dest_len = static_cast<unsigned int>(src.size() * 2);
        if (dest_len < 4096) dest_len = 4096;
        std::vector<uint8_t> dest(dest_len);
        int ret;
        while (true) {
            dest_len = static_cast<unsigned int>(dest.size());
            ret = BZ2_bzBuffToBuffDecompress(reinterpret_cast<char*>(dest.data()), &dest_len,
                                             const_cast<char*>(reinterpret_cast<const char*>(src.data())),
                                             static_cast<unsigned int>(src.size()), 0, 0);
            if (ret == BZ_OK) break;
            if (ret == BZ_OUTBUFF_FULL) {
                dest.resize(dest.size() * 2);
                continue;
            }
            return std::unexpected("bz2: decompression failed: " + std::to_string(ret));
        }
        dest.resize(dest_len);
        return dest;
    }
};

// -- LzmaCodec --

struct LzmaCodec : Codec {
    int preset;
    LzmaCodec(int p) : preset(p) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        size_t dest_size = lzma_stream_buffer_bound(src.size());
        std::vector<uint8_t> dest(dest_size);
        size_t out_pos = 0;
        lzma_ret ret = lzma_easy_buffer_encode(static_cast<uint32_t>(preset), LZMA_CHECK_CRC64,
                                               nullptr, src.data(), src.size(),
                                               dest.data(), &out_pos, dest.size());
        if (ret != LZMA_OK) return std::unexpected("lzma: encode failed: " + std::to_string(ret));
        dest.resize(out_pos);
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        size_t memlimit = UINT64_MAX;
        size_t in_pos = 0;
        // Try growing buffer approach
        size_t dest_size = src.size() * 4;
        if (dest_size < 4096) dest_size = 4096;
        std::vector<uint8_t> dest(dest_size);
        size_t out_pos = 0;
        while (true) {
            lzma_ret ret = lzma_stream_buffer_decode(&memlimit, 0, nullptr,
                                                     src.data(), &in_pos, src.size(),
                                                     dest.data(), &out_pos, dest.size());
            if (ret == LZMA_OK) break;
            if (ret == LZMA_BUF_ERROR || out_pos >= dest.size() - 1) {
                dest.resize(dest.size() * 2);
                in_pos = 0;
                out_pos = 0;
                memlimit = UINT64_MAX;
                continue;
            }
            return std::unexpected("lzma: decode failed: " + std::to_string(ret));
        }
        dest.resize(out_pos);
        return dest;
    }
};

// -- Crc32cCodec (appends/verifies 4-byte LE CRC32C) --

struct Crc32cCodec : Codec {
    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        std::vector<uint8_t> dest(src.begin(), src.end());
        uint32_t crc = compute_crc32c(src);
        dest.push_back(static_cast<uint8_t>(crc & 0xFF));
        dest.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));
        dest.push_back(static_cast<uint8_t>((crc >> 16) & 0xFF));
        dest.push_back(static_cast<uint8_t>((crc >> 24) & 0xFF));
        return dest;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        if (src.size() < 4) return std::unexpected("crc32c: input too small");
        auto data = src.subspan(0, src.size() - 4);
        uint32_t expected_crc;
        std::memcpy(&expected_crc, src.data() + src.size() - 4, 4);
        uint32_t actual_crc = compute_crc32c(data);
        if (actual_crc != expected_crc)
            return std::unexpected("crc32c: checksum mismatch");
        return std::vector<uint8_t>(data.begin(), data.end());
    }
};

// -- BytesCodec (passthrough for little-endian host) --

struct BytesCodec : Codec {
    std::string endian;
    BytesCodec(std::string e) : endian(std::move(e)) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        return std::vector<uint8_t>(src.begin(), src.end());
    }
    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        return std::vector<uint8_t>(src.begin(), src.end());
    }
};

// -- TransposeCodec (stub: passthrough) --

struct TransposeCodec : Codec {
    std::vector<size_t> order;
    TransposeCodec(std::vector<size_t> o) : order(std::move(o)) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        return std::vector<uint8_t>(src.begin(), src.end());
    }
    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        return std::vector<uint8_t>(src.begin(), src.end());
    }
};

// -- Sharding helpers --

static auto product(std::span<const size_t> v) -> size_t {
    size_t p = 1;
    for (auto x : v) p *= x;
    return p;
}

static auto compute_chunks_per_dim(const std::vector<size_t>& outer,
                                    const std::vector<size_t>& inner)
    -> std::vector<size_t> {
    std::vector<size_t> result(outer.size());
    for (size_t i = 0; i < outer.size(); ++i)
        result[i] = (outer[i] + inner[i] - 1) / inner[i];
    return result;
}

static auto linear_to_coord(size_t linear, const std::vector<size_t>& dims)
    -> std::vector<size_t> {
    std::vector<size_t> coord(dims.size());
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
        coord[d] = linear % dims[d];
        linear /= dims[d];
    }
    return coord;
}

static auto coord_to_linear(const std::vector<size_t>& coord,
                             const std::vector<size_t>& shape) -> size_t {
    size_t idx = 0;
    size_t stride = 1;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        idx += coord[d] * stride;
        stride *= shape[d];
    }
    return idx;
}

static auto extract_inner_chunk(const uint8_t* shard_data,
                                 const std::vector<size_t>& shard_shape,
                                 const std::vector<size_t>& chunk_shape,
                                 const std::vector<size_t>& grid_pos,
                                 size_t elem_size) -> std::vector<uint8_t> {
    size_t chunk_numel = product(chunk_shape);
    std::vector<uint8_t> out(chunk_numel * elem_size);
    size_t ndim = shard_shape.size();
    std::vector<size_t> local(ndim, 0);
    for (size_t i = 0; i < chunk_numel; ++i) {
        std::vector<size_t> shard_coord(ndim);
        for (size_t d = 0; d < ndim; ++d)
            shard_coord[d] = grid_pos[d] * chunk_shape[d] + local[d];
        size_t src = coord_to_linear(shard_coord, shard_shape);
        std::memcpy(out.data() + i * elem_size,
                     shard_data + src * elem_size, elem_size);
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++local[d] < chunk_shape[d]) break;
            local[d] = 0;
        }
    }
    return out;
}

static void insert_inner_chunk(uint8_t* shard_data,
                                const std::vector<size_t>& shard_shape,
                                const std::vector<size_t>& chunk_shape,
                                const std::vector<size_t>& grid_pos,
                                const uint8_t* chunk_data,
                                size_t elem_size) {
    size_t chunk_numel = product(chunk_shape);
    size_t ndim = shard_shape.size();
    std::vector<size_t> local(ndim, 0);
    for (size_t i = 0; i < chunk_numel; ++i) {
        std::vector<size_t> shard_coord(ndim);
        for (size_t d = 0; d < ndim; ++d)
            shard_coord[d] = grid_pos[d] * chunk_shape[d] + local[d];
        size_t dst = coord_to_linear(shard_coord, shard_shape);
        std::memcpy(shard_data + dst * elem_size,
                     chunk_data + i * elem_size, elem_size);
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++local[d] < chunk_shape[d]) break;
            local[d] = 0;
        }
    }
}

// ── Codec pipeline (moved before ShardingCodec so it can be stored by value) ──

struct CodecPipeline {
    std::vector<std::unique_ptr<Codec>> codecs;

    auto encode(std::span<const uint8_t> data) -> std::expected<std::vector<uint8_t>, std::string> {
        std::vector<uint8_t> buf(data.begin(), data.end());
        for (auto& c : codecs) {
            auto result = c->encode(buf);
            if (!result) return std::unexpected(result.error());
            buf = std::move(*result);
        }
        return buf;
    }

    auto decode(std::span<const uint8_t> data) -> std::expected<std::vector<uint8_t>, std::string> {
        std::vector<uint8_t> buf(data.begin(), data.end());
        for (auto it = codecs.rbegin(); it != codecs.rend(); ++it) {
            auto result = (*it)->decode(buf);
            if (!result) return std::unexpected(result.error());
            buf = std::move(*result);
        }
        return buf;
    }
};

// Forward declarations for codec factory (needed by ShardingCodec)
auto make_codec(const CodecConfig& cfg,
                std::span<const size_t> chunk_shape = {},
                size_t elem_size = 0) -> std::unique_ptr<Codec>;
auto build_pipeline(std::span<const CodecConfig> cfgs,
                    std::span<const size_t> chunk_shape = {},
                    size_t elem_size = 0) -> CodecPipeline;

// -- ShardingCodec (full implementation) --

struct ShardingCodec : Codec {
    std::vector<size_t> outer_shape;  // shard shape (= array's chunk_shape)
    std::vector<size_t> inner_shape;  // inner chunk shape
    size_t elem_size;
    CodecPipeline inner_pipeline;
    CodecPipeline index_pipeline;

    ShardingCodec(std::vector<size_t> outer, std::vector<size_t> inner,
                  size_t esz, CodecPipeline ip, CodecPipeline idxp)
        : outer_shape(std::move(outer)), inner_shape(std::move(inner)),
          elem_size(esz), inner_pipeline(std::move(ip)), index_pipeline(std::move(idxp)) {}

    auto encode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        auto grid_dims = compute_chunks_per_dim(outer_shape, inner_shape);
        size_t total_chunks = product(grid_dims);

        std::vector<uint8_t> output;
        std::vector<std::pair<uint64_t, uint64_t>> index(total_chunks);

        for (size_t i = 0; i < total_chunks; ++i) {
            auto grid_pos = linear_to_coord(i, grid_dims);
            auto raw_chunk = extract_inner_chunk(src.data(), outer_shape, inner_shape,
                                                  grid_pos, elem_size);
            auto encoded = inner_pipeline.encode(raw_chunk);
            if (!encoded) return std::unexpected(encoded.error());

            uint64_t offset = output.size();
            uint64_t nbytes = encoded->size();
            index[i] = {offset, nbytes};

            output.insert(output.end(), encoded->begin(), encoded->end());
        }

        // Build raw index: array of (offset, nbytes) uint64 LE pairs
        std::vector<uint8_t> raw_index(total_chunks * 16);
        for (size_t i = 0; i < total_chunks; ++i) {
            uint64_t off = index[i].first;
            uint64_t nb = index[i].second;
            std::memcpy(raw_index.data() + i * 16, &off, 8);
            std::memcpy(raw_index.data() + i * 16 + 8, &nb, 8);
        }

        // Encode index with index_pipeline
        auto encoded_index = index_pipeline.encode(raw_index);
        if (!encoded_index) return std::unexpected(encoded_index.error());

        output.insert(output.end(), encoded_index->begin(), encoded_index->end());
        return output;
    }

    auto decode(std::span<const uint8_t> src) -> std::expected<std::vector<uint8_t>, std::string> override {
        auto grid_dims = compute_chunks_per_dim(outer_shape, inner_shape);
        size_t total_chunks = product(grid_dims);
        size_t raw_index_size = total_chunks * 16;

        // Determine encoded index size by encoding a dummy raw index
        std::vector<uint8_t> dummy_index(raw_index_size, 0);
        auto dummy_encoded = index_pipeline.encode(dummy_index);
        if (!dummy_encoded) return std::unexpected(dummy_encoded.error());
        size_t encoded_index_size = dummy_encoded->size();

        if (src.size() < encoded_index_size)
            return std::unexpected("shard too small to contain index");

        // Read and decode the index from the end of the shard
        auto index_start = src.size() - encoded_index_size;
        auto encoded_idx_span = src.subspan(index_start, encoded_index_size);
        auto raw_index = index_pipeline.decode(encoded_idx_span);
        if (!raw_index) return std::unexpected(raw_index.error());

        if (raw_index->size() != raw_index_size)
            return std::unexpected("decoded index has unexpected size");

        // Parse index entries
        constexpr uint64_t sentinel = 0xFFFFFFFFFFFFFFFFULL;
        std::vector<std::pair<uint64_t, uint64_t>> index(total_chunks);
        for (size_t i = 0; i < total_chunks; ++i) {
            uint64_t off, nb;
            std::memcpy(&off, raw_index->data() + i * 16, 8);
            std::memcpy(&nb, raw_index->data() + i * 16 + 8, 8);
            index[i] = {off, nb};
        }

        // Allocate output buffer
        size_t shard_numel = product(outer_shape);
        std::vector<uint8_t> output(shard_numel * elem_size, 0);

        // Decode each inner chunk
        for (size_t i = 0; i < total_chunks; ++i) {
            auto [off, nb] = index[i];
            if (off == sentinel && nb == sentinel) continue; // empty chunk
            if (off + nb > index_start)
                return std::unexpected("inner chunk extends past index");

            auto chunk_span = src.subspan(static_cast<size_t>(off), static_cast<size_t>(nb));
            auto decoded = inner_pipeline.decode(chunk_span);
            if (!decoded) return std::unexpected(decoded.error());

            auto grid_pos = linear_to_coord(i, grid_dims);
            insert_inner_chunk(output.data(), outer_shape, inner_shape,
                               grid_pos, decoded->data(), elem_size);
        }

        return output;
    }
};

// -- VideoCodec --

#ifdef UTILS_HAS_VIDEO_CODECS

struct AVCodecContextDeleter {
    void operator()(AVCodecContext* ctx) const {
        if (ctx) avcodec_free_context(&ctx);
    }
};
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;

struct AVFrameDeleter {
    void operator()(AVFrame* f) const {
        if (f) av_frame_free(&f);
    }
};
using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

struct AVPacketDeleter {
    void operator()(AVPacket* p) const {
        if (p) av_packet_free(&p);
    }
};
using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;

struct SwsContextDeleter {
    void operator()(SwsContext* ctx) const {
        if (ctx) sws_freeContext(ctx);
    }
};
using SwsContextPtr = std::unique_ptr<SwsContext, SwsContextDeleter>;

struct AVParserDeleter {
    void operator()(AVCodecParserContext* p) const {
        if (p) av_parser_close(p);
    }
};
using AVParserPtr = std::unique_ptr<AVCodecParserContext, AVParserDeleter>;

static void ensure_ffmpeg_quiet() {
    static const bool once = (av_log_set_level(AV_LOG_ERROR), true);
    (void)once;
}

// RAII guard to suppress stderr (SVT-AV1/x265 use their own log systems)
struct StderrSuppressor {
    int saved_fd = -1;
    StderrSuppressor() {
        fflush(stderr);
        saved_fd = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
    }
    ~StderrSuppressor() {
        if (saved_fd >= 0) {
            fflush(stderr);
            dup2(saved_fd, STDERR_FILENO);
            close(saved_fd);
        }
    }
    StderrSuppressor(const StderrSuppressor&) = delete;
    auto operator=(const StderrSuppressor&) -> StderrSuppressor& = delete;
};

static auto video_codec_id_to_av(CodecId id) -> AVCodecID {
    switch (id) {
    case CodecId::H264: return AV_CODEC_ID_H264;
    case CodecId::H265: return AV_CODEC_ID_HEVC;
    case CodecId::AV1:  return AV_CODEC_ID_AV1;
    default: return AV_CODEC_ID_NONE;
    }
}

static auto find_video_encoder(CodecId id) -> const AVCodec* {
    const AVCodec* codec = nullptr;
    switch (id) {
    case CodecId::H264:
        codec = avcodec_find_encoder_by_name("libx264");
        break;
    case CodecId::H265:
        codec = avcodec_find_encoder_by_name("libx265");
        break;
    case CodecId::AV1:
        codec = avcodec_find_encoder_by_name("libsvtav1");
        if (!codec) codec = avcodec_find_encoder_by_name("libaom-av1");
        break;
    default:
        break;
    }
    if (!codec) codec = avcodec_find_encoder(video_codec_id_to_av(id));
    return codec;
}

static auto pick_pix_fmt(const AVCodec* codec, int bit_depth) -> AVPixelFormat {
    auto gray = (bit_depth == 10) ? AV_PIX_FMT_GRAY10LE : AV_PIX_FMT_GRAY8;
    if (codec->pix_fmts) {
        for (auto p = codec->pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
            if (*p == gray) return gray;
        }
    }
    return (bit_depth == 10) ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
}

// Binary header prepended to each encoded chunk (UInt8-only, no normalization)
// [28 bytes fixed header][bitstream]
//   magic "VCHK" (4), version (4), frame_count (4), width (4), height (4),
//   bit_depth (4), bitstream_size (4)
static constexpr size_t VIDEO_HEADER_SIZE = 28;
static constexpr uint32_t VIDEO_MAGIC = 0x4B484356; // "VCHK" little-endian

static void write_le32(uint8_t* dst, uint32_t v) { std::memcpy(dst, &v, 4); }
static auto read_le32(const uint8_t* src) -> uint32_t { uint32_t v; std::memcpy(&v, src, 4); return v; }

struct VideoCodec : Codec {
    CodecId codec_id;
    std::vector<size_t> chunk_shape; // [Z, Y, X]
    int crf;
    std::string preset;
    int bit_depth;

    VideoCodec(CodecId id, std::vector<size_t> shape,
               int crf_val, std::string preset_str, int bdepth)
        : codec_id(id), chunk_shape(std::move(shape)),
          crf(crf_val), preset(std::move(preset_str)), bit_depth(bdepth) {}

    auto encode(std::span<const uint8_t> src)
        -> std::expected<std::vector<uint8_t>, std::string> override {
        ensure_ffmpeg_quiet();
        if (chunk_shape.size() != 3)
            return std::unexpected("video codec requires exactly 3D chunk shape");

        auto num_frames = static_cast<int>(chunk_shape[0]);
        auto height = static_cast<int>(chunk_shape[1]);
        auto width = static_cast<int>(chunk_shape[2]);
        size_t frame_pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
        size_t expected_size = static_cast<size_t>(num_frames) * frame_pixels;
        if (src.size() != expected_size)
            return std::unexpected("video codec: input size mismatch (expected " +
                                   std::to_string(expected_size) + ", got " +
                                   std::to_string(src.size()) + ")");

        // Find encoder
        const AVCodec* codec = find_video_encoder(codec_id);
        if (!codec)
            return std::unexpected("video codec: encoder not found");

        // Allocate context
        AVCodecContextPtr ctx(avcodec_alloc_context3(codec));
        if (!ctx)
            return std::unexpected("video codec: failed to allocate encoder context");

        auto pix_fmt = pick_pix_fmt(codec, bit_depth);
        ctx->width = width;
        ctx->height = height;
        ctx->pix_fmt = pix_fmt;
        ctx->time_base = {1, 25};
        ctx->gop_size = num_frames;
        ctx->max_b_frames = 0;
        // No GLOBAL_HEADER — SPS/PPS/VPS will be inline in the bitstream

        // Set CRF and preset via private options
        auto crf_str = std::to_string(crf);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
        if (codec_id == CodecId::AV1) {
            // SVT-AV1 uses "preset" for speed, libaom uses "cpu-used"
            if (std::string_view(codec->name) == "libsvtav1") {
                av_opt_set(ctx->priv_data, "preset", preset.c_str(), 0);
                av_opt_set(ctx->priv_data, "crf", crf_str.c_str(), 0);
                av_opt_set(ctx->priv_data, "svtav1-params", "loglevel=-2", 0);
            } else {
                av_opt_set(ctx->priv_data, "cpu-used", preset.c_str(), 0);
                av_opt_set(ctx->priv_data, "crf", crf_str.c_str(), 0);
            }
        } else if (codec_id == CodecId::H265) {
            av_opt_set(ctx->priv_data, "preset", preset.c_str(), 0);
            av_opt_set(ctx->priv_data, "crf", crf_str.c_str(), 0);
            av_opt_set(ctx->priv_data, "x265-params", "log-level=-1", 0);
        } else {
            av_opt_set(ctx->priv_data, "preset", preset.c_str(), 0);
            av_opt_set(ctx->priv_data, "crf", crf_str.c_str(), 0);
        }
#pragma GCC diagnostic pop

        int ret;
        { StderrSuppressor quiet; ret = avcodec_open2(ctx.get(), codec, nullptr); }
        if (ret < 0)
            return std::unexpected("video codec: failed to open encoder (error " +
                                   std::to_string(ret) + ")");

        // Determine if we need swscale (grayscale -> YUV conversion)
        bool need_sws = (pix_fmt == AV_PIX_FMT_YUV420P || pix_fmt == AV_PIX_FMT_YUV420P10LE);
        auto src_fmt = (bit_depth == 10) ? AV_PIX_FMT_GRAY10LE : AV_PIX_FMT_GRAY8;
        SwsContextPtr sws_ctx;
        if (need_sws) {
            sws_ctx.reset(sws_getContext(width, height, src_fmt,
                                         width, height, pix_fmt,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr));
            if (!sws_ctx)
                return std::unexpected("video codec: failed to create swscale context");
        }

        // Allocate frame and packet
        AVFramePtr frame(av_frame_alloc());
        if (!frame)
            return std::unexpected("video codec: failed to allocate frame");
        frame->format = pix_fmt;
        frame->width = width;
        frame->height = height;
        ret = av_frame_get_buffer(frame.get(), 0);
        if (ret < 0)
            return std::unexpected("video codec: failed to allocate frame buffer");

        AVPacketPtr pkt(av_packet_alloc());
        if (!pkt)
            return std::unexpected("video codec: failed to allocate packet");

        // Encode all frames
        std::vector<uint8_t> bitstream;
        bitstream.reserve(src.size() / 2); // rough estimate

        auto drain_packets = [&]() -> std::expected<void, std::string> {
            while (true) {
                ret = avcodec_receive_packet(ctx.get(), pkt.get());
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0)
                    return std::unexpected("video codec: receive_packet failed");
                bitstream.insert(bitstream.end(), pkt->data, pkt->data + pkt->size);
                av_packet_unref(pkt.get());
            }
            return {};
        };

        for (int i = 0; i < num_frames; ++i) {
            ret = av_frame_make_writable(frame.get());
            if (ret < 0)
                return std::unexpected("video codec: frame not writable");

            const uint8_t* frame_data = src.data() + static_cast<size_t>(i) * frame_pixels;

            if (need_sws) {
                // Convert grayscale to YUV
                const uint8_t* src_slices[1] = {frame_data};
                int src_strides[1] = {width};
                sws_scale(sws_ctx.get(), src_slices, src_strides, 0, height,
                          frame->data, frame->linesize);
            } else {
                // Direct copy to Y plane (GRAY8)
                for (int y = 0; y < height; ++y)
                    std::memcpy(frame->data[0] + y * frame->linesize[0],
                                frame_data + y * width,
                                static_cast<size_t>(width));
            }

            frame->pts = i;
            ret = avcodec_send_frame(ctx.get(), frame.get());
            if (ret < 0)
                return std::unexpected("video codec: send_frame failed");
            auto dr = drain_packets();
            if (!dr) return std::unexpected(dr.error());
        }

        // Flush encoder
        ret = avcodec_send_frame(ctx.get(), nullptr);
        if (ret < 0 && ret != AVERROR_EOF)
            return std::unexpected("video codec: flush send_frame failed");
        auto dr = drain_packets();
        if (!dr) return std::unexpected(dr.error());

        // Build output: [header][bitstream]
        size_t total = VIDEO_HEADER_SIZE + bitstream.size();
        std::vector<uint8_t> output(total);

        // Write header
        write_le32(output.data() + 0, VIDEO_MAGIC);
        write_le32(output.data() + 4, 1); // version
        write_le32(output.data() + 8, static_cast<uint32_t>(num_frames));
        write_le32(output.data() + 12, static_cast<uint32_t>(width));
        write_le32(output.data() + 16, static_cast<uint32_t>(height));
        write_le32(output.data() + 20, static_cast<uint32_t>(bit_depth));
        write_le32(output.data() + 24, static_cast<uint32_t>(bitstream.size()));

        // Write bitstream (SPS/PPS/VPS are inline)
        std::memcpy(output.data() + VIDEO_HEADER_SIZE,
                     bitstream.data(), bitstream.size());

        return output;
    }

    auto decode(std::span<const uint8_t> src)
        -> std::expected<std::vector<uint8_t>, std::string> override {
        ensure_ffmpeg_quiet();
        if (src.size() < VIDEO_HEADER_SIZE)
            return std::unexpected("video codec: data too small for header");

        // Parse header
        uint32_t magic = read_le32(src.data() + 0);
        if (magic != VIDEO_MAGIC)
            return std::unexpected("video codec: invalid magic");
        uint32_t version = read_le32(src.data() + 4);
        if (version != 1)
            return std::unexpected("video codec: unsupported header version " +
                                   std::to_string(version));
        uint32_t num_frames = read_le32(src.data() + 8);
        uint32_t width = read_le32(src.data() + 12);
        uint32_t height = read_le32(src.data() + 16);
        uint32_t hdr_bit_depth = read_le32(src.data() + 20);
        uint32_t bitstream_size = read_le32(src.data() + 24);

        size_t expected_total = VIDEO_HEADER_SIZE + bitstream_size;
        if (src.size() < expected_total)
            return std::unexpected("video codec: truncated data");

        const uint8_t* bitstream_ptr = src.data() + VIDEO_HEADER_SIZE;

        // Find decoder
        const AVCodec* codec = avcodec_find_decoder(video_codec_id_to_av(codec_id));
        if (!codec)
            return std::unexpected("video codec: decoder not found");

        AVCodecContextPtr ctx(avcodec_alloc_context3(codec));
        if (!ctx)
            return std::unexpected("video codec: failed to allocate decoder context");

        int ret;
        { StderrSuppressor quiet; ret = avcodec_open2(ctx.get(), codec, nullptr); }
        if (ret < 0)
            return std::unexpected("video codec: failed to open decoder (error " +
                                   std::to_string(ret) + ")");

        (void)hdr_bit_depth; // reserved for future use

        // Allocate frame and packet
        AVFramePtr frame(av_frame_alloc());
        AVPacketPtr pkt(av_packet_alloc());
        if (!frame || !pkt)
            return std::unexpected("video codec: allocation failed");

        size_t frame_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        size_t total_pixels = static_cast<size_t>(num_frames) * frame_pixels;
        std::vector<uint8_t> output(total_pixels);
        size_t frame_idx = 0;

        auto collect_frames = [&]() -> std::expected<void, std::string> {
            while (true) {
                ret = avcodec_receive_frame(ctx.get(), frame.get());
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0)
                    return std::unexpected("video codec: receive_frame failed");

                if (frame_idx >= num_frames)
                    return std::unexpected("video codec: too many decoded frames");

                uint8_t* dst = output.data() + frame_idx * frame_pixels;

                // Extract Y (luma) plane directly — works for GRAY8, YUV420P,
                // YUV444P, etc. The Y plane always contains the grayscale values.
                for (int y = 0; y < static_cast<int>(height); ++y)
                    std::memcpy(dst + static_cast<size_t>(y) * width,
                                frame->data[0] + y * frame->linesize[0],
                                width);
                ++frame_idx;
            }
            return {};
        };

        // Use parser to split bitstream into packets
        AVParserPtr parser(av_parser_init(static_cast<int>(codec->id)));
        if (!parser)
            return std::unexpected("video codec: failed to init parser");

        size_t pos = 0;
        while (pos < bitstream_size) {
            uint8_t* out_data = nullptr;
            int out_size = 0;
            int consumed = av_parser_parse2(parser.get(), ctx.get(),
                &out_data, &out_size,
                bitstream_ptr + pos, static_cast<int>(bitstream_size - pos),
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (consumed < 0)
                return std::unexpected("video codec: parser failed");
            pos += static_cast<size_t>(consumed);
            if (out_size > 0) {
                pkt->data = out_data;
                pkt->size = out_size;
                ret = avcodec_send_packet(ctx.get(), pkt.get());
                if (ret < 0 && ret != AVERROR(EAGAIN))
                    return std::unexpected("video codec: send_packet failed");
                auto cf = collect_frames();
                if (!cf) return std::unexpected(cf.error());
            }
        }

        // Flush parser
        {
            uint8_t* out_data = nullptr;
            int out_size = 0;
            av_parser_parse2(parser.get(), ctx.get(),
                &out_data, &out_size,
                nullptr, 0,
                AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if (out_size > 0) {
                pkt->data = out_data;
                pkt->size = out_size;
                ret = avcodec_send_packet(ctx.get(), pkt.get());
                if (ret >= 0 || ret == AVERROR(EAGAIN)) {
                    auto cf = collect_frames();
                    if (!cf) return std::unexpected(cf.error());
                }
            }
        }

        // Flush decoder
        ret = avcodec_send_packet(ctx.get(), nullptr);
        if (ret >= 0 || ret == AVERROR_EOF) {
            auto cf = collect_frames();
            if (!cf) return std::unexpected(cf.error());
        }

        if (frame_idx != num_frames)
            return std::unexpected("video codec: decoded " + std::to_string(frame_idx) +
                                   " frames, expected " + std::to_string(num_frames));

        return output;
    }
};

#endif // UTILS_HAS_VIDEO_CODECS

// -- VideoCodecStub (when FFmpeg not available) --

#ifndef UTILS_HAS_VIDEO_CODECS
struct VideoCodecStub : Codec {
    auto encode(std::span<const uint8_t>) -> std::expected<std::vector<uint8_t>, std::string> override {
        return std::unexpected(std::string("video codecs require UTILS_ENABLE_VIDEO_CODECS=ON (FFmpeg)"));
    }
    auto decode(std::span<const uint8_t>) -> std::expected<std::vector<uint8_t>, std::string> override {
        return std::unexpected(std::string("video codecs require UTILS_ENABLE_VIDEO_CODECS=ON (FFmpeg)"));
    }
};
#endif

// ── Codec factory and pipeline ──────────────────────────────────────────────

auto make_codec(const CodecConfig& cfg,
                std::span<const size_t> chunk_shape,
                size_t elem_size) -> std::unique_ptr<Codec> {
    switch (cfg.id) {
    case CodecId::Blosc:
        return std::make_unique<BloscCodec>(cfg.blosc_cname, cfg.blosc_clevel, cfg.blosc_shuffle,
                                            cfg.blosc_typesize, cfg.blosc_blocksize);
    case CodecId::Gzip:     return std::make_unique<GzipCodec>(cfg.level >= 0 ? cfg.level : 5);
    case CodecId::Zstd:     return std::make_unique<ZstdCodec>(cfg.level >= 0 ? cfg.level : 3);
    case CodecId::Lz4:      return std::make_unique<Lz4Codec>(cfg.acceleration);
    case CodecId::Bz2:      return std::make_unique<Bz2Codec>(cfg.level >= 0 ? cfg.level : 9);
    case CodecId::Lzma:     return std::make_unique<LzmaCodec>(cfg.level >= 0 ? cfg.level : 6);
    case CodecId::Crc32c:   return std::make_unique<Crc32cCodec>();
    case CodecId::Bytes:    return std::make_unique<BytesCodec>(cfg.endian);
    case CodecId::Transpose:return std::make_unique<TransposeCodec>(cfg.transpose_order);
    case CodecId::Sharding: {
        std::vector<size_t> outer(chunk_shape.begin(), chunk_shape.end());
        std::vector<size_t> inner = cfg.shard_shape;
        auto ip = build_pipeline(cfg.sub_codecs, cfg.shard_shape, elem_size);
        auto idxp = build_pipeline(cfg.index_codecs);
        return std::make_unique<ShardingCodec>(std::move(outer), std::move(inner),
                                                elem_size, std::move(ip), std::move(idxp));
    }
    case CodecId::H264:
    case CodecId::H265:
    case CodecId::AV1: {
#ifdef UTILS_HAS_VIDEO_CODECS
        std::vector<size_t> cs(chunk_shape.begin(), chunk_shape.end());
        return std::make_unique<VideoCodec>(
            cfg.id, std::move(cs), cfg.video_crf, cfg.video_preset, cfg.video_bit_depth);
#else
        return std::make_unique<VideoCodecStub>();
#endif
    }
    }
    return std::make_unique<BytesCodec>("little");
}

auto build_pipeline(std::span<const CodecConfig> cfgs,
                    std::span<const size_t> chunk_shape,
                    size_t elem_size) -> CodecPipeline {
    CodecPipeline p;
    for (auto& cfg : cfgs) p.codecs.push_back(make_codec(cfg, chunk_shape, elem_size));
    return p;
}

// Parse a single v2 codec JSON object (compressor or filter) and append to pipeline
static void parse_v2_codec_json(const Json& codec_json, DType dt, CodecPipeline& p) {
    if (codec_json.is_null()) return;
    if (codec_json.type() != Json::Type::Object) return;
    auto& id_json = codec_json["id"];
    if (id_json.type() != Json::Type::String) return;
    auto& id = id_json.as_string();
    if (id == "blosc") {
        std::string cname = "lz4";
        int clevel = 5, shuffle = 1, blocksize = 0;
        if (codec_json.contains("cname")) cname = codec_json["cname"].as_string();
        if (codec_json.contains("clevel")) clevel = static_cast<int>(codec_json["clevel"].as_int());
        if (codec_json.contains("shuffle")) shuffle = static_cast<int>(codec_json["shuffle"].as_int());
        if (codec_json.contains("blocksize")) blocksize = static_cast<int>(codec_json["blocksize"].as_int());
        int ts = static_cast<int>(dtype_size(dt));
        p.codecs.push_back(std::make_unique<BloscCodec>(cname, clevel, shuffle, ts, blocksize));
    } else if (id == "zlib" || id == "gzip") {
        int level = 1;
        if (codec_json.contains("level")) level = static_cast<int>(codec_json["level"].as_int());
        p.codecs.push_back(std::make_unique<GzipCodec>(level));
    } else if (id == "zstd") {
        int level = 3;
        if (codec_json.contains("level")) level = static_cast<int>(codec_json["level"].as_int());
        p.codecs.push_back(std::make_unique<ZstdCodec>(level));
    } else if (id == "lz4") {
        int accel = 1;
        if (codec_json.contains("acceleration")) accel = static_cast<int>(codec_json["acceleration"].as_int());
        p.codecs.push_back(std::make_unique<Lz4Codec>(accel));
    } else if (id == "bz2") {
        int level = 9;
        if (codec_json.contains("level")) level = static_cast<int>(codec_json["level"].as_int());
        p.codecs.push_back(std::make_unique<Bz2Codec>(level));
    } else if (id == "lzma") {
        int preset = 6;
        if (codec_json.contains("preset")) preset = static_cast<int>(codec_json["preset"].as_int());
        p.codecs.push_back(std::make_unique<LzmaCodec>(preset));
    }
}

auto build_v2_pipeline(const Json& compressor_json, const Json& filters_json, DType dt) -> CodecPipeline {
    CodecPipeline p;
    // Filters are applied before the compressor
    if (!filters_json.is_null() && filters_json.type() == Json::Type::Array) {
        for (auto& f : filters_json.as_array()) {
            parse_v2_codec_json(f, dt, p);
        }
    }
    // Compressor
    parse_v2_codec_json(compressor_json, dt, p);
    return p;
}

// ── Metadata fields and parsing ─────────────────────────────────────────────

struct MetadataFields {
    ZarrVersion version;
    std::vector<size_t> shape;
    std::vector<size_t> chunks;
    DType dtype;
    Order order;
    double fill_value;
    std::vector<CodecConfig> codecs;
    ChunkKeyEncoding chunk_key_encoding;
    std::vector<std::string> dimension_names;
};

auto parse_fill_value(const Json& fv, DType /*dt*/) -> double {
    if (fv.is_null()) return 0.0;
    if (fv.type() == Json::Type::Number) return fv.as_number();
    if (fv.type() == Json::Type::String) {
        auto& s = fv.as_string();
        if (s == "NaN") return std::numeric_limits<double>::quiet_NaN();
        if (s == "Infinity") return std::numeric_limits<double>::infinity();
        if (s == "-Infinity") return -std::numeric_limits<double>::infinity();
        try { return std::stod(s); } catch (...) { return 0.0; }
    }
    if (fv.type() == Json::Type::Bool) return fv.as_bool() ? 1.0 : 0.0;
    return 0.0;
}

auto parse_codec_config_v3(const Json& codec_json) -> std::expected<CodecConfig, std::string> {
    if (codec_json.type() != Json::Type::Object || !codec_json.contains("name"))
        return std::unexpected("invalid codec object");
    auto& name = codec_json["name"].as_string();
    auto& cfg_json = codec_json.contains("configuration") ? codec_json["configuration"] : codec_json;
    CodecConfig cc;
    if (name == "blosc") {
        cc.id = CodecId::Blosc;
        cc.blosc_cname = cfg_json.contains("cname") ? cfg_json["cname"].as_string() : "lz4";
        cc.blosc_clevel = cfg_json.contains("clevel") ? static_cast<int>(cfg_json["clevel"].as_int()) : 5;
        if (cfg_json.contains("shuffle")) {
            auto& sv = cfg_json["shuffle"];
            if (sv.type() == Json::Type::String) cc.blosc_shuffle = shuffle_str_to_int(sv.as_string());
            else cc.blosc_shuffle = static_cast<int>(sv.as_int());
        } else {
            cc.blosc_shuffle = 1;
        }
        cc.blosc_typesize = cfg_json.contains("typesize") ? static_cast<int>(cfg_json["typesize"].as_int()) : 0;
        cc.blosc_blocksize = cfg_json.contains("blocksize") ? static_cast<int>(cfg_json["blocksize"].as_int()) : 0;
    } else if (name == "gzip") {
        cc.id = CodecId::Gzip;
        cc.level = cfg_json.contains("level") ? static_cast<int>(cfg_json["level"].as_int()) : 5;
    } else if (name == "zstd") {
        cc.id = CodecId::Zstd;
        cc.level = cfg_json.contains("level") ? static_cast<int>(cfg_json["level"].as_int()) : 3;
    } else if (name == "bytes") {
        cc.id = CodecId::Bytes;
        cc.endian = cfg_json.contains("endian") ? cfg_json["endian"].as_string() : "little";
    } else if (name == "transpose") {
        cc.id = CodecId::Transpose;
        if (cfg_json.contains("order")) {
            for (auto& v : cfg_json["order"].as_array())
                cc.transpose_order.push_back(static_cast<size_t>(v.as_int()));
        }
    } else if (name == "crc32c") {
        cc.id = CodecId::Crc32c;
    } else if (name == "sharding_indexed") {
        cc.id = CodecId::Sharding;
        if (cfg_json.contains("chunk_shape")) {
            for (auto& v : cfg_json["chunk_shape"].as_array())
                cc.shard_shape.push_back(static_cast<size_t>(v.as_int()));
        }
        if (cfg_json.contains("codecs")) {
            for (auto& sc : cfg_json["codecs"].as_array()) {
                auto sub = parse_codec_config_v3(sc);
                if (!sub) return std::unexpected(sub.error());
                cc.sub_codecs.push_back(std::move(*sub));
            }
        }
        if (cfg_json.contains("index_codecs")) {
            for (auto& ic : cfg_json["index_codecs"].as_array()) {
                auto idx = parse_codec_config_v3(ic);
                if (!idx) return std::unexpected(idx.error());
                cc.index_codecs.push_back(std::move(*idx));
            }
        }
    } else if (name == "h264") {
        cc.id = CodecId::H264;
        cc.video_crf = cfg_json.contains("crf") ? static_cast<int>(cfg_json["crf"].as_int()) : 23;
        cc.video_preset = cfg_json.contains("preset") ? cfg_json["preset"].as_string() : "medium";
        cc.video_bit_depth = cfg_json.contains("bit_depth") ? static_cast<int>(cfg_json["bit_depth"].as_int()) : 8;
    } else if (name == "h265" || name == "hevc") {
        cc.id = CodecId::H265;
        cc.video_crf = cfg_json.contains("crf") ? static_cast<int>(cfg_json["crf"].as_int()) : 23;
        cc.video_preset = cfg_json.contains("preset") ? cfg_json["preset"].as_string() : "medium";
        cc.video_bit_depth = cfg_json.contains("bit_depth") ? static_cast<int>(cfg_json["bit_depth"].as_int()) : 8;
    } else if (name == "av1") {
        cc.id = CodecId::AV1;
        cc.video_crf = cfg_json.contains("crf") ? static_cast<int>(cfg_json["crf"].as_int()) : 30;
        cc.video_preset = cfg_json.contains("preset") ? cfg_json["preset"].as_string() : "6";
        cc.video_bit_depth = cfg_json.contains("bit_depth") ? static_cast<int>(cfg_json["bit_depth"].as_int()) : 8;
    } else {
        return std::unexpected("unknown codec: " + name);
    }
    return cc;
}

auto parse_v2_metadata(const Json& j) -> std::expected<MetadataFields, std::string> {
    MetadataFields m;
    m.version = ZarrVersion::V2;
    m.order = Order::C;
    // shape
    if (!j.contains("shape")) return std::unexpected("v2 metadata missing 'shape'");
    for (auto& v : j["shape"].as_array()) m.shape.push_back(static_cast<size_t>(v.as_int()));
    // chunks
    if (!j.contains("chunks")) return std::unexpected("v2 metadata missing 'chunks'");
    for (auto& v : j["chunks"].as_array()) m.chunks.push_back(static_cast<size_t>(v.as_int()));
    // dtype
    if (!j.contains("dtype")) return std::unexpected("v2 metadata missing 'dtype'");
    auto dt = numpy_dtype_to_dtype(j["dtype"].as_string());
    if (!dt) return std::unexpected(dt.error());
    m.dtype = *dt;
    // order
    if (j.contains("order") && j["order"].as_string() == "F") m.order = Order::F;
    // fill_value
    m.fill_value = parse_fill_value(j["fill_value"], m.dtype);
    // Helper to parse a v2 codec/filter JSON object into a CodecConfig
    auto parse_v2_codec = [&](const Json& comp) -> std::optional<CodecConfig> {
        if (comp.is_null() || comp.type() != Json::Type::Object || !comp.contains("id"))
            return std::nullopt;
        CodecConfig cc;
        auto& id = comp["id"].as_string();
        if (id == "blosc") {
            cc.id = CodecId::Blosc;
            cc.blosc_cname = comp.contains("cname") ? comp["cname"].as_string() : "lz4";
            cc.blosc_clevel = comp.contains("clevel") ? static_cast<int>(comp["clevel"].as_int()) : 5;
            cc.blosc_shuffle = comp.contains("shuffle") ? static_cast<int>(comp["shuffle"].as_int()) : 1;
            cc.blosc_typesize = comp.contains("typesize") ? static_cast<int>(comp["typesize"].as_int()) : static_cast<int>(dtype_size(m.dtype));
            cc.blosc_blocksize = comp.contains("blocksize") ? static_cast<int>(comp["blocksize"].as_int()) : 0;
        } else if (id == "zlib" || id == "gzip") {
            cc.id = CodecId::Gzip;
            cc.level = comp.contains("level") ? static_cast<int>(comp["level"].as_int()) : 1;
        } else if (id == "zstd") {
            cc.id = CodecId::Zstd;
            cc.level = comp.contains("level") ? static_cast<int>(comp["level"].as_int()) : 3;
        } else if (id == "lz4") {
            cc.id = CodecId::Lz4;
            cc.acceleration = comp.contains("acceleration") ? static_cast<int>(comp["acceleration"].as_int()) : 1;
        } else if (id == "bz2") {
            cc.id = CodecId::Bz2;
            cc.level = comp.contains("level") ? static_cast<int>(comp["level"].as_int()) : 9;
        } else if (id == "lzma") {
            cc.id = CodecId::Lzma;
            cc.level = comp.contains("preset") ? static_cast<int>(comp["preset"].as_int()) : 6;
        } else {
            return std::nullopt; // unknown codec
        }
        return cc;
    };
    // filters -> codecs (filters are applied before compressor)
    if (j.contains("filters") && !j["filters"].is_null() && j["filters"].type() == Json::Type::Array) {
        for (auto& f : j["filters"].as_array()) {
            auto fc = parse_v2_codec(f);
            if (fc) m.codecs.push_back(std::move(*fc));
        }
    }
    // compressor -> codecs (appended after filters)
    if (j.contains("compressor") && !j["compressor"].is_null()) {
        auto cc = parse_v2_codec(j["compressor"]);
        if (cc) m.codecs.push_back(std::move(*cc));
    }
    // dimension_separator -> chunk_key_encoding
    m.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;
    if (j.contains("dimension_separator") && j["dimension_separator"].as_string() == "/")
        m.chunk_key_encoding = ChunkKeyEncoding::SlashSeparated;
    return m;
}

auto parse_v3_metadata(const Json& j) -> std::expected<MetadataFields, std::string> {
    MetadataFields m;
    m.version = ZarrVersion::V3;
    m.order = Order::C;
    if (j.contains("zarr_format") && j["zarr_format"].as_int() != 3)
        return std::unexpected("expected zarr_format 3");
    if (j.contains("node_type") && j["node_type"].as_string() != "array")
        return std::unexpected("expected node_type 'array'");
    // shape
    if (!j.contains("shape")) return std::unexpected("v3 metadata missing 'shape'");
    for (auto& v : j["shape"].as_array()) m.shape.push_back(static_cast<size_t>(v.as_int()));
    // chunk_grid
    if (j.contains("chunk_grid")) {
        auto& cg = j["chunk_grid"];
        if (cg.contains("configuration") && cg["configuration"].contains("chunk_shape")) {
            for (auto& v : cg["configuration"]["chunk_shape"].as_array())
                m.chunks.push_back(static_cast<size_t>(v.as_int()));
        }
    }
    // data_type
    if (!j.contains("data_type")) return std::unexpected("v3 metadata missing 'data_type'");
    auto dt = numpy_dtype_to_dtype(j["data_type"].as_string());
    if (!dt) return std::unexpected(dt.error());
    m.dtype = *dt;
    // fill_value
    m.fill_value = parse_fill_value(j["fill_value"], m.dtype);
    // codecs
    if (j.contains("codecs")) {
        for (auto& c : j["codecs"].as_array()) {
            auto cc = parse_codec_config_v3(c);
            if (!cc) return std::unexpected(cc.error());
            m.codecs.push_back(std::move(*cc));
        }
    }
    // chunk_key_encoding
    m.chunk_key_encoding = ChunkKeyEncoding::V3Default;
    if (j.contains("chunk_key_encoding")) {
        auto& cke = j["chunk_key_encoding"];
        if (cke.contains("configuration") && cke["configuration"].contains("separator")) {
            auto& sep = cke["configuration"]["separator"].as_string();
            if (sep == ".") m.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;
            else if (sep == "/") m.chunk_key_encoding = ChunkKeyEncoding::SlashSeparated;
        } else if (cke.contains("name") && cke["name"].as_string() == "default") {
            m.chunk_key_encoding = ChunkKeyEncoding::V3Default;
        }
    }
    // dimension_names
    if (j.contains("dimension_names")) {
        for (auto& v : j["dimension_names"].as_array()) {
            if (v.type() == Json::Type::String) m.dimension_names.push_back(v.as_string());
            else m.dimension_names.emplace_back();
        }
    }
    return m;
}

auto serialize_v2_metadata(const MetadataFields& m) -> Json {
    std::map<std::string, Json, std::less<>> obj;
    obj["zarr_format"] = Json::number(2);
    // shape
    std::vector<Json> shape_arr;
    for (auto s : m.shape) shape_arr.push_back(Json::number(static_cast<double>(s)));
    obj["shape"] = Json::array(std::move(shape_arr));
    // chunks
    std::vector<Json> chunks_arr;
    for (auto c : m.chunks) chunks_arr.push_back(Json::number(static_cast<double>(c)));
    obj["chunks"] = Json::array(std::move(chunks_arr));
    // dtype
    obj["dtype"] = Json::string_val(dtype_to_numpy_str(m.dtype));
    // order
    obj["order"] = Json::string_val(m.order == Order::F ? "F" : "C");
    // fill_value
    if (std::isnan(m.fill_value)) obj["fill_value"] = Json::string_val("NaN");
    else if (std::isinf(m.fill_value) && m.fill_value > 0) obj["fill_value"] = Json::string_val("Infinity");
    else if (std::isinf(m.fill_value) && m.fill_value < 0) obj["fill_value"] = Json::string_val("-Infinity");
    else obj["fill_value"] = Json::number(m.fill_value);
    // Helper to serialize a single v2 codec config to JSON
    auto serialize_v2_codec = [](const CodecConfig& cc) -> Json {
        std::map<std::string, Json, std::less<>> comp;
        switch (cc.id) {
        case CodecId::Blosc:
            comp["id"] = Json::string_val("blosc");
            comp["cname"] = Json::string_val(cc.blosc_cname);
            comp["clevel"] = Json::number(cc.blosc_clevel);
            comp["shuffle"] = Json::number(cc.blosc_shuffle);
            comp["blocksize"] = Json::number(cc.blosc_blocksize);
            break;
        case CodecId::Gzip:
            comp["id"] = Json::string_val("zlib");
            comp["level"] = Json::number(cc.level >= 0 ? cc.level : 1);
            break;
        case CodecId::Zstd:
            comp["id"] = Json::string_val("zstd");
            comp["level"] = Json::number(cc.level >= 0 ? cc.level : 3);
            break;
        case CodecId::Lz4:
            comp["id"] = Json::string_val("lz4");
            comp["acceleration"] = Json::number(cc.acceleration);
            break;
        case CodecId::Bz2:
            comp["id"] = Json::string_val("bz2");
            comp["level"] = Json::number(cc.level >= 0 ? cc.level : 9);
            break;
        case CodecId::Lzma:
            comp["id"] = Json::string_val("lzma");
            comp["preset"] = Json::number(cc.level >= 0 ? cc.level : 6);
            break;
        default:
            comp["id"] = Json::string_val("blosc");
            break;
        }
        return Json::object(std::move(comp));
    };
    // compressor: last codec in the list (or null if empty)
    if (m.codecs.empty()) {
        obj["compressor"] = Json::null_val();
    } else {
        obj["compressor"] = serialize_v2_codec(m.codecs.back());
    }
    // filters: first N-1 codecs (or null if only one or zero)
    if (m.codecs.size() > 1) {
        std::vector<Json> filters_arr;
        for (size_t i = 0; i + 1 < m.codecs.size(); ++i)
            filters_arr.push_back(serialize_v2_codec(m.codecs[i]));
        obj["filters"] = Json::array(std::move(filters_arr));
    } else {
        obj["filters"] = Json::null_val();
    }
    // dimension_separator
    if (m.chunk_key_encoding == ChunkKeyEncoding::SlashSeparated)
        obj["dimension_separator"] = Json::string_val("/");
    else
        obj["dimension_separator"] = Json::string_val(".");
    return Json::object(std::move(obj));
}

auto serialize_v3_metadata(const MetadataFields& m) -> Json {
    std::map<std::string, Json, std::less<>> obj;
    obj["zarr_format"] = Json::number(3);
    obj["node_type"] = Json::string_val("array");
    // shape
    std::vector<Json> shape_arr;
    for (auto s : m.shape) shape_arr.push_back(Json::number(static_cast<double>(s)));
    obj["shape"] = Json::array(std::move(shape_arr));
    // chunk_grid
    std::vector<Json> cs_arr;
    for (auto c : m.chunks) cs_arr.push_back(Json::number(static_cast<double>(c)));
    std::map<std::string, Json, std::less<>> cg_config;
    cg_config["chunk_shape"] = Json::array(std::move(cs_arr));
    std::map<std::string, Json, std::less<>> cg;
    cg["name"] = Json::string_val("regular");
    cg["configuration"] = Json::object(std::move(cg_config));
    obj["chunk_grid"] = Json::object(std::move(cg));
    // data_type
    obj["data_type"] = Json::string_val(dtype_to_v3_str(m.dtype));
    // fill_value
    if (std::isnan(m.fill_value)) obj["fill_value"] = Json::string_val("NaN");
    else if (std::isinf(m.fill_value) && m.fill_value > 0) obj["fill_value"] = Json::string_val("Infinity");
    else if (std::isinf(m.fill_value) && m.fill_value < 0) obj["fill_value"] = Json::string_val("-Infinity");
    else obj["fill_value"] = Json::number(m.fill_value);
    // codecs
    std::vector<Json> codecs_arr;
    for (auto& cc : m.codecs) {
        std::map<std::string, Json, std::less<>> codec_obj;
        std::map<std::string, Json, std::less<>> codec_config;
        switch (cc.id) {
        case CodecId::Blosc:
            codec_obj["name"] = Json::string_val("blosc");
            codec_config["cname"] = Json::string_val(cc.blosc_cname);
            codec_config["clevel"] = Json::number(cc.blosc_clevel);
            codec_config["shuffle"] = Json::string_val(shuffle_int_to_str(cc.blosc_shuffle));
            codec_config["typesize"] = Json::number(cc.blosc_typesize);
            codec_config["blocksize"] = Json::number(cc.blosc_blocksize);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::Gzip:
            codec_obj["name"] = Json::string_val("gzip");
            codec_config["level"] = Json::number(cc.level);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::Zstd:
            codec_obj["name"] = Json::string_val("zstd");
            codec_config["level"] = Json::number(cc.level);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::Bytes:
            codec_obj["name"] = Json::string_val("bytes");
            codec_config["endian"] = Json::string_val(cc.endian);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::Transpose:
            codec_obj["name"] = Json::string_val("transpose");
            {
                std::vector<Json> order_arr;
                for (auto o : cc.transpose_order) order_arr.push_back(Json::number(static_cast<double>(o)));
                codec_config["order"] = Json::array(std::move(order_arr));
            }
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::Crc32c:
            codec_obj["name"] = Json::string_val("crc32c");
            break;
        case CodecId::Sharding: {
            codec_obj["name"] = Json::string_val("sharding_indexed");
            std::map<std::string, Json, std::less<>> shard_config;
            // chunk_shape
            std::vector<Json> cs;
            for (auto s : cc.shard_shape) cs.push_back(Json::number(static_cast<double>(s)));
            shard_config["chunk_shape"] = Json::array(std::move(cs));
            // codecs (inner/sub codecs)
            if (!cc.sub_codecs.empty()) {
                std::vector<Json> sub_arr;
                for (auto& sc : cc.sub_codecs) {
                    // Re-serialize each sub codec using the same v3 logic
                    std::map<std::string, Json, std::less<>> sobj;
                    std::map<std::string, Json, std::less<>> scfg;
                    switch (sc.id) {
                    case CodecId::Blosc:
                        sobj["name"] = Json::string_val("blosc");
                        scfg["cname"] = Json::string_val(sc.blosc_cname);
                        scfg["clevel"] = Json::number(sc.blosc_clevel);
                        scfg["shuffle"] = Json::string_val(shuffle_int_to_str(sc.blosc_shuffle));
                        scfg["typesize"] = Json::number(sc.blosc_typesize);
                        scfg["blocksize"] = Json::number(sc.blosc_blocksize);
                        sobj["configuration"] = Json::object(std::move(scfg));
                        break;
                    case CodecId::Gzip:
                        sobj["name"] = Json::string_val("gzip");
                        scfg["level"] = Json::number(sc.level);
                        sobj["configuration"] = Json::object(std::move(scfg));
                        break;
                    case CodecId::Zstd:
                        sobj["name"] = Json::string_val("zstd");
                        scfg["level"] = Json::number(sc.level);
                        sobj["configuration"] = Json::object(std::move(scfg));
                        break;
                    case CodecId::Bytes:
                        sobj["name"] = Json::string_val("bytes");
                        scfg["endian"] = Json::string_val(sc.endian);
                        sobj["configuration"] = Json::object(std::move(scfg));
                        break;
                    case CodecId::Crc32c:
                        sobj["name"] = Json::string_val("crc32c");
                        break;
                    default:
                        sobj["name"] = Json::string_val("bytes");
                        break;
                    }
                    sub_arr.push_back(Json::object(std::move(sobj)));
                }
                shard_config["codecs"] = Json::array(std::move(sub_arr));
            }
            // index_codecs
            if (!cc.index_codecs.empty()) {
                std::vector<Json> idx_arr;
                for (auto& ic : cc.index_codecs) {
                    std::map<std::string, Json, std::less<>> iobj;
                    std::map<std::string, Json, std::less<>> icfg;
                    switch (ic.id) {
                    case CodecId::Bytes:
                        iobj["name"] = Json::string_val("bytes");
                        icfg["endian"] = Json::string_val(ic.endian);
                        iobj["configuration"] = Json::object(std::move(icfg));
                        break;
                    case CodecId::Crc32c:
                        iobj["name"] = Json::string_val("crc32c");
                        break;
                    default:
                        iobj["name"] = Json::string_val("bytes");
                        break;
                    }
                    idx_arr.push_back(Json::object(std::move(iobj)));
                }
                shard_config["index_codecs"] = Json::array(std::move(idx_arr));
            }
            codec_obj["configuration"] = Json::object(std::move(shard_config));
            break;
        }
        case CodecId::H264:
            codec_obj["name"] = Json::string_val("h264");
            codec_config["crf"] = Json::number(cc.video_crf);
            codec_config["preset"] = Json::string_val(cc.video_preset);
            codec_config["bit_depth"] = Json::number(cc.video_bit_depth);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::H265:
            codec_obj["name"] = Json::string_val("h265");
            codec_config["crf"] = Json::number(cc.video_crf);
            codec_config["preset"] = Json::string_val(cc.video_preset);
            codec_config["bit_depth"] = Json::number(cc.video_bit_depth);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        case CodecId::AV1:
            codec_obj["name"] = Json::string_val("av1");
            codec_config["crf"] = Json::number(cc.video_crf);
            codec_config["preset"] = Json::string_val(cc.video_preset);
            codec_config["bit_depth"] = Json::number(cc.video_bit_depth);
            codec_obj["configuration"] = Json::object(std::move(codec_config));
            break;
        default:
            codec_obj["name"] = Json::string_val("bytes");
            break;
        }
        codecs_arr.push_back(Json::object(std::move(codec_obj)));
    }
    obj["codecs"] = Json::array(std::move(codecs_arr));
    // chunk_key_encoding
    {
        std::map<std::string, Json, std::less<>> cke;
        cke["name"] = Json::string_val("default");
        std::map<std::string, Json, std::less<>> cke_config;
        if (m.chunk_key_encoding == ChunkKeyEncoding::DotSeparated)
            cke_config["separator"] = Json::string_val(".");
        else
            cke_config["separator"] = Json::string_val("/");
        cke["configuration"] = Json::object(std::move(cke_config));
        obj["chunk_key_encoding"] = Json::object(std::move(cke));
    }
    // dimension_names
    if (!m.dimension_names.empty()) {
        std::vector<Json> dn;
        for (auto& n : m.dimension_names) dn.push_back(Json::string_val(n));
        obj["dimension_names"] = Json::array(std::move(dn));
    }
    return Json::object(std::move(obj));
}

// ── Chunk key helper ────────────────────────────────────────────────────────

auto chunk_key(std::string_view base_path, std::span<const size_t> indices,
               ChunkKeyEncoding enc, ZarrVersion version) -> std::string {
    std::string key;
    if (version == ZarrVersion::V3) {
        key = join_path(base_path, "c");
        for (size_t i = 0; i < indices.size(); ++i)
            key += "/" + std::to_string(indices[i]);
    } else {
        // v2
        std::string idx_part;
        char sep = (enc == ChunkKeyEncoding::SlashSeparated) ? '/' : '.';
        for (size_t i = 0; i < indices.size(); ++i) {
            if (i > 0) idx_part += sep;
            idx_part += std::to_string(indices[i]);
        }
        key = join_path(base_path, idx_part);
    }
    return key;
}

} // anonymous namespace

// ── CodecConfig factory methods ─────────────────────────────────────────────

auto CodecConfig::blosc(std::string cname, int clevel, int shuffle,
                        int typesize, int blocksize) -> CodecConfig {
    CodecConfig c;
    c.id = CodecId::Blosc;
    c.blosc_cname = std::move(cname);
    c.blosc_clevel = clevel;
    c.blosc_shuffle = shuffle;
    c.blosc_typesize = typesize;
    c.blosc_blocksize = blocksize;
    return c;
}

auto CodecConfig::gzip(int lvl) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Gzip; c.level = lvl; return c;
}

auto CodecConfig::zstd(int lvl) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Zstd; c.level = lvl; return c;
}

auto CodecConfig::lz4(int accel) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Lz4; c.acceleration = accel; return c;
}

auto CodecConfig::bz2(int lvl) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Bz2; c.level = lvl; return c;
}

auto CodecConfig::lzma(int preset) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Lzma; c.level = preset; return c;
}

auto CodecConfig::crc32c() -> CodecConfig {
    CodecConfig c; c.id = CodecId::Crc32c; return c;
}

auto CodecConfig::bytes(std::string endian) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Bytes; c.endian = std::move(endian); return c;
}

auto CodecConfig::transpose(std::vector<size_t> order) -> CodecConfig {
    CodecConfig c; c.id = CodecId::Transpose; c.transpose_order = std::move(order); return c;
}

auto CodecConfig::sharding(std::vector<size_t> shape, std::vector<CodecConfig> codecs,
                           std::vector<CodecConfig> idx_codecs) -> CodecConfig {
    CodecConfig c;
    c.id = CodecId::Sharding;
    c.shard_shape = std::move(shape);
    c.sub_codecs = std::move(codecs);
    c.index_codecs = std::move(idx_codecs);
    return c;
}

auto CodecConfig::h264(int crf, std::string preset, int bit_depth) -> CodecConfig {
    CodecConfig c;
    c.id = CodecId::H264;
    c.video_crf = crf;
    c.video_preset = std::move(preset);
    c.video_bit_depth = bit_depth;
    return c;
}

auto CodecConfig::h265(int crf, std::string preset, int bit_depth) -> CodecConfig {
    CodecConfig c;
    c.id = CodecId::H265;
    c.video_crf = crf;
    c.video_preset = std::move(preset);
    c.video_bit_depth = bit_depth;
    return c;
}

auto CodecConfig::av1(int crf, std::string preset, int bit_depth) -> CodecConfig {
    CodecConfig c;
    c.id = CodecId::AV1;
    c.video_crf = crf;
    c.video_preset = std::move(preset);
    c.video_bit_depth = bit_depth;
    return c;
}

// ── Store::get_partial default ──────────────────────────────────────────────

auto Store::get_partial(std::string_view key, size_t offset, size_t length) const
    -> std::expected<std::vector<uint8_t>, std::string> {
    auto data = get(key);
    if (!data) return std::unexpected(data.error());
    if (offset >= data->size()) return std::vector<uint8_t>{};
    size_t end = std::min(offset + length, data->size());
    return std::vector<uint8_t>(data->begin() + static_cast<ptrdiff_t>(offset),
                                data->begin() + static_cast<ptrdiff_t>(end));
}

// ── MemoryStore ─────────────────────────────────────────────────────────────

struct MemoryStore::Impl {
    std::map<std::string, std::vector<uint8_t>, std::less<>> data;
};

MemoryStore::MemoryStore() : impl_(std::make_unique<Impl>()) {}
MemoryStore::~MemoryStore() = default;
MemoryStore::MemoryStore(MemoryStore&&) noexcept = default;
auto MemoryStore::operator=(MemoryStore&&) noexcept -> MemoryStore& = default;

auto MemoryStore::get(std::string_view key) const
    -> std::expected<std::vector<uint8_t>, std::string> {
    auto it = impl_->data.find(key);
    if (it == impl_->data.end())
        return std::unexpected("key not found: " + std::string(key));
    return it->second;
}

auto MemoryStore::set(std::string_view key, std::span<const uint8_t> value)
    -> std::expected<void, std::string> {
    impl_->data[std::string(key)] = std::vector<uint8_t>(value.begin(), value.end());
    return {};
}

auto MemoryStore::erase(std::string_view key) -> std::expected<void, std::string> {
    auto it = impl_->data.find(key);
    if (it != impl_->data.end()) impl_->data.erase(it);
    return {};
}

auto MemoryStore::exists(std::string_view key) const -> bool {
    return impl_->data.find(key) != impl_->data.end();
}

auto MemoryStore::list_prefix(std::string_view prefix) const -> std::vector<std::string> {
    std::vector<std::string> result;
    for (auto& [k, v] : impl_->data) {
        if (prefix.empty() || k.starts_with(prefix))
            result.push_back(k);
    }
    return result;
}

auto MemoryStore::list_dir(std::string_view prefix) const
    -> std::pair<std::vector<std::string>, std::vector<std::string>> {
    std::string pfx(prefix);
    if (!pfx.empty() && pfx.back() != '/') pfx += '/';
    std::vector<std::string> keys;
    std::set<std::string> prefixes;
    for (auto& [k, v] : impl_->data) {
        if (!k.starts_with(pfx) && !(pfx.empty() || pfx == "/")) continue;
        std::string_view remainder;
        if (pfx.empty() || pfx == "/") remainder = k;
        else remainder = std::string_view(k).substr(pfx.size());
        if (remainder.empty()) continue;
        auto slash = remainder.find('/');
        if (slash == std::string_view::npos) {
            keys.emplace_back(remainder);
        } else {
            prefixes.insert(std::string(remainder.substr(0, slash)));
        }
    }
    return {keys, {prefixes.begin(), prefixes.end()}};
}

auto MemoryStore::get_partial(std::string_view key, size_t offset, size_t length) const
    -> std::expected<std::vector<uint8_t>, std::string> {
    auto it = impl_->data.find(key);
    if (it == impl_->data.end())
        return std::unexpected("key not found: " + std::string(key));
    auto& data = it->second;
    if (offset >= data.size()) return std::vector<uint8_t>{};
    size_t end = std::min(offset + length, data.size());
    return std::vector<uint8_t>(data.begin() + static_cast<ptrdiff_t>(offset),
                                data.begin() + static_cast<ptrdiff_t>(end));
}

// ── FilesystemStore ─────────────────────────────────────────────────────────

struct FilesystemStore::Impl {
    std::filesystem::path root;
};

FilesystemStore::FilesystemStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FilesystemStore::~FilesystemStore() = default;
FilesystemStore::FilesystemStore(FilesystemStore&&) noexcept = default;
auto FilesystemStore::operator=(FilesystemStore&&) noexcept -> FilesystemStore& = default;

auto FilesystemStore::open(const std::filesystem::path& root)
    -> std::expected<std::unique_ptr<FilesystemStore>, std::string> {
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    auto impl = std::make_unique<Impl>();
    impl->root = std::filesystem::canonical(root, ec);
    if (ec) impl->root = std::filesystem::absolute(root);
    return std::unique_ptr<FilesystemStore>(new FilesystemStore(std::move(impl)));
}

auto FilesystemStore::root() const -> const std::filesystem::path& {
    return impl_->root;
}

auto FilesystemStore::get(std::string_view key) const
    -> std::expected<std::vector<uint8_t>, std::string> {
    auto path = impl_->root / std::string(key);
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return std::unexpected("failed to read: " + path.string());
    auto size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

auto FilesystemStore::set(std::string_view key, std::span<const uint8_t> value)
    -> std::expected<void, std::string> {
    auto path = impl_->root / std::string(key);
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream f(path, std::ios::binary);
    if (!f) return std::unexpected("failed to write: " + path.string());
    f.write(reinterpret_cast<const char*>(value.data()), static_cast<std::streamsize>(value.size()));
    return {};
}

auto FilesystemStore::erase(std::string_view key) -> std::expected<void, std::string> {
    auto path = impl_->root / std::string(key);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return {};
}

auto FilesystemStore::exists(std::string_view key) const -> bool {
    auto path = impl_->root / std::string(key);
    return std::filesystem::exists(path);
}

auto FilesystemStore::list_prefix(std::string_view prefix) const -> std::vector<std::string> {
    std::vector<std::string> result;
    auto dir = impl_->root / std::string(prefix);
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return result;
    for (auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (entry.is_regular_file()) {
            auto rel = std::filesystem::relative(entry.path(), impl_->root, ec);
            if (!ec) result.push_back(rel.string());
        }
    }
    return result;
}

auto FilesystemStore::list_dir(std::string_view prefix) const
    -> std::pair<std::vector<std::string>, std::vector<std::string>> {
    auto dir = impl_->root / std::string(prefix);
    std::vector<std::string> keys, prefixes;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return {keys, prefixes};
    for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        auto name = entry.path().filename().string();
        if (entry.is_regular_file()) keys.push_back(name);
        else if (entry.is_directory()) prefixes.push_back(name);
    }
    return {keys, prefixes};
}

auto FilesystemStore::get_partial(std::string_view key, size_t offset, size_t length) const
    -> std::expected<std::vector<uint8_t>, std::string> {
    auto path = impl_->root / std::string(key);
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected("failed to read: " + path.string());
    f.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> data(length);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(length));
    auto bytes_read = f.gcount();
    data.resize(static_cast<size_t>(bytes_read));
    return data;
}

// ── ArrayMetadata ───────────────────────────────────────────────────────────

struct ArrayMetadata::Impl {
    ZarrVersion version = ZarrVersion::V3;
    std::vector<size_t> shape;
    std::vector<size_t> chunks;
    DType dtype = DType::Float32;
    Order order = Order::C;
    double fill_value = 0.0;
    std::vector<CodecConfig> codecs;
    ChunkKeyEncoding chunk_key_encoding = ChunkKeyEncoding::V3Default;
    std::vector<std::string> dimension_names;
};

ArrayMetadata::ArrayMetadata(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ArrayMetadata::~ArrayMetadata() = default;
ArrayMetadata::ArrayMetadata(ArrayMetadata&&) noexcept = default;
auto ArrayMetadata::operator=(ArrayMetadata&&) noexcept -> ArrayMetadata& = default;

ArrayMetadata::ArrayMetadata(const ArrayMetadata& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

auto ArrayMetadata::operator=(const ArrayMetadata& other) -> ArrayMetadata& {
    if (this != &other) impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}

auto ArrayMetadata::version() const -> ZarrVersion { return impl_->version; }
auto ArrayMetadata::shape() const -> std::span<const size_t> { return impl_->shape; }
auto ArrayMetadata::chunks() const -> std::span<const size_t> { return impl_->chunks; }
auto ArrayMetadata::dtype() const -> DType { return impl_->dtype; }
auto ArrayMetadata::order() const -> Order { return impl_->order; }
auto ArrayMetadata::fill_value() const -> double { return impl_->fill_value; }
auto ArrayMetadata::codecs() const -> std::span<const CodecConfig> { return impl_->codecs; }
auto ArrayMetadata::chunk_key_encoding() const -> ChunkKeyEncoding { return impl_->chunk_key_encoding; }
auto ArrayMetadata::dimension_names() const -> std::span<const std::string> { return impl_->dimension_names; }

// ── ZarrArray ───────────────────────────────────────────────────────────────

struct ZarrArray::Impl {
    Store* store = nullptr;
    std::shared_ptr<Store> owned_store_;
    bool read_only_ = false;
    std::string path;
    ArrayMetadata meta{nullptr};
    CodecPipeline pipeline;

    auto metadata_key() const -> std::string {
        if (meta.version() == ZarrVersion::V3)
            return join_path(path, "zarr.json");
        return join_path(path, ".zarray");
    }

    auto attrs_key() const -> std::string {
        if (meta.version() == ZarrVersion::V3)
            return join_path(path, "zarr.json");
        return join_path(path, ".zattrs");
    }

    auto chunk_elements() const -> size_t {
        auto c = meta.chunks();
        size_t n = 1;
        for (auto s : c) n *= s;
        return n;
    }

    auto chunk_bytes() const -> size_t {
        return chunk_elements() * dtype_size(meta.dtype());
    }

    auto fill_chunk_buffer() const -> std::vector<uint8_t> {
        size_t nbytes = chunk_bytes();
        std::vector<uint8_t> buf(nbytes, 0);
        double fv = meta.fill_value();
        if (fv != 0.0) {
            size_t esz = dtype_size(meta.dtype());
            size_t nel = chunk_elements();
            for (size_t i = 0; i < nel; ++i) {
                switch (meta.dtype()) {
                case DType::Float32: { auto v = static_cast<float>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::Float64: { std::memcpy(buf.data() + i * esz, &fv, esz); break; }
                case DType::Int8:    { auto v = static_cast<int8_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::Int16:   { auto v = static_cast<int16_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::Int32:   { auto v = static_cast<int32_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::Int64:   { auto v = static_cast<int64_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::UInt8:   { auto v = static_cast<uint8_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::UInt16:  { auto v = static_cast<uint16_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::UInt32:  { auto v = static_cast<uint32_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                case DType::UInt64:  { auto v = static_cast<uint64_t>(fv); std::memcpy(buf.data() + i * esz, &v, esz); break; }
                }
            }
        }
        return buf;
    }

    auto to_metadata_fields() const -> MetadataFields {
        MetadataFields mf;
        mf.version = meta.version();
        mf.shape = {meta.shape().begin(), meta.shape().end()};
        mf.chunks = {meta.chunks().begin(), meta.chunks().end()};
        mf.dtype = meta.dtype();
        mf.order = meta.order();
        mf.fill_value = meta.fill_value();
        mf.codecs = {meta.codecs().begin(), meta.codecs().end()};
        mf.chunk_key_encoding = meta.chunk_key_encoding();
        mf.dimension_names = {meta.dimension_names().begin(), meta.dimension_names().end()};
        return mf;
    }

    auto write_metadata() -> std::expected<void, std::string> {
        auto mf = to_metadata_fields();
        Json j;
        if (meta.version() == ZarrVersion::V3)
            j = serialize_v3_metadata(mf);
        else
            j = serialize_v2_metadata(mf);
        auto json_str = j.serialize();
        auto key = metadata_key();
        return store->set(key, std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(json_str.data()), json_str.size()));
    }
};

ZarrArray::ZarrArray(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ZarrArray::~ZarrArray() = default;
ZarrArray::ZarrArray(ZarrArray&&) noexcept = default;
auto ZarrArray::operator=(ZarrArray&&) noexcept -> ZarrArray& = default;

auto ZarrArray::open(Store& store, std::string_view path)
    -> std::expected<ZarrArray, std::string> {
    auto norm = normalize_path(path);
    // Try v3 first
    auto v3_key = join_path(norm, "zarr.json");
    auto v2_key = join_path(norm, ".zarray");
    bool is_v3 = store.exists(v3_key);
    bool is_v2 = store.exists(v2_key);
    if (!is_v3 && !is_v2)
        return std::unexpected("no zarr metadata found at path: " + norm);

    std::string meta_key = is_v3 ? v3_key : v2_key;
    auto raw = store.get(meta_key);
    if (!raw) return std::unexpected(raw.error());

    std::string json_str(raw->begin(), raw->end());
    auto j = Json::parse(json_str).value();

    std::expected<MetadataFields, std::string> mf;
    if (is_v3) {
        mf = parse_v3_metadata(j);
    } else {
        mf = parse_v2_metadata(j);
    }
    if (!mf) return std::unexpected(mf.error());

    auto meta_impl = std::make_unique<ArrayMetadata::Impl>();
    meta_impl->version = mf->version;
    meta_impl->shape = std::move(mf->shape);
    meta_impl->chunks = std::move(mf->chunks);
    meta_impl->dtype = mf->dtype;
    meta_impl->order = mf->order;
    meta_impl->fill_value = mf->fill_value;
    meta_impl->codecs = std::move(mf->codecs);
    meta_impl->chunk_key_encoding = mf->chunk_key_encoding;
    meta_impl->dimension_names = std::move(mf->dimension_names);

    CodecPipeline pipeline;
    if (!meta_impl->codecs.empty()) {
        if (is_v3) {
            pipeline = build_pipeline(meta_impl->codecs, meta_impl->chunks,
                                     dtype_size(meta_impl->dtype));
        } else {
            pipeline = build_v2_pipeline(
                j.contains("compressor") ? j["compressor"] : Json::null_val(),
                j.contains("filters") ? j["filters"] : Json::null_val(),
                meta_impl->dtype);
        }
    }

    auto impl = std::make_unique<ZarrArray::Impl>();
    impl->store = &store;
    impl->path = norm;
    impl->meta = ArrayMetadata(std::move(meta_impl));
    impl->pipeline = std::move(pipeline);
    return ZarrArray(std::move(impl));
}

auto ZarrArray::create(Store& store, std::string_view path, const CreateOptions& opts)
    -> std::expected<ZarrArray, std::string> {
    auto norm = normalize_path(path);
    if (opts.shape.empty())
        return std::unexpected("shape must not be empty");
    if (!opts.chunks.empty() && opts.chunks.size() != opts.shape.size())
        return std::unexpected("chunks must have same length as shape");

    std::vector<size_t> chunks = opts.chunks;
    if (chunks.empty()) {
        // Auto-chunk: use shape dims clamped to reasonable sizes (min 1)
        for (auto s : opts.shape)
            chunks.push_back(std::max(size_t(1), std::min(s, size_t(1024))));
    }

    std::vector<CodecConfig> codecs;
    if (!opts.codecs.has_value()) {
        // No codecs specified: use sensible defaults
        if (opts.version == ZarrVersion::V3) {
            codecs.push_back(CodecConfig::bytes());
            codecs.push_back(CodecConfig::blosc("lz4", 5, 1, static_cast<int>(dtype_size(opts.dtype)), 0));
        } else {
            codecs.push_back(CodecConfig::blosc("lz4", 5, 1, static_cast<int>(dtype_size(opts.dtype)), 0));
        }
    } else {
        codecs = *opts.codecs;
    }

    auto meta_impl = std::make_unique<ArrayMetadata::Impl>();
    meta_impl->version = opts.version;
    meta_impl->shape = opts.shape;
    meta_impl->chunks = chunks;
    meta_impl->dtype = opts.dtype;
    meta_impl->order = opts.order;
    meta_impl->fill_value = opts.fill_value;
    meta_impl->codecs = codecs;
    meta_impl->chunk_key_encoding = opts.chunk_key_encoding;
    meta_impl->dimension_names = opts.dimension_names;

    auto impl = std::make_unique<ZarrArray::Impl>();
    impl->store = &store;
    impl->path = norm;
    impl->meta = ArrayMetadata(std::move(meta_impl));
    impl->pipeline = build_pipeline(codecs, chunks, dtype_size(opts.dtype));

    // Write metadata
    auto wr = impl->write_metadata();
    if (!wr) return std::unexpected(wr.error());

    // Write attributes if provided
    if (!opts.attributes_json.empty()) {
        if (opts.version == ZarrVersion::V2) {
            auto attrs_key = join_path(norm, ".zattrs");
            auto r = store.set(attrs_key, std::span<const uint8_t>(
                reinterpret_cast<const uint8_t*>(opts.attributes_json.data()),
                opts.attributes_json.size()));
            if (!r) return std::unexpected(r.error());
        }
        // For v3, attributes are embedded in zarr.json
        // We would need to re-serialize with attributes; keep it simple for now
    }

    return ZarrArray(std::move(impl));
}

auto ZarrArray::read_chunk(std::span<const size_t> indices) const
    -> std::expected<std::vector<uint8_t>, std::string> {
    auto key = chunk_key(impl_->path, indices, impl_->meta.chunk_key_encoding(), impl_->meta.version());
    if (!impl_->store->exists(key)) {
        return impl_->fill_chunk_buffer();
    }
    auto raw = impl_->store->get(key);
    if (!raw) return std::unexpected(raw.error());
    if (impl_->pipeline.codecs.empty()) return *raw;
    return impl_->pipeline.decode(*raw);
}

auto ZarrArray::write_chunk(std::span<const size_t> indices, std::span<const uint8_t> data)
    -> std::expected<void, std::string> {
    auto key = chunk_key(impl_->path, indices, impl_->meta.chunk_key_encoding(), impl_->meta.version());
    if (impl_->pipeline.codecs.empty()) {
        return impl_->store->set(key, data);
    }
    auto encoded = impl_->pipeline.encode(data);
    if (!encoded) return std::unexpected(encoded.error());
    return impl_->store->set(key, *encoded);
}

auto ZarrArray::read_region(std::span<const size_t> offset, std::span<const size_t> region_shape,
                            void* out) const -> std::expected<void, std::string> {
    size_t ndim = impl_->meta.shape().size();
    if (offset.size() != ndim || region_shape.size() != ndim)
        return std::unexpected("offset/shape must match array ndim");

    auto chunk_shape = impl_->meta.chunks();
    size_t esz = dtype_size(impl_->meta.dtype());

    // Compute output strides (row-major)
    std::vector<size_t> out_strides(ndim);
    if (ndim > 0) {
        out_strides[ndim - 1] = esz;
        for (size_t d = ndim - 1; d > 0; --d)
            out_strides[d - 1] = out_strides[d] * region_shape[d];
    }

    // Compute which chunk indices we need
    std::vector<size_t> chunk_start(ndim), chunk_end(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        chunk_start[d] = offset[d] / chunk_shape[d];
        chunk_end[d] = (offset[d] + region_shape[d] + chunk_shape[d] - 1) / chunk_shape[d];
    }

    // Iterate over all chunks in the region
    std::vector<size_t> ci(ndim);
    for (size_t d = 0; d < ndim; ++d) ci[d] = chunk_start[d];

    while (true) {
        // Read this chunk
        auto chunk_data = read_chunk(ci);
        if (!chunk_data) return std::unexpected(chunk_data.error());
        auto* chunk_ptr = chunk_data->data();

        // Compute chunk strides (row-major)
        std::vector<size_t> cstrides(ndim);
        cstrides[ndim - 1] = esz;
        for (size_t d = ndim - 1; d > 0; --d)
            cstrides[d - 1] = cstrides[d] * chunk_shape[d];

        // Compute intersection of this chunk with the region
        std::vector<size_t> src_off(ndim), dst_off(ndim), copy_shape(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            size_t chunk_global_start = ci[d] * chunk_shape[d];
            size_t chunk_global_end = chunk_global_start + chunk_shape[d];
            size_t region_start = offset[d];
            size_t region_end = offset[d] + region_shape[d];
            size_t inter_start = std::max(chunk_global_start, region_start);
            size_t inter_end = std::min(chunk_global_end, region_end);
            src_off[d] = inter_start - chunk_global_start;
            dst_off[d] = inter_start - region_start;
            copy_shape[d] = inter_end - inter_start;
        }

        // Copy elements from chunk to output using nested iteration
        // For simplicity, use a flat iteration over copy_shape
        std::vector<size_t> idx(ndim, 0);
        size_t total_elems = 1;
        for (size_t d = 0; d < ndim; ++d) total_elems *= copy_shape[d];

        for (size_t n = 0; n < total_elems; ++n) {
            // Compute source and dest byte offsets
            size_t src_byte = 0, dst_byte = 0;
            for (size_t d = 0; d < ndim; ++d) {
                src_byte += (src_off[d] + idx[d]) * cstrides[d];
                dst_byte += (dst_off[d] + idx[d]) * out_strides[d];
            }
            std::memcpy(static_cast<uint8_t*>(out) + dst_byte, chunk_ptr + src_byte, esz);
            // Increment idx
            for (size_t d = ndim; d > 0; --d) {
                ++idx[d - 1];
                if (idx[d - 1] < copy_shape[d - 1]) break;
                idx[d - 1] = 0;
            }
        }

        // Advance to next chunk
        bool done = false;
        for (size_t d = ndim; d > 0; --d) {
            ++ci[d - 1];
            if (ci[d - 1] < chunk_end[d - 1]) break;
            ci[d - 1] = chunk_start[d - 1];
            if (d == 1) done = true;
        }
        if (done) break;
    }
    return {};
}

auto ZarrArray::write_region(std::span<const size_t> offset, std::span<const size_t> region_shape,
                             const void* data) -> std::expected<void, std::string> {
    size_t ndim = impl_->meta.shape().size();
    if (offset.size() != ndim || region_shape.size() != ndim)
        return std::unexpected("offset/shape must match array ndim");

    auto chunk_shape = impl_->meta.chunks();
    size_t esz = dtype_size(impl_->meta.dtype());

    // Input strides (row-major)
    std::vector<size_t> in_strides(ndim);
    if (ndim > 0) {
        in_strides[ndim - 1] = esz;
        for (size_t d = ndim - 1; d > 0; --d)
            in_strides[d - 1] = in_strides[d] * region_shape[d];
    }

    std::vector<size_t> chunk_start(ndim), chunk_end(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        chunk_start[d] = offset[d] / chunk_shape[d];
        chunk_end[d] = (offset[d] + region_shape[d] + chunk_shape[d] - 1) / chunk_shape[d];
    }

    std::vector<size_t> ci(ndim);
    for (size_t d = 0; d < ndim; ++d) ci[d] = chunk_start[d];

    while (true) {
        // Compute chunk strides
        std::vector<size_t> cstrides(ndim);
        cstrides[ndim - 1] = esz;
        for (size_t d = ndim - 1; d > 0; --d)
            cstrides[d - 1] = cstrides[d] * chunk_shape[d];

        // Compute intersection
        std::vector<size_t> src_off(ndim), dst_off(ndim), copy_shape(ndim);
        bool partial = false;
        for (size_t d = 0; d < ndim; ++d) {
            size_t chunk_global_start = ci[d] * chunk_shape[d];
            size_t chunk_global_end = chunk_global_start + chunk_shape[d];
            size_t region_start = offset[d];
            size_t region_end = offset[d] + region_shape[d];
            size_t inter_start = std::max(chunk_global_start, region_start);
            size_t inter_end = std::min(chunk_global_end, region_end);
            dst_off[d] = inter_start - chunk_global_start; // offset within chunk
            src_off[d] = inter_start - region_start;       // offset within input
            copy_shape[d] = inter_end - inter_start;
            if (copy_shape[d] != chunk_shape[d]) partial = true;
        }

        // Read existing chunk if partial
        std::vector<uint8_t> chunk_buf;
        if (partial) {
            auto existing = read_chunk(ci);
            if (!existing) return std::unexpected(existing.error());
            chunk_buf = std::move(*existing);
        } else {
            chunk_buf.resize(impl_->chunk_bytes(), 0);
        }

        // Copy from input to chunk buffer
        std::vector<size_t> idx(ndim, 0);
        size_t total_elems = 1;
        for (size_t d = 0; d < ndim; ++d) total_elems *= copy_shape[d];

        for (size_t n = 0; n < total_elems; ++n) {
            size_t chunk_byte = 0, input_byte = 0;
            for (size_t d = 0; d < ndim; ++d) {
                chunk_byte += (dst_off[d] + idx[d]) * cstrides[d];
                input_byte += (src_off[d] + idx[d]) * in_strides[d];
            }
            std::memcpy(chunk_buf.data() + chunk_byte,
                        static_cast<const uint8_t*>(data) + input_byte, esz);
            for (size_t d = ndim; d > 0; --d) {
                ++idx[d - 1];
                if (idx[d - 1] < copy_shape[d - 1]) break;
                idx[d - 1] = 0;
            }
        }

        // Write chunk
        auto wr = write_chunk(ci, chunk_buf);
        if (!wr) return std::unexpected(wr.error());

        // Advance
        bool done = false;
        for (size_t d = ndim; d > 0; --d) {
            ++ci[d - 1];
            if (ci[d - 1] < chunk_end[d - 1]) break;
            ci[d - 1] = chunk_start[d - 1];
            if (d == 1) done = true;
        }
        if (done) break;
    }
    return {};
}

auto ZarrArray::read_tensor() const -> std::expected<Tensor, std::string> {
    auto s = impl_->meta.shape();
    std::vector<size_t> shape_vec(s.begin(), s.end());
    auto t = Tensor::zeros(shape_vec, impl_->meta.dtype());
    if (!t) return std::unexpected(t.error());
    std::vector<size_t> offset(shape_vec.size(), 0);
    auto r = read_region(offset, shape_vec, t->data_ptr());
    if (!r) return std::unexpected(r.error());
    return std::move(*t);
}

auto ZarrArray::read_tensor(std::span<const size_t> offset,
                            std::span<const size_t> region_shape) const
    -> std::expected<Tensor, std::string> {
    std::vector<size_t> shape_vec(region_shape.begin(), region_shape.end());
    auto t = Tensor::zeros(shape_vec, impl_->meta.dtype());
    if (!t) return std::unexpected(t.error());
    auto r = read_region(offset, region_shape, t->data_ptr());
    if (!r) return std::unexpected(r.error());
    return std::move(*t);
}

auto ZarrArray::write_tensor(const Tensor& tensor) -> std::expected<void, std::string> {
    auto s = impl_->meta.shape();
    if (tensor.ndim() != s.size())
        return std::unexpected("tensor ndim doesn't match array ndim");
    for (size_t d = 0; d < s.size(); ++d) {
        if (tensor.shape()[d] != s[d])
            return std::unexpected("tensor shape doesn't match array shape at dim " + std::to_string(d));
    }
    Tensor contig = tensor.is_contiguous() ? Tensor(tensor) : tensor.contiguous();
    std::vector<size_t> offset(s.size(), 0);
    std::vector<size_t> shape_vec(s.begin(), s.end());
    return write_region(offset, shape_vec, contig.data_ptr());
}

auto ZarrArray::write_tensor(const Tensor& tensor, std::span<const size_t> offset)
    -> std::expected<void, std::string> {
    Tensor contig = tensor.is_contiguous() ? Tensor(tensor) : tensor.contiguous();
    std::vector<size_t> shape_vec(tensor.shape().begin(), tensor.shape().end());
    return write_region(offset, shape_vec, contig.data_ptr());
}

auto ZarrArray::metadata() const -> const ArrayMetadata& { return impl_->meta; }
auto ZarrArray::shape() const -> std::span<const size_t> { return impl_->meta.shape(); }
auto ZarrArray::chunks() const -> std::span<const size_t> { return impl_->meta.chunks(); }
auto ZarrArray::dtype() const -> DType { return impl_->meta.dtype(); }
auto ZarrArray::ndim() const -> size_t { return impl_->meta.shape().size(); }
auto ZarrArray::version() const -> ZarrVersion { return impl_->meta.version(); }

auto ZarrArray::attributes() const -> std::expected<std::string, std::string> {
    if (impl_->meta.version() == ZarrVersion::V3) {
        auto key = join_path(impl_->path, "zarr.json");
        auto raw = impl_->store->get(key);
        if (!raw) return std::unexpected(raw.error());
        std::string json_str(raw->begin(), raw->end());
        auto j = Json::parse(json_str).value();
        if (j.contains("attributes"))
            return j["attributes"].serialize();
        return std::string("{}");
    }
    auto key = join_path(impl_->path, ".zattrs");
    if (!impl_->store->exists(key)) return std::string("{}");
    auto raw = impl_->store->get(key);
    if (!raw) return std::unexpected(raw.error());
    return std::string(raw->begin(), raw->end());
}

auto ZarrArray::set_attributes(std::string_view json_str) -> std::expected<void, std::string> {
    if (impl_->meta.version() == ZarrVersion::V3) {
        // Read existing zarr.json, add/update attributes
        auto key = join_path(impl_->path, "zarr.json");
        auto raw = impl_->store->get(key);
        if (!raw) return std::unexpected(raw.error());
        std::string existing(raw->begin(), raw->end());
        auto j = Json::parse(existing).value();
        auto attrs = Json::parse(json_str).value();
        j.as_object_mut()["attributes"] = attrs;
        auto serialized = j.serialize();
        return impl_->store->set(key, std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()));
    }
    auto key = join_path(impl_->path, ".zattrs");
    return impl_->store->set(key, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(json_str.data()), json_str.size()));
}

auto ZarrArray::resize(std::span<const size_t> new_shape) -> std::expected<void, std::string> {
    if (new_shape.size() != impl_->meta.shape().size())
        return std::unexpected("new_shape must have same ndim as existing array");
    impl_->meta.impl_->shape.assign(new_shape.begin(), new_shape.end());
    return impl_->write_metadata();
}

// Path-based open factory
auto ZarrArray::open(const std::filesystem::path& path)
    -> std::expected<ZarrArray, std::string> {
    auto store_result = FilesystemStore::open(path);
    if (!store_result) return std::unexpected(store_result.error());
    std::shared_ptr<Store> owned(store_result->release());
    auto arr = ZarrArray::open(*owned, "");
    if (!arr) return std::unexpected(arr.error());
    arr->impl_->owned_store_ = std::move(owned);
    return arr;
}

// Computed properties
auto ZarrArray::size() const -> size_t {
    auto s = shape();
    size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

auto ZarrArray::itemsize() const -> size_t {
    return dtype_size(dtype());
}

auto ZarrArray::nbytes() const -> size_t {
    return size() * itemsize();
}

auto ZarrArray::nchunks() const -> size_t {
    auto s = shape();
    auto c = chunks();
    size_t n = 1;
    for (size_t d = 0; d < s.size(); ++d)
        n *= (s[d] + c[d] - 1) / c[d];
    return n;
}

auto ZarrArray::nchunks_initialized() const -> size_t {
    auto prefix = impl_->path.empty() ? std::string("") : impl_->path;
    auto keys = impl_->store->list_prefix(prefix);
    size_t count = 0;
    for (auto& k : keys) {
        // Skip metadata files
        if (k.ends_with("zarr.json") || k.ends_with(".zarray") ||
            k.ends_with(".zattrs") || k.ends_with(".zgroup") ||
            k.ends_with(".zmetadata"))
            continue;
        ++count;
    }
    return count;
}

auto ZarrArray::fill_value() const -> double {
    return impl_->meta.fill_value();
}

auto ZarrArray::order() const -> Order {
    return impl_->meta.order();
}

auto ZarrArray::path() const -> std::string {
    return impl_->path;
}

auto ZarrArray::name() const -> std::string {
    if (impl_->path.empty()) return "/";
    auto pos = impl_->path.rfind('/');
    if (pos == std::string::npos) return impl_->path;
    return impl_->path.substr(pos + 1);
}

auto ZarrArray::read_only() const -> bool {
    return impl_->read_only_;
}

auto ZarrArray::store_ref() const -> const Store& {
    return *impl_->store;
}

auto ZarrArray::info() const -> std::string {
    std::string s = "Type        : ZarrArray\n";
    s += "Path        : " + (impl_->path.empty() ? "/" : impl_->path) + "\n";
    s += "Shape       : (";
    auto sh = shape();
    for (size_t i = 0; i < sh.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(sh[i]);
    }
    s += ")\nChunks      : (";
    auto ch = chunks();
    for (size_t i = 0; i < ch.size(); ++i) {
        if (i > 0) s += ", ";
        s += std::to_string(ch[i]);
    }
    s += ")\nDType       : " + std::string(dtype_name(dtype())) + "\n";
    s += "Order       : " + std::string(order() == Order::C ? "C" : "F") + "\n";
    s += "Read-only   : " + std::string(impl_->read_only_ ? "true" : "false") + "\n";
    s += "Store       : " + std::string(impl_->owned_store_ ? "owned" : "external") + "\n";
    return s;
}

// Granular attribute methods for ZarrArray
auto ZarrArray::get_attribute(std::string_view key) const
    -> std::expected<std::string, std::string> {
    auto attrs_str = attributes();
    if (!attrs_str) return std::unexpected(attrs_str.error());
    auto j = Json::parse(*attrs_str);
    if (!j) return std::unexpected("failed to parse attributes");
    if (!j->contains(key))
        return std::unexpected("attribute not found: " + std::string(key));
    return (*j)[key].serialize();
}

auto ZarrArray::set_attribute(std::string_view key, std::string_view value_json)
    -> std::expected<void, std::string> {
    auto attrs_str = attributes();
    if (!attrs_str) return std::unexpected(attrs_str.error());
    auto j = Json::parse(*attrs_str);
    if (!j) return std::unexpected("failed to parse attributes");
    auto val = Json::parse(value_json);
    if (!val) return std::unexpected("failed to parse value JSON");
    j->as_object_mut()[std::string(key)] = std::move(*val);
    return set_attributes(j->serialize());
}

auto ZarrArray::delete_attribute(std::string_view key)
    -> std::expected<void, std::string> {
    auto attrs_str = attributes();
    if (!attrs_str) return std::unexpected(attrs_str.error());
    auto j = Json::parse(*attrs_str);
    if (!j) return std::unexpected("failed to parse attributes");
    auto& obj = j->as_object_mut();
    auto it = obj.find(key);
    if (it != obj.end()) obj.erase(it);
    return set_attributes(j->serialize());
}

auto ZarrArray::contains_attribute(std::string_view key) const -> bool {
    auto attrs_str = attributes();
    if (!attrs_str) return false;
    auto j = Json::parse(*attrs_str);
    if (!j) return false;
    return j->contains(key);
}

auto ZarrArray::attribute_keys() const -> std::vector<std::string> {
    std::vector<std::string> result;
    auto attrs_str = attributes();
    if (!attrs_str) return result;
    auto j = Json::parse(*attrs_str);
    if (!j || !j->is_object()) return result;
    for (auto& [k, v] : j->as_object()) result.push_back(k);
    return result;
}

auto ZarrArray::num_attributes() const -> size_t {
    auto attrs_str = attributes();
    if (!attrs_str) return 0;
    auto j = Json::parse(*attrs_str);
    if (!j || !j->is_object()) return 0;
    return j->as_object().size();
}

// Append
auto ZarrArray::append(const Tensor& data, size_t axis)
    -> std::expected<void, std::string> {
    auto s = shape();
    if (axis >= s.size())
        return std::unexpected("axis " + std::to_string(axis) + " out of range for " +
                              std::to_string(s.size()) + "-d array");
    if (data.ndim() != s.size())
        return std::unexpected("data ndim doesn't match array ndim");
    // Validate all non-axis dims match
    for (size_t d = 0; d < s.size(); ++d) {
        if (d != axis && data.shape()[d] != s[d])
            return std::unexpected("data shape mismatch at dim " + std::to_string(d) +
                                  ": expected " + std::to_string(s[d]) +
                                  " got " + std::to_string(data.shape()[d]));
    }
    size_t old_extent = s[axis];
    size_t append_extent = data.shape()[axis];
    // Resize
    std::vector<size_t> new_shape(s.begin(), s.end());
    new_shape[axis] = old_extent + append_extent;
    auto r = resize(new_shape);
    if (!r) return std::unexpected(r.error());
    // Write at offset
    std::vector<size_t> offset(s.size(), 0);
    offset[axis] = old_extent;
    return write_tensor(data, offset);
}

void ZarrArray::set_owned_store(std::shared_ptr<Store> store) {
    impl_->owned_store_ = std::move(store);
}

void ZarrArray::set_read_only(bool ro) {
    impl_->read_only_ = ro;
}

// ── ZarrGroup ───────────────────────────────────────────────────────────────

struct ZarrGroup::Impl {
    Store* store = nullptr;
    std::shared_ptr<Store> owned_store_;
    bool read_only_ = false;
    std::string path;
    ZarrVersion version = ZarrVersion::V3;
};

ZarrGroup::ZarrGroup(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ZarrGroup::~ZarrGroup() = default;
ZarrGroup::ZarrGroup(ZarrGroup&&) noexcept = default;
auto ZarrGroup::operator=(ZarrGroup&&) noexcept -> ZarrGroup& = default;

auto ZarrGroup::open(Store& store, std::string_view path)
    -> std::expected<ZarrGroup, std::string> {
    auto norm = normalize_path(path);
    // Try v3 first
    auto v3_key = join_path(norm, "zarr.json");
    auto v2_key = join_path(norm, ".zgroup");

    if (store.exists(v3_key)) {
        auto raw = store.get(v3_key);
        if (!raw) return std::unexpected(raw.error());
        std::string json_str(raw->begin(), raw->end());
        auto j = Json::parse(json_str).value();
        if (j.contains("node_type") && j["node_type"].as_string() == "group") {
            auto impl = std::make_unique<ZarrGroup::Impl>();
            impl->store = &store;
            impl->path = norm;
            impl->version = ZarrVersion::V3;
            return ZarrGroup(std::move(impl));
        }
        // If it's an array, not a group
        if (j.contains("node_type") && j["node_type"].as_string() == "array")
            return std::unexpected("path is an array, not a group: " + norm);
    }

    if (store.exists(v2_key)) {
        auto impl = std::make_unique<ZarrGroup::Impl>();
        impl->store = &store;
        impl->path = norm;
        impl->version = ZarrVersion::V2;
        return ZarrGroup(std::move(impl));
    }

    return std::unexpected("no zarr group found at path: " + norm);
}

auto ZarrGroup::create(Store& store, std::string_view path, ZarrVersion version)
    -> std::expected<ZarrGroup, std::string> {
    auto norm = normalize_path(path);
    if (version == ZarrVersion::V3) {
        std::map<std::string, Json, std::less<>> obj;
        obj["zarr_format"] = Json::number(3);
        obj["node_type"] = Json::string_val("group");
        auto json_str = Json::object(std::move(obj)).serialize();
        auto key = join_path(norm, "zarr.json");
        auto r = store.set(key, std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(json_str.data()), json_str.size()));
        if (!r) return std::unexpected(r.error());
    } else {
        std::map<std::string, Json, std::less<>> obj;
        obj["zarr_format"] = Json::number(2);
        auto json_str = Json::object(std::move(obj)).serialize();
        auto key = join_path(norm, ".zgroup");
        auto r = store.set(key, std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(json_str.data()), json_str.size()));
        if (!r) return std::unexpected(r.error());
    }
    auto impl = std::make_unique<ZarrGroup::Impl>();
    impl->store = &store;
    impl->path = norm;
    impl->version = version;
    return ZarrGroup(std::move(impl));
}

auto ZarrGroup::open_array(std::string_view name) const
    -> std::expected<ZarrArray, std::string> {
    auto child = join_path(impl_->path, name);
    return ZarrArray::open(*impl_->store, child);
}

auto ZarrGroup::create_array(std::string_view name, const CreateOptions& opts) const
    -> std::expected<ZarrArray, std::string> {
    auto child = join_path(impl_->path, name);
    CreateOptions child_opts = opts;
    if (child_opts.version != impl_->version) {
        // Match group version by default if user didn't customize
    }
    return ZarrArray::create(*impl_->store, child, child_opts);
}

auto ZarrGroup::open_group(std::string_view name) const
    -> std::expected<ZarrGroup, std::string> {
    auto child = join_path(impl_->path, name);
    return ZarrGroup::open(*impl_->store, child);
}

auto ZarrGroup::create_group(std::string_view name) const
    -> std::expected<ZarrGroup, std::string> {
    auto child = join_path(impl_->path, name);
    return ZarrGroup::create(*impl_->store, child, impl_->version);
}

auto ZarrGroup::list_arrays() const -> std::vector<std::string> {
    std::vector<std::string> result;
    auto [keys, prefixes] = impl_->store->list_dir(impl_->path);
    for (auto& name : prefixes) {
        auto child = join_path(impl_->path, name);
        // Check for array markers
        if (impl_->version == ZarrVersion::V3) {
            auto zk = join_path(child, "zarr.json");
            if (impl_->store->exists(zk)) {
                auto raw = impl_->store->get(zk);
                if (raw) {
                    std::string s(raw->begin(), raw->end());
                    auto j = Json::parse(s).value();
                    if (j.contains("node_type") && j["node_type"].as_string() == "array")
                        result.push_back(name);
                }
            }
        } else {
            if (impl_->store->exists(join_path(child, ".zarray")))
                result.push_back(name);
        }
    }
    return result;
}

auto ZarrGroup::list_groups() const -> std::vector<std::string> {
    std::vector<std::string> result;
    auto [keys, prefixes] = impl_->store->list_dir(impl_->path);
    for (auto& name : prefixes) {
        auto child = join_path(impl_->path, name);
        if (impl_->version == ZarrVersion::V3) {
            auto zk = join_path(child, "zarr.json");
            if (impl_->store->exists(zk)) {
                auto raw = impl_->store->get(zk);
                if (raw) {
                    std::string s(raw->begin(), raw->end());
                    auto j = Json::parse(s).value();
                    if (j.contains("node_type") && j["node_type"].as_string() == "group")
                        result.push_back(name);
                }
            }
        } else {
            if (impl_->store->exists(join_path(child, ".zgroup")))
                result.push_back(name);
        }
    }
    return result;
}

auto ZarrGroup::list_children() const
    -> std::pair<std::vector<std::string>, std::vector<std::string>> {
    return {list_arrays(), list_groups()};
}

auto ZarrGroup::contains_array(std::string_view name) const -> bool {
    auto child = join_path(impl_->path, name);
    if (impl_->version == ZarrVersion::V3) {
        auto zk = join_path(child, "zarr.json");
        if (!impl_->store->exists(zk)) return false;
        auto raw = impl_->store->get(zk);
        if (!raw) return false;
        std::string s(raw->begin(), raw->end());
        auto j = Json::parse(s).value();
        return j.contains("node_type") && j["node_type"].as_string() == "array";
    }
    return impl_->store->exists(join_path(child, ".zarray"));
}

auto ZarrGroup::contains_group(std::string_view name) const -> bool {
    auto child = join_path(impl_->path, name);
    if (impl_->version == ZarrVersion::V3) {
        auto zk = join_path(child, "zarr.json");
        if (!impl_->store->exists(zk)) return false;
        auto raw = impl_->store->get(zk);
        if (!raw) return false;
        std::string s(raw->begin(), raw->end());
        auto j = Json::parse(s).value();
        return j.contains("node_type") && j["node_type"].as_string() == "group";
    }
    return impl_->store->exists(join_path(child, ".zgroup"));
}

auto ZarrGroup::attributes() const -> std::expected<std::string, std::string> {
    if (impl_->version == ZarrVersion::V3) {
        auto key = join_path(impl_->path, "zarr.json");
        auto raw = impl_->store->get(key);
        if (!raw) return std::unexpected(raw.error());
        std::string json_str(raw->begin(), raw->end());
        auto j = Json::parse(json_str).value();
        if (j.contains("attributes"))
            return j["attributes"].serialize();
        return std::string("{}");
    }
    auto key = join_path(impl_->path, ".zattrs");
    if (!impl_->store->exists(key)) return std::string("{}");
    auto raw = impl_->store->get(key);
    if (!raw) return std::unexpected(raw.error());
    return std::string(raw->begin(), raw->end());
}

auto ZarrGroup::set_attributes(std::string_view json_str) -> std::expected<void, std::string> {
    if (impl_->version == ZarrVersion::V3) {
        auto key = join_path(impl_->path, "zarr.json");
        auto raw = impl_->store->get(key);
        if (!raw) return std::unexpected(raw.error());
        std::string existing(raw->begin(), raw->end());
        auto j = Json::parse(existing).value();
        auto attrs = Json::parse(json_str).value();
        j.as_object_mut()["attributes"] = attrs;
        auto serialized = j.serialize();
        return impl_->store->set(key, std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()));
    }
    auto key = join_path(impl_->path, ".zattrs");
    return impl_->store->set(key, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(json_str.data()), json_str.size()));
}

// Path-based open factory
auto ZarrGroup::open(const std::filesystem::path& path)
    -> std::expected<ZarrGroup, std::string> {
    auto store_result = FilesystemStore::open(path);
    if (!store_result) return std::unexpected(store_result.error());
    std::shared_ptr<Store> owned(store_result->release());
    auto grp = ZarrGroup::open(*owned, "");
    if (!grp) return std::unexpected(grp.error());
    grp->impl_->owned_store_ = std::move(owned);
    return grp;
}

// Navigation properties
auto ZarrGroup::path() const -> std::string {
    return impl_->path;
}

auto ZarrGroup::name() const -> std::string {
    if (impl_->path.empty()) return "/";
    auto pos = impl_->path.rfind('/');
    if (pos == std::string::npos) return impl_->path;
    return impl_->path.substr(pos + 1);
}

auto ZarrGroup::store_ref() const -> const Store& {
    return *impl_->store;
}

auto ZarrGroup::version() const -> ZarrVersion {
    return impl_->version;
}

auto ZarrGroup::nchildren() const -> size_t {
    auto [arrs, grps] = list_children();
    return arrs.size() + grps.size();
}

auto ZarrGroup::read_only() const -> bool {
    return impl_->read_only_;
}

auto ZarrGroup::info() const -> std::string {
    std::string s = "Type        : ZarrGroup\n";
    s += "Path        : " + (impl_->path.empty() ? "/" : impl_->path) + "\n";
    s += "Version     : " + std::string(impl_->version == ZarrVersion::V3 ? "3" : "2") + "\n";
    auto [arrs, grps] = list_children();
    s += "Arrays      : " + std::to_string(arrs.size()) + "\n";
    s += "Groups      : " + std::to_string(grps.size()) + "\n";
    s += "Read-only   : " + std::string(impl_->read_only_ ? "true" : "false") + "\n";
    return s;
}

// Granular attribute methods for ZarrGroup
auto ZarrGroup::get_attribute(std::string_view key) const
    -> std::expected<std::string, std::string> {
    auto attrs_str = attributes();
    if (!attrs_str) return std::unexpected(attrs_str.error());
    auto j = Json::parse(*attrs_str);
    if (!j) return std::unexpected("failed to parse attributes");
    if (!j->contains(key))
        return std::unexpected("attribute not found: " + std::string(key));
    return (*j)[key].serialize();
}

auto ZarrGroup::set_attribute(std::string_view key, std::string_view value_json)
    -> std::expected<void, std::string> {
    auto attrs_str = attributes();
    if (!attrs_str) return std::unexpected(attrs_str.error());
    auto j = Json::parse(*attrs_str);
    if (!j) return std::unexpected("failed to parse attributes");
    auto val = Json::parse(value_json);
    if (!val) return std::unexpected("failed to parse value JSON");
    j->as_object_mut()[std::string(key)] = std::move(*val);
    return set_attributes(j->serialize());
}

auto ZarrGroup::delete_attribute(std::string_view key)
    -> std::expected<void, std::string> {
    auto attrs_str = attributes();
    if (!attrs_str) return std::unexpected(attrs_str.error());
    auto j = Json::parse(*attrs_str);
    if (!j) return std::unexpected("failed to parse attributes");
    auto& obj = j->as_object_mut();
    auto it = obj.find(key);
    if (it != obj.end()) obj.erase(it);
    return set_attributes(j->serialize());
}

auto ZarrGroup::contains_attribute(std::string_view key) const -> bool {
    auto attrs_str = attributes();
    if (!attrs_str) return false;
    auto j = Json::parse(*attrs_str);
    if (!j) return false;
    return j->contains(key);
}

auto ZarrGroup::attribute_keys() const -> std::vector<std::string> {
    std::vector<std::string> result;
    auto attrs_str = attributes();
    if (!attrs_str) return result;
    auto j = Json::parse(*attrs_str);
    if (!j || !j->is_object()) return result;
    for (auto& [k, v] : j->as_object()) result.push_back(k);
    return result;
}

auto ZarrGroup::num_attributes() const -> size_t {
    auto attrs_str = attributes();
    if (!attrs_str) return 0;
    auto j = Json::parse(*attrs_str);
    if (!j || !j->is_object()) return 0;
    return j->as_object().size();
}

void ZarrGroup::set_owned_store(std::shared_ptr<Store> store) {
    impl_->owned_store_ = std::move(store);
}

void ZarrGroup::set_read_only(bool ro) {
    impl_->read_only_ = ro;
}

auto ZarrGroup::consolidate_metadata() -> std::expected<void, std::string> {
    if (impl_->version != ZarrVersion::V2)
        return std::unexpected("consolidate_metadata is only supported for v2");

    std::map<std::string, Json, std::less<>> metadata_map;

    // Collect all metadata from store
    auto all_keys = impl_->store->list_prefix(impl_->path);
    for (auto& key : all_keys) {
        // Check if key ends with .zarray, .zgroup, or .zattrs
        bool is_meta = false;
        for (auto suffix : {".zarray", ".zgroup", ".zattrs"}) {
            if (key.ends_with(suffix)) { is_meta = true; break; }
        }
        if (!is_meta) continue;
        auto raw = impl_->store->get(key);
        if (!raw) continue;
        std::string json_str(raw->begin(), raw->end());
        auto j = Json::parse(json_str).value();
        // Store relative to group path
        std::string rel_key = key;
        if (!impl_->path.empty() && key.starts_with(impl_->path + "/"))
            rel_key = key.substr(impl_->path.size() + 1);
        metadata_map[rel_key] = std::move(j);
    }

    std::map<std::string, Json, std::less<>> zmeta;
    zmeta["zarr_consolidated_format"] = Json::number(1);
    zmeta["metadata"] = Json::object(std::move(metadata_map));

    auto serialized = Json::object(std::move(zmeta)).serialize();
    auto key = join_path(impl_->path, ".zmetadata");
    return impl_->store->set(key, std::span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size()));
}

// ── Free functions ──────────────────────────────────────────────────────────

auto save_zarr(const Tensor& t, const std::filesystem::path& path,
               ZarrVersion v, std::optional<std::vector<CodecConfig>> codecs)
    -> std::expected<void, std::string> {
    auto store_result = FilesystemStore::open(path);
    if (!store_result) return std::unexpected(store_result.error());
    auto& store = *store_result;

    CreateOptions opts;
    opts.shape.assign(t.shape().begin(), t.shape().end());
    opts.dtype = t.dtype();
    opts.version = v;
    opts.codecs = std::move(codecs);
    opts.chunk_key_encoding = (v == ZarrVersion::V3) ? ChunkKeyEncoding::V3Default
                                                     : ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(*store, "", opts);
    if (!arr) return std::unexpected(arr.error());
    return arr->write_tensor(t);
}

auto load_zarr(const std::filesystem::path& path) -> std::expected<Tensor, std::string> {
    auto store_result = FilesystemStore::open(path);
    if (!store_result) return std::unexpected(store_result.error());
    auto& store = *store_result;

    auto arr = ZarrArray::open(*store, "");
    if (!arr) return std::unexpected(arr.error());
    return arr->read_tensor();
}

auto detect_zarr_version(const Store& store, std::string_view path)
    -> std::expected<ZarrVersion, std::string> {
    auto norm = normalize_path(path);
    if (store.exists(join_path(norm, "zarr.json")))
        return ZarrVersion::V3;
    if (store.exists(join_path(norm, ".zarray")) || store.exists(join_path(norm, ".zgroup")))
        return ZarrVersion::V2;
    return std::unexpected("no zarr metadata found at path: " + std::string(norm));
}

auto open_zarr(const std::filesystem::path& path, OpenMode mode)
    -> std::expected<ZarrArray, std::string> {
    auto store_result = FilesystemStore::open(path);
    if (!store_result) return std::unexpected(store_result.error());
    std::shared_ptr<Store> owned(store_result->release());
    auto arr = ZarrArray::open(*owned, "");
    if (!arr) return std::unexpected(arr.error());
    arr->impl_->owned_store_ = std::move(owned);
    arr->impl_->read_only_ = (mode == OpenMode::Read);
    return arr;
}

auto open_zarr_group(const std::filesystem::path& path, OpenMode mode,
                      std::optional<ZarrVersion> /*version*/)
    -> std::expected<ZarrGroup, std::string> {
    auto store_result = FilesystemStore::open(path);
    if (!store_result) return std::unexpected(store_result.error());
    std::shared_ptr<Store> owned(store_result->release());
    auto grp = ZarrGroup::open(*owned, "");
    if (!grp) return std::unexpected(grp.error());
    grp->impl_->owned_store_ = std::move(owned);
    grp->impl_->read_only_ = (mode == OpenMode::Read);
    return grp;
}

} // namespace utils

// ============================================================================
// vc::Zarr, vc::OMEZarr, downsample helpers
// ============================================================================

#include <blosc.h>
#include <zstd.h>
#include <lz4.h>
#include <zlib.h>
#include <omp.h>

#include <atomic>
#include <iostream>
#include <nlohmann/json.hpp>

namespace vc {

// ============================================================================
// Compressor configuration (parsed from .zarray / zarr.json codecs)
// ============================================================================

namespace {

enum class CompressorId { None, Blosc, Zstd, Lz4, Gzip };

struct CompressorConfig {
    CompressorId id = CompressorId::None;
    // Blosc params
    std::string blosc_cname = "lz4";
    int blosc_clevel = 5;
    int blosc_shuffle = 1;
    int blosc_typesize = 1;
    int blosc_blocksize = 0;
    // Zstd/Gzip level
    int level = 3;
};

static CompressorConfig parseCompressorFromCodecs(
    std::span<const utils::CodecConfig> codecs, int dtypeSize)
{
    CompressorConfig cfg;
    cfg.blosc_typesize = dtypeSize;

    // Walk the codec list and pick the first compressor we recognise
    for (auto& cc : codecs) {
        switch (cc.id) {
            case utils::CodecId::Blosc:
                cfg.id = CompressorId::Blosc;
                cfg.blosc_cname = cc.blosc_cname;
                cfg.blosc_clevel = cc.blosc_clevel;
                cfg.blosc_shuffle = cc.blosc_shuffle;
                cfg.blosc_typesize = cc.blosc_typesize ? cc.blosc_typesize : dtypeSize;
                cfg.blosc_blocksize = cc.blosc_blocksize;
                return cfg;
            case utils::CodecId::Zstd:
                cfg.id = CompressorId::Zstd;
                cfg.level = cc.level >= 0 ? cc.level : 3;
                return cfg;
            case utils::CodecId::Lz4:
                cfg.id = CompressorId::Lz4;
                cfg.level = cc.acceleration;
                return cfg;
            case utils::CodecId::Gzip:
                cfg.id = CompressorId::Gzip;
                cfg.level = cc.level >= 0 ? cc.level : 5;
                return cfg;
            case utils::CodecId::Sharding:
                // Recurse into sub-codecs
                return parseCompressorFromCodecs(cc.sub_codecs, dtypeSize);
            default:
                break;
        }
    }
    return cfg;
}

static std::string delimiterFromEncoding(utils::ChunkKeyEncoding enc)
{
    switch (enc) {
        case utils::ChunkKeyEncoding::SlashSeparated: return "/";
        case utils::ChunkKeyEncoding::DotSeparated:   return ".";
        case utils::ChunkKeyEncoding::V3Default:       return "/";
    }
    return ".";
}

}  // anonymous namespace

// ============================================================================
// Zarr::Impl
// ============================================================================

struct Zarr::Impl {
    std::filesystem::path fsPath;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> chunkShape_;
    std::vector<std::size_t> shards_;
    std::size_t chunkSize_ = 0;
    DType dtype_ = DType::UInt8;
    std::size_t dtypeSize_ = 1;
    std::string delimiter_ = ".";
    bool isSharded_ = false;
    CompressorConfig compressor_;

    // utils zarr array for chunk/region I/O
    std::shared_ptr<utils::FilesystemStore> store_;
    std::unique_ptr<utils::ZarrArray> zarrArray_;
    std::string arrayPathInStore_;

    void openFromPath()
    {
        // Open the parent directory as a FilesystemStore, then open the array
        // at the relative path within it.
        auto parentPath = fsPath.parent_path();
        auto arrayName = fsPath.filename().string();

        auto storeResult = utils::FilesystemStore::open(parentPath);
        if (!storeResult) {
            throw std::runtime_error("Failed to open store at " +
                                     parentPath.string() + ": " + storeResult.error());
        }
        store_ = std::shared_ptr<utils::FilesystemStore>(storeResult->release());
        arrayPathInStore_ = arrayName;

        auto arrResult = utils::ZarrArray::open(*store_, arrayPathInStore_);
        if (!arrResult) {
            throw std::runtime_error("Failed to open zarr array at " +
                                     fsPath.string() + ": " + arrResult.error());
        }
        zarrArray_ = std::make_unique<utils::ZarrArray>(std::move(*arrResult));

        // Extract metadata from the opened array
        const auto& meta = zarrArray_->metadata();
        auto shapeSpan = meta.shape();
        shape_.assign(shapeSpan.begin(), shapeSpan.end());

        auto chunksSpan = meta.chunks();
        chunkShape_.assign(chunksSpan.begin(), chunksSpan.end());

        dtype_ = static_cast<DType>(meta.dtype());
        dtypeSize_ = utils::dtype_size(meta.dtype());

        // Chunk size (product)
        chunkSize_ = 1;
        for (auto c : chunkShape_)
            chunkSize_ *= c;

        // Delimiter
        delimiter_ = delimiterFromEncoding(meta.chunk_key_encoding());

        // Compressor
        compressor_ = parseCompressorFromCodecs(
            meta.codecs(), static_cast<int>(dtypeSize_));

        // Shard detection: scan codecs for sharding
        isSharded_ = false;
        shards_ = chunkShape_;  // default: shards == chunks

        auto codecs = meta.codecs();
        for (auto& cc : codecs) {
            if (cc.id == utils::CodecId::Sharding) {
                isSharded_ = true;
                // In zarr v3 sharding: chunk_grid.chunk_shape is the shard shape,
                // and the sharding codec's shard_shape (confusingly named chunk_shape
                // in the spec) is the inner chunk shape.
                shards_ = chunkShape_;  // shard = outer chunk
                if (!cc.shard_shape.empty()) {
                    chunkShape_ = cc.shard_shape;  // inner chunk
                    chunkSize_ = 1;
                    for (auto c : chunkShape_)
                        chunkSize_ *= c;
                }
                break;
            }
        }
    }
};

// ============================================================================
// Zarr public API
// ============================================================================

Zarr Zarr::open(const std::filesystem::path& path)
{
    Zarr z;
    z.impl_ = std::make_unique<Impl>();
    z.impl_->fsPath = path;
    z.impl_->openFromPath();
    return z;
}

Zarr Zarr::create(const std::filesystem::path& path,
                  std::vector<std::size_t> shape,
                  std::vector<std::size_t> chunks,
                  DType dtype,
                  const std::string& compressor,
                  const std::string& delimiter)
{
    namespace fs = std::filesystem;

    // Determine parent and array name
    fs::path parentPath = path.parent_path();
    std::string name = path.filename().string();
    fs::path dsPath = parentPath / name;
    fs::create_directories(dsPath);

    // Map DType to numpy dtype string
    std::string dtypeStr;
    switch (dtype) {
        case DType::UInt8:   dtypeStr = "|u1"; break;
        case DType::UInt16:  dtypeStr = "<u2"; break;
        case DType::UInt32:  dtypeStr = "<u4"; break;
        case DType::UInt64:  dtypeStr = "<u8"; break;
        case DType::Int8:    dtypeStr = "|i1"; break;
        case DType::Int16:   dtypeStr = "<i2"; break;
        case DType::Int32:   dtypeStr = "<i4"; break;
        case DType::Int64:   dtypeStr = "<i8"; break;
        case DType::Float32: dtypeStr = "<f4"; break;
        case DType::Float64: dtypeStr = "<f8"; break;
    }

    // Write .zarray metadata
    nlohmann::json zarray;
    zarray["zarr_format"] = 2;
    zarray["shape"] = shape;
    zarray["chunks"] = chunks;
    zarray["dtype"] = dtypeStr;
    zarray["fill_value"] = 0;
    zarray["order"] = "C";
    zarray["dimension_separator"] = delimiter;

    if (compressor == "blosc") {
        zarray["compressor"] = {
            {"id", "blosc"},
            {"cname", "lz4"},
            {"clevel", 5},
            {"shuffle", 1},
            {"blocksize", 0}
        };
    } else if (compressor == "zstd") {
        zarray["compressor"] = {
            {"id", "zstd"},
            {"level", 3}
        };
    } else if (compressor == "lz4") {
        zarray["compressor"] = {
            {"id", "lz4"},
            {"acceleration", 1}
        };
    } else if (compressor == "gzip" || compressor == "zlib") {
        zarray["compressor"] = {
            {"id", "gzip"},
            {"level", 5}
        };
    } else if (compressor.empty() || compressor == "none") {
        zarray["compressor"] = nullptr;
    } else {
        zarray["compressor"] = {{"id", compressor}};
    }
    zarray["filters"] = nullptr;

    {
        std::ofstream f(dsPath / ".zarray");
        f << zarray.dump(2) << '\n';
    }

    // Also write .zgroup in parent if it doesn't exist
    auto zgroupPath = parentPath / ".zgroup";
    if (!fs::exists(zgroupPath)) {
        std::ofstream g(zgroupPath);
        g << R"({"zarr_format": 2})" << '\n';
    }

    return Zarr::open(dsPath);
}

Zarr::~Zarr() = default;
Zarr::Zarr(Zarr&&) noexcept = default;
Zarr& Zarr::operator=(Zarr&&) noexcept = default;

std::span<const std::size_t> Zarr::shape() const { return impl_->shape_; }
std::span<const std::size_t> Zarr::chunks() const { return impl_->chunkShape_; }
std::span<const std::size_t> Zarr::shards() const { return impl_->shards_; }
DType Zarr::dtype() const { return impl_->dtype_; }
std::size_t Zarr::itemsize() const { return impl_->dtypeSize_; }
std::size_t Zarr::chunkSize() const { return impl_->chunkSize_; }
const std::filesystem::path& Zarr::path() const { return impl_->fsPath; }
const std::string& Zarr::delimiter() const { return impl_->delimiter_; }
bool Zarr::isSharded() const { return impl_->isSharded_; }

bool Zarr::readChunk(std::size_t iz, std::size_t iy, std::size_t ix, void* output) const
{
    std::vector<std::size_t> indices = {iz, iy, ix};
    auto result = impl_->zarrArray_->read_chunk(indices);
    if (!result) return false;

    const auto& bytes = *result;
    const std::size_t expectedBytes = impl_->chunkSize_ * impl_->dtypeSize_;
    if (bytes.size() < expectedBytes) return false;

    std::memcpy(output, bytes.data(), expectedBytes);
    return true;
}

bool Zarr::writeChunk(std::size_t iz, std::size_t iy, std::size_t ix,
                      const void* input, std::size_t nbytes)
{
    std::vector<std::size_t> indices = {iz, iy, ix};
    auto data = std::span<const uint8_t>(
        static_cast<const uint8_t*>(input), nbytes);
    auto result = impl_->zarrArray_->write_chunk(indices, data);
    return result.has_value();
}

bool Zarr::chunkExists(std::size_t iz, std::size_t iy, std::size_t ix) const
{
    // Build chunk path: <basepath>/<iz><delim><iy><delim><ix>
    auto p = impl_->fsPath /
        (std::to_string(iz) + impl_->delimiter_ +
         std::to_string(iy) + impl_->delimiter_ +
         std::to_string(ix));
    return std::filesystem::exists(p);
}

bool Zarr::readRegion(std::span<const std::size_t> offset,
                      std::span<const std::size_t> regionShape,
                      void* output) const
{
    auto result = impl_->zarrArray_->read_region(offset, regionShape, output);
    return result.has_value();
}

bool Zarr::writeRegion(std::span<const std::size_t> offset,
                       std::span<const std::size_t> regionShape,
                       const void* data)
{
    auto result = impl_->zarrArray_->write_region(offset, regionShape, data);
    return result.has_value();
}

void Zarr::decompress(const std::vector<char>& compressed,
                      void* output, std::size_t nElements) const
{
    const std::size_t outBytes = nElements * impl_->dtypeSize_;

    switch (impl_->compressor_.id) {
        case CompressorId::None:
            std::memcpy(output, compressed.data(), outBytes);
            break;

        case CompressorId::Blosc: {
            int ret = blosc_decompress(compressed.data(), output,
                                        outBytes);
            if (ret < 0) {
                throw std::runtime_error("blosc_decompress failed with code " +
                                          std::to_string(ret));
            }
            break;
        }

        case CompressorId::Zstd: {
            std::size_t ret = ZSTD_decompress(output, outBytes,
                                               compressed.data(), compressed.size());
            if (ZSTD_isError(ret)) {
                throw std::runtime_error(
                    std::string("ZSTD_decompress failed: ") + ZSTD_getErrorName(ret));
            }
            break;
        }

        case CompressorId::Lz4: {
            // numcodecs lz4 format: 4-byte LE original size prefix, then lz4 block
            if (compressed.size() < 4) {
                throw std::runtime_error("LZ4 compressed data too short");
            }
            uint32_t origSize;
            std::memcpy(&origSize, compressed.data(), 4);
            int ret = LZ4_decompress_safe(
                compressed.data() + 4,
                static_cast<char*>(output),
                static_cast<int>(compressed.size() - 4),
                static_cast<int>(origSize));
            if (ret < 0) {
                throw std::runtime_error("LZ4_decompress_safe failed");
            }
            break;
        }

        case CompressorId::Gzip: {
            z_stream strm{};
            if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) {
                throw std::runtime_error("inflateInit2 failed");
            }
            strm.avail_in = static_cast<uInt>(compressed.size());
            strm.next_in = reinterpret_cast<Bytef*>(
                const_cast<char*>(compressed.data()));
            strm.avail_out = static_cast<uInt>(outBytes);
            strm.next_out = static_cast<Bytef*>(output);
            int ret = inflate(&strm, Z_FINISH);
            inflateEnd(&strm);
            if (ret != Z_STREAM_END && ret != Z_OK) {
                throw std::runtime_error("gzip inflate failed with code " +
                                          std::to_string(ret));
            }
            break;
        }
    }
}

nlohmann::json Zarr::readAttributes(const std::filesystem::path& groupPath)
{
    auto attrsPath = groupPath / ".zattrs";
    if (!std::filesystem::exists(attrsPath)) {
        return nlohmann::json::object();
    }
    std::ifstream f(attrsPath);
    return nlohmann::json::parse(f);
}

void Zarr::writeAttributes(const std::filesystem::path& groupPath,
                            const nlohmann::json& attrs)
{
    auto attrsPath = groupPath / ".zattrs";
    std::filesystem::create_directories(groupPath);
    std::ofstream f(attrsPath);
    f << attrs.dump(2) << '\n';
}

// ============================================================================
// OMEZarr
// ============================================================================

OMEZarr OMEZarr::open(const std::filesystem::path& path)
{
    OMEZarr oz;
    oz.path_ = path;

    std::vector<std::string> levelNames;

    for (auto& entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_directory()) continue;
        auto p = entry.path();
        if (std::filesystem::exists(p / ".zarray") ||
            std::filesystem::exists(p / "zarr.json")) {
            levelNames.push_back(p.filename().string());
        }
    }

    // Sort numerically
    std::sort(levelNames.begin(), levelNames.end(),
              [](const std::string& a, const std::string& b) {
                  try {
                      return std::stoi(a) < std::stoi(b);
                  } catch (...) {
                      return a < b;
                  }
              });

    oz.levels_.reserve(levelNames.size());
    for (auto& name : levelNames) {
        oz.levels_.push_back(Zarr::open(path / name));
    }
    return oz;
}

OMEZarr::~OMEZarr() = default;
OMEZarr::OMEZarr(OMEZarr&&) noexcept = default;
OMEZarr& OMEZarr::operator=(OMEZarr&&) noexcept = default;

std::size_t OMEZarr::numLevels() const { return levels_.size(); }

Zarr& OMEZarr::level(int i) { return levels_.at(static_cast<std::size_t>(i)); }
const Zarr& OMEZarr::level(int i) const { return levels_.at(static_cast<std::size_t>(i)); }

const std::filesystem::path& OMEZarr::path() const { return path_; }

nlohmann::json OMEZarr::attributes() const
{
    return Zarr::readAttributes(path_);
}

void OMEZarr::writeAttributes(const nlohmann::json& attrs)
{
    Zarr::writeAttributes(path_, attrs);
}

void OMEZarr::createPyramidLevels(int nLevels, std::size_t chunkH,
                                   std::size_t chunkW, bool isU16)
{
    DType dt = isU16 ? DType::UInt16 : DType::UInt8;

    // Level 0 must already exist
    if (levels_.empty()) {
        throw std::runtime_error("OMEZarr::createPyramidLevels: no level-0 found");
    }

    auto prevShape = levels_[0].shape();
    std::vector<std::size_t> prev(prevShape.begin(), prevShape.end());

    for (int lvl = 1; lvl <= nLevels; lvl++) {
        std::vector<std::size_t> ds = {prev[0], (prev[1] + 1) / 2, (prev[2] + 1) / 2};
        std::size_t chZ = std::min(ds[0], prev[0]);
        std::vector<std::size_t> dc = {chZ, std::min(chunkH, ds[1]), std::min(chunkW, ds[2])};

        auto levelPath = path_ / std::to_string(lvl);
        auto z = Zarr::create(levelPath, ds, dc, dt, "blosc", ".");
        levels_.push_back(std::move(z));

        prev = ds;
    }
}

template<typename T>
void OMEZarr::buildPyramidLevel(int level, std::size_t /*chunkH*/, std::size_t /*chunkW*/,
                                 int numParts, int partId)
{
    auto& src = levels_.at(static_cast<std::size_t>(level - 1));
    auto& dst = levels_.at(static_cast<std::size_t>(level));
    auto ss = src.shape();
    auto sc = src.chunks();
    auto dc = dst.chunks();
    auto ds = dst.shape();

    // Source chunk grid
    std::size_t srcChunksY = (ss[1] + sc[1] - 1) / sc[1];
    std::size_t srcChunksX = (ss[2] + sc[2] - 1) / sc[2];

    // Dest chunk grid
    std::size_t dstChunksY = (ds[1] + dc[1] - 1) / dc[1];
    std::size_t dstChunksX = (ds[2] + dc[2] - 1) / dc[2];

    // Contiguous block assignment
    std::size_t rowsPerPart = (dstChunksY + static_cast<std::size_t>(numParts) - 1) /
                              static_cast<std::size_t>(numParts);
    std::size_t rowStart = static_cast<std::size_t>(partId) * rowsPerPart;
    std::size_t rowEnd = std::min(rowStart + rowsPerPart, dstChunksY);
    std::size_t myTiles = (rowEnd - rowStart) * dstChunksX;
    std::atomic<std::size_t> done{0};

    std::size_t srcElems = sc[0] * sc[1] * sc[2];
    std::size_t dstElems = dc[0] * dc[1] * dc[2];
    std::size_t dstNbytes = dstElems * dst.itemsize();

    #pragma omp parallel for schedule(dynamic, 2)
    for (long long ti = 0; ti < static_cast<long long>(myTiles); ti++) {
        std::size_t dcy = rowStart + static_cast<std::size_t>(ti) / dstChunksX;
        std::size_t dcx = static_cast<std::size_t>(ti) % dstChunksX;

        std::vector<T> dstBuf(dstElems, T(0));

        // Each dest chunk assembles from a 2x2 grid of source chunks
        for (int sy = 0; sy < 2; sy++) {
            for (int sx = 0; sx < 2; sx++) {
                std::size_t scy = dcy * 2 + static_cast<std::size_t>(sy);
                std::size_t scx = dcx * 2 + static_cast<std::size_t>(sx);
                if (scy >= srcChunksY || scx >= srcChunksX) continue;

                std::vector<T> srcBuf(srcElems, T(0));
                if (src.chunkExists(0, scy, scx))
                    src.readChunk(0, scy, scx, srcBuf.data());
                else
                    continue;

                // Actual valid extent in this source chunk
                std::size_t saZ = std::min(sc[0], ss[0]);
                std::size_t saY = std::min(sc[1], ss[1] - scy * sc[1]);
                std::size_t saX = std::min(sc[2], ss[2] - scx * sc[2]);

                // Offset within dest chunk
                std::size_t halfY = sc[1] / 2, halfX = sc[2] / 2;
                std::size_t offY = static_cast<std::size_t>(sy) * halfY;
                std::size_t offX = static_cast<std::size_t>(sx) * halfX;

                downsampleTileIntoPreserveZ(
                    srcBuf.data(), sc[0], sc[1], sc[2],
                    dstBuf.data(), dc[0], dc[1], dc[2],
                    saZ, saY, saX, offY, offX);
            }
        }

        dst.writeChunk(0, dcy, dcx, dstBuf.data(), dstNbytes);

        std::size_t d = ++done;
        #pragma omp critical(pp)
        { std::cout << "\r[pyramid L" << level << "] " << d << "/" << myTiles
                    << " (" << static_cast<int>(100.0 * static_cast<double>(d) /
                                                static_cast<double>(myTiles)) << "%)"
                    << std::flush; }
    }
    if (myTiles > 0) std::cout << std::endl;
}

template void OMEZarr::buildPyramidLevel<uint8_t>(int, std::size_t, std::size_t, int, int);
template void OMEZarr::buildPyramidLevel<uint16_t>(int, std::size_t, std::size_t, int, int);

// ============================================================================
// Downsample helpers
// ============================================================================

template <typename T>
void downsampleChunk(const T* src, std::size_t srcZ, std::size_t srcY, std::size_t srcX,
                     T* dst, std::size_t dstZ, std::size_t dstY, std::size_t dstX,
                     std::size_t srcActualZ, std::size_t srcActualY, std::size_t srcActualX)
{
    for (std::size_t zz = 0; zz < dstZ; zz++)
        for (std::size_t yy = 0; yy < dstY; yy++)
            for (std::size_t xx = 0; xx < dstX; xx++) {
                uint32_t sum = 0; int cnt = 0;
                for (int d0 = 0; d0 < 2 && 2*zz+static_cast<std::size_t>(d0) < srcActualZ; d0++)
                    for (int d1 = 0; d1 < 2 && 2*yy+static_cast<std::size_t>(d1) < srcActualY; d1++)
                        for (int d2 = 0; d2 < 2 && 2*xx+static_cast<std::size_t>(d2) < srcActualX; d2++) {
                            sum += src[(2*zz+d0)*srcY*srcX + (2*yy+d1)*srcX + (2*xx+d2)];
                            cnt++;
                        }
                dst[zz*dstY*dstX + yy*dstX + xx] = T((sum + cnt/2) / std::max(1, cnt));
            }
}

template void downsampleChunk<uint8_t>(const uint8_t*, std::size_t, std::size_t, std::size_t,
    uint8_t*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
template void downsampleChunk<uint16_t>(const uint16_t*, std::size_t, std::size_t, std::size_t,
    uint16_t*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

template <typename T>
void downsampleTileInto(const T* src, std::size_t srcZ, std::size_t srcY, std::size_t srcX,
                        T* dst, std::size_t dstZ, std::size_t dstY, std::size_t dstX,
                        std::size_t srcActualZ, std::size_t srcActualY, std::size_t srcActualX,
                        std::size_t dstOffY, std::size_t dstOffX)
{
    std::size_t halfZ = (srcActualZ + 1) / 2;
    std::size_t halfY = (srcActualY + 1) / 2;
    std::size_t halfX = (srcActualX + 1) / 2;
    for (std::size_t zz = 0; zz < halfZ && zz < dstZ; zz++)
        for (std::size_t yy = 0; yy < halfY && (dstOffY + yy) < dstY; yy++)
            for (std::size_t xx = 0; xx < halfX && (dstOffX + xx) < dstX; xx++) {
                uint32_t sum = 0; int cnt = 0;
                for (int d0 = 0; d0 < 2 && 2*zz+static_cast<std::size_t>(d0) < srcActualZ; d0++)
                    for (int d1 = 0; d1 < 2 && 2*yy+static_cast<std::size_t>(d1) < srcActualY; d1++)
                        for (int d2 = 0; d2 < 2 && 2*xx+static_cast<std::size_t>(d2) < srcActualX; d2++) {
                            sum += src[(2*zz+d0)*srcY*srcX + (2*yy+d1)*srcX + (2*xx+d2)];
                            cnt++;
                        }
                dst[zz*dstY*dstX + (dstOffY+yy)*dstX + (dstOffX+xx)] = T((sum + cnt/2) / std::max(1, cnt));
            }
}

template void downsampleTileInto<uint8_t>(const uint8_t*, std::size_t, std::size_t, std::size_t,
    uint8_t*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
template void downsampleTileInto<uint16_t>(const uint16_t*, std::size_t, std::size_t, std::size_t,
    uint16_t*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

template <typename T>
void downsampleTileIntoPreserveZ(const T* src, std::size_t srcZ, std::size_t srcY, std::size_t srcX,
                                T* dst, std::size_t dstZ, std::size_t dstY, std::size_t dstX,
                                std::size_t srcActualZ, std::size_t srcActualY, std::size_t srcActualX,
                                std::size_t dstOffY, std::size_t dstOffX)
{
    std::size_t halfY = (srcActualY + 1) / 2;
    std::size_t halfX = (srcActualX + 1) / 2;
    for (std::size_t zz = 0; zz < srcActualZ && zz < dstZ; zz++)
        for (std::size_t yy = 0; yy < halfY && (dstOffY + yy) < dstY; yy++)
            for (std::size_t xx = 0; xx < halfX && (dstOffX + xx) < dstX; xx++) {
                uint32_t sum = 0; int cnt = 0;
                for (int d1 = 0; d1 < 2 && 2 * yy + static_cast<std::size_t>(d1) < srcActualY; d1++)
                    for (int d2 = 0; d2 < 2 && 2 * xx + static_cast<std::size_t>(d2) < srcActualX; d2++) {
                        sum += src[zz * srcY * srcX + (2 * yy + d1) * srcX + (2 * xx + d2)];
                        cnt++;
                    }
                dst[zz * dstY * dstX + (dstOffY + yy) * dstX + (dstOffX + xx)] = T((sum + cnt / 2) / std::max(1, cnt));
            }
}

template void downsampleTileIntoPreserveZ<uint8_t>(const uint8_t*, std::size_t, std::size_t, std::size_t,
    uint8_t*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
template void downsampleTileIntoPreserveZ<uint16_t>(const uint16_t*, std::size_t, std::size_t, std::size_t,
    uint16_t*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

}  // namespace vc
