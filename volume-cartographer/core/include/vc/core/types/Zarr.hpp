#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "vc/core/types/Tensor.hpp"

namespace zarr {

// ── Enums ──────────────────────────────────────────────────────────────────

enum class Dtype : uint8_t {
    uint8, uint16, float32,             // original three
    float64, int8, int16, int32, int64, // extended
    uint32, uint64,
};

enum class ZarrVersion : uint8_t { v2, v3 };
enum class Order : uint8_t { C, F };
enum class OpenMode : uint8_t { Read, Write, ReadWrite, Append };

enum class CodecId : uint8_t {
    Blosc,
    Gzip,
    Zstd,
    Lz4,
    Bz2,
    Lzma,
    Crc32c,
    Bytes,
    Transpose,
    Sharding,
    H264,
    H265,
    AV1,
};

enum class ChunkKeyEncoding : uint8_t {
    DotSeparated,   // "0.0"        (v2 default)
    SlashSeparated, // "0/0"        (v2 dimension_separator="/")
    V3Default,      // "c/0/0"      (v3 default)
};

// ── Common type aliases ────────────────────────────────────────────────────

using Shape = std::vector<size_t>;
using ChunkId = std::vector<size_t>;
using json = nlohmann::json;

// ── Dtype helpers ──────────────────────────────────────────────────────────

inline size_t dtypeSize(Dtype d) {
    switch (d) {
        case Dtype::uint8:   return 1;
        case Dtype::int8:    return 1;
        case Dtype::uint16:  return 2;
        case Dtype::int16:   return 2;
        case Dtype::float32: return 4;
        case Dtype::int32:   return 4;
        case Dtype::uint32:  return 4;
        case Dtype::float64: return 8;
        case Dtype::int64:   return 8;
        case Dtype::uint64:  return 8;
    }
    return 1;
}

inline std::string dtypeToString(Dtype d) {
    switch (d) {
        case Dtype::uint8:   return "|u1";
        case Dtype::uint16:  return "<u2";
        case Dtype::float32: return "<f4";
        case Dtype::float64: return "<f8";
        case Dtype::int8:    return "|i1";
        case Dtype::int16:   return "<i2";
        case Dtype::int32:   return "<i4";
        case Dtype::int64:   return "<i8";
        case Dtype::uint32:  return "<u4";
        case Dtype::uint64:  return "<u8";
    }
    return "|u1";
}

inline Dtype dtypeFromString(const std::string& s) {
    if (s == "|u1" || s == "<u1" || s == "uint8")   return Dtype::uint8;
    if (s == "<u2" || s == ">u2" || s == "uint16")   return Dtype::uint16;
    if (s == "<f4" || s == ">f4" || s == "float32")  return Dtype::float32;
    if (s == "<f8" || s == ">f8" || s == "float64")  return Dtype::float64;
    if (s == "|i1" || s == "<i1" || s == "int8")     return Dtype::int8;
    if (s == "<i2" || s == ">i2" || s == "int16")    return Dtype::int16;
    if (s == "<i4" || s == ">i4" || s == "int32")    return Dtype::int32;
    if (s == "<i8" || s == ">i8" || s == "int64")    return Dtype::int64;
    if (s == "<u4" || s == ">u4" || s == "uint32")   return Dtype::uint32;
    if (s == "<u8" || s == ">u8" || s == "uint64")   return Dtype::uint64;
    throw std::runtime_error("unsupported zarr dtype: " + s);
}

// v3 dtype string names
inline std::string dtypeToV3String(Dtype d) {
    switch (d) {
        case Dtype::uint8:   return "uint8";
        case Dtype::uint16:  return "uint16";
        case Dtype::float32: return "float32";
        case Dtype::float64: return "float64";
        case Dtype::int8:    return "int8";
        case Dtype::int16:   return "int16";
        case Dtype::int32:   return "int32";
        case Dtype::int64:   return "int64";
        case Dtype::uint32:  return "uint32";
        case Dtype::uint64:  return "uint64";
    }
    return "uint8";
}

inline Dtype dtypeFromV3String(const std::string& s) {
    if (s == "uint8")   return Dtype::uint8;
    if (s == "uint16")  return Dtype::uint16;
    if (s == "float32") return Dtype::float32;
    if (s == "float64") return Dtype::float64;
    if (s == "int8")    return Dtype::int8;
    if (s == "int16")   return Dtype::int16;
    if (s == "int32")   return Dtype::int32;
    if (s == "int64")   return Dtype::int64;
    if (s == "uint32")  return Dtype::uint32;
    if (s == "uint64")  return Dtype::uint64;
    throw std::runtime_error("unsupported zarr v3 dtype: " + s);
}

// ── CodecConfig ────────────────────────────────────────────────────────────

struct CodecConfig {
    CodecId id{};

    // Blosc parameters
    std::string blosc_cname = "lz4";
    int blosc_clevel = 5;
    int blosc_shuffle = 1; // 0=none, 1=byte, 2=bit
    int blosc_typesize = 0;
    int blosc_blocksize = 0;

    // Gzip / Bz2 / Zstd / Lz4 / Lzma
    int level = -1;

    // Lz4 acceleration
    int acceleration = 1;

    // Bytes codec endian
    std::string endian = "little";

    // Transpose order
    std::vector<std::size_t> transpose_order;

    // Sharding
    std::vector<std::size_t> shard_shape;
    std::vector<CodecConfig> sub_codecs;
    std::vector<CodecConfig> index_codecs;

    // Video codec parameters (H264, H265, AV1)
    int video_crf = 23;
    std::string video_preset = "medium";
    int video_bit_depth = 8;

    // Factory methods
    static auto blosc(std::string cname = "lz4", int clevel = 5,
                      int shuffle = 1, int typesize = 0,
                      int blocksize = 0) -> CodecConfig;
    static auto gzip(int lvl = 5) -> CodecConfig;
    static auto zstd(int lvl = 3) -> CodecConfig;
    static auto lz4(int accel = 1) -> CodecConfig;
    static auto bz2(int lvl = 9) -> CodecConfig;
    static auto lzma(int preset = 6) -> CodecConfig;
    static auto crc32c() -> CodecConfig;
    static auto bytes(std::string endian = "little") -> CodecConfig;
    static auto transpose(std::vector<std::size_t> order) -> CodecConfig;
    static auto sharding(std::vector<std::size_t> shape,
                         std::vector<CodecConfig> codecs = {},
                         std::vector<CodecConfig> idx_codecs = {}) -> CodecConfig;
    static auto h264(int crf = 23, std::string preset = "medium",
                     int bit_depth = 8) -> CodecConfig;
    static auto h265(int crf = 23, std::string preset = "medium",
                     int bit_depth = 8) -> CodecConfig;
    static auto av1(int crf = 30, std::string preset = "6",
                    int bit_depth = 8) -> CodecConfig;
};

// ── CompressorConfig (backward compat alias for v2-style compressor) ──────

struct CompressorConfig {
    std::string id;  // "blosc", "zlib", "gzip", "zstd", or "" (no compression)

    // blosc options
    std::string cname = "lz4";
    int clevel = 5;
    int shuffle = 1;  // 0=none, 1=byte, 2=bit
    int blocksize = 0;

    // zlib/gzip/zstd level
    int level = 1;

    static CompressorConfig none() { return CompressorConfig{""}; }
    static CompressorConfig blosc(const std::string& cname = "lz4", int clevel = 5, int shuffle = 1) {
        CompressorConfig c;
        c.id = "blosc";
        c.cname = cname;
        c.clevel = clevel;
        c.shuffle = shuffle;
        return c;
    }
    static CompressorConfig zstd(int level = 1) {
        CompressorConfig c;
        c.id = "zstd";
        c.level = level;
        return c;
    }
    static CompressorConfig gzip(int level = 1) {
        CompressorConfig c;
        c.id = "gzip";
        c.level = level;
        return c;
    }
    static CompressorConfig zlib(int level = 1) {
        CompressorConfig c;
        c.id = "zlib";
        c.level = level;
        return c;
    }
};

// ── Store ──────────────────────────────────────────────────────────────────

class Store {
public:
    virtual ~Store() = default;

    [[nodiscard]] virtual auto get(std::string_view key) const
        -> std::expected<std::vector<uint8_t>, std::string> = 0;
    [[nodiscard]] virtual auto set(std::string_view key,
                                   std::span<const uint8_t> value)
        -> std::expected<void, std::string> = 0;
    [[nodiscard]] virtual auto erase(std::string_view key)
        -> std::expected<void, std::string> = 0;
    [[nodiscard]] virtual auto exists(std::string_view key) const -> bool = 0;
    [[nodiscard]] virtual auto list_prefix(std::string_view prefix) const
        -> std::vector<std::string> = 0;
    [[nodiscard]] virtual auto list_dir(std::string_view prefix) const
        -> std::pair<std::vector<std::string>, std::vector<std::string>> = 0;

    // Partial read for sharding support
    [[nodiscard]] virtual auto get_partial(std::string_view key,
                                           std::size_t offset,
                                           std::size_t length) const
        -> std::expected<std::vector<uint8_t>, std::string>;
};

// ── MemoryStore ────────────────────────────────────────────────────────────

class MemoryStore : public Store {
public:
    MemoryStore();
    ~MemoryStore() override;
    MemoryStore(MemoryStore&&) noexcept;
    auto operator=(MemoryStore&&) noexcept -> MemoryStore&;

    [[nodiscard]] auto get(std::string_view key) const
        -> std::expected<std::vector<uint8_t>, std::string> override;
    [[nodiscard]] auto set(std::string_view key,
                           std::span<const uint8_t> value)
        -> std::expected<void, std::string> override;
    [[nodiscard]] auto erase(std::string_view key)
        -> std::expected<void, std::string> override;
    [[nodiscard]] auto exists(std::string_view key) const -> bool override;
    [[nodiscard]] auto list_prefix(std::string_view prefix) const
        -> std::vector<std::string> override;
    [[nodiscard]] auto list_dir(std::string_view prefix) const
        -> std::pair<std::vector<std::string>, std::vector<std::string>> override;
    [[nodiscard]] auto get_partial(std::string_view key, std::size_t offset,
                                   std::size_t length) const
        -> std::expected<std::vector<uint8_t>, std::string> override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── FilesystemStore ────────────────────────────────────────────────────────

class FilesystemStore : public Store {
public:
    ~FilesystemStore() override;
    FilesystemStore(FilesystemStore&&) noexcept;
    auto operator=(FilesystemStore&&) noexcept -> FilesystemStore&;

    [[nodiscard]] static auto open(const std::filesystem::path& root)
        -> std::expected<std::unique_ptr<FilesystemStore>, std::string>;

    [[nodiscard]] auto root() const -> const std::filesystem::path&;

    [[nodiscard]] auto get(std::string_view key) const
        -> std::expected<std::vector<uint8_t>, std::string> override;
    [[nodiscard]] auto set(std::string_view key,
                           std::span<const uint8_t> value)
        -> std::expected<void, std::string> override;
    [[nodiscard]] auto erase(std::string_view key)
        -> std::expected<void, std::string> override;
    [[nodiscard]] auto exists(std::string_view key) const -> bool override;
    [[nodiscard]] auto list_prefix(std::string_view prefix) const
        -> std::vector<std::string> override;
    [[nodiscard]] auto list_dir(std::string_view prefix) const
        -> std::pair<std::vector<std::string>, std::vector<std::string>> override;
    [[nodiscard]] auto get_partial(std::string_view key, std::size_t offset,
                                   std::size_t length) const
        -> std::expected<std::vector<uint8_t>, std::string> override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit FilesystemStore(std::unique_ptr<Impl> impl);
};

// ── ArrayMetadata ──────────────────────────────────────────────────────────

class ArrayMetadata {
public:
    ~ArrayMetadata();
    ArrayMetadata(ArrayMetadata&&) noexcept;
    auto operator=(ArrayMetadata&&) noexcept -> ArrayMetadata&;
    ArrayMetadata(const ArrayMetadata&);
    auto operator=(const ArrayMetadata&) -> ArrayMetadata&;

    [[nodiscard]] auto version() const -> ZarrVersion;
    [[nodiscard]] auto shape() const -> std::span<const std::size_t>;
    [[nodiscard]] auto chunks() const -> std::span<const std::size_t>;
    [[nodiscard]] auto dtype() const -> Dtype;
    [[nodiscard]] auto order() const -> Order;
    [[nodiscard]] auto fill_value() const -> double;
    [[nodiscard]] auto codecs() const -> std::span<const CodecConfig>;
    [[nodiscard]] auto chunk_key_encoding() const -> ChunkKeyEncoding;
    [[nodiscard]] auto dimension_names() const
        -> std::span<const std::string>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class ZarrArray;
    explicit ArrayMetadata(std::unique_ptr<Impl> impl);
};

// ── CreateOptions ──────────────────────────────────────────────────────────

struct CreateOptions {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> chunks;
    Dtype dtype = Dtype::float32;
    Order order = Order::C;
    double fill_value = 0.0;
    ZarrVersion version = ZarrVersion::v3;
    std::optional<std::vector<CodecConfig>> codecs;  // nullopt = defaults
    ChunkKeyEncoding chunk_key_encoding = ChunkKeyEncoding::V3Default;
    std::vector<std::string> dimension_names;
    std::string attributes_json;
};

// ── ZarrArray ──────────────────────────────────────────────────────────────

class ZarrArray {
public:
    ~ZarrArray();
    ZarrArray(ZarrArray&&) noexcept;
    auto operator=(ZarrArray&&) noexcept -> ZarrArray&;

    [[nodiscard]] static auto open(Store& store, std::string_view path = "")
        -> std::expected<ZarrArray, std::string>;
    [[nodiscard]] static auto open(const std::filesystem::path& path)
        -> std::expected<ZarrArray, std::string>;
    [[nodiscard]] static auto create(Store& store, std::string_view path,
                                     const CreateOptions& opts)
        -> std::expected<ZarrArray, std::string>;

    // Chunk I/O
    [[nodiscard]] auto read_chunk(std::span<const std::size_t> indices) const
        -> std::expected<std::vector<uint8_t>, std::string>;
    [[nodiscard]] auto write_chunk(std::span<const std::size_t> indices,
                                   std::span<const uint8_t> data)
        -> std::expected<void, std::string>;

    // Region I/O (raw buffer)
    [[nodiscard]] auto read_region(std::span<const std::size_t> offset,
                                   std::span<const std::size_t> shape,
                                   void* out) const
        -> std::expected<void, std::string>;
    [[nodiscard]] auto write_region(std::span<const std::size_t> offset,
                                    std::span<const std::size_t> shape,
                                    const void* data)
        -> std::expected<void, std::string>;

    // Metadata
    [[nodiscard]] auto metadata() const -> const ArrayMetadata&;
    [[nodiscard]] auto shape() const -> std::span<const std::size_t>;
    [[nodiscard]] auto chunks() const -> std::span<const std::size_t>;
    [[nodiscard]] auto dtype() const -> Dtype;
    [[nodiscard]] auto ndim() const -> std::size_t;
    [[nodiscard]] auto version() const -> ZarrVersion;

    // Computed properties
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto itemsize() const -> std::size_t;
    [[nodiscard]] auto nbytes() const -> std::size_t;
    [[nodiscard]] auto nchunks() const -> std::size_t;
    [[nodiscard]] auto nchunks_initialized() const -> std::size_t;
    [[nodiscard]] auto fill_value() const -> double;
    [[nodiscard]] auto order() const -> Order;
    [[nodiscard]] auto path() const -> std::string;
    [[nodiscard]] auto name() const -> std::string;
    [[nodiscard]] auto read_only() const -> bool;
    [[nodiscard]] auto store_ref() const -> const Store&;
    [[nodiscard]] auto info() const -> std::string;

    // Attributes (bulk)
    [[nodiscard]] auto attributes() const
        -> std::expected<std::string, std::string>;
    [[nodiscard]] auto set_attributes(std::string_view json_str)
        -> std::expected<void, std::string>;

    // Attributes (granular)
    [[nodiscard]] auto get_attribute(std::string_view key) const
        -> std::expected<std::string, std::string>;
    [[nodiscard]] auto set_attribute(std::string_view key,
                                     std::string_view value_json)
        -> std::expected<void, std::string>;
    [[nodiscard]] auto delete_attribute(std::string_view key)
        -> std::expected<void, std::string>;
    [[nodiscard]] auto contains_attribute(std::string_view key) const -> bool;
    [[nodiscard]] auto attribute_keys() const -> std::vector<std::string>;
    [[nodiscard]] auto num_attributes() const -> std::size_t;

    // Resize
    [[nodiscard]] auto resize(std::span<const std::size_t> new_shape)
        -> std::expected<void, std::string>;

    // Append
    [[nodiscard]] auto append(const void* data,
                              std::span<const std::size_t> data_shape,
                              std::size_t axis = 0)
        -> std::expected<void, std::string>;

    // Store ownership (for path-based factories)
    void set_owned_store(std::shared_ptr<Store> store);
    void set_read_only(bool ro);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit ZarrArray(std::unique_ptr<Impl> impl);

    friend auto open_zarr(const std::filesystem::path&, OpenMode)
        -> std::expected<ZarrArray, std::string>;
};

// ── ZarrGroup ──────────────────────────────────────────────────────────────

class ZarrGroup {
public:
    ~ZarrGroup();
    ZarrGroup(ZarrGroup&&) noexcept;
    auto operator=(ZarrGroup&&) noexcept -> ZarrGroup&;

    [[nodiscard]] static auto open(Store& store, std::string_view path = "")
        -> std::expected<ZarrGroup, std::string>;
    [[nodiscard]] static auto open(const std::filesystem::path& path)
        -> std::expected<ZarrGroup, std::string>;
    [[nodiscard]] static auto create(Store& store, std::string_view path = "",
                                     ZarrVersion version = ZarrVersion::v3)
        -> std::expected<ZarrGroup, std::string>;

    // Child access
    [[nodiscard]] auto open_array(std::string_view name) const
        -> std::expected<ZarrArray, std::string>;
    [[nodiscard]] auto create_array(std::string_view name,
                                    const CreateOptions& opts) const
        -> std::expected<ZarrArray, std::string>;
    [[nodiscard]] auto open_group(std::string_view name) const
        -> std::expected<ZarrGroup, std::string>;
    [[nodiscard]] auto create_group(std::string_view name) const
        -> std::expected<ZarrGroup, std::string>;

    // Enumeration
    [[nodiscard]] auto list_arrays() const -> std::vector<std::string>;
    [[nodiscard]] auto list_groups() const -> std::vector<std::string>;
    [[nodiscard]] auto list_children() const
        -> std::pair<std::vector<std::string>, std::vector<std::string>>;

    // Membership
    [[nodiscard]] auto contains_array(std::string_view name) const -> bool;
    [[nodiscard]] auto contains_group(std::string_view name) const -> bool;

    // Navigation properties
    [[nodiscard]] auto path() const -> std::string;
    [[nodiscard]] auto name() const -> std::string;
    [[nodiscard]] auto store_ref() const -> const Store&;
    [[nodiscard]] auto version() const -> ZarrVersion;
    [[nodiscard]] auto nchildren() const -> std::size_t;
    [[nodiscard]] auto read_only() const -> bool;
    [[nodiscard]] auto info() const -> std::string;

    // Attributes (bulk)
    [[nodiscard]] auto attributes() const
        -> std::expected<std::string, std::string>;
    [[nodiscard]] auto set_attributes(std::string_view json_str)
        -> std::expected<void, std::string>;

    // Attributes (granular)
    [[nodiscard]] auto get_attribute(std::string_view key) const
        -> std::expected<std::string, std::string>;
    [[nodiscard]] auto set_attribute(std::string_view key,
                                     std::string_view value_json)
        -> std::expected<void, std::string>;
    [[nodiscard]] auto delete_attribute(std::string_view key)
        -> std::expected<void, std::string>;
    [[nodiscard]] auto contains_attribute(std::string_view key) const -> bool;
    [[nodiscard]] auto attribute_keys() const -> std::vector<std::string>;
    [[nodiscard]] auto num_attributes() const -> std::size_t;

    // Consolidation (v2)
    [[nodiscard]] auto consolidate_metadata()
        -> std::expected<void, std::string>;

    // Store ownership (for path-based factories)
    void set_owned_store(std::shared_ptr<Store> store);
    void set_read_only(bool ro);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit ZarrGroup(std::unique_ptr<Impl> impl);

    friend auto open_zarr_group(const std::filesystem::path&, OpenMode,
                                 std::optional<ZarrVersion>)
        -> std::expected<ZarrGroup, std::string>;
};

// ── Free functions (new API) ───────────────────────────────────────────────

[[nodiscard]] auto detect_zarr_version(const Store& store,
                                       std::string_view path = "")
    -> std::expected<ZarrVersion, std::string>;

[[nodiscard]] auto open_zarr(const std::filesystem::path& path,
                              OpenMode mode = OpenMode::Read)
    -> std::expected<ZarrArray, std::string>;

[[nodiscard]] auto open_zarr_group(
    const std::filesystem::path& path,
    OpenMode mode = OpenMode::Read,
    std::optional<ZarrVersion> version = std::nullopt)
    -> std::expected<ZarrGroup, std::string>;

// ── Zarr — wraps a single zarr array (backward-compat v2 API) ─────────────

class Zarr {
public:
    ~Zarr();

    // Open existing array
    static std::unique_ptr<Zarr> open(const std::filesystem::path& path,
                                       const std::string& dimensionSeparator = ".");

    // Create new array
    static std::unique_ptr<Zarr> create(const std::filesystem::path& path,
                                          Dtype dtype,
                                          const Shape& shape,
                                          const Shape& chunkShape,
                                          const CompressorConfig& compressor = CompressorConfig::blosc(),
                                          double fillValue = 0,
                                          const std::string& dimensionSeparator = ".");

    // Accessors
    const Shape& shape() const { return shape_; }
    const Shape& chunkShape() const { return chunkShape_; }
    size_t chunkSize() const { return chunkSize_; }
    Dtype dtype() const { return dtype_; }
    const std::filesystem::path& path() const { return path_; }
    const std::string& dimensionSeparator() const { return dimSep_; }
    ZarrVersion version() const { return version_; }
    const CompressorConfig& compressor() const { return compressor_; }

    bool isZarr() const { return true; }

    // Chunk I/O
    bool readChunk(const ChunkId& id, void* buffer) const;
    void writeChunk(const ChunkId& id, const void* buffer);
    bool chunkExists(const ChunkId& id) const;

    // Compression helpers (exposed for ChunkCache direct use)
    void decompress(const std::vector<char>& compressed, void* output, size_t maxElements) const;
    std::vector<char> compress(const void* input, size_t numElements) const;

    // Chunk path construction
    std::filesystem::path chunkPath(const ChunkId& id) const;

    // Chunk grid dimensions
    size_t numChunks(size_t dim) const {
        return (shape_[dim] + chunkShape_[dim] - 1) / chunkShape_[dim];
    }

    size_t ndim() const { return shape_.size(); }

    const Shape& defaultChunkShape() const { return chunkShape_; }
    size_t defaultChunkSize() const { return chunkSize_; }
    Dtype getDtype() const { return dtype_; }

private:
    Zarr() = default;

    void parseMetadataV2(const json& meta);
    void writeMetadataV2() const;

    std::filesystem::path path_;
    Shape shape_;
    Shape chunkShape_;
    size_t chunkSize_ = 0;
    Dtype dtype_ = Dtype::uint8;
    CompressorConfig compressor_;
    double fillValue_ = 0;
    std::string dimSep_ = ".";
    ZarrVersion version_ = ZarrVersion::v2;
    std::string order_ = "C";
};

// ── OMEZarr — multi-resolution zarr pyramid ───────────────────────────────

class OMEZarr {
public:
    ~OMEZarr();

    static std::unique_ptr<OMEZarr> open(const std::filesystem::path& root);
    static std::unique_ptr<OMEZarr> create(const std::filesystem::path& root);

    const std::filesystem::path& path() const { return root_; }
    size_t numLevels() const { return levels_.size(); }

    Zarr* level(int l) const;

    const Shape& shape() const;
    Dtype dtype() const;

    const json& attributes() const { return attrs_; }

    void addLevel(std::unique_ptr<Zarr> arr, int level);
    void writeMultiscalesMetadata(const json& extraAttrs = {});
    void keys(std::vector<std::string>& out) const;

private:
    OMEZarr() = default;

    std::filesystem::path root_;
    std::vector<std::unique_ptr<Zarr>> levels_;
    json attrs_;
};

// ── Store-level operations (backward compat) ──────────────────────────────

void createStore(const std::filesystem::path& root);
void createGroup(const std::filesystem::path& root);

std::unique_ptr<Zarr> openArray(const std::filesystem::path& root,
                                  const std::string& name,
                                  const std::string& dimSep = ".");

std::unique_ptr<Zarr> createArray(const std::filesystem::path& root,
                                    const std::string& name,
                                    Dtype dtype,
                                    const Shape& shape,
                                    const Shape& chunkShape,
                                    const CompressorConfig& compressor = CompressorConfig::blosc(),
                                    double fillValue = 0,
                                    const std::string& dimSep = ".");

std::unique_ptr<Zarr> createArray(const std::filesystem::path& root,
                                    const std::string& name,
                                    const std::string& dtype,
                                    const Shape& shape,
                                    const Shape& chunkShape,
                                    const std::string& compressorId,
                                    const json& compressorOpts,
                                    const std::string& dimSep = ".");

std::unique_ptr<Zarr> openDataset(const std::filesystem::path& root,
                                    const std::string& name);

json readAttributes(const std::filesystem::path& path);
void writeAttributes(const std::filesystem::path& path, const json& attrs);
json readArrayMetadata(const std::filesystem::path& datasetPath);

// ── Subarray I/O (backward compat templates) ──────────────────────────────

template<typename T>
void readSubarray(const Zarr& ds, vc::Tensor<T>& out, const Shape& offset);

template<typename T>
void writeSubarray(Zarr& ds, const vc::Tensor<T>& in, const Shape& offset);

template<typename T>
void writeSubarray(Zarr& ds, const vc::TensorAdaptor<T>& in, const Shape& offset);

} // namespace zarr
