#pragma once

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

#include "tensor.hpp"

namespace utils {

// ── Enums ──────────────────────────────────────────────────────────────────

enum class ZarrVersion : uint8_t { V2, V3 };

// Order enum is in tensor.hpp

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

enum class OpenMode : uint8_t { Read, Write, ReadWrite, Append };

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
    [[nodiscard]] auto dtype() const -> DType;
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
    DType dtype = DType::Float32;
    Order order = Order::C;
    double fill_value = 0.0;
    ZarrVersion version = ZarrVersion::V3;
    std::optional<std::vector<CodecConfig>> codecs;  // nullopt = use defaults, empty = no compression
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

    // Region I/O
    [[nodiscard]] auto read_region(std::span<const std::size_t> offset,
                                   std::span<const std::size_t> shape,
                                   void* out) const
        -> std::expected<void, std::string>;
    [[nodiscard]] auto write_region(std::span<const std::size_t> offset,
                                    std::span<const std::size_t> shape,
                                    const void* data)
        -> std::expected<void, std::string>;

    // Tensor integration
    [[nodiscard]] auto read_tensor() const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto read_tensor(std::span<const std::size_t> offset,
                                   std::span<const std::size_t> shape) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto write_tensor(const Tensor& tensor)
        -> std::expected<void, std::string>;
    [[nodiscard]] auto write_tensor(const Tensor& tensor,
                                    std::span<const std::size_t> offset)
        -> std::expected<void, std::string>;

    // Metadata
    [[nodiscard]] auto metadata() const -> const ArrayMetadata&;
    [[nodiscard]] auto shape() const -> std::span<const std::size_t>;
    [[nodiscard]] auto chunks() const -> std::span<const std::size_t>;
    [[nodiscard]] auto dtype() const -> DType;
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
    [[nodiscard]] auto append(const Tensor& data, std::size_t axis = 0)
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
                                     ZarrVersion version = ZarrVersion::V3)
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

// ── Free functions ─────────────────────────────────────────────────────────

[[nodiscard]] auto save_zarr(
    const Tensor& t, const std::filesystem::path& path,
    ZarrVersion v = ZarrVersion::V3,
    std::optional<std::vector<CodecConfig>> codecs = std::nullopt) -> std::expected<void, std::string>;

[[nodiscard]] auto load_zarr(const std::filesystem::path& path)
    -> std::expected<Tensor, std::string>;

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

} // namespace utils

#include <nlohmann/json.hpp>

namespace vc {

// Re-export utils::DType for consumers.
using DType = utils::DType;

// ── Zarr ───────────────────────────────────────────────────────────────────
// Single zarr array (wraps utils::ZarrArray internally).

class Zarr {
public:
    // Open an existing zarr array directory (must contain .zarray or zarr.json)
    static Zarr open(const std::filesystem::path& path);

    // Create a new zarr v2 array
    static Zarr create(const std::filesystem::path& path,
                       std::vector<std::size_t> shape,
                       std::vector<std::size_t> chunks,
                       DType dtype,
                       const std::string& compressor = "blosc",
                       const std::string& delimiter = ".");

    ~Zarr();
    Zarr(Zarr&&) noexcept;
    Zarr& operator=(Zarr&&) noexcept;

    // --- Metadata ---
    [[nodiscard]] std::span<const std::size_t> shape() const;
    [[nodiscard]] std::span<const std::size_t> chunks() const;
    [[nodiscard]] std::span<const std::size_t> shards() const;  // == chunks() if not sharded
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::size_t itemsize() const;
    [[nodiscard]] std::size_t chunkSize() const;       // product of chunk dims
    [[nodiscard]] const std::filesystem::path& path() const;
    [[nodiscard]] const std::string& delimiter() const;
    [[nodiscard]] bool isSharded() const;

    // --- Chunk I/O ---
    bool readChunk(std::size_t iz, std::size_t iy, std::size_t ix, void* output) const;
    bool writeChunk(std::size_t iz, std::size_t iy, std::size_t ix, const void* input, std::size_t nbytes);
    bool chunkExists(std::size_t iz, std::size_t iy, std::size_t ix) const;

    // --- Region I/O ---
    bool readRegion(std::span<const std::size_t> offset, std::span<const std::size_t> shape, void* out) const;
    bool writeRegion(std::span<const std::size_t> offset, std::span<const std::size_t> shape, const void* data);

    // --- Decompression (for TieredChunkCache) ---
    void decompress(const std::vector<char>& compressed, void* output, std::size_t nElements) const;

    // --- Attributes (static helpers) ---
    static nlohmann::json readAttributes(const std::filesystem::path& path);
    static void writeAttributes(const std::filesystem::path& path, const nlohmann::json& attrs);

    struct Impl;
private:
    Zarr() = default;
    std::unique_ptr<Impl> impl_;
};

// ── OMEZarr ────────────────────────────────────────────────────────────────
// Multi-scale OME-Zarr pyramid. Owns N Zarr objects.

class OMEZarr {
public:
    static OMEZarr open(const std::filesystem::path& path);

    ~OMEZarr();
    OMEZarr(OMEZarr&&) noexcept;
    OMEZarr& operator=(OMEZarr&&) noexcept;

    [[nodiscard]] std::size_t numLevels() const;
    [[nodiscard]] Zarr& level(int i);
    [[nodiscard]] const Zarr& level(int i) const;
    [[nodiscard]] const std::filesystem::path& path() const;

    // Pyramid building
    template<typename T>
    void buildPyramidLevel(int level, std::size_t chunkH, std::size_t chunkW,
                           int numParts = 1, int partId = 0);

    void createPyramidLevels(int nLevels, std::size_t chunkH, std::size_t chunkW, bool isU16);

    // OME metadata
    [[nodiscard]] nlohmann::json attributes() const;
    void writeAttributes(const nlohmann::json& attrs);

private:
    OMEZarr() = default;
    std::filesystem::path path_;
    std::vector<Zarr> levels_;
};

// Downsample helpers (used by OMEZarr::buildPyramidLevel and render tools)
template <typename T>
void downsampleChunk(const T* src, std::size_t srcZ, std::size_t srcY, std::size_t srcX,
                     T* dst, std::size_t dstZ, std::size_t dstY, std::size_t dstX,
                     std::size_t srcActualZ, std::size_t srcActualY, std::size_t srcActualX);

template <typename T>
void downsampleTileInto(const T* src, std::size_t srcZ, std::size_t srcY, std::size_t srcX,
                        T* dst, std::size_t dstZ, std::size_t dstY, std::size_t dstX,
                        std::size_t srcActualZ, std::size_t srcActualY, std::size_t srcActualX,
                        std::size_t dstOffY, std::size_t dstOffX);

template <typename T>
void downsampleTileIntoPreserveZ(const T* src, std::size_t srcZ, std::size_t srcY, std::size_t srcX,
                                T* dst, std::size_t dstZ, std::size_t dstY, std::size_t dstX,
                                std::size_t srcActualZ, std::size_t srcActualY, std::size_t srcActualX,
                                std::size_t dstOffY, std::size_t dstOffX);

}  // namespace vc
