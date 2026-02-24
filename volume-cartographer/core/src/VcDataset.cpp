#include "vc/core/types/VcDataset.hpp"

#include <utils/zarr.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <blosc.h>
#include <zstd.h>
#include <lz4.h>
#include <zlib.h>

namespace vc {

// ============================================================================
// Compressor configuration (parsed from .zarray JSON)
// ============================================================================

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

static CompressorConfig parseCompressor(const nlohmann::json& zarray, int dtypeSize)
{
    CompressorConfig cfg;
    cfg.blosc_typesize = dtypeSize;

    if (!zarray.contains("compressor") || zarray["compressor"].is_null()) {
        cfg.id = CompressorId::None;
        return cfg;
    }

    const auto& comp = zarray["compressor"];
    std::string id = comp.value("id", "");

    if (id == "blosc") {
        cfg.id = CompressorId::Blosc;
        cfg.blosc_cname = comp.value("cname", "lz4");
        cfg.blosc_clevel = comp.value("clevel", 5);
        cfg.blosc_shuffle = comp.value("shuffle", 1);
        if (comp.contains("blocksize"))
            cfg.blosc_blocksize = comp["blocksize"].get<int>();
        if (comp.contains("typesize"))
            cfg.blosc_typesize = comp["typesize"].get<int>();
        else
            cfg.blosc_typesize = dtypeSize;
    } else if (id == "zstd") {
        cfg.id = CompressorId::Zstd;
        cfg.level = comp.value("level", 3);
    } else if (id == "lz4") {
        cfg.id = CompressorId::Lz4;
        cfg.level = comp.value("acceleration", 1);
    } else if (id == "gzip" || id == "zlib") {
        cfg.id = CompressorId::Gzip;
        cfg.level = comp.value("level", 5);
    } else {
        throw std::runtime_error("Unsupported zarr compressor: " + id);
    }

    return cfg;
}

// ============================================================================
// VcDataset::Impl
// ============================================================================

struct VcDataset::Impl {
    std::filesystem::path fsPath;
    std::vector<size_t> shape_;
    std::vector<size_t> chunkShape_;
    size_t chunkSize_ = 0;
    VcDtype dtype_ = VcDtype::uint8;
    size_t dtypeSize_ = 1;
    std::string delimiter_ = ".";
    CompressorConfig compressor_;

    // utils zarr array for read_region / write_region / write_chunk
    std::shared_ptr<utils::FilesystemStore> store_;
    std::unique_ptr<utils::ZarrArray> zarrArray_;
    std::string arrayPathInStore_;  // relative path within store (e.g. "" or "0")

    void parseZarray(const std::filesystem::path& path)
    {
        auto zarrayPath = path / ".zarray";
        if (!std::filesystem::exists(zarrayPath)) {
            throw std::runtime_error("Missing .zarray in " + path.string());
        }

        std::ifstream f(zarrayPath);
        nlohmann::json zarray = nlohmann::json::parse(f);

        // Shape
        auto shapeJson = zarray["shape"];
        shape_.clear();
        for (auto& v : shapeJson)
            shape_.push_back(v.get<size_t>());

        // Chunks
        auto chunksJson = zarray["chunks"];
        chunkShape_.clear();
        for (auto& v : chunksJson)
            chunkShape_.push_back(v.get<size_t>());

        // Chunk size (product)
        chunkSize_ = 1;
        for (auto c : chunkShape_)
            chunkSize_ *= c;

        // Dtype
        std::string dtypeStr = zarray["dtype"].get<std::string>();
        if (dtypeStr == "|u1" || dtypeStr == "<u1" || dtypeStr == "uint8") {
            dtype_ = VcDtype::uint8;
            dtypeSize_ = 1;
        } else if (dtypeStr == "<u2" || dtypeStr == "|u2" || dtypeStr == "uint16") {
            dtype_ = VcDtype::uint16;
            dtypeSize_ = 2;
        } else {
            throw std::runtime_error("Unsupported zarr dtype: " + dtypeStr);
        }

        // Dimension separator
        if (zarray.contains("dimension_separator")) {
            delimiter_ = zarray["dimension_separator"].get<std::string>();
        }

        // Compressor
        compressor_ = parseCompressor(zarray, static_cast<int>(dtypeSize_));
    }

    void openZarrArray()
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
    }
};

// ============================================================================
// VcDataset public API
// ============================================================================

VcDataset::VcDataset(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>())
{
    impl_->fsPath = path;
    impl_->parseZarray(path);
    impl_->openZarrArray();
}

VcDataset::~VcDataset() = default;
VcDataset::VcDataset(VcDataset&&) noexcept = default;
VcDataset& VcDataset::operator=(VcDataset&&) noexcept = default;

const std::vector<size_t>& VcDataset::shape() const { return impl_->shape_; }
const std::vector<size_t>& VcDataset::defaultChunkShape() const { return impl_->chunkShape_; }
size_t VcDataset::defaultChunkSize() const { return impl_->chunkSize_; }
VcDtype VcDataset::getDtype() const { return impl_->dtype_; }
size_t VcDataset::dtypeSize() const { return impl_->dtypeSize_; }
const std::filesystem::path& VcDataset::path() const { return impl_->fsPath; }
const std::string& VcDataset::delimiter() const { return impl_->delimiter_; }

void VcDataset::decompress(const std::vector<char>& compressed,
                            void* output, size_t nElements) const
{
    const size_t outBytes = nElements * impl_->dtypeSize_;

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
            size_t ret = ZSTD_decompress(output, outBytes,
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

bool VcDataset::chunkExists(size_t iz, size_t iy, size_t ix) const
{
    // Build chunk path: <basepath>/<iz><delim><iy><delim><ix>
    auto p = impl_->fsPath /
        (std::to_string(iz) + impl_->delimiter_ +
         std::to_string(iy) + impl_->delimiter_ +
         std::to_string(ix));
    return std::filesystem::exists(p);
}

bool VcDataset::readChunk(size_t iz, size_t iy, size_t ix, void* output) const
{
    std::vector<size_t> indices = {iz, iy, ix};
    auto result = impl_->zarrArray_->read_chunk(indices);
    if (!result) return false;

    const auto& bytes = *result;
    const size_t expectedBytes = impl_->chunkSize_ * impl_->dtypeSize_;
    if (bytes.size() < expectedBytes) return false;

    std::memcpy(output, bytes.data(), expectedBytes);
    return true;
}

bool VcDataset::writeChunk(size_t iz, size_t iy, size_t ix,
                            const void* input, size_t nbytes)
{
    std::vector<size_t> indices = {iz, iy, ix};
    auto data = std::span<const uint8_t>(
        static_cast<const uint8_t*>(input), nbytes);
    auto result = impl_->zarrArray_->write_chunk(indices, data);
    return result.has_value();
}

bool VcDataset::readRegion(const std::vector<size_t>& offset,
                            const std::vector<size_t>& regionShape,
                            void* output) const
{
    auto result = impl_->zarrArray_->read_region(offset, regionShape, output);
    return result.has_value();
}

bool VcDataset::writeRegion(const std::vector<size_t>& offset,
                             const std::vector<size_t>& regionShape,
                             const void* data)
{
    auto result = impl_->zarrArray_->write_region(offset, regionShape, data);
    return result.has_value();
}

// ============================================================================
// Factory functions
// ============================================================================

std::vector<std::unique_ptr<VcDataset>> openZarrLevels(
    const std::filesystem::path& zarrRoot)
{
    std::vector<std::string> levelNames;

    for (auto& entry : std::filesystem::directory_iterator(zarrRoot)) {
        if (!entry.is_directory()) continue;
        auto p = entry.path();
        if (std::filesystem::exists(p / ".zarray")) {
            levelNames.push_back(p.filename().string());
        }
    }

    // Sort numerically
    std::sort(levelNames.begin(), levelNames.end(),
              [](const std::string& a, const std::string& b) {
                  // Try numeric comparison first
                  try {
                      return std::stoi(a) < std::stoi(b);
                  } catch (...) {
                      return a < b;
                  }
              });

    std::vector<std::unique_ptr<VcDataset>> result;
    result.reserve(levelNames.size());
    for (auto& name : levelNames) {
        result.push_back(std::make_unique<VcDataset>(zarrRoot / name));
    }
    return result;
}

nlohmann::json readZarrAttributes(const std::filesystem::path& groupPath)
{
    auto attrsPath = groupPath / ".zattrs";
    if (!std::filesystem::exists(attrsPath)) {
        return nlohmann::json::object();
    }
    std::ifstream f(attrsPath);
    return nlohmann::json::parse(f);
}

void writeZarrAttributes(const std::filesystem::path& groupPath,
                          const nlohmann::json& attrs)
{
    auto attrsPath = groupPath / ".zattrs";
    std::filesystem::create_directories(groupPath);
    std::ofstream f(attrsPath);
    f << attrs.dump(2) << '\n';
}

std::unique_ptr<VcDataset> createZarrDataset(
    const std::filesystem::path& parentPath,
    const std::string& name,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& chunks,
    VcDtype dtype,
    const std::string& compressor,
    const std::string& dimensionSeparator)
{
    namespace fs = std::filesystem;

    // Create the directory structure
    fs::path dsPath = parentPath / name;
    fs::create_directories(dsPath);

    // Write .zarray metadata
    nlohmann::json zarray;
    zarray["zarr_format"] = 2;
    zarray["shape"] = shape;
    zarray["chunks"] = chunks;
    zarray["dtype"] = (dtype == VcDtype::uint8) ? "|u1" : "<u2";
    zarray["fill_value"] = 0;
    zarray["order"] = "C";
    zarray["dimension_separator"] = dimensionSeparator;

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
    } else if (compressor.empty() || compressor == "none") {
        zarray["compressor"] = nullptr;
    } else {
        zarray["compressor"] = {{"id", compressor}};
    }
    zarray["filters"] = nullptr;

    std::ofstream f(dsPath / ".zarray");
    f << zarray.dump(2) << '\n';
    f.close();

    // Also write .zgroup in parent if it doesn't exist
    auto zgroupPath = parentPath / ".zgroup";
    if (!fs::exists(zgroupPath)) {
        std::ofstream g(zgroupPath);
        g << R"({"zarr_format": 2})" << '\n';
    }

    return std::make_unique<VcDataset>(dsPath);
}

}  // namespace vc
