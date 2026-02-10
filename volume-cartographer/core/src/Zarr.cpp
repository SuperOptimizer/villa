#include "vc/core/util/Zarr.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

// POSIX for hot-path chunk I/O
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

namespace vc::zarr {

// ============================================================================
// Dtype helpers
// ============================================================================

static Dtype parseDtypeV2(const std::string& s)
{
    // v2 uses numpy typestr: "|u1", "<u2"/"<u1", "<f4", etc.
    if (s == "|u1" || s == "<u1" || s == ">u1" || s == "uint8") return Dtype::uint8;
    if (s == "|u2" || s == "<u2" || s == ">u2" || s == "uint16") return Dtype::uint16;
    if (s == "<f4" || s == ">f4" || s == "|f4" || s == "float32") return Dtype::float32;
    throw std::runtime_error("Unsupported zarr dtype: " + s);
}

static Dtype parseDtypeV3(const std::string& s)
{
    if (s == "uint8") return Dtype::uint8;
    if (s == "uint16") return Dtype::uint16;
    if (s == "float32") return Dtype::float32;
    throw std::runtime_error("Unsupported zarr v3 data_type: " + s);
}

static std::string dtypeToV2String(Dtype d)
{
    switch (d) {
    case Dtype::uint8: return "<u1";
    case Dtype::uint16: return "<u2";
    case Dtype::float32: return "<f4";
    }
    return "<u1";
}

static std::string dtypeToV3String(Dtype d)
{
    switch (d) {
    case Dtype::uint8: return "uint8";
    case Dtype::uint16: return "uint16";
    case Dtype::float32: return "float32";
    }
    return "uint8";
}

static Dtype parseDtypeFromString(const std::string& s)
{
    if (s == "uint8") return Dtype::uint8;
    if (s == "uint16") return Dtype::uint16;
    if (s == "float32") return Dtype::float32;
    return parseDtypeV2(s);  // fall back to numpy typestr parsing
}

static std::size_t dtypeSizeOf(Dtype d)
{
    switch (d) {
    case Dtype::uint8: return 1;
    case Dtype::uint16: return 2;
    case Dtype::float32: return 4;
    }
    return 1;
}

// ============================================================================
// Fill value parsing
// ============================================================================

double parseFillValue(const nlohmann::json& j, Dtype /*dtype*/)
{
    if (j.is_null()) return 0.0;
    if (j.is_number()) return j.get<double>();
    if (j.is_string()) {
        std::string s = j.get<std::string>();
        if (s == "NaN") return std::numeric_limits<double>::quiet_NaN();
        if (s == "Infinity") return std::numeric_limits<double>::infinity();
        if (s == "-Infinity") return -std::numeric_limits<double>::infinity();
        // Hex-encoded float bits (e.g. "0x7fc00000" for float32 NaN)
        if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
            unsigned long long bits = std::stoull(s, nullptr, 16);
            // Interpret as float32 bits
            float f;
            auto b32 = static_cast<uint32_t>(bits);
            std::memcpy(&f, &b32, sizeof(f));
            return static_cast<double>(f);
        }
    }
    return 0.0;
}

// ============================================================================
// Dataset
// ============================================================================

// -- Shard write buffer -------------------------------------------------------

struct Dataset::ShardBuffer {
    struct ShardKey {
        std::size_t iz, iy, ix;
        bool operator==(const ShardKey& o) const
        {
            return iz == o.iz && iy == o.iy && ix == o.ix;
        }
    };
    struct ShardKeyHash {
        std::size_t operator()(const ShardKey& k) const
        {
            return k.iz * 73856093 ^ k.iy * 19349663 ^ k.ix * 83492791;
        }
    };
    struct PendingShard {
        std::mutex mtx;
        std::vector<std::vector<uint8_t>> compressedChunks;
        std::vector<bool> touched;     // which inner chunk slots have been written
        std::size_t numTouched = 0;    // count of touched slots
        std::size_t bufferedBytes = 0; // compressed bytes held in this shard
        uint64_t lastTouch = 0;        // generation counter for LRU
    };

    const ShardingIndexedCodec* shardCodec = nullptr;
    ShapeType shardGrid;       // inner chunks per shard per dimension
    std::size_t numInner = 0;  // product of shardGrid (for full shards)
    ShapeType datasetShape;    // dataset shape (for boundary shard computation)
    ShapeType innerShape;      // inner chunk shape

    std::mutex mapMtx;
    std::unordered_map<ShardKey, std::unique_ptr<PendingShard>, ShardKeyHash> pending;
    std::atomic<uint64_t> generation{0};
    std::atomic<std::size_t> totalBufferedBytes{0};
    std::size_t maxBufferedBytes = 0;  // 0 = unlimited

    PendingShard& getOrCreate(const ShardKey& key)
    {
        std::lock_guard<std::mutex> lock(mapMtx);
        auto it = pending.find(key);
        if (it != pending.end()) return *it->second;
        auto s = std::make_unique<PendingShard>();
        s->compressedChunks.resize(numInner);
        s->touched.resize(numInner, false);
        auto& ref = *s;
        pending.emplace(key, std::move(s));
        return ref;
    }

    std::size_t flatIndex(std::size_t iz, std::size_t iy, std::size_t ix) const
    {
        return iz * shardGrid[1] * shardGrid[2] +
               iy * shardGrid[2] + ix;
    }

    // Compute how many inner chunks exist in a shard at the dataset boundary.
    // Interior shards have numInner; boundary shards may have fewer.
    std::size_t expectedChunks(std::size_t sz, std::size_t sy, std::size_t sx) const
    {
        auto chunksInDim = [](std::size_t dimSize, std::size_t shardPos,
                              std::size_t gridDim, std::size_t innerDim) {
            // Total inner chunks in this dimension of the dataset
            std::size_t totalChunks = (dimSize + innerDim - 1) / innerDim;
            // First inner chunk index in this shard
            std::size_t first = shardPos * gridDim;
            // How many inner chunks this shard covers in this dimension
            if (first >= totalChunks) return std::size_t{0};
            return std::min(gridDim, totalChunks - first);
        };
        std::size_t nz = chunksInDim(datasetShape[0], sz, shardGrid[0], innerShape[0]);
        std::size_t ny = chunksInDim(datasetShape[1], sy, shardGrid[1], innerShape[1]);
        std::size_t nx = chunksInDim(datasetShape[2], sx, shardGrid[2], innerShape[2]);
        return nz * ny * nx;
    }
};

Dataset::Dataset() = default;

Dataset::~Dataset()
{
    try {
        flush();
    } catch (const std::exception& e) {
        std::cerr << "Dataset::~Dataset: flush failed: " << e.what()
                  << std::endl;
    }
}

Dataset::Dataset(Dataset&& o) noexcept = default;
Dataset& Dataset::operator=(Dataset&& o) noexcept = default;

Dataset::operator bool() const { return valid_; }
const ShapeType& Dataset::shape() const { return shape_; }
const ShapeType& Dataset::chunkShape() const { return chunkShape_; }
Dtype Dataset::dtype() const { return dtype_; }
bool Dataset::isUint8() const { return dtype_ == Dtype::uint8; }
bool Dataset::isUint16() const { return dtype_ == Dtype::uint16; }
bool Dataset::isFloat32() const { return dtype_ == Dtype::float32; }
const fs::path& Dataset::path() const { return path_; }
ZarrVersion Dataset::version() const { return version_; }
double Dataset::fillValue() const { return fillValue_; }
bool Dataset::isSharded() const { return codecs_.isSharded(); }
ShapeType Dataset::innerChunkShape() const { return codecs_.innerChunkShape(); }
const CodecPipeline& Dataset::codecs() const { return codecs_; }

std::size_t Dataset::dtypeSize() const { return dtypeSizeOf(dtype_); }

std::size_t Dataset::chunkSize() const
{
    if (chunkShape_.empty()) return 0;
    std::size_t s = 1;
    for (auto d : chunkShape_) s *= d;
    return s;
}

void Dataset::fillWithFillValue(void* out, std::size_t totalElems) const
{
    const std::size_t outBytes = totalElems * dtypeSize();
    if (fillValue_ == 0 && !std::isnan(fillValue_)) {
        std::memset(out, 0, outBytes);
    } else {
        switch (dtype_) {
        case Dtype::uint8: {
            auto v = static_cast<uint8_t>(fillValue_);
            std::memset(out, v, outBytes);
            break;
        }
        case Dtype::uint16: {
            auto v = static_cast<uint16_t>(fillValue_);
            auto* p = static_cast<uint16_t*>(out);
            for (std::size_t i = 0; i < totalElems; ++i) p[i] = v;
            break;
        }
        case Dtype::float32: {
            auto v = static_cast<float>(fillValue_);
            auto* p = static_cast<float*>(out);
            for (std::size_t i = 0; i < totalElems; ++i) p[i] = v;
            break;
        }
        }
    }
}

// -- Shard buffer init / flush ------------------------------------------------

void Dataset::initShardBuffer()
{
    if (shardBuf_ || !isSharded()) return;

    auto buf = std::make_unique<ShardBuffer>();
    buf->shardCodec = codecs_.shardingCodec();

    const auto& innerShp = buf->shardCodec->innerChunkShape;
    const std::size_t ndim = chunkShape_.size();
    buf->shardGrid.resize(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        buf->shardGrid[d] =
            (chunkShape_[d] + innerShp[d] - 1) / innerShp[d];
    }
    buf->numInner = 1;
    for (auto d : buf->shardGrid) buf->numInner *= d;
    buf->datasetShape = shape_;
    buf->innerShape = innerShp;

    shardBuf_ = std::move(buf);
}

void Dataset::flush()
{
    if (!shardBuf_) return;

    std::vector<std::pair<ShardBuffer::ShardKey,
                          std::unique_ptr<ShardBuffer::PendingShard>>> toWrite;
    {
        std::lock_guard<std::mutex> lock(shardBuf_->mapMtx);
        for (auto& [key, shard] : shardBuf_->pending) {
            toWrite.emplace_back(key, std::move(shard));
        }
        shardBuf_->pending.clear();
    }

    for (auto& [key, shard] : toWrite) {
        auto assembled = shardBuf_->shardCodec->assembleShard(
            shard->compressedChunks, shardBuf_->numInner);
        writeShardRaw(ShapeType{key.iz, key.iy, key.ix}, assembled);
    }
    shardBuf_->totalBufferedBytes.store(0, std::memory_order_relaxed);
}

void Dataset::setShardBufferLimit(std::size_t maxBytes)
{
    initShardBuffer();
    shardBuf_->maxBufferedBytes = maxBytes;
}

void Dataset::evictOldestShards()
{
    if (!shardBuf_ || shardBuf_->maxBufferedBytes == 0) return;
    if (shardBuf_->totalBufferedBytes.load(std::memory_order_relaxed)
        <= shardBuf_->maxBufferedBytes)
        return;

    struct Candidate {
        ShardBuffer::ShardKey key;
        uint64_t lastTouch;
        std::size_t bytes;
    };
    std::vector<Candidate> candidates;

    {
        std::lock_guard<std::mutex> lock(shardBuf_->mapMtx);
        candidates.reserve(shardBuf_->pending.size());
        for (auto& [key, shard] : shardBuf_->pending) {
            candidates.push_back({key, shard->lastTouch, shard->bufferedBytes});
        }
    }

    // Sort oldest first
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.lastTouch < b.lastTouch;
              });

    // Evict oldest until under 3/4 of limit
    std::size_t target = shardBuf_->maxBufferedBytes * 3 / 4;
    std::vector<std::pair<ShardBuffer::ShardKey,
                          std::unique_ptr<ShardBuffer::PendingShard>>> toWrite;

    {
        std::lock_guard<std::mutex> lock(shardBuf_->mapMtx);
        for (auto& c : candidates) {
            if (shardBuf_->totalBufferedBytes.load(std::memory_order_relaxed)
                <= target)
                break;
            auto it = shardBuf_->pending.find(c.key);
            if (it == shardBuf_->pending.end()) continue;
            shardBuf_->totalBufferedBytes.fetch_sub(
                it->second->bufferedBytes, std::memory_order_relaxed);
            toWrite.emplace_back(c.key, std::move(it->second));
            shardBuf_->pending.erase(it);
        }
    }

    for (auto& [key, shard] : toWrite) {
        auto assembled = shardBuf_->shardCodec->assembleShard(
            shard->compressedChunks, shardBuf_->numInner);
        writeShardRaw(ShapeType{key.iz, key.iy, key.ix}, assembled);
    }
}

void Dataset::writeShardRaw(const ShapeType& shardIdx,
                             const std::vector<uint8_t>& assembled)
{
    auto cp = chunkPath(shardIdx);

    auto parent = cp.parent_path();
    if (!fs::exists(parent))
        fs::create_directories(parent);

    int fd = ::open(cp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error(
            "Dataset::writeShardRaw: cannot create shard file: " +
            cp.string());
    }

    std::size_t written = 0;
    while (written < assembled.size()) {
        ssize_t w = ::write(fd, assembled.data() + written,
                            assembled.size() - written);
        if (w <= 0) break;
        written += static_cast<std::size_t>(w);
    }
    ::close(fd);
}

// -- Metadata parsing ---------------------------------------------------------

void Dataset::readV1Metadata(const fs::path& dsPath)
{
    auto metaPath = dsPath / "meta";
    std::ifstream f(metaPath);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + metaPath.string());

    nlohmann::json j = nlohmann::json::parse(f);

    shape_ = j.at("shape").get<ShapeType>();
    chunkShape_ = j.at("chunks").get<ShapeType>();
    dtype_ = parseDtypeV2(j.at("dtype").get<std::string>());
    fillValue_ = parseFillValue(j.value("fill_value", nlohmann::json()), dtype_);
    delimiter_ = ".";  // v1 always uses "."

    std::string compression = j.value("compression", std::string("none"));
    nlohmann::json compressionOpts = j.value("compression_opts", nlohmann::json());
    std::string order = j.value("order", std::string("C"));

    codecs_ = CodecPipeline::fromV1(compression, compressionOpts, order,
                                     dtypeSizeOf(dtype_));
    chunkKeyEncoding_ = "v2";
    version_ = ZarrVersion::v1;
}

void Dataset::readV2Metadata(const fs::path& dsPath)
{
    auto zarrayPath = dsPath / ".zarray";
    std::ifstream f(zarrayPath);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + zarrayPath.string());

    nlohmann::json j = nlohmann::json::parse(f);

    shape_ = j.at("shape").get<ShapeType>();
    chunkShape_ = j.at("chunks").get<ShapeType>();
    dtype_ = parseDtypeV2(j.at("dtype").get<std::string>());
    fillValue_ = parseFillValue(j.value("fill_value", nlohmann::json()), dtype_);
    delimiter_ = j.value("dimension_separator", std::string("."));

    nlohmann::json compressor;
    if (j.contains("compressor"))
        compressor = j["compressor"];

    nlohmann::json filters;
    if (j.contains("filters"))
        filters = j["filters"];

    std::string order = j.value("order", std::string("C"));

    codecs_ = CodecPipeline::fromV2(compressor, filters, order,
                                     dtypeSizeOf(dtype_));
    chunkKeyEncoding_ = "v2";
    version_ = ZarrVersion::v2;
}

void Dataset::readV3Metadata(const fs::path& dsPath)
{
    auto zarrJsonPath = dsPath / "zarr.json";
    std::ifstream f(zarrJsonPath);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + zarrJsonPath.string());

    nlohmann::json j = nlohmann::json::parse(f);

    if (j.value("node_type", "") != "array")
        throw std::runtime_error("zarr.json node_type is not 'array' at " + dsPath.string());

    shape_ = j.at("shape").get<ShapeType>();
    dtype_ = parseDtypeV3(j.at("data_type").get<std::string>());
    fillValue_ = parseFillValue(j.value("fill_value", nlohmann::json()), dtype_);

    // Chunk grid
    if (j.contains("chunk_grid")) {
        auto& cg = j["chunk_grid"];
        if (cg.value("name", "") == "regular" && cg.contains("configuration")) {
            chunkShape_ = cg["configuration"].at("chunk_shape").get<ShapeType>();
        }
    }

    // Chunk key encoding → separator
    delimiter_ = "/";  // v3 default
    chunkKeyEncoding_ = "default";
    if (j.contains("chunk_key_encoding")) {
        auto& cke = j["chunk_key_encoding"];
        std::string ckeName = cke.value("name", std::string("default"));
        chunkKeyEncoding_ = ckeName;
        if (cke.contains("configuration") && cke["configuration"].contains("separator")) {
            delimiter_ = cke["configuration"]["separator"].get<std::string>();
        }
    }

    // Codecs — parse via pipeline
    if (j.contains("codecs") && j["codecs"].is_array()) {
        codecs_ = CodecPipeline::fromV3Json(j["codecs"]);
    } else {
        codecs_ = CodecPipeline::defaultPipeline(dtypeSizeOf(dtype_));
    }

    version_ = ZarrVersion::v3;
}

void Dataset::writeV1Metadata() const
{
    nlohmann::json j;
    j["zarr_format"] = 1;
    j["shape"] = shape_;
    j["chunks"] = chunkShape_;
    j["dtype"] = dtypeToV2String(dtype_);
    j["fill_value"] = fillValue_;
    j["order"] = "C";

    // Extract compression info from codec pipeline
    if (codecs_.bytesToBytes.empty()) {
        j["compression"] = "none";
        j["compression_opts"] = nullptr;
    } else {
        auto& codec = codecs_.bytesToBytes[0];
        std::string cname = codec->name();
        if (cname == "blosc") {
            j["compression"] = "blosc";
            auto* blosc = dynamic_cast<const BloscCodec*>(codec.get());
            if (blosc) {
                j["compression_opts"] = {
                    {"cname", blosc->cname}, {"clevel", blosc->clevel},
                    {"shuffle", blosc->shuffle}, {"blocksize", blosc->blocksize}};
            }
        } else if (cname == "gzip") {
            j["compression"] = "zlib";
            auto* gz = dynamic_cast<const GzipCodec*>(codec.get());
            j["compression_opts"] = gz ? gz->level : 6;
        } else if (cname == "zstd") {
            j["compression"] = "zstd";
            auto* zstd = dynamic_cast<const ZstdCodec*>(codec.get());
            j["compression_opts"] = zstd ? zstd->level : 3;
        } else {
            j["compression"] = "none";
            j["compression_opts"] = nullptr;
        }
    }

    fs::create_directories(path_);
    {
        std::ofstream f(path_ / "meta");
        f << j.dump(4) << "\n";
    }
    {
        std::ofstream f(path_ / "attrs");
        f << "{}\n";
    }
}

void Dataset::writeV2Metadata() const
{
    nlohmann::json j;
    j["zarr_format"] = 2;
    j["shape"] = shape_;
    j["chunks"] = chunkShape_;
    j["dtype"] = dtypeToV2String(dtype_);
    j["fill_value"] = fillValue_;
    j["order"] = "C";
    j["filters"] = nullptr;
    j["dimension_separator"] = delimiter_;

    // Extract compressor from codec pipeline
    if (codecs_.bytesToBytes.empty()) {
        j["compressor"] = nullptr;
    } else {
        auto& codec = codecs_.bytesToBytes[0];
        std::string cname = codec->name();
        if (cname == "blosc") {
            auto* blosc = dynamic_cast<const BloscCodec*>(codec.get());
            if (blosc) {
                j["compressor"] = {
                    {"id", "blosc"}, {"cname", blosc->cname},
                    {"clevel", blosc->clevel}, {"shuffle", blosc->shuffle},
                    {"blocksize", blosc->blocksize}};
            }
        } else if (cname == "gzip") {
            auto* gz = dynamic_cast<const GzipCodec*>(codec.get());
            j["compressor"] = {{"id", "zlib"}, {"level", gz ? gz->level : 6}};
        } else if (cname == "zstd") {
            auto* zstd = dynamic_cast<const ZstdCodec*>(codec.get());
            j["compressor"] = {{"id", "zstd"}, {"level", zstd ? zstd->level : 3}};
        } else {
            j["compressor"] = nullptr;
        }
    }

    fs::create_directories(path_);
    std::ofstream f(path_ / ".zarray");
    f << j.dump(4) << "\n";
}

void Dataset::writeV3Metadata() const
{
    nlohmann::json j;
    j["zarr_format"] = 3;
    j["node_type"] = "array";
    j["shape"] = shape_;
    j["data_type"] = dtypeToV3String(dtype_);
    j["fill_value"] = fillValue_;

    j["chunk_grid"] = {
        {"name", "regular"},
        {"configuration", {{"chunk_shape", chunkShape_}}}
    };

    j["chunk_key_encoding"] = {
        {"name", chunkKeyEncoding_ == "v2" ? "v2" : "default"},
        {"configuration", {{"separator", delimiter_}}}
    };

    j["codecs"] = codecs_.toJson();
    j["attributes"] = nlohmann::json::object();
    j["dimension_names"] = nlohmann::json::array();

    fs::create_directories(path_);

    // v3: chunk data lives under c/ subdirectory
    if (chunkKeyEncoding_ == "default") {
        fs::create_directories(path_ / "c");
    }

    std::ofstream f(path_ / "zarr.json");
    f << j.dump(4) << "\n";
}

// -- Factory methods ----------------------------------------------------------

Dataset Dataset::open(const fs::path& dsPath)
{
    Dataset ds;
    ds.path_ = dsPath;

    if (fs::exists(dsPath / ".zarray")) {
        ds.readV2Metadata(dsPath);
    } else if (fs::exists(dsPath / "zarr.json")) {
        ds.readV3Metadata(dsPath);
    } else if (fs::exists(dsPath / "meta")) {
        ds.readV1Metadata(dsPath);
    } else {
        throw std::runtime_error(
            "No .zarray, zarr.json, or meta found at " + dsPath.string());
    }

    ds.valid_ = true;
    return ds;
}

Dataset Dataset::create(
    const fs::path& dsPath,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& /*compressor*/,
    const nlohmann::json& compressorOpts,
    double fillValue,
    const std::string& zarrDelimiter)
{
    Dataset ds;
    ds.path_ = dsPath;
    ds.shape_ = shape;
    ds.chunkShape_ = chunks;
    ds.dtype_ = parseDtypeFromString(dtype);
    ds.fillValue_ = fillValue;
    ds.delimiter_ = zarrDelimiter;
    ds.version_ = ZarrVersion::v2;
    ds.chunkKeyEncoding_ = "v2";

    // Build codec pipeline from compressor opts
    nlohmann::json comp;
    comp["id"] = "blosc";
    if (compressorOpts.contains("cname"))
        comp["cname"] = compressorOpts["cname"];
    else
        comp["cname"] = "zstd";
    if (compressorOpts.contains("clevel"))
        comp["clevel"] = compressorOpts["clevel"];
    else
        comp["clevel"] = 1;
    if (compressorOpts.contains("shuffle"))
        comp["shuffle"] = compressorOpts["shuffle"];
    else
        comp["shuffle"] = 0;
    if (compressorOpts.contains("blocksize"))
        comp["blocksize"] = compressorOpts["blocksize"];
    else
        comp["blocksize"] = 0;

    ds.codecs_ = CodecPipeline::fromV2(comp, nullptr, "C",
                                        dtypeSizeOf(ds.dtype_));

    ds.writeV2Metadata();
    ds.valid_ = true;
    return ds;
}

Dataset Dataset::createV3(
    const fs::path& dsPath,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& /*compressor*/,
    const nlohmann::json& compressorOpts,
    double fillValue,
    const std::string& separator)
{
    Dataset ds;
    ds.path_ = dsPath;
    ds.shape_ = shape;
    ds.chunkShape_ = chunks;
    ds.dtype_ = parseDtypeFromString(dtype);
    ds.fillValue_ = fillValue;
    ds.delimiter_ = separator;
    ds.version_ = ZarrVersion::v3;
    ds.chunkKeyEncoding_ = "default";

    // Build v3 codec pipeline
    nlohmann::json codecsArr = nlohmann::json::array();
    codecsArr.push_back({{"name", "bytes"},
                          {"configuration", {{"endian", "little"}}}});

    nlohmann::json bloscCfg;
    bloscCfg["cname"] = compressorOpts.value("cname", std::string("zstd"));
    bloscCfg["clevel"] = compressorOpts.value("clevel", 1);
    int sh = compressorOpts.value("shuffle", 0);
    bloscCfg["shuffle"] = sh == 0 ? "noshuffle" : (sh == 2 ? "bitshuffle" : "shuffle");
    bloscCfg["blocksize"] = compressorOpts.value("blocksize", 0);
    bloscCfg["typesize"] = static_cast<int>(dtypeSizeOf(ds.dtype_));
    codecsArr.push_back({{"name", "blosc"}, {"configuration", bloscCfg}});

    ds.codecs_ = CodecPipeline::fromV3Json(codecsArr);

    ds.writeV3Metadata();
    ds.valid_ = true;
    return ds;
}

Dataset Dataset::createV3Sharded(
    const fs::path& dsPath,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& shardShape,
    const ShapeType& innerChunkShape,
    const nlohmann::json& compressorOpts,
    double fillValue,
    const std::string& separator)
{
    Dataset ds;
    ds.path_ = dsPath;
    ds.shape_ = shape;
    ds.chunkShape_ = shardShape;  // chunk_grid = shard shape
    ds.dtype_ = parseDtypeFromString(dtype);
    ds.fillValue_ = fillValue;
    ds.delimiter_ = separator;
    ds.version_ = ZarrVersion::v3;
    ds.chunkKeyEncoding_ = "default";

    // Build sharding codec pipeline:
    //   sharding_indexed(inner=[bytes, blosc], index=[bytes, crc32c])
    nlohmann::json innerCodecsArr = nlohmann::json::array();
    innerCodecsArr.push_back({{"name", "bytes"},
                               {"configuration", {{"endian", "little"}}}});

    nlohmann::json bloscCfg;
    bloscCfg["cname"] = compressorOpts.value("cname", std::string("zstd"));
    bloscCfg["clevel"] = compressorOpts.value("clevel", 1);
    int sh = compressorOpts.value("shuffle", 0);
    bloscCfg["shuffle"] = sh == 0 ? "noshuffle" : (sh == 2 ? "bitshuffle" : "shuffle");
    bloscCfg["blocksize"] = compressorOpts.value("blocksize", 0);
    bloscCfg["typesize"] = static_cast<int>(dtypeSizeOf(ds.dtype_));
    innerCodecsArr.push_back({{"name", "blosc"}, {"configuration", bloscCfg}});

    nlohmann::json indexCodecsArr = nlohmann::json::array();
    indexCodecsArr.push_back({{"name", "bytes"},
                               {"configuration", {{"endian", "little"}}}});
    indexCodecsArr.push_back({{"name", "crc32c"}});

    nlohmann::json shardingCfg;
    shardingCfg["chunk_shape"] = innerChunkShape;
    shardingCfg["codecs"] = innerCodecsArr;
    shardingCfg["index_codecs"] = indexCodecsArr;
    shardingCfg["index_location"] = "end";

    nlohmann::json codecsArr = nlohmann::json::array();
    codecsArr.push_back({{"name", "sharding_indexed"},
                          {"configuration", shardingCfg}});

    ds.codecs_ = CodecPipeline::fromV3Json(codecsArr);

    ds.writeV3Metadata();
    ds.valid_ = true;
    return ds;
}

Dataset Dataset::createV1(
    const fs::path& dsPath,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& compression,
    const nlohmann::json& compressionOpts,
    double fillValue,
    const std::string& order)
{
    Dataset ds;
    ds.path_ = dsPath;
    ds.shape_ = shape;
    ds.chunkShape_ = chunks;
    ds.dtype_ = parseDtypeFromString(dtype);
    ds.fillValue_ = fillValue;
    ds.delimiter_ = ".";
    ds.version_ = ZarrVersion::v1;
    ds.chunkKeyEncoding_ = "v2";

    ds.codecs_ = CodecPipeline::fromV1(compression, compressionOpts, order,
                                        dtypeSizeOf(ds.dtype_));

    ds.writeV1Metadata();
    ds.valid_ = true;
    return ds;
}

// -- Chunk path ---------------------------------------------------------------

fs::path Dataset::chunkPath(const ShapeType& idx) const
{
    std::string key;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (i > 0) key += delimiter_;
        key += std::to_string(idx[i]);
    }

    if (chunkKeyEncoding_ == "default") {
        // v3 default: chunks under c/ subdirectory
        return path_ / "c" / key;
    }

    if (delimiter_ == "/") {
        // v2 with slash delimiter: nested directories
        fs::path p = path_;
        for (auto d : idx) p /= std::to_string(d);
        return p;
    }

    return path_ / key;
}

fs::path Dataset::chunkPath(std::size_t iz, std::size_t iy, std::size_t ix) const
{
    return chunkPath(ShapeType{iz, iy, ix});
}

// -- N-D Chunk I/O ------------------------------------------------------------

bool Dataset::chunkExists(const ShapeType& chunkIdx) const
{
    return fs::exists(chunkPath(chunkIdx));
}

bool Dataset::readChunk(const ShapeType& chunkIdx, void* out) const
{
    auto cp = chunkPath(chunkIdx);
    const std::size_t elemSize = dtypeSize();
    const std::size_t totalElems = chunkSize();
    const std::size_t outBytes = totalElems * elemSize;

    int fd = ::open(cp.c_str(), O_RDONLY);
    if (fd < 0) {
        fillWithFillValue(out, totalElems);
        return false;
    }

    struct stat st;
    ::fstat(fd, &st);
    const std::size_t fileSize = static_cast<std::size_t>(st.st_size);

    // Read compressed data
    std::vector<uint8_t> compressed(fileSize);
    std::size_t bytesRead = 0;
    while (bytesRead < fileSize) {
        ssize_t r = ::read(fd, compressed.data() + bytesRead, fileSize - bytesRead);
        if (r <= 0) break;
        bytesRead += static_cast<std::size_t>(r);
    }
    ::close(fd);

    if (bytesRead != fileSize) {
        throw std::runtime_error(
            "Truncated chunk file (read " + std::to_string(bytesRead) +
            " of " + std::to_string(fileSize) + " bytes): " + cp.string());
    }

    // Decompress through codec pipeline
    codecs_.decode(compressed.data(), bytesRead, out, outBytes,
                    chunkShape_, elemSize);

    return true;
}

void Dataset::writeChunk(const ShapeType& chunkIdx, const void* data)
{
    const std::size_t elemSize = dtypeSize();
    const std::size_t totalElems = chunkSize();
    const std::size_t srcBytes = totalElems * elemSize;

    // Check if all fill_value — skip write
    bool allFill = true;
    if (fillValue_ == 0 && !std::isnan(fillValue_)) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (std::size_t i = 0; i < srcBytes && allFill; ++i)
            if (p[i] != 0) allFill = false;
    } else {
        allFill = false;
    }
    if (allFill) return;

    // Compress through codec pipeline
    auto compressed = codecs_.encode(data, srcBytes, chunkShape_, elemSize);

    auto cp = chunkPath(chunkIdx);

    // Create parent dirs if needed
    auto parent = cp.parent_path();
    if (!fs::exists(parent))
        fs::create_directories(parent);

    int fd = ::open(cp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Cannot create chunk file: " + cp.string());
    }

    std::size_t written = 0;
    while (written < compressed.size()) {
        ssize_t w = ::write(fd, compressed.data() + written,
                            compressed.size() - written);
        if (w <= 0) break;
        written += static_cast<std::size_t>(w);
    }
    ::close(fd);
}

// -- 3D convenience overloads -------------------------------------------------

bool Dataset::chunkExists(std::size_t iz, std::size_t iy, std::size_t ix) const
{
    return chunkExists(ShapeType{iz, iy, ix});
}

bool Dataset::readChunk(std::size_t iz, std::size_t iy, std::size_t ix, void* out) const
{
    return readChunk(ShapeType{iz, iy, ix}, out);
}

void Dataset::writeChunk(std::size_t iz, std::size_t iy, std::size_t ix, const void* data)
{
    writeChunk(ShapeType{iz, iy, ix}, data);
}

// -- Sharded access -----------------------------------------------------------

bool Dataset::readShardedChunk(const ShapeType& shardIdx,
                               const ShapeType& innerIdx, void* out) const
{
    auto* sc = codecs_.shardingCodec();
    if (!sc) {
        throw std::runtime_error("readShardedChunk: dataset is not sharded");
    }

    auto cp = chunkPath(shardIdx);
    int fd = ::open(cp.c_str(), O_RDONLY);
    if (fd < 0) {
        // Shard doesn't exist — fill with fillValue
        std::size_t innerElems = 1;
        for (auto d : sc->innerChunkShape) innerElems *= d;
        fillWithFillValue(out, innerElems);
        return false;
    }

    struct stat st;
    ::fstat(fd, &st);

    // Compute shard grid shape
    const std::size_t ndim = chunkShape_.size();
    ShapeType shardGrid(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        shardGrid[d] =
            (chunkShape_[d] + sc->innerChunkShape[d] - 1) / sc->innerChunkShape[d];
    }

    std::size_t innerElems = 1;
    for (auto d : sc->innerChunkShape) innerElems *= d;
    std::size_t outBytes = innerElems * dtypeSize();

    bool found = sc->readInnerChunk(fd, st.st_size, innerIdx, shardGrid,
                                     dtypeSize(), out, outBytes);
    ::close(fd);

    if (!found) {
        fillWithFillValue(out, innerElems);
    }
    return found;
}

void Dataset::writeShardedChunk(const ShapeType& chunkIdx, const void* data)
{
    if (!isSharded()) {
        throw std::runtime_error("writeShardedChunk: dataset is not sharded");
    }

    initShardBuffer();
    auto* sc = shardBuf_->shardCodec;
    const auto& grid = shardBuf_->shardGrid;
    const std::size_t ndim = chunkIdx.size();

    // Map global chunk coords → (shardIdx, innerIdx)
    ShapeType shardIdx(ndim), innerIdx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        shardIdx[d] = chunkIdx[d] / grid[d];
        innerIdx[d] = chunkIdx[d] % grid[d];
    }

    std::size_t innerElems = 1;
    for (auto d : sc->innerChunkShape) innerElems *= d;
    const std::size_t innerBytes = innerElems * dtypeSize();

    // Check if all fill_value — store empty sentinel
    bool allFill = true;
    if (fillValue_ == 0 && !std::isnan(fillValue_)) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (std::size_t i = 0; i < innerBytes && allFill; ++i)
            if (p[i] != 0) allFill = false;
    } else {
        allFill = false;
    }

    // Compute flat inner index
    const std::size_t flatIdx =
        shardBuf_->flatIndex(innerIdx[0], innerIdx[1], innerIdx[2]);

    ShardBuffer::ShardKey key{shardIdx[0], shardIdx[1], shardIdx[2]};
    auto& shard = shardBuf_->getOrCreate(key);

    const std::size_t expected =
        shardBuf_->expectedChunks(key.iz, key.iy, key.ix);

    bool shardComplete = false;
    {
        // Compress outside the lock if non-fill
        std::vector<uint8_t> compressed;
        if (!allFill) {
            compressed = sc->encodeInnerChunk(data, innerBytes, dtypeSize());
        }

        std::lock_guard<std::mutex> lock(shard.mtx);

        // Track byte delta for this slot
        std::size_t oldBytes = shard.compressedChunks[flatIdx].size();
        std::size_t newBytes = compressed.size();

        if (allFill) {
            shard.compressedChunks[flatIdx].clear();
            newBytes = 0;
        } else {
            shard.compressedChunks[flatIdx] = std::move(compressed);
        }

        shard.bufferedBytes += newBytes;
        shard.bufferedBytes -= oldBytes;
        shard.lastTouch = shardBuf_->generation.fetch_add(1, std::memory_order_relaxed);

        // Update global byte tracking
        if (newBytes > oldBytes)
            shardBuf_->totalBufferedBytes.fetch_add(newBytes - oldBytes, std::memory_order_relaxed);
        else if (oldBytes > newBytes)
            shardBuf_->totalBufferedBytes.fetch_sub(oldBytes - newBytes, std::memory_order_relaxed);

        // Only increment on first write to this slot
        if (!shard.touched[flatIdx]) {
            shard.touched[flatIdx] = true;
            ++shard.numTouched;
        }
        shardComplete = (shard.numTouched == expected);
    }

    // Auto-flush completed shard to free memory
    if (shardComplete) {
        std::vector<std::vector<uint8_t>> chunks;
        std::size_t shardBytes;
        {
            std::lock_guard<std::mutex> lock(shard.mtx);
            chunks = std::move(shard.compressedChunks);
            shardBytes = shard.bufferedBytes;
            shard.bufferedBytes = 0;
        }

        shardBuf_->totalBufferedBytes.fetch_sub(shardBytes, std::memory_order_relaxed);

        auto assembled = shardBuf_->shardCodec->assembleShard(
            chunks, shardBuf_->numInner);

        writeShardRaw(shardIdx, assembled);

        // Remove from pending map
        {
            std::lock_guard<std::mutex> lock(shardBuf_->mapMtx);
            shardBuf_->pending.erase(key);
        }
    }

    // LRU eviction if over budget
    evictOldestShards();
}

// ============================================================================
// File
// ============================================================================

File::File(const fs::path& path) : path_(path) {}
File::~File() = default;
File::File(File&& o) noexcept = default;
File& File::operator=(File&& o) noexcept = default;

const fs::path& File::path() const { return path_; }

void File::keys(std::vector<std::string>& out) const
{
    out.clear();
    if (!fs::is_directory(path_)) return;
    for (auto& entry : fs::directory_iterator(path_)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        // Skip hidden dirs and the 'c' chunk directory (v3)
        if (name.empty() || name[0] == '.') continue;
        // A subdirectory is a key if it contains .zarray, zarr.json, or .zgroup
        if (fs::exists(entry.path() / ".zarray") ||
            fs::exists(entry.path() / "zarr.json") ||
            fs::exists(entry.path() / ".zgroup")) {
            out.push_back(name);
        }
    }
    std::sort(out.begin(), out.end());
}

// ============================================================================
// File / Group operations
// ============================================================================

void createFile(const fs::path& path, bool overwrite)
{
    if (overwrite && fs::exists(path))
        fs::remove_all(path);

    fs::create_directories(path);

    // Write .zgroup
    nlohmann::json zgroup;
    zgroup["zarr_format"] = 2;
    std::ofstream f(path / ".zgroup");
    f << zgroup.dump(4) << "\n";
}

void createGroup(File& file, const std::string& name)
{
    auto groupPath = file.path() / name;
    fs::create_directories(groupPath);

    nlohmann::json zgroup;
    zgroup["zarr_format"] = 2;
    std::ofstream f(groupPath / ".zgroup");
    f << zgroup.dump(4) << "\n";
}

// ============================================================================
// Dataset operations
// ============================================================================

Dataset createDataset(
    File& file,
    const std::string& name,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& compressor,
    const nlohmann::json& compressorOpts,
    double fillValue,
    const std::string& zarrDelimiter)
{
    auto dsPath = file.path() / name;
    return Dataset::create(dsPath, dtype, shape, chunks, compressor,
                           compressorOpts, fillValue, zarrDelimiter);
}

Dataset createDatasetInGroup(
    File& file,
    const std::string& groupName,
    const std::string& name,
    const std::string& dtype,
    const ShapeType& shape,
    const ShapeType& chunks,
    const std::string& compressor,
    const nlohmann::json& compressorOpts,
    double fillValue,
    const std::string& zarrDelimiter)
{
    // Ensure group directory exists
    auto groupPath = file.path() / groupName;
    if (!fs::exists(groupPath / ".zgroup")) {
        fs::create_directories(groupPath);
        nlohmann::json zgroup;
        zgroup["zarr_format"] = 2;
        std::ofstream f(groupPath / ".zgroup");
        f << zgroup.dump(4) << "\n";
    }

    auto dsPath = groupPath / name;
    return Dataset::create(dsPath, dtype, shape, chunks, compressor,
                           compressorOpts, fillValue, zarrDelimiter);
}

Dataset openDataset(File& file, const std::string& name)
{
    return Dataset::open(file.path() / name);
}

Dataset openDatasetAutoSep(
    const fs::path& groupPath,
    const std::string& name)
{
    return Dataset::open(groupPath / name);
}

// ============================================================================
// Attribute I/O
// ============================================================================

void readAttributes(const fs::path& groupPath, nlohmann::json& out)
{
    // Try .zattrs (v2) first, then zarr.json attributes (v3)
    auto zattrsPath = groupPath / ".zattrs";
    if (fs::exists(zattrsPath)) {
        std::ifstream f(zattrsPath);
        if (f.is_open()) {
            out = nlohmann::json::parse(f);
            return;
        }
    }

    auto zarrJsonPath = groupPath / "zarr.json";
    if (fs::exists(zarrJsonPath)) {
        std::ifstream f(zarrJsonPath);
        if (f.is_open()) {
            auto j = nlohmann::json::parse(f);
            if (j.contains("attributes"))
                out = j["attributes"];
            else
                out = nlohmann::json::object();
            return;
        }
    }

    out = nlohmann::json::object();
}

void writeAttributes(File& file, const nlohmann::json& attrs)
{
    auto zattrsPath = file.path() / ".zattrs";
    std::ofstream f(zattrsPath);
    f << attrs.dump(4) << "\n";
}

void writeGroupAttributes(
    const fs::path& groupPath,
    const std::string& groupName,
    const nlohmann::json& attrs)
{
    auto p = groupPath / groupName / ".zattrs";
    std::ofstream f(p);
    f << attrs.dump(4) << "\n";
}

void writeGroupAttributes(File& file, const nlohmann::json& attrs)
{
    auto p = file.path() / ".zattrs";
    std::ofstream f(p);
    f << attrs.dump(4) << "\n";
}

// ============================================================================
// Subarray I/O — native blocking overlap implementation
// ============================================================================

template <typename T>
void readSubarray(
    const Dataset& ds,
    Array3D<T>& out,
    const ShapeType& offset)
{
    const auto& dsShape = ds.shape();
    const auto& chunkSh = ds.chunkShape();
    const auto outShape = out.shape();

    if (outShape.size() != 3 || dsShape.size() != 3 || chunkSh.size() != 3) {
        throw std::runtime_error("readSubarray: only 3D arrays supported");
    }

    const std::size_t oz = offset[0], oy = offset[1], ox = offset[2];
    const std::size_t rz = outShape[0], ry = outShape[1], rx = outShape[2];
    const std::size_t cz = chunkSh[0], cy = chunkSh[1], cx = chunkSh[2];

    // Compute chunk range
    const std::size_t minCz = oz / cz, minCy = oy / cy, minCx = ox / cx;
    const std::size_t maxCz = (oz + rz - 1) / cz;
    const std::size_t maxCy = (oy + ry - 1) / cy;
    const std::size_t maxCx = (ox + rx - 1) / cx;

    const std::size_t chunkElems = ds.chunkSize();
    std::vector<T> chunkBuf(chunkElems);

    for (std::size_t iz = minCz; iz <= maxCz; ++iz) {
        for (std::size_t iy = minCy; iy <= maxCy; ++iy) {
            for (std::size_t ix = minCx; ix <= maxCx; ++ix) {
                ds.readChunk(iz, iy, ix, chunkBuf.data());

                // Compute overlap between chunk and request
                std::size_t chunkOz = iz * cz, chunkOy = iy * cy, chunkOx = ix * cx;
                std::size_t startZ = std::max(oz, chunkOz);
                std::size_t startY = std::max(oy, chunkOy);
                std::size_t startX = std::max(ox, chunkOx);
                std::size_t endZ = std::min(oz + rz, chunkOz + cz);
                std::size_t endY = std::min(oy + ry, chunkOy + cy);
                std::size_t endX = std::min(ox + rx, chunkOx + cx);

                for (std::size_t z = startZ; z < endZ; ++z) {
                    for (std::size_t y = startY; y < endY; ++y) {
                        std::size_t srcOff = (z - chunkOz) * cy * cx +
                                             (y - chunkOy) * cx +
                                             (startX - chunkOx);
                        std::size_t dstOff = (z - oz) * ry * rx +
                                             (y - oy) * rx +
                                             (startX - ox);
                        std::size_t count = endX - startX;
                        std::memcpy(&out.data()[dstOff], &chunkBuf[srcOff],
                                    count * sizeof(T));
                    }
                }
            }
        }
    }
}

template <typename T>
void writeSubarray(
    const Dataset& ds,
    const Array3D<T>& data,
    const ShapeType& offset)
{
    const auto dataShape = data.shape();

    if (dataShape.size() != 3) {
        throw std::runtime_error("writeSubarray: only 3D arrays supported");
    }

    const std::size_t oz = offset[0], oy = offset[1], ox = offset[2];
    const std::size_t rz = dataShape[0], ry = dataShape[1], rx = dataShape[2];
    auto& dsRef = const_cast<Dataset&>(ds);

    if (ds.isSharded()) {
        // Sharded: decompose into inner chunks
        const auto innerSh = ds.innerChunkShape();
        const std::size_t cz = innerSh[0], cy = innerSh[1], cx = innerSh[2];
        const std::size_t chunkElems = cz * cy * cx;

        const std::size_t minCz = oz / cz, minCy = oy / cy, minCx = ox / cx;
        const std::size_t maxCz = (oz + rz - 1) / cz;
        const std::size_t maxCy = (oy + ry - 1) / cy;
        const std::size_t maxCx = (ox + rx - 1) / cx;

        std::vector<T> chunkBuf(chunkElems);

        for (std::size_t iz = minCz; iz <= maxCz; ++iz) {
            for (std::size_t iy = minCy; iy <= maxCy; ++iy) {
                for (std::size_t ix = minCx; ix <= maxCx; ++ix) {
                    std::size_t chunkOz = iz * cz, chunkOy = iy * cy, chunkOx = ix * cx;
                    std::size_t startZ = std::max(oz, chunkOz);
                    std::size_t startY = std::max(oy, chunkOy);
                    std::size_t startX = std::max(ox, chunkOx);
                    std::size_t endZ = std::min(oz + rz, chunkOz + cz);
                    std::size_t endY = std::min(oy + ry, chunkOy + cy);
                    std::size_t endX = std::min(ox + rx, chunkOx + cx);

                    std::memset(chunkBuf.data(), 0, chunkElems * sizeof(T));

                    for (std::size_t z = startZ; z < endZ; ++z) {
                        for (std::size_t y = startY; y < endY; ++y) {
                            std::size_t dstOff = (z - chunkOz) * cy * cx +
                                                 (y - chunkOy) * cx +
                                                 (startX - chunkOx);
                            std::size_t srcOff = (z - oz) * ry * rx +
                                                 (y - oy) * rx +
                                                 (startX - ox);
                            std::size_t count = endX - startX;
                            std::memcpy(&chunkBuf[dstOff], &data.data()[srcOff],
                                        count * sizeof(T));
                        }
                    }

                    dsRef.writeShardedChunk(
                        ShapeType{iz, iy, ix}, chunkBuf.data());
                }
            }
        }
    } else {
        // Non-sharded: original path
        const auto& chunkSh = ds.chunkShape();
        const std::size_t cz = chunkSh[0], cy = chunkSh[1], cx = chunkSh[2];
        const std::size_t chunkElems = ds.chunkSize();

        const std::size_t minCz = oz / cz, minCy = oy / cy, minCx = ox / cx;
        const std::size_t maxCz = (oz + rz - 1) / cz;
        const std::size_t maxCy = (oy + ry - 1) / cy;
        const std::size_t maxCx = (ox + rx - 1) / cx;

        std::vector<T> chunkBuf(chunkElems);

        for (std::size_t iz = minCz; iz <= maxCz; ++iz) {
            for (std::size_t iy = minCy; iy <= maxCy; ++iy) {
                for (std::size_t ix = minCx; ix <= maxCx; ++ix) {
                    std::size_t chunkOz = iz * cz, chunkOy = iy * cy, chunkOx = ix * cx;
                    std::size_t startZ = std::max(oz, chunkOz);
                    std::size_t startY = std::max(oy, chunkOy);
                    std::size_t startX = std::max(ox, chunkOx);
                    std::size_t endZ = std::min(oz + rz, chunkOz + cz);
                    std::size_t endY = std::min(oy + ry, chunkOy + cy);
                    std::size_t endX = std::min(ox + rx, chunkOx + cx);

                    bool fullChunk = (startZ == chunkOz && endZ == chunkOz + cz &&
                                      startY == chunkOy && endY == chunkOy + cy &&
                                      startX == chunkOx && endX == chunkOx + cx);

                    if (!fullChunk) {
                        ds.readChunk(iz, iy, ix, chunkBuf.data());
                    } else {
                        std::memset(chunkBuf.data(), 0, chunkElems * sizeof(T));
                    }

                    for (std::size_t z = startZ; z < endZ; ++z) {
                        for (std::size_t y = startY; y < endY; ++y) {
                            std::size_t dstOff = (z - chunkOz) * cy * cx +
                                                 (y - chunkOy) * cx +
                                                 (startX - chunkOx);
                            std::size_t srcOff = (z - oz) * ry * rx +
                                                 (y - oy) * rx +
                                                 (startX - ox);
                            std::size_t count = endX - startX;
                            std::memcpy(&chunkBuf[dstOff], &data.data()[srcOff],
                                        count * sizeof(T));
                        }
                    }

                    dsRef.writeChunk(iz, iy, ix, chunkBuf.data());
                }
            }
        }
    }
}

// ============================================================================
// Pyramid helpers
// ============================================================================

void writeZarrMultiscaleAttrs(
    File& file,
    int maxLevel,
    const nlohmann::json& extraAttrs)
{
    nlohmann::json attrs = extraAttrs;

    nlohmann::json multiscale;
    multiscale["version"] = "0.4";
    multiscale["name"] = "render";
    multiscale["axes"] = nlohmann::json::array({
        nlohmann::json{{"name", "z"}, {"type", "space"}},
        nlohmann::json{{"name", "y"}, {"type", "space"}},
        nlohmann::json{{"name", "x"}, {"type", "space"}}});
    multiscale["datasets"] = nlohmann::json::array();
    for (int level = 0; level <= maxLevel; ++level) {
        const double s = std::pow(2.0, level);
        nlohmann::json dset;
        dset["path"] = std::to_string(level);
        dset["coordinateTransformations"] = nlohmann::json::array({
            nlohmann::json{
                {"type", "scale"}, {"scale", nlohmann::json::array({s, s, s})}},
            nlohmann::json{
                {"type", "translation"},
                {"translation", nlohmann::json::array({0.0, 0.0, 0.0})}}});
        multiscale["datasets"].push_back(dset);
    }
    multiscale["metadata"] = nlohmann::json{{"downsampling_method", "mean"}};
    attrs["multiscales"] = nlohmann::json::array({multiscale});

    writeAttributes(file, attrs);
}

void buildPyramidLevel(
    File& file,
    int targetLevel,
    const std::string& dtype,
    std::size_t chunkH,
    std::size_t chunkW)
{
    auto src = Dataset::open(file.path() / std::to_string(targetLevel - 1));
    const auto& sShape = src.shape();

    ShapeType dShape = {
        (sShape[0] + 1) / 2,
        (sShape[1] + 1) / 2,
        (sShape[2] + 1) / 2};
    ShapeType dChunks = {
        dShape[0],
        std::min(chunkH, dShape[1]),
        std::min(chunkW, dShape[2])};
    nlohmann::json compOpts = {
        {"cname", "zstd"}, {"clevel", 1}, {"shuffle", 0}};

    Dataset dst = [&]() {
        auto dsPath = file.path() / std::to_string(targetLevel);
        if (fs::exists(dsPath / ".zarray") || fs::exists(dsPath / "zarr.json")) {
            return Dataset::open(dsPath);
        }
        return Dataset::create(dsPath, dtype, dShape, dChunks,
                               std::string("blosc"), compOpts);
    }();

    const std::size_t tileZ = dShape[0], tileY = chunkH, tileX = chunkW;
    const std::size_t tilesY = (dShape[1] + tileY - 1) / tileY;
    const std::size_t tilesX = (dShape[2] + tileX - 1) / tileX;
    const std::size_t totalTiles = tilesY * tilesX;
    std::atomic<std::size_t> tilesDone{0};

    const bool isU16 = (dtype == "uint16");

    for (std::size_t z = 0; z < dShape[0]; z += tileZ) {
        const std::size_t lz = std::min(tileZ, dShape[0] - z);
#pragma omp parallel for schedule(dynamic, 2)
        for (long long y = 0; y < static_cast<long long>(dShape[1]);
             y += static_cast<long long>(tileY)) {
            for (long long x = 0; x < static_cast<long long>(dShape[2]);
                 x += static_cast<long long>(tileX)) {
                const std::size_t ly =
                    std::min(tileY, static_cast<std::size_t>(dShape[1] - y));
                const std::size_t lx =
                    std::min(tileX, static_cast<std::size_t>(dShape[2] - x));

                const std::size_t sz =
                    std::min<std::size_t>(2 * lz, sShape[0] - 2 * z);
                const std::size_t sy =
                    std::min<std::size_t>(2 * ly, sShape[1] - y * 2);
                const std::size_t sx =
                    std::min<std::size_t>(2 * lx, sShape[2] - x * 2);

                if (isU16) {
                    Array3D<uint16_t> srcChunk(sz, sy, sx);
                    ShapeType sOff = {
                        static_cast<std::size_t>(2 * z),
                        static_cast<std::size_t>(2 * y),
                        static_cast<std::size_t>(2 * x)};
                    readSubarray<uint16_t>(src, srcChunk, sOff);
                    Array3D<uint16_t> dstChunk(lz, ly, lx);
                    for (std::size_t zz = 0; zz < lz; ++zz)
                        for (std::size_t yy = 0; yy < ly; ++yy)
                            for (std::size_t xx = 0; xx < lx; ++xx) {
                                uint32_t sum = 0;
                                int cnt = 0;
                                for (int dz2 = 0;
                                     dz2 < 2 && (2 * zz + dz2) < sz; ++dz2)
                                    for (int dy2 = 0;
                                         dy2 < 2 && (2 * yy + dy2) < sy;
                                         ++dy2)
                                        for (int dx2 = 0;
                                             dx2 < 2 && (2 * xx + dx2) < sx;
                                             ++dx2) {
                                            sum += srcChunk(
                                                2 * zz + dz2, 2 * yy + dy2,
                                                2 * xx + dx2);
                                            cnt += 1;
                                        }
                                dstChunk(zz, yy, xx) = static_cast<uint16_t>(
                                    (sum + (cnt / 2)) / std::max(1, cnt));
                            }
                    ShapeType dOff = {
                        z, static_cast<std::size_t>(y),
                        static_cast<std::size_t>(x)};
                    writeSubarray<uint16_t>(dst, dstChunk, dOff);
                } else {
                    Array3D<uint8_t> srcChunk(sz, sy, sx);
                    ShapeType sOff = {
                        static_cast<std::size_t>(2 * z),
                        static_cast<std::size_t>(2 * y),
                        static_cast<std::size_t>(2 * x)};
                    readSubarray<uint8_t>(src, srcChunk, sOff);
                    Array3D<uint8_t> dstChunk(lz, ly, lx);
                    for (std::size_t zz = 0; zz < lz; ++zz)
                        for (std::size_t yy = 0; yy < ly; ++yy)
                            for (std::size_t xx = 0; xx < lx; ++xx) {
                                int sum = 0;
                                int cnt = 0;
                                for (int dz2 = 0;
                                     dz2 < 2 && (2 * zz + dz2) < sz; ++dz2)
                                    for (int dy2 = 0;
                                         dy2 < 2 && (2 * yy + dy2) < sy;
                                         ++dy2)
                                        for (int dx2 = 0;
                                             dx2 < 2 && (2 * xx + dx2) < sx;
                                             ++dx2) {
                                            sum += srcChunk(
                                                2 * zz + dz2, 2 * yy + dy2,
                                                2 * xx + dx2);
                                            cnt += 1;
                                        }
                                dstChunk(zz, yy, xx) = static_cast<uint8_t>(
                                    (sum + (cnt / 2)) / std::max(1, cnt));
                            }
                    ShapeType dOff = {
                        z, static_cast<std::size_t>(y),
                        static_cast<std::size_t>(x)};
                    writeSubarray<uint8_t>(dst, dstChunk, dOff);
                }

                std::size_t done = ++tilesDone;
                int pct = static_cast<int>(
                    100.0 * double(done) / double(totalTiles));
#pragma omp critical(progress_print)
                {
                    std::cout << "\r[pyramid L" << targetLevel << "] tile "
                              << done << "/" << totalTiles << " (" << pct
                              << "%)" << std::flush;
                }
            }
        }
    }
    std::cout << std::endl;
}

// ============================================================================
// Raw-pointer writeSubarray (3D)
// ============================================================================

template <typename T>
void writeSubarray(
    const Dataset& ds,
    const T* data,
    const ShapeType& dataShape,
    const ShapeType& offset)
{
    if (dataShape.size() != 3) {
        throw std::runtime_error("writeSubarray: only 3D arrays supported");
    }

    const std::size_t oz = offset[0], oy = offset[1], ox = offset[2];
    const std::size_t rz = dataShape[0], ry = dataShape[1], rx = dataShape[2];
    auto& dsRef = const_cast<Dataset&>(ds);

    if (ds.isSharded()) {
        const auto innerSh = ds.innerChunkShape();
        const std::size_t cz = innerSh[0], cy = innerSh[1], cx = innerSh[2];
        const std::size_t chunkElems = cz * cy * cx;

        const std::size_t minCz = oz / cz, minCy = oy / cy, minCx = ox / cx;
        const std::size_t maxCz = (oz + rz - 1) / cz;
        const std::size_t maxCy = (oy + ry - 1) / cy;
        const std::size_t maxCx = (ox + rx - 1) / cx;

        std::vector<T> chunkBuf(chunkElems);

        for (std::size_t iz = minCz; iz <= maxCz; ++iz) {
            for (std::size_t iy = minCy; iy <= maxCy; ++iy) {
                for (std::size_t ix = minCx; ix <= maxCx; ++ix) {
                    std::size_t chunkOz = iz * cz, chunkOy = iy * cy, chunkOx = ix * cx;
                    std::size_t startZ = std::max(oz, chunkOz);
                    std::size_t startY = std::max(oy, chunkOy);
                    std::size_t startX = std::max(ox, chunkOx);
                    std::size_t endZ = std::min(oz + rz, chunkOz + cz);
                    std::size_t endY = std::min(oy + ry, chunkOy + cy);
                    std::size_t endX = std::min(ox + rx, chunkOx + cx);

                    std::memset(chunkBuf.data(), 0, chunkElems * sizeof(T));

                    for (std::size_t z = startZ; z < endZ; ++z) {
                        for (std::size_t y = startY; y < endY; ++y) {
                            std::size_t dstOff = (z - chunkOz) * cy * cx +
                                                 (y - chunkOy) * cx +
                                                 (startX - chunkOx);
                            std::size_t srcOff = (z - oz) * ry * rx +
                                                 (y - oy) * rx +
                                                 (startX - ox);
                            std::size_t count = endX - startX;
                            std::memcpy(&chunkBuf[dstOff], &data[srcOff],
                                        count * sizeof(T));
                        }
                    }

                    dsRef.writeShardedChunk(
                        ShapeType{iz, iy, ix}, chunkBuf.data());
                }
            }
        }
    } else {
        const auto& chunkSh = ds.chunkShape();
        const std::size_t cz = chunkSh[0], cy = chunkSh[1], cx = chunkSh[2];
        const std::size_t chunkElems = ds.chunkSize();

        const std::size_t minCz = oz / cz, minCy = oy / cy, minCx = ox / cx;
        const std::size_t maxCz = (oz + rz - 1) / cz;
        const std::size_t maxCy = (oy + ry - 1) / cy;
        const std::size_t maxCx = (ox + rx - 1) / cx;

        std::vector<T> chunkBuf(chunkElems);

        for (std::size_t iz = minCz; iz <= maxCz; ++iz) {
            for (std::size_t iy = minCy; iy <= maxCy; ++iy) {
                for (std::size_t ix = minCx; ix <= maxCx; ++ix) {
                    std::size_t chunkOz = iz * cz, chunkOy = iy * cy, chunkOx = ix * cx;
                    std::size_t startZ = std::max(oz, chunkOz);
                    std::size_t startY = std::max(oy, chunkOy);
                    std::size_t startX = std::max(ox, chunkOx);
                    std::size_t endZ = std::min(oz + rz, chunkOz + cz);
                    std::size_t endY = std::min(oy + ry, chunkOy + cy);
                    std::size_t endX = std::min(ox + rx, chunkOx + cx);

                    bool fullChunk = (startZ == chunkOz && endZ == chunkOz + cz &&
                                      startY == chunkOy && endY == chunkOy + cy &&
                                      startX == chunkOx && endX == chunkOx + cx);

                    if (!fullChunk) {
                        ds.readChunk(iz, iy, ix, chunkBuf.data());
                    } else {
                        std::memset(chunkBuf.data(), 0, chunkElems * sizeof(T));
                    }

                    for (std::size_t z = startZ; z < endZ; ++z) {
                        for (std::size_t y = startY; y < endY; ++y) {
                            std::size_t dstOff = (z - chunkOz) * cy * cx +
                                                 (y - chunkOy) * cx +
                                                 (startX - chunkOx);
                            std::size_t srcOff = (z - oz) * ry * rx +
                                                 (y - oy) * rx +
                                                 (startX - ox);
                            std::size_t count = endX - startX;
                            std::memcpy(&chunkBuf[dstOff], &data[srcOff],
                                        count * sizeof(T));
                        }
                    }

                    dsRef.writeChunk(iz, iy, ix, chunkBuf.data());
                }
            }
        }
    }
}

// ============================================================================
// N-D Subarray I/O
// ============================================================================

template <typename T>
void readSubarrayND(
    const Dataset& ds,
    T* out,
    const ShapeType& outShape,
    const ShapeType& offset)
{
    const auto& chunkSh = ds.chunkShape();
    const std::size_t ndim = outShape.size();

    if (ndim != chunkSh.size()) {
        throw std::runtime_error("readSubarrayND: ndim mismatch");
    }

    // Compute chunk ranges per dimension
    std::vector<std::size_t> minC(ndim), maxC(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        minC[d] = offset[d] / chunkSh[d];
        maxC[d] = (offset[d] + outShape[d] - 1) / chunkSh[d];
    }

    // Output strides (C-order)
    std::vector<std::size_t> outStrides(ndim);
    outStrides[ndim - 1] = 1;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
        outStrides[d] = outStrides[d + 1] * outShape[d + 1];
    }

    // Chunk strides (C-order)
    std::vector<std::size_t> chunkStrides(ndim);
    chunkStrides[ndim - 1] = 1;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
        chunkStrides[d] = chunkStrides[d + 1] * chunkSh[d + 1];
    }

    const std::size_t chunkElems = ds.chunkSize();
    std::vector<T> chunkBuf(chunkElems);

    // Total number of chunks
    std::size_t totalChunks = 1;
    for (std::size_t d = 0; d < ndim; ++d) {
        totalChunks *= (maxC[d] - minC[d] + 1);
    }

    // Iterate over all chunks in C-order
    ShapeType chunkIdx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) chunkIdx[d] = minC[d];

    for (std::size_t ci = 0; ci < totalChunks; ++ci) {
        ds.readChunk(chunkIdx, chunkBuf.data());

        // Compute overlap per dimension
        std::vector<std::size_t> start(ndim), end(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            std::size_t chunkO = chunkIdx[d] * chunkSh[d];
            start[d] = std::max(offset[d], chunkO);
            end[d] = std::min(offset[d] + outShape[d], chunkO + chunkSh[d]);
        }

        // Copy row-by-row (innermost dimension contiguous)
        // Iterate over all dims except the last
        std::size_t numRows = 1;
        for (std::size_t d = 0; d < ndim - 1; ++d) {
            numRows *= (end[d] - start[d]);
        }

        std::vector<std::size_t> coords(ndim);
        for (std::size_t row = 0; row < numRows; ++row) {
            std::size_t rem = row;
            for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
                std::size_t span = end[d] - start[d];
                coords[d] = start[d] + (rem % span);
                rem /= span;
            }

            // Source offset in chunk buffer
            std::size_t srcOff = 0;
            for (std::size_t d = 0; d < ndim - 1; ++d) {
                srcOff += (coords[d] - chunkIdx[d] * chunkSh[d]) * chunkStrides[d];
            }
            srcOff += (start[ndim - 1] - chunkIdx[ndim - 1] * chunkSh[ndim - 1]);

            // Dest offset in output buffer
            std::size_t dstOff = 0;
            for (std::size_t d = 0; d < ndim - 1; ++d) {
                dstOff += (coords[d] - offset[d]) * outStrides[d];
            }
            dstOff += (start[ndim - 1] - offset[ndim - 1]);

            std::size_t count = end[ndim - 1] - start[ndim - 1];
            std::memcpy(&out[dstOff], &chunkBuf[srcOff], count * sizeof(T));
        }

        // Advance chunkIdx
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++chunkIdx[d] <= maxC[d]) break;
            chunkIdx[d] = minC[d];
        }
    }
}

template <typename T>
void writeSubarrayND(
    const Dataset& ds,
    const T* data,
    const ShapeType& dataShape,
    const ShapeType& offset)
{
    const auto& chunkSh = ds.chunkShape();
    const std::size_t ndim = dataShape.size();

    if (ndim != chunkSh.size()) {
        throw std::runtime_error("writeSubarrayND: ndim mismatch");
    }

    std::vector<std::size_t> minC(ndim), maxC(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        minC[d] = offset[d] / chunkSh[d];
        maxC[d] = (offset[d] + dataShape[d] - 1) / chunkSh[d];
    }

    std::vector<std::size_t> dataStrides(ndim);
    dataStrides[ndim - 1] = 1;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
        dataStrides[d] = dataStrides[d + 1] * dataShape[d + 1];
    }

    std::vector<std::size_t> chunkStrides(ndim);
    chunkStrides[ndim - 1] = 1;
    for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
        chunkStrides[d] = chunkStrides[d + 1] * chunkSh[d + 1];
    }

    const std::size_t chunkElems = ds.chunkSize();
    std::vector<T> chunkBuf(chunkElems);
    auto& dsRef = const_cast<Dataset&>(ds);

    std::size_t totalChunks = 1;
    for (std::size_t d = 0; d < ndim; ++d) {
        totalChunks *= (maxC[d] - minC[d] + 1);
    }

    ShapeType chunkIdx(ndim);
    for (std::size_t d = 0; d < ndim; ++d) chunkIdx[d] = minC[d];

    for (std::size_t ci = 0; ci < totalChunks; ++ci) {
        std::vector<std::size_t> start(ndim), end(ndim);
        bool fullChunk = true;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::size_t chunkO = chunkIdx[d] * chunkSh[d];
            start[d] = std::max(offset[d], chunkO);
            end[d] = std::min(offset[d] + dataShape[d], chunkO + chunkSh[d]);
            if (start[d] != chunkO || end[d] != chunkO + chunkSh[d])
                fullChunk = false;
        }

        if (!fullChunk) {
            ds.readChunk(chunkIdx, chunkBuf.data());
        } else {
            std::memset(chunkBuf.data(), 0, chunkElems * sizeof(T));
        }

        std::size_t numRows = 1;
        for (std::size_t d = 0; d < ndim - 1; ++d) {
            numRows *= (end[d] - start[d]);
        }

        std::vector<std::size_t> coords(ndim);
        for (std::size_t row = 0; row < numRows; ++row) {
            std::size_t rem = row;
            for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
                std::size_t span = end[d] - start[d];
                coords[d] = start[d] + (rem % span);
                rem /= span;
            }

            std::size_t dstOff = 0;
            for (std::size_t d = 0; d < ndim - 1; ++d) {
                dstOff += (coords[d] - chunkIdx[d] * chunkSh[d]) * chunkStrides[d];
            }
            dstOff += (start[ndim - 1] - chunkIdx[ndim - 1] * chunkSh[ndim - 1]);

            std::size_t srcOff = 0;
            for (std::size_t d = 0; d < ndim - 1; ++d) {
                srcOff += (coords[d] - offset[d]) * dataStrides[d];
            }
            srcOff += (start[ndim - 1] - offset[ndim - 1]);

            std::size_t count = end[ndim - 1] - start[ndim - 1];
            std::memcpy(&chunkBuf[dstOff], &data[srcOff], count * sizeof(T));
        }

        dsRef.writeChunk(chunkIdx, chunkBuf.data());

        // Advance chunkIdx
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (++chunkIdx[d] <= maxC[d]) break;
            chunkIdx[d] = minC[d];
        }
    }
}

// ============================================================================
// Consolidated metadata
// ============================================================================

static void collectMetadataFiles(
    const fs::path& root,
    const fs::path& current,
    nlohmann::json& metadata)
{
    for (auto& entry : fs::directory_iterator(current)) {
        if (entry.is_regular_file()) {
            std::string name = entry.path().filename().string();
            if (name == ".zarray" || name == ".zattrs" || name == ".zgroup") {
                std::string relPath =
                    fs::relative(entry.path(), root).string();
                std::ifstream f(entry.path());
                if (f.is_open()) {
                    metadata[relPath] = nlohmann::json::parse(f);
                }
            }
        } else if (entry.is_directory()) {
            std::string dirName = entry.path().filename().string();
            // Skip the v3 chunk directory and hidden dirs
            if (dirName == "c" || (!dirName.empty() && dirName[0] == '.'))
                continue;
            // Only recurse into directories that contain metadata files
            // (groups or arrays), not bare chunk directories
            if (fs::exists(entry.path() / ".zarray") ||
                fs::exists(entry.path() / ".zattrs") ||
                fs::exists(entry.path() / ".zgroup")) {
                collectMetadataFiles(root, entry.path(), metadata);
            }
        }
    }
}

void writeConsolidatedMetadata(const fs::path& storePath)
{
    // Check if this is a v3 store (has root zarr.json with node_type=group)
    auto rootZarr = storePath / "zarr.json";
    if (fs::exists(rootZarr)) {
        std::ifstream f(rootZarr);
        auto j = nlohmann::json::parse(f);
        f.close();

        if (j.value("node_type", "") == "group") {
            // v3: add consolidated_metadata to root zarr.json
            nlohmann::json consolidated;
            consolidated["metadata"] = nlohmann::json::object();

            // Walk subdirectories for zarr.json files
            for (auto& entry : fs::recursive_directory_iterator(storePath)) {
                if (!entry.is_regular_file()) continue;
                if (entry.path().filename() != "zarr.json") continue;
                if (entry.path() == rootZarr) continue;

                std::string relPath =
                    fs::relative(entry.path().parent_path(), storePath).string();
                std::ifstream mf(entry.path());
                if (mf.is_open()) {
                    consolidated["metadata"][relPath] = nlohmann::json::parse(mf);
                }
            }

            j["consolidated_metadata"] = consolidated;
            std::ofstream out(rootZarr);
            out << j.dump(4) << "\n";
            return;
        }
    }

    // v2: write .zmetadata
    nlohmann::json consolidated;
    consolidated["zarr_consolidated_format"] = 1;
    consolidated["metadata"] = nlohmann::json::object();

    collectMetadataFiles(storePath, storePath, consolidated["metadata"]);

    std::ofstream out(storePath / ".zmetadata");
    out << consolidated.dump(4) << "\n";
}

nlohmann::json readConsolidatedMetadata(const fs::path& storePath)
{
    // Try v2 first
    auto zmetaPath = storePath / ".zmetadata";
    if (fs::exists(zmetaPath)) {
        std::ifstream f(zmetaPath);
        return nlohmann::json::parse(f);
    }

    // Try v3
    auto rootZarr = storePath / "zarr.json";
    if (fs::exists(rootZarr)) {
        std::ifstream f(rootZarr);
        auto j = nlohmann::json::parse(f);
        if (j.contains("consolidated_metadata")) {
            return j["consolidated_metadata"];
        }
    }

    return nlohmann::json::object();
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template void readSubarray<uint8_t>(const Dataset&, Array3D<uint8_t>&, const ShapeType&);
template void readSubarray<uint16_t>(const Dataset&, Array3D<uint16_t>&, const ShapeType&);
template void readSubarray<float>(const Dataset&, Array3D<float>&, const ShapeType&);

template void writeSubarray<uint8_t>(const Dataset&, const Array3D<uint8_t>&, const ShapeType&);
template void writeSubarray<uint16_t>(const Dataset&, const Array3D<uint16_t>&, const ShapeType&);
template void writeSubarray<float>(const Dataset&, const Array3D<float>&, const ShapeType&);

template void writeSubarray<uint8_t>(const Dataset&, const uint8_t*, const ShapeType&, const ShapeType&);
template void writeSubarray<uint16_t>(const Dataset&, const uint16_t*, const ShapeType&, const ShapeType&);
template void writeSubarray<float>(const Dataset&, const float*, const ShapeType&, const ShapeType&);

template void readSubarrayND<uint8_t>(const Dataset&, uint8_t*, const ShapeType&, const ShapeType&);
template void readSubarrayND<uint16_t>(const Dataset&, uint16_t*, const ShapeType&, const ShapeType&);
template void readSubarrayND<float>(const Dataset&, float*, const ShapeType&, const ShapeType&);

template void writeSubarrayND<uint8_t>(const Dataset&, const uint8_t*, const ShapeType&, const ShapeType&);
template void writeSubarrayND<uint16_t>(const Dataset&, const uint16_t*, const ShapeType&, const ShapeType&);
template void writeSubarrayND<float>(const Dataset&, const float*, const ShapeType&, const ShapeType&);

}  // namespace vc::zarr
