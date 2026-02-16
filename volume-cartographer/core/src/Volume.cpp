#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/Z5Decompressor.hpp"
#include "vc/core/cache/DiskStore.hpp"

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/filesystem/metadata.hxx"
#include "z5/multiarray/xtensor_access.hxx"

static const std::filesystem::path METADATA_FILE = "meta.json";
static const std::filesystem::path METADATA_FILE_ALT = "metadata.json";

Volume::Volume(std::filesystem::path path) : path_(std::move(path))
{
    loadMetadata();

    _width = metadata_["width"].get<int>();
    _height = metadata_["height"].get<int>();
    _slices = metadata_["slices"].get<int>();

    std::vector<std::mutex> init_mutexes(_slices);


    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(std::filesystem::path path, std::string uuid, std::string name)
    : path_(std::move(path))
{
    metadata_["uuid"] = uuid;
    metadata_["name"] = name;
    metadata_["type"] = "vol";
    metadata_["width"] = _width;
    metadata_["height"] = _height;
    metadata_["slices"] = _slices;
    metadata_["voxelsize"] = double{};
    metadata_["min"] = double{};
    metadata_["max"] = double{};

    zarrOpen();
}

Volume::~Volume() = default;

void Volume::loadMetadata()
{
    auto metaPath = path_ / METADATA_FILE;
    if (std::filesystem::exists(metaPath)) {
        metadata_ = vc::json::load_json_file(metaPath);
    } else {
        auto altPath = path_ / METADATA_FILE_ALT;
        auto full = vc::json::load_json_file(altPath);
        if (!full.contains("scan")) {
            throw std::runtime_error(
                "metadata.json missing 'scan' key: " + altPath.string());
        }
        metadata_ = full["scan"];
        if (!metadata_.contains("format")) {
            metadata_["format"] = "zarr";
        }
        metaPath = altPath;
    }
    vc::json::require_type(metadata_, "type", "vol", metaPath.string());
    vc::json::require_fields(metadata_, {"uuid", "width", "height", "slices"}, metaPath.string());
}

std::string Volume::id() const
{
    return metadata_["uuid"].get<std::string>();
}

std::string Volume::name() const
{
    return metadata_["name"].get<std::string>();
}

void Volume::setName(const std::string& n)
{
    metadata_["name"] = n;
}

void Volume::saveMetadata()
{
    auto metaPath = path_ / METADATA_FILE;
    std::ofstream jsonFile(metaPath.string(), std::ofstream::out);
    jsonFile << metadata_ << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

bool Volume::checkDir(std::filesystem::path path)
{
    return std::filesystem::is_directory(path) &&
           (std::filesystem::exists(path / METADATA_FILE) ||
            std::filesystem::exists(path / METADATA_FILE_ALT));
}

void Volume::zarrOpen()
{
    if (!metadata_.contains("format") || metadata_["format"].get<std::string>() != "zarr")
        return;

    zarrFile_ = std::make_unique<z5::filesystem::handle::File>(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);

    std::vector<std::string> groups;
    zarrFile_->keys(groups);
    std::sort(groups.begin(), groups.end());

    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(auto name : groups) {
        // Read metadata first to discover the dimension separator
        z5::filesystem::handle::Dataset tmp_handle(path_ / name, z5::FileMode::FileMode::r);
        z5::DatasetMetadata dsMeta;
        z5::filesystem::readMetadata(tmp_handle, dsMeta);

        // Re-create handle with correct delimiter so chunk keys resolve properly
        z5::filesystem::handle::Dataset ds_handle(group, name, dsMeta.zarrDelimiter);

        zarrDs_.push_back(z5::filesystem::openDataset(ds_handle));
        if (zarrDs_.back()->getDtype() != z5::types::Datatype::uint8 && zarrDs_.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);

        // Verify level 0 shape matches meta.json dimensions
        // zarr shape is [z, y, x] = [slices, height, width]
        if (zarrDs_.size() == 1 && !skipShapeCheck) {
            const auto& shape = zarrDs_[0]->shape();
            if (static_cast<int>(shape[0]) != _slices ||
                static_cast<int>(shape[1]) != _height ||
                static_cast<int>(shape[2]) != _width) {
                throw std::runtime_error(
                    "zarr level 0 shape [z,y,x]=(" + std::to_string(shape[0]) + ", " +
                    std::to_string(shape[1]) + ", " + std::to_string(shape[2]) +
                    ") does not match meta.json dimensions (slices=" + std::to_string(_slices) +
                    ", height=" + std::to_string(_height) + ", width=" + std::to_string(_width) +
                    ") in " + path_.string());
            }
        }
    }
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path)
{
    return std::make_shared<Volume>(path);
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path, std::string uuid, std::string name)
{
    return std::make_shared<Volume>(path, uuid, name);
}

int Volume::sliceWidth() const { return _width; }
int Volume::sliceHeight() const { return _height; }
int Volume::numSlices() const { return _slices; }
std::array<int, 3> Volume::shape() const { return {_width, _height, _slices}; }
double Volume::voxelSize() const
{
    return metadata_["voxelsize"].get<double>();
}

z5::Dataset *Volume::zarrDataset(int level) const {
    if (level >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales() const {
    return zarrDs_.size();
}

std::unique_ptr<vc::cache::TieredChunkCache> Volume::createTieredCache(
    std::shared_ptr<vc::cache::DiskStore> diskStore) const
{
    if (zarrDs_.empty()) return nullptr;

    // Build level metadata from our zarr datasets
    std::vector<vc::cache::FileSystemChunkSource::LevelMeta> levels;
    levels.reserve(zarrDs_.size());
    for (auto& ds : zarrDs_) {
        const auto& shape = ds->shape();
        const auto& chunks = ds->defaultChunkShape();
        vc::cache::FileSystemChunkSource::LevelMeta lm;
        lm.shape = {
            static_cast<int>(shape[0]),
            static_cast<int>(shape[1]),
            static_cast<int>(shape[2])};
        lm.chunkShape = {
            static_cast<int>(chunks[0]),
            static_cast<int>(chunks[1]),
            static_cast<int>(chunks[2])};
        levels.push_back(lm);
    }

    // Detect delimiter from the first dataset's chunk path
    std::string delimiter = ".";
    if (zarrDs_[0]->isZarr()) {
        z5::types::ShapeType probe = {0, 0, 1};
        std::filesystem::path p;
        zarrDs_[0]->chunkPath(probe, p);
        std::string basePath = zarrDs_[0]->path().string();
        std::string suffix = p.string().substr(basePath.size() + 1);
        size_t pos = suffix.find_first_not_of("0123456789");
        if (pos != std::string::npos) {
            size_t end = suffix.find_first_of("0123456789", pos);
            delimiter = suffix.substr(pos, end - pos);
        }
    }

    auto source = std::make_unique<vc::cache::FileSystemChunkSource>(
        path_, delimiter, std::move(levels));

    // Build dataset pointers for the decompressor
    std::vector<z5::Dataset*> dsPtrs;
    dsPtrs.reserve(zarrDs_.size());
    for (auto& ds : zarrDs_) {
        dsPtrs.push_back(ds.get());
    }
    auto decompress = vc::cache::makeZ5Decompressor(dsPtrs);

    vc::cache::TieredChunkCache::Config config;
    config.volumeId = id();

    return std::make_unique<vc::cache::TieredChunkCache>(
        std::move(config),
        std::move(source),
        std::move(decompress),
        std::move(diskStore));
}

std::function<int(const z5::Dataset*)> Volume::datasetLevelMapper() const
{
    // Build a map from dataset pointer to level index
    std::unordered_map<const z5::Dataset*, int> dsToLevel;
    for (size_t i = 0; i < zarrDs_.size(); i++) {
        dsToLevel[zarrDs_[i].get()] = static_cast<int>(i);
    }
    return [dsToLevel = std::move(dsToLevel)](const z5::Dataset* ds) -> int {
        auto it = dsToLevel.find(ds);
        return (it != dsToLevel.end()) ? it->second : -1;
    };
}

// ============================================================================
// Cache management
// ============================================================================

ChunkCache<uint8_t>& Volume::cache()
{
    if (!cache8_) {
        cache8_ = std::make_unique<ChunkCache<uint8_t>>(cacheMaxBytes_);
        if (tieredCache_)
            cache8_->setTieredBackend(tieredCache_.get(), datasetLevelMapper());
    }
    return *cache8_;
}

ChunkCache<uint16_t>& Volume::cache16()
{
    if (!cache16_) {
        cache16_ = std::make_unique<ChunkCache<uint16_t>>(cacheMaxBytes_);
        if (tieredCache_)
            cache16_->setTieredBackend(tieredCache_.get(), datasetLevelMapper());
    }
    return *cache16_;
}

void Volume::setCacheSize(size_t maxBytes)
{
    cacheMaxBytes_ = maxBytes;
    if (cache8_) cache8_->setMaxBytes(maxBytes);
    if (cache16_) cache16_->setMaxBytes(maxBytes);
}

void Volume::enableTieredCache(std::shared_ptr<vc::cache::DiskStore> diskStore)
{
    fprintf(stderr, "[TILED] enableTieredCache: path=%s numScales=%zu\n",
            path_.string().c_str(), zarrDs_.size());
    tieredCache_ = createTieredCache(std::move(diskStore));
    fprintf(stderr, "[TILED] enableTieredCache: tieredCache=%p numLevels=%d\n",
            (void*)tieredCache_.get(),
            tieredCache_ ? tieredCache_->numLevels() : -1);
    // Rewire existing caches if they exist
    if (tieredCache_) {
        auto mapper = datasetLevelMapper();
        if (cache8_)  cache8_->setTieredBackend(tieredCache_.get(), mapper);
        if (cache16_) cache16_->setTieredBackend(tieredCache_.get(), mapper);
    }
}

vc::cache::TieredChunkCache* Volume::tieredCache() const
{
    return tieredCache_.get();
}

// ============================================================================
// Sampling
// ============================================================================

void Volume::sample(cv::Mat_<uint8_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    int level, vc::Sampling method)
{
    z5::Dataset* ds = zarrDataset(level);
    if (!ds) return;
    float scale = 1.0f / std::pow(2.0f, level);
    cv::Mat_<cv::Vec3f> scaled;
    if (level > 0) {
        scaled = coords * scale;
    } else {
        scaled = coords;
    }
    readInterpolated3D(out, ds, scaled, &cache(), method);
}

void Volume::sampleComposite(cv::Mat_<uint8_t>& out,
                              const cv::Mat_<cv::Vec3f>& baseCoords,
                              const cv::Mat_<cv::Vec3f>& normals,
                              int zStart, int zEnd,
                              const CompositeParams& params,
                              int level)
{
    z5::Dataset* ds = zarrDataset(level);
    if (!ds) return;
    float scale = 1.0f / std::pow(2.0f, level);
    cv::Mat_<cv::Vec3f> scaled;
    if (level > 0) {
        scaled = baseCoords * scale;
    } else {
        scaled = baseCoords;
    }
    readCompositeFast(out, ds, scaled, normals, scale,
                      zStart, zEnd, params, cache());
}

int Volume::sampleBestEffort(cv::Mat_<uint8_t>& out,
                              const cv::Mat_<cv::Vec3f>& coords,
                              int level, vc::Sampling method)
{
    const int nScales = static_cast<int>(numScales());

    // Try from requested level to coarser
    for (int lvl = level; lvl < nScales; lvl++) {
        if (allChunksCached(coords, lvl)) {
            float scale = 1.0f / std::pow(2.0f, lvl);
            cv::Mat_<cv::Vec3f> scaled = (lvl > 0) ? cv::Mat_<cv::Vec3f>(coords * scale) : coords;
            readInterpolated3D(out, zarrDataset(lvl), scaled, &cache(), method);
            // If we fell back to a coarser level, prefetch the requested level
            if (lvl > level) {
                prefetchChunks(coords, level);
            }
            return lvl;
        }
    }

    // Coarsest level: block if needed (should be pinned hot)
    int last = nScales - 1;
    if (last < 0) last = 0;
    static std::once_flag allMissWarn;
    std::call_once(allMissWarn, [last]() {
        fprintf(stderr, "[TILED] sampleBestEffort: ALL LEVELS MISS, blocking at coarsest=%d\n", last);
    });
    float scale = 1.0f / std::pow(2.0f, last);
    cv::Mat_<cv::Vec3f> scaled = (last > 0) ? cv::Mat_<cv::Vec3f>(coords * scale) : coords;
    readInterpolated3D(out, zarrDataset(last), scaled, &cache(), method);
    // Prefetch the requested level in background
    if (last > level) {
        prefetchChunks(coords, level);
    }
    return last;
}

// ============================================================================
// Dimensioned read API (readNd)
// ============================================================================

static cv::Mat_<cv::Vec3f> makeCoordGrid(
    cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV, int w, int h)
{
    cv::Mat_<cv::Vec3f> coords(h, w);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++)
            coords(r, c) = origin + axisU * static_cast<float>(c)
                                  + axisV * static_cast<float>(r);
    return coords;
}

uint8_t Volume::read0d(cv::Vec3f point, int level, vc::Sampling method)
{
    cv::Mat_<cv::Vec3f> coords(1, 1);
    coords(0, 0) = point;
    cv::Mat_<uint8_t> out;
    read2d(out, coords, level, method);
    return out.empty() ? 0 : out(0, 0);
}

void Volume::read1d(cv::Mat_<uint8_t>& out,
                    cv::Vec3f origin, cv::Vec3f step, int count,
                    int level, vc::Sampling method)
{
    cv::Mat_<cv::Vec3f> coords(1, count);
    for (int i = 0; i < count; i++)
        coords(0, i) = origin + step * static_cast<float>(i);
    read2d(out, coords, level, method);
}

void Volume::read2d(cv::Mat_<uint8_t>& out,
                    cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                    int w, int h, int level, vc::Sampling method)
{
    auto coords = makeCoordGrid(origin, axisU, axisV, w, h);
    read2d(out, coords, level, method);
}

int Volume::read2dBestEffort(cv::Mat_<uint8_t>& out,
                             cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                             int w, int h, int level, vc::Sampling method)
{
    auto coords = makeCoordGrid(origin, axisU, axisV, w, h);
    return read2dBestEffort(out, coords, level, method);
}

void Volume::read2d(cv::Mat_<uint8_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    int level, vc::Sampling method)
{
    sample(out, coords, level, method);
}

int Volume::read2dBestEffort(cv::Mat_<uint8_t>& out,
                              const cv::Mat_<cv::Vec3f>& coords,
                              int level, vc::Sampling method)
{
    return sampleBestEffort(out, coords, level, method);
}

void Volume::read3d(cv::Mat_<uint8_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    const cv::Mat_<cv::Vec3f>& normals,
                    int zStart, int zEnd, const CompositeParams& params,
                    int level)
{
    sampleComposite(out, coords, normals, zStart, zEnd, params, level);
}

int Volume::read3dBestEffort(cv::Mat_<uint8_t>& out,
                              const cv::Mat_<cv::Vec3f>& coords,
                              const cv::Mat_<cv::Vec3f>& normals,
                              int zStart, int zEnd,
                              const CompositeParams& params, int level)
{
    const int nScales = static_cast<int>(numScales());

    for (int lvl = level; lvl < nScales; lvl++) {
        if (allCompositeChunksCached(coords, normals, zStart, zEnd, lvl)) {
            read3d(out, coords, normals, zStart, zEnd, params, lvl);
            if (lvl > level)
                prefetchCompositeChunks(coords, normals, zStart, zEnd, level);
            return lvl;
        }
    }

    // Coarsest level — block
    int last = std::max(0, nScales - 1);
    read3d(out, coords, normals, zStart, zEnd, params, last);
    if (last > level)
        prefetchCompositeChunks(coords, normals, zStart, zEnd, level);
    return last;
}

void Volume::read3d(cv::Mat_<uint8_t>& out,
                    cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                    cv::Vec3f normal, int w, int h,
                    int zStart, int zEnd, const CompositeParams& params,
                    int level)
{
    auto coords = makeCoordGrid(origin, axisU, axisV, w, h);
    cv::Mat_<cv::Vec3f> normals(h, w, normal);
    read3d(out, coords, normals, zStart, zEnd, params, level);
}

int Volume::read3dBestEffort(cv::Mat_<uint8_t>& out,
                              cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                              cv::Vec3f normal, int w, int h,
                              int zStart, int zEnd, const CompositeParams& params,
                              int level)
{
    auto coords = makeCoordGrid(origin, axisU, axisV, w, h);
    cv::Mat_<cv::Vec3f> normals(h, w, normal);
    return read3dBestEffort(out, coords, normals, zStart, zEnd, params, level);
}

// ============================================================================
// Composite-aware chunk helpers
// ============================================================================

bool Volume::allCompositeChunksCached(
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    int zStart, int zEnd, int level) const
{
    z5::Dataset* ds = zarrDataset(level);
    if (!ds || coords.empty()) return false;

    float scale = 1.0f / std::pow(2.0f, level);
    auto cs = ds->defaultChunkShape();  // {cz, cy, cx}
    const auto& shape = ds->shape();    // {z, y, x}

    // Compute bounding box of all sampling points including normal offsets
    float loX = std::numeric_limits<float>::max();
    float loY = std::numeric_limits<float>::max();
    float loZ = std::numeric_limits<float>::max();
    float hiX = std::numeric_limits<float>::lowest();
    float hiY = std::numeric_limits<float>::lowest();
    float hiZ = std::numeric_limits<float>::lowest();

    const bool hasNormals = !normals.empty() && normals.size() == coords.size();

    for (int r = 0; r < coords.rows; r++) {
        for (int c = 0; c < coords.cols; c++) {
            const auto& v = coords(r, c);
            float sx = v[0] * scale;
            float sy = v[1] * scale;
            float sz = v[2] * scale;

            cv::Vec3f n = hasNormals ? normals(r, c) : cv::Vec3f(1, 0, 0);
            // Normal offsets are scaled by zStep (which is `scale` itself)
            for (int z : {zStart, zEnd}) {
                float off = z * scale;
                float px = sx + n[0] * off;
                float py = sy + n[1] * off;
                float pz = sz + n[2] * off;
                loX = std::min(loX, px); hiX = std::max(hiX, px);
                loY = std::min(loY, py); hiY = std::max(hiY, py);
                loZ = std::min(loZ, pz); hiZ = std::max(hiZ, pz);
            }
        }
    }

    // Add interpolation margin
    loX -= 1.0f; loY -= 1.0f; loZ -= 1.0f;
    hiX += 1.0f; hiY += 1.0f; hiZ += 1.0f;

    int minIx = std::max(0, static_cast<int>(std::floor(loX / cs[2])));
    int maxIx = std::min(static_cast<int>(std::floor(hiX / cs[2])),
                         static_cast<int>((shape[2] - 1) / cs[2]));
    int minIy = std::max(0, static_cast<int>(std::floor(loY / cs[1])));
    int maxIy = std::min(static_cast<int>(std::floor(hiY / cs[1])),
                         static_cast<int>((shape[1] - 1) / cs[1]));
    int minIz = std::max(0, static_cast<int>(std::floor(loZ / cs[0])));
    int maxIz = std::min(static_cast<int>(std::floor(hiZ / cs[0])),
                         static_cast<int>((shape[0] - 1) / cs[0]));

    if (minIx > maxIx || minIy > maxIy || minIz > maxIz) return false;

    // Prefer tiered cache: non-blocking get() checks hot + warm tiers
    if (tieredCache_) {
        for (int iz = minIz; iz <= maxIz; iz++)
            for (int iy = minIy; iy <= maxIy; iy++)
                for (int ix = minIx; ix <= maxIx; ix++)
                    if (!tieredCache_->get(vc::cache::ChunkKey{level, iz, iy, ix}))
                        return false;
        return true;
    }

    // Fallback: check ChunkCache local map
    if (!cache8_) return false;
    for (int iz = minIz; iz <= maxIz; iz++)
        for (int iy = minIy; iy <= maxIy; iy++)
            for (int ix = minIx; ix <= maxIx; ix++)
                if (!cache8_->getIfCached(ds, iz, iy, ix))
                    return false;

    return true;
}

void Volume::prefetchCompositeChunks(
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    int zStart, int zEnd, int level)
{
    z5::Dataset* ds = zarrDataset(level);
    if (!ds || coords.empty()) return;

    float scale = 1.0f / std::pow(2.0f, level);
    auto cs = ds->defaultChunkShape();
    const auto& shape = ds->shape();

    float loX = std::numeric_limits<float>::max();
    float loY = std::numeric_limits<float>::max();
    float loZ = std::numeric_limits<float>::max();
    float hiX = std::numeric_limits<float>::lowest();
    float hiY = std::numeric_limits<float>::lowest();
    float hiZ = std::numeric_limits<float>::lowest();

    const bool hasNormals = !normals.empty() && normals.size() == coords.size();

    for (int r = 0; r < coords.rows; r++) {
        for (int c = 0; c < coords.cols; c++) {
            const auto& v = coords(r, c);
            float sx = v[0] * scale;
            float sy = v[1] * scale;
            float sz = v[2] * scale;

            cv::Vec3f n = hasNormals ? normals(r, c) : cv::Vec3f(1, 0, 0);
            for (int z : {zStart, zEnd}) {
                float off = z * scale;
                float px = sx + n[0] * off;
                float py = sy + n[1] * off;
                float pz = sz + n[2] * off;
                loX = std::min(loX, px); hiX = std::max(hiX, px);
                loY = std::min(loY, py); hiY = std::max(hiY, py);
                loZ = std::min(loZ, pz); hiZ = std::max(hiZ, pz);
            }
        }
    }

    loX -= 1.0f; loY -= 1.0f; loZ -= 1.0f;
    hiX += 1.0f; hiY += 1.0f; hiZ += 1.0f;

    int minIx = std::max(0, static_cast<int>(std::floor(loX / cs[2])));
    int maxIx = std::min(static_cast<int>(std::floor(hiX / cs[2])),
                         static_cast<int>((shape[2] - 1) / cs[2]));
    int minIy = std::max(0, static_cast<int>(std::floor(loY / cs[1])));
    int maxIy = std::min(static_cast<int>(std::floor(hiY / cs[1])),
                         static_cast<int>((shape[1] - 1) / cs[1]));
    int minIz = std::max(0, static_cast<int>(std::floor(loZ / cs[0])));
    int maxIz = std::min(static_cast<int>(std::floor(hiZ / cs[0])),
                         static_cast<int>((shape[0] - 1) / cs[0]));

    if (minIx > maxIx || minIy > maxIy || minIz > maxIz) return;

    if (tieredCache_) {
        tieredCache_->prefetchRegion(level, minIz, minIy, minIx, maxIz, maxIy, maxIx);
    } else {
        cache().prefetch(ds, minIz, minIy, minIx, maxIz, maxIy, maxIx);
    }
}

void Volume::pinCoarsestLevel(bool blocking)
{
    if (!tieredCache_ || zarrDs_.empty()) {
        fprintf(stderr, "[TILED] pinCoarsestLevel: SKIP tieredCache=%p zarrDs=%zu\n",
                (void*)tieredCache_.get(), zarrDs_.size());
        return;
    }
    int last = static_cast<int>(zarrDs_.size()) - 1;
    z5::Dataset* ds = zarrDataset(last);
    if (!ds) {
        fprintf(stderr, "[TILED] pinCoarsestLevel: no dataset at level %d\n", last);
        return;
    }
    const auto& shape = ds->shape();         // {z, y, x}
    const auto& chunks = ds->defaultChunkShape(); // {cz, cy, cx}
    std::array<int, 3> gridDims = {
        static_cast<int>((shape[0] + chunks[0] - 1) / chunks[0]),
        static_cast<int>((shape[1] + chunks[1] - 1) / chunks[1]),
        static_cast<int>((shape[2] + chunks[2] - 1) / chunks[2])
    };
    fprintf(stderr, "[TILED] pinCoarsestLevel: level=%d shape=(%zu,%zu,%zu) chunks=(%zu,%zu,%zu) grid=(%d,%d,%d) blocking=%d\n",
            last, shape[0], shape[1], shape[2], chunks[0], chunks[1], chunks[2],
            gridDims[0], gridDims[1], gridDims[2], blocking);
    tieredCache_->pinLevel(last, gridDims, blocking);
    fprintf(stderr, "[TILED] pinCoarsestLevel: done\n");
}

// ============================================================================
// Chunk query / prefetch
// ============================================================================

Volume::ChunkBBox Volume::coordsToChunkBBox(
    const cv::Mat_<cv::Vec3f>& coords, int level) const
{
    z5::Dataset* ds = zarrDataset(level);
    if (!ds || coords.empty()) {
        return {0, -1, 0, -1, 0, -1};  // invalid bbox
    }

    float scale = 1.0f / std::pow(2.0f, level);
    auto cs = ds->defaultChunkShape();  // {cz, cy, cx}

    float loX = std::numeric_limits<float>::max();
    float loY = std::numeric_limits<float>::max();
    float loZ = std::numeric_limits<float>::max();
    float hiX = std::numeric_limits<float>::lowest();
    float hiY = std::numeric_limits<float>::lowest();
    float hiZ = std::numeric_limits<float>::lowest();

    for (auto it = coords.begin(); it != coords.end(); ++it) {
        const auto& v = *it;
        float sx = v[0] * scale;
        float sy = v[1] * scale;
        float sz = v[2] * scale;
        loX = std::min(loX, sx); hiX = std::max(hiX, sx);
        loY = std::min(loY, sy); hiY = std::max(hiY, sy);
        loZ = std::min(loZ, sz); hiZ = std::max(hiZ, sz);
    }

    // Add 1-voxel margin for interpolation
    loX -= 1.0f; loY -= 1.0f; loZ -= 1.0f;
    hiX += 1.0f; hiY += 1.0f; hiZ += 1.0f;

    const auto& shape = ds->shape();  // {z, y, x}
    ChunkBBox bb;
    bb.minIx = std::max(0, static_cast<int>(std::floor(loX / cs[2])));
    bb.maxIx = std::min(static_cast<int>(std::floor(hiX / cs[2])),
                        static_cast<int>((shape[2] - 1) / cs[2]));
    bb.minIy = std::max(0, static_cast<int>(std::floor(loY / cs[1])));
    bb.maxIy = std::min(static_cast<int>(std::floor(hiY / cs[1])),
                        static_cast<int>((shape[1] - 1) / cs[1]));
    bb.minIz = std::max(0, static_cast<int>(std::floor(loZ / cs[0])));
    bb.maxIz = std::min(static_cast<int>(std::floor(hiZ / cs[0])),
                        static_cast<int>((shape[0] - 1) / cs[0]));
    return bb;
}

bool Volume::allChunksCached(const cv::Mat_<cv::Vec3f>& coords, int level) const
{
    z5::Dataset* ds = zarrDataset(level);
    if (!ds) return false;

    auto bb = coordsToChunkBBox(coords, level);
    if (bb.minIx > bb.maxIx) return false;  // invalid bbox

    int totalChunks = (bb.maxIz - bb.minIz + 1) * (bb.maxIy - bb.minIy + 1) * (bb.maxIx - bb.minIx + 1);

    // Prefer tiered cache: its non-blocking get() checks hot + warm tiers
    // and returns immediately (nullptr on miss). Prefetched chunks land in
    // hot tier, so this correctly detects them.
    if (tieredCache_) {
        for (int iz = bb.minIz; iz <= bb.maxIz; iz++)
            for (int iy = bb.minIy; iy <= bb.maxIy; iy++)
                for (int ix = bb.minIx; ix <= bb.maxIx; ix++)
                    if (!tieredCache_->get(vc::cache::ChunkKey{level, iz, iy, ix}))
                        return false;
        return true;
    }

    // Fallback: check ChunkCache local map
    if (!cache8_) return false;
    for (int iz = bb.minIz; iz <= bb.maxIz; iz++)
        for (int iy = bb.minIy; iy <= bb.maxIy; iy++)
            for (int ix = bb.minIx; ix <= bb.maxIx; ix++)
                if (!cache8_->getIfCached(ds, iz, iy, ix))
                    return false;
    return true;
}

void Volume::prefetchChunks(const cv::Mat_<cv::Vec3f>& coords, int level)
{
    auto bb = coordsToChunkBBox(coords, level);
    if (bb.minIx > bb.maxIx) return;  // invalid bbox

    // Prefer tiered cache (non-blocking async prefetch)
    if (tieredCache_) {
        tieredCache_->prefetchRegion(level,
            bb.minIz, bb.minIy, bb.minIx,
            bb.maxIz, bb.maxIy, bb.maxIx);
    } else {
        // Fallback: ChunkCache::prefetch (blocking, OMP parallel)
        z5::Dataset* ds = zarrDataset(level);
        if (ds) {
            cache().prefetch(ds,
                bb.minIz, bb.minIy, bb.minIx,
                bb.maxIz, bb.maxIy, bb.maxIx);
        }
    }
}

void Volume::cancelPendingPrefetch()
{
    if (tieredCache_) {
        tieredCache_->cancelPendingPrefetch();
    }
}

void Volume::prefetchWorldBBox(const cv::Vec3f& lo, const cv::Vec3f& hi, int level)
{
    if (!tieredCache_) return;

    z5::Dataset* ds = zarrDataset(level);
    if (!ds) return;

    float scale = 1.0f / std::pow(2.0f, level);
    auto cs = ds->defaultChunkShape();  // {cz, cy, cx}
    const auto& shape = ds->shape();    // {z, y, x}

    // Scale world coords to level coords, add 1-voxel interpolation margin
    // Coords are (x, y, z) in Vec3f; chunks are (z, y, x)
    int minIx = std::max(0, static_cast<int>(std::floor(lo[0] * scale - 1) / static_cast<int>(cs[2])));
    int maxIx = std::min(static_cast<int>((shape[2] - 1) / cs[2]),
                         static_cast<int>(std::floor(hi[0] * scale + 1) / static_cast<int>(cs[2])));
    int minIy = std::max(0, static_cast<int>(std::floor(lo[1] * scale - 1) / static_cast<int>(cs[1])));
    int maxIy = std::min(static_cast<int>((shape[1] - 1) / cs[1]),
                         static_cast<int>(std::floor(hi[1] * scale + 1) / static_cast<int>(cs[1])));
    int minIz = std::max(0, static_cast<int>(std::floor(lo[2] * scale - 1) / static_cast<int>(cs[0])));
    int maxIz = std::min(static_cast<int>((shape[0] - 1) / cs[0]),
                         static_cast<int>(std::floor(hi[2] * scale + 1) / static_cast<int>(cs[0])));

    if (minIx > maxIx || minIy > maxIy || minIz > maxIz) return;

    tieredCache_->prefetchRegion(level, minIz, minIy, minIx, maxIz, maxIy, maxIx);
}
