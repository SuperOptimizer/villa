#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/cache/TieredChunkCache.hpp"
#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/VcDecompressor.hpp"
#include "vc/core/cache/DiskStore.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/util/RemoteUrl.hpp"
#include "vc/core/types/VcDataset.hpp"

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

    zarrGroup_ = vc::readZarrAttributes(path_);
    zarrDs_ = vc::openZarrLevels(path_);

    for (size_t i = 0; i < zarrDs_.size(); i++) {
        auto dtype = zarrDs_[i]->getDtype();
        if (dtype != vc::VcDtype::uint8 && dtype != vc::VcDtype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets, incompatible type found in " + path_.string());

        // Verify level 0 shape matches meta.json dimensions
        // zarr shape is [z, y, x] = [slices, height, width]
        if (i == 0 && !skipShapeCheck) {
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

std::shared_ptr<Volume> Volume::NewFromUrl(
    const std::string& url,
    const std::filesystem::path& cacheRoot)
{
    namespace fs = std::filesystem;

    // Resolve s3:// URLs to https:// and detect AWS credentials
    auto resolved = vc::resolveRemoteUrl(url);
    vc::cache::HttpAuth auth;
    auth.awsSigv4 = resolved.useAwsSigv4;
    auth.region = resolved.awsRegion;

    // Determine cache root
    fs::path root = cacheRoot;
    if (root.empty()) {
        root = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") / ".VC3D" / "remote_cache";
    }

    // Fetch remote metadata (downloads .zarray files, synthesizes meta.json)
    auto info = vc::cache::fetchRemoteZarrMetadata(resolved.httpsUrl, root, auth);

    // Temporarily skip shape validation (staging dir has no chunk data)
    auto prevSkip = skipShapeCheck;
    skipShapeCheck = true;

    auto vol = std::make_shared<Volume>(info.stagingDir);

    skipShapeCheck = prevSkip;

    vol->isRemote_ = true;
    vol->remoteUrl_ = info.url;
    vol->remoteDelimiter_ = info.delimiter;
    vol->remoteAuth_ = auth;

    return vol;
}

int Volume::sliceWidth() const { return _width; }
int Volume::sliceHeight() const { return _height; }
int Volume::numSlices() const { return _slices; }
std::array<int, 3> Volume::shape() const { return {_width, _height, _slices}; }
double Volume::voxelSize() const
{
    return metadata_["voxelsize"].get<double>();
}

vc::VcDataset *Volume::zarrDataset(int level) const {
    if (level >= static_cast<int>(zarrDs_.size()))
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

    // Get delimiter from VcDataset
    std::string delimiter = zarrDs_[0]->delimiter();

    // Create chunk source: HTTP for remote volumes, filesystem for local
    std::unique_ptr<vc::cache::ChunkSource> source;
    if (isRemote_) {
        source = std::make_unique<vc::cache::HttpChunkSource>(
            remoteUrl_, remoteDelimiter_, std::move(levels), remoteAuth_);

        // For remote volumes, use the staging dir itself as the disk store.
        if (!diskStore) {
            vc::cache::DiskStore::Config dsCfg;
            dsCfg.root = path_;
            dsCfg.directMode = true;
            dsCfg.delimiter = remoteDelimiter_;
            diskStore = std::make_shared<vc::cache::DiskStore>(std::move(dsCfg));
        }
    } else {
        source = std::make_unique<vc::cache::FileSystemChunkSource>(
            path_, delimiter, std::move(levels));
    }

    // Build dataset pointers for the decompressor
    std::vector<vc::VcDataset*> dsPtrs;
    dsPtrs.reserve(zarrDs_.size());
    for (auto& ds : zarrDs_) {
        dsPtrs.push_back(ds.get());
    }
    auto decompress = vc::cache::makeVcDecompressor(dsPtrs);

    vc::cache::TieredChunkCache::Config config;
    config.volumeId = id();
    config.hotMaxBytes = cacheBudgetHot_;
    config.warmMaxBytes = cacheBudgetWarm_;

    return std::make_unique<vc::cache::TieredChunkCache>(
        std::move(config),
        std::move(source),
        std::move(decompress),
        std::move(diskStore));
}

// ============================================================================
// Cache management
// ============================================================================

void Volume::ensureTieredCache() const
{
    if (!tieredCache_) {
        auto* self = const_cast<Volume*>(this);
        tieredCache_ = self->createTieredCache(self->pendingDiskStore_);
    }
}

vc::cache::TieredChunkCache* Volume::tieredCache()
{
    ensureTieredCache();
    return tieredCache_.get();
}

void Volume::setCacheBudget(size_t hotBytes, size_t warmBytes)
{
    cacheBudgetHot_ = hotBytes;
    cacheBudgetWarm_ = warmBytes;
}

void Volume::setDiskStore(std::shared_ptr<vc::cache::DiskStore> store)
{
    if (tieredCache_) {
        fprintf(stderr, "[Volume] WARNING: setDiskStore() called after cache "
                        "already created — ignoring\n");
        return;
    }
    pendingDiskStore_ = std::move(store);
}

// ============================================================================
// Sampling
// ============================================================================

void Volume::sample(cv::Mat_<uint8_t>& out,
                    const cv::Mat_<cv::Vec3f>& coords,
                    int level, vc::Sampling method)
{
    float scale = 1.0f / std::pow(2.0f, level);
    cv::Mat_<cv::Vec3f> scaled;
    if (level > 0) {
        scaled = coords * scale;
    } else {
        scaled = coords;
    }
    readInterpolated3D(out, tieredCache(), level, scaled, method);
}

void Volume::sampleComposite(cv::Mat_<uint8_t>& out,
                              const cv::Mat_<cv::Vec3f>& baseCoords,
                              const cv::Mat_<cv::Vec3f>& normals,
                              int zStart, int zEnd,
                              const CompositeParams& params,
                              int level)
{
    float scale = 1.0f / std::pow(2.0f, level);
    cv::Mat_<cv::Vec3f> scaled;
    if (level > 0) {
        scaled = baseCoords * scale;
    } else {
        scaled = baseCoords;
    }
    readCompositeFast(out, tieredCache(), level, scaled, normals, scale,
                      zStart, zEnd, params);
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
            readInterpolated3D(out, tieredCache(), lvl, scaled, method);
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
    readInterpolated3D(out, tieredCache(), last, scaled, method);
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
    ensureTieredCache();
    if (!tieredCache_ || coords.empty()) return false;

    vc::VcDataset* ds = zarrDataset(level);
    if (!ds) return false;

    float scale = 1.0f / std::pow(2.0f, level);
    auto cs = ds->defaultChunkShape();  // {cz, cy, cx}
    const auto& shape = ds->shape();    // {z, y, x}

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

    if (minIx > maxIx || minIy > maxIy || minIz > maxIz) return false;

    for (int iz = minIz; iz <= maxIz; iz++)
        for (int iy = minIy; iy <= maxIy; iy++)
            for (int ix = minIx; ix <= maxIx; ix++)
                if (!tieredCache_->get(vc::cache::ChunkKey{level, iz, iy, ix}))
                    return false;
    return true;
}

void Volume::prefetchCompositeChunks(
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    int zStart, int zEnd, int level)
{
    ensureTieredCache();
    if (!tieredCache_) return;

    vc::VcDataset* ds = zarrDataset(level);
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

    tieredCache_->prefetchRegion(level, minIz, minIy, minIx, maxIz, maxIy, maxIx);
}

void Volume::pinCoarsestLevel(bool blocking)
{
    ensureTieredCache();
    if (!tieredCache_ || zarrDs_.empty()) return;

    int last = static_cast<int>(zarrDs_.size()) - 1;
    vc::VcDataset* ds = zarrDataset(last);
    if (!ds) return;

    const auto& shape = ds->shape();
    const auto& chunks = ds->defaultChunkShape();
    std::array<int, 3> gridDims = {
        static_cast<int>((shape[0] + chunks[0] - 1) / chunks[0]),
        static_cast<int>((shape[1] + chunks[1] - 1) / chunks[1]),
        static_cast<int>((shape[2] + chunks[2] - 1) / chunks[2])
    };
    tieredCache_->pinLevel(last, gridDims, blocking);
}

// ============================================================================
// Chunk query / prefetch
// ============================================================================

Volume::ChunkBBox Volume::coordsToChunkBBox(
    const cv::Mat_<cv::Vec3f>& coords, int level) const
{
    vc::VcDataset* ds = zarrDataset(level);
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
    ensureTieredCache();
    if (!tieredCache_) return false;

    auto bb = coordsToChunkBBox(coords, level);
    if (bb.minIx > bb.maxIx) return false;  // invalid bbox

    for (int iz = bb.minIz; iz <= bb.maxIz; iz++)
        for (int iy = bb.minIy; iy <= bb.maxIy; iy++)
            for (int ix = bb.minIx; ix <= bb.maxIx; ix++)
                if (!tieredCache_->get(vc::cache::ChunkKey{level, iz, iy, ix}))
                    return false;
    return true;
}

void Volume::prefetchChunks(const cv::Mat_<cv::Vec3f>& coords, int level)
{
    ensureTieredCache();
    if (!tieredCache_) return;

    auto bb = coordsToChunkBBox(coords, level);
    if (bb.minIx > bb.maxIx) return;  // invalid bbox

    tieredCache_->prefetchRegion(level,
        bb.minIz, bb.minIy, bb.minIx,
        bb.maxIz, bb.maxIy, bb.maxIx);
}

void Volume::cancelPendingPrefetch()
{
    if (tieredCache_) {
        tieredCache_->cancelPendingPrefetch();
    }
}

void Volume::prefetchWorldBBox(const cv::Vec3f& lo, const cv::Vec3f& hi, int level)
{
    ensureTieredCache();
    if (!tieredCache_) return;

    vc::VcDataset* ds = zarrDataset(level);
    if (!ds) return;

    float scale = 1.0f / std::pow(2.0f, level);
    auto cs = ds->defaultChunkShape();  // {cz, cy, cx}
    const auto& shape = ds->shape();    // {z, y, x}

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
