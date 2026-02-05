#include "vc/core/types/Volume.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "vc/core/types/ChunkStore.hpp"
#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/SlicingLite.hpp"

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/filesystem/metadata.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include <xtensor/containers/xarray.hpp>

static const std::filesystem::path METADATA_FILE = "meta.json";
static const std::filesystem::path METADATA_FILE_ALT = "metadata.json";

// ============================================================================
// Impl — all z5 state lives here
// ============================================================================

struct Volume::Impl {
    std::filesystem::path path;
    std::unique_ptr<nlohmann::json> metadata;

    int width{0};
    int height{0};
    int slices{0};

    std::unique_ptr<z5::filesystem::handle::File> zarrFile;
    std::vector<std::unique_ptr<z5::Dataset>> zarrDs;
    std::unique_ptr<nlohmann::json> zarrGroup;

    std::shared_ptr<ChunkStore> store;

    Impl()
        : metadata(std::make_unique<nlohmann::json>()),
          zarrGroup(std::make_unique<nlohmann::json>()) {}

    void loadMetadata()
    {
        auto metaPath = path / METADATA_FILE;
        if (std::filesystem::exists(metaPath)) [[likely]] {
            *metadata = vc::json::load_json_file(metaPath);
        } else {
            auto altPath = path / METADATA_FILE_ALT;
            auto full = vc::json::load_json_file(altPath);
            if (!full.contains("scan")) [[unlikely]] {
                throw std::runtime_error(
                    "metadata.json missing 'scan' key: " + altPath.string());
            }
            *metadata = full["scan"];
            if (!metadata->contains("format")) {
                (*metadata)["format"] = "zarr";
            }
            metaPath = altPath;
        }
        vc::json::require_type(*metadata, "type", "vol", metaPath.string());
        vc::json::require_fields(*metadata, {"uuid", "width", "height", "slices"}, metaPath.string());
    }

    void zarrOpen()
    {
        if (!metadata->contains("format") || (*metadata)["format"].get<std::string>() != "zarr")
            return;

        zarrFile = std::make_unique<z5::filesystem::handle::File>(path);
        z5::filesystem::handle::Group group(path, z5::FileMode::FileMode::r);
        z5::readAttributes(group, *zarrGroup);

        std::vector<std::string> groups;
        zarrFile->keys(groups);
        std::sort(groups.begin(), groups.end());

        for (const auto& name : groups) {
            z5::filesystem::handle::Dataset tmp_handle(path / name, z5::FileMode::FileMode::r);
            z5::DatasetMetadata dsMeta;
            z5::filesystem::readMetadata(tmp_handle, dsMeta);

            z5::filesystem::handle::Dataset ds_handle(group, name, dsMeta.zarrDelimiter);
            zarrDs.push_back(z5::filesystem::openDataset(ds_handle));

            if (zarrDs.back()->getDtype() != z5::types::Datatype::uint8 &&
                zarrDs.back()->getDtype() != z5::types::Datatype::uint16)
                throw std::runtime_error(
                    "only uint8 & uint16 is currently supported for zarr datasets incompatible type found in " +
                    path.string() + " / " + name);

            if (zarrDs.size() == 1 && !Volume::skipShapeCheck) {
                const auto& shape = zarrDs[0]->shape();
                if (static_cast<int>(shape[0]) != slices ||
                    static_cast<int>(shape[1]) != height ||
                    static_cast<int>(shape[2]) != width) {
                    throw std::runtime_error(
                        "zarr level 0 shape [z,y,x]=(" + std::to_string(shape[0]) + ", " +
                        std::to_string(shape[1]) + ", " + std::to_string(shape[2]) +
                        ") does not match meta.json dimensions (slices=" + std::to_string(slices) +
                        ", height=" + std::to_string(height) + ", width=" + std::to_string(width) +
                        ") in " + path.string());
                }
            }
        }
    }

    z5::Dataset* dataset(int level) const {
        if (level >= static_cast<int>(zarrDs.size())) [[unlikely]]
            return nullptr;
        return zarrDs[level].get();
    }

    void ensureStore() {
        if (!store) {
            store = std::make_shared<ChunkStore>();
        }
    }
};

// ============================================================================
// Construction / destruction
// ============================================================================

Volume::Volume(std::filesystem::path path) : pImpl_(std::make_unique<Impl>())
{
    pImpl_->path = std::move(path);
    pImpl_->loadMetadata();
    pImpl_->width = (*pImpl_->metadata)["width"].get<int>();
    pImpl_->height = (*pImpl_->metadata)["height"].get<int>();
    pImpl_->slices = (*pImpl_->metadata)["slices"].get<int>();
    pImpl_->zarrOpen();
}

Volume::Volume(std::filesystem::path path, std::shared_ptr<ChunkStore> store)
    : pImpl_(std::make_unique<Impl>())
{
    pImpl_->path = std::move(path);
    pImpl_->store = std::move(store);
    pImpl_->loadMetadata();
    pImpl_->width = (*pImpl_->metadata)["width"].get<int>();
    pImpl_->height = (*pImpl_->metadata)["height"].get<int>();
    pImpl_->slices = (*pImpl_->metadata)["slices"].get<int>();
    pImpl_->zarrOpen();
}

Volume::Volume(std::filesystem::path path, const std::string& uuid, const std::string& name)
    : pImpl_(std::make_unique<Impl>())
{
    pImpl_->path = std::move(path);
    (*pImpl_->metadata)["uuid"] = uuid;
    (*pImpl_->metadata)["name"] = name;
    (*pImpl_->metadata)["type"] = "vol";
    (*pImpl_->metadata)["width"] = pImpl_->width;
    (*pImpl_->metadata)["height"] = pImpl_->height;
    (*pImpl_->metadata)["slices"] = pImpl_->slices;
    (*pImpl_->metadata)["voxelsize"] = double{};
    (*pImpl_->metadata)["min"] = double{};
    (*pImpl_->metadata)["max"] = double{};
    pImpl_->zarrOpen();
}

Volume::~Volume() = default;

// ============================================================================
// Static factories
// ============================================================================

std::shared_ptr<Volume> Volume::New(std::filesystem::path path)
{
    return std::make_shared<Volume>(std::move(path));
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path, std::shared_ptr<ChunkStore> store)
{
    return std::make_shared<Volume>(std::move(path), std::move(store));
}

std::shared_ptr<Volume> Volume::New(std::filesystem::path path, const std::string& uuid, const std::string& name)
{
    return std::make_shared<Volume>(std::move(path), uuid, name);
}

// ============================================================================
// Metadata
// ============================================================================

std::string Volume::id() const { return (*pImpl_->metadata)["uuid"].get<std::string>(); }
std::string Volume::name() const { return (*pImpl_->metadata)["name"].get<std::string>(); }
void Volume::setName(const std::string& n) { (*pImpl_->metadata)["name"] = n; }
std::filesystem::path Volume::path() const noexcept { return pImpl_->path; }

void Volume::saveMetadata()
{
    auto metaPath = pImpl_->path / METADATA_FILE;
    std::ofstream jsonFile(metaPath.string(), std::ofstream::out);
    jsonFile << *pImpl_->metadata << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

int Volume::sliceWidth() const noexcept { return pImpl_->width; }
int Volume::sliceHeight() const noexcept { return pImpl_->height; }
int Volume::numSlices() const noexcept { return pImpl_->slices; }
std::array<int, 3> Volume::shape() const noexcept { return {pImpl_->width, pImpl_->height, pImpl_->slices}; }

double Volume::voxelSize() const { return (*pImpl_->metadata)["voxelsize"].get<double>(); }

bool Volume::checkDir(const std::filesystem::path& path)
{
    return std::filesystem::is_directory(path) &&
           (std::filesystem::exists(path / METADATA_FILE) ||
            std::filesystem::exists(path / METADATA_FILE_ALT));
}

// ============================================================================
// Scale levels
// ============================================================================

size_t Volume::numScales() const noexcept { return pImpl_->zarrDs.size(); }

std::array<size_t, 3> Volume::shapeZYX(int level) const
{
    auto* ds = pImpl_->dataset(level);
    if (!ds) return {0, 0, 0};
    const auto& s = ds->shape();
    return {s[0], s[1], s[2]};
}

std::array<int, 3> Volume::chunkShape(int level) const
{
    auto* ds = pImpl_->dataset(level);
    if (!ds) return {0, 0, 0};
    const auto& cs = ds->defaultChunkShape();
    return {static_cast<int>(cs[0]), static_cast<int>(cs[1]), static_cast<int>(cs[2])};
}

// ============================================================================
// Cache management
// ============================================================================

void Volume::setChunkStore(std::shared_ptr<ChunkStore> store)
{
    pImpl_->store = std::move(store);
}

std::shared_ptr<ChunkStore> Volume::chunkStore() const
{
    return pImpl_->store;
}

// ============================================================================
// Interpolated reads — delegate to existing Slicing free functions
// ============================================================================

void Volume::readInterpolated(cv::Mat_<uint8_t>& out, const cv::Mat_<cv::Vec3f>& coords,
                              InterpolationMethod method, int level)
{
    pImpl_->ensureStore();
    readInterpolated3D(out, pImpl_->dataset(level), coords, &pImpl_->store->cache8(), method);
}

void Volume::readInterpolated(cv::Mat_<uint16_t>& out, const cv::Mat_<cv::Vec3f>& coords,
                              InterpolationMethod method, int level)
{
    pImpl_->ensureStore();
    readInterpolated3D(out, pImpl_->dataset(level), coords, &pImpl_->store->cache16(), method);
}

// ============================================================================
// Composite rendering
// ============================================================================

void Volume::readComposite(cv::Mat_<uint8_t>& out, const cv::Mat_<cv::Vec3f>& baseCoords,
                           const cv::Mat_<cv::Vec3f>& normals, float zStep, int zStart, int zEnd,
                           const CompositeParams& params, int level)
{
    pImpl_->ensureStore();
    readCompositeFast(out, pImpl_->dataset(level), baseCoords, normals,
                      zStep, zStart, zEnd, params, pImpl_->store->cache8());
}

void Volume::readCompositeConstantNormal(cv::Mat_<uint8_t>& out, const cv::Mat_<cv::Vec3f>& baseCoords,
                                         const cv::Vec3f& normal, float zStep, int zStart, int zEnd,
                                         const CompositeParams& params, int level)
{
    pImpl_->ensureStore();
    readCompositeFastConstantNormal(out, pImpl_->dataset(level), baseCoords, normal,
                                    zStep, zStart, zEnd, params, pImpl_->store->cache8());
}

// ============================================================================
// Multi-slice reads
// ============================================================================

void Volume::readMultiSlice(std::vector<cv::Mat_<uint8_t>>& out, const cv::Mat_<cv::Vec3f>& basePoints,
                            const cv::Mat_<cv::Vec3f>& stepDirs, const std::vector<float>& offsets, int level)
{
    pImpl_->ensureStore();
    ::readMultiSlice(out, pImpl_->dataset(level), &pImpl_->store->cache8(), basePoints, stepDirs, offsets);
}

void Volume::readMultiSlice(std::vector<cv::Mat_<uint16_t>>& out, const cv::Mat_<cv::Vec3f>& basePoints,
                            const cv::Mat_<cv::Vec3f>& stepDirs, const std::vector<float>& offsets, int level)
{
    pImpl_->ensureStore();
    ::readMultiSlice(out, pImpl_->dataset(level), &pImpl_->store->cache16(), basePoints, stepDirs, offsets);
}

// ============================================================================
// 3D block reads — cv::Mat based, z5 hidden internally
// ============================================================================

template<typename T>
static void readBlockImpl(std::vector<cv::Mat_<T>>& out,
                          z5::Dataset* ds,
                          const std::array<int,3>& offsetZYX,
                          const std::array<int,3>& sizeZYX)
{
    if (!ds) {
        out.clear();
        return;
    }

    const int nz = sizeZYX[0], ny = sizeZYX[1], nx = sizeZYX[2];

    // Read via z5 into an xtensor, then copy to cv::Mat slices
    xt::xarray<T> buf = xt::empty<T>({
        static_cast<size_t>(nz),
        static_cast<size_t>(ny),
        static_cast<size_t>(nx)});
    std::vector<size_t> off = {
        static_cast<size_t>(offsetZYX[0]),
        static_cast<size_t>(offsetZYX[1]),
        static_cast<size_t>(offsetZYX[2])};
    z5::multiarray::readSubarray<T>(*ds, buf, off.begin());

    out.resize(nz);
    const T* data = buf.data();
    const size_t sliceSize = static_cast<size_t>(ny) * nx;

    for (int z = 0; z < nz; z++) {
        out[z].create(ny, nx);
        std::memcpy(out[z].data, data + z * sliceSize, sliceSize * sizeof(T));
    }
}

void Volume::readBlock(std::vector<cv::Mat_<uint8_t>>& out,
                       const std::array<int,3>& offsetZYX, const std::array<int,3>& sizeZYX, int level)
{
    readBlockImpl(out, pImpl_->dataset(level), offsetZYX, sizeZYX);
}

void Volume::readBlock(std::vector<cv::Mat_<uint16_t>>& out,
                       const std::array<int,3>& offsetZYX, const std::array<int,3>& sizeZYX, int level)
{
    readBlockImpl(out, pImpl_->dataset(level), offsetZYX, sizeZYX);
}

// ============================================================================
// 3D block writes — cv::Mat based
// ============================================================================

template<typename T>
static void writeBlockImpl(z5::Dataset* ds,
                           const std::vector<cv::Mat_<T>>& data,
                           const std::array<int,3>& offsetZYX)
{
    if (!ds || data.empty()) return;

    const int nz = static_cast<int>(data.size());
    const int ny = data[0].rows;
    const int nx = data[0].cols;

    xt::xarray<T> buf = xt::empty<T>({
        static_cast<size_t>(nz),
        static_cast<size_t>(ny),
        static_cast<size_t>(nx)});

    T* dst = buf.data();
    const size_t sliceSize = static_cast<size_t>(ny) * nx;

    for (int z = 0; z < nz; z++) {
        std::memcpy(dst + z * sliceSize, data[z].data, sliceSize * sizeof(T));
    }

    std::vector<size_t> off = {
        static_cast<size_t>(offsetZYX[0]),
        static_cast<size_t>(offsetZYX[1]),
        static_cast<size_t>(offsetZYX[2])};
    z5::multiarray::writeSubarray<T>(*ds, buf, off.begin());
}

void Volume::writeBlock(const std::vector<cv::Mat_<uint8_t>>& data,
                        const std::array<int,3>& offsetZYX, int level)
{
    writeBlockImpl(pImpl_->dataset(level), data, offsetZYX);
}

void Volume::writeBlock(const std::vector<cv::Mat_<uint16_t>>& data,
                        const std::array<int,3>& offsetZYX, int level)
{
    writeBlockImpl(pImpl_->dataset(level), data, offsetZYX);
}

// ============================================================================
// Gradient computation
// ============================================================================

cv::Mat_<cv::Vec3f> Volume::computeGradients(const cv::Mat_<cv::Vec3f>& rawPoints, float dsScale, int level)
{
    return computeVolumeGradientsNative(pImpl_->dataset(level), rawPoints, dsScale);
}

// ============================================================================
// Deprecated bridge
// ============================================================================

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

z5::Dataset* Volume::zarrDataset(int level) const
{
    return pImpl_->dataset(level);
}

#pragma GCC diagnostic pop

ChunkCache<uint8_t>* Volume::rawCache8() const
{
    if (!pImpl_->store) return nullptr;
    return &pImpl_->store->cache8();
}

ChunkCache<uint16_t>* Volume::rawCache16() const
{
    if (!pImpl_->store) return nullptr;
    return &pImpl_->store->cache16();
}
