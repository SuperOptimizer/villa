#include "vc/core/types/ZarrMetadata.hpp"

#include "vc/core/types/VcDataset.hpp"

#include <utils/http_fetch.hpp>
#include <utils/zarr.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace vc::render {

namespace {

class HttpStatusError final : public std::runtime_error {
public:
    HttpStatusError(long status, const std::string& key)
        : std::runtime_error("HTTP " + std::to_string(status) + " fetching " + key)
        , status_(status)
    {
    }
    long status() const noexcept { return status_; }

private:
    long status_ = 0;
};

bool hasSuffix(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

bool isOptionalMetadataProbe(const std::string& key)
{
    return key == "zarr.json" || key == ".zattrs" ||
           hasSuffix(key, "/zarr.json") || hasSuffix(key, "/.zattrs");
}

class ClassifyingHttpStore final : public utils::Store {
public:
    explicit ClassifyingHttpStore(std::string baseUrl, vc::HttpAuth auth = {})
        : baseUrl_(stripTrailingSlash(std::move(baseUrl)))
        , client_(makeClient(std::move(auth)))
    {
    }

    bool exists(const std::string& key) const override
    {
        auto response = client_.head(makeUrl(key));
        if (response.ok())
            return true;
        if (response.not_found())
            return false;
        if (response.status_code == 403 && isOptionalMetadataProbe(key))
            return false;
        throw HttpStatusError(response.status_code, key);
    }

    std::vector<std::byte> get(const std::string& key) const override
    {
        auto found = get_if_exists(key);
        if (!found)
            throw std::runtime_error("HTTP zarr key not found: " + key);
        return std::move(*found);
    }

    std::optional<std::vector<std::byte>> get_if_exists(const std::string& key) const override
    {
        auto response = client_.get(makeUrl(key));
        if (response.ok())
            return std::move(response.body);
        if (response.not_found())
            return std::nullopt;
        if (response.status_code == 403 && isOptionalMetadataProbe(key))
            return std::nullopt;
        throw HttpStatusError(response.status_code, key);
    }

    std::optional<std::vector<std::byte>>
    get_partial(const std::string& key, std::size_t offset, std::size_t length) const override
    {
        auto response = client_.get_range(makeUrl(key), offset, length);
        if (response.ok())
            return std::move(response.body);
        if (response.not_found())
            return std::nullopt;
        throw HttpStatusError(response.status_code, key);
    }

    std::optional<std::vector<std::byte>>
    get_parallel(const std::string& key) const override
    {
        auto response = client_.get_parallel(makeUrl(key));
        if (response.ok())
            return std::move(response.body);
        if (response.not_found())
            return std::nullopt;
        throw HttpStatusError(response.status_code, key);
    }

    void set(const std::string&, std::span<const std::byte>) override
    {
        throw std::runtime_error("HTTP zarr store is read-only");
    }

    void erase(const std::string&) override
    {
        throw std::runtime_error("HTTP zarr store is read-only");
    }

private:
    std::string makeUrl(const std::string& key) const { return baseUrl_ + "/" + key; }

    static std::string stripTrailingSlash(std::string value)
    {
        while (!value.empty() && value.back() == '/')
            value.pop_back();
        return value;
    }

    static utils::HttpClient makeClient(vc::HttpAuth auth)
    {
        utils::HttpClient::Config config;
        config.aws_auth = std::move(auth);
        config.transfer_timeout = std::chrono::seconds{60};
        return utils::HttpClient(std::move(config));
    }

    std::string baseUrl_;
    utils::HttpClient client_;
};

std::array<int, 3> toArray3(const std::vector<std::size_t>& values, const char* name)
{
    if (values.size() != 3)
        throw std::runtime_error(std::string("zarr ") + name + " must be 3D");
    return {static_cast<int>(values[0]), static_cast<int>(values[1]),
            static_cast<int>(values[2])};
}

// Read one array's metadata into a fresh single-level ZarrPyramidMeta, mirroring
// ZarrChunkFetcher::addLevel (no fetcher built).
ZarrPyramidMeta levelMeta(const utils::ZarrArray& array)
{
    const auto& meta = array.metadata();
    ChunkDtype dtype = ChunkDtype::UInt8;
    if (meta.dtype == utils::ZarrDtype::uint16) {
        dtype = ChunkDtype::UInt16;
    } else if (meta.dtype != utils::ZarrDtype::uint8) {
        throw std::runtime_error("zarr metadata supports uint8 and uint16 only");
    }

    std::vector<std::size_t> chunkShape = meta.chunks;
    if (meta.shard_config)
        chunkShape = meta.shard_config->sub_chunks;

    ZarrPyramidMeta single;
    single.shapes.push_back(toArray3(meta.shape, "shape"));
    single.chunkShapes.push_back(toArray3(chunkShape, "chunk shape"));
    single.storageChunkShapes.push_back(toArray3(meta.chunks, "storage chunk shape"));
    single.fillValue = meta.fill_value.value_or(0.0);
    single.dtype = dtype;
    return single;
}

IChunkedArray::LevelTransform powerOfTwoTransform(int physicalLevel)
{
    const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << physicalLevel);
    IChunkedArray::LevelTransform transform;
    transform.scaleFromLevel0 = {invScale, invScale, invScale};
    return transform;
}

void addPhysicalLevel(ZarrPyramidMeta& meta, int physicalLevel, const utils::ZarrArray& array)
{
    if (physicalLevel < 0)
        throw std::runtime_error("zarr physical level must be non-negative");

    const auto index = static_cast<std::size_t>(physicalLevel);
    if (meta.shapes.size() <= index) {
        meta.levelNumbers.resize(index + 1, -1);
        meta.transforms.resize(index + 1);
        meta.shapes.resize(index + 1, {0, 0, 0});
        meta.chunkShapes.resize(index + 1, {1, 1, 1});
        meta.storageChunkShapes.resize(index + 1, {1, 1, 1});
    }
    if (meta.levelNumbers[index] >= 0)
        throw std::runtime_error("duplicate zarr physical level " + std::to_string(physicalLevel));

    ZarrPyramidMeta single = levelMeta(array);
    const bool hasExistingLevel =
        std::any_of(meta.levelNumbers.begin(), meta.levelNumbers.end(),
                    [](int n) { return n >= 0; });
    if (hasExistingLevel && meta.dtype != single.dtype)
        throw std::runtime_error("zarr requires all levels to have the same dtype");

    meta.levelNumbers[index] = physicalLevel;
    meta.shapes[index] = single.shapes[0];
    meta.chunkShapes[index] = single.chunkShapes[0];
    meta.storageChunkShapes[index] = single.storageChunkShapes[0];
    meta.transforms[index] = powerOfTwoTransform(physicalLevel);
    meta.fillValue = single.fillValue;
    meta.dtype = single.dtype;
}

bool hasAnyLevel(const ZarrPyramidMeta& meta)
{
    return std::any_of(meta.levelNumbers.begin(), meta.levelNumbers.end(),
                       [](int n) { return n >= 0; });
}

std::vector<int> localLevelNumbers(const std::filesystem::path& root)
{
    std::vector<int> levels;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory())
            continue;
        const auto name = entry.path().filename().string();
        if (name.empty() || !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            }))
            continue;
        if (std::filesystem::exists(entry.path() / ".zarray") ||
            std::filesystem::exists(entry.path() / "zarr.json")) {
            levels.push_back(std::stoi(name));
        }
    }
    std::sort(levels.begin(), levels.end());
    return levels;
}

utils::ZarrArray openLocalArray(const std::filesystem::path& path)
{
    auto array = utils::ZarrArray::open(path, vc::buildZarrCodecRegistry(1));
    if (array.metadata().dtype == utils::ZarrDtype::uint16)
        array = utils::ZarrArray::open(path, vc::buildZarrCodecRegistry(2));
    return array;
}

utils::ZarrArray openRemoteArray(const std::shared_ptr<utils::Store>& store, const std::string& key)
{
    auto array = utils::ZarrArray::open(store, key, vc::buildZarrCodecRegistry(1));
    if (array.metadata().dtype == utils::ZarrDtype::uint16)
        array = utils::ZarrArray::open(store, key, vc::buildZarrCodecRegistry(2));
    return array;
}

std::vector<std::pair<int, std::string>> remoteLevelKeysFromZattrs(
    const std::shared_ptr<utils::Store>& store)
{
    auto data = store->get_if_exists(".zattrs");
    if (!data)
        return {};

    const std::string json(reinterpret_cast<const char*>(data->data()), data->size());
    auto attrs = utils::json_parse(json);
    if (!attrs.contains("multiscales") || !attrs["multiscales"].is_array() ||
        attrs["multiscales"].empty()) {
        return {};
    }

    const auto& ms0 = attrs["multiscales"][0];
    if (!ms0.contains("datasets") || !ms0["datasets"].is_array())
        return {};

    std::vector<std::pair<int, std::string>> keys;
    int datasetIndex = 0;
    for (const auto& dataset : ms0["datasets"]) {
        if (!dataset.contains("path") || !dataset["path"].is_string()) {
            ++datasetIndex;
            continue;
        }
        std::string path = dataset["path"].get_string();
        while (!path.empty() && path.front() == '/')
            path.erase(path.begin());
        while (!path.empty() && path.back() == '/')
            path.pop_back();
        if (!path.empty())
            keys.emplace_back(datasetIndex, std::move(path));
        ++datasetIndex;
    }
    return keys;
}

} // namespace

ZarrPyramidMeta openLocalZarrMeta(const std::filesystem::path& root)
{
    ZarrPyramidMeta meta;
    for (int level : localLevelNumbers(root))
        addPhysicalLevel(meta, level, openLocalArray(root / std::to_string(level)));
    if (!hasAnyLevel(meta))
        addPhysicalLevel(meta, 0, openLocalArray(root));
    return meta;
}

ZarrPyramidMeta openHttpZarrMeta(const std::string& url, const vc::HttpAuth& auth)
{
    auto store = std::make_shared<ClassifyingHttpStore>(url, auth);
    ZarrPyramidMeta meta;

    const auto zattrsLevelKeys = remoteLevelKeysFromZattrs(store);
    if (!zattrsLevelKeys.empty()) {
        for (const auto& [physicalLevel, key] : zattrsLevelKeys)
            addPhysicalLevel(meta, physicalLevel, openRemoteArray(store, key));
        return meta;
    }

    for (int physicalLevel = 0; physicalLevel < 32; ++physicalLevel) {
        const auto key = std::to_string(physicalLevel);
        try {
            addPhysicalLevel(meta, physicalLevel, openRemoteArray(store, key));
        } catch (const HttpStatusError& e) {
            if (e.status() == 404 ||
                (e.status() == 403 && (hasAnyLevel(meta) || physicalLevel == 0)))
                break;
            throw;
        } catch (const std::exception&) {
            if (physicalLevel == 0)
                throw;
            break;
        }
    }
    if (!hasAnyLevel(meta))
        addPhysicalLevel(meta, 0, openRemoteArray(store, ""));
    return meta;
}

} // namespace vc::render
