#include "vc/core/types/Volume.hpp"

#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <unordered_map>

#include "vc/core/util/LoadJson.hpp"
#include "vc/core/zarr/ZarrDataset.hpp"
#include "vc/core/csvs/CsvsDataset.hpp"

static const std::filesystem::path METADATA_JSON = "meta.json";
static const std::filesystem::path METADATA_INI = "meta.ini";

// Parse a simple key=value INI file into a map
static std::unordered_map<std::string, std::string> parseIni(const std::filesystem::path& path)
{
    std::unordered_map<std::string, std::string> kv;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq != std::string::npos)
            kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return kv;
}

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
    auto jsonPath = path_ / METADATA_JSON;
    if (std::filesystem::exists(jsonPath)) {
        metadata_ = vc::json::load_json_file(jsonPath);
        vc::json::require_type(metadata_, "type", "vol", jsonPath.string());
        vc::json::require_fields(metadata_, {"uuid", "width", "height", "slices"}, jsonPath.string());
        return;
    }

    // Fall back to meta.ini (used by csvs volumes)
    auto iniPath = path_ / METADATA_INI;
    if (!std::filesystem::exists(iniPath)) {
        throw std::runtime_error("No meta.json or meta.ini found in " + path_.string());
    }

    auto kv = parseIni(iniPath);

    // Parse shape=Z,Y,X
    if (kv.count("shape")) {
        std::istringstream ss(kv["shape"]);
        std::string tok;
        int dims[3] = {};
        for (int i = 0; i < 3 && std::getline(ss, tok, ','); i++)
            dims[i] = std::stoi(tok);
        metadata_["slices"] = dims[0];
        metadata_["height"] = dims[1];
        metadata_["width"] = dims[2];
    }

    metadata_["type"] = "vol";
    metadata_["format"] = kv.count("format") ? kv["format"] : "csvs";

    // Optional fields from ini
    if (kv.count("uuid"))      metadata_["uuid"] = kv["uuid"];
    else                       metadata_["uuid"] = path_.filename().string();
    if (kv.count("name"))      metadata_["name"] = kv["name"];
    else                       metadata_["name"] = path_.filename().string();
    if (kv.count("voxelsize")) metadata_["voxelsize"] = std::stod(kv["voxelsize"]);
    else                       metadata_["voxelsize"] = 0.0;
    if (kv.count("min"))       metadata_["min"] = std::stod(kv["min"]);
    else                       metadata_["min"] = 0.0;
    if (kv.count("max"))       metadata_["max"] = std::stod(kv["max"]);
    else                       metadata_["max"] = 0.0;
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
    auto metaPath = path_ / METADATA_JSON;
    std::ofstream jsonFile(metaPath.string(), std::ofstream::out);
    jsonFile << metadata_ << '\n';
    if (jsonFile.fail()) {
        throw std::runtime_error("could not write json file '" + metaPath.string() + "'");
    }
}

bool Volume::checkDir(std::filesystem::path path)
{
    return std::filesystem::is_directory(path) &&
           (std::filesystem::exists(path / METADATA_JSON) ||
            std::filesystem::exists(path / METADATA_INI));
}

void Volume::zarrOpen()
{
    if (!metadata_.contains("format"))
        return;
    auto fmt = metadata_["format"].get<std::string>();
    if (fmt != "zarr" && fmt != "csvs")
        return;

    // Read group attributes if available
    auto groupAttrsPath = path_ / ".zattrs";
    if (std::filesystem::exists(groupAttrsPath)) {
        std::ifstream attrsFile(groupAttrsPath);
        if (attrsFile.is_open()) {
            zarrGroup_ = nlohmann::json::parse(attrsFile);
        }
    }

    // Find all scale level directories (0, 1, 2, etc.)
    std::vector<std::string> groups;
    bool isCsvs = false;
    for (const auto& entry : std::filesystem::directory_iterator(path_)) {
        if (entry.is_directory()) {
            auto name = entry.path().filename().string();
            // Check for CSVS format (has meta.ini with format=csvs)
            auto csvsMetaPath = entry.path() / "meta.ini";
            if (std::filesystem::exists(csvsMetaPath)) {
                // Quick check: read first line for "format=csvs"
                std::ifstream mf(csvsMetaPath);
                std::string line;
                bool found = false;
                while (std::getline(mf, line)) {
                    if (line == "format=csvs") { found = true; break; }
                }
                if (found) {
                    groups.push_back(name);
                    isCsvs = true;
                    continue;
                }
            }
            // Check if it's a zarr array (has .zarray)
            if (std::filesystem::exists(entry.path() / ".zarray") ||
                std::filesystem::exists(entry.path() / "zarr.json")) {
                groups.push_back(name);
            }
        }
    }
    std::sort(groups.begin(), groups.end());

    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    bool isFirstLevel = true;
    for(const auto& name : groups) {
        auto dsPath = path_ / name;

        if (isCsvs) {
            auto ds = std::make_unique<volcart::csvs::CsvsDataset>(dsPath);
            auto dtype = ds->dtype();
            if (dtype != volcart::zarr::Dtype::UInt8 && dtype != volcart::zarr::Dtype::UInt16)
                throw std::runtime_error("only uint8 & uint16 is currently supported for csvs datasets");

            if (isFirstLevel && !skipShapeCheck) {
                const auto& shape = ds->shape();
                if (static_cast<int>(shape[0]) != _slices ||
                    static_cast<int>(shape[1]) != _height ||
                    static_cast<int>(shape[2]) != _width) {
                    throw std::runtime_error(
                        "csvs level 0 shape does not match meta.json dimensions in " + path_.string());
                }
            }

            // CsvsDataset implements IChunkSource directly
            chunkSources_.push_back(ds.get());
            ownedSources_.push_back(std::move(ds));
        } else {
            auto zarrDs = std::make_unique<volcart::zarr::ZarrDataset>(dsPath);

            auto dtype = zarrDs->getDtype();
            if (dtype != volcart::zarr::Dtype::UInt8 && dtype != volcart::zarr::Dtype::UInt16)
                throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);

            // Verify level 0 shape matches meta.json dimensions
            if (isFirstLevel && !skipShapeCheck) {
                const auto& shape = zarrDs->shape();
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

            // ZarrDataset implements IChunkSource directly
            chunkSources_.push_back(zarrDs.get());
            ownedSources_.push_back(std::move(zarrDs));
        }
        isFirstLevel = false;
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

IChunkSource* Volume::chunkSource(int level) const {
    if (level >= static_cast<int>(chunkSources_.size()))
        return nullptr;
    return chunkSources_[level];
}

size_t Volume::numScales() const {
    return chunkSources_.size();
}
