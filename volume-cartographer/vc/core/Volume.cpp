#include "vc/core/types/Volume.hpp"

#include <fstream>
#include <opencv2/imgcodecs.hpp>

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"


Volume::Volume(std::filesystem::path path) : path_(std::move(path))
{
    // Load metadata from meta.json
    auto metaPath = path_ / "meta.json";
    if (!std::filesystem::exists(metaPath)) {
        throw std::runtime_error("could not find json file '" + metaPath.string() + "'");
    }
    std::ifstream jsonFile(metaPath);
    if (!jsonFile) {
        throw std::runtime_error("could not open json file '" + metaPath.string() + "'");
    }
    jsonFile >> metadata_;
    if (jsonFile.bad()) {
        throw std::runtime_error("could not read json file '" + metaPath.string() + "'");
    }

    if (metadata_.at("type").get<std::string>() != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    _width = metadata_.at("width").get<int>();
    _height = metadata_.at("height").get<int>();
    _slices = metadata_.at("slices").get<int>();

    std::vector<std::mutex> init_mutexes(_slices);


    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(std::filesystem::path path, std::string uuid, std::string name)
    : path_(std::move(path))
{
    metadata_["uuid"] = std::move(uuid);
    metadata_["name"] = std::move(name);
    metadata_["type"] = "vol";
    metadata_["width"] = _width;
    metadata_["height"] = _height;
    metadata_["slices"] = _slices;
    metadata_["voxelsize"] = double{};
    metadata_["min"] = double{};
    metadata_["max"] = double{};

    zarrOpen();
}

void Volume::zarrOpen()
{
    if (!metadata_.contains("format") || metadata_.at("format").get<std::string>() != "zarr")
        return;

    zarrFile_ = std::make_unique<z5::filesystem::handle::File>(path_);
    z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
    z5::readAttributes(group, zarrGroup_);

    std::vector<std::string> groups;
    zarrFile_->keys(groups);
    std::ranges::sort(groups);

    //FIXME hardcoded assumption that groups correspond to power-2 scaledowns ...
    for(const auto& name : groups) {
        z5::filesystem::handle::Dataset ds_handle(group, name, nlohmann::json::parse(std::ifstream(path_/name/".zarray")).value<std::string>("dimension_separator","."));

        zarrDs_.push_back(z5::filesystem::openDataset(ds_handle));
        if (zarrDs_.back()->getDtype() != z5::types::Datatype::uint8 && zarrDs_.back()->getDtype() != z5::types::Datatype::uint16)
            throw std::runtime_error("only uint8 & uint16 is currently supported for zarr datasets incompatible type found in "+path_.string()+" / " +name);
    }
}

std::shared_ptr<Volume> Volume::New(const std::filesystem::path& path)
{
    return std::make_shared<Volume>(path);
}

std::shared_ptr<Volume> Volume::New(const std::filesystem::path& path, const std::string& uuid, const std::string& name)
{
    return std::make_shared<Volume>(path, uuid, name);
}

int Volume::sliceWidth() const { return _width; }
int Volume::sliceHeight() const { return _height; }
int Volume::numSlices() const { return _slices; }
double Volume::voxelSize() const
{
    return metadata_.at("voxelsize").get<double>();
}

z5::Dataset *Volume::zarrDataset(int level) const {
    if (level >= zarrDs_.size())
        return nullptr;

    return zarrDs_[level].get();
}

size_t Volume::numScales() const {
    return zarrDs_.size();
}
