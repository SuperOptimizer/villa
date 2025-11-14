#pragma once

#include <filesystem>
#include <z5/dataset.hxx>
#include <z5/filesystem/handle.hxx>

#include <nlohmann/json.hpp>

#include "z5/types/types.hxx"


class Volume
{
public:
    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, std::string uuid, std::string name);

    ~Volume() = default;


    static std::shared_ptr<Volume> New(const std::filesystem::path& path);

    static std::shared_ptr<Volume> New(const std::filesystem::path& path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const { return metadata_.at("uuid").get<std::string>(); }
    [[nodiscard]] std::string name() const { return metadata_.at("name").get<std::string>(); }
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    [[nodiscard]] nlohmann::json& metadata() { return metadata_; }

    [[nodiscard]] int sliceWidth() const;
    [[nodiscard]] int sliceHeight() const;
    [[nodiscard]] int numSlices() const;
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] z5::Dataset *zarrDataset(int level = 0) const;
    [[nodiscard]] size_t numScales() const;

protected:
    std::filesystem::path path_;
    nlohmann::json metadata_;

    int _width{0};
    int _height{0};
    int _slices{0};

    std::unique_ptr<z5::filesystem::handle::File> zarrFile_;
    std::vector<std::unique_ptr<z5::Dataset>> zarrDs_;
    nlohmann::json zarrGroup_;
    void zarrOpen();
};

