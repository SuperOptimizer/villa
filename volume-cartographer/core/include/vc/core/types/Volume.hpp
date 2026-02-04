#pragma once

#include <nlohmann/json_fwd.hpp>
#include <stddef.h>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <map>

// Forward declarations - avoid including heavy headers
namespace z5 {
    class Dataset;
    namespace filesystem::handle {
        class File;
    }
}

class Volume final
{
public:
    // Static flag to skip zarr shape validation against meta.json
    static inline bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, const std::string& uuid, const std::string& name);

    ~Volume();


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    static std::shared_ptr<Volume> New(std::filesystem::path path, const std::string& uuid, const std::string& name);

    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }
    void saveMetadata();

    [[nodiscard]] int sliceWidth() const noexcept { return _width; }
    [[nodiscard]] int sliceHeight() const noexcept { return _height; }
    [[nodiscard]] int numSlices() const noexcept { return _slices; }
    [[nodiscard]] std::array<int, 3> shape() const noexcept { return {_width, _height, _slices}; }
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] z5::Dataset *zarrDataset(int level = 0) const;
    [[nodiscard]] size_t numScales() const noexcept { return zarrDs_.size(); }

    static bool checkDir(const std::filesystem::path& path);

protected:
    std::filesystem::path path_;
    std::unique_ptr<nlohmann::json> metadata_;

    int _width{0};
    int _height{0};
    int _slices{0};

    std::unique_ptr<z5::filesystem::handle::File> zarrFile_;
    std::vector<std::unique_ptr<z5::Dataset>> zarrDs_;
    std::unique_ptr<nlohmann::json> zarrGroup_;
    void zarrOpen();

    void loadMetadata();
};

