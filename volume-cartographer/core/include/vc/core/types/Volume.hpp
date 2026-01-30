#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

#include "vc/core/types/IChunkSource.hpp"

class Volume
{
public:
    // Static flag to skip zarr shape validation against meta.json
    static inline bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, std::string uuid, std::string name);

    ~Volume();


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    static std::shared_ptr<Volume> New(std::filesystem::path path, std::string uuid, std::string name);

    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void saveMetadata();

    [[nodiscard]] int sliceWidth() const;
    [[nodiscard]] int sliceHeight() const;
    [[nodiscard]] int numSlices() const;
    /// Returns {width(X), height(Y), depth(Z)}
    [[nodiscard]] std::array<int, 3> shape() const;
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] IChunkSource* chunkSource(int level = 0) const;
    [[nodiscard]] size_t numScales() const;

    static bool checkDir(std::filesystem::path path);

protected:
    std::filesystem::path path_;
    nlohmann::json metadata_;

    int _width{0};
    int _height{0};
    int _slices{0};

    // Owning storage for all chunk sources (zarr adapters, csvs datasets, etc.)
    std::vector<std::unique_ptr<IChunkSource>> ownedSources_;
    // Non-owning pointers: one per level
    std::vector<IChunkSource*> chunkSources_;
    nlohmann::json zarrGroup_;
    void zarrOpen();

    void loadMetadata();
};

