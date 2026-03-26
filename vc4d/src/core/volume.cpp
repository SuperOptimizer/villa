#include "vc4d/core/volume.hpp"
#include "vc4d/core/zarr.hpp"
#include "vc4d/cache/tiered_cache.hpp"

#include <stdexcept>

namespace vc4d {

Volume::Volume(std::filesystem::path zarr_root)
    : path_(std::move(zarr_root))
{
    load_metadata();
}

Volume::Volume(std::string url, std::filesystem::path local_cache_root)
    : path_(std::move(local_cache_root))
    , is_remote_(true)
    , remote_url_(std::move(url))
{
    // Remote volumes: metadata is fetched to local_cache_root first,
    // then loaded normally.
    load_metadata();
}

void Volume::load_metadata() {
    auto meta_path = path_ / "meta.json";
    if (!std::filesystem::exists(meta_path)) {
        // Auto-detect from zarr arrays
        meta_path = path_ / ".zattrs";
    }

    if (std::filesystem::exists(meta_path)) {
        std::ifstream f(meta_path);
        metadata_ = nlohmann::json::parse(f);
        if (metadata_.contains("uuid"))
            id_ = metadata_["uuid"].get<std::string>();
        if (metadata_.contains("name"))
            name_ = metadata_["name"].get<std::string>();
        if (metadata_.contains("voxelsize"))
            voxel_size_ = metadata_["voxelsize"].get<double>();
    }

    // Scan for scale levels (subdirectories named 0, 1, 2, ...)
    for (int level = 0; ; ++level) {
        auto level_dir = path_ / std::to_string(level);
        if (!std::filesystem::exists(level_dir / ".zarray"))
            break;

        auto ds = std::make_unique<ZarrDataset>(level_dir);
        scales_.push_back(ScaleInfo{
            ds->shape(),
            ds->chunk_shape(),
            1 << level
        });
        datasets_.push_back(std::move(ds));
    }

    if (scales_.empty())
        throw std::runtime_error("No zarr scale levels found in " + path_.string());
}

uint8_t Volume::sample(Vec3f coord, int level) const {
    if (datasets_.empty() || level >= static_cast<int>(datasets_.size()))
        return 0;

    // Scale coordinates for the requested pyramid level
    float s = static_cast<float>(scales_[level].downsample_factor);
    float x = coord.x / s;
    float y = coord.y / s;
    float z = coord.z / s;

    auto& ds = *datasets_[level];
    auto [dz, dy, dx] = ds.shape();

    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    int iz = static_cast<int>(z);

    if (ix < 0 || ix >= dx || iy < 0 || iy >= dy || iz < 0 || iz >= dz)
        return 0;

    // TODO: Trilinear interpolation + cache integration
    // For now, return 0 as placeholder
    return 0;
}

void Volume::sample_into(std::span<uint8_t> out,
                          std::span<const Vec3f> coords,
                          int level) const {
    for (size_t i = 0; i < coords.size() && i < out.size(); ++i)
        out[i] = sample(coords[i], level);
}

const Volume::DataBounds& Volume::data_bounds() const {
    return data_bounds_;
}

void Volume::compute_data_bounds() {
    // Scan coarsest level to find non-zero extent
    if (scales_.empty()) return;
    // TODO: Implement scanning
    data_bounds_.valid = false;
}

void Volume::prefetch(Box3f /*world_box*/, int /*level*/) {
    // TODO: Submit async chunk load tasks
}

void Volume::cancel_prefetch() {
    // TODO: Cancel pending async tasks
}

} // namespace vc4d
