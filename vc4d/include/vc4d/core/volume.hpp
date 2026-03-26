#pragma once
/// vc4d::Volume — Volumetric data backed by a zarr store.
///
/// Key improvements over vc3d::Volume:
///   • No OpenCV — sampling returns Grid<uint8_t> or writes to raw spans.
///   • No xtensor dependency — chunk data is plain std::vector<uint8_t>.
///   • Cache is injected, not owned — Volume doesn't manage its own cache
///     lifetime.  This decouples the data model from caching policy.
///   • No thread_local statics (skipShapeCheck was a global flag in vc3d).
///   • Metadata is loaded eagerly and stored as nlohmann::json.
///   • Multi-scale pyramid is first-class (not bolted on).

#include "math.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc4d {

// Forward declarations
class ZarrDataset;
class TieredCache;

// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------
class Volume {
public:
    struct ScaleInfo {
        std::array<int, 3> shape;        // {depth, height, width} in voxels
        std::array<int, 3> chunk_shape;  // {cz, cy, cx}
        int downsample_factor;           // 1 for level 0, 2 for level 1, etc.
    };

    // Construct from a local zarr directory.
    explicit Volume(std::filesystem::path zarr_root);

    // Construct for a remote volume (metadata fetched separately).
    Volume(std::string url, std::filesystem::path local_cache_root);

    [[nodiscard]] const std::string& id() const { return id_; }
    [[nodiscard]] const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }
    [[nodiscard]] bool is_remote() const { return is_remote_; }
    [[nodiscard]] const std::string& remote_url() const { return remote_url_; }

    // ---- Dimensions (level 0) -----------------------------------------------
    [[nodiscard]] int width()  const { return scales_[0].shape[2]; }
    [[nodiscard]] int height() const { return scales_[0].shape[1]; }
    [[nodiscard]] int depth()  const { return scales_[0].shape[0]; }
    [[nodiscard]] std::array<int, 3> shape() const { return scales_[0].shape; }
    [[nodiscard]] double voxel_size() const { return voxel_size_; }

    // ---- Multi-scale --------------------------------------------------------
    [[nodiscard]] size_t num_scales() const { return scales_.size(); }
    [[nodiscard]] const ScaleInfo& scale_info(int level) const { return scales_[level]; }

    // ---- Sampling -----------------------------------------------------------
    // Sample a single voxel (trilinear interpolation).
    [[nodiscard]] uint8_t sample(Vec3f coord, int level = 0) const;

    // Sample a grid of coordinates into an output buffer.
    void sample_into(std::span<uint8_t> out,
                     std::span<const Vec3f> coords,
                     int level = 0) const;

    // ---- Cache injection ----------------------------------------------------
    void set_cache(std::shared_ptr<TieredCache> cache) { cache_ = std::move(cache); }
    [[nodiscard]] TieredCache* cache() const { return cache_.get(); }

    // ---- Data bounds --------------------------------------------------------
    struct DataBounds {
        Box3f box;
        bool valid = false;
    };
    [[nodiscard]] const DataBounds& data_bounds() const;
    void compute_data_bounds();

    // ---- Prefetch -----------------------------------------------------------
    void prefetch(Box3f world_box, int level);
    void cancel_prefetch();

private:
    void load_metadata();

    std::filesystem::path path_;
    std::string id_;
    std::string name_;
    nlohmann::json metadata_;

    std::vector<ScaleInfo> scales_;
    std::vector<std::unique_ptr<ZarrDataset>> datasets_;
    double voxel_size_{1.0};

    std::shared_ptr<TieredCache> cache_;

    bool is_remote_{false};
    std::string remote_url_;

    mutable DataBounds data_bounds_;
};

} // namespace vc4d
