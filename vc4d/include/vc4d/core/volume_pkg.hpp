#pragma once
/// vc4d::VolumePkg — A collection of volumes and segmentations.
///
/// Simplified from vc3d::VolumePkg:
///   • No static mutable state (loadFirstSegmentationDir_ was a static).
///   • No dual local/remote code paths tangled together.
///   • Segmentations are value types, not shared_ptr soup.
///   • Clear ownership: VolumePkg owns Volumes and Segmentations.

#include "volume.hpp"
#include "surface.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc4d {

// ---------------------------------------------------------------------------
// Segmentation — metadata + optional loaded surface
// ---------------------------------------------------------------------------
class Segmentation {
public:
    explicit Segmentation(std::filesystem::path path);

    [[nodiscard]] const std::string& id() const { return id_; }
    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

    void set_name(const std::string& n) { name_ = n; }

    // Surface access — load on demand, not in constructor.
    [[nodiscard]] bool has_surface() const { return surface_ != nullptr; }
    [[nodiscard]] QuadSurface* surface() { return surface_.get(); }
    [[nodiscard]] const QuadSurface* surface() const { return surface_.get(); }

    // Load the surface from disk. No-op if already loaded.
    void load_surface();
    void unload_surface() { surface_.reset(); }

    void save_metadata() const;

private:
    std::filesystem::path path_;
    std::string id_;
    std::string name_;
    nlohmann::json metadata_;
    std::unique_ptr<QuadSurface> surface_;
};

// ---------------------------------------------------------------------------
// VolumePkg
// ---------------------------------------------------------------------------
class VolumePkg {
public:
    explicit VolumePkg(std::filesystem::path root);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const std::filesystem::path& path() const { return root_; }

    // ---- Volumes ------------------------------------------------------------
    [[nodiscard]] std::vector<std::string> volume_ids() const;
    [[nodiscard]] Volume* volume(const std::string& id);
    [[nodiscard]] bool has_volume(const std::string& id) const;

    // ---- Segmentations ------------------------------------------------------
    [[nodiscard]] std::vector<std::string> segmentation_ids() const;
    [[nodiscard]] Segmentation* segmentation(const std::string& id);
    [[nodiscard]] bool has_segmentation(const std::string& id) const;

    // Refresh segmentation list from disk (after external changes).
    void refresh_segmentations();

private:
    std::filesystem::path root_;
    std::string name_;
    nlohmann::json config_;

    std::map<std::string, std::unique_ptr<Volume>> volumes_;
    std::map<std::string, std::unique_ptr<Segmentation>> segmentations_;

    void load_config();
    void scan_volumes();
    void scan_segmentations();
};

} // namespace vc4d
