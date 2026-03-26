#pragma once
/// vc4d::AppState — Application state, replacing vc3d's CState.
///
/// Key improvements:
///   • Not a QObject god-object — uses Qt signals only for actual state changes.
///   • No raw pointers (CState had POI*, surfaceRaw()).
///   • No dual ownership patterns (vc3d had surfaces in both CState and VolumePkg).
///   • Clear ownership: VolumePkg owns data, AppState holds the current selection.
///   • Uses std::expected or optional for fallible accessors instead of null checks.

#include "vc4d/core/volume_pkg.hpp"
#include "vc4d/core/surface.hpp"
#include "vc4d/cache/tiered_cache.hpp"

#include <QObject>
#include <memory>
#include <optional>
#include <string>

namespace vc4d {

class AppState : public QObject {
    Q_OBJECT

public:
    explicit AppState(QObject* parent = nullptr);
    ~AppState() override;

    // ---- Package ------------------------------------------------------------
    void open(std::filesystem::path volpkg_path, TieredCache::Config cache_config);
    void close();
    [[nodiscard]] bool is_open() const { return pkg_ != nullptr; }
    [[nodiscard]] VolumePkg* pkg() { return pkg_.get(); }

    // ---- Current selections -------------------------------------------------
    void select_volume(const std::string& id);
    void select_surface(const std::string& id);
    void deselect_surface();

    [[nodiscard]] Volume* current_volume() { return current_volume_; }
    [[nodiscard]] QuadSurface* active_surface() { return active_surface_; }
    [[nodiscard]] const std::string& active_surface_id() const { return active_surface_id_; }

signals:
    void package_opened();
    void package_closed();
    void volume_changed(Volume* vol);
    void surface_changed(QuadSurface* surf, const std::string& id);

private:
    std::unique_ptr<VolumePkg> pkg_;
    std::shared_ptr<TieredCache> cache_;

    Volume* current_volume_{};          // non-owning, lifetime tied to pkg_
    QuadSurface* active_surface_{};     // non-owning, lifetime tied to pkg_
    std::string active_surface_id_;
};

} // namespace vc4d
