#include "vc4d/ui/app_state.hpp"

namespace vc4d {

AppState::AppState(QObject* parent)
    : QObject(parent) {}

AppState::~AppState() = default;

void AppState::open(std::filesystem::path volpkg_path, TieredCache::Config cache_config) {
    close();

    cache_ = std::make_shared<TieredCache>(cache_config);
    pkg_ = std::make_unique<VolumePkg>(std::move(volpkg_path));

    emit package_opened();
}

void AppState::close() {
    if (!pkg_) return;

    deselect_surface();
    current_volume_ = nullptr;
    pkg_.reset();
    cache_.reset();

    emit package_closed();
}

void AppState::select_volume(const std::string& id) {
    if (!pkg_) return;

    auto* vol = pkg_->volume(id);
    if (!vol) return;

    vol->set_cache(cache_);
    current_volume_ = vol;

    emit volume_changed(vol);
}

void AppState::select_surface(const std::string& id) {
    if (!pkg_) return;

    auto* seg = pkg_->segmentation(id);
    if (!seg) return;

    seg->load_surface();
    active_surface_ = seg->surface();
    active_surface_id_ = id;

    emit surface_changed(active_surface_, id);
}

void AppState::deselect_surface() {
    active_surface_ = nullptr;
    active_surface_id_.clear();

    emit surface_changed(nullptr, "");
}

} // namespace vc4d
