#pragma once

#include <nlohmann/json_fwd.hpp>

struct SurfaceMeta;

namespace vc::meta {

SurfaceMeta parseFromJson(const nlohmann::json& j);
nlohmann::json toJson(const SurfaceMeta& m);

}  // namespace vc::meta
