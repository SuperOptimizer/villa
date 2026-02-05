#pragma once

#include <nlohmann/json_fwd.hpp>

struct TracerParams;

namespace vc::tracer {
TracerParams parseFromJson(const nlohmann::json& j);
nlohmann::json toJson(const TracerParams& p);
void applyJsonOverlay(TracerParams& base, const nlohmann::json& overlay);
}  // namespace vc::tracer
