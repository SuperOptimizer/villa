#pragma once

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vc {

enum class OverlayColormapKind { OpenCv, Tint, DiscreteLut };
enum class ColormapAudience { Shared, OverlayOnly };
enum class EntryScope { SharedOnly, OverlayCompatible };

struct OverlayColormapSpec {
    std::string id;
    std::string label;
    OverlayColormapKind kind;
    ColormapAudience audience;
    int opencvCode;
    cv::Vec3f tint; // R, G, B in [0,1]
    const uint32_t* discreteLut;
};

struct OverlayColormapEntry {
    std::string label;
    std::string id;
};

const std::vector<OverlayColormapSpec>& specs() noexcept;
const OverlayColormapSpec& resolve(const std::string& id);

const std::vector<OverlayColormapEntry>& entries(EntryScope scope = EntryScope::OverlayCompatible) noexcept;

}  // namespace vc
