#include "TiledViewerCamera.hpp"
#include <algorithm>

void TiledViewerCamera::recalcPyramidLevel(int numScales)
{
    if (numScales <= 0) {
        dsScaleIdx = 0;
        dsScale = 1.0f;
        return;
    }

    const float maxScale = (numScales >= 2) ? 0.5f : 1.0f;
    const float minScale = std::pow(2.0f, 1.0f - numScales);

    if (scale >= maxScale) {
        dsScaleIdx = 0;
    } else if (scale < minScale) {
        dsScaleIdx = numScales - 1;
    } else {
        dsScaleIdx = static_cast<int>(std::round(-std::log2(scale)));
    }

    if (downscaleOverride > 0) {
        dsScaleIdx += downscaleOverride;
        dsScaleIdx = std::min(dsScaleIdx, numScales - 1);
    }

    dsScale = std::pow(2.0f, -dsScaleIdx);
}

float TiledViewerCamera::roundScale(float s)
{
    if (std::abs(s - std::round(std::log2(s))) < 0.02f) {
        s = std::pow(2.0f, std::round(std::log2(s)));
    }
    return std::clamp(s, MIN_SCALE, MAX_SCALE);
}
