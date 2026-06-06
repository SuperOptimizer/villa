#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>
#include <span>

#include <utils/compositing.hpp>

namespace CompositeMethod {

float mean(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) return 0.0f;
    return utils::composite_mean(
        std::span<const float>(stack.values.data(), stack.validCount));
}

float max(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) return 0.0f;
    return utils::composite_max(
        std::span<const float>(stack.values.data(), stack.validCount));
}

float min(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) return 255.0f;
    return utils::composite_min(
        std::span<const float>(stack.values.data(), stack.validCount));
}

float alpha(const LayerStack& stack, const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) return 0.0f;

    // Scale thresholds to [0,255] range to avoid per-layer normalization.
    // composite_alpha computes (density - alpha_min) / (alpha_max - alpha_min),
    // so scaling both min/max by 255 cancels the layer's [0,255] range.
    // alpha_cutoff is compared against accumulated alpha (already [0,1]), not
    // layer values, so it must NOT be scaled.
    float result = utils::composite_alpha(
        std::span<const float>(stack.values.data(), stack.validCount),
        params.alphaMin * 255.0f, params.alphaMax * 255.0f,
        params.alphaOpacity, params.alphaCutoff);
    return result * 255.0f;
}

} // namespace CompositeMethod

float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) return 0.0f;

    // Use utils enum-based dispatch for simple methods
    auto method = utils::parse_compositing_method(params.method);

    switch (method) {
        case utils::CompositingMethod::max:
            return CompositeMethod::max(stack);
        case utils::CompositingMethod::min:
            return CompositeMethod::min(stack);
        case utils::CompositingMethod::alpha:
            return CompositeMethod::alpha(stack, params);
        case utils::CompositingMethod::mean:
        default:
            return CompositeMethod::mean(stack);
    }
}

bool methodRequiresLayerStorage(const std::string& method) noexcept
{
    return utils::method_requires_storage(utils::parse_compositing_method(method));
}

void buildTfLut256(bool enabled,
                   uint8_t x1, uint8_t y1,
                   uint8_t x2, uint8_t y2,
                   uint8_t lut[256]) noexcept
{
    if (!enabled) {
        for (int i = 0; i < 256; ++i) lut[i] = uint8_t(i);
        return;
    }
    // Sort the two middle knots by x so the PL function is monotone in x.
    if (x1 > x2) { std::swap(x1, x2); std::swap(y1, y2); }
    // Four segments: [0,x1] → [0,y1], [x1,x2] → [y1,y2], [x2,255] → [y2,255].
    // Each segment is a linear interpolation; degenerate runs collapse to a
    // step by short-circuiting the denominator.
    auto lerp = [](float x, float x0, float x1, float y0, float y1) {
        const float d = x1 - x0;
        if (d <= 0.f) return y0;
        const float t = (x - x0) / d;
        return y0 + t * (y1 - y0);
    };
    for (int i = 0; i < 256; ++i) {
        float y;
        if (i <= int(x1))      y = lerp(float(i), 0.f,      float(x1), 0.f,      float(y1));
        else if (i <= int(x2)) y = lerp(float(i), float(x1), float(x2), float(y1), float(y2));
        else                   y = lerp(float(i), float(x2), 255.f,     float(y2), 255.f);
        if (y < 0.f) y = 0.f;
        if (y > 255.f) y = 255.f;
        lut[i] = uint8_t(y + 0.5f);
    }
}

