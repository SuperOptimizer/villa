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

float beerLambert(const LayerStack& stack, const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) return 0.0f;

    // Pre-scale extinction into [0,255] domain so we avoid per-layer /255.
    const float extinctionScaled = params.blExtinction / 255.0f;
    const float emissionScaled = params.blEmission / 255.0f;

    float transmittance = 1.0f;
    float accumulatedColor = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        const float value = stack.values[i];

        if (value < 0.255f) continue;  // ~0.001 * 255

        const float emission = value * emissionScaled;
        const float layerTransmittance = std::exp(-extinctionScaled * value);

        accumulatedColor += emission * transmittance * (1.0f - layerTransmittance);
        transmittance *= layerTransmittance;

        if (transmittance < 0.001f) break;
    }

    accumulatedColor += params.blAmbient * transmittance;
    return std::min(255.0f, accumulatedColor * 255.0f);
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
        case utils::CompositingMethod::mean:
            return CompositeMethod::mean(stack);
        case utils::CompositingMethod::max:
            return CompositeMethod::max(stack);
        case utils::CompositingMethod::min:
            return CompositeMethod::min(stack);
        case utils::CompositingMethod::alpha:
            return CompositeMethod::alpha(stack, params);
        case utils::CompositingMethod::beer_lambert:
            return CompositeMethod::beerLambert(stack, params);
        case utils::CompositingMethod::dvr:
            return utils::composite_dvr(
                std::span<const float>(stack.values.data(), stack.validCount),
                params.dvrAmbient);
        case utils::CompositingMethod::first_hit_iso:
            return utils::composite_first_hit_iso(
                std::span<const float>(stack.values.data(), stack.validCount),
                float(params.isoCutoff));
        case utils::CompositingMethod::dev_from_mean:
            return utils::composite_dev_from_mean(
                std::span<const float>(stack.values.data(), stack.validCount),
                float(params.isoCutoff));
        case utils::CompositingMethod::emission_dvr:
            return utils::composite_emission_dvr(
                std::span<const float>(stack.values.data(), stack.validCount));
        case utils::CompositingMethod::max_above_iso:
            return utils::composite_max_above_iso(
                std::span<const float>(stack.values.data(), stack.validCount),
                float(params.isoCutoff));
        case utils::CompositingMethod::gamma_weighted:
            return utils::composite_gamma_weighted(
                std::span<const float>(stack.values.data(), stack.validCount),
                float(params.isoCutoff));
        case utils::CompositingMethod::gradient_mag:
            return utils::composite_gradient_mag(
                std::span<const float>(stack.values.data(), stack.validCount));
        case utils::CompositingMethod::pbr_iso:
            return utils::composite_first_hit_iso(
                std::span<const float>(stack.values.data(), stack.validCount),
                float(params.isoCutoff));
        case utils::CompositingMethod::shaded_dvr:
            return utils::composite_dvr(
                std::span<const float>(stack.values.data(), stack.validCount),
                params.dvrAmbient);
    }

    return CompositeMethod::mean(stack);
}

