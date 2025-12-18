#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>

namespace CompositeMethod {

float mean(const LayerStack& stack)
{
    if (stack.validCount == 0) return 0.0f;

    float sum = 0.0f;
    for (int i = 0; i < stack.validCount; i++) {
        sum += stack.values[i];
    }
    return sum / static_cast<float>(stack.validCount);
}

float max(const LayerStack& stack)
{
    if (stack.validCount == 0) return 0.0f;

    float maxVal = stack.values[0];
    for (int i = 1; i < stack.validCount; i++) {
        if (stack.values[i] > maxVal) {
            maxVal = stack.values[i];
        }
    }
    return maxVal;
}

float min(const LayerStack& stack)
{
    if (stack.validCount == 0) return 255.0f;

    float minVal = stack.values[0];
    for (int i = 1; i < stack.validCount; i++) {
        if (stack.values[i] < minVal) {
            minVal = stack.values[i];
        }
    }
    return minVal;
}

float alpha(const LayerStack& stack, const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;

    const float alphaScale = 1.0f / (255.0f * (params.alphaMax - params.alphaMin));
    const float alphaOffset = params.alphaMin / (params.alphaMax - params.alphaMin);

    float alpha = 0.0f;
    float valueAcc = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        float normalized = stack.values[i] * alphaScale - alphaOffset;
        if (normalized <= 0.0f) continue;
        if (normalized > 1.0f) normalized = 1.0f;
        if (alpha >= params.alphaCutoff) break;

        float opacity = normalized * params.alphaOpacity;
        if (opacity > 1.0f) opacity = 1.0f;
        float weight = (1.0f - alpha) * opacity;
        valueAcc += weight * normalized;
        alpha += weight;
    }

    return valueAcc * 255.0f;
}

float beerLambert(const LayerStack& stack, const CompositeParams& params)
{
    // Beer-Lambert volume rendering with emission
    // - Each voxel emits light proportional to its density (bright = visible)
    // - Each voxel absorbs light based on density (creates depth effect)
    // - Front-to-back compositing: front layers partially occlude back layers

    if (stack.validCount == 0) return 0.0f;

    const float extinction = params.blExtinction;
    const float emissionScale = params.blEmission;
    const float ambient = params.blAmbient;

    float transmittance = 1.0f;  // Light remaining (starts at full)
    float accumulatedColor = 0.0f;  // Accumulated emitted light

    // Front-to-back compositing (layer 0 is frontmost)
    for (int i = 0; i < stack.validCount; i++) {
        const float value = stack.values[i];
        const float density = value / 255.0f;

        if (density < 0.001f) {
            continue;
        }

        // Emission: bright voxels emit light
        const float emission = density * emissionScale;

        // Beer-Lambert absorption
        const float layerTransmittance = std::exp(-extinction * density);

        // Add emission weighted by current transmittance (how much light can escape)
        accumulatedColor += emission * transmittance * (1.0f - layerTransmittance);

        // Update transmittance for next layer
        transmittance *= layerTransmittance;

        // Early termination if nearly opaque
        if (transmittance < 0.001f) {
            break;
        }
    }

    // Add ambient light that made it through
    accumulatedColor += ambient * transmittance;

    // Scale to 0-255 range
    return std::min(255.0f, accumulatedColor * 255.0f);
}

} // namespace CompositeMethod

float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;

    const std::string& method = params.method;

    if (method == "mean") {
        return CompositeMethod::mean(stack);
    } else if (method == "max") {
        return CompositeMethod::max(stack);
    } else if (method == "min") {
        return CompositeMethod::min(stack);
    } else if (method == "alpha") {
        return CompositeMethod::alpha(stack, params);
    } else if (method == "beerLambert") {
        return CompositeMethod::beerLambert(stack, params);
    }

    // Default to mean
    return CompositeMethod::mean(stack);
}

bool methodRequiresLayerStorage(const std::string& method)
{
    // These methods need all layer values stored to compute their result
    // (can't use a running accumulator)
    // Only max, min, mean can be computed with running accumulators
    return method != "max" && method != "min" && method != "mean";
}

std::vector<std::string> availableCompositeMethods()
{
    return {
        "mean",
        "max",
        "min",
        "alpha",
        "beerLambert"
    };
}
