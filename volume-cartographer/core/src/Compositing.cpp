#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>

namespace CompositeMethod {

[[gnu::always_inline]] inline float mean(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) [[unlikely]] return 0.0f;

    const float* __restrict__ values = stack.values.data();
    const int n = stack.validCount;

    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += values[i];
    }
    // Multiply by reciprocal instead of divide
    return sum * (1.0f / static_cast<float>(n));
}

[[gnu::always_inline]] inline float max(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) [[unlikely]] return 0.0f;

    const float* __restrict__ values = stack.values.data();
    const int n = stack.validCount;

    float maxVal = values[0];
    #pragma omp simd reduction(max:maxVal)
    for (int i = 1; i < n; i++) {
        maxVal = std::max(maxVal, values[i]);
    }
    return maxVal;
}

[[gnu::always_inline]] inline float min(const LayerStack& stack) noexcept
{
    if (stack.validCount == 0) [[unlikely]] return 255.0f;

    const float* __restrict__ values = stack.values.data();
    const int n = stack.validCount;

    float minVal = values[0];
    #pragma omp simd reduction(min:minVal)
    for (int i = 1; i < n; i++) {
        minVal = std::min(minVal, values[i]);
    }
    return minVal;
}

[[gnu::always_inline]] inline float alpha(const LayerStack& stack, const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) [[unlikely]] return 0.0f;

    const float alphaScale = 1.0f / (255.0f * (params.alphaMax - params.alphaMin));
    const float alphaOffset = params.alphaMin / (params.alphaMax - params.alphaMin);

    float alpha = 0.0f;
    float valueAcc = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        float normalized = stack.values[i] * alphaScale - alphaOffset;
        if (normalized <= 0.0f) [[unlikely]] continue;
        if (normalized > 1.0f) [[unlikely]] normalized = 1.0f;
        if (alpha >= params.alphaCutoff) [[unlikely]] break;

        float opacity = normalized * params.alphaOpacity;
        if (opacity > 1.0f) [[unlikely]] opacity = 1.0f;
        float weight = (1.0f - alpha) * opacity;
        valueAcc += weight * normalized;
        alpha += weight;
    }

    return valueAcc * 255.0f;
}

[[gnu::always_inline]] inline float beerLambert(const LayerStack& stack, const CompositeParams& params) noexcept
{
    // Beer-Lambert volume rendering with emission
    // - Each voxel emits light proportional to its density (bright = visible)
    // - Each voxel absorbs light based on density (creates depth effect)
    // - Front-to-back compositing: front layers partially occlude back layers

    if (stack.validCount == 0) [[unlikely]] return 0.0f;

    const float* __restrict__ values = stack.values.data();
    const int n = stack.validCount;
    const float extinction = params.blExtinction;
    const float emissionScale = params.blEmission;
    const float ambient = params.blAmbient;

    // Pre-compute constant factors
    constexpr float inv255 = 1.0f / 255.0f;
    constexpr float densityThreshold = 0.001f;
    constexpr float transmittanceThreshold = 0.001f;

    float transmittance = 1.0f;  // Light remaining (starts at full)
    float accumulatedColor = 0.0f;  // Accumulated emitted light

    // Front-to-back compositing (layer 0 is frontmost)
    for (int i = 0; i < n; i++) {
        const float density = values[i] * inv255;

        if (density < densityThreshold) [[unlikely]] {
            continue;
        }

        // Emission: bright voxels emit light
        const float emission = density * emissionScale;

        // Beer-Lambert absorption: exp(-extinction * density)
        const float layerTransmittance = std::exp(-extinction * density);
        const float oneMinusLayerT = 1.0f - layerTransmittance;

        // Add emission weighted by current transmittance (how much light can escape)
        // FMA-friendly: accumulatedColor += (emission * oneMinusLayerT) * transmittance
        accumulatedColor += emission * oneMinusLayerT * transmittance;

        // Update transmittance for next layer
        transmittance *= layerTransmittance;

        // Early termination if nearly opaque
        if (transmittance < transmittanceThreshold) [[unlikely]] {
            break;
        }
    }

    // Add ambient light that made it through, scale to 0-255 range
    return std::min(255.0f, (accumulatedColor + ambient * transmittance) * 255.0f);
}

} // namespace CompositeMethod

float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params) noexcept
{
    if (stack.validCount == 0) [[unlikely]] return 0.0f;

    switch (params.methodType) {
        case CompositeMethodType::Mean:        return CompositeMethod::mean(stack);
        case CompositeMethodType::Max:         return CompositeMethod::max(stack);
        case CompositeMethodType::Min:         return CompositeMethod::min(stack);
        case CompositeMethodType::Alpha:       return CompositeMethod::alpha(stack, params);
        case CompositeMethodType::BeerLambert: return CompositeMethod::beerLambert(stack, params);
    }
    return CompositeMethod::mean(stack);
}

bool methodRequiresLayerStorage(const std::string& method) noexcept
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

float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params) noexcept
{
    if (!params.lightingEnabled) [[likely]] {
        return 1.0f;
    }

    // Convert azimuth and elevation to light direction vector
    // Azimuth: 0=+X (right), 90=+Y (up in screen space)
    // Elevation: angle above the XY plane (toward viewer)
    constexpr float degToRad = static_cast<float>(M_PI) / 180.0f;
    const float azimuthRad = params.lightAzimuth * degToRad;
    const float elevationRad = params.lightElevation * degToRad;

    // Light direction (pointing toward the light source)
    // Use sincos where available for efficiency
    const float cosElev = std::cos(elevationRad);
    const float sinElev = std::sin(elevationRad);
    const float cosAz = std::cos(azimuthRad);
    const float sinAz = std::sin(azimuthRad);

    const float lightX = cosAz * cosElev;
    const float lightY = sinAz * cosElev;
    const float lightZ = sinElev;

    // Compute squared length of normal for normalization
    const float nx = normal[0], ny = normal[1], nz = normal[2];
    const float lenSq = nx*nx + ny*ny + nz*nz;

    if (lenSq < 1e-8f) [[unlikely]] {
        return params.lightAmbient;
    }

    // Use inverse sqrt for normalization (avoids division)
    const float invLen = 1.0f / std::sqrt(lenSq);

    // Normalized dot product: (n/|n|) · L = (n · L) / |n| = (n · L) * invLen
    const float nDotL_unnorm = nx*lightX + ny*lightY + nz*lightZ;
    const float nDotL = std::max(0.0f, nDotL_unnorm * invLen);

    // Combine: ambient + diffuse, clamped to [0, 1]
    // FMA-friendly: ambient + diffuse * nDotL
    const float lighting = params.lightAmbient + params.lightDiffuse * nDotL;

    return std::min(1.0f, std::max(0.0f, lighting));
}
