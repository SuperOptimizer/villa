#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/matx.hpp>

// Enum for fast composite method dispatch (avoids string comparisons in hot loops)
enum class CompositeMethodType : uint8_t {
    Mean,
    Max,
    Min,
    Alpha,
    BeerLambert
};

// Convert string to enum (call once, not per-pixel)
[[nodiscard]] inline CompositeMethodType parseCompositeMethod(const std::string& s) noexcept {
    if (s == "max") return CompositeMethodType::Max;
    if (s == "min") return CompositeMethodType::Min;
    if (s == "alpha") return CompositeMethodType::Alpha;
    if (s == "beerLambert") return CompositeMethodType::BeerLambert;
    return CompositeMethodType::Mean;
}

// Parameters for multi-layer compositing
struct CompositeParams {
    // Compositing method: "mean", "max", "min", "alpha", "beerLambert"
    std::string method = "mean";
    CompositeMethodType methodType = CompositeMethodType::Mean;  // Pre-resolved for hot paths

    // Call after setting method string to update the enum
    void resolveMethodType() noexcept { methodType = parseCompositeMethod(method); }

    // Alpha compositing parameters
    float alphaMin = 0.0f;
    float alphaMax = 1.0f;
    float alphaOpacity = 1.0f;
    float alphaCutoff = 1.0f;

    // Beer-Lambert parameters (volume rendering with emission + absorption)
    float blExtinction = 1.5f;        // Absorption coefficient (higher = more opaque)
    float blEmission = 1.5f;          // Emission scale (higher = brighter)
    float blAmbient = 0.1f;           // Ambient light (background illumination)

    // Directional lighting parameters
    bool lightingEnabled = false;     // Enable surface lighting
    float lightAzimuth = 45.0f;       // Light direction azimuth (degrees, 0=right, 90=up)
    float lightElevation = 45.0f;     // Light direction elevation (degrees above horizon)
    float lightDiffuse = 0.7f;        // Diffuse lighting strength (0-1)
    float lightAmbient = 0.3f;        // Ambient lighting (0-1, ensures shadows aren't pure black)

    // Pre-processing
    uint8_t isoCutoff = 0;           // Highpass filter: values below this are set to 0
};

// Layer values for a single pixel across all layers
// Used by compositing methods to process per-pixel data
struct LayerStack {
    std::vector<float> values;  // Values at each layer (after cutoff/equalization)
    int validCount = 0;         // Number of valid (sampled) layers
};

// Compositing method interface
// Each method takes a stack of layer values and returns a single output value
namespace CompositeMethod {

float mean(const LayerStack& stack) noexcept;
float max(const LayerStack& stack) noexcept;
float min(const LayerStack& stack) noexcept;
float alpha(const LayerStack& stack, const CompositeParams& params) noexcept;
float beerLambert(const LayerStack& stack, const CompositeParams& params) noexcept;

} // namespace CompositeMethod

// Apply compositing to a single pixel's layer stack
// Returns the final composited value (0-255)
float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params
) noexcept;

// Utility: check if method requires all layer values to be stored
// (as opposed to running accumulator like max/min)
bool methodRequiresLayerStorage(const std::string& method) noexcept;

[[nodiscard]] inline bool methodRequiresLayerStorage(CompositeMethodType t) noexcept {
    return t != CompositeMethodType::Max && t != CompositeMethodType::Min && t != CompositeMethodType::Mean;
}

// Utility: get list of available compositing methods
std::vector<std::string> availableCompositeMethods();

// Compute directional lighting factor for a surface normal
// Returns a multiplier (0-1) based on Lambertian diffuse lighting
// normal: surface normal (should be normalized)
// params: contains light direction and strength settings
float computeLightingFactor(const cv::Vec3f& normal, const CompositeParams& params) noexcept;
