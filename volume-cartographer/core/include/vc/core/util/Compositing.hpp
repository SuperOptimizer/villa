#pragma once

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <numbers>

// Parameters for multi-layer compositing
struct CompositeParams {
    // Compositing method: "mean", "max", "min", "alpha"
    std::string method = "mean";

    // Alpha compositing parameters
    float alphaMin = 0.0f;
    float alphaMax = 1.0f;
    float alphaOpacity = 1.0f;
    float alphaCutoff = 1.0f;

    // Pre-processing
    uint8_t isoCutoff = 0;           // Highpass filter: values below this are set to 0

    // Per-ray layer preprocess (applied to the N sampled composite layers
    // for each pixel before the composite method runs). Cancels z-axis
    // brightness drift so the composite averages evenly across a ray that
    // crosses bright and dark strata.
    //
    // preNormalizeLayers: min-max stretch the N layer values to [0, 255].
    //   Preserves per-ray structure, eliminates absolute brightness offset.
    // preHistEqLayers:   CDF-based histogram equalization over the N layer
    //   values. Flattens per-ray contrast; strongest effect, can clip out
    //   true structure if the ray is genuinely uniform.
    // Both can be chained (normalize → equalize).
    bool preNormalizeLayers = false;
    bool preHistEqLayers = false;

    // Piecewise-linear 4-knot transfer functions with fixed endpoints at
    // (0,0) and (255,255); only the two middle knots (x1,y1)(x2,y2) are
    // user-settable. When the enable flag is off, the LUT is identity.
    //
    // preTf:  applied to every sampled u8 voxel BEFORE layer storage /
    //         preprocess / compositing. Lets you cut air (y below some x),
    //         isolate an intensity band, or compress dynamic range before
    //         per-ray normalization runs.
    // postTf: applied to the final composite output value (before the
    //         existing 2D postprocess stages: stretch).
    //         Lets you remap the composite's output intensity curve.
    bool preTfEnabled = false;
    uint8_t preTfX1 = 85, preTfY1 = 85;
    uint8_t preTfX2 = 170, preTfY2 = 170;
    bool postTfEnabled = false;
    uint8_t postTfX1 = 85, postTfY1 = 85;
    uint8_t postTfX2 = 170, postTfY2 = 170;

    bool operator==(const CompositeParams&) const = default;
};

// Consolidated rendering settings for composite mode (Qt-free)
struct CompositeRenderSettings {
    bool enabled = false;
    int layersFront = 8;
    int layersBehind = 0;
    bool reverseDirection = false;

    bool planeEnabled = false;
    int planeLayersFront = 4;
    int planeLayersBehind = 4;

    CompositeParams params;  // method, alpha, isoCutoff

    // Postprocessing (applied after composite render)
    bool postStretchValues = false;

    bool operator==(const CompositeRenderSettings&) const = default;
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

// Build a 256-entry u8→u8 LUT from a 4-knot piecewise-linear transfer
// function with implicit endpoints (0,0) and (255,255). When `enabled` is
// false, writes the identity mapping. Knot x coordinates are clamped and
// sorted internally, so the caller does not need to pre-sort; degenerate
// runs (x1 == x2) collapse to a step. Safe for tight rendering loops — a
// 256-byte array fits trivially in L1D.
void buildTfLut256(bool enabled,
                   uint8_t x1, uint8_t y1,
                   uint8_t x2, uint8_t y2,
                   uint8_t lut[256]) noexcept;
