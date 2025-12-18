#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <cstdint>

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

float mean(const LayerStack& stack);
float max(const LayerStack& stack);
float min(const LayerStack& stack);
float alpha(const LayerStack& stack, const CompositeParams& params);

} // namespace CompositeMethod

// Apply compositing to a single pixel's layer stack
// Returns the final composited value (0-255)
float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params
);

// Utility: check if method requires all layer values to be stored
// (as opposed to running accumulator like max/min)
bool methodRequiresLayerStorage(const std::string& method);

// Utility: get list of available compositing methods
std::vector<std::string> availableCompositeMethods();
