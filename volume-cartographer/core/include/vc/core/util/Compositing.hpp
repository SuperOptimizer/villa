#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <array>
#include <vector>
#include <cstdint>

// Parameters for multi-layer compositing
struct CompositeParams {
    // Compositing method:
    // Basic: "mean", "max", "min", "median", "alpha"
    // Statistical: "stddev", "range", "localContrast", "entropy"
    // Edge detection: "gradient", "gradientSum", "laplacian", "sobel"
    // Advanced: "percentile", "weightedMean", "peakCount", "thresholdCount"
    std::string method = "mean";

    // Alpha compositing parameters
    float alphaMin = 0.0f;
    float alphaMax = 1.0f;
    float alphaOpacity = 1.0f;
    float alphaCutoff = 1.0f;

    // Gradient parameters
    float gradientScale = 2.0f;      // Output multiplier (higher = more contrast)

    // Std Dev parameters
    float stddevScale = 2.0f;        // Output multiplier (higher = more contrast)

    // Laplacian parameters
    float laplacianScale = 2.0f;     // Output multiplier (higher = more contrast)

    // Range parameters
    float rangeScale = 1.0f;         // Output multiplier for range method

    // Gradient Sum parameters
    float gradientSumScale = 1.0f;   // Output multiplier for gradient sum

    // Sobel parameters
    float sobelScale = 2.0f;         // Output multiplier for Sobel edge detection

    // Local Contrast parameters
    float localContrastScale = 255.0f;  // Output multiplier (result is 0-1, scale to 0-255)

    // Entropy parameters
    float entropyScale = 32.0f;      // Output multiplier (entropy typically 0-8 bits)

    // Peak Detection parameters
    float peakThreshold = 10.0f;     // Minimum height difference to count as a peak
    float peakCountScale = 25.0f;    // Output multiplier for peak count

    // Threshold Count parameters
    float countThreshold = 128.0f;   // Value threshold for counting
    float thresholdCountScale = 15.0f; // Output multiplier for threshold count

    // Percentile parameters
    float percentile = 0.5f;         // 0.0 = min, 0.5 = median, 1.0 = max

    // Weighted Mean parameters
    float weightedMeanSigma = 0.5f;  // Gaussian sigma (fraction of layer count)

    // Pre-processing
    bool histogramEqualize = false;  // Apply contrast enhancement to input values
    bool use3DGLCAE = false;         // Use full 3D Global-Local Contrast Adaptive Enhancement
                                     // (requires histogramEqualize = true)
    float glcaeClipLimit = 2.0f;     // CLAHE clip limit for 3D GLCAE local enhancement
    int glcaeTileSize = 16;          // Cubic tile size for 3D GLCAE local enhancement
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

// Basic methods
float mean(const LayerStack& stack);
float max(const LayerStack& stack);
float min(const LayerStack& stack);
float median(const LayerStack& stack);
float alpha(const LayerStack& stack, const CompositeParams& params);

// Statistical methods
// Standard deviation: measure of variance across layers
float stddev(const LayerStack& stack, const CompositeParams& params);

// Range: max - min, measures total variation
float range(const LayerStack& stack, const CompositeParams& params);

// Local Contrast: (max - min) / (max + min), normalized contrast
float localContrast(const LayerStack& stack, const CompositeParams& params);

// Entropy: measure of randomness/disorder in layer values
float entropy(const LayerStack& stack, const CompositeParams& params);

// Edge detection methods
// Gradient magnitude: max absolute difference between adjacent layers
float gradient(const LayerStack& stack, const CompositeParams& params);

// Gradient Sum: sum of all absolute differences between adjacent layers
float gradientSum(const LayerStack& stack, const CompositeParams& params);

// Laplacian (second derivative): max absolute second difference
float laplacian(const LayerStack& stack, const CompositeParams& params);

// Sobel: weighted 3-point gradient kernel [-1, 0, 1] with neighbor weighting
float sobel(const LayerStack& stack, const CompositeParams& params);

// Advanced methods
// Percentile: returns value at specified percentile (0.5 = median)
float percentile(const LayerStack& stack, const CompositeParams& params);

// Weighted Mean: Gaussian-weighted mean centered on middle layer
float weightedMean(const LayerStack& stack, const CompositeParams& params);

// Peak Count: count local maxima in z-direction
float peakCount(const LayerStack& stack, const CompositeParams& params);

// Threshold Count: count layers exceeding threshold
float thresholdCount(const LayerStack& stack, const CompositeParams& params);

} // namespace CompositeMethod

// Build histogram equalization LUT using CLAHE algorithm
// Only considers non-zero values for contrast calculation
// histogram: input histogram (will be modified by clipping)
// totalSamples: number of non-zero samples
// Returns: 256-entry lookup table mapping input values to equalized output
std::array<uint8_t, 256> buildCLAHELookupTable(
    std::array<int, 256>& histogram,
    int totalSamples
);

// Global-Local Fusion Contrast Enhancement
// Combines global histogram equalization with local CLAHE using adaptive fusion weights.
// Based on: "A Global and Local Contrast Stretching Based Method for Enhancement"
//
// Algorithm:
// 1. Global: Optimal histogram blending with uniform distribution (Brent optimization)
// 2. Local: CLAHE for local contrast enhancement
// 3. Fusion: Adaptive weighting based on Laplacian (detail) and brightness
//
// Parameters:
//   input: grayscale image (CV_8UC1)
//   claheClipLimit: CLAHE clip limit (default 2.0)
//   claheTileSize: CLAHE tile size (default 8x8)
// Returns: contrast-enhanced grayscale image (CV_8UC1)
cv::Mat globalLocalFusionEnhance(
    const cv::Mat& input,
    float claheClipLimit = 2.0f,
    int claheTileSize = 8
);

// Build global equalization LUT using optimal histogram blending
// Finds optimal lambda to blend input histogram with uniform distribution
// such that the mapping function minimizes information loss.
// histogram: normalized histogram (sum = 1.0)
// Returns: 256-entry lookup table
std::array<uint8_t, 256> buildGlobalEqualizationLUT(
    const std::array<float, 256>& histogram
);

// 3D Global-Local Contrast Adaptive Enhancement (3D GLCAE)
// Applies contrast enhancement to a 3D volume slab (width × height × depth)
// Used for enhancing layer stacks before compositing.
//
// Algorithm:
// 1. Global: Compute histogram of entire 3D slab, optimal lambda blending
// 2. Local: 3D CLAHE with cubic tiles in x, y, z dimensions
// 3. Fusion: 3D Laplacian magnitude and brightness-based weight blending
//
// Parameters:
//   volume: 3D volume as flat array [z][y][x] ordering, size = width*height*depth
//   width, height, depth: dimensions of the volume
//   isoCutoff: values below this are treated as zero (not enhanced)
//   claheClipLimit: clip limit for local CLAHE (default 2.0)
//   tileSize: cubic tile size for local CLAHE (default 16)
// Returns: enhanced volume (same size as input)
std::vector<uint8_t> apply3DGLCAE(
    const std::vector<uint8_t>& volume,
    int width, int height, int depth,
    uint8_t isoCutoff = 0,
    float claheClipLimit = 2.0f,
    int tileSize = 16
);

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
