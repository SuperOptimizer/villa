#include "vc/core/util/Compositing.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>

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

float median(const LayerStack& stack)
{
    if (stack.validCount == 0) return 0.0f;
    if (stack.validCount == 1) return stack.values[0];

    // Make a copy for sorting
    std::vector<float> sorted(stack.values.begin(), stack.values.begin() + stack.validCount);
    std::sort(sorted.begin(), sorted.end());

    int mid = stack.validCount / 2;
    if (stack.validCount % 2 == 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2.0f;
    } else {
        return sorted[mid];
    }
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

float gradient(const LayerStack& stack, const CompositeParams& params)
{
    // Gradient magnitude: max absolute difference between adjacent layers
    // Detects edges/transitions in the z-direction
    if (stack.validCount < 2) return 0.0f;

    float maxGrad = 0.0f;
    for (int i = 1; i < stack.validCount; i++) {
        float grad = std::abs(stack.values[i] - stack.values[i - 1]);
        if (grad > maxGrad) {
            maxGrad = grad;
        }
    }
    return std::min(255.0f, maxGrad * params.gradientScale);
}

float stddev(const LayerStack& stack, const CompositeParams& params)
{
    // Standard deviation across layers
    // High values = textured/varying, low values = homogeneous
    if (stack.validCount < 2) return 0.0f;

    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < stack.validCount; i++) {
        sum += stack.values[i];
    }
    float mean = sum / static_cast<float>(stack.validCount);

    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < stack.validCount; i++) {
        float diff = stack.values[i] - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(stack.validCount);

    // Standard deviation, scaled by user parameter
    float sd = std::sqrt(variance);
    return std::min(255.0f, sd * params.stddevScale);
}

float laplacian(const LayerStack& stack, const CompositeParams& params)
{
    // Laplacian (second derivative): detects edges more sharply than gradient
    // Computes d²f/dz² ≈ f[i+1] - 2*f[i] + f[i-1] for each interior point
    // Returns max absolute value (or could sum them)
    if (stack.validCount < 3) return 0.0f;

    float maxLaplacian = 0.0f;
    for (int i = 1; i < stack.validCount - 1; i++) {
        float lapl = stack.values[i + 1] - 2.0f * stack.values[i] + stack.values[i - 1];
        float absLapl = std::abs(lapl);
        if (absLapl > maxLaplacian) {
            maxLaplacian = absLapl;
        }
    }

    // Scale by user parameter
    return std::min(255.0f, maxLaplacian * params.laplacianScale);
}

float range(const LayerStack& stack, const CompositeParams& params)
{
    // Range: max - min across layers
    // High values indicate significant variation
    if (stack.validCount == 0) return 0.0f;

    float minVal = stack.values[0];
    float maxVal = stack.values[0];
    for (int i = 1; i < stack.validCount; i++) {
        if (stack.values[i] < minVal) minVal = stack.values[i];
        if (stack.values[i] > maxVal) maxVal = stack.values[i];
    }

    return std::min(255.0f, (maxVal - minVal) * params.rangeScale);
}

float gradientSum(const LayerStack& stack, const CompositeParams& params)
{
    // Sum of all absolute differences between adjacent layers
    // Captures total "bumpiness" across layers
    if (stack.validCount < 2) return 0.0f;

    float sum = 0.0f;
    for (int i = 1; i < stack.validCount; i++) {
        sum += std::abs(stack.values[i] - stack.values[i - 1]);
    }

    return std::min(255.0f, sum * params.gradientSumScale);
}

float sobel(const LayerStack& stack, const CompositeParams& params)
{
    // Sobel-style gradient using [-1, 0, 1] kernel with neighbor weighting
    // More sophisticated edge detection than simple gradient
    if (stack.validCount < 3) return 0.0f;

    float maxSobel = 0.0f;
    for (int i = 1; i < stack.validCount - 1; i++) {
        // Sobel kernel: weighted difference with neighbors
        // Approximates [-1, 0, 1] convolved with [1, 2, 1] smoothing
        float sobelVal = std::abs(stack.values[i + 1] - stack.values[i - 1]);
        if (sobelVal > maxSobel) {
            maxSobel = sobelVal;
        }
    }

    return std::min(255.0f, maxSobel * params.sobelScale);
}

float localContrast(const LayerStack& stack, const CompositeParams& params)
{
    // Local Contrast: (max - min) / (max + min)
    // Normalized contrast measure, good for detecting texture regardless of brightness
    if (stack.validCount == 0) return 0.0f;

    float minVal = stack.values[0];
    float maxVal = stack.values[0];
    for (int i = 1; i < stack.validCount; i++) {
        if (stack.values[i] < minVal) minVal = stack.values[i];
        if (stack.values[i] > maxVal) maxVal = stack.values[i];
    }

    float sum = maxVal + minVal;
    if (sum < 1.0f) return 0.0f;  // Avoid division by zero

    float contrast = (maxVal - minVal) / sum;
    return std::min(255.0f, contrast * params.localContrastScale);
}

float entropy(const LayerStack& stack, const CompositeParams& params)
{
    // Entropy: measure of randomness/disorder in layer values
    // Uses histogram of values to compute Shannon entropy
    if (stack.validCount < 2) return 0.0f;

    // Build a simple histogram (16 bins for efficiency)
    const int numBins = 16;
    std::array<int, numBins> hist = {0};

    for (int i = 0; i < stack.validCount; i++) {
        int bin = static_cast<int>(stack.values[i] / 256.0f * numBins);
        if (bin >= numBins) bin = numBins - 1;
        hist[bin]++;
    }

    // Compute Shannon entropy: H = -sum(p * log2(p))
    float entropy = 0.0f;
    float invCount = 1.0f / static_cast<float>(stack.validCount);
    for (int i = 0; i < numBins; i++) {
        if (hist[i] > 0) {
            float p = static_cast<float>(hist[i]) * invCount;
            entropy -= p * std::log2(p);
        }
    }

    return std::min(255.0f, entropy * params.entropyScale);
}

float percentile(const LayerStack& stack, const CompositeParams& params)
{
    // Returns value at specified percentile (0.0 = min, 0.5 = median, 1.0 = max)
    if (stack.validCount == 0) return 0.0f;
    if (stack.validCount == 1) return stack.values[0];

    // Make a copy for sorting
    std::vector<float> sorted(stack.values.begin(), stack.values.begin() + stack.validCount);
    std::sort(sorted.begin(), sorted.end());

    float idx = params.percentile * (stack.validCount - 1);
    int lowIdx = static_cast<int>(idx);
    int highIdx = std::min(lowIdx + 1, stack.validCount - 1);
    float frac = idx - lowIdx;

    // Linear interpolation between adjacent values
    return sorted[lowIdx] * (1.0f - frac) + sorted[highIdx] * frac;
}

float weightedMean(const LayerStack& stack, const CompositeParams& params)
{
    // Gaussian-weighted mean centered on middle layer
    // Focuses on the "core" of the material
    if (stack.validCount == 0) return 0.0f;
    if (stack.validCount == 1) return stack.values[0];

    float center = (stack.validCount - 1) / 2.0f;
    float sigma = params.weightedMeanSigma * stack.validCount;
    if (sigma < 0.1f) sigma = 0.1f;

    float weightSum = 0.0f;
    float valueSum = 0.0f;

    for (int i = 0; i < stack.validCount; i++) {
        float dist = static_cast<float>(i) - center;
        float weight = std::exp(-(dist * dist) / (2.0f * sigma * sigma));
        weightSum += weight;
        valueSum += weight * stack.values[i];
    }

    if (weightSum < 0.001f) return mean(stack);
    return valueSum / weightSum;
}

float peakCount(const LayerStack& stack, const CompositeParams& params)
{
    // Count local maxima (peaks) in z-direction
    // A peak is where value[i] > value[i-1] and value[i] > value[i+1]
    if (stack.validCount < 3) return 0.0f;

    int peaks = 0;
    for (int i = 1; i < stack.validCount - 1; i++) {
        float prev = stack.values[i - 1];
        float curr = stack.values[i];
        float next = stack.values[i + 1];

        // Check if this is a local maximum with sufficient height
        if (curr > prev + params.peakThreshold && curr > next + params.peakThreshold) {
            peaks++;
        }
    }

    return std::min(255.0f, static_cast<float>(peaks) * params.peakCountScale);
}

float thresholdCount(const LayerStack& stack, const CompositeParams& params)
{
    // Count how many layers exceed the threshold
    // High counts indicate dense material (e.g., ink)
    if (stack.validCount == 0) return 0.0f;

    int count = 0;
    for (int i = 0; i < stack.validCount; i++) {
        if (stack.values[i] >= params.countThreshold) {
            count++;
        }
    }

    return std::min(255.0f, static_cast<float>(count) * params.thresholdCountScale);
}

} // namespace CompositeMethod
float compositeLayerStack(
    const LayerStack& stack,
    const CompositeParams& params)
{
    if (stack.validCount == 0) return 0.0f;

    const std::string& method = params.method;

    // Basic methods
    if (method == "mean") {
        return CompositeMethod::mean(stack);
    } else if (method == "max") {
        return CompositeMethod::max(stack);
    } else if (method == "min") {
        return CompositeMethod::min(stack);
    } else if (method == "median") {
        return CompositeMethod::median(stack);
    } else if (method == "alpha") {
        return CompositeMethod::alpha(stack, params);
    }
    // Statistical methods
    else if (method == "stddev") {
        return CompositeMethod::stddev(stack, params);
    } else if (method == "range") {
        return CompositeMethod::range(stack, params);
    } else if (method == "localContrast") {
        return CompositeMethod::localContrast(stack, params);
    } else if (method == "entropy") {
        return CompositeMethod::entropy(stack, params);
    }
    // Edge detection methods
    else if (method == "gradient") {
        return CompositeMethod::gradient(stack, params);
    } else if (method == "gradientSum") {
        return CompositeMethod::gradientSum(stack, params);
    } else if (method == "laplacian") {
        return CompositeMethod::laplacian(stack, params);
    } else if (method == "sobel") {
        return CompositeMethod::sobel(stack, params);
    }
    // Advanced methods
    else if (method == "percentile") {
        return CompositeMethod::percentile(stack, params);
    } else if (method == "weightedMean") {
        return CompositeMethod::weightedMean(stack, params);
    } else if (method == "peakCount") {
        return CompositeMethod::peakCount(stack, params);
    } else if (method == "thresholdCount") {
        return CompositeMethod::thresholdCount(stack, params);
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
        // Basic
        "mean",
        "max",
        "min",
        "median",
        "alpha",
        // Statistical
        "stddev",
        "range",
        "localContrast",
        "entropy",
        // Edge detection
        "gradient",
        "gradientSum",
        "laplacian",
        "sobel",
        // Advanced
        "percentile",
        "weightedMean",
        "peakCount",
        "thresholdCount"
    };
}
