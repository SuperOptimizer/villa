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

std::array<uint8_t, 256> buildCLAHELookupTable(
    std::array<int, 256>& histogram,
    int totalSamples)
{
    std::array<uint8_t, 256> lut;

    // Zero values stay zero - they're not part of the equalization
    lut[0] = 0;

    if (totalSamples <= 0) {
        // Identity LUT for non-zero values
        for (int i = 1; i < 256; i++) {
            lut[i] = static_cast<uint8_t>(i);
        }
        return lut;
    }

    // CLAHE: Clip histogram and redistribute excess (only for non-zero bins)
    // Clip limit as a multiplier of average bin count
    const float clipLimitMultiplier = 2.0f;
    const int avgBinCount = totalSamples / 255;  // 255 bins (1-255)
    const int clipLimit = std::max(1, static_cast<int>(avgBinCount * clipLimitMultiplier));

    // Clip histogram and count excess (bins 1-255 only)
    int excess = 0;
    for (int i = 1; i < 256; i++) {
        if (histogram[i] > clipLimit) {
            excess += histogram[i] - clipLimit;
            histogram[i] = clipLimit;
        }
    }

    // Redistribute excess evenly across non-zero bins (1-255)
    const int redistPerBin = excess / 255;
    int residual = excess % 255;
    for (int i = 1; i < 256; i++) {
        histogram[i] += redistPerBin;
        if (residual > 0) {
            histogram[i]++;
            residual--;
        }
    }

    // Recompute total after redistribution (bins 1-255)
    int newTotal = 0;
    for (int i = 1; i < 256; i++) {
        newTotal += histogram[i];
    }

    // Build CDF and equalization LUT (only for bins 1-255)
    int cumulative = 0;
    int minCdf = 0;

    // Find first non-zero bin (starting from 1)
    for (int i = 1; i < 256; i++) {
        if (histogram[i] > 0) {
            minCdf = histogram[i];
            break;
        }
    }

    // Map non-zero values to 1-255 range (preserve 0 as 0)
    for (int i = 1; i < 256; i++) {
        cumulative += histogram[i];
        if (newTotal > minCdf) {
            lut[i] = static_cast<uint8_t>(
                1 + std::round(254.0f * (cumulative - minCdf) / (newTotal - minCdf))
            );
        } else {
            lut[i] = static_cast<uint8_t>(i);
        }
    }

    return lut;
}

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

// Helper: Build mapping table from blended histogram
// Maps input values to equalized output using cumulative distribution
static std::array<uint8_t, 256> buildMappingTable(const std::array<float, 256>& h_tilde)
{
    std::array<uint8_t, 256> t;
    float cumSum = 0.0f;
    for (int i = 0; i < 256; i++) {
        cumSum += h_tilde[i];
        // Map to 0-255 range
        t[i] = static_cast<uint8_t>(std::min(255.0f, std::ceil(255.0f * cumSum + 0.5f)));
    }
    return t;
}

// Helper: Compute the objective function for lambda optimization
// Measures maximum "collision distance" - bins that map to same output
static float computeObjective(float lambda, const std::array<float, 256>& h_i)
{
    const float h_u = 1.0f / 256.0f;  // Uniform histogram value
    const float denom = 1.0f + lambda;

    // Blend histograms: h_tilde = (1/(1+lambda)) * h_i + (lambda/(1+lambda)) * h_u
    std::array<float, 256> h_tilde;
    for (int i = 0; i < 256; i++) {
        h_tilde[i] = (h_i[i] + lambda * h_u) / denom;
    }

    // Build mapping table
    auto t = buildMappingTable(h_tilde);

    // Find maximum distance between bins that map to the same output
    float maxDist = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (h_tilde[i] <= 0.0f) continue;
        for (int j = 0; j < i; j++) {
            if (h_tilde[j] > 0.0f && t[i] == t[j]) {
                maxDist = std::max(maxDist, static_cast<float>(i - j));
            }
        }
    }

    return maxDist;
}

std::array<uint8_t, 256> buildGlobalEqualizationLUT(const std::array<float, 256>& histogram)
{
    // Brent's method for 1D optimization (simplified golden section search)
    // Find lambda that minimizes collision distance

    const float goldenRatio = 0.618033988749895f;
    float a = 0.0f;
    float b = 10.0f;  // Search range for lambda

    float x1 = b - goldenRatio * (b - a);
    float x2 = a + goldenRatio * (b - a);
    float f1 = computeObjective(x1, histogram);
    float f2 = computeObjective(x2, histogram);

    const float tol = 0.01f;
    const int maxIter = 50;

    for (int iter = 0; iter < maxIter && (b - a) > tol; iter++) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - goldenRatio * (b - a);
            f1 = computeObjective(x1, histogram);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + goldenRatio * (b - a);
            f2 = computeObjective(x2, histogram);
        }
    }

    // Use the midpoint as optimal lambda
    float optimalLambda = (a + b) / 2.0f;

    // Build final blended histogram and mapping table
    const float h_u = 1.0f / 256.0f;
    const float denom = 1.0f + optimalLambda;

    std::array<float, 256> h_tilde;
    for (int i = 0; i < 256; i++) {
        h_tilde[i] = (histogram[i] + optimalLambda * h_u) / denom;
    }

    return buildMappingTable(h_tilde);
}

// Helper: Compute 3D Laplacian magnitude at a point
static float compute3DLaplacian(
    const std::vector<uint8_t>& vol,
    int x, int y, int z,
    int w, int h, int d)
{
    // 3D Laplacian: sum of second derivatives in x, y, z
    // L = (v[x+1] - 2*v[x] + v[x-1]) + (v[y+1] - 2*v[y] + v[y-1]) + (v[z+1] - 2*v[z] + v[z-1])

    auto idx = [w, h](int x, int y, int z) { return z * w * h + y * w + x; };

    float center = static_cast<float>(vol[idx(x, y, z)]);

    float lapX = 0.0f, lapY = 0.0f, lapZ = 0.0f;

    if (x > 0 && x < w - 1) {
        lapX = vol[idx(x+1, y, z)] - 2.0f * center + vol[idx(x-1, y, z)];
    }
    if (y > 0 && y < h - 1) {
        lapY = vol[idx(x, y+1, z)] - 2.0f * center + vol[idx(x, y-1, z)];
    }
    if (z > 0 && z < d - 1) {
        lapZ = vol[idx(x, y, z+1)] - 2.0f * center + vol[idx(x, y, z-1)];
    }

    return std::abs(lapX) + std::abs(lapY) + std::abs(lapZ);
}

// Helper: Apply 3D CLAHE to a volume
// Returns locally enhanced volume
static std::vector<uint8_t> apply3DCLAHE(
    const std::vector<uint8_t>& volume,
    int width, int height, int depth,
    uint8_t isoCutoff,
    float clipLimit,
    int tileSize)
{
    std::vector<uint8_t> result(volume.size());

    // Adjust tile size to fit volume (minimum 2)
    tileSize = std::max(2, tileSize);

    // Number of tiles in each dimension
    const int numTilesX = (width + tileSize - 1) / tileSize;
    const int numTilesY = (height + tileSize - 1) / tileSize;
    const int numTilesZ = (depth + tileSize - 1) / tileSize;

    // Compute LUT for each tile
    std::vector<std::array<uint8_t, 256>> tileLUTs(numTilesX * numTilesY * numTilesZ);

    auto tileIdx = [numTilesX, numTilesY](int tx, int ty, int tz) {
        return tz * numTilesX * numTilesY + ty * numTilesX + tx;
    };

    auto volIdx = [width, height](int x, int y, int z) {
        return z * width * height + y * width + x;
    };

    // Build histogram and LUT for each tile
    for (int tz = 0; tz < numTilesZ; tz++) {
        for (int ty = 0; ty < numTilesY; ty++) {
            for (int tx = 0; tx < numTilesX; tx++) {
                // Tile bounds
                int x0 = tx * tileSize;
                int y0 = ty * tileSize;
                int z0 = tz * tileSize;
                int x1 = std::min(x0 + tileSize, width);
                int y1 = std::min(y0 + tileSize, height);
                int z1 = std::min(z0 + tileSize, depth);

                // Build histogram for this tile
                std::array<int, 256> histogram{};
                int totalSamples = 0;

                for (int z = z0; z < z1; z++) {
                    for (int y = y0; y < y1; y++) {
                        for (int x = x0; x < x1; x++) {
                            uint8_t val = volume[volIdx(x, y, z)];
                            if (val >= isoCutoff && val > 0) {
                                histogram[val]++;
                                totalSamples++;
                            }
                        }
                    }
                }

                // Build CLAHE LUT for this tile
                auto& lut = tileLUTs[tileIdx(tx, ty, tz)];
                lut[0] = 0;

                if (totalSamples > 0) {
                    // Clip limit
                    const int avgBinCount = totalSamples / 255;
                    const int clipLim = std::max(1, static_cast<int>(avgBinCount * clipLimit));

                    // Clip and redistribute
                    int excess = 0;
                    for (int i = 1; i < 256; i++) {
                        if (histogram[i] > clipLim) {
                            excess += histogram[i] - clipLim;
                            histogram[i] = clipLim;
                        }
                    }

                    const int redistPerBin = excess / 255;
                    int residual = excess % 255;
                    for (int i = 1; i < 256; i++) {
                        histogram[i] += redistPerBin;
                        if (residual > 0) { histogram[i]++; residual--; }
                    }

                    // Build CDF
                    int cumulative = 0;
                    int minCdf = 0;
                    for (int i = 1; i < 256; i++) {
                        if (histogram[i] > 0 && minCdf == 0) minCdf = histogram[i];
                        cumulative += histogram[i];
                    }

                    int newTotal = cumulative;
                    cumulative = 0;
                    for (int i = 1; i < 256; i++) {
                        cumulative += histogram[i];
                        if (newTotal > minCdf) {
                            lut[i] = static_cast<uint8_t>(1 + std::round(254.0f * (cumulative - minCdf) / (newTotal - minCdf)));
                        } else {
                            lut[i] = static_cast<uint8_t>(i);
                        }
                    }
                } else {
                    for (int i = 1; i < 256; i++) lut[i] = static_cast<uint8_t>(i);
                }
            }
        }
    }

    // Apply with nearest-neighbor tile lookup
    for (int z = 0; z < depth; z++) {
        const int tz = z / tileSize;
        for (int y = 0; y < height; y++) {
            const int ty = y / tileSize;
            for (int x = 0; x < width; x++) {
                uint8_t val = volume[volIdx(x, y, z)];

                if (val < isoCutoff || val == 0) {
                    result[volIdx(x, y, z)] = 0;
                    continue;
                }

                const int tx = x / tileSize;
                result[volIdx(x, y, z)] = tileLUTs[tileIdx(tx, ty, tz)][val];
            }
        }
    }

    return result;
}

std::vector<uint8_t> apply3DGLCAE(
    const std::vector<uint8_t>& volume,
    int width, int height, int depth,
    uint8_t isoCutoff,
    float claheClipLimit,
    int tileSize)
{
    if (volume.empty() || width <= 0 || height <= 0 || depth <= 0) {
        return volume;
    }

    const size_t totalVoxels = static_cast<size_t>(width) * height * depth;
    if (volume.size() != totalVoxels) {
        return volume;
    }

    auto volIdx = [width, height](int x, int y, int z) {
        return z * width * height + y * width + x;
    };

    // Step 1: Build global histogram and create global equalization LUT
    std::array<float, 256> globalHist{};
    int totalSamples = 0;

    for (size_t i = 0; i < totalVoxels; i++) {
        uint8_t val = volume[i];
        if (val >= isoCutoff && val > 0) {
            globalHist[val] += 1.0f;
            totalSamples++;
        }
    }

    if (totalSamples == 0) {
        return volume;  // Nothing to enhance
    }

    // Normalize histogram
    const float invTotal = 1.0f / static_cast<float>(totalSamples);
    for (int i = 0; i < 256; i++) {
        globalHist[i] *= invTotal;
    }

    // Build global LUT using optimal lambda blending
    auto globalLUT = buildGlobalEqualizationLUT(globalHist);
    globalLUT[0] = 0;

    // Step 2: Apply global enhancement
    std::vector<uint8_t> globalEnhanced(totalVoxels);
    for (size_t i = 0; i < totalVoxels; i++) {
        uint8_t val = volume[i];
        globalEnhanced[i] = (val < isoCutoff) ? 0 : globalLUT[val];
    }

    // Step 3: Apply 3D CLAHE for local enhancement
    std::vector<uint8_t> localEnhanced = apply3DCLAHE(
        volume, width, height, depth, isoCutoff, claheClipLimit, tileSize);

    // Step 4: Compute fusion weights and blend
    std::vector<uint8_t> result(totalVoxels);

    // Pre-compute max Laplacian for normalization
    float maxLapGlobal = 1.0f, maxLapLocal = 1.0f;
    for (int z = 1; z < depth - 1; z++) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float lapG = compute3DLaplacian(globalEnhanced, x, y, z, width, height, depth);
                float lapL = compute3DLaplacian(localEnhanced, x, y, z, width, height, depth);
                maxLapGlobal = std::max(maxLapGlobal, lapG);
                maxLapLocal = std::max(maxLapLocal, lapL);
            }
        }
    }

    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t idx = volIdx(x, y, z);
                uint8_t gVal = globalEnhanced[idx];
                uint8_t lVal = localEnhanced[idx];

                if (gVal == 0 && lVal == 0) {
                    result[idx] = 0;
                    continue;
                }

                // Compute Laplacian-based detail weights
                float lapG = compute3DLaplacian(globalEnhanced, x, y, z, width, height, depth);
                float lapL = compute3DLaplacian(localEnhanced, x, y, z, width, height, depth);

                float c_g = lapG / maxLapGlobal + 0.00001f;
                float c_l = lapL / maxLapLocal + 0.00001f;

                // Brightness weight (prefer mid-tones, sigma = 0.2)
                float g_scaled = static_cast<float>(gVal) / 255.0f;
                float l_scaled = static_cast<float>(lVal) / 255.0f;
                float b_g = std::exp(-(g_scaled - 0.5f) * (g_scaled - 0.5f) / 0.08f);
                float b_l = std::exp(-(l_scaled - 0.5f) * (l_scaled - 0.5f) / 0.08f);

                // Combined weights: minimum of detail and brightness
                float w_g = std::min(c_g, b_g);
                float w_l = std::min(c_l, b_l);

                // Normalize and blend
                float wSum = w_g + w_l;
                if (wSum < 0.00001f) {
                    result[idx] = static_cast<uint8_t>((static_cast<int>(gVal) + lVal) / 2);
                } else {
                    float blended = (w_g * gVal + w_l * lVal) / wSum;
                    result[idx] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, blended)));
                }
            }
        }
    }

    return result;
}

cv::Mat globalLocalFusionEnhance(
    const cv::Mat& input,
    float claheClipLimit,
    int claheTileSize)
{
    if (input.empty() || input.type() != CV_8UC1) {
        return input.clone();
    }

    const int rows = input.rows;
    const int cols = input.cols;
    const int totalPixels = rows * cols;

    // Step 1: Compute normalized histogram
    std::array<float, 256> h_i = {0.0f};
    for (int j = 0; j < rows; j++) {
        const uint8_t* row = input.ptr<uint8_t>(j);
        for (int i = 0; i < cols; i++) {
            h_i[row[i]] += 1.0f;
        }
    }
    for (int i = 0; i < 256; i++) {
        h_i[i] /= static_cast<float>(totalPixels);
    }

    // Step 2: Build global equalization LUT
    auto globalLUT = buildGlobalEqualizationLUT(h_i);

    // Step 3: Apply global equalization
    cv::Mat globalEnhanced(rows, cols, CV_8UC1);
    for (int j = 0; j < rows; j++) {
        const uint8_t* srcRow = input.ptr<uint8_t>(j);
        uint8_t* dstRow = globalEnhanced.ptr<uint8_t>(j);
        for (int i = 0; i < cols; i++) {
            dstRow[i] = globalLUT[srcRow[i]];
        }
    }

    // Step 4: Apply CLAHE for local enhancement
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(claheClipLimit, cv::Size(claheTileSize, claheTileSize));
    cv::Mat localEnhanced;
    clahe->apply(input, localEnhanced);

    // Step 5: Compute fusion weights
    // Weight based on Laplacian (detail/edges) and brightness

    // Laplacian of global enhanced
    cv::Mat lapGlobal;
    cv::Laplacian(globalEnhanced, lapGlobal, CV_16S, 3);
    cv::Mat absLapGlobal;
    cv::convertScaleAbs(lapGlobal, absLapGlobal);

    // Laplacian of local enhanced
    cv::Mat lapLocal;
    cv::Laplacian(localEnhanced, lapLocal, CV_16S, 3);
    cv::Mat absLapLocal;
    cv::convertScaleAbs(lapLocal, absLapLocal);

    // Compute fusion weights and blend
    cv::Mat output(rows, cols, CV_8UC1);

    for (int j = 0; j < rows; j++) {
        const uint8_t* globalRow = globalEnhanced.ptr<uint8_t>(j);
        const uint8_t* localRow = localEnhanced.ptr<uint8_t>(j);
        const uint8_t* lapGlobalRow = absLapGlobal.ptr<uint8_t>(j);
        const uint8_t* lapLocalRow = absLapLocal.ptr<uint8_t>(j);
        uint8_t* outRow = output.ptr<uint8_t>(j);

        for (int i = 0; i < cols; i++) {
            // Normalize Laplacian values (detail measure)
            float c_g = static_cast<float>(lapGlobalRow[i]) / 255.0f + 0.00001f;
            float c_l = static_cast<float>(lapLocalRow[i]) / 255.0f + 0.00001f;

            // Brightness weight (prefer mid-tones)
            float g_scaled = static_cast<float>(globalRow[i]) / 255.0f;
            float l_scaled = static_cast<float>(localRow[i]) / 255.0f;
            float b_g = std::exp(-std::pow(g_scaled - 0.5f, 2) / (2.0f * 0.04f));  // sigma = 0.2
            float b_l = std::exp(-std::pow(l_scaled - 0.5f, 2) / (2.0f * 0.04f));

            // Combined weights: minimum of detail and brightness
            float w_g = std::min(c_g, b_g);
            float w_l = std::min(c_l, b_l);

            // Normalize weights
            float wSum = w_g + w_l;
            if (wSum < 0.00001f) {
                // Equal weighting if both weights are zero
                outRow[i] = static_cast<uint8_t>((static_cast<int>(globalRow[i]) + localRow[i]) / 2);
            } else {
                float w_g_norm = w_g / wSum;
                float w_l_norm = w_l / wSum;

                // Blend global and local
                float blended = w_g_norm * globalRow[i] + w_l_norm * localRow[i];
                outRow[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, blended)));
            }
        }
    }

    return output;
}
