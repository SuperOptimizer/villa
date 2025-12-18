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
        "alpha"
    };
}
