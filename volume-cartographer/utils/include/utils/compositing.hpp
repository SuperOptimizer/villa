#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

enum class CompositeMethod : std::uint8_t {
    Mean,    // Average of all layers
    Max,     // Maximum intensity projection
    Min,     // Minimum intensity projection
    Median,  // Median of layers
};

struct CompositeOptions {
    CompositeMethod method = CompositeMethod::Mean;
};

/// Composite a 3D tensor (layers x H x W) down to a 2D tensor (H x W).
/// Supported dtypes: Float32, UInt8.
[[nodiscard]] auto composite(const Tensor& layers,
                             const CompositeOptions& opts = {})
    -> std::expected<Tensor, std::string>;

} // namespace utils
