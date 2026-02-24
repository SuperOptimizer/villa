#pragma once

#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

/// Thin a binary image (2D UInt8 tensor, nonzero = foreground) to its skeleton
/// using the Zhang-Suen morphological thinning algorithm.
///
/// The output is a 2D UInt8 tensor of the same shape where skeleton pixels
/// are 255 and background pixels are 0.
[[nodiscard]] auto thin(const Tensor& input) -> std::expected<Tensor, std::string>;

} // namespace utils
