#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

enum class ResizeMethod : std::uint8_t {
    Nearest,   // Nearest-neighbor (fast, no smoothing)
    Bilinear,  // Bilinear interpolation (smooth)
    Lanczos,   // Lanczos-3 (high quality downscaling)
};

/// Resize a 2D tensor to target dimensions.
/// Supports UInt8 and Float32 dtypes. Output dtype matches input dtype.
[[nodiscard]] auto resize(const Tensor& input,
                           std::size_t target_height, std::size_t target_width,
                           ResizeMethod method = ResizeMethod::Bilinear)
    -> std::expected<Tensor, std::string>;

/// Resize by a scale factor (e.g., 0.5 for half size, 2.0 for double).
/// Supports UInt8 and Float32 dtypes. Output dtype matches input dtype.
[[nodiscard]] auto resize(const Tensor& input, float scale,
                           ResizeMethod method = ResizeMethod::Bilinear)
    -> std::expected<Tensor, std::string>;

/// Downscale by integer factor (fast path: 2x, 4x, etc.).
/// Uses area averaging for quality.
/// Supports UInt8 and Float32 dtypes. Output dtype matches input dtype.
[[nodiscard]] auto downscale(const Tensor& input, std::size_t factor)
    -> std::expected<Tensor, std::string>;

} // namespace utils
