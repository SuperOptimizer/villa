#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

enum class BorderMode : std::uint8_t {
    Zero,      // Pad with zeros
    Replicate, // Clamp to edge
    Reflect,   // Mirror at boundary
};

/// General 2D convolution with arbitrary kernel.
/// Input must be 2D (Float32 or UInt8). Kernel must be 2D with odd dimensions.
/// Output is always Float32.
[[nodiscard]] auto convolve(const Tensor& input, const Tensor& kernel,
                            BorderMode border = BorderMode::Zero)
    -> std::expected<Tensor, std::string>;

/// Separable 2D convolution (faster for separable kernels like Gaussian).
/// kernel_x and kernel_y must be 1D Float32 tensors with odd length.
/// Output is always Float32.
[[nodiscard]] auto convolve_separable(const Tensor& input,
                                       const Tensor& kernel_x,
                                       const Tensor& kernel_y,
                                       BorderMode border = BorderMode::Zero)
    -> std::expected<Tensor, std::string>;

/// Gaussian blur with given sigma.
/// Kernel size is auto-computed as ceil(3*sigma)*2+1.
[[nodiscard]] auto gaussian_blur(const Tensor& input, float sigma,
                                  BorderMode border = BorderMode::Replicate)
    -> std::expected<Tensor, std::string>;

/// Box (mean) filter with given kernel size.
[[nodiscard]] auto box_blur(const Tensor& input, std::size_t kernel_size,
                             BorderMode border = BorderMode::Replicate)
    -> std::expected<Tensor, std::string>;

/// Sobel edge detection. Returns gradient magnitude as Float32.
[[nodiscard]] auto sobel(const Tensor& input,
                          BorderMode border = BorderMode::Replicate)
    -> std::expected<Tensor, std::string>;

/// Sharpen filter with adjustable amount.
[[nodiscard]] auto sharpen(const Tensor& input, float amount = 1.0f,
                            BorderMode border = BorderMode::Replicate)
    -> std::expected<Tensor, std::string>;

/// Create a 1D Gaussian kernel tensor (Float32) with the given sigma.
/// Kernel size is ceil(3*sigma)*2+1.
[[nodiscard]] auto gaussian_kernel(float sigma) -> Tensor;

} // namespace utils
