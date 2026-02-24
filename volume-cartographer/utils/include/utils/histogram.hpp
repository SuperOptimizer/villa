#pragma once

#include <cstddef>
#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

// Compute histogram of a 2D tensor.
// For UInt8: 256 bins (0-255)
// For UInt16: 65536 bins or user-specified
// For Float32: user-specified num_bins over [min, max] range
// Output is a Float64 tensor of shape {num_bins}.
[[nodiscard]] auto histogram(const Tensor& input, std::size_t num_bins = 0)
    -> std::expected<Tensor, std::string>;

// Cumulative histogram (CDF).
// Computes histogram then running sum.
// Output is a Float64 tensor of shape {num_bins}.
[[nodiscard]] auto cumulative_histogram(const Tensor& input, std::size_t num_bins = 0)
    -> std::expected<Tensor, std::string>;

// Histogram equalization -- enhances contrast.
// Returns UInt8 output regardless of input type.
[[nodiscard]] auto histogram_equalize(const Tensor& input)
    -> std::expected<Tensor, std::string>;

// Otsu's threshold -- find optimal binary threshold.
// Returns the threshold value.
[[nodiscard]] auto otsu_threshold(const Tensor& input)
    -> std::expected<double, std::string>;

// Apply binary threshold: pixels >= threshold -> 255, else -> 0.
// Returns UInt8 tensor.
[[nodiscard]] auto threshold(const Tensor& input, double thresh)
    -> std::expected<Tensor, std::string>;

} // namespace utils
