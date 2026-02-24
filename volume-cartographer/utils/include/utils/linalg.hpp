#pragma once

#include <expected>
#include <string>

#include "tensor.hpp"

namespace utils {

// Inner product of 1D tensors (returns scalar)
[[nodiscard]] auto dot(const Tensor& a, const Tensor& b) -> Tensor;

// L2 norm (returns scalar Float64)
[[nodiscard]] auto norm(const Tensor& a) -> Tensor;

// Matrix multiply [M,K] x [K,N] -> [M,N]
[[nodiscard]] auto matmul(const Tensor& a, const Tensor& b)
    -> std::expected<Tensor, std::string>;

// Cross product of 3-element vectors
[[nodiscard]] auto cross(const Tensor& a, const Tensor& b) -> Tensor;

// Normalize vector to unit length
[[nodiscard]] auto normalize(const Tensor& a) -> Tensor;

} // namespace utils
