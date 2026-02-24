#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include "tensor.hpp"

namespace utils {

enum class InterpMode : std::uint8_t { Nearest, Linear };
enum class BoundaryMode : std::uint8_t { Clamp, Zero };

// Single-point sampling
[[nodiscard]] auto interpolate2d(const Tensor& t, double y, double x,
                                 InterpMode interp = InterpMode::Linear,
                                 BoundaryMode boundary = BoundaryMode::Clamp) -> double;

[[nodiscard]] auto interpolate3d(const Tensor& t, double z, double y, double x,
                                 InterpMode interp = InterpMode::Linear,
                                 BoundaryMode boundary = BoundaryMode::Clamp) -> double;

// Batch: coords is [N, 2] (2D) or [N, 3] (3D) Float64 tensor
// Returns [N] Float64 tensor
[[nodiscard]] auto interpolate(const Tensor& t, const Tensor& coords,
                               InterpMode interp = InterpMode::Linear,
                               BoundaryMode boundary = BoundaryMode::Clamp)
    -> std::expected<Tensor, std::string>;

} // namespace utils
