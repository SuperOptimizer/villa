#pragma once

#include <expected>
#include <string>
#include <vector>

#include "utils/tensor.hpp"

namespace utils {

struct DistanceTransformOptions {
    std::vector<double> spacing;
    bool squared = false;
    DType output_dtype = DType::Float32;
};

[[nodiscard]] auto distance_transform(const Tensor& input,
                                      const DistanceTransformOptions& opts = {})
    -> std::expected<Tensor, std::string>;

[[nodiscard]] auto distance_transform_inv(const Tensor& input,
                                          const DistanceTransformOptions& opts = {})
    -> std::expected<Tensor, std::string>;

} // namespace utils
