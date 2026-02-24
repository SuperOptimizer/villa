#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

enum class StructuringElement : std::uint8_t {
    Cross,   // 3x3 cross (4-connected / face connectivity)
    Square,  // 3x3 square (8-connected / full connectivity)
    Disk,    // Circular structuring element
};

// Core operations (enum-based structuring element)
[[nodiscard]] auto dilate(const Tensor& input,
                          StructuringElement se = StructuringElement::Square,
                          std::size_t iterations = 1) -> std::expected<Tensor, std::string>;

[[nodiscard]] auto erode(const Tensor& input,
                         StructuringElement se = StructuringElement::Square,
                         std::size_t iterations = 1) -> std::expected<Tensor, std::string>;

// Core operations (custom kernel)
[[nodiscard]] auto dilate(const Tensor& input, const Tensor& kernel,
                          std::size_t iterations = 1) -> std::expected<Tensor, std::string>;

[[nodiscard]] auto erode(const Tensor& input, const Tensor& kernel,
                         std::size_t iterations = 1) -> std::expected<Tensor, std::string>;

// Compound operations
[[nodiscard]] auto morphological_open(const Tensor& input,
                                      StructuringElement se = StructuringElement::Square,
                                      std::size_t iterations = 1) -> std::expected<Tensor, std::string>;

[[nodiscard]] auto morphological_close(const Tensor& input,
                                       StructuringElement se = StructuringElement::Square,
                                       std::size_t iterations = 1) -> std::expected<Tensor, std::string>;

[[nodiscard]] auto morphological_gradient(const Tensor& input,
                                          StructuringElement se = StructuringElement::Square)
    -> std::expected<Tensor, std::string>;

// Helper: create structuring element as Tensor
[[nodiscard]] auto make_structuring_element(StructuringElement se,
                                            std::size_t radius = 1) -> Tensor;

} // namespace utils
