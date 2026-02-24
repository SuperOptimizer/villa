#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>

#include "utils/tensor.hpp"

namespace utils {

enum class Connectivity : std::uint8_t { Face, Full };

struct CCResult {
    Tensor labels;
    std::size_t count;
};

[[nodiscard]] auto connected_components(const Tensor& input,
                                        Connectivity conn = Connectivity::Face)
    -> std::expected<CCResult, std::string>;

[[nodiscard]] auto connected_components_binary(const Tensor& input,
                                               Connectivity conn = Connectivity::Face)
    -> std::expected<CCResult, std::string>;

} // namespace utils
