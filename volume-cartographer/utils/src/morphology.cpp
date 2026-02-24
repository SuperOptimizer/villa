#include "utils/morphology.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace utils {
namespace {

auto validate_input(const Tensor& input) -> std::expected<void, std::string> {
    if (input.ndim() != 2) {
        return std::unexpected("morphology: input must be 2D, got " +
                               std::to_string(input.ndim()) + "D");
    }
    if (input.dtype() != DType::UInt8) {
        return std::unexpected("morphology: input must have UInt8 dtype, got " +
                               std::string(dtype_name(input.dtype())));
    }
    return {};
}

auto validate_kernel(const Tensor& kernel) -> std::expected<void, std::string> {
    if (kernel.ndim() != 2) {
        return std::unexpected("morphology: kernel must be 2D, got " +
                               std::to_string(kernel.ndim()) + "D");
    }
    if (kernel.dtype() != DType::UInt8) {
        return std::unexpected("morphology: kernel must have UInt8 dtype, got " +
                               std::string(dtype_name(kernel.dtype())));
    }
    auto kshape = kernel.shape();
    if (kshape[0] % 2 == 0 || kshape[1] % 2 == 0) {
        return std::unexpected("morphology: kernel must have odd dimensions, got " +
                               std::to_string(kshape[0]) + "x" +
                               std::to_string(kshape[1]));
    }
    return {};
}

// Build a flat list of (dy, dx) offsets where kernel is nonzero
auto kernel_offsets(const Tensor& kernel)
    -> std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> {
    auto kshape = kernel.shape();
    auto kh = kshape[0];
    auto kw = kshape[1];
    auto cy = static_cast<std::ptrdiff_t>(kh / 2);
    auto cx = static_cast<std::ptrdiff_t>(kw / 2);

    auto cont = kernel.contiguous();
    const auto* kdata = cont.data<std::uint8_t>();

    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> offsets;
    for (std::size_t r = 0; r < kh; ++r) {
        for (std::size_t c = 0; c < kw; ++c) {
            if (kdata[r * kw + c] != 0) {
                offsets.emplace_back(static_cast<std::ptrdiff_t>(r) - cy,
                                     static_cast<std::ptrdiff_t>(c) - cx);
            }
        }
    }
    return offsets;
}

auto dilate_once(const Tensor& input, const Tensor& kernel) -> std::expected<Tensor, std::string> {
    auto shape = input.shape();
    auto h = shape[0];
    auto w = shape[1];

    auto cont = input.contiguous();
    const auto* in_data = cont.data<std::uint8_t>();

    auto out = Tensor::zeros(shape, DType::UInt8);
    if (!out) return std::unexpected(out.error());
    auto* out_data = out->data<std::uint8_t>();

    auto offsets = kernel_offsets(kernel);

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            if (in_data[y * w + x] == 0) continue;
            // Set all pixels under the kernel to foreground
            for (auto& [dy, dx] : offsets) {
                auto ny = static_cast<std::ptrdiff_t>(y) + dy;
                auto nx = static_cast<std::ptrdiff_t>(x) + dx;
                if (ny >= 0 && ny < static_cast<std::ptrdiff_t>(h) &&
                    nx >= 0 && nx < static_cast<std::ptrdiff_t>(w)) {
                    out_data[static_cast<std::size_t>(ny) * w + static_cast<std::size_t>(nx)] = 255;
                }
            }
        }
    }

    return std::move(*out);
}

auto erode_once(const Tensor& input, const Tensor& kernel) -> std::expected<Tensor, std::string> {
    auto shape = input.shape();
    auto h = shape[0];
    auto w = shape[1];

    auto cont = input.contiguous();
    const auto* in_data = cont.data<std::uint8_t>();

    auto out = Tensor::zeros(shape, DType::UInt8);
    if (!out) return std::unexpected(out.error());
    auto* out_data = out->data<std::uint8_t>();

    auto offsets = kernel_offsets(kernel);

    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            // Pixel is foreground only if ALL kernel positions are foreground
            bool all_set = true;
            for (auto& [dy, dx] : offsets) {
                auto ny = static_cast<std::ptrdiff_t>(y) + dy;
                auto nx = static_cast<std::ptrdiff_t>(x) + dx;
                if (ny < 0 || ny >= static_cast<std::ptrdiff_t>(h) ||
                    nx < 0 || nx >= static_cast<std::ptrdiff_t>(w)) {
                    all_set = false;
                    break;
                }
                if (in_data[static_cast<std::size_t>(ny) * w + static_cast<std::size_t>(nx)] == 0) {
                    all_set = false;
                    break;
                }
            }
            if (all_set) {
                out_data[y * w + x] = 255;
            }
        }
    }

    return std::move(*out);
}

} // namespace

auto make_structuring_element(StructuringElement se, std::size_t radius) -> Tensor {
    auto size = 2 * radius + 1;
    auto t = Tensor::zeros({size, size}, DType::UInt8).value();
    auto* d = t.data<std::uint8_t>();

    switch (se) {
        case StructuringElement::Cross:
            for (std::size_t i = 0; i < size; ++i) {
                d[radius * size + i] = 1; // horizontal bar
                d[i * size + radius] = 1; // vertical bar
            }
            break;

        case StructuringElement::Square:
            for (std::size_t i = 0; i < size * size; ++i) {
                d[i] = 1;
            }
            break;

        case StructuringElement::Disk: {
            auto center = static_cast<double>(radius);
            auto r2 = (center + 0.5) * (center + 0.5);
            for (std::size_t y = 0; y < size; ++y) {
                for (std::size_t x = 0; x < size; ++x) {
                    auto dy = static_cast<double>(y) - center;
                    auto dx = static_cast<double>(x) - center;
                    if (dy * dy + dx * dx <= r2) {
                        d[y * size + x] = 1;
                    }
                }
            }
            break;
        }
    }

    return t;
}

auto dilate(const Tensor& input, StructuringElement se, std::size_t iterations)
    -> std::expected<Tensor, std::string> {
    auto kernel = make_structuring_element(se);
    return dilate(input, kernel, iterations);
}

auto erode(const Tensor& input, StructuringElement se, std::size_t iterations)
    -> std::expected<Tensor, std::string> {
    auto kernel = make_structuring_element(se);
    return erode(input, kernel, iterations);
}

auto dilate(const Tensor& input, const Tensor& kernel, std::size_t iterations)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());
    if (auto v = validate_kernel(kernel); !v) return std::unexpected(v.error());

    if (iterations == 0) {
        return input.clone();
    }

    auto result = dilate_once(input, kernel);
    if (!result) return result;

    for (std::size_t i = 1; i < iterations; ++i) {
        result = dilate_once(*result, kernel);
        if (!result) return result;
    }

    return result;
}

auto erode(const Tensor& input, const Tensor& kernel, std::size_t iterations)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());
    if (auto v = validate_kernel(kernel); !v) return std::unexpected(v.error());

    if (iterations == 0) {
        return input.clone();
    }

    auto result = erode_once(input, kernel);
    if (!result) return result;

    for (std::size_t i = 1; i < iterations; ++i) {
        result = erode_once(*result, kernel);
        if (!result) return result;
    }

    return result;
}

auto morphological_open(const Tensor& input, StructuringElement se, std::size_t iterations)
    -> std::expected<Tensor, std::string> {
    auto eroded = erode(input, se, iterations);
    if (!eroded) return eroded;
    return dilate(*eroded, se, iterations);
}

auto morphological_close(const Tensor& input, StructuringElement se, std::size_t iterations)
    -> std::expected<Tensor, std::string> {
    auto dilated = dilate(input, se, iterations);
    if (!dilated) return dilated;
    return erode(*dilated, se, iterations);
}

auto morphological_gradient(const Tensor& input, StructuringElement se)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());

    auto dilated = dilate(input, se, 1);
    if (!dilated) return dilated;
    auto eroded = erode(input, se, 1);
    if (!eroded) return eroded;

    auto shape = input.shape();
    auto numel = input.numel();

    auto out = Tensor::zeros(shape, DType::UInt8);
    if (!out) return std::unexpected(out.error());

    const auto* dil = dilated->data<std::uint8_t>();
    const auto* ero = eroded->data<std::uint8_t>();
    auto* dst = out->data<std::uint8_t>();

    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = (dil[i] > ero[i]) ? static_cast<std::uint8_t>(dil[i] - ero[i]) : 0;
    }

    return std::move(*out);
}

} // namespace utils
