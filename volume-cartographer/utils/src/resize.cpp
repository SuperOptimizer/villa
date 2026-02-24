#include "utils/resize.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace utils {
namespace {

/// Convert input tensor to a contiguous Float32 buffer.
auto to_float32_buffer(const Tensor& input) -> std::vector<float> {
    auto numel = input.numel();
    std::vector<float> buf(numel);

    auto contiguous = input.contiguous();

    if (input.dtype() == DType::UInt8) {
        const auto* src = contiguous.data<std::uint8_t>();
        for (std::size_t i = 0; i < numel; ++i) {
            buf[i] = static_cast<float>(src[i]);
        }
    } else {
        const auto* src = contiguous.data<float>();
        for (std::size_t i = 0; i < numel; ++i) {
            buf[i] = src[i];
        }
    }
    return buf;
}

/// Validate that a tensor is 2D with Float32 or UInt8 dtype.
auto validate_input(const Tensor& input, const char* func_name)
    -> std::expected<void, std::string> {
    if (input.ndim() != 2) {
        return std::unexpected(std::string(func_name) + ": input must be 2D, got " +
                               std::to_string(input.ndim()) + "D");
    }
    if (input.dtype() != DType::Float32 && input.dtype() != DType::UInt8) {
        return std::unexpected(std::string(func_name) +
                               ": input must be Float32 or UInt8, got " +
                               std::string(dtype_name(input.dtype())));
    }
    return {};
}

/// Clamp a float to [0, max] and fetch from row-major buffer.
auto sample_clamped(const float* img, std::size_t rows, std::size_t cols,
                    std::ptrdiff_t r, std::ptrdiff_t c) -> float {
    auto ri = static_cast<std::size_t>(
        std::clamp(r, std::ptrdiff_t{0}, static_cast<std::ptrdiff_t>(rows - 1)));
    auto ci = static_cast<std::size_t>(
        std::clamp(c, std::ptrdiff_t{0}, static_cast<std::ptrdiff_t>(cols - 1)));
    return img[ri * cols + ci];
}

/// Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0.
auto lanczos_weight(double x, double a) -> double {
    if (x == 0.0) return 1.0;
    if (std::abs(x) >= a) return 0.0;
    auto pi_x = std::numbers::pi * x;
    return (std::sin(pi_x) / pi_x) * (std::sin(pi_x / a) / (pi_x / a));
}

/// Write Float32 buffer back to a tensor with the given dtype.
/// For UInt8: rounds and clamps to [0, 255].
auto buffer_to_tensor(const std::vector<float>& buf, std::size_t height,
                      std::size_t width, DType target_dt)
    -> std::expected<Tensor, std::string> {
    std::size_t shape[] = {height, width};
    if (target_dt == DType::UInt8) {
        std::vector<std::uint8_t> out(buf.size());
        for (std::size_t i = 0; i < buf.size(); ++i) {
            auto val = std::round(buf[i]);
            val = std::clamp(val, 0.0f, 255.0f);
            out[i] = static_cast<std::uint8_t>(val);
        }
        return Tensor::from_data(out.data(), shape, DType::UInt8);
    }
    return Tensor::from_data(buf.data(), shape, DType::Float32);
}

/// Nearest-neighbor resize.
auto resize_nearest(const float* src, std::size_t src_h, std::size_t src_w,
                    std::size_t dst_h, std::size_t dst_w) -> std::vector<float> {
    std::vector<float> dst(dst_h * dst_w);
    auto scale_y = static_cast<double>(src_h) / static_cast<double>(dst_h);
    auto scale_x = static_cast<double>(src_w) / static_cast<double>(dst_w);

    for (std::size_t r = 0; r < dst_h; ++r) {
        auto src_r = static_cast<std::size_t>(
            std::min(static_cast<double>(r) * scale_y, static_cast<double>(src_h - 1)));
        for (std::size_t c = 0; c < dst_w; ++c) {
            auto src_c = static_cast<std::size_t>(
                std::min(static_cast<double>(c) * scale_x, static_cast<double>(src_w - 1)));
            dst[r * dst_w + c] = src[src_r * src_w + src_c];
        }
    }
    return dst;
}

/// Bilinear resize.
auto resize_bilinear(const float* src, std::size_t src_h, std::size_t src_w,
                     std::size_t dst_h, std::size_t dst_w) -> std::vector<float> {
    std::vector<float> dst(dst_h * dst_w);
    auto scale_y = static_cast<double>(src_h) / static_cast<double>(dst_h);
    auto scale_x = static_cast<double>(src_w) / static_cast<double>(dst_w);

    for (std::size_t r = 0; r < dst_h; ++r) {
        // Map output center to input coordinate
        auto fy = (static_cast<double>(r) + 0.5) * scale_y - 0.5;
        auto y0 = static_cast<std::ptrdiff_t>(std::floor(fy));
        auto dy = static_cast<float>(fy - static_cast<double>(y0));

        for (std::size_t c = 0; c < dst_w; ++c) {
            auto fx = (static_cast<double>(c) + 0.5) * scale_x - 0.5;
            auto x0 = static_cast<std::ptrdiff_t>(std::floor(fx));
            auto dx = static_cast<float>(fx - static_cast<double>(x0));

            auto p00 = sample_clamped(src, src_h, src_w, y0, x0);
            auto p01 = sample_clamped(src, src_h, src_w, y0, x0 + 1);
            auto p10 = sample_clamped(src, src_h, src_w, y0 + 1, x0);
            auto p11 = sample_clamped(src, src_h, src_w, y0 + 1, x0 + 1);

            auto top = p00 + dx * (p01 - p00);
            auto bot = p10 + dx * (p11 - p10);
            dst[r * dst_w + c] = top + dy * (bot - top);
        }
    }
    return dst;
}

/// Lanczos-3 resize (6x6 support window).
auto resize_lanczos(const float* src, std::size_t src_h, std::size_t src_w,
                    std::size_t dst_h, std::size_t dst_w) -> std::vector<float> {
    constexpr double A = 3.0; // Lanczos-3
    std::vector<float> dst(dst_h * dst_w);
    auto scale_y = static_cast<double>(src_h) / static_cast<double>(dst_h);
    auto scale_x = static_cast<double>(src_w) / static_cast<double>(dst_w);

    // When downscaling, widen the kernel to cover more input pixels
    auto support_y = std::max(A, A * scale_y);
    auto support_x = std::max(A, A * scale_x);

    for (std::size_t r = 0; r < dst_h; ++r) {
        auto fy = (static_cast<double>(r) + 0.5) * scale_y - 0.5;

        for (std::size_t c = 0; c < dst_w; ++c) {
            auto fx = (static_cast<double>(c) + 0.5) * scale_x - 0.5;

            auto y_min = static_cast<std::ptrdiff_t>(std::ceil(fy - support_y));
            auto y_max = static_cast<std::ptrdiff_t>(std::floor(fy + support_y));
            auto x_min = static_cast<std::ptrdiff_t>(std::ceil(fx - support_x));
            auto x_max = static_cast<std::ptrdiff_t>(std::floor(fx + support_x));

            double sum = 0.0;
            double weight_sum = 0.0;

            for (auto sy = y_min; sy <= y_max; ++sy) {
                auto wy = lanczos_weight((static_cast<double>(sy) - fy) / std::max(1.0, scale_y),
                                         A);
                for (auto sx = x_min; sx <= x_max; ++sx) {
                    auto wx = lanczos_weight((static_cast<double>(sx) - fx) / std::max(1.0, scale_x),
                                             A);
                    auto w = wy * wx;
                    sum += w * static_cast<double>(sample_clamped(src, src_h, src_w, sy, sx));
                    weight_sum += w;
                }
            }

            dst[r * dst_w + c] = (weight_sum > 0.0)
                                     ? static_cast<float>(sum / weight_sum)
                                     : 0.0f;
        }
    }
    return dst;
}

} // anonymous namespace

auto resize(const Tensor& input,
            std::size_t target_height, std::size_t target_width,
            ResizeMethod method)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "resize"); !v) {
        return std::unexpected(v.error());
    }
    if (target_height == 0 || target_width == 0) {
        return std::unexpected(std::string("resize: target dimensions must be > 0"));
    }

    auto src_h = input.shape()[0];
    auto src_w = input.shape()[1];
    auto input_dt = input.dtype();

    // Identity case: same dimensions -> clone
    if (target_height == src_h && target_width == src_w) {
        return input.clone();
    }

    auto buf = to_float32_buffer(input);

    std::vector<float> result;
    switch (method) {
        case ResizeMethod::Nearest:
            result = resize_nearest(buf.data(), src_h, src_w, target_height, target_width);
            break;
        case ResizeMethod::Bilinear:
            result = resize_bilinear(buf.data(), src_h, src_w, target_height, target_width);
            break;
        case ResizeMethod::Lanczos:
            result = resize_lanczos(buf.data(), src_h, src_w, target_height, target_width);
            break;
    }

    return buffer_to_tensor(result, target_height, target_width, input_dt);
}

auto resize(const Tensor& input, float scale, ResizeMethod method)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "resize"); !v) {
        return std::unexpected(v.error());
    }
    if (scale <= 0.0f) {
        return std::unexpected(std::string("resize: scale must be > 0"));
    }

    auto src_h = input.shape()[0];
    auto src_w = input.shape()[1];
    auto dst_h = static_cast<std::size_t>(std::max(1.0f, std::round(static_cast<float>(src_h) * scale)));
    auto dst_w = static_cast<std::size_t>(std::max(1.0f, std::round(static_cast<float>(src_w) * scale)));

    return resize(input, dst_h, dst_w, method);
}

auto downscale(const Tensor& input, std::size_t factor)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "downscale"); !v) {
        return std::unexpected(v.error());
    }
    if (factor == 0) {
        return std::unexpected(std::string("downscale: factor must be > 0"));
    }

    auto src_h = input.shape()[0];
    auto src_w = input.shape()[1];
    auto input_dt = input.dtype();

    if (factor == 1) {
        return input.clone();
    }

    auto dst_h = src_h / factor;
    auto dst_w = src_w / factor;
    if (dst_h == 0 || dst_w == 0) {
        return std::unexpected(std::string("downscale: factor too large for input dimensions"));
    }

    auto buf = to_float32_buffer(input);
    std::vector<float> result(dst_h * dst_w);
    auto block_size = static_cast<float>(factor * factor);

    for (std::size_t r = 0; r < dst_h; ++r) {
        for (std::size_t c = 0; c < dst_w; ++c) {
            float sum = 0.0f;
            for (std::size_t br = 0; br < factor; ++br) {
                for (std::size_t bc = 0; bc < factor; ++bc) {
                    sum += buf[(r * factor + br) * src_w + (c * factor + bc)];
                }
            }
            result[r * dst_w + c] = sum / block_size;
        }
    }

    return buffer_to_tensor(result, dst_h, dst_w, input_dt);
}

} // namespace utils
