#include "utils/convolution.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace utils {
namespace {

/// Fetch a pixel from the image with border handling.
/// img: pointer to Float32 data, row-major (rows x cols).
auto fetch(const float* img, std::size_t rows, std::size_t cols,
           std::ptrdiff_t r, std::ptrdiff_t c, BorderMode border) -> float {
    auto ri = r;
    auto ci = c;
    auto sr = static_cast<std::ptrdiff_t>(rows);
    auto sc = static_cast<std::ptrdiff_t>(cols);

    if (ri < 0 || ri >= sr || ci < 0 || ci >= sc) {
        switch (border) {
            case BorderMode::Zero:
                return 0.0f;
            case BorderMode::Replicate:
                ri = std::clamp(ri, std::ptrdiff_t{0}, sr - 1);
                ci = std::clamp(ci, std::ptrdiff_t{0}, sc - 1);
                break;
            case BorderMode::Reflect:
                // Reflect at boundary: index -1 -> 0, -2 -> 1, etc.
                if (ri < 0) ri = -ri - 1;
                if (ri >= sr) ri = 2 * sr - ri - 1;
                ri = std::clamp(ri, std::ptrdiff_t{0}, sr - 1);
                if (ci < 0) ci = -ci - 1;
                if (ci >= sc) ci = 2 * sc - ci - 1;
                ci = std::clamp(ci, std::ptrdiff_t{0}, sc - 1);
                break;
        }
    }
    return img[static_cast<std::size_t>(ri) * cols + static_cast<std::size_t>(ci)];
}

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
        // Float32
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

/// Validate that a kernel is 2D with odd dimensions and Float32 dtype.
auto validate_kernel(const Tensor& kernel, const char* func_name)
    -> std::expected<void, std::string> {
    if (kernel.ndim() != 2) {
        return std::unexpected(std::string(func_name) + ": kernel must be 2D, got " +
                               std::to_string(kernel.ndim()) + "D");
    }
    if (kernel.dtype() != DType::Float32) {
        return std::unexpected(std::string(func_name) +
                               ": kernel must be Float32, got " +
                               std::string(dtype_name(kernel.dtype())));
    }
    auto kh = kernel.shape()[0];
    auto kw = kernel.shape()[1];
    if (kh % 2 == 0 || kw % 2 == 0) {
        return std::unexpected(std::string(func_name) +
                               ": kernel dimensions must be odd, got " +
                               std::to_string(kh) + "x" + std::to_string(kw));
    }
    return {};
}

/// Validate that a 1D kernel has odd length and Float32 dtype.
auto validate_kernel_1d(const Tensor& kernel, const char* name, const char* func_name)
    -> std::expected<void, std::string> {
    if (kernel.ndim() != 1) {
        return std::unexpected(std::string(func_name) + ": " + name +
                               " must be 1D, got " +
                               std::to_string(kernel.ndim()) + "D");
    }
    if (kernel.dtype() != DType::Float32) {
        return std::unexpected(std::string(func_name) + ": " + name +
                               " must be Float32, got " +
                               std::string(dtype_name(kernel.dtype())));
    }
    auto len = kernel.shape()[0];
    if (len % 2 == 0) {
        return std::unexpected(std::string(func_name) + ": " + name +
                               " length must be odd, got " + std::to_string(len));
    }
    return {};
}

} // namespace

auto convolve(const Tensor& input, const Tensor& kernel, BorderMode border)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "convolve"); !v) return std::unexpected(v.error());
    if (auto v = validate_kernel(kernel, "convolve"); !v) return std::unexpected(v.error());

    auto rows = input.shape()[0];
    auto cols = input.shape()[1];
    auto kh = kernel.shape()[0];
    auto kw = kernel.shape()[1];
    auto half_h = static_cast<std::ptrdiff_t>(kh / 2);
    auto half_w = static_cast<std::ptrdiff_t>(kw / 2);

    auto img = to_float32_buffer(input);

    // Get kernel data (contiguous, Float32)
    auto kern_cont = kernel.contiguous();
    const auto* kdata = kern_cont.data<float>();

    // Allocate output
    auto out = Tensor::zeros({rows, cols}, DType::Float32);
    if (!out) return std::unexpected(out.error());
    auto* dst = out->data<float>();

    // Standard 2D convolution: flip kernel and slide
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float acc = 0.0f;
            for (std::ptrdiff_t kr = 0; kr < static_cast<std::ptrdiff_t>(kh); ++kr) {
                for (std::ptrdiff_t kc = 0; kc < static_cast<std::ptrdiff_t>(kw); ++kc) {
                    // Flip kernel: access kernel[kh-1-kr][kw-1-kc]
                    auto ir = static_cast<std::ptrdiff_t>(r) + kr - half_h;
                    auto ic = static_cast<std::ptrdiff_t>(c) + kc - half_w;
                    auto pixel = fetch(img.data(), rows, cols, ir, ic, border);
                    auto kval = kdata[(kh - 1 - static_cast<std::size_t>(kr)) * kw +
                                      (kw - 1 - static_cast<std::size_t>(kc))];
                    acc += pixel * kval;
                }
            }
            dst[r * cols + c] = acc;
        }
    }

    return std::move(*out);
}

auto convolve_separable(const Tensor& input, const Tensor& kernel_x,
                         const Tensor& kernel_y, BorderMode border)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "convolve_separable"); !v)
        return std::unexpected(v.error());
    if (auto v = validate_kernel_1d(kernel_x, "kernel_x", "convolve_separable"); !v)
        return std::unexpected(v.error());
    if (auto v = validate_kernel_1d(kernel_y, "kernel_y", "convolve_separable"); !v)
        return std::unexpected(v.error());

    auto rows = input.shape()[0];
    auto cols = input.shape()[1];
    auto kx_len = kernel_x.shape()[0];
    auto ky_len = kernel_y.shape()[0];
    auto half_x = static_cast<std::ptrdiff_t>(kx_len / 2);
    auto half_y = static_cast<std::ptrdiff_t>(ky_len / 2);

    auto img = to_float32_buffer(input);

    auto kx_cont = kernel_x.contiguous();
    auto ky_cont = kernel_y.contiguous();
    const auto* kx_data = kx_cont.data<float>();
    const auto* ky_data = ky_cont.data<float>();

    // Pass 1: horizontal convolution with kernel_x
    std::vector<float> temp(rows * cols, 0.0f);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float acc = 0.0f;
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(kx_len); ++k) {
                auto ic = static_cast<std::ptrdiff_t>(c) + k - half_x;
                auto pixel = fetch(img.data(), rows, cols,
                                   static_cast<std::ptrdiff_t>(r), ic, border);
                // Flip: kernel_x[kx_len - 1 - k]
                acc += pixel * kx_data[kx_len - 1 - static_cast<std::size_t>(k)];
            }
            temp[r * cols + c] = acc;
        }
    }

    // Pass 2: vertical convolution with kernel_y on temp
    auto out = Tensor::zeros({rows, cols}, DType::Float32);
    if (!out) return std::unexpected(out.error());
    auto* dst = out->data<float>();

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            float acc = 0.0f;
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(ky_len); ++k) {
                auto ir = static_cast<std::ptrdiff_t>(r) + k - half_y;
                // Use border-aware fetch on temp buffer
                auto pixel = fetch(temp.data(), rows, cols, ir,
                                   static_cast<std::ptrdiff_t>(c), border);
                // Flip: kernel_y[ky_len - 1 - k]
                acc += pixel * ky_data[ky_len - 1 - static_cast<std::size_t>(k)];
            }
            dst[r * cols + c] = acc;
        }
    }

    return std::move(*out);
}

auto gaussian_kernel(float sigma) -> Tensor {
    auto radius = static_cast<std::size_t>(std::ceil(3.0f * sigma));
    auto size = radius * 2 + 1;

    auto t = Tensor::zeros({size}, DType::Float32).value();
    auto* data = t.data<float>();

    float sum = 0.0f;
    auto center = static_cast<float>(radius);
    for (std::size_t i = 0; i < size; ++i) {
        float x = static_cast<float>(i) - center;
        data[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += data[i];
    }

    // Normalize so kernel sums to 1
    for (std::size_t i = 0; i < size; ++i) {
        data[i] /= sum;
    }

    return t;
}

auto gaussian_blur(const Tensor& input, float sigma, BorderMode border)
    -> std::expected<Tensor, std::string> {
    if (sigma <= 0.0f) {
        return std::unexpected("gaussian_blur: sigma must be positive, got " +
                               std::to_string(sigma));
    }
    auto kern = gaussian_kernel(sigma);
    return convolve_separable(input, kern, kern, border);
}

auto box_blur(const Tensor& input, std::size_t kernel_size, BorderMode border)
    -> std::expected<Tensor, std::string> {
    if (kernel_size == 0) {
        return std::unexpected("box_blur: kernel_size must be positive");
    }
    if (kernel_size % 2 == 0) {
        return std::unexpected("box_blur: kernel_size must be odd, got " +
                               std::to_string(kernel_size));
    }

    auto val = 1.0f / static_cast<float>(kernel_size);
    auto kern = Tensor::full({kernel_size}, static_cast<double>(val), DType::Float32);
    if (!kern) return std::unexpected(kern.error());

    return convolve_separable(input, *kern, *kern, border);
}

auto sobel(const Tensor& input, BorderMode border)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "sobel"); !v) return std::unexpected(v.error());

    // Sobel kernels (3x3)
    // Sobel-X: [[-1 0 1], [-2 0 2], [-1 0 1]]
    // Sobel-Y: [[-1 -2 -1], [0 0 0], [1 2 1]]
    float sx_data[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sy_data[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    std::size_t kshape[] = {3, 3};
    auto sobel_x = Tensor::from_data(sx_data, kshape, DType::Float32);
    auto sobel_y = Tensor::from_data(sy_data, kshape, DType::Float32);
    if (!sobel_x) return std::unexpected(sobel_x.error());
    if (!sobel_y) return std::unexpected(sobel_y.error());

    auto gx = convolve(input, *sobel_x, border);
    if (!gx) return std::unexpected(gx.error());
    auto gy = convolve(input, *sobel_y, border);
    if (!gy) return std::unexpected(gy.error());

    // Compute gradient magnitude: sqrt(gx^2 + gy^2)
    auto rows = input.shape()[0];
    auto cols = input.shape()[1];
    auto numel = rows * cols;

    auto out = Tensor::zeros({rows, cols}, DType::Float32);
    if (!out) return std::unexpected(out.error());

    const auto* gx_data = gx->data<float>();
    const auto* gy_data = gy->data<float>();
    auto* dst = out->data<float>();

    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = std::sqrt(gx_data[i] * gx_data[i] + gy_data[i] * gy_data[i]);
    }

    return std::move(*out);
}

auto sharpen(const Tensor& input, float amount, BorderMode border)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input, "sharpen"); !v) return std::unexpected(v.error());

    // Sharpen kernel: identity + amount * (identity - blur)
    // Using the standard sharpening kernel:
    //   [0  -a  0 ]
    //   [-a 1+4a -a]
    //   [0  -a  0 ]
    float a = amount;
    float k_data[] = {
        0.0f,  -a,        0.0f,
        -a,    1.0f + 4*a, -a,
        0.0f,  -a,        0.0f
    };

    std::size_t kshape[] = {3, 3};
    auto kern = Tensor::from_data(k_data, kshape, DType::Float32);
    if (!kern) return std::unexpected(kern.error());

    return convolve(input, *kern, border);
}

} // namespace utils
