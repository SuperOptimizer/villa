#include "utils/compositing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace utils {
namespace {

template <typename T>
auto composite_mean(const T* src, T* dst, std::size_t layers, std::size_t h,
                    std::size_t w) -> void {
    const auto pixels = h * w;
    for (std::size_t p = 0; p < pixels; ++p) {
        double sum = 0.0;
        for (std::size_t l = 0; l < layers; ++l) {
            sum += static_cast<double>(src[l * pixels + p]);
        }
        if constexpr (std::is_floating_point_v<T>) {
            dst[p] = static_cast<T>(sum / static_cast<double>(layers));
        } else {
            // Round for integer types
            dst[p] = static_cast<T>(sum / static_cast<double>(layers) + 0.5);
        }
    }
}

template <typename T>
auto composite_max(const T* src, T* dst, std::size_t layers, std::size_t h,
                   std::size_t w) -> void {
    const auto pixels = h * w;
    for (std::size_t p = 0; p < pixels; ++p) {
        T val = src[p];
        for (std::size_t l = 1; l < layers; ++l) {
            T v = src[l * pixels + p];
            if (v > val) {
                val = v;
            }
        }
        dst[p] = val;
    }
}

template <typename T>
auto composite_min(const T* src, T* dst, std::size_t layers, std::size_t h,
                   std::size_t w) -> void {
    const auto pixels = h * w;
    for (std::size_t p = 0; p < pixels; ++p) {
        T val = src[p];
        for (std::size_t l = 1; l < layers; ++l) {
            T v = src[l * pixels + p];
            if (v < val) {
                val = v;
            }
        }
        dst[p] = val;
    }
}

template <typename T>
auto composite_median(const T* src, T* dst, std::size_t layers, std::size_t h,
                      std::size_t w) -> void {
    const auto pixels = h * w;
    std::vector<T> scratch(layers);
    for (std::size_t p = 0; p < pixels; ++p) {
        for (std::size_t l = 0; l < layers; ++l) {
            scratch[l] = src[l * pixels + p];
        }
        std::sort(scratch.begin(), scratch.end());
        if (layers % 2 == 1) {
            dst[p] = scratch[layers / 2];
        } else {
            if constexpr (std::is_floating_point_v<T>) {
                dst[p] = static_cast<T>(
                    (static_cast<double>(scratch[layers / 2 - 1]) +
                     static_cast<double>(scratch[layers / 2])) *
                    0.5);
            } else {
                dst[p] = static_cast<T>(
                    (static_cast<unsigned>(scratch[layers / 2 - 1]) +
                     static_cast<unsigned>(scratch[layers / 2])) /
                    2);
            }
        }
    }
}

template <typename T>
auto dispatch_method(CompositeMethod method, const T* src, T* dst,
                     std::size_t layers, std::size_t h,
                     std::size_t w) -> void {
    switch (method) {
        case CompositeMethod::Mean:
            composite_mean(src, dst, layers, h, w);
            break;
        case CompositeMethod::Max:
            composite_max(src, dst, layers, h, w);
            break;
        case CompositeMethod::Min:
            composite_min(src, dst, layers, h, w);
            break;
        case CompositeMethod::Median:
            composite_median(src, dst, layers, h, w);
            break;
    }
}

} // namespace

auto composite(const Tensor& layers, const CompositeOptions& opts)
    -> std::expected<Tensor, std::string> {
    // Validate dimensionality
    if (layers.ndim() != 3) {
        return std::unexpected("composite: input must be a 3D tensor "
                               "(layers x H x W), got ndim=" +
                               std::to_string(layers.ndim()));
    }

    // Validate dtype
    const auto dt = layers.dtype();
    if (dt != DType::Float32 && dt != DType::UInt8) {
        return std::unexpected(
            "composite: unsupported dtype " + std::string(dtype_name(dt)) +
            ", expected Float32 or UInt8");
    }

    const auto shape = layers.shape();
    const auto num_layers = shape[0];
    const auto h = shape[1];
    const auto w = shape[2];

    // Single layer: return a squeezed copy
    if (num_layers == 1) {
        auto squeezed = layers.squeeze(0);
        if (!squeezed) {
            return std::unexpected(squeezed.error());
        }
        return squeezed->contiguous().clone();
    }

    // Ensure contiguous C-order for simple striding
    auto src = layers.contiguous();

    // Allocate output
    std::vector<std::size_t> out_shape{h, w};
    auto out = Tensor::zeros(out_shape, dt);
    if (!out) {
        return std::unexpected(out.error());
    }

    if (dt == DType::Float32) {
        dispatch_method(opts.method, src.data<float>(), out->data<float>(),
                        num_layers, h, w);
    } else {
        dispatch_method(opts.method, src.data<std::uint8_t>(),
                        out->data<std::uint8_t>(), num_layers, h, w);
    }

    return out;
}

} // namespace utils
