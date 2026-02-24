#include "utils/distance_transform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace utils {
namespace {

constexpr double kInf = 1e30;

// 1D distance transform using Felzenszwalb & Huttenlocher's parabola method.
// f: input/output array of squared distances
// n: length of the array
// spacing: grid spacing for this axis
auto dt_1d(double* f, std::size_t n, double spacing) -> void {
    if (n == 0) return;
    if (n == 1) return; // single element, distance is already correct

    double sp2 = spacing * spacing;

    std::vector<std::size_t> v(n);  // locations of parabolas
    std::vector<double> z(n + 1);   // boundaries between parabolas
    v[0] = 0;
    z[0] = -kInf;
    z[1] = kInf;
    std::size_t k = 0;

    for (std::size_t q = 1; q < n; ++q) {
        double fq = f[q];
        double sq = static_cast<double>(q);
        while (true) {
            double sv = static_cast<double>(v[k]);
            // Intersection of parabolas at v[k] and q
            double s = ((fq + sq * sq * sp2) - (f[v[k]] + sv * sv * sp2))
                       / (2.0 * sp2 * (sq - sv));
            if (s > z[k]) {
                k++;
                v[k] = q;
                z[k] = s;
                z[k + 1] = kInf;
                break;
            }
            if (k == 0) {
                v[0] = q;
                z[1] = kInf;
                break;
            }
            k--;
        }
    }

    k = 0;
    for (std::size_t q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<double>(q)) {
            k++;
        }
        double dq = static_cast<double>(q) - static_cast<double>(v[k]);
        f[q] = dq * dq * sp2 + f[v[k]];
    }
}

auto is_foreground_nonzero(const void* data, DType dt, std::size_t idx) -> bool {
    switch (dt) {
        case DType::Float32: return static_cast<const float*>(data)[idx] != 0.0f;
        case DType::Float64: return static_cast<const double*>(data)[idx] != 0.0;
        case DType::Int8:    return static_cast<const std::int8_t*>(data)[idx] != 0;
        case DType::Int16:   return static_cast<const std::int16_t*>(data)[idx] != 0;
        case DType::Int32:   return static_cast<const std::int32_t*>(data)[idx] != 0;
        case DType::Int64:   return static_cast<const std::int64_t*>(data)[idx] != 0;
        case DType::UInt8:   return static_cast<const std::uint8_t*>(data)[idx] != 0;
        case DType::UInt16:  return static_cast<const std::uint16_t*>(data)[idx] != 0;
        case DType::UInt32:  return static_cast<const std::uint32_t*>(data)[idx] != 0;
        case DType::UInt64:  return static_cast<const std::uint64_t*>(data)[idx] != 0;
    }
    return false;
}

auto dt_impl(const Tensor& input, const DistanceTransformOptions& opts, bool invert)
    -> std::expected<Tensor, std::string> {

    if (input.ndim() < 1) {
        return std::unexpected("distance_transform: input must have at least 1 dimension");
    }
    if (opts.output_dtype != DType::Float32 && opts.output_dtype != DType::Float64) {
        return std::unexpected("distance_transform: output_dtype must be Float32 or Float64");
    }

    auto shape = input.shape();
    auto ndim = input.ndim();
    auto numel = input.numel();

    // Validate spacing
    std::vector<double> spacing = opts.spacing;
    if (spacing.empty()) {
        spacing.assign(ndim, 1.0);
    }
    if (spacing.size() != ndim) {
        return std::unexpected("distance_transform: spacing size (" +
                               std::to_string(spacing.size()) +
                               ") must match ndim (" + std::to_string(ndim) + ")");
    }
    for (auto s : spacing) {
        if (s <= 0.0) {
            return std::unexpected("distance_transform: spacing must be positive");
        }
    }

    // Make input contiguous
    auto contiguous = input.contiguous();
    const void* in_data = contiguous.data_ptr();

    // Initialize working buffer (Float64)
    std::vector<double> work(numel);
    for (std::size_t i = 0; i < numel; ++i) {
        bool fg = is_foreground_nonzero(in_data, contiguous.dtype(), i);
        if (invert) fg = !fg;
        work[i] = fg ? 0.0 : kInf;
    }

    // Apply 1D distance transform along each axis
    // For each axis d, iterate over all lines along that axis
    std::vector<std::size_t> shape_vec(shape.begin(), shape.end());

    for (std::size_t d = 0; d < ndim; ++d) {
        auto n = shape_vec[d];
        if (n <= 1) continue;

        // Compute the stride for axis d in C-order linear indexing
        std::size_t axis_stride = 1;
        for (std::size_t k = d + 1; k < ndim; ++k) {
            axis_stride *= shape_vec[k];
        }

        // Number of lines = total elements / elements along this axis
        std::size_t num_lines = numel / n;

        // Temp buffer for one line
        std::vector<double> line_buf(n);

        for (std::size_t line = 0; line < num_lines; ++line) {
            // Compute base offset for this line
            // The line index encodes all dims except d
            // We need to decompose 'line' into the other dimensions
            std::size_t base = 0;
            std::size_t remainder = line;
            for (std::size_t k = 0; k < ndim; ++k) {
                if (k == d) continue;
                std::size_t dim_stride = 1;
                for (std::size_t j = k + 1; j < ndim; ++j) {
                    if (j == d) continue;
                    dim_stride *= shape_vec[j];
                }
                auto coord_k = remainder / dim_stride;
                remainder %= dim_stride;

                // Compute stride in full array for dimension k
                std::size_t full_stride = 1;
                for (std::size_t j = k + 1; j < ndim; ++j) {
                    full_stride *= shape_vec[j];
                }
                base += coord_k * full_stride;
            }

            // Extract line
            for (std::size_t i = 0; i < n; ++i) {
                line_buf[i] = work[base + i * axis_stride];
            }

            // Transform
            dt_1d(line_buf.data(), n, spacing[d]);

            // Write back
            for (std::size_t i = 0; i < n; ++i) {
                work[base + i * axis_stride] = line_buf[i];
            }
        }
    }

    // Take sqrt if not squared
    if (!opts.squared) {
        for (auto& v : work) {
            v = std::sqrt(v);
        }
    }

    // Create output tensor
    auto out = Tensor::zeros(shape, opts.output_dtype);
    if (!out) {
        return std::unexpected(out.error());
    }

    if (opts.output_dtype == DType::Float64) {
        auto* dst = out->data<double>();
        for (std::size_t i = 0; i < numel; ++i) {
            dst[i] = work[i];
        }
    } else {
        auto* dst = out->data<float>();
        for (std::size_t i = 0; i < numel; ++i) {
            dst[i] = static_cast<float>(work[i]);
        }
    }

    return std::move(*out);
}

} // namespace

auto distance_transform(const Tensor& input, const DistanceTransformOptions& opts)
    -> std::expected<Tensor, std::string> {
    return dt_impl(input, opts, false);
}

auto distance_transform_inv(const Tensor& input, const DistanceTransformOptions& opts)
    -> std::expected<Tensor, std::string> {
    return dt_impl(input, opts, true);
}

} // namespace utils
