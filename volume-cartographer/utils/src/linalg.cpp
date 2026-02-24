#include "utils/linalg.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

namespace utils {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto read_element_at(const Tensor& t, std::size_t i) -> double {
    auto* base = static_cast<const std::byte*>(t.data_ptr());
    auto off = static_cast<std::ptrdiff_t>(i) * t.strides()[0];
    auto* p = base + off;

    switch (t.dtype()) {
        case DType::Float32: return static_cast<double>(*reinterpret_cast<const float*>(p));
        case DType::Float64: return *reinterpret_cast<const double*>(p);
        case DType::Int8:    return static_cast<double>(*reinterpret_cast<const std::int8_t*>(p));
        case DType::Int16:   return static_cast<double>(*reinterpret_cast<const std::int16_t*>(p));
        case DType::Int32:   return static_cast<double>(*reinterpret_cast<const std::int32_t*>(p));
        case DType::Int64:   return static_cast<double>(*reinterpret_cast<const std::int64_t*>(p));
        case DType::UInt8:   return static_cast<double>(*reinterpret_cast<const std::uint8_t*>(p));
        case DType::UInt16:  return static_cast<double>(*reinterpret_cast<const std::uint16_t*>(p));
        case DType::UInt32:  return static_cast<double>(*reinterpret_cast<const std::uint32_t*>(p));
        case DType::UInt64:  return static_cast<double>(*reinterpret_cast<const std::uint64_t*>(p));
    }
    __builtin_unreachable();
}

static auto read_matrix_element(const Tensor& t, std::size_t row, std::size_t col) -> double {
    auto* base = static_cast<const std::byte*>(t.data_ptr());
    auto off = static_cast<std::ptrdiff_t>(row) * t.strides()[0] +
               static_cast<std::ptrdiff_t>(col) * t.strides()[1];
    auto* p = base + off;

    switch (t.dtype()) {
        case DType::Float32: return static_cast<double>(*reinterpret_cast<const float*>(p));
        case DType::Float64: return *reinterpret_cast<const double*>(p);
        case DType::Int8:    return static_cast<double>(*reinterpret_cast<const std::int8_t*>(p));
        case DType::Int16:   return static_cast<double>(*reinterpret_cast<const std::int16_t*>(p));
        case DType::Int32:   return static_cast<double>(*reinterpret_cast<const std::int32_t*>(p));
        case DType::Int64:   return static_cast<double>(*reinterpret_cast<const std::int64_t*>(p));
        case DType::UInt8:   return static_cast<double>(*reinterpret_cast<const std::uint8_t*>(p));
        case DType::UInt16:  return static_cast<double>(*reinterpret_cast<const std::uint16_t*>(p));
        case DType::UInt32:  return static_cast<double>(*reinterpret_cast<const std::uint32_t*>(p));
        case DType::UInt64:  return static_cast<double>(*reinterpret_cast<const std::uint64_t*>(p));
    }
    __builtin_unreachable();
}

// ---------------------------------------------------------------------------
// dot
// ---------------------------------------------------------------------------

auto dot(const Tensor& a, const Tensor& b) -> Tensor {
    assert(a.ndim() == 1 && "dot: first argument must be 1D");
    assert(b.ndim() == 1 && "dot: second argument must be 1D");
    assert(a.shape()[0] == b.shape()[0] && "dot: size mismatch");

    auto out_dt = promote_dtype(a.dtype(), b.dtype());
    double total = 0.0;
    for (std::size_t i = 0; i < a.shape()[0]; ++i) {
        total += read_element_at(a, i) * read_element_at(b, i);
    }

    auto result = Tensor::full({}, total, out_dt);
    assert(result.has_value());
    return std::move(*result);
}

// ---------------------------------------------------------------------------
// norm
// ---------------------------------------------------------------------------

auto norm(const Tensor& a) -> Tensor {
    assert(a.ndim() == 1 && "norm: argument must be 1D");

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < a.shape()[0]; ++i) {
        double v = read_element_at(a, i);
        sum_sq += v * v;
    }

    auto result = Tensor::full({}, std::sqrt(sum_sq), DType::Float64);
    assert(result.has_value());
    return std::move(*result);
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

auto matmul(const Tensor& a, const Tensor& b)
    -> std::expected<Tensor, std::string> {
    if (a.ndim() != 2) return std::unexpected("matmul: first argument must be 2D");
    if (b.ndim() != 2) return std::unexpected("matmul: second argument must be 2D");
    if (a.shape()[1] != b.shape()[0]) {
        return std::unexpected("matmul: inner dimensions must match (" +
                               std::to_string(a.shape()[1]) + " vs " +
                               std::to_string(b.shape()[0]) + ")");
    }

    auto M = a.shape()[0];
    auto K = a.shape()[1];
    auto N = b.shape()[1];
    auto out_dt = promote_dtype(a.dtype(), b.dtype());

    auto result = Tensor::zeros({M, N}, out_dt);
    if (!result) return std::unexpected(result.error());
    auto& out = *result;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += read_matrix_element(a, i, k) * read_matrix_element(b, k, j);
            }
            // Write to output
            auto* base = static_cast<std::byte*>(out.data_ptr());
            auto off = static_cast<std::ptrdiff_t>(i) * out.strides()[0] +
                       static_cast<std::ptrdiff_t>(j) * out.strides()[1];
            switch (out_dt) {
                case DType::Float32: *reinterpret_cast<float*>(base + off) = static_cast<float>(sum); break;
                case DType::Float64: *reinterpret_cast<double*>(base + off) = sum; break;
                case DType::Int32:   *reinterpret_cast<std::int32_t*>(base + off) = static_cast<std::int32_t>(sum); break;
                case DType::Int64:   *reinterpret_cast<std::int64_t*>(base + off) = static_cast<std::int64_t>(sum); break;
                default: *reinterpret_cast<double*>(base + off) = sum; break;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// cross
// ---------------------------------------------------------------------------

auto cross(const Tensor& a, const Tensor& b) -> Tensor {
    assert(a.ndim() == 1 && a.shape()[0] == 3 && "cross: first argument must be 1D with 3 elements");
    assert(b.ndim() == 1 && b.shape()[0] == 3 && "cross: second argument must be 1D with 3 elements");

    double a0 = read_element_at(a, 0);
    double a1 = read_element_at(a, 1);
    double a2 = read_element_at(a, 2);
    double b0 = read_element_at(b, 0);
    double b1 = read_element_at(b, 1);
    double b2 = read_element_at(b, 2);

    double c0 = a1 * b2 - a2 * b1;
    double c1 = a2 * b0 - a0 * b2;
    double c2 = a0 * b1 - a1 * b0;

    auto out_dt = promote_dtype(a.dtype(), b.dtype());
    double data[3] = {c0, c1, c2};
    auto result = Tensor::from_data(data, {{3}}, DType::Float64);
    assert(result.has_value());
    if (out_dt != DType::Float64) {
        return result->to(out_dt);
    }
    return std::move(*result);
}

// ---------------------------------------------------------------------------
// normalize
// ---------------------------------------------------------------------------

auto normalize(const Tensor& a) -> Tensor {
    auto n = norm(a);
    double n_val = *static_cast<const double*>(n.data_ptr());
    return a / n_val;
}

} // namespace utils
