#include "utils/interpolation.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace utils {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto read_element(const Tensor& t, const std::ptrdiff_t* indices,
                         std::size_t ndim) -> double {
    auto* base = static_cast<const std::byte*>(t.data_ptr());
    std::ptrdiff_t off = 0;
    for (std::size_t i = 0; i < ndim; ++i) {
        off += indices[i] * t.strides()[i];
    }

    switch (t.dtype()) {
        case DType::Float32: return static_cast<double>(*reinterpret_cast<const float*>(base + off));
        case DType::Float64: return *reinterpret_cast<const double*>(base + off);
        case DType::Int8:    return static_cast<double>(*reinterpret_cast<const std::int8_t*>(base + off));
        case DType::Int16:   return static_cast<double>(*reinterpret_cast<const std::int16_t*>(base + off));
        case DType::Int32:   return static_cast<double>(*reinterpret_cast<const std::int32_t*>(base + off));
        case DType::Int64:   return static_cast<double>(*reinterpret_cast<const std::int64_t*>(base + off));
        case DType::UInt8:   return static_cast<double>(*reinterpret_cast<const std::uint8_t*>(base + off));
        case DType::UInt16:  return static_cast<double>(*reinterpret_cast<const std::uint16_t*>(base + off));
        case DType::UInt32:  return static_cast<double>(*reinterpret_cast<const std::uint32_t*>(base + off));
        case DType::UInt64:  return static_cast<double>(*reinterpret_cast<const std::uint64_t*>(base + off));
    }
    __builtin_unreachable();
}

static auto clamp_idx(std::ptrdiff_t i, std::ptrdiff_t size) -> std::ptrdiff_t {
    if (i < 0) return 0;
    if (i >= size) return size - 1;
    return i;
}

static auto in_bounds(std::ptrdiff_t i, std::ptrdiff_t size) -> bool {
    return i >= 0 && i < size;
}

// Sample 2D with boundary handling; returns value or 0 for out-of-bounds with Zero mode
static auto sample2d(const Tensor& t, std::ptrdiff_t iy, std::ptrdiff_t ix,
                     BoundaryMode boundary) -> double {
    auto h = static_cast<std::ptrdiff_t>(t.shape()[0]);
    auto w = static_cast<std::ptrdiff_t>(t.shape()[1]);

    if (boundary == BoundaryMode::Zero) {
        if (!in_bounds(iy, h) || !in_bounds(ix, w)) return 0.0;
    } else {
        iy = clamp_idx(iy, h);
        ix = clamp_idx(ix, w);
    }
    std::ptrdiff_t idx[2] = {iy, ix};
    return read_element(t, idx, 2);
}

// Sample 3D with boundary handling
static auto sample3d(const Tensor& t, std::ptrdiff_t iz, std::ptrdiff_t iy, std::ptrdiff_t ix,
                     BoundaryMode boundary) -> double {
    auto d = static_cast<std::ptrdiff_t>(t.shape()[0]);
    auto h = static_cast<std::ptrdiff_t>(t.shape()[1]);
    auto w = static_cast<std::ptrdiff_t>(t.shape()[2]);

    if (boundary == BoundaryMode::Zero) {
        if (!in_bounds(iz, d) || !in_bounds(iy, h) || !in_bounds(ix, w)) return 0.0;
    } else {
        iz = clamp_idx(iz, d);
        iy = clamp_idx(iy, h);
        ix = clamp_idx(ix, w);
    }
    std::ptrdiff_t idx[3] = {iz, iy, ix};
    return read_element(t, idx, 3);
}

// ---------------------------------------------------------------------------
// Single-point 2D
// ---------------------------------------------------------------------------

auto interpolate2d(const Tensor& t, double y, double x,
                   InterpMode interp, BoundaryMode boundary) -> double {
    assert(t.ndim() == 2 && "interpolate2d requires a 2D tensor");

    if (interp == InterpMode::Nearest) {
        auto iy = static_cast<std::ptrdiff_t>(std::round(y));
        auto ix = static_cast<std::ptrdiff_t>(std::round(x));
        return sample2d(t, iy, ix, boundary);
    }

    // Bilinear
    auto y0 = static_cast<std::ptrdiff_t>(std::floor(y));
    auto x0 = static_cast<std::ptrdiff_t>(std::floor(x));
    auto y1 = y0 + 1;
    auto x1 = x0 + 1;

    double fy = y - static_cast<double>(y0);
    double fx = x - static_cast<double>(x0);

    double v00 = sample2d(t, y0, x0, boundary);
    double v01 = sample2d(t, y0, x1, boundary);
    double v10 = sample2d(t, y1, x0, boundary);
    double v11 = sample2d(t, y1, x1, boundary);

    return v00 * (1.0 - fy) * (1.0 - fx) +
           v01 * (1.0 - fy) * fx +
           v10 * fy * (1.0 - fx) +
           v11 * fy * fx;
}

// ---------------------------------------------------------------------------
// Single-point 3D
// ---------------------------------------------------------------------------

auto interpolate3d(const Tensor& t, double z, double y, double x,
                   InterpMode interp, BoundaryMode boundary) -> double {
    assert(t.ndim() == 3 && "interpolate3d requires a 3D tensor");

    if (interp == InterpMode::Nearest) {
        auto iz = static_cast<std::ptrdiff_t>(std::round(z));
        auto iy = static_cast<std::ptrdiff_t>(std::round(y));
        auto ix = static_cast<std::ptrdiff_t>(std::round(x));
        return sample3d(t, iz, iy, ix, boundary);
    }

    // Trilinear
    auto z0 = static_cast<std::ptrdiff_t>(std::floor(z));
    auto y0 = static_cast<std::ptrdiff_t>(std::floor(y));
    auto x0 = static_cast<std::ptrdiff_t>(std::floor(x));
    auto z1 = z0 + 1;
    auto y1 = y0 + 1;
    auto x1 = x0 + 1;

    double fz = z - static_cast<double>(z0);
    double fy = y - static_cast<double>(y0);
    double fx = x - static_cast<double>(x0);

    double v000 = sample3d(t, z0, y0, x0, boundary);
    double v001 = sample3d(t, z0, y0, x1, boundary);
    double v010 = sample3d(t, z0, y1, x0, boundary);
    double v011 = sample3d(t, z0, y1, x1, boundary);
    double v100 = sample3d(t, z1, y0, x0, boundary);
    double v101 = sample3d(t, z1, y0, x1, boundary);
    double v110 = sample3d(t, z1, y1, x0, boundary);
    double v111 = sample3d(t, z1, y1, x1, boundary);

    // Interpolate along x
    double v00 = v000 * (1.0 - fx) + v001 * fx;
    double v01 = v010 * (1.0 - fx) + v011 * fx;
    double v10 = v100 * (1.0 - fx) + v101 * fx;
    double v11 = v110 * (1.0 - fx) + v111 * fx;

    // Interpolate along y
    double v0 = v00 * (1.0 - fy) + v01 * fy;
    double v1 = v10 * (1.0 - fy) + v11 * fy;

    // Interpolate along z
    return v0 * (1.0 - fz) + v1 * fz;
}

// ---------------------------------------------------------------------------
// Batch interpolation
// ---------------------------------------------------------------------------

auto interpolate(const Tensor& t, const Tensor& coords,
                 InterpMode interp, BoundaryMode boundary)
    -> std::expected<Tensor, std::string> {
    if (coords.ndim() != 2) {
        return std::unexpected("coords must be a 2D tensor [N, ndim]");
    }
    if (coords.dtype() != DType::Float64) {
        return std::unexpected("coords must be Float64");
    }

    auto coord_dim = coords.shape()[1];
    if (coord_dim != 2 && coord_dim != 3) {
        return std::unexpected("coords second dimension must be 2 or 3");
    }
    if (t.ndim() != coord_dim) {
        return std::unexpected("tensor ndim must match coords dimension");
    }

    auto n = coords.shape()[0];
    auto result = Tensor::zeros({n}, DType::Float64);
    if (!result) return std::unexpected(result.error());
    auto& out = *result;
    auto* out_ptr = out.data<double>();

    auto* coord_base = static_cast<const std::byte*>(coords.data_ptr());

    for (std::size_t i = 0; i < n; ++i) {
        // Read coordinates for point i
        auto coord_row_off = static_cast<std::ptrdiff_t>(i) * coords.strides()[0];

        if (coord_dim == 2) {
            auto* py = reinterpret_cast<const double*>(coord_base + coord_row_off);
            auto* px = reinterpret_cast<const double*>(coord_base + coord_row_off + coords.strides()[1]);
            out_ptr[i] = interpolate2d(t, *py, *px, interp, boundary);
        } else {
            auto* pz = reinterpret_cast<const double*>(coord_base + coord_row_off);
            auto* py = reinterpret_cast<const double*>(coord_base + coord_row_off + coords.strides()[1]);
            auto* px = reinterpret_cast<const double*>(coord_base + coord_row_off + 2 * coords.strides()[1]);
            out_ptr[i] = interpolate3d(t, *pz, *py, *px, interp, boundary);
        }
    }
    return out;
}

} // namespace utils
