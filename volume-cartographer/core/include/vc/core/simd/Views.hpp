#pragma once

#include "Tile.hpp"

#include <array>
#include <cstddef>

namespace vc::simd {

// --------------------------------------------------------------------------
// VolumeView<T> — non-owning 3D sub-region view for passing to algorithms
// --------------------------------------------------------------------------
template <typename T>
struct VolumeView {
    T* data;
    int sz, sy, sx;
    size_t stride_z, stride_y, stride_x;

    T& operator()(int z, int y, int x) {
        return data[z * stride_z + y * stride_y + x * stride_x];
    }
    const T& operator()(int z, int y, int x) const {
        return data[z * stride_z + y * stride_y + x * stride_x];
    }

    std::array<int, 3> shape() const { return {sz, sy, sx}; }
};

// --------------------------------------------------------------------------
// SliceView<T> — non-owning 2D slice view
// --------------------------------------------------------------------------
template <typename T>
struct SliceView {
    T* data;
    int sy, sx;
    size_t stride_y, stride_x;

    T& operator()(int y, int x) {
        return data[y * stride_y + x * stride_x];
    }
    const T& operator()(int y, int x) const {
        return data[y * stride_y + x * stride_x];
    }

    std::array<int, 2> shape() const { return {sy, sx}; }
};

// --------------------------------------------------------------------------
// Factory functions
// --------------------------------------------------------------------------

// Extract a Z-slice from a Tile3D as a 2D view (zero-copy)
template <typename T, int N>
SliceView<T> tile_slice_z(Tile3D<T, N>& tile, int z_local) {
    return {tile.data() + z_local, N, N, N, static_cast<size_t>(N) * N};
}

template <typename T, int N>
SliceView<const T> tile_slice_z(const Tile3D<T, N>& tile, int z_local) {
    return {tile.data() + z_local, N, N, N, static_cast<size_t>(N) * N};
}

// Wrap raw external buffer as a column-major ZYX view
template <typename T>
VolumeView<T> wrap_raw(T* data, int sz, int sy, int sx) {
    return {data, sz, sy, sx,
            1,
            static_cast<size_t>(sz),
            static_cast<size_t>(sz) * sy};
}

// Wrap a Tile3D as a VolumeView
template <typename T, int N>
VolumeView<T> wrap_tile(Tile3D<T, N>& tile) {
    return {tile.data(), N, N, N,
            1,
            static_cast<size_t>(N),
            static_cast<size_t>(N) * N};
}

template <typename T, int N>
VolumeView<const T> wrap_tile(const Tile3D<T, N>& tile) {
    return {tile.data(), N, N, N,
            1,
            static_cast<size_t>(N),
            static_cast<size_t>(N) * N};
}

}  // namespace vc::simd
