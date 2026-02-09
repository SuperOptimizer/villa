#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vc::simd {

// --------------------------------------------------------------------------
// TileGeometry<N> — compile-time constants for power-of-2 tile size N
// --------------------------------------------------------------------------
template <int N>
struct TileGeometry {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
    static_assert(N >= 8, "N must be at least 8");

    static constexpr int kSize = N;
    static constexpr int kShift = __builtin_ctz(static_cast<unsigned>(N));
    static constexpr int kMask = N - 1;

    static constexpr size_t volume1d = N;
    static constexpr size_t volume2d = static_cast<size_t>(N) * N;
    static constexpr size_t volume3d = static_cast<size_t>(N) * N * N;

    // Column-major ZYX: z is stride-1 (compatible with z5/xtensor)
    static constexpr size_t offset3d(int z, int y, int x) {
        return static_cast<size_t>(z) +
               static_cast<size_t>(y) * N +
               static_cast<size_t>(x) * N * N;
    }

    static constexpr size_t offset2d(int y, int x) {
        return static_cast<size_t>(y) + static_cast<size_t>(x) * N;
    }

    static constexpr size_t offset1d(int x) {
        return static_cast<size_t>(x);
    }

    // From global coords to (chunk_id, local_offset)
    static constexpr int chunk_id(int coord) { return coord >> kShift; }
    static constexpr int local(int coord) { return coord & kMask; }
};

// --------------------------------------------------------------------------
// Tile3D<T, N> — Fixed-size aligned N^3 storage
// --------------------------------------------------------------------------
template <typename T, int N>
struct alignas(64) Tile3D {
    using Geom = TileGeometry<N>;
    T data_[Geom::volume3d];

    T& operator()(int z, int y, int x) {
        return data_[Geom::offset3d(z, y, x)];
    }
    const T& operator()(int z, int y, int x) const {
        return data_[Geom::offset3d(z, y, x)];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    static constexpr size_t bytes() { return Geom::volume3d * sizeof(T); }

    T* __restrict__ aligned_data() {
        return static_cast<T*>(__builtin_assume_aligned(data_, 64));
    }
    const T* __restrict__ aligned_data() const {
        return static_cast<const T*>(__builtin_assume_aligned(data_, 64));
    }
};

// --------------------------------------------------------------------------
// Tile2D<T, N> — Fixed-size aligned N^2 storage
// --------------------------------------------------------------------------
template <typename T, int N>
struct alignas(64) Tile2D {
    using Geom = TileGeometry<N>;
    T data_[Geom::volume2d];

    T& operator()(int y, int x) {
        return data_[Geom::offset2d(y, x)];
    }
    const T& operator()(int y, int x) const {
        return data_[Geom::offset2d(y, x)];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    static constexpr size_t bytes() { return Geom::volume2d * sizeof(T); }
};

// --------------------------------------------------------------------------
// Tile1D<T, N> — Fixed-size aligned N storage
// --------------------------------------------------------------------------
template <typename T, int N>
struct alignas(64) Tile1D {
    using Geom = TileGeometry<N>;
    T data_[Geom::volume1d];

    T& operator()(int x) { return data_[x]; }
    const T& operator()(int x) const { return data_[x]; }

    T* data() { return data_; }
    const T* data() const { return data_; }
    static constexpr size_t bytes() { return Geom::volume1d * sizeof(T); }
};

// Convenience alias: Tile defaults to 3D
template <typename T, int N>
using Tile = Tile3D<T, N>;

}  // namespace vc::simd
