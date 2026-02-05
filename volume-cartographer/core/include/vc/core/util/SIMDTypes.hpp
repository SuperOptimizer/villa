#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace vc::simd {

// ============================================================================
// ChunkView<T, N> — Fixed-stride view into an N×N×N voxel chunk
// ============================================================================
//
// Wraps a raw pointer with compile-time strides so the compiler can:
// - Replace multiplies with shifts (lz * 1024 → lz << 10 for N=32)
// - Prove memory bounds for auto-vectorization
// - Eliminate redundant address calculations
//
// No bounds checking. Caller must ensure coordinates are valid.
// Use MASK to convert global coords to local: local = global & MASK

template<typename T, int N>
struct ChunkView {
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

    static constexpr int SIZE = N;
    static constexpr int MASK = N - 1;
    static constexpr size_t STRIDE_Z = static_cast<size_t>(N) * N;
    static constexpr size_t STRIDE_Y = N;
    static constexpr size_t TOTAL = static_cast<size_t>(N) * N * N;

    // 3D array reference: compiler knows the full extent (N*N*N elements).
    // This resolves "cannot identify array bounds" — the compiler can prove
    // all accesses are within [0, N*N*N) for auto-vectorization.
    using Array3D = const T[N][N][N];
    Array3D& arr;

    explicit ChunkView(const T* d) noexcept
        : arr(*reinterpret_cast<const T(*)[N][N][N]>(d)) {}

    [[gnu::always_inline]] T operator()(int z, int y, int x) const noexcept {
        return arr[z][y][x];
    }

    [[gnu::always_inline]] T nearest(int lz, int ly, int lx) const noexcept {
        return arr[lz][ly][lx];
    }

    // FMA-friendly trilinear: a + t*(b-a) generates fmadd on ARM
    // Uses 3D array indexing so the compiler sees bounded accesses.
    [[gnu::always_inline]] float trilinear(
        int lz, int ly, int lx,
        float fz, float fy, float fx) const noexcept
    {
        const float c000 = arr[lz][ly][lx];
        const float c001 = arr[lz][ly][lx + 1];
        const float c010 = arr[lz][ly + 1][lx];
        const float c011 = arr[lz][ly + 1][lx + 1];
        const float c100 = arr[lz + 1][ly][lx];
        const float c101 = arr[lz + 1][ly][lx + 1];
        const float c110 = arr[lz + 1][ly + 1][lx];
        const float c111 = arr[lz + 1][ly + 1][lx + 1];

        const float c00 = c000 + fx * (c001 - c000);
        const float c01 = c010 + fx * (c011 - c010);
        const float c10 = c100 + fx * (c101 - c100);
        const float c11 = c110 + fx * (c111 - c110);

        const float cl0 = c00 + fy * (c01 - c00);
        const float cl1 = c10 + fy * (c11 - c10);

        return cl0 + fz * (cl1 - cl0);
    }

    // Gradient via central differences at integer coordinates
    [[gnu::always_inline]] void gradient(
        int lz, int ly, int lx,
        float& gx, float& gy, float& gz) const noexcept
    {
        gx = static_cast<float>(arr[lz][ly][lx + 1]) - static_cast<float>(arr[lz][ly][lx - 1]);
        gy = static_cast<float>(arr[lz][ly + 1][lx]) - static_cast<float>(arr[lz][ly - 1][lx]);
        gz = static_cast<float>(arr[lz + 1][ly][lx]) - static_cast<float>(arr[lz - 1][ly][lx]);
    }
};

// Runtime-stride fallback for non-power-of-2 or unusual chunk sizes
template<typename T>
struct ChunkViewRT {
    const T* __restrict__ data;
    const size_t stride_z, stride_y;

    ChunkViewRT(const T* d, size_t sz, size_t sy)
        : data(d), stride_z(sz), stride_y(sy) {}

    [[gnu::always_inline]] T operator()(int z, int y, int x) const noexcept {
        return data[static_cast<size_t>(z) * stride_z + static_cast<size_t>(y) * stride_y + x];
    }

    [[gnu::always_inline]] T nearest(int lz, int ly, int lx) const noexcept {
        return data[static_cast<size_t>(lz) * stride_z + static_cast<size_t>(ly) * stride_y + lx];
    }

    [[gnu::always_inline]] float trilinear(
        int lz, int ly, int lx,
        float fz, float fy, float fx) const noexcept
    {
        const size_t z0y0 = static_cast<size_t>(lz) * stride_z + static_cast<size_t>(ly) * stride_y;
        const size_t z0y1 = z0y0 + stride_y;
        const size_t z1y0 = z0y0 + stride_z;
        const size_t z1y1 = z1y0 + stride_y;

        const float c000 = data[z0y0 + lx];
        const float c001 = data[z0y0 + lx + 1];
        const float c010 = data[z0y1 + lx];
        const float c011 = data[z0y1 + lx + 1];
        const float c100 = data[z1y0 + lx];
        const float c101 = data[z1y0 + lx + 1];
        const float c110 = data[z1y1 + lx];
        const float c111 = data[z1y1 + lx + 1];

        const float c00 = c000 + fx * (c001 - c000);
        const float c01 = c010 + fx * (c011 - c010);
        const float c10 = c100 + fx * (c101 - c100);
        const float c11 = c110 + fx * (c111 - c110);

        const float cl0 = c00 + fy * (c01 - c00);
        const float cl1 = c10 + fy * (c11 - c10);

        return cl0 + fz * (cl1 - cl0);
    }
};


// ============================================================================
// Tile<T, N> — N×N output tile with SoA coordinate storage
// ============================================================================
//
// Holds up to N*N samples in structure-of-arrays layout for SIMD-friendly
// batch processing. Sequential SoA arrays let the compiler vectorize
// the FMA chains even when chunk reads are scattered.

template<typename T, int N = 32>
struct Tile {
    static constexpr int CAPACITY = N * N;

    // SoA coordinate arrays — 64-byte aligned for NEON/SVE
    alignas(64) float vx[CAPACITY];
    alignas(64) float vy[CAPACITY];
    alignas(64) float vz[CAPACITY];
    alignas(64) T results[CAPACITY];
    int count = 0;

    void clear() noexcept { count = 0; }

    [[gnu::always_inline]] void add(float x, float y, float z) noexcept {
        vx[count] = x;
        vy[count] = y;
        vz[count] = z;
        count++;
    }

    // Batch trilinear sample from a single chunk.
    // All coordinates must have their trilinear +1 neighbors within the chunk.
    template<int CN>
    void sampleTrilinear(const ChunkView<T, CN>& chunk, int czMask, int cyMask, int cxMask) noexcept {
        const int n = count;
        for (int i = 0; i < n; i++) {
            const int iz = static_cast<int>(vz[i]);
            const int iy = static_cast<int>(vy[i]);
            const int ix = static_cast<int>(vx[i]);
            const int lz = iz & czMask, ly = iy & cyMask, lx = ix & cxMask;
            const float fz = vz[i] - iz, fy = vy[i] - iy, fx = vx[i] - ix;
            results[i] = static_cast<T>(chunk.trilinear(lz, ly, lx, fz, fy, fx));
        }
    }

    // Batch nearest-neighbor sample
    template<int CN>
    void sampleNearest(const ChunkView<T, CN>& chunk, int czMask, int cyMask, int cxMask) noexcept {
        const int n = count;
        for (int i = 0; i < n; i++) {
            const int iz = static_cast<int>(vz[i] + 0.5f);
            const int iy = static_cast<int>(vy[i] + 0.5f);
            const int ix = static_cast<int>(vx[i] + 0.5f);
            results[i] = chunk.nearest(iz & czMask, iy & cyMask, ix & cxMask);
        }
    }
};


// ============================================================================
// Vec<T, N> — N-element SIMD-friendly vector
// ============================================================================
//
// 1D batch of samples, aligned for SIMD. Use for processing runs of
// consecutive pixels along a row.

template<typename T, int N = 32>
struct Vec {
    static constexpr int CAPACITY = N;

    alignas(64) float vx[N];
    alignas(64) float vy[N];
    alignas(64) float vz[N];
    alignas(64) T results[N];
    int count = 0;

    void clear() noexcept { count = 0; }

    [[gnu::always_inline]] void add(float x, float y, float z) noexcept {
        vx[count] = x;
        vy[count] = y;
        vz[count] = z;
        count++;
    }

    template<int CN>
    void sampleTrilinear(const ChunkView<T, CN>& chunk, int czMask, int cyMask, int cxMask) noexcept {
        const int n = count;
        for (int i = 0; i < n; i++) {
            const int iz = static_cast<int>(vz[i]);
            const int iy = static_cast<int>(vy[i]);
            const int ix = static_cast<int>(vx[i]);
            const int lz = iz & czMask, ly = iy & cyMask, lx = ix & cxMask;
            const float fz = vz[i] - iz, fy = vy[i] - iy, fx = vx[i] - ix;
            results[i] = static_cast<T>(chunk.trilinear(lz, ly, lx, fz, fy, fx));
        }
    }

    template<int CN>
    void sampleNearest(const ChunkView<T, CN>& chunk, int czMask, int cyMask, int cxMask) noexcept {
        const int n = count;
        for (int i = 0; i < n; i++) {
            const int iz = static_cast<int>(vz[i] + 0.5f);
            const int iy = static_cast<int>(vy[i] + 0.5f);
            const int ix = static_cast<int>(vx[i] + 0.5f);
            results[i] = chunk.nearest(iz & czMask, iy & cyMask, ix & cxMask);
        }
    }
};


// ============================================================================
// Dispatch helper — calls fn with a "view maker" lambda
// ============================================================================
//
// Dispatches once on chunk dimensions, then passes fn a maker that creates
// the correct ChunkView type from just a data pointer. This avoids per-run
// dispatch overhead — call once before the OMP region, use inside the loop.
//
// Usage:
//   dispatchChunkView<T>(cy, cx, [&](auto makeView) {
//       #pragma omp parallel
//       {
//           // ... inside loop:
//           auto cv = makeView(raw_ptr);
//           float v = cv.trilinear(lz, ly, lx, fz, fy, fx);
//       }
//   });

template<typename T, typename Fn>
[[gnu::always_inline]] inline void dispatchChunkView(int cy, int cx, Fn&& fn) noexcept
{
    if (cy == cx) {
        switch (cx) {
            case 32:
                fn([](const T* d) { return ChunkView<T, 32>{d}; });
                return;
            case 64:
                fn([](const T* d) { return ChunkView<T, 64>{d}; });
                return;
            case 128:
                fn([](const T* d) { return ChunkView<T, 128>{d}; });
                return;
        }
    }
    const size_t sz = static_cast<size_t>(cy) * cx;
    const size_t sy = static_cast<size_t>(cx);
    fn([sz, sy](const T* d) { return ChunkViewRT<T>{d, sz, sy}; });
}

} // namespace vc::simd
