#pragma once

#include <algorithm>
#include <cstddef>

namespace vc::simd {

// Detect SIMD register width at compile time (bytes)
inline constexpr int kSimdRegisterBytes =
#if defined(__AVX512F__)
    64
#elif defined(__AVX2__) || defined(__AVX__)
    32
#elif defined(__SSE2__) || defined(__ARM_NEON)
    16
#else
    16  // conservative fallback
#endif
    ;

// Minimum tile size per axis: must fill at least one SIMD register
// for the given element type, but never smaller than a sane floor.
template <typename T>
inline constexpr int kMinTileSize = [] {
    constexpr int hw_floor = kSimdRegisterBytes / static_cast<int>(sizeof(T));
    constexpr int sane_floor = 8;
    int result = hw_floor > sane_floor ? hw_floor : sane_floor;
    // Round up to power of 2 (should already be, but safety)
    int p = 1;
    while (p < result)
        p <<= 1;
    return p;
}();

// Supported tile sizes (power-of-2, descending)
inline constexpr int kTileSizes[] = {128, 64, 32, 16, 8};
inline constexpr int kNumTileSizes = 5;

}  // namespace vc::simd
