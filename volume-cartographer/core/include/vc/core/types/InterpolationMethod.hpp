#pragma once

/**
 * @brief Interpolation methods for volume sampling
 *
 * Extracted to a separate header to reduce compile-time dependencies.
 * Headers that only need the enum can include this lightweight header
 * instead of the full Slicing.hpp which pulls in xtensor/z5.
 */
enum class InterpolationMethod {
    Nearest = 0,    ///< Nearest neighbor (fastest, blocky)
    Trilinear = 1,  ///< Trilinear interpolation (default, good balance)
    Tricubic = 2,   ///< Tricubic interpolation (smooth, slower)
    Lanczos = 3     ///< Lanczos-3 interpolation (best quality, slowest)
};

/**
 * @brief Convert interpolation method enum to string
 */
inline const char* interpolationMethodName(InterpolationMethod method) {
    switch (method) {
        case InterpolationMethod::Nearest: return "Nearest Neighbor";
        case InterpolationMethod::Trilinear: return "Trilinear";
        case InterpolationMethod::Tricubic: return "Tricubic";
        case InterpolationMethod::Lanczos: return "Lanczos-3";
        default: return "Unknown";
    }
}
