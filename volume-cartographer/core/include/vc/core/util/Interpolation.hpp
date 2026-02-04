#pragma once

#include <xtensor/containers/xarray.hpp>
#include <opencv2/core.hpp>
#include <vc/core/util/ChunkCache.hpp>
#include <stdint.h>
#include <stdlib.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/matx.inl.hpp>
#include <cmath>
#include <numbers>
#include <array>
#include <algorithm>
#include <type_traits>

namespace z5 {
class Dataset;
}  // namespace z5
namespace cv {
template <typename _Tp> class Mat_;
}  // namespace cv
template <typename T> class ChunkCache;

namespace vc {

/**
 * @brief Lanczos kernel function
 *
 * L(x) = sinc(x) * sinc(x/a)  for |x| < a
 *      = 0                     otherwise
 *
 * where sinc(x) = sin(pi*x) / (pi*x) for x != 0, 1 for x = 0
 *
 * @param x Distance from center
 * @param a Lanczos parameter (window size), typically 2 or 3
 * @return Lanczos kernel weight
 */
[[gnu::always_inline]] inline float lanczosKernel(float x, int a = 3) noexcept {
    if (x == 0.0f) [[unlikely]] return 1.0f;
    if (std::abs(x) >= static_cast<float>(a)) [[unlikely]] return 0.0f;

    float pix = std::numbers::pi_v<float> * x;
    float pixa = pix / static_cast<float>(a);
    return (std::sin(pix) / pix) * (std::sin(pixa) / pixa);
}

/**
 * @brief Precomputed Lanczos-3 kernel weights for fast evaluation
 *
 * For Lanczos-3, the kernel spans [-3, 3) which means 6 samples.
 * Weights are precomputed for a given fractional offset.
 */
struct Lanczos3Weights {
    std::array<float, 6> weights;

    /**
     * @brief Compute weights for a fractional offset
     * @param frac Fractional part in [0, 1)
     */
    explicit Lanczos3Weights(float frac) {
        // Compute all weights - positions relative to center
        weights[0] = lanczosKernel(frac + 2.0f, 3);  // i=0: frac - (-2) = frac + 2
        weights[1] = lanczosKernel(frac + 1.0f, 3);  // i=1: frac - (-1) = frac + 1
        weights[2] = lanczosKernel(frac, 3);         // i=2: frac - 0 = frac
        weights[3] = lanczosKernel(frac - 1.0f, 3);  // i=3: frac - 1
        weights[4] = lanczosKernel(frac - 2.0f, 3);  // i=4: frac - 2
        weights[5] = lanczosKernel(frac - 3.0f, 3);  // i=5: frac - 3

        // Sum and normalize with reciprocal multiplication (faster than division)
        const float sum = weights[0] + weights[1] + weights[2] +
                          weights[3] + weights[4] + weights[5];
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            weights[0] *= inv_sum;
            weights[1] *= inv_sum;
            weights[2] *= inv_sum;
            weights[3] *= inv_sum;
            weights[4] *= inv_sum;
            weights[5] *= inv_sum;
        }
    }
};

/**
 * @brief Sample a 3D volume using Lanczos-3 interpolation
 *
 * Uses separable evaluation: apply 1D Lanczos in each dimension sequentially.
 * This is O(6*6*6) = O(216) for full 3D, but separable is O(6+6+6) = O(18).
 *
 * @tparam T Data type (uint8_t, uint16_t, or float)
 * @param data Raw pointer to 3D volume data (ZYX ordering)
 * @param sz, sy, sx Volume dimensions
 * @param vz, vy, vx Sample position (can be non-integer)
 * @return Interpolated value
 */
template<typename T>
float sampleLanczos3D(const T* __restrict__ data, int sz, int sy, int sx,
                      float vz, float vy, float vx) {
    // Integer and fractional parts
    const int iz = static_cast<int>(std::floor(vz));
    const int iy = static_cast<int>(std::floor(vy));
    const int ix = static_cast<int>(std::floor(vx));

    const float fz = vz - static_cast<float>(iz);
    const float fy = vy - static_cast<float>(iy);
    const float fx = vx - static_cast<float>(ix);

    // Precompute weights
    const Lanczos3Weights wz(fz), wy(fy), wx(fx);

    // Pre-compute strides for better optimization
    const size_t stride_z = static_cast<size_t>(sy) * sx;
    const size_t stride_y = static_cast<size_t>(sx);

    // Pre-compute valid ranges to eliminate branches in inner loops
    const int z_start = std::max(0, -2 - iz);
    const int z_end = std::min(6, sz - iz + 2);
    const int y_start = std::max(0, -2 - iy);
    const int y_end = std::min(6, sy - iy + 2);
    const int x_start = std::max(0, -2 - ix);
    const int x_end = std::min(6, sx - ix + 2);

    float result = 0.0f;

    for (int dzi = z_start; dzi < z_end; ++dzi) {
        const int z = iz + dzi - 2;
        const float wZ = wz.weights[dzi];
        const size_t z_off = z * stride_z;

        for (int dyi = y_start; dyi < y_end; ++dyi) {
            const int y = iy + dyi - 2;
            const float wZY = wZ * wy.weights[dyi];
            const size_t zy_off = z_off + y * stride_y;

            // Innermost loop - simple enough for auto-vectorization
            float row_sum = 0.0f;
            #pragma omp simd reduction(+:row_sum)
            for (int dxi = x_start; dxi < x_end; ++dxi) {
                const int x = ix + dxi - 2;
                const float w = wZY * wx.weights[dxi];
                row_sum += w * static_cast<float>(data[zy_off + x]);
            }
            result += row_sum;
        }
    }

    return result;
}

/**
 * @brief Sample a 3D volume and compute gradient using Lanczos-3 with Gaussian derivatives
 *
 * Computes both the interpolated value and the gradient at the sample position.
 * Uses derivative of Gaussian kernels: G'(x, sigma) = -x / sigma^2 * G(x, sigma)
 *
 * @tparam T Data type
 * @param data Raw pointer to 3D volume data
 * @param sz, sy, sx Volume dimensions
 * @param vz, vy, vx Sample position
 * @param sigma Gaussian sigma for gradient computation
 * @param[out] gradient Output gradient (dx, dy, dz)
 * @return Interpolated value
 */
template<typename T>
float sampleLanczos3DWithGradient(const T* __restrict__ data, int sz, int sy, int sx,
                                   float vz, float vy, float vx,
                                   float sigma, cv::Vec3f& gradient) {
    // Integer and fractional parts
    int iz = static_cast<int>(std::floor(vz));
    int iy = static_cast<int>(std::floor(vy));
    int ix = static_cast<int>(std::floor(vx));

    float fz = vz - static_cast<float>(iz);
    float fy = vy - static_cast<float>(iy);
    float fx = vx - static_cast<float>(ix);

    // Precompute Lanczos weights
    Lanczos3Weights wz(fz), wy(fy), wx(fx);

    // Precompute Gaussian derivative weights
    float sigma2 = sigma * sigma;
    float norm = 1.0f / (sigma * std::sqrt(2.0f * std::numbers::pi_v<float>));

    auto gaussianWeight = [&](float x) -> float {
        return norm * std::exp(-x * x / (2.0f * sigma2));
    };

    auto gaussianDerivWeight = [&](float x) -> float {
        return -x / sigma2 * gaussianWeight(x);
    };

    float value = 0.0f;
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;

    for (int dz = -2; dz <= 3; ++dz) {
        int z = iz + dz;
        if (z < 0 || z >= sz) [[unlikely]] continue;
        float wZ = wz.weights[dz + 2];
        float dZ = static_cast<float>(dz) - fz;
        float gZ = gaussianWeight(dZ);
        float dgZ = gaussianDerivWeight(dZ);

        for (int dy = -2; dy <= 3; ++dy) {
            int y = iy + dy;
            if (y < 0 || y >= sy) [[unlikely]] continue;
            float wZY = wZ * wy.weights[dy + 2];
            float dY = static_cast<float>(dy) - fy;
            float gY = gaussianWeight(dY);
            float dgY = gaussianDerivWeight(dY);
            float gZY = gZ * gY;
            float dgZY_dz = dgZ * gY;
            float dgZY_dy = gZ * dgY;

            for (int dx = -2; dx <= 3; ++dx) {
                int x = ix + dx;
                if (x < 0 || x >= sx) [[unlikely]] continue;

                float dX = static_cast<float>(dx) - fx;
                float gX = gaussianWeight(dX);
                float dgX = gaussianDerivWeight(dX);

                float w = wZY * wx.weights[dx + 2];
                float v = static_cast<float>(data[z * sy * sx + y * sx + x]);

                value += w * v;

                // Gradient contributions
                gx += v * gZY * dgX;
                gy += v * dgZY_dy * gX;
                gz += v * dgZY_dz * gX;
            }
        }
    }

    gradient = cv::Vec3f(gx, gy, gz);
    return value;
}

/**
 * @brief Parameters for Lanczos resampling
 */
struct LanczosResampleParams {
    float scaleX = 1.0f;  ///< Scale factor in X
    float scaleY = 1.0f;  ///< Scale factor in Y
    float scaleZ = 1.0f;  ///< Scale factor in Z
};

/**
 * @brief Resample a volume chunk using Lanczos-3 interpolation
 *
 * @tparam T Input/output data type
 * @param input Input chunk data
 * @param inSz, inSy, inSx Input dimensions
 * @param output Output chunk data (must be pre-allocated)
 * @param outSz, outSy, outSx Output dimensions
 * @param params Resampling parameters
 */
template<typename T>
void resampleChunkLanczos3D(const T* __restrict__ input, int inSz, int inSy, int inSx,
                             T* __restrict__ output, int outSz, int outSy, int outSx,
                             const LanczosResampleParams& params) {
    // Pre-compute inverse scales for multiplication instead of division
    const float invScaleZ = 1.0f / params.scaleZ;
    const float invScaleY = 1.0f / params.scaleY;
    const float invScaleX = 1.0f / params.scaleX;

    // Pre-compute output strides
    const size_t out_stride_z = static_cast<size_t>(outSy) * outSx;
    const size_t out_stride_y = static_cast<size_t>(outSx);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int oz = 0; oz < outSz; ++oz) {
        for (int oy = 0; oy < outSy; ++oy) {
            // Pre-compute z and y input coords (constant for inner loop)
            const float iz = static_cast<float>(oz) * invScaleZ;
            const float iy = static_cast<float>(oy) * invScaleY;
            const size_t out_base = oz * out_stride_z + oy * out_stride_y;

            for (int ox = 0; ox < outSx; ++ox) {
                const float ix = static_cast<float>(ox) * invScaleX;

                float v = sampleLanczos3D(input, inSz, inSy, inSx, iz, iy, ix);

                // Clamp and store
                if constexpr (std::is_same_v<T, uint8_t>) {
                    v = std::max(0.0f, std::min(255.0f, v));
                    output[out_base + ox] = static_cast<uint8_t>(v + 0.5f);
                } else if constexpr (std::is_same_v<T, uint16_t>) {
                    v = std::max(0.0f, std::min(65535.0f, v));
                    output[out_base + ox] = static_cast<uint16_t>(v + 0.5f);
                } else {
                    output[oz * outSy * outSx + oy * outSx + ox] = static_cast<T>(v);
                }
            }
        }
    }
}

/**
 * @brief Read interpolated 3D data using Lanczos-3 from a Zarr dataset
 *
 * @tparam T Data type (uint8_t or uint16_t)
 * @param[out] out Output matrix
 * @param ds Zarr dataset
 * @param coords Coordinates to sample (XYZ in Vec3f)
 * @param cache Chunk cache
 */
template<typename T>
void readInterpolated3DLanczos(cv::Mat_<T>& out, z5::Dataset* ds,
                                const cv::Mat_<cv::Vec3f>& coords,
                                ChunkCache<T>& cache);

// Explicit instantiations declared
extern template void readInterpolated3DLanczos<uint8_t>(cv::Mat_<uint8_t>&, z5::Dataset*,
                                                         const cv::Mat_<cv::Vec3f>&,
                                                         ChunkCache<uint8_t>&);
extern template void readInterpolated3DLanczos<uint16_t>(cv::Mat_<uint16_t>&, z5::Dataset*,
                                                          const cv::Mat_<cv::Vec3f>&,
                                                          ChunkCache<uint16_t>&);

}  // namespace vc
