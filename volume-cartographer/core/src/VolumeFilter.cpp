#include "vc/core/util/VolumeFilter.hpp"

#include <opencv2/imgproc.hpp>
#include <_stdlib.h>
#include <opencv2/core/hal/interface.h>
#include <stdint.h>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/matx.inl.hpp>
#include <opencv2/core/types.hpp>
#include <xsimd/memory/xsimd_aligned_allocator.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xstorage.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/utils/xutils.hpp>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <atomic>
#include <tuple>
#include <type_traits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vc {

// ============================================================================
// Helper functions
// ============================================================================

static constexpr float PI = 3.14159265358979323846f;
static constexpr float EPS = 1e-10f;

// Clamp helper
template<typename T>
static inline constexpr T clamp(T v, T lo, T hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Get max value for type
template<typename T>
static constexpr float maxTypeValue() noexcept {
    if constexpr (std::is_same_v<T, uint8_t>) return 255.0f;
    else if constexpr (std::is_same_v<T, uint16_t>) return 65535.0f;
    else return 1.0f;
}

// 1D Gaussian kernel
static std::vector<float> gaussianKernel1D(float sigma) {
    if (sigma <= 0.0f) [[unlikely]] return {1.0f};
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    int size = 2 * radius + 1;
    std::vector<float> kernel(size);
    float sum = 0.0f;
    float denom = 2.0f * sigma * sigma;
    for (int i = 0; i < size; ++i) {
        float x = static_cast<float>(i - radius);
        kernel[i] = std::exp(-x * x / denom);
        sum += kernel[i];
    }
    for (int i = 0; i < size; ++i) kernel[i] /= sum;
    return kernel;
}

// 1D Gaussian derivative kernel
static std::vector<float> gaussianDerivKernel1D(float sigma) {
    if (sigma <= 0.0f) [[unlikely]] return {0.0f};
    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    int size = 2 * radius + 1;
    std::vector<float> kernel(size);
    float sigma2 = sigma * sigma;
    float norm = 1.0f / (sigma * std::sqrt(2.0f * PI));
    for (int i = 0; i < size; ++i) {
        float x = static_cast<float>(i - radius);
        float g = norm * std::exp(-x * x / (2.0f * sigma2));
        kernel[i] = -x / sigma2 * g;  // G'(x) = -x/sigma^2 * G(x)
    }
    return kernel;
}

// ============================================================================
// Non-Local Means Denoising
// ============================================================================

template<typename T>
void nlmDenoise3D(const xt::xarray<T>& input, xt::xarray<T>& output, const NLMParams& params) {
    const int sz = static_cast<int>(input.shape()[0]);
    const int sy = static_cast<int>(input.shape()[1]);
    const int sx = static_cast<int>(input.shape()[2]);

    output = xt::xarray<T>::from_shape(input.shape());

    const int sr = params.searchRadius;
    const int pr = params.patchRadius;
    const float h2 = params.h * params.h;
    const int patchSize = (2 * pr + 1);
    const int patchVol = patchSize * patchSize * patchSize;

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int z = 0; z < sz; ++z) {
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                float weightSum = 0.0f;
                float valueSum = 0.0f;

                // Search window bounds
                int z0 = std::max(0, z - sr);
                int z1 = std::min(sz - 1, z + sr);
                int y0 = std::max(0, y - sr);
                int y1 = std::min(sy - 1, y + sr);
                int x0 = std::max(0, x - sr);
                int x1 = std::min(sx - 1, x + sr);

                // Iterate over search window
                for (int sz_ = z0; sz_ <= z1; ++sz_) {
                    for (int sy_ = y0; sy_ <= y1; ++sy_) {
                        for (int sx_ = x0; sx_ <= x1; ++sx_) {
                            // Compute patch distance
                            float dist = 0.0f;
                            int validCount = 0;

                            for (int pz = -pr; pz <= pr; ++pz) {
                                int zp1 = z + pz;
                                int zp2 = sz_ + pz;
                                if (zp1 < 0 || zp1 >= sz || zp2 < 0 || zp2 >= sz) continue;

                                for (int py = -pr; py <= pr; ++py) {
                                    int yp1 = y + py;
                                    int yp2 = sy_ + py;
                                    if (yp1 < 0 || yp1 >= sy || yp2 < 0 || yp2 >= sy) continue;

                                    for (int px = -pr; px <= pr; ++px) {
                                        int xp1 = x + px;
                                        int xp2 = sx_ + px;
                                        if (xp1 < 0 || xp1 >= sx || xp2 < 0 || xp2 >= sx) continue;

                                        float v1 = static_cast<float>(input(zp1, yp1, xp1));
                                        float v2 = static_cast<float>(input(zp2, yp2, xp2));
                                        float diff = v1 - v2;
                                        dist += diff * diff;
                                        validCount++;
                                    }
                                }
                            }

                            if (validCount > 0) {
                                dist /= static_cast<float>(validCount);
                            }

                            // Weight based on patch similarity
                            float weight = std::exp(-dist / h2);
                            weightSum += weight;
                            valueSum += weight * static_cast<float>(input(sz_, sy_, sx_));
                        }
                    }
                }

                // Compute filtered value
                float result = (weightSum > EPS) ? valueSum / weightSum : static_cast<float>(input(z, y, x));

                // Clamp and store
                constexpr float maxVal = maxTypeValue<T>();
                result = clamp(result, 0.0f, maxVal);
                if constexpr (std::is_floating_point_v<T>) {
                    output(z, y, x) = static_cast<T>(result);
                } else {
                    output(z, y, x) = static_cast<T>(result + 0.5f);
                }
            }
        }
    }
}

// ============================================================================
// Vo Ring Artifact Removal (SOTA algorithm from Vo et al. Optics Express 2018)
// ============================================================================

// Helper: Create 1D mean filter for large rings
static void meanFilter1D(const std::vector<float>& in, std::vector<float>& out, int size) {
    int n = static_cast<int>(in.size());
    int half = size / 2;
    out.resize(n);

    float sum = 0.0f;
    int count = 0;

    // Initialize window
    for (int i = 0; i < std::min(half + 1, n); ++i) {
        sum += in[i];
        count++;
    }

    for (int i = 0; i < n; ++i) {
        // Add right element
        int right = i + half;
        if (right < n && right >= half + 1) {
            sum += in[right];
            count++;
        }
        // Remove left element
        int left = i - half - 1;
        if (left >= 0) {
            sum -= in[left];
            count--;
        }
        out[i] = sum / static_cast<float>(count);
    }
}

// Helper: 1D median filter
static void medianFilter1D(const std::vector<float>& in, std::vector<float>& out, int size) {
    int n = static_cast<int>(in.size());
    int half = size / 2;
    out.resize(n);

    std::vector<float> window;
    window.reserve(size);

    for (int i = 0; i < n; ++i) {
        window.clear();
        for (int j = std::max(0, i - half); j <= std::min(n - 1, i + half); ++j) {
            window.push_back(in[j]);
        }
        std::sort(window.begin(), window.end());
        out[i] = window[window.size() / 2];
    }
}

// Vo algorithm: sorting-based removal for large rings
template<typename T>
static void voRemoveLargeRings(cv::Mat& sinogram, const VoRingParams& params) {
    int H = sinogram.rows;  // angles
    int W = sinogram.cols;  // detector elements

    // Step 1: Sort each column independently, keeping track of indices
    cv::Mat sorted(H, W, CV_32FC1);
    std::vector<std::vector<int>> sortIndices(W, std::vector<int>(H));

    for (int x = 0; x < W; ++x) {
        std::vector<std::pair<float, int>> colData(H);
        for (int y = 0; y < H; ++y) {
            colData[y] = {sinogram.at<float>(y, x), y};
        }
        std::sort(colData.begin(), colData.end());
        for (int y = 0; y < H; ++y) {
            sorted.at<float>(y, x) = colData[y].first;
            sortIndices[x][y] = colData[y].second;
        }
    }

    // Step 2: Apply mean filter along detector direction (horizontal)
    cv::Mat smoothed(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        std::vector<float> row(W), filtered(W);
        for (int x = 0; x < W; ++x) {
            row[x] = sorted.at<float>(y, x);
        }
        meanFilter1D(row, filtered, params.la);
        for (int x = 0; x < W; ++x) {
            smoothed.at<float>(y, x) = filtered[x];
        }
    }

    // Step 3: Compute correction (sorted - smoothed)
    cv::Mat correction = sorted - smoothed;

    // Step 4: Apply threshold based on SNR
    float sigma = 0.0f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            sigma += correction.at<float>(y, x) * correction.at<float>(y, x);
        }
    }
    sigma = std::sqrt(sigma / static_cast<float>(H * W));
    float threshold = static_cast<float>(params.snr) * sigma;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float& c = correction.at<float>(y, x);
            if (std::abs(c) < threshold) c = 0.0f;
        }
    }

    // Step 5: Unsort the correction and apply
    for (int x = 0; x < W; ++x) {
        std::vector<float> unsortedCorr(H);
        for (int y = 0; y < H; ++y) {
            int origIdx = sortIndices[x][y];
            unsortedCorr[origIdx] = correction.at<float>(y, x);
        }
        for (int y = 0; y < H; ++y) {
            sinogram.at<float>(y, x) -= unsortedCorr[y];
        }
    }
}

// Vo algorithm: FFT-based removal for small/medium rings
template<typename T>
static void voRemoveSmallRings(cv::Mat& sinogram, const VoRingParams& params) {
    int H = sinogram.rows;
    int W = sinogram.cols;

    // Apply median filter along angle direction (vertical)
    cv::Mat medianFiltered(H, W, CV_32FC1);
    for (int x = 0; x < W; ++x) {
        std::vector<float> col(H), filtered(H);
        for (int y = 0; y < H; ++y) {
            col[y] = sinogram.at<float>(y, x);
        }
        medianFilter1D(col, filtered, params.sm);
        for (int y = 0; y < H; ++y) {
            medianFiltered.at<float>(y, x) = filtered[y];
        }
    }

    // Ring pattern is the difference
    cv::Mat ringPattern = sinogram - medianFiltered;

    // Mean of ring pattern along angles -> ring profile
    std::vector<float> ringProfile(W, 0.0f);
    for (int x = 0; x < W; ++x) {
        float sum = 0.0f;
        for (int y = 0; y < H; ++y) {
            sum += ringPattern.at<float>(y, x);
        }
        ringProfile[x] = sum / static_cast<float>(H);
    }

    // Threshold and apply correction
    float sigma = 0.0f;
    for (int x = 0; x < W; ++x) {
        sigma += ringProfile[x] * ringProfile[x];
    }
    sigma = std::sqrt(sigma / static_cast<float>(W));
    float threshold = static_cast<float>(params.snr) * sigma;

    for (int x = 0; x < W; ++x) {
        if (std::abs(ringProfile[x]) < threshold) ringProfile[x] = 0.0f;
    }

    // Subtract ring profile from all rows
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            sinogram.at<float>(y, x) -= ringProfile[x];
        }
    }
}

template<typename T>
void voRingRemoveSlice(cv::Mat_<T>& slice, const VoRingParams& params) {
    const int H = slice.rows;
    const int W = slice.cols;

    // Determine center
    float cx = (params.centerX >= 0) ? params.centerX : static_cast<float>(W) / 2.0f;
    float cy = (params.centerY >= 0) ? params.centerY : static_cast<float>(H) / 2.0f;

    // Convert slice to polar (sinogram-like representation)
    float maxRadius = std::sqrt(cx * cx + cy * cy);
    int numAngles = static_cast<int>(2.0f * PI * maxRadius);
    int numRadii = static_cast<int>(maxRadius) + 1;

    cv::Mat polarImg(numAngles, numRadii, CV_32FC1);

    // Cartesian to polar
    for (int a = 0; a < numAngles; ++a) {
        float angle = static_cast<float>(a) * 2.0f * PI / static_cast<float>(numAngles);
        for (int r = 0; r < numRadii; ++r) {
            float radius = static_cast<float>(r);
            float x = cx + radius * std::cos(angle);
            float y = cy + radius * std::sin(angle);

            int x0 = static_cast<int>(x);
            int y0 = static_cast<int>(y);
            float fx = x - static_cast<float>(x0);
            float fy = y - static_cast<float>(y0);

            float val = 0.0f;
            if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1) {
                val = (1 - fx) * (1 - fy) * static_cast<float>(slice(y0, x0)) +
                      fx * (1 - fy) * static_cast<float>(slice(y0, x0 + 1)) +
                      (1 - fx) * fy * static_cast<float>(slice(y0 + 1, x0)) +
                      fx * fy * static_cast<float>(slice(y0 + 1, x0 + 1));
            }
            polarImg.at<float>(a, r) = val;
        }
    }

    // Apply Vo ring removal to the sinogram
    voRemoveLargeRings<T>(polarImg, params);
    voRemoveSmallRings<T>(polarImg, params);

    // Convert back to Cartesian
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float radius = std::sqrt(dx * dx + dy * dy);
            float angle = std::atan2(dy, dx);
            if (angle < 0) angle += 2.0f * PI;

            float aF = angle * static_cast<float>(numAngles) / (2.0f * PI);
            int a0 = static_cast<int>(aF) % numAngles;
            int a1 = (a0 + 1) % numAngles;
            float fa = aF - std::floor(aF);

            int r0 = static_cast<int>(radius);
            int r1 = std::min(r0 + 1, numRadii - 1);
            float fr = radius - std::floor(radius);

            if (r0 < numRadii) {
                float val = (1 - fa) * (1 - fr) * polarImg.at<float>(a0, r0) +
                            fa * (1 - fr) * polarImg.at<float>(a1, r0) +
                            (1 - fa) * fr * polarImg.at<float>(a0, r1) +
                            fa * fr * polarImg.at<float>(a1, r1);

                val = clamp(val, 0.0f, maxTypeValue<T>());
                if constexpr (std::is_floating_point_v<T>) {
                    slice(y, x) = static_cast<T>(val);
                } else {
                    slice(y, x) = static_cast<T>(val + 0.5f);
                }
            }
        }
    }
}

template<typename T>
void voRingRemoveVolume(xt::xarray<T>& volume, const VoRingParams& params,
                         std::function<void(float)> progressCallback) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    std::atomic<int> completed{0};

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < sz; ++z) {
        cv::Mat_<T> slice(sy, sx);
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                slice(y, x) = volume(z, y, x);
            }
        }

        voRingRemoveSlice(slice, params);

        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                volume(z, y, x) = slice(y, x);
            }
        }

        if (progressCallback) {
            int done = ++completed;
            if (done % 10 == 0 || done == sz) {
                progressCallback(static_cast<float>(done) / static_cast<float>(sz));
            }
        }
    }
}

// ============================================================================
// FFT Stripe Removal
// ============================================================================

template<typename T>
void stripeRemoveSlice(cv::Mat_<T>& slice, const StripeRemovalParams& params) {
    const int H = slice.rows;
    const int W = slice.cols;

    // Convert to float for FFT
    cv::Mat floatImg(H, W, CV_32FC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            floatImg.at<float>(y, x) = static_cast<float>(slice(y, x));
        }
    }

    // Optimal DFT size
    int M = cv::getOptimalDFTSize(H);
    int N = cv::getOptimalDFTSize(W);

    // Pad image
    cv::Mat padded;
    cv::copyMakeBorder(floatImg, padded, 0, M - H, 0, N - W, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Perform 2D FFT
    cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32FC1)};
    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);

    // Create Gaussian damping mask for horizontal frequencies (vertical stripes)
    // Vertical stripes appear as horizontal lines in FFT (around center row)
    cv::Mat mask(M, N, CV_32FC1, cv::Scalar(1.0f));

    int centerY = M / 2;
    float sigma2 = params.sigma * params.sigma;

    // Shift to have DC at center
    cv::Mat shifted;
    int cx = N / 2;
    int cy = M / 2;

    // Dampen horizontal frequencies (vertical stripes in spatial domain)
    // These appear as bright lines along x-axis in frequency domain
    for (int y = 0; y < M; ++y) {
        for (int x = 0; x < N; ++x) {
            // Distance from horizontal center line
            int dy = std::abs(y - cy);
            if (dy < static_cast<int>(3.0f * params.sigma) && x != cx) {
                // Apply Gaussian damping
                float damp = 1.0f - std::exp(-static_cast<float>(dy * dy) / (2.0f * sigma2));
                mask.at<float>(y, x) = damp;
            }
        }
    }

    // Apply mask to FFT (after shifting DC to center)
    cv::Mat q0(complex, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(complex, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(complex, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(complex, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // Apply mask
    cv::Mat maskPlanes[] = {mask, mask};
    cv::Mat maskComplex;
    cv::merge(maskPlanes, 2, maskComplex);
    cv::mulSpectrums(complex, maskComplex, complex, 0);

    // Shift back
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // Inverse FFT
    cv::idft(complex, complex, cv::DFT_SCALE);
    cv::split(complex, planes);

    // Copy back to output
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float val = planes[0].at<float>(y, x);
            val = clamp(val, 0.0f, maxTypeValue<T>());
            if constexpr (std::is_floating_point_v<T>) {
                slice(y, x) = static_cast<T>(val);
            } else {
                slice(y, x) = static_cast<T>(val + 0.5f);
            }
        }
    }
}

template<typename T>
void stripeRemoveVolume(xt::xarray<T>& volume, const StripeRemovalParams& params,
                         std::function<void(float)> progressCallback) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    std::atomic<int> completed{0};

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < sz; ++z) {
        cv::Mat_<T> slice(sy, sx);
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                slice(y, x) = volume(z, y, x);
            }
        }

        stripeRemoveSlice(slice, params);

        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                volume(z, y, x) = slice(y, x);
            }
        }

        if (progressCallback) {
            int done = ++completed;
            if (done % 10 == 0 || done == sz) {
                progressCallback(static_cast<float>(done) / static_cast<float>(sz));
            }
        }
    }
}

// ============================================================================
// 3D CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================

template<typename T>
void clahe3D(xt::xarray<T>& volume, const CLAHE3DParams& params,
              std::function<void(float)> progressCallback) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    const int tileSize = params.tileSize;
    const int numBins = params.numBins;
    const float clipLimit = params.clipLimit;

    // Calculate number of tiles
    int tilesZ = (sz + tileSize - 1) / tileSize;
    int tilesY = (sy + tileSize - 1) / tileSize;
    int tilesX = (sx + tileSize - 1) / tileSize;

    // Find min/max for scaling
    T minVal = volume(0, 0, 0);
    T maxVal = minVal;
    const T* __restrict__ volData = volume.data();
    const size_t volSize = volume.size();
#pragma omp parallel for simd reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < volSize; ++i) {
        minVal = std::min(minVal, volData[i]);
        maxVal = std::max(maxVal, volData[i]);
    }
    float range = static_cast<float>(maxVal - minVal);
    if (range < EPS) [[unlikely]] return;

    // Compute lookup tables for each tile
    std::vector<std::vector<float>> tileLUTs(tilesZ * tilesY * tilesX, std::vector<float>(numBins));

    auto tileIndex = [&](int tz, int ty, int tx) {
        return tz * tilesY * tilesX + ty * tilesX + tx;
    };

    if (progressCallback) progressCallback(0.0f);

    // First pass: compute histogram and LUT for each tile
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int tz = 0; tz < tilesZ; ++tz) {
        for (int ty = 0; ty < tilesY; ++ty) {
            for (int tx = 0; tx < tilesX; ++tx) {
                int z0 = tz * tileSize;
                int z1 = std::min(z0 + tileSize, sz);
                int y0 = ty * tileSize;
                int y1 = std::min(y0 + tileSize, sy);
                int x0 = tx * tileSize;
                int x1 = std::min(x0 + tileSize, sx);

                int tileVoxels = (z1 - z0) * (y1 - y0) * (x1 - x0);

                // Build histogram
                std::vector<int> hist(numBins, 0);
                for (int z = z0; z < z1; ++z) {
                    for (int y = y0; y < y1; ++y) {
                        for (int x = x0; x < x1; ++x) {
                            float normalized = (static_cast<float>(volume(z, y, x)) - static_cast<float>(minVal)) / range;
                            int bin = clamp(static_cast<int>(normalized * (numBins - 1)), 0, numBins - 1);
                            hist[bin]++;
                        }
                    }
                }

                // Clip histogram
                int clipThreshold = static_cast<int>(clipLimit * tileVoxels / numBins);
                int excess = 0;
                for (int i = 0; i < numBins; ++i) {
                    if (hist[i] > clipThreshold) {
                        excess += hist[i] - clipThreshold;
                        hist[i] = clipThreshold;
                    }
                }

                // Redistribute excess
                int redistPerBin = excess / numBins;
                int redistRemainder = excess % numBins;
                for (int i = 0; i < numBins; ++i) {
                    hist[i] += redistPerBin;
                    if (i < redistRemainder) hist[i]++;
                }

                // Build cumulative histogram (LUT)
                auto& lut = tileLUTs[tileIndex(tz, ty, tx)];
                int sum = 0;
                for (int i = 0; i < numBins; ++i) {
                    sum += hist[i];
                    lut[i] = static_cast<float>(sum) / static_cast<float>(tileVoxels);
                }
            }
        }
    }

    if (progressCallback) progressCallback(0.3f);

    // Second pass: apply with trilinear interpolation
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < sz; ++z) {
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                // Find tile coordinates
                float tz = static_cast<float>(z) / tileSize - 0.5f;
                float ty = static_cast<float>(y) / tileSize - 0.5f;
                float tx = static_cast<float>(x) / tileSize - 0.5f;

                int tz0 = clamp(static_cast<int>(std::floor(tz)), 0, tilesZ - 1);
                int ty0 = clamp(static_cast<int>(std::floor(ty)), 0, tilesY - 1);
                int tx0 = clamp(static_cast<int>(std::floor(tx)), 0, tilesX - 1);
                int tz1 = std::min(tz0 + 1, tilesZ - 1);
                int ty1 = std::min(ty0 + 1, tilesY - 1);
                int tx1 = std::min(tx0 + 1, tilesX - 1);

                float fz = tz - std::floor(tz);
                float fy = ty - std::floor(ty);
                float fx = tx - std::floor(tx);

                // Get bin for current voxel
                float normalized = (static_cast<float>(volume(z, y, x)) - static_cast<float>(minVal)) / range;
                int bin = clamp(static_cast<int>(normalized * (numBins - 1)), 0, numBins - 1);

                // Trilinear interpolation of LUT values
                float v000 = tileLUTs[tileIndex(tz0, ty0, tx0)][bin];
                float v001 = tileLUTs[tileIndex(tz0, ty0, tx1)][bin];
                float v010 = tileLUTs[tileIndex(tz0, ty1, tx0)][bin];
                float v011 = tileLUTs[tileIndex(tz0, ty1, tx1)][bin];
                float v100 = tileLUTs[tileIndex(tz1, ty0, tx0)][bin];
                float v101 = tileLUTs[tileIndex(tz1, ty0, tx1)][bin];
                float v110 = tileLUTs[tileIndex(tz1, ty1, tx0)][bin];
                float v111 = tileLUTs[tileIndex(tz1, ty1, tx1)][bin];

                float v00 = v000 * (1 - fx) + v001 * fx;
                float v01 = v010 * (1 - fx) + v011 * fx;
                float v10 = v100 * (1 - fx) + v101 * fx;
                float v11 = v110 * (1 - fx) + v111 * fx;

                float v0 = v00 * (1 - fy) + v01 * fy;
                float v1 = v10 * (1 - fy) + v11 * fy;

                float result = v0 * (1 - fz) + v1 * fz;

                // Scale back to original range
                result = result * range + static_cast<float>(minVal);
                result = clamp(result, 0.0f, maxTypeValue<T>());

                if constexpr (std::is_floating_point_v<T>) {
                    volume(z, y, x) = static_cast<T>(result);
                } else {
                    volume(z, y, x) = static_cast<T>(result + 0.5f);
                }
            }
        }
    }

    if (progressCallback) progressCallback(1.0f);
}

// ============================================================================
// BM3D Denoising (Block-Matching 3D - simplified 2D slice version)
// ============================================================================

// Helper: DCT-II transform for 1D
static void dct1D(std::vector<float>& data) {
    int n = static_cast<int>(data.size());
    std::vector<float> result(n);
    for (int k = 0; k < n; ++k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += data[i] * std::cos(PI * (2 * i + 1) * k / (2.0f * n));
        }
        result[k] = sum * (k == 0 ? std::sqrt(1.0f / n) : std::sqrt(2.0f / n));
    }
    data = std::move(result);
}

// Helper: Inverse DCT-II
static void idct1D(std::vector<float>& data) {
    int n = static_cast<int>(data.size());
    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        float sum = data[0] * std::sqrt(1.0f / n);
        for (int k = 1; k < n; ++k) {
            sum += data[k] * std::sqrt(2.0f / n) * std::cos(PI * (2 * i + 1) * k / (2.0f * n));
        }
        result[i] = sum;
    }
    data = std::move(result);
}

template<typename T>
void bm3dSlice(cv::Mat_<T>& slice, const BM3DParams& params) {
    const int H = slice.rows;
    const int W = slice.cols;
    const int blockSize = params.blockSize;
    const int searchRadius = params.searchRadius;
    const int maxMatches = params.maxMatches;
    const float hardThresh = params.hardThreshold * params.sigma;

    // Working arrays
    cv::Mat floatImg(H, W, CV_32FC1);
    cv::Mat estimate(H, W, CV_32FC1, cv::Scalar(0));
    cv::Mat weights(H, W, CV_32FC1, cv::Scalar(0));

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            floatImg.at<float>(y, x) = static_cast<float>(slice(y, x));
        }
    }

    // Step size for processing (skip every few pixels for speed)
    int step = std::max(1, blockSize / 2);

    // Process blocks
    for (int refY = 0; refY <= H - blockSize; refY += step) {
        for (int refX = 0; refX <= W - blockSize; refX += step) {
            // Extract reference block
            std::vector<float> refBlock(blockSize * blockSize);
            for (int by = 0; by < blockSize; ++by) {
                for (int bx = 0; bx < blockSize; ++bx) {
                    refBlock[by * blockSize + bx] = floatImg.at<float>(refY + by, refX + bx);
                }
            }

            // Find similar blocks in search window
            std::vector<std::tuple<float, int, int>> matches;  // (distance, y, x)

            int y0 = std::max(0, refY - searchRadius);
            int y1 = std::min(H - blockSize, refY + searchRadius);
            int x0 = std::max(0, refX - searchRadius);
            int x1 = std::min(W - blockSize, refX + searchRadius);

            for (int sy = y0; sy <= y1; ++sy) {
                for (int sx = x0; sx <= x1; ++sx) {
                    float dist = 0.0f;
                    for (int by = 0; by < blockSize; ++by) {
                        for (int bx = 0; bx < blockSize; ++bx) {
                            float diff = refBlock[by * blockSize + bx] -
                                        floatImg.at<float>(sy + by, sx + bx);
                            dist += diff * diff;
                        }
                    }
                    dist /= (blockSize * blockSize);
                    matches.emplace_back(dist, sy, sx);
                }
            }

            // Sort by distance and keep top matches
            std::sort(matches.begin(), matches.end());
            int numMatches = std::min(maxMatches, static_cast<int>(matches.size()));

            // Stack matched blocks
            std::vector<std::vector<float>> blockStack(numMatches,
                std::vector<float>(blockSize * blockSize));

            for (int m = 0; m < numMatches; ++m) {
                int sy = std::get<1>(matches[m]);
                int sx = std::get<2>(matches[m]);
                for (int by = 0; by < blockSize; ++by) {
                    for (int bx = 0; bx < blockSize; ++bx) {
                        blockStack[m][by * blockSize + bx] =
                            floatImg.at<float>(sy + by, sx + bx);
                    }
                }
            }

            // Apply 2D DCT to each block
            for (int m = 0; m < numMatches; ++m) {
                // Row-wise DCT
                for (int by = 0; by < blockSize; ++by) {
                    std::vector<float> row(blockSize);
                    for (int bx = 0; bx < blockSize; ++bx) {
                        row[bx] = blockStack[m][by * blockSize + bx];
                    }
                    dct1D(row);
                    for (int bx = 0; bx < blockSize; ++bx) {
                        blockStack[m][by * blockSize + bx] = row[bx];
                    }
                }
                // Column-wise DCT
                for (int bx = 0; bx < blockSize; ++bx) {
                    std::vector<float> col(blockSize);
                    for (int by = 0; by < blockSize; ++by) {
                        col[by] = blockStack[m][by * blockSize + bx];
                    }
                    dct1D(col);
                    for (int by = 0; by < blockSize; ++by) {
                        blockStack[m][by * blockSize + bx] = col[by];
                    }
                }
            }

            // Apply 1D transform along stack dimension and hard threshold
            for (int by = 0; by < blockSize; ++by) {
                for (int bx = 0; bx < blockSize; ++bx) {
                    std::vector<float> stackVec(numMatches);
                    for (int m = 0; m < numMatches; ++m) {
                        stackVec[m] = blockStack[m][by * blockSize + bx];
                    }
                    if (numMatches > 1) dct1D(stackVec);

                    // Hard thresholding
                    int nonZero = 0;
                    for (int m = 0; m < numMatches; ++m) {
                        if (std::abs(stackVec[m]) < hardThresh) {
                            stackVec[m] = 0.0f;
                        } else {
                            nonZero++;
                        }
                    }

                    if (numMatches > 1) idct1D(stackVec);
                    for (int m = 0; m < numMatches; ++m) {
                        blockStack[m][by * blockSize + bx] = stackVec[m];
                    }
                }
            }

            // Apply inverse 2D DCT
            for (int m = 0; m < numMatches; ++m) {
                // Column-wise IDCT
                for (int bx = 0; bx < blockSize; ++bx) {
                    std::vector<float> col(blockSize);
                    for (int by = 0; by < blockSize; ++by) {
                        col[by] = blockStack[m][by * blockSize + bx];
                    }
                    idct1D(col);
                    for (int by = 0; by < blockSize; ++by) {
                        blockStack[m][by * blockSize + bx] = col[by];
                    }
                }
                // Row-wise IDCT
                for (int by = 0; by < blockSize; ++by) {
                    std::vector<float> row(blockSize);
                    for (int bx = 0; bx < blockSize; ++bx) {
                        row[bx] = blockStack[m][by * blockSize + bx];
                    }
                    idct1D(row);
                    for (int bx = 0; bx < blockSize; ++bx) {
                        blockStack[m][by * blockSize + bx] = row[bx];
                    }
                }
            }

            // Aggregate results
            float weight = 1.0f;  // Could be based on number of non-zero coefficients
            for (int m = 0; m < numMatches; ++m) {
                int sy = std::get<1>(matches[m]);
                int sx = std::get<2>(matches[m]);
                for (int by = 0; by < blockSize; ++by) {
                    for (int bx = 0; bx < blockSize; ++bx) {
                        estimate.at<float>(sy + by, sx + bx) +=
                            weight * blockStack[m][by * blockSize + bx];
                        weights.at<float>(sy + by, sx + bx) += weight;
                    }
                }
            }
        }
    }

    // Normalize and copy back
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float w = weights.at<float>(y, x);
            float val;
            if (w > 0) {
                val = estimate.at<float>(y, x) / w;
            } else {
                val = floatImg.at<float>(y, x);
            }
            val = clamp(val, 0.0f, maxTypeValue<T>());
            if constexpr (std::is_floating_point_v<T>) {
                slice(y, x) = static_cast<T>(val);
            } else {
                slice(y, x) = static_cast<T>(val + 0.5f);
            }
        }
    }
}

template<typename T>
void bm3dVolume(xt::xarray<T>& volume, const BM3DParams& params,
                 std::function<void(float)> progressCallback) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    std::atomic<int> completed{0};

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < sz; ++z) {
        cv::Mat_<T> slice(sy, sx);
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                slice(y, x) = volume(z, y, x);
            }
        }

        bm3dSlice(slice, params);

        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                volume(z, y, x) = slice(y, x);
            }
        }

        if (progressCallback) {
            int done = ++completed;
            if (done % 5 == 0 || done == sz) {
                progressCallback(static_cast<float>(done) / static_cast<float>(sz));
            }
        }
    }
}

// ============================================================================
// Ring Artifact Correction (Legacy)
// ============================================================================

template<typename T>
void ringCorrectSlice(cv::Mat_<T>& slice, const RingCorrectionParams& params) {
    const int H = slice.rows;
    const int W = slice.cols;

    // Determine center
    float cx = (params.centerX >= 0) ? params.centerX : static_cast<float>(W) / 2.0f;
    float cy = (params.centerY >= 0) ? params.centerY : static_cast<float>(H) / 2.0f;

    // Polar transform dimensions
    float maxRadius = std::sqrt(cx * cx + cy * cy);
    int numAngles = static_cast<int>(2.0f * PI * maxRadius);  // ~1 pixel per angle
    int numRadii = static_cast<int>(maxRadius) + 1;

    // Convert to polar
    cv::Mat polarImg(numRadii, numAngles, CV_32FC1);

    for (int r = 0; r < numRadii; ++r) {
        float radius = static_cast<float>(r);
        for (int a = 0; a < numAngles; ++a) {
            float angle = static_cast<float>(a) * 2.0f * PI / static_cast<float>(numAngles);
            float x = cx + radius * std::cos(angle);
            float y = cy + radius * std::sin(angle);

            // Bilinear interpolation
            int x0 = static_cast<int>(x);
            int y0 = static_cast<int>(y);
            float fx = x - static_cast<float>(x0);
            float fy = y - static_cast<float>(y0);

            float val = 0.0f;
            if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1) {
                val = (1 - fx) * (1 - fy) * static_cast<float>(slice(y0, x0)) +
                      fx * (1 - fy) * static_cast<float>(slice(y0, x0 + 1)) +
                      (1 - fx) * fy * static_cast<float>(slice(y0 + 1, x0)) +
                      fx * fy * static_cast<float>(slice(y0 + 1, x0 + 1));
            }
            polarImg.at<float>(r, a) = val;
        }
    }

    // Apply median filter along radial direction (vertical in polar image)
    cv::Mat filtered;
    cv::medianBlur(polarImg, filtered, params.medianWidth | 1);  // Ensure odd kernel size

    // Compute ring pattern (median filtered) and subtract
    cv::Mat ringPattern = filtered - polarImg;

    // Convert back to Cartesian and apply correction
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float radius = std::sqrt(dx * dx + dy * dy);
            float angle = std::atan2(dy, dx);
            if (angle < 0) angle += 2.0f * PI;

            // Map to polar coordinates
            int r = static_cast<int>(radius + 0.5f);
            float aF = angle * static_cast<float>(numAngles) / (2.0f * PI);
            int a = static_cast<int>(aF) % numAngles;

            if (r < numRadii && a < numAngles) {
                float correction = ringPattern.at<float>(r, a);
                float val = static_cast<float>(slice(y, x)) - correction;
                val = clamp(val, 0.0f, maxTypeValue<T>());
                if constexpr (std::is_floating_point_v<T>) {
                    slice(y, x) = static_cast<T>(val);
                } else {
                    slice(y, x) = static_cast<T>(val + 0.5f);
                }
            }
        }
    }
}

template<typename T>
void ringCorrectVolume(xt::xarray<T>& volume, const RingCorrectionParams& params) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < sz; ++z) {
        // Extract slice to cv::Mat
        cv::Mat_<T> slice(sy, sx);
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                slice(y, x) = volume(z, y, x);
            }
        }

        // Process slice
        ringCorrectSlice(slice, params);

        // Copy back
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                volume(z, y, x) = slice(y, x);
            }
        }
    }
}

// ============================================================================
// Intensity Normalization
// ============================================================================

template<typename T>
void normalizeIntensity(xt::xarray<T>& volume, const NormalizationParams& params) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    // Sample points for polynomial fitting (downsample for efficiency)
    const int minDim = std::min({sz, sy, sx});
    const int step = std::max(1, minDim / 10 + 1);
    std::vector<float> xs, ys, zs, vals;

    for (int z = 0; z < sz; z += step) {
        for (int y = 0; y < sy; y += step) {
            for (int x = 0; x < sx; x += step) {
                float v = static_cast<float>(volume(z, y, x));
                if (v > 0) {  // Only non-zero voxels
                    xs.push_back(static_cast<float>(x) / static_cast<float>(sx));
                    ys.push_back(static_cast<float>(y) / static_cast<float>(sy));
                    zs.push_back(static_cast<float>(z) / static_cast<float>(sz));
                    vals.push_back(v);
                }
            }
        }
    }

    if (vals.empty()) [[unlikely]] return;

    // Compute mean of samples
    float sampleMean = std::accumulate(vals.begin(), vals.end(), 0.0f) / static_cast<float>(vals.size());

    // For polynomial fitting, we'll use a simple trilinear model for now
    // (full polynomial fitting would require more complex least squares)
    // Background estimate: mean value
    float bgMean = sampleMean;

    // Scale factor to achieve target mean
    float scale = (bgMean > EPS) ? params.targetMean / bgMean : 1.0f;

    // Apply normalization
    constexpr float maxVal = maxTypeValue<T>();

    #pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < sz; ++z) {
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                float v = static_cast<float>(volume(z, y, x)) * scale;
                v = clamp(v, 0.0f, maxVal);
                if constexpr (std::is_floating_point_v<T>) {
                    volume(z, y, x) = static_cast<T>(v);
                } else {
                    volume(z, y, x) = static_cast<T>(v + 0.5f);
                }
            }
        }
    }
}

// ============================================================================
// 3D Anisotropic Diffusion
// ============================================================================

template<typename T>
void anisotropicDiffusion3D(xt::xarray<T>& volume, const DiffusionParams& params,
                             std::function<void(float)> progressCallback) {
    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    // Convert to float for processing
    std::vector<float> img(sz * sy * sx);
    std::vector<float> imgNew(sz * sy * sx);

    const size_t totalSize = static_cast<size_t>(sz) * sy * sx;
#pragma omp parallel for simd
    for (size_t i = 0; i < totalSize; ++i) {
        img[i] = static_cast<float>(volume.data()[i]);
    }

    // Precompute Gaussian kernel for gradient smoothing
    auto gaussKernel = gaussianKernel1D(params.sigma);
    int kRadius = static_cast<int>(gaussKernel.size()) / 2;

    // Diffusion constants
    const float GAMMA = 0.01f;
    const float CM = 7.2848f;
    const float lambda2 = params.lambda * params.lambda;

    // Helper to clamp indices
    auto ci = [&](int v, int max) { return std::max(0, std::min(max - 1, v)); };

    // Allocate buffers for gradients and structure tensor
    std::vector<float> gx(sz * sy * sx), gy(sz * sy * sx), gz(sz * sy * sx);
    std::vector<float> s11(sz * sy * sx), s12(sz * sy * sx), s13(sz * sy * sx);
    std::vector<float> s22(sz * sy * sx), s23(sz * sy * sx), s33(sz * sy * sx);

    // Restrict pointers for better vectorization
    float* __restrict__ imgPtr = img.data();
    float* __restrict__ imgNewPtr = imgNew.data();
    float* __restrict__ gxPtr = gx.data();
    float* __restrict__ gyPtr = gy.data();
    float* __restrict__ gzPtr = gz.data();
    float* __restrict__ s11Ptr = s11.data();
    float* __restrict__ s12Ptr = s12.data();
    float* __restrict__ s13Ptr = s13.data();
    float* __restrict__ s22Ptr = s22.data();
    float* __restrict__ s23Ptr = s23.data();
    float* __restrict__ s33Ptr = s33.data();

    for (int step = 0; step < params.numSteps; ++step) {
        if (progressCallback) {
            progressCallback(static_cast<float>(step) / static_cast<float>(params.numSteps));
        }

        // Compute gradients using central differences
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < sz; ++z) {
            for (int y = 0; y < sy; ++y) {
#pragma omp simd
                for (int x = 0; x < sx; ++x) {
                    int idx = z * sy * sx + y * sx + x;

                    gxPtr[idx] = 0.5f * (imgPtr[ci(z, sz) * sy * sx + ci(y, sy) * sx + ci(x + 1, sx)] -
                                          imgPtr[ci(z, sz) * sy * sx + ci(y, sy) * sx + ci(x - 1, sx)]);
                    gyPtr[idx] = 0.5f * (imgPtr[ci(z, sz) * sy * sx + ci(y + 1, sy) * sx + ci(x, sx)] -
                                          imgPtr[ci(z, sz) * sy * sx + ci(y - 1, sy) * sx + ci(x, sx)]);
                    gzPtr[idx] = 0.5f * (imgPtr[ci(z + 1, sz) * sy * sx + ci(y, sy) * sx + ci(x, sx)] -
                                          imgPtr[ci(z - 1, sz) * sy * sx + ci(y, sy) * sx + ci(x, sx)]);
                }
            }
        }

        // Compute structure tensor elements (with smoothing)
        // For simplicity, we compute unsmoothed tensor; full implementation would smooth
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < sz; ++z) {
            for (int y = 0; y < sy; ++y) {
#pragma omp simd
                for (int x = 0; x < sx; ++x) {
                    int idx = z * sy * sx + y * sx + x;
                    float gxv = gxPtr[idx], gyv = gyPtr[idx], gzv = gzPtr[idx];

                    s11Ptr[idx] = gxv * gxv;
                    s12Ptr[idx] = gxv * gyv;
                    s13Ptr[idx] = gxv * gzv;
                    s22Ptr[idx] = gyv * gyv;
                    s23Ptr[idx] = gyv * gzv;
                    s33Ptr[idx] = gzv * gzv;
                }
            }
        }

        // Diffusion step
        #pragma omp parallel for collapse(2) schedule(static)
        for (int z = 0; z < sz; ++z) {
            for (int y = 0; y < sy; ++y) {
                for (int x = 0; x < sx; ++x) {
                    int idx = z * sy * sx + y * sx + x;

                    // Compute gradient magnitude squared
                    float gradMag2 = s11Ptr[idx] + s22Ptr[idx] + s33Ptr[idx];

                    // Diffusivity function (Perona-Malik)
                    float c = GAMMA + (1.0f - GAMMA) * std::exp(-CM * gradMag2 / lambda2);

                    // 3D Laplacian with diffusivity
                    float lap = 0.0f;
                    float imgC = imgPtr[idx];

                    // X direction
                    lap += c * (imgPtr[ci(z, sz) * sy * sx + ci(y, sy) * sx + ci(x + 1, sx)] - imgC);
                    lap += c * (imgPtr[ci(z, sz) * sy * sx + ci(y, sy) * sx + ci(x - 1, sx)] - imgC);

                    // Y direction
                    lap += c * (imgPtr[ci(z, sz) * sy * sx + ci(y + 1, sy) * sx + ci(x, sx)] - imgC);
                    lap += c * (imgPtr[ci(z, sz) * sy * sx + ci(y - 1, sy) * sx + ci(x, sx)] - imgC);

                    // Z direction
                    lap += c * (imgPtr[ci(z + 1, sz) * sy * sx + ci(y, sy) * sx + ci(x, sx)] - imgC);
                    lap += c * (imgPtr[ci(z - 1, sz) * sy * sx + ci(y, sy) * sx + ci(x, sx)] - imgC);

                    imgNewPtr[idx] = imgC + params.stepSize * lap;
                }
            }
        }

        std::swap(imgPtr, imgNewPtr);
        img.swap(imgNew);
    }

    if (progressCallback) {
        progressCallback(1.0f);
    }

    // Convert back to original type
    constexpr float maxVal = maxTypeValue<T>();
    T* __restrict__ volumePtr = volume.data();

#pragma omp parallel for simd
    for (size_t i = 0; i < totalSize; ++i) {
        float v = clamp(imgPtr[i], 0.0f, maxVal);
        if constexpr (std::is_floating_point_v<T>) {
            volumePtr[i] = static_cast<T>(v);
        } else {
            volumePtr[i] = static_cast<T>(v + 0.5f);
        }
    }
}

// ============================================================================
// Gradient Computation
// ============================================================================

template<typename T>
void computeGradientVolume(const xt::xarray<T>& input,
                            xt::xarray<float>& gradX,
                            xt::xarray<float>& gradY,
                            xt::xarray<float>& gradZ,
                            const GradientParams& params) {
    const int sz = static_cast<int>(input.shape()[0]);
    const int sy = static_cast<int>(input.shape()[1]);
    const int sx = static_cast<int>(input.shape()[2]);

    gradX = xt::xarray<float>::from_shape(input.shape());
    gradY = xt::xarray<float>::from_shape(input.shape());
    gradZ = xt::xarray<float>::from_shape(input.shape());

    auto gaussKernel = gaussianKernel1D(params.sigma);
    auto derivKernel = gaussianDerivKernel1D(params.sigma);
    int kRadius = static_cast<int>(gaussKernel.size()) / 2;

    // Helper to clamp indices
    auto ci = [&](int v, int max) { return std::max(0, std::min(max - 1, v)); };

    // For efficiency, we use simple central differences with Gaussian smoothing
    // Full implementation would use separable 1D convolutions
    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < sz; ++z) {
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                float gx = 0.0f, gy = 0.0f, gz = 0.0f;
                float normSum = 0.0f;

                // Apply derivative kernel in each direction
                for (int k = -kRadius; k <= kRadius; ++k) {
                    float dw = derivKernel[k + kRadius];
                    float gw = gaussKernel[k + kRadius];

                    // X gradient: derivative in X, smooth in Y and Z
                    gx += dw * static_cast<float>(input(ci(z, sz), ci(y, sy), ci(x + k, sx)));

                    // Y gradient: derivative in Y, smooth in X and Z
                    gy += dw * static_cast<float>(input(ci(z, sz), ci(y + k, sy), ci(x, sx)));

                    // Z gradient: derivative in Z, smooth in X and Y
                    gz += dw * static_cast<float>(input(ci(z + k, sz), ci(y, sy), ci(x, sx)));
                }

                if (params.normalize) {
                    float mag = std::sqrt(gx * gx + gy * gy + gz * gz);
                    if (mag > EPS) {
                        gx /= mag;
                        gy /= mag;
                        gz /= mag;
                    }
                }

                gradX(z, y, x) = gx;
                gradY(z, y, x) = gy;
                gradZ(z, y, x) = gz;
            }
        }
    }
}

template<typename T>
void computeGradientVolume4D(const xt::xarray<T>& input,
                              xt::xarray<float>& output,
                              const GradientParams& params) {
    xt::xarray<float> gx, gy, gz;
    computeGradientVolume(input, gx, gy, gz, params);

    const int sz = static_cast<int>(input.shape()[0]);
    const int sy = static_cast<int>(input.shape()[1]);
    const int sx = static_cast<int>(input.shape()[2]);

    output = xt::xarray<float>::from_shape({3ul, static_cast<size_t>(sz),
                                             static_cast<size_t>(sy),
                                             static_cast<size_t>(sx)});

    #pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < sz; ++z) {
        for (int y = 0; y < sy; ++y) {
            for (int x = 0; x < sx; ++x) {
                output(0, z, y, x) = gx(z, y, x);  // dI/dx
                output(1, z, y, x) = gy(z, y, x);  // dI/dy
                output(2, z, y, x) = gz(z, y, x);  // dI/dz
            }
        }
    }
}

// ============================================================================
// Chunked Processing Infrastructure
// ============================================================================

template<typename T>
void processVolumeChunked(
    xt::xarray<T>& volume,
    const ChunkProcessingParams& chunkParams,
    std::function<void(xt::xarray<T>&, const std::array<int, 3>&)> processChunk) {

    const int sz = static_cast<int>(volume.shape()[0]);
    const int sy = static_cast<int>(volume.shape()[1]);
    const int sx = static_cast<int>(volume.shape()[2]);

    const int cs = chunkParams.chunkSize;
    const int overlap = chunkParams.overlap;

    // Compute number of chunks in each dimension
    int numChunksZ = (sz + cs - 1) / cs;
    int numChunksY = (sy + cs - 1) / cs;
    int numChunksX = (sx + cs - 1) / cs;

    // Set thread count if specified
    #ifdef _OPENMP
    if (chunkParams.numThreads > 0) {
        omp_set_num_threads(chunkParams.numThreads);
    }
    #endif

    // Process chunks in parallel
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (int cz = 0; cz < numChunksZ; ++cz) {
        for (int cy = 0; cy < numChunksY; ++cy) {
            for (int cx = 0; cx < numChunksX; ++cx) {
                // Chunk bounds with overlap
                int z0 = std::max(0, cz * cs - overlap);
                int z1 = std::min(sz, (cz + 1) * cs + overlap);
                int y0 = std::max(0, cy * cs - overlap);
                int y1 = std::min(sy, (cy + 1) * cs + overlap);
                int x0 = std::max(0, cx * cs - overlap);
                int x1 = std::min(sx, (cx + 1) * cs + overlap);

                int lz = z1 - z0;
                int ly = y1 - y0;
                int lx = x1 - x0;

                // Extract chunk
                xt::xarray<T> chunk = xt::xarray<T>::from_shape({
                    static_cast<size_t>(lz),
                    static_cast<size_t>(ly),
                    static_cast<size_t>(lx)
                });

                for (int z = 0; z < lz; ++z) {
                    for (int y = 0; y < ly; ++y) {
                        for (int x = 0; x < lx; ++x) {
                            chunk(z, y, x) = volume(z0 + z, y0 + y, x0 + x);
                        }
                    }
                }

                // Process chunk
                processChunk(chunk, {z0, y0, x0});

                // Copy back only the core region (excluding overlap)
                int coreZ0 = (cz == 0) ? 0 : overlap;
                int coreY0 = (cy == 0) ? 0 : overlap;
                int coreX0 = (cx == 0) ? 0 : overlap;
                int coreZ1 = (cz == numChunksZ - 1) ? lz : lz - overlap;
                int coreY1 = (cy == numChunksY - 1) ? ly : ly - overlap;
                int coreX1 = (cx == numChunksX - 1) ? lx : lx - overlap;

                for (int z = coreZ0; z < coreZ1; ++z) {
                    for (int y = coreY0; y < coreY1; ++y) {
                        for (int x = coreX0; x < coreX1; ++x) {
                            volume(z0 + z, y0 + y, x0 + x) = chunk(z, y, x);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

template void nlmDenoise3D<uint8_t>(const xt::xarray<uint8_t>&, xt::xarray<uint8_t>&, const NLMParams&);
template void nlmDenoise3D<uint16_t>(const xt::xarray<uint16_t>&, xt::xarray<uint16_t>&, const NLMParams&);
template void nlmDenoise3D<float>(const xt::xarray<float>&, xt::xarray<float>&, const NLMParams&);

template void ringCorrectSlice<uint8_t>(cv::Mat_<uint8_t>&, const RingCorrectionParams&);
template void ringCorrectSlice<uint16_t>(cv::Mat_<uint16_t>&, const RingCorrectionParams&);
template void ringCorrectSlice<float>(cv::Mat_<float>&, const RingCorrectionParams&);

template void ringCorrectVolume<uint8_t>(xt::xarray<uint8_t>&, const RingCorrectionParams&);
template void ringCorrectVolume<uint16_t>(xt::xarray<uint16_t>&, const RingCorrectionParams&);
template void ringCorrectVolume<float>(xt::xarray<float>&, const RingCorrectionParams&);

template void normalizeIntensity<uint8_t>(xt::xarray<uint8_t>&, const NormalizationParams&);
template void normalizeIntensity<uint16_t>(xt::xarray<uint16_t>&, const NormalizationParams&);
template void normalizeIntensity<float>(xt::xarray<float>&, const NormalizationParams&);

template void anisotropicDiffusion3D<uint8_t>(xt::xarray<uint8_t>&, const DiffusionParams&, std::function<void(float)>);
template void anisotropicDiffusion3D<uint16_t>(xt::xarray<uint16_t>&, const DiffusionParams&, std::function<void(float)>);
template void anisotropicDiffusion3D<float>(xt::xarray<float>&, const DiffusionParams&, std::function<void(float)>);

template void computeGradientVolume<uint8_t>(const xt::xarray<uint8_t>&, xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);
template void computeGradientVolume<uint16_t>(const xt::xarray<uint16_t>&, xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);
template void computeGradientVolume<float>(const xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);

template void computeGradientVolume4D<uint8_t>(const xt::xarray<uint8_t>&, xt::xarray<float>&, const GradientParams&);
template void computeGradientVolume4D<uint16_t>(const xt::xarray<uint16_t>&, xt::xarray<float>&, const GradientParams&);
template void computeGradientVolume4D<float>(const xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);

template void processVolumeChunked<uint8_t>(xt::xarray<uint8_t>&, const ChunkProcessingParams&, std::function<void(xt::xarray<uint8_t>&, const std::array<int, 3>&)>);
template void processVolumeChunked<uint16_t>(xt::xarray<uint16_t>&, const ChunkProcessingParams&, std::function<void(xt::xarray<uint16_t>&, const std::array<int, 3>&)>);
template void processVolumeChunked<float>(xt::xarray<float>&, const ChunkProcessingParams&, std::function<void(xt::xarray<float>&, const std::array<int, 3>&)>);

// Vo ring removal
template void voRingRemoveSlice<uint8_t>(cv::Mat_<uint8_t>&, const VoRingParams&);
template void voRingRemoveSlice<uint16_t>(cv::Mat_<uint16_t>&, const VoRingParams&);
template void voRingRemoveSlice<float>(cv::Mat_<float>&, const VoRingParams&);

template void voRingRemoveVolume<uint8_t>(xt::xarray<uint8_t>&, const VoRingParams&, std::function<void(float)>);
template void voRingRemoveVolume<uint16_t>(xt::xarray<uint16_t>&, const VoRingParams&, std::function<void(float)>);
template void voRingRemoveVolume<float>(xt::xarray<float>&, const VoRingParams&, std::function<void(float)>);

// Stripe removal
template void stripeRemoveSlice<uint8_t>(cv::Mat_<uint8_t>&, const StripeRemovalParams&);
template void stripeRemoveSlice<uint16_t>(cv::Mat_<uint16_t>&, const StripeRemovalParams&);
template void stripeRemoveSlice<float>(cv::Mat_<float>&, const StripeRemovalParams&);

template void stripeRemoveVolume<uint8_t>(xt::xarray<uint8_t>&, const StripeRemovalParams&, std::function<void(float)>);
template void stripeRemoveVolume<uint16_t>(xt::xarray<uint16_t>&, const StripeRemovalParams&, std::function<void(float)>);
template void stripeRemoveVolume<float>(xt::xarray<float>&, const StripeRemovalParams&, std::function<void(float)>);

// 3D CLAHE
template void clahe3D<uint8_t>(xt::xarray<uint8_t>&, const CLAHE3DParams&, std::function<void(float)>);
template void clahe3D<uint16_t>(xt::xarray<uint16_t>&, const CLAHE3DParams&, std::function<void(float)>);
template void clahe3D<float>(xt::xarray<float>&, const CLAHE3DParams&, std::function<void(float)>);

// BM3D
template void bm3dSlice<uint8_t>(cv::Mat_<uint8_t>&, const BM3DParams&);
template void bm3dSlice<uint16_t>(cv::Mat_<uint16_t>&, const BM3DParams&);
template void bm3dSlice<float>(cv::Mat_<float>&, const BM3DParams&);

template void bm3dVolume<uint8_t>(xt::xarray<uint8_t>&, const BM3DParams&, std::function<void(float)>);
template void bm3dVolume<uint16_t>(xt::xarray<uint16_t>&, const BM3DParams&, std::function<void(float)>);
template void bm3dVolume<float>(xt::xarray<float>&, const BM3DParams&, std::function<void(float)>);

}  // namespace vc
