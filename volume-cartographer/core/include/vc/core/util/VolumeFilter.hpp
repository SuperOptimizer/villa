#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <opencv2/core.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <array>

namespace cv {
template <typename _Tp> class Mat_;
}  // namespace cv

namespace vc {

// ============================================================================
// Non-Local Means Denoising
// ============================================================================

/**
 * @brief Parameters for 3D Non-Local Means denoising
 */
struct NLMParams {
    int searchRadius = 10;   ///< Search window radius (21x21x21 search window)
    int patchRadius = 3;     ///< Patch half-size (7x7x7 patches)
    float h = 10.0f;         ///< Filter strength (higher = more smoothing)
};

/**
 * @brief Apply 3D Non-Local Means denoising to a volume chunk
 *
 * For each voxel, finds similar patches in the search window and weights
 * contributions by exp(-||patch_i - patch_j||^2 / h^2).
 *
 * @tparam T Data type (uint8_t, uint16_t, or float)
 * @param input Input chunk (ZYX ordering)
 * @param output Output chunk (must be same size as input)
 * @param params NLM parameters
 */
template<typename T>
void nlmDenoise3D(const xt::xarray<T>& input, xt::xarray<T>& output, const NLMParams& params);

// ============================================================================
// Ring Artifact Correction (Vo Algorithm)
// ============================================================================

/**
 * @brief Parameters for Vo ring artifact removal
 *
 * Based on: Vo et al. "Superior techniques for eliminating ring artifacts
 * in X-ray micro-tomography" Optics Express 2018
 */
struct VoRingParams {
    int snr = 3;                 ///< Signal-to-noise ratio for thresholding (1-5, higher = less aggressive)
    int la = 81;                 ///< Size of mean filter for large rings (must be odd)
    int sm = 21;                 ///< Size of median filter for small/medium rings (must be odd)
    float centerX = -1.0f;       ///< Rotation center X (-1 = auto-detect as image center)
    float centerY = -1.0f;       ///< Rotation center Y (-1 = auto-detect as image center)
};

/**
 * @brief Remove ring artifacts using Vo algorithm (sorting + FFT + regularization)
 *
 * Much more effective than simple polar+median approach. Combines:
 * 1. Sorting-based approach for large rings
 * 2. FFT-based filtering for small/medium rings
 * 3. Regularization to prevent over-correction
 *
 * @tparam T Data type
 * @param slice Input/output 2D slice (modified in place)
 * @param params Vo algorithm parameters
 */
template<typename T>
void voRingRemoveSlice(cv::Mat_<T>& slice, const VoRingParams& params);

/**
 * @brief Remove ring artifacts from volume using Vo algorithm (per-slice)
 *
 * @tparam T Data type
 * @param volume Input/output volume (ZYX ordering, modified in place)
 * @param params Vo algorithm parameters
 * @param progressCallback Optional progress callback
 */
template<typename T>
void voRingRemoveVolume(xt::xarray<T>& volume, const VoRingParams& params,
                         std::function<void(float)> progressCallback = nullptr);

// Legacy ring correction (kept for compatibility)
struct RingCorrectionParams {
    float centerX = -1.0f;
    float centerY = -1.0f;
    int medianWidth = 5;
};

template<typename T>
void ringCorrectSlice(cv::Mat_<T>& slice, const RingCorrectionParams& params);

template<typename T>
void ringCorrectVolume(xt::xarray<T>& volume, const RingCorrectionParams& params);

// ============================================================================
// Stripe Artifact Removal (FFT-based)
// ============================================================================

/**
 * @brief Parameters for FFT stripe removal
 */
struct StripeRemovalParams {
    float sigma = 3.0f;          ///< Gaussian damping sigma in Fourier domain
    int decNum = 4;              ///< Number of decomposition levels for wavelet
    bool useWavelet = false;     ///< Use wavelet-FFT hybrid (slower but better)
};

/**
 * @brief Remove vertical stripe artifacts using FFT filtering
 *
 * Filters out vertical stripes (horizontal lines in FFT) common in
 * synchrotron CT data. Applies Gaussian damping to horizontal frequencies.
 *
 * @tparam T Data type
 * @param slice Input/output 2D slice (modified in place)
 * @param params Stripe removal parameters
 */
template<typename T>
void stripeRemoveSlice(cv::Mat_<T>& slice, const StripeRemovalParams& params);

/**
 * @brief Remove stripe artifacts from volume (per-slice)
 *
 * @tparam T Data type
 * @param volume Input/output volume (ZYX ordering, modified in place)
 * @param params Stripe removal parameters
 * @param progressCallback Optional progress callback
 */
template<typename T>
void stripeRemoveVolume(xt::xarray<T>& volume, const StripeRemovalParams& params,
                         std::function<void(float)> progressCallback = nullptr);

// ============================================================================
// 3D CLAHE (Contrast Limited Adaptive Histogram Equalization)
// ============================================================================

/**
 * @brief Parameters for 3D CLAHE
 */
struct CLAHE3DParams {
    int tileSize = 64;           ///< Size of tiles in each dimension
    float clipLimit = 4.0f;      ///< Contrast limit (higher = more contrast)
    int numBins = 256;           ///< Number of histogram bins
};

/**
 * @brief Apply 3D CLAHE for local contrast enhancement
 *
 * Divides volume into tiles, computes contrast-limited histogram equalization
 * per tile, then uses trilinear interpolation for smooth transitions.
 * Essential for enhancing low-contrast features like ink traces.
 *
 * @tparam T Data type
 * @param volume Input/output volume (modified in place)
 * @param params CLAHE parameters
 * @param progressCallback Optional progress callback
 */
template<typename T>
void clahe3D(xt::xarray<T>& volume, const CLAHE3DParams& params,
              std::function<void(float)> progressCallback = nullptr);

// ============================================================================
// BM3D Denoising (Block-Matching 3D)
// ============================================================================

/**
 * @brief Parameters for BM3D denoising
 */
struct BM3DParams {
    float sigma = 25.0f;         ///< Noise standard deviation estimate
    int blockSize = 8;           ///< Block size for matching (8 is standard)
    int searchRadius = 16;       ///< Search window radius for block matching
    int maxMatches = 16;         ///< Maximum number of matched blocks
    float hardThreshold = 2.7f;  ///< Hard threshold multiplier (stage 1)
    bool wienerFiltering = true; ///< Enable Wiener filtering (stage 2)
};

/**
 * @brief Apply BM3D denoising to a 2D slice
 *
 * Block-Matching 3D denoising (applied per-slice as BM2D):
 * 1. Block matching: find similar blocks in search window
 * 2. Collaborative filtering: stack and transform, threshold, inverse
 * 3. Aggregation: weighted averaging of overlapping estimates
 * 4. Optional Wiener filtering: second stage with improved estimates
 *
 * @tparam T Data type
 * @param slice Input/output 2D slice (modified in place)
 * @param params BM3D parameters
 */
template<typename T>
void bm3dSlice(cv::Mat_<T>& slice, const BM3DParams& params);

/**
 * @brief Apply BM3D denoising to volume (per-slice with temporal consistency)
 *
 * @tparam T Data type
 * @param volume Input/output volume (modified in place)
 * @param params BM3D parameters
 * @param progressCallback Optional progress callback
 */
template<typename T>
void bm3dVolume(xt::xarray<T>& volume, const BM3DParams& params,
                 std::function<void(float)> progressCallback = nullptr);

// ============================================================================
// Intensity Normalization (Beam Hardening Correction)
// ============================================================================

/**
 * @brief Parameters for intensity normalization
 */
struct NormalizationParams {
    int polyOrder = 2;          ///< Polynomial order for surface fitting
    float targetMean = 32768.0f;///< Target mean intensity
};

/**
 * @brief Normalize intensity using polynomial surface fitting
 *
 * Fits a polynomial surface to the volume to estimate the background
 * illumination pattern, then normalizes to correct beam hardening.
 *
 * @tparam T Data type
 * @param volume Input/output volume (modified in place)
 * @param params Normalization parameters
 */
template<typename T>
void normalizeIntensity(xt::xarray<T>& volume, const NormalizationParams& params);

// ============================================================================
// 3D Anisotropic Diffusion
// ============================================================================

/**
 * @brief Parameters for 3D anisotropic diffusion
 */
struct DiffusionParams {
    float lambda = 1.0f;     ///< Edge threshold parameter
    float sigma = 3.0f;      ///< Gaussian sigma for gradient smoothing
    int numSteps = 50;       ///< Number of diffusion iterations
    float stepSize = 0.1f;   ///< Time step (must be <= 1/6 for 3D stability)
};

/**
 * @brief Apply 3D anisotropic diffusion for edge-preserving smoothing
 *
 * Extends the 2D CED approach to 3D:
 * - Computes 3x3 structure tensor J = smooth(grad * grad^T)
 * - Eigendecomposition for principal directions
 * - Diffuses preferentially along low-gradient directions
 *
 * @tparam T Data type
 * @param volume Input/output volume (modified in place)
 * @param params Diffusion parameters
 * @param progressCallback Optional callback for progress updates (0.0 to 1.0)
 */
template<typename T>
void anisotropicDiffusion3D(xt::xarray<T>& volume, const DiffusionParams& params,
                             std::function<void(float)> progressCallback = nullptr);

// ============================================================================
// Gradient Computation
// ============================================================================

/**
 * @brief Parameters for gradient computation
 */
struct GradientParams {
    float sigma = 1.0f;      ///< Pre-smoothing sigma (Gaussian derivative kernel)
    bool normalize = false;  ///< Normalize gradients to unit vectors
};

/**
 * @brief Compute gradient field for a volume
 *
 * Uses Gaussian derivative kernels for smooth gradient estimation:
 * G'(x, sigma) = -x / sigma^2 * G(x, sigma)
 *
 * @tparam T Input data type
 * @param input Input volume (ZYX ordering)
 * @param gradX Output gradient in X direction
 * @param gradY Output gradient in Y direction
 * @param gradZ Output gradient in Z direction
 * @param params Gradient computation parameters
 */
template<typename T>
void computeGradientVolume(const xt::xarray<T>& input,
                            xt::xarray<float>& gradX,
                            xt::xarray<float>& gradY,
                            xt::xarray<float>& gradZ,
                            const GradientParams& params);

/**
 * @brief Compute gradient field and store as 4D array (channel-first)
 *
 * Output shape: [3, Z, Y, X] where channel 0=dI/dx, 1=dI/dy, 2=dI/dz
 *
 * @tparam T Input data type
 * @param input Input volume
 * @param output Output gradient volume [3, Z, Y, X]
 * @param params Gradient computation parameters
 */
template<typename T>
void computeGradientVolume4D(const xt::xarray<T>& input,
                              xt::xarray<float>& output,
                              const GradientParams& params);

// ============================================================================
// Chunked Processing Infrastructure
// ============================================================================

/**
 * @brief Parameters for chunked volume processing
 */
struct ChunkProcessingParams {
    int chunkSize = 128;     ///< Processing chunk size (cubic)
    int overlap = 16;        ///< Chunk overlap for boundary handling
    int numThreads = 0;      ///< Number of threads (0 = auto)
};

/**
 * @brief Process a volume in overlapping chunks
 *
 * @tparam T Data type
 * @param volume Input/output volume
 * @param chunkParams Chunking parameters
 * @param processChunk Function to process each chunk: (chunk, globalOffset) -> void
 *                     The chunk includes overlap regions; only the core region
 *                     should be written back.
 */
template<typename T>
void processVolumeChunked(
    xt::xarray<T>& volume,
    const ChunkProcessingParams& chunkParams,
    std::function<void(xt::xarray<T>&, const std::array<int, 3>&)> processChunk);

// ============================================================================
// Explicit template instantiation declarations
// ============================================================================

extern template void nlmDenoise3D<uint8_t>(const xt::xarray<uint8_t>&, xt::xarray<uint8_t>&, const NLMParams&);
extern template void nlmDenoise3D<uint16_t>(const xt::xarray<uint16_t>&, xt::xarray<uint16_t>&, const NLMParams&);
extern template void nlmDenoise3D<float>(const xt::xarray<float>&, xt::xarray<float>&, const NLMParams&);

// Vo ring removal
extern template void voRingRemoveSlice<uint8_t>(cv::Mat_<uint8_t>&, const VoRingParams&);
extern template void voRingRemoveSlice<uint16_t>(cv::Mat_<uint16_t>&, const VoRingParams&);
extern template void voRingRemoveSlice<float>(cv::Mat_<float>&, const VoRingParams&);

extern template void voRingRemoveVolume<uint8_t>(xt::xarray<uint8_t>&, const VoRingParams&, std::function<void(float)>);
extern template void voRingRemoveVolume<uint16_t>(xt::xarray<uint16_t>&, const VoRingParams&, std::function<void(float)>);
extern template void voRingRemoveVolume<float>(xt::xarray<float>&, const VoRingParams&, std::function<void(float)>);

// Legacy ring correction
extern template void ringCorrectSlice<uint8_t>(cv::Mat_<uint8_t>&, const RingCorrectionParams&);
extern template void ringCorrectSlice<uint16_t>(cv::Mat_<uint16_t>&, const RingCorrectionParams&);
extern template void ringCorrectSlice<float>(cv::Mat_<float>&, const RingCorrectionParams&);

extern template void ringCorrectVolume<uint8_t>(xt::xarray<uint8_t>&, const RingCorrectionParams&);
extern template void ringCorrectVolume<uint16_t>(xt::xarray<uint16_t>&, const RingCorrectionParams&);
extern template void ringCorrectVolume<float>(xt::xarray<float>&, const RingCorrectionParams&);

// Stripe removal
extern template void stripeRemoveSlice<uint8_t>(cv::Mat_<uint8_t>&, const StripeRemovalParams&);
extern template void stripeRemoveSlice<uint16_t>(cv::Mat_<uint16_t>&, const StripeRemovalParams&);
extern template void stripeRemoveSlice<float>(cv::Mat_<float>&, const StripeRemovalParams&);

extern template void stripeRemoveVolume<uint8_t>(xt::xarray<uint8_t>&, const StripeRemovalParams&, std::function<void(float)>);
extern template void stripeRemoveVolume<uint16_t>(xt::xarray<uint16_t>&, const StripeRemovalParams&, std::function<void(float)>);
extern template void stripeRemoveVolume<float>(xt::xarray<float>&, const StripeRemovalParams&, std::function<void(float)>);

// 3D CLAHE
extern template void clahe3D<uint8_t>(xt::xarray<uint8_t>&, const CLAHE3DParams&, std::function<void(float)>);
extern template void clahe3D<uint16_t>(xt::xarray<uint16_t>&, const CLAHE3DParams&, std::function<void(float)>);
extern template void clahe3D<float>(xt::xarray<float>&, const CLAHE3DParams&, std::function<void(float)>);

// BM3D
extern template void bm3dSlice<uint8_t>(cv::Mat_<uint8_t>&, const BM3DParams&);
extern template void bm3dSlice<uint16_t>(cv::Mat_<uint16_t>&, const BM3DParams&);
extern template void bm3dSlice<float>(cv::Mat_<float>&, const BM3DParams&);

extern template void bm3dVolume<uint8_t>(xt::xarray<uint8_t>&, const BM3DParams&, std::function<void(float)>);
extern template void bm3dVolume<uint16_t>(xt::xarray<uint16_t>&, const BM3DParams&, std::function<void(float)>);
extern template void bm3dVolume<float>(xt::xarray<float>&, const BM3DParams&, std::function<void(float)>);

extern template void normalizeIntensity<uint8_t>(xt::xarray<uint8_t>&, const NormalizationParams&);
extern template void normalizeIntensity<uint16_t>(xt::xarray<uint16_t>&, const NormalizationParams&);
extern template void normalizeIntensity<float>(xt::xarray<float>&, const NormalizationParams&);

extern template void anisotropicDiffusion3D<uint8_t>(xt::xarray<uint8_t>&, const DiffusionParams&, std::function<void(float)>);
extern template void anisotropicDiffusion3D<uint16_t>(xt::xarray<uint16_t>&, const DiffusionParams&, std::function<void(float)>);
extern template void anisotropicDiffusion3D<float>(xt::xarray<float>&, const DiffusionParams&, std::function<void(float)>);

extern template void computeGradientVolume<uint8_t>(const xt::xarray<uint8_t>&, xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);
extern template void computeGradientVolume<uint16_t>(const xt::xarray<uint16_t>&, xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);
extern template void computeGradientVolume<float>(const xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);

extern template void computeGradientVolume4D<uint8_t>(const xt::xarray<uint8_t>&, xt::xarray<float>&, const GradientParams&);
extern template void computeGradientVolume4D<uint16_t>(const xt::xarray<uint16_t>&, xt::xarray<float>&, const GradientParams&);
extern template void computeGradientVolume4D<float>(const xt::xarray<float>&, xt::xarray<float>&, const GradientParams&);

}  // namespace vc
