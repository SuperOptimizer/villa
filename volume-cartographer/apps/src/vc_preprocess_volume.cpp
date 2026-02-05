/**
 * @file vc_preprocess_volume.cpp
 * @brief SOTA volume preprocessing pipeline for noisy 3D X-ray/CT data
 *
 * Implements a modular pipeline with SOTA algorithms:
 * 1. BM3D denoising (Block-Matching 3D)
 * 2. NLM denoising (3D Non-Local Means) - alternative to BM3D
 * 3. Vo ring artifact removal (sorting + FFT + regularization)
 * 4. FFT stripe removal (for synchrotron data)
 * 5. 3D CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * 6. Normalize (polynomial surface fit for beam hardening)
 * 7. Resample (Lanczos-3 interpolation)
 * 8. Compute gradients (Gaussian derivative kernels)
 * 9. Diffuse (3D anisotropic diffusion)
 *
 * Features:
 * - Preview mode for quick parameter tuning
 * - Runtime estimation and ETA display
 * - Chunked processing for large volumes
 * - OpenMP parallelization
 */

#include <boost/program_options.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include <chrono>
#include <sstream>
#include <map>

#include <nlohmann/json.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/attributes.hxx"

#include <xtensor/containers/xarray.hpp>

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>

#include "vc/core/util/VolumeFilter.hpp"
#include "vc/core/util/Interpolation.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace po = boost::program_options;
namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================================
// Configuration
// ============================================================================

struct Config {
    // Input/output
    std::string inputPath;
    std::string outputPath;
    std::string gradientOutputPath;

    // Stage control
    bool bm3d = false;
    bool denoise = false;
    bool voRing = false;
    bool ringCorrect = false;  // Legacy
    bool stripeRemove = false;
    bool clahe = false;
    bool normalize = false;
    bool resample = false;
    bool computeGradients = false;
    bool diffuse = false;

    // BM3D options
    float bm3dSigma = 25.0f;
    int bm3dBlockSize = 8;
    int bm3dSearchRadius = 16;
    int bm3dMaxMatches = 16;
    float bm3dHardThreshold = 2.7f;
    bool bm3dWiener = true;

    // Denoising options (NLM)
    int denoiseSearchRadius = 10;
    int denoisePatchRadius = 3;
    float denoiseH = 10.0f;

    // Vo ring correction options
    int voSnr = 3;
    int voLa = 81;
    int voSm = 21;
    float voCenterX = -1.0f;
    float voCenterY = -1.0f;

    // Legacy ring correction options
    float ringCenterX = -1.0f;
    float ringCenterY = -1.0f;
    int ringMedianWidth = 5;

    // Stripe removal options
    float stripeSigma = 3.0f;
    int stripeDecNum = 4;
    bool stripeUseWavelet = false;

    // 3D CLAHE options
    int claheTileSize = 64;
    float claheClipLimit = 4.0f;
    int claheNumBins = 256;

    // Normalization options
    int normPolyOrder = 2;
    float normTargetMean = 32768.0f;

    // Resampling options
    float resampleScaleX = 1.0f;
    float resampleScaleY = 1.0f;
    float resampleScaleZ = 1.0f;

    // Gradient options
    float gradientSigma = 1.0f;
    bool gradientNormalize = false;

    // Diffusion options
    float diffuseLambda = 1.0f;
    float diffuseSigma = 3.0f;
    int diffuseSteps = 50;

    // General options
    int chunkSize = 128;
    int overlap = 16;
    int numThreads = 0;
    size_t cacheMB = 4096;
    bool verbose = false;

    // Preview mode
    bool preview = false;
    int previewSize = 256;  // Size of preview subvolume in each dimension
    int previewZ = -1;      // Starting Z for preview (-1 = center)
    int previewY = -1;      // Starting Y for preview (-1 = center)
    int previewX = -1;      // Starting X for preview (-1 = center)

    // Runtime estimation
    bool showEta = true;
};

static std::mutex g_printMtx;

static void log(const Config& cfg, const std::string& msg) {
    if (cfg.verbose) {
        std::lock_guard<std::mutex> lock(g_printMtx);
        std::cout << "[INFO] " << msg << "\n";
    }
}

// ============================================================================
// Progress reporting
// ============================================================================

class ProgressReporter {
public:
    ProgressReporter(const std::string& stage, size_t total, bool showEta = true)
        : stage_(stage), total_(total), done_(0), lastPct_(-1), showEta_(showEta),
          startTime_(std::chrono::steady_clock::now()) {}

    void update(size_t increment = 1) {
        size_t d = done_.fetch_add(increment) + increment;
        int pct = static_cast<int>(100.0 * static_cast<double>(d) / static_cast<double>(total_));
        if (pct != lastPct_) {
            lastPct_ = pct;

            std::string etaStr;
            if (showEta_ && d > 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime_).count();
                if (elapsed > 2 && d > 1) {  // Only show ETA after some progress
                    double rate = static_cast<double>(d) / static_cast<double>(elapsed);
                    int remaining = static_cast<int>((total_ - d) / rate);
                    int hours = remaining / 3600;
                    int minutes = (remaining % 3600) / 60;
                    int seconds = remaining % 60;
                    std::ostringstream oss;
                    if (hours > 0) {
                        oss << " ETA: " << hours << "h" << minutes << "m";
                    } else if (minutes > 0) {
                        oss << " ETA: " << minutes << "m" << seconds << "s";
                    } else {
                        oss << " ETA: " << seconds << "s";
                    }
                    etaStr = oss.str();
                }
            }

            std::lock_guard<std::mutex> lock(g_printMtx);
            std::cout << "\r[" << stage_ << "] " << d << "/" << total_
                      << " (" << pct << "%)" << etaStr << "          " << std::flush;
        }
    }

    void finish() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime_).count();
        std::lock_guard<std::mutex> lock(g_printMtx);
        std::cout << "\r[" << stage_ << "] Complete (" << total_ << "/" << total_
                  << ") in " << elapsed << "s                    " << "\n";
    }

    double getElapsedSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - startTime_).count();
    }

private:
    std::string stage_;
    size_t total_;
    std::atomic<size_t> done_;
    std::atomic<int> lastPct_;
    bool showEta_;
    std::chrono::steady_clock::time_point startTime_;
};

// ============================================================================
// Zarr I/O helpers
// ============================================================================

struct VolumeInfo {
    size_t sz, sy, sx;
    z5::types::Datatype dtype;
    std::vector<size_t> chunkShape;
};

static VolumeInfo getVolumeInfo(z5::Dataset* ds) {
    VolumeInfo info;
    const auto& shape = ds->shape();
    info.sz = shape[0];
    info.sy = shape[1];
    info.sx = shape[2];
    info.dtype = ds->getDtype();
    info.chunkShape = ds->defaultChunkShape();
    return info;
}

// Runtime estimator for the full pipeline
class RuntimeEstimator {
public:
    struct StageTime {
        std::string name;
        double voxelsPerSecond;  // Estimated from preview
    };

    void recordStage(const std::string& name, size_t voxels, double seconds) {
        if (seconds > 0.001) {
            stageTimes_[name] = {name, static_cast<double>(voxels) / seconds};
        }
    }

    void printEstimate(const VolumeInfo& info, const std::vector<std::string>& stages) {
        size_t totalVoxels = info.sz * info.sy * info.sx;

        std::cout << "\n=== Runtime Estimate (based on preview) ===" << "\n";
        double totalSeconds = 0.0;

        for (const auto& stage : stages) {
            if (stageTimes_.count(stage)) {
                double seconds = static_cast<double>(totalVoxels) / stageTimes_[stage].voxelsPerSecond;
                totalSeconds += seconds;
                int hours = static_cast<int>(seconds) / 3600;
                int minutes = (static_cast<int>(seconds) % 3600) / 60;
                std::cout << "  " << stage << ": ";
                if (hours > 0) {
                    std::cout << hours << "h " << minutes << "m" << "\n";
                } else {
                    std::cout << minutes << "m " << (static_cast<int>(seconds) % 60) << "s" << "\n";
                }
            }
        }

        int totalHours = static_cast<int>(totalSeconds) / 3600;
        int totalMinutes = (static_cast<int>(totalSeconds) % 3600) / 60;
        std::cout << "  TOTAL: ";
        if (totalHours > 0) {
            std::cout << totalHours << "h " << totalMinutes << "m" << "\n";
        } else {
            std::cout << totalMinutes << "m " << (static_cast<int>(totalSeconds) % 60) << "s" << "\n";
        }
        std::cout << "==========================================\n" << "\n";
    }

private:
    std::map<std::string, StageTime> stageTimes_;
};

static RuntimeEstimator g_estimator;

template<typename T>
static void readChunk(std::unique_ptr<z5::Dataset>& ds, xt::xarray<T>& chunk,
                      size_t z0, size_t y0, size_t x0,
                      size_t lz, size_t ly, size_t lx) {
    chunk = xt::xarray<T>::from_shape({lz, ly, lx});
    readSubarray3D(chunk, *ds, {z0, y0, x0});
}

template<typename T>
static void writeChunk(std::unique_ptr<z5::Dataset>& ds, const xt::xarray<T>& chunk,
                       size_t z0, size_t y0, size_t x0) {
    writeSubarray3D(*ds, chunk, {z0, y0, x0});
}

// ============================================================================
// Stage implementations
// ============================================================================

template<typename T>
static void runDenoise(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                       std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running NLM denoising...");

    vc::NLMParams params;
    params.searchRadius = cfg.denoiseSearchRadius;
    params.patchRadius = cfg.denoisePatchRadius;
    params.h = cfg.denoiseH;

    // Required overlap for NLM
    int nlmOverlap = params.searchRadius + params.patchRadius + 1;
    int effectiveOverlap = std::max(cfg.overlap, nlmOverlap);

    size_t numChunksZ = (info.sz + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksY = (info.sy + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksX = (info.sx + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t totalChunks = numChunksZ * numChunksY * numChunksX;

    ProgressReporter progress("Denoise", totalChunks);

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t cz = 0; cz < numChunksZ; ++cz) {
        for (size_t cy = 0; cy < numChunksY; ++cy) {
            for (size_t cx = 0; cx < numChunksX; ++cx) {
                // Compute chunk bounds with overlap
                size_t z0 = cz * cfg.chunkSize;
                size_t y0 = cy * cfg.chunkSize;
                size_t x0 = cx * cfg.chunkSize;

                size_t z0o = (z0 >= static_cast<size_t>(effectiveOverlap)) ? z0 - effectiveOverlap : 0;
                size_t y0o = (y0 >= static_cast<size_t>(effectiveOverlap)) ? y0 - effectiveOverlap : 0;
                size_t x0o = (x0 >= static_cast<size_t>(effectiveOverlap)) ? x0 - effectiveOverlap : 0;

                size_t z1 = std::min(z0 + cfg.chunkSize, info.sz);
                size_t y1 = std::min(y0 + cfg.chunkSize, info.sy);
                size_t x1 = std::min(x0 + cfg.chunkSize, info.sx);

                size_t z1o = std::min(z1 + effectiveOverlap, info.sz);
                size_t y1o = std::min(y1 + effectiveOverlap, info.sy);
                size_t x1o = std::min(x1 + effectiveOverlap, info.sx);

                // Read chunk with overlap
                xt::xarray<T> chunkIn;
                #pragma omp critical
                {
                    readChunk<T>(dsIn, chunkIn, z0o, y0o, x0o, z1o - z0o, y1o - y0o, x1o - x0o);
                }

                // Process
                xt::xarray<T> chunkOut;
                vc::nlmDenoise3D(chunkIn, chunkOut, params);

                // Extract core region
                size_t coreZ0 = z0 - z0o;
                size_t coreY0 = y0 - y0o;
                size_t coreX0 = x0 - x0o;
                size_t coreLz = z1 - z0;
                size_t coreLy = y1 - y0;
                size_t coreLx = x1 - x0;

                xt::xarray<T> core = xt::xarray<T>::from_shape({coreLz, coreLy, coreLx});
                for (size_t z = 0; z < coreLz; ++z) {
                    for (size_t y = 0; y < coreLy; ++y) {
                        for (size_t x = 0; x < coreLx; ++x) {
                            core(z, y, x) = chunkOut(coreZ0 + z, coreY0 + y, coreX0 + x);
                        }
                    }
                }

                // Write core region
                #pragma omp critical
                {
                    writeChunk<T>(dsOut, core, z0, y0, x0);
                }

                progress.update();
            }
        }
    }

    progress.finish();
}

template<typename T>
static void runBM3D(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                    std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running BM3D denoising...");

    vc::BM3DParams params;
    params.sigma = cfg.bm3dSigma;
    params.blockSize = cfg.bm3dBlockSize;
    params.searchRadius = cfg.bm3dSearchRadius;
    params.maxMatches = cfg.bm3dMaxMatches;
    params.hardThreshold = cfg.bm3dHardThreshold;
    params.wienerFiltering = cfg.bm3dWiener;

    ProgressReporter progress("BM3D", info.sz, cfg.showEta);

    #pragma omp parallel for schedule(dynamic)
    for (size_t z = 0; z < info.sz; ++z) {
        xt::xarray<T> slice;
        #pragma omp critical
        {
            readChunk<T>(dsIn, slice, z, 0, 0, 1, info.sy, info.sx);
        }

        cv::Mat_<T> sliceMat(static_cast<int>(info.sy), static_cast<int>(info.sx));
        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                sliceMat(static_cast<int>(y), static_cast<int>(x)) = slice(0, y, x);
            }
        }

        vc::bm3dSlice(sliceMat, params);

        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                slice(0, y, x) = sliceMat(static_cast<int>(y), static_cast<int>(x));
            }
        }

        #pragma omp critical
        {
            writeChunk<T>(dsOut, slice, z, 0, 0);
        }

        progress.update();
    }

    progress.finish();
}

template<typename T>
static void runVoRingCorrect(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                              std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running Vo ring artifact removal...");

    vc::VoRingParams params;
    params.snr = cfg.voSnr;
    params.la = cfg.voLa;
    params.sm = cfg.voSm;
    params.centerX = cfg.voCenterX;
    params.centerY = cfg.voCenterY;

    ProgressReporter progress("VoRing", info.sz, cfg.showEta);

    #pragma omp parallel for schedule(dynamic)
    for (size_t z = 0; z < info.sz; ++z) {
        xt::xarray<T> slice;
        #pragma omp critical
        {
            readChunk<T>(dsIn, slice, z, 0, 0, 1, info.sy, info.sx);
        }

        cv::Mat_<T> sliceMat(static_cast<int>(info.sy), static_cast<int>(info.sx));
        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                sliceMat(static_cast<int>(y), static_cast<int>(x)) = slice(0, y, x);
            }
        }

        vc::voRingRemoveSlice(sliceMat, params);

        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                slice(0, y, x) = sliceMat(static_cast<int>(y), static_cast<int>(x));
            }
        }

        #pragma omp critical
        {
            writeChunk<T>(dsOut, slice, z, 0, 0);
        }

        progress.update();
    }

    progress.finish();
}

template<typename T>
static void runStripeRemove(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                             std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running FFT stripe removal...");

    vc::StripeRemovalParams params;
    params.sigma = cfg.stripeSigma;
    params.decNum = cfg.stripeDecNum;
    params.useWavelet = cfg.stripeUseWavelet;

    ProgressReporter progress("StripeRemove", info.sz, cfg.showEta);

    #pragma omp parallel for schedule(dynamic)
    for (size_t z = 0; z < info.sz; ++z) {
        xt::xarray<T> slice;
        #pragma omp critical
        {
            readChunk<T>(dsIn, slice, z, 0, 0, 1, info.sy, info.sx);
        }

        cv::Mat_<T> sliceMat(static_cast<int>(info.sy), static_cast<int>(info.sx));
        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                sliceMat(static_cast<int>(y), static_cast<int>(x)) = slice(0, y, x);
            }
        }

        vc::stripeRemoveSlice(sliceMat, params);

        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                slice(0, y, x) = sliceMat(static_cast<int>(y), static_cast<int>(x));
            }
        }

        #pragma omp critical
        {
            writeChunk<T>(dsOut, slice, z, 0, 0);
        }

        progress.update();
    }

    progress.finish();
}

template<typename T>
static void runCLAHE(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                     std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running 3D CLAHE...");

    // For 3D CLAHE, we need to read the full volume
    // (or use chunked CLAHE with tile interpolation)
    xt::xarray<T> volume;
    readChunk<T>(dsIn, volume, 0, 0, 0, info.sz, info.sy, info.sx);

    vc::CLAHE3DParams params;
    params.tileSize = cfg.claheTileSize;
    params.clipLimit = cfg.claheClipLimit;
    params.numBins = cfg.claheNumBins;

    auto progressCb = [&](float p) {
        std::lock_guard<std::mutex> lock(g_printMtx);
        std::cout << "\r[CLAHE] " << static_cast<int>(p * 100) << "%     " << std::flush;
    };

    vc::clahe3D(volume, params, progressCb);

    writeChunk<T>(dsOut, volume, 0, 0, 0);

    std::lock_guard<std::mutex> lock(g_printMtx);
    std::cout << "\r[CLAHE] Complete                    " << "\n";
}

template<typename T>
static void runRingCorrect(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                           std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running ring artifact correction (legacy)...");

    vc::RingCorrectionParams params;
    params.centerX = cfg.ringCenterX;
    params.centerY = cfg.ringCenterY;
    params.medianWidth = cfg.ringMedianWidth;

    ProgressReporter progress("RingCorrect", info.sz, cfg.showEta);

    #pragma omp parallel for schedule(dynamic)
    for (size_t z = 0; z < info.sz; ++z) {
        // Read slice
        xt::xarray<T> slice;
        #pragma omp critical
        {
            readChunk<T>(dsIn, slice, z, 0, 0, 1, info.sy, info.sx);
        }

        // Convert to cv::Mat for processing
        cv::Mat_<T> sliceMat(static_cast<int>(info.sy), static_cast<int>(info.sx));
        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                sliceMat(static_cast<int>(y), static_cast<int>(x)) = slice(0, y, x);
            }
        }

        // Process
        vc::ringCorrectSlice(sliceMat, params);

        // Convert back
        for (size_t y = 0; y < info.sy; ++y) {
            for (size_t x = 0; x < info.sx; ++x) {
                slice(0, y, x) = sliceMat(static_cast<int>(y), static_cast<int>(x));
            }
        }

        // Write slice
        #pragma omp critical
        {
            writeChunk<T>(dsOut, slice, z, 0, 0);
        }

        progress.update();
    }

    progress.finish();
}

template<typename T>
static void runNormalize(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                         std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running intensity normalization...");

    // Read entire volume for normalization (global operation)
    // For very large volumes, this would need to be done in tiles
    xt::xarray<T> volume;
    readChunk<T>(dsIn, volume, 0, 0, 0, info.sz, info.sy, info.sx);

    vc::NormalizationParams params;
    params.polyOrder = cfg.normPolyOrder;
    params.targetMean = cfg.normTargetMean;

    vc::normalizeIntensity(volume, params);

    // Write back
    writeChunk<T>(dsOut, volume, 0, 0, 0);

    std::cout << "[Normalize] Complete" << "\n";
}

template<typename T>
static void runResample(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                        std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info,
                        const VolumeInfo& outInfo) {
    log(cfg, "Running Lanczos resampling...");

    vc::LanczosResampleParams params;
    params.scaleX = cfg.resampleScaleX;
    params.scaleY = cfg.resampleScaleY;
    params.scaleZ = cfg.resampleScaleZ;

    // Process in chunks
    size_t numChunksZ = (outInfo.sz + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksY = (outInfo.sy + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksX = (outInfo.sx + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t totalChunks = numChunksZ * numChunksY * numChunksX;

    ProgressReporter progress("Resample", totalChunks);

    // Lanczos-3 needs 6 samples in each direction
    int lanczosOverlap = 4;

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t cz = 0; cz < numChunksZ; ++cz) {
        for (size_t cy = 0; cy < numChunksY; ++cy) {
            for (size_t cx = 0; cx < numChunksX; ++cx) {
                // Output chunk bounds
                size_t oz0 = cz * cfg.chunkSize;
                size_t oy0 = cy * cfg.chunkSize;
                size_t ox0 = cx * cfg.chunkSize;
                size_t oz1 = std::min(oz0 + cfg.chunkSize, outInfo.sz);
                size_t oy1 = std::min(oy0 + cfg.chunkSize, outInfo.sy);
                size_t ox1 = std::min(ox0 + cfg.chunkSize, outInfo.sx);

                // Corresponding input region (with overlap for Lanczos)
                size_t iz0 = static_cast<size_t>(std::max(0.0f, static_cast<float>(oz0) / params.scaleZ - lanczosOverlap));
                size_t iy0 = static_cast<size_t>(std::max(0.0f, static_cast<float>(oy0) / params.scaleY - lanczosOverlap));
                size_t ix0 = static_cast<size_t>(std::max(0.0f, static_cast<float>(ox0) / params.scaleX - lanczosOverlap));
                size_t iz1 = std::min(static_cast<size_t>(static_cast<float>(oz1) / params.scaleZ + lanczosOverlap + 1), info.sz);
                size_t iy1 = std::min(static_cast<size_t>(static_cast<float>(oy1) / params.scaleY + lanczosOverlap + 1), info.sy);
                size_t ix1 = std::min(static_cast<size_t>(static_cast<float>(ox1) / params.scaleX + lanczosOverlap + 1), info.sx);

                // Read input chunk
                xt::xarray<T> chunkIn;
                #pragma omp critical
                {
                    readChunk<T>(dsIn, chunkIn, iz0, iy0, ix0, iz1 - iz0, iy1 - iy0, ix1 - ix0);
                }

                // Allocate output chunk
                size_t olz = oz1 - oz0;
                size_t oly = oy1 - oy0;
                size_t olx = ox1 - ox0;
                xt::xarray<T> chunkOut = xt::xarray<T>::from_shape({olz, oly, olx});

                // Resample
                vc::resampleChunkLanczos3D<T>(
                    chunkIn.data(),
                    static_cast<int>(iz1 - iz0),
                    static_cast<int>(iy1 - iy0),
                    static_cast<int>(ix1 - ix0),
                    chunkOut.data(),
                    static_cast<int>(olz),
                    static_cast<int>(oly),
                    static_cast<int>(olx),
                    params);

                // Write output chunk
                #pragma omp critical
                {
                    writeChunk<T>(dsOut, chunkOut, oz0, oy0, ox0);
                }

                progress.update();
            }
        }
    }

    progress.finish();
}

template<typename T>
static void runComputeGradients(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                                 std::unique_ptr<z5::Dataset>& dsGradOut, const VolumeInfo& info) {
    log(cfg, "Computing gradient volume...");

    vc::GradientParams params;
    params.sigma = cfg.gradientSigma;
    params.normalize = cfg.gradientNormalize;

    // Gradient kernel radius
    int gradOverlap = static_cast<int>(std::ceil(3.0f * params.sigma)) + 1;
    int effectiveOverlap = std::max(cfg.overlap, gradOverlap);

    size_t numChunksZ = (info.sz + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksY = (info.sy + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksX = (info.sx + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t totalChunks = numChunksZ * numChunksY * numChunksX;

    ProgressReporter progress("Gradients", totalChunks);

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t cz = 0; cz < numChunksZ; ++cz) {
        for (size_t cy = 0; cy < numChunksY; ++cy) {
            for (size_t cx = 0; cx < numChunksX; ++cx) {
                // Compute chunk bounds with overlap
                size_t z0 = cz * cfg.chunkSize;
                size_t y0 = cy * cfg.chunkSize;
                size_t x0 = cx * cfg.chunkSize;

                size_t z0o = (z0 >= static_cast<size_t>(effectiveOverlap)) ? z0 - effectiveOverlap : 0;
                size_t y0o = (y0 >= static_cast<size_t>(effectiveOverlap)) ? y0 - effectiveOverlap : 0;
                size_t x0o = (x0 >= static_cast<size_t>(effectiveOverlap)) ? x0 - effectiveOverlap : 0;

                size_t z1 = std::min(z0 + cfg.chunkSize, info.sz);
                size_t y1 = std::min(y0 + cfg.chunkSize, info.sy);
                size_t x1 = std::min(x0 + cfg.chunkSize, info.sx);

                size_t z1o = std::min(z1 + effectiveOverlap, info.sz);
                size_t y1o = std::min(y1 + effectiveOverlap, info.sy);
                size_t x1o = std::min(x1 + effectiveOverlap, info.sx);

                // Read chunk with overlap
                xt::xarray<T> chunkIn;
                #pragma omp critical
                {
                    readChunk<T>(dsIn, chunkIn, z0o, y0o, x0o, z1o - z0o, y1o - y0o, x1o - x0o);
                }

                // Compute gradients
                xt::xarray<float> gradX, gradY, gradZ;
                vc::computeGradientVolume(chunkIn, gradX, gradY, gradZ, params);

                // Extract core region and combine into 4D
                size_t coreZ0 = z0 - z0o;
                size_t coreY0 = y0 - y0o;
                size_t coreX0 = x0 - x0o;
                size_t coreLz = z1 - z0;
                size_t coreLy = y1 - y0;
                size_t coreLx = x1 - x0;

                // Write each gradient component separately
                // Gradient output is [3, Z, Y, X] where 3 is for dx, dy, dz
                xt::xarray<float> coreGx = xt::xarray<float>::from_shape({coreLz, coreLy, coreLx});
                xt::xarray<float> coreGy = xt::xarray<float>::from_shape({coreLz, coreLy, coreLx});
                xt::xarray<float> coreGz = xt::xarray<float>::from_shape({coreLz, coreLy, coreLx});

                for (size_t z = 0; z < coreLz; ++z) {
                    for (size_t y = 0; y < coreLy; ++y) {
                        for (size_t x = 0; x < coreLx; ++x) {
                            coreGx(z, y, x) = gradX(coreZ0 + z, coreY0 + y, coreX0 + x);
                            coreGy(z, y, x) = gradY(coreZ0 + z, coreY0 + y, coreX0 + x);
                            coreGz(z, y, x) = gradZ(coreZ0 + z, coreY0 + y, coreX0 + x);
                        }
                    }
                }

                // Write gradient chunks (channel-first layout [3, Z, Y, X])
                // We write to positions [0, z0, y0, x0], [1, z0, y0, x0], [2, z0, y0, x0]
                #pragma omp critical
                {
                    z5::types::ShapeType offX = {0, z0, y0, x0};
                    z5::types::ShapeType offY = {1, z0, y0, x0};
                    z5::types::ShapeType offZ = {2, z0, y0, x0};

                    // Reshape to include channel dimension for writing
                    xt::xarray<float> gxWrite = xt::xarray<float>::from_shape({1ul, coreLz, coreLy, coreLx});
                    xt::xarray<float> gyWrite = xt::xarray<float>::from_shape({1ul, coreLz, coreLy, coreLx});
                    xt::xarray<float> gzWrite = xt::xarray<float>::from_shape({1ul, coreLz, coreLy, coreLx});

                    for (size_t z = 0; z < coreLz; ++z) {
                        for (size_t y = 0; y < coreLy; ++y) {
                            for (size_t x = 0; x < coreLx; ++x) {
                                gxWrite(0, z, y, x) = coreGx(z, y, x);
                                gyWrite(0, z, y, x) = coreGy(z, y, x);
                                gzWrite(0, z, y, x) = coreGz(z, y, x);
                            }
                        }
                    }

                    writeSubarray3D(*dsGradOut, gxWrite, offX);
                    writeSubarray3D(*dsGradOut, gyWrite, offY);
                    writeSubarray3D(*dsGradOut, gzWrite, offZ);
                }

                progress.update();
            }
        }
    }

    progress.finish();
}

template<typename T>
static void runDiffuse(const Config& cfg, std::unique_ptr<z5::Dataset>& dsIn,
                       std::unique_ptr<z5::Dataset>& dsOut, const VolumeInfo& info) {
    log(cfg, "Running 3D anisotropic diffusion...");

    vc::DiffusionParams params;
    params.lambda = cfg.diffuseLambda;
    params.sigma = cfg.diffuseSigma;
    params.numSteps = cfg.diffuseSteps;
    params.stepSize = 1.0f / 6.0f;  // Maximum stable for 3D

    // Diffusion overlap based on sigma and iterations
    int diffOverlap = static_cast<int>(std::ceil(3.0f * params.sigma)) + 2;
    int effectiveOverlap = std::max(cfg.overlap, diffOverlap);

    size_t numChunksZ = (info.sz + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksY = (info.sy + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t numChunksX = (info.sx + cfg.chunkSize - 1) / cfg.chunkSize;
    size_t totalChunks = numChunksZ * numChunksY * numChunksX;

    ProgressReporter progress("Diffuse", totalChunks);

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (size_t cz = 0; cz < numChunksZ; ++cz) {
        for (size_t cy = 0; cy < numChunksY; ++cy) {
            for (size_t cx = 0; cx < numChunksX; ++cx) {
                // Compute chunk bounds with overlap
                size_t z0 = cz * cfg.chunkSize;
                size_t y0 = cy * cfg.chunkSize;
                size_t x0 = cx * cfg.chunkSize;

                size_t z0o = (z0 >= static_cast<size_t>(effectiveOverlap)) ? z0 - effectiveOverlap : 0;
                size_t y0o = (y0 >= static_cast<size_t>(effectiveOverlap)) ? y0 - effectiveOverlap : 0;
                size_t x0o = (x0 >= static_cast<size_t>(effectiveOverlap)) ? x0 - effectiveOverlap : 0;

                size_t z1 = std::min(z0 + cfg.chunkSize, info.sz);
                size_t y1 = std::min(y0 + cfg.chunkSize, info.sy);
                size_t x1 = std::min(x0 + cfg.chunkSize, info.sx);

                size_t z1o = std::min(z1 + effectiveOverlap, info.sz);
                size_t y1o = std::min(y1 + effectiveOverlap, info.sy);
                size_t x1o = std::min(x1 + effectiveOverlap, info.sx);

                // Read chunk with overlap
                xt::xarray<T> chunk;
                #pragma omp critical
                {
                    readChunk<T>(dsIn, chunk, z0o, y0o, x0o, z1o - z0o, y1o - y0o, x1o - x0o);
                }

                // Process (in-place)
                vc::anisotropicDiffusion3D(chunk, params, nullptr);

                // Extract core region
                size_t coreZ0 = z0 - z0o;
                size_t coreY0 = y0 - y0o;
                size_t coreX0 = x0 - x0o;
                size_t coreLz = z1 - z0;
                size_t coreLy = y1 - y0;
                size_t coreLx = x1 - x0;

                xt::xarray<T> core = xt::xarray<T>::from_shape({coreLz, coreLy, coreLx});
                for (size_t z = 0; z < coreLz; ++z) {
                    for (size_t y = 0; y < coreLy; ++y) {
                        for (size_t x = 0; x < coreLx; ++x) {
                            core(z, y, x) = chunk(coreZ0 + z, coreY0 + y, coreX0 + x);
                        }
                    }
                }

                // Write core region
                #pragma omp critical
                {
                    writeChunk<T>(dsOut, core, z0, y0, x0);
                }

                progress.update();
            }
        }
    }

    progress.finish();
}

// ============================================================================
// Pipeline orchestration
// ============================================================================

template<typename T>
static int runPipeline(const Config& cfg) {
    fs::path inputPath(cfg.inputPath);
    fs::path outputPath(cfg.outputPath);

    // Open input dataset
    std::unique_ptr<z5::Dataset> dsIn;

    // Detect dimension separator from .zarray
    auto getDimSep = [](const fs::path& zarrayPath) -> std::string {
        try {
            json j = json::parse(std::ifstream(zarrayPath.string()));
            if (j.contains("dimension_separator")) {
                return j["dimension_separator"].get<std::string>();
            }
        } catch (...) {
            // dimension_separator is optional; keep default '.'
        }
        return ".";
    };

    // Check if input is a dataset (.zarray) or group (.zgroup)
    if (fs::exists(inputPath / ".zarray")) {
        std::string dimSep = getDimSep(inputPath / ".zarray");
        z5::filesystem::handle::Group parent(inputPath.parent_path(), z5::FileMode::r);
        z5::filesystem::handle::Dataset dsHandle(parent, inputPath.filename().string(), dimSep);
        dsIn = z5::filesystem::openDataset(dsHandle);
    } else if (fs::exists(inputPath / "0" / ".zarray")) {
        // OME-Zarr with pyramid levels
        std::string dimSep = getDimSep(inputPath / "0" / ".zarray");
        z5::filesystem::handle::Group root(inputPath, z5::FileMode::r);
        z5::filesystem::handle::Dataset dsHandle(root, "0", dimSep);
        dsIn = z5::filesystem::openDataset(dsHandle);
    } else {
        std::cerr << "Error: Cannot find Zarr dataset at " << inputPath << "\n";
        return 1;
    }

    VolumeInfo fullInfo = getVolumeInfo(dsIn.get());
    std::cout << "Input volume: " << fullInfo.sx << " x " << fullInfo.sy << " x " << fullInfo.sz << "\n";

    // Preview mode: extract a subvolume
    VolumeInfo inInfo = fullInfo;
    size_t previewZ0 = 0, previewY0 = 0, previewX0 = 0;

    if (cfg.preview) {
        size_t pSize = static_cast<size_t>(cfg.previewSize);

        // Calculate preview region (centered if not specified)
        previewZ0 = (cfg.previewZ >= 0) ? static_cast<size_t>(cfg.previewZ)
                                         : (fullInfo.sz > pSize) ? (fullInfo.sz - pSize) / 2 : 0;
        previewY0 = (cfg.previewY >= 0) ? static_cast<size_t>(cfg.previewY)
                                         : (fullInfo.sy > pSize) ? (fullInfo.sy - pSize) / 2 : 0;
        previewX0 = (cfg.previewX >= 0) ? static_cast<size_t>(cfg.previewX)
                                         : (fullInfo.sx > pSize) ? (fullInfo.sx - pSize) / 2 : 0;

        // Clamp to volume bounds
        inInfo.sz = std::min(pSize, fullInfo.sz - previewZ0);
        inInfo.sy = std::min(pSize, fullInfo.sy - previewY0);
        inInfo.sx = std::min(pSize, fullInfo.sx - previewX0);

        std::cout << "Preview mode: processing subvolume " << inInfo.sx << " x " << inInfo.sy << " x " << inInfo.sz
                  << " at offset (" << previewX0 << ", " << previewY0 << ", " << previewZ0 << ")" << "\n";
    }

    // Compute output dimensions (may change if resampling)
    VolumeInfo outInfo = inInfo;
    if (cfg.resample) {
        outInfo.sx = static_cast<size_t>(std::round(static_cast<float>(inInfo.sx) * cfg.resampleScaleX));
        outInfo.sy = static_cast<size_t>(std::round(static_cast<float>(inInfo.sy) * cfg.resampleScaleY));
        outInfo.sz = static_cast<size_t>(std::round(static_cast<float>(inInfo.sz) * cfg.resampleScaleZ));
        std::cout << "Output volume (resampled): " << outInfo.sx << " x " << outInfo.sy << " x " << outInfo.sz << "\n";
    }

    // Create output dataset
    fs::create_directories(outputPath.parent_path());
    z5::filesystem::handle::File outputFile(outputPath, z5::FileMode::w);
    z5::createFile(outputFile, true);

    std::vector<size_t> outShape = {outInfo.sz, outInfo.sy, outInfo.sx};
    std::vector<size_t> outChunks = {
        std::min(static_cast<size_t>(cfg.chunkSize), outInfo.sz),
        std::min(static_cast<size_t>(cfg.chunkSize), outInfo.sy),
        std::min(static_cast<size_t>(cfg.chunkSize), outInfo.sx)
    };

    std::string dtypeStr;
    if constexpr (std::is_same_v<T, uint8_t>) dtypeStr = "uint8";
    else if constexpr (std::is_same_v<T, uint16_t>) dtypeStr = "uint16";
    else dtypeStr = "float32";

    json compOpts = {{"cname", "zstd"}, {"clevel", 1}, {"shuffle", 0}};
    auto dsOut = z5::createDataset(outputFile, "0", dtypeStr, outShape, outChunks, "blosc", compOpts);

    // Create gradient output dataset if needed
    std::unique_ptr<z5::Dataset> dsGradOut;
    if (cfg.computeGradients && !cfg.gradientOutputPath.empty()) {
        fs::path gradPath(cfg.gradientOutputPath);
        fs::create_directories(gradPath.parent_path());
        z5::filesystem::handle::File gradFile(gradPath, z5::FileMode::w);
        z5::createFile(gradFile, true);

        std::vector<size_t> gradShape = {3ul, outInfo.sz, outInfo.sy, outInfo.sx};
        std::vector<size_t> gradChunks = {1ul, outChunks[0], outChunks[1], outChunks[2]};
        dsGradOut = z5::createDataset(gradFile, "0", "float32", gradShape, gradChunks, "blosc", compOpts);
    }

    // Build list of enabled stages (in execution order)
    std::vector<std::string> stages;
    if (cfg.bm3d) stages.push_back("bm3d");
    if (cfg.denoise) stages.push_back("denoise");
    if (cfg.voRing) stages.push_back("vo_ring");
    if (cfg.ringCorrect) stages.push_back("ring_correct");
    if (cfg.stripeRemove) stages.push_back("stripe_remove");
    if (cfg.clahe) stages.push_back("clahe");
    if (cfg.normalize) stages.push_back("normalize");
    if (cfg.resample) stages.push_back("resample");
    if (cfg.computeGradients) stages.push_back("compute_gradients");
    if (cfg.diffuse) stages.push_back("diffuse");

    std::cout << "Enabled stages: ";
    for (const auto& s : stages) std::cout << s << " ";
    std::cout << "\n";

    // Track whether we've written to output yet (for pipeline chaining)
    bool outputWritten = false;
    VolumeInfo currentInfo = inInfo;

    // For preview mode, first copy the subvolume to output
    // This way all stages read from the output dataset at correct offsets
    if (cfg.preview && !stages.empty()) {
        std::cout << "Copying preview subvolume to output..." << "\n";
        xt::xarray<T> subvolume;
        readChunk<T>(dsIn, subvolume, previewZ0, previewY0, previewX0, inInfo.sz, inInfo.sy, inInfo.sx);
        writeChunk<T>(dsOut, subvolume, 0, 0, 0);
        outputWritten = true;
        std::cout << "Preview subvolume copied." << "\n";
    }

    // Run stages in order
    // Note: Each stage reads from dsIn (if first) or dsOut (if chained), writes to dsOut

    auto startTime = std::chrono::steady_clock::now();

    if (cfg.bm3d) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runBM3D<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runBM3D<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;

        if (cfg.preview) {
            auto stageEnd = std::chrono::steady_clock::now();
            double seconds = std::chrono::duration<double>(stageEnd - stageStart).count();
            g_estimator.recordStage("bm3d", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.denoise) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runDenoise<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runDenoise<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("denoise", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.voRing) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runVoRingCorrect<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runVoRingCorrect<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("vo_ring", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.ringCorrect) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runRingCorrect<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runRingCorrect<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("ring_correct", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.stripeRemove) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runStripeRemove<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runStripeRemove<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("stripe_remove", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.clahe) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runCLAHE<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runCLAHE<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("clahe", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.normalize) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runNormalize<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runNormalize<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("normalize", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.resample) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runResample<T>(cfg, dsIn, dsOut, currentInfo, outInfo);
        } else {
            runResample<T>(cfg, dsOut, dsOut, currentInfo, outInfo);
        }
        currentInfo = outInfo;
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("resample", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.computeGradients && dsGradOut) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runComputeGradients<T>(cfg, dsIn, dsGradOut, currentInfo);
        } else {
            runComputeGradients<T>(cfg, dsOut, dsGradOut, currentInfo);
        }
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("compute_gradients", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    if (cfg.diffuse) {
        auto stageStart = std::chrono::steady_clock::now();
        if (!outputWritten) {
            runDiffuse<T>(cfg, dsIn, dsOut, currentInfo);
        } else {
            runDiffuse<T>(cfg, dsOut, dsOut, currentInfo);
        }
        outputWritten = true;
        if (cfg.preview) {
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStart).count();
            g_estimator.recordStage("diffuse", currentInfo.sz * currentInfo.sy * currentInfo.sx, seconds);
        }
    }

    // Print runtime estimate for full volume (preview mode)
    if (cfg.preview && !stages.empty()) {
        g_estimator.printEstimate(fullInfo, stages);
    }

    // If no stages ran, just copy
    if (stages.empty()) {
        std::cout << "No stages enabled, copying input to output..." << "\n";
        size_t numChunksZ = (inInfo.sz + cfg.chunkSize - 1) / cfg.chunkSize;
        size_t numChunksY = (inInfo.sy + cfg.chunkSize - 1) / cfg.chunkSize;
        size_t numChunksX = (inInfo.sx + cfg.chunkSize - 1) / cfg.chunkSize;

        #pragma omp parallel for collapse(3) schedule(dynamic)
        for (size_t cz = 0; cz < numChunksZ; ++cz) {
            for (size_t cy = 0; cy < numChunksY; ++cy) {
                for (size_t cx = 0; cx < numChunksX; ++cx) {
                    size_t z0 = cz * cfg.chunkSize;
                    size_t y0 = cy * cfg.chunkSize;
                    size_t x0 = cx * cfg.chunkSize;
                    size_t lz = std::min(static_cast<size_t>(cfg.chunkSize), inInfo.sz - z0);
                    size_t ly = std::min(static_cast<size_t>(cfg.chunkSize), inInfo.sy - y0);
                    size_t lx = std::min(static_cast<size_t>(cfg.chunkSize), inInfo.sx - x0);

                    xt::xarray<T> chunk;
                    #pragma omp critical
                    {
                        readChunk<T>(dsIn, chunk, z0, y0, x0, lz, ly, lx);
                        writeChunk<T>(dsOut, chunk, z0, y0, x0);
                    }
                }
            }
        }
    }

    // Write metadata
    json attrs;
    attrs["vc_preprocess_volume"] = {
        {"version", "1.0.0"},
        {"input", cfg.inputPath},
        {"stages", stages},
        {"parameters", {
            {"denoise", {
                {"enabled", cfg.denoise},
                {"search_radius", cfg.denoiseSearchRadius},
                {"patch_radius", cfg.denoisePatchRadius},
                {"h", cfg.denoiseH}
            }},
            {"ring_correct", {
                {"enabled", cfg.ringCorrect},
                {"center_x", cfg.ringCenterX},
                {"center_y", cfg.ringCenterY},
                {"median_width", cfg.ringMedianWidth}
            }},
            {"normalize", {
                {"enabled", cfg.normalize},
                {"poly_order", cfg.normPolyOrder},
                {"target_mean", cfg.normTargetMean}
            }},
            {"resample", {
                {"enabled", cfg.resample},
                {"scale_x", cfg.resampleScaleX},
                {"scale_y", cfg.resampleScaleY},
                {"scale_z", cfg.resampleScaleZ}
            }},
            {"compute_gradients", {
                {"enabled", cfg.computeGradients},
                {"sigma", cfg.gradientSigma},
                {"normalize", cfg.gradientNormalize}
            }},
            {"diffuse", {
                {"enabled", cfg.diffuse},
                {"lambda", cfg.diffuseLambda},
                {"sigma", cfg.diffuseSigma},
                {"steps", cfg.diffuseSteps}
            }}
        }},
        {"timestamp", []() {
            auto now = std::chrono::system_clock::now();
            auto time = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::gmtime(&time), "%FT%TZ");
            return ss.str();
        }()}
    };
    z5::filesystem::writeAttributes(outputFile, attrs);

    std::cout << "Output saved to: " << outputPath << "\n";
    if (dsGradOut) {
        std::cout << "Gradients saved to: " << cfg.gradientOutputPath << "\n";
    }

    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Config cfg;

    po::options_description desc("vc_preprocess_volume - Volume preprocessing pipeline\n\nUsage");

    desc.add_options()
        ("help,h", "Show this help message")

        // Input/Output
        ("input,i", po::value<std::string>(&cfg.inputPath)->required(),
            "Input Zarr volume (required)")
        ("output,o", po::value<std::string>(&cfg.outputPath)->required(),
            "Output Zarr volume (required)")
        ("output-gradients", po::value<std::string>(&cfg.gradientOutputPath),
            "Output gradient volume (optional)")

        // Stage control (SOTA algorithms)
        ("bm3d", po::bool_switch(&cfg.bm3d),
            "Enable BM3D denoising (SOTA)")
        ("denoise", po::bool_switch(&cfg.denoise),
            "Enable NLM denoising (alternative to BM3D)")
        ("vo-ring", po::bool_switch(&cfg.voRing),
            "Enable Vo ring artifact removal (SOTA)")
        ("ring-correct", po::bool_switch(&cfg.ringCorrect),
            "Enable legacy ring artifact correction")
        ("stripe-remove", po::bool_switch(&cfg.stripeRemove),
            "Enable FFT stripe artifact removal")
        ("clahe", po::bool_switch(&cfg.clahe),
            "Enable 3D CLAHE contrast enhancement")
        ("normalize", po::bool_switch(&cfg.normalize),
            "Enable intensity normalization")
        ("resample", po::bool_switch(&cfg.resample),
            "Enable Lanczos resampling")
        ("compute-gradients", po::bool_switch(&cfg.computeGradients),
            "Compute gradient volume")
        ("diffuse", po::bool_switch(&cfg.diffuse),
            "Enable 3D anisotropic diffusion")

        // BM3D options
        ("bm3d-sigma", po::value<float>(&cfg.bm3dSigma)->default_value(25.0f),
            "Noise sigma estimate (default: 25)")
        ("bm3d-block-size", po::value<int>(&cfg.bm3dBlockSize)->default_value(8),
            "Block size for matching (default: 8)")
        ("bm3d-search-radius", po::value<int>(&cfg.bm3dSearchRadius)->default_value(16),
            "Search window radius (default: 16)")
        ("bm3d-max-matches", po::value<int>(&cfg.bm3dMaxMatches)->default_value(16),
            "Maximum matched blocks (default: 16)")
        ("bm3d-hard-threshold", po::value<float>(&cfg.bm3dHardThreshold)->default_value(2.7f),
            "Hard threshold multiplier (default: 2.7)")
        ("bm3d-wiener", po::value<bool>(&cfg.bm3dWiener)->default_value(true),
            "Enable Wiener filtering (default: true)")

        // NLM denoising options
        ("denoise-search-radius", po::value<int>(&cfg.denoiseSearchRadius)->default_value(10),
            "Search window radius (default: 10)")
        ("denoise-patch-radius", po::value<int>(&cfg.denoisePatchRadius)->default_value(3),
            "Patch half-size (default: 3)")
        ("denoise-h", po::value<float>(&cfg.denoiseH)->default_value(10.0f),
            "Filter strength (default: 10.0)")

        // Vo ring correction options
        ("vo-snr", po::value<int>(&cfg.voSnr)->default_value(3),
            "Signal-to-noise ratio for thresholding (default: 3)")
        ("vo-la", po::value<int>(&cfg.voLa)->default_value(81),
            "Mean filter size for large rings (default: 81)")
        ("vo-sm", po::value<int>(&cfg.voSm)->default_value(21),
            "Median filter size for small rings (default: 21)")
        ("vo-center-x", po::value<float>(&cfg.voCenterX)->default_value(-1.0f),
            "Rotation center X (default: auto)")
        ("vo-center-y", po::value<float>(&cfg.voCenterY)->default_value(-1.0f),
            "Rotation center Y (default: auto)")

        // Legacy ring correction options
        ("ring-center-x", po::value<float>(&cfg.ringCenterX)->default_value(-1.0f),
            "Polar center X (default: auto)")
        ("ring-center-y", po::value<float>(&cfg.ringCenterY)->default_value(-1.0f),
            "Polar center Y (default: auto)")
        ("ring-median-width", po::value<int>(&cfg.ringMedianWidth)->default_value(5),
            "Median filter width (default: 5)")

        // Stripe removal options
        ("stripe-sigma", po::value<float>(&cfg.stripeSigma)->default_value(3.0f),
            "Gaussian damping sigma in FFT (default: 3)")
        ("stripe-dec-num", po::value<int>(&cfg.stripeDecNum)->default_value(4),
            "Wavelet decomposition levels (default: 4)")
        ("stripe-wavelet", po::bool_switch(&cfg.stripeUseWavelet),
            "Use wavelet-FFT hybrid method")

        // 3D CLAHE options
        ("clahe-tile-size", po::value<int>(&cfg.claheTileSize)->default_value(64),
            "Size of tiles in each dimension (default: 64)")
        ("clahe-clip-limit", po::value<float>(&cfg.claheClipLimit)->default_value(4.0f),
            "Contrast limit (default: 4.0)")
        ("clahe-num-bins", po::value<int>(&cfg.claheNumBins)->default_value(256),
            "Number of histogram bins (default: 256)")

        // Normalization options
        ("norm-poly-order", po::value<int>(&cfg.normPolyOrder)->default_value(2),
            "Polynomial order (default: 2)")
        ("norm-target-mean", po::value<float>(&cfg.normTargetMean)->default_value(32768.0f),
            "Target mean (default: 32768)")

        // Resampling options
        ("resample-scale-x", po::value<float>(&cfg.resampleScaleX)->default_value(1.0f),
            "Scale factor X (default: 1.0)")
        ("resample-scale-y", po::value<float>(&cfg.resampleScaleY)->default_value(1.0f),
            "Scale factor Y (default: 1.0)")
        ("resample-scale-z", po::value<float>(&cfg.resampleScaleZ)->default_value(1.0f),
            "Scale factor Z (default: 1.0)")

        // Gradient options
        ("gradient-sigma", po::value<float>(&cfg.gradientSigma)->default_value(1.0f),
            "Pre-smoothing sigma (default: 1.0)")
        ("gradient-normalize", po::bool_switch(&cfg.gradientNormalize),
            "Normalize to unit vectors")

        // Diffusion options
        ("diffuse-lambda", po::value<float>(&cfg.diffuseLambda)->default_value(1.0f),
            "Edge threshold (default: 1.0)")
        ("diffuse-sigma", po::value<float>(&cfg.diffuseSigma)->default_value(3.0f),
            "Gradient smoothing (default: 3.0)")
        ("diffuse-steps", po::value<int>(&cfg.diffuseSteps)->default_value(50),
            "Iterations (default: 50)")

        // General options
        ("chunk-size", po::value<int>(&cfg.chunkSize)->default_value(128),
            "Processing chunk size (default: 128)")
        ("overlap", po::value<int>(&cfg.overlap)->default_value(16),
            "Chunk overlap (default: 16)")
        ("threads", po::value<int>(&cfg.numThreads)->default_value(0),
            "Thread count (default: auto)")
        ("cache-mb", po::value<size_t>(&cfg.cacheMB)->default_value(4096),
            "Cache size MB (default: 4096)")
        ("verbose,v", po::bool_switch(&cfg.verbose),
            "Verbose output")

        // Preview mode
        ("preview", po::bool_switch(&cfg.preview),
            "Preview mode: process small subvolume for parameter tuning")
        ("preview-size", po::value<int>(&cfg.previewSize)->default_value(256),
            "Preview subvolume size in each dimension (default: 256)")
        ("preview-z", po::value<int>(&cfg.previewZ)->default_value(-1),
            "Preview starting Z coordinate (-1 = center)")
        ("preview-y", po::value<int>(&cfg.previewY)->default_value(-1),
            "Preview starting Y coordinate (-1 = center)")
        ("preview-x", po::value<int>(&cfg.previewX)->default_value(-1),
            "Preview starting X coordinate (-1 = center)")

        // Runtime options
        ("no-eta", po::bool_switch()->notifier([&cfg](bool v) { cfg.showEta = !v; }),
            "Disable ETA display")
    ;

    po::positional_options_description pos;
    pos.add("input", 1);
    pos.add("output", 1);

    po::variables_map vm;

    try {
        po::store(po::command_line_parser(argc, argv)
            .options(desc)
            .positional(pos)
            .run(), vm);

        if (vm.count("help") || argc < 3) {
            std::cout << desc << "\n";
            std::cout << "\nExamples:" << "\n";
            std::cout << "  # SOTA preprocessing pipeline" << "\n";
            std::cout << "  vc_preprocess_volume input.zarr output.zarr --bm3d --vo-ring --clahe --diffuse" << "\n";
            std::cout << "\n";
            std::cout << "  # Preview mode for parameter tuning" << "\n";
            std::cout << "  vc_preprocess_volume input.zarr preview.zarr --preview --bm3d --bm3d-sigma 30" << "\n";
            std::cout << "\n";
            std::cout << "  # Ring and stripe removal for synchrotron data" << "\n";
            std::cout << "  vc_preprocess_volume input.zarr output.zarr --vo-ring --stripe-remove" << "\n";
            std::cout << "\n";
            std::cout << "  # Resample to half resolution" << "\n";
            std::cout << "  vc_preprocess_volume vol.zarr vol_half.zarr --resample --resample-scale-x 0.5 --resample-scale-y 0.5 --resample-scale-z 0.5" << "\n";
            std::cout << "\n";
            std::cout << "  # Compute gradients for downstream processing" << "\n";
            std::cout << "  vc_preprocess_volume vol.zarr vol_proc.zarr --compute-gradients --output-gradients grad.zarr" << "\n";
            return 0;
        }

        po::notify(vm);

    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Use --help for usage information." << "\n";
        return 1;
    }

    // Set thread count
    #ifdef _OPENMP
    if (cfg.numThreads > 0) {
        omp_set_num_threads(cfg.numThreads);
    }
    std::cout << "Using " << omp_get_max_threads() << " threads" << "\n";
    #endif

    // Detect dtype from input
    fs::path inputPath(cfg.inputPath);
    z5::types::Datatype dtype;

    auto getDimSepLocal = [](const fs::path& zarrayPath) -> std::string {
        try {
            json j = json::parse(std::ifstream(zarrayPath.string()));
            if (j.contains("dimension_separator")) {
                return j["dimension_separator"].get<std::string>();
            }
        } catch (...) {
            // dimension_separator is optional; keep default '.'
        }
        return ".";
    };

    try {
        if (fs::exists(inputPath / ".zarray")) {
            std::string dimSep = getDimSepLocal(inputPath / ".zarray");
            z5::filesystem::handle::Group parent(inputPath.parent_path(), z5::FileMode::r);
            z5::filesystem::handle::Dataset dsHandle(parent, inputPath.filename().string(), dimSep);
            auto ds = z5::filesystem::openDataset(dsHandle);
            dtype = ds->getDtype();
        } else if (fs::exists(inputPath / "0" / ".zarray")) {
            std::string dimSep = getDimSepLocal(inputPath / "0" / ".zarray");
            z5::filesystem::handle::Group root(inputPath, z5::FileMode::r);
            z5::filesystem::handle::Dataset dsHandle(root, "0", dimSep);
            auto ds = z5::filesystem::openDataset(dsHandle);
            dtype = ds->getDtype();
        } else {
            std::cerr << "Error: Cannot find Zarr dataset at " << inputPath << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error opening input: " << e.what() << "\n";
        return 1;
    }

    // Run pipeline with appropriate type
    try {
        if (dtype == z5::types::Datatype::uint8) {
            return runPipeline<uint8_t>(cfg);
        } else if (dtype == z5::types::Datatype::uint16) {
            return runPipeline<uint16_t>(cfg);
        } else if (dtype == z5::types::Datatype::float32) {
            return runPipeline<float>(cfg);
        } else {
            std::cerr << "Error: Unsupported data type. Only uint8, uint16, and float32 are supported." << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
