#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/ABFFlattening.hpp"
#include "vc/core/zarr/ZarrDataset.hpp"
#include "vc/core/zarr/Tensor3D.hpp"
#include "vc/core/csvs/CsvsDataset.hpp"
#include <omp.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/program_options.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <cmath>
#include <set>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <queue>
#include <deque>

namespace po = boost::program_options;
using json = nlohmann::json;

// ============================================================================
// Type Traits
// ============================================================================

template<typename T> struct PixelTraits;
template<> struct PixelTraits<uint8_t> {
    static constexpr int cvType = CV_8UC1;
    static constexpr volcart::zarr::Dtype dtype = volcart::zarr::Dtype::UInt8;
};
template<> struct PixelTraits<uint16_t> {
    static constexpr int cvType = CV_16UC1;
    static constexpr volcart::zarr::Dtype dtype = volcart::zarr::Dtype::UInt16;
};

enum class AccumType { Max, Mean, Median };

// ============================================================================
// Simple Parameter Structs
// ============================================================================

struct CropParams {
    int x = 0, y = 0, width = 0, height = 0;

    cv::Rect toRect(const cv::Size& canvas) const {
        if (width <= 0 || height <= 0)
            return {0, 0, canvas.width, canvas.height};
        cv::Rect req{x, y, width, height};
        return req & cv::Rect{0, 0, canvas.width, canvas.height};
    }
};

struct AccumParams {
    std::vector<float> offsets;
    AccumType type = AccumType::Max;
    double step = 0.0;
};

struct AffineTransform {
    cv::Mat_<double> matrix = cv::Mat_<double>::eye(4, 4);

    cv::Matx33d linearPart() const {
        return cv::Matx33d(
            matrix(0,0), matrix(0,1), matrix(0,2),
            matrix(1,0), matrix(1,1), matrix(1,2),
            matrix(2,0), matrix(2,1), matrix(2,2));
    }
};

struct TransformParams {
    int rotQuad = -1;
    int flipAxis = -1;
    bool hasAffine = false;
    AffineTransform affine;
};

struct RenderParams {
    float renderScale;
    float scaleSeg;
    float dsScale;
    double sliceStep;
    bool nearestNeighbor;
    TransformParams transform;
    AccumParams accum;
    int startTileRow = 0;
    int endTileRow = -1;  // -1 means all rows
    int numParts = 1;
};

// ============================================================================
// Work Distribution: Row-based with work stealing
// ============================================================================

struct RowWorkQueue {
    std::mutex mutex;
    uint32_t nextRow = 0;
    uint32_t totalRows = 0;

    void reset(uint32_t rows) {
        std::lock_guard<std::mutex> lock(mutex);
        nextRow = 0;
        totalRows = rows;
    }

    // Get next row to process, returns -1 if done
    int getNextRow() {
        std::lock_guard<std::mutex> lock(mutex);
        if (nextRow >= totalRows) return -1;
        return nextRow++;
    }
};

// ============================================================================
// Async Write Queue for overlapping I/O with compute
// ============================================================================

template<typename T>
struct AsyncWriteQueue {
    struct WriteJob {
        volcart::zarr::Tensor3D<T> data;
        std::vector<size_t> offset;
        volcart::zarr::ZarrDataset* ds;
    };

    std::queue<WriteJob> jobs;
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cvFull;
    std::thread writerThread;
    bool done = false;
    size_t maxQueueSize = 16;  // Buffer up to 16 tiles

    void start() {
        writerThread = std::thread([this]() {
            while (true) {
                WriteJob job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this]() { return !jobs.empty() || done; });
                    if (done && jobs.empty()) break;
                    job = std::move(jobs.front());
                    jobs.pop();
                    cvFull.notify_one();
                }
                job.ds->writeSubarray(job.data, job.offset);
            }
        });
    }

    void enqueue(volcart::zarr::ZarrDataset* ds, volcart::zarr::Tensor3D<T>&& data,
                 std::vector<size_t> offset) {
        std::unique_lock<std::mutex> lock(mutex);
        cvFull.wait(lock, [this]() { return jobs.size() < maxQueueSize; });
        jobs.push({std::move(data), std::move(offset), ds});
        cv.notify_one();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            done = true;
        }
        cv.notify_one();
        if (writerThread.joinable()) writerThread.join();
    }
};

// ============================================================================
// Small Utilities
// ============================================================================

static std::string formatDuration(double seconds) {
    int s = static_cast<int>(seconds);
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", s/3600, (s%3600)/60, s%60);
    return buf;
}

static inline bool isNaN3f(const cv::Vec3f& v) {
    return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]);
}

static inline float trilinear8(float c000, float c001, float c010, float c011,
                               float c100, float c101, float c110, float c111,
                               float fx, float fy, float fz) {
    float c00 = c000 + fx * (c001 - c000);
    float c01 = c010 + fx * (c011 - c010);
    float c10 = c100 + fx * (c101 - c100);
    float c11 = c110 + fx * (c111 - c110);
    float c0 = c00 + fy * (c01 - c00);
    float c1 = c10 + fy * (c11 - c10);
    return c0 + fz * (c1 - c0);
}

static json defaultBloscOpts() {
    return {{"cname", "zstd"}, {"clevel", 1}, {"shuffle", 0}};
}

static std::filesystem::path numberedPath(const std::filesystem::path& dir,
                                          int index, int maxIndex,
                                          const std::string& ext = ".tif") {
    int pad = 2, v = maxIndex;
    while (v >= 100) { pad++; v /= 10; }
    std::ostringstream fn;
    fn << std::setw(pad) << std::setfill('0') << index << ext;
    return dir / fn.str();
}

static bool allFilesExist(const std::filesystem::path& dir, int count, int maxIndex,
                          const std::string& ext = ".tif") {
    for (int i = 0; i < count; ++i)
        if (!std::filesystem::exists(numberedPath(dir, i, maxIndex, ext)))
            return false;
    return true;
}

static int normalizeQuadrantRotation(double angleDeg, double tolDeg = 0.5) {
    double a = std::fmod(angleDeg, 360.0);
    if (a < 0) a += 360.0;
    static const double q[4] = {0.0, 90.0, 180.0, 270.0};
    int best = 0;
    double bestDiff = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        double d = std::abs(a - q[i]);
        if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return (bestDiff <= tolDeg) ? best : -1;
}

static void rotateFlipIfNeeded(cv::Mat& m, int rotQuad, int flipAxis) {
    if (rotQuad == 1)      cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (rotQuad == 2) cv::rotate(m, m, cv::ROTATE_180);
    else if (rotQuad == 3) cv::rotate(m, m, cv::ROTATE_90_CLOCKWISE);
    if (flipAxis >= 0 && flipAxis <= 2)
        cv::flip(m, m, flipAxis == 2 ? -1 : flipAxis);
}

static void mapTileIndex(int tx, int ty, int tilesX, int tilesY,
                         int quadRot, int flipType,
                         int& outTx, int& outTy, int& outTilesX, int& outTilesY) {
    const bool swap = (quadRot % 2) == 1;
    outTilesX = swap ? tilesY : tilesX;
    outTilesY = swap ? tilesX : tilesY;

    int rx = tx, ry = ty;
    switch (quadRot) {
        case 1: rx = ty; ry = tilesX - 1 - tx; break;
        case 2: rx = tilesX - 1 - tx; ry = tilesY - 1 - ty; break;
        case 3: rx = tilesY - 1 - ty; ry = tx; break;
        default: break;
    }

    outTx = rx; outTy = ry;
    if (flipType == 0) outTy = outTilesY - 1 - ry;
    else if (flipType == 1) outTx = outTilesX - 1 - rx;
    else if (flipType == 2) { outTx = outTilesX - 1 - rx; outTy = outTilesY - 1 - ry; }
}

// Thread-safe progress tracking
struct ProgressTracker {
    std::atomic<size_t> done{0};
    size_t total = 0;
    std::chrono::steady_clock::time_point startTime;
    const char* label = "";
    std::mutex printMutex;

    void reset(size_t totalCount, const char* lbl) {
        done = 0;
        total = totalCount;
        label = lbl;
        startTime = std::chrono::steady_clock::now();
    }

    void increment(size_t count = 1) {
        size_t d = done.fetch_add(count) + count;
        // Only print every 10 tiles or at completion
        if (d % 10 != 0 && d != total) return;

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - startTime).count();
        double eta = (d > 0) ? elapsed * (double(total) / d - 1.0) : 0.0;

        std::lock_guard<std::mutex> lock(printMutex);
        std::cout << "\r[" << label << "] " << d << "/" << total
                  << " (" << int(100.0*d/total) << "%)  elapsed " << formatDuration(elapsed)
                  << "  eta " << formatDuration(eta) << "  " << std::flush;
    }

    void finish() {
        std::cout << std::endl;
    }
};

// ============================================================================
// Affine Transform Functions
// ============================================================================

static bool invertAffineInPlace(AffineTransform& T) {
    cv::Mat Ainv;
    if (cv::invert(cv::Mat(T.linearPart()), Ainv, cv::DECOMP_LU) < 1e-10)
        return false;
    cv::Matx33d Ai(Ainv.at<double>(0,0), Ainv.at<double>(0,1), Ainv.at<double>(0,2),
                   Ainv.at<double>(1,0), Ainv.at<double>(1,1), Ainv.at<double>(1,2),
                   Ainv.at<double>(2,0), Ainv.at<double>(2,1), Ainv.at<double>(2,2));
    cv::Vec3d t(T.matrix(0,3), T.matrix(1,3), T.matrix(2,3));
    cv::Vec3d ti = -(Ai * t);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.matrix(r,c) = Ai(r,c);
        T.matrix(r,3) = ti(r);
    }
    return true;
}

static AffineTransform loadAffineTransform(const std::string& filename) {
    AffineTransform transform;
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open affine: " + filename);

    json j; file >> j;
    if (j.contains("transformation_matrix")) {
        auto mat = j["transformation_matrix"];
        if (mat.size() != 3 && mat.size() != 4)
            throw std::runtime_error("Affine matrix must have 3 or 4 rows");
        for (int r = 0; r < (int)mat.size(); ++r) {
            if (mat[r].size() != 4)
                throw std::runtime_error("Each row must have 4 elements");
            for (int c = 0; c < 4; ++c)
                transform.matrix(r,c) = mat[r][c].get<double>();
        }
    }
    return transform;
}

static AffineTransform composeAffine(const AffineTransform& A, const AffineTransform& B) {
    AffineTransform R;
    cv::Mat tmp = B.matrix * A.matrix;
    tmp.copyTo(R.matrix);
    return R;
}

static std::pair<std::string, bool> parseAffineSpec(const std::string& spec) {
    for (const char* suf : {":inv", ":invert", ":i"}) {
        size_t len = strlen(suf);
        if (spec.size() > len && spec.substr(spec.size() - len) == suf)
            return {spec.substr(0, spec.size() - len), true};
    }
    return {spec, false};
}

static bool loadAndComposeAffines(const std::vector<std::string>& specs,
                                  AffineTransform& out, bool& hasAffine) {
    hasAffine = false;
    if (specs.empty()) return true;

    AffineTransform composed;
    for (size_t k = 0; k < specs.size(); ++k) {
        auto [path, invert] = parseAffineSpec(specs[k]);
        try {
            AffineTransform T = loadAffineTransform(path);
            std::cout << "Loaded affine[" << k << "]: " << path
                      << (invert ? " (invert)" : "") << std::endl;
            if (invert && !invertAffineInPlace(T)) {
                std::cerr << "Error: affine[" << k << "] non-invertible\n";
                return false;
            }
            composed = composeAffine(composed, T);
        } catch (const std::exception& e) {
            std::cerr << "Error loading affine[" << k << "]: " << e.what() << "\n";
            return false;
        }
    }
    out = composed;
    hasAffine = true;
    return true;
}

static void applyAffineTransform(cv::Mat_<cv::Vec3f>& pts, cv::Mat_<cv::Vec3f>& norms,
                                 const AffineTransform& T) {
    const cv::Matx33d A = T.linearPart();
    const cv::Matx33d invAT = A.inv().t();
    const cv::Vec3d t(T.matrix(0,3), T.matrix(1,3), T.matrix(2,3));

    const double a00 = A(0,0), a01 = A(0,1), a02 = A(0,2);
    const double a10 = A(1,0), a11 = A(1,1), a12 = A(1,2);
    const double a20 = A(2,0), a21 = A(2,1), a22 = A(2,2);
    const double t0 = t[0], t1 = t[1], t2 = t[2];

    const double n00 = invAT(0,0), n01 = invAT(0,1), n02 = invAT(0,2);
    const double n10 = invAT(1,0), n11 = invAT(1,1), n12 = invAT(1,2);
    const double n20 = invAT(2,0), n21 = invAT(2,1), n22 = invAT(2,2);

    for (int y = 0; y < pts.rows; ++y) {
        cv::Vec3f* pRow = pts.ptr<cv::Vec3f>(y);
        cv::Vec3f* nRow = norms.ptr<cv::Vec3f>(y);

        for (int x = 0; x < pts.cols; ++x) {
            cv::Vec3f& p = pRow[x];
            cv::Vec3f& n = nRow[x];

            if (!isNaN3f(p)) {
                float px = p[0], py = p[1], pz = p[2];
                p[0] = static_cast<float>(a00*px + a01*py + a02*pz + t0);
                p[1] = static_cast<float>(a10*px + a11*py + a12*pz + t1);
                p[2] = static_cast<float>(a20*px + a21*py + a22*pz + t2);
            }

            if (!isNaN3f(n)) {
                float nx = n[0], ny = n[1], nz = n[2];
                float rx = static_cast<float>(n00*nx + n01*ny + n02*nz);
                float ry = static_cast<float>(n10*nx + n11*ny + n12*nz);
                float rz = static_cast<float>(n20*nx + n21*ny + n22*nz);
                float len = std::sqrt(rx*rx + ry*ry + rz*rz);
                if (len > 0) { rx /= len; ry /= len; rz /= len; }
                n[0] = rx; n[1] = ry; n[2] = rz;
            }
        }
    }
}

// ============================================================================
// Surface Preparation
// ============================================================================

static bool computeGlobalFlipDecision(QuadSurface* surf, int dx, int dy,
                                      float u0, float v0, float renderScale, float scaleSeg,
                                      const TransformParams& transform, cv::Vec3f& centroid) {
    cv::Mat_<cv::Vec3f> tp, tn;
    surf->gen(&tp, &tn, cv::Size(dx, dy), cv::Vec3f(0,0,0), renderScale, cv::Vec3f(0, v0, u0));
    tp *= scaleSeg;
    if (transform.hasAffine)
        applyAffineTransform(tp, tn, transform.affine);

    centroid = cv::Vec3f(0,0,0);
    int count = 0;
    for (int y = 0; y < tp.rows; ++y)
        for (int x = 0; x < tp.cols; ++x)
            if (!isNaN3f(tp(y,x))) { centroid += tp(y,x); count++; }
    if (count > 0) centroid /= float(count);

    size_t toward = 0, away = 0;
    for (int y = 0; y < tp.rows; ++y)
        for (int x = 0; x < tp.cols; ++x) {
            if (isNaN3f(tp(y,x)) || isNaN3f(tn(y,x))) continue;
            if ((centroid - tp(y,x)).dot(tn(y,x)) > 0) toward++; else away++;
        }
    return away > toward;
}

static void prepareBasePointsAndStepDirs(const cv::Mat_<cv::Vec3f>& pts,
                                         const cv::Mat_<cv::Vec3f>& norms,
                                         float scaleSeg, float dsScale,
                                         const TransformParams& transform,
                                         bool flipNormals,
                                         cv::Mat_<cv::Vec3f>& baseOut,
                                         cv::Mat_<cv::Vec3f>& stepOut) {
    baseOut = pts.clone();
    baseOut *= scaleSeg;
    stepOut = norms.clone();
    if (transform.hasAffine)
        applyAffineTransform(baseOut, stepOut, transform.affine);

    const float sign = flipNormals ? -1.0f : 1.0f;
    const float combinedScale = dsScale;

    for (int y = 0; y < stepOut.rows; ++y) {
        cv::Vec3f* baseRow = baseOut.ptr<cv::Vec3f>(y);
        cv::Vec3f* stepRow = stepOut.ptr<cv::Vec3f>(y);

        for (int x = 0; x < stepOut.cols; ++x) {
            baseRow[x] *= combinedScale;

            cv::Vec3f& v = stepRow[x];
            if (!isNaN3f(v)) {
                v *= sign;
                float L = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                if (L > 0) {
                    float invL = 1.0f / L;
                    v[0] *= invL; v[1] *= invL; v[2] *= invL;
                }
            }
        }
    }
}

static void computeCanvasOrigin(const cv::Size& size, float& u0, float& v0) {
    u0 = -0.5f * (size.width - 1.0f);
    v0 = -0.5f * (size.height - 1.0f);
}

// ============================================================================
// Pre-computed Offset Tables
// ============================================================================

struct OffsetTable {
    std::vector<float> allOffsets;
    std::vector<size_t> sliceStartIdx;
    size_t samplesPerSlice;
    size_t numSlices;
    double sliceCenter;

    // Pre-scaled offsets for direct use
    std::vector<float> scaledOffsets;

    void build(size_t nSlices, double sliceStep, const std::vector<float>& accumOffsets) {
        numSlices = nSlices;
        sliceCenter = 0.5 * (double(nSlices) - 1.0);
        samplesPerSlice = accumOffsets.empty() ? 1 : accumOffsets.size();

        allOffsets.clear();
        sliceStartIdx.clear();
        allOffsets.reserve(nSlices * samplesPerSlice);
        sliceStartIdx.reserve(nSlices);

        for (size_t zi = 0; zi < nSlices; ++zi) {
            sliceStartIdx.push_back(allOffsets.size());
            float off = static_cast<float>((double(zi) - sliceCenter) * sliceStep);
            if (accumOffsets.empty()) {
                allOffsets.push_back(off);
            } else {
                for (float ao : accumOffsets)
                    allOffsets.push_back(off + ao);
            }
        }
    }

    void preScale(float dsScale) {
        scaledOffsets.resize(allOffsets.size());
        for (size_t i = 0; i < allOffsets.size(); ++i) {
            scaledOffsets[i] = allOffsets[i] * dsScale;
        }
    }
};

// ============================================================================
// Accumulation Helpers
// ============================================================================

template<typename T>
static void computeMedianFromSamples(const std::vector<cv::Mat>& samples, cv::Mat& out) {
    if (samples.empty()) { out.release(); return; }
    const int rows = samples[0].rows, cols = samples[0].cols;
    out.create(rows, cols, samples[0].type());
    std::vector<T> vals(samples.size());
    for (int y = 0; y < rows; ++y) {
        T* outRow = out.ptr<T>(y);
        for (int x = 0; x < cols; ++x) {
            for (size_t i = 0; i < samples.size(); ++i)
                vals[i] = samples[i].ptr<T>(y)[x];
            std::nth_element(vals.begin(), vals.begin() + vals.size()/2, vals.end());
            outRow[x] = (vals.size() % 2) ? vals[vals.size()/2]
                : static_cast<T>((uint32_t(vals[vals.size()/2-1]) + vals[vals.size()/2] + 1) / 2);
        }
    }
}

template<typename T>
static void accumulateSlices(cv::Mat& out, std::vector<cv::Mat>& bulk,
                             size_t start, size_t count, AccumType type) {
    if (count == 1) {
        out = bulk[start];
        return;
    }

    if (type == AccumType::Max) {
        out = bulk[start];
        for (size_t i = 1; i < count; ++i)
            cv::max(out, bulk[start+i], out);
    } else if (type == AccumType::Mean) {
        cv::Mat sum;
        bulk[start].convertTo(sum, CV_32F);
        for (size_t i = 1; i < count; ++i) {
            cv::Mat tmp;
            bulk[start+i].convertTo(tmp, CV_32F);
            sum += tmp;
        }
        sum *= (1.0f / count);
        sum.convertTo(out, PixelTraits<T>::cvType);
    } else {
        std::vector<cv::Mat> samples(bulk.begin()+start, bulk.begin()+start+count);
        computeMedianFromSamples<T>(samples, out);
    }
}

// ============================================================================
// Optimized Bulk Slice Rendering
// ============================================================================

template<typename T>
static void renderSlicesBulkOptimized(
    std::vector<cv::Mat>& out,
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& scaledOffsets,  // Pre-scaled!
    bool nearest)
{
    constexpr int cvType = PixelTraits<T>::cvType;
    const int rows = basePoints.rows, cols = basePoints.cols;
    const size_t numOffsets = scaledOffsets.size();

    if (numOffsets == 0 || rows == 0 || cols == 0) return;

    const int czShift = cache->chunkShiftZ(), cyShift = cache->chunkShiftY(), cxShift = cache->chunkShiftX();
    const int czMask = cache->chunkMaskZ(), cyMask = cache->chunkMaskY(), cxMask = cache->chunkMaskX();
    const int dsZ = cache->datasetSizeZ(), dsY = cache->datasetSizeY(), dsX = cache->datasetSizeX();

    // Pre-allocate output matrices
    out.resize(numOffsets);
    for (size_t i = 0; i < numOffsets; ++i) {
        out[i].create(rows, cols, cvType);
    }

    // Prepare output row pointers for all slices - avoids repeated ptr() calls
    std::vector<std::vector<T*>> outRowPtrs(numOffsets);
    for (size_t si = 0; si < numOffsets; ++si) {
        outRowPtrs[si].resize(rows);
        for (int r = 0; r < rows; ++r) {
            outRowPtrs[si][r] = out[si].ptr<T>(r);
        }
    }

    // Process row by row for cache locality
    for (int r = 0; r < rows; ++r) {
        const cv::Vec3f* baseRow = basePoints.ptr<cv::Vec3f>(r);
        const cv::Vec3f* stepRow = stepDirs.ptr<cv::Vec3f>(r);

        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& p = baseRow[c];
            const cv::Vec3f& d = stepRow[c];

            if (isNaN3f(p) || isNaN3f(d)) {
                for (size_t si = 0; si < numOffsets; ++si)
                    outRowPtrs[si][r][c] = 0;
                continue;
            }

            // Cache the last chunk pointer for this pixel's ray
            int lastCiz = -1, lastCiy = -1, lastCix = -1;
            const volcart::zarr::Tensor3D<T>* lastChunk = nullptr;

            for (size_t si = 0; si < numOffsets; ++si) {
                const float scale = scaledOffsets[si];
                float cx = p[0] + scale * d[0];
                float cy = p[1] + scale * d[1];
                float cz = p[2] + scale * d[2];

                if (nearest) {
                    int iz = static_cast<int>(cz + 0.5f);
                    int iy = static_cast<int>(cy + 0.5f);
                    int ix = static_cast<int>(cx + 0.5f);

                    if (iz < 0 || iz >= dsZ || iy < 0 || iy >= dsY || ix < 0 || ix >= dsX) {
                        outRowPtrs[si][r][c] = 0;
                        continue;
                    }

                    int ciz = iz >> czShift, ciy = iy >> cyShift, cix = ix >> cxShift;

                    if (ciz != lastCiz || ciy != lastCiy || cix != lastCix) {
                        lastChunk = cache->getRawFast(ciz, ciy, cix);
                        if (!lastChunk) {
                            auto owner = cache->get(ciz, ciy, cix);
                            lastChunk = owner.get();
                        }
                        lastCiz = ciz; lastCiy = ciy; lastCix = cix;
                    }

                    outRowPtrs[si][r][c] = lastChunk ? (*lastChunk)(iz & czMask, iy & cyMask, ix & cxMask) : 0;
                } else {
                    int iz = static_cast<int>(cz);
                    int iy = static_cast<int>(cy);
                    int ix = static_cast<int>(cx);

                    if (iz < 0 || iy < 0 || ix < 0 ||
                        iz + 1 >= dsZ || iy + 1 >= dsY || ix + 1 >= dsX) {
                        outRowPtrs[si][r][c] = 0;
                        continue;
                    }

                    float fz = cz - iz, fy = cy - iy, fx = cx - ix;
                    int ciz = iz >> czShift, ciy = iy >> cyShift, cix = ix >> cxShift;

                    if (ciz == (iz+1) >> czShift && ciy == (iy+1) >> cyShift && cix == (ix+1) >> cxShift) {
                        if (ciz != lastCiz || ciy != lastCiy || cix != lastCix) {
                            lastChunk = cache->getRawFast(ciz, ciy, cix);
                            if (!lastChunk) {
                                auto owner = cache->get(ciz, ciy, cix);
                                lastChunk = owner.get();
                            }
                            lastCiz = ciz; lastCiy = ciy; lastCix = cix;
                        }

                        if (!lastChunk) {
                            outRowPtrs[si][r][c] = 0;
                            continue;
                        }

                        const auto& ch = *lastChunk;
                        int lz = iz & czMask, ly = iy & cyMask, lx = ix & cxMask;

                        float val = trilinear8(
                            ch(lz,ly,lx), ch(lz,ly,lx+1), ch(lz,ly+1,lx), ch(lz,ly+1,lx+1),
                            ch(lz+1,ly,lx), ch(lz+1,ly,lx+1), ch(lz+1,ly+1,lx), ch(lz+1,ly+1,lx+1),
                            fx, fy, fz);
                        outRowPtrs[si][r][c] = static_cast<T>(std::min(val + 0.5f, float(std::numeric_limits<T>::max())));
                    } else {
                        auto get = [&](int vz, int vy, int vx) -> float {
                            auto raw = cache->getRawFast(vz >> czShift, vy >> cyShift, vx >> cxShift);
                            if (!raw) {
                                auto owner = cache->get(vz >> czShift, vy >> cyShift, vx >> cxShift);
                                raw = owner.get();
                            }
                            return raw ? float((*raw)(vz & czMask, vy & cyMask, vx & cxMask)) : 0;
                        };
                        float val = trilinear8(
                            get(iz,iy,ix), get(iz,iy,ix+1), get(iz,iy+1,ix), get(iz,iy+1,ix+1),
                            get(iz+1,iy,ix), get(iz+1,iy,ix+1), get(iz+1,iy+1,ix), get(iz+1,iy+1,ix+1),
                            fx, fy, fz);
                        outRowPtrs[si][r][c] = static_cast<T>(std::min(val + 0.5f, float(std::numeric_limits<T>::max())));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Unified Slice Rendering with Callback
// ============================================================================

template<typename T, typename SliceCallback>
static void renderSlicesWithCallback(
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    bool nearestNeighbor,
    const OffsetTable& offsets,
    const AccumParams& accum,
    int rotQuad,
    int flipAxis,
    SliceCallback&& onSlice)
{
    // Bulk render all samples using pre-scaled offsets
    std::vector<cv::Mat> bulk;
    renderSlicesBulkOptimized<T>(bulk, cache, basePoints, stepDirs,
                                  offsets.scaledOffsets, nearestNeighbor);

    // Process each output slice
    for (size_t zi = 0; zi < offsets.numSlices; ++zi) {
        cv::Mat sliceOut;
        accumulateSlices<T>(sliceOut, bulk, offsets.sliceStartIdx[zi],
                           offsets.samplesPerSlice, accum.type);

        if (rotQuad >= 0 || flipAxis >= 0)
            rotateFlipIfNeeded(sliceOut, rotQuad, flipAxis);

        onSlice(zi, sliceOut);
    }
}

// ============================================================================
// Row-Based Tiled Rendering Loop
// ============================================================================

template<typename T, typename TileCallback>
static void renderTiledRowBased(
    QuadSurface* surf,
    ChunkCache<T>* cache,
    const cv::Size& tgtSize,
    const cv::Size& fullSize,
    const cv::Rect& crop,
    const RenderParams& params,
    const OffsetTable& offsets,
    uint32_t tileW, uint32_t tileH,
    const char* label,
    TileCallback&& onTile)
{
    const uint32_t tilesX = (tgtSize.width + tileW - 1) / tileW;
    const uint32_t tilesY = (tgtSize.height + tileH - 1) / tileH;

    // Compute global flip decision once
    cv::Vec3f centroid;
    float baseU0, baseV0;
    computeCanvasOrigin(fullSize, baseU0, baseV0);
    baseU0 += crop.x;
    baseV0 += crop.y;
    bool globalFlip = computeGlobalFlipDecision(
        surf, std::min(int(tileW), tgtSize.width), std::min(int(tileH), tgtSize.height),
        baseU0, baseV0, params.renderScale, params.scaleSeg, params.transform, centroid);

    // Apply tile row partitioning
    uint32_t startRow = params.startTileRow;
    uint32_t endRow = (params.endTileRow < 0) ? tilesY : std::min(uint32_t(params.endTileRow), tilesY);
    uint32_t partRows = endRow - startRow;

    ProgressTracker progress;
    progress.reset(partRows * tilesX, label);

    // Row work queue for work-stealing pattern
    RowWorkQueue rowQueue;
    rowQueue.reset(partRows);

    // Each thread grabs entire rows and processes all tiles in that row
    // This maximizes cache reuse since tiles in a row share similar volume regions
    #pragma omp parallel
    {
        while (true) {
            int localRow = rowQueue.getNextRow();
            if (localRow < 0) break;
            int ty = startRow + localRow;

            uint32_t y0 = ty * tileH;
            uint32_t dy = std::min(tileH, uint32_t(tgtSize.height) - y0);

            // Process all tiles in this row sequentially
            // Tiles in a row share Y coordinate, so they access similar chunks
            for (uint32_t tx = 0; tx < tilesX; ++tx) {
                uint32_t x0 = tx * tileW;
                uint32_t dx = std::min(tileW, uint32_t(tgtSize.width) - x0);

                // Generate tile geometry
                float tu0 = baseU0 + x0;
                float tv0 = baseV0 + y0;

                cv::Mat_<cv::Vec3f> pts, norms;
                surf->gen(&pts, &norms, cv::Size(dx, dy), cv::Vec3f(0,0,0),
                          params.renderScale, cv::Vec3f(0, tv0, tu0));

                cv::Mat_<cv::Vec3f> basePoints, stepDirs;
                prepareBasePointsAndStepDirs(pts, norms, params.scaleSeg, params.dsScale,
                                             params.transform, globalFlip, basePoints, stepDirs);

                onTile(static_cast<int>(tx), static_cast<int>(ty),
                       static_cast<int>(dx), static_cast<int>(dy),
                       basePoints, stepDirs);

                progress.increment();
            }
        }
    }
    progress.finish();
}

// ============================================================================
// Alternative: Band-Based Processing for Even Better Cache Locality
// ============================================================================

// Process multiple rows of tiles together as a "band"
// This allows even better chunk reuse across tile boundaries
template<typename T, typename TileCallback>
static void renderTiledBandBased(
    QuadSurface* surf,
    ChunkCache<T>* cache,
    const cv::Size& tgtSize,
    const cv::Size& fullSize,
    const cv::Rect& crop,
    const RenderParams& params,
    const OffsetTable& offsets,
    uint32_t tileW, uint32_t tileH,
    uint32_t bandHeight,  // Number of tile rows per band
    const char* label,
    TileCallback&& onTile)
{
    const uint32_t tilesX = (tgtSize.width + tileW - 1) / tileW;
    const uint32_t tilesY = (tgtSize.height + tileH - 1) / tileH;

    // Apply tile row partitioning
    uint32_t partStartRow = params.startTileRow;
    uint32_t partEndRow = (params.endTileRow < 0) ? tilesY : std::min(uint32_t(params.endTileRow), tilesY);
    uint32_t partRows = partEndRow - partStartRow;
    const uint32_t numBands = (partRows + bandHeight - 1) / bandHeight;
    const size_t totalTiles = tilesX * partRows;

    // Compute global flip decision once
    cv::Vec3f centroid;
    float baseU0, baseV0;
    computeCanvasOrigin(fullSize, baseU0, baseV0);
    baseU0 += crop.x;
    baseV0 += crop.y;
    bool globalFlip = computeGlobalFlipDecision(
        surf, std::min(int(tileW), tgtSize.width), std::min(int(tileH), tgtSize.height),
        baseU0, baseV0, params.renderScale, params.scaleSeg, params.transform, centroid);

    ProgressTracker progress;
    progress.reset(totalTiles, label);

    // Process bands sequentially, tiles within band in parallel
    for (uint32_t band = 0; band < numBands; ++band) {
        uint32_t startRow = partStartRow + band * bandHeight;
        uint32_t endRow = std::min(startRow + bandHeight, partEndRow);
        uint32_t tilesInBand = (endRow - startRow) * tilesX;

        #pragma omp parallel for schedule(dynamic, 1)
        for (long long tileIdx = 0; tileIdx < static_cast<long long>(tilesInBand); ++tileIdx) {
            // Map linear index to (tx, ty) within band
            uint32_t localRow = tileIdx / tilesX;
            uint32_t tx = tileIdx % tilesX;
            uint32_t ty = startRow + localRow;

            uint32_t x0 = tx * tileW, y0 = ty * tileH;
            uint32_t dx = std::min(tileW, uint32_t(tgtSize.width) - x0);
            uint32_t dy = std::min(tileH, uint32_t(tgtSize.height) - y0);

            float tu0 = baseU0 + x0;
            float tv0 = baseV0 + y0;

            cv::Mat_<cv::Vec3f> pts, norms;
            surf->gen(&pts, &norms, cv::Size(dx, dy), cv::Vec3f(0,0,0),
                      params.renderScale, cv::Vec3f(0, tv0, tu0));

            cv::Mat_<cv::Vec3f> basePoints, stepDirs;
            prepareBasePointsAndStepDirs(pts, norms, params.scaleSeg, params.dsScale,
                                         params.transform, globalFlip, basePoints, stepDirs);

            onTile(static_cast<int>(tx), static_cast<int>(ty),
                   static_cast<int>(dx), static_cast<int>(dy),
                   basePoints, stepDirs);

            progress.increment();
        }

        // After each band, the cache will naturally evict old chunks
        // This prevents the cache from thrashing
    }
    progress.finish();
}

// ============================================================================
// TIFF Writer Helpers
// ============================================================================

template<typename T>
static std::vector<TiffWriter> createTiffWriters(
    const std::filesystem::path& dir,
    size_t count,
    uint32_t width, uint32_t height,
    uint32_t tileW, uint32_t tileH)
{
    std::vector<TiffWriter> writers;
    writers.reserve(count);
    int maxIdx = count > 0 ? static_cast<int>(count) - 1 : 0;
    for (size_t z = 0; z < count; ++z) {
        writers.emplace_back(
            numberedPath(dir, static_cast<int>(z), maxIdx),
            width, height, PixelTraits<T>::cvType, tileW, tileH, 0.0f);
    }
    return writers;
}

// ============================================================================
// Zarr Pyramid Building
// ============================================================================

template<typename T>
static void downsampleChunk(
    volcart::zarr::ZarrDataset* src,
    volcart::zarr::ZarrDataset* dst,
    size_t z, size_t y, size_t x,
    size_t lz, size_t ly, size_t lx,
    size_t sz, size_t sy, size_t sx)
{
    volcart::zarr::Tensor3D<T> srcChunk(sz, sy, sx);
    src->readSubarray(srcChunk, {2*z, 2*y, 2*x}, {sz, sy, sx});

    volcart::zarr::Tensor3D<T> dstChunk(lz, ly, lx);
    for (size_t zz = 0; zz < lz; ++zz)
        for (size_t yy = 0; yy < ly; ++yy)
            for (size_t xx = 0; xx < lx; ++xx) {
                uint32_t sum = 0;
                int cnt = 0;
                for (int dz = 0; dz < 2 && (2*zz + dz) < sz; ++dz)
                    for (int dy = 0; dy < 2 && (2*yy + dy) < sy; ++dy)
                        for (int dx = 0; dx < 2 && (2*xx + dx) < sx; ++dx) {
                            sum += srcChunk(2*zz + dz, 2*yy + dy, 2*xx + dx);
                            cnt++;
                        }
                dstChunk(zz, yy, xx) = static_cast<T>((sum + cnt/2) / std::max(1, cnt));
            }

    dst->writeSubarray(dstChunk, {z, y, x});
}

template<typename T>
static void buildZarrPyramid(
    const std::filesystem::path& basePath,
    std::vector<std::unique_ptr<volcart::zarr::ZarrDataset>>& levels,
    int maxLevel,
    size_t chunkH, size_t chunkW)
{
    for (int level = 1; level <= maxLevel; ++level) {
        auto& src = levels[level - 1];
        const auto& sShape = src->shape();

        std::vector<size_t> dShape = {
            (sShape[0] + 1) / 2,
            (sShape[1] + 1) / 2,
            (sShape[2] + 1) / 2
        };
        std::vector<size_t> dChunks = {
            dShape[0],
            std::min(chunkH, dShape[1]),
            std::min(chunkW, dShape[2])
        };

        auto dst = std::make_unique<volcart::zarr::ZarrDataset>(
            basePath / std::to_string(level),
            dShape, dChunks, PixelTraits<T>::dtype, "blosc", defaultBloscOpts());

        const size_t tileY = chunkH, tileX = chunkW;
        const size_t tilesY = (dShape[1] + tileY - 1) / tileY;
        const size_t tilesX = (dShape[2] + tileX - 1) / tileX;
        const size_t totalTiles = tilesY * tilesX;

        ProgressTracker progress;
        std::string progLabel = "render L" + std::to_string(level);
        progress.reset(totalTiles, progLabel.c_str());

        // Row-based processing for pyramid too
        RowWorkQueue rowQueue;
        rowQueue.reset(tilesY);

        #pragma omp parallel
        {
            while (true) {
                int ty = rowQueue.getNextRow();
                if (ty < 0) break;

                size_t y = ty * tileY;

                for (size_t tx = 0; tx < tilesX; ++tx) {
                    size_t x = tx * tileX;
                    size_t ly = std::min(tileY, dShape[1] - y);
                    size_t lx = std::min(tileX, dShape[2] - x);
                    size_t lz = dShape[0];

                    size_t sz = std::min<size_t>(2 * lz, sShape[0]);
                    size_t sy = std::min<size_t>(2 * ly, sShape[1] - 2 * y);
                    size_t sx = std::min<size_t>(2 * lx, sShape[2] - 2 * x);

                    downsampleChunk<T>(src.get(), dst.get(), 0, y, x, lz, ly, lx, sz, sy, sx);
                    progress.increment();
                }
            }
        }
        progress.finish();
        levels.push_back(std::move(dst));
    }
}

static void writeZarrAttrs(const std::filesystem::path& path,
                           const std::filesystem::path& volPath,
                           int groupIdx,
                           size_t numSlices,
                           double sliceStep,
                           const AccumParams& accum,
                           const cv::Size& canvasSize,
                           size_t chunkZ, size_t chunkH, size_t chunkW)
{
    json attrs;
    attrs["source_zarr"] = volPath.string();
    attrs["source_group"] = groupIdx;
    attrs["num_slices"] = numSlices;
    attrs["slice_step"] = sliceStep;

    if (!accum.offsets.empty()) {
        attrs["accum_step"] = accum.step;
        attrs["accum_type"] = (accum.type == AccumType::Max) ? "max"
                           : (accum.type == AccumType::Mean) ? "mean" : "median";
        attrs["accum_samples"] = accum.offsets.size();
    }

    attrs["canvas_size"] = {canvasSize.width, canvasSize.height};
    attrs["chunk_size"] = {chunkZ, chunkH, chunkW};
    attrs["note_axes_order"] = "ZYX (slice, row, col)";

    json multiscale;
    multiscale["version"] = "0.4";
    multiscale["name"] = "render";
    multiscale["axes"] = json::array({
        {{"name","z"},{"type","space"}},
        {{"name","y"},{"type","space"}},
        {{"name","x"},{"type","space"}}
    });
    multiscale["datasets"] = json::array();
    for (int level = 0; level <= 5; ++level) {
        double s = std::pow(2.0, level);
        json dset;
        dset["path"] = std::to_string(level);
        dset["coordinateTransformations"] = json::array({
            {{"type","scale"},{"scale", json::array({s, s, s})}},
            {{"type","translation"},{"translation", json::array({0.0, 0.0, 0.0})}}
        });
        multiscale["datasets"].push_back(dset);
    }
    multiscale["metadata"] = {{"downsampling_method", "mean"}};
    attrs["multiscales"] = json::array({multiscale});

    std::ofstream(path / ".zattrs") << attrs.dump(2);
}

// ============================================================================
// Render to Zarr with Row-Based Processing
// ============================================================================

template<typename T>
static void renderToZarr(
    QuadSurface* surf,
    ChunkCache<T>* cache,
    const std::filesystem::path& outPath,
    const std::filesystem::path& volPath,
    int groupIdx,
    const cv::Size& tgtSize,
    const cv::Size& fullSize,
    const cv::Rect& crop,
    const RenderParams& params,
    const OffsetTable& offsets,
    bool includeTifs,
    int shardSize = 0)
{
    constexpr size_t CH = 128, CW = 128;

    cv::Size zarrSize = tgtSize;
    if (params.transform.rotQuad >= 0 && (params.transform.rotQuad % 2 == 1))
        std::swap(zarrSize.width, zarrSize.height);

    std::filesystem::create_directories(outPath);

    std::vector<size_t> shape = {offsets.numSlices, size_t(zarrSize.height), size_t(zarrSize.width)};
    std::vector<size_t> chunks = {shape[0], std::min(CH, shape[1]), std::min(CW, shape[2])};

    auto version = (shardSize > 0) ? volcart::zarr::ZarrVersion::V3
                                   : volcart::zarr::ZarrVersion::V2;
    std::vector<size_t> shards;
    if (shardSize > 0) {
        shards = {shape[0], std::min(size_t(shardSize), shape[1]),
                             std::min(size_t(shardSize), shape[2])};
    }

    auto dsOut = std::make_unique<volcart::zarr::ZarrDataset>(
        outPath / "0", shape, chunks, PixelTraits<T>::dtype, "blosc", defaultBloscOpts(),
        version, shards);

    const int tilesX_src = (tgtSize.width + CW - 1) / CW;
    const int tilesY_src = (tgtSize.height + CH - 1) / CH;
    const int rotQuad = params.transform.rotQuad;
    const int flipAxis = params.transform.flipAxis;

    // Async write queue
    AsyncWriteQueue<T> writeQueue;
    writeQueue.start();

    // Use band-based processing: process 4 rows of tiles at a time
    // This balances parallelism with cache locality
    uint32_t bandHeight = 4;

    renderTiledBandBased<T>(surf, cache, tgtSize, fullSize, crop, params, offsets,
        CW, CH, bandHeight, "render L0",
        [&](int tx, int ty, int dx, int dy,
            const cv::Mat_<cv::Vec3f>& basePoints,
            const cv::Mat_<cv::Vec3f>& stepDirs)
        {
            const bool swapWH = (rotQuad >= 0) && (rotQuad % 2 == 1);
            size_t dx_dst = swapWH ? dy : dx;
            size_t dy_dst = swapWH ? dx : dy;

            int dstTx, dstTy, rTilesX, rTilesY;
            mapTileIndex(tx, ty, tilesX_src, tilesY_src,
                         std::max(rotQuad, 0), flipAxis,
                         dstTx, dstTy, rTilesX, rTilesY);

            volcart::zarr::Tensor3D<T> outChunk(offsets.numSlices, dy_dst, dx_dst);

            renderSlicesWithCallback<T>(cache, basePoints, stepDirs,
                params.nearestNeighbor, offsets, params.accum, rotQuad, flipAxis,
                [&](size_t zi, cv::Mat& slice) {
                    for (int yy = 0; yy < slice.rows; ++yy) {
                        const T* src = slice.ptr<T>(yy);
                        for (int xx = 0; xx < slice.cols; ++xx)
                            outChunk(zi, yy, xx) = src[xx];
                    }
                });

            std::vector<size_t> offset = {0, size_t(dstTy) * CH, size_t(dstTx) * CW};
            writeQueue.enqueue(dsOut.get(), std::move(outChunk), std::move(offset));
        });

    writeQueue.finish();
    dsOut->flush();

    // Skip pyramid and tiff export when running as one part of a multi-VM job.
    // These should be run separately after all parts finish writing L0 tiles.
    if (params.numParts > 1) {
        std::cout << "[zarr] multi-part mode: skipping pyramid build and tif export\n";
        return;
    }

    // Build pyramid
    std::vector<std::unique_ptr<volcart::zarr::ZarrDataset>> levels;
    levels.push_back(std::move(dsOut));
    buildZarrPyramid<T>(outPath, levels, 5, CH, CW);

    // Write attributes
    writeZarrAttrs(outPath, volPath, groupIdx, offsets.numSlices, params.sliceStep,
                   params.accum, zarrSize, offsets.numSlices, CH, CW);

    // Optionally export TIFFs
    if (includeTifs) {
        std::string zname = outPath.stem().string();
        std::filesystem::path layersDir = outPath.parent_path() / ("layers_" + zname);

        int maxIdx = offsets.numSlices > 0 ? static_cast<int>(offsets.numSlices) - 1 : 0;
        if (allFilesExist(layersDir, offsets.numSlices, maxIdx)) {
            std::cout << "[tif export] all slices exist, skipping" << std::endl;
            return;
        }

        std::filesystem::create_directories(layersDir);

        auto& dsL0 = levels[0];
        const auto& s = dsL0->shape();
        const uint32_t Z = s[0], Y = s[1], X = s[2];

        auto writers = createTiffWriters<T>(layersDir, Z, X, Y, CW, CH);
        std::vector<std::mutex> locks(Z);

        const uint32_t tilesX = (X + CW - 1) / CW;
        const uint32_t tilesY = (Y + CH - 1) / CH;

        ProgressTracker progress;
        progress.reset(tilesX * tilesY, "tif export");

        // Row-based TIFF export too
        RowWorkQueue rowQueue;
        rowQueue.reset(tilesY);

        #pragma omp parallel
        {
            while (true) {
                int ty = rowQueue.getNextRow();
                if (ty < 0) break;

                uint32_t y0 = ty * CH;
                uint32_t dy = std::min<uint32_t>(CH, Y - y0);

                for (uint32_t tx = 0; tx < tilesX; ++tx) {
                    uint32_t x0 = tx * CW;
                    uint32_t dx = std::min<uint32_t>(CW, X - x0);

                    volcart::zarr::Tensor3D<T> tile(Z, dy, dx);
                    dsL0->readSubarray(tile, {0, y0, x0}, {Z, dy, dx});

                    for (uint32_t z = 0; z < Z; ++z) {
                        cv::Mat srcTile(dy, dx, PixelTraits<T>::cvType);
                        for (uint32_t yy = 0; yy < dy; ++yy) {
                            T* row = srcTile.ptr<T>(yy);
                            for (uint32_t xx = 0; xx < dx; ++xx)
                                row[xx] = tile(z, yy, xx);
                        }
                        std::lock_guard<std::mutex> guard(locks[z]);
                        writers[z].writeTile(x0, y0, srcTile);
                    }
                    progress.increment();
                }
            }
        }
        progress.finish();
    }
}

// ============================================================================
// CSVS Subvolume Write Helper
// ============================================================================

template<typename T>
static void csvsWriteSubvolume(volcart::csvs::CsvsDataset* ds,
                                const volcart::zarr::Tensor3D<T>& data,
                                std::array<size_t, 3> offset)
{
    const auto& s = data.shape();
    uint32_t cs = ds->chunkSize();
    size_t elemSize = volcart::zarr::dtypeSize(ds->dtype());
    size_t chunkBytes = size_t(cs) * cs * cs * elemSize;
    std::vector<uint8_t> chunkBuf(chunkBytes, 0);

    for (size_t oz = 0; oz < s[0]; ) {
        size_t gz = offset[0] + oz;
        size_t cz = gz / cs;
        size_t lz = gz % cs;
        size_t dz = std::min(size_t(cs) - lz, s[0] - oz);

        for (size_t oy = 0; oy < s[1]; ) {
            size_t gy = offset[1] + oy;
            size_t cy = gy / cs;
            size_t ly = gy % cs;
            size_t dy = std::min(size_t(cs) - ly, s[1] - oy);

            for (size_t ox = 0; ox < s[2]; ) {
                size_t gx = offset[2] + ox;
                size_t cx = gx / cs;
                size_t lx = gx % cs;
                size_t dx = std::min(size_t(cs) - lx, s[2] - ox);

                std::memset(chunkBuf.data(), 0, chunkBytes);
                ds->readChunk(cz, cy, cx, chunkBuf.data());

                T* dst = reinterpret_cast<T*>(chunkBuf.data());
                for (size_t z = 0; z < dz; ++z)
                    for (size_t y = 0; y < dy; ++y)
                        for (size_t x = 0; x < dx; ++x)
                            dst[(lz + z) * cs * cs + (ly + y) * cs + (lx + x)] =
                                data(oz + z, oy + y, ox + x);

                ds->writeChunk(cz, cy, cx, chunkBuf.data(), chunkBytes);
                ox += dx;
            }
            oy += dy;
        }
        oz += dz;
    }
}

template<typename T>
static void csvsReadSubvolume(volcart::csvs::CsvsDataset* ds,
                               volcart::zarr::Tensor3D<T>& out,
                               std::array<size_t, 3> offset,
                               std::array<size_t, 3> size)
{
    out.resize(size[0], size[1], size[2]);
    out.fill(T{0});

    uint32_t cs = ds->chunkSize();
    size_t elemSize = volcart::zarr::dtypeSize(ds->dtype());
    size_t chunkBytes = size_t(cs) * cs * cs * elemSize;
    std::vector<uint8_t> chunkBuf(chunkBytes);

    for (size_t oz = 0; oz < size[0]; ) {
        size_t gz = offset[0] + oz;
        size_t cz = gz / cs;
        size_t lz = gz % cs;
        size_t dz = std::min(size_t(cs) - lz, size[0] - oz);

        for (size_t oy = 0; oy < size[1]; ) {
            size_t gy = offset[1] + oy;
            size_t cy = gy / cs;
            size_t ly = gy % cs;
            size_t dy = std::min(size_t(cs) - ly, size[1] - oy);

            for (size_t ox = 0; ox < size[2]; ) {
                size_t gx = offset[2] + ox;
                size_t cx = gx / cs;
                size_t lx = gx % cs;
                size_t dx = std::min(size_t(cs) - lx, size[2] - ox);

                if (ds->readChunk(cz, cy, cx, chunkBuf.data())) {
                    const T* src = reinterpret_cast<const T*>(chunkBuf.data());
                    for (size_t z = 0; z < dz; ++z)
                        for (size_t y = 0; y < dy; ++y)
                            for (size_t x = 0; x < dx; ++x)
                                out(oz + z, oy + y, ox + x) =
                                    src[(lz + z) * cs * cs + (ly + y) * cs + (lx + x)];
                }
                ox += dx;
            }
            oy += dy;
        }
        oz += dz;
    }
}

// ============================================================================
// Async Write Queue for CSVS
// ============================================================================

template<typename T>
struct AsyncCsvsWriteQueue {
    struct WriteJob {
        volcart::zarr::Tensor3D<T> data;
        std::array<size_t, 3> offset;
        volcart::csvs::CsvsDataset* ds;
    };

    std::queue<WriteJob> jobs;
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable cvFull;
    std::thread writerThread;
    bool done = false;
    size_t maxQueueSize = 16;

    void start() {
        writerThread = std::thread([this]() {
            while (true) {
                WriteJob job;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this]() { return !jobs.empty() || done; });
                    if (done && jobs.empty()) break;
                    job = std::move(jobs.front());
                    jobs.pop();
                    cvFull.notify_one();
                }
                csvsWriteSubvolume<T>(job.ds, job.data, job.offset);
            }
        });
    }

    void enqueue(volcart::csvs::CsvsDataset* ds, volcart::zarr::Tensor3D<T>&& data,
                 std::array<size_t, 3> offset) {
        std::unique_lock<std::mutex> lock(mutex);
        cvFull.wait(lock, [this]() { return jobs.size() < maxQueueSize; });
        jobs.push({std::move(data), offset, ds});
        cv.notify_one();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            done = true;
        }
        cv.notify_one();
        if (writerThread.joinable()) writerThread.join();
    }
};

// ============================================================================
// CSVS Pyramid Building
// ============================================================================

template<typename T>
static void downsampleChunkCsvs(
    volcart::csvs::CsvsDataset* src,
    volcart::csvs::CsvsDataset* dst,
    size_t z, size_t y, size_t x,
    size_t lz, size_t ly, size_t lx,
    size_t sz, size_t sy, size_t sx)
{
    volcart::zarr::Tensor3D<T> srcChunk(sz, sy, sx);
    csvsReadSubvolume<T>(src, srcChunk, {2*z, 2*y, 2*x}, {sz, sy, sx});

    volcart::zarr::Tensor3D<T> dstChunk(lz, ly, lx);
    for (size_t zz = 0; zz < lz; ++zz)
        for (size_t yy = 0; yy < ly; ++yy)
            for (size_t xx = 0; xx < lx; ++xx) {
                uint32_t sum = 0;
                int cnt = 0;
                for (int dz = 0; dz < 2 && (2*zz + dz) < sz; ++dz)
                    for (int dy = 0; dy < 2 && (2*yy + dy) < sy; ++dy)
                        for (int dx = 0; dx < 2 && (2*xx + dx) < sx; ++dx) {
                            sum += srcChunk(2*zz + dz, 2*yy + dy, 2*xx + dx);
                            cnt++;
                        }
                dstChunk(zz, yy, xx) = static_cast<T>((sum + cnt/2) / std::max(1, cnt));
            }

    csvsWriteSubvolume<T>(dst, dstChunk, {z, y, x});
}

template<typename T>
static void buildCsvsPyramid(
    const std::filesystem::path& basePath,
    std::vector<std::unique_ptr<volcart::csvs::CsvsDataset>>& levels,
    int maxLevel,
    size_t chunkSize,
    size_t shardSize)
{
    for (int level = 1; level <= maxLevel; ++level) {
        auto& src = levels[level - 1];
        const auto& sShape = src->shape();

        std::array<size_t, 3> dShape = {
            (sShape[0] + 1) / 2,
            (sShape[1] + 1) / 2,
            (sShape[2] + 1) / 2
        };

        auto dst = std::make_unique<volcart::csvs::CsvsDataset>(
            basePath / std::to_string(level),
            dShape, static_cast<uint32_t>(chunkSize),
            static_cast<uint32_t>(shardSize),
            src->dtype());

        const size_t tileY = chunkSize, tileX = chunkSize;
        const size_t tilesY = (dShape[1] + tileY - 1) / tileY;
        const size_t tilesX = (dShape[2] + tileX - 1) / tileX;
        const size_t totalTiles = tilesY * tilesX;

        ProgressTracker progress;
        std::string progLabel = "render L" + std::to_string(level);
        progress.reset(totalTiles, progLabel.c_str());

        RowWorkQueue rowQueue;
        rowQueue.reset(tilesY);

        #pragma omp parallel
        {
            while (true) {
                int ty = rowQueue.getNextRow();
                if (ty < 0) break;

                size_t y = ty * tileY;

                for (size_t tx = 0; tx < tilesX; ++tx) {
                    size_t x = tx * tileX;
                    size_t ly = std::min(tileY, dShape[1] - y);
                    size_t lx = std::min(tileX, dShape[2] - x);
                    size_t lz = dShape[0];

                    size_t sz = std::min<size_t>(2 * lz, sShape[0]);
                    size_t sy = std::min<size_t>(2 * ly, sShape[1] - 2 * y);
                    size_t sx = std::min<size_t>(2 * lx, sShape[2] - 2 * x);

                    downsampleChunkCsvs<T>(src.get(), dst.get(), 0, y, x, lz, ly, lx, sz, sy, sx);
                    progress.increment();
                }
            }
        }
        progress.finish();
        levels.push_back(std::move(dst));
    }
}

// ============================================================================
// Render to CSVS
// ============================================================================

template<typename T>
static void renderToCsvs(
    QuadSurface* surf,
    ChunkCache<T>* cache,
    const std::filesystem::path& outPath,
    const cv::Size& tgtSize,
    const cv::Size& fullSize,
    const cv::Rect& crop,
    const RenderParams& params,
    const OffsetTable& offsets,
    int shardSize)
{
    constexpr size_t CH = 128, CW = 128;

    cv::Size csvsSize = tgtSize;
    if (params.transform.rotQuad >= 0 && (params.transform.rotQuad % 2 == 1))
        std::swap(csvsSize.width, csvsSize.height);

    std::filesystem::create_directories(outPath);

    std::array<size_t, 3> shape = {offsets.numSlices, size_t(csvsSize.height), size_t(csvsSize.width)};

    auto dsOut = std::make_unique<volcart::csvs::CsvsDataset>(
        outPath / "0", shape,
        static_cast<uint32_t>(std::min(CH, CW)),
        static_cast<uint32_t>(shardSize),
        PixelTraits<T>::dtype == volcart::zarr::Dtype::UInt16 ?
            volcart::zarr::Dtype::UInt16 : volcart::zarr::Dtype::UInt8);

    const int tilesX_src = (tgtSize.width + CW - 1) / CW;
    const int tilesY_src = (tgtSize.height + CH - 1) / CH;
    const int rotQuad = params.transform.rotQuad;
    const int flipAxis = params.transform.flipAxis;

    AsyncCsvsWriteQueue<T> writeQueue;
    writeQueue.start();

    uint32_t bandHeight = 4;

    renderTiledBandBased<T>(surf, cache, tgtSize, fullSize, crop, params, offsets,
        CW, CH, bandHeight, "render L0",
        [&](int tx, int ty, int dx, int dy,
            const cv::Mat_<cv::Vec3f>& basePoints,
            const cv::Mat_<cv::Vec3f>& stepDirs)
        {
            const bool swapWH = (rotQuad >= 0) && (rotQuad % 2 == 1);
            size_t dx_dst = swapWH ? dy : dx;
            size_t dy_dst = swapWH ? dx : dy;

            int dstTx, dstTy, rTilesX, rTilesY;
            mapTileIndex(tx, ty, tilesX_src, tilesY_src,
                         std::max(rotQuad, 0), flipAxis,
                         dstTx, dstTy, rTilesX, rTilesY);

            volcart::zarr::Tensor3D<T> outChunk(offsets.numSlices, dy_dst, dx_dst);

            renderSlicesWithCallback<T>(cache, basePoints, stepDirs,
                params.nearestNeighbor, offsets, params.accum, rotQuad, flipAxis,
                [&](size_t zi, cv::Mat& slice) {
                    for (int yy = 0; yy < slice.rows; ++yy) {
                        const T* src = slice.ptr<T>(yy);
                        for (int xx = 0; xx < slice.cols; ++xx)
                            outChunk(zi, yy, xx) = src[xx];
                    }
                });

            std::array<size_t, 3> offset = {0, size_t(dstTy) * CH, size_t(dstTx) * CW};
            writeQueue.enqueue(dsOut.get(), std::move(outChunk), offset);
        });

    writeQueue.finish();

    if (params.numParts > 1) {
        std::cout << "[csvs] multi-part mode: skipping pyramid build\n";
        return;
    }

    // Build pyramid
    std::vector<std::unique_ptr<volcart::csvs::CsvsDataset>> levels;
    levels.push_back(std::move(dsOut));
    buildCsvsPyramid<T>(outPath, levels, 5, std::min(CH, CW), shardSize);
}

// ============================================================================
// Render to TIFF with Row-Based Processing
// ============================================================================

template<typename T>
static void renderToTiff(
    QuadSurface* surf,
    ChunkCache<T>* cache,
    const std::filesystem::path& outPath,
    const cv::Size& tgtSize,
    const cv::Size& fullSize,
    const cv::Rect& crop,
    const RenderParams& params,
    const OffsetTable& offsets)
{
    constexpr uint32_t tileW = 64, tileH = 64;

    const int rotQuad = params.transform.rotQuad;
    const int flipAxis = params.transform.flipAxis;

    int outW = tgtSize.width, outH = tgtSize.height;
    if (rotQuad >= 0 && (rotQuad % 2 == 1))
        std::swap(outW, outH);

    // For multi-part TIFF, compute the pixel height of this part's rows
    const uint32_t tilesY_full = (outH + tileH - 1) / tileH;
    uint32_t partStartRow = params.startTileRow;
    uint32_t partEndRow = (params.endTileRow < 0) ? tilesY_full : std::min(uint32_t(params.endTileRow), tilesY_full);
    uint32_t partPixelY0 = partStartRow * tileH;
    int partOutH = std::min(int(partEndRow * tileH), outH) - int(partPixelY0);
    if (partOutH <= 0) {
        std::cout << "[tif] no rows assigned to this part, skipping" << std::endl;
        return;
    }

    bool usesPattern = outPath.string().find('%') != std::string::npos;
    auto makePath = [&](int z) -> std::filesystem::path {
        if (usesPattern) {
            char buf[1024];
            snprintf(buf, sizeof(buf), outPath.string().c_str(), z);
            return buf;
        }
        return numberedPath(outPath, z, std::max(0, static_cast<int>(offsets.numSlices) - 1));
    };

    int maxIdx = std::max(0, static_cast<int>(offsets.numSlices) - 1);
    if (!usesPattern && allFilesExist(outPath, offsets.numSlices, maxIdx)) {
        std::cout << "[tif] all slices exist, skipping" << std::endl;
        return;
    }

    if (!usesPattern) {
        std::filesystem::create_directories(outPath);
    } else {
        std::filesystem::create_directories(outPath.parent_path());
    }

    std::vector<TiffWriter> writers;
    writers.reserve(offsets.numSlices);
    for (size_t z = 0; z < offsets.numSlices; ++z) {
        writers.emplace_back(makePath(z), outW, partOutH, PixelTraits<T>::cvType, tileW, tileH, 0.0f);
    }
    std::vector<std::mutex> locks(offsets.numSlices);

    const int tilesX_src = (tgtSize.width + tileW - 1) / tileW;
    const int tilesY_src = (tgtSize.height + tileH - 1) / tileH;

    // Use row-based processing
    renderTiledRowBased<T>(surf, cache, tgtSize, fullSize, crop, params, offsets,
        tileW, tileH, "tif tiled",
        [&](int tx, int ty, int dx, int dy,
            const cv::Mat_<cv::Vec3f>& basePoints,
            const cv::Mat_<cv::Vec3f>& stepDirs)
        {
            renderSlicesWithCallback<T>(cache, basePoints, stepDirs,
                params.nearestNeighbor, offsets, params.accum, rotQuad, flipAxis,
                [&](size_t zi, cv::Mat& slice) {
                    int dstTx, dstTy, rTilesX, rTilesY;
                    mapTileIndex(tx, ty, tilesX_src, tilesY_src,
                                 std::max(rotQuad, 0), flipAxis,
                                 dstTx, dstTy, rTilesX, rTilesY);
                    uint32_t x0 = dstTx * tileW;
                    uint32_t y0 = dstTy * tileH - partPixelY0;
                    std::lock_guard<std::mutex> guard(locks[zi]);
                    writers[zi].writeTile(x0, y0, slice);
                });
        });
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[])
{
    auto wallStart = std::chrono::steady_clock::now();

    po::options_description required("Required arguments");
    required.add_options()
        ("volume,v", po::value<std::string>()->required(), "Path to the OME-Zarr volume")
        ("output,o", po::value<std::string>()->required(), "Output path")
        ("scale", po::value<float>()->required(), "Pixels per level-g voxel")
        ("group-idx,g", po::value<int>()->required(), "OME-Zarr group index");

    po::options_description optional("Optional arguments");
    optional.add_options()
        ("help,h", "Show this help message")
        ("segmentation,s", po::value<std::string>(), "Path to segmentation folder")
        ("render-folder", po::value<std::string>(), "Folder containing segmentations to batch render")
        ("format", po::value<std::string>(), "Output format for batch mode (zarr|tif)")
        ("cache-gb", po::value<size_t>()->default_value(16), "Cache size in GB")
        ("num-slices,n", po::value<int>()->default_value(1), "Number of slices")
        ("slice-step", po::value<float>()->default_value(1.0f), "Slice spacing")
        ("accum", po::value<float>()->default_value(0.0f), "Accumulation step")
        ("accum-type", po::value<std::string>()->default_value("max"), "Accumulation type (max|mean|median)")
        ("crop-x", po::value<int>()->default_value(0), "Crop X")
        ("crop-y", po::value<int>()->default_value(0), "Crop Y")
        ("crop-width", po::value<int>()->default_value(0), "Crop width")
        ("crop-height", po::value<int>()->default_value(0), "Crop height")
        ("affine", po::value<std::vector<std::string>>()->multitoken()->composing(), "Affine transform files")
        ("scale-segmentation", po::value<float>()->default_value(1.0f), "Segmentation scale")
        ("rotate", po::value<double>()->default_value(0.0), "Rotation angle")
        ("flip", po::value<int>()->default_value(-1), "Flip axis (0=V, 1=H, 2=both)")
        ("include-tifs", po::bool_switch()->default_value(false), "Export TIFFs from Zarr")
        ("flatten", po::bool_switch()->default_value(false), "Apply ABF++ flattening")
        ("flatten-iterations", po::value<int>()->default_value(10), "ABF++ iterations")
        ("flatten-downsample", po::value<int>()->default_value(1), "ABF++ downsample factor")
        ("nearest-neighbor", po::bool_switch()->default_value(false), "Use nearest-neighbor sampling")
        ("threads", po::value<int>()->default_value(0), "Number of threads (0 = auto)")
        ("shard-size", po::value<int>()->default_value(0), "Shard size for zarr v3 sharding (0 = disabled, implies v3)")
        ("band-height", po::value<int>()->default_value(4), "Tile rows per band for cache locality")
        ("num-parts", po::value<int>()->default_value(1), "Total number of shards for distributed rendering")
        ("part-id", po::value<int>()->default_value(0), "This shard's ID (0-indexed)");

    po::options_description all("Usage");
    all.add(required).add(optional);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), vm);
        if (vm.count("help") || argc < 2) {
            std::cout << "vc_render_tifxyz: Render volume data using segmentation surfaces\n\n";
            std::cout << all << '\n';
            return EXIT_SUCCESS;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\nUse --help for usage\n";
        return EXIT_FAILURE;
    }

    // Set thread count
    int numThreads = vm["threads"].as<int>();
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }
    std::cout << "Using " << omp_get_max_threads() << " threads\n";

    // Parse basic arguments
    std::filesystem::path volPath = vm["volume"].as<std::string>();
    std::string outArg = vm["output"].as<std::string>();
    int groupIdx = vm["group-idx"].as<int>();
    float tgtScale = vm["scale"].as<float>();
    int numSlices = vm["num-slices"].as<int>();
    float sliceStep = vm["slice-step"].as<float>();
    bool includeTifs = vm["include-tifs"].as<bool>();
    bool flatten = vm["flatten"].as<bool>();
    int flattenIters = vm["flatten-iterations"].as<int>();
    int flattenDownsample = vm["flatten-downsample"].as<int>();

    if (sliceStep <= 0) {
        std::cerr << "Error: --slice-step must be positive\n";
        return EXIT_FAILURE;
    }

    // Parse batch mode
    bool hasBatchFolder = vm.count("render-folder") > 0;
    std::filesystem::path batchFolder;
    std::string batchFormat;
    if (hasBatchFolder) {
        batchFolder = vm["render-folder"].as<std::string>();
        if (!vm.count("format")) {
            std::cerr << "Error: --format required with --render-folder\n";
            return EXIT_FAILURE;
        }
        batchFormat = vm["format"].as<std::string>();
        std::transform(batchFormat.begin(), batchFormat.end(), batchFormat.begin(), ::tolower);
        if (batchFormat != "zarr" && batchFormat != "tif") {
            std::cerr << "Error: --format must be 'zarr' or 'tif'\n";
            return EXIT_FAILURE;
        }
    }

    std::filesystem::path segPath;
    if (!hasBatchFolder) {
        if (!vm.count("segmentation")) {
            std::cerr << "Error: --segmentation required unless --render-folder used\n";
            return EXIT_FAILURE;
        }
        segPath = vm["segmentation"].as<std::string>();
    }

    // Parse crop
    CropParams cropParams;
    cropParams.x = vm["crop-x"].as<int>();
    cropParams.y = vm["crop-y"].as<int>();
    cropParams.width = vm["crop-width"].as<int>();
    cropParams.height = vm["crop-height"].as<int>();

    // Parse accumulation
    AccumParams accum;
    float accumStep = vm["accum"].as<float>();
    if (accumStep > 0) {
        if (accumStep > sliceStep) {
            std::cerr << "Error: --accum must be <= --slice-step\n";
            return EXIT_FAILURE;
        }
        double ratio = sliceStep / accumStep;
        double rounded = std::round(ratio);
        if (std::abs(ratio - rounded) > 1e-4) {
            std::cerr << "Error: --accum must evenly divide --slice-step\n";
            return EXIT_FAILURE;
        }
        size_t samples = std::max<size_t>(1, static_cast<size_t>(rounded));
        double spacing = sliceStep / samples;
        for (size_t i = 0; i < samples; ++i)
            accum.offsets.push_back(static_cast<float>(spacing * i));
        accum.step = spacing;

        std::string typeStr = vm["accum-type"].as<std::string>();
        std::transform(typeStr.begin(), typeStr.end(), typeStr.begin(), ::tolower);
        accum.type = (typeStr == "mean") ? AccumType::Mean
                   : (typeStr == "median") ? AccumType::Median
                   : AccumType::Max;

        std::cout << "Accumulation: " << samples << " samples at step " << spacing
                  << " using '" << typeStr << "'\n";
    }

    // Parse transform
    TransformParams transform;
    double rotateAngle = vm["rotate"].as<double>();
    if (std::abs(rotateAngle) > 1e-6) {
        transform.rotQuad = normalizeQuadrantRotation(rotateAngle);
        if (transform.rotQuad < 0) {
            std::cerr << "Error: only 0/90/180/270 degree rotations supported\n";
            return EXIT_FAILURE;
        }
        std::cout << "Rotation: " << (transform.rotQuad * 90) << " degrees\n";
    }
    transform.flipAxis = vm["flip"].as<int>();

    // Load affines
    if (vm.count("affine")) {
        if (!loadAndComposeAffines(vm["affine"].as<std::vector<std::string>>(),
                                   transform.affine, transform.hasAffine)) {
            return EXIT_FAILURE;
        }
    }

    // Load dataset
    auto ds = std::make_unique<volcart::zarr::ZarrDataset>(volPath / std::to_string(groupIdx));
    const bool isU16 = (ds->getDtype() == volcart::zarr::Dtype::UInt16);
    const float dsScale = std::ldexp(1.0f, -groupIdx);

    std::cout << "Dataset shape: [";
    for (size_t i = 0; i < ds->shape().size(); ++i)
        std::cout << (i ? ", " : "") << ds->shape()[i];
    std::cout << "], dtype=" << (isU16 ? "uint16" : "uint8") << "\n";

    // Initialize caches
    size_t cacheBytes = vm["cache-gb"].as<size_t>() * 1024ull * 1024ull * 1024ull;
    ChunkCache<uint8_t> cache8(cacheBytes);
    ChunkCache<uint16_t> cache16(cacheBytes);
    cache8.init(ds.get());
    cache16.init(ds.get());
    std::cout << "Cache size: " << (cacheBytes / (1024*1024*1024)) << " GB\n";

    float scaleSeg = vm["scale-segmentation"].as<float>();
    bool nearestNeighbor = vm["nearest-neighbor"].as<bool>();
    if (nearestNeighbor)
        std::cout << "Using nearest-neighbor sampling\n";

    int shardSize = vm["shard-size"].as<int>();
    if (shardSize > 0)
        std::cout << "Zarr v3 sharding enabled, shard size: " << shardSize << "\n";

    int numParts = vm["num-parts"].as<int>();
    int partId = vm["part-id"].as<int>();
    if (numParts < 1) {
        std::cerr << "Error: --num-parts must be >= 1\n";
        return EXIT_FAILURE;
    }
    if (partId < 0 || partId >= numParts) {
        std::cerr << "Error: --part-id must be in [0, num-parts)\n";
        return EXIT_FAILURE;
    }
    if (numParts > 1)
        std::cout << "Sharding: part " << partId << " of " << numParts << "\n";

    // Pre-build offset table (shared across all tiles)
    OffsetTable offsets;
    offsets.build(numSlices, sliceStep, accum.offsets);
    offsets.preScale(dsScale);  // Pre-multiply by dsScale

    // Process function
    auto processOne = [&](const std::filesystem::path& segFolder,
                          const std::filesystem::path& outPath,
                          bool forceZarr)
    {
        std::filesystem::path outputPath = outPath;
        if (forceZarr && outputPath.extension() != ".zarr")
            outputPath = outputPath.string() + ".zarr";

        bool outputZarr = forceZarr || (outputPath.extension() == ".zarr");

        std::cout << "Rendering: " << segFolder << " -> " << outputPath
                  << (outputZarr ? " (zarr)" : " (tif)") << "\n";

        // Load surface
        std::unique_ptr<QuadSurface> surf;
        try {
            surf = load_quad_from_tifxyz(segFolder);
        } catch (...) {
            std::cerr << "Error loading: " << segFolder << "\n";
            return;
        }

        // Apply flattening if requested
        if (flatten) {
            std::cout << "Applying ABF++ flattening...\n";
            vc::ABFConfig cfg;
            cfg.maxIterations = flattenIters;
            cfg.downsampleFactor = flattenDownsample;
            cfg.useABF = true;
            cfg.scaleToOriginalArea = true;
            if (auto* flat = vc::abfFlattenToNewSurface(*surf, cfg)) {
                surf.reset(flat);
                std::cout << "Flattening complete: " << surf->rawPointsPtr()->cols
                          << " x " << surf->rawPointsPtr()->rows << "\n";
            } else {
                std::cerr << "Warning: flattening failed, using original\n";
            }
        }

        // Sanitize points
        auto* rawPts = surf->rawPointsPtr();
        for (int j = 0; j < rawPts->rows; ++j)
            for (int i = 0; i < rawPts->cols; ++i)
                if ((*rawPts)(j,i)[0] == -1)
                    (*rawPts)(j,i) = {NAN, NAN, NAN};

        cv::Size fullSize = rawPts->size();

        // Compute render scale
        double sA = 1.0;
        if (transform.hasAffine) {
            double det = cv::determinant(cv::Mat(transform.affine.linearPart()));
            if (std::isfinite(det) && std::abs(det) > 1e-18)
                sA = std::cbrt(std::abs(det));
        }
        float renderScale = static_cast<float>(tgtScale * scaleSeg * sA * dsScale);

        // Scale canvas
        double sx = renderScale / surf->_scale[1];
        double sy = renderScale / surf->_scale[0];
        fullSize.width = std::max(1, static_cast<int>(std::lround(fullSize.width * sx)));
        fullSize.height = std::max(1, static_cast<int>(std::lround(fullSize.height * sy)));

        // Compute crop
        cv::Rect crop = cropParams.toRect(fullSize);
        if (crop.width <= 0 || crop.height <= 0) {
            std::cerr << "Error: crop outside canvas\n";
            return;
        }
        cv::Size tgtSize = crop.size();

        std::cout << "Canvas: " << fullSize << ", crop: " << crop
                  << ", renderScale=" << renderScale << "\n";

        // Build render params
        RenderParams params;
        params.renderScale = renderScale;
        params.scaleSeg = scaleSeg;
        params.dsScale = dsScale;
        params.sliceStep = sliceStep;
        params.nearestNeighbor = nearestNeighbor;
        params.transform = transform;
        params.accum = accum;
        params.numParts = numParts;

        // Compute tile row partitioning for multi-VM sharding
        if (numParts > 1) {
            // Tile height depends on output format: 128 for zarr, 64 for tiff
            uint32_t tileH = outputZarr ? 128 : 64;
            uint32_t tilesY = (tgtSize.height + tileH - 1) / tileH;
            uint32_t rowsPerPart = tilesY / numParts;
            params.startTileRow = partId * rowsPerPart;
            params.endTileRow = (partId == numParts - 1) ? tilesY : (partId + 1) * rowsPerPart;
            std::cout << "Part " << partId << ": tile rows [" << params.startTileRow
                      << ", " << params.endTileRow << ") of " << tilesY << "\n";
        }

        // For tiff with multi-part, modify output path
        std::filesystem::path actualOutputPath = outputPath;
        if (numParts > 1 && !outputZarr) {
            // Add _part_NNN suffix to the output directory/pattern
            std::ostringstream suffix;
            suffix << "_part_" << std::setw(3) << std::setfill('0') << partId;
            if (outputPath.string().find('%') != std::string::npos) {
                // Pattern mode: insert suffix before the extension in the pattern
                std::string s = outputPath.string();
                auto dot = s.rfind('.');
                if (dot != std::string::npos)
                    actualOutputPath = s.substr(0, dot) + suffix.str() + s.substr(dot);
                else
                    actualOutputPath = s + suffix.str();
            } else {
                actualOutputPath = outputPath.string() + suffix.str();
            }
            std::cout << "Part output: " << actualOutputPath << "\n";
        }

        // Render
        if (outputZarr && shardSize > 0) {
            // Use CSVS format for sharded output
            if (isU16)
                renderToCsvs<uint16_t>(surf.get(), &cache16, outputPath,
                                       tgtSize, fullSize, crop, params, offsets, shardSize);
            else
                renderToCsvs<uint8_t>(surf.get(), &cache8, outputPath,
                                      tgtSize, fullSize, crop, params, offsets, shardSize);
        } else if (outputZarr) {
            if (isU16)
                renderToZarr<uint16_t>(surf.get(), &cache16, outputPath, volPath, groupIdx,
                                       tgtSize, fullSize, crop, params, offsets, includeTifs, shardSize);
            else
                renderToZarr<uint8_t>(surf.get(), &cache8, outputPath, volPath, groupIdx,
                                      tgtSize, fullSize, crop, params, offsets, includeTifs, shardSize);
        } else {
            if (actualOutputPath.string().find('%') == std::string::npos)
                std::filesystem::create_directories(actualOutputPath);
            else
                std::filesystem::create_directories(actualOutputPath.parent_path());
            if (isU16)
                renderToTiff<uint16_t>(surf.get(), &cache16, actualOutputPath,
                                       tgtSize, fullSize, crop, params, offsets);
            else
                renderToTiff<uint8_t>(surf.get(), &cache8, actualOutputPath,
                                      tgtSize, fullSize, crop, params, offsets);
        }
    };

    // Execute
    if (hasBatchFolder) {
        for (const auto& entry : std::filesystem::directory_iterator(batchFolder)) {
            if (!entry.is_directory()) continue;

            std::string segName = entry.path().filename().string();
            std::filesystem::path base(outArg);
            std::filesystem::path outPath;

            if (batchFormat == "zarr") {
                std::filesystem::path baseDir;
                if ((std::filesystem::exists(base) && std::filesystem::is_directory(base)) ||
                    !base.has_extension()) {
                    baseDir = base;
                    std::filesystem::create_directories(baseDir);
                } else {
                    baseDir = base.parent_path();
                    if (baseDir.empty()) baseDir = std::filesystem::current_path();
                }
                std::string prefix = base.stem().string().empty()
                                   ? base.filename().string()
                                   : base.stem().string();
                outPath = baseDir / (prefix + "_" + segName);
            } else {
                outPath = base.is_absolute() ? (base / segName) : (entry.path() / base);
            }

            processOne(entry.path(), outPath, batchFormat == "zarr");
        }
    } else {
        processOne(segPath, outArg, false);
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    std::cout << "Total time: " << formatDuration(elapsed) << std::endl;

    return EXIT_SUCCESS;
}