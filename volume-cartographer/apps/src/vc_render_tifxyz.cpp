#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/ABFFlattening.hpp"
#include "vc/core/simd/simd.hpp"

#include <nlohmann/json.hpp>
#include <csignal>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <boost/program_options.hpp>
#include <mutex>
#include <cmath>
#include <set>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <thread>
#include <tiffio.h>
#include <omp.h>

namespace po = boost::program_options;
using json = nlohmann::json;

enum class SampleMode { Nearest, Trilinear };

// ============================================================
// Tile coordinate transform (rotation + flip)
// ============================================================

static inline void mapTileIndex(int tx, int ty, int tilesX, int tilesY,
                         int quadRot, int flipType,
                         int& outTx, int& outTy, int& outTilesX, int& outTilesY)
{
    bool swap = (quadRot % 2) == 1;
    int rTX = swap ? tilesY : tilesX, rTY = swap ? tilesX : tilesY;
    int rx = tx, ry = ty;
    switch (quadRot) {
        case 1: rx = ty;              ry = tilesX - 1 - tx; break;
        case 2: rx = tilesX - 1 - tx; ry = tilesY - 1 - ty; break;
        case 3: rx = tilesY - 1 - ty; ry = tx;              break;
        default: break;
    }
    int fx = rx, fy = ry;
    if (flipType == 0)      fy = rTY - 1 - ry;
    else if (flipType == 1) fx = rTX - 1 - rx;
    else if (flipType == 2) { fx = rTX - 1 - rx; fy = rTY - 1 - ry; }
    outTx = fx; outTy = fy; outTilesX = rTX; outTilesY = rTY;
}

// ============================================================
// 2x2x2 mean downsample helpers
// ============================================================

// 2x2x2 mean downsample from src tile into a sub-region of dst at (dstOffY, dstOffX).
template <typename T>
static void downsampleTileInto(const T* src, size_t srcZ, size_t srcY, size_t srcX,
                        T* dst, size_t dstZ, size_t dstY, size_t dstX,
                        size_t srcActualZ, size_t srcActualY, size_t srcActualX,
                        size_t dstOffY, size_t dstOffX)
{
    size_t halfZ = (srcActualZ + 1) / 2;
    size_t halfY = (srcActualY + 1) / 2;
    size_t halfX = (srcActualX + 1) / 2;
    for (size_t z = 0; z < halfZ; z++)
        for (size_t y = 0; y < halfY; y++)
            for (size_t x = 0; x < halfX; x++) {
                uint32_t sum = 0; int cnt = 0;
                for (int dz = 0; dz < 2 && (2*z+dz) < srcActualZ; dz++)
                    for (int dy = 0; dy < 2 && (2*y+dy) < srcActualY; dy++)
                        for (int dx = 0; dx < 2 && (2*x+dx) < srcActualX; dx++) {
                            sum += src[(2*z+dz)*srcY*srcX + (2*y+dy)*srcX + (2*x+dx)];
                            cnt++;
                        }
                size_t dz = z, dy2 = dstOffY + y, dx2 = dstOffX + x;
                if (dz < dstZ && dy2 < dstY && dx2 < dstX)
                    dst[dz * dstY * dstX + dy2 * dstX + dx2] = T((sum + cnt/2) / cnt);
            }
}

// ============================================================
// Graceful shutdown
// ============================================================

// Graceful shutdown: SIGINT/SIGTERM set this flag, render loop checks it
static std::atomic<bool> g_shutdownRequested{false};
static void shutdownHandler(int /*sig*/)
{
    g_shutdownRequested.store(true, std::memory_order_relaxed);
}

// ============================================================
// Logging infrastructure
// ============================================================

static FILE* g_logFile = nullptr;       // non-null when --log-path active
static std::string g_logPrefix;         // e.g. "[part 2/8] " — prepended when logging to file
static std::atomic<bool> g_logRunning{false};
static std::thread g_logFlushThread;

// Log to file if active, otherwise to the given default stream.
// When logging to a shared file in multi-part mode, each line is prefixed with the part id.
static void logPrintf(FILE* defaultStream, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
static void logPrintf(FILE* defaultStream, const char* fmt, ...)
{
    FILE* out = g_logFile ? g_logFile : defaultStream;
    if (g_logFile && !g_logPrefix.empty())
        std::fputs(g_logPrefix.c_str(), out);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(out, fmt, ap);
    va_end(ap);
}

// Start background flush thread (every ~5 seconds)
static void startLogFlusher()
{
    if (!g_logFile) return;
    g_logRunning = true;
    g_logFlushThread = std::thread([] {
        while (g_logRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (g_logFile) std::fflush(g_logFile);
        }
    });
}

static void stopLogFlusher()
{
    g_logRunning = false;
    if (g_logFlushThread.joinable()) g_logFlushThread.join();
    if (g_logFile) { std::fflush(g_logFile); std::fclose(g_logFile); g_logFile = nullptr; }
}

// ============================================================
// Affine transform utilities
// ============================================================

struct AffineTransform {
    cv::Mat_<double> matrix = cv::Mat_<double>::eye(4, 4);
};

static bool invertAffineInPlace(AffineTransform& T)
{
    cv::Mat A(3, 3, CV_64F), Ainv;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            A.at<double>(r, c) = T.matrix(r, c);
    if (cv::invert(A, Ainv, cv::DECOMP_LU) < 1e-10) return false;

    cv::Matx33d Ai;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Ai(r, c) = Ainv.at<double>(r, c);
    cv::Vec3d t(T.matrix(0,3), T.matrix(1,3), T.matrix(2,3));
    cv::Vec3d ti = -(Ai * t);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.matrix(r, c) = Ai(r, c);
        T.matrix(r, 3) = ti(r);
    }
    T.matrix(3,0) = T.matrix(3,1) = T.matrix(3,2) = 0.0;
    T.matrix(3,3) = 1.0;
    return true;
}

static AffineTransform loadAffineTransform(const std::string& filename)
{
    AffineTransform transform;
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open affine transform file: " + filename);
    try {
        json j; file >> j;
        if (j.contains("transformation_matrix")) {
            auto mat = j["transformation_matrix"];
            if (mat.size() != 3 && mat.size() != 4)
                throw std::runtime_error("Affine matrix must have 3 or 4 rows");
            for (int row = 0; row < (int)mat.size(); row++) {
                if (mat[row].size() != 4)
                    throw std::runtime_error("Each row must have 4 elements");
                for (int col = 0; col < 4; col++)
                    transform.matrix(row, col) = mat[row][col].get<double>();
            }
            if (mat.size() == 4) {
                if (std::abs(transform.matrix(3,0)) > 1e-12 ||
                    std::abs(transform.matrix(3,1)) > 1e-12 ||
                    std::abs(transform.matrix(3,2)) > 1e-12 ||
                    std::abs(transform.matrix(3,3) - 1.0) > 1e-12)
                    throw std::runtime_error("Bottom affine row must be [0,0,0,1]");
            }
        }
    } catch (json::parse_error&) {
        throw std::runtime_error("Error parsing affine transform file: " + filename);
    }
    return transform;
}

static AffineTransform composeAffine(const AffineTransform& A, const AffineTransform& B)
{
    AffineTransform R;
    cv::Mat tmp = B.matrix * A.matrix;
    tmp.copyTo(R.matrix);
    return R;
}

static void printMat4x4(const cv::Mat_<double>& M, const char* header)
{
    if (header) logPrintf(stdout, "%s\n", header);
    for (int r = 0; r < 4; ++r) {
        logPrintf(stdout, "  [%12.6f, %12.6f, %12.6f, %12.6f]\n",
                  M(r,0), M(r,1), M(r,2), M(r,3));
    }
}

static std::pair<std::string, bool> parseAffineSpec(const std::string& spec)
{
    for (const auto& suffix : {":inv", ":invert", ":i"}) {
        std::string s(suffix);
        if (spec.size() > s.size() && spec.substr(spec.size() - s.size()) == s)
            return {spec.substr(0, spec.size() - s.size()), true};
    }
    return {spec, false};
}

// ============================================================
// Geometry helpers
// ============================================================

static cv::Vec3f applyAffineToPoint(const cv::Vec3f& pt, const AffineTransform& T)
{
    double x = pt[0], y = pt[1], z = pt[2];
    return cv::Vec3f(
        float(T.matrix(0,0)*x + T.matrix(0,1)*y + T.matrix(0,2)*z + T.matrix(0,3)),
        float(T.matrix(1,0)*x + T.matrix(1,1)*y + T.matrix(1,2)*z + T.matrix(1,3)),
        float(T.matrix(2,0)*x + T.matrix(2,1)*y + T.matrix(2,2)*z + T.matrix(2,3)));
}

static void applyAffineTransform(cv::Mat_<cv::Vec3f>& points,
                                  cv::Mat_<cv::Vec3f>& normals,
                                  const AffineTransform& T)
{
    cv::Matx33d A(T.matrix(0,0), T.matrix(0,1), T.matrix(0,2),
                  T.matrix(1,0), T.matrix(1,1), T.matrix(1,2),
                  T.matrix(2,0), T.matrix(2,1), T.matrix(2,2));
    cv::Matx33d invAT = A.inv().t();

    for (int y = 0; y < points.rows; y++)
        for (int x = 0; x < points.cols; x++) {
            cv::Vec3f& pt = points(y, x);
            if (std::isnan(pt[0])) continue;
            pt = applyAffineToPoint(pt, T);
        }

    for (int y = 0; y < normals.rows; y++)
        for (int x = 0; x < normals.cols; x++) {
            cv::Vec3f& n = normals(y, x);
            if (std::isnan(n[0])) continue;
            double nx = invAT(0,0)*n[0] + invAT(0,1)*n[1] + invAT(0,2)*n[2];
            double ny = invAT(1,0)*n[0] + invAT(1,1)*n[1] + invAT(1,2)*n[2];
            double nz = invAT(2,0)*n[0] + invAT(2,1)*n[1] + invAT(2,2)*n[2];
            double len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 0) n = cv::Vec3f(float(nx/len), float(ny/len), float(nz/len));
        }
}

static cv::Vec3f calculateMeshCentroid(const cv::Mat_<cv::Vec3f>& points)
{
    cv::Vec3f sum(0,0,0); int count = 0;
    for (int y = 0; y < points.rows; y++)
        for (int x = 0; x < points.cols; x++) {
            const auto& pt = points(y, x);
            if (!std::isnan(pt[0])) { sum += pt; count++; }
        }
    return count > 0 ? sum / float(count) : sum;
}

static bool shouldFlipNormals(const cv::Mat_<cv::Vec3f>& points,
                               const cv::Mat_<cv::Vec3f>& normals,
                               const cv::Vec3f& ref)
{
    size_t toward = 0, away = 0;
    for (int y = 0; y < points.rows; y++)
        for (int x = 0; x < points.cols; x++) {
            const auto& pt = points(y, x);
            const auto& n = normals(y, x);
            if (std::isnan(pt[0]) || std::isnan(n[0])) continue;
            ((ref - pt).dot(n) > 0 ? toward : away)++;
        }
    return away > toward;
}

static void flipNormalsIf(cv::Mat_<cv::Vec3f>& normals, bool flip)
{
    if (!flip) return;
    for (int y = 0; y < normals.rows; y++)
        for (int x = 0; x < normals.cols; x++) {
            auto& n = normals(y, x);
            if (!std::isnan(n[0])) n = -n;
        }
}

static void normalizeNormals(cv::Mat_<cv::Vec3f>& nrm)
{
    for (int y = 0; y < nrm.rows; y++)
        for (int x = 0; x < nrm.cols; x++) {
            auto& v = nrm(y, x);
            if (std::isnan(v[0])) continue;
            float L = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            if (L > 0) v /= L;
        }
}

// ============================================================
// Rotation / flip helpers
// ============================================================

static int normalizeQuadrantRotation(double angleDeg, double tol = 0.5)
{
    double a = std::fmod(angleDeg, 360.0);
    if (a < 0) a += 360.0;
    for (int i = 0; i < 4; ++i)
        if (std::abs(a - i * 90.0) <= tol) return i;
    return -1;
}

static void rotateFlipIfNeeded(cv::Mat& m, int rotQuad, int flip_axis)
{
    if (rotQuad == 1)      cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (rotQuad == 2) cv::rotate(m, m, cv::ROTATE_180);
    else if (rotQuad == 3) cv::rotate(m, m, cv::ROTATE_90_CLOCKWISE);
    if (flip_axis == 0)      cv::flip(m, m, 0);
    else if (flip_axis == 1) cv::flip(m, m, 1);
    else if (flip_axis == 2) cv::flip(m, m, -1);
}

// ============================================================
// Surface generation helpers
// ============================================================

static void computeCanvasOrigin(const cv::Size& size, float& u0, float& v0)
{
    u0 = -0.5f * (size.width  - 1.0f);
    v0 = -0.5f * (size.height - 1.0f);
}

static void genTile(QuadSurface* surf, const cv::Size& size, float render_scale,
                    float u0, float v0, cv::Mat_<cv::Vec3f>& points, cv::Mat_<cv::Vec3f>& normals)
{
    surf->gen(&points, &normals, size, cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0, v0, 0));
}

static void prepareBaseAndDirs(const cv::Mat_<cv::Vec3f>& pts, const cv::Mat_<cv::Vec3f>& nrm,
                                float scale_seg, float ds_scale,
                                bool hasAffine, const AffineTransform& aff,
                                bool flipDecision,
                                cv::Mat_<cv::Vec3f>& base, cv::Mat_<cv::Vec3f>& dirs)
{
    base = pts.clone(); base *= scale_seg;
    dirs = nrm.clone();
    if (hasAffine) applyAffineTransform(base, dirs, aff);
    flipNormalsIf(dirs, flipDecision);
    normalizeNormals(dirs);
    base *= ds_scale;
}

static bool computeGlobalFlipDecision(QuadSurface* surf, int dx, int dy,
                                       float u0, float v0, float render_scale,
                                       float scale_seg, bool hasAffine,
                                       const AffineTransform& aff, cv::Vec3f& centroid)
{
    cv::Mat_<cv::Vec3f> tp, tn;
    surf->gen(&tp, &tn, cv::Size(dx, dy), cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0, v0, 0));
    tp *= scale_seg;
    if (hasAffine) applyAffineTransform(tp, tn, aff);
    centroid = calculateMeshCentroid(tp);
    return shouldFlipNormals(tp, tn, centroid);
}

// ============================================================
// Accumulation (max / mean / median) — templated for u8 & u16
// ============================================================

enum class AccumType { Max, Mean, Median, Alpha, BeerLambert };

template <typename T>
static cv::Mat accumulate(const std::vector<cv::Mat>& samples, size_t start, size_t count, AccumType type, int cvType)
{
    if (count == 1) return samples[start];
    if (type == AccumType::Max) {
        cv::Mat out = samples[start].clone();
        for (size_t j = 1; j < count; ++j) cv::max(out, samples[start + j], out);
        return out;
    }
    if (type == AccumType::Mean) {
        cv::Mat sum; samples[start].convertTo(sum, CV_64F);
        for (size_t j = 1; j < count; ++j) { cv::Mat t; samples[start+j].convertTo(t, CV_64F); sum += t; }
        sum /= double(count);
        cv::Mat out; sum.convertTo(out, cvType);
        return out;
    }
    // Median
    int rows = samples[start].rows, cols = samples[start].cols;
    cv::Mat out(rows, cols, cvType);
    std::vector<T> vals(count);
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++) {
            for (size_t i = 0; i < count; i++) vals[i] = samples[start + i].at<T>(y, x);
            std::sort(vals.begin(), vals.end());
            out.at<T>(y, x) = (count % 2 == 1)
                ? vals[count / 2]
                : T((uint32_t(vals[count/2 - 1]) + uint32_t(vals[count/2]) + 1) / 2);
        }
    return out;
}

// ============================================================
// Unified band rendering
// ============================================================

// Build the flat offset vector used by readMultiSlice
static std::vector<float> buildOffsetList(int numSlices, double sliceStep, double dsScale,
                                           const std::vector<float>& accumOffsets)
{
    double center = 0.5 * (std::max(1, numSlices) - 1.0);
    size_t sps = std::max<size_t>(1, accumOffsets.size());
    std::vector<float> out;
    out.reserve(numSlices * sps);
    for (int zi = 0; zi < numSlices; zi++) {
        float base = float((zi - center) * sliceStep);
        if (accumOffsets.empty())
            out.push_back(base * float(dsScale));
        else
            for (float ao : accumOffsets)
                out.push_back((base + ao) * float(dsScale));
    }
    return out;
}

// Process raw multi-slice results into final per-slice images (accumulation + rotate/flip).
// Returns one cv::Mat per output slice.
template <typename T>
static std::vector<cv::Mat> processRawSlices(std::vector<cv::Mat_<T>>& raw, int numSlices,
                                              const std::vector<float>& accumOffsets,
                                              AccumType accumType, int cvType,
                                              int rotQuad, int flipAxis)
{
    size_t sps = std::max<size_t>(1, accumOffsets.size());
    std::vector<cv::Mat> result(numSlices);
    for (int zi = 0; zi < numSlices; zi++) {
        if (accumOffsets.empty()) {
            result[zi] = raw[zi];
        } else {
            // Gather samples into a generic vector
            std::vector<cv::Mat> samps(raw.begin() + zi * sps, raw.begin() + zi * sps + sps);
            result[zi] = accumulate<T>(samps, 0, sps, accumType, cvType);
        }
        rotateFlipIfNeeded(result[zi], rotQuad, flipAxis);
    }
    return result;
}

// ============================================================
// SIMD sampler: inline multi-slice sampling via vc::simd::Sampler
// ============================================================

// Sample all slices for a tile/band region using vc::simd::Sampler.
// Replaces sampleTileSlices / readMultiSlice with compile-time tile strides.
template <typename T, int N>
static void sampleSlicesSimd(
    std::vector<cv::Mat_<T>>& raw,
    vc::simd::Sampler<T, N>& sampler,
    const cv::Mat_<cv::Vec3f>& base,
    const cv::Mat_<cv::Vec3f>& dirs,
    const std::vector<float>& offsets,
    SampleMode mode)
{
    using Geom = vc::simd::TileGeometry<N>;
    const int h = base.rows, w = base.cols;
    const int nSlices = static_cast<int>(offsets.size());

    raw.resize(nSlices);
    for (int si = 0; si < nSlices; ++si)
        raw[si].create(h, w);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const auto& b = base(y, x);
            const auto& d = dirs(y, x);

            // NaN check — invalid surface point
            if (std::isnan(b[0])) {
                for (int si = 0; si < nSlices; ++si)
                    raw[si](y, x) = T{};
                continue;
            }

            // Single-tile check: compute bounding chunk of first/last sample
            float vz0 = b[0] + d[0] * offsets[0];
            float vy0 = b[1] + d[1] * offsets[0];
            float vx0 = b[2] + d[2] * offsets[0];
            float vz1 = b[0] + d[0] * offsets[nSlices - 1];
            float vy1 = b[1] + d[1] * offsets[nSlices - 1];
            float vx1 = b[2] + d[2] * offsets[nSlices - 1];

            int tz0 = Geom::chunk_id(static_cast<int>(std::floor(std::min(vz0, vz1))));
            int tz1 = Geom::chunk_id(static_cast<int>(std::floor(std::max(vz0, vz1))) + 1);
            int ty0 = Geom::chunk_id(static_cast<int>(std::floor(std::min(vy0, vy1))));
            int ty1 = Geom::chunk_id(static_cast<int>(std::floor(std::max(vy0, vy1))) + 1);
            int tx0 = Geom::chunk_id(static_cast<int>(std::floor(std::min(vx0, vx1))));
            int tx1 = Geom::chunk_id(static_cast<int>(std::floor(std::max(vx0, vx1))) + 1);

            bool singleTile = (tz0 == tz1 && ty0 == ty1 && tx0 == tx1);

            if (singleTile && mode == SampleMode::Trilinear) {
                // FAST PATH: all slices in same tile, direct pointer math
                sampler.update_tile(tz0, ty0, tx0);
                const T* data = sampler.current_data();
                if (!data) {
                    for (int si = 0; si < nSlices; ++si)
                        raw[si](y, x) = T{};
                    continue;
                }

                for (int si = 0; si < nSlices; ++si) {
                    float vz = b[0] + d[0] * offsets[si];
                    float vy = b[1] + d[1] * offsets[si];
                    float vx = b[2] + d[2] * offsets[si];
                    int iz = static_cast<int>(std::floor(vz));
                    int iy = static_cast<int>(std::floor(vy));
                    int ix = static_cast<int>(std::floor(vx));
                    float fz = vz - iz, fy = vy - iy, fx = vx - ix;
                    int lz = Geom::local(iz), ly = Geom::local(iy), lx = Geom::local(ix);
                    float c000 = data[Geom::offset3d(lz,     ly,     lx)];
                    float c001 = data[Geom::offset3d(lz,     ly,     lx + 1)];
                    float c010 = data[Geom::offset3d(lz,     ly + 1, lx)];
                    float c011 = data[Geom::offset3d(lz,     ly + 1, lx + 1)];
                    float c100 = data[Geom::offset3d(lz + 1, ly,     lx)];
                    float c101 = data[Geom::offset3d(lz + 1, ly,     lx + 1)];
                    float c110 = data[Geom::offset3d(lz + 1, ly + 1, lx)];
                    float c111 = data[Geom::offset3d(lz + 1, ly + 1, lx + 1)];
                    float c00 = (1 - fx) * c000 + fx * c001;
                    float c01 = (1 - fx) * c010 + fx * c011;
                    float c10 = (1 - fx) * c100 + fx * c101;
                    float c11 = (1 - fx) * c110 + fx * c111;
                    float c0 = (1 - fy) * c00 + fy * c01;
                    float c1 = (1 - fy) * c10 + fy * c11;
                    raw[si](y, x) = static_cast<T>((1 - fz) * c0 + fz * c1);
                }
            } else {
                // SLOW PATH: per-sample via Sampler methods
                for (int si = 0; si < nSlices; ++si) {
                    float vz = b[0] + d[0] * offsets[si];
                    float vy = b[1] + d[1] * offsets[si];
                    float vx = b[2] + d[2] * offsets[si];
                    if (mode == SampleMode::Nearest)
                        raw[si](y, x) = sampler.sample_nearest(vz, vy, vx);
                    else
                        raw[si](y, x) = static_cast<T>(sampler.sample_trilinear_fast(vz, vy, vx));
                }
            }
        }
    }
}

// Runtime dispatch over chunk size N: creates SparseVolume<T,N> + Sampler<T,N>
// and calls sampleSlicesSimd. Thread-local sampler is stored in a thread_local
// unique_ptr keyed by vol pointer.
template <typename T, int N>
static void sampleSlicesDispatch(
    std::vector<cv::Mat_<T>>& raw,
    vc::zarr::Dataset* ds,
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& base,
    const cv::Mat_<cv::Vec3f>& dirs,
    const std::vector<float>& offsets,
    SampleMode mode,
    vc::simd::SparseVolume<T, N>& vol)
{
    // Thread-local sampler — each OMP thread gets its own
    thread_local std::unique_ptr<vc::simd::Sampler<T, N>> tlSampler;
    thread_local vc::simd::SparseVolume<T, N>* tlVol = nullptr;
    if (tlVol != &vol) {
        tlSampler = std::make_unique<vc::simd::Sampler<T, N>>(vol);
        tlVol = &vol;
    }
    sampleSlicesSimd<T, N>(raw, *tlSampler, base, dirs, offsets, mode);
}

// Render bands for a segmentation, calling writeSlices for each band.
// bandH should match the output chunk Y dimension so each chunk is written exactly once.
// WriteFn: void(const std::vector<cv::Mat>& slices, uint32_t bandIdx, uint32_t bandY0)
template <typename T, int N, typename WriteFn>
static void renderBands(
    QuadSurface* surf, vc::zarr::Dataset* ds,
    ChunkCache<T>* cache,
    vc::simd::SparseVolume<T, N>& vol,
    SampleMode sampleMode,
    const cv::Size& fullSize, const cv::Rect& crop, const cv::Size& tgtSize,
    float renderScale, float scaleSeg, float dsScale,
    bool hasAffine, const AffineTransform& aff,
    int numSlices, double sliceStep,
    const std::vector<float>& accumOffsets, AccumType accumType,
    bool isComposite, int compositeStart, int compositeEnd,
    const CompositeParams& compositeParams,
    int rotQuad, int flipAxis,
    int numParts, int partId,
    int cvType,
    uint32_t bandH,
    WriteFn&& writeSlices)
{
    const uint32_t numBands = (uint32_t(tgtSize.height) + bandH - 1) / bandH;

    // Global flip decision
    cv::Vec3f centroid;
    float u0b, v0b; computeCanvasOrigin(fullSize, u0b, v0b);
    u0b += float(crop.x); v0b += float(crop.y);
    bool globalFlip = computeGlobalFlipDecision(
        surf, std::min(128, tgtSize.width), std::min(128, tgtSize.height),
        u0b, v0b, renderScale, scaleSeg, hasAffine, aff, centroid);

    // Build offset list for readMultiSlice
    auto allOffsets = buildOffsetList(numSlices, sliceStep, dsScale, accumOffsets);

    auto wallStart = std::chrono::steady_clock::now();
    auto lastPrint = wallStart;

    // Contiguous block assignment: each part gets a contiguous range of bands
    // for better spatial locality (avoids all VMs reading the same volume region)
    uint32_t bandsPerPart = (numBands + uint32_t(numParts) - 1) / uint32_t(numParts);
    uint32_t bandStart = uint32_t(partId) * bandsPerPart;
    uint32_t bandEnd = std::min(bandStart + bandsPerPart, numBands);

    for (uint32_t bi = bandStart; bi < bandEnd; bi++) {
        uint32_t y0 = bi * bandH;
        uint32_t dy = std::min(bandH, uint32_t(tgtSize.height) - y0);

        // Generate surface for this band
        float u0, v0; computeCanvasOrigin(fullSize, u0, v0);
        u0 += float(crop.x); v0 += float(crop.y) + float(y0);
        cv::Mat_<cv::Vec3f> bandPts, bandNrm;
        genTile(surf, cv::Size(tgtSize.width, int(dy)), renderScale, u0, v0, bandPts, bandNrm);

        cv::Mat_<cv::Vec3f> base, dirs;
        prepareBaseAndDirs(bandPts, bandNrm, scaleSeg, dsScale, hasAffine, aff, globalFlip, base, dirs);

        std::vector<cv::Mat> slices;

        if (isComposite) {
            // Composite mode: always u8 — callers always instantiate with T=uint8_t
            cv::Mat_<uint8_t> compOut;
            if constexpr (std::is_same_v<T, uint8_t>) {
                readCompositeFast(compOut, ds, base, dirs,
                                  float(sliceStep * dsScale),
                                  compositeStart, compositeEnd,
                                  compositeParams, *cache);
            }
            cv::Mat s = compOut;
            rotateFlipIfNeeded(s, rotQuad, flipAxis);
            slices = {s};
        } else {
            // Normal: bulk read + accumulate via simd Sampler
            std::vector<cv::Mat_<T>> raw;
            sampleSlicesDispatch<T, N>(raw, ds, cache, base, dirs, allOffsets, sampleMode, vol);
            slices = processRawSlices<T>(raw, numSlices, accumOffsets, accumType, cvType, rotQuad, flipAxis);
        }

        writeSlices(slices, bi, y0);

        // Progress (throttled to ~1/sec)
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - lastPrint).count();
        uint32_t bandsThis = bandEnd - bandStart;
        uint32_t done = bi - bandStart + 1;
        if (since >= 1.0 || done == bandsThis) {
            lastPrint = now;
            double elapsed = std::chrono::duration<double>(now - wallStart).count();
            double eta = done > 0 ? elapsed * (double(bandsThis) / done - 1.0) : 0.0;
            double bandsPerSec = elapsed > 0 ? done / elapsed : 0.0;
            auto cs = cache->stats();
            uint64_t tot = cs.hits + cs.misses;
            double hr = tot > 0 ? 100.0 * cs.hits / tot : 0.0;
            double gbR = cs.bytesRead / (1024.0*1024.0*1024.0);
            const char* prefix = g_logFile ? "  " : "\r  ";
            const char* suffix = g_logFile ? "\n" : "";
            logPrintf(stderr, "%sband %u/%u (%d%%)  %.1f bands/s  %dm%02ds  eta %dm%02ds  cache %.1f%% evict %lu read %.2fGB%s",
                prefix, done, bandsThis, int(100.0 * done / bandsThis),
                bandsPerSec,
                int(elapsed)/60, int(elapsed)%60, int(eta)/60, int(eta)%60,
                hr, (unsigned long)cs.evictions, gbR, suffix);
        }
    }
    if (!g_logFile) std::fprintf(stderr, "\n");
}

// ============================================================
// TIFF band writer helper
// ============================================================

static void writeTifBand(std::vector<TiffWriter>& writers,
                          const std::vector<cv::Mat>& slices,
                          uint32_t bandY0, uint32_t tiffTileH,
                          uint32_t srcHeight, int rotQuad, int flipAxis)
{
    uint32_t numTiffTiles = (srcHeight + tiffTileH - 1) / tiffTileH;
    for (size_t zi = 0; zi < slices.size(); zi++) {
        for (uint32_t ty = 0; ty < uint32_t(slices[zi].rows); ty += tiffTileH) {
            uint32_t tdy = std::min(tiffTileH, uint32_t(slices[zi].rows) - ty);
            cv::Mat sub = slices[zi](cv::Rect(0, ty, slices[zi].cols, tdy));
            int dstBx, dstBy, rBX, rBY;
            uint32_t srcTileIdx = (bandY0 + ty) / tiffTileH;
            mapTileIndex(0, int(srcTileIdx), 1, int(numTiffTiles),
                         std::max(rotQuad, 0), flipAxis, dstBx, dstBy, rBX, rBY);
            writers[zi].writeTile(0, uint32_t(dstBy) * tiffTileH, sub);
        }
    }
}


// ============================================================
// Tile-based rendering (OMP-parallel over output zarr chunks)
// ============================================================

template <typename T, int N>
static void renderTiles(
    QuadSurface* surf, vc::zarr::Dataset* ds,
    ChunkCache<T>* cache,
    vc::simd::SparseVolume<T, N>& vol,
    SampleMode sampleMode,
    const cv::Size& fullSize, const cv::Rect& crop, const cv::Size& tgtSize,
    float renderScale, float scaleSeg, float dsScale,
    bool hasAffine, const AffineTransform& aff,
    int numSlices, double sliceStep,
    const std::vector<float>& accumOffsets, AccumType accumType,
    bool isComposite, int compositeStart, int compositeEnd,
    const CompositeParams& compositeParams,
    int rotQuad, int flipAxis,
    int numParts, int partId,
    int cvType,
    // Zarr output
    vc::zarr::Dataset* dsOut, const std::vector<size_t>& chunks0,
    size_t tilesXSrc, size_t tilesYSrc,
    // Pyramid datasets L1-L5 (empty = no inline pyramid)
    const std::vector<vc::zarr::Dataset*>& pyramidDs,
    // TIF output (optional)
    std::vector<TiffWriter>* tifWriters, uint32_t tiffTileH,
    bool quickTif,
    bool resume)
{
    const size_t CH = chunks0[1], CW = chunks0[2];
    const uint32_t numTileRows = static_cast<uint32_t>(tilesYSrc);
    const uint32_t numTileCols = static_cast<uint32_t>(tilesXSrc);

    // Pre-warm validMask cache (thread-safety: subsequent calls are read-only)
    surf->validMask();

    // Global flip decision
    cv::Vec3f centroid;
    float u0b, v0b; computeCanvasOrigin(fullSize, u0b, v0b);
    u0b += float(crop.x); v0b += float(crop.y);
    bool globalFlip = computeGlobalFlipDecision(
        surf, std::min(128, tgtSize.width), std::min(128, tgtSize.height),
        u0b, v0b, renderScale, scaleSeg, hasAffine, aff, centroid);

    // Build offset list
    auto allOffsets = buildOffsetList(numSlices, sliceStep, dsScale, accumOffsets);

    const bool wantTif = tifWriters && !tifWriters->empty();
    const bool useU16 = (cvType == CV_16UC1);

    auto wallStart = std::chrono::steady_clock::now();
    auto lastPrint = wallStart;

    // Contiguous block assignment: each part gets a contiguous range of tile rows
    // for better spatial locality (avoids all VMs reading the same volume region)
    uint32_t rowsPerPart = (numTileRows + uint32_t(numParts) - 1) / uint32_t(numParts);
    // Align to 32 tile-rows for multi-VM pyramid (2^5 = 32 covers L1..L5)
    if (numParts > 1 && !pyramidDs.empty()) {
        constexpr uint32_t kPyrAlign = 32;
        rowsPerPart = ((rowsPerPart + kPyrAlign - 1) / kPyrAlign) * kPyrAlign;
    }
    uint32_t tyStart = uint32_t(partId) * rowsPerPart;
    uint32_t tyEnd = std::min(tyStart + rowsPerPart, numTileRows);

    // ---- Pyramid accumulation buffers (L1..L5) ----
    // At level L, one 128×128 pyramid chunk covers 2^L × 2^L L0 tiles.
    // pyrAccum[li] has one buffer per pyramid chunk column for the current row-group.
    struct PyrAccumLevel {
        size_t chZ, chY, chX;          // chunk dims at this level
        size_t pyrChunksX;             // number of pyramid chunk columns
        std::vector<std::vector<T>> bufs;  // one per pyramid chunk column
    };
    std::vector<PyrAccumLevel> pyrAccum;
    if (!pyramidDs.empty()) {
        pyrAccum.resize(pyramidDs.size());
        for (size_t li = 0; li < pyramidDs.size(); li++) {
            const auto& pc = pyramidDs[li]->chunkShape();
            pyrAccum[li].chZ = pc[0];
            pyrAccum[li].chY = pc[1];
            pyrAccum[li].chX = pc[2];
            // Number of pyramid chunk columns at this level
            const auto& ps = pyramidDs[li]->shape();
            pyrAccum[li].pyrChunksX = (ps[2] + pc[2] - 1) / pc[2];
            pyrAccum[li].bufs.resize(pyrAccum[li].pyrChunksX);
            for (auto& b : pyrAccum[li].bufs)
                b.assign(pc[0] * pc[1] * pc[2], T(0));
        }
    }

    for (uint32_t ty = tyStart; ty < tyEnd; ty++) {
        uint32_t y0 = ty * uint32_t(CH);
        uint32_t dy = std::min(uint32_t(CH), uint32_t(tgtSize.height) - y0);

        // TIF row buffer: one tile-column's worth of slices per tx
        // tifRowBuf[tx] = vector of cv::Mat (one per output slice), each 128xdx
        std::vector<std::vector<cv::Mat>> tifRowBuf;
        if (wantTif) tifRowBuf.resize(numTileCols);

        #pragma omp parallel for schedule(dynamic)
        for (uint32_t tx = 0; tx < numTileCols; tx++) {
            // Resume: skip tile if L0 chunk already exists on disk
            if (resume) {
                const bool needsRotFlip = (rotQuad >= 0 || flipAxis >= 0);
                int dTx = int(tx), dTy = int(ty), dTX, dTY;
                if (needsRotFlip)
                    mapTileIndex(int(tx), int(ty), int(tilesXSrc), int(tilesYSrc),
                                 std::max(rotQuad, 0), flipAxis, dTx, dTy, dTX, dTY);
                vc::zarr::ShapeType cid = {0, size_t(dTy), size_t(dTx)};
                if (dsOut->chunkExists(cid)) {
                    // Still need to scatter into pyramid accum buffers
                    // No rotation when inline pyramid is active
                    if (!pyrAccum.empty()) {
                        size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];
                        std::vector<T> existingBuf(chunkZ * chunkY * chunkX, T(0));
                        dsOut->readChunk(cid, existingBuf.data());
                        uint32_t dxTile = std::min(uint32_t(CW), uint32_t(tgtSize.width) - tx * uint32_t(CW));
                        size_t dy_actual = std::min(chunkY, size_t(dy));
                        size_t dx_actual = std::min(chunkX, size_t(dxTile));
                        size_t numZ = isComposite ? 1 : size_t(std::max(1, numSlices));
                        size_t l1cx = size_t(tx) >> 1;
                        size_t halfCH = CH / 2, halfCW = CW / 2;
                        size_t offY = (size_t(ty) & 1) * halfCH;
                        size_t offX = (size_t(tx) & 1) * halfCW;
                        auto& pa = pyrAccum[0];
                        if (l1cx < pa.bufs.size()) {
                            downsampleTileInto(
                                existingBuf.data(), chunkZ, chunkY, chunkX,
                                pa.bufs[l1cx].data(), pa.chZ, pa.chY, pa.chX,
                                numZ, dy_actual, dx_actual,
                                offY, offX);
                        }
                    }
                    continue;
                }
            }

            uint32_t x0 = tx * uint32_t(CW);
            uint32_t dx = std::min(uint32_t(CW), uint32_t(tgtSize.width) - x0);

            // 1. Generate surface for this tile
            float u0, v0; computeCanvasOrigin(fullSize, u0, v0);
            u0 += float(crop.x) + float(x0);
            v0 += float(crop.y) + float(y0);
            cv::Mat_<cv::Vec3f> tilePts, tileNrm;
            genTile(surf, cv::Size(int(dx), int(dy)), renderScale, u0, v0, tilePts, tileNrm);

            // 2. Prepare base coords and step directions
            cv::Mat_<cv::Vec3f> base, dirs;
            prepareBaseAndDirs(tilePts, tileNrm, scaleSeg, dsScale, hasAffine, aff, globalFlip, base, dirs);

            // 3. Sample all slices for this tile (single-threaded)
            std::vector<cv::Mat_<T>> raw;
            if (isComposite) {
                if constexpr (std::is_same_v<T, uint8_t>) {
                    cv::Mat_<uint8_t> compOut;
                    readCompositeFast(compOut, ds, base, dirs,
                                      float(sliceStep * dsScale),
                                      compositeStart, compositeEnd,
                                      compositeParams, *cache);
                    raw.resize(1);
                    raw[0] = compOut;
                }
            } else {
                sampleSlicesDispatch<T, N>(raw, ds, cache, base, dirs, allOffsets, sampleMode, vol);
            }

            // Accumulate (no rotation — applied per-zarr-chunk and per-tif-band separately)
            std::vector<cv::Mat> slices = processRawSlices<T>(raw, numSlices, accumOffsets, accumType, cvType, -1, -1);

            // 4. Pack into zarr chunk (with rotation applied to pixel data)
            {
                size_t chunkZ = chunks0[0], chunkY = chunks0[1], chunkX = chunks0[2];
                size_t numZ = slices.size();
                const bool needsRotFlip = (rotQuad >= 0 || flipAxis >= 0);

                // Only clone+rotate when rotation/flip is active
                const std::vector<cv::Mat>* zarrSlices = &slices;
                std::vector<cv::Mat> rotSlices;
                if (needsRotFlip) {
                    rotSlices.resize(numZ);
                    for (size_t zi = 0; zi < numZ; zi++) {
                        rotSlices[zi] = slices[zi].clone();
                        rotateFlipIfNeeded(rotSlices[zi], rotQuad, flipAxis);
                    }
                    zarrSlices = &rotSlices;
                }

                int dstTx = int(tx), dstTy = int(ty), dTX, dTY;
                if (needsRotFlip)
                    mapTileIndex(int(tx), int(ty), int(tilesXSrc), int(tilesYSrc),
                                 std::max(rotQuad, 0), flipAxis, dstTx, dstTy, dTX, dTY);

                std::vector<T> chunkBuf(chunkZ * chunkY * chunkX, T(0));
                size_t dy_actual = std::min(chunkY, size_t((*zarrSlices)[0].rows));
                size_t dx_actual = std::min(chunkX, size_t((*zarrSlices)[0].cols));
                for (size_t zi = 0; zi < numZ; zi++) {
                    size_t sliceOff = zi * chunkY * chunkX;
                    for (size_t yy = 0; yy < dy_actual; yy++) {
                        const T* row = (*zarrSlices)[zi].ptr<T>(int(yy));
                        std::memcpy(&chunkBuf[sliceOff + yy * chunkX], row, dx_actual * sizeof(T));
                    }
                }
                vc::zarr::ShapeType chunkId = {0, size_t(dstTy), size_t(dstTx)};
                dsOut->writeChunk(chunkId, chunkBuf.data());

                // Scatter L0 tile into L1 pyramid accumulation buffer
                // No rotation when inline pyramid is active, so tx/ty == dstTx/dstTy.
                // L0 tile (ty, tx) maps to L1 pyramid chunk (ty/2, tx/2)
                // at sub-offset ((ty%2)*halfCH, (tx%2)*halfCW) within the L1 chunk
                if (!pyrAccum.empty()) {
                    size_t l1cx = size_t(tx) >> 1;
                    size_t halfCH = CH / 2, halfCW = CW / 2;
                    size_t offY = (size_t(ty) & 1) * halfCH;
                    size_t offX = (size_t(tx) & 1) * halfCW;
                    auto& pa = pyrAccum[0];
                    if (l1cx < pa.bufs.size()) {
                        downsampleTileInto(
                            chunkBuf.data(), chunkZ, chunkY, chunkX,
                            pa.bufs[l1cx].data(), pa.chZ, pa.chY, pa.chX,
                            numZ, dy_actual, dx_actual,
                            offY, offX);
                    }
                }
            }

            // 5. Store unrotated slices for TIF assembly
            if (wantTif) {
                tifRowBuf[tx] = std::move(slices);
            }
        }

        // After all tx done for this ty: assemble TIF if needed
        if (wantTif) {
            int outSlices = isComposite ? 1 : numSlices;
            // Assemble full-width unrotated rows from tile columns
            std::vector<cv::Mat> fullWidthSlices(outSlices);
            for (int zi = 0; zi < outSlices; zi++) {
                fullWidthSlices[zi] = cv::Mat(int(dy), tgtSize.width, cvType, cv::Scalar(0));
                for (uint32_t tx = 0; tx < numTileCols; tx++) {
                    uint32_t x0t = tx * uint32_t(CW);
                    const cv::Mat& tileMat = tifRowBuf[tx][zi];
                    tileMat.copyTo(fullWidthSlices[zi](cv::Rect(int(x0t), 0, tileMat.cols, tileMat.rows)));
                }
                // Apply rotation/flip to assembled full-width row
                rotateFlipIfNeeded(fullWidthSlices[zi], rotQuad, flipAxis);
            }

            if (quickTif && !useU16) {
                for (auto& s : fullWidthSlices)
                    for (int r = 0; r < s.rows; r++) {
                        auto* row = s.ptr<uint8_t>(r);
                        for (int c = 0; c < s.cols; c++)
                            row[c] &= 0xF0;
                    }
            }
            writeTifBand(*tifWriters, fullWidthSlices, y0, tiffTileH, uint32_t(tgtSize.height), rotQuad, flipAxis);
        }

        // ---- Pyramid row-group flush ----
        // At each level L (1..5), a pyramid chunk covers 2^L tile rows.
        // When the current tile-row completes a group, flush & cascade.
        if (!pyrAccum.empty()) {
            // No rotation/flip when inline pyramid is active (ensured by caller),
            // so tile-row ty maps directly to dest tile-row ty.
            uint32_t relRow = ty - tyStart + 1;
            bool isLastRow = (ty == tyEnd - 1);

            for (size_t li = 0; li < pyrAccum.size(); li++) {
                uint32_t groupSize = 1u << (li + 1);  // L1=2, L2=4, L3=8, L4=16, L5=32
                bool groupComplete = (relRow % groupSize == 0) || isLastRow;
                if (!groupComplete) continue;

                auto& pa = pyrAccum[li];
                // tile-row ty divided by 2^(li+1) gives the pyramid chunk row
                size_t pyrChunkRow = size_t(ty) >> (li + 1);

                // Write each column's accumulation buffer as a zarr chunk
                for (size_t cx = 0; cx < pa.bufs.size(); cx++) {
                    vc::zarr::ShapeType chunkId = {0, pyrChunkRow, cx};
                    pyramidDs[li]->writeChunk(chunkId, pa.bufs[cx].data());
                }

                // Cascade: scatter this level's buffers into the next level's accum
                if (li + 1 < pyrAccum.size()) {
                    auto& nextPa = pyrAccum[li + 1];
                    size_t halfY = pa.chY / 2, halfX = pa.chX / 2;
                    for (size_t cx = 0; cx < pa.bufs.size(); cx++) {
                        size_t nextCx = cx >> 1;
                        size_t offY = (pyrChunkRow & 1) * halfY;
                        size_t offX = (cx & 1) * halfX;
                        if (nextCx < nextPa.bufs.size()) {
                            downsampleTileInto(
                                pa.bufs[cx].data(), pa.chZ, pa.chY, pa.chX,
                                nextPa.bufs[nextCx].data(), nextPa.chZ, nextPa.chY, nextPa.chX,
                                pa.chZ, pa.chY, pa.chX,
                                offY, offX);
                        }
                    }
                }

                // Clear this level's buffers for next group
                for (auto& b : pa.bufs)
                    std::fill(b.begin(), b.end(), T(0));
            }
        }

        // Progress (throttled to ~1/sec)
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - lastPrint).count();
        uint32_t tileRowsThis = tyEnd - tyStart;
        uint32_t done = ty - tyStart + 1;
        if (since >= 1.0 || done == tileRowsThis) {
            lastPrint = now;
            double elapsed = std::chrono::duration<double>(now - wallStart).count();
            double eta = done > 0 ? elapsed * (double(tileRowsThis) / done - 1.0) : 0.0;
            // Chunks written this part: done tile-rows × numTileCols chunks per row
            double chunksWritten = double(done) * double(numTileCols);
            double chunksPerSec = elapsed > 0 ? chunksWritten / elapsed : 0.0;
            auto cs = cache->stats();
            uint64_t tot = cs.hits + cs.misses;
            double hr = tot > 0 ? 100.0 * cs.hits / tot : 0.0;
            double gbR = cs.bytesRead / (1024.0*1024.0*1024.0);
            const char* prefix = g_logFile ? "  " : "\r  ";
            const char* suffix = g_logFile ? "\n" : "";
            logPrintf(stderr, "%stile-row %u/%u (%d%%)  %.1f chunks/s  %dm%02ds  eta %dm%02ds  cache %.1f%% evict %lu read %.2fGB%s",
                prefix, done, tileRowsThis, int(100.0 * done / tileRowsThis),
                chunksPerSec,
                int(elapsed)/60, int(elapsed)%60, int(eta)/60, int(eta)%60,
                hr, (unsigned long)cs.evictions, gbR, suffix);
        }
    }
    if (!g_logFile) std::fprintf(stderr, "\n");
}


// ============================================================
// main
// ============================================================

int main(int argc, char *argv[])
{
    // clang-format off
    po::options_description required("Required arguments");
    required.add_options()
        ("volume,v", po::value<std::string>()->required(), "Path to the OME-Zarr volume")
        ("scale", po::value<float>()->required(), "Pixels per level-g voxel (Pg)")
        ("group-idx,g", po::value<int>()->required(), "OME-Zarr group index");

    po::options_description optional("Optional arguments");
    optional.add_options()
        ("help,h", "Show this help message")
        ("segmentation,s", po::value<std::string>(), "Path to a single tifxyz segmentation folder")
        ("cache-gb", po::value<size_t>()->default_value(16), "Zarr chunk cache size in GB")
        ("log-path", po::value<std::string>(), "Log all output to file instead of stdout/stderr")
        ("timeout", po::value<int>()->default_value(0), "Kill process if not finished within N minutes")
        ("num-slices,n", po::value<int>()->default_value(1), "Number of slices to render")
        ("slice-step", po::value<float>()->default_value(1.0f), "Spacing between slices along normal")
        ("accum", po::value<float>()->default_value(0.0f), "Accumulation sub-step (0 = disabled)")
        ("accum-type", po::value<std::string>()->default_value("max"), "Reducer: max, mean, median, alpha, beerlambert")
        ("crop-x", po::value<int>()->default_value(0), "Crop X") ("crop-y", po::value<int>()->default_value(0), "Crop Y")
        ("crop-width", po::value<int>()->default_value(0), "Crop width") ("crop-height", po::value<int>()->default_value(0), "Crop height")
        ("auto-crop", po::bool_switch()->default_value(false), "Auto-crop to valid surface bbox")
        ("affine", po::value<std::vector<std::string>>()->multitoken()->composing(), "Affine JSON files (append :inv to invert)")
        ("affine-invert", po::value<std::vector<int>>()->multitoken()->composing(), "0-based indices to invert")
        ("affine-transform", po::value<std::string>(), "[DEPRECATED] Single affine JSON")
        ("invert-affine", po::bool_switch()->default_value(false), "[DEPRECATED] Invert single affine")
        ("scale-segmentation", po::value<float>()->default_value(1.0), "Scale segmentation")
        ("rotate", po::value<double>()->default_value(0.0), "Rotate output (0/90/180/270)")
        ("flip", po::value<int>()->default_value(-1), "Flip: 0=V, 1=H, 2=Both")
        ("zarr-output", po::value<std::string>(), "Output path for .zarr (optional)")
        ("tif-output", po::value<std::string>(), "Output path for per-slice TIFFs (optional)")
        ("quick-tif", po::bool_switch()->default_value(false), "Fast TIF: PACKBITS + zero low nibble")
        ("flatten", po::bool_switch()->default_value(false), "ABF++ flattening")
        ("flatten-iterations", po::value<int>()->default_value(10), "ABF++ iterations")
        ("flatten-downsample", po::value<int>()->default_value(1), "ABF++ downsample factor")
        ("alpha-min", po::value<float>()->default_value(0.0f), "Alpha min (0-255)")
        ("alpha-max", po::value<float>()->default_value(255.0f), "Alpha max (0-255)")
        ("alpha-opacity", po::value<float>()->default_value(230.0f), "Alpha opacity (0-255)")
        ("alpha-cutoff", po::value<float>()->default_value(9950.0f), "Alpha cutoff (0-10000)")
        ("bl-extinction", po::value<float>()->default_value(1.5f), "Beer-Lambert extinction")
        ("bl-emission", po::value<float>()->default_value(1.5f), "Beer-Lambert emission")
        ("bl-ambient", po::value<float>()->default_value(0.1f), "Beer-Lambert ambient")
        ("iso-cutoff", po::value<int>()->default_value(0), "Highpass (0-255)")
        ("composite-start", po::value<int>(), "Composite start offset")
        ("composite-end", po::value<int>(), "Composite end offset")
        ("num-parts", po::value<int>()->default_value(1), "Parts for multi-VM")
        ("part-id", po::value<int>()->default_value(0), "Part ID (0-indexed)")
        ("merge-tiff-parts", po::bool_switch()->default_value(false), "Merge partial TIFFs from multi-VM render")
        ("pyramid", po::value<bool>()->default_value(true), "Build pyramid levels L1-L5 (default: true)")
        ("resume", po::bool_switch()->default_value(false), "Skip chunks that already exist on disk")
        ("pre", po::bool_switch()->default_value(false), "Create zarr + all level datasets")
        ("zarr-v3", po::bool_switch()->default_value(false),
            "Write Zarr v3 output with sharding (128^3 shards, 32^3 inner chunks)")
        ("sample-mode", po::value<std::string>()->default_value("trilinear"),
            "Sampling: nearest, trilinear");
    // clang-format on

    po::options_description all("Usage");
    all.add(required).add(optional);
    po::variables_map parsed;
    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);
        if (parsed.count("help") || argc < 2) {
            std::cout << "vc_render_tifxyz: Render volume data using segmentation surfaces\n\n" << all << '\n';
            return EXIT_SUCCESS;
        }
        po::notify(parsed);
    } catch (po::error& e) {
        std::cerr << "Error: " << e.what() << "\nUse --help for usage information\n";
        return EXIT_FAILURE;
    }

    // --- Log path setup ---
    if (parsed.count("log-path")) {
        const auto& logPath = parsed["log-path"].as<std::string>();
        g_logFile = std::fopen(logPath.c_str(), "a");
        if (!g_logFile) {
            std::cerr << "Error: cannot open log file: " << logPath << "\n";
            return EXIT_FAILURE;
        }
        startLogFlusher();
    }

    // --- Parse common options ---
    const int numParts = parsed["num-parts"].as<int>();
    const int partId = parsed["part-id"].as<int>();
    if (numParts < 1 || partId < 0 || partId >= numParts) {
        logPrintf(stderr, "Error: need 0 <= part-id < num-parts\n"); return EXIT_FAILURE;
    }
    if (g_logFile && numParts > 1) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "[part %d/%d] ", partId, numParts);
        g_logPrefix = buf;
    }
    if (numParts > 1) logPrintf(stdout, "Multi-part mode: part %d of %d\n", partId, numParts);

    const bool pre_flag = parsed["pre"].as<bool>();
    const bool wantPyramid = parsed["pyramid"].as<bool>();
    const bool resumeFlag = parsed["resume"].as<bool>();

    // Parse sample mode
    SampleMode sampleMode = SampleMode::Trilinear;
    {
        std::string sm = parsed["sample-mode"].as<std::string>();
        std::transform(sm.begin(), sm.end(), sm.begin(),
                       [](unsigned char c){ return char(std::tolower(c)); });
        if (sm == "nearest") sampleMode = SampleMode::Nearest;
        else if (sm == "trilinear") sampleMode = SampleMode::Trilinear;
        else { logPrintf(stderr, "Error: invalid --sample-mode '%s' (use nearest or trilinear)\n", sm.c_str()); return EXIT_FAILURE; }
    }

    const bool wantZarr = parsed.count("zarr-output") > 0;
    const bool wantTif = parsed.count("tif-output") > 0;
    if (!wantZarr && !wantTif) { logPrintf(stderr, "Error: at least one of --zarr-output or --tif-output required\n"); return EXIT_FAILURE; }
    const std::string zarrOutputArg = wantZarr ? parsed["zarr-output"].as<std::string>() : "";
    const std::string tifOutputArg = wantTif ? parsed["tif-output"].as<std::string>() : "";

    const bool mergeTiffFlag = parsed["merge-tiff-parts"].as<bool>();
    if (mergeTiffFlag) {
        if (numParts < 2) { logPrintf(stderr, "Error: --merge-tiff-parts needs --num-parts >= 2\n"); return EXIT_FAILURE; }
        if (!wantTif) { logPrintf(stderr, "Error: --merge-tiff-parts requires --tif-output\n"); return EXIT_FAILURE; }
        return mergeTiffParts(tifOutputArg, numParts) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::filesystem::path vol_path = parsed["volume"].as<std::string>();

    if (!parsed.count("segmentation")) { logPrintf(stderr, "Error: --segmentation required\n"); return EXIT_FAILURE; }
    std::filesystem::path seg_path = parsed["segmentation"].as<std::string>();

    float tgt_scale = parsed["scale"].as<float>();
    int group_idx = parsed["group-idx"].as<int>();
    int num_slices = parsed["num-slices"].as<int>();
    double slice_step = parsed["slice-step"].as<float>();
    if (!std::isfinite(slice_step) || slice_step <= 0) { logPrintf(stderr, "Error: --slice-step must be positive\n"); return EXIT_FAILURE; }

    double accum_step = parsed["accum"].as<float>();
    if (!std::isfinite(accum_step) || accum_step < 0) { logPrintf(stderr, "Error: --accum must be non-negative\n"); return EXIT_FAILURE; }

    std::string accum_type_str = parsed["accum-type"].as<std::string>();
    std::transform(accum_type_str.begin(), accum_type_str.end(), accum_type_str.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });

    AccumType accumType;
    if      (accum_type_str == "max")    accumType = AccumType::Max;
    else if (accum_type_str == "mean")   accumType = AccumType::Mean;
    else if (accum_type_str == "median") accumType = AccumType::Median;
    else if (accum_type_str == "alpha")  accumType = AccumType::Alpha;
    else if (accum_type_str == "beerlam" || accum_type_str == "beerlambert") accumType = AccumType::BeerLambert;
    else { logPrintf(stderr, "Error: invalid --accum-type\n"); return EXIT_FAILURE; }

    const bool isCompositeMode = (accumType == AccumType::Alpha || accumType == AccumType::BeerLambert);
    int compositeStart = 0, compositeEnd = num_slices - 1;

    CompositeParams compositeParams;
    if (isCompositeMode) {
        compositeParams.method = (accumType == AccumType::Alpha) ? "alpha" : "beerLambert";
        compositeParams.alphaMin = parsed["alpha-min"].as<float>() / 255.0f;
        compositeParams.alphaMax = parsed["alpha-max"].as<float>() / 255.0f;
        compositeParams.alphaOpacity = parsed["alpha-opacity"].as<float>() / 255.0f;
        compositeParams.alphaCutoff = parsed["alpha-cutoff"].as<float>() / 10000.0f;
        compositeParams.blExtinction = parsed["bl-extinction"].as<float>();
        compositeParams.blEmission = parsed["bl-emission"].as<float>();
        compositeParams.blAmbient = parsed["bl-ambient"].as<float>();
        compositeParams.isoCutoff = uint8_t(std::clamp(parsed["iso-cutoff"].as<int>(), 0, 255));
        if (parsed.count("composite-start")) compositeStart = parsed["composite-start"].as<int>();
        if (parsed.count("composite-end"))   compositeEnd = parsed["composite-end"].as<int>();
        if (compositeEnd < compositeStart) { logPrintf(stderr, "Error: --composite-end < --composite-start\n"); return EXIT_FAILURE; }
        int layers = compositeEnd - compositeStart + 1;
        logPrintf(stdout, "Composite: %s (%d layers [%d..%d])\n", compositeParams.method.c_str(), layers, compositeStart, compositeEnd);
        if (compositeParams.method == "alpha")
            logPrintf(stdout, "  alpha: min=%.0f max=%.0f opacity=%.0f cutoff=%.0f\n",
                      compositeParams.alphaMin*255, compositeParams.alphaMax*255,
                      compositeParams.alphaOpacity*255, compositeParams.alphaCutoff*10000);
        else
            logPrintf(stdout, "  BL: ext=%.1f em=%.1f amb=%.1f\n",
                      compositeParams.blExtinction, compositeParams.blEmission, compositeParams.blAmbient);
        if (compositeParams.isoCutoff > 0) logPrintf(stdout, "  iso cutoff: %d\n", int(compositeParams.isoCutoff));
    }

    std::vector<float> accumOffsets;
    if (accum_step > 0) {
        if (accum_step > slice_step) { logPrintf(stderr, "Error: --accum > --slice-step\n"); return EXIT_FAILURE; }
        double ratio = slice_step / accum_step, rounded = std::round(ratio);
        if (std::abs(ratio - rounded) > 1e-4) { logPrintf(stderr, "Error: --accum must evenly divide --slice-step\n"); return EXIT_FAILURE; }
        size_t samples = std::max<size_t>(1, size_t(rounded));
        double spacing = slice_step / samples;
        for (size_t i = 0; i < samples; i++) accumOffsets.push_back(float(spacing * i));
        accum_step = spacing;
        logPrintf(stdout, "Accumulation: %zu samples/slice at step %.4f (%s)\n", samples, spacing, accum_type_str.c_str());
    }

    const float ds_scale = std::ldexp(1.0f, -group_idx);
    float scale_seg = parsed["scale-segmentation"].as<float>();
    double rotate_angle = parsed["rotate"].as<double>();
    int flip_axis = parsed["flip"].as<int>();
    const bool quickTif = parsed["quick-tif"].as<bool>();

    // --- Load affines ---
    AffineTransform affineTransform;
    bool hasAffine = false;
    std::vector<std::pair<std::string,bool>> affineSpecs;
    if (parsed.count("affine"))
        for (const auto& s : parsed["affine"].as<std::vector<std::string>>())
            affineSpecs.push_back(parseAffineSpec(s));
    if (parsed.count("affine-transform")) {
        affineSpecs.emplace_back(parsed["affine-transform"].as<std::string>(), parsed["invert-affine"].as<bool>());
        logPrintf(stdout, "[deprecated] Using --affine-transform; prefer --affine.\n");
    }
    if (parsed.count("affine-invert") && !affineSpecs.empty()) {
        std::set<int> inv;
        for (int idx : parsed["affine-invert"].as<std::vector<int>>()) {
            if (idx < 0 || idx >= int(affineSpecs.size())) { logPrintf(stderr, "Error: --affine-invert index out of range\n"); return EXIT_FAILURE; }
            inv.insert(idx);
        }
        for (int k = 0; k < int(affineSpecs.size()); k++) if (inv.count(k)) affineSpecs[k].second = true;
    }
    if (!affineSpecs.empty()) {
        AffineTransform composed;
        for (int k = 0; k < int(affineSpecs.size()); k++) {
            auto& [path, inv] = affineSpecs[k];
            try {
                auto T = loadAffineTransform(path);
                logPrintf(stdout, "Loaded affine[%d]: %s%s\n", k, path.c_str(), inv ? " (invert)" : "");
                if (inv && !invertAffineInPlace(T)) { logPrintf(stderr, "Error: non-invertible affine[%d]\n", k); return EXIT_FAILURE; }
                composed = composeAffine(composed, T);
            } catch (const std::exception& e) { logPrintf(stderr, "Error loading affine[%d]: %s\n", k, e.what()); return EXIT_FAILURE; }
        }
        hasAffine = true;
        affineTransform = composed;
        printMat4x4(affineTransform.matrix, "Final composed affine:");
    }

    // --- Open source volume ---
    auto ds = vc::zarr::openDatasetAutoSep(vol_path, std::to_string(group_idx));
    {
        const auto& sh = ds.shape();
        logPrintf(stdout, "zarr dataset size for group %d [%zu,%zu,%zu]\n", group_idx, sh[0], sh[1], sh[2]);
    }

    const bool output_is_u16 = ds.isUint16();
    logPrintf(stdout, "Source dtype: %s\n", output_is_u16 ? "uint16" : "uint8");
    if (output_is_u16 && isCompositeMode)
        logPrintf(stderr, "Warning: composite forces 8-bit output (source is 16-bit)\n");
    // Source volume chunk size (for compile-time tile stride dispatch)
    int chunkN = 0;
    {
        const auto& cs = ds.chunkShape();
        logPrintf(stdout, "chunk shape [%zu,%zu,%zu]\n", cs[0], cs[1], cs[2]);
        // Use the spatial dimension (Y or X); they should match for cubic chunks
        chunkN = static_cast<int>(cs[1]);
        if (chunkN != 32 && chunkN != 64 && chunkN != 128) {
            logPrintf(stderr, "Warning: chunk size %d not power-of-2 (32/64/128), defaulting to 128\n", chunkN);
            chunkN = 128;
        }
    }
    if (sampleMode == SampleMode::Nearest)
        logPrintf(stdout, "Sample mode: nearest\n");
    else
        logPrintf(stdout, "Sample mode: trilinear\n");

    int rotQuadGlobal = -1;
    if (std::abs(rotate_angle) > 1e-6) {
        rotQuadGlobal = normalizeQuadrantRotation(rotate_angle);
        if (rotQuadGlobal < 0) { logPrintf(stderr, "Error: only 0/90/180/270 rotations supported\n"); return EXIT_FAILURE; }
        rotate_angle = rotQuadGlobal * 90.0;
        logPrintf(stdout, "Rotation: %.0f degrees\n", rotate_angle);
    }
    if (flip_axis >= 0) logPrintf(stdout, "Flip: %s\n", flip_axis == 0 ? "V" : flip_axis == 1 ? "H" : "Both");

    if (wantZarr) {
        if (auto p = std::filesystem::path(zarrOutputArg).parent_path(); !p.empty())
            std::filesystem::create_directories(p);
    }
    if (wantTif) std::filesystem::create_directories(tifOutputArg);

    const size_t cache_bytes = parsed["cache-gb"].as<size_t>() * 1024ull * 1024ull * 1024ull;
    ChunkCache<uint8_t> chunk_cache_u8(cache_bytes);
    ChunkCache<uint16_t> chunk_cache_u16(cache_bytes);

    if (int t = parsed["timeout"].as<int>(); t > 0) {
        logPrintf(stdout, "Timeout: %d minutes\n", t);
        std::thread([t]{ std::this_thread::sleep_for(std::chrono::minutes(t)); logPrintf(stderr, "\n[timeout]\n"); if (g_logFile) std::fflush(g_logFile); _exit(2); }).detach();
    }

    // ============================================================
    // process_one: render a single segmentation to output
    // ============================================================
    auto process_one = [&](const std::filesystem::path& seg_folder) {
        {
            std::ostringstream oss;
            oss << "Rendering: " << seg_folder.string();
            if (wantZarr) oss << " -> " << zarrOutputArg << " (zarr)";
            if (wantTif)  oss << " -> " << tifOutputArg << " (tif)";
            logPrintf(stdout, "%s\n", oss.str().c_str());
        }

        std::unique_ptr<QuadSurface> surf;
        try { surf = load_quad_from_tifxyz(seg_folder); }
        catch (...) { logPrintf(stderr, "Error loading: %s\n", seg_folder.string().c_str()); return; }

        if (parsed["flatten"].as<bool>()) {
            logPrintf(stdout, "Applying ABF++ flattening...\n");
            vc::ABFConfig cfg;
            cfg.maxIterations = size_t(parsed["flatten-iterations"].as<int>());
            cfg.downsampleFactor = parsed["flatten-downsample"].as<int>();
            cfg.useABF = true; cfg.scaleToOriginalArea = true;
            if (auto* fs = vc::abfFlattenToNewSurface(*surf, cfg)) {
                surf.reset(fs);
                logPrintf(stdout, "Flattened: %dx%d\n", surf->rawPointsPtr()->cols, surf->rawPointsPtr()->rows);
            } else {
                logPrintf(stderr, "Warning: ABF++ failed, using original\n");
            }
        }

        // Replace sentinel -1 with NaN
        auto* raw_points = surf->rawPointsPtr();
        for (int j = 0; j < raw_points->rows; j++)
            for (int i = 0; i < raw_points->cols; i++)
                if ((*raw_points)(j,i)[0] == -1) (*raw_points)(j,i) = {NAN,NAN,NAN};

        // Bounding box of valid points
        int col_min = raw_points->cols, col_max = -1, row_min = raw_points->rows, row_max = -1;
        for (int j = 0; j < raw_points->rows; j++)
            for (int i = 0; i < raw_points->cols; i++)
                if (std::isfinite((*raw_points)(j,i)[0])) {
                    col_min = std::min(col_min, i); col_max = std::max(col_max, i);
                    row_min = std::min(row_min, j); row_max = std::max(row_max, j);
                }

        cv::Size full_size = raw_points->size();

        // Compute render scale
        double sA = 1.0;
        if (hasAffine) {
            cv::Matx33d A(affineTransform.matrix(0,0), affineTransform.matrix(0,1), affineTransform.matrix(0,2),
                          affineTransform.matrix(1,0), affineTransform.matrix(1,1), affineTransform.matrix(1,2),
                          affineTransform.matrix(2,0), affineTransform.matrix(2,1), affineTransform.matrix(2,2));
            double d = cv::determinant(cv::Mat(A));
            if (std::isfinite(d) && std::abs(d) > 1e-18) sA = std::cbrt(std::abs(d));
        }
        double render_scale = double(tgt_scale) * (double(scale_seg) * sA * double(ds_scale));

        {
            double sx = render_scale / surf->_scale[0], sy = render_scale / surf->_scale[1];
            full_size.width  = std::max(1, int(std::lround(full_size.width  * sx)));
            full_size.height = std::max(1, int(std::lround(full_size.height * sy)));
        }

        cv::Size tgt_size = full_size;
        cv::Rect crop = {0, 0, full_size.width, full_size.height};
        const cv::Rect canvasROI = crop;

        logPrintf(stdout, "ds_level=%d ds_scale=%g sA=%g Pg=%g render_scale=%g\n",
                  group_idx, ds_scale, sA, double(tgt_scale), render_scale);

        // Handle crop
        int cx = parsed["crop-x"].as<int>(), cy = parsed["crop-y"].as<int>();
        int cw = parsed["crop-width"].as<int>(), ch = parsed["crop-height"].as<int>();
        bool manual = cw > 0 && ch > 0, autoCrop = parsed["auto-crop"].as<bool>();
        if (autoCrop && manual) { logPrintf(stderr, "Error: --auto-crop and --crop-* are mutually exclusive\n"); return; }

        if (autoCrop && col_max >= col_min) {
            double sx = render_scale / surf->_scale[0], sy = render_scale / surf->_scale[1];
            crop = cv::Rect(int(std::floor(col_min*sx)), int(std::floor(row_min*sy)),
                            int(std::ceil((col_max+1)*sx)) - int(std::floor(col_min*sx)),
                            int(std::ceil((row_max+1)*sy)) - int(std::floor(row_min*sy))) & canvasROI;
            tgt_size = crop.size();
            logPrintf(stdout, "auto-crop: [%d×%d from (%d,%d)]\n", crop.width, crop.height, crop.x, crop.y);
        } else if (manual) {
            crop = cv::Rect(cx, cy, cw, ch) & canvasROI;
            if (crop.width <= 0 || crop.height <= 0) { logPrintf(stderr, "Error: crop outside canvas\n"); return; }
            tgt_size = crop.size();
        }

        logPrintf(stdout, "rendering %dx%d at scale %g crop [%d×%d from (%d,%d)]\n",
                  tgt_size.width, tgt_size.height, double(tgt_scale),
                  crop.width, crop.height, crop.x, crop.y);

        const int rotQuad = rotQuadGlobal;

        // Determine output dtype
        const bool useU16 = output_is_u16 && !isCompositeMode;
        const int cvType = useU16 ? CV_16UC1 : CV_8UC1;

        // ---- Zarr setup (if requested) ----
        const size_t CH = 128, CW = 128;
        size_t baseZ = isCompositeMode ? 1 : size_t(std::max(1, num_slices));
        std::vector<size_t> chunks0;
        vc::zarr::Dataset dsOut0;
        std::filesystem::path output_path_local = wantZarr ? std::filesystem::path(zarrOutputArg) : std::filesystem::path("/dev/null");
        vc::zarr::File outFile(output_path_local);
        size_t tilesYSrc = 0, tilesXSrc = 0;

        const bool zarrV3 = parsed["zarr-v3"].as<bool>();

        if (wantZarr) {
            cv::Size zarrXY = tgt_size;
            if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(zarrXY.width, zarrXY.height);
            size_t baseY = zarrXY.height, baseX = zarrXY.width;

            std::vector<size_t> shape0 = {baseZ, baseY, baseX};
            json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};
            std::string dtype = useU16 ? "uint16" : "uint8";

            // For v3 sharded: shard=128^3, inner=32^3
            // For v2: chunk = [Z, min(128,Y), min(128,X)]
            const size_t SHARD = 128, INNER = 32;
            if (zarrV3) {
                chunks0 = {std::min(SHARD, shape0[0]),
                           std::min(SHARD, shape0[1]),
                           std::min(SHARD, shape0[2])};
            } else {
                chunks0 = {shape0[0], std::min(CH, shape0[1]), std::min(CW, shape0[2])};
            }
            std::vector<size_t> innerChunks0 = {
                std::min(INNER, chunks0[0]),
                std::min(INNER, chunks0[1]),
                std::min(INNER, chunks0[2])};

            // Helper to create the L0 dataset (v2 or v3 sharded)
            auto createL0 = [&]() -> vc::zarr::Dataset {
                auto dsPath = output_path_local / "0";
                if (zarrV3) {
                    return vc::zarr::Dataset::createV3Sharded(
                        dsPath, dtype, shape0, chunks0, innerChunks0, compOpts);
                } else {
                    return vc::zarr::createDataset(
                        outFile, "0", dtype, shape0, chunks0,
                        std::string("blosc"), compOpts);
                }
            };

            if (pre_flag) {
                logPrintf(stdout, "[pre] creating %s zarr + all levels...\n",
                          zarrV3 ? "v3 sharded" : "v2");
                vc::zarr::createFile(output_path_local, true);
                if (zarrV3) {
                    // Write a v3 root group zarr.json
                    json rootJson;
                    rootJson["zarr_format"] = 3;
                    rootJson["node_type"] = "group";
                    rootJson["attributes"] = json::object();
                    std::ofstream rootF(output_path_local / "zarr.json");
                    rootF << rootJson.dump(4) << "\n";
                }
                createL0();
                logPrintf(stdout, "[pre] L0 shape: [%zu,%zu,%zu]\n", shape0[0], shape0[1], shape0[2]);
                if (zarrV3) {
                    logPrintf(stdout, "[pre] shard: [%zu,%zu,%zu]  inner: [%zu,%zu,%zu]\n",
                              chunks0[0], chunks0[1], chunks0[2],
                              innerChunks0[0], innerChunks0[1], innerChunks0[2]);
                }
                if (wantPyramid && !zarrV3) {
                    // Create pyramid datasets (v2 only; v3 pyramid not yet implemented)
                    std::vector<size_t> prevShape = shape0;
                    for (int level = 1; level <= 5; level++) {
                        std::vector<size_t> ds = {(prevShape[0]+1)/2, (prevShape[1]+1)/2, (prevShape[2]+1)/2};
                        std::vector<size_t> dc = {std::min(ds[0], shape0[0]), std::min(CH, ds[1]), std::min(CW, ds[2])};
                        vc::zarr::createDataset(outFile, std::to_string(level), dtype, ds, dc, std::string("blosc"), compOpts);
                        prevShape = ds;
                    }
                }

                // Write attrs
                cv::Size attrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(attrXY.width, attrXY.height);
                json attrs;
                attrs["source_zarr"] = vol_path.string();
                attrs["source_group"] = group_idx;
                attrs["num_slices"] = baseZ;
                attrs["slice_step"] = slice_step;
                if (!accumOffsets.empty()) {
                    attrs["accum_step"] = accum_step;
                    attrs["accum_type"] = accum_type_str;
                    attrs["accum_samples"] = int(accumOffsets.size());
                }
                attrs["canvas_size"] = {attrXY.width, attrXY.height};
                attrs["chunk_size"] = {int(baseZ), int(CH), int(CW)};
                attrs["note_axes_order"] = "ZYX (slice, row, col)";
                if (zarrV3) {
                    attrs["zarr_version"] = 3;
                    attrs["shard_shape"] = {int(chunks0[0]), int(chunks0[1]), int(chunks0[2])};
                    attrs["inner_chunk_shape"] = {int(innerChunks0[0]), int(innerChunks0[1]), int(innerChunks0[2])};
                }
                if (!zarrV3) {
                    vc::zarr::writeZarrMultiscaleAttrs(outFile, 5, attrs);
                } else {
                    auto rootZarr = output_path_local / "zarr.json";
                    std::ifstream rf(rootZarr);
                    json rootJson = json::parse(rf);
                    rf.close();
                    rootJson["attributes"] = attrs;
                    std::ofstream wf(rootZarr);
                    wf << rootJson.dump(4) << "\n";
                }
                logPrintf(stdout, "[pre] done.\n");
                return;
            } else if (numParts > 1) {
                const auto dsPathV2 = output_path_local / "0" / ".zarray";
                const auto dsPathV3 = output_path_local / "0" / "zarr.json";
                if (!std::filesystem::exists(dsPathV2) && !std::filesystem::exists(dsPathV3)) {
                    logPrintf(stderr, "Error: run --pre first in multi-part mode\n"); return;
                }
                dsOut0 = vc::zarr::openDataset(outFile, "0");
            } else if (resumeFlag) {
                const auto dsPathV2 = output_path_local / "0" / ".zarray";
                const auto dsPathV3 = output_path_local / "0" / "zarr.json";
                if (std::filesystem::exists(dsPathV2) || std::filesystem::exists(dsPathV3)) {
                    dsOut0 = vc::zarr::openDataset(outFile, "0");
                    logPrintf(stdout, "[resume] opening existing zarr\n");
                } else {
                    vc::zarr::createFile(output_path_local, true);
                    if (zarrV3) {
                        json rootJson;
                        rootJson["zarr_format"] = 3;
                        rootJson["node_type"] = "group";
                        rootJson["attributes"] = json::object();
                        std::ofstream rootF(output_path_local / "zarr.json");
                        rootF << rootJson.dump(4) << "\n";
                    }
                    dsOut0 = createL0();
                }
            } else {
                vc::zarr::createFile(output_path_local, true);
                if (zarrV3) {
                    json rootJson;
                    rootJson["zarr_format"] = 3;
                    rootJson["node_type"] = "group";
                    rootJson["attributes"] = json::object();
                    std::ofstream rootF(output_path_local / "zarr.json");
                    rootF << rootJson.dump(4) << "\n";
                }
                dsOut0 = createL0();
            }

            tilesYSrc = (tgt_size.height + CH - 1) / CH;
            tilesXSrc = (tgt_size.width  + CW - 1) / CW;
        }

        // Limit shard buffer to 1GB — LRU eviction flushes oldest shards
        if (dsOut0 && dsOut0.isSharded()) {
            dsOut0.setShardBufferLimit(1ULL << 30);
        }

        // Install graceful shutdown handlers
        std::signal(SIGINT, shutdownHandler);
        std::signal(SIGTERM, shutdownHandler);

        // ---- TIF setup (if requested) ----
        std::vector<TiffWriter> tifWriters;
        uint32_t tiffTileH = 16;

        if (wantTif) {
            int outW = (rotQuad >= 0 && rotQuad % 2 == 1) ? tgt_size.height : tgt_size.width;
            int outH = (rotQuad >= 0 && rotQuad % 2 == 1) ? tgt_size.width  : tgt_size.height;
            int tifSlices = isCompositeMode ? 1 : num_slices;

            auto makePath = [&](int zi) -> std::filesystem::path {
                int pad = 2, v = std::max(0, num_slices-1);
                while (v >= 100) { pad++; v /= 10; }
                std::ostringstream fn; fn << std::setw(pad) << std::setfill('0') << zi;
                return std::filesystem::path(tifOutputArg) / (fn.str() + ".tif");
            };
            auto makePartPath = [&](int zi) -> std::filesystem::path {
                auto p = makePath(zi);
                if (numParts > 1) return p.parent_path() / (p.stem().string() + ".part" + std::to_string(partId) + p.extension().string());
                return p;
            };

            // Skip if all exist
            bool tifSkip = false;
            if (numParts <= 1) {
                bool all = true;
                for (int z = 0; z < tifSlices; z++) if (!std::filesystem::exists(makePartPath(z))) { all = false; break; }
                if (all) {
                    if (!wantZarr) { logPrintf(stdout, "[tif] all slices exist, skipping.\n"); return; }
                    logPrintf(stdout, "[tif] all slices exist, skipping tif output.\n");
                    tifSkip = true;
                }
            }
            if (!tifSkip) {
                uint32_t tiffTileW = (uint32_t(outW) + 15u) & ~15u;
                uint16_t tifComp = quickTif ? COMPRESSION_PACKBITS : COMPRESSION_LZW;
                for (int z = 0; z < tifSlices; z++)
                    tifWriters.emplace_back(makePartPath(z), uint32_t(outW), uint32_t(outH), cvType, tiffTileW, tiffTileH, 0.0f, tifComp);
            }
        }

        // ---- Create pyramid datasets before render ----
        const bool hasRotFlip = (rotQuad >= 0 || flip_axis >= 0);
        // Inline pyramid only works without rotation/flip and non-sharded v2
        const bool inlinePyramid = wantZarr && wantPyramid && !pre_flag && !hasRotFlip && !zarrV3;
        std::vector<vc::zarr::Dataset*> pyramidDs;
        std::vector<vc::zarr::Dataset> pyramidOwned;
        if (wantZarr && wantPyramid && !pre_flag && !zarrV3) {
            std::string dtype = useU16 ? "uint16" : "uint8";
            json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};
            // Single-part: create datasets now; multi-part/resume: already created them
            if (numParts <= 1 && !resumeFlag) {
                cv::Size zarrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(zarrXY.width, zarrXY.height);
                std::vector<size_t> prevShape = {baseZ, size_t(zarrXY.height), size_t(zarrXY.width)};
                for (int level = 1; level <= 5; level++) {
                    std::vector<size_t> ds = {(prevShape[0]+1)/2, (prevShape[1]+1)/2, (prevShape[2]+1)/2};
                    std::vector<size_t> dc = {std::min(ds[0], baseZ), std::min(CH, ds[1]), std::min(CW, ds[2])};
                    vc::zarr::createDataset(outFile, std::to_string(level), dtype, ds, dc, std::string("blosc"), compOpts);
                    prevShape = ds;
                }
            }
            if (inlinePyramid) {
                pyramidOwned.reserve(5);
                for (int level = 1; level <= 5; level++) {
                    pyramidOwned.push_back(vc::zarr::openDataset(outFile, std::to_string(level)));
                    pyramidDs.push_back(&pyramidOwned.back());
                }
            }
        }

        // ---- Render pass ----
        // Dispatch helper: creates SparseVolume<T,N> and calls a render function
        // with compile-time chunk size N.
        auto dispatchTiles = [&](auto tag) {
            using T = decltype(tag);
            auto& cache = [&]() -> ChunkCache<T>& {
                if constexpr (std::is_same_v<T, uint16_t>) return chunk_cache_u16;
                else return chunk_cache_u8;
            }();
            auto doIt = [&](auto nTag) {
                constexpr int NN = decltype(nTag)::value;
                vc::simd::SparseVolume<T, NN> vol(&ds, &cache);
                renderTiles<T, NN>(surf.get(), &ds, &cache,
                    vol, sampleMode,
                    full_size, crop, tgt_size, float(render_scale), scale_seg, ds_scale,
                    hasAffine, affineTransform, num_slices, slice_step,
                    accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                    compositeParams, rotQuad, flip_axis, numParts, partId, cvType,
                    &dsOut0, chunks0, tilesXSrc, tilesYSrc,
                    pyramidDs,
                    tifWriters.empty() ? nullptr : &tifWriters, tiffTileH, quickTif,
                    resumeFlag);
            };
            switch (chunkN) {
                case 32:  doIt(std::integral_constant<int,32>{}); break;
                case 64:  doIt(std::integral_constant<int,64>{}); break;
                default:  doIt(std::integral_constant<int,128>{}); break;
            }
        };

        {
            if (wantZarr) {
                // Tile-based: OMP-parallel over output zarr chunks
                if (useU16) dispatchTiles(uint16_t{});
                else        dispatchTiles(uint8_t{});
            } else {
                // Band-based: TIF-only path
                uint32_t bandH = 128;
                auto writerFn = [&](const std::vector<cv::Mat>& slices, uint32_t bandIdx, uint32_t bandY0) {
                    if (!tifWriters.empty()) {
                        if (quickTif && !useU16) {
                            std::vector<cv::Mat> quantized(slices.size());
                            for (size_t i = 0; i < slices.size(); i++) {
                                quantized[i] = slices[i].clone();
                                for (int r = 0; r < quantized[i].rows; r++) {
                                    auto* row = quantized[i].ptr<uint8_t>(r);
                                    for (int c = 0; c < quantized[i].cols; c++)
                                        row[c] &= 0xF0;
                                }
                            }
                            writeTifBand(tifWriters, quantized, bandY0, tiffTileH, uint32_t(tgt_size.height), rotQuad, flip_axis);
                        } else {
                            writeTifBand(tifWriters, slices, bandY0, tiffTileH, uint32_t(tgt_size.height), rotQuad, flip_axis);
                        }
                    }
                };

                auto dispatchBands = [&](auto tag) {
                    using T = decltype(tag);
                    auto& cache = [&]() -> ChunkCache<T>& {
                        if constexpr (std::is_same_v<T, uint16_t>) return chunk_cache_u16;
                        else return chunk_cache_u8;
                    }();
                    auto doIt = [&](auto nTag) {
                        constexpr int NN = decltype(nTag)::value;
                        vc::simd::SparseVolume<T, NN> vol(&ds, &cache);
                        renderBands<T, NN>(surf.get(), &ds, &cache,
                            vol, sampleMode,
                            full_size, crop, tgt_size, float(render_scale), scale_seg, ds_scale,
                            hasAffine, affineTransform, num_slices, slice_step,
                            accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                            compositeParams, rotQuad, flip_axis, numParts, partId, cvType, bandH, writerFn);
                    };
                    switch (chunkN) {
                        case 32:  doIt(std::integral_constant<int,32>{}); break;
                        case 64:  doIt(std::integral_constant<int,64>{}); break;
                        default:  doIt(std::integral_constant<int,128>{}); break;
                    }
                };

                if (useU16) dispatchBands(uint16_t{});
                else        dispatchBands(uint8_t{});
            }

            tifWriters.clear();
        }

        // Flush buffered shard writes to disk (no-op for non-sharded)
        if (wantZarr) {
            if (g_shutdownRequested.load(std::memory_order_relaxed)) {
                logPrintf(stdout, "[shutdown] flushing buffered shards...\n");
            }
            dsOut0.flush();
            if (g_shutdownRequested.load(std::memory_order_relaxed)) {
                logPrintf(stdout, "[shutdown] flush complete. Exiting gracefully.\n");
                return;
            }
        }

        // ---- Zarr pyramid + attrs ----
        if (wantZarr && !pre_flag) {
            // Rotation/flip: pyramid couldn't be built inline, build from L0 on disk
            if (wantPyramid && hasRotFlip && !zarrV3) {
                logPrintf(stdout, "[pyramid] building from L0...\n");
                std::string dtype = useU16 ? "uint16" : "uint8";
                for (int level = 1; level <= 5; level++) {
                    vc::zarr::buildPyramidLevel(outFile, level, dtype, CH, CW);
                }
            }
            if (zarrV3 && wantPyramid) {
                logPrintf(stdout, "[zarr-v3] skipping pyramid build (not yet implemented for v3 sharded)\n");
            }

            // --pre already writes attrs for multi-part; single-part writes here
            if (numParts <= 1) {
                cv::Size attrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(attrXY.width, attrXY.height);
                json attrs;
                attrs["source_zarr"] = vol_path.string();
                attrs["source_group"] = group_idx;
                attrs["num_slices"] = baseZ;
                attrs["slice_step"] = slice_step;
                if (!accumOffsets.empty()) {
                    attrs["accum_step"] = accum_step;
                    attrs["accum_type"] = accum_type_str;
                    attrs["accum_samples"] = int(accumOffsets.size());
                }
                attrs["canvas_size"] = {attrXY.width, attrXY.height};
                attrs["chunk_size"] = {int(baseZ), int(CH), int(CW)};
                attrs["note_axes_order"] = "ZYX (slice, row, col)";
                if (!zarrV3) {
                    vc::zarr::writeZarrMultiscaleAttrs(outFile, 5, attrs);
                } else {
                    attrs["zarr_version"] = 3;
                    auto rootZarr = output_path_local / "zarr.json";
                    std::ifstream rf(rootZarr);
                    json rootJson = json::parse(rf);
                    rf.close();
                    rootJson["attributes"] = attrs;
                    std::ofstream wf(rootZarr);
                    wf << rootJson.dump(4) << "\n";
                }
            }
        }
    };

    process_one(seg_path);

    // Print final cache stats
    auto printStats = [](const char* name, const auto& s) {
        uint64_t tot = s.hits + s.misses;
        if (tot == 0) return;
        logPrintf(stdout, "[%s cache] hits=%lu miss=%lu rate=%.1f%% evict=%lu read=%.2fGB re-read=%lu(%.2fGB)\n",
                  name, (unsigned long)s.hits, (unsigned long)s.misses,
                  100.0*s.hits/tot, (unsigned long)s.evictions,
                  s.bytesRead/(1024.0*1024.0*1024.0),
                  (unsigned long)s.reReads, s.reReadBytes/(1024.0*1024.0*1024.0));
    };
    printStats("u8", chunk_cache_u8.stats());
    printStats("u16", chunk_cache_u16.stats());

    stopLogFlusher();
    return EXIT_SUCCESS;
}