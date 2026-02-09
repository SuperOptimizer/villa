#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/util/Zarr.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/ABFFlattening.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <boost/program_options.hpp>
#include <mutex>
#include <cmath>
#include <set>
#include <cctype>
#include <chrono>
#include <thread>
#include <tiffio.h>
#include <omp.h>

namespace po = boost::program_options;
using json = nlohmann::json;

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
    if (header) std::cout << header << "\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int r = 0; r < 4; ++r) {
        std::cout << "  [";
        for (int c = 0; c < 4; ++c) {
            std::cout << std::setw(12) << M(r,c);
            if (c < 3) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout.unsetf(std::ios::floatfield);
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

// Render bands for a segmentation, calling writeSlices for each band.
// bandH should match the output chunk Y dimension so each chunk is written exactly once.
// WriteFn: void(const std::vector<cv::Mat>& slices, uint32_t bandIdx, uint32_t bandY0)
template <typename T, typename WriteFn>
static void renderBands(
    QuadSurface* surf, z5::Dataset* ds,
    ChunkCache<T>* cache,
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

    for (uint32_t bi = uint32_t(partId); bi < numBands; bi += uint32_t(numParts)) {
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
            // Normal: bulk read + accumulate
            std::vector<cv::Mat_<T>> raw;
            readMultiSlice(raw, ds, cache, base, dirs, allOffsets);
            slices = processRawSlices<T>(raw, numSlices, accumOffsets, accumType, cvType, rotQuad, flipAxis);
        }

        writeSlices(slices, bi, y0);

        // Progress (throttled to ~1/sec)
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - lastPrint).count();
        uint32_t bandsThis = (numBands - uint32_t(partId) + uint32_t(numParts) - 1) / uint32_t(numParts);
        uint32_t done = (bi - uint32_t(partId)) / uint32_t(numParts) + 1;
        if (since >= 1.0 || done == bandsThis) {
            lastPrint = now;
            double elapsed = std::chrono::duration<double>(now - wallStart).count();
            double eta = done > 0 ? elapsed * (double(bandsThis) / done - 1.0) : 0.0;
            auto cs = cache->stats();
            uint64_t tot = cs.hits + cs.misses;
            double hr = tot > 0 ? 100.0 * cs.hits / tot : 0.0;
            double gbR = cs.bytesRead / (1024.0*1024.0*1024.0);
            std::fprintf(stderr, "\r  band %u/%u (%d%%)  %dm%02ds  eta %dm%02ds  cache %.1f%% evict %lu read %.2fGB",
                done, bandsThis, int(100.0 * done / bandsThis),
                int(elapsed)/60, int(elapsed)%60, int(eta)/60, int(eta)%60,
                hr, (unsigned long)cs.evictions, gbR);
        }
    }
    std::fprintf(stderr, "\n");
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

template <typename T>
static void renderTiles(
    QuadSurface* surf, z5::Dataset* ds,
    ChunkCache<T>* cache,
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
    z5::Dataset* dsOut, const std::vector<size_t>& chunks0,
    size_t tilesXSrc, size_t tilesYSrc,
    // TIF output (optional)
    std::vector<TiffWriter>* tifWriters, uint32_t tiffTileH,
    bool quickTif)
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

    for (uint32_t ty = uint32_t(partId); ty < numTileRows; ty += uint32_t(numParts)) {
        uint32_t y0 = ty * uint32_t(CH);
        uint32_t dy = std::min(uint32_t(CH), uint32_t(tgtSize.height) - y0);

        // TIF row buffer: one tile-column's worth of slices per tx
        // tifRowBuf[tx] = vector of cv::Mat (one per output slice), each 128xdx
        std::vector<std::vector<cv::Mat>> tifRowBuf;
        if (wantTif) tifRowBuf.resize(numTileCols);

        #pragma omp parallel for schedule(dynamic)
        for (uint32_t tx = 0; tx < numTileCols; tx++) {
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
                sampleTileSlices(raw, ds, cache, base, dirs, allOffsets);
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
                z5::types::ShapeType chunkId = {0, size_t(dstTy), size_t(dstTx)};
                dsOut->writeChunk(chunkId, chunkBuf.data());
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

        // Progress (throttled to ~1/sec)
        auto now = std::chrono::steady_clock::now();
        double since = std::chrono::duration<double>(now - lastPrint).count();
        uint32_t tileRowsThis = (numTileRows - uint32_t(partId) + uint32_t(numParts) - 1) / uint32_t(numParts);
        uint32_t done = (ty - uint32_t(partId)) / uint32_t(numParts) + 1;
        if (since >= 1.0 || done == tileRowsThis) {
            lastPrint = now;
            double elapsed = std::chrono::duration<double>(now - wallStart).count();
            double eta = done > 0 ? elapsed * (double(tileRowsThis) / done - 1.0) : 0.0;
            auto cs = cache->stats();
            uint64_t tot = cs.hits + cs.misses;
            double hr = tot > 0 ? 100.0 * cs.hits / tot : 0.0;
            double gbR = cs.bytesRead / (1024.0*1024.0*1024.0);
            std::fprintf(stderr, "\r  tile-row %u/%u (%d%%)  %dm%02ds  eta %dm%02ds  cache %.1f%% evict %lu read %.2fGB",
                done, tileRowsThis, int(100.0 * done / tileRowsThis),
                int(elapsed)/60, int(elapsed)%60, int(eta)/60, int(eta)%60,
                hr, (unsigned long)cs.evictions, gbR);
        }
    }
    std::fprintf(stderr, "\n");
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
        ("merge-parts", po::bool_switch()->default_value(false), "Merge partial TIFFs")
        ("finalize", po::bool_switch()->default_value(false), "Build pyramid for existing zarr L0")
        ("pyramid-level", po::value<int>()->default_value(0), "Build only this pyramid level (1-5, 0=all)")
        ("pre", po::bool_switch()->default_value(false), "Create zarr + all level datasets");
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

    // --- Parse common options ---
    const int numParts = parsed["num-parts"].as<int>();
    const int partId = parsed["part-id"].as<int>();
    if (numParts < 1 || partId < 0 || partId >= numParts) {
        std::cerr << "Error: need 0 <= part-id < num-parts\n"; return EXIT_FAILURE;
    }
    if (numParts > 1) std::cout << "Multi-part mode: part " << partId << " of " << numParts << std::endl;

    const bool finalize_flag = parsed["finalize"].as<bool>();
    const bool pre_flag = parsed["pre"].as<bool>();
    if (pre_flag && finalize_flag) { std::cerr << "Error: --pre and --finalize are mutually exclusive.\n"; return EXIT_FAILURE; }

    const bool wantZarr = parsed.count("zarr-output") > 0;
    const bool wantTif = parsed.count("tif-output") > 0;
    if (!wantZarr && !wantTif) { std::cerr << "Error: at least one of --zarr-output or --tif-output required\n"; return EXIT_FAILURE; }
    const std::string zarrOutputArg = wantZarr ? parsed["zarr-output"].as<std::string>() : "";
    const std::string tifOutputArg = wantTif ? parsed["tif-output"].as<std::string>() : "";

    const bool mergeParts = parsed["merge-parts"].as<bool>();
    if (mergeParts) {
        if (numParts < 2) { std::cerr << "Error: --merge-parts needs --num-parts >= 2\n"; return EXIT_FAILURE; }
        if (!wantTif) { std::cerr << "Error: --merge-parts requires --tif-output\n"; return EXIT_FAILURE; }
        return mergeTiffParts(tifOutputArg, numParts) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (finalize_flag && numParts >= 2 && wantTif && !wantZarr) {
        return mergeTiffParts(tifOutputArg, numParts) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::filesystem::path vol_path = parsed["volume"].as<std::string>();

    if (!parsed.count("segmentation")) { std::cerr << "Error: --segmentation required\n"; return EXIT_FAILURE; }
    std::filesystem::path seg_path = parsed["segmentation"].as<std::string>();

    float tgt_scale = parsed["scale"].as<float>();
    int group_idx = parsed["group-idx"].as<int>();
    int num_slices = parsed["num-slices"].as<int>();
    double slice_step = parsed["slice-step"].as<float>();
    if (!std::isfinite(slice_step) || slice_step <= 0) { std::cerr << "Error: --slice-step must be positive\n"; return EXIT_FAILURE; }

    double accum_step = parsed["accum"].as<float>();
    if (!std::isfinite(accum_step) || accum_step < 0) { std::cerr << "Error: --accum must be non-negative\n"; return EXIT_FAILURE; }

    std::string accum_type_str = parsed["accum-type"].as<std::string>();
    std::transform(accum_type_str.begin(), accum_type_str.end(), accum_type_str.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });

    AccumType accumType;
    if      (accum_type_str == "max")    accumType = AccumType::Max;
    else if (accum_type_str == "mean")   accumType = AccumType::Mean;
    else if (accum_type_str == "median") accumType = AccumType::Median;
    else if (accum_type_str == "alpha")  accumType = AccumType::Alpha;
    else if (accum_type_str == "beerlam" || accum_type_str == "beerlambert") accumType = AccumType::BeerLambert;
    else { std::cerr << "Error: invalid --accum-type\n"; return EXIT_FAILURE; }

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
        if (compositeEnd < compositeStart) { std::cerr << "Error: --composite-end < --composite-start\n"; return EXIT_FAILURE; }
        int layers = compositeEnd - compositeStart + 1;
        std::cout << "Composite: " << compositeParams.method << " (" << layers << " layers [" << compositeStart << ".." << compositeEnd << "])\n";
        if (compositeParams.method == "alpha")
            std::cout << "  alpha: min=" << compositeParams.alphaMin*255 << " max=" << compositeParams.alphaMax*255
                      << " opacity=" << compositeParams.alphaOpacity*255 << " cutoff=" << compositeParams.alphaCutoff*10000 << "\n";
        else
            std::cout << "  BL: ext=" << compositeParams.blExtinction << " em=" << compositeParams.blEmission << " amb=" << compositeParams.blAmbient << "\n";
        if (compositeParams.isoCutoff > 0) std::cout << "  iso cutoff: " << int(compositeParams.isoCutoff) << "\n";
    }

    std::vector<float> accumOffsets;
    if (accum_step > 0) {
        if (accum_step > slice_step) { std::cerr << "Error: --accum > --slice-step\n"; return EXIT_FAILURE; }
        double ratio = slice_step / accum_step, rounded = std::round(ratio);
        if (std::abs(ratio - rounded) > 1e-4) { std::cerr << "Error: --accum must evenly divide --slice-step\n"; return EXIT_FAILURE; }
        size_t samples = std::max<size_t>(1, size_t(rounded));
        double spacing = slice_step / samples;
        for (size_t i = 0; i < samples; i++) accumOffsets.push_back(float(spacing * i));
        accum_step = spacing;
        std::cout << "Accumulation: " << samples << " samples/slice at step " << spacing << " (" << accum_type_str << ")\n";
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
        std::cout << "[deprecated] Using --affine-transform; prefer --affine.\n";
    }
    if (parsed.count("affine-invert") && !affineSpecs.empty()) {
        std::set<int> inv;
        for (int idx : parsed["affine-invert"].as<std::vector<int>>()) {
            if (idx < 0 || idx >= int(affineSpecs.size())) { std::cerr << "Error: --affine-invert index out of range\n"; return EXIT_FAILURE; }
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
                std::cout << "Loaded affine[" << k << "]: " << path << (inv ? " (invert)" : "") << "\n";
                if (inv && !invertAffineInPlace(T)) { std::cerr << "Error: non-invertible affine[" << k << "]\n"; return EXIT_FAILURE; }
                composed = composeAffine(composed, T);
            } catch (const std::exception& e) { std::cerr << "Error loading affine[" << k << "]: " << e.what() << "\n"; return EXIT_FAILURE; }
        }
        hasAffine = true;
        affineTransform = composed;
        printMat4x4(affineTransform.matrix, "Final composed affine:");
    }

    // --- Open source volume ---
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, std::to_string(group_idx),
        json::parse(std::ifstream(vol_path/std::to_string(group_idx)/".zarray")).value<std::string>("dimension_separator","."));
    auto ds = z5::filesystem::openDataset(ds_handle);
    std::cout << "zarr dataset size for group " << group_idx << ds->shape() << std::endl;

    const bool output_is_u16 = (ds->getDtype() == z5::types::Datatype::uint16);
    std::cout << "Source dtype: " << (output_is_u16 ? "uint16" : "uint8") << std::endl;
    if (output_is_u16 && isCompositeMode)
        std::cerr << "Warning: composite forces 8-bit output (source is 16-bit)\n";
    std::cout << "chunk shape " << ds->chunking().blockShape() << std::endl;

    int rotQuadGlobal = -1;
    if (std::abs(rotate_angle) > 1e-6) {
        rotQuadGlobal = normalizeQuadrantRotation(rotate_angle);
        if (rotQuadGlobal < 0) { std::cerr << "Error: only 0/90/180/270 rotations supported\n"; return EXIT_FAILURE; }
        rotate_angle = rotQuadGlobal * 90.0;
        std::cout << "Rotation: " << rotate_angle << " degrees\n";
    }
    if (flip_axis >= 0) std::cout << "Flip: " << (flip_axis == 0 ? "V" : flip_axis == 1 ? "H" : "Both") << "\n";

    if (wantZarr) {
        if (auto p = std::filesystem::path(zarrOutputArg).parent_path(); !p.empty())
            std::filesystem::create_directories(p);
    }
    if (wantTif) std::filesystem::create_directories(tifOutputArg);

    const size_t cache_bytes = parsed["cache-gb"].as<size_t>() * 1024ull * 1024ull * 1024ull;
    ChunkCache<uint8_t> chunk_cache_u8(cache_bytes);
    ChunkCache<uint16_t> chunk_cache_u16(cache_bytes);

    if (int t = parsed["timeout"].as<int>(); t > 0) {
        std::cout << "Timeout: " << t << " minutes\n";
        std::thread([t]{ std::this_thread::sleep_for(std::chrono::minutes(t)); std::cerr << "\n[timeout]\n"; _exit(2); }).detach();
    }

    // ============================================================
    // process_one: render a single segmentation to output
    // ============================================================
    auto process_one = [&](const std::filesystem::path& seg_folder) {
        std::cout << "Rendering: " << seg_folder;
        if (wantZarr) std::cout << " -> " << zarrOutputArg << " (zarr)";
        if (wantTif)  std::cout << " -> " << tifOutputArg << " (tif)";
        std::cout << std::endl;

        std::unique_ptr<QuadSurface> surf;
        try { surf = load_quad_from_tifxyz(seg_folder); }
        catch (...) { std::cout << "Error loading: " << seg_folder << std::endl; return; }

        if (parsed["flatten"].as<bool>()) {
            std::cout << "Applying ABF++ flattening...\n";
            vc::ABFConfig cfg;
            cfg.maxIterations = size_t(parsed["flatten-iterations"].as<int>());
            cfg.downsampleFactor = parsed["flatten-downsample"].as<int>();
            cfg.useABF = true; cfg.scaleToOriginalArea = true;
            if (auto* fs = vc::abfFlattenToNewSurface(*surf, cfg)) {
                surf.reset(fs);
                std::cout << "Flattened: " << surf->rawPointsPtr()->cols << "x" << surf->rawPointsPtr()->rows << "\n";
            } else {
                std::cerr << "Warning: ABF++ failed, using original\n";
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

        std::cout << "ds_level=" << group_idx << " ds_scale=" << ds_scale << " sA=" << sA
                  << " Pg=" << tgt_scale << " render_scale=" << render_scale << "\n";

        // Handle crop
        int cx = parsed["crop-x"].as<int>(), cy = parsed["crop-y"].as<int>();
        int cw = parsed["crop-width"].as<int>(), ch = parsed["crop-height"].as<int>();
        bool manual = cw > 0 && ch > 0, autoCrop = parsed["auto-crop"].as<bool>();
        if (autoCrop && manual) { std::cerr << "Error: --auto-crop and --crop-* are mutually exclusive\n"; return; }

        if (autoCrop && col_max >= col_min) {
            double sx = render_scale / surf->_scale[0], sy = render_scale / surf->_scale[1];
            crop = cv::Rect(int(std::floor(col_min*sx)), int(std::floor(row_min*sy)),
                            int(std::ceil((col_max+1)*sx)) - int(std::floor(col_min*sx)),
                            int(std::ceil((row_max+1)*sy)) - int(std::floor(row_min*sy))) & canvasROI;
            tgt_size = crop.size();
            std::cout << "auto-crop: " << crop << "\n";
        } else if (manual) {
            crop = cv::Rect(cx, cy, cw, ch) & canvasROI;
            if (crop.width <= 0 || crop.height <= 0) { std::cerr << "Error: crop outside canvas\n"; return; }
            tgt_size = crop.size();
        }

        std::cout << "rendering " << tgt_size << " at scale " << tgt_scale << " crop " << crop << "\n";

        const int rotQuad = rotQuadGlobal;
        // Determine output dtype
        const bool useU16 = output_is_u16 && !isCompositeMode;
        const int cvType = useU16 ? CV_16UC1 : CV_8UC1;

        // ---- Zarr setup (if requested) ----
        const size_t CH = 128, CW = 128;
        size_t baseZ = isCompositeMode ? 1 : size_t(std::max(1, num_slices));
        std::vector<size_t> chunks0;
        std::unique_ptr<z5::Dataset> dsOut;
        z5::filesystem::handle::File outFile(wantZarr ? zarrOutputArg : "/dev/null");
        size_t tilesYSrc = 0, tilesXSrc = 0;

        if (wantZarr) {
            cv::Size zarrXY = tgt_size;
            if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(zarrXY.width, zarrXY.height);
            size_t baseY = zarrXY.height, baseX = zarrXY.width;

            outFile = z5::filesystem::handle::File(zarrOutputArg);
            std::vector<size_t> shape0 = {baseZ, baseY, baseX};
            chunks0 = {shape0[0], std::min(CH, shape0[1]), std::min(CW, shape0[2])};
            json compOpts = {{"cname","zstd"},{"clevel",1},{"shuffle",0}};
            std::string dtype = useU16 ? "uint16" : "uint8";

            if (pre_flag) {
                std::cout << "[pre] creating zarr + all levels...\n";
                z5::createFile(outFile, true);
                z5::createDataset(outFile, "0", dtype, shape0, chunks0, std::string("blosc"), compOpts);
                std::cout << "[pre] L0 shape: [" << shape0[0] << "," << shape0[1] << "," << shape0[2] << "]\n";
                createPyramidDatasets(outFile, shape0, CH, CW, useU16);

                cv::Size attrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(attrXY.width, attrXY.height);
                writeZarrAttrs(outFile, vol_path, group_idx, baseZ, slice_step, accum_step,
                               accum_type_str, accumOffsets.size(), attrXY, baseZ, CH, CW);
                return;
            } else if (finalize_flag) {
                dsOut = z5::openDataset(outFile, "0");
            } else if (numParts > 1) {
                if (!std::filesystem::exists(std::filesystem::path(zarrOutputArg) / "0" / ".zarray")) {
                    std::cerr << "Error: run --pre first in multi-part mode\n"; return;
                }
                dsOut = z5::openDataset(outFile, "0");
            } else {
                z5::createFile(outFile, true);
                dsOut = z5::createDataset(outFile, "0", dtype, shape0, chunks0, std::string("blosc"), compOpts);
            }

            tilesYSrc = (tgt_size.height + CH - 1) / CH;
            tilesXSrc = (tgt_size.width  + CW - 1) / CW;
        }

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
                    if (!wantZarr) { std::cout << "[tif] all slices exist, skipping.\n"; return; }
                    std::cout << "[tif] all slices exist, skipping tif output.\n";
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

        // ---- Render pass ----
        if (!finalize_flag) {
            if (wantZarr) {
                // Tile-based: OMP-parallel over output zarr chunks
                if (useU16)
                    renderTiles<uint16_t>(surf.get(), ds.get(), &chunk_cache_u16,
                        full_size, crop, tgt_size, float(render_scale), scale_seg, ds_scale,
                        hasAffine, affineTransform, num_slices, slice_step,
                        accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                        compositeParams, rotQuad, flip_axis, numParts, partId, cvType,
                        dsOut.get(), chunks0, tilesXSrc, tilesYSrc,
                        tifWriters.empty() ? nullptr : &tifWriters, tiffTileH, quickTif);
                else
                    renderTiles<uint8_t>(surf.get(), ds.get(), &chunk_cache_u8,
                        full_size, crop, tgt_size, float(render_scale), scale_seg, ds_scale,
                        hasAffine, affineTransform, num_slices, slice_step,
                        accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                        compositeParams, rotQuad, flip_axis, numParts, partId, cvType,
                        dsOut.get(), chunks0, tilesXSrc, tilesYSrc,
                        tifWriters.empty() ? nullptr : &tifWriters, tiffTileH, quickTif);
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

                if (useU16)
                    renderBands<uint16_t>(surf.get(), ds.get(), &chunk_cache_u16,
                        full_size, crop, tgt_size, float(render_scale), scale_seg, ds_scale,
                        hasAffine, affineTransform, num_slices, slice_step,
                        accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                        compositeParams, rotQuad, flip_axis, numParts, partId, cvType, bandH, writerFn);
                else
                    renderBands<uint8_t>(surf.get(), ds.get(), &chunk_cache_u8,
                        full_size, crop, tgt_size, float(render_scale), scale_seg, ds_scale,
                        hasAffine, affineTransform, num_slices, slice_step,
                        accumOffsets, accumType, isCompositeMode, compositeStart, compositeEnd,
                        compositeParams, rotQuad, flip_axis, numParts, partId, cvType, bandH, writerFn);
            }

            tifWriters.clear();
        }

        // ---- Zarr pyramid + attrs ----
        if (wantZarr && !pre_flag) {
            if (finalize_flag) std::cout << "[finalize] building pyramid...\n";

            // In single-part mode, create pyramid level datasets now
            // (multi-part creates them in --pre step)
            if (numParts <= 1 && !finalize_flag) {
                cv::Size zarrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(zarrXY.width, zarrXY.height);
                std::vector<size_t> shape0 = {baseZ, size_t(zarrXY.height), size_t(zarrXY.width)};
                createPyramidDatasets(outFile, shape0, CH, CW, useU16);
            }

            for (int level = 1; level <= 5; level++) {
                if (useU16) buildPyramidLevel<uint16_t>(outFile, level, CH, CW, numParts, partId);
                else        buildPyramidLevel<uint8_t>(outFile, level, CH, CW, numParts, partId);
            }

            // --pre already writes attrs for multi-part; single-part writes here
            if (numParts <= 1) {
                cv::Size attrXY = tgt_size;
                if (rotQuad >= 0 && (rotQuad % 2) == 1) std::swap(attrXY.width, attrXY.height);
                writeZarrAttrs(outFile, vol_path, group_idx, baseZ, slice_step, accum_step,
                               accum_type_str, accumOffsets.size(), attrXY, baseZ, CH, CW);
            }
        }
    };

    process_one(seg_path);

    // Print final cache stats
    auto printStats = [](const char* name, const auto& s) {
        uint64_t tot = s.hits + s.misses;
        if (tot == 0) return;
        std::cout << "[" << name << " cache] hits=" << s.hits << " miss=" << s.misses
                  << " rate=" << std::fixed << std::setprecision(1) << (100.0*s.hits/tot) << "%"
                  << " evict=" << s.evictions
                  << " read=" << std::setprecision(2) << s.bytesRead/(1024.0*1024.0*1024.0) << "GB"
                  << " re-read=" << s.reReads << "(" << s.reReadBytes/(1024.0*1024.0*1024.0) << "GB)\n";
    };
    printStats("u8", chunk_cache_u8.stats());
    printStats("u16", chunk_cache_u16.stats());

    return EXIT_SUCCESS;
}