#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/ABFFlattening.hpp"
#include "vc/core/zarr/ZarrDataset.hpp"
#include "vc/core/zarr/Tensor3D.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <functional>
#include <boost/program_options.hpp>
#include <mutex>
#include <cmath>
#include <set>
#include <cctype>
#include <chrono>

namespace po = boost::program_options;

using json = nlohmann::json;

static inline bool isNaN3f(const cv::Vec3f& v)
{
    return std::isnan(v[0]) || std::isnan(v[1]) || std::isnan(v[2]);
}

struct AffineTransform {
    cv::Mat_<double> matrix;  // 4x4 matrix in XYZ format

    AffineTransform() {
        matrix = cv::Mat_<double>::eye(4, 4);
    }

    cv::Matx33d linearPart() const {
        return cv::Matx33d(
            matrix(0,0), matrix(0,1), matrix(0,2),
            matrix(1,0), matrix(1,1), matrix(1,2),
            matrix(2,0), matrix(2,1), matrix(2,2));
    }
};

/**
 * @brief Invert an affine transform in-place.
 *        M = [A | t; 0 0 0 1]  ->  M^{-1} = [A^{-1} | -A^{-1} t; 0 0 0 1]
 * @return bool True on success, false if A is non-invertible.
 */
static inline bool invertAffineInPlace(AffineTransform& T)
{
    cv::Mat A_cv(T.linearPart());
    cv::Mat Ainv_cv;
    double det = cv::invert(A_cv, Ainv_cv, cv::DECOMP_LU);
    if (det < 1e-10) return false;
    cv::Matx33d Ainv(Ainv_cv.at<double>(0,0), Ainv_cv.at<double>(0,1), Ainv_cv.at<double>(0,2),
                     Ainv_cv.at<double>(1,0), Ainv_cv.at<double>(1,1), Ainv_cv.at<double>(1,2),
                     Ainv_cv.at<double>(2,0), Ainv_cv.at<double>(2,1), Ainv_cv.at<double>(2,2));
    const cv::Vec3d t(T.matrix(0,3), T.matrix(1,3), T.matrix(2,3));
    const cv::Vec3d tinv = -(Ainv * t);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.matrix(r, c) = Ainv(r, c);
        T.matrix(r, 3) = tinv(r);
    }
    T.matrix(3,0) = 0.0; T.matrix(3,1) = 0.0; T.matrix(3,2) = 0.0; T.matrix(3,3) = 1.0;
    return true;
}


/**
 * @brief Load affine transform from file (JSON)
 * 
 * @param filename Path to affine transform file
 * @return AffineTransform Loaded transform data
 */
AffineTransform loadAffineTransform(const std::string& filename) {
    AffineTransform transform;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open affine transform file: " + filename);
    }
    
    try {
        json j;
        file >> j;
        
        if (j.contains("transformation_matrix")) {
            auto mat = j["transformation_matrix"];
            if (mat.size() != 3 && mat.size() != 4) {
                throw std::runtime_error("Affine matrix must have 3 or 4 rows");
            }

            for (int row = 0; row < (int)mat.size(); row++) {
                if (mat[row].size() != 4) {
                    throw std::runtime_error("Each row of affine matrix must have 4 elements");
                }
                for (int col = 0; col < 4; col++) {
                    transform.matrix.at<double>(row, col) = mat[row][col].get<double>();
                }
            }
            // If 3x4 provided, bottom row remains [0 0 0 1] from identity ctor.
            if (mat.size() == 4) {
                // Optional: sanity-check bottom row is [0 0 0 1] within tolerance
                const double a30 = transform.matrix(3,0);
                const double a31 = transform.matrix(3,1);
                const double a32 = transform.matrix(3,2);
                const double a33 = transform.matrix(3,3);
                if (std::abs(a30) > 1e-12 || std::abs(a31) > 1e-12 ||
                    std::abs(a32) > 1e-12 || std::abs(a33 - 1.0) > 1e-12)
                    throw std::runtime_error("Bottom affine row must be [0,0,0,1]");
            }
        }
    } catch (json::parse_error&) {
        throw std::runtime_error("Error parsing affine transform file: " + filename);
    }

    return transform;
}

/**
 * @brief Compose two affines: result = B * A (apply A first, then B)
 */
static inline AffineTransform composeAffine(const AffineTransform& A, const AffineTransform& B)
{
    AffineTransform R;
    cv::Mat tmp = B.matrix * A.matrix; // left-multiply: row-vector convention used in code
    tmp.copyTo(R.matrix);
    return R;
}

/**
 * @brief Pretty-print a 4x4 matrix to stdout.
 */
static inline void printMat4x4(const cv::Mat_<double>& M, const char* header)
{
    if (header) std::cout << header << "\n";
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(6);
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

/**
 * @brief Parse an affine spec string possibly ending with an inversion hint.
 *        Accepted suffixes: :inv, :invert, :i  (e.g., "path/to/A.json:inv").
 *        Only the trailing token is interpreted, so Windows "C:\..." paths are safe.
 */

static inline std::pair<std::string, bool> parseAffineSpec(const std::string& spec)
{
    std::string path = spec;
    bool inv = false;
    const std::vector<std::string> suffixes = {":inv", ":invert", ":i"};
    for (const auto& suffix : suffixes) {
        if (spec.size() > suffix.size() &&
            spec.substr(spec.size() - suffix.size()) == suffix) {
            inv = true;
            path = spec.substr(0, spec.size() - suffix.size());
            break;
        }
    }
    return {path, inv};
}


/**
 * @brief Apply affine transform to points and normals
 * 
 * @param points Points to transform (modified in-place)
 * @param normals Normals to transform (modified in-place)
 * @param transform Affine transform to apply
 */
void applyAffineTransform(cv::Mat_<cv::Vec3f>& points,
                         cv::Mat_<cv::Vec3f>& normals,
                         const AffineTransform& transform) {
    const cv::Matx33d A = transform.linearPart();
    const cv::Matx33d invAT = A.inv().t();
    const cv::Vec3d t(transform.matrix(0,3), transform.matrix(1,3), transform.matrix(2,3));

    for (int y = 0; y < points.rows; y++) {
        for (int x = 0; x < points.cols; x++) {
            cv::Vec3f& pt = points(y, x);
            cv::Vec3f& n = normals(y, x);

            if (!isNaN3f(pt)) {
                cv::Vec3d pd(pt[0], pt[1], pt[2]);
                cv::Vec3d r = A * pd + t;
                pt = cv::Vec3f(static_cast<float>(r[0]), static_cast<float>(r[1]), static_cast<float>(r[2]));
            }

            if (!isNaN3f(n)) {
                cv::Vec3d nd(n[0], n[1], n[2]);
                cv::Vec3d rn = invAT * nd;
                double len = cv::norm(rn);
                if (len > 0.0) rn /= len;
                n = cv::Vec3f(static_cast<float>(rn[0]), static_cast<float>(rn[1]), static_cast<float>(rn[2]));
            }
        }
    }
}



static inline int normalizeQuadrantRotation(double angleDeg, double tolDeg = 0.5)
{
    // Map to [0, 360)
    double a = std::fmod(angleDeg, 360.0);
    if (a < 0) a += 360.0;
    // Find nearest multiple of 90
    static const double q[4] = {0.0, 90.0, 180.0, 270.0};
    int best = 0;
    double bestDiff = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) {
        double d = std::abs(a - q[i]);
        if (d < bestDiff) { bestDiff = d; best = i; }
    }
    return (bestDiff <= tolDeg) ? best : -1;
}

// Apply optional right-angle rotation and optional flip in one place
static inline void rotateFlipIfNeeded(cv::Mat& m, int rotQuad, int flip_axis)
{
    if (rotQuad == 1)      cv::rotate(m, m, cv::ROTATE_90_COUNTERCLOCKWISE);
    else if (rotQuad == 2) cv::rotate(m, m, cv::ROTATE_180);
    else if (rotQuad == 3) cv::rotate(m, m, cv::ROTATE_90_CLOCKWISE);
    if (flip_axis >= 0 && flip_axis <= 2)
        cv::flip(m, m, flip_axis == 2 ? -1 : flip_axis);
}

// Map source tile index (tx,ty) in a grid (tilesX,tilesY) to destination index
// after applying a 90-degree-multiple rotation followed by optional flip.
static inline void mapTileIndex(int tx, int ty,
                                int tilesX, int tilesY,
                                int quadRot, int flipType,
                                int& outTx, int& outTy,
                                int& outTilesX, int& outTilesY)
{
    const bool swap = (quadRot % 2) == 1;
    const int rTilesX = swap ? tilesY : tilesX;
    const int rTilesY = swap ? tilesX : tilesY;

    int rx = tx, ry = ty;
    switch (quadRot) {
        case 0: rx = tx;                ry = ty;                break;
        case 1: rx = ty;                ry = (tilesX - 1 - tx); break; // 90 CCW
        case 2: rx = (tilesX - 1 - tx); ry = (tilesY - 1 - ty); break; // 180
        case 3: rx = (tilesY - 1 - ty); ry = tx;                break; // 270 CCW
        default: rx = tx; ry = ty; break;
    }

    int fx = rx, fy = ry;
    if (flipType == 0) {
        // Vertical flip: flip rows
        fy = (rTilesY - 1 - ry);
    } else if (flipType == 1) {
        // Horizontal flip: flip columns
        fx = (rTilesX - 1 - rx);
    } else if (flipType == 2) {
        // Both
        fx = (rTilesX - 1 - rx);
        fy = (rTilesY - 1 - ry);
    }

    outTx = fx;
    outTy = fy;
    outTilesX = rTilesX;
    outTilesY = rTilesY;
}


// Compute a global normal orientation flip decision using a small probe tile
static inline bool computeGlobalFlipDecision(
    QuadSurface* surf,
    int dx0, int dy0,
    float u0, float v0,
    float render_scale, float scale_seg,
    bool hasAffine, const AffineTransform& affineTransform,
    cv::Vec3f& outCentroid)
{
    cv::Mat_<cv::Vec3f> _tp, _tn;
    surf->gen(&_tp, &_tn, cv::Size(dx0, dy0),
              cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0, v0, 0.0f));
    _tp *= scale_seg;
    if (hasAffine) applyAffineTransform(_tp, _tn, affineTransform);

    // Compute centroid of valid points
    cv::Vec3f centroid(0,0,0);
    int count = 0;
    for (int y = 0; y < _tp.rows; y++)
        for (int x = 0; x < _tp.cols; x++) {
            const cv::Vec3f& pt = _tp(y, x);
            if (!isNaN3f(pt)) { centroid += pt; count++; }
        }
    if (count > 0) centroid /= static_cast<float>(count);
    outCentroid = centroid;

    // Count normals pointing toward vs away from centroid
    size_t toward = 0, away = 0;
    for (int y = 0; y < _tp.rows; y++)
        for (int x = 0; x < _tp.cols; x++) {
            const cv::Vec3f& pt = _tp(y, x);
            const cv::Vec3f& n = _tn(y, x);
            if (isNaN3f(pt) || isNaN3f(n)) continue;
            if ((centroid - pt).dot(n) > 0) toward++; else away++;
        }
    return away > toward;
}

// Given raw tile points/normals, produce dataset-space base points and normalized step dirs
static inline void prepareBasePointsAndStepDirs(
    const cv::Mat_<cv::Vec3f>& tilePoints,
    const cv::Mat_<cv::Vec3f>& tileNormals,
    float scale_seg,
    float ds_scale,
    bool hasAffine,
    const AffineTransform& affineTransform,
    bool globalFlipDecision,
    cv::Mat_<cv::Vec3f>& basePointsOut,
    cv::Mat_<cv::Vec3f>& stepDirsOut)
{
    basePointsOut = tilePoints.clone();
    basePointsOut *= scale_seg;
    stepDirsOut = tileNormals.clone();
    if (hasAffine) applyAffineTransform(basePointsOut, stepDirsOut, affineTransform);
    // Flip and normalize normals in a single pass
    const float flipSign = globalFlipDecision ? -1.0f : 1.0f;
    for (int yy = 0; yy < stepDirsOut.rows; ++yy)
        for (int xx = 0; xx < stepDirsOut.cols; ++xx) {
            cv::Vec3f& v = stepDirsOut(yy, xx);
            if (isNaN3f(v)) continue;
            v *= flipSign;
            float L = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
            if (L > 0) v /= L;
        }
    basePointsOut *= ds_scale;
}

// Compute canvas-centered origin (u0,v0) for given target size
static inline void computeCanvasOrigin(const cv::Size& size, float& u0, float& v0)
{
    u0 = -0.5f * (static_cast<float>(size.width)  - 1.0f);
    v0 = -0.5f * (static_cast<float>(size.height) - 1.0f);
}

// Thin wrapper around QuadSurface::gen with consistent parameters
static inline void genTile(
    QuadSurface* surf,
    const cv::Size& size,
    float render_scale,
    float u0, float v0,
    cv::Mat_<cv::Vec3f>& points,
    cv::Mat_<cv::Vec3f>& normals)
{
    surf->gen(&points, &normals, size, cv::Vec3f(0,0,0), render_scale, cv::Vec3f(u0, v0, 0.0f));
}

// Print tiled progress (OMP-safe)
static inline void printTileProgress(
    std::atomic<size_t>& tilesDone, size_t totalTiles, const char* label)
{
    size_t done = ++tilesDone;
    int pct = static_cast<int>(100.0 * double(done) / double(totalTiles));
    #pragma omp critical(progress_print)
    {
        std::cout << "\r[" << label << "] tile " << done << "/" << totalTiles
                  << " (" << pct << "%)" << std::flush;
    }
}

// Generate tile points/normals and prepare base points + step dirs for rendering
static inline void genTileAndPrepare(
    QuadSurface* surf,
    const cv::Size& tileSize,
    const cv::Size& fullSize,
    size_t x0_src, size_t y0_src,
    const cv::Rect& crop,
    float render_scale,
    float scale_seg, float ds_scale,
    bool hasAffine, const AffineTransform& affineTransform,
    bool globalFlipDecision,
    cv::Mat_<cv::Vec3f>& basePoints,
    cv::Mat_<cv::Vec3f>& stepDirs)
{
    float u0, v0;
    computeCanvasOrigin(fullSize, u0, v0);
    u0 += static_cast<float>(x0_src + static_cast<size_t>(crop.x));
    v0 += static_cast<float>(y0_src + static_cast<size_t>(crop.y));
    cv::Mat_<cv::Vec3f> tilePoints, tileNormals;
    genTile(surf, tileSize, render_scale, u0, v0, tilePoints, tileNormals);
    prepareBasePointsAndStepDirs(tilePoints, tileNormals,
        scale_seg, ds_scale, hasAffine, affineTransform,
        globalFlipDecision, basePoints, stepDirs);
}

// Core tiled rendering loop: row-sequential with parallel tiles per row.
template<typename TileFunc>
static void renderTiledLoop(
    QuadSurface* surf,
    const cv::Size& tgt_size,
    const cv::Size& full_size,
    const cv::Rect& crop,
    float render_scale,
    float scale_seg, float ds_scale,
    bool hasAffine, const AffineTransform& affineTransform,
    cv::Vec3f& meshCentroid,
    std::atomic<size_t>& tilesDone, size_t totalTiles,
    const char* progressLabel,
    uint32_t tileW, uint32_t tileH,
    TileFunc&& tileFunc)
{
    const uint32_t tilesX = (static_cast<uint32_t>(tgt_size.width)  + tileW - 1) / tileW;
    const uint32_t tilesY = (static_cast<uint32_t>(tgt_size.height) + tileH - 1) / tileH;

    bool globalFlip = false;
    {
        const int dx0 = std::min<int>(static_cast<int>(tileW), tgt_size.width);
        const int dy0 = std::min<int>(static_cast<int>(tileH), tgt_size.height);
        float u0, v0;
        computeCanvasOrigin(full_size, u0, v0);
        u0 += static_cast<float>(crop.x);
        v0 += static_cast<float>(crop.y);
        globalFlip = computeGlobalFlipDecision(
            surf, dx0, dy0, u0, v0, render_scale,
            scale_seg, hasAffine, affineTransform, meshCentroid);
    }

    for (uint32_t ty = 0; ty < tilesY; ++ty) {
        #pragma omp parallel for schedule(dynamic)
        for (long long tx = 0; tx < static_cast<long long>(tilesX); ++tx) {
            const uint32_t x0_src = static_cast<uint32_t>(tx) * tileW;
            const uint32_t y0_src = ty * tileH;
            const uint32_t dx = std::min<uint32_t>(tileW, static_cast<uint32_t>(tgt_size.width) - x0_src);
            const uint32_t dy = std::min<uint32_t>(tileH, static_cast<uint32_t>(tgt_size.height) - y0_src);

            cv::Mat_<cv::Vec3f> basePoints, stepDirs;
            genTileAndPrepare(surf,
                cv::Size(static_cast<int>(dx), static_cast<int>(dy)),
                full_size, x0_src, y0_src, crop,
                render_scale, scale_seg, ds_scale,
                hasAffine, affineTransform, globalFlip,
                basePoints, stepDirs);

            tileFunc(static_cast<int>(tx), static_cast<int>(ty),
                     static_cast<int>(dx), static_cast<int>(dy),
                     basePoints, stepDirs);

            printTileProgress(tilesDone, totalTiles, progressLabel);
        }
    }
    std::cout << std::endl;
}

// Local trilinear interpolation from contiguous Tensor3D memory.
// Tensor3D is row-major ZYX: shape (nz, ny, nx), strides [ny*nx, nx, 1].
template<typename T>
static inline float sampleTrilinearLocal(
    const T* data, int nz, int ny, int nx,
    float z, float y, float x)
{
    int iz = static_cast<int>(z);
    int iy = static_cast<int>(y);
    int ix = static_cast<int>(x);
    if (iz < 0 || iy < 0 || ix < 0 ||
        iz + 1 >= nz || iy + 1 >= ny || ix + 1 >= nx)
        return 0;
    float fz = z - iz, fy = y - iy, fx = x - ix;
    size_t sZ = static_cast<size_t>(ny) * nx, sY = static_cast<size_t>(nx);
    size_t base = static_cast<size_t>(iz) * sZ + static_cast<size_t>(iy) * sY + static_cast<size_t>(ix);
    float c000 = data[base],          c001 = data[base + 1];
    float c010 = data[base + sY],     c011 = data[base + sY + 1];
    float c100 = data[base + sZ],     c101 = data[base + sZ + 1];
    float c110 = data[base + sZ+sY],  c111 = data[base + sZ+sY+1];
    float c00 = c000 + fx*(c001-c000), c01 = c010 + fx*(c011-c010);
    float c10 = c100 + fx*(c101-c100), c11 = c110 + fx*(c111-c110);
    float c0 = c00 + fy*(c01-c00), c1 = c10 + fy*(c11-c10);
    return c0 + fz*(c1-c0);
}

// Local nearest-neighbor sampling from contiguous Tensor3D memory (ZYX layout).
template<typename T>
static inline T sampleNearestLocal(
    const T* data, int nz, int ny, int nx,
    float z, float y, float x)
{
    int iz = static_cast<int>(std::round(z));
    int iy = static_cast<int>(std::round(y));
    int ix = static_cast<int>(std::round(x));
    if (iz < 0 || iy < 0 || ix < 0 || iz >= nz || iy >= ny || ix >= nx)
        return 0;
    size_t sZ = static_cast<size_t>(ny) * nx, sY = static_cast<size_t>(nx);
    return data[static_cast<size_t>(iz) * sZ + static_cast<size_t>(iy) * sY + static_cast<size_t>(ix)];
}

// Bulk-read a 3D bounding box and render all slices via local trilinear interpolation.
// Returns true on success, false if bbox is too large (caller should fall back).
//
// Coordinate conventions:
//   Dataset shape & Tensor3D: ZYX (dim0=Z, dim1=Y, dim2=X)
//   readArea3D offset Vec3i:  [0]=Z, [1]=Y, [2]=X
//   coords Vec3f (basePoints/stepDirs): [0]=X, [1]=Y, [2]=Z  (legacy XYZ order)
template<typename T>
static bool renderSlicesBulk(
    std::vector<cv::Mat>& outSlices,
    volcart::zarr::ZarrDataset* ds,
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    float ds_scale,
    bool nearest_neighbor,
    const std::vector<float>& allOffsets)
{
    constexpr int cvType = std::is_same_v<T, uint16_t> ? CV_16UC1 : CV_8UC1;
    const int rows = basePoints.rows;
    const int cols = basePoints.cols;
    const size_t nOffsets = allOffsets.size();
    if (nOffsets == 0 || rows == 0 || cols == 0) return false;

    // Find min/max offset values
    float minOff = allOffsets[0], maxOff = allOffsets[0];
    for (size_t i = 1; i < nOffsets; ++i) {
        if (allOffsets[i] < minOff) minOff = allOffsets[i];
        if (allOffsets[i] > maxOff) maxOff = allOffsets[i];
    }
    const float scaleMin = minOff * ds_scale;
    const float scaleMax = maxOff * ds_scale;

    // Compute AABB across all pixels at min and max offsets.
    // coords Vec3f: [0]=X, [1]=Y, [2]=Z
    float z_min_f = std::numeric_limits<float>::max(), z_max_f = std::numeric_limits<float>::lowest();
    float y_min_f = z_min_f, y_max_f = z_max_f;
    float x_min_f = z_min_f, x_max_f = z_max_f;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& p = basePoints(r, c);
            const cv::Vec3f& d = stepDirs(r, c);
            if (isNaN3f(p) || isNaN3f(d)) continue;
            for (int e = 0; e < 2; ++e) {
                float s = (e == 0) ? scaleMin : scaleMax;
                float cx = p[0] + s * d[0]; // coords[0] = X
                float cy = p[1] + s * d[1]; // coords[1] = Y
                float cz = p[2] + s * d[2]; // coords[2] = Z
                if (cz < z_min_f) z_min_f = cz;
                if (cz > z_max_f) z_max_f = cz;
                if (cy < y_min_f) y_min_f = cy;
                if (cy > y_max_f) y_max_f = cy;
                if (cx < x_min_f) x_min_f = cx;
                if (cx > x_max_f) x_max_f = cx;
            }
        }
    }

    if (z_min_f > z_max_f) return false; // all NaN

    // Sample directly from cached chunks — no readArea3D copy needed
    const int cw = cache->chunkSizeX(), ch = cache->chunkSizeY(), cd = cache->chunkSizeZ();
    const int cwShift = cache->chunkShiftX(), chShift = cache->chunkShiftY(), cdShift = cache->chunkShiftZ();
    const int cwMask = cache->chunkMaskX(), chMask = cache->chunkMaskY(), cdMask = cache->chunkMaskZ();
    const int dsZ = cache->datasetSizeX(), dsY = cache->datasetSizeY(), dsX = cache->datasetSizeZ();

    outSlices.resize(nOffsets);
    for (size_t si = 0; si < nOffsets; ++si) {
        outSlices[si].create(rows, cols, cvType);
    }

    for (size_t si = 0; si < nOffsets; ++si) {
        const float scale = allOffsets[si] * ds_scale;
        cv::Mat& out = outSlices[si];
        for (int r = 0; r < rows; ++r) {
            T* row = out.ptr<T>(r);
            for (int c = 0; c < cols; ++c) {
                const cv::Vec3f& p = basePoints(r, c);
                const cv::Vec3f& d = stepDirs(r, c);
                if (isNaN3f(p) || isNaN3f(d)) {
                    row[c] = 0;
                    continue;
                }
                // coords: [0]=X, [1]=Y, [2]=Z; chunk dims: X=dim0, Y=dim1, Z=dim2
                float cx = p[0] + scale * d[0];
                float cy = p[1] + scale * d[1];
                float cz = p[2] + scale * d[2];

                if (nearest_neighbor) {
                    int iz = static_cast<int>(cz + 0.5f);
                    int iy = static_cast<int>(cy + 0.5f);
                    int ix = static_cast<int>(cx + 0.5f);
                    if (iz < 0 || iz >= dsZ || iy < 0 || iy >= dsY || ix < 0 || ix >= dsX) {
                        row[c] = 0; continue;
                    }
                    auto ptr = cache->get(iz >> cwShift, iy >> chShift, ix >> cdShift);
                    if (!ptr) { row[c] = 0; continue; }
                    row[c] = (*ptr)(iz & cwMask, iy & chMask, ix & cdMask);
                } else {
                    int iz = static_cast<int>(cz);
                    int iy = static_cast<int>(cy);
                    int ix = static_cast<int>(cx);
                    if (iz < 0 || iy < 0 || ix < 0 ||
                        iz + 1 >= dsZ || iy + 1 >= dsY || ix + 1 >= dsX) {
                        row[c] = 0; continue;
                    }
                    float fz = cz - iz, fy = cy - iy, fx = cx - ix;

                    // Fast path: all 8 neighbors in same chunk
                    int ciz = iz >> cwShift, ciy = iy >> chShift, cix = ix >> cdShift;
                    int ciz1 = (iz+1) >> cwShift, ciy1 = (iy+1) >> chShift, cix1 = (ix+1) >> cdShift;
                    if (ciz == ciz1 && ciy == ciy1 && cix == cix1) {
                        auto ptr = cache->get(ciz, ciy, cix);
                        if (!ptr) { row[c] = 0; continue; }
                        const auto& chunk = *ptr;
                        int lz = iz & cwMask, ly = iy & chMask, lx = ix & cdMask;
                        float c000 = chunk(lz, ly, lx),     c001 = chunk(lz, ly, lx+1);
                        float c010 = chunk(lz, ly+1, lx),   c011 = chunk(lz, ly+1, lx+1);
                        float c100 = chunk(lz+1, ly, lx),   c101 = chunk(lz+1, ly, lx+1);
                        float c110 = chunk(lz+1, ly+1, lx), c111 = chunk(lz+1, ly+1, lx+1);
                        float c00 = c000 + fx*(c001-c000), c01 = c010 + fx*(c011-c010);
                        float c10 = c100 + fx*(c101-c100), c11 = c110 + fx*(c111-c110);
                        float v0 = c00 + fy*(c01-c00), v1 = c10 + fy*(c11-c10);
                        float val = v0 + fz*(v1-v0);
                        row[c] = static_cast<T>(std::min<float>(val + 0.5f,
                            static_cast<float>(std::numeric_limits<T>::max())));
                    } else {
                        // Slow path: neighbors span chunk boundaries
                        auto getVoxel = [&](int vz, int vy, int vx) -> float {
                            auto ptr = cache->get(vz >> cwShift, vy >> chShift, vx >> cdShift);
                            if (!ptr) return 0;
                            return static_cast<float>((*ptr)(vz & cwMask, vy & chMask, vx & cdMask));
                        };
                        float c000 = getVoxel(iz,iy,ix),     c001 = getVoxel(iz,iy,ix+1);
                        float c010 = getVoxel(iz,iy+1,ix),   c011 = getVoxel(iz,iy+1,ix+1);
                        float c100 = getVoxel(iz+1,iy,ix),   c101 = getVoxel(iz+1,iy,ix+1);
                        float c110 = getVoxel(iz+1,iy+1,ix), c111 = getVoxel(iz+1,iy+1,ix+1);
                        float c00 = c000 + fx*(c001-c000), c01 = c010 + fx*(c011-c010);
                        float c10 = c100 + fx*(c101-c100), c11 = c110 + fx*(c111-c110);
                        float v0 = c00 + fy*(c01-c00), v1 = c10 + fy*(c11-c10);
                        float val = v0 + fz*(v1-v0);
                        row[c] = static_cast<T>(std::min<float>(val + 0.5f,
                            static_cast<float>(std::numeric_limits<T>::max())));
                    }
                }
            }
        }
    }

    return true;
}

// Render one slice from base points and unit step directions at offset `off`
template<typename T>
static inline void renderSliceFromBaseT(
    cv::Mat& out,
    volcart::zarr::ZarrDataset* ds,
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    float off,
    float ds_scale,
    bool nearest_neighbor = false)
{
    cv::Mat_<cv::Vec3f> coords(basePoints.size());
    const float scale = off * static_cast<float>(ds_scale);
    for (int yy = 0; yy < coords.rows; ++yy) {
        for (int xx = 0; xx < coords.cols; ++xx) {
            const cv::Vec3f& p = basePoints(yy, xx);
            const cv::Vec3f& d = stepDirs(yy, xx);
            coords(yy, xx) = p + scale * d;
        }
    }
    cv::Mat_<T> tmp;
    readInterpolated3D(tmp, ds, coords, cache, nearest_neighbor);
    out = tmp;
}

enum class AccumType {
    Max,
    Mean,
    Median
};

template <typename T>
static inline void computeMedianFromSamples(const std::vector<cv::Mat>& samples, cv::Mat& out)
{
    if (samples.empty()) {
        out.release();
        return;
    }
    const int rows = samples.front().rows;
    const int cols = samples.front().cols;
    out.create(rows, cols, samples.front().type());
    std::vector<T> values(samples.size());
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            for (size_t i = 0; i < samples.size(); ++i) {
                values[i] = samples[i].at<T>(y, x);
            }
            std::sort(values.begin(), values.end());
            if ((values.size() % 2) == 1) {
                out.at<T>(y, x) = values[values.size() / 2];
            } else {
                const T a = values[values.size() / 2 - 1];
                const T b = values[values.size() / 2];
                out.at<T>(y, x) = static_cast<T>((static_cast<uint32_t>(a) + static_cast<uint32_t>(b) + 1) / 2);
            }
        }
    }
}

template <typename RenderFunc>
static inline void renderAccumulatedSlice(
    cv::Mat& out,
    RenderFunc&& renderFunc,
    float baseOff,
    const std::vector<float>& accumOffsets,
    AccumType accumType,
    int cvType)
{
    if (accumOffsets.empty()) {
        renderFunc(out, baseOff);
        return;
    }

    const size_t sampleCount = accumOffsets.size();

    if (sampleCount == 1) {
        renderFunc(out, baseOff + accumOffsets.front());
        return;
    }

    if (accumType == AccumType::Max) {
        renderFunc(out, baseOff + accumOffsets.front());
        cv::Mat tmp;
        for (size_t i = 1; i < sampleCount; ++i) {
            renderFunc(tmp, baseOff + accumOffsets[i]);
            cv::max(out, tmp, out);
        }
    } else if (accumType == AccumType::Mean) {
        cv::Mat sample;
        renderFunc(sample, baseOff + accumOffsets.front());
        cv::Mat sum;
        sample.convertTo(sum, CV_64F);
        for (size_t i = 1; i < sampleCount; ++i) {
            renderFunc(sample, baseOff + accumOffsets[i]);
            cv::Mat tmp;
            sample.convertTo(tmp, CV_64F);
            sum += tmp;
        }
        sum /= static_cast<double>(sampleCount);
        sum.convertTo(out, cvType);
    } else { // Median
        std::vector<cv::Mat> samples;
        samples.reserve(sampleCount);
        cv::Mat sample;
        for (size_t i = 0; i < sampleCount; ++i) {
            renderFunc(sample, baseOff + accumOffsets[i]);
            samples.emplace_back(sample.clone());
        }
        if (cvType == CV_16UC1) {
            computeMedianFromSamples<uint16_t>(samples, out);
        } else {
            computeMedianFromSamples<uint8_t>(samples, out);
        }
    }
}

// ============================================================================
// Template helpers to eliminate uint8/uint16 code duplication
// ============================================================================

// Render slices into a zarr chunk, applying rotation/flip per slice
template<typename T>
static inline void renderSlicesToZarrChunk(
    volcart::zarr::Tensor3D<T>& outChunk,
    volcart::zarr::ZarrDataset* ds,
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    float ds_scale,
    bool nearest_neighbor,
    size_t dz, size_t z0, double baseZ_center, double slice_step,
    const std::vector<float>& accumOffsets, AccumType accumType,
    int rotQuad, int flip_axis)
{
    constexpr int cvType = std::is_same_v<T, uint16_t> ? CV_16UC1 : CV_8UC1;

    // Build flat list of ALL offsets to sample (each output slice × accumulation sub-samples)
    std::vector<float> allOffsets;
    std::vector<size_t> sliceStartIdx; // index into allOffsets for each output slice
    for (size_t zi = 0; zi < dz; ++zi) {
        sliceStartIdx.push_back(allOffsets.size());
        const float off = static_cast<float>(
            (static_cast<double>(z0 + zi) - baseZ_center) * slice_step);
        if (accumOffsets.empty()) {
            allOffsets.push_back(off);
        } else {
            for (float ao : accumOffsets) {
                allOffsets.push_back(off + ao);
            }
        }
    }

    // Try bulk read + local interpolation
    std::vector<cv::Mat> bulkSlices;
    bool usedBulk = renderSlicesBulk<T>(bulkSlices, ds, cache,
        basePoints, stepDirs, ds_scale, nearest_neighbor, allOffsets);

    cv::Mat tileOut;
    for (size_t zi = 0; zi < dz; ++zi) {
        if (usedBulk) {
            size_t start = sliceStartIdx[zi];
            size_t count = accumOffsets.empty() ? 1 : accumOffsets.size();
            if (count == 1) {
                tileOut = bulkSlices[start];
            } else if (accumType == AccumType::Max) {
                tileOut = bulkSlices[start].clone();
                for (size_t i = 1; i < count; ++i)
                    cv::max(tileOut, bulkSlices[start + i], tileOut);
            } else if (accumType == AccumType::Mean) {
                cv::Mat sum;
                bulkSlices[start].convertTo(sum, CV_64F);
                for (size_t i = 1; i < count; ++i) {
                    cv::Mat tmp;
                    bulkSlices[start + i].convertTo(tmp, CV_64F);
                    sum += tmp;
                }
                sum /= static_cast<double>(count);
                sum.convertTo(tileOut, cvType);
            } else { // Median
                std::vector<cv::Mat> samples(bulkSlices.begin() + start,
                                              bulkSlices.begin() + start + count);
                if (cvType == CV_16UC1)
                    computeMedianFromSamples<uint16_t>(samples, tileOut);
                else
                    computeMedianFromSamples<uint8_t>(samples, tileOut);
            }
        } else {
            const float off = static_cast<float>(
                (static_cast<double>(z0 + zi) - baseZ_center) * slice_step);
            auto renderOne = [&](cv::Mat& dst, float offset) {
                renderSliceFromBaseT<T>(dst, ds, cache,
                                        basePoints, stepDirs, offset, ds_scale, nearest_neighbor);
            };
            renderAccumulatedSlice(tileOut, renderOne, off, accumOffsets, accumType, cvType);
        }
        if (rotQuad >= 0 || flip_axis >= 0) {
            rotateFlipIfNeeded(tileOut, rotQuad, flip_axis);
        }
        const size_t cH = static_cast<size_t>(tileOut.rows);
        const size_t cW = static_cast<size_t>(tileOut.cols);
        for (size_t yy = 0; yy < cH; ++yy) {
            const T* src = tileOut.ptr<T>(static_cast<int>(yy));
            for (size_t xx = 0; xx < cW; ++xx) {
                outChunk(zi, yy, xx) = src[xx];
            }
        }
    }
}

// Downsample a zarr chunk by 2x in all dimensions using averaging
template<typename T>
static inline void downsampleChunk(
    volcart::zarr::ZarrDataset* src,
    volcart::zarr::ZarrDataset* dst,
    size_t z, size_t y, size_t x,
    size_t lz, size_t ly, size_t lx,
    size_t sz, size_t sy, size_t sx)
{
    volcart::zarr::Tensor3D<T> srcChunk(sz, sy, sx);
    std::vector<size_t> sOff = {static_cast<size_t>(2*z), static_cast<size_t>(2*y), static_cast<size_t>(2*x)};
    src->readSubarray(srcChunk, sOff, {sz, sy, sx});
    volcart::zarr::Tensor3D<T> dstChunk(lz, ly, lx);
    for (size_t zz = 0; zz < lz; ++zz)
        for (size_t yy = 0; yy < ly; ++yy)
            for (size_t xx = 0; xx < lx; ++xx) {
                uint32_t sum = 0; int cnt = 0;
                for (int dz2 = 0; dz2 < 2 && (2*zz + dz2) < sz; ++dz2)
                    for (int dy2 = 0; dy2 < 2 && (2*yy + dy2) < sy; ++dy2)
                        for (int dx2 = 0; dx2 < 2 && (2*xx + dx2) < sx; ++dx2) {
                            sum += srcChunk(2*zz + dz2, 2*yy + dy2, 2*xx + dx2);
                            cnt += 1;
                        }
                dstChunk(zz, yy, xx) = static_cast<T>((sum + (cnt/2)) / std::max(1, cnt));
            }
    std::vector<size_t> dOff = {z, static_cast<size_t>(y), static_cast<size_t>(x)};
    dst->writeSubarray(dstChunk, dOff);
}

// Export zarr L0 tile to TIFF writers
template<typename T>
static inline void exportZarrTileToTiff(
    volcart::zarr::ZarrDataset* dsL0,
    std::vector<TiffWriter>& writers,
    std::vector<std::mutex>& writerLocks,
    size_t Z, uint32_t dy, uint32_t dx,
    uint32_t y0_src, uint32_t x0_src,
    uint32_t x0_dst, uint32_t y0_dst)
{
    constexpr int cvType = std::is_same_v<T, uint16_t> ? CV_16UC1 : CV_8UC1;
    volcart::zarr::Tensor3D<T> tile(Z, static_cast<size_t>(dy), static_cast<size_t>(dx));
    std::vector<size_t> off = {0, static_cast<size_t>(y0_src), static_cast<size_t>(x0_src)};
    dsL0->readSubarray(tile, off, {Z, static_cast<size_t>(dy), static_cast<size_t>(dx)});
    for (size_t z = 0; z < Z; ++z) {
        cv::Mat srcTile(static_cast<int>(dy), static_cast<int>(dx), cvType);
        for (uint32_t yy = 0; yy < dy; ++yy) {
            T* dstRow = srcTile.ptr<T>(static_cast<int>(yy));
            for (uint32_t xx = 0; xx < dx; ++xx) dstRow[xx] = tile(z, yy, xx);
        }
        std::lock_guard<std::mutex> guard(writerLocks[z]);
        writers[z].writeTile(x0_dst, y0_dst, srcTile);
    }
}

// Render slices for TIFF tiled output, writing tiles directly to writers
template<typename T>
static inline void renderSlicesToTiffTiles(
    volcart::zarr::ZarrDataset* ds,
    ChunkCache<T>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    float ds_scale,
    bool nearest_neighbor,
    int num_slices, double num_slices_center, double slice_step,
    const std::vector<float>& accumOffsets, AccumType accumType,
    int rotQuad, int flip_axis,
    int tx, int ty, int tilesX_src, int tilesY_src,
    uint32_t tileW, uint32_t tileH,
    std::vector<TiffWriter>& writers,
    std::vector<std::mutex>& writerLocks)
{
    constexpr int cvType = std::is_same_v<T, uint16_t> ? CV_16UC1 : CV_8UC1;

    // Build flat list of ALL offsets to sample
    std::vector<float> allOffsets;
    std::vector<size_t> sliceStartIdx;
    for (int zi = 0; zi < num_slices; ++zi) {
        sliceStartIdx.push_back(allOffsets.size());
        const float off = static_cast<float>(
            (static_cast<double>(zi) - num_slices_center) * slice_step);
        if (accumOffsets.empty()) {
            allOffsets.push_back(off);
        } else {
            for (float ao : accumOffsets) {
                allOffsets.push_back(off + ao);
            }
        }
    }

    // Try bulk read + local interpolation
    std::vector<cv::Mat> bulkSlices;
    bool usedBulk = renderSlicesBulk<T>(bulkSlices, ds, cache,
        basePoints, stepDirs, ds_scale, nearest_neighbor, allOffsets);

    cv::Mat tileOut;
    for (int zi = 0; zi < num_slices; ++zi) {
        if (usedBulk) {
            size_t start = sliceStartIdx[zi];
            size_t count = accumOffsets.empty() ? 1 : accumOffsets.size();
            if (count == 1) {
                tileOut = bulkSlices[start];
            } else if (accumType == AccumType::Max) {
                tileOut = bulkSlices[start].clone();
                for (size_t i = 1; i < count; ++i)
                    cv::max(tileOut, bulkSlices[start + i], tileOut);
            } else if (accumType == AccumType::Mean) {
                cv::Mat sum;
                bulkSlices[start].convertTo(sum, CV_64F);
                for (size_t i = 1; i < count; ++i) {
                    cv::Mat tmp;
                    bulkSlices[start + i].convertTo(tmp, CV_64F);
                    sum += tmp;
                }
                sum /= static_cast<double>(count);
                sum.convertTo(tileOut, cvType);
            } else { // Median
                std::vector<cv::Mat> samples(bulkSlices.begin() + start,
                                              bulkSlices.begin() + start + count);
                if (cvType == CV_16UC1)
                    computeMedianFromSamples<uint16_t>(samples, tileOut);
                else
                    computeMedianFromSamples<uint8_t>(samples, tileOut);
            }
        } else {
            const float off = static_cast<float>(
                (static_cast<double>(zi) - num_slices_center) * slice_step);
            auto renderOne = [&](cv::Mat& dst, float offset) {
                renderSliceFromBaseT<T>(dst, ds, cache,
                                        basePoints, stepDirs, offset, ds_scale, nearest_neighbor);
            };
            renderAccumulatedSlice(tileOut, renderOne, off, accumOffsets, accumType, cvType);
        }
        cv::Mat tileTransformed = tileOut;
        rotateFlipIfNeeded(tileTransformed, rotQuad, flip_axis);
        int dstTx, dstTy, rTilesX, rTilesY;
        mapTileIndex(tx, ty, tilesX_src, tilesY_src,
                     std::max(rotQuad, 0), flip_axis,
                     dstTx, dstTy, rTilesX, rTilesY);
        const uint32_t x0_dst = static_cast<uint32_t>(dstTx) * tileW;
        const uint32_t y0_dst = static_cast<uint32_t>(dstTy) * tileH;
        std::lock_guard<std::mutex> guard(writerLocks[static_cast<size_t>(zi)]);
        writers[static_cast<size_t>(zi)].writeTile(x0_dst, y0_dst, tileTransformed);
    }
}

int main(int argc, char *argv[])
{
    auto wall_t0 = std::chrono::steady_clock::now();
    // clang-format off
    po::options_description required("Required arguments");
    required.add_options()
        ("volume,v", po::value<std::string>()->required(),
            "Path to the OME-Zarr volume")
        ("output,o", po::value<std::string>()->required(),
            "Output path or name (Zarr: name without extension; TIF: filename or printf pattern)")
        ("scale", po::value<float>()->required(),
            "Pixels per level-g voxel (Pg)")
        ("group-idx,g", po::value<int>()->required(),
            "OME-Zarr group index");

    po::options_description optional("Optional arguments");
    optional.add_options()
        ("help,h", "Show this help message")
        ("segmentation,s", po::value<std::string>(),
            "Path to a single tifxyz segmentation folder (ignored if --render-folder is set)")
        ("render-folder", po::value<std::string>(),
            "Folder containing tifxyz segmentation folders to batch render")
        ("cache-gb", po::value<size_t>()->default_value(16),
            "Zarr chunk cache size in gigabytes (default: 16)")
        ("format", po::value<std::string>(),
            "When using --render-folder, choose 'zarr' or 'tif' output")
        ("num-slices,n", po::value<int>()->default_value(1),
            "Number of slices to render")
        ("slice-step", po::value<float>()->default_value(1.0f),
            "Spacing between successive slices along the surface normal (fractional values allowed)")
        ("accum", po::value<float>()->default_value(0.0f),
            "Optional fractional accumulation step (< slice-step) aggregated into each output slice (0 = disabled)")
        ("accum-type", po::value<std::string>()->default_value("max"),
            "Accumulation reducer when --accum > 0: max, mean, or median")
        ("crop-x", po::value<int>()->default_value(0),
            "Crop region X coordinate")
        ("crop-y", po::value<int>()->default_value(0),
            "Crop region Y coordinate")
        ("crop-width", po::value<int>()->default_value(0),
            "Crop region width (0 = no crop)")
        ("crop-height", po::value<int>()->default_value(0),
            "Crop region height (0 = no crop)")
        // Multi-affine interface (preferred):
        ("affine", po::value<std::vector<std::string>>()->multitoken()->composing(),
            "One or more affine JSON files, in application order (first listed applies first). "
            "You may append :inv / :invert / :i to a spec to invert that transform (e.g., --affine A.json:inv). "
            "Key in JSON: 'transformation_matrix' (3x4 or 4x4).")
        ("affine-invert", po::value<std::vector<int>>()->multitoken()->composing(),
            "0-based indices into --affine to invert before composing (e.g., --affine-invert 0 2).")
        // Backward-compatible single-affine options (deprecated):
        ("affine-transform", po::value<std::string>(),
            "[DEPRECATED] Single affine JSON; prefer --affine. Key 'transformation_matrix' (3x4 or 4x4).")
        ("invert-affine", po::bool_switch()->default_value(false),
            "[DEPRECATED] Invert the single --affine-transform.")
        ("scale-segmentation", po::value<float>()->default_value(1.0),
            "Scale segmentation to target scale")
        ("rotate", po::value<double>()->default_value(0.0),
            "Rotate output image by angle in degrees (counterclockwise)")
        ("flip", po::value<int>()->default_value(-1),
            "Flip output image. 0=Vertical, 1=Horizontal, 2=Both")
        ("include-tifs", po::bool_switch()->default_value(false),
            "If output is Zarr, also export per-Z TIFF slices to layers_{zarrname}")
        ("flatten", po::bool_switch()->default_value(false),
            "Apply ABF++ flattening to the surface before rendering")
        ("flatten-iterations", po::value<int>()->default_value(10),
            "Maximum ABF++ iterations when --flatten is enabled")
        ("flatten-downsample", po::value<int>()->default_value(1),
            "Downsample factor for ABF++ (1=full, 2=half, 4=quarter). Higher = faster but lower quality")
        ("nearest-neighbor", po::bool_switch()->default_value(false),
            "Use nearest-neighbor sampling instead of trilinear interpolation (faster but lower quality)");
    // clang-format on

    po::options_description all("Usage");
    all.add(required).add(optional);

    po::variables_map parsed;
    try {
        po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

        if (parsed.count("help") > 0 || argc < 2) {
            std::cout << "vc_render_tifxyz: Render volume data using segmentation surfaces\n\n";
            std::cout << all << '\n';
            return EXIT_SUCCESS;
        }
        
        po::notify(parsed);
    } catch (po::error& e) {
        std::cerr << "Error: " << e.what() << '\n';
        std::cerr << "Use --help for usage information\n";
        return EXIT_FAILURE;
    }

    std::filesystem::path vol_path = parsed["volume"].as<std::string>();
    std::string base_output_arg = parsed["output"].as<std::string>();
    const bool has_render_folder = parsed.count("render-folder") > 0;
    std::filesystem::path render_folder_path;
    std::string batch_format;
    if (has_render_folder) {
        render_folder_path = std::filesystem::path(parsed["render-folder"].as<std::string>());
        if (parsed.count("format") == 0) {
            std::cerr << "Error: --format is required when using --render-folder (zarr|tif).\n";
            return EXIT_FAILURE;
        }
        batch_format = parsed["format"].as<std::string>();
        std::transform(batch_format.begin(), batch_format.end(), batch_format.begin(), ::tolower);
        if (batch_format != "zarr" && batch_format != "tif") {
            std::cerr << "Error: --format must be 'zarr' or 'tif'.\n";
            return EXIT_FAILURE;
        }
        if (!std::filesystem::exists(render_folder_path) || !std::filesystem::is_directory(render_folder_path)) {
            std::cerr << "Error: --render-folder path is not a directory: " << render_folder_path << "\n";
            return EXIT_FAILURE;
        }
    }
    std::filesystem::path seg_path;
    if (!has_render_folder) {
        if (parsed.count("segmentation") == 0) {
            std::cerr << "Error: --segmentation is required unless --render-folder is used.\n";
            return EXIT_FAILURE;
        }
        seg_path = parsed["segmentation"].as<std::string>();
    }
    float tgt_scale = parsed["scale"].as<float>();
    int group_idx = parsed["group-idx"].as<int>();
    int num_slices = parsed["num-slices"].as<int>();
    double slice_step = static_cast<double>(parsed["slice-step"].as<float>());
    if (!std::isfinite(slice_step) || slice_step <= 0.0) {
        std::cerr << "Error: --slice-step must be positive.\n";
        return EXIT_FAILURE;
    }
    double accum_step = static_cast<double>(parsed["accum"].as<float>());
    if (!std::isfinite(accum_step) || accum_step < 0.0) {
        std::cerr << "Error: --accum must be non-negative.\n";
        return EXIT_FAILURE;
    }
    std::string accum_type_str = parsed["accum-type"].as<std::string>();
    std::transform(accum_type_str.begin(), accum_type_str.end(), accum_type_str.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    AccumType accumType = AccumType::Max;
    if (accum_type_str == "max") {
        accumType = AccumType::Max;
    } else if (accum_type_str == "mean") {
        accumType = AccumType::Mean;
    } else if (accum_type_str == "median") {
        accumType = AccumType::Median;
    } else {
        std::cerr << "Error: --accum-type must be one of: max, mean, median.\n";
        return EXIT_FAILURE;
    }
    std::vector<float> accumOffsets;
    if (accum_step > 0.0) {
        if (accum_step > slice_step) {
            std::cerr << "Error: --accum must be <= --slice-step.\n";
            return EXIT_FAILURE;
        }
        const double ratio = slice_step / accum_step;
        const double rounded = std::round(ratio);
        const double tol = 1e-4;
        if (std::abs(ratio - rounded) > tol) {
            std::cerr << "Error: --accum must evenly divide --slice-step (ratio="
                      << ratio << ").\n";
            return EXIT_FAILURE;
        }
        const size_t samples = std::max<size_t>(1, static_cast<size_t>(rounded));
        const double spacing = slice_step / static_cast<double>(samples);
        accumOffsets.reserve(samples);
        for (size_t i = 0; i < samples; ++i) {
            accumOffsets.push_back(static_cast<float>(spacing * static_cast<double>(i)));
        }
        accum_step = spacing;
        std::cout << "Accumulation enabled: " << samples << " samples per slice at step "
                  << spacing << " using '" << accum_type_str << "' reducer." << std::endl;
    }
    // Downsample factor for this OME-Zarr pyramid level: g=0 -> 1, g=1 -> 0.5, ...
    const float ds_scale = std::ldexp(1.0f, -group_idx);  // 2^(-group_idx)
    float scale_seg = parsed["scale-segmentation"].as<float>();

    double rotate_angle = parsed["rotate"].as<double>();
    int flip_axis = parsed["flip"].as<int>();
    const bool include_tifs = parsed["include-tifs"].as<bool>();
    const bool nearest_neighbor = parsed["nearest-neighbor"].as<bool>();

    if (nearest_neighbor) {
        std::cout << "Using nearest-neighbor sampling (faster, lower quality)" << std::endl;
    }

    AffineTransform affineTransform;
    bool hasAffine = false;
    
    // --- New multi-affine loading & composition ---
    std::vector<std::pair<std::string,bool>> affineSpecs; // (path, invert?)
    if (parsed.count("affine") > 0) {
        for (const auto& s : parsed["affine"].as<std::vector<std::string>>()) {
            affineSpecs.emplace_back(parseAffineSpec(s));
        }
    }
    // Back-compat: single --affine-transform [--invert-affine]
    if (parsed.count("affine-transform") > 0) {
        const std::string singlePath = parsed["affine-transform"].as<std::string>();
        const bool singleInv = parsed["invert-affine"].as<bool>();
        affineSpecs.emplace_back(singlePath, singleInv);
        std::cout << "[deprecated] Using --affine-transform"
                  << (singleInv ? " (with inversion)" : "") << "; prefer --affine.\n";
    }
    // Optional index-based inversion flags for --affine
    if (parsed.count("affine-invert") > 0 && !affineSpecs.empty()) {
        std::set<int> idxInv;
        for (int idx : parsed["affine-invert"].as<std::vector<int>>()) {
            if (idx < 0 || idx >= static_cast<int>(affineSpecs.size())) {
                std::cerr << "Error: --affine-invert index " << idx
                          << " out of range [0.." << (affineSpecs.size()-1) << "].\n";
                return EXIT_FAILURE;
            }
            idxInv.insert(idx);
        }
        int k = 0;
        for (auto& spec : affineSpecs) {
            if (idxInv.count(k)) spec.second = true;
            ++k;
        }
    }
    // Load, optionally invert, then compose (first listed applies first)
    if (!affineSpecs.empty()) {
        AffineTransform composed; // identity
        int k = 0;
        for (const auto& [path, invertFlag] : affineSpecs) {
            try {
                AffineTransform T = loadAffineTransform(path);
                std::cout << "Loaded affine[" << k << "]: " << path
                          << (invertFlag ? " (invert)" : "") << std::endl;
                if (invertFlag) {
                    if (!invertAffineInPlace(T)) {
                        std::cerr << "Error: affine[" << k << "] has non-invertible linear part.\n";
                        return EXIT_FAILURE;
                    }
                }
                composed = composeAffine(composed, T);
                ++k;
            } catch (const std::exception& e) {
                std::cerr << "Error loading affine[" << k << "]: " << e.what() << std::endl;
                return EXIT_FAILURE;

            }
        }
        hasAffine = true;
        affineTransform = composed;
        printMat4x4(affineTransform.matrix, "Final composed affine (applied to points first):");
    }
    
    auto ds = std::make_unique<volcart::zarr::ZarrDataset>(vol_path / std::to_string(group_idx));

    std::cout << "zarr dataset size for scale group " << group_idx << " [";
    for (size_t i = 0; i < ds->shape().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ds->shape()[i];
    }
    std::cout << "]" << std::endl;
    const bool output_is_u16 = (ds->getDtype() == volcart::zarr::Dtype::UInt16);
    if (output_is_u16)
        std::cout << "Detected source dtype=uint16 -> rendering as uint16" << std::endl;
    else
        std::cout << "Detected source dtype!=uint16 -> rendering as uint8 (default)" << std::endl;
    std::cout << "chunk shape [";
    size_t chunk_voxels = 1;
    for (size_t i = 0; i < ds->chunkShape().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ds->chunkShape()[i];
        chunk_voxels *= ds->chunkShape()[i];
    }
    const size_t bytes_per_voxel = output_is_u16 ? 2 : 1;
    const size_t chunk_bytes_each = chunk_voxels * bytes_per_voxel;
    std::cout << "] (" << (chunk_bytes_each / (1024*1024)) << " MB per chunk)" << std::endl;
    std::cout << "output argument: " << base_output_arg << std::endl;

    // Enforce 90-degree-increment rotations only
    int rotQuadGlobal = -1;
    if (std::abs(rotate_angle) > 1e-6) {
        rotQuadGlobal = normalizeQuadrantRotation(rotate_angle);
        if (rotQuadGlobal < 0) {
            std::cerr << "Error: only 0/90/180/270 degree rotations are supported." << std::endl;
            return EXIT_FAILURE;
        }
        rotate_angle = rotQuadGlobal * 90.0; // normalize
        std::cout << "Rotation: " << rotate_angle << " degrees" << std::endl;
    }
    if (flip_axis >= 0) {
        std::cout << "Flip: " << (flip_axis == 0 ? "Vertical" : flip_axis == 1 ? "Horizontal" : "Both") << std::endl;
    }

    std::filesystem::path output_path(base_output_arg);
    {
        const auto parent = output_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
    }

    const size_t cache_gb = parsed["cache-gb"].as<size_t>();
    const size_t cache_bytes = cache_gb * 1024ull * 1024ull * 1024ull;
    {
        size_t total_ds_chunks = 1;
        for (size_t i = 0; i < ds->shape().size(); ++i)
            total_ds_chunks *= (ds->shape()[i] + ds->chunkShape()[i] - 1) / ds->chunkShape()[i];
        std::cout << "Dataset has " << total_ds_chunks << " disk chunks" << std::endl;
    }
    ChunkCache<uint8_t> chunk_cache_u8(cache_bytes);
    ChunkCache<uint16_t> chunk_cache_u16(cache_bytes);
    chunk_cache_u8.init(ds.get());
    chunk_cache_u16.init(ds.get());

    auto process_one = [&](const std::filesystem::path& seg_folder, const std::string& out_arg, bool force_zarr) -> void {
        std::filesystem::path output_path_local(out_arg);
        if (force_zarr) {
            // ensure .zarr extension
            if (output_path_local.extension() != ".zarr")
                output_path_local = output_path_local.string() + ".zarr";
        }
        bool output_is_zarr = force_zarr || (output_path_local.extension() == ".zarr");
        if (!output_is_zarr) {
            // May be a directory target (no printf pattern): create directory
            if (output_path_local.string().find('%') == std::string::npos) {
                std::filesystem::create_directories(output_path_local);
            } else {
                std::filesystem::create_directories(output_path_local.parent_path());
            }
        }

        std::cout << "Rendering segmentation: "
                  << seg_folder.string() << " -> "
                  << output_path_local.string()
                  << (output_is_zarr?" (zarr)":" (tif)")
                  << std::endl;

        std::unique_ptr<QuadSurface> surf;
        try {
            surf = load_quad_from_tifxyz(seg_folder);
        }
        catch (...) {
            std::cout << "error when loading: " << seg_folder << std::endl;
            return;
        }

        // Apply ABF++ flattening if requested
        if (parsed["flatten"].as<bool>()) {
            std::cout << "Applying ABF++ flattening..." << std::endl;
            vc::ABFConfig flatConfig;
            flatConfig.maxIterations = static_cast<std::size_t>(parsed["flatten-iterations"].as<int>());
            flatConfig.downsampleFactor = parsed["flatten-downsample"].as<int>();
            flatConfig.useABF = true;
            flatConfig.scaleToOriginalArea = true;

            QuadSurface* flatSurf = vc::abfFlattenToNewSurface(*surf, flatConfig);
            if (flatSurf) {
                surf.reset(flatSurf);
                std::cout << "Flattening complete. New grid: "
                          << surf->rawPointsPtr()->cols << " x " << surf->rawPointsPtr()->rows << std::endl;
            } else {
                std::cerr << "Warning: ABF++ flattening failed, using original mesh" << std::endl;
            }
        }

    cv::Mat_<cv::Vec3f> *raw_points = surf->rawPointsPtr();
    for(int j=0;j<raw_points->rows;j++)
        for(int i=0;i<raw_points->cols;i++)
            if ((*raw_points)(j,i)[0] == -1)
                (*raw_points)(j,i) = {NAN,NAN,NAN};
    
    cv::Size full_size = raw_points->size();

    // Interpret --scale as Pg = pixels per level-g voxel.
    // Compute isotropic affine scale sA = cbrt(|det(A)|) (ignore shear/rot)
    // and the effective render scale used by surf->gen() and canvas sizing:
    //   render_scale = Pg / (scale_seg * sA * ds_scale)
    // This keeps pixels locked to level-g voxels while geometry is still
    // mapped to dataset index space by: scale_seg -> affine -> ds_scale.
    double sA = 1.0;
    if (hasAffine) {
        const double detA = cv::determinant(cv::Mat(affineTransform.linearPart()));
        if (std::isfinite(detA) && std::abs(detA) > 1e-18)
            sA = std::cbrt(std::abs(detA));
    }
    const double Pg = static_cast<double>(tgt_scale);
    const double render_scale = Pg * (static_cast<double>(scale_seg) * sA * static_cast<double>(ds_scale));

    // Canvas sizing depends ONLY on render_scale and the saved surface stride.
    {
        const double sx = render_scale / static_cast<double>(surf->_scale[0]);
        const double sy = render_scale / static_cast<double>(surf->_scale[1]);
        full_size.width  = std::max(1, static_cast<int>(std::lround(full_size.width  * sx)));
        full_size.height = std::max(1, static_cast<int>(std::lround(full_size.height * sy)));
    }
    
    // The uncropped, scaled canvas:
    cv::Size tgt_size = full_size;
    // 'crop' is expressed in the *uncropped* canvas coordinate system
    cv::Rect crop = {0, 0, full_size.width, full_size.height};
    
    std::cout << "downsample level " << group_idx
              << " (ds_scale=" << ds_scale << ", sA=" << sA
              << ", Pg=" << Pg << ", render_scale=" << render_scale << ")\n";

    // Handle crop parameters (clamped to canvas)
    const int crop_x = parsed["crop-x"].as<int>();
    const int crop_y = parsed["crop-y"].as<int>();
    const int crop_width  = parsed["crop-width"].as<int>();
    const int crop_height = parsed["crop-height"].as<int>();
    const cv::Rect canvasROI{0, 0, full_size.width, full_size.height};
    if (crop_width > 0 && crop_height > 0) {
        const cv::Rect req{crop_x, crop_y, crop_width, crop_height};
        crop = (req & canvasROI); // intersect with canvas
        if (crop.width <= 0 || crop.height <= 0) {
            std::cerr << "Error: crop rectangle " << req
                      << " lies outside the render canvas " << canvasROI << std::endl;
            return;
        }
        tgt_size = crop.size();
    } else {
        crop = canvasROI;              // no crop requested
        tgt_size = crop.size();
    }
    
    std::cout << "rendering size " << tgt_size
              << " at scale " << tgt_scale
              << " crop " << crop << std::endl;    
              
    cv::Vec3f meshCentroid;

    if (output_is_zarr) {
        const size_t CH = 128, CW = 128;
        const size_t baseZ = std::max(1, num_slices);
        const double baseZ_center = 0.5 * (static_cast<double>(baseZ) - 1.0);
        const size_t CZ = baseZ;
        const int rotQuad = normalizeQuadrantRotation(rotate_angle);
        cv::Size zarr_xy_size = tgt_size;

        if (rotQuad >= 0) {
            if ((rotQuad % 2) == 1) std::swap(zarr_xy_size.width, zarr_xy_size.height);
        }
        const size_t baseY = static_cast<size_t>(zarr_xy_size.height);
        const size_t baseX = static_cast<size_t>(zarr_xy_size.width);

        // Create output zarr directory
        std::filesystem::create_directories(output_path_local);

        std::vector<size_t> shape0 = {baseZ, baseY, baseX};
        std::vector<size_t> chunks0 = {shape0[0], std::min(CH, shape0[1]), std::min(CW, shape0[2])};
        nlohmann::json compOpts0 = {
            {"cname",   "zstd"},
            {"clevel",  1},
            {"shuffle", 0}
        };
        const volcart::zarr::Dtype out_dtype0 = output_is_u16 ? volcart::zarr::Dtype::UInt16 : volcart::zarr::Dtype::UInt8;
        auto dsOut0 = std::make_unique<volcart::zarr::ZarrDataset>(
            output_path_local / "0", shape0, chunks0, out_dtype0, "blosc", compOpts0);

        const size_t tilesY_src = (static_cast<size_t>(tgt_size.height) + CH - 1) / CH;
        const size_t tilesX_src = (static_cast<size_t>(tgt_size.width)  + CW - 1) / CW;
        const size_t totalTiles = tilesY_src * tilesX_src;
        std::atomic<size_t> tilesDone{0};

        renderTiledLoop(surf.get(), tgt_size, full_size, crop,
            static_cast<float>(render_scale), scale_seg, ds_scale,
            hasAffine, affineTransform, meshCentroid,
            tilesDone, totalTiles, "render L0",
            static_cast<uint32_t>(CW), static_cast<uint32_t>(CH),
            [&](int tx, int ty, int dx, int dy,
                const cv::Mat_<cv::Vec3f>& basePoints, const cv::Mat_<cv::Vec3f>& stepDirs)
            {
                const bool swapWH = (rotQuad >= 0) && ((rotQuad % 2) == 1);
                const size_t dy_dst = swapWH ? dx : dy;
                const size_t dx_dst = swapWH ? dy : dx;

                int dstTx = tx, dstTy = ty;
                int dstTilesX = static_cast<int>(tilesX_src), dstTilesY = static_cast<int>(tilesY_src);
                if (rotQuad >= 0 || flip_axis >= 0) {
                    mapTileIndex(tx, ty,
                                 static_cast<int>(tilesX_src), static_cast<int>(tilesY_src),
                                 std::max(rotQuad, 0), flip_axis,
                                 dstTx, dstTy, dstTilesX, dstTilesY);
                }
                const size_t x0_dst = static_cast<size_t>(dstTx) * CW;
                const size_t y0_dst = static_cast<size_t>(dstTy) * CH;
                std::vector<size_t> outOffset = {0, y0_dst, x0_dst};

                if (output_is_u16) {
                    volcart::zarr::Tensor3D<uint16_t> outChunk(baseZ, dy_dst, dx_dst);
                    renderSlicesToZarrChunk<uint16_t>(outChunk, ds.get(), &chunk_cache_u16,
                        basePoints, stepDirs, static_cast<float>(ds_scale), nearest_neighbor,
                        baseZ, 0, baseZ_center, slice_step, accumOffsets, accumType, rotQuad, flip_axis);
                    dsOut0->writeSubarray(outChunk, outOffset);
                } else {
                    volcart::zarr::Tensor3D<uint8_t> outChunk(baseZ, dy_dst, dx_dst);
                    renderSlicesToZarrChunk<uint8_t>(outChunk, ds.get(), &chunk_cache_u8,
                        basePoints, stepDirs, static_cast<float>(ds_scale), nearest_neighbor,
                        baseZ, 0, baseZ_center, slice_step, accumOffsets, accumType, rotQuad, flip_axis);
                    dsOut0->writeSubarray(outChunk, outOffset);
                }
            });

        if (output_is_u16)
            chunk_cache_u16.printStats("after L0 render");
        else
            chunk_cache_u8.printStats("after L0 render");

        // Build multi-resolution pyramid levels 1..5 by averaging 2x blocks in Z, Y, and X
        // Keep track of datasets for each level
        std::vector<std::unique_ptr<volcart::zarr::ZarrDataset>> pyramidLevels;
        pyramidLevels.push_back(std::move(dsOut0));  // level 0

        auto downsample_avg = [&](int targetLevel){
            auto& src = pyramidLevels[targetLevel - 1];
            const auto& sShape = src->shape();
            // Downsample Z, Y and X by 2 (handle odd sizes)
            std::vector<size_t> dShape = {
                (sShape[0] + 1) / 2,
                (sShape[1] + 1) / 2,
                (sShape[2] + 1) / 2
            };
            // Chunk Z equals number of slices at this level (full Z), XY = 128
            std::vector<size_t> dChunks = {dShape[0], std::min(CH, dShape[1]), std::min(CW, dShape[2])};
            nlohmann::json compOpts = {
                {"cname",   "zstd"},
                {"clevel",  1},
                {"shuffle", 0}
            };
            auto dst = std::make_unique<volcart::zarr::ZarrDataset>(
                output_path_local / std::to_string(targetLevel),
                dShape, dChunks, out_dtype0, "blosc", compOpts);

            const size_t tileZ = dShape[0], tileY = CH, tileX = CW;
            const size_t tilesY = (dShape[1] + tileY - 1) / tileY;
            const size_t tilesX = (dShape[2] + tileX - 1) / tileX;
            const size_t totalTiles = tilesY * tilesX;
            std::atomic<size_t> tilesDone{0};
            #pragma omp parallel for schedule(dynamic)
            for (size_t z = 0; z < dShape[0]; z += tileZ) {
                const size_t lz = std::min(tileZ, dShape[0] - z);

                for (long long y = 0; y < static_cast<long long>(dShape[1]); y += tileY) {
                    for (long long x = 0; x < static_cast<long long>(dShape[2]); x += tileX) {
                        const size_t ly = std::min(tileY, static_cast<size_t>(dShape[1] - y));
                        const size_t lx = std::min(tileX, static_cast<size_t>(dShape[2] - x));

                        const size_t sz = std::min<size_t>(2*lz, sShape[0] - 2*z);
                        const size_t sy = std::min<size_t>(2*ly, sShape[1] - y*2);
                        const size_t sx = std::min<size_t>(2*lx, sShape[2] - x*2);

                        if (output_is_u16) {
                            downsampleChunk<uint16_t>(src.get(), dst.get(), z, y, x, lz, ly, lx, sz, sy, sx);
                        } else {
                            downsampleChunk<uint8_t>(src.get(), dst.get(), z, y, x, lz, ly, lx, sz, sy, sx);
                        }

                        const std::string progLabel = "render L" + std::to_string(targetLevel);
                        printTileProgress(tilesDone, totalTiles, progLabel.c_str());
                    }
                }
            }
            std::cout << std::endl;
            pyramidLevels.push_back(std::move(dst));
        };

        for (int level = 1; level <= 5; ++level) {
            downsample_avg(level);
        }

        nlohmann::json attrs;
        attrs["source_zarr"] = vol_path.string();
        attrs["source_group"] = group_idx;
        attrs["num_slices"] = baseZ;
        attrs["slice_step"] = slice_step;
        if (!accumOffsets.empty()) {
            attrs["accum_step"] = accum_step;
            attrs["accum_type"] = accum_type_str;
            attrs["accum_samples"] = static_cast<int>(accumOffsets.size());
        }
        {
            cv::Size attr_xy = tgt_size;
            const int rotQuadAttr = normalizeQuadrantRotation(rotate_angle);
            if (rotQuadAttr >= 0 && (rotQuadAttr % 2) == 1) std::swap(attr_xy.width, attr_xy.height);
            attrs["canvas_size"] = {attr_xy.width, attr_xy.height};
        }
        attrs["chunk_size"] = {static_cast<int>(CZ), static_cast<int>(CH), static_cast<int>(CW)};
        attrs["note_axes_order"] = "ZYX (slice, row, col)";

        nlohmann::json multiscale;
        multiscale["version"] = "0.4";
        multiscale["name"] = "render";
        multiscale["axes"] = nlohmann::json::array({
            nlohmann::json{{"name","z"},{"type","space"}},
            nlohmann::json{{"name","y"},{"type","space"}},
            nlohmann::json{{"name","x"},{"type","space"}}
        });
        multiscale["datasets"] = nlohmann::json::array();
        for (int level = 0; level <= 5; ++level) {
            const double s = std::pow(2.0, level);
            nlohmann::json dset;
            dset["path"] = std::to_string(level);
            dset["coordinateTransformations"] = nlohmann::json::array({
                // Z, Y and X scale by 2^level
                nlohmann::json{{"type","scale"},{"scale", nlohmann::json::array({s, s, s})}},
                nlohmann::json{{"type","translation"},{"translation", nlohmann::json::array({0.0, 0.0, 0.0})}}
            });
            multiscale["datasets"].push_back(dset);
        }
        multiscale["metadata"] = nlohmann::json{{"downsampling_method","mean"}};
        attrs["multiscales"] = nlohmann::json::array({multiscale});

        // Write attributes to .zattrs file
        {
            std::ofstream attrsFile(output_path_local / ".zattrs");
            attrsFile << attrs.dump(2);
        }

        // Optionally export per-Z TIFFs from level 0 into layers_{zarrname}
        if (include_tifs) {
            try {
                auto& dsL0 = pyramidLevels[0];
                const auto& shape0_check = dsL0->shape(); // [Z, Y, X]
                const size_t Z = shape0_check[0];
                const int Y = static_cast<int>(shape0_check[1]);
                const int X = static_cast<int>(shape0_check[2]);

                std::string zname = output_path_local.stem().string();
                std::filesystem::path layers_dir = output_path_local.parent_path() / (std::string("layers_") + zname);
                std::filesystem::create_directories(layers_dir);

                int pad = 2;
                size_t maxIndex = (Z > 0) ? (Z - 1) : 0;
                while (maxIndex >= 100) { pad++; maxIndex /= 10; }

                bool all_exist = true;
                for (size_t z = 0; z < Z; ++z) {
                    std::ostringstream fn;
                    fn << std::setw(pad) << std::setfill('0') << z;
                    std::filesystem::path outPath = layers_dir / (fn.str() + std::string(".tif"));
                    if (!std::filesystem::exists(outPath)) { all_exist = false; break; }
                }
                if (all_exist) {
                    std::cout << "[tif export] all slices exist in " << layers_dir.string() << ", skipping." << std::endl;
                    return;
                }

                const uint32_t tileW = static_cast<uint32_t>(CW);
                const uint32_t tileH = static_cast<uint32_t>(CH);
                const uint32_t tilesX_src = (static_cast<uint32_t>(X) + tileW - 1) / tileW;
                const uint32_t tilesY_src = (static_cast<uint32_t>(Y) + tileH - 1) / tileH;
                // Zarr L0 already has rotation/flip applied; TIFFs should match L0 exactly
                const uint32_t outW = static_cast<uint32_t>(X);
                const uint32_t outH = static_cast<uint32_t>(Y);
                const size_t totalTiles = static_cast<size_t>(tilesX_src) * static_cast<size_t>(tilesY_src);
                std::atomic<size_t> tilesDone{0};

                std::vector<TiffWriter> writers;
                std::vector<std::mutex> writerLocks(Z);
                writers.reserve(Z);
                const int cvType = output_is_u16 ? CV_16UC1 : CV_8UC1;
                for (size_t z = 0; z < Z; ++z) {
                    std::ostringstream fn;
                    fn << std::setw(pad) << std::setfill('0') << z;
                    std::filesystem::path outPath = layers_dir / (fn.str() + std::string(".tif"));
                    writers.emplace_back(outPath, outW, outH, cvType, tileW, tileH, 0.0f);
                }

                #pragma omp parallel for schedule(dynamic)
                for (long long ty = 0; ty < static_cast<long long>(tilesY_src); ++ty) {
                    for (long long tx = 0; tx < static_cast<long long>(tilesX_src); ++tx) {
                        const uint32_t x0_src = static_cast<uint32_t>(tx) * tileW;
                        const uint32_t y0_src = static_cast<uint32_t>(ty) * tileH;
                        const uint32_t dx = std::min<uint32_t>(tileW, static_cast<uint32_t>(X) - x0_src);
                        const uint32_t dy = std::min<uint32_t>(tileH, static_cast<uint32_t>(Y) - y0_src);
                        const uint32_t x0_dst = static_cast<uint32_t>(tx) * tileW;
                        const uint32_t y0_dst = static_cast<uint32_t>(ty) * tileH;

                        if (output_is_u16) {
                            exportZarrTileToTiff<uint16_t>(dsL0.get(), writers, writerLocks, Z, dy, dx, y0_src, x0_src, x0_dst, y0_dst);
                        } else {
                            exportZarrTileToTiff<uint8_t>(dsL0.get(), writers, writerLocks, Z, dy, dx, y0_src, x0_src, x0_dst, y0_dst);
                        }

                        printTileProgress(tilesDone, totalTiles, "tif export");
                    }
                }

                writers.clear(); // Explicitly close all writers

                std::cout << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[tif export] warning: failed to export TIFFs: " << e.what() << std::endl;
            }
        }

        return;
    }

    {
        try {
            const int rotQuad = normalizeQuadrantRotation(rotate_angle);
            if (std::abs(rotate_angle) > 1e-6 && rotQuad < 0) {
                throw std::runtime_error("non-right-angle rotation not supported in tiled-TIFF path");
                }

                const int outW = ((rotQuad >= 0) && (rotQuad % 2 == 1)) ? tgt_size.height : tgt_size.width;
                const int outH = ((rotQuad >= 0) && (rotQuad % 2 == 1)) ? tgt_size.width  : tgt_size.height;

                const uint32_t tileW = 128;
                const uint32_t tileH = 128;
                const double num_slices_center = 0.5 * (static_cast<double>(std::max(1, num_slices)) - 1.0);

                auto make_out_path = [&](int sliceIdx) -> std::filesystem::path {
                    if (output_path_local.string().find('%') == std::string::npos) {
                        int pad = 2; int v = std::max(0, num_slices-1);
                        while (v >= 100) { pad++; v /= 10; }
                        std::ostringstream fn;
                        fn << std::setw(pad) << std::setfill('0') << sliceIdx;
                        return output_path_local / (fn.str() + ".tif");
                    } else {
                        char buf[1024];
                        snprintf(buf, sizeof(buf), output_path_local.string().c_str(), sliceIdx);
                        return std::filesystem::path(buf);
                    }
                };

                // If all expected TIFFs exist, skip this segmentation
                {
                    bool all_exist = true;
                    for (int z = 0; z < num_slices; ++z) {
                        std::filesystem::path outPath = make_out_path(z);
                        if (!std::filesystem::exists(outPath)) { all_exist = false; break; }
                    }
                    if (all_exist) {
                        std::cout << "[tif tiled] all slices exist in " << output_path_local.string() << ", skipping." << std::endl;
                        return;
                    }
                }

                std::vector<TiffWriter> writers;
                std::vector<std::mutex> writerLocks(static_cast<size_t>(num_slices));
                writers.reserve(static_cast<size_t>(num_slices));
                const int cvType = output_is_u16 ? CV_16UC1 : CV_8UC1;
                for (int z = 0; z < num_slices; ++z) {
                    std::filesystem::path outPath = make_out_path(z);
                    writers.emplace_back(outPath, static_cast<uint32_t>(outW), static_cast<uint32_t>(outH),
                                         cvType, tileW, tileH, 0.0f);
                }

                const uint32_t tilesX_src = (static_cast<uint32_t>(tgt_size.width)  + tileW - 1) / tileW;
                const uint32_t tilesY_src = (static_cast<uint32_t>(tgt_size.height) + tileH - 1) / tileH;
                const size_t totalTiles = static_cast<size_t>(tilesX_src) * static_cast<size_t>(tilesY_src);
                std::atomic<size_t> tilesDone{0};

                renderTiledLoop(surf.get(), tgt_size, full_size, crop,
                    static_cast<float>(render_scale), scale_seg, ds_scale,
                    hasAffine, affineTransform, meshCentroid,
                    tilesDone, totalTiles, "tif tiled",
                    tileW, tileH,
                    [&](int tx, int ty, int dx, int dy,
                        const cv::Mat_<cv::Vec3f>& basePoints, const cv::Mat_<cv::Vec3f>& stepDirs)
                    {
                        if (output_is_u16) {
                            renderSlicesToTiffTiles<uint16_t>(
                                ds.get(), &chunk_cache_u16, basePoints, stepDirs,
                                static_cast<float>(ds_scale), nearest_neighbor,
                                num_slices, num_slices_center, slice_step,
                                accumOffsets, accumType, rotQuad, flip_axis,
                                tx, ty,
                                static_cast<int>(tilesX_src), static_cast<int>(tilesY_src),
                                tileW, tileH, writers, writerLocks);
                        } else {
                            renderSlicesToTiffTiles<uint8_t>(
                                ds.get(), &chunk_cache_u8, basePoints, stepDirs,
                                static_cast<float>(ds_scale), nearest_neighbor,
                                num_slices, num_slices_center, slice_step,
                                accumOffsets, accumType, rotQuad, flip_axis,
                                tx, ty,
                                static_cast<int>(tilesX_src), static_cast<int>(tilesY_src),
                                tileW, tileH, writers, writerLocks);
                        }
                    });

                if (output_is_u16)
                    chunk_cache_u16.printStats("after tif render");
                else
                    chunk_cache_u8.printStats("after tif render");

                writers.clear(); // Explicitly close all writers

                return;
        } catch (const std::exception& e) {
            std::cerr << "[tif tiled] error: " << e.what() << std::endl;
            return;
        }
    }
    };


    if (has_render_folder) {
        // iterate through folders in render_folder_path
        for (const auto& entry : std::filesystem::directory_iterator(render_folder_path)) {
            if (!entry.is_directory()) continue;

            const std::string seg_name = entry.path().filename().string();
            const std::filesystem::path base(base_output_arg);

            std::filesystem::path out_arg_path;

            if (batch_format == "zarr") {
                // Treat -o as a DIRECTORY in batch-zarr mode so the wrapper can
                // find layers_* under that directory.
                // Write:  <-o>/<prefix>_<seg-name>.zarr
                // And TIFFs: <-o>/layers_<prefix>_<seg-name>
                std::filesystem::path base_dir;
                if ((std::filesystem::exists(base) && std::filesystem::is_directory(base)) || !base.has_extension()) {
                    base_dir = base;
                    std::error_code ec;
                    std::filesystem::create_directories(base_dir, ec); // best-effort
                } else {
                    // If -o looks like a file, fall back to its parent
                    base_dir = base.parent_path();
                    if (base_dir.empty()) base_dir = std::filesystem::current_path();
                }
                // Keep prior naming flavor by using the stem (or filename if empty) as prefix
                const std::string prefix = base.stem().string().empty()
                                             ? base.filename().string()
                                             : base.stem().string();
                out_arg_path = base_dir / (prefix + "_" + seg_name);
                std::cout << "[batch] writing Zarr under directory: "
                          << base_dir.string()
                          << "  name=" << (prefix + "_" + seg_name) << " (.zarr appended)\n";
            } else {
                // For TIFF, keep old behavior but handle absolute -o correctly
                out_arg_path = base.is_absolute() ? (base / seg_name)
                                                : (entry.path() / base);
            }

            process_one(entry.path(), out_arg_path.string(), batch_format == "zarr");
        }
    } else {
        process_one(seg_path, base_output_arg, false);
    }

    {
        auto wall_t1 = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(wall_t1 - wall_t0).count();
        int h = static_cast<int>(elapsed / 3600);
        int m = static_cast<int>((elapsed % 3600) / 60);
        int s = static_cast<int>(elapsed % 60);
        std::cout << "Total time: "
                  << std::setw(2) << std::setfill('0') << h << ":"
                  << std::setw(2) << std::setfill('0') << m << ":"
                  << std::setw(2) << std::setfill('0') << s << std::endl;
    }
    return EXIT_SUCCESS;
}
