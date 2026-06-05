#include "vc/core/render/ChunkedPlaneSampler.hpp"

#include <utils/thread_pool.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <future>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vc::render {
namespace {

struct LocalChunkCache {
    explicit LocalChunkCache(IChunkedArray& a, std::size_t expectedChunks = 0)
        : array(a)
    {
        if (expectedChunks > 0) {
            chunks.reserve(expectedChunks);
            requestedKeys.reserve(expectedChunks);
            errorKeys.reserve(expectedChunks);
        }
    }

    const ChunkResult& get(const ChunkKey& key, int& requested, int& errors)
    {
        // Trilinear sampling reads 8 voxels per pixel and adjacent pixels share
        // chunks, so consecutive lookups overwhelmingly hit the same key. Skip
        // the hash-map probe in that case.
        if (lastResult && lastKey == key)
            return *lastResult;

        auto it = chunks.find(key);
        if (it == chunks.end()) {
            ChunkResult result = array.tryGetChunk(key.level, key.iz, key.iy, key.ix);
            if (result.status == ChunkStatus::MissQueued && requestedKeys.insert(key).second)
                ++requested;
            if (result.status == ChunkStatus::Error && errorKeys.insert(key).second)
                ++errors;
            it = chunks.emplace(key, std::move(result)).first;
        }

        lastKey = key;
        lastResult = &it->second;
        return it->second;
    }

    IChunkedArray& array;
    std::unordered_map<ChunkKey, ChunkResult, ChunkKeyHash> chunks;
    std::unordered_set<ChunkKey, ChunkKeyHash> requestedKeys;
    std::unordered_set<ChunkKey, ChunkKeyHash> errorKeys;
    ChunkKey lastKey{};
    const ChunkResult* lastResult = nullptr;
};

constexpr int kParallelMinPixels = 128 * 128;
constexpr int kMaxRenderSamplerWorkers = 8;

int renderSamplerWorkerCount()
{
    const unsigned hc = std::thread::hardware_concurrency();
    if (hc <= 2)
        return 1;
    return std::clamp(static_cast<int>(hc) - 2, 1, kMaxRenderSamplerWorkers);
}

utils::ThreadPool& renderSamplerPool()
{
    static utils::ThreadPool pool(static_cast<std::size_t>(renderSamplerWorkerCount()));
    return pool;
}

bool shouldParallelizeSamples(int rows, int cols)
{
    return renderSamplerWorkerCount() > 1 &&
           rows > 0 && cols > 0 &&
           rows * cols >= kParallelMinPixels;
}

struct LevelAccess {
    std::array<int, 3> shape{};
    std::array<int, 3> chunkShape{};
    IChunkedArray::LevelTransform transform;
    uint8_t fill = 0;
};

struct LevelPlane {
    cv::Vec3f origin;
    cv::Vec3f vxStep;
    cv::Vec3f vyStep;
};

struct SampleTile {
    int tx = 0;
    int ty = 0;
    int xEnd = 0;
    int yEnd = 0;
};

bool finiteCoord(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool surfaceSentinel(const cv::Vec3f& p)
{
    // Hot path: called per screen pixel in the render sampler. gen() marks holes
    // as (-1,-1,-1); (0,0,0) is the unset marker; NaN comes from bad source data.
    // Coordinates never reach +/-Inf in the surface pipeline (see QuadSurface),
    // so we detect NaN with the branchless self-inequality test (x != x) instead
    // of std::isfinite (a libm classification call) -- the same trick the mask
    // builder uses. Cheap == checks first so valid pixels exit fast.
    const float a = p[0], b = p[1], c = p[2];
    if (a == -1.0f || b == -1.0f || c == -1.0f)
        return true;
    if (a == 0.0f && b == 0.0f && c == 0.0f)
        return true;
    return a != a || b != b || c != c;   // NaN
}

LevelAccess makeLevelAccess(IChunkedArray& array, int level)
{
    LevelAccess access;
    access.shape = array.shape(level);
    access.chunkShape = array.chunkShape(level);
    access.transform = array.levelTransform(level);
    access.fill = static_cast<uint8_t>(std::clamp(std::lround(array.fillValue()), 0L, 255L));
    return access;
}

bool hasSampleableLevel(const LevelAccess& access)
{
    return access.shape[0] > 0 && access.shape[1] > 0 && access.shape[2] > 0
        && access.chunkShape[0] > 0 && access.chunkShape[1] > 0 && access.chunkShape[2] > 0;
}

cv::Vec3f toLevelCoord(const LevelAccess& access, const cv::Vec3f& p0)
{
    // Hot path: called per screen pixel. The level transform is a uniform scale
    // with (almost always) zero offset (offsetFromLevel0 = {0,0,0} for the vca/
    // zarr fetchers), so the common case is a single float multiply per axis --
    // skip the double round-trip and the +0 adds. Coordinates stay well within
    // float's exact-integer range (volumes are < 2^24 voxels), so float scaling
    // is precise enough here.
    const auto& t = access.transform;
    const float sx = float(t.scaleFromLevel0[0]);
    const float sy = float(t.scaleFromLevel0[1]);
    const float sz = float(t.scaleFromLevel0[2]);
    if (t.offsetFromLevel0[0] == 0.0 && t.offsetFromLevel0[1] == 0.0 &&
        t.offsetFromLevel0[2] == 0.0) {
        return {p0[0] * sx, p0[1] * sy, p0[2] * sz};
    }
    return {
        float(double(p0[0]) * t.scaleFromLevel0[0] + t.offsetFromLevel0[0]),
        float(double(p0[1]) * t.scaleFromLevel0[1] + t.offsetFromLevel0[1]),
        float(double(p0[2]) * t.scaleFromLevel0[2] + t.offsetFromLevel0[2]),
    };
}

cv::Vec3f toLevelVector(const LevelAccess& access, const cv::Vec3f& v0)
{
    const auto& t = access.transform;
    return {
        float(double(v0[0]) * t.scaleFromLevel0[0]),
        float(double(v0[1]) * t.scaleFromLevel0[1]),
        float(double(v0[2]) * t.scaleFromLevel0[2]),
    };
}

LevelPlane toLevelPlane(const LevelAccess& access,
                        const cv::Vec3f& origin,
                        const cv::Vec3f& vxStep,
                        const cv::Vec3f& vyStep)
{
    return {toLevelCoord(access, origin),
            toLevelVector(access, vxStep),
            toLevelVector(access, vyStep)};
}

bool inLevelBounds(const std::array<int, 3>& shape, float z, float y, float x)
{
    return z >= 0.0f && y >= 0.0f && x >= 0.0f
        && z < float(shape[0]) && y < float(shape[1]) && x < float(shape[2]);
}

bool readVoxel(IChunkedArray& array,
               LocalChunkCache& cache,
               const LevelAccess& access,
               int level,
               int iz,
               int iy,
               int ix,
               uint8_t& out,
               int& requested,
               int& errors)
{
    const auto& shape = access.shape;
    if (unsigned(iz) >= unsigned(shape[0])
        || unsigned(iy) >= unsigned(shape[1])
        || unsigned(ix) >= unsigned(shape[2])) {
        out = access.fill;
        return true;
    }

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] <= 0 || chunkShape[1] <= 0 || chunkShape[2] <= 0)
        return false;

    const int cz = iz / chunkShape[0];
    const int cy = iy / chunkShape[1];
    const int cx = ix / chunkShape[2];
    const ChunkResult& result = cache.get({level, cz, cy, cx}, requested, errors);
    if (result.status == ChunkStatus::MissQueued ||
        result.status == ChunkStatus::Missing ||
        result.status == ChunkStatus::Error)
        return false;

    if (result.status == ChunkStatus::AllFill) {
        out = access.fill;
        return true;
    }

    if (result.status != ChunkStatus::Data || !result.bytes)
        return false;

    const int lz = iz - cz * chunkShape[0];
    const int ly = iy - cy * chunkShape[1];
    const int lx = ix - cx * chunkShape[2];
    const std::size_t offset = (std::size_t(lz) * std::size_t(chunkShape[1])
                              + std::size_t(ly)) * std::size_t(chunkShape[2])
                              + std::size_t(lx);
    if (offset >= result.bytes->size())
        return false;

    out = std::to_integer<uint8_t>((*result.bytes)[offset]);
    return true;
}

bool sampleNearest(IChunkedArray& array,
                   LocalChunkCache& cache,
                   const LevelAccess& access,
                   int level,
                   const cv::Vec3f& p,
                   uint8_t& out,
                   int& requested,
                   int& errors)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return true;
    }

    int ix = int(x + 0.5f);
    int iy = int(y + 0.5f);
    int iz = int(z + 0.5f);
    ix = std::clamp(ix, 0, shape[2] - 1);
    iy = std::clamp(iy, 0, shape[1] - 1);
    iz = std::clamp(iz, 0, shape[0] - 1);
    return readVoxel(array, cache, access, level, iz, iy, ix, out, requested, errors);
}

bool sampleTrilinear(IChunkedArray& array,
                     LocalChunkCache& cache,
                     const LevelAccess& access,
                     int level,
                     const cv::Vec3f& p,
                     uint8_t& out,
                     int& requested,
                     int& errors)
{
    const auto& shape = access.shape;
    const float x = p[0], y = p[1], z = p[2];
    if (!inLevelBounds(shape, z, y, x)) {
        out = access.fill;
        return true;
    }

    const int ix = int(x);
    const int iy = int(y);
    const int iz = int(z);
    const float fx = x - float(ix);
    const float fy = y - float(iy);
    const float fz = z - float(iz);

    const auto& chunkShape = access.chunkShape;
    if (chunkShape[0] > 0 && chunkShape[1] > 0 && chunkShape[2] > 0 &&
        ix + 1 < shape[2] && iy + 1 < shape[1] && iz + 1 < shape[0]) {
        const int cz = iz / chunkShape[0];
        const int cy = iy / chunkShape[1];
        const int cx = ix / chunkShape[2];
        const int lz = iz - cz * chunkShape[0];
        const int ly = iy - cy * chunkShape[1];
        const int lx = ix - cx * chunkShape[2];
        if (lx + 1 < chunkShape[2] && ly + 1 < chunkShape[1] && lz + 1 < chunkShape[0]) {
            const ChunkResult& result = cache.get({level, cz, cy, cx}, requested, errors);
            if (result.status == ChunkStatus::MissQueued ||
                result.status == ChunkStatus::Missing ||
                result.status == ChunkStatus::Error)
                return false;

            if (result.status == ChunkStatus::AllFill) {
                out = access.fill;
                return true;
            }

            if (result.status == ChunkStatus::Data && result.bytes) {
                const std::size_t strideX = 1;
                const std::size_t strideY = std::size_t(chunkShape[2]);
                const std::size_t strideZ = std::size_t(chunkShape[1]) * std::size_t(chunkShape[2]);
                const std::size_t offset000 = std::size_t(lz) * strideZ
                                            + std::size_t(ly) * strideY
                                            + std::size_t(lx);
                const std::size_t offset111 = offset000 + strideZ + strideY + strideX;
                if (offset111 < result.bytes->size()) {
                    const auto* bytes = result.bytes->data();
                    const uint8_t v000 = std::to_integer<uint8_t>(bytes[offset000]);
                    const uint8_t v001 = std::to_integer<uint8_t>(bytes[offset000 + strideX]);
                    const uint8_t v010 = std::to_integer<uint8_t>(bytes[offset000 + strideY]);
                    const uint8_t v011 = std::to_integer<uint8_t>(bytes[offset000 + strideY + strideX]);
                    const uint8_t v100 = std::to_integer<uint8_t>(bytes[offset000 + strideZ]);
                    const uint8_t v101 = std::to_integer<uint8_t>(bytes[offset000 + strideZ + strideX]);
                    const uint8_t v110 = std::to_integer<uint8_t>(bytes[offset000 + strideZ + strideY]);
                    const uint8_t v111 = std::to_integer<uint8_t>(bytes[offset111]);

                    const float c00 = std::fma(fx, float(v001) - float(v000), float(v000));
                    const float c01 = std::fma(fx, float(v011) - float(v010), float(v010));
                    const float c10 = std::fma(fx, float(v101) - float(v100), float(v100));
                    const float c11 = std::fma(fx, float(v111) - float(v110), float(v110));
                    const float c0 = std::fma(fy, c01 - c00, c00);
                    const float c1 = std::fma(fy, c11 - c10, c10);
                    const float value = std::clamp(std::fma(fz, c1 - c0, c0), 0.0f, 255.0f);
                    out = static_cast<uint8_t>(value);
                    return true;
                }
            }
        }
    }

    uint8_t v000 = 0, v001 = 0, v010 = 0, v011 = 0;
    uint8_t v100 = 0, v101 = 0, v110 = 0, v111 = 0;
    bool ready = true;
    ready = readVoxel(array, cache, access, level, iz,     iy,     ix,     v000, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz,     iy,     ix + 1, v001, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz,     iy + 1, ix,     v010, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz,     iy + 1, ix + 1, v011, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy,     ix,     v100, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy,     ix + 1, v101, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy + 1, ix,     v110, requested, errors) && ready;
    ready = readVoxel(array, cache, access, level, iz + 1, iy + 1, ix + 1, v111, requested, errors) && ready;
    if (!ready)
        return false;

    const float c00 = std::fma(fx, float(v001) - float(v000), float(v000));
    const float c01 = std::fma(fx, float(v011) - float(v010), float(v010));
    const float c10 = std::fma(fx, float(v101) - float(v100), float(v100));
    const float c11 = std::fma(fx, float(v111) - float(v110), float(v110));
    const float c0 = std::fma(fy, c01 - c00, c00);
    const float c1 = std::fma(fy, c11 - c10, c10);
    const float value = std::clamp(std::fma(fz, c1 - c0, c0), 0.0f, 255.0f);
    out = static_cast<uint8_t>(value);
    return true;
}

bool samplePoint(IChunkedArray& array,
                 LocalChunkCache& cache,
                 const LevelAccess& access,
                 int level,
                 const cv::Vec3f& p0,
                 vc::Sampling sampling,
                 bool zeroIsSentinel,
                 uint8_t& out,
                 int& requested,
                 int& errors)
{
    // surfaceSentinel() already subsumes the NaN/Inf check (it tests x != x and
    // the pipeline never produces +/-Inf), so when zeroIsSentinel we don't also
    // run finiteCoord -- that was 3 redundant std::isfinite calls per pixel on
    // the hot render path. Without the sentinel convention we still need the
    // explicit finite test.
    if (zeroIsSentinel) {
        if (surfaceSentinel(p0))
            return false;
    } else if (!finiteCoord(p0)) {
        out = access.fill;
        return true;
    }

    const cv::Vec3f p = toLevelCoord(access, p0);
    if (sampling == vc::Sampling::Nearest)
        return sampleNearest(array, cache, access, level, p, out, requested, errors);

    return sampleTrilinear(array, cache, access, level, p, out, requested, errors);
}

bool sampleLevelPoint(IChunkedArray& array,
                      LocalChunkCache& cache,
                      const LevelAccess& access,
                      int level,
                      const cv::Vec3f& p,
                      vc::Sampling sampling,
                      uint8_t& out,
                      int& requested,
                      int& errors)
{
    // Non-finite coords fail inLevelBounds (NaN compares false) and return
    // fill, identical to an explicit finiteCoord check.
    if (sampling == vc::Sampling::Nearest)
        return sampleNearest(array, cache, access, level, p, out, requested, errors);

    return sampleTrilinear(array, cache, access, level, p, out, requested, errors);
}

void addStats(ChunkedPlaneSampler::Stats& dst, const ChunkedPlaneSampler::Stats& src)
{
    dst.coveredPixels += src.coveredPixels;
    dst.requestedChunks += src.requestedChunks;
    dst.errorChunks += src.errorChunks;
}

} // namespace

ChunkedPlaneSampler::Stats samplePlaneLevelImpl(
    IChunkedArray& array,
    int level,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options,
    bool overwriteCovered)
{
    ChunkedPlaneSampler::Stats stats;
    if (level < 0 || level >= array.numLevels() || out.empty() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;
    const LevelPlane levelPlane = toLevelPlane(access, origin, vxStep, vyStep);
    const int tile = std::max(1, options.tileSize);
    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((out.rows + tile - 1) / tile) *
                  std::size_t((out.cols + tile - 1) / tile));
    for (int ty = 0; ty < out.rows; ty += tile) {
        const int yEnd = std::min(ty + tile, out.rows);
        for (int tx = 0; tx < out.cols; tx += tile) {
            const int xEnd = std::min(tx + tile, out.cols);
            tiles.push_back({tx, ty, xEnd, yEnd});
        }
    }

    auto processTileRange = [&](std::size_t begin, std::size_t end) {
        ChunkedPlaneSampler::Stats localStats;
        LocalChunkCache chunkCache(array, std::max<std::size_t>(16, (end - begin) * 4));
        for (std::size_t i = begin; i < end; ++i) {
            const SampleTile& sampleTile = tiles[i];
            for (int y = sampleTile.ty; y < sampleTile.yEnd; ++y) {
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                const cv::Vec3f rowBase = levelPlane.origin + levelPlane.vyStep * float(y);
                for (int x = sampleTile.tx; x < sampleTile.xEnd; ++x) {
                    if (!overwriteCovered && coverageRow[x])
                        continue;

                    uint8_t value = 0;
                    if (sampleLevelPoint(array, chunkCache, access, level,
                                         rowBase + levelPlane.vxStep * float(x),
                                         options.sampling, value,
                                         localStats.requestedChunks, localStats.errorChunks)) {
                        const bool wasCovered = coverageRow[x] != 0;
                        outRow[x] = value;
                        coverageRow[x] = 1;
                        if (!wasCovered)
                            ++localStats.coveredPixels;
                    }
                }
            }
        }
        return localStats;
    };

    if (!shouldParallelizeSamples(out.rows, out.cols) || tiles.size() <= 1)
        return processTileRange(0, tiles.size());

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end)
            break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] {
            return processTileRange(begin, end);
        }));
    }
    for (auto& future : futures) {
        addStats(stats, future.get());
    }
    return stats;
}

ChunkedPlaneSampler::Stats sampleCoordsLevelImpl(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options,
    bool overwriteCovered)
{
    ChunkedPlaneSampler::Stats stats;
    if (level < 0 || level >= array.numLevels() || coords.empty() || out.empty() || coverage.empty())
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;
    const int tile = std::max(1, options.tileSize);
    const int h = std::min({coords.rows, out.rows, coverage.rows});
    const int w = std::min({coords.cols, out.cols, coverage.cols});
    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((h + tile - 1) / tile) *
                  std::size_t((w + tile - 1) / tile));
    for (int ty = 0; ty < h; ty += tile) {
        const int yEnd = std::min(ty + tile, h);
        for (int tx = 0; tx < w; tx += tile) {
            const int xEnd = std::min(tx + tile, w);
            tiles.push_back({tx, ty, xEnd, yEnd});
        }
    }

    auto processTileRange = [&](std::size_t begin, std::size_t end) {
        ChunkedPlaneSampler::Stats localStats;
        LocalChunkCache chunkCache(array, std::max<std::size_t>(16, (end - begin) * 4));
        for (std::size_t i = begin; i < end; ++i) {
            const SampleTile& sampleTile = tiles[i];
            for (int y = sampleTile.ty; y < sampleTile.yEnd; ++y) {
                const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = sampleTile.tx; x < sampleTile.xEnd; ++x) {
                    if (!overwriteCovered && coverageRow[x])
                        continue;

                    uint8_t value = 0;
                    if (samplePoint(array, chunkCache, access, level, coordRow[x], options.sampling,
                                    true, value, localStats.requestedChunks, localStats.errorChunks)) {
                        const bool wasCovered = coverageRow[x] != 0;
                        outRow[x] = value;
                        coverageRow[x] = 1;
                        if (!wasCovered)
                            ++localStats.coveredPixels;
                    }
                }
            }
        }
        return localStats;
    };

    if (!shouldParallelizeSamples(h, w) || tiles.size() <= 1)
        return processTileRange(0, tiles.size());

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end)
            break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] {
            return processTileRange(begin, end);
        }));
    }
    for (auto& future : futures) {
        addStats(stats, future.get());
    }
    return stats;
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::samplePlaneLevel(
    IChunkedArray& array,
    int level,
    const cv::Vec3f& origin,
    const cv::Vec3f& vxStep,
    const cv::Vec3f& vyStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    return samplePlaneLevelImpl(array, level, origin, vxStep, vyStep, out, coverage,
                                options, false);
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsLevel(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    return sampleCoordsLevelImpl(array, level, coords, out, coverage, options, false);
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsMaxComposite(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    int layerStart,
    int numLayers,
    float layerStep,
    float isoCutoff,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    ChunkedPlaneSampler::Stats stats;
    if (level < 0 || level >= array.numLevels() || coords.empty() || out.empty() ||
        coverage.empty() || normals.empty() || numLayers <= 0)
        return stats;

    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return stats;

    const int tile = std::max(1, options.tileSize);
    const int h = std::min({coords.rows, out.rows, coverage.rows, normals.rows});
    const int w = std::min({coords.cols, out.cols, coverage.cols, normals.cols});

    // chunkShape is the codec atom (32^3) -- a power of two -- so chunk index =
    // voxel >> log2(shape) and in-chunk offset = voxel & (shape-1), avoiding the
    // 3 runtime integer divisions per depth (a real idiv ~20-40 cyc each, N* per
    // pixel in composite). Fall back to div/mod if a non-pow2 shape ever appears.
    auto log2pow2 = [](int v) -> int { return (v > 0 && (v & (v - 1)) == 0)
                                            ? __builtin_ctz(unsigned(v)) : -1; };
    const int sh0 = log2pow2(access.chunkShape[0]);
    const int sh1 = log2pow2(access.chunkShape[1]);
    const int sh2 = log2pow2(access.chunkShape[2]);
    const bool pow2Chunks = sh0 >= 0 && sh1 >= 0 && sh2 >= 0;

    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((h + tile - 1) / tile) * std::size_t((w + tile - 1) / tile));
    for (int ty = 0; ty < h; ty += tile)
        for (int tx = 0; tx < w; tx += tile)
            tiles.push_back({tx, ty, std::min(tx + tile, w), std::min(ty + tile, h)});

    auto processTileRange = [&](std::size_t begin, std::size_t end) {
        ChunkedPlaneSampler::Stats localStats;
        LocalChunkCache chunkCache(array, std::max<std::size_t>(16, (end - begin) * 4));
        for (std::size_t i = begin; i < end; ++i) {
            const SampleTile& st = tiles[i];
            for (int y = st.ty; y < st.yEnd; ++y) {
                const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
                const cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
                uint8_t* outRow = out.ptr<uint8_t>(y);
                uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
                for (int x = st.tx; x < st.xEnd; ++x) {
                    const cv::Vec3f base = coordRow[x];
                    // Hole pixel -> leave uncovered (matches per-layer sentinel
                    // skip, where the offset coord stays the sentinel).
                    if (surfaceSentinel(base))
                        continue;
                    const cv::Vec3f nrm = normalRow[x];
                    // Walk the depths for THIS pixel inline. The N depths are
                    // offset along the normal by ~1 voxel each, so they almost
                    // all fall in the SAME chunk -- we resolve the chunk once and
                    // hold its resident buffer across depths, recomputing only the
                    // in-chunk offset + byte read per depth. That avoids N chunk
                    // lookups, N status re-checks and (when the chunk is unchanged)
                    // the cache.get map probe -- readVoxel/cache.get were the bulk
                    // of the composite cost. We track max over covered samples
                    // whose value >= isoCutoff (== composite_max of the layers).
                    const auto& shp = access.shape;
                    const auto& csh = access.chunkShape;
                    const int csh0 = csh[0], csh1 = csh[1], csh2 = csh[2];
                    // Hoist the level transform out of the depth loop: each depth
                    // is (base + nrm*off) mapped to level space. Since the
                    // transform is affine, transform base and nrm ONCE, then each
                    // depth is baseL + nrmL*off -- no per-depth toLevelCoord call.
                    const cv::Vec3f baseL = toLevelCoord(access, base);
                    const cv::Vec3f nrmL = toLevelVector(access, nrm);
                    int lastCz = INT_MIN, lastCy = INT_MIN, lastCx = INT_MIN;
                    const std::vector<std::byte>* curBytes = nullptr;
                    bool curAllFill = false, curUsable = false;
                    float best = 0.0f;
                    bool any = false;
                    // off is an induction variable (off += layerStep) -- avoids the
                    // per-depth int->float convert + multiply.
                    float off = float(layerStart) * layerStep;
                    for (int l = 0; l < numLayers; ++l, off += layerStep) {
                        const float fx = baseL[0] + nrmL[0] * off;
                        const float fy = baseL[1] + nrmL[1] * off;
                        const float fz = baseL[2] + nrmL[2] * off;
                        // Lower bound via float (a negative coord rounds toward 0
                        // and would otherwise pass the unsigned check). Upper bound
                        // is the single unsigned int compare below -- we drop the
                        // redundant float < float(shape) compares, which forced a
                        // per-depth int->float convert of each shape dim.
                        if (fx < 0.0f || fy < 0.0f || fz < 0.0f)
                            continue;
                        const int iz = int(fz + 0.5f), iy = int(fy + 0.5f), ix = int(fx + 0.5f);
                        if (unsigned(iz) >= unsigned(shp[0]) || unsigned(iy) >= unsigned(shp[1]) ||
                            unsigned(ix) >= unsigned(shp[2]))
                            continue;
                        // chunk index + in-chunk coords: shift/mask for pow2 chunk
                        // shapes (the 32^3 atom), div/mod otherwise.
                        int cz, cy, cx, lz, ly, lx;
                        if (pow2Chunks) {
                            cz = iz >> sh0; cy = iy >> sh1; cx = ix >> sh2;
                            lz = iz & (csh0 - 1); ly = iy & (csh1 - 1); lx = ix & (csh2 - 1);
                        } else {
                            cz = iz / csh0; cy = iy / csh1; cx = ix / csh2;
                            lz = iz - cz * csh0; ly = iy - cy * csh1; lx = ix - cx * csh2;
                        }
                        if (cz != lastCz || cy != lastCy || cx != lastCx) {
                            lastCz = cz; lastCy = cy; lastCx = cx;
                            const ChunkResult& r = chunkCache.get(
                                {level, cz, cy, cx},
                                localStats.requestedChunks, localStats.errorChunks);
                            curAllFill = (r.status == ChunkStatus::AllFill);
                            curUsable = (r.status == ChunkStatus::Data) && r.bytes;
                            curBytes = curUsable ? r.bytes.get() : nullptr;
                        }
                        uint8_t value;
                        if (curAllFill) {
                            value = access.fill;
                        } else if (curUsable) {
                            // In-chunk byte offset. For the pow2 atom this is a
                            // pair of shifts, not the two runtime multiplies the
                            // compiler emits for the generic chunkShape stride.
                            const std::size_t o = pow2Chunks
                                ? (((std::size_t(lz) << sh1) + std::size_t(ly)) << sh2)
                                      + std::size_t(lx)
                                : (std::size_t(lz) * std::size_t(csh1)
                                      + std::size_t(ly)) * std::size_t(csh2)
                                      + std::size_t(lx);
                            if (o >= curBytes->size())
                                continue;
                            value = std::to_integer<uint8_t>((*curBytes)[o]);
                        } else {
                            continue;   // missing/queued/error chunk
                        }
                        const float fv = float(value);
                        if (fv < isoCutoff)
                            continue;
                        if (!any || fv > best) { best = fv; any = true; }
                    }
                    if (any) {
                        outRow[x] = static_cast<uint8_t>(std::clamp(best, 0.0f, 255.0f));
                        const bool wasCovered = coverageRow[x] != 0;
                        coverageRow[x] = 1;
                        if (!wasCovered)
                            ++localStats.coveredPixels;
                    }
                }
            }
        }
        return localStats;
    };

    if (!shouldParallelizeSamples(h, w) || tiles.size() <= 1)
        return processTileRange(0, tiles.size());

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end)
            break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] {
            return processTileRange(begin, end);
        }));
    }
    for (auto& future : futures)
        addStats(stats, future.get());
    return stats;
}


} // namespace vc::render
