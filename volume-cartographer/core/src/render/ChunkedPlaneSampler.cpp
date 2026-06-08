#include "vc/core/render/ChunkedPlaneSampler.hpp"
#include "vc/core/render/ChunkCache.hpp"

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

struct SampleTile {
    int tx = 0;
    int ty = 0;
    int xEnd = 0;
    int yEnd = 0;
};

// Bit-level NaN test. Under -ffast-math (-ffinite-math-only, the Unsafe build) the
// compiler assumes no NaN/Inf and PROVES the usual `x != x` self-inequality is
// always false -- so it DELETES the guard, and NaN coords from bad source data
// would no longer be skipped (garbage/black pixels). Reading the bits and testing
// the IEEE-754 NaN pattern (exponent all-ones, mantissa nonzero -> abs > 0x7f800000)
// cannot be elided: it's integer arithmetic the optimizer must keep.
__attribute__((always_inline)) inline bool isNanBits(float x)
{
    std::uint32_t u;
    __builtin_memcpy(&u, &x, sizeof(u));
    return (u & 0x7fffffffu) > 0x7f800000u;   // NaN (also catches no value <= +Inf)
}

// Round a coord to its nearest integer voxel index. int(f + 0.5f) compiles to an
// addps + a float-domain round-trip (cvttps2dq) + per-lane addss/cvttss2si; the
// hardware round-to-nearest convert (cvtss2si, one insn, no +0.5) is leaner and
// was ~3-4% of the composite kernel. Differs from round-half-up only on exact .5
// ties (measure-zero for continuous sample coords; <=1-voxel shift, well within
// the nearest-sampling tolerance). Caller guarantees f >= 0 here.
__attribute__((always_inline)) inline int nearestIdx(float f)
{
    return int(__builtin_lrintf(f));
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

void addStats(ChunkedPlaneSampler::Stats& dst, ChunkedPlaneSampler::Stats& src)
{
    dst.coveredPixels += src.coveredPixels;
    dst.requestedChunks += src.requestedChunks;
    dst.errorChunks += src.errorChunks;
    if (dst.missedKeys.empty())
        dst.missedKeys = std::move(src.missedKeys);
    else
        dst.missedKeys.insert(dst.missedKeys.end(),
            std::make_move_iterator(src.missedKeys.begin()),
            std::make_move_iterator(src.missedKeys.end()));
}

} // namespace

namespace {

// Forward decl: the chunk-major (sample-binned) composite kernel, defined after
// the driver. Routed in via VCA_BINNED.
template <int CHUNK_LOG2, typename ArrayT>
ChunkedPlaneSampler::Stats renderTileRangeBinned(
    ArrayT& array, const LevelAccess& access, int level,
    const cv::Mat_<cv::Vec3f>& coords, const cv::Mat_<cv::Vec3f>& normals,
    cv::Mat_<uint8_t>& out, cv::Mat_<uint8_t>& coverage,
    const std::vector<SampleTile>& tiles, std::size_t begin, std::size_t end,
    int layerStart, int numLayers, float layerStep);

// Shared driver for BOTH composite and non-composite rendering: build the tile
// list, pick the compile-time chunk-log specialization + devirtualized array, run
// renderTileRange (parallel across tiles), reduce stats. Composite passes
// numLayers/layerStart/layerStep + normals; non-composite passes a single
// layer (numLayers ignored when Composite=false) and may pass empty normals.
template <bool Composite, bool Trilinear>
ChunkedPlaneSampler::Stats runRenderDriver(
    IChunkedArray& array, const LevelAccess& access, int level,
    const cv::Mat_<cv::Vec3f>& coords, const cv::Mat_<cv::Vec3f>& normals,
    cv::Mat_<uint8_t>& out, cv::Mat_<uint8_t>& coverage,
    int layerStart, int numLayers, float layerStep, int tileSize)
{
    ChunkedPlaneSampler::Stats stats;
    const int tile = std::max(1, tileSize);
    const int h = std::min(coords.rows, std::min(out.rows, coverage.rows));
    const int w = std::min(coords.cols, std::min(out.cols, coverage.cols));
    std::vector<SampleTile> tiles;
    tiles.reserve(std::size_t((h + tile - 1) / tile) * std::size_t((w + tile - 1) / tile));
    for (int ty = 0; ty < h; ty += tile)
        for (int tx = 0; tx < w; tx += tile)
            tiles.push_back({tx, ty, std::min(tx + tile, w), std::min(ty + tile, h)});

    const int cs0 = access.chunkShape[0];
    const bool cubicPow2 = cs0 > 0 && (cs0 & (cs0 - 1)) == 0 &&
                           access.chunkShape[1] == cs0 && access.chunkShape[2] == cs0;
    const int chunkLog = cubicPow2 ? __builtin_ctz(unsigned(cs0)) : -1;
    ChunkCache* cc = dynamic_cast<ChunkCache*>(&array);

    // VCA_BINNED: route the composite path through the chunk-major (sample-binned)
    // kernel instead of the pixel-major one. A/B flag while binning is validated.
    static const bool kBinned = Composite && (::getenv("VCA_BINNED") != nullptr);
    auto runRange = [&](std::size_t begin, std::size_t end) {
        auto dispatch = [&](auto& arr) -> ChunkedPlaneSampler::Stats {
            if constexpr (Composite) {
                if (kBinned) {
                    if (chunkLog == 5)
                        return renderTileRangeBinned<5>(arr, access, level, coords, normals,
                            out, coverage, tiles, begin, end, layerStart, numLayers, layerStep);
                    return renderTileRangeBinned<-1>(arr, access, level, coords, normals,
                        out, coverage, tiles, begin, end, layerStart, numLayers, layerStep);
                }
            }
            // Only the production 32^3 chunk size gets a dedicated static (shift/mask)
            // kernel. Every other chunk size -- rare; no real volume uses one -- falls
            // to the generic dynamic-dims path (a few divides instead of shifts).
            if (chunkLog == 5)
                return renderTileRange<Composite, Trilinear, 5>(arr, access, level, coords, normals,
                    out, coverage, tiles, begin, end, layerStart, numLayers, layerStep);
            return renderTileRange<Composite, Trilinear, -1>(arr, access, level, coords, normals,
                out, coverage, tiles, begin, end, layerStart, numLayers, layerStep);
        };
        return cc ? dispatch(*cc) : dispatch(array);
    };

    if (!shouldParallelizeSamples(h, w) || tiles.size() <= 1)
        return runRange(0, tiles.size());

    const std::size_t workerCount = std::min<std::size_t>(
        renderSamplerPool().worker_count(), tiles.size());
    const std::size_t tilesPerWorker = (tiles.size() + workerCount - 1) / workerCount;
    std::vector<std::future<ChunkedPlaneSampler::Stats>> futures;
    futures.reserve(workerCount);
    for (std::size_t worker = 0; worker < workerCount; ++worker) {
        const std::size_t begin = worker * tilesPerWorker;
        const std::size_t end = std::min(begin + tilesPerWorker, tiles.size());
        if (begin >= end) break;
        futures.push_back(renderSamplerPool().submit([&, begin, end] { return runRange(begin, end); }));
    }
    for (auto& future : futures) {
        auto fstats = future.get();
        addStats(stats, fstats);
    }
    return stats;
}

// Non-composite entry shared by sampleCoordsLevel + samplePlaneLevel: dispatch on
// the sampling mode (nearest vs trilinear) into the unified kernel.
ChunkedPlaneSampler::Stats runNonComposite(
    IChunkedArray& array, int level,
    const cv::Mat_<cv::Vec3f>& coords, const cv::Mat_<cv::Vec3f>& normals,
    cv::Mat_<uint8_t>& out, cv::Mat_<uint8_t>& coverage,
    const ChunkedPlaneSampler::Options& options)
{
    if (level < 0 || level >= array.numLevels() || coords.empty() || out.empty() || coverage.empty())
        return {};
    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return {};
    if (options.sampling == vc::Sampling::Trilinear)
        return runRenderDriver<false, true>(array, access, level, coords, normals,
                                            out, coverage, 0, 1, 0.0f, options.tileSize);
    return runRenderDriver<false, false>(array, access, level, coords, normals,
                                         out, coverage, 0, 1, 0.0f, options.tileSize);
}

}  // namespace

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
    // A plane is a surface whose coords are origin + vx*x + vy*y. Materialize that
    // coords Mat (in level-0 space) and run the unified non-composite kernel -- one
    // path for planes and quads.
    if (out.empty() || coverage.empty())
        return {};
    cv::Mat_<cv::Vec3f> coords(out.rows, out.cols);
    for (int y = 0; y < out.rows; ++y) {
        cv::Vec3f* row = coords.ptr<cv::Vec3f>(y);
        const cv::Vec3f rowBase = origin + vyStep * float(y);
        for (int x = 0; x < out.cols; ++x)
            row[x] = rowBase + vxStep * float(x);
    }
    static const cv::Mat_<cv::Vec3f> kNoNormals;
    return runNonComposite(array, level, coords, kNoNormals, out, coverage, options);
}

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsLevel(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    static const cv::Mat_<cv::Vec3f> kNoNormals;
    return runNonComposite(array, level, coords, kNoNormals, out, coverage, options);
}

namespace {

// Per-tile-range max-composite kernel, templated on the chunk edge log2 so the
// chunk-index shift, in-chunk mask and the row/plane strides are COMPILE-TIME
// constants for the common 32^3 atom (CHUNK_LOG2 == 5). That lets the compiler
// emit >>5 / &31 and constant-fold the stride math instead of loading runtime
// chunkShape values -- which also frees the registers those values occupied,
// cutting the spilling in this hot loop. CHUNK_LOG2 == -1 selects a generic
// runtime path (non-power-of-two or non-cubic chunk shapes; not hit in practice).
// Unified render kernel. Composite=true walks numLayers along the normal and
// max-reduces (the composite=max path). Composite=false samples a single point on
// the surface and writes it (the plain plane/coords path) -- if constexpr collapses
// the depth loop to one iteration and the reduction to a direct write, so the
// non-composite case is the same optimized body (inlined lookup, 8-byte key, flat
// miss list, LOD fallback) with zero composite overhead. Both use nearest sampling
// Record a missed chunk. Out-of-line + cold so the vector's capacity-grow/realloc
// (operator new) machinery stays OUT of the hot per-depth lookup loop -- a miss is
// the rare case (most chunks are resident), and inlining the push_back's grow branch
// into the kernel bloated it and forced register spills. The kernel calls this only
// on an actual miss.
// Record a UNIQUE missed chunk. Deduped via a flat hash set (O(1) insert, no sort)
// so a chunk missed by thousands of pixels is pushed once -- without this the miss
// list floods to millions of duplicate keys. Out-of-line + cold: a miss is the rare
// case, and keeping the set insert + vector grow out of the hot loop avoids bloat.
__attribute__((noinline, cold))
void recordMiss(std::vector<ChunkKey>& missedVec,
                std::unordered_set<std::uint64_t>& seen, ChunkKey key)
{
    if (seen.insert(key.word()).second)
        missedVec.push_back(key);
}

// One coarser LOD level for the miss-fallback (precomputed per render call).
// pow2 + the cshLog* fields let the per-sample chunk math use shifts/masks
// instead of runtime integer divides (idivl was ~6% of the kernel during
// streaming, when misses are frequent). cshLog<0 / pow2=false -> divide fallback.
struct CoarseLevel {
    int shift; int csh, cshY, cshX; int sz, sy, sx;
    int cshLog, cshLogY, cshLogX;   // log2 of chunk dims (valid iff pow2)
    bool pow2;
};

// Build the coarser-LOD fallback table. noinline so the vector construction +
// push_back grow machinery (operator new/delete, throw_length_error, memcpy)
// lives HERE, not inlined into the hot kernel where it bloated the function and
// its frame. Built once per kernel call (cold).
__attribute__((noinline))
std::vector<CoarseLevel> buildCoarseTable(IChunkedArray& array, int level)
{
    std::vector<CoarseLevel> coarse;
    const int n = array.numLevels();
    auto log2pos = [](int v) { return (v > 0 && (v & (v - 1)) == 0)
                                      ? __builtin_ctz(unsigned(v)) : -1; };
    for (int L2 = level + 1; L2 < n; ++L2) {
        const auto cs = array.chunkShape(L2);
        const auto ls = array.shape(L2);
        if (cs[0] <= 0) break;
        const int lg = log2pos(cs[0]), lgY = log2pos(cs[1]), lgX = log2pos(cs[2]);
        coarse.push_back(CoarseLevel{L2 - level, cs[0], cs[1], cs[2],
                                     ls[0], ls[1], ls[2], lg, lgY, lgX,
                                     (lg >= 0 && lgY >= 0 && lgX >= 0)});
    }
    return coarse;
}

// LOD miss-fallback: a fine chunk missed at `level`, so sample the same point
// from the nearest COARSER level that IS resident (blurry but present, no hole,
// while fine chunks stream in). noinline + cold: this is the RARE path (most
// chunks are resident), and pulling its loop + the coarse-level divides OUT of
// the hot per-layer loop frees the registers the depth walk was spilling. The
// hot loop only calls this on an actual miss. The lookup callable forwards to the
// kernel's pin (concrete or virtual), so this stays dispatch-agnostic.
// Templated on the PIN type (concrete ChunkCache::ResidentPin vs the virtual
// pin), NOT on a per-kernel lambda -- so all CHUNK_LOG2 variants of a given array
// type share ONE instantiation (was 4 near-identical copies: {5,-1} x {concrete,
// virtual}; now 2). Smaller .text / less icache pressure. The lookup dispatch is
// the same is_same_v branch the kernel's doLookup used.
template <typename PinT>
__attribute__((noinline, cold))
int coarseFallbackImpl(PinT& pin, int level, uint8_t fillVal,
                       const std::vector<CoarseLevel>& coarse,
                       int iz, int iy, int ix)
{
    auto doLookup = [&](int lv, int z, int y, int x) {
        if constexpr (std::is_same_v<PinT, ChunkCache::ResidentPin>)
            return pin.lookupChecked(lv, z, y, x);
        else
            return pin->lookup(lv, z, y, x);
    };
    for (const auto& cl : coarse) {
        const int vz = iz >> cl.shift, vy = iy >> cl.shift, vx = ix >> cl.shift;
        if (unsigned(vz) >= unsigned(cl.sz) || unsigned(vy) >= unsigned(cl.sy) ||
            unsigned(vx) >= unsigned(cl.sx))
            continue;
        int cz, cy, cx, lz, ly, lx;
        if (cl.pow2) {
            // Power-of-2 chunk dims (32^3 in production): chunk index = high bits,
            // in-chunk offset = low bits. Shifts/masks, no idivl.
            cz = vz >> cl.cshLog;  cy = vy >> cl.cshLogY;  cx = vx >> cl.cshLogX;
            lz = vz & (cl.csh - 1); ly = vy & (cl.cshY - 1); lx = vx & (cl.cshX - 1);
        } else {
            cz = vz / cl.csh; cy = vy / cl.cshY; cx = vx / cl.cshX;
            lz = vz - cz * cl.csh; ly = vy - cy * cl.cshY; lx = vx - cx * cl.cshX;
        }
        const auto rv = doLookup(level + cl.shift, cz, cy, cx);
        if (rv.status == ChunkStatus::AllFill)
            return fillVal;
        if (rv.status == ChunkStatus::Data && rv.bytes) {
            const std::size_t o = (std::size_t(lz) * std::size_t(cl.cshY)
                                  + std::size_t(ly)) * std::size_t(cl.cshX) + std::size_t(lx);
            if (o < rv.bytes->size())
                return std::to_integer<uint8_t>((*rv.bytes)[o]);
        }
    }
    return -1;
}

// here; the Trilinear template path handles 8-corner interpolation.
//
// noinline: each <Composite,Trilinear,CHUNK_LOG2,ArrayT> instantiation must be its
// OWN function. Without this, the dispatch switch inlines ALL the variants (composite
// max, plain nearest, trilinear, x3 chunk-logs, x2 array types) into one giant blob
// in the worker thunk -- the register allocator then juggles every variant's live set
// at once and spills heavily (~31 stack slots, 44 inlined vector-grow calls). Split
// out, each kernel gets its own register budget and the hot composite loop stays in
// registers.
template <bool Composite, bool Trilinear, int CHUNK_LOG2, typename ArrayT>
__attribute__((noinline)) ChunkedPlaneSampler::Stats renderTileRange(
    ArrayT& array,
    const LevelAccess& access,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const std::vector<SampleTile>& tiles,
    std::size_t begin,
    std::size_t end,
    int layerStart,
    int numLayers,
    float layerStep)
{
    constexpr bool kStatic = (CHUNK_LOG2 >= 0);
    // For the static path everything is a compile-time constant; for the generic
    // path these are the runtime chunk dims.
    const int dynLog = kStatic ? CHUNK_LOG2 : 0;
    const int csh = kStatic ? (1 << CHUNK_LOG2) : access.chunkShape[0];
    const int cshY = kStatic ? (1 << CHUNK_LOG2) : access.chunkShape[1];
    const int cshX = kStatic ? (1 << CHUNK_LOG2) : access.chunkShape[2];
    (void)dynLog; (void)csh; (void)cshY; (void)cshX;

    const std::array<int, 3>& shp = access.shape;
    // Hoist the grid extents to plain scalars. shp is a reference into `access`;
    // the per-layer bounds check reloaded shp[0..2] from memory every layer
    // because the optimizer can't always prove the cv::Mat output writes don't
    // alias `access`. As locals they stay in registers across the depth loop.
    const int shp0 = shp[0], shp1 = shp[1], shp2 = shp[2];
    const uint8_t fillVal = access.fill;

    // Precompute the level transform ONCE as float scalars (it's constant for the
    // whole call). The per-pixel toLevelCoord/toLevelVector were doing double-
    // precision math + a per-pixel "offset == 0?" check; here that check is hoisted
    // and the per-pixel transform is a plain float multiply (+ optional add).
    const auto& tf = access.transform;
    const float tsx = float(tf.scaleFromLevel0[0]);
    const float tsy = float(tf.scaleFromLevel0[1]);
    const float tsz = float(tf.scaleFromLevel0[2]);
    const float tox = float(tf.offsetFromLevel0[0]);
    const float toy = float(tf.offsetFromLevel0[1]);
    const float toz = float(tf.offsetFromLevel0[2]);
    const bool zeroOffset = tf.offsetFromLevel0[0] == 0.0 &&
                            tf.offsetFromLevel0[1] == 0.0 && tf.offsetFromLevel0[2] == 0.0;

    ChunkedPlaneSampler::Stats localStats;
    // Pin the resident map for the whole tile range so the raw byte pointers from
    // lookup() stay valid even if the tick swaps in a new map concurrently (the
    // cache is shared across viewers). For a concrete ChunkCache take a by-value
    // pin so lookup() is a direct (inlinable) call; else the virtual pin.
    constexpr bool kConcrete = std::is_same_v<ArrayT, ChunkCache>;
    auto pin = [&] {
        if constexpr (kConcrete) return array.pinConcrete();
        else return array.makeResidentPin();
    }();
    auto doLookup = [&](int lv, int z, int y, int x) {
        // Concrete path: lookupChecked is tiny + inlines (the kernel already
        // bounds-checked the voxel coords, so the chunk coords are in-grid -- skip
        // isValidKey). Virtual path keeps the full checked lookup.
        if constexpr (kConcrete) return pin.lookupChecked(lv, z, y, x);
        else return pin->lookup(lv, z, y, x);
    };
    std::vector<ChunkKey> missedVec;   // unique missed chunks (deduped via missedSeen)
    std::unordered_set<std::uint64_t> missedSeen;
    missedVec.reserve(std::max<std::size_t>(16, (end - begin) * 4));
    missedSeen.reserve(std::max<std::size_t>(16, (end - begin) * 4));

    // Read ONE voxel at level-L integer coords (iz,iy,ix). Returns the value, or -1
    // on a miss (records the missed chunk). Used by the trilinear path, whose 8
    // corners can each fall in a different chunk. Nearest sampling uses the inlined
    // run-cache in the loop body instead (this is only the per-corner trilinear read).
    auto readVoxelAt = [&](int iz, int iy, int ix) -> int {
        if (unsigned(iz) >= unsigned(shp0) || unsigned(iy) >= unsigned(shp1) ||
            unsigned(ix) >= unsigned(shp2))
            return -1;
        int cz, cy, cx, lz, ly, lx;
        if constexpr (kStatic) {
            constexpr int kLog = CHUNK_LOG2, kMask = (1 << CHUNK_LOG2) - 1;
            cz = iz >> kLog; cy = iy >> kLog; cx = ix >> kLog;
            lz = iz & kMask; ly = iy & kMask; lx = ix & kMask;
        } else {
            cz = iz / csh; cy = iy / cshY; cx = ix / cshX;
            lz = iz - cz * csh; ly = iy - cy * cshY; lx = ix - cx * cshX;
        }
        const auto rv = doLookup(level, cz, cy, cx);
        if (rv.status == ChunkStatus::AllFill)
            return fillVal;
        if (rv.status == ChunkStatus::Data && rv.bytes) {
            std::size_t o;
            if constexpr (kStatic)
                o = ((std::size_t(lz) << CHUNK_LOG2) + std::size_t(ly))
                        * (std::size_t(1) << CHUNK_LOG2) + std::size_t(lx);
            else
                o = (std::size_t(lz) * std::size_t(cshY) + std::size_t(ly))
                        * std::size_t(cshX) + std::size_t(lx);
            if (o < rv.bytes->size())
                return std::to_integer<uint8_t>((*rv.bytes)[o]);
            return -1;
        }
        if (rv.status == ChunkStatus::MissQueued)
            recordMiss(missedVec, missedSeen, {level, cz, cy, cx});
        else if (rv.status == ChunkStatus::Error)
            ++localStats.errorChunks;
        return -1;
    };

    // LOD fallback: when a chunk misses at `level`, sample the same point from the
    // nearest COARSER level that IS resident -- blurry but present, instead of a
    // hole, while the fine chunks stream in. Precompute per coarser level its chunk
    // dims, grid shape, and the bit shift from level-L voxels to that level's voxels
    // (pyramid is 2x per level, so coord >>= (L2-L)). Sample value at the coarse
    // chunk's nearest voxel. Cheap: only walked on a miss.
    // Precompute the coarser-level table once (cold miss-fallback uses it). See
    // coarseFallbackImpl -- that function is noinline/cold and kept OUT of this
    // frame so the hot depth loop doesn't pay its register footprint.
    const std::vector<CoarseLevel> coarse = buildCoarseTable(array, level);

    // Chunk-major lookup cache (cross-pixel dedup). The per-pixel depth walk only
    // probes the resident map on a chunk CHANGE, but ADJACENT pixels re-probe the
    // SAME chunks -- a 32x32 tile touches only a handful of distinct chunks, yet
    // each of its ~1000 pixels independently hash-looks-up those chunks. Cache the
    // resolved (data ptr/size/fill) by chunk key in a tiny MRU table checked before
    // doLookup, so each distinct chunk hits the resident map ONCE per tile-range
    // instead of once per pixel. Valid for the whole SETTLE phase (the resident map
    // is frozen). This is the first, least-invasive step of chunk-major traversal:
    // it removes the cross-pixel lookup duplication without reordering the loops.
    struct ChunkCacheEntry { std::uint64_t key; const std::byte* data; std::size_t size; bool fill; };
    constexpr int kTileCacheN = 16;   // distinct chunks per neighborhood is small
    // thread_local so the 16-entry table lives in TLS, NOT this kernel's stack
    // frame -- it was ~384 bytes of frame that pushed the hot loop's working set
    // into spills. Persists across tile-range calls on a worker; cleared here each
    // call (the resident map is frozen per SETTLE, so stale entries must not leak).
    static thread_local ChunkCacheEntry tcache[kTileCacheN];
    for (auto& e : tcache) { e.key = ~std::uint64_t(0); e.data = nullptr; e.size = 0; e.fill = false; }
    int tcacheHead = 0;
    // Resolve a chunk via the cache; on miss, probe the resident map + record a fine
    // miss. Returns by setting curData/curSize/curFill (matches the inline path).
    auto resolveChunk = [&](std::uint64_t ckey, int cz, int cy, int cx,
                            const std::byte*& curData, std::size_t& curSize, bool& curFill) {
        for (int k = 0; k < kTileCacheN; ++k) {
            if (tcache[k].key == ckey) {
                curData = tcache[k].data; curSize = tcache[k].size; curFill = tcache[k].fill;
                return;
            }
        }
        // Miss in the tile cache -> probe the resident map.
        const auto rv = doLookup(level, cz, cy, cx);
        curFill = false; curData = nullptr; curSize = 0;
        if (rv.status == ChunkStatus::AllFill) {
            curFill = true;
        } else if (rv.status == ChunkStatus::Data && rv.bytes) {
            curData = rv.bytes->data(); curSize = rv.bytes->size();
        } else {
            // Record the miss ONCE (the resident map is frozen for the whole SETTLE
            // phase, so this chunk stays missing) then cache the miss result, so later
            // pixels over the same missing chunk neither re-probe the map NOR re-record.
            if (rv.status == ChunkStatus::MissQueued)
                recordMiss(missedVec, missedSeen, {level, cz, cy, cx});
            else if (rv.status == ChunkStatus::Error)
                ++localStats.errorChunks;
        }
        // Insert the result (resident OR miss) into the MRU ring.
        tcache[tcacheHead] = ChunkCacheEntry{ckey, curData, curSize, curFill};
        tcacheHead = (tcacheHead + 1) % kTileCacheN;
    };


    for (std::size_t i = begin; i < end; ++i) {
        const SampleTile& st = tiles[i];
        for (int y = st.ty; y < st.yEnd; ++y) {
            const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
            // Normals are only needed to step along the depth in the composite case.
            const cv::Vec3f* normalRow = Composite ? normals.ptr<cv::Vec3f>(y) : nullptr;
            uint8_t* outRow = out.ptr<uint8_t>(y);
            uint8_t* coverageRow = coverage.ptr<uint8_t>(y);
            for (int x = st.tx; x < st.xEnd; ++x) {
                const cv::Vec3f base = coordRow[x];
                // QuadSurface::gen() stamps NaN-NaN-NaN wherever the surface does
                // not exist (its invalidation pass always runs -- single-component
                // forces it, multi-component uses NaN borders), so for gen() output
                // a single NaN test per pixel is sufficient. NaN propagates through
                // all three components, so checking base[0] alone suffices. This
                // replaces the general (-1 / (0,0,0) / NaN) sentinel checks
                // -- those markers only appear in raw _points, never in gen output.
                // Bit test (not base[0]!=base[0]) so -ffast-math can't elide it.
                if (isNanBits(base[0]))
                    continue;
                // base/normal -> level space using the hoisted float scalars.
                const cv::Vec3f baseL = zeroOffset
                    ? cv::Vec3f(base[0]*tsx, base[1]*tsy, base[2]*tsz)
                    : cv::Vec3f(base[0]*tsx+tox, base[1]*tsy+toy, base[2]*tsz+toz);
                cv::Vec3f nrmL(0.0f, 0.0f, 0.0f);
                if constexpr (Composite) {
                    const cv::Vec3f nrm = normalRow[x];
                    nrmL = cv::Vec3f(nrm[0]*tsx, nrm[1]*tsy, nrm[2]*tsz);
                }

                // Chunk-change detection across depth layers. STATIC path: detect
                // "still in the same 2^LOG chunk" straight from the int voxel coords
                // -- chunk changed iff any of (iz,iy,ix) crossed a 2^LOG boundary,
                // i.e. (iz^pz | iy^py | ix^px) >> LOG != 0. Profiling showed this
                // beats packing a u64 key per layer: the coords arrive packed in an
                // xmm, so building+comparing a key re-vectorizes into vpextrq extracts
                // (~5% of the kernel); the XOR test stays scalar. pPrev seeds to -1 so
                // the first in-bounds layer forces the lookup. DYNAMIC path: packed key.
                int pz = -1, py = -1, px = -1;
                // Previous SAMPLED voxel (composite no-double-sample guard). -2 so
                // the first valid layer (coords >= 0) never matches.
                int pvz = -2, pvy = -2, pvx = -2;
                std::uint64_t lastChunk = ~std::uint64_t(0);
                // Chunk-resident state cached across the depth loop. The previous
                // code held a `const std::vector<std::byte>*` and dereferenced its
                // header (data ptr at +0x0, size at +0x8) EVERY layer for the
                // bounds check + load. Those are invariant within a chunk, so cache
                // the RAW data pointer + size on a chunk crossing; the per-layer
                // path is then `if (o < curSize) curData[o]` -- two registers, no
                // vector-header indirection. curFill = the all-fill chunk (no buffer).
                const std::byte* curData = nullptr;
                std::size_t curSize = 0;
                bool curFill = false;
                // Seed best at -1 (below any real sample, which is [0,255]) so the
                // reduction needs no separate `any` bool ("covered" is best >= 0
                // after the loop). COMPOSITE keeps the running max as an INT -- the
                // samples are bytes and the reduce is a pure max, so staying in the
                // integer domain drops the per-layer byte->float vcvtsi2ss (~3%); we
                // convert once after the loop. The non-composite/trilinear paths
                // produce a (possibly fractional) float result, so they use `best`.
                int bestI = -1;
                float best = -1.0f;
                // Composite walks numLayers along the normal + max-reduces; the
                // non-composite (single plane) case is the SAME body with exactly
                // one layer at the surface and a direct write (if constexpr collapses
                // the loop + reduction away).
                const int kLayers = Composite ? numLayers : 1;
                // Strength-reduce the per-depth coord: instead of baseL + nrmL*off
                // (3 mul + 3 add per layer), walk it incrementally f += nrmStep
                // (3 add per layer). nrmStep/the initial f are hoisted out of the
                // depth loop. Exact affine walk; the tiny accumulated rounding is
                // absorbed by the int truncation at the sample.
                const float off0 = Composite ? float(layerStart) * layerStep : 0.0f;
                float fx = baseL[0] + nrmL[0] * off0;
                float fy = baseL[1] + nrmL[1] * off0;
                float fz = baseL[2] + nrmL[2] * off0;
                const float sx = nrmL[0] * layerStep;
                const float sy = nrmL[1] * layerStep;
                const float sz = nrmL[2] * layerStep;
                for (int l = 0; l < kLayers; ++l,
                         fx += sx, fy += sy, fz += sz) {
                    if constexpr (Trilinear) {
                        // Cull samples that stepped off the low edge. ONLY the
                        // trilinear path needs this: it uses int() truncation +
                        // reads iz/iz+1 with a fractional weight, so a negative
                        // coord would extrapolate (dx<0). The nearest/composite path
                        // below instead relies on its unsigned(i) >= shp bounds check
                        // -- a negative coord rounds to a negative int that the
                        // unsigned compare already rejects -- so it does NOT pay this
                        // min/branch per layer (it was ~2 vminss + 1 branch in the
                        // hot composite loop, redundant with the bounds check there).
                        // min(fx,fy,fz)<0 folds the 3 sign tests into one compare.
                        if (std::fmin(fx, std::fmin(fy, fz)) < 0.0f)
                            continue;
                        // 8-corner trilinear interp. Corners can cross chunks, so
                        // each is read individually; any missing corner -> skip
                        // (the run-cache below is nearest-only). Composite never
                        // uses Trilinear (it's nearest), so this is the !Composite
                        // single-sample path.
                        const int ix = int(fx), iy = int(fy), iz = int(fz);
                        if (iz + 1 >= shp0 || iy + 1 >= shp1 || ix + 1 >= shp2)
                            continue;
                        const float dx = fx - float(ix), dy = fy - float(iy), dz = fz - float(iz);
                        const int v000 = readVoxelAt(iz,   iy,   ix);
                        const int v001 = readVoxelAt(iz,   iy,   ix+1);
                        const int v010 = readVoxelAt(iz,   iy+1, ix);
                        const int v011 = readVoxelAt(iz,   iy+1, ix+1);
                        const int v100 = readVoxelAt(iz+1, iy,   ix);
                        const int v101 = readVoxelAt(iz+1, iy,   ix+1);
                        const int v110 = readVoxelAt(iz+1, iy+1, ix);
                        const int v111 = readVoxelAt(iz+1, iy+1, ix+1);
                        if ((v000|v001|v010|v011|v100|v101|v110|v111) < 0)
                            continue;   // a corner missed
                        const float c00 = std::fma(dx, float(v001 - v000), float(v000));
                        const float c01 = std::fma(dx, float(v011 - v010), float(v010));
                        const float c10 = std::fma(dx, float(v101 - v100), float(v100));
                        const float c11 = std::fma(dx, float(v111 - v110), float(v110));
                        const float c0 = std::fma(dy, c01 - c00, c00);
                        const float c1 = std::fma(dy, c11 - c10, c10);
                        best = std::clamp(std::fma(dz, c1 - c0, c0), 0.0f, 255.0f);
                        break;   // single sample (Trilinear implies !Composite); best>=0 marks covered
                    }

                    const int iz = nearestIdx(fz), iy = nearestIdx(fy), ix = nearestIdx(fx);
                    if (unsigned(iz) >= unsigned(shp0) || unsigned(iy) >= unsigned(shp1) ||
                        unsigned(ix) >= unsigned(shp2))
                        continue;

                    // STRICT no-double-sample guard (composite): if this layer rounds
                    // to the SAME voxel as the previous one, skip the load+reduce --
                    // max(x,x)=x, so re-reading the identical byte is pure waste. The
                    // depth-decimation upstream already thins most duplicates; this
                    // catches the residual (endpoints / sub-voxel steps) so NO voxel
                    // is ever sampled twice in a column. Reuses pv* (prev voxel).
                    if constexpr (Composite) {
                        if (iz == pvz && iy == pvy && ix == pvx)
                            continue;
                        pvz = iz; pvy = iy; pvx = ix;
                    }

                    // In-chunk offset (needed every layer for the byte load).
                    int lz, ly, lx;
                    if constexpr (kStatic) {
                        constexpr int kMask = (1 << CHUNK_LOG2) - 1;
                        lz = iz & kMask; ly = iy & kMask; lx = ix & kMask;
                    } else {
                        const int cz0 = iz / csh, cy0 = iy / cshY, cx0 = ix / cshX;
                        lz = iz - cz0 * csh; ly = iy - cy0 * cshY; lx = ix - cx0 * cshX;
                    }

                    // Did we leave the previous chunk? Static: scalar XOR of high
                    // bits (no SIMD pack). Dynamic: compare the divide-based key.
                    bool chunkChanged;
                    int cz = 0, cy = 0, cx = 0;
                    if constexpr (kStatic) {
                        chunkChanged = ((unsigned(iz ^ pz) | unsigned(iy ^ py)
                                       | unsigned(ix ^ px)) >> CHUNK_LOG2) != 0;
                        pz = iz; py = iy; px = ix;
                    } else {
                        cz = iz / csh; cy = iy / cshY; cx = ix / cshX;
                        const std::uint64_t chunkKey = (std::uint64_t(unsigned(cz)) << 42)
                                                     | (std::uint64_t(unsigned(cy)) << 21)
                                                     |  std::uint64_t(unsigned(cx));
                        chunkChanged = (chunkKey != lastChunk);
                        lastChunk = chunkKey;
                    }
                    if (chunkChanged) {
                        // Resolve chunk coords (static path defers these to here --
                        // only needed on an actual chunk crossing, not every layer).
                        std::uint64_t ckey;
                        if constexpr (kStatic) {
                            cz = iz >> CHUNK_LOG2; cy = iy >> CHUNK_LOG2; cx = ix >> CHUNK_LOG2;
                            ckey = (std::uint64_t(unsigned(cz)) << 42)
                                 | (std::uint64_t(unsigned(cy)) << 21) | std::uint64_t(unsigned(cx));
                        } else {
                            ckey = lastChunk;   // dynamic path already packed it above
                        }
                        // Tile-cached resolve: hits the resident map only on the first
                        // pixel to touch this chunk in the tile-range; later pixels
                        // reuse the cached data ptr/size (cross-pixel lookup dedup).
                        resolveChunk(ckey, cz, cy, cx, curData, curSize, curFill);
                    }

                    // Resolve the sample value. The two HOT branches (resident data
                    // hit, all-fill chunk) are call-free, so the composite max-reduce
                    // happens inside them with `bestI` staying in a register. The cold
                    // miss path -- which calls coarseFallbackImpl -- is split out below
                    // so `bestI` does NOT have to be spilled across that call every
                    // layer (it was a per-iteration load+store to the stack, ~4.6% of
                    // the kernel: the accumulator survived a call it almost never makes).
                    uint8_t value;
                    if (curData) [[likely]] {
                        std::size_t o;
                        if constexpr (kStatic) {
                            o = ((std::size_t(lz) << CHUNK_LOG2) + std::size_t(ly)
                                ) * (std::size_t(1) << CHUNK_LOG2) + std::size_t(lx);
                        } else {
                            o = (std::size_t(lz) * std::size_t(cshY)
                                + std::size_t(ly)) * std::size_t(cshX)
                                + std::size_t(lx);
                        }
                        if (o >= curSize)
                            continue;
                        value = std::to_integer<uint8_t>(curData[o]);
                    } else if (curFill) {
                        value = fillVal;
                    } else {
                        // Cold miss path: try a coarser resident level (blurry but
                        // present). noinline/cold helper kept out of this loop's frame.
                        const int fb = coarseFallbackImpl(pin, level, fillVal,
                                                          coarse, iz, iy, ix);
                        if (fb < 0)
                            continue;
                        if constexpr (Composite) {
                            if (fb > bestI) bestI = fb;
                            continue;
                        }
                        best = float(fb);
                        break;
                    }

                    if constexpr (Composite) {
                        // Integer max-reduce: value is a byte, stays in a GP register
                        // across the loop -- no per-layer byte->float convert. Branchy
                        // max compiles to cmov (no spill). Converted to float once
                        // after the loop.
                        const int vi = value;
                        if (vi > bestI) bestI = vi;
                    } else {
                        // Single plane sample: take this value and stop.
                        best = float(value);
                        break;
                    }
                }
                if constexpr (Composite)
                    best = float(bestI);
                if (best >= 0.0f) {
                    outRow[x] = static_cast<uint8_t>(std::clamp(best, 0.0f, 255.0f));
                    if (coverageRow[x] == 0)
                        ++localStats.coveredPixels;
                    coverageRow[x] = 1;
                }
            }
        }
    }
    // Hand off the raw miss list -- NO sort+unique here. The tick dedups when it
    // issues fetches (a key already resident/in-flight is skipped), so duplicate
    // keys are harmless; sorting them was ~2% of render time for no benefit. The
    // lastChunk gate already collapses a pixel's same-chunk run to one push, so the
    // only dups are different pixels hitting the same missing chunk -- bounded.
    // requestedChunks is then a slight overcount (a HUD stat only).
    localStats.requestedChunks += static_cast<int>(missedVec.size());
    localStats.missedKeys.insert(localStats.missedKeys.end(),
        std::make_move_iterator(missedVec.begin()),
        std::make_move_iterator(missedVec.end()));
    return localStats;
}

// ---------------------------------------------------------------------------
// CHUNK-MAJOR (sample-binned) composite kernel.
//
// The pixel-major kernel above walks each pixel's depth column and resolves a
// chunk on every chunk-change -- so adjacent pixels re-resolve the same chunks,
// and a missing chunk re-runs the coarse-LOD fallback per layer per pixel.
//
// This kernel inverts the order. Per tile:
//   PASS 1 (scatter): walk every (pixel,layer) sample, compute its level-L voxel
//     coord, and append {pixelLocalIdx, inChunkOffset} to a bucket keyed by chunk.
//   PASS 2 (gather): for each DISTINCT chunk, resolve it ONCE (one resident-map
//     probe; one coarse-LOD fallback if missing), then sweep its bucket writing
//     max into a per-pixel accumulator. Same-chunk bytes are touched contiguously.
//
// Wins vs pixel-major: one lookup per chunk per TILE (not per pixel), the coarse
// fallback's expensive pyramid walk happens once per missing chunk, and the chunk
// bytes are read in a contiguous sweep. Composite-only (binning pays off only with
// many samples/chunk; the 1-layer non-composite path stays pixel-major).
template <int CHUNK_LOG2, typename ArrayT>
__attribute__((noinline)) ChunkedPlaneSampler::Stats renderTileRangeBinned(
    ArrayT& array, const LevelAccess& access, int level,
    const cv::Mat_<cv::Vec3f>& coords, const cv::Mat_<cv::Vec3f>& normals,
    cv::Mat_<uint8_t>& out, cv::Mat_<uint8_t>& coverage,
    const std::vector<SampleTile>& tiles, std::size_t begin, std::size_t end,
    int layerStart, int numLayers, float layerStep)
{
    constexpr bool kStatic = (CHUNK_LOG2 >= 0);
    const int csh  = kStatic ? (1 << CHUNK_LOG2) : access.chunkShape[0];
    const int cshY = kStatic ? (1 << CHUNK_LOG2) : access.chunkShape[1];
    const int cshX = kStatic ? (1 << CHUNK_LOG2) : access.chunkShape[2];

    const int shp0 = access.shape[0], shp1 = access.shape[1], shp2 = access.shape[2];
    const uint8_t fillVal = access.fill;
    const auto& tf = access.transform;
    const float tsx = float(tf.scaleFromLevel0[0]), tsy = float(tf.scaleFromLevel0[1]),
                tsz = float(tf.scaleFromLevel0[2]);
    const float tox = float(tf.offsetFromLevel0[0]), toy = float(tf.offsetFromLevel0[1]),
                toz = float(tf.offsetFromLevel0[2]);
    const bool zeroOffset = tf.offsetFromLevel0[0] == 0.0 && tf.offsetFromLevel0[1] == 0.0
                         && tf.offsetFromLevel0[2] == 0.0;

    ChunkedPlaneSampler::Stats localStats;
    constexpr bool kConcrete = std::is_same_v<ArrayT, ChunkCache>;
    auto pin = [&] { if constexpr (kConcrete) return array.pinConcrete();
                     else return array.makeResidentPin(); }();
    auto doLookup = [&](int lv, int z, int y, int x) {
        if constexpr (kConcrete) return pin.lookupChecked(lv, z, y, x);
        else return pin->lookup(lv, z, y, x);
    };
    std::vector<ChunkKey> missedVec;
    std::unordered_set<std::uint64_t> missedSeen;
    missedVec.reserve(64); missedSeen.reserve(64);

    // Coarse-LOD table (same as the pixel-major path).
    std::vector<CoarseLevel> coarse;
    {
        const int n = array.numLevels();
        auto log2pos = [](int v){ return (v>0 && (v&(v-1))==0) ? __builtin_ctz(unsigned(v)) : -1; };
        for (int L2 = level + 1; L2 < n; ++L2) {
            const auto cs = array.chunkShape(L2); const auto ls = array.shape(L2);
            if (cs[0] <= 0) break;
            const int lg=log2pos(cs[0]), lgY=log2pos(cs[1]), lgX=log2pos(cs[2]);
            coarse.push_back(CoarseLevel{L2-level, cs[0],cs[1],cs[2], ls[0],ls[1],ls[2],
                                         lg,lgY,lgX, (lg>=0&&lgY>=0&&lgX>=0)});
        }
    }

    // One binned sample: which tile-local pixel, and the voxel coord (kept as
    // int coords; the in-chunk offset is derived in the gather sweep). 16 bytes.
    struct BSample { int pix; int iz, iy, ix; };
    // Per-distinct-chunk record: packed key + the [start,count) slice of `order`
    // (sample indices grouped to this chunk) filled during the grouping step.
    struct BChunk { std::uint64_t key; int cz, cy, cx; int start, count; };

    std::vector<BSample> samples;      // all in-grid samples for the current tile
    std::vector<BChunk>  chunks;       // distinct chunks in the tile
    std::vector<int>     order;        // sample indices, grouped by chunk
    std::vector<int>     bestI;        // per-tile-pixel running max (-1 = uncovered)
    // Small open-addressed map chunkKey -> index in `chunks` for the scatter group.
    std::vector<std::uint64_t> hmap;   // {gen:32, chunkIdx:32} open-addressed slots

    for (std::size_t ti = begin; ti < end; ++ti) {
        const SampleTile& st = tiles[ti];
        const int tw = st.xEnd - st.tx, th = st.yEnd - st.ty;
        if (tw <= 0 || th <= 0) continue;
        const int npix = tw * th;
        samples.clear(); chunks.clear();
        bestI.assign(npix, -1);

        // --- PASS 1: scatter samples into per-chunk groups ------------------
        // Open-addressed hash (power-of-2, linear probe) chunkKey -> chunks[] idx.
        // A tile touches only a few dozen distinct chunks, so a fixed small table
        // suffices -- and we clear it via a per-tile GENERATION stamp instead of
        // zeroing it (no O(hsize) memset per tile). hmap stores {gen, idx} packed:
        // an entry is live iff its high 32 bits == the current tile generation.
        constexpr int hbits = 9;                  // 512 slots; >> distinct chunks/tile
        constexpr int hsize = 1 << hbits, hmask = hsize - 1;
        if (int(hmap.size()) < hsize) hmap.assign(hsize, 0);
        const std::uint32_t gen = std::uint32_t(ti) + 1;   // unique per tile in range
        auto chunkIndexFor = [&](std::uint64_t key, int cz, int cy, int cx) -> int {
            int h = int((key * 0x9E3779B97F4A7C15ull) >> (64 - hbits)) & hmask;
            while (true) {
                const std::uint64_t slot = hmap[h];
                if (std::uint32_t(slot >> 32) != gen) {     // empty (stale generation)
                    const int ci = int(chunks.size());
                    chunks.push_back(BChunk{key, cz, cy, cx, 0, 0});
                    hmap[h] = (std::uint64_t(gen) << 32) | std::uint32_t(ci);
                    return ci;
                }
                const int ci = int(std::uint32_t(slot));
                if (chunks[ci].key == key) return ci;
                h = (h + 1) & hmask;
            }
        };

        for (int ly = 0; ly < th; ++ly) {
            const int y = st.ty + ly;
            const cv::Vec3f* coordRow = coords.ptr<cv::Vec3f>(y);
            const cv::Vec3f* normalRow = normals.ptr<cv::Vec3f>(y);
            for (int lx = 0; lx < tw; ++lx) {
                const int x = st.tx + lx;
                const cv::Vec3f base = coordRow[x];
                if (isNanBits(base[0])) continue;
                const cv::Vec3f baseL = zeroOffset
                    ? cv::Vec3f(base[0]*tsx, base[1]*tsy, base[2]*tsz)
                    : cv::Vec3f(base[0]*tsx+tox, base[1]*tsy+toy, base[2]*tsz+toz);
                const cv::Vec3f nrm = normalRow[x];
                const cv::Vec3f nrmL(nrm[0]*tsx, nrm[1]*tsy, nrm[2]*tsz);
                const float off0 = float(layerStart) * layerStep;
                float fx = baseL[0]+nrmL[0]*off0, fy = baseL[1]+nrmL[1]*off0, fz = baseL[2]+nrmL[2]*off0;
                const float sxp = nrmL[0]*layerStep, syp = nrmL[1]*layerStep, szp = nrmL[2]*layerStep;
                const int pix = ly * tw + lx;
                for (int l = 0; l < numLayers; ++l, fx += sxp, fy += syp, fz += szp) {
                    const int iz = nearestIdx(fz), iy = nearestIdx(fy), ix = nearestIdx(fx);
                    if (unsigned(iz) >= unsigned(shp0) || unsigned(iy) >= unsigned(shp1) ||
                        unsigned(ix) >= unsigned(shp2))
                        continue;
                    int cz, cy, cx; std::uint64_t key;
                    if constexpr (kStatic) {
                        cz = iz >> CHUNK_LOG2; cy = iy >> CHUNK_LOG2; cx = ix >> CHUNK_LOG2;
                    } else {
                        cz = iz / csh; cy = iy / cshY; cx = ix / cshX;
                    }
                    key = (std::uint64_t(unsigned(cz)) << 42) | (std::uint64_t(unsigned(cy)) << 21)
                        | std::uint64_t(unsigned(cx));
                    const int ci = chunkIndexFor(key, cz, cy, cx);
                    chunks[ci].count++;
                    samples.push_back(BSample{(pix << 0), iz, iy, ix});
                    // stash the chunk index in the high bits of pix? No -- keep a
                    // parallel array via order grouping below. Use samples index +
                    // remember ci by re-deriving in grouping. Simpler: store ci now.
                    samples.back().pix = pix | (ci << 16);   // pix < 1024 fits in 16 bits
                }
            }
        }
        if (samples.empty()) continue;

        // --- group sample indices by chunk (counting sort over chunks) ------
        int total = 0;
        for (auto& c : chunks) { c.start = total; total += c.count; c.count = 0; }
        order.resize(total);
        for (int si = 0; si < int(samples.size()); ++si) {
            const int ci = samples[si].pix >> 16;
            order[chunks[ci].start + chunks[ci].count++] = si;
        }

        // --- PASS 2: gather -- resolve each chunk once, sweep its samples ----
        for (const auto& c : chunks) {
            const auto rv = doLookup(level, c.cz, c.cy, c.cx);
            const std::byte* cdata = nullptr; std::size_t csize = 0; bool cfill = false;
            bool cmiss = false;
            if (rv.status == ChunkStatus::AllFill) cfill = true;
            else if (rv.status == ChunkStatus::Data && rv.bytes) { cdata = rv.bytes->data(); csize = rv.bytes->size(); }
            else {
                cmiss = true;
                if (rv.status == ChunkStatus::MissQueued)
                    recordMiss(missedVec, missedSeen, {level, c.cz, c.cy, c.cx});
                else if (rv.status == ChunkStatus::Error) ++localStats.errorChunks;
            }
            for (int k = c.start; k < c.start + c.count; ++k) {
                const BSample& s = samples[order[k]];
                const int pix = s.pix & 0xffff;
                int value;
                if (cfill) value = fillVal;
                else if (cdata) {
                    std::size_t o;
                    if constexpr (kStatic) {
                        const int kMask = (1 << CHUNK_LOG2) - 1;
                        o = ((std::size_t(s.iz & kMask) << CHUNK_LOG2) + std::size_t(s.iy & kMask))
                            * (std::size_t(1) << CHUNK_LOG2) + std::size_t(s.ix & kMask);
                    } else {
                        const int lz = s.iz - c.cz*csh, lyy = s.iy - c.cy*cshY, lxx = s.ix - c.cx*cshX;
                        o = (std::size_t(lz)*std::size_t(cshY) + std::size_t(lyy))*std::size_t(cshX) + std::size_t(lxx);
                    }
                    if (o >= csize) continue;
                    value = std::to_integer<uint8_t>(cdata[o]);
                } else {
                    // Missing fine chunk: coarse-LOD fallback. The expensive pyramid
                    // walk's chunk lookups happen here; still per-sample (each maps to
                    // a different coarse voxel) but grouped, so the coarse CHUNK probes
                    // hit the same few coarse chunks back-to-back (cache-warm).
                    const int fb = coarseFallbackImpl(pin, level, fillVal, coarse, s.iz, s.iy, s.ix);
                    if (fb < 0) continue;
                    value = fb;
                }
                if (value > bestI[pix]) bestI[pix] = value;
            }
            (void)cmiss;
        }

        // --- write tile output ---------------------------------------------
        for (int ly = 0; ly < th; ++ly) {
            uint8_t* outRow = out.ptr<uint8_t>(st.ty + ly);
            uint8_t* covRow = coverage.ptr<uint8_t>(st.ty + ly);
            for (int lx = 0; lx < tw; ++lx) {
                const int b = bestI[ly*tw + lx];
                if (b >= 0) {
                    outRow[st.tx + lx] = uint8_t(std::clamp(b, 0, 255));
                    if (covRow[st.tx + lx] == 0) ++localStats.coveredPixels;
                    covRow[st.tx + lx] = 1;
                }
            }
        }
    }

    localStats.requestedChunks += int(missedVec.size());
    localStats.missedKeys.insert(localStats.missedKeys.end(),
        std::make_move_iterator(missedVec.begin()), std::make_move_iterator(missedVec.end()));
    return localStats;
}

}  // namespace

ChunkedPlaneSampler::Stats ChunkedPlaneSampler::sampleCoordsMaxComposite(
    IChunkedArray& array,
    int level,
    const cv::Mat_<cv::Vec3f>& coords,
    const cv::Mat_<cv::Vec3f>& normals,
    int layerStart,
    int numLayers,
    float layerStep,
    cv::Mat_<uint8_t>& out,
    cv::Mat_<uint8_t>& coverage,
    const Options& options)
{
    if (level < 0 || level >= array.numLevels() || coords.empty() || out.empty() ||
        coverage.empty() || normals.empty() || numLayers <= 0)
        return {};
    const LevelAccess access = makeLevelAccess(array, level);
    if (!hasSampleableLevel(access))
        return {};

    // DEPTH-DECIMATION at coarse render levels (do less, keep the same result).
    // The kernel walks numLayers along the unit normal, each step scaled into
    // level-L space by scaleFromLevel0 = 1/2^level. So the per-layer advance is
    // (layerStep / 2^level) level-L VOXELS. At a coarse level that is << 1, so many
    // consecutive layers round (nearestIdx) to the SAME level-L voxel -- e.g. at
    // level 5 with step 1, ~32 layers collapse onto one voxel: 32x redundant loads
    // + max-reduces reading the identical byte. For a MAX reduction over nearest
    // samples, dropping those duplicates is LOSSLESS (the max over the distinct
    // voxels is unchanged). Re-express the SAME depth span at ~1-voxel resolution:
    // step' = layerStep * 2^level (>= 1 voxel), count' = ceil(numLayers / 2^level).
    // Level 0 (scale 1) is unaffected. The start is rescaled to keep the same span.
    const double s = access.transform.scaleFromLevel0[0];   // = 1/2^level (isotropic)
    int effLayerStart = layerStart;
    int effNumLayers = numLayers;
    float effLayerStep = layerStep;
    if (s > 0.0 && s < 1.0 && numLayers > 1) {
        const double perLayerVox = std::abs(double(layerStep)) * s;   // voxels/layer
        const double spanVox = double(numLayers - 1) * double(layerStep) * s;  // signed
        // One sample per level-L voxel across the span (+1 to cover both ends).
        const int n = std::max(1, int(std::ceil(std::abs(spanVox))) + 1);
        // Decimate ONLY when it's a strict win: per-layer advance < 1 voxel (so
        // consecutive layers collapse) AND the voxel-resolution count is smaller.
        // Never increase the layer count -- this is a "do less" optimization.
        if (perLayerVox > 0.0 && perLayerVox < 1.0 && n < numLayers) {
            const double startVox = double(layerStart) * double(layerStep) * s;
            const float stepV = (n > 1) ? float(spanVox / double(n - 1)) : 0.0f;
            // Kernel multiplies layerStart/Step by nrmL (which already carries `s`),
            // so express start/step back in the kernel's (unit-normal * s) units by
            // dividing the voxel quantities by s.
            effLayerStep  = (stepV != 0.0f) ? float(double(stepV) / s) : float(layerStep);
            effLayerStart = (effLayerStep != 0.0f) ? int(std::lround(startVox / s / double(effLayerStep))) : 0;
            effNumLayers  = n;
        }
    }

    // Composite = max reduction over numLayers, nearest sampling. Same unified
    // driver + kernel as the plain plane/quad path (Composite=false there).
    return runRenderDriver<true, false>(array, access, level, coords, normals,
                                        out, coverage, effLayerStart, effNumLayers,
                                        effLayerStep, options.tileSize);
}


} // namespace vc::render
