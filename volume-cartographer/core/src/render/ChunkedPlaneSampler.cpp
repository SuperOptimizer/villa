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

    auto runRange = [&](std::size_t begin, std::size_t end) {
        auto dispatch = [&](auto& arr) -> ChunkedPlaneSampler::Stats {
            // Only the production 32^3 chunk size gets a dedicated static (shift/mask)
            // kernel. Every other chunk size -- rare; no real volume uses one -- falls
            // to the generic dynamic-dims path (a few divides instead of shifts),
            // which is correct for any size. This keeps the template fan-out small:
            // CHUNK_LOG2 was {4,5,6,-1} (4 variants per Composite x Trilinear x Array);
            // {4,6} were speculative and never instantiated by a real run. Now {5,-1}.
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
struct CoarseLevel { int shift; int csh, cshY, cshX; int sz, sy, sx; };

// LOD miss-fallback: a fine chunk missed at `level`, so sample the same point
// from the nearest COARSER level that IS resident (blurry but present, no hole,
// while fine chunks stream in). noinline + cold: this is the RARE path (most
// chunks are resident), and pulling its loop + the coarse-level divides OUT of
// the hot per-layer loop frees the registers the depth walk was spilling. The
// hot loop only calls this on an actual miss. The lookup callable forwards to the
// kernel's pin (concrete or virtual), so this stays dispatch-agnostic.
template <typename LookupFn>
__attribute__((noinline, cold))
int coarseFallbackImpl(LookupFn&& doLookup, int level, uint8_t fillVal,
                       const std::vector<CoarseLevel>& coarse,
                       int iz, int iy, int ix)
{
    for (const auto& cl : coarse) {
        const int vz = iz >> cl.shift, vy = iy >> cl.shift, vx = ix >> cl.shift;
        if (unsigned(vz) >= unsigned(cl.sz) || unsigned(vy) >= unsigned(cl.sy) ||
            unsigned(vx) >= unsigned(cl.sx))
            continue;
        const int cz = vz / cl.csh, cy = vy / cl.cshY, cx = vx / cl.cshX;
        const auto rv = doLookup(level + cl.shift, cz, cy, cx);
        if (rv.status == ChunkStatus::AllFill)
            return fillVal;
        if (rv.status == ChunkStatus::Data && rv.bytes) {
            const int lz = vz - cz * cl.csh, ly = vy - cy * cl.cshY, lx = vx - cx * cl.cshX;
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
    std::vector<CoarseLevel> coarse;
    {
        const int n = array.numLevels();
        for (int L2 = level + 1; L2 < n; ++L2) {
            const auto cs = array.chunkShape(L2);
            const auto ls = array.shape(L2);
            if (cs[0] <= 0) break;
            coarse.push_back(CoarseLevel{L2 - level, cs[0], cs[1], cs[2],
                                         ls[0], ls[1], ls[2]});
        }
    }

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
                        if constexpr (kStatic) {
                            cz = iz >> CHUNK_LOG2; cy = iy >> CHUNK_LOG2; cx = ix >> CHUNK_LOG2;
                        }
                        // Raw lock-free resident read -- no probe layer,
                        // no shared_ptr copy (the tick won't evict mid-frame).
                        const auto rv = doLookup(level, cz, cy, cx);
                        curFill = false; curData = nullptr; curSize = 0;
                        if (rv.status == ChunkStatus::AllFill) {
                            curFill = true;
                        } else if (rv.status == ChunkStatus::Data && rv.bytes) {
                            // Cache the raw buffer ptr + size ONCE per chunk; the
                            // per-layer load below skips the vector-header deref.
                            curData = rv.bytes->data();
                            curSize = rv.bytes->size();
                        } else {
                            // Append misses to a flat vector (no per-insert node
                            // alloc); deduped once at flush. Gated by chunk-change
                            // (lastChunk), so at most one push per chunk crossing.
                            if (rv.status == ChunkStatus::MissQueued)
                                recordMiss(missedVec, missedSeen, {level, cz, cy, cx});
                            else if (rv.status == ChunkStatus::Error)
                                ++localStats.errorChunks;
                        }
                    }

                    uint8_t value;
                    if (curFill) {
                        value = fillVal;
                    } else if (curData) {
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
                    } else {
                        // Miss at this level -> try a coarser resident level (blurry
                        // but present). -1 means no coarser level has it either.
                        // noinline/cold helper -- kept out of this loop's frame.
                        const int fb = coarseFallbackImpl(doLookup, level, fillVal,
                                                          coarse, iz, iy, ix);
                        if (fb < 0)
                            continue;
                        value = uint8_t(fb);
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
    // Composite = max reduction over numLayers, nearest sampling. Same unified
    // driver + kernel as the plain plane/quad path (Composite=false there).
    return runRenderDriver<true, false>(array, access, level, coords, normals,
                                        out, coverage, layerStart, numLayers,
                                        layerStep, options.tileSize);
}


} // namespace vc::render
