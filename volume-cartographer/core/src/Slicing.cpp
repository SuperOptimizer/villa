#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"
#include "vc/core/util/Interpolation.hpp"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "z5/dataset.hxx"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>


// ============================================================================
// CacheParams — extract dataset constants once (ZYX ordering)
// ============================================================================

template<typename T>
struct CacheParams {
    int cz, cy, cx, sz, sy, sx;
    int czShift, cyShift, cxShift, czMask, cyMask, cxMask;
    int chunksZ, chunksY, chunksX;

    explicit CacheParams(z5::Dataset* ds) {
        const auto& cs = ds->defaultChunkShape();
        cz = static_cast<int>(cs[0]);
        cy = static_cast<int>(cs[1]);
        cx = static_cast<int>(cs[2]);
        const auto& shape = ds->shape();
        sz = static_cast<int>(shape[0]);
        sy = static_cast<int>(shape[1]);
        sx = static_cast<int>(shape[2]);
        czShift = log2_pow2(cz); cyShift = log2_pow2(cy); cxShift = log2_pow2(cx);
        czMask = cz - 1; cyMask = cy - 1; cxMask = cx - 1;
        chunksZ = (sz + cz - 1) / cz;
        chunksY = (sy + cy - 1) / cy;
        chunksX = (sx + cx - 1) / cx;
    }

    [[gnu::always_inline]] static int log2_pow2(int v) noexcept {
        int r = 0;
        while ((v >> r) > 1) r++;
        return r;
    }
};


// ============================================================================
// ChunkSampler — thread-local fast voxel access via raw pointer + strides
// ============================================================================

template<typename T>
struct ChunkSampler {
    static constexpr int kSlots = 8;
    struct Slot {
        int iz = -1, iy = -1, ix = -1;
        typename ChunkCache<T>::ChunkPtr chunk;
        const T* data = nullptr;
    };

    const CacheParams<T>& p;
    ChunkCache<T>& cache;
    z5::Dataset* ds;
    Slot slots[kSlots];
    int mru = 0;  // most-recently-used slot index
    const T* data = nullptr;  // current data pointer
    size_t s0 = 0, s1 = 0, s2 = 0;

    ChunkSampler(const CacheParams<T>& p_, ChunkCache<T>& cache_, z5::Dataset* ds_)
        : p(p_), cache(cache_), ds(ds_)
    {
        s0 = static_cast<size_t>(p.cy) * p.cx;
        s1 = static_cast<size_t>(p.cx);
        s2 = 1;
    }

    [[gnu::always_inline]] void updateChunk(int iz, int iy, int ix) {
        // Check MRU slot first (most common case)
        auto& m = slots[mru];
        if (m.iz == iz && m.iy == iy && m.ix == ix) [[likely]] {
            data = m.data;
            return;
        }
        // Linear scan remaining slots
        for (int i = 0; i < kSlots; i++) {
            if (i == mru) continue;
            auto& s = slots[i];
            if (s.iz == iz && s.iy == iy && s.ix == ix) {
                mru = i;
                data = s.data;
                return;
            }
        }
        // Miss: evict LRU (slot furthest from mru in ring)
        int victim = (mru + 1) % kSlots;
        auto& v = slots[victim];
        v.chunk = cache.get(ds, iz, iy, ix);
        v.iz = iz; v.iy = iy; v.ix = ix;
        v.data = v.chunk ? v.chunk->data() : nullptr;
        mru = victim;
        data = v.data;
    }

    [[gnu::always_inline]] bool inBounds(float vz, float vy, float vx) const noexcept {
        return vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx;
    }

    [[gnu::always_inline]] T sampleNearest(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz + 0.5f);
        int iy = static_cast<int>(vy + 0.5f);
        int ix = static_cast<int>(vx + 0.5f);
        if (iz >= p.sz) [[unlikely]] iz = p.sz - 1;
        if (iy >= p.sy) [[unlikely]] iy = p.sy - 1;
        if (ix >= p.sx) [[unlikely]] ix = p.sx - 1;

        updateChunk(iz >> p.czShift, iy >> p.cyShift, ix >> p.cxShift);
        if (!data) [[unlikely]] return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask) * s2];
    }

    [[gnu::always_inline]] T sampleInt(int iz, int iy, int ix) {
        // Use unsigned comparison trick: (unsigned)(x) >= N catches both x<0 and x>=N
        if (static_cast<unsigned>(iz) >= static_cast<unsigned>(p.sz) ||
            static_cast<unsigned>(iy) >= static_cast<unsigned>(p.sy) ||
            static_cast<unsigned>(ix) >= static_cast<unsigned>(p.sx)) [[unlikely]]
            return 0;

        updateChunk(iz >> p.czShift, iy >> p.cyShift, ix >> p.cxShift);
        if (!data) [[unlikely]] return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask) * s2];
    }

    // FMA-friendly form: a + t*(b-a) instead of (1-t)*a + t*b
    [[gnu::always_inline]] float sampleTrilinear(float vz, float vy, float vx) {
        const int iz = static_cast<int>(vz);
        const int iy = static_cast<int>(vy);
        const int ix = static_cast<int>(vx);

        const float c000 = sampleInt(iz, iy, ix);
        const float c001 = sampleInt(iz, iy, ix + 1);
        const float c010 = sampleInt(iz, iy + 1, ix);
        const float c011 = sampleInt(iz, iy + 1, ix + 1);
        const float c100 = sampleInt(iz + 1, iy, ix);
        const float c101 = sampleInt(iz + 1, iy, ix + 1);
        const float c110 = sampleInt(iz + 1, iy + 1, ix);
        const float c111 = sampleInt(iz + 1, iy + 1, ix + 1);

        const float fx = vx - ix;
        const float fy = vy - iy;
        const float fz = vz - iz;

        // FMA-friendly lerp: a + t*(b-a)
        const float c00 = c000 + fx * (c001 - c000);
        const float c01 = c010 + fx * (c011 - c010);
        const float c10 = c100 + fx * (c101 - c100);
        const float c11 = c110 + fx * (c111 - c110);

        const float c0 = c00 + fy * (c01 - c00);
        const float c1 = c10 + fy * (c11 - c10);

        return c0 + fz * (c1 - c0);
    }

    // Fast path: when all 8 trilinear corners are in the same chunk
    // Uses FMA-friendly form: a + t*(b-a) instead of (1-t)*a + t*b
    [[gnu::always_inline]] float sampleTrilinearFast(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz);
        int iy = static_cast<int>(vy);
        int ix = static_cast<int>(vx);

        if (iz < 0 || iy < 0 || ix < 0 ||
            iz + 1 >= p.sz || iy + 1 >= p.sy || ix + 1 >= p.sx) [[unlikely]]
            return sampleTrilinear(vz, vy, vx);

        int ciz0 = iz >> p.czShift, ciz1 = (iz + 1) >> p.czShift;
        int ciy0 = iy >> p.cyShift, ciy1 = (iy + 1) >> p.cyShift;
        int cix0 = ix >> p.cxShift, cix1 = (ix + 1) >> p.cxShift;

        if (ciz0 == ciz1 && ciy0 == ciy1 && cix0 == cix1) [[likely]] {
            updateChunk(ciz0, ciy0, cix0);
            if (!data) [[unlikely]] return 0;

            const int lz0 = iz & p.czMask, ly0 = iy & p.cyMask, lx0 = ix & p.cxMask;
            const int lz1 = lz0 + 1, ly1 = ly0 + 1, lx1 = lx0 + 1;

            // Pre-compute base offsets to help compiler optimize
            const size_t base_z0y0 = lz0*s0 + ly0*s1;
            const size_t base_z1y0 = lz1*s0 + ly0*s1;
            const size_t base_z0y1 = lz0*s0 + ly1*s1;
            const size_t base_z1y1 = lz1*s0 + ly1*s1;

            // Load all 8 corners with optimized address calculation
            const float c000 = data[base_z0y0 + lx0];
            const float c001 = data[base_z0y0 + lx1];
            const float c010 = data[base_z0y1 + lx0];
            const float c011 = data[base_z0y1 + lx1];
            const float c100 = data[base_z1y0 + lx0];
            const float c101 = data[base_z1y0 + lx1];
            const float c110 = data[base_z1y1 + lx0];
            const float c111 = data[base_z1y1 + lx1];

            const float fx = vx - ix;
            const float fy = vy - iy;
            const float fz = vz - iz;

            // FMA-friendly form: a + t*(b-a) generates fmadd instructions
            const float c00 = c000 + fx * (c001 - c000);
            const float c01 = c010 + fx * (c011 - c010);
            const float c10 = c100 + fx * (c101 - c100);
            const float c11 = c110 + fx * (c111 - c110);

            const float c0 = c00 + fy * (c01 - c00);
            const float c1 = c10 + fy * (c11 - c10);

            return c0 + fz * (c1 - c0);
        }

        return sampleTrilinear(vz, vy, vx);
    }
};


// ============================================================================
// readVolumeImpl — unified inner loop
// ============================================================================

enum class SampleMode { Nearest, Trilinear };

template<typename T, SampleMode Mode, typename NormalFn>
static void readVolumeImpl(
    cv::Mat_<T>& out,
    z5::Dataset* ds,
    ChunkCache<T>& cache,
    const CacheParams<T>& p,
    const cv::Mat_<cv::Vec3f>& coords,
    NormalFn getNormal,
    int numLayers,
    float zStep,
    int zStart,
    const CompositeParams* params)
{
    const int h = coords.rows;
    const int w = coords.cols;

    out = cv::Mat_<T>(coords.size(), 0);

    // Phase 1: Discover needed chunks and prefetch (single-threaded).
    {
        const size_t totalChunks = static_cast<size_t>(p.chunksZ) * p.chunksY * p.chunksX;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIz = p.chunksZ, maxIz = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIx = p.chunksX, maxIx = -1;

        auto markVoxel = [&](float vz, float vy, float vx) {
            if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) return;
            int iz = static_cast<int>(vz + 0.5f);
            int iy = static_cast<int>(vy + 0.5f);
            int ix = static_cast<int>(vx + 0.5f);
            if (iz >= p.sz) iz = p.sz - 1;
            if (iy >= p.sy) iy = p.sy - 1;
            if (ix >= p.sx) ix = p.sx - 1;
            int ciz = iz >> p.czShift;
            int ciy = iy >> p.cyShift;
            int cix = ix >> p.cxShift;
            size_t idx = ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix;
            if (!needed[idx]) {
                needed[idx] = 1;
                minIz = std::min(minIz, ciz); maxIz = std::max(maxIz, ciz);
                minIy = std::min(minIy, ciy); maxIy = std::max(maxIy, ciy);
                minIx = std::min(minIx, cix); maxIx = std::max(maxIx, cix);
            }
        };

        if (numLayers == 1) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float vz = coords(y,x)[2], vy = coords(y,x)[1], vx = coords(y,x)[0];
                    if constexpr (Mode == SampleMode::Trilinear) {
                        if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) continue;
                        int iz = static_cast<int>(vz), iy = static_cast<int>(vy), ix = static_cast<int>(vx);
                        for (int dz = 0; dz <= 1; dz++)
                            for (int dy = 0; dy <= 1; dy++)
                                for (int dx = 0; dx <= 1; dx++)
                                    markVoxel(float(iz+dz), float(iy+dy), float(ix+dx));
                    } else {
                        markVoxel(vz, vy, vx);
                    }
                }
            }
        } else {
            std::vector<float> layerOffsets(numLayers);
            for (int l = 0; l < numLayers; l++)
                layerOffsets[l] = (zStart + l) * zStep;

            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    const cv::Vec3f& bc = coords(y, x);
                    float bz = bc[2], by = bc[1], bx = bc[0];
                    if (!(bz >= 0 && by >= 0 && bx >= 0 && bz < p.sz && by < p.sy)) continue;
                    cv::Vec3f n = getNormal(y, x);
                    for (int l = 0; l < numLayers; l++) {
                        float off = layerOffsets[l];
                        markVoxel(bz + n[2]*off, by + n[1]*off, bx + n[0]*off);
                    }
                }
            }
        }

        // Collect needed chunk indices for parallel loading
        std::vector<std::array<int,3>> neededChunks;
        for (int cix = minIx; cix <= maxIx; cix++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int ciz = minIz; ciz <= maxIz; ciz++)
                    if (needed[ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededChunks.push_back({ciz, ciy, cix});

        // Load chunks — check if any are uncached before spawning threads.
        bool anyMissing = false;
        for (size_t ci = 0; ci < neededChunks.size(); ci++) {
            if (!cache.getIfCached(ds, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2])) {
                anyMissing = true;
                break;
            }
        }

        if (anyMissing) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t ci = 0; ci < neededChunks.size(); ci++)
                cache.get(ds, neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]);
        }
    }

    // Phase 2: Sample (OMP parallel, all chunks already cached)
    const bool isComposite = (numLayers > 1);
    const bool isMin = params && (params->method == "min");
    const bool isMax = params && (params->method == "max");
    const bool isMean = params && (params->method == "mean");
    const bool needsLayerStorage = params && methodRequiresLayerStorage(params->method);
    const float firstLayerOffset = zStart * zStep;

    #pragma omp parallel
    {
        ChunkSampler<T> samplerNoPrefetch(p, cache, ds);
        ChunkSampler<T> samplerPrefetch(p, cache, ds);

        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        #pragma omp for collapse(2)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bc = coords(y, x);
                float base_vz = bc[2], base_vy = bc[1], base_vx = bc[0];

                if (numLayers == 1) {
                    // Single-sample path
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    if constexpr (Mode == SampleMode::Trilinear) {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        float v = samplerNoPrefetch.sampleTrilinearFast(base_vz, base_vy, base_vx);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            if (v < 0.f) v = 0.f;
                            if (v > 65535.f) v = 65535.f;
                            out(y, x) = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            out(y, x) = static_cast<T>(v);
                        }
                    } else {
                        if (!(base_vz < p.sz && base_vy < p.sy && base_vx < p.sx)) continue;
                        if ((static_cast<int>(base_vz + 0.5f) | static_cast<int>(base_vy + 0.5f) | static_cast<int>(base_vx + 0.5f)) < 0) continue;
                        out(y, x) = samplerNoPrefetch.sampleNearest(base_vz, base_vy, base_vx);
                    }
                } else {
                    // Composite path
                    if (!(base_vz >= 0 && base_vy >= 0 && base_vx >= 0)) continue;

                    cv::Vec3f n = getNormal(y, x);
                    float nz = n[2], ny = n[1], nx = n[0];

                    float acc = isMin ? 255.0f : 0.0f;
                    int validCount = 0;

                    if (needsLayerStorage) {
                        stack.validCount = 0;
                    }

                    float dz = nz * zStep;
                    float dy = ny * zStep;
                    float dx = nx * zStep;

                    float vz = base_vz + nz * firstLayerOffset;
                    float vy = base_vy + ny * firstLayerOffset;
                    float vx = base_vx + nx * firstLayerOffset;

                    for (int layer = 0; layer < numLayers; layer++) {
                        bool validSample = false;
                        float value = 0;

                        if (samplerPrefetch.inBounds(vz, vy, vx)) {
                            uint8_t raw = samplerPrefetch.sampleNearest(vz, vy, vx);
                            value = static_cast<float>(raw < params->isoCutoff ? 0 : raw);
                            validSample = true;
                        }

                        vz += dz; vy += dy; vx += dx;

                        if (validSample) {
                            if (needsLayerStorage) {
                                stack.values[stack.validCount++] = value;
                            } else if (isMax) {
                                acc = value > acc ? value : acc;
                                validCount++;
                            } else if (isMin) {
                                acc = value < acc ? value : acc;
                                validCount++;
                            } else {
                                acc += value;
                                validCount++;
                            }
                        }
                    }

                    float result = 0.0f;
                    if (needsLayerStorage) {
                        result = compositeLayerStack(stack, *params);
                    } else if (isMax || isMin) {
                        result = acc;
                    } else if (isMean && validCount > 0) {
                        result = acc / static_cast<float>(validCount);
                    }

                    if (params->lightingEnabled) {
                        cv::Vec3f ln = getNormal(y, x);
                        float lightFactor = computeLightingFactor(ln, *params);
                        result *= lightFactor;
                    }

                    out(y, x) = static_cast<T>(std::max(0.0f, std::min(255.0f, result)));
                }
            }
        }
    }
}


// ============================================================================
// readArea3DImpl
// ============================================================================

template<typename T>
static void readArea3DImpl(xt::xtensor<T, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<T>* cache) {

    CacheParams<T> p(ds);

    cv::Vec3i size = {static_cast<int>(out.shape()[0]), static_cast<int>(out.shape()[1]), static_cast<int>(out.shape()[2])};
    cv::Vec3i to = offset + size;

    // Step 1: List all required chunks
    std::vector<cv::Vec3i> chunks_to_process;
    cv::Vec3i start_chunk = {offset[0] / p.cz, offset[1] / p.cy, offset[2] / p.cx};
    cv::Vec3i end_chunk = {(to[0] - 1) / p.cz, (to[1] - 1) / p.cy, (to[2] - 1) / p.cx};

    for (int cz = start_chunk[0]; cz <= end_chunk[0]; ++cz) {
        for (int cy = start_chunk[1]; cy <= end_chunk[1]; ++cy) {
            for (int cx = start_chunk[2]; cx <= end_chunk[2]; ++cx) {
                chunks_to_process.push_back({cz, cy, cx});
            }
        }
    }

    // Pre-compute output strides for column-major layout
    // Column-major: element (z, y, x) at z + y * out_sz + x * out_sz * out_sy
    const size_t out_sz = static_cast<size_t>(size[0]);
    const size_t out_sy = static_cast<size_t>(size[1]);
    const size_t out_stride_y = out_sz;
    const size_t out_stride_x = out_sz * out_sy;
    T* __restrict__ out_data = out.data();

    // Pre-compute chunk strides for row-major layout
    // Row-major: element (lz, ly, lx) at lz * cy * cx + ly * cx + lx
    const size_t chunk_stride_z = static_cast<size_t>(p.cy) * p.cx;
    const size_t chunk_stride_y = static_cast<size_t>(p.cx);

    // Step 2 & 3: Load and copy chunks (no inner OMP — called from parallel tile loop)
    for (const auto& idx : chunks_to_process) {
        int cz = idx[0], cy = idx[1], cx = idx[2];
        auto chunkPtr = cache->get(ds, cz, cy, cx);

        cv::Vec3i chunk_offset = {p.cz * cz, p.cy * cy, p.cx * cx};

        cv::Vec3i copy_from_start = {
            std::max(offset[0], chunk_offset[0]),
            std::max(offset[1], chunk_offset[1]),
            std::max(offset[2], chunk_offset[2])
        };

        cv::Vec3i copy_from_end = {
            std::min(to[0], chunk_offset[0] + p.cz),
            std::min(to[1], chunk_offset[1] + p.cy),
            std::min(to[2], chunk_offset[2] + p.cx)
        };

        // Pre-compute local offsets
        const int lz_base = copy_from_start[0] - chunk_offset[0];
        const int ly_base = copy_from_start[1] - chunk_offset[1];
        const int lx_base = copy_from_start[2] - chunk_offset[2];

        // Pre-compute output offsets
        const int oz_base = copy_from_start[0] - offset[0];
        const int oy_base = copy_from_start[1] - offset[1];
        const int ox_base = copy_from_start[2] - offset[2];

        const int nz = copy_from_end[0] - copy_from_start[0];
        const int ny = copy_from_end[1] - copy_from_start[1];
        const int nx = copy_from_end[2] - copy_from_start[2];

        if (chunkPtr) {
            const T* __restrict__ chunk_data = chunkPtr->data();

            for (int dz = 0; dz < nz; ++dz) {
                const int oz = oz_base + dz;
                const int lz = lz_base + dz;
                const size_t chunk_z_off = lz * chunk_stride_z;

                for (int dy = 0; dy < ny; ++dy) {
                    const int oy = oy_base + dy;
                    const int ly = ly_base + dy;
                    const size_t chunk_zy_off = chunk_z_off + ly * chunk_stride_y + lx_base;
                    const size_t out_zy_off = oz + oy * out_stride_y + ox_base * out_stride_x;

                    // Innermost loop with simple indexing
                    for (int dx = 0; dx < nx; ++dx) {
                        out_data[out_zy_off + dx * out_stride_x] = chunk_data[chunk_zy_off + dx];
                    }
                }
            }
        } else {
            for (int dz = 0; dz < nz; ++dz) {
                const int oz = oz_base + dz;

                for (int dy = 0; dy < ny; ++dy) {
                    const int oy = oy_base + dy;
                    const size_t out_zy_off = oz + oy * out_stride_y + ox_base * out_stride_x;

                    // Innermost loop
                    for (int dx = 0; dx < nx; ++dx) {
                        out_data[out_zy_off + dx * out_stride_x] = 0;
                    }
                }
            }
        }
    }
}

void readArea3D(xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<uint8_t>* cache) {
    readArea3DImpl(out, offset, ds, cache);
}

void readArea3D(xt::xtensor<uint16_t, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<uint16_t>* cache) {
    readArea3DImpl(out, offset, ds, cache);
}


// ============================================================================
// Public API — thin wrappers around readVolumeImpl
// ============================================================================

template<typename T>
static void readInterpolated3DImpl(cv::Mat_<T>& out, z5::Dataset* ds,
                                   const cv::Mat_<cv::Vec3f>& coords, ChunkCache<T>* cache, bool nearest_neighbor) {
    CacheParams<T> p(ds);

    if (nearest_neighbor) {
        readVolumeImpl<T, SampleMode::Nearest>(out, ds, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    } else {
        readVolumeImpl<T, SampleMode::Trilinear>(out, ds, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    }
}

void readInterpolated3D(cv::Mat_<uint8_t>& out, z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint8_t>* cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint16_t>* cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}

// InterpolationMethod overloads
void readInterpolated3D(cv::Mat_<uint8_t>& out, z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint8_t>* cache, InterpolationMethod method) {
    switch (method) {
        case InterpolationMethod::Nearest:
            readInterpolated3DImpl(out, ds, coords, cache, /*nearest_neighbor=*/true);
            break;
        case InterpolationMethod::Lanczos:
            vc::readInterpolated3DLanczos(out, ds, coords, *cache);
            break;
        case InterpolationMethod::Trilinear:
        case InterpolationMethod::Tricubic:  // Tricubic falls back to trilinear for now
        default:
            readInterpolated3DImpl(out, ds, coords, cache, /*nearest_neighbor=*/false);
            break;
    }
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint16_t>* cache, InterpolationMethod method) {
    switch (method) {
        case InterpolationMethod::Nearest:
            readInterpolated3DImpl(out, ds, coords, cache, /*nearest_neighbor=*/true);
            break;
        case InterpolationMethod::Lanczos:
            vc::readInterpolated3DLanczos(out, ds, coords, *cache);
            break;
        case InterpolationMethod::Trilinear:
        case InterpolationMethod::Tricubic:  // Tricubic falls back to trilinear for now
        default:
            readInterpolated3DImpl(out, ds, coords, cache, /*nearest_neighbor=*/false);
            break;
    }
}

void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache)
{
    CacheParams<uint8_t> p(ds);

    const bool hasNormals = !normals.empty() && normals.size() == baseCoords.size();
    const int numLayers = zEnd - zStart + 1;

    auto getNormal = [&](int y, int x) -> cv::Vec3f {
        if (hasNormals) {
            const cv::Vec3f& n = normals(y, x);
            if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                return n;
            }
        }
        return {1, 0, 0};
    };

    readVolumeImpl<uint8_t, SampleMode::Nearest>(out, ds, cache, p, baseCoords,
        getNormal, numLayers, zStep, zStart, &params);
}

void readCompositeFastConstantNormal(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache)
{
    CacheParams<uint8_t> p(ds);

    const int numLayers = zEnd - zStart + 1;

    auto getNormal = [&](int, int) -> cv::Vec3f {
        return normal;
    };

    readVolumeImpl<uint8_t, SampleMode::Nearest>(out, ds, cache, p, baseCoords,
        getNormal, numLayers, zStep, zStart, &params);
}


// ============================================================================
// readMultiSlice — bulk multi-slice trilinear sampling
// ============================================================================

template<typename T>
static void readMultiSliceImpl(
    std::vector<cv::Mat_<T>>& out,
    z5::Dataset* ds,
    ChunkCache<T>& cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    CacheParams<T> p(ds);
    const int h = basePoints.rows;
    const int w = basePoints.cols;
    const int numSlices = static_cast<int>(offsets.size());

    out.resize(numSlices);
    for (int s = 0; s < numSlices; s++)
        out[s] = cv::Mat_<T>(basePoints.size(), 0);

    if (numSlices == 0) return;

    // No prefetch — with sequential band iteration the cache hit rate is ~99.99%.
    // Just sample directly. Each OMP thread gets its own ChunkSampler with a
    // one-entry chunk cache, so the only shared-state access is cache.get()
    // on chunk boundary crossings (rare).

    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;

    #pragma omp parallel
    {
        ChunkSampler<T> sampler(p, cache, ds);

        #pragma omp for schedule(static)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bp = basePoints(y, x);
                const cv::Vec3f& sd = stepDirs(y, x);
                if (std::isnan(bp[0])) continue;

                for (int si = 0; si < numSlices; si++) {
                    float off = offsets[si];
                    float vx = bp[0] + sd[0] * off;
                    float vy = bp[1] + sd[1] * off;
                    float vz = bp[2] + sd[2] * off;

                    if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx))
                        continue;

                    float v = sampler.sampleTrilinearFast(vz, vy, vx);
                    v = std::max(0.f, std::min(maxVal, v + 0.5f));
                    out[si](y, x) = static_cast<T>(v);
                }
            }
        }
    }
}

void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImpl(out, ds, *cache, basePoints, stepDirs, offsets);
}

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets)
{
    readMultiSliceImpl(out, ds, *cache, basePoints, stepDirs, offsets);
}


// ============================================================================
// readSubarray3D — centralized z5 subarray reading
// ============================================================================
// These wrappers consolidate z5::multiarray::readSubarray template instantiations
// into a single translation unit, preventing code bloat from multiple instantiations.

#include "z5/multiarray/xtensor_access.hxx"

template<typename T>
static void readSubarray3DImpl(xt::xarray<T>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset) {
    z5::multiarray::readSubarray<T>(ds, out, offset.begin());
}

void readSubarray3D(xt::xarray<uint8_t>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset) {
    readSubarray3DImpl(out, ds, offset);
}

void readSubarray3D(xt::xarray<uint16_t>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset) {
    readSubarray3DImpl(out, ds, offset);
}

void readSubarray3D(xt::xarray<float>& out, z5::Dataset& ds, const std::vector<std::size_t>& offset) {
    readSubarray3DImpl(out, ds, offset);
}

// ============================================================================
// computeVolumeGradientsNative — compute gradients from volume at raw surface points
// ============================================================================
// Moved here from CVolumeViewerRender.cpp to consolidate z5 header dependencies.

cv::Mat_<cv::Vec3f> computeVolumeGradientsNative(
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& rawPoints,
    float dsScale)
{
    const int h = rawPoints.rows;
    const int w = rawPoints.cols;
    cv::Mat_<cv::Vec3f> gradients(h, w, cv::Vec3f(0, 0, 1));

    if (h == 0 || w == 0) return gradients;

    const auto volShape = ds->shape();
    const int volZ = static_cast<int>(volShape[0]);
    const int volY = static_cast<int>(volShape[1]);
    const int volX = static_cast<int>(volShape[2]);

    // Step 1: Find bounding box of all valid coordinates
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& c = rawPoints(y, x);
            // Skip invalid points (marked as -1, -1, -1)
            if (c[0] == -1.f) continue;

            const float cx = c[0] * dsScale;
            const float cy = c[1] * dsScale;
            const float cz = c[2] * dsScale;

            minX = std::min(minX, cx);
            minY = std::min(minY, cy);
            minZ = std::min(minZ, cz);
            maxX = std::max(maxX, cx);
            maxY = std::max(maxY, cy);
            maxZ = std::max(maxZ, cz);
        }
    }

    if (minX > maxX) return gradients;  // No valid points

    // Add padding for gradient computation (need ±1 voxel)
    const int pad = 2;
    const int bboxX0 = std::max(0, static_cast<int>(std::floor(minX)) - pad);
    const int bboxY0 = std::max(0, static_cast<int>(std::floor(minY)) - pad);
    const int bboxZ0 = std::max(0, static_cast<int>(std::floor(minZ)) - pad);
    const int bboxX1 = std::min(volX, static_cast<int>(std::ceil(maxX)) + pad + 1);
    const int bboxY1 = std::min(volY, static_cast<int>(std::ceil(maxY)) + pad + 1);
    const int bboxZ1 = std::min(volZ, static_cast<int>(std::ceil(maxZ)) + pad + 1);

    const size_t localW = static_cast<size_t>(bboxX1 - bboxX0);
    const size_t localH = static_cast<size_t>(bboxY1 - bboxY0);
    const size_t localD = static_cast<size_t>(bboxZ1 - bboxZ0);

    if (localW == 0 || localH == 0 || localD == 0) return gradients;

    // Step 2: Batch read the volume data for the bounding box
    xt::xarray<uint8_t> localVolume = xt::empty<uint8_t>({localD, localH, localW});
    std::vector<std::size_t> off = {static_cast<size_t>(bboxZ0), static_cast<size_t>(bboxY0), static_cast<size_t>(bboxX0)};
    readSubarray3D(localVolume, *ds, off);

    // Get raw pointer for faster access (xtensor stores data in row-major by default)
    const uint8_t* __restrict__ volData = localVolume.data();
    const size_t strideZ = localH * localW;
    const size_t strideY = localW;

    // Pre-convert bounds for fast clamping
    const int localWi = static_cast<int>(localW);
    const int localHi = static_cast<int>(localH);
    const int localDi = static_cast<int>(localD);

    // Step 3: Compute gradients in parallel at each raw grid point
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const cv::Vec3f& c = rawPoints(y, x);

            // Skip invalid points
            if (c[0] == -1.f) {
                gradients(y, x) = cv::Vec3f(0, 0, 1);
                continue;
            }

            // Scale coordinates to dataset space and convert to local coords
            const float cx = c[0] * dsScale - static_cast<float>(bboxX0);
            const float cy = c[1] * dsScale - static_cast<float>(bboxY0);
            const float cz = c[2] * dsScale - static_cast<float>(bboxZ0);

            // Compute integer center position with rounding
            const int icx = static_cast<int>(std::round(cx));
            const int icy = static_cast<int>(std::round(cy));
            const int icz = static_cast<int>(std::round(cz));

            // Compute clamped indices for all 6 sample points
            const int lx_p = std::min(std::max(icx + 1, 0), localWi - 1);
            const int lx_m = std::min(std::max(icx - 1, 0), localWi - 1);
            const int ly_p = std::min(std::max(icy + 1, 0), localHi - 1);
            const int ly_m = std::min(std::max(icy - 1, 0), localHi - 1);
            const int lz_p = std::min(std::max(icz + 1, 0), localDi - 1);
            const int lz_m = std::min(std::max(icz - 1, 0), localDi - 1);
            const int lx_c = std::min(std::max(icx, 0), localWi - 1);
            const int ly_c = std::min(std::max(icy, 0), localHi - 1);
            const int lz_c = std::min(std::max(icz, 0), localDi - 1);

            // Sample using pointer arithmetic (faster than operator())
            const float v_xp = static_cast<float>(volData[lz_c * strideZ + ly_c * strideY + lx_p]);
            const float v_xm = static_cast<float>(volData[lz_c * strideZ + ly_c * strideY + lx_m]);
            const float v_yp = static_cast<float>(volData[lz_c * strideZ + ly_p * strideY + lx_c]);
            const float v_ym = static_cast<float>(volData[lz_c * strideZ + ly_m * strideY + lx_c]);
            const float v_zp = static_cast<float>(volData[lz_p * strideZ + ly_c * strideY + lx_c]);
            const float v_zm = static_cast<float>(volData[lz_m * strideZ + ly_c * strideY + lx_c]);

            // Central differences (FMA-friendly: diff * 0.5 instead of diff / 2)
            const float gx = (v_xp - v_xm) * 0.5f;
            const float gy = (v_yp - v_ym) * 0.5f;
            const float gz = (v_zp - v_zm) * 0.5f;

            // Normalize to get unit normal (negative gradient points toward surface)
            const float len_sq = gx*gx + gy*gy + gz*gz;
            if (len_sq > 1e-12f) {
                const float inv_len = -1.0f / std::sqrt(len_sq);  // Negative for normal direction
                gradients(y, x) = cv::Vec3f(gx * inv_len, gy * inv_len, gz * inv_len);
            } else {
                gradients(y, x) = cv::Vec3f(0, 0, 1);
            }
        }
    }

    return gradients;
}
