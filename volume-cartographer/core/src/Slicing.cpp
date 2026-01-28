#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"

#include "vc/core/zarr/Tensor3D.hpp"
#include "vc/core/zarr/ZarrDataset.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

using namespace volcart::zarr;


// ============================================================================
// CacheParams — extract cache constants once
// ============================================================================

template<typename T>
struct CacheParams {
    int cw, ch, cd, sx, sy, sz;
    int cwShift, chShift, cdShift, cwMask, chMask, cdMask;
    int chunksX, chunksY, chunksZ;

    explicit CacheParams(ChunkCache<T>& c)
        : cw(c.chunkSizeX()), ch(c.chunkSizeY()), cd(c.chunkSizeZ()),
          sx(c.datasetSizeX()), sy(c.datasetSizeY()), sz(c.datasetSizeZ()),
          cwShift(c.chunkShiftX()), chShift(c.chunkShiftY()), cdShift(c.chunkShiftZ()),
          cwMask(c.chunkMaskX()), chMask(c.chunkMaskY()), cdMask(c.chunkMaskZ()),
          chunksX(c.chunksX()), chunksY(c.chunksY()), chunksZ(c.chunksZ()) {}
};


// ============================================================================
// ChunkSampler — thread-local fast voxel access via raw pointer + strides
// ============================================================================

template<typename T, bool PrefetchZ = false>
struct ChunkSampler {
    const CacheParams<T>& p;
    ChunkCache<T>& cache;
    int cachedIx = -1, cachedIy = -1, cachedIz = -1;
    typename ChunkCache<T>::ChunkPtr cachedChunk;  // holds chunk alive
    const T* data = nullptr;
    size_t s0 = 0, s1 = 0, s2 = 0;

    ChunkSampler(const CacheParams<T>& p_, ChunkCache<T>& cache_)
        : p(p_), cache(cache_) {}

    void updateChunk(int ix, int iy, int iz) {
        if (ix == cachedIx && iy == cachedIy && iz == cachedIz) return;
        cachedChunk = cache.get(ix, iy, iz);
        if (cachedChunk) {
            data = cachedChunk->data();
            s0 = static_cast<size_t>(p.ch) * p.cd;
            s1 = static_cast<size_t>(p.cd);
            s2 = 1;
            if constexpr (PrefetchZ) {
                if (iz + 1 < p.chunksZ) {
                    auto next = cache.getIfCached(ix, iy, iz + 1);
                    if (next) __builtin_prefetch(next->data(), 0, 1);
                }
            }
        } else {
            data = nullptr;
        }
        cachedIx = ix; cachedIy = iy; cachedIz = iz;
    }

    bool inBounds(float ox, float oy, float oz) const {
        return ox >= 0 && oy >= 0 && oz >= 0 && ox < p.sx && oy < p.sy && oz < p.sz;
    }

    T sampleNearest(float ox, float oy, float oz) {
        int iox = static_cast<int>(ox + 0.5f);
        int ioy = static_cast<int>(oy + 0.5f);
        int ioz = static_cast<int>(oz + 0.5f);
        if (iox >= p.sx) iox = p.sx - 1;
        if (ioy >= p.sy) ioy = p.sy - 1;
        if (ioz >= p.sz) ioz = p.sz - 1;

        updateChunk(iox >> p.cwShift, ioy >> p.chShift, ioz >> p.cdShift);
        if (!data) return 0;

        return data[(iox & p.cwMask) * s0 + (ioy & p.chMask) * s1 + (ioz & p.cdMask) * s2];
    }

    T sampleInt(int ox, int oy, int oz) {
        if (ox < 0 || oy < 0 || oz < 0 || ox >= p.sx || oy >= p.sy || oz >= p.sz)
            return 0;

        updateChunk(ox >> p.cwShift, oy >> p.chShift, oz >> p.cdShift);
        if (!data) return 0;

        return data[(ox & p.cwMask) * s0 + (oy & p.chMask) * s1 + (oz & p.cdMask) * s2];
    }

    float sampleTrilinear(float ox, float oy, float oz) {
        int iox = static_cast<int>(ox);
        int ioy = static_cast<int>(oy);
        int ioz = static_cast<int>(oz);

        float c000 = sampleInt(iox, ioy, ioz);
        float c100 = sampleInt(iox + 1, ioy, ioz);
        float c010 = sampleInt(iox, ioy + 1, ioz);
        float c110 = sampleInt(iox + 1, ioy + 1, ioz);
        float c001 = sampleInt(iox, ioy, ioz + 1);
        float c101 = sampleInt(iox + 1, ioy, ioz + 1);
        float c011 = sampleInt(iox, ioy + 1, ioz + 1);
        float c111 = sampleInt(iox + 1, ioy + 1, ioz + 1);

        float fx = ox - iox;
        float fy = oy - ioy;
        float fz = oz - ioz;

        float c00 = (1 - fz) * c000 + fz * c001;
        float c01 = (1 - fz) * c010 + fz * c011;
        float c10 = (1 - fz) * c100 + fz * c101;
        float c11 = (1 - fz) * c110 + fz * c111;

        float c0 = (1 - fy) * c00 + fy * c01;
        float c1 = (1 - fy) * c10 + fy * c11;

        return (1 - fx) * c0 + fx * c1;
    }

    // Fast path: when all 8 trilinear corners are in the same chunk,
    // do a single updateChunk + 8 direct array lookups (no per-sample bounds checks).
    float sampleTrilinearFast(float ox, float oy, float oz) {
        int iox = static_cast<int>(ox);
        int ioy = static_cast<int>(oy);
        int ioz = static_cast<int>(oz);

        // Bounds check entire 2x2x2 block at once
        if (iox < 0 || ioy < 0 || ioz < 0 ||
            iox + 1 >= p.sx || ioy + 1 >= p.sy || ioz + 1 >= p.sz)
            return sampleTrilinear(ox, oy, oz);

        // Check if all 8 corners are in the same chunk
        int ix0 = iox >> p.cwShift, ix1 = (iox + 1) >> p.cwShift;
        int iy0 = ioy >> p.chShift, iy1 = (ioy + 1) >> p.chShift;
        int iz0 = ioz >> p.cdShift, iz1 = (ioz + 1) >> p.cdShift;

        if (ix0 == ix1 && iy0 == iy1 && iz0 == iz1) {
            // Fast path: single chunk, no per-sample bounds checks
            updateChunk(ix0, iy0, iz0);
            if (!data) return 0;
            int lx0 = iox & p.cwMask, ly0 = ioy & p.chMask, lz0 = ioz & p.cdMask;
            int lx1 = lx0 + 1, ly1 = ly0 + 1, lz1 = lz0 + 1;

            float c000 = data[lx0*s0 + ly0*s1 + lz0*s2];
            float c100 = data[lx1*s0 + ly0*s1 + lz0*s2];
            float c010 = data[lx0*s0 + ly1*s1 + lz0*s2];
            float c110 = data[lx1*s0 + ly1*s1 + lz0*s2];
            float c001 = data[lx0*s0 + ly0*s1 + lz1*s2];
            float c101 = data[lx1*s0 + ly0*s1 + lz1*s2];
            float c011 = data[lx0*s0 + ly1*s1 + lz1*s2];
            float c111 = data[lx1*s0 + ly1*s1 + lz1*s2];

            float fx = ox - iox;
            float fy = oy - ioy;
            float fz = oz - ioz;

            float c00 = (1 - fz) * c000 + fz * c001;
            float c01 = (1 - fz) * c010 + fz * c011;
            float c10 = (1 - fz) * c100 + fz * c101;
            float c11 = (1 - fz) * c110 + fz * c111;

            float c0 = (1 - fy) * c00 + fy * c01;
            float c1 = (1 - fy) * c10 + fy * c11;

            return (1 - fx) * c0 + fx * c1;
        }

        return sampleTrilinear(ox, oy, oz);
    }
};


// ============================================================================
// readVolumeImpl — unified inner loop
// ============================================================================

enum class SampleMode { Nearest, Trilinear };

template<typename T, SampleMode Mode, typename NormalFn>
static void readVolumeImpl(
    cv::Mat_<T>& out,
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
        const size_t totalChunks = static_cast<size_t>(p.chunksX) * p.chunksY * p.chunksZ;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIx = p.chunksX, maxIx = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIz = p.chunksZ, maxIz = -1;

        auto markVoxel = [&](float ox, float oy, float oz) {
            if (!(ox >= 0 && oy >= 0 && oz >= 0 && ox < p.sx && oy < p.sy && oz < p.sz)) return;
            int iox = static_cast<int>(ox + 0.5f);
            int ioy = static_cast<int>(oy + 0.5f);
            int ioz = static_cast<int>(oz + 0.5f);
            if (iox >= p.sx) iox = p.sx - 1;
            if (ioy >= p.sy) ioy = p.sy - 1;
            if (ioz >= p.sz) ioz = p.sz - 1;
            int ix = iox >> p.cwShift;
            int iy = ioy >> p.chShift;
            int iz = ioz >> p.cdShift;
            size_t idx = ix + iy * p.chunksX + iz * p.chunksX * p.chunksY;
            if (!needed[idx]) {
                needed[idx] = 1;
                minIx = std::min(minIx, ix); maxIx = std::max(maxIx, ix);
                minIy = std::min(minIy, iy); maxIy = std::max(maxIy, iy);
                minIz = std::min(minIz, iz); maxIz = std::max(maxIz, iz);
            }
        };

        if (numLayers == 1) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    float ox = coords(y,x)[2], oy = coords(y,x)[1], oz = coords(y,x)[0];
                    if constexpr (Mode == SampleMode::Trilinear) {
                        if (!(ox >= 0 && oy >= 0 && oz >= 0 && ox < p.sx && oy < p.sy && oz < p.sz)) continue;
                        int iox = static_cast<int>(ox), ioy = static_cast<int>(oy), ioz = static_cast<int>(oz);
                        for (int dz = 0; dz <= 1; dz++)
                            for (int dy = 0; dy <= 1; dy++)
                                for (int dx = 0; dx <= 1; dx++)
                                    markVoxel(float(iox+dx), float(ioy+dy), float(ioz+dz));
                    } else {
                        markVoxel(ox, oy, oz);
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
                    float bx = bc[2], by = bc[1], bz = bc[0];
                    if (!(bx >= 0 && by >= 0 && bz >= 0 && bx < p.sx && by < p.sy)) continue;
                    cv::Vec3f n = getNormal(y, x);
                    for (int l = 0; l < numLayers; l++) {
                        float off = layerOffsets[l];
                        markVoxel(bx + n[2]*off, by + n[1]*off, bz + n[0]*off);
                    }
                }
            }
        }

        // Collect needed chunk indices for parallel loading
        std::vector<std::array<int,3>> neededChunks;
        for (int iz = minIz; iz <= maxIz; iz++)
            for (int iy = minIy; iy <= maxIy; iy++)
                for (int ix = minIx; ix <= maxIx; ix++)
                    if (needed[ix + iy * p.chunksX + iz * p.chunksX * p.chunksY])
                        neededChunks.push_back({ix, iy, iz});

        // Load chunks in parallel (fine-grained locking in ChunkCache allows this)
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t ci = 0; ci < neededChunks.size(); ci++)
            cache.get(neededChunks[ci][0], neededChunks[ci][1], neededChunks[ci][2]);
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
        ChunkSampler<T, false> samplerNoPrefetch(p, cache);
        ChunkSampler<T, true> samplerPrefetch(p, cache);

        LayerStack stack;
        if (needsLayerStorage) {
            stack.values.resize(numLayers);
        }

        #pragma omp for collapse(2)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& bc = coords(y, x);
                float base_ox = bc[2], base_oy = bc[1], base_oz = bc[0];

                if (numLayers == 1) {
                    // Single-sample path
                    if (!(base_ox >= 0 && base_oy >= 0 && base_oz >= 0)) continue;

                    if constexpr (Mode == SampleMode::Trilinear) {
                        if (!(base_ox < p.sx && base_oy < p.sy && base_oz < p.sz)) continue;
                        float v = samplerNoPrefetch.sampleTrilinearFast(base_ox, base_oy, base_oz);
                        if constexpr (std::is_same_v<T, uint16_t>) {
                            if (v < 0.f) v = 0.f;
                            if (v > 65535.f) v = 65535.f;
                            out(y, x) = static_cast<uint16_t>(v + 0.5f);
                        } else {
                            out(y, x) = static_cast<T>(v);
                        }
                    } else {
                        if (!(base_ox < p.sx && base_oy < p.sy && base_oz < p.sz)) continue;
                        if ((static_cast<int>(base_ox + 0.5f) | static_cast<int>(base_oy + 0.5f) | static_cast<int>(base_oz + 0.5f)) < 0) continue;
                        out(y, x) = samplerNoPrefetch.sampleNearest(base_ox, base_oy, base_oz);
                    }
                } else {
                    // Composite path
                    if (!(base_ox >= 0 && base_oy >= 0 && base_oz >= 0)) continue;

                    cv::Vec3f n = getNormal(y, x);
                    float nx = n[0], ny = n[1], nz = n[2];

                    float acc = isMin ? 255.0f : 0.0f;
                    int validCount = 0;

                    if (needsLayerStorage) {
                        stack.validCount = 0;
                    }

                    float dx = nz * zStep;
                    float dy = ny * zStep;
                    float dz = nx * zStep;

                    float ox = base_ox + nz * firstLayerOffset;
                    float oy = base_oy + ny * firstLayerOffset;
                    float oz = base_oz + nx * firstLayerOffset;

                    for (int layer = 0; layer < numLayers; layer++) {
                        bool validSample = false;
                        float value = 0;

                        if (samplerPrefetch.inBounds(ox, oy, oz)) {
                            uint8_t raw = samplerPrefetch.sampleNearest(ox, oy, oz);
                            value = static_cast<float>(raw < params->isoCutoff ? 0 : raw);
                            validSample = true;
                        }

                        ox += dx; oy += dy; oz += dz;

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
static void readArea3DImpl(Tensor3D<T>& out, const cv::Vec3i& offset, ZarrDataset* ds, ChunkCache<T>* cache) {

    CacheParams<T> p(*cache);

    cv::Vec3i size = {(int)out.shape()[0], (int)out.shape()[1], (int)out.shape()[2]};
    cv::Vec3i to = offset + size;

    // Step 1: List all required chunks
    std::vector<cv::Vec3i> chunks_to_process;
    cv::Vec3i start_chunk = {offset[0] / p.cw, offset[1] / p.ch, offset[2] / p.cd};
    cv::Vec3i end_chunk = {(to[0] - 1) / p.cw, (to[1] - 1) / p.ch, (to[2] - 1) / p.cd};

    for (int cz = start_chunk[0]; cz <= end_chunk[0]; ++cz) {
        for (int cy = start_chunk[1]; cy <= end_chunk[1]; ++cy) {
            for (int cx = start_chunk[2]; cx <= end_chunk[2]; ++cx) {
                chunks_to_process.push_back({cz, cy, cx});
            }
        }
    }

    // No shuffle — sequential order maximizes spatial locality for cache hits

    // Step 2 & 3: Load and copy chunks (no inner OMP — called from parallel tile loop)
    for (const auto& idx : chunks_to_process) {
        int cz = idx[0], cy = idx[1], cx = idx[2];
        auto chunkPtr = cache->get(cz, cy, cx);
        Tensor3D<T>* chunk = chunkPtr.get();

        cv::Vec3i chunk_offset = {p.cw * cz, p.ch * cy, p.cd * cx};

        cv::Vec3i copy_from_start = {
            std::max(offset[0], chunk_offset[0]),
            std::max(offset[1], chunk_offset[1]),
            std::max(offset[2], chunk_offset[2])
        };

        cv::Vec3i copy_from_end = {
            std::min(to[0], chunk_offset[0] + p.cw),
            std::min(to[1], chunk_offset[1] + p.ch),
            std::min(to[2], chunk_offset[2] + p.cd)
        };

        if (chunk) {
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        int lz = z - chunk_offset[0];
                        int ly = y - chunk_offset[1];
                        int lx = x - chunk_offset[2];
                        out(z - offset[0], y - offset[1], x - offset[2]) = (*chunk)(lz, ly, lx);
                    }
                }
            }
        } else {
            for (int z = copy_from_start[0]; z < copy_from_end[0]; ++z) {
                for (int y = copy_from_start[1]; y < copy_from_end[1]; ++y) {
                    for (int x = copy_from_start[2]; x < copy_from_end[2]; ++x) {
                        out(z - offset[0], y - offset[1], x - offset[2]) = 0;
                    }
                }
            }
        }
    }
}

void readArea3D(Tensor3D<uint8_t>& out, const cv::Vec3i& offset, ZarrDataset* ds, ChunkCache<uint8_t>* cache) {
    readArea3DImpl(out, offset, ds, cache);
}

void readArea3D(Tensor3D<uint16_t>& out, const cv::Vec3i& offset, ZarrDataset* ds, ChunkCache<uint16_t>* cache) {
    readArea3DImpl(out, offset, ds, cache);
}


// ============================================================================
// Public API — thin wrappers around readVolumeImpl
// ============================================================================

template<typename T>
static void readInterpolated3DImpl(cv::Mat_<T>& out, ZarrDataset* ds,
                                   const cv::Mat_<cv::Vec3f>& coords, ChunkCache<T>* cache, bool nearest_neighbor) {
    CacheParams<T> p(*cache);

    if (nearest_neighbor) {
        readVolumeImpl<T, SampleMode::Nearest>(out, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    } else {
        readVolumeImpl<T, SampleMode::Trilinear>(out, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    }
}

void readInterpolated3D(cv::Mat_<uint8_t>& out, ZarrDataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint8_t>* cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}

void readInterpolated3D(cv::Mat_<uint16_t>& out, ZarrDataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords, ChunkCache<uint16_t>* cache, bool nearest_neighbor) {
    readInterpolated3DImpl(out, ds, coords, cache, nearest_neighbor);
}


void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    ZarrDataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache)
{
    cache.init(ds);
    CacheParams<uint8_t> p(cache);

    const bool hasNormals = !normals.empty() && normals.size() == baseCoords.size();
    const int numLayers = zEnd - zStart + 1;

    auto getNormal = [&](int y, int x) -> cv::Vec3f {
        if (hasNormals) {
            const cv::Vec3f& n = normals(y, x);
            if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2])) {
                return n;
            }
        }
        return {0, 0, 1};
    };

    readVolumeImpl<uint8_t, SampleMode::Nearest>(out, cache, p, baseCoords,
        getNormal, numLayers, zStep, zStart, &params);
}

void readCompositeFastConstantNormal(
    cv::Mat_<uint8_t>& out,
    ZarrDataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache)
{
    cache.init(ds);
    CacheParams<uint8_t> p(cache);

    const int numLayers = zEnd - zStart + 1;

    auto getNormal = [&](int, int) -> cv::Vec3f {
        return normal;
    };

    readVolumeImpl<uint8_t, SampleMode::Nearest>(out, cache, p, baseCoords,
        getNormal, numLayers, zStep, zStart, &params);
}
