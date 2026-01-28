#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Compositing.hpp"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include XTENSORINCLUDE(io, xio.hpp)
#include XTENSORINCLUDE(generators, xbuilder.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/multiarray/xtensor_access.hxx"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <random>


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
    const T* data = nullptr;
    size_t s0 = 0, s1 = 0, s2 = 0;

    ChunkSampler(const CacheParams<T>& p_, ChunkCache<T>& cache_)
        : p(p_), cache(cache_) {}

    void updateChunk(int ix, int iy, int iz) {
        if (ix == cachedIx && iy == cachedIy && iz == cachedIz) return;
        xt::xarray<T>* chunk = cache.get(ix, iy, iz);
        if (chunk) {
            data = chunk->data();
            const auto& strides = chunk->strides();
            s0 = strides[0]; s1 = strides[1]; s2 = strides[2];
            if constexpr (PrefetchZ) {
                if (iz + 1 < p.chunksZ) {
                    xt::xarray<T>* next = cache.getIfCached(ix, iy, iz + 1);
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

    // Initialize output
    out = cv::Mat_<T>(coords.size(), 0);

    // Phase 1: Discover needed chunks and prefetch (single-threaded).
    // All disk I/O happens here so the OMP sampling loop never touches disk.
    {
        // Bitmap of needed chunk indices
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

        // Load all needed chunks from disk (single-threaded, no mutex needed)
        for (int iz = minIz; iz <= maxIz; iz++)
            for (int iy = minIy; iy <= maxIy; iy++)
                for (int ix = minIx; ix <= maxIx; ix++)
                    if (needed[ix + iy * p.chunksX + iz * p.chunksX * p.chunksY])
                        cache.get(ix, iy, iz);
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
        // Use PrefetchZ=true for composite, false for single-sample
        // We need both instantiated; pick at runtime but the branch is constant
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
                        float v = samplerNoPrefetch.sampleTrilinear(base_ox, base_oy, base_oz);
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
static void readArea3DImpl(xt::xtensor<T, 3, xt::layout_type::column_major>& out, const cv::Vec3i& offset, z5::Dataset* ds, ChunkCache<T>* cache) {
    cache->init(ds);

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

    // Shuffle to reduce I/O contention from parallel requests
    std::shuffle(chunks_to_process.begin(), chunks_to_process.end(), std::mt19937(std::random_device()()));

    // Step 2 & 3: Combined parallel I/O and copy
    #pragma omp parallel for schedule(dynamic, 1)
    for (const auto& idx : chunks_to_process) {
        int cz = idx[0], cy = idx[1], cx = idx[2];
        xt::xarray<T>* chunk = cache->get(cx, cy, cz);

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
    cache->init(ds);
    CacheParams<T> p(*cache);

    if (nearest_neighbor) {
        readVolumeImpl<T, SampleMode::Nearest>(out, *cache, p, coords,
            [](int, int) -> cv::Vec3f { return {}; }, 1, 0.f, 0, nullptr);
    } else {
        readVolumeImpl<T, SampleMode::Trilinear>(out, *cache, p, coords,
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
    z5::Dataset* ds,
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
