#include "vc/core/util/Interpolation.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/traits.hpp>

#include "vc/core/util/ChunkCache.hpp"
#include "z5/dataset.hxx"

namespace vc {

// ============================================================================
// CacheParams — extract dataset constants once (ZYX ordering)
// ============================================================================

template<typename T>
struct CacheParamsLanczos {
    int cz, cy, cx, sz, sy, sx;
    int czShift, cyShift, cxShift, czMask, cyMask, cxMask;
    int chunksZ, chunksY, chunksX;

    explicit CacheParamsLanczos(z5::Dataset* ds) {
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
// ChunkSamplerLanczos — thread-local fast voxel access for Lanczos interpolation
// ============================================================================

template<typename T>
struct ChunkSamplerLanczos {
    static constexpr int kSlots = 16;  // More slots for Lanczos's larger footprint

    // Pack 3 chunk indices into a single 64-bit key for fast comparison
    [[gnu::always_inline]] static uint64_t packKey(int iz, int iy, int ix) noexcept {
        return (static_cast<uint64_t>(static_cast<uint32_t>(iz)) << 42) |
               (static_cast<uint64_t>(static_cast<uint32_t>(iy)) << 21) |
               static_cast<uint64_t>(static_cast<uint32_t>(ix));
    }

    struct Slot {
        uint64_t key = ~uint64_t(0);  // Invalid key (all 1s)
        typename ChunkCache<T>::ChunkPtr chunk;
        const T* __restrict__ data = nullptr;
    };

    const CacheParamsLanczos<T>& p;
    ChunkCache<T>& cache;
    z5::Dataset* ds;
    Slot slots[kSlots];
    int mru = 0;
    size_t s0 = 0, s1 = 0, s2 = 0;

    ChunkSamplerLanczos(const CacheParamsLanczos<T>& p_, ChunkCache<T>& cache_, z5::Dataset* ds_)
        : p(p_), cache(cache_), ds(ds_)
    {
        s0 = static_cast<size_t>(p.cy) * p.cx;
        s1 = static_cast<size_t>(p.cx);
        s2 = 1;
    }

    [[gnu::always_inline]] const T* getChunkData(int ciz, int ciy, int cix) {
        const uint64_t key = packKey(ciz, ciy, cix);

        // Check MRU slot first - single 64-bit comparison
        if (slots[mru].key == key) [[likely]] {
            return slots[mru].data;
        }
        // Linear scan remaining slots
        for (int i = 0; i < kSlots; i++) {
            if (i == mru) continue;
            if (slots[i].key == key) {
                mru = i;
                return slots[i].data;
            }
        }
        // Miss: evict LRU
        int victim = (mru + 1) % kSlots;
        auto& v = slots[victim];
        v.chunk = cache.get(ds, ciz, ciy, cix);
        v.key = key;
        v.data = v.chunk ? v.chunk->data() : nullptr;
        mru = victim;
        return v.data;
    }

    [[gnu::always_inline]] T sampleInt(int iz, int iy, int ix) {
        // Use unsigned comparison trick: (unsigned)(x) >= N catches both x<0 and x>=N
        if (static_cast<unsigned>(iz) >= static_cast<unsigned>(p.sz) ||
            static_cast<unsigned>(iy) >= static_cast<unsigned>(p.sy) ||
            static_cast<unsigned>(ix) >= static_cast<unsigned>(p.sx)) [[unlikely]]
            return 0;

        int ciz = iz >> p.czShift;
        int ciy = iy >> p.cyShift;
        int cix = ix >> p.cxShift;

        const T* data = getChunkData(ciz, ciy, cix);
        if (!data) [[unlikely]] return 0;

        return data[(iz & p.czMask) * s0 + (iy & p.cyMask) * s1 + (ix & p.cxMask) * s2];
    }

    float sampleLanczos(float vz, float vy, float vx) {
        // Integer and fractional parts
        int iz = static_cast<int>(std::floor(vz));
        int iy = static_cast<int>(std::floor(vy));
        int ix = static_cast<int>(std::floor(vx));

        float fz = vz - static_cast<float>(iz);
        float fy = vy - static_cast<float>(iy);
        float fx = vx - static_cast<float>(ix);

        // Precompute Lanczos-3 weights
        Lanczos3Weights wz(fz), wy(fy), wx(fx);

        // Fast path: check if entire 6x6x6 neighborhood is in bounds and one chunk
        const int zMin = iz - 2, zMax = iz + 3;
        const int yMin = iy - 2, yMax = iy + 3;
        const int xMin = ix - 2, xMax = ix + 3;

        if (zMin >= 0 && yMin >= 0 && xMin >= 0 &&
            zMax < p.sz && yMax < p.sy && xMax < p.sx) [[likely]] {
            // Check if all corners are in the same chunk
            const int czMin = zMin >> p.czShift, czMax = zMax >> p.czShift;
            const int cyMin = yMin >> p.cyShift, cyMax = yMax >> p.cyShift;
            const int cxMin = xMin >> p.cxShift, cxMax = xMax >> p.cxShift;

            if (czMin == czMax && cyMin == cyMax && cxMin == cxMax) {
                // All 216 samples in one chunk - no per-voxel bounds checks
                const T* data = getChunkData(czMin, cyMin, cxMin);
                if (!data) [[unlikely]] return 0.0f;

                const size_t lz0 = static_cast<size_t>(zMin & p.czMask);
                const size_t ly0 = static_cast<size_t>(yMin & p.cyMask);
                const size_t lx0 = static_cast<size_t>(xMin & p.cxMask);

                float result = 0.0f;
                float weightSum = 0.0f;
                for (int dz = 0; dz < 6; ++dz) {
                    const float wZ = wz.weights[dz];
                    const size_t zOff = (lz0 + dz) * s0;
                    for (int dy = 0; dy < 6; ++dy) {
                        const float wZY = wZ * wy.weights[dy];
                        const size_t yzOff = zOff + (ly0 + dy) * s1;
                        for (int dx = 0; dx < 6; ++dx) {
                            const float w = wZY * wx.weights[dx];
                            result += w * static_cast<float>(data[yzOff + lx0 + dx]);
                            weightSum += w;
                        }
                    }
                }
                return weightSum > 0 ? result / weightSum : 0.0f;
            }
        }

        // Slow path: boundary handling required
        float result = 0.0f;
        float weightSum = 0.0f;

        for (int dz = -2; dz <= 3; ++dz) {
            int z = iz + dz;
            if (z < 0 || z >= p.sz) continue;
            float wZ = wz.weights[dz + 2];

            for (int dy = -2; dy <= 3; ++dy) {
                int y = iy + dy;
                if (y < 0 || y >= p.sy) continue;
                float wZY = wZ * wy.weights[dy + 2];

                for (int dx = -2; dx <= 3; ++dx) {
                    int x = ix + dx;
                    if (x < 0 || x >= p.sx) continue;

                    float w = wZY * wx.weights[dx + 2];
                    result += w * static_cast<float>(sampleInt(z, y, x));
                    weightSum += w;
                }
            }
        }

        // Renormalize for edge cases
        if (weightSum > 0.0f && weightSum < 0.99f) {
            result /= weightSum;
        }

        return result;
    }
};

// ============================================================================
// readInterpolated3DLanczos implementation
// ============================================================================

template<typename T>
void readInterpolated3DLanczosImpl(cv::Mat_<T>& out, z5::Dataset* ds,
                                    const cv::Mat_<cv::Vec3f>& coords,
                                    ChunkCache<T>& cache) {
    CacheParamsLanczos<T> p(ds);
    const int h = coords.rows;
    const int w = coords.cols;

    out = cv::Mat_<T>(coords.size(), 0);

    // Phase 1: Discover needed chunks for Lanczos (larger footprint than trilinear)
    {
        const size_t totalChunks = static_cast<size_t>(p.chunksZ) * p.chunksY * p.chunksX;
        std::vector<uint8_t> needed(totalChunks, 0);
        int minIz = p.chunksZ, maxIz = -1;
        int minIy = p.chunksY, maxIy = -1;
        int minIx = p.chunksX, maxIx = -1;

        auto markVoxel = [&](int iz, int iy, int ix) {
            if (iz < 0 || iy < 0 || ix < 0 || iz >= p.sz || iy >= p.sy || ix >= p.sx) return;
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

        // Lanczos-3 footprint: [-2, 3] in each dimension = 6 samples
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float vz = coords(y, x)[2], vy = coords(y, x)[1], vx = coords(y, x)[0];
                if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx)) continue;

                int iz = static_cast<int>(vz);
                int iy = static_cast<int>(vy);
                int ix = static_cast<int>(vx);

                // Mark all voxels in Lanczos footprint
                for (int dz = -2; dz <= 3; dz++)
                    for (int dy = -2; dy <= 3; dy++)
                        for (int dx = -2; dx <= 3; dx++)
                            markVoxel(iz + dz, iy + dy, ix + dx);
            }
        }

        // Collect needed chunks
        std::vector<std::array<int, 3>> neededChunks;
        neededChunks.reserve(static_cast<size_t>(maxIx - minIx + 1) * (maxIy - minIy + 1) * (maxIz - minIz + 1));
        for (int cix = minIx; cix <= maxIx; cix++)
            for (int ciy = minIy; ciy <= maxIy; ciy++)
                for (int ciz = minIz; ciz <= maxIz; ciz++)
                    if (needed[ciz * p.chunksY * p.chunksX + ciy * p.chunksX + cix])
                        neededChunks.push_back({ciz, ciy, cix});

        // Parallel prefetch
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

    // Phase 2: Sample using Lanczos
    constexpr float maxVal = std::is_same_v<T, uint16_t> ? 65535.f : 255.f;

    #pragma omp parallel
    {
        ChunkSamplerLanczos<T> sampler(p, cache, ds);

        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                const cv::Vec3f& c = coords(y, x);
                float vz = c[2], vy = c[1], vx = c[0];

                if (!(vz >= 0 && vy >= 0 && vx >= 0 && vz < p.sz && vy < p.sy && vx < p.sx))
                    continue;

                float v = sampler.sampleLanczos(vz, vy, vx);
                v = std::max(0.f, std::min(maxVal, v + 0.5f));
                out(y, x) = static_cast<T>(v);
            }
        }
    }
}

// Public API wrapper
template<typename T>
void readInterpolated3DLanczos(cv::Mat_<T>& out, z5::Dataset* ds,
                                const cv::Mat_<cv::Vec3f>& coords,
                                ChunkCache<T>& cache) {
    readInterpolated3DLanczosImpl(out, ds, coords, cache);
}

// Explicit instantiations
template void readInterpolated3DLanczos<uint8_t>(cv::Mat_<uint8_t>& out, z5::Dataset* ds,
                                                  const cv::Mat_<cv::Vec3f>& coords,
                                                  ChunkCache<uint8_t>& cache);

template void readInterpolated3DLanczos<uint16_t>(cv::Mat_<uint16_t>& out, z5::Dataset* ds,
                                                   const cv::Mat_<cv::Vec3f>& coords,
                                                   ChunkCache<uint16_t>& cache);

}  // namespace vc
