#pragma once

#include "SparseVolume.hpp"

#include <algorithm>
#include <cmath>

namespace vc::simd {

// --------------------------------------------------------------------------
// Sampler<T, N> — thread-local fast access with MRU tile cache
//
// Provides nearest, integer, and trilinear sampling with compile-time
// stride computation from TileGeometry<N>. Each thread should have its
// own Sampler instance.
// --------------------------------------------------------------------------
template <typename T, int N>
class Sampler {
public:
    using Geom = TileGeometry<N>;

    explicit Sampler(SparseVolume<T, N>& vol) : vol_(vol) {}

    // Sample nearest neighbor (rounds to nearest voxel)
    T sample_nearest(float vz, float vy, float vx) {
        int iz = static_cast<int>(vz + 0.5f);
        int iy = static_cast<int>(vy + 0.5f);
        int ix = static_cast<int>(vx + 0.5f);
        auto [sz, sy, sx] = vol_.shape();
        iz = std::clamp(iz, 0, sz - 1);
        iy = std::clamp(iy, 0, sy - 1);
        ix = std::clamp(ix, 0, sx - 1);
        return sample_int(iz, iy, ix);
    }

    // Sample at integer coordinates (bounds checked)
    T sample_int(int z, int y, int x) {
        auto [sz, sy, sx] = vol_.shape();
        if (z < 0 || y < 0 || x < 0 || z >= sz || y >= sy || x >= sx)
            return T{};

        update_tile(Geom::chunk_id(z), Geom::chunk_id(y), Geom::chunk_id(x));
        if (!data_) return T{};

        return data_[Geom::offset3d(Geom::local(z), Geom::local(y), Geom::local(x))];
    }

    // Trilinear interpolation (general, handles chunk boundaries)
    float sample_trilinear(float vz, float vy, float vx) {
        int iz = static_cast<int>(std::floor(vz));
        int iy = static_cast<int>(std::floor(vy));
        int ix = static_cast<int>(std::floor(vx));

        float c000 = sample_int(iz,     iy,     ix);
        float c100 = sample_int(iz + 1, iy,     ix);
        float c010 = sample_int(iz,     iy + 1, ix);
        float c110 = sample_int(iz + 1, iy + 1, ix);
        float c001 = sample_int(iz,     iy,     ix + 1);
        float c101 = sample_int(iz + 1, iy,     ix + 1);
        float c011 = sample_int(iz,     iy + 1, ix + 1);
        float c111 = sample_int(iz + 1, iy + 1, ix + 1);

        float fz = vz - iz;
        float fy = vy - iy;
        float fx = vx - ix;

        float c00 = (1 - fx) * c000 + fx * c001;
        float c01 = (1 - fx) * c010 + fx * c011;
        float c10 = (1 - fx) * c100 + fx * c101;
        float c11 = (1 - fx) * c110 + fx * c111;

        float c0 = (1 - fy) * c00 + fy * c01;
        float c1 = (1 - fy) * c10 + fy * c11;

        return (1 - fz) * c0 + fz * c1;
    }

    // Fast trilinear: when all 8 corners are in the same tile, avoids
    // repeated tile lookups and uses compile-time strides
    float sample_trilinear_fast(float vz, float vy, float vx) {
        int iz = static_cast<int>(std::floor(vz));
        int iy = static_cast<int>(std::floor(vy));
        int ix = static_cast<int>(std::floor(vx));

        auto [sz, sy, sx] = vol_.shape();
        if (iz < 0 || iy < 0 || ix < 0 ||
            iz + 1 >= sz || iy + 1 >= sy || ix + 1 >= sx)
            return sample_trilinear(vz, vy, vx);

        // Check if all 8 corners are in the same tile
        int tz0 = Geom::chunk_id(iz), tz1 = Geom::chunk_id(iz + 1);
        int ty0 = Geom::chunk_id(iy), ty1 = Geom::chunk_id(iy + 1);
        int tx0 = Geom::chunk_id(ix), tx1 = Geom::chunk_id(ix + 1);

        if (tz0 != tz1 || ty0 != ty1 || tx0 != tx1)
            return sample_trilinear(vz, vy, vx);

        update_tile(tz0, ty0, tx0);
        if (!data_) return 0;

        int lz0 = Geom::local(iz), ly0 = Geom::local(iy), lx0 = Geom::local(ix);
        int lz1 = lz0 + 1, ly1 = ly0 + 1, lx1 = lx0 + 1;

        // Compile-time stride computation via TileGeometry
        float c000 = data_[Geom::offset3d(lz0, ly0, lx0)];
        float c100 = data_[Geom::offset3d(lz1, ly0, lx0)];
        float c010 = data_[Geom::offset3d(lz0, ly1, lx0)];
        float c110 = data_[Geom::offset3d(lz1, ly1, lx0)];
        float c001 = data_[Geom::offset3d(lz0, ly0, lx1)];
        float c101 = data_[Geom::offset3d(lz1, ly0, lx1)];
        float c011 = data_[Geom::offset3d(lz0, ly1, lx1)];
        float c111 = data_[Geom::offset3d(lz1, ly1, lx1)];

        float fz = vz - iz;
        float fy = vy - iy;
        float fx = vx - ix;

        float c00 = (1 - fx) * c000 + fx * c001;
        float c01 = (1 - fx) * c010 + fx * c011;
        float c10 = (1 - fx) * c100 + fx * c101;
        float c11 = (1 - fx) * c110 + fx * c111;

        float c0 = (1 - fy) * c00 + fy * c01;
        float c1 = (1 - fy) * c10 + fy * c11;

        return (1 - fz) * c0 + fz * c1;
    }

    // Load/switch to the tile containing chunk coords (tz, ty, tx).
    // After calling, current_data() returns the tile pointer (or nullptr).
    void update_tile(int tz, int ty, int tx) {
        // Check MRU slot first
        auto& m = slots_[mru_];
        if (m.tz == tz && m.ty == ty && m.tx == tx) {
            data_ = m.data;
            return;
        }
        // Linear scan remaining slots
        for (int i = 0; i < kSlots; i++) {
            if (i == mru_) continue;
            auto& s = slots_[i];
            if (s.tz == tz && s.ty == ty && s.tx == tx) {
                mru_ = i;
                data_ = s.data;
                return;
            }
        }
        // Miss: evict oldest slot
        int victim = (mru_ + 1) % kSlots;
        auto& v = slots_[victim];
        v.lifetime = vol_.tile(tz, ty, tx);
        v.tz = tz;
        v.ty = ty;
        v.tx = tx;
        v.data = v.lifetime ? v.lifetime->data() : nullptr;
        mru_ = victim;
        data_ = v.data;
    }

    // Current tile data pointer (valid after update_tile; nullptr if tile missing)
    const T* current_data() const { return data_; }

private:
    static constexpr int kSlots = 8;

    struct Slot {
        int tz = -1, ty = -1, tx = -1;
        const T* data = nullptr;
        typename ChunkCache<T>::ChunkPtr lifetime;
    };

    SparseVolume<T, N>& vol_;
    Slot slots_[kSlots]{};
    int mru_ = 0;
    const T* data_ = nullptr;
};

}  // namespace vc::simd
