#pragma once

#include "Tile.hpp"
#include "vc/core/util/ChunkCache.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

// Forward declaration
namespace vc::zarr {
class Dataset;
}

namespace vc::simd {

// --------------------------------------------------------------------------
// SparseVolume<T, N> — lazy-loaded chunked volume with homogeneous tiles
//
// Wraps z5::Dataset + ChunkCache<T> with compile-time tile size N for
// zero-cost stride computation. Tiles are loaded on demand from the cache.
// --------------------------------------------------------------------------
template <typename T, int N>
class SparseVolume {
public:
    using Geom = TileGeometry<N>;

    SparseVolume(vc::zarr::Dataset* ds, ChunkCache<T>* cache);

    SparseVolume(SparseVolume&& other) noexcept
        : ds_(other.ds_)
        , cache_(other.cache_)
        , shape_(other.shape_)
        , tiles_(std::move(other.tiles_))
    {}

    SparseVolume& operator=(SparseVolume&& other) noexcept {
        if (this != &other) {
            ds_ = other.ds_;
            cache_ = other.cache_;
            shape_ = other.shape_;
            tiles_ = std::move(other.tiles_);
        }
        return *this;
    }

    SparseVolume(const SparseVolume&) = delete;
    SparseVolume& operator=(const SparseVolume&) = delete;

    // Voxel access (bounds-checked, lazy-loads tile)
    T at(int z, int y, int x) {
        if (z < 0 || y < 0 || x < 0 ||
            z >= shape_[0] || y >= shape_[1] || x >= shape_[2])
            return T{};

        int tz = Geom::chunk_id(z);
        int ty = Geom::chunk_id(y);
        int tx = Geom::chunk_id(x);

        const T* td = tile_data(tz, ty, tx);
        if (!td) return T{};

        return td[Geom::offset3d(Geom::local(z), Geom::local(y), Geom::local(x))];
    }

    // Raw tile pointer (fast, no lifetime management — caller must hold cache ref)
    const T* tile_data(int tz, int ty, int tx) {
        uint64_t k = key(tz, ty, tx);

        // Fast path: shared lock lookup
        {
            std::shared_lock lock(mutex_);
            auto it = tiles_.find(k);
            if (it != tiles_.end())
                return it->second.first;
        }

        // Slow path: load
        return load_tile(tz, ty, tx);
    }

    // Lifetime-managed tile access
    typename ChunkCache<T>::ChunkPtr tile(int tz, int ty, int tx) {
        return cache_->get(ds_, tz, ty, tx);
    }

    std::array<int, 3> shape() const { return shape_; }

    std::array<int, 3> tile_grid() const {
        return {(shape_[0] + N - 1) / N,
                (shape_[1] + N - 1) / N,
                (shape_[2] + N - 1) / N};
    }

    vc::zarr::Dataset* dataset() const { return ds_; }
    ChunkCache<T>* cache() const { return cache_; }

private:
    vc::zarr::Dataset* ds_;
    ChunkCache<T>* cache_;
    std::array<int, 3> shape_{};

    mutable std::shared_mutex mutex_;
    mutable std::unordered_map<uint64_t,
        std::pair<const T*, typename ChunkCache<T>::ChunkPtr>> tiles_;

    static uint64_t key(int tz, int ty, int tx) {
        return (static_cast<uint64_t>(static_cast<uint16_t>(tz)) << 32) |
               (static_cast<uint64_t>(static_cast<uint16_t>(ty)) << 16) |
               static_cast<uint64_t>(static_cast<uint16_t>(tx));
    }

    const T* load_tile(int tz, int ty, int tx) {
        auto chunk = cache_->get(ds_, tz, ty, tx);
        if (!chunk)
            return nullptr;

        const T* ptr = chunk->data();
        uint64_t k = key(tz, ty, tx);

        std::unique_lock lock(mutex_);
        auto [it, inserted] = tiles_.emplace(k, std::make_pair(ptr, std::move(chunk)));
        return it->second.first;
    }
};

}  // namespace vc::simd
