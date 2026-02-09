#pragma once

#include "SparseVolume.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace vc::simd {

// --------------------------------------------------------------------------
// ComputedVolume<T, N, ComputeFn> — compute-on-load volume
//
// Wraps a SparseVolume as the source and applies a compute functor to each
// tile on first access. Replacement for Chunked3d<T,C>.
//
// ComputeFn must provide:
//   - enum { BORDER }
//   - enum { CHUNK_SIZE }  (should match N)
//   - template <typename Arr, typename Elem>
//     void compute(const Arr& large, Arr& small, const cv::Vec3i& offset);
// --------------------------------------------------------------------------
template <typename T, int N, typename ComputeFn>
class ComputedVolume {
public:
    using Geom = TileGeometry<N>;

    ComputedVolume(ComputeFn& fn, vc::zarr::Dataset* ds, ChunkCache<T>* cache,
                   std::filesystem::path cache_root = {})
        : source_(ds, cache), fn_(fn), cache_root_(std::move(cache_root)) {}

    T& at(int z, int y, int x) {
        int tz = Geom::chunk_id(z);
        int ty = Geom::chunk_id(y);
        int tx = Geom::chunk_id(x);

        T* chunk = get_computed(tz, ty, tx);
        return chunk[Geom::offset3d(Geom::local(z), Geom::local(y), Geom::local(x))];
    }

    T& safe_at(int z, int y, int x) {
        int tz = Geom::chunk_id(z);
        int ty = Geom::chunk_id(y);
        int tx = Geom::chunk_id(x);

        T* chunk = get_computed_safe(tz, ty, tx);
        return chunk[Geom::offset3d(Geom::local(z), Geom::local(y), Geom::local(x))];
    }

    SparseVolume<T, N>& source() { return source_; }
    const SparseVolume<T, N>& source() const { return source_; }

    std::array<int, 3> shape() const { return source_.shape(); }

private:
    SparseVolume<T, N> source_;
    ComputeFn& fn_;
    std::filesystem::path cache_root_;

    mutable std::shared_mutex mutex_;
    mutable std::unordered_map<uint64_t, T*> computed_;

    static uint64_t key(int tz, int ty, int tx) {
        return (static_cast<uint64_t>(static_cast<uint16_t>(tz)) << 32) |
               (static_cast<uint64_t>(static_cast<uint16_t>(ty)) << 16) |
               static_cast<uint64_t>(static_cast<uint16_t>(tx));
    }

    T* get_computed(int tz, int ty, int tx) {
        uint64_t k = key(tz, ty, tx);

        auto it = computed_.find(k);
        if (it != computed_.end())
            return it->second;

        return compute_tile(tz, ty, tx, k);
    }

    T* get_computed_safe(int tz, int ty, int tx) {
        uint64_t k = key(tz, ty, tx);

        {
            std::shared_lock lock(mutex_);
            auto it = computed_.find(k);
            if (it != computed_.end())
                return it->second;
        }

        return compute_tile(tz, ty, tx, k);
    }

    T* compute_tile(int tz, int ty, int tx, uint64_t k) {
        constexpr size_t vol = Geom::volume3d;

        T* chunk = static_cast<T*>(std::malloc(vol * sizeof(T)));
        if (!chunk)
            throw std::bad_alloc();
        std::memset(chunk, 0, vol * sizeof(T));

        // TODO: Invoke fn_.compute() with source data when migration happens.
        // For now, just load the raw source tile data.
        const T* src = source_.tile_data(tz, ty, tx);
        if (src)
            std::memcpy(chunk, src, vol * sizeof(T));

        std::unique_lock lock(mutex_);
        auto [it, inserted] = computed_.emplace(k, chunk);
        if (!inserted) {
            // Another thread beat us
            std::free(chunk);
            return it->second;
        }
        return chunk;
    }
};

}  // namespace vc::simd
