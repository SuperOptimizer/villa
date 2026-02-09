#pragma once

#include "TileDecomposition.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace vc::simd {

namespace detail {

// Aligned memory allocation/deallocation
inline void* aligned_alloc_impl(size_t alignment, size_t size) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
}

struct AlignedDeleter {
    void operator()(void* p) const { std::free(p); }
};

template <typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter>;

template <typename T>
AlignedPtr<T> make_aligned(size_t count) {
    void* raw = aligned_alloc_impl(64, count * sizeof(T));
    if (!raw)
        throw std::bad_alloc();
    std::memset(raw, 0, count * sizeof(T));
    return AlignedPtr<T>(static_cast<T*>(raw));
}

// Find span index for a coordinate via linear scan (tile count is tiny, 2-5 per axis)
inline int find_span(const std::vector<TileSpan>& spans, int coord) {
    for (int i = static_cast<int>(spans.size()) - 1; i >= 0; --i) {
        if (coord >= spans[i].offset)
            return i;
    }
    return 0;
}

}  // namespace detail

// --------------------------------------------------------------------------
// ComposedRegion3D — arbitrary-size region backed by mixed-size aligned tiles
// --------------------------------------------------------------------------
template <typename T>
class ComposedRegion3D {
public:
    ComposedRegion3D(int sz, int sy, int sx, int min_tile_size = kMinTileSize<T>)
        : sz_(sz), sy_(sy), sx_(sx)
        , layout_(sz, sy, sx, min_tile_size, 0)
    {
        allocate_tiles();
    }

    T& operator()(int z, int y, int x) {
        int zi = detail::find_span(layout_.z_spans, z);
        int yi = detail::find_span(layout_.y_spans, y);
        int xi = detail::find_span(layout_.x_spans, x);
        auto& tb = tile_at(zi, yi, xi);
        int lz = z - layout_.z_spans[zi].offset;
        int ly = y - layout_.y_spans[yi].offset;
        int lx = x - layout_.x_spans[xi].offset;
        return tb.data[static_cast<size_t>(lz) +
                       static_cast<size_t>(ly) * tb.nz +
                       static_cast<size_t>(lx) * tb.nz * tb.ny];
    }

    const T& operator()(int z, int y, int x) const {
        int zi = detail::find_span(layout_.z_spans, z);
        int yi = detail::find_span(layout_.y_spans, y);
        int xi = detail::find_span(layout_.x_spans, x);
        const auto& tb = tile_at(zi, yi, xi);
        int lz = z - layout_.z_spans[zi].offset;
        int ly = y - layout_.y_spans[yi].offset;
        int lx = x - layout_.x_spans[xi].offset;
        return tb.data[static_cast<size_t>(lz) +
                       static_cast<size_t>(ly) * tb.nz +
                       static_cast<size_t>(lx) * tb.nz * tb.ny];
    }

    std::array<int, 3> shape() const { return {sz_, sy_, sx_}; }

    // Iterate over tiles. Callback receives:
    //   (T* data, int tile_nz, int tile_ny, int tile_nx,
    //    int offset_z, int offset_y, int offset_x)
    template <typename Fn>
    void for_each_tile(Fn&& fn) {
        int nz = static_cast<int>(layout_.z_spans.size());
        int ny = static_cast<int>(layout_.y_spans.size());
        int nx = static_cast<int>(layout_.x_spans.size());
        for (int xi = 0; xi < nx; ++xi) {
            for (int yi = 0; yi < ny; ++yi) {
                for (int zi = 0; zi < nz; ++zi) {
                    auto& tb = tile_at(zi, yi, xi);
                    fn(tb.data.get(), tb.nz, tb.ny, tb.nx,
                       layout_.z_spans[zi].offset,
                       layout_.y_spans[yi].offset,
                       layout_.x_spans[xi].offset);
                }
            }
        }
    }

    template <typename Fn>
    void for_each_tile(Fn&& fn) const {
        int nz = static_cast<int>(layout_.z_spans.size());
        int ny = static_cast<int>(layout_.y_spans.size());
        int nx = static_cast<int>(layout_.x_spans.size());
        for (int xi = 0; xi < nx; ++xi) {
            for (int yi = 0; yi < ny; ++yi) {
                for (int zi = 0; zi < nz; ++zi) {
                    const auto& tb = tile_at(zi, yi, xi);
                    fn(tb.data.get(), tb.nz, tb.ny, tb.nx,
                       layout_.z_spans[zi].offset,
                       layout_.y_spans[yi].offset,
                       layout_.x_spans[xi].offset);
                }
            }
        }
    }

    const TileLayout3D& layout() const { return layout_; }

private:
    int sz_, sy_, sx_;
    TileLayout3D layout_;

    struct TileBlock {
        detail::AlignedPtr<T> data;
        int nz, ny, nx;
    };
    std::vector<TileBlock> tiles_;

    void allocate_tiles() {
        int nz = static_cast<int>(layout_.z_spans.size());
        int ny = static_cast<int>(layout_.y_spans.size());
        int nx = static_cast<int>(layout_.x_spans.size());
        tiles_.resize(static_cast<size_t>(nz) * ny * nx);

        for (int xi = 0; xi < nx; ++xi) {
            for (int yi = 0; yi < ny; ++yi) {
                for (int zi = 0; zi < nz; ++zi) {
                    auto& tb = tile_at(zi, yi, xi);
                    tb.nz = layout_.z_spans[zi].size;
                    tb.ny = layout_.y_spans[yi].size;
                    tb.nx = layout_.x_spans[xi].size;
                    size_t vol = static_cast<size_t>(tb.nz) * tb.ny * tb.nx;
                    tb.data = detail::make_aligned<T>(vol);
                }
            }
        }
    }

    TileBlock& tile_at(int zi, int yi, int xi) {
        int nz = static_cast<int>(layout_.z_spans.size());
        int ny = static_cast<int>(layout_.y_spans.size());
        return tiles_[static_cast<size_t>(zi) +
                      static_cast<size_t>(yi) * nz +
                      static_cast<size_t>(xi) * nz * ny];
    }

    const TileBlock& tile_at(int zi, int yi, int xi) const {
        int nz = static_cast<int>(layout_.z_spans.size());
        int ny = static_cast<int>(layout_.y_spans.size());
        return tiles_[static_cast<size_t>(zi) +
                      static_cast<size_t>(yi) * nz +
                      static_cast<size_t>(xi) * nz * ny];
    }
};

// --------------------------------------------------------------------------
// ComposedRegion2D — arbitrary-size 2D region backed by mixed-size tiles
// --------------------------------------------------------------------------
template <typename T>
class ComposedRegion2D {
public:
    ComposedRegion2D(int sy, int sx, int min_tile_size = kMinTileSize<T>)
        : sy_(sy), sx_(sx)
        , layout_(sy, sx, min_tile_size, 0)
    {
        allocate_tiles();
    }

    T& operator()(int y, int x) {
        int yi = detail::find_span(layout_.y_spans, y);
        int xi = detail::find_span(layout_.x_spans, x);
        auto& tb = tile_at(yi, xi);
        int ly = y - layout_.y_spans[yi].offset;
        int lx = x - layout_.x_spans[xi].offset;
        return tb.data[static_cast<size_t>(ly) +
                       static_cast<size_t>(lx) * tb.ny];
    }

    const T& operator()(int y, int x) const {
        int yi = detail::find_span(layout_.y_spans, y);
        int xi = detail::find_span(layout_.x_spans, x);
        const auto& tb = tile_at(yi, xi);
        int ly = y - layout_.y_spans[yi].offset;
        int lx = x - layout_.x_spans[xi].offset;
        return tb.data[static_cast<size_t>(ly) +
                       static_cast<size_t>(lx) * tb.ny];
    }

    std::array<int, 2> shape() const { return {sy_, sx_}; }

    template <typename Fn>
    void for_each_tile(Fn&& fn) {
        int ny = static_cast<int>(layout_.y_spans.size());
        int nx = static_cast<int>(layout_.x_spans.size());
        for (int xi = 0; xi < nx; ++xi) {
            for (int yi = 0; yi < ny; ++yi) {
                auto& tb = tile_at(yi, xi);
                fn(tb.data.get(), tb.ny, tb.nx,
                   layout_.y_spans[yi].offset,
                   layout_.x_spans[xi].offset);
            }
        }
    }

    template <typename Fn>
    void for_each_tile(Fn&& fn) const {
        int ny = static_cast<int>(layout_.y_spans.size());
        int nx = static_cast<int>(layout_.x_spans.size());
        for (int xi = 0; xi < nx; ++xi) {
            for (int yi = 0; yi < ny; ++yi) {
                const auto& tb = tile_at(yi, xi);
                fn(tb.data.get(), tb.ny, tb.nx,
                   layout_.y_spans[yi].offset,
                   layout_.x_spans[xi].offset);
            }
        }
    }

    const TileLayout2D& layout() const { return layout_; }

private:
    int sy_, sx_;
    TileLayout2D layout_;

    struct TileBlock {
        detail::AlignedPtr<T> data;
        int ny, nx;
    };
    std::vector<TileBlock> tiles_;

    void allocate_tiles() {
        int ny = static_cast<int>(layout_.y_spans.size());
        int nx = static_cast<int>(layout_.x_spans.size());
        tiles_.resize(static_cast<size_t>(ny) * nx);

        for (int xi = 0; xi < nx; ++xi) {
            for (int yi = 0; yi < ny; ++yi) {
                auto& tb = tile_at(yi, xi);
                tb.ny = layout_.y_spans[yi].size;
                tb.nx = layout_.x_spans[xi].size;
                size_t vol = static_cast<size_t>(tb.ny) * tb.nx;
                tb.data = detail::make_aligned<T>(vol);
            }
        }
    }

    TileBlock& tile_at(int yi, int xi) {
        int ny = static_cast<int>(layout_.y_spans.size());
        return tiles_[static_cast<size_t>(yi) +
                      static_cast<size_t>(xi) * ny];
    }

    const TileBlock& tile_at(int yi, int xi) const {
        int ny = static_cast<int>(layout_.y_spans.size());
        return tiles_[static_cast<size_t>(yi) +
                      static_cast<size_t>(xi) * ny];
    }
};

}  // namespace vc::simd
