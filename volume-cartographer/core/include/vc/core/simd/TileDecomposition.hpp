#pragma once

#include "Platform.hpp"

#include <cassert>
#include <vector>

namespace vc::simd {

// A span along one axis: [offset, offset + size)
struct TileSpan {
    int offset;  // start position in the region
    int size;    // power-of-2 tile size (e.g. 64, 32, 8)
};

// Decompose a length into power-of-2 tiles, greedy largest-first.
// Remainder below min_tile_size gets padded up to min_tile_size.
//
// Example (min=32): decompose_axis(100) = [{0,64}, {64,32}, {96,32}]
//   -> last tile covers [96,128) but only [96,100) is real data
//
// Example (min=8):  decompose_axis(100) = [{0,64}, {64,32}, {96,8}]
//   -> last tile covers [96,104) but only [96,100) is real data
template <typename T>
inline std::vector<TileSpan> decompose_axis(int length, int min_size = kMinTileSize<T>) {
    assert(length > 0);
    assert(min_size > 0 && (min_size & (min_size - 1)) == 0);

    std::vector<TileSpan> spans;
    int pos = 0;
    int remaining = length;

    while (remaining > 0) {
        // Find largest tile size that fits and is >= min_size
        int tile_size = 0;
        for (int i = 0; i < kNumTileSizes; i++) {
            if (kTileSizes[i] <= remaining && kTileSizes[i] >= min_size) {
                tile_size = kTileSizes[i];
                break;
            }
        }
        // Remainder is smaller than any valid tile: pad up to min_size
        if (tile_size == 0)
            tile_size = min_size;

        spans.push_back({pos, tile_size});
        pos += tile_size;
        remaining -= tile_size;
    }
    return spans;
}

// Non-template overload taking explicit min_size
inline std::vector<TileSpan> decompose_axis_explicit(int length, int min_size) {
    assert(length > 0);
    assert(min_size > 0 && (min_size & (min_size - 1)) == 0);

    std::vector<TileSpan> spans;
    int pos = 0;
    int remaining = length;

    while (remaining > 0) {
        int tile_size = 0;
        for (int i = 0; i < kNumTileSizes; i++) {
            if (kTileSizes[i] <= remaining && kTileSizes[i] >= min_size) {
                tile_size = kTileSizes[i];
                break;
            }
        }
        if (tile_size == 0)
            tile_size = min_size;

        spans.push_back({pos, tile_size});
        pos += tile_size;
        remaining -= tile_size;
    }
    return spans;
}

// 3D decomposition: independent per-axis, forms a grid of mixed-size tiles
struct TileLayout3D {
    std::vector<TileSpan> z_spans, y_spans, x_spans;
    int padded_z = 0, padded_y = 0, padded_x = 0;

    TileLayout3D() = default;

    template <typename T>
    TileLayout3D(int sz, int sy, int sx, int min_size = kMinTileSize<T>)
        : z_spans(decompose_axis_explicit(sz, min_size))
        , y_spans(decompose_axis_explicit(sy, min_size))
        , x_spans(decompose_axis_explicit(sx, min_size))
    {
        padded_z = z_spans.back().offset + z_spans.back().size;
        padded_y = y_spans.back().offset + y_spans.back().size;
        padded_x = x_spans.back().offset + x_spans.back().size;
    }

    TileLayout3D(int sz, int sy, int sx, int min_size, int /*tag*/)
        : z_spans(decompose_axis_explicit(sz, min_size))
        , y_spans(decompose_axis_explicit(sy, min_size))
        , x_spans(decompose_axis_explicit(sx, min_size))
    {
        padded_z = z_spans.back().offset + z_spans.back().size;
        padded_y = y_spans.back().offset + y_spans.back().size;
        padded_x = x_spans.back().offset + x_spans.back().size;
    }

    size_t tile_count() const {
        return z_spans.size() * y_spans.size() * x_spans.size();
    }
};

// 2D decomposition
struct TileLayout2D {
    std::vector<TileSpan> y_spans, x_spans;
    int padded_y = 0, padded_x = 0;

    TileLayout2D() = default;

    template <typename T>
    TileLayout2D(int sy, int sx, int min_size = kMinTileSize<T>)
        : y_spans(decompose_axis_explicit(sy, min_size))
        , x_spans(decompose_axis_explicit(sx, min_size))
    {
        padded_y = y_spans.back().offset + y_spans.back().size;
        padded_x = x_spans.back().offset + x_spans.back().size;
    }

    TileLayout2D(int sy, int sx, int min_size, int /*tag*/)
        : y_spans(decompose_axis_explicit(sy, min_size))
        , x_spans(decompose_axis_explicit(sx, min_size))
    {
        padded_y = y_spans.back().offset + y_spans.back().size;
        padded_x = x_spans.back().offset + x_spans.back().size;
    }

    size_t tile_count() const {
        return y_spans.size() * x_spans.size();
    }
};

}  // namespace vc::simd
