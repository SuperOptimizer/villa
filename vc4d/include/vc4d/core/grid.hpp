#pragma once
/// vc4d::Grid<T> — 2D grid with first-class sparsity.
///
/// Replaces cv::Mat_<cv::Vec3f> + the "-1 sentinel" pattern from vc3d.
/// In vc3d, invalid grid cells were marked by setting the first component
/// to -1.f, then every consumer had to check `p[0] != -1.f`.  That pattern
/// was fragile, couldn't distinguish "no data" from "data at x=-1", and
/// led to hundreds of special-case branches.
///
/// Grid<T> stores std::optional<T> per cell.  Absent = no data.
/// It provides range-based iteration over valid cells and valid quads
/// (2×2 neighborhoods where all four corners are present), which are the
/// two dominant access patterns in surface processing.
///
/// Memory layout: flat row-major vector of optional<T>.  For a dense grid
/// this is ~1 byte overhead per cell (the optional flag).  For very large
/// sparse grids a future version could switch to a hash map, but the
/// current surfaces are typically <10M cells so flat is fine.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace vc4d {

// ---------------------------------------------------------------------------
// PointRef / QuadRef — returned by iterators for structured bindings
// ---------------------------------------------------------------------------
template <typename T>
struct PointRef {
    int row;
    int col;
    T& value;
};

template <typename T>
struct ConstPointRef {
    int row;
    int col;
    const T& value;
};

template <typename T>
struct QuadRef {
    int row;
    int col;
    T& p00; // (row,   col)
    T& p01; // (row,   col+1)
    T& p10; // (row+1, col)
    T& p11; // (row+1, col+1)
};

template <typename T>
struct ConstQuadRef {
    int row;
    int col;
    const T& p00;
    const T& p01;
    const T& p10;
    const T& p11;
};

// ---------------------------------------------------------------------------
// Grid<T>
// ---------------------------------------------------------------------------
template <typename T>
class Grid {
public:
    Grid() = default;

    Grid(int rows, int cols)
        : rows_(rows), cols_(cols), cells_(static_cast<size_t>(rows) * cols) {}

    Grid(int rows, int cols, const T& fill)
        : rows_(rows), cols_(cols), cells_(static_cast<size_t>(rows) * cols, fill) {}

    // ---- Dimensions ---------------------------------------------------------
    [[nodiscard]] int rows() const { return rows_; }
    [[nodiscard]] int cols() const { return cols_; }
    [[nodiscard]] bool empty() const { return cells_.empty(); }

    // ---- Element access (optional-based — no sentinel values) ---------------
    [[nodiscard]] std::optional<T>& operator()(int r, int c) {
        assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return cells_[static_cast<size_t>(r) * cols_ + c];
    }
    [[nodiscard]] const std::optional<T>& operator()(int r, int c) const {
        assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return cells_[static_cast<size_t>(r) * cols_ + c];
    }

    // Set a cell (replaces the vc3d pattern of writing to cv::Mat then
    // checking for -1 everywhere).
    void set(int r, int c, const T& val) { (*this)(r, c) = val; }
    void clear(int r, int c)             { (*this)(r, c).reset(); }

    [[nodiscard]] bool has(int r, int c) const {
        return in_bounds(r, c) && cells_[static_cast<size_t>(r) * cols_ + c].has_value();
    }

    [[nodiscard]] bool in_bounds(int r, int c) const {
        return r >= 0 && r < rows_ && c >= 0 && c < cols_;
    }

    // ---- Flat access (for bulk operations) ----------------------------------
    [[nodiscard]] std::span<std::optional<T>>       data()       { return cells_; }
    [[nodiscard]] std::span<const std::optional<T>> data() const { return cells_; }

    // ---- Counting -----------------------------------------------------------
    [[nodiscard]] int count_valid() const {
        return static_cast<int>(
            std::ranges::count_if(cells_, [](const auto& c) { return c.has_value(); }));
    }

    // ---- Valid-point iteration ----------------------------------------------
    // Usage: for (auto [row, col, val] : grid.valid_points()) { ... }

    class ValidPointIterator {
    public:
        ValidPointIterator(Grid* g, int idx) : g_(g), idx_(idx) { advance_to_valid(); }
        auto operator*() -> PointRef<T> {
            int r = idx_ / g_->cols_;
            int c = idx_ % g_->cols_;
            return {r, c, *g_->cells_[idx_]};
        }
        ValidPointIterator& operator++() { ++idx_; advance_to_valid(); return *this; }
        bool operator!=(const ValidPointIterator& o) const { return idx_ != o.idx_; }
    private:
        void advance_to_valid() {
            int n = static_cast<int>(g_->cells_.size());
            while (idx_ < n && !g_->cells_[idx_].has_value()) ++idx_;
        }
        Grid* g_;
        int idx_;
    };

    class ConstValidPointIterator {
    public:
        ConstValidPointIterator(const Grid* g, int idx) : g_(g), idx_(idx) { advance_to_valid(); }
        auto operator*() const -> ConstPointRef<T> {
            int r = idx_ / g_->cols_;
            int c = idx_ % g_->cols_;
            return {r, c, *g_->cells_[idx_]};
        }
        ConstValidPointIterator& operator++() { ++idx_; advance_to_valid(); return *this; }
        bool operator!=(const ConstValidPointIterator& o) const { return idx_ != o.idx_; }
    private:
        void advance_to_valid() {
            int n = static_cast<int>(g_->cells_.size());
            while (idx_ < n && !g_->cells_[idx_].has_value()) ++idx_;
        }
        const Grid* g_;
        int idx_;
    };

    struct ValidPointRange {
        Grid* g;
        auto begin() { return ValidPointIterator(g, 0); }
        auto end()   { return ValidPointIterator(g, static_cast<int>(g->cells_.size())); }
    };

    struct ConstValidPointRange {
        const Grid* g;
        auto begin() const { return ConstValidPointIterator(g, 0); }
        auto end()   const { return ConstValidPointIterator(g, static_cast<int>(g->cells_.size())); }
    };

    ValidPointRange      valid_points()       { return {this}; }
    ConstValidPointRange valid_points() const { return {this}; }

    // ---- Valid-quad iteration ------------------------------------------------
    // Yields every 2×2 cell where all four corners have values.
    // This is the primary iteration pattern for mesh/surface operations.

    class ValidQuadIterator {
    public:
        ValidQuadIterator(Grid* g, int r, int c) : g_(g), r_(r), c_(c) { advance_to_valid(); }
        auto operator*() -> QuadRef<T> {
            return {r_, c_,
                    *(*g_)(r_,     c_),     *(*g_)(r_,     c_ + 1),
                    *(*g_)(r_ + 1, c_),     *(*g_)(r_ + 1, c_ + 1)};
        }
        ValidQuadIterator& operator++() { next(); advance_to_valid(); return *this; }
        bool operator!=(const ValidQuadIterator& o) const { return r_ != o.r_ || c_ != o.c_; }
    private:
        void next() {
            if (++c_ >= g_->cols_ - 1) { c_ = 0; ++r_; }
        }
        void advance_to_valid() {
            while (r_ < g_->rows_ - 1) {
                if (g_->has(r_, c_) && g_->has(r_, c_+1) &&
                    g_->has(r_+1, c_) && g_->has(r_+1, c_+1))
                    return;
                next();
            }
        }
        Grid* g_;
        int r_, c_;
    };

    class ConstValidQuadIterator {
    public:
        ConstValidQuadIterator(const Grid* g, int r, int c) : g_(g), r_(r), c_(c) { advance_to_valid(); }
        auto operator*() const -> ConstQuadRef<T> {
            return {r_, c_,
                    *(*g_)(r_,     c_),     *(*g_)(r_,     c_ + 1),
                    *(*g_)(r_ + 1, c_),     *(*g_)(r_ + 1, c_ + 1)};
        }
        ConstValidQuadIterator& operator++() { next(); advance_to_valid(); return *this; }
        bool operator!=(const ConstValidQuadIterator& o) const { return r_ != o.r_ || c_ != o.c_; }
    private:
        void next() {
            if (++c_ >= g_->cols_ - 1) { c_ = 0; ++r_; }
        }
        void advance_to_valid() {
            while (r_ < g_->rows_ - 1) {
                if (g_->has(r_, c_) && g_->has(r_, c_+1) &&
                    g_->has(r_+1, c_) && g_->has(r_+1, c_+1))
                    return;
                next();
            }
        }
        const Grid* g_;
        int r_, c_;
    };

    struct ValidQuadRange {
        Grid* g;
        auto begin() { return ValidQuadIterator(g, 0, 0); }
        auto end()   { return ValidQuadIterator(g, g->rows_ - 1, 0); }
    };

    struct ConstValidQuadRange {
        const Grid* g;
        auto begin() const { return ConstValidQuadIterator(g, 0, 0); }
        auto end()   const { return ConstValidQuadIterator(g, g->rows_ - 1, 0); }
    };

    ValidQuadRange      valid_quads()       { return {this}; }
    ConstValidQuadRange valid_quads() const { return {this}; }

private:
    int rows_{};
    int cols_{};
    std::vector<std::optional<T>> cells_;
};

} // namespace vc4d
