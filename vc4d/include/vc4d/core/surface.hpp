#pragma once
/// vc4d::QuadSurface — Clean quad-mesh surface.
///
/// Key improvements over vc3d::QuadSurface:
///   • Uses Grid<Vec3f> instead of cv::Mat_<cv::Vec3f> + sentinel -1.
///   • No virtual inheritance — vc3d had a Surface base class with virtuals
///     for move/valid/loc/coord/normal/gen/pointTo.  In practice only
///     QuadSurface was ever instantiated.  We drop the base class.
///   • No lazy loading inside the type — loading is the caller's
///     responsibility (see io.hpp).  This eliminates the const_cast hack.
///   • No I/O methods on the type — save/load are free functions in io.hpp.
///   • Channels (mask, approval_mask, etc.) are stored in a simple map.
///   • Scale is Vec2f, not cv::Vec2f.
///   • No mutable caches with const_cast — caches are explicit members.

#include "grid.hpp"
#include "math.hpp"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace vc4d {

// ---------------------------------------------------------------------------
// Channel — a named auxiliary grid (mask, direction field, etc.)
// Replaces the unordered_map<string, cv::Mat> in vc3d with typed grids.
// ---------------------------------------------------------------------------
using ByteGrid  = Grid<uint8_t>;
using FloatGrid = Grid<float>;

// ---------------------------------------------------------------------------
// QuadSurface
// ---------------------------------------------------------------------------
class QuadSurface {
public:
    QuadSurface() = default;
    QuadSurface(Grid<Vec3f> points, Vec2f scale);

    // ---- Grid access --------------------------------------------------------
    [[nodiscard]] const Grid<Vec3f>& points() const { return points_; }
    [[nodiscard]]       Grid<Vec3f>& points()       { return points_; }

    [[nodiscard]] int rows() const { return points_.rows(); }
    [[nodiscard]] int cols() const { return points_.cols(); }

    [[nodiscard]] Vec2f scale()  const { return scale_; }
    void set_scale(Vec2f s) { scale_ = s; }

    // ---- Derived geometry (computed on demand, not cached behind const_cast) -
    [[nodiscard]] Box3f bbox() const;
    [[nodiscard]] Vec3f center() const;
    [[nodiscard]] int   count_valid_points() const { return points_.count_valid(); }

    // Normal at a grid cell (finite-difference from neighbors).
    [[nodiscard]] std::optional<Vec3f> normal_at(int row, int col) const;

    // Compute all normals into a grid (same dimensions as points).
    [[nodiscard]] Grid<Vec3f> compute_normals() const;

    // Compute validity mask (true where point exists).
    [[nodiscard]] Grid<bool> valid_mask() const;

    // ---- Point-to-surface projection ----------------------------------------
    // Find the grid location closest to a world-space target point.
    // Returns (row, col) as floats for sub-cell precision, or nullopt.
    struct ProjectionResult {
        float row;
        float col;
        float distance;
    };
    [[nodiscard]] std::optional<ProjectionResult> project(Vec3f target, float threshold = 1e6f) const;

    // ---- Coordinate generation (for sampling volumes) -----------------------
    // Generate a patch of world-space coordinates and normals centered at
    // (row, col) with the given output size and scale.
    struct Patch {
        Grid<Vec3f> coords;
        Grid<Vec3f> normals;
    };
    [[nodiscard]] Patch generate_patch(float row, float col, int out_rows, int out_cols, float sample_scale) const;

    // ---- Channels -----------------------------------------------------------
    void set_channel(const std::string& name, ByteGrid channel);
    [[nodiscard]] std::optional<ByteGrid*> channel(const std::string& name);
    [[nodiscard]] std::optional<const ByteGrid*> channel(const std::string& name) const;
    [[nodiscard]] std::vector<std::string> channel_names() const;

    // ---- Transforms ---------------------------------------------------------
    void rotate(float angle_deg);
    void resample(float factor_x, float factor_y);
    void flip_u();  // Reverse rows
    void flip_v();  // Reverse columns

    // ---- Overlapping surface tracking (by ID) -------------------------------
    [[nodiscard]] const std::set<std::string>& overlapping_ids() const { return overlapping_ids_; }
    void add_overlapping(const std::string& id)    { overlapping_ids_.insert(id); }
    void remove_overlapping(const std::string& id) { overlapping_ids_.erase(id); }

    // ---- Metadata -----------------------------------------------------------
    std::string id;
    std::string name;

private:
    Grid<Vec3f> points_;
    Vec2f scale_{1.f, 1.f};
    std::map<std::string, ByteGrid> channels_;
    std::set<std::string> overlapping_ids_;
};

// ---------------------------------------------------------------------------
// Free functions for surface operations (composable, not methods)
// ---------------------------------------------------------------------------

// Set difference: points in a but not within tolerance of b.
[[nodiscard]] QuadSurface surface_diff(const QuadSurface& a, const QuadSurface& b, float tolerance = 2.f);

// Union: all points from both surfaces.
[[nodiscard]] QuadSurface surface_union(const QuadSurface& a, const QuadSurface& b, float tolerance = 2.f);

// Intersection: points in a that are within tolerance of b.
[[nodiscard]] QuadSurface surface_intersection(const QuadSurface& a, const QuadSurface& b, float tolerance = 2.f);

// Test if two surfaces overlap in 3D space.
[[nodiscard]] bool surfaces_overlap(const QuadSurface& a, const QuadSurface& b);

} // namespace vc4d
