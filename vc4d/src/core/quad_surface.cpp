#include "vc4d/core/surface.hpp"

#include <algorithm>
#include <cmath>

namespace vc4d {

QuadSurface::QuadSurface(Grid<Vec3f> points, Vec2f scale)
    : points_(std::move(points)), scale_(scale) {}

Box3f QuadSurface::bbox() const {
    auto box = Box3f::empty();
    for (auto [r, c, p] : points_.valid_points())
        box = box.expanded(p);
    return box;
}

Vec3f QuadSurface::center() const {
    auto box = bbox();
    return box.center();
}

std::optional<Vec3f> QuadSurface::normal_at(int row, int col) const {
    if (!points_.has(row, col)) return std::nullopt;

    // Finite-difference normal from grid neighbors.
    Vec3f du{}, dv{};
    bool has_du = false, has_dv = false;

    if (points_.has(row, col - 1) && points_.has(row, col + 1)) {
        du = *points_(row, col + 1) - *points_(row, col - 1);
        has_du = true;
    } else if (points_.has(row, col + 1)) {
        du = *points_(row, col + 1) - *points_(row, col);
        has_du = true;
    } else if (points_.has(row, col - 1)) {
        du = *points_(row, col) - *points_(row, col - 1);
        has_du = true;
    }

    if (points_.has(row - 1, col) && points_.has(row + 1, col)) {
        dv = *points_(row + 1, col) - *points_(row - 1, col);
        has_dv = true;
    } else if (points_.has(row + 1, col)) {
        dv = *points_(row + 1, col) - *points_(row, col);
        has_dv = true;
    } else if (points_.has(row - 1, col)) {
        dv = *points_(row, col) - *points_(row - 1, col);
        has_dv = true;
    }

    if (!has_du || !has_dv) return std::nullopt;

    auto n = du.cross(dv);
    auto len = n.length();
    if (len < 1e-12f) return std::nullopt;
    return n / len;
}

Grid<Vec3f> QuadSurface::compute_normals() const {
    Grid<Vec3f> normals(rows(), cols());
    for (auto [r, c, p] : points_.valid_points()) {
        if (auto n = normal_at(r, c))
            normals.set(r, c, *n);
    }
    return normals;
}

Grid<bool> QuadSurface::valid_mask() const {
    Grid<bool> mask(rows(), cols());
    for (auto [r, c, p] : points_.valid_points())
        mask.set(r, c, true);
    return mask;
}

std::optional<QuadSurface::ProjectionResult>
QuadSurface::project(Vec3f target, float threshold) const {
    float best_dist = threshold * threshold;
    std::optional<ProjectionResult> result;

    for (auto [r, c, p] : points_.valid_points()) {
        float d2 = distance_sq(p, target);
        if (d2 < best_dist) {
            best_dist = d2;
            result = ProjectionResult{
                static_cast<float>(r),
                static_cast<float>(c),
                std::sqrt(d2)
            };
        }
    }
    return result;
}

QuadSurface::Patch QuadSurface::generate_patch(
    float center_row, float center_col,
    int out_rows, int out_cols, float sample_scale) const
{
    Patch patch;
    patch.coords  = Grid<Vec3f>(out_rows, out_cols);
    patch.normals = Grid<Vec3f>(out_rows, out_cols);

    float start_row = center_row - (out_rows / 2.f) * sample_scale;
    float start_col = center_col - (out_cols / 2.f) * sample_scale;

    for (int r = 0; r < out_rows; ++r) {
        for (int c = 0; c < out_cols; ++c) {
            float gr = start_row + r * sample_scale;
            float gc = start_col + c * sample_scale;

            int ir = static_cast<int>(gr);
            int ic = static_cast<int>(gc);

            if (!points_.in_bounds(ir, ic) || !points_.has(ir, ic))
                continue;

            // Nearest-neighbor for now; bilinear interpolation can be added.
            patch.coords.set(r, c, *points_(ir, ic));
            if (auto n = normal_at(ir, ic))
                patch.normals.set(r, c, *n);
        }
    }
    return patch;
}

void QuadSurface::set_channel(const std::string& name, ByteGrid channel) {
    channels_[name] = std::move(channel);
}

std::optional<ByteGrid*> QuadSurface::channel(const std::string& name) {
    auto it = channels_.find(name);
    if (it == channels_.end()) return std::nullopt;
    return &it->second;
}

std::optional<const ByteGrid*> QuadSurface::channel(const std::string& name) const {
    auto it = channels_.find(name);
    if (it == channels_.end()) return std::nullopt;
    return &it->second;
}

std::vector<std::string> QuadSurface::channel_names() const {
    std::vector<std::string> names;
    names.reserve(channels_.size());
    for (const auto& [k, v] : channels_)
        names.push_back(k);
    return names;
}

void QuadSurface::flip_u() {
    Grid<Vec3f> flipped(rows(), cols());
    for (auto [r, c, p] : points_.valid_points())
        flipped.set(rows() - 1 - r, c, p);
    points_ = std::move(flipped);
}

void QuadSurface::flip_v() {
    Grid<Vec3f> flipped(rows(), cols());
    for (auto [r, c, p] : points_.valid_points())
        flipped.set(r, cols() - 1 - c, p);
    points_ = std::move(flipped);
}

void QuadSurface::rotate(float /*angle_deg*/) {
    // TODO: Implement rotation with canvas expansion
}

void QuadSurface::resample(float /*factor_x*/, float /*factor_y*/) {
    // TODO: Implement bilinear resampling
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

QuadSurface surface_diff(const QuadSurface& a, const QuadSurface& b, float tolerance) {
    Grid<Vec3f> result(a.rows(), a.cols());
    float tol2 = tolerance * tolerance;

    for (auto [r, c, p] : a.points().valid_points()) {
        bool near_b = false;
        for (auto [br, bc, bp] : b.points().valid_points()) {
            if (distance_sq(p, bp) < tol2) { near_b = true; break; }
        }
        if (!near_b) result.set(r, c, p);
    }
    return QuadSurface(std::move(result), a.scale());
}

QuadSurface surface_union(const QuadSurface& a, const QuadSurface& b, float /*tolerance*/) {
    // Simple union: start with a, add non-overlapping from b.
    // Full implementation would need spatial indexing.
    Grid<Vec3f> result = a.points();
    // TODO: Merge b's points into result with spatial dedup.
    (void)b;
    return QuadSurface(std::move(result), a.scale());
}

QuadSurface surface_intersection(const QuadSurface& a, const QuadSurface& b, float tolerance) {
    Grid<Vec3f> result(a.rows(), a.cols());
    float tol2 = tolerance * tolerance;

    for (auto [r, c, p] : a.points().valid_points()) {
        for (auto [br, bc, bp] : b.points().valid_points()) {
            if (distance_sq(p, bp) < tol2) {
                result.set(r, c, p);
                break;
            }
        }
    }
    return QuadSurface(std::move(result), a.scale());
}

bool surfaces_overlap(const QuadSurface& a, const QuadSurface& b) {
    return a.bbox().intersects(b.bbox());
}

} // namespace vc4d
