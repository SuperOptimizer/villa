#include "vc/core/util/GeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vc::utils {

// ============================================================================
// Vector operations
// ============================================================================

cv::Vec3f normed(const cv::Vec3f& v)
{
    return v/std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

cv::Vec2f vmin(const cv::Vec2f& a, const cv::Vec2f& b)
{
    return {std::min(a[0],b[0]), std::min(a[1],b[1])};
}

cv::Vec2f vmax(const cv::Vec2f& a, const cv::Vec2f& b)
{
    return {std::max(a[0],b[0]), std::max(a[1],b[1])};
}

// ============================================================================
// Distance functions
// ============================================================================

float sdist(const cv::Vec3f& a, const cv::Vec3f& b)
{
    cv::Vec3f d = a - b;
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

float tdist(const cv::Vec3f& a, const cv::Vec3f& b, float td)
{
    cv::Vec3f d = a - b;
    float l = std::sqrt(d.dot(d));
    return std::abs(l - td);
}

float tdist_sum(const cv::Vec3f& p, const std::vector<cv::Vec3f>& tgts, const std::vector<float>& tds)
{
    float sum = 0;
    for(size_t i = 0; i < tgts.size(); i++) {
        float d = tdist(p, tgts[i], tds[i]);
        sum += d*d;
    }
    return sum;
}

// ============================================================================
// Bilinear interpolation template implementations
// ============================================================================

template <typename E>
E at_int(const cv::Mat_<E>& points, cv::Vec2f p)
{
    int x = static_cast<int>(p[0]);
    int y = static_cast<int>(p[1]);
    float fx = p[0] - x;
    float fy = p[1] - y;

    E p00 = points(y, x);
    E p01 = points(y, x+1);
    E p10 = points(y+1, x);
    E p11 = points(y+1, x+1);

    E p0 = (1-fx)*p00 + fx*p01;
    E p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

// Explicit template instantiations for common types
template cv::Vec3f at_int<cv::Vec3f>(const cv::Mat_<cv::Vec3f>&, cv::Vec2f);
template cv::Vec3d at_int<cv::Vec3d>(const cv::Mat_<cv::Vec3d>&, cv::Vec2f);
template cv::Vec2f at_int<cv::Vec2f>(const cv::Mat_<cv::Vec2f>&, cv::Vec2f);
template float at_int<float>(const cv::Mat_<float>&, cv::Vec2f);

// ============================================================================
// Location validation
// ============================================================================

template<typename T, int C>
bool loc_valid(const cv::Mat_<cv::Vec<T,C>>& m, const cv::Vec2d& l)
{
    if (l[0] == -1)
        return false;

    cv::Rect bounds = {0, 0, m.rows-2, m.cols-2};
    cv::Vec2i li = {static_cast<int>(std::floor(l[0])), static_cast<int>(std::floor(l[1]))};

    if (!bounds.contains(cv::Point(li)))
        return false;

    if (m(li[0], li[1])[0] == -1)
        return false;
    if (m(li[0]+1, li[1])[0] == -1)
        return false;
    if (m(li[0], li[1]+1)[0] == -1)
        return false;
    if (m(li[0]+1, li[1]+1)[0] == -1)
        return false;
    return true;
}

template<typename T, int C>
bool loc_valid_xy(const cv::Mat_<cv::Vec<T,C>>& m, const cv::Vec2d& l)
{
    return loc_valid(m, {l[1], l[0]});
}

// Explicit template instantiations
template bool loc_valid<float, 3>(const cv::Mat_<cv::Vec3f>&, const cv::Vec2d&);
template bool loc_valid<double, 3>(const cv::Mat_<cv::Vec3d>&, const cv::Vec2d&);
template bool loc_valid_xy<float, 3>(const cv::Mat_<cv::Vec3f>&, const cv::Vec2d&);
template bool loc_valid_xy<double, 3>(const cv::Mat_<cv::Vec3d>&, const cv::Vec2d&);
template bool loc_valid<float, 2>(const cv::Mat_<cv::Vec2f>&, const cv::Vec2d&);
template bool loc_valid_xy<float, 2>(const cv::Mat_<cv::Vec2f>&, const cv::Vec2d&);

bool loc_valid_nan_xy(const cv::Mat_<cv::Vec3f>& m, const cv::Vec2f& l)
{
    if (l[0] < 0 || l[1] < 0)
        return false;
    return !std::isnan(m(static_cast<int>(l[1]), static_cast<int>(l[0]))[0]);
}

// ============================================================================
// Search and optimization
// ============================================================================

void min_loc(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f& loc, cv::Vec3f& out,
             const cv::Vec3f &tgt, bool z_search)
{
    cv::Rect boundary(1, 1, points.cols-2, points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        loc = {-1,-1};
        return;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);

    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};

    float step = 1.0;

    while (changed) {
        changed = false;

        for(auto& off : search) {
            cv::Vec2f cand = loc + off*step;

            if (!boundary.contains(cv::Point(cand))) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return;
            }

            val = at_int(points, cand);
            float res = sdist(val, tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
}

template<typename E>
void min_loc_t(const cv::Mat_<E>& points, cv::Vec2f& loc, E& out, E tgt, bool z_search)
{
    cv::Rect boundary(1, 1, points.cols-2, points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = E(-1,-1,-1);
        loc = {-1,-1};
        return;
    }

    bool changed = true;
    E val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);

    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};

    float step = 1.0;

    while (changed) {
        changed = false;

        for(auto& off : search) {
            cv::Vec2f cand = loc + off*step;

            if (!boundary.contains(cv::Point(cand))) {
                out = E(-1,-1,-1);
                loc = {-1,-1};
                return;
            }

            val = at_int(points, cand);
            float res = sdist(val, tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
}

// Explicit instantiations for min_loc_t (only for Vec3f which is what's actually used)
template void min_loc_t<cv::Vec3f>(const cv::Mat_<cv::Vec3f>&, cv::Vec2f&, cv::Vec3f&, cv::Vec3f, bool);

void min_loc_multi(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f& loc, cv::Vec3f& out,
                   const std::vector<cv::Vec3f>& tgts, const std::vector<float>& tds,
                   bool z_search, void* plane)
{
    // For now, this is a placeholder that matches the QuadSurface signature
    // The plane parameter needs proper typing (PlaneSurface*)
    // This will be implemented when we refactor QuadSurface.cpp
    if (!loc_valid(points, {loc[1], loc[0]})) {
        out = {-1,-1,-1};
        return;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    // Plane distance would be added here if plane is not null

    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = 1.0;
    constexpr float min_step = 0.125;

    while (changed) {
        changed = false;

        for(auto& off : search) {
            cv::Vec2f cand = loc + off*step;

            if (!loc_valid(points, {cand[1], cand[0]})) {
                continue;
            }

            val = at_int(points, cand);
            float res = tdist_sum(val, tgts, tds);
            // Plane distance would be added here if plane is not null

            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step < min_step)
            break;
    }
}

// ============================================================================
// Normal computation
// ============================================================================

cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f>& points, const cv::Vec3f& loc)
{
    cv::Vec2f inb_loc = {loc[0], loc[1]};
    // Move inside from the grid border so we can access required locations
    inb_loc = vmax(inb_loc, {1,1});
    inb_loc = vmin(inb_loc, {static_cast<float>(points.cols-3), static_cast<float>(points.rows-3)});

    if (!loc_valid_xy(points, inb_loc))
        return {NAN,NAN,NAN};

    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(-1,0)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,1)))
        return {NAN,NAN,NAN};
    if (!loc_valid_xy(points, inb_loc+cv::Vec2f(0,-1)))
        return {NAN,NAN,NAN};

    cv::Vec3f xv = normed(at_int(points, inb_loc+cv::Vec2f(1,0)) - at_int(points, inb_loc-cv::Vec2f(1,0)));
    cv::Vec3f yv = normed(at_int(points, inb_loc+cv::Vec2f(0,1)) - at_int(points, inb_loc-cv::Vec2f(0,1)));

    cv::Vec3f n = yv.cross(xv);

    if (std::isnan(n[0]))
        return {NAN,NAN,NAN};

    return normed(n);
}

} // namespace vc::utils
