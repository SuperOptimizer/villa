#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

// Use inverse sqrt + multiply instead of sqrt + divide (faster)
[[gnu::always_inline]] static cv::Vec3f normed(const cv::Vec3f& v) noexcept
{
    const float lenSq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    if (lenSq < 1e-12f) [[unlikely]] return {0.0f, 0.0f, 0.0f};
    const float invLen = 1.0f / std::sqrt(lenSq);
    return {v[0] * invLen, v[1] * invLen, v[2] * invLen};
}

[[gnu::always_inline]] static cv::Vec2f vmin(const cv::Vec2f &a, const cv::Vec2f &b) noexcept
{
    return {std::min(a[0],b[0]),std::min(a[1],b[1])};
}

[[gnu::always_inline]] static cv::Vec2f vmax(const cv::Vec2f &a, const cv::Vec2f &b) noexcept
{
    return {std::max(a[0],b[0]),std::max(a[1],b[1])};
}

cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc)
{
    cv::Vec2f inb_loc = {loc[0], loc[1]};
    //move inside from the grid border so w can access required locations
    inb_loc = vmax(inb_loc, {1.f,1.f});
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

    cv::Vec3f xv = normed(at_int(points,inb_loc+cv::Vec2f(1,0))-at_int(points,inb_loc-cv::Vec2f(1,0)));
    cv::Vec3f yv = normed(at_int(points,inb_loc+cv::Vec2f(0,1))-at_int(points,inb_loc-cv::Vec2f(0,1)));

    // Left-hand rule: +U (xv) × +V (yv) = +Z (toward viewer)
    cv::Vec3f n = xv.cross(yv);

    if (std::isnan(n[0]))
        return {NAN,NAN,NAN};

    return normed(n);
}

// FMA-friendly bilinear interpolation: a + t*(b-a) instead of (1-t)*a + t*b
template <typename E>
[[gnu::always_inline]] static E at_int_impl(const cv::Mat_<E> &points, const cv::Vec2f& p) noexcept
{
    const int x = static_cast<int>(p[0]);
    const int y = static_cast<int>(p[1]);
    const float fx = p[0] - x;
    const float fy = p[1] - y;

    const E& p00 = points(y, x);
    const E& p01 = points(y, x + 1);
    const E& p10 = points(y + 1, x);
    const E& p11 = points(y + 1, x + 1);

    // FMA-friendly form: a + t*(b-a) generates fmadd instructions
    const E p0 = p00 + fx * (p01 - p00);
    const E p1 = p10 + fx * (p11 - p10);

    return p0 + fy * (p1 - p0);
}

template<typename T, int C>
[[gnu::always_inline]] static bool loc_valid_impl(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l) noexcept
{
    if (l[0] == -1) [[unlikely]]
        return false;

    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {static_cast<int>(floor(l[0])), static_cast<int>(floor(l[1]))};

    if (!bounds.contains(cv::Point(li))) [[unlikely]]
        return false;

    if (m(li[0],li[1])[0] == -1) [[unlikely]]
        return false;
    if (m(li[0]+1,li[1])[0] == -1) [[unlikely]]
        return false;
    if (m(li[0],li[1]+1)[0] == -1) [[unlikely]]
        return false;
    if (m(li[0]+1,li[1]+1)[0] == -1) [[unlikely]]
        return false;
    return true;
}

[[gnu::always_inline]] static bool loc_valid_scalar(const cv::Mat_<float> &m, const cv::Vec2d &l) noexcept
{
    if (l[0] == -1) [[unlikely]]
        return false;

    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {static_cast<int>(floor(l[0])), static_cast<int>(floor(l[1]))};

    if (!bounds.contains(cv::Point(li))) [[unlikely]]
        return false;

    if (m(li[0],li[1]) == -1) [[unlikely]]
        return false;
    if (m(li[0]+1,li[1]) == -1) [[unlikely]]
        return false;
    if (m(li[0],li[1]+1) == -1) [[unlikely]]
        return false;
    if (m(li[0]+1,li[1]+1) == -1) [[unlikely]]
        return false;
    return true;
}

template<typename T, int C>
[[gnu::always_inline]] static bool loc_valid_xy_impl(const cv::Mat_<cv::Vec<T,C>> &m, const cv::Vec2d &l) noexcept
{
    return loc_valid_impl(m, {l[1],l[0]});
}

[[gnu::always_inline]] static bool loc_valid_xy_scalar(const cv::Mat_<float> &m, const cv::Vec2d &l) noexcept
{
    return loc_valid_scalar(m, {l[1],l[0]});
}

[[gnu::flatten]] cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &p) noexcept {
    return at_int_impl(points, p);
}

[[gnu::flatten]] float at_int(const cv::Mat_<float> &points, const cv::Vec2f& p) noexcept {
    return at_int_impl(points, p);
}

[[gnu::flatten]] cv::Vec3d at_int(const cv::Mat_<cv::Vec3d> &points, const cv::Vec2f& p) noexcept {
    return at_int_impl(points, p);
}

[[gnu::flatten]] bool loc_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l) noexcept {
    return loc_valid_impl(m, l);
}

[[gnu::flatten]] bool loc_valid(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l) noexcept {
    return loc_valid_impl(m, l);
}

[[gnu::flatten]] bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l) noexcept {
    return loc_valid_scalar(m, l);
}

[[gnu::flatten]] bool loc_valid_xy(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l) noexcept {
    return loc_valid_xy_impl(m, l);
}

[[gnu::flatten]] bool loc_valid_xy(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l) noexcept {
    return loc_valid_xy_impl(m, l);
}

[[gnu::flatten]] bool loc_valid_xy(const cv::Mat_<float> &m, const cv::Vec2d &l) noexcept {
    return loc_valid_xy_scalar(m, l);
}


float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist) noexcept
{
    cv::Vec3f d = a-b;
    float l = std::sqrt(d.dot(d));

    return std::abs(l-t_dist);
}

float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds) noexcept
{
    float sum = 0;
    for(size_t i=0;i<tgts.size();i++) {
        float d = tdist(v, tgts[i], tds[i]);
        sum += d*d;
    }

    return sum;
}

// Helper: remove spatial outliers based on robust neighbor-distance stats
cv::Mat_<cv::Vec3f> clean_surface_outliers(const cv::Mat_<cv::Vec3f>& points, float distance_threshold, bool print_stats)
{
    cv::Mat_<cv::Vec3f> cleaned = points.clone();

    std::vector<float> all_neighbor_dists;
    all_neighbor_dists.reserve(static_cast<size_t>(points.rows) * points.cols);

    // First pass: gather squared neighbor distances (avoid sqrt in hot loop)
    for (auto [j, i, center] : ValidPointRange<const cv::Vec3f>(&points)) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int ny = j + dy;
                const int nx = i + dx;
                if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                    if (points(ny, nx)[0] != -1.f) {
                        const cv::Vec3f d = center - points(ny, nx);
                        const float distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
                        if (std::isfinite(distSq) && distSq > 0.f) {
                            all_neighbor_dists.push_back(distSq);
                        }
                    }
                }
            }
        }
    }

    float median_distSq = 0.0f;
    float mad = 0.0f;
    if (!all_neighbor_dists.empty()) {
        std::sort(all_neighbor_dists.begin(), all_neighbor_dists.end());
        median_distSq = all_neighbor_dists[all_neighbor_dists.size() / 2];
        std::vector<float> abs_devs;
        abs_devs.reserve(all_neighbor_dists.size());
        for (float d : all_neighbor_dists) {
            abs_devs.push_back(std::abs(d - median_distSq));
        }
        std::sort(abs_devs.begin(), abs_devs.end());
        mad = abs_devs[abs_devs.size() / 2];
    }
    // Threshold in squared-distance space
    const float threshold = median_distSq + distance_threshold * (mad / 0.6745f);

    if (print_stats) {
        std::cout << "Outlier detection statistics:" << "\n";
        std::cout << "  Median neighbor distance²: " << median_distSq << "\n";
        std::cout << "  MAD: " << mad << "\n";
        std::cout << "  K (sigma multiplier): " << distance_threshold << "\n";
        std::cout << "  Distance² threshold: " << threshold << "\n";
    }

    // Second pass: invalidate isolated/far points (using squared distances)
    int removed_count = 0;
    for (auto [j, i, center] : ValidPointRange<const cv::Vec3f>(&points)) {
        float min_neighborSq = std::numeric_limits<float>::infinity();
        int neighbor_count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int ny = j + dy;
                const int nx = i + dx;
                if (ny >= 0 && ny < points.rows && nx >= 0 && nx < points.cols) {
                    if (points(ny, nx)[0] != -1.f) {
                        const cv::Vec3f d = center - points(ny, nx);
                        const float distSq = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
                        if (std::isfinite(distSq)) {
                            min_neighborSq = std::min(min_neighborSq, distSq);
                            neighbor_count++;
                        }
                    }
                }
            }
        }
        if (neighbor_count == 0 || (min_neighborSq > threshold && threshold > 0.f)) {
            cleaned(j, i) = cv::Vec3f(-1.f, -1.f, -1.f);
            if (print_stats) removed_count++;
        }
    }

    if (print_stats) {
        std::cout << "Surface cleaning: removed " << removed_count << " outlier points" << "\n";
    }

    return cleaned;
}
