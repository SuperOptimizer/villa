#pragma once
/// vc4d::math — Minimal linear-algebra types.
///
/// Replaces cv::Vec3f, cv::Vec2f, Eigen::Vector3f, and the grab-bag of
/// geometric helpers spread across vc3d.  All types are value types with
/// constexpr construction so they work naturally in containers, structured
/// bindings, and compile-time evaluation.
///
/// Design decisions vs vc3d:
///   • No OpenCV dependency — cv::Vec3f was used as a general 3-vector
///     throughout vc3d even though OpenCV is a computer-vision library.
///   • No Eigen dependency — Eigen was pulled in for matrix ops that are
///     trivially expressible here.
///   • Aggregates where possible — structured bindings work out of the box.
///   • constexpr everything — enables compile-time math.

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <format>
#include <numbers>
#include <string>

namespace vc4d {

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------
template <std::floating_point T = float>
struct Vec2 {
    T x{}, y{};

    // Arithmetic
    constexpr Vec2 operator+(Vec2 b) const { return {x + b.x, y + b.y}; }
    constexpr Vec2 operator-(Vec2 b) const { return {x - b.x, y - b.y}; }
    constexpr Vec2 operator*(T s)    const { return {x * s, y * s}; }
    constexpr Vec2 operator/(T s)    const { return {x / s, y / s}; }
    constexpr Vec2 operator-()       const { return {-x, -y}; }

    constexpr Vec2& operator+=(Vec2 b) { x += b.x; y += b.y; return *this; }
    constexpr Vec2& operator-=(Vec2 b) { x -= b.x; y -= b.y; return *this; }
    constexpr Vec2& operator*=(T s)    { x *= s; y *= s; return *this; }
    constexpr Vec2& operator/=(T s)    { x /= s; y /= s; return *this; }

    constexpr T    dot(Vec2 b)    const { return x * b.x + y * b.y; }
    constexpr T    length_sq()    const { return dot(*this); }
    T              length()       const { return std::sqrt(length_sq()); }
    Vec2           normalized()   const { auto l = length(); return l > T(0) ? *this / l : Vec2{}; }

    constexpr bool operator==(Vec2 b) const = default;

    // Array-style access
    constexpr T  operator[](int i) const { return i == 0 ? x : y; }
    constexpr T& operator[](int i)       { return i == 0 ? x : y; }

    std::string to_string() const { return std::format("({}, {})", x, y); }
};

template <std::floating_point T>
constexpr Vec2<T> operator*(T s, Vec2<T> v) { return v * s; }

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------
template <std::floating_point T = float>
struct Vec3 {
    T x{}, y{}, z{};

    constexpr Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    constexpr Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    constexpr Vec3 operator*(T s)    const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(T s)    const { return {x / s, y / s, z / s}; }
    constexpr Vec3 operator-()       const { return {-x, -y, -z}; }

    constexpr Vec3& operator+=(Vec3 b) { x += b.x; y += b.y; z += b.z; return *this; }
    constexpr Vec3& operator-=(Vec3 b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    constexpr Vec3& operator*=(T s)    { x *= s; y *= s; z *= s; return *this; }
    constexpr Vec3& operator/=(T s)    { x /= s; y /= s; z /= s; return *this; }

    constexpr T    dot(Vec3 b)    const { return x * b.x + y * b.y + z * b.z; }
    constexpr Vec3 cross(Vec3 b)  const {
        return {y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x};
    }
    constexpr T    length_sq()    const { return dot(*this); }
    T              length()       const { return std::sqrt(length_sq()); }
    Vec3           normalized()   const { auto l = length(); return l > T(0) ? *this / l : Vec3{}; }

    constexpr bool operator==(Vec3 b) const = default;

    constexpr T  operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    constexpr T& operator[](int i)       { return i == 0 ? x : (i == 1 ? y : z); }

    // Convenience: truncate / extend
    constexpr Vec2<T> xy() const { return {x, y}; }

    std::string to_string() const { return std::format("({}, {}, {})", x, y, z); }
};

template <std::floating_point T>
constexpr Vec3<T> operator*(T s, Vec3<T> v) { return v * s; }

// ---------------------------------------------------------------------------
// Box3 — axis-aligned bounding box (replaces vc3d Rect3D + ad-hoc bboxes)
// ---------------------------------------------------------------------------
template <std::floating_point T = float>
struct Box3 {
    Vec3<T> lo{};
    Vec3<T> hi{};

    static constexpr Box3 empty() {
        constexpr auto big = std::numeric_limits<T>::max();
        return {{big, big, big}, {-big, -big, -big}};
    }

    constexpr Box3 expanded(Vec3<T> p) const {
        return {
            {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)},
            {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)}
        };
    }

    constexpr Box3 merged(Box3 b) const {
        return {
            {std::min(lo.x, b.lo.x), std::min(lo.y, b.lo.y), std::min(lo.z, b.lo.z)},
            {std::max(hi.x, b.hi.x), std::max(hi.y, b.hi.y), std::max(hi.z, b.hi.z)}
        };
    }

    constexpr bool intersects(Box3 b) const {
        return lo.x <= b.hi.x && hi.x >= b.lo.x
            && lo.y <= b.hi.y && hi.y >= b.lo.y
            && lo.z <= b.hi.z && hi.z >= b.lo.z;
    }

    constexpr bool contains(Vec3<T> p) const {
        return p.x >= lo.x && p.x <= hi.x
            && p.y >= lo.y && p.y <= hi.y
            && p.z >= lo.z && p.z <= hi.z;
    }

    constexpr Vec3<T> center() const { return (lo + hi) / T(2); }
    constexpr Vec3<T> extent() const { return hi - lo; }
    constexpr bool    valid()  const { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }

    constexpr bool operator==(Box3 b) const = default;
};

// ---------------------------------------------------------------------------
// Common aliases
// ---------------------------------------------------------------------------
using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Box3f = Box3<float>;
using Box3d = Box3<double>;

// ---------------------------------------------------------------------------
// Free-function utilities
// ---------------------------------------------------------------------------
template <std::floating_point T>
constexpr T lerp(T a, T b, T t) { return a + t * (b - a); }

template <std::floating_point T>
constexpr Vec3<T> lerp(Vec3<T> a, Vec3<T> b, T t) {
    return {lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t)};
}

template <std::floating_point T>
T distance(Vec3<T> a, Vec3<T> b) { return (a - b).length(); }

template <std::floating_point T>
constexpr T distance_sq(Vec3<T> a, Vec3<T> b) { return (a - b).length_sq(); }

} // namespace vc4d
