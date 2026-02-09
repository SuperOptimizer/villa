#pragma once

#include "Sampler.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <unordered_map>

namespace vc::simd {

// --------------------------------------------------------------------------
// Interpolator<T, N> — Ceres Jet compatible trilinear interpolation
//
// Uses thread-local Sampler pool for concurrent access from Ceres solver.
// Drop-in replacement for CachedChunked3dInterpolator with compile-time
// tile strides.
// --------------------------------------------------------------------------
template <typename T, int N>
class Interpolator {
public:
    explicit Interpolator(SparseVolume<T, N>& vol)
        : vol_(vol), shape_(vol.shape()) {}

    Interpolator(const Interpolator&) = delete;
    Interpolator& operator=(const Interpolator&) = delete;

    // Trilinear interpolation compatible with Ceres Jet types.
    // V can be double, ceres::Jet<double, N>, etc.
    template <typename V>
    void Evaluate(const V& z, const V& y, const V& x, V* out) const {
        Sampler<T, N>& s = local_sampler();

        int cz_ = static_cast<int>(std::floor(val(z)));
        int cy_ = static_cast<int>(std::floor(val(y)));
        int cx_ = static_cast<int>(std::floor(val(x)));

        cz_ = std::clamp(cz_, 0, shape_[0] > 0 ? shape_[0] - 2 : 0);
        cy_ = std::clamp(cy_, 0, shape_[1] > 0 ? shape_[1] - 2 : 0);
        cx_ = std::clamp(cx_, 0, shape_[2] > 0 ? shape_[2] - 2 : 0);

        const V fz = z - V(cz_);
        const V fy = y - V(cy_);
        const V fx = x - V(cx_);

        const V cz = clamp_v(fz);
        const V cy = clamp_v(fy);
        const V cx = clamp_v(fx);

        const V c000 = V(s.sample_int(cz_,     cy_,     cx_));
        const V c100 = V(s.sample_int(cz_ + 1, cy_,     cx_));
        const V c010 = V(s.sample_int(cz_,     cy_ + 1, cx_));
        const V c110 = V(s.sample_int(cz_ + 1, cy_ + 1, cx_));
        const V c001 = V(s.sample_int(cz_,     cy_,     cx_ + 1));
        const V c101 = V(s.sample_int(cz_ + 1, cy_,     cx_ + 1));
        const V c011 = V(s.sample_int(cz_,     cy_ + 1, cx_ + 1));
        const V c111 = V(s.sample_int(cz_ + 1, cy_ + 1, cx_ + 1));

        const V c00 = (V(1) - cx) * c000 + cx * c001;
        const V c01 = (V(1) - cx) * c010 + cx * c011;
        const V c10 = (V(1) - cx) * c100 + cx * c101;
        const V c11 = (V(1) - cx) * c110 + cx * c111;

        const V c0 = (V(1) - cy) * c00 + cy * c01;
        const V c1 = (V(1) - cy) * c10 + cy * c11;

        *out = (V(1) - cz) * c0 + cz * c1;
    }

private:
    SparseVolume<T, N>& vol_;
    std::array<int, 3> shape_;

    static double val(const double& v) { return v; }
    template <typename JetT>
    static double val(const JetT& v) { return v.a; }

    template <typename V>
    static V clamp_v(const V& v) {
        if constexpr (std::is_same_v<V, double> || std::is_same_v<V, float>) {
            return std::clamp(v, V(0), V(1));
        } else {
            // For Jet types: clamp preserves derivative information
            if (val(v) < 0.0) return V(0);
            if (val(v) > 1.0) return V(1);
            return v;
        }
    }

    Sampler<T, N>& local_sampler() const {
        // Thread-local pool keyed by volume address
        struct TLS {
            std::unordered_map<const void*, std::unique_ptr<Sampler<T, N>>> map;
            std::deque<const void*> order;
        };
        thread_local TLS tls;

        constexpr std::size_t kMax = 256;
        const void* k = static_cast<const void*>(&vol_);

        if (auto it = tls.map.find(k); it != tls.map.end())
            return *it->second;

        // Evict oldest if at capacity
        if (tls.map.size() >= kMax && !tls.order.empty()) {
            const void* old = tls.order.front();
            tls.order.pop_front();
            tls.map.erase(old);
        }

        auto [it, inserted] = tls.map.emplace(
            k, std::make_unique<Sampler<T, N>>(
                   const_cast<SparseVolume<T, N>&>(vol_)));
        if (inserted)
            tls.order.push_back(k);
        return *it->second;
    }
};

}  // namespace vc::simd
