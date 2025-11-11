#include "vc/core/util/SurfacePatchIndex.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "vc/core/util/Surface.hpp"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

namespace {

struct TriangleHit {
    cv::Vec3f closest{0, 0, 0};
    cv::Vec3f bary{0, 0, 0}; // weights for vertices (sum to 1, >= 0)
    float distSq = std::numeric_limits<float>::max();
};

inline float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

TriangleHit closestPointOnTriangle(const cv::Vec3f& p,
                                   const cv::Vec3f& a,
                                   const cv::Vec3f& b,
                                   const cv::Vec3f& c)
{
    TriangleHit hit;

    const cv::Vec3f ab = b - a;
    const cv::Vec3f ac = c - a;
    const cv::Vec3f ap = p - a;

    float d1 = ab.dot(ap);
    float d2 = ac.dot(ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        hit.closest = a;
        hit.bary = {1.0f, 0.0f, 0.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    const cv::Vec3f bp = p - b;
    float d3 = ab.dot(bp);
    float d4 = ac.dot(bp);
    if (d3 >= 0.0f && d4 <= d3) {
        hit.closest = b;
        hit.bary = {0.0f, 1.0f, 0.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        hit.closest = a + v * ab;
        hit.bary = {1.0f - v, v, 0.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    const cv::Vec3f cp = p - c;
    float d5 = ab.dot(cp);
    float d6 = ac.dot(cp);
    if (d6 >= 0.0f && d5 <= d6) {
        hit.closest = c;
        hit.bary = {0.0f, 0.0f, 1.0f};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        hit.closest = a + w * ac;
        hit.bary = {1.0f - w, 0.0f, w};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        hit.closest = b + w * (c - b);
        hit.bary = {0.0f, 1.0f - w, w};
        hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
        return hit;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float u = 1.0f - v - w;
    hit.closest = a + ab * v + ac * w;
    hit.bary = {u, v, w};
    hit.distSq = cv::norm(p - hit.closest, cv::NORM_L2SQR);
    return hit;
}

} // namespace

struct SurfacePatchIndex::Impl {
    struct PatchRecord {
        QuadSurface* surface = nullptr;
        int i = 0;
        int j = 0;
        std::array<cv::Vec3f, 4> corners{};
    };

    using Point3 = bg::model::point<float, 3, bg::cs::cartesian>;
    using Box3 = bg::model::box<Point3>;
    using Entry = std::pair<Box3, PatchRecord>;
    using PatchTree = bgi::rtree<Entry, bgi::quadratic<32>>;

    std::unique_ptr<PatchTree> tree;
    size_t patchCount = 0;

    struct PatchHit {
        bool valid = false;
        float u = 0.0f;
        float v = 0.0f;
        float distSq = std::numeric_limits<float>::max();
    };

    static PatchHit evaluatePatch(const PatchRecord& rec, const cv::Vec3f& point) {
        PatchHit best;

        const auto& p00 = rec.corners[0];
        const auto& p10 = rec.corners[1];
        const auto& p11 = rec.corners[2];
        const auto& p01 = rec.corners[3];

        // Triangle 0: (p00, p10, p01)
        {
            TriangleHit tri = closestPointOnTriangle(point, p00, p10, p01);
            if (tri.distSq < best.distSq) {
                best.valid = true;
                best.distSq = tri.distSq;
                best.u = clamp01(tri.bary[1]);
                best.v = clamp01(tri.bary[2]);
            }
        }

        // Triangle 1: (p10, p11, p01)
        {
            TriangleHit tri = closestPointOnTriangle(point, p10, p11, p01);
            if (tri.distSq < best.distSq) {
                best.valid = true;
                best.distSq = tri.distSq;
                float u = clamp01(tri.bary[0] + tri.bary[1]);
                float v = clamp01(tri.bary[1] + tri.bary[2]);
                best.u = u;
                best.v = v;
            }
        }

        return best;
    }
};

SurfacePatchIndex::SurfacePatchIndex()
    : impl_(std::make_unique<Impl>())
{}

SurfacePatchIndex::~SurfacePatchIndex() = default;
SurfacePatchIndex::SurfacePatchIndex(SurfacePatchIndex&&) noexcept = default;
SurfacePatchIndex& SurfacePatchIndex::operator=(SurfacePatchIndex&&) noexcept = default;

void SurfacePatchIndex::rebuild(const std::vector<QuadSurface*>& surfaces, float bboxPadding)
{
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }

    std::vector<Impl::Entry> entries;

    for (QuadSurface* surface : surfaces) {
        if (!surface) {
            continue;
        }
        const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
        if (!points || points->empty()) {
            continue;
        }

        const int rows = points->rows;
        const int cols = points->cols;
        if (rows < 2 || cols < 2) {
            continue;
        }

        for (int j = 0; j < rows - 1; ++j) {
            for (int i = 0; i < cols - 1; ++i) {
                const cv::Vec3f& p00 = (*points)(j, i);
                const cv::Vec3f& p10 = (*points)(j, i + 1);
                const cv::Vec3f& p01 = (*points)(j + 1, i);
                const cv::Vec3f& p11 = (*points)(j + 1, i + 1);

                if (p00[0] == -1.0f || p10[0] == -1.0f || p01[0] == -1.0f || p11[0] == -1.0f) {
                    continue;
                }

                Impl::PatchRecord rec;
                rec.surface = surface;
                rec.i = i;
                rec.j = j;
                rec.corners = {p00, p10, p11, p01};

                cv::Vec3f low = rec.corners[0];
                cv::Vec3f high = rec.corners[0];
                for (const cv::Vec3f& c : rec.corners) {
                    low[0] = std::min(low[0], c[0]);
                    low[1] = std::min(low[1], c[1]);
                    low[2] = std::min(low[2], c[2]);
                    high[0] = std::max(high[0], c[0]);
                    high[1] = std::max(high[1], c[1]);
                    high[2] = std::max(high[2], c[2]);
                }

                if (bboxPadding > 0.0f) {
                    low -= cv::Vec3f(bboxPadding, bboxPadding, bboxPadding);
                    high += cv::Vec3f(bboxPadding, bboxPadding, bboxPadding);
                }

                Impl::Point3 min_pt(low[0], low[1], low[2]);
                Impl::Point3 max_pt(high[0], high[1], high[2]);

                entries.emplace_back(Impl::Box3(min_pt, max_pt), rec);
            }
        }
    }

    impl_->patchCount = entries.size();
    if (entries.empty()) {
        impl_->tree.reset();
        return;
    }

    impl_->tree = std::make_unique<Impl::PatchTree>(entries.begin(), entries.end());
}

void SurfacePatchIndex::clear()
{
    if (impl_) {
        impl_->tree.reset();
        impl_->patchCount = 0;
    }
}

bool SurfacePatchIndex::empty() const
{
    return !impl_ || !impl_->tree || impl_->patchCount == 0;
}

std::optional<SurfacePatchIndex::LookupResult>
SurfacePatchIndex::locate(const cv::Vec3f& worldPoint, float tolerance, QuadSurface* targetSurface) const
{
    if (!impl_ || !impl_->tree || tolerance <= 0.0f) {
        return std::nullopt;
    }

    const float tol = std::max(tolerance, 0.0f);
    Impl::Point3 min_pt(worldPoint[0] - tol, worldPoint[1] - tol, worldPoint[2] - tol);
    Impl::Point3 max_pt(worldPoint[0] + tol, worldPoint[1] + tol, worldPoint[2] + tol);
    Impl::Box3 query(min_pt, max_pt);

    std::vector<Impl::Entry> candidates;
    impl_->tree->query(bgi::intersects(query), std::back_inserter(candidates));

    const float toleranceSq = tol * tol;
    SurfacePatchIndex::LookupResult best;
    float bestDistSq = toleranceSq;
    bool found = false;

    for (const auto& entry : candidates) {
        const Impl::PatchRecord& rec = entry.second;
        if (targetSurface && rec.surface != targetSurface) {
            continue;
        }

        Impl::PatchHit hit = Impl::evaluatePatch(rec, worldPoint);
        if (!hit.valid || hit.distSq > bestDistSq) {
            continue;
        }

        const cv::Vec3f center = rec.surface->center();
        const cv::Vec2f scale = rec.surface->scale();
        const float absX = static_cast<float>(rec.i) + hit.u;
        const float absY = static_cast<float>(rec.j) + hit.v;
        cv::Vec3f ptr = {
            absX - center[0] * scale[0],
            absY - center[1] * scale[1],
            0.0f
        };

        best.surface = rec.surface;
        best.ptr = ptr;
        best.distance = std::sqrt(hit.distSq);
        bestDistSq = hit.distSq;
        found = true;
    }

    if (!found) {
        return std::nullopt;
    }

    return best;
}
