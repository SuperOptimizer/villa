#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace utils {

/// N-dimensional KD-tree spatial index.
///
/// Supports incremental insertion, bulk construction, nearest-neighbor,
/// k-nearest-neighbor, and radius search queries. Points are identified by a
/// `uint64_t` id that is either user-supplied or auto-assigned.
template <std::size_t Dims = 3>
class KdTree {
public:
    using Point = std::array<float, Dims>;

    KdTree() = default;
    ~KdTree() = default;

    KdTree(KdTree&& other) noexcept = default;
    auto operator=(KdTree&& other) noexcept -> KdTree& = default;

    KdTree(const KdTree& other) = default;
    auto operator=(const KdTree& other) -> KdTree& = default;

    /// Insert a single point with an explicit id.
    auto insert(Point p, std::uint64_t id) -> void {
        nodes_.push_back({p, id});
        built_ = false;
    }

    /// Insert a single point with auto-assigned id (equal to current size).
    auto insert(Point p) -> void {
        insert(p, static_cast<std::uint64_t>(nodes_.size()));
    }

    /// Bulk-build from a span of points. Ids are assigned 0..n-1.
    /// Replaces any previously stored data and builds the tree immediately.
    auto build(std::span<const Point> points) -> void {
        nodes_.clear();
        nodes_.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            nodes_.push_back({points[i], static_cast<std::uint64_t>(i)});
        }
        rebuild();
    }

    /// Ensure the internal tree structure is up-to-date. Called automatically
    /// before queries when needed.
    auto rebuild() -> void {
        if (nodes_.empty()) {
            built_ = true;
            return;
        }
        build_recursive(0, nodes_.size(), 0);
        built_ = true;
    }

    /// Find the single nearest neighbor to `query`.
    /// Returns (id, squared distance), or nullopt if the tree is empty.
    [[nodiscard]] auto nearest(Point query) const
        -> std::optional<std::pair<std::uint64_t, float>> {
        ensure_built();
        if (nodes_.empty()) {
            return std::nullopt;
        }
        float best_dist_sq = std::numeric_limits<float>::max();
        std::size_t best_idx = 0;
        nearest_recursive(query, 0, nodes_.size(), 0, best_dist_sq, best_idx);
        return std::pair{nodes_[best_idx].id, best_dist_sq};
    }

    /// Find the k nearest neighbors to `query`.
    /// Returns up to k pairs of (id, squared distance), sorted nearest first.
    [[nodiscard]] auto knn(Point query, std::size_t k) const
        -> std::vector<std::pair<std::uint64_t, float>> {
        ensure_built();
        if (nodes_.empty() || k == 0) {
            return {};
        }
        // Max-heap of (distance_sq, index)
        using Entry = std::pair<float, std::size_t>;
        std::priority_queue<Entry> heap;
        knn_recursive(query, k, 0, nodes_.size(), 0, heap);

        std::vector<std::pair<std::uint64_t, float>> result;
        result.reserve(heap.size());
        while (!heap.empty()) {
            auto [dist_sq, idx] = heap.top();
            heap.pop();
            result.push_back({nodes_[idx].id, dist_sq});
        }
        std::ranges::reverse(result);
        return result;
    }

    /// Find all points within `radius` of `query`.
    /// Returns pairs of (id, squared distance), sorted nearest first.
    [[nodiscard]] auto radius_search(Point query, float radius) const
        -> std::vector<std::pair<std::uint64_t, float>> {
        ensure_built();
        if (nodes_.empty() || radius <= 0.0F) {
            return {};
        }
        float radius_sq = radius * radius;
        std::vector<std::pair<std::uint64_t, float>> result;
        radius_recursive(query, radius_sq, 0, nodes_.size(), 0, result);
        std::ranges::sort(result, {}, &std::pair<std::uint64_t, float>::second);
        return result;
    }

    /// Number of points in the tree.
    [[nodiscard]] auto size() const noexcept -> std::size_t {
        return nodes_.size();
    }

    /// Whether the tree contains no points.
    [[nodiscard]] auto empty() const noexcept -> bool {
        return nodes_.empty();
    }

    /// Remove all points and reset the tree.
    auto clear() -> void {
        nodes_.clear();
        built_ = true;
    }

private:
    struct Node {
        Point point{};
        std::uint64_t id{0};
    };

    mutable std::vector<Node> nodes_;
    mutable bool built_{true};

    /// Squared Euclidean distance between two points.
    [[nodiscard]] static auto distance_sq(const Point& a,
                                          const Point& b) noexcept -> float {
        float d = 0.0F;
        for (std::size_t i = 0; i < Dims; ++i) {
            float diff = a[i] - b[i];
            d += diff * diff;
        }
        return d;
    }

    /// Ensure the tree is built before a query. Since we mark nodes_ mutable,
    /// this can be called from const methods.
    auto ensure_built() const -> void {
        if (!built_) {
            const_cast<KdTree*>(this)->rebuild();
        }
    }

    /// Recursively build the tree by partitioning nodes in-place.
    /// The range [begin, end) is partitioned so that the median element (by the
    /// current splitting axis) is placed at mid = (begin+end)/2. Left children
    /// occupy [begin, mid) and right children occupy [mid+1, end).
    auto build_recursive(std::size_t begin, std::size_t end,
                         std::size_t depth) -> void {
        if (end - begin <= 1) {
            return;
        }
        std::size_t axis = depth % Dims;
        std::size_t mid = begin + (end - begin) / 2;
        std::nth_element(
            nodes_.begin() + static_cast<std::ptrdiff_t>(begin),
            nodes_.begin() + static_cast<std::ptrdiff_t>(mid),
            nodes_.begin() + static_cast<std::ptrdiff_t>(end),
            [axis](const Node& a, const Node& b) {
                return a.point[axis] < b.point[axis];
            });
        build_recursive(begin, mid, depth + 1);
        build_recursive(mid + 1, end, depth + 1);
    }

    /// Recursive nearest-neighbor search.
    auto nearest_recursive(const Point& query, std::size_t begin,
                           std::size_t end, std::size_t depth,
                           float& best_dist_sq,
                           std::size_t& best_idx) const -> void {
        if (begin >= end) {
            return;
        }
        std::size_t mid = begin + (end - begin) / 2;
        float d = distance_sq(query, nodes_[mid].point);
        if (d < best_dist_sq) {
            best_dist_sq = d;
            best_idx = mid;
        }
        if (end - begin == 1) {
            return;
        }
        std::size_t axis = depth % Dims;
        float diff = query[axis] - nodes_[mid].point[axis];
        float diff_sq = diff * diff;

        // Search the side containing the query first
        std::size_t near_begin, near_end, far_begin, far_end;
        if (diff <= 0.0F) {
            near_begin = begin;
            near_end = mid;
            far_begin = mid + 1;
            far_end = end;
        } else {
            near_begin = mid + 1;
            near_end = end;
            far_begin = begin;
            far_end = mid;
        }

        nearest_recursive(query, near_begin, near_end, depth + 1, best_dist_sq,
                           best_idx);
        // Only search the far side if the splitting plane is closer than the
        // current best
        if (diff_sq < best_dist_sq) {
            nearest_recursive(query, far_begin, far_end, depth + 1,
                              best_dist_sq, best_idx);
        }
    }

    /// Recursive k-nearest-neighbor search.
    auto knn_recursive(
        const Point& query, std::size_t k, std::size_t begin, std::size_t end,
        std::size_t depth,
        std::priority_queue<std::pair<float, std::size_t>>& heap) const
        -> void {
        if (begin >= end) {
            return;
        }
        std::size_t mid = begin + (end - begin) / 2;
        float d = distance_sq(query, nodes_[mid].point);
        if (heap.size() < k) {
            heap.push({d, mid});
        } else if (d < heap.top().first) {
            heap.pop();
            heap.push({d, mid});
        }
        if (end - begin == 1) {
            return;
        }
        std::size_t axis = depth % Dims;
        float diff = query[axis] - nodes_[mid].point[axis];
        float diff_sq = diff * diff;

        std::size_t near_begin, near_end, far_begin, far_end;
        if (diff <= 0.0F) {
            near_begin = begin;
            near_end = mid;
            far_begin = mid + 1;
            far_end = end;
        } else {
            near_begin = mid + 1;
            near_end = end;
            far_begin = begin;
            far_end = mid;
        }

        knn_recursive(query, k, near_begin, near_end, depth + 1, heap);
        float worst =
            heap.size() < k ? std::numeric_limits<float>::max() : heap.top().first;
        if (diff_sq < worst) {
            knn_recursive(query, k, far_begin, far_end, depth + 1, heap);
        }
    }

    /// Recursive radius search.
    auto radius_recursive(
        const Point& query, float radius_sq, std::size_t begin,
        std::size_t end, std::size_t depth,
        std::vector<std::pair<std::uint64_t, float>>& result) const -> void {
        if (begin >= end) {
            return;
        }
        std::size_t mid = begin + (end - begin) / 2;
        float d = distance_sq(query, nodes_[mid].point);
        if (d <= radius_sq) {
            result.push_back({nodes_[mid].id, d});
        }
        if (end - begin == 1) {
            return;
        }
        std::size_t axis = depth % Dims;
        float diff = query[axis] - nodes_[mid].point[axis];
        float diff_sq = diff * diff;

        std::size_t near_begin, near_end, far_begin, far_end;
        if (diff <= 0.0F) {
            near_begin = begin;
            near_end = mid;
            far_begin = mid + 1;
            far_end = end;
        } else {
            near_begin = mid + 1;
            near_end = end;
            far_begin = begin;
            far_end = mid;
        }

        radius_recursive(query, radius_sq, near_begin, near_end, depth + 1,
                          result);
        if (diff_sq <= radius_sq) {
            radius_recursive(query, radius_sq, far_begin, far_end, depth + 1,
                              result);
        }
    }
};

} // namespace utils
