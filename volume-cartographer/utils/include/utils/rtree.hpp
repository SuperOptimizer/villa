#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utils {

/// Axis-aligned bounding box in N dimensions.
template <std::size_t Dims>
struct Box {
    std::array<float, Dims> min{};
    std::array<float, Dims> max{};

    /// Create a box that contains nothing (inverted extents).
    [[nodiscard]] static auto empty_box() noexcept -> Box {
        Box b;
        b.min.fill(std::numeric_limits<float>::max());
        b.max.fill(std::numeric_limits<float>::lowest());
        return b;
    }

    /// Create a box from a single point (zero-volume).
    [[nodiscard]] static auto from_point(
        const std::array<float, Dims>& p) noexcept -> Box {
        return {p, p};
    }

    /// Expand this box to include another box.
    auto expand(const Box& other) noexcept -> void {
        for (std::size_t i = 0; i < Dims; ++i) {
            if (other.min[i] < min[i]) {
                min[i] = other.min[i];
            }
            if (other.max[i] > max[i]) {
                max[i] = other.max[i];
            }
        }
    }

    /// Compute the volume (area in 2D, hypervolume in N-D) of this box.
    [[nodiscard]] auto volume() const noexcept -> float {
        float v = 1.0F;
        for (std::size_t i = 0; i < Dims; ++i) {
            float extent = max[i] - min[i];
            if (extent <= 0.0F) {
                return 0.0F;
            }
            v *= extent;
        }
        return v;
    }

    /// Compute the volume of the union of this box and another.
    [[nodiscard]] auto union_volume(const Box& other) const noexcept -> float {
        float v = 1.0F;
        for (std::size_t i = 0; i < Dims; ++i) {
            float lo = std::min(min[i], other.min[i]);
            float hi = std::max(max[i], other.max[i]);
            float extent = hi - lo;
            if (extent <= 0.0F) {
                return 0.0F;
            }
            v *= extent;
        }
        return v;
    }

    /// Check whether this box intersects another box.
    [[nodiscard]] auto intersects(const Box& other) const noexcept -> bool {
        for (std::size_t i = 0; i < Dims; ++i) {
            if (min[i] > other.max[i] || max[i] < other.min[i]) {
                return false;
            }
        }
        return true;
    }

    /// Check whether this box fully contains a point.
    [[nodiscard]] auto contains_point(
        const std::array<float, Dims>& p) const noexcept -> bool {
        for (std::size_t i = 0; i < Dims; ++i) {
            if (p[i] < min[i] || p[i] > max[i]) {
                return false;
            }
        }
        return true;
    }

    /// Minimum squared distance from a point to this box. Returns 0 if the
    /// point is inside the box.
    [[nodiscard]] auto min_distance_sq(
        const std::array<float, Dims>& p) const noexcept -> float {
        float d = 0.0F;
        for (std::size_t i = 0; i < Dims; ++i) {
            if (p[i] < min[i]) {
                float diff = min[i] - p[i];
                d += diff * diff;
            } else if (p[i] > max[i]) {
                float diff = p[i] - max[i];
                d += diff * diff;
            }
        }
        return d;
    }
};

/// N-dimensional R-tree spatial index.
///
/// Supports both point and axis-aligned bounding box (AABB) storage with
/// dynamic insert/remove and efficient spatial queries. Points are stored
/// internally as zero-volume boxes. Each entry is identified by a user-supplied
/// `uint64_t` id.
///
/// Uses the quadratic split algorithm for node splitting.
template <std::size_t Dims = 3>
class RTree {
public:
    using Point = std::array<float, Dims>;
    using BoxType = Box<Dims>;

    explicit RTree(std::size_t max_entries_per_node = 16)
        : max_entries_(max_entries_per_node),
          min_entries_(max_entries_per_node / 2),
          root_(std::make_unique<Node>(true)) {
        if (min_entries_ < 1) {
            min_entries_ = 1;
        }
    }

    ~RTree() = default;

    RTree(RTree&& other) noexcept = default;
    auto operator=(RTree&& other) noexcept -> RTree& = default;

    RTree(const RTree& other)
        : max_entries_(other.max_entries_),
          min_entries_(other.min_entries_),
          size_(other.size_),
          root_(other.root_ ? other.root_->clone() : nullptr) {
        rebuild_id_map();
    }

    auto operator=(const RTree& other) -> RTree& {
        if (this != &other) {
            max_entries_ = other.max_entries_;
            min_entries_ = other.min_entries_;
            size_ = other.size_;
            root_ = other.root_ ? other.root_->clone() : nullptr;
            rebuild_id_map();
        }
        return *this;
    }

    /// Insert a point with an explicit id.
    auto insert(Point point, std::uint64_t id) -> void {
        insert(BoxType::from_point(point), id);
    }

    /// Insert a bounding box with an explicit id.
    auto insert(BoxType box, std::uint64_t id) -> void {
        auto entry = Entry{box, id};
        auto* leaf = choose_leaf(root_.get(), entry.box);
        leaf->entries.push_back(entry);
        id_map_[id] = leaf;
        ++size_;

        if (leaf->entries.size() > max_entries_) {
            handle_overflow(leaf);
        }
    }

    /// Remove the entry with the given id. No-op if id is not found.
    auto remove(std::uint64_t id) -> void {
        auto it = id_map_.find(id);
        if (it == id_map_.end()) {
            return;
        }

        Node* leaf = it->second;
        // Remove the entry from the leaf
        auto& entries = leaf->entries;
        auto eit = std::find_if(entries.begin(), entries.end(),
                                [id](const Entry& e) { return e.id == id; });
        if (eit != entries.end()) {
            entries.erase(eit);
        }
        id_map_.erase(it);
        --size_;

        // Condense the tree
        condense_tree(leaf);
    }

    /// Bulk load points (more efficient than repeated insert). Clears existing
    /// data.
    auto bulk_load_points(
        std::span<const std::pair<Point, std::uint64_t>> entries) -> void {
        std::vector<std::pair<BoxType, std::uint64_t>> box_entries;
        box_entries.reserve(entries.size());
        for (const auto& [pt, id] : entries) {
            box_entries.emplace_back(BoxType::from_point(pt), id);
        }
        bulk_load_boxes(box_entries);
    }

    /// Bulk load bounding boxes (more efficient than repeated insert). Clears
    /// existing data.
    auto bulk_load_boxes(
        std::span<const std::pair<BoxType, std::uint64_t>> entries) -> void {
        clear();
        if (entries.empty()) {
            return;
        }

        // Build sorted entry list
        std::vector<Entry> sorted;
        sorted.reserve(entries.size());
        for (const auto& [box, id] : entries) {
            sorted.push_back({box, id});
        }

        // Sort-tile-recursive (STR) bulk loading
        root_ = str_bulk_load(sorted);
        size_ = entries.size();
        rebuild_id_map();
    }

    /// Find the single nearest entry to `query`.
    /// Returns (id, squared distance), or nullopt if the tree is empty.
    [[nodiscard]] auto nearest(Point query) const
        -> std::optional<std::pair<std::uint64_t, float>> {
        if (empty()) {
            return std::nullopt;
        }
        auto result = knn(query, 1);
        if (result.empty()) {
            return std::nullopt;
        }
        return result[0];
    }

    /// Find the k nearest entries to `query`.
    /// Returns up to k pairs of (id, squared distance), sorted nearest first.
    /// Distance is measured from the query point to the nearest point of the
    /// entry's bounding box.
    [[nodiscard]] auto knn(Point query, std::size_t k) const
        -> std::vector<std::pair<std::uint64_t, float>> {
        if (empty() || k == 0) {
            return {};
        }

        // Max-heap of (distance_sq, id)
        using HeapEntry = std::pair<float, std::uint64_t>;
        std::priority_queue<HeapEntry> result_heap;

        // Min-heap of (distance_sq, node*) for branch-and-bound
        using BranchEntry = std::pair<float, const Node*>;
        auto branch_cmp = [](const BranchEntry& a, const BranchEntry& b) {
            return a.first > b.first;
        };
        std::priority_queue<BranchEntry, std::vector<BranchEntry>,
                            decltype(branch_cmp)>
            branch_heap(branch_cmp);

        branch_heap.push({root_->bounds.min_distance_sq(query), root_.get()});

        while (!branch_heap.empty()) {
            auto [node_dist, node] = branch_heap.top();
            branch_heap.pop();

            // Prune: if this node is farther than our k-th best, skip
            if (result_heap.size() >= k && node_dist > result_heap.top().first) {
                continue;
            }

            if (node->is_leaf) {
                for (const auto& entry : node->entries) {
                    float d = entry.box.min_distance_sq(query);
                    if (result_heap.size() < k) {
                        result_heap.push({d, entry.id});
                    } else if (d < result_heap.top().first) {
                        result_heap.pop();
                        result_heap.push({d, entry.id});
                    }
                }
            } else {
                for (const auto& child : node->children) {
                    float d = child->bounds.min_distance_sq(query);
                    if (result_heap.size() < k ||
                        d < result_heap.top().first) {
                        branch_heap.push({d, child.get()});
                    }
                }
            }
        }

        std::vector<std::pair<std::uint64_t, float>> result;
        result.reserve(result_heap.size());
        while (!result_heap.empty()) {
            auto [d, id] = result_heap.top();
            result_heap.pop();
            result.push_back({id, d});
        }
        std::ranges::reverse(result);
        return result;
    }

    /// Find all entries within `radius` of `query`.
    /// Returns pairs of (id, squared distance), sorted nearest first.
    [[nodiscard]] auto radius_search(Point query, float radius) const
        -> std::vector<std::pair<std::uint64_t, float>> {
        if (empty() || radius <= 0.0F) {
            return {};
        }
        float radius_sq = radius * radius;
        std::vector<std::pair<std::uint64_t, float>> result;
        radius_search_recursive(root_.get(), query, radius_sq, result);
        std::ranges::sort(result, {}, &std::pair<std::uint64_t, float>::second);
        return result;
    }

    /// Find all entries whose bounding box intersects the query box.
    [[nodiscard]] auto query_intersects(BoxType query) const
        -> std::vector<std::uint64_t> {
        std::vector<std::uint64_t> result;
        query_intersects(query, [&result](std::uint64_t id, const BoxType&) {
            result.push_back(id);
        });
        return result;
    }

    /// Find all entries whose bounding box contains the query point.
    [[nodiscard]] auto query_contains(Point query) const
        -> std::vector<std::uint64_t> {
        std::vector<std::uint64_t> result;
        query_contains_recursive(root_.get(), query, result);
        return result;
    }

    /// Visitor-based intersection query (avoids allocation).
    /// Visitor signature: void(uint64_t id, const BoxType& box)
    template <typename Fn>
    auto query_intersects(BoxType query, Fn&& visitor) const -> void {
        query_intersects_recursive(root_.get(), query,
                                   std::forward<Fn>(visitor));
    }

    /// Number of entries in the tree.
    [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }

    /// Whether the tree contains no entries.
    [[nodiscard]] auto empty() const noexcept -> bool { return size_ == 0; }

    /// Remove all entries and reset the tree.
    auto clear() -> void {
        root_ = std::make_unique<Node>(true);
        id_map_.clear();
        size_ = 0;
    }

private:
    struct Entry {
        BoxType box{};
        std::uint64_t id{0};
    };

    struct Node {
        bool is_leaf{true};
        BoxType bounds{BoxType::empty_box()};
        Node* parent{nullptr};

        // Leaf nodes store entries
        std::vector<Entry> entries;
        // Internal nodes store children
        std::vector<std::unique_ptr<Node>> children;

        explicit Node(bool leaf) : is_leaf(leaf) {}

        [[nodiscard]] auto clone() const -> std::unique_ptr<Node> {
            auto copy = std::make_unique<Node>(is_leaf);
            copy->bounds = bounds;
            copy->entries = entries;
            for (const auto& child : children) {
                auto child_copy = child->clone();
                child_copy->parent = copy.get();
                copy->children.push_back(std::move(child_copy));
            }
            return copy;
        }

        auto recompute_bounds() -> void {
            bounds = BoxType::empty_box();
            if (is_leaf) {
                for (const auto& entry : entries) {
                    bounds.expand(entry.box);
                }
            } else {
                for (const auto& child : children) {
                    bounds.expand(child->bounds);
                }
            }
        }
    };

    std::size_t max_entries_;
    std::size_t min_entries_;
    std::size_t size_{0};
    std::unique_ptr<Node> root_;
    std::unordered_map<std::uint64_t, Node*> id_map_;

    // ── Tree construction ──────────────────────────────────────────────────

    /// Choose the leaf node to insert an entry into.
    auto choose_leaf(Node* node, const BoxType& box) -> Node* {
        while (!node->is_leaf) {
            Node* best_child = nullptr;
            float best_enlargement = std::numeric_limits<float>::max();
            float best_volume = std::numeric_limits<float>::max();

            for (auto& child : node->children) {
                float current_vol = child->bounds.volume();
                float enlarged_vol = child->bounds.union_volume(box);
                float enlargement = enlarged_vol - current_vol;

                if (enlargement < best_enlargement ||
                    (enlargement == best_enlargement &&
                     current_vol < best_volume)) {
                    best_enlargement = enlargement;
                    best_volume = current_vol;
                    best_child = child.get();
                }
            }

            // Expand bounds as we descend
            node->bounds.expand(box);
            node = best_child;
        }
        node->bounds.expand(box);
        return node;
    }

    /// Handle overflow of a node by splitting.
    auto handle_overflow(Node* node) -> void {
        auto [left, right] = split_node(node);

        if (node->parent == nullptr) {
            // Node is root: create new root
            auto new_root = std::make_unique<Node>(false);
            left->parent = new_root.get();
            right->parent = new_root.get();
            new_root->bounds = BoxType::empty_box();
            new_root->bounds.expand(left->bounds);
            new_root->bounds.expand(right->bounds);
            new_root->children.push_back(std::move(left));
            new_root->children.push_back(std::move(right));
            root_ = std::move(new_root);
        } else {
            Node* parent = node->parent;

            // Replace the overflowing node with left in the parent
            for (auto& child : parent->children) {
                if (child.get() == node) {
                    left->parent = parent;
                    child = std::move(left);
                    break;
                }
            }

            // Add right to the parent
            right->parent = parent;
            parent->children.push_back(std::move(right));
            parent->recompute_bounds();

            // If parent overflows, recurse
            if (parent->children.size() > max_entries_) {
                handle_overflow_internal(parent);
            }
        }
    }

    /// Handle overflow of an internal node.
    auto handle_overflow_internal(Node* node) -> void {
        auto [left, right] = split_internal_node(node);

        if (node->parent == nullptr) {
            auto new_root = std::make_unique<Node>(false);
            left->parent = new_root.get();
            right->parent = new_root.get();
            new_root->bounds = BoxType::empty_box();
            new_root->bounds.expand(left->bounds);
            new_root->bounds.expand(right->bounds);
            new_root->children.push_back(std::move(left));
            new_root->children.push_back(std::move(right));
            root_ = std::move(new_root);
        } else {
            Node* parent = node->parent;

            for (auto& child : parent->children) {
                if (child.get() == node) {
                    left->parent = parent;
                    child = std::move(left);
                    break;
                }
            }

            right->parent = parent;
            parent->children.push_back(std::move(right));
            parent->recompute_bounds();

            if (parent->children.size() > max_entries_) {
                handle_overflow_internal(parent);
            }
        }
    }

    // ── Quadratic split (leaf) ─────────────────────────────────────────────

    /// Quadratic split: partition entries of a leaf node into two new leaf
    /// nodes.
    auto split_node(Node* node)
        -> std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> {
        auto& entries = node->entries;
        std::size_t n = entries.size();

        // Pick seeds: find pair with maximum waste (area of union minus
        // individual areas)
        std::size_t seed1 = 0;
        std::size_t seed2 = 1;
        float worst_waste = std::numeric_limits<float>::lowest();

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                float combined = entries[i].box.union_volume(entries[j].box);
                float waste = combined - entries[i].box.volume() -
                              entries[j].box.volume();
                if (waste > worst_waste) {
                    worst_waste = waste;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }

        auto left = std::make_unique<Node>(true);
        auto right = std::make_unique<Node>(true);

        left->entries.push_back(entries[seed1]);
        left->bounds = entries[seed1].box;

        right->entries.push_back(entries[seed2]);
        right->bounds = entries[seed2].box;

        // Mark used entries
        std::vector<bool> assigned(n, false);
        assigned[seed1] = true;
        assigned[seed2] = true;

        std::size_t remaining = n - 2;

        while (remaining > 0) {
            // If one group needs all remaining to meet minimum
            if (left->entries.size() + remaining == min_entries_) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!assigned[i]) {
                        left->entries.push_back(entries[i]);
                        left->bounds.expand(entries[i].box);
                        assigned[i] = true;
                    }
                }
                break;
            }
            if (right->entries.size() + remaining == min_entries_) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!assigned[i]) {
                        right->entries.push_back(entries[i]);
                        right->bounds.expand(entries[i].box);
                        assigned[i] = true;
                    }
                }
                break;
            }

            // Pick next: entry with maximum preference for one group
            std::size_t best_idx = 0;
            float best_diff = std::numeric_limits<float>::lowest();
            for (std::size_t i = 0; i < n; ++i) {
                if (assigned[i]) {
                    continue;
                }
                float d1 = left->bounds.union_volume(entries[i].box) -
                           left->bounds.volume();
                float d2 = right->bounds.union_volume(entries[i].box) -
                           right->bounds.volume();
                float diff = std::abs(d1 - d2);
                if (diff > best_diff) {
                    best_diff = diff;
                    best_idx = i;
                }
            }

            // Assign to the group that needs least enlargement
            float enlarge_left =
                left->bounds.union_volume(entries[best_idx].box) -
                left->bounds.volume();
            float enlarge_right =
                right->bounds.union_volume(entries[best_idx].box) -
                right->bounds.volume();

            if (enlarge_left < enlarge_right ||
                (enlarge_left == enlarge_right &&
                 left->entries.size() <= right->entries.size())) {
                left->entries.push_back(entries[best_idx]);
                left->bounds.expand(entries[best_idx].box);
            } else {
                right->entries.push_back(entries[best_idx]);
                right->bounds.expand(entries[best_idx].box);
            }
            assigned[best_idx] = true;
            --remaining;
        }

        // Update id_map for entries in left and right
        for (const auto& entry : left->entries) {
            id_map_[entry.id] = left.get();
        }
        for (const auto& entry : right->entries) {
            id_map_[entry.id] = right.get();
        }

        return {std::move(left), std::move(right)};
    }

    // ── Quadratic split (internal) ─────────────────────────────────────────

    /// Quadratic split for internal nodes (children instead of entries).
    auto split_internal_node(Node* node)
        -> std::pair<std::unique_ptr<Node>, std::unique_ptr<Node>> {
        auto& children = node->children;
        std::size_t n = children.size();

        std::size_t seed1 = 0;
        std::size_t seed2 = 1;
        float worst_waste = std::numeric_limits<float>::lowest();

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                float combined =
                    children[i]->bounds.union_volume(children[j]->bounds);
                float waste = combined - children[i]->bounds.volume() -
                              children[j]->bounds.volume();
                if (waste > worst_waste) {
                    worst_waste = waste;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }

        auto left = std::make_unique<Node>(false);
        auto right = std::make_unique<Node>(false);

        left->bounds = children[seed1]->bounds;
        right->bounds = children[seed2]->bounds;

        // Assign seeds
        std::vector<bool> assigned(n, false);
        assigned[seed1] = true;
        assigned[seed2] = true;

        children[seed1]->parent = left.get();
        left->children.push_back(std::move(children[seed1]));

        children[seed2]->parent = right.get();
        right->children.push_back(std::move(children[seed2]));

        std::size_t remaining = n - 2;

        while (remaining > 0) {
            if (left->children.size() + remaining == min_entries_) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!assigned[i]) {
                        left->bounds.expand(children[i]->bounds);
                        children[i]->parent = left.get();
                        left->children.push_back(std::move(children[i]));
                        assigned[i] = true;
                    }
                }
                break;
            }
            if (right->children.size() + remaining == min_entries_) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (!assigned[i]) {
                        right->bounds.expand(children[i]->bounds);
                        children[i]->parent = right.get();
                        right->children.push_back(std::move(children[i]));
                        assigned[i] = true;
                    }
                }
                break;
            }

            std::size_t best_idx = 0;
            float best_diff = std::numeric_limits<float>::lowest();
            for (std::size_t i = 0; i < n; ++i) {
                if (assigned[i]) {
                    continue;
                }
                float d1 = left->bounds.union_volume(children[i]->bounds) -
                           left->bounds.volume();
                float d2 = right->bounds.union_volume(children[i]->bounds) -
                           right->bounds.volume();
                float diff = std::abs(d1 - d2);
                if (diff > best_diff) {
                    best_diff = diff;
                    best_idx = i;
                }
            }

            float enlarge_left =
                left->bounds.union_volume(children[best_idx]->bounds) -
                left->bounds.volume();
            float enlarge_right =
                right->bounds.union_volume(children[best_idx]->bounds) -
                right->bounds.volume();

            if (enlarge_left < enlarge_right ||
                (enlarge_left == enlarge_right &&
                 left->children.size() <= right->children.size())) {
                left->bounds.expand(children[best_idx]->bounds);
                children[best_idx]->parent = left.get();
                left->children.push_back(std::move(children[best_idx]));
            } else {
                right->bounds.expand(children[best_idx]->bounds);
                children[best_idx]->parent = right.get();
                right->children.push_back(std::move(children[best_idx]));
            }
            assigned[best_idx] = true;
            --remaining;
        }

        return {std::move(left), std::move(right)};
    }

    // ── Condense tree after removal ────────────────────────────────────────

    auto condense_tree(Node* node) -> void {
        // Collect orphaned entries to reinsert
        std::vector<Entry> orphans;
        Node* current = node;

        while (current != root_.get()) {
            Node* parent = current->parent;

            bool underflow =
                (current->is_leaf && current->entries.empty()) ||
                (!current->is_leaf && current->children.empty());

            if (underflow) {
                // Collect all entries from this subtree
                collect_entries(current, orphans);
                // Remove this child from parent
                auto& siblings = parent->children;
                auto it =
                    std::find_if(siblings.begin(), siblings.end(),
                                 [current](const std::unique_ptr<Node>& c) {
                                     return c.get() == current;
                                 });
                if (it != siblings.end()) {
                    siblings.erase(it);
                }
                parent->recompute_bounds();
            } else {
                current->recompute_bounds();
            }
            current = parent;
        }

        // If root has only one child, make that child the root
        while (!root_->is_leaf && root_->children.size() == 1) {
            auto new_root = std::move(root_->children[0]);
            new_root->parent = nullptr;
            root_ = std::move(new_root);
        }

        // Reinsert orphans
        for (auto& entry : orphans) {
            auto* leaf = choose_leaf(root_.get(), entry.box);
            leaf->entries.push_back(entry);
            id_map_[entry.id] = leaf;
            if (leaf->entries.size() > max_entries_) {
                handle_overflow(leaf);
            }
        }
    }

    /// Collect all entries from a subtree.
    auto collect_entries(Node* node, std::vector<Entry>& out) -> void {
        if (node->is_leaf) {
            for (auto& entry : node->entries) {
                id_map_.erase(entry.id);
                out.push_back(std::move(entry));
            }
            node->entries.clear();
        } else {
            for (auto& child : node->children) {
                collect_entries(child.get(), out);
            }
            node->children.clear();
        }
    }

    // ── Query helpers ──────────────────────────────────────────────────────

    auto radius_search_recursive(
        const Node* node, const Point& query, float radius_sq,
        std::vector<std::pair<std::uint64_t, float>>& result) const -> void {
        if (node->is_leaf) {
            for (const auto& entry : node->entries) {
                float d = entry.box.min_distance_sq(query);
                if (d <= radius_sq) {
                    result.push_back({entry.id, d});
                }
            }
        } else {
            for (const auto& child : node->children) {
                if (child->bounds.min_distance_sq(query) <= radius_sq) {
                    radius_search_recursive(child.get(), query, radius_sq,
                                            result);
                }
            }
        }
    }

    template <typename Fn>
    auto query_intersects_recursive(const Node* node, const BoxType& query,
                                    Fn&& visitor) const -> void {
        if (node->is_leaf) {
            for (const auto& entry : node->entries) {
                if (entry.box.intersects(query)) {
                    visitor(entry.id, entry.box);
                }
            }
        } else {
            for (const auto& child : node->children) {
                if (child->bounds.intersects(query)) {
                    query_intersects_recursive(child.get(), query,
                                               std::forward<Fn>(visitor));
                }
            }
        }
    }

    auto query_contains_recursive(const Node* node, const Point& query,
                                  std::vector<std::uint64_t>& result) const
        -> void {
        if (node->is_leaf) {
            for (const auto& entry : node->entries) {
                if (entry.box.contains_point(query)) {
                    result.push_back(entry.id);
                }
            }
        } else {
            for (const auto& child : node->children) {
                if (child->bounds.contains_point(query)) {
                    query_contains_recursive(child.get(), query, result);
                }
            }
        }
    }

    // ── Bulk loading (Sort-Tile-Recursive) ─────────────────────────────────

    /// Pack a sorted vector of entries into leaf nodes of at most
    /// max_entries_ each.
    auto pack_leaves(std::vector<Entry>& entries)
        -> std::vector<std::unique_ptr<Node>> {
        std::vector<std::unique_ptr<Node>> leaves;
        for (std::size_t start = 0; start < entries.size();
             start += max_entries_) {
            std::size_t end =
                std::min(start + max_entries_, entries.size());
            auto leaf = std::make_unique<Node>(true);
            for (std::size_t i = start; i < end; ++i) {
                leaf->entries.push_back(std::move(entries[i]));
            }
            leaf->recompute_bounds();
            leaves.push_back(std::move(leaf));
        }
        return leaves;
    }

    /// Recursively group a vector of child nodes into an R-tree by sorting
    /// on successive dimensions and packing into internal nodes.
    auto group_nodes(std::vector<std::unique_ptr<Node>>& nodes,
                     std::size_t depth) -> std::unique_ptr<Node> {
        if (nodes.size() <= max_entries_) {
            auto internal = std::make_unique<Node>(false);
            for (auto& child : nodes) {
                child->parent = internal.get();
                internal->children.push_back(std::move(child));
            }
            internal->recompute_bounds();
            return internal;
        }

        std::size_t axis = depth % Dims;
        std::sort(nodes.begin(), nodes.end(),
                  [axis](const std::unique_ptr<Node>& a,
                         const std::unique_ptr<Node>& b) {
                      float ca = (a->bounds.min[axis] + a->bounds.max[axis]) *
                                 0.5F;
                      float cb = (b->bounds.min[axis] + b->bounds.max[axis]) *
                                 0.5F;
                      return ca < cb;
                  });

        std::vector<std::unique_ptr<Node>> groups;
        for (std::size_t start = 0; start < nodes.size();
             start += max_entries_) {
            std::size_t end = std::min(start + max_entries_, nodes.size());
            auto group = std::make_unique<Node>(false);
            for (std::size_t i = start; i < end; ++i) {
                nodes[i]->parent = group.get();
                group->children.push_back(std::move(nodes[i]));
            }
            group->recompute_bounds();
            groups.push_back(std::move(group));
        }

        if (groups.size() == 1) {
            return std::move(groups[0]);
        }

        return group_nodes(groups, depth + 1);
    }

    /// Sort-Tile-Recursive bulk load: sort entries by dimension, pack into
    /// leaves, then recursively group internal nodes.
    auto str_bulk_load(std::vector<Entry>& entries)
        -> std::unique_ptr<Node> {
        if (entries.size() <= max_entries_) {
            auto leaf = std::make_unique<Node>(true);
            leaf->entries = std::move(entries);
            leaf->recompute_bounds();
            return leaf;
        }

        // Sort entries by the first dimension's center coordinate
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) {
                      float ca = (a.box.min[0] + a.box.max[0]) * 0.5F;
                      float cb = (b.box.min[0] + b.box.max[0]) * 0.5F;
                      return ca < cb;
                  });

        // For multi-dimensional STR, sort slices by second dimension, etc.
        // Compute the number of slices per dimension
        auto n = entries.size();
        auto leaves_needed = (n + max_entries_ - 1) / max_entries_;
        auto slices_per_dim = static_cast<std::size_t>(
            std::ceil(std::pow(static_cast<double>(leaves_needed),
                               1.0 / static_cast<double>(Dims))));
        auto slice_size = slices_per_dim * max_entries_;

        // Sort within each first-dimension slice by second dimension, etc.
        if constexpr (Dims > 1) {
            for (std::size_t start = 0; start < n; start += slice_size) {
                std::size_t end = std::min(start + slice_size, n);
                auto begin_it = entries.begin() +
                                static_cast<std::ptrdiff_t>(start);
                auto end_it = entries.begin() +
                              static_cast<std::ptrdiff_t>(end);
                std::sort(begin_it, end_it,
                          [](const Entry& a, const Entry& b) {
                              float ca =
                                  (a.box.min[1] + a.box.max[1]) * 0.5F;
                              float cb =
                                  (b.box.min[1] + b.box.max[1]) * 0.5F;
                              return ca < cb;
                          });
            }
        }

        // Pack into leaf nodes
        auto leaves = pack_leaves(entries);

        if (leaves.size() == 1) {
            return std::move(leaves[0]);
        }

        // Recursively group leaves into internal nodes
        return group_nodes(leaves, 0);
    }

    // ── ID map rebuild ─────────────────────────────────────────────────────

    auto rebuild_id_map() -> void {
        id_map_.clear();
        rebuild_id_map_recursive(root_.get());
    }

    auto rebuild_id_map_recursive(Node* node) -> void {
        if (node->is_leaf) {
            for (const auto& entry : node->entries) {
                id_map_[entry.id] = node;
            }
        } else {
            for (auto& child : node->children) {
                child->parent = node;
                rebuild_id_map_recursive(child.get());
            }
        }
    }
};

} // namespace utils
