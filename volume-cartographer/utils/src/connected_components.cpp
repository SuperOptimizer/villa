#include "utils/connected_components.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace utils {
namespace {

class UnionFind {
public:
    explicit UnionFind(std::size_t capacity) : parent_(capacity), rank_(capacity, 0) {
        std::iota(parent_.begin(), parent_.end(), std::uint32_t{0});
    }

    auto find(std::uint32_t x) -> std::uint32_t {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    auto unite(std::uint32_t a, std::uint32_t b) -> void {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
    }

private:
    std::vector<std::uint32_t> parent_;
    std::vector<std::uint8_t> rank_;
};

struct NeighborInfo {
    std::ptrdiff_t linear_offset;               // byte offset in the output labels array
    std::vector<std::ptrdiff_t> coord_deltas;   // per-dimension delta for bounds check
};

auto compute_backward_neighbors(std::span<const std::size_t> shape,
                                Connectivity conn) -> std::vector<NeighborInfo> {
    auto ndim = shape.size();
    std::vector<NeighborInfo> neighbors;

    if (conn == Connectivity::Face) {
        // Face connectivity: differ in exactly 1 dimension by -1
        for (std::size_t d = 0; d < ndim; ++d) {
            if (shape[d] <= 1) continue;
            NeighborInfo ni;
            ni.coord_deltas.assign(ndim, 0);
            ni.coord_deltas[d] = -1;
            std::ptrdiff_t label_stride = 1;
            for (std::size_t i = ndim; i-- > d + 1;) {
                label_stride *= static_cast<std::ptrdiff_t>(shape[i]);
            }
            ni.linear_offset = -label_stride;
            neighbors.push_back(std::move(ni));
        }
    } else {
        // Full connectivity: all 3^ndim - 1 neighbors, backward half
        // A neighbor is "backward" if the first nonzero delta is negative
        std::vector<std::ptrdiff_t> deltas(ndim, -1);

        auto total = std::size_t{1};
        for (std::size_t d = 0; d < ndim; ++d) total *= 3;

        for (std::size_t i = 0; i < total; ++i) {
            // decode base-3
            std::vector<std::ptrdiff_t> d_vec(ndim);
            auto val = i;
            for (std::size_t d = ndim; d-- > 0;) {
                d_vec[d] = static_cast<std::ptrdiff_t>(val % 3) - 1;
                val /= 3;
            }

            // skip zero offset
            bool all_zero = true;
            for (auto v : d_vec) {
                if (v != 0) { all_zero = false; break; }
            }
            if (all_zero) continue;

            // check backward: first nonzero must be -1
            bool backward = false;
            for (auto v : d_vec) {
                if (v != 0) {
                    backward = (v < 0);
                    break;
                }
            }
            if (!backward) continue;

            // check that shape allows this neighbor (skip if dim size is 1 and delta != 0)
            bool valid = true;
            for (std::size_t d = 0; d < ndim; ++d) {
                if (shape[d] <= 1 && d_vec[d] != 0) { valid = false; break; }
            }
            if (!valid) continue;

            NeighborInfo ni;
            ni.coord_deltas = d_vec;

            // compute linear offset in label array (C-order)
            std::ptrdiff_t offset = 0;
            std::ptrdiff_t stride = 1;
            for (std::size_t d = ndim; d-- > 0;) {
                offset += d_vec[d] * stride;
                stride *= static_cast<std::ptrdiff_t>(shape[d]);
            }
            ni.linear_offset = offset;
            neighbors.push_back(std::move(ni));
        }
    }

    return neighbors;
}

auto is_integer_dtype(DType dt) -> bool {
    switch (dt) {
        case DType::Int8:
        case DType::Int16:
        case DType::Int32:
        case DType::Int64:
        case DType::UInt8:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
            return true;
        default:
            return false;
    }
}

// Get value at linear index as uint64 for comparison
template <typename T>
auto get_value(const void* data, std::size_t idx) -> std::uint64_t {
    return static_cast<std::uint64_t>(static_cast<const T*>(data)[idx]);
}

using GetValueFn = std::uint64_t(*)(const void*, std::size_t);

auto get_value_fn(DType dt) -> GetValueFn {
    switch (dt) {
        case DType::Int8:   return get_value<std::int8_t>;
        case DType::Int16:  return get_value<std::int16_t>;
        case DType::Int32:  return get_value<std::int32_t>;
        case DType::Int64:  return get_value<std::int64_t>;
        case DType::UInt8:  return get_value<std::uint8_t>;
        case DType::UInt16: return get_value<std::uint16_t>;
        case DType::UInt32: return get_value<std::uint32_t>;
        case DType::UInt64: return get_value<std::uint64_t>;
        default: return nullptr;
    }
}

// Compute multi-index from linear index
auto linear_to_coord(std::size_t linear, std::span<const std::size_t> shape)
    -> std::vector<std::ptrdiff_t> {
    auto ndim = shape.size();
    std::vector<std::ptrdiff_t> coord(ndim);
    for (std::size_t d = ndim; d-- > 0;) {
        coord[d] = static_cast<std::ptrdiff_t>(linear % shape[d]);
        linear /= shape[d];
    }
    return coord;
}

auto cc_impl(const Tensor& input, Connectivity conn, bool binary)
    -> std::expected<CCResult, std::string> {
    if (input.ndim() < 1) {
        return std::unexpected("connected_components: input must have at least 1 dimension");
    }
    if (!binary && !is_integer_dtype(input.dtype())) {
        return std::unexpected("connected_components: input must have integer dtype, got " +
                               std::string(dtype_name(input.dtype())));
    }

    auto shape = input.shape();
    auto numel = input.numel();

    // Make contiguous
    auto contiguous = input.contiguous();
    const void* in_data = contiguous.data_ptr();

    auto get_val = get_value_fn(contiguous.dtype());
    if (!get_val && !binary) {
        // For binary mode with float input, we just check nonzero
        // We'll handle this differently
    }

    // Allocate label output
    auto labels_result = Tensor::zeros(shape, DType::UInt32);
    if (!labels_result) {
        return std::unexpected(labels_result.error());
    }
    auto labels = std::move(*labels_result);
    auto* label_data = labels.data<std::uint32_t>();

    auto neighbors = compute_backward_neighbors(shape, conn);

    // Union-find
    // Max labels = numel (worst case: every voxel is separate)
    UnionFind uf(numel + 1); // 0 = background, labels start at 1
    std::uint32_t next_label = 1;

    // Forward pass
    for (std::size_t i = 0; i < numel; ++i) {
        bool foreground = false;
        std::uint64_t val = 0;

        if (binary) {
            // For binary, check nonzero by dispatching on dtype
            switch (contiguous.dtype()) {
                case DType::Float32: foreground = static_cast<const float*>(in_data)[i] != 0.0f; break;
                case DType::Float64: foreground = static_cast<const double*>(in_data)[i] != 0.0; break;
                case DType::Int8:    foreground = static_cast<const std::int8_t*>(in_data)[i] != 0; break;
                case DType::Int16:   foreground = static_cast<const std::int16_t*>(in_data)[i] != 0; break;
                case DType::Int32:   foreground = static_cast<const std::int32_t*>(in_data)[i] != 0; break;
                case DType::Int64:   foreground = static_cast<const std::int64_t*>(in_data)[i] != 0; break;
                case DType::UInt8:   foreground = static_cast<const std::uint8_t*>(in_data)[i] != 0; break;
                case DType::UInt16:  foreground = static_cast<const std::uint16_t*>(in_data)[i] != 0; break;
                case DType::UInt32:  foreground = static_cast<const std::uint32_t*>(in_data)[i] != 0; break;
                case DType::UInt64:  foreground = static_cast<const std::uint64_t*>(in_data)[i] != 0; break;
            }
        } else {
            val = get_val(in_data, i);
            foreground = (val != 0);
        }

        if (!foreground) {
            label_data[i] = 0;
            continue;
        }

        // Check backward neighbors
        auto coord = linear_to_coord(i, shape);
        std::uint32_t min_label = 0;

        for (auto& nb : neighbors) {
            // bounds check
            bool in_bounds = true;
            for (std::size_t d = 0; d < shape.size(); ++d) {
                auto nc = coord[d] + nb.coord_deltas[d];
                if (nc < 0 || nc >= static_cast<std::ptrdiff_t>(shape[d])) {
                    in_bounds = false;
                    break;
                }
            }
            if (!in_bounds) continue;

            auto nb_idx = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) + nb.linear_offset);
            auto nb_label = label_data[nb_idx];
            if (nb_label == 0) continue;

            if (!binary) {
                // Multi-label: only connect if same value
                auto nb_val = get_val(in_data, nb_idx);
                if (nb_val != val) continue;
            }

            auto root = uf.find(nb_label);
            if (min_label == 0) {
                min_label = root;
            } else {
                uf.unite(min_label, root);
                min_label = uf.find(min_label);
            }
        }

        if (min_label == 0) {
            label_data[i] = next_label++;
        } else {
            label_data[i] = min_label;
        }
    }

    // Flatten: map all labels to their roots
    // Then relabel sequentially
    std::vector<std::uint32_t> root_map(next_label, 0);
    std::uint32_t count = 0;
    for (std::uint32_t l = 1; l < next_label; ++l) {
        auto root = uf.find(l);
        if (root_map[root] == 0) {
            root_map[root] = ++count;
        }
        root_map[l] = root_map[root];
    }

    // Relabel
    for (std::size_t i = 0; i < numel; ++i) {
        if (label_data[i] != 0) {
            label_data[i] = root_map[uf.find(label_data[i])];
        }
    }

    return CCResult{std::move(labels), count};
}

} // namespace

auto connected_components(const Tensor& input, Connectivity conn)
    -> std::expected<CCResult, std::string> {
    return cc_impl(input, conn, false);
}

auto connected_components_binary(const Tensor& input, Connectivity conn)
    -> std::expected<CCResult, std::string> {
    return cc_impl(input, conn, true);
}

} // namespace utils
