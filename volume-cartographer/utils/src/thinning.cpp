#include "utils/thinning.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace utils {
namespace {

// Zhang-Suen thinning algorithm.
// Operates on a working buffer where 1 = foreground, 0 = background.
// Iterates two sub-iterations until no pixels are removed.
//
// Neighborhood labeling (3x3):
//   P9 P2 P3
//   P8 P1 P4
//   P7 P6 P5
//
// Sub-iteration 1 conditions (all must hold to mark P1 for deletion):
//   (a) 2 <= B(P1) <= 6
//   (b) A(P1) == 1
//   (c) P2 * P4 * P6 == 0
//   (d) P4 * P6 * P8 == 0
//
// Sub-iteration 2 conditions:
//   (a) 2 <= B(P1) <= 6
//   (b) A(P1) == 1
//   (c) P2 * P4 * P8 == 0
//   (d) P2 * P6 * P8 == 0
//
// B(P1) = number of nonzero neighbors among P2..P9
// A(P1) = number of 0->1 transitions in the ordered sequence P2,P3,...,P9,P2

auto zhang_suen(std::vector<std::uint8_t>& img, std::size_t rows, std::size_t cols) -> void {
    if (rows < 3 || cols < 3) return;

    std::vector<std::size_t> to_remove;
    bool changed = true;

    while (changed) {
        changed = false;

        // --- Sub-iteration 1 ---
        to_remove.clear();
        for (std::size_t r = 1; r + 1 < rows; ++r) {
            for (std::size_t c = 1; c + 1 < cols; ++c) {
                if (img[r * cols + c] == 0) continue;

                // Neighbors: P2..P9 in clockwise order starting from top
                auto p2 = img[(r - 1) * cols + c];
                auto p3 = img[(r - 1) * cols + (c + 1)];
                auto p4 = img[r * cols + (c + 1)];
                auto p5 = img[(r + 1) * cols + (c + 1)];
                auto p6 = img[(r + 1) * cols + c];
                auto p7 = img[(r + 1) * cols + (c - 1)];
                auto p8 = img[r * cols + (c - 1)];
                auto p9 = img[(r - 1) * cols + (c - 1)];

                // B(P1): count of nonzero neighbors
                int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (b < 2 || b > 6) continue;

                // A(P1): 0->1 transitions in P2,P3,P4,P5,P6,P7,P8,P9,P2
                int a = 0;
                if (p2 == 0 && p3 == 1) a++;
                if (p3 == 0 && p4 == 1) a++;
                if (p4 == 0 && p5 == 1) a++;
                if (p5 == 0 && p6 == 1) a++;
                if (p6 == 0 && p7 == 1) a++;
                if (p7 == 0 && p8 == 1) a++;
                if (p8 == 0 && p9 == 1) a++;
                if (p9 == 0 && p2 == 1) a++;
                if (a != 1) continue;

                // Conditions (c) and (d) for sub-iteration 1
                if (p2 * p4 * p6 != 0) continue;
                if (p4 * p6 * p8 != 0) continue;

                to_remove.push_back(r * cols + c);
            }
        }
        for (auto idx : to_remove) {
            img[idx] = 0;
            changed = true;
        }

        // --- Sub-iteration 2 ---
        to_remove.clear();
        for (std::size_t r = 1; r + 1 < rows; ++r) {
            for (std::size_t c = 1; c + 1 < cols; ++c) {
                if (img[r * cols + c] == 0) continue;

                auto p2 = img[(r - 1) * cols + c];
                auto p3 = img[(r - 1) * cols + (c + 1)];
                auto p4 = img[r * cols + (c + 1)];
                auto p5 = img[(r + 1) * cols + (c + 1)];
                auto p6 = img[(r + 1) * cols + c];
                auto p7 = img[(r + 1) * cols + (c - 1)];
                auto p8 = img[r * cols + (c - 1)];
                auto p9 = img[(r - 1) * cols + (c - 1)];

                int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (b < 2 || b > 6) continue;

                int a = 0;
                if (p2 == 0 && p3 == 1) a++;
                if (p3 == 0 && p4 == 1) a++;
                if (p4 == 0 && p5 == 1) a++;
                if (p5 == 0 && p6 == 1) a++;
                if (p6 == 0 && p7 == 1) a++;
                if (p7 == 0 && p8 == 1) a++;
                if (p8 == 0 && p9 == 1) a++;
                if (p9 == 0 && p2 == 1) a++;
                if (a != 1) continue;

                // Conditions (c) and (d) for sub-iteration 2
                if (p2 * p4 * p8 != 0) continue;
                if (p2 * p6 * p8 != 0) continue;

                to_remove.push_back(r * cols + c);
            }
        }
        for (auto idx : to_remove) {
            img[idx] = 0;
            changed = true;
        }
    }
}

} // namespace

auto thin(const Tensor& input) -> std::expected<Tensor, std::string> {
    if (input.ndim() != 2) {
        return std::unexpected("thin: input must be 2D, got " +
                               std::to_string(input.ndim()) + "D");
    }
    if (input.dtype() != DType::UInt8) {
        return std::unexpected("thin: input must have UInt8 dtype, got " +
                               std::string(dtype_name(input.dtype())));
    }

    auto shape = input.shape();
    auto rows = shape[0];
    auto cols = shape[1];

    // Build working buffer: binarize (nonzero -> 1, zero -> 0)
    auto contiguous = input.contiguous();
    const auto* src = contiguous.data<std::uint8_t>();
    auto numel = rows * cols;

    std::vector<std::uint8_t> work(numel);
    for (std::size_t i = 0; i < numel; ++i) {
        work[i] = (src[i] != 0) ? std::uint8_t{1} : std::uint8_t{0};
    }

    // Run Zhang-Suen thinning
    zhang_suen(work, rows, cols);

    // Create output tensor: skeleton pixels = 255
    auto out = Tensor::zeros(shape, DType::UInt8);
    if (!out) {
        return std::unexpected(out.error());
    }
    auto* dst = out->data<std::uint8_t>();
    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = (work[i] != 0) ? std::uint8_t{255} : std::uint8_t{0};
    }

    return std::move(*out);
}

} // namespace utils
