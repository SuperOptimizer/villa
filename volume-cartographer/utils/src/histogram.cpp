#include "utils/histogram.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace utils {
namespace {

auto validate_input(const Tensor& input) -> std::expected<void, std::string> {
    if (input.ndim() != 2) {
        return std::unexpected("histogram: input must be 2D, got " +
                               std::to_string(input.ndim()) + "D");
    }
    auto dt = input.dtype();
    if (dt != DType::UInt8 && dt != DType::UInt16 && dt != DType::Float32) {
        return std::unexpected(
            "histogram: unsupported dtype " + std::string(dtype_name(dt)) +
            ", expected UInt8, UInt16, or Float32");
    }
    return {};
}

auto resolve_num_bins(const Tensor& input, std::size_t num_bins)
    -> std::expected<std::size_t, std::string> {
    auto dt = input.dtype();
    if (dt == DType::UInt8) {
        return (num_bins == 0) ? 256 : num_bins;
    }
    if (dt == DType::UInt16) {
        return (num_bins == 0) ? 65536 : num_bins;
    }
    // Float32
    if (num_bins == 0) {
        return std::unexpected(
            "histogram: num_bins must be specified for Float32 input");
    }
    return num_bins;
}

// Compute histogram for integer types by direct binning.
template <typename T>
auto histogram_integer(const Tensor& input, std::size_t num_bins) -> Tensor {
    auto cont = input.contiguous();
    const auto* data = cont.data<T>();
    auto numel = cont.numel();

    auto hist = Tensor::zeros({num_bins}, DType::Float64).value();
    auto* h = hist.data<double>();

    for (std::size_t i = 0; i < numel; ++i) {
        auto val = static_cast<std::size_t>(data[i]);
        if (val < num_bins) {
            h[val] += 1.0;
        }
    }

    return hist;
}

// Compute histogram for Float32 by linearly mapping [min,max] to bins.
auto histogram_float(const Tensor& input, std::size_t num_bins) -> Tensor {
    auto cont = input.contiguous();
    const auto* data = cont.data<float>();
    auto numel = cont.numel();

    // Find min and max
    auto fmin = std::numeric_limits<float>::max();
    auto fmax = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < numel; ++i) {
        if (data[i] < fmin) fmin = data[i];
        if (data[i] > fmax) fmax = data[i];
    }

    auto hist = Tensor::zeros({num_bins}, DType::Float64).value();
    auto* h = hist.data<double>();

    if (fmin == fmax) {
        // All values are the same -- put everything in the first bin
        h[0] = static_cast<double>(numel);
        return hist;
    }

    auto range = static_cast<double>(fmax) - static_cast<double>(fmin);
    auto bins_d = static_cast<double>(num_bins);

    for (std::size_t i = 0; i < numel; ++i) {
        auto normalized = (static_cast<double>(data[i]) - static_cast<double>(fmin)) / range;
        auto bin = static_cast<std::size_t>(normalized * bins_d);
        // Clamp the max value to the last bin
        if (bin >= num_bins) bin = num_bins - 1;
        h[bin] += 1.0;
    }

    return hist;
}

} // namespace

auto histogram(const Tensor& input, std::size_t num_bins)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());

    auto resolved = resolve_num_bins(input, num_bins);
    if (!resolved) return std::unexpected(resolved.error());
    auto bins = *resolved;

    switch (input.dtype()) {
        case DType::UInt8:
            return histogram_integer<std::uint8_t>(input, bins);
        case DType::UInt16:
            return histogram_integer<std::uint16_t>(input, bins);
        case DType::Float32:
            return histogram_float(input, bins);
        default:
            return std::unexpected("histogram: unsupported dtype");
    }
}

auto cumulative_histogram(const Tensor& input, std::size_t num_bins)
    -> std::expected<Tensor, std::string> {
    auto hist = histogram(input, num_bins);
    if (!hist) return hist;

    auto bins = hist->shape()[0];
    auto* h = hist->data<double>();

    // In-place cumulative sum
    for (std::size_t i = 1; i < bins; ++i) {
        h[i] += h[i - 1];
    }

    return hist;
}

auto histogram_equalize(const Tensor& input)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());

    // Use 256 bins for equalization regardless of input dtype
    constexpr std::size_t bins = 256;
    auto numel = input.numel();
    auto shape = input.shape();

    // Convert to UInt8 for histogram computation
    auto u8_input = input.to(DType::UInt8);
    auto cont = u8_input.contiguous();
    const auto* src = cont.data<std::uint8_t>();

    // Compute histogram
    double h[bins] = {};
    for (std::size_t i = 0; i < numel; ++i) {
        h[src[i]] += 1.0;
    }

    // Compute CDF
    double cdf[bins];
    cdf[0] = h[0];
    for (std::size_t i = 1; i < bins; ++i) {
        cdf[i] = cdf[i - 1] + h[i];
    }

    // Find minimum non-zero CDF value
    double cdf_min = 0.0;
    for (std::size_t i = 0; i < bins; ++i) {
        if (cdf[i] > 0.0) {
            cdf_min = cdf[i];
            break;
        }
    }

    // Build lookup table
    std::uint8_t lut[bins];
    auto total = static_cast<double>(numel);
    for (std::size_t i = 0; i < bins; ++i) {
        if (cdf[i] == 0.0) {
            lut[i] = 0;
        } else {
            auto val = (cdf[i] - cdf_min) / (total - cdf_min) * 255.0;
            lut[i] = static_cast<std::uint8_t>(
                std::clamp(std::round(val), 0.0, 255.0));
        }
    }

    // Apply lookup
    auto out = Tensor::zeros(shape, DType::UInt8);
    if (!out) return std::unexpected(out.error());
    auto* dst = out->data<std::uint8_t>();

    for (std::size_t i = 0; i < numel; ++i) {
        dst[i] = lut[src[i]];
    }

    return std::move(*out);
}

auto otsu_threshold(const Tensor& input)
    -> std::expected<double, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());

    // Use 256-bin histogram (convert to UInt8 range)
    constexpr std::size_t bins = 256;
    auto numel = input.numel();

    auto u8_input = input.to(DType::UInt8);
    auto cont = u8_input.contiguous();
    const auto* src = cont.data<std::uint8_t>();

    // Compute histogram
    double h[bins] = {};
    for (std::size_t i = 0; i < numel; ++i) {
        h[src[i]] += 1.0;
    }

    // Normalize to probabilities
    auto total = static_cast<double>(numel);
    double p[bins];
    for (std::size_t i = 0; i < bins; ++i) {
        p[i] = h[i] / total;
    }

    // Compute total mean
    double mu_total = 0.0;
    for (std::size_t i = 0; i < bins; ++i) {
        mu_total += static_cast<double>(i) * p[i];
    }

    // Find threshold that maximizes inter-class variance
    double best_variance = -1.0;
    double best_thresh = 0.0;

    double w0 = 0.0;
    double mu0_sum = 0.0;

    for (std::size_t t = 0; t < bins - 1; ++t) {
        w0 += p[t];
        mu0_sum += static_cast<double>(t) * p[t];

        if (w0 == 0.0) continue;

        auto w1 = 1.0 - w0;
        if (w1 == 0.0) break;

        auto mu0 = mu0_sum / w0;
        auto mu1 = (mu_total - mu0_sum) / w1;

        auto diff = mu0 - mu1;
        auto variance = w0 * w1 * diff * diff;

        if (variance > best_variance) {
            best_variance = variance;
            best_thresh = static_cast<double>(t);
        }
    }

    return best_thresh;
}

auto threshold(const Tensor& input, double thresh)
    -> std::expected<Tensor, std::string> {
    if (auto v = validate_input(input); !v) return std::unexpected(v.error());

    auto shape = input.shape();
    auto numel = input.numel();
    auto cont = input.contiguous();

    auto out = Tensor::zeros(shape, DType::UInt8);
    if (!out) return std::unexpected(out.error());
    auto* dst = out->data<std::uint8_t>();

    switch (input.dtype()) {
        case DType::UInt8: {
            const auto* src = cont.data<std::uint8_t>();
            for (std::size_t i = 0; i < numel; ++i) {
                dst[i] = (static_cast<double>(src[i]) >= thresh) ? 255 : 0;
            }
            break;
        }
        case DType::UInt16: {
            const auto* src = cont.data<std::uint16_t>();
            for (std::size_t i = 0; i < numel; ++i) {
                dst[i] = (static_cast<double>(src[i]) >= thresh) ? 255 : 0;
            }
            break;
        }
        case DType::Float32: {
            const auto* src = cont.data<float>();
            for (std::size_t i = 0; i < numel; ++i) {
                dst[i] = (static_cast<double>(src[i]) >= thresh) ? 255 : 0;
            }
            break;
        }
        default:
            return std::unexpected("threshold: unsupported dtype");
    }

    return std::move(*out);
}

} // namespace utils
