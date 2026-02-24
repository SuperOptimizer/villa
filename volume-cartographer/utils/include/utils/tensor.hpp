#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace utils {

enum class DType : std::uint8_t {
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
};

enum class Order : std::uint8_t { C, F };

[[nodiscard]] auto dtype_size(DType dt) noexcept -> std::size_t;
[[nodiscard]] auto dtype_name(DType dt) noexcept -> const char*;

template <typename T> constexpr auto dtype_of() noexcept -> DType;

template <> constexpr auto dtype_of<float>() noexcept -> DType { return DType::Float32; }
template <> constexpr auto dtype_of<double>() noexcept -> DType { return DType::Float64; }
template <> constexpr auto dtype_of<std::int8_t>() noexcept -> DType { return DType::Int8; }
template <> constexpr auto dtype_of<std::int16_t>() noexcept -> DType { return DType::Int16; }
template <> constexpr auto dtype_of<std::int32_t>() noexcept -> DType { return DType::Int32; }
template <> constexpr auto dtype_of<std::int64_t>() noexcept -> DType { return DType::Int64; }
template <> constexpr auto dtype_of<std::uint8_t>() noexcept -> DType { return DType::UInt8; }
template <> constexpr auto dtype_of<std::uint16_t>() noexcept -> DType { return DType::UInt16; }
template <> constexpr auto dtype_of<std::uint32_t>() noexcept -> DType { return DType::UInt32; }
template <> constexpr auto dtype_of<std::uint64_t>() noexcept -> DType { return DType::UInt64; }

struct Slice {
    std::ptrdiff_t start{0};
    std::ptrdiff_t stop{-1};
    std::ptrdiff_t step{1};
};

struct SliceArg {
    enum class Kind : std::uint8_t { Index, Slice };
    Kind kind;
    std::ptrdiff_t index{};
    Slice slice{};

    static auto idx(std::ptrdiff_t i) -> SliceArg {
        return {Kind::Index, i, {}};
    }
    static auto slc(std::ptrdiff_t start = 0, std::ptrdiff_t stop = -1,
                    std::ptrdiff_t step = 1) -> SliceArg {
        return {Kind::Slice, 0, {start, stop, step}};
    }
};

template <typename T>
class TensorView {
public:
    TensorView(T* data, std::vector<std::size_t> shape,
               std::vector<std::ptrdiff_t> strides)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {}

    template <typename... Indices>
    [[nodiscard]] auto operator()(Indices... indices) const -> T& {
        static_assert(sizeof...(Indices) > 0);
        const std::ptrdiff_t idx[] = {static_cast<std::ptrdiff_t>(indices)...};
        auto* ptr = reinterpret_cast<std::byte*>(data_);
        for (std::size_t i = 0; i < sizeof...(Indices); ++i) {
            ptr += idx[i] * strides_[i];
        }
        return *reinterpret_cast<T*>(ptr);
    }

    [[nodiscard]] auto data() const noexcept -> T* { return data_; }
    [[nodiscard]] auto shape() const noexcept -> const std::vector<std::size_t>& { return shape_; }
    [[nodiscard]] auto strides() const noexcept -> const std::vector<std::ptrdiff_t>& { return strides_; }

private:
    T* data_;
    std::vector<std::size_t> shape_;
    std::vector<std::ptrdiff_t> strides_;
};

class Tensor {
public:
    ~Tensor();
    Tensor(Tensor&& other) noexcept;
    auto operator=(Tensor&& other) noexcept -> Tensor&;
    Tensor(const Tensor& other);
    auto operator=(const Tensor& other) -> Tensor&;

    // Factories
    [[nodiscard]] static auto zeros(std::initializer_list<std::size_t> shape,
                                    DType dt = DType::Float32,
                                    Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto zeros(std::span<const std::size_t> shape,
                                    DType dt = DType::Float32,
                                    Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto ones(std::initializer_list<std::size_t> shape,
                                   DType dt = DType::Float32,
                                   Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto ones(std::span<const std::size_t> shape,
                                   DType dt = DType::Float32,
                                   Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto full(std::initializer_list<std::size_t> shape,
                                   double value,
                                   DType dt = DType::Float32,
                                   Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto full(std::span<const std::size_t> shape,
                                   double value,
                                   DType dt = DType::Float32,
                                   Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto from_data(const void* data, std::span<const std::size_t> shape,
                                        DType dt,
                                        Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto empty(std::initializer_list<std::size_t> shape,
                                    DType dt = DType::Float32,
                                    Order layout = Order::C) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto empty(std::span<const std::size_t> shape,
                                    DType dt = DType::Float32,
                                    Order layout = Order::C) -> std::expected<Tensor, std::string>;

    // Adapt: zero-copy view over external memory
    [[nodiscard]] static auto adapt(std::shared_ptr<void> storage, void* data,
                                    std::span<const std::size_t> shape, DType dt,
                                    Order layout = Order::C) -> Tensor;
    [[nodiscard]] static auto adapt(void* data, std::span<const std::size_t> shape, DType dt,
                                    Order layout = Order::C) -> Tensor;
    [[nodiscard]] static auto adapt(void* data, std::span<const std::size_t> shape,
                                    std::span<const std::ptrdiff_t> strides, DType dt) -> Tensor;

    // *_like: copy shape/dtype/layout from existing
    [[nodiscard]] static auto zeros_like(const Tensor& t) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto ones_like(const Tensor& t) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto full_like(const Tensor& t, double value) -> std::expected<Tensor, std::string>;
    [[nodiscard]] static auto empty_like(const Tensor& t) -> std::expected<Tensor, std::string>;

    // Properties
    [[nodiscard]] auto ndim() const noexcept -> std::size_t;
    [[nodiscard]] auto shape() const noexcept -> std::span<const std::size_t>;
    [[nodiscard]] auto strides() const noexcept -> std::span<const std::ptrdiff_t>;
    [[nodiscard]] auto dtype() const noexcept -> DType;
    [[nodiscard]] auto layout() const noexcept -> Order;
    [[nodiscard]] auto numel() const noexcept -> std::size_t;
    [[nodiscard]] auto nbytes() const noexcept -> std::size_t;
    [[nodiscard]] auto data_ptr() const noexcept -> void*;
    [[nodiscard]] auto is_contiguous() const noexcept -> bool;
    [[nodiscard]] auto is_contiguous(Order order) const noexcept -> bool;
    [[nodiscard]] auto use_count() const noexcept -> long;

    // Typed access
    template <typename T>
    [[nodiscard]] auto data() const noexcept -> T* {
        assert(dtype_of<T>() == dtype() && "dtype mismatch");
        return static_cast<T*>(data_ptr());
    }

    template <typename T>
    [[nodiscard]] auto view() const -> std::expected<TensorView<T>, std::string> {
        if (dtype_of<T>() != dtype()) {
            return std::unexpected("dtype mismatch: requested " +
                                   std::string(dtype_name(dtype_of<T>())) +
                                   " but tensor is " + dtype_name(dtype()));
        }
        return TensorView<T>(static_cast<T*>(data_ptr()),
                             std::vector<std::size_t>(shape().begin(), shape().end()),
                             std::vector<std::ptrdiff_t>(strides().begin(), strides().end()));
    }

    // In-place mutation
    auto fill_(double value) -> Tensor&;
    auto zero_() -> Tensor&;

    // Layout
    [[nodiscard]] auto reshape(std::span<const std::size_t> new_shape) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto transpose(std::size_t dim0, std::size_t dim1) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto squeeze(std::ptrdiff_t dim = -1) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto unsqueeze(std::size_t dim) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto permute(std::span<const std::size_t> dims) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto flatten() const -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto contiguous() const -> Tensor;
    [[nodiscard]] auto clone() const -> Tensor;

    // Type casting
    [[nodiscard]] auto to(DType target) const -> Tensor;
    [[nodiscard]] auto to(DType target, double scale, double offset = 0.0) const -> Tensor;

    // Slicing
    [[nodiscard]] auto slice(std::span<const SliceArg> args) const
        -> std::expected<Tensor, std::string>;
    [[nodiscard]] auto slice(std::initializer_list<SliceArg> args) const
        -> std::expected<Tensor, std::string>;

    // Comparison
    [[nodiscard]] auto operator==(const Tensor& other) const -> bool;
    [[nodiscard]] auto allclose(const Tensor& other, double rtol = 1e-5,
                                double atol = 1e-8) const -> bool;

    // I/O
    friend auto operator<<(std::ostream& os, const Tensor& t) -> std::ostream&;

    // Arithmetic (tensor-tensor)
    friend auto operator+(const Tensor& a, const Tensor& b) -> Tensor;
    friend auto operator-(const Tensor& a, const Tensor& b) -> Tensor;
    friend auto operator*(const Tensor& a, const Tensor& b) -> Tensor;
    friend auto operator/(const Tensor& a, const Tensor& b) -> Tensor;

    // Arithmetic (tensor-scalar)
    friend auto operator+(const Tensor& a, double b) -> Tensor;
    friend auto operator-(const Tensor& a, double b) -> Tensor;
    friend auto operator*(const Tensor& a, double b) -> Tensor;
    friend auto operator/(const Tensor& a, double b) -> Tensor;

    // Unary minus
    friend auto operator-(const Tensor& a) -> Tensor;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit Tensor(std::unique_ptr<Impl> impl);

    [[nodiscard]] static auto make_tensor(std::span<const std::size_t> shape, DType dt, Order layout,
                                          const std::function<void(void*, std::size_t)>& fill)
        -> std::expected<Tensor, std::string>;

    [[nodiscard]] static auto make_view(const Impl& src, void* data,
                                        std::vector<std::size_t> shape,
                                        std::vector<std::ptrdiff_t> strides) -> Tensor;
};

// Unary ops (free functions)
[[nodiscard]] auto abs(const Tensor& t) -> Tensor;
[[nodiscard]] auto neg(const Tensor& t) -> Tensor;
[[nodiscard]] auto sqrt(const Tensor& t) -> Tensor;
[[nodiscard]] auto clamp(const Tensor& t, double lo, double hi) -> Tensor;

// Reductions — global (return 0-dim scalar tensor)
[[nodiscard]] auto sum(const Tensor& t) -> Tensor;
[[nodiscard]] auto mean(const Tensor& t) -> Tensor;
[[nodiscard]] auto amin(const Tensor& t) -> Tensor;
[[nodiscard]] auto amax(const Tensor& t) -> Tensor;

// Reductions — along dimension (return tensor with that dim removed)
[[nodiscard]] auto sum(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string>;
[[nodiscard]] auto mean(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string>;
[[nodiscard]] auto amin(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string>;
[[nodiscard]] auto amax(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string>;

// DType promotion
[[nodiscard]] auto promote_dtype(DType a, DType b) -> DType;

// Creation — arange
[[nodiscard]] auto arange(double start, double stop, double step = 1.0,
                          DType dt = DType::Float64)
    -> std::expected<Tensor, std::string>;
[[nodiscard]] auto arange(double stop, DType dt = DType::Float64)
    -> std::expected<Tensor, std::string>;

// Creation — linspace
[[nodiscard]] auto linspace(double start, double stop, std::size_t n,
                            DType dt = DType::Float64)
    -> std::expected<Tensor, std::string>;

// Selection — where
[[nodiscard]] auto where(const Tensor& condition,
                         const Tensor& x, const Tensor& y)
    -> std::expected<Tensor, std::string>;

// Coordinate grids — meshgrid (ij indexing)
[[nodiscard]] auto meshgrid(std::span<const Tensor> tensors)
    -> std::expected<std::vector<Tensor>, std::string>;

// Creation — identity matrix
[[nodiscard]] auto eye(std::size_t n, DType dt = DType::Float64)
    -> std::expected<Tensor, std::string>;

// Joining
[[nodiscard]] auto concatenate(std::span<const Tensor> tensors, std::size_t axis = 0)
    -> std::expected<Tensor, std::string>;
[[nodiscard]] auto stack(std::span<const Tensor> tensors, std::size_t axis = 0)
    -> std::expected<Tensor, std::string>;

// Matrix multiplication (1D and 2D)
[[nodiscard]] auto matmul(const Tensor& a, const Tensor& b)
    -> std::expected<Tensor, std::string>;

// Index of min/max — global (flat index)
[[nodiscard]] auto argmin(const Tensor& t) -> std::size_t;
[[nodiscard]] auto argmax(const Tensor& t) -> std::size_t;

// Index of min/max — along dimension
[[nodiscard]] auto argmin(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string>;
[[nodiscard]] auto argmax(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string>;

// Boolean reductions
[[nodiscard]] auto any(const Tensor& t) -> bool;
[[nodiscard]] auto all(const Tensor& t) -> bool;

// Tiling
[[nodiscard]] auto tile(const Tensor& t, std::span<const std::size_t> reps)
    -> std::expected<Tensor, std::string>;

// Flip (reverse along axis)
[[nodiscard]] auto flip(const Tensor& t, std::size_t axis)
    -> std::expected<Tensor, std::string>;

// Padding (widths: flat span of 2*ndim, [before_0, after_0, before_1, after_1, ...])
[[nodiscard]] auto pad(const Tensor& t, std::span<const std::size_t> widths,
                       double value = 0.0)
    -> std::expected<Tensor, std::string>;

// Cumulative operations
[[nodiscard]] auto cumsum(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string>;
[[nodiscard]] auto cumprod(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string>;

// Sorting (ascending)
[[nodiscard]] auto sort(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string>;
[[nodiscard]] auto argsort(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string>;

// Diagonal (1D → 2D diagonal matrix, 2D → 1D diagonal extract)
[[nodiscard]] auto diag(const Tensor& t, std::ptrdiff_t offset = 0)
    -> std::expected<Tensor, std::string>;

} // namespace utils
