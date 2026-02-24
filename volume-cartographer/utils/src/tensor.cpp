#include "utils/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace utils {

auto dtype_size(DType dt) noexcept -> std::size_t {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int8:    return 1;
        case DType::Int16:   return 2;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
        case DType::UInt8:   return 1;
        case DType::UInt16:  return 2;
        case DType::UInt32:  return 4;
        case DType::UInt64:  return 8;
    }
    return 0;
}

auto dtype_name(DType dt) noexcept -> const char* {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int8:    return "int8";
        case DType::Int16:   return "int16";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        case DType::UInt8:   return "uint8";
        case DType::UInt16:  return "uint16";
        case DType::UInt32:  return "uint32";
        case DType::UInt64:  return "uint64";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// DType helpers
// ---------------------------------------------------------------------------

static auto is_float_dtype(DType dt) -> bool {
    return dt == DType::Float32 || dt == DType::Float64;
}

static auto is_signed_dtype(DType dt) -> bool {
    return dt == DType::Int8 || dt == DType::Int16 || dt == DType::Int32 || dt == DType::Int64;
}

static auto dtype_width(DType dt) -> int {
    switch (dt) {
        case DType::Int8:  case DType::UInt8:  return 8;
        case DType::Int16: case DType::UInt16: return 16;
        case DType::Int32: case DType::UInt32: return 32;
        case DType::Int64: case DType::UInt64: return 64;
        case DType::Float32: return 32;
        case DType::Float64: return 64;
    }
    return 0;
}

auto promote_dtype(DType a, DType b) -> DType {
    if (a == b) return a;

    // If either is float, result is float (wider wins)
    if (is_float_dtype(a) || is_float_dtype(b)) {
        if (a == DType::Float64 || b == DType::Float64) return DType::Float64;
        return DType::Float32;
    }

    // Both integer types
    int wa = dtype_width(a), wb = dtype_width(b);
    bool sa = is_signed_dtype(a), sb = is_signed_dtype(b);

    // Same signedness: pick wider
    if (sa == sb) {
        return wa >= wb ? a : b;
    }

    // Mixed sign: signed type at max width
    int max_width = std::max(wa, wb);
    switch (max_width) {
        case 8:  return DType::Int8;
        case 16: return DType::Int16;
        case 32: return DType::Int32;
        default: return DType::Int64;
    }
}

static auto reduction_dtype(DType dt) -> DType {
    if (is_float_dtype(dt)) return DType::Float64;
    if (is_signed_dtype(dt)) return DType::Int64;
    return DType::UInt64;
}

// ---------------------------------------------------------------------------
// Element read/write as double
// ---------------------------------------------------------------------------

static auto read_as_double(const void* p, DType dt) -> double {
    switch (dt) {
        case DType::Float32: return static_cast<double>(*static_cast<const float*>(p));
        case DType::Float64: return *static_cast<const double*>(p);
        case DType::Int8:    return static_cast<double>(*static_cast<const std::int8_t*>(p));
        case DType::Int16:   return static_cast<double>(*static_cast<const std::int16_t*>(p));
        case DType::Int32:   return static_cast<double>(*static_cast<const std::int32_t*>(p));
        case DType::Int64:   return static_cast<double>(*static_cast<const std::int64_t*>(p));
        case DType::UInt8:   return static_cast<double>(*static_cast<const std::uint8_t*>(p));
        case DType::UInt16:  return static_cast<double>(*static_cast<const std::uint16_t*>(p));
        case DType::UInt32:  return static_cast<double>(*static_cast<const std::uint32_t*>(p));
        case DType::UInt64:  return static_cast<double>(*static_cast<const std::uint64_t*>(p));
    }
    __builtin_unreachable();
}

static void write_from_double(void* p, DType dt, double val) {
    switch (dt) {
        case DType::Float32: *static_cast<float*>(p) = static_cast<float>(val); return;
        case DType::Float64: *static_cast<double*>(p) = val; return;
        case DType::Int8:    *static_cast<std::int8_t*>(p) = static_cast<std::int8_t>(val); return;
        case DType::Int16:   *static_cast<std::int16_t*>(p) = static_cast<std::int16_t>(val); return;
        case DType::Int32:   *static_cast<std::int32_t*>(p) = static_cast<std::int32_t>(val); return;
        case DType::Int64:   *static_cast<std::int64_t*>(p) = static_cast<std::int64_t>(val); return;
        case DType::UInt8:   *static_cast<std::uint8_t*>(p) = static_cast<std::uint8_t>(val); return;
        case DType::UInt16:  *static_cast<std::uint16_t*>(p) = static_cast<std::uint16_t>(val); return;
        case DType::UInt32:  *static_cast<std::uint32_t*>(p) = static_cast<std::uint32_t>(val); return;
        case DType::UInt64:  *static_cast<std::uint64_t*>(p) = static_cast<std::uint64_t>(val); return;
    }
}

// ---------------------------------------------------------------------------
// Strided index helpers
// ---------------------------------------------------------------------------

static auto compute_offset(std::span<const std::ptrdiff_t> strides,
                           const std::size_t* indices, std::size_t ndim) -> std::ptrdiff_t {
    std::ptrdiff_t offset = 0;
    for (std::size_t i = 0; i < ndim; ++i) {
        offset += static_cast<std::ptrdiff_t>(indices[i]) * strides[i];
    }
    return offset;
}

static void advance_indices(std::size_t* indices,
                            std::span<const std::size_t> shape) {
    for (auto i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
        auto dim = static_cast<std::size_t>(i);
        if (++indices[dim] < shape[dim]) return;
        indices[dim] = 0;
    }
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

static constexpr std::size_t kAlignment = 64;

struct Tensor::Impl {
    std::shared_ptr<void> storage;
    void* data;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    DType dtype;
    std::size_t numel;
    Order layout{Order::C};

    static auto make_storage(std::size_t nbytes) -> std::shared_ptr<void> {
        if (nbytes == 0) nbytes = kAlignment; // always allocate something
        void* ptr = std::aligned_alloc(kAlignment, (nbytes + kAlignment - 1) / kAlignment * kAlignment);
        if (!ptr) return nullptr;
        return {ptr, std::free};
    }

    static auto compute_row_major_strides(std::span<const std::size_t> shape,
                                          std::size_t elem_size) -> std::vector<std::ptrdiff_t> {
        if (shape.empty()) return {};
        std::vector<std::ptrdiff_t> strides(shape.size());
        auto stride = static_cast<std::ptrdiff_t>(elem_size);
        for (auto i = static_cast<std::ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
            strides[static_cast<std::size_t>(i)] = stride;
            stride *= static_cast<std::ptrdiff_t>(shape[static_cast<std::size_t>(i)]);
        }
        return strides;
    }

    static auto compute_column_major_strides(std::span<const std::size_t> shape,
                                             std::size_t elem_size) -> std::vector<std::ptrdiff_t> {
        if (shape.empty()) return {};
        std::vector<std::ptrdiff_t> strides(shape.size());
        auto stride = static_cast<std::ptrdiff_t>(elem_size);
        for (std::size_t i = 0; i < shape.size(); ++i) {
            strides[i] = stride;
            stride *= static_cast<std::ptrdiff_t>(shape[i]);
        }
        return strides;
    }

    static auto compute_strides(std::span<const std::size_t> shape,
                                std::size_t elem_size, Order layout) -> std::vector<std::ptrdiff_t> {
        return layout == Order::C
            ? compute_row_major_strides(shape, elem_size)
            : compute_column_major_strides(shape, elem_size);
    }

    static auto compute_numel(std::span<const std::size_t> shape) -> std::size_t {
        if (shape.empty()) return 1; // scalar
        std::size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

// ---------------------------------------------------------------------------
// Tensor special members
// ---------------------------------------------------------------------------

Tensor::Tensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tensor::~Tensor() = default;
Tensor::Tensor(Tensor&& other) noexcept = default;
auto Tensor::operator=(Tensor&& other) noexcept -> Tensor& = default;

Tensor::Tensor(const Tensor& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {
    // Deep copy: allocate new storage and copy data
    auto nbytes = impl_->numel * dtype_size(impl_->dtype);
    impl_->storage = Impl::make_storage(nbytes);
    impl_->data = impl_->storage.get();
    if (other.is_contiguous()) {
        std::memcpy(impl_->data, other.impl_->data, nbytes);
    } else {
        auto contig = other.contiguous();
        std::memcpy(impl_->data, contig.impl_->data, nbytes);
    }
    impl_->strides = Impl::compute_strides(impl_->shape, dtype_size(impl_->dtype), impl_->layout);
}

auto Tensor::operator=(const Tensor& other) -> Tensor& {
    if (this != &other) {
        auto tmp = Tensor(other);
        std::swap(impl_, tmp.impl_);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Factories
// ---------------------------------------------------------------------------

auto Tensor::make_tensor(std::span<const std::size_t> shape, DType dt, Order layout,
                         const std::function<void(void*, std::size_t)>& fill)
    -> std::expected<Tensor, std::string> {
    for (auto s : shape) {
        if (s == 0) return std::unexpected("shape dimensions must be non-zero");
    }

    auto impl = std::make_unique<Tensor::Impl>();
    impl->dtype = dt;
    impl->layout = layout;
    impl->shape.assign(shape.begin(), shape.end());
    impl->numel = Tensor::Impl::compute_numel(shape);
    impl->strides = Tensor::Impl::compute_strides(shape, dtype_size(dt), layout);
    auto nbytes = impl->numel * dtype_size(dt);
    impl->storage = Tensor::Impl::make_storage(nbytes);
    if (!impl->storage) return std::unexpected("allocation failed");
    impl->data = impl->storage.get();

    fill(impl->data, nbytes);

    return Tensor(std::move(impl));
}

auto Tensor::zeros(std::initializer_list<std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return zeros(std::span<const std::size_t>(shape.begin(), shape.size()), dt, layout);
}

auto Tensor::zeros(std::span<const std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return make_tensor(shape, dt, layout, [](void* p, std::size_t n) { std::memset(p, 0, n); });
}

auto Tensor::ones(std::initializer_list<std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return ones(std::span<const std::size_t>(shape.begin(), shape.size()), dt, layout);
}

auto Tensor::ones(std::span<const std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return full(shape, 1.0, dt, layout);
}

auto Tensor::full(std::initializer_list<std::size_t> shape, double value, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return full(std::span<const std::size_t>(shape.begin(), shape.size()), value, dt, layout);
}

auto Tensor::full(std::span<const std::size_t> shape, double value, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return make_tensor(shape, dt, layout, [value, dt](void* p, std::size_t n) {
        auto elem_sz = dtype_size(dt);
        auto count = n / elem_sz;

        switch (dt) {
            case DType::Float32: {
                auto v = static_cast<float>(value);
                auto* t = static_cast<float*>(p);
                std::fill_n(t, count, v);
                break;
            }
            case DType::Float64: {
                auto* t = static_cast<double*>(p);
                std::fill_n(t, count, value);
                break;
            }
            case DType::Int8: {
                auto v = static_cast<std::int8_t>(value);
                std::fill_n(static_cast<std::int8_t*>(p), count, v);
                break;
            }
            case DType::Int16: {
                auto v = static_cast<std::int16_t>(value);
                std::fill_n(static_cast<std::int16_t*>(p), count, v);
                break;
            }
            case DType::Int32: {
                auto v = static_cast<std::int32_t>(value);
                std::fill_n(static_cast<std::int32_t*>(p), count, v);
                break;
            }
            case DType::Int64: {
                auto v = static_cast<std::int64_t>(value);
                std::fill_n(static_cast<std::int64_t*>(p), count, v);
                break;
            }
            case DType::UInt8: {
                auto v = static_cast<std::uint8_t>(value);
                std::fill_n(static_cast<std::uint8_t*>(p), count, v);
                break;
            }
            case DType::UInt16: {
                auto v = static_cast<std::uint16_t>(value);
                std::fill_n(static_cast<std::uint16_t*>(p), count, v);
                break;
            }
            case DType::UInt32: {
                auto v = static_cast<std::uint32_t>(value);
                std::fill_n(static_cast<std::uint32_t*>(p), count, v);
                break;
            }
            case DType::UInt64: {
                auto v = static_cast<std::uint64_t>(value);
                std::fill_n(static_cast<std::uint64_t*>(p), count, v);
                break;
            }
        }
    });
}

auto Tensor::from_data(const void* data, std::span<const std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    if (!data) return std::unexpected("data pointer is null");
    return make_tensor(shape, dt, layout, [data](void* p, std::size_t n) {
        std::memcpy(p, data, n);
    });
}

auto Tensor::empty(std::initializer_list<std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return empty(std::span<const std::size_t>(shape.begin(), shape.size()), dt, layout);
}

auto Tensor::empty(std::span<const std::size_t> shape, DType dt, Order layout)
    -> std::expected<Tensor, std::string> {
    return make_tensor(shape, dt, layout, [](void*, std::size_t) {});
}

// ---------------------------------------------------------------------------
// Adapt (zero-copy)
// ---------------------------------------------------------------------------

auto Tensor::adapt(std::shared_ptr<void> storage, void* data,
                   std::span<const std::size_t> shape, DType dt,
                   Order layout) -> Tensor {
    assert(data && "adapt: data pointer must not be null");
    auto impl = std::make_unique<Impl>();
    impl->storage = std::move(storage);
    impl->data = data;
    impl->shape.assign(shape.begin(), shape.end());
    impl->strides = Impl::compute_strides(shape, dtype_size(dt), layout);
    impl->dtype = dt;
    impl->numel = Impl::compute_numel(shape);
    impl->layout = layout;
    return Tensor(std::move(impl));
}

auto Tensor::adapt(void* data, std::span<const std::size_t> shape, DType dt,
                   Order layout) -> Tensor {
    assert(data && "adapt: data pointer must not be null");
    auto impl = std::make_unique<Impl>();
    impl->storage = std::shared_ptr<void>(data, [](void*) {}); // non-owning
    impl->data = data;
    impl->shape.assign(shape.begin(), shape.end());
    impl->strides = Impl::compute_strides(shape, dtype_size(dt), layout);
    impl->dtype = dt;
    impl->numel = Impl::compute_numel(shape);
    impl->layout = layout;
    return Tensor(std::move(impl));
}

auto Tensor::adapt(void* data, std::span<const std::size_t> shape,
                   std::span<const std::ptrdiff_t> strides, DType dt) -> Tensor {
    assert(data && "adapt: data pointer must not be null");
    auto impl = std::make_unique<Impl>();
    impl->storage = std::shared_ptr<void>(data, [](void*) {}); // non-owning
    impl->data = data;
    impl->shape.assign(shape.begin(), shape.end());
    impl->strides.assign(strides.begin(), strides.end());
    impl->dtype = dt;
    impl->numel = Impl::compute_numel(shape);
    impl->layout = Order::C; // arbitrary for explicit strides
    return Tensor(std::move(impl));
}

// ---------------------------------------------------------------------------
// *_like factories
// ---------------------------------------------------------------------------

auto Tensor::zeros_like(const Tensor& t) -> std::expected<Tensor, std::string> {
    return zeros(t.shape(), t.dtype(), t.layout());
}

auto Tensor::ones_like(const Tensor& t) -> std::expected<Tensor, std::string> {
    return ones(t.shape(), t.dtype(), t.layout());
}

auto Tensor::full_like(const Tensor& t, double value) -> std::expected<Tensor, std::string> {
    return full(t.shape(), value, t.dtype(), t.layout());
}

auto Tensor::empty_like(const Tensor& t) -> std::expected<Tensor, std::string> {
    return empty(t.shape(), t.dtype(), t.layout());
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

auto Tensor::ndim() const noexcept -> std::size_t { return impl_->shape.size(); }

auto Tensor::shape() const noexcept -> std::span<const std::size_t> { return impl_->shape; }

auto Tensor::strides() const noexcept -> std::span<const std::ptrdiff_t> { return impl_->strides; }

auto Tensor::dtype() const noexcept -> DType { return impl_->dtype; }

auto Tensor::layout() const noexcept -> Order { return impl_->layout; }

auto Tensor::numel() const noexcept -> std::size_t { return impl_->numel; }

auto Tensor::nbytes() const noexcept -> std::size_t {
    return impl_->numel * dtype_size(impl_->dtype);
}

auto Tensor::data_ptr() const noexcept -> void* { return impl_->data; }

auto Tensor::is_contiguous() const noexcept -> bool {
    if (impl_->shape.empty()) return true;
    auto expected = Impl::compute_strides(impl_->shape, dtype_size(impl_->dtype), impl_->layout);
    return impl_->strides == expected;
}

auto Tensor::is_contiguous(Order order) const noexcept -> bool {
    if (impl_->shape.empty()) return true;
    auto expected = Impl::compute_strides(impl_->shape, dtype_size(impl_->dtype), order);
    return impl_->strides == expected;
}

auto Tensor::use_count() const noexcept -> long { return impl_->storage.use_count(); }

// ---------------------------------------------------------------------------
// View helper (shares storage, different shape/strides/data_ptr)
// ---------------------------------------------------------------------------

auto Tensor::make_view(const Impl& src, void* data,
                       std::vector<std::size_t> shape,
                       std::vector<std::ptrdiff_t> strides) -> Tensor {
    auto impl = std::make_unique<Tensor::Impl>();
    impl->storage = src.storage;
    impl->data = data;
    impl->shape = std::move(shape);
    impl->strides = std::move(strides);
    impl->dtype = src.dtype;
    impl->numel = Tensor::Impl::compute_numel(impl->shape);
    impl->layout = src.layout;
    return Tensor(std::move(impl));
}

// ---------------------------------------------------------------------------
// Element iteration helpers
// ---------------------------------------------------------------------------

// Calls fn(byte_ptr) for each element in strided order
static void for_each_element(void* data,
                             std::span<const std::size_t> shape,
                             std::span<const std::ptrdiff_t> strides,
                             const std::function<void(std::byte*)>& fn) {
    if (shape.empty()) {
        fn(static_cast<std::byte*>(data));
        return;
    }
    auto* base = static_cast<std::byte*>(data);
    if (shape.size() == 1) {
        for (std::size_t i = 0; i < shape[0]; ++i) {
            fn(base + static_cast<std::ptrdiff_t>(i) * strides[0]);
        }
        return;
    }
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for_each_element(base + static_cast<std::ptrdiff_t>(i) * strides[0],
                         shape.subspan(1), strides.subspan(1), fn);
    }
}

// Dispatch a function across all dtype types
template <typename Fn>
static auto dispatch_dtype(DType dt, Fn&& fn) {
    switch (dt) {
        case DType::Float32: return fn.template operator()<float>();
        case DType::Float64: return fn.template operator()<double>();
        case DType::Int8:    return fn.template operator()<std::int8_t>();
        case DType::Int16:   return fn.template operator()<std::int16_t>();
        case DType::Int32:   return fn.template operator()<std::int32_t>();
        case DType::Int64:   return fn.template operator()<std::int64_t>();
        case DType::UInt8:   return fn.template operator()<std::uint8_t>();
        case DType::UInt16:  return fn.template operator()<std::uint16_t>();
        case DType::UInt32:  return fn.template operator()<std::uint32_t>();
        case DType::UInt64:  return fn.template operator()<std::uint64_t>();
    }
    __builtin_unreachable();
}

// ---------------------------------------------------------------------------
// In-place mutation
// ---------------------------------------------------------------------------

auto Tensor::fill_(double value) -> Tensor& {
    dispatch_dtype(impl_->dtype, [&]<typename T>() {
        auto v = static_cast<T>(value);
        if (is_contiguous()) {
            std::fill_n(static_cast<T*>(impl_->data), impl_->numel, v);
        } else {
            for_each_element(impl_->data, impl_->shape, impl_->strides,
                [v](std::byte* p) { *reinterpret_cast<T*>(p) = v; });
        }
    });
    return *this;
}

auto Tensor::zero_() -> Tensor& {
    if (is_contiguous()) {
        std::memset(impl_->data, 0, nbytes());
    } else {
        auto elem_sz = dtype_size(impl_->dtype);
        for_each_element(impl_->data, impl_->shape, impl_->strides,
            [elem_sz](std::byte* p) { std::memset(p, 0, elem_sz); });
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

auto Tensor::reshape(std::span<const std::size_t> new_shape) const
    -> std::expected<Tensor, std::string> {
    // Compute numel of new shape
    std::size_t new_numel = 1;
    for (auto s : new_shape) {
        if (s == 0) return std::unexpected("shape dimensions must be non-zero");
        new_numel *= s;
    }
    if (new_numel != impl_->numel) {
        return std::unexpected("cannot reshape tensor of " + std::to_string(impl_->numel) +
                               " elements into shape with " + std::to_string(new_numel) + " elements");
    }
    if (!is_contiguous()) {
        return std::unexpected("reshape requires a contiguous tensor");
    }
    auto new_strides = Impl::compute_strides(new_shape, dtype_size(impl_->dtype), impl_->layout);
    return make_view(*impl_, impl_->data,
                     {new_shape.begin(), new_shape.end()},
                     std::move(new_strides));
}

auto Tensor::transpose(std::size_t dim0, std::size_t dim1) const
    -> std::expected<Tensor, std::string> {
    if (dim0 >= ndim() || dim1 >= ndim()) {
        return std::unexpected("transpose dimension out of range");
    }
    auto new_shape = std::vector<std::size_t>(impl_->shape.begin(), impl_->shape.end());
    auto new_strides = std::vector<std::ptrdiff_t>(impl_->strides.begin(), impl_->strides.end());
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);
    return make_view(*impl_, impl_->data, std::move(new_shape), std::move(new_strides));
}

auto Tensor::squeeze(std::ptrdiff_t dim) const -> std::expected<Tensor, std::string> {
    if (dim >= 0 && static_cast<std::size_t>(dim) >= ndim()) {
        return std::unexpected("squeeze dimension out of range");
    }

    std::vector<std::size_t> new_shape;
    std::vector<std::ptrdiff_t> new_strides;

    if (dim < 0) {
        // Squeeze all dimensions of size 1
        for (std::size_t i = 0; i < ndim(); ++i) {
            if (impl_->shape[i] != 1) {
                new_shape.push_back(impl_->shape[i]);
                new_strides.push_back(impl_->strides[i]);
            }
        }
    } else {
        auto d = static_cast<std::size_t>(dim);
        if (impl_->shape[d] != 1) {
            return std::unexpected("cannot squeeze dimension " + std::to_string(dim) +
                                   " with size " + std::to_string(impl_->shape[d]));
        }
        for (std::size_t i = 0; i < ndim(); ++i) {
            if (i != d) {
                new_shape.push_back(impl_->shape[i]);
                new_strides.push_back(impl_->strides[i]);
            }
        }
    }

    return make_view(*impl_, impl_->data, std::move(new_shape), std::move(new_strides));
}

auto Tensor::unsqueeze(std::size_t dim) const -> std::expected<Tensor, std::string> {
    if (dim > ndim()) {
        return std::unexpected("unsqueeze dimension out of range");
    }

    auto new_shape = std::vector<std::size_t>(impl_->shape.begin(), impl_->shape.end());
    auto new_strides = std::vector<std::ptrdiff_t>(impl_->strides.begin(), impl_->strides.end());

    // Compute stride for the new dimension
    std::ptrdiff_t new_stride = static_cast<std::ptrdiff_t>(dtype_size(impl_->dtype));
    if (dim < ndim()) {
        new_stride = impl_->strides[dim];
    } else if (!impl_->strides.empty()) {
        new_stride = impl_->strides.back();
    }

    new_shape.insert(new_shape.begin() + static_cast<std::ptrdiff_t>(dim), 1);
    new_strides.insert(new_strides.begin() + static_cast<std::ptrdiff_t>(dim), new_stride);

    return make_view(*impl_, impl_->data, std::move(new_shape), std::move(new_strides));
}

auto Tensor::permute(std::span<const std::size_t> dims) const
    -> std::expected<Tensor, std::string> {
    if (dims.size() != ndim()) {
        return std::unexpected("permute: dims size must match ndim");
    }
    std::vector<bool> seen(ndim(), false);
    for (auto d : dims) {
        if (d >= ndim()) return std::unexpected("permute: dimension out of range");
        if (seen[d]) return std::unexpected("permute: duplicate dimension");
        seen[d] = true;
    }
    std::vector<std::size_t> new_shape(ndim());
    std::vector<std::ptrdiff_t> new_strides(ndim());
    for (std::size_t i = 0; i < ndim(); ++i) {
        new_shape[i] = impl_->shape[dims[i]];
        new_strides[i] = impl_->strides[dims[i]];
    }
    return make_view(*impl_, impl_->data, std::move(new_shape), std::move(new_strides));
}

auto Tensor::flatten() const -> std::expected<Tensor, std::string> {
    std::vector<std::size_t> new_shape = {impl_->numel};
    return reshape(new_shape);
}

// ---------------------------------------------------------------------------
// Contiguous copy helper
// ---------------------------------------------------------------------------

static void copy_strided(const void* src, void* dst,
                         std::span<const std::size_t> shape,
                         std::span<const std::ptrdiff_t> src_strides,
                         std::span<const std::ptrdiff_t> dst_strides,
                         std::size_t elem_size) {
    if (shape.empty()) {
        std::memcpy(dst, src, elem_size);
        return;
    }
    if (shape.size() == 1) {
        auto* s = static_cast<const std::byte*>(src);
        auto* d = static_cast<std::byte*>(dst);
        for (std::size_t i = 0; i < shape[0]; ++i) {
            std::memcpy(d + static_cast<std::ptrdiff_t>(i) * dst_strides[0],
                        s + static_cast<std::ptrdiff_t>(i) * src_strides[0],
                        elem_size);
        }
        return;
    }
    auto* s = static_cast<const std::byte*>(src);
    auto* d = static_cast<std::byte*>(dst);
    for (std::size_t i = 0; i < shape[0]; ++i) {
        copy_strided(s + static_cast<std::ptrdiff_t>(i) * src_strides[0],
                     d + static_cast<std::ptrdiff_t>(i) * dst_strides[0],
                     shape.subspan(1), src_strides.subspan(1), dst_strides.subspan(1),
                     elem_size);
    }
}

auto Tensor::contiguous() const -> Tensor {
    if (is_contiguous()) {
        // Return a view sharing storage
        return make_view(*impl_, impl_->data,
                         {impl_->shape.begin(), impl_->shape.end()},
                         {impl_->strides.begin(), impl_->strides.end()});
    }

    auto elem_sz = dtype_size(impl_->dtype);
    auto nbytes = impl_->numel * elem_sz;
    auto storage = Impl::make_storage(nbytes);
    auto new_strides = Impl::compute_strides(impl_->shape, elem_sz, impl_->layout);

    copy_strided(impl_->data, storage.get(),
                 impl_->shape, impl_->strides, new_strides, elem_sz);

    auto impl = std::make_unique<Impl>();
    impl->storage = std::move(storage);
    impl->data = impl->storage.get();
    impl->shape = impl_->shape;
    impl->strides = std::move(new_strides);
    impl->dtype = impl_->dtype;
    impl->numel = impl_->numel;
    impl->layout = impl_->layout;
    return Tensor(std::move(impl));
}

auto Tensor::clone() const -> Tensor {
    return Tensor(*this);
}

// ---------------------------------------------------------------------------
// Type casting
// ---------------------------------------------------------------------------

auto Tensor::to(DType target) const -> Tensor {
    if (target == dtype()) return clone();
    auto result = empty(shape(), target, layout());
    assert(result.has_value());
    auto& out = *result;

    auto numel_count = impl_->numel;
    std::vector<std::size_t> indices(ndim(), 0);

    for (std::size_t n = 0; n < numel_count; ++n) {
        auto off_in = compute_offset(impl_->strides, indices.data(), ndim());
        auto off_out = compute_offset(out.strides(), indices.data(), ndim());

        double val = read_as_double(static_cast<const std::byte*>(impl_->data) + off_in, impl_->dtype);
        write_from_double(static_cast<std::byte*>(out.data_ptr()) + off_out, target, val);

        advance_indices(indices.data(), impl_->shape);
    }
    return out;
}

auto Tensor::to(DType target, double scale, double offset) const -> Tensor {
    auto result = empty(shape(), target, layout());
    assert(result.has_value());
    auto& out = *result;

    auto numel_count = impl_->numel;
    std::vector<std::size_t> indices(ndim(), 0);

    for (std::size_t n = 0; n < numel_count; ++n) {
        auto off_in = compute_offset(impl_->strides, indices.data(), ndim());
        auto off_out = compute_offset(out.strides(), indices.data(), ndim());

        double val = read_as_double(static_cast<const std::byte*>(impl_->data) + off_in, impl_->dtype);
        val = val * scale + offset;
        write_from_double(static_cast<std::byte*>(out.data_ptr()) + off_out, target, val);

        advance_indices(indices.data(), impl_->shape);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Slicing
// ---------------------------------------------------------------------------

auto Tensor::slice(std::span<const SliceArg> args) const
    -> std::expected<Tensor, std::string> {
    if (args.size() > ndim()) {
        return std::unexpected("too many slice arguments for tensor of ndim " +
                               std::to_string(ndim()));
    }

    auto* new_data = static_cast<std::byte*>(impl_->data);
    std::vector<std::size_t> new_shape;
    std::vector<std::ptrdiff_t> new_strides;

    for (std::size_t i = 0; i < args.size(); ++i) {
        auto dim_size = static_cast<std::ptrdiff_t>(impl_->shape[i]);
        auto dim_stride = impl_->strides[i];

        if (args[i].kind == SliceArg::Kind::Index) {
            auto idx = args[i].index;
            if (idx < 0) idx += dim_size;
            if (idx < 0 || idx >= dim_size) {
                return std::unexpected("index " + std::to_string(args[i].index) +
                                       " out of range for dimension " + std::to_string(i) +
                                       " with size " + std::to_string(dim_size));
            }
            new_data += idx * dim_stride;
            // Dimension is removed (indexed away)
        } else {
            auto start = args[i].slice.start;
            auto stop = args[i].slice.stop;
            auto step = args[i].slice.step;

            if (step == 0) {
                return std::unexpected("slice step cannot be zero");
            }

            // Normalize negative indices
            if (start < 0) start += dim_size;
            if (stop < 0) stop += dim_size;

            // Clamp
            start = std::clamp(start, std::ptrdiff_t{0}, dim_size);
            stop = std::clamp(stop, std::ptrdiff_t{0}, dim_size);

            std::size_t length = 0;
            if (step > 0 && stop > start) {
                length = static_cast<std::size_t>((stop - start + step - 1) / step);
            } else if (step < 0 && start > stop) {
                length = static_cast<std::size_t>((start - stop - step - 1) / (-step));
            }

            new_data += start * dim_stride;
            new_shape.push_back(length);
            new_strides.push_back(dim_stride * step);
        }
    }

    // Append remaining dimensions unchanged
    for (std::size_t i = args.size(); i < ndim(); ++i) {
        new_shape.push_back(impl_->shape[i]);
        new_strides.push_back(impl_->strides[i]);
    }

    return make_view(*impl_, new_data, std::move(new_shape), std::move(new_strides));
}

auto Tensor::slice(std::initializer_list<SliceArg> args) const
    -> std::expected<Tensor, std::string> {
    return slice(std::span<const SliceArg>(args.begin(), args.size()));
}

// ---------------------------------------------------------------------------
// Arithmetic helpers
// ---------------------------------------------------------------------------

template <typename F>
static auto binary_op_impl(const Tensor& a, const Tensor& b, F&& op) -> Tensor {
    assert(a.ndim() == b.ndim() && "binary_op: ndim mismatch");
    for (std::size_t i = 0; i < a.ndim(); ++i) {
        assert(a.shape()[i] == b.shape()[i] && "binary_op: shape mismatch");
    }

    auto out_dt = promote_dtype(a.dtype(), b.dtype());
    auto result = Tensor::empty(a.shape(), out_dt, a.layout());
    assert(result.has_value());
    auto& out = *result;

    auto numel = a.numel();
    std::vector<std::size_t> indices(a.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off_a = compute_offset(a.strides(), indices.data(), a.ndim());
        auto off_b = compute_offset(b.strides(), indices.data(), b.ndim());
        auto off_o = compute_offset(out.strides(), indices.data(), out.ndim());

        double va = read_as_double(static_cast<const std::byte*>(a.data_ptr()) + off_a, a.dtype());
        double vb = read_as_double(static_cast<const std::byte*>(b.data_ptr()) + off_b, b.dtype());
        double vr = op(va, vb);
        write_from_double(static_cast<std::byte*>(out.data_ptr()) + off_o, out_dt, vr);

        advance_indices(indices.data(), a.shape());
    }
    return out;
}

template <typename F>
static auto scalar_op_impl(const Tensor& a, double scalar, F&& op) -> Tensor {
    auto result = Tensor::empty(a.shape(), a.dtype(), a.layout());
    assert(result.has_value());
    auto& out = *result;

    auto numel = a.numel();
    std::vector<std::size_t> indices(a.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off_a = compute_offset(a.strides(), indices.data(), a.ndim());
        auto off_o = compute_offset(out.strides(), indices.data(), out.ndim());

        double va = read_as_double(static_cast<const std::byte*>(a.data_ptr()) + off_a, a.dtype());
        double vr = op(va, scalar);
        write_from_double(static_cast<std::byte*>(out.data_ptr()) + off_o, a.dtype(), vr);

        advance_indices(indices.data(), a.shape());
    }
    return out;
}

template <typename F>
static auto unary_op_impl(const Tensor& a, DType out_dt, F&& op) -> Tensor {
    auto result = Tensor::empty(a.shape(), out_dt, a.layout());
    assert(result.has_value());
    auto& out = *result;

    auto numel = a.numel();
    std::vector<std::size_t> indices(a.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off_a = compute_offset(a.strides(), indices.data(), a.ndim());
        auto off_o = compute_offset(out.strides(), indices.data(), out.ndim());

        double va = read_as_double(static_cast<const std::byte*>(a.data_ptr()) + off_a, a.dtype());
        double vr = op(va);
        write_from_double(static_cast<std::byte*>(out.data_ptr()) + off_o, out_dt, vr);

        advance_indices(indices.data(), a.shape());
    }
    return out;
}

// ---------------------------------------------------------------------------
// Arithmetic operators (tensor-tensor)
// ---------------------------------------------------------------------------

auto operator+(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_op_impl(a, b, [](double x, double y) { return x + y; });
}

auto operator-(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_op_impl(a, b, [](double x, double y) { return x - y; });
}

auto operator*(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_op_impl(a, b, [](double x, double y) { return x * y; });
}

auto operator/(const Tensor& a, const Tensor& b) -> Tensor {
    return binary_op_impl(a, b, [](double x, double y) { return x / y; });
}

// ---------------------------------------------------------------------------
// Arithmetic operators (tensor-scalar)
// ---------------------------------------------------------------------------

auto operator+(const Tensor& a, double b) -> Tensor {
    return scalar_op_impl(a, b, [](double x, double y) { return x + y; });
}

auto operator-(const Tensor& a, double b) -> Tensor {
    return scalar_op_impl(a, b, [](double x, double y) { return x - y; });
}

auto operator*(const Tensor& a, double b) -> Tensor {
    return scalar_op_impl(a, b, [](double x, double y) { return x * y; });
}

auto operator/(const Tensor& a, double b) -> Tensor {
    return scalar_op_impl(a, b, [](double x, double y) { return x / y; });
}

// ---------------------------------------------------------------------------
// Unary minus
// ---------------------------------------------------------------------------

auto operator-(const Tensor& a) -> Tensor {
    return unary_op_impl(a, a.dtype(), [](double x) { return -x; });
}

// ---------------------------------------------------------------------------
// Unary ops (free functions)
// ---------------------------------------------------------------------------

auto abs(const Tensor& t) -> Tensor {
    return unary_op_impl(t, t.dtype(), [](double x) { return std::abs(x); });
}

auto neg(const Tensor& t) -> Tensor {
    return -t;
}

auto sqrt(const Tensor& t) -> Tensor {
    DType out_dt = is_float_dtype(t.dtype()) ? t.dtype() : DType::Float64;
    return unary_op_impl(t, out_dt, [](double x) { return std::sqrt(x); });
}

auto clamp(const Tensor& t, double lo, double hi) -> Tensor {
    return unary_op_impl(t, t.dtype(), [lo, hi](double x) { return std::clamp(x, lo, hi); });
}

// ---------------------------------------------------------------------------
// Reductions — global
// ---------------------------------------------------------------------------

auto sum(const Tensor& t) -> Tensor {
    double total = 0.0;
    auto numel = t.numel();
    std::vector<std::size_t> indices(t.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        total += read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        advance_indices(indices.data(), t.shape());
    }

    auto out_dt = reduction_dtype(t.dtype());
    auto result = Tensor::full({}, total, out_dt);
    assert(result.has_value());
    return std::move(*result);
}

auto mean(const Tensor& t) -> Tensor {
    double total = 0.0;
    auto numel = t.numel();
    std::vector<std::size_t> indices(t.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        total += read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        advance_indices(indices.data(), t.shape());
    }

    double avg = (numel > 0) ? total / static_cast<double>(numel) : 0.0;
    auto result = Tensor::full({}, avg, DType::Float64);
    assert(result.has_value());
    return std::move(*result);
}

auto amin(const Tensor& t) -> Tensor {
    double min_val = std::numeric_limits<double>::infinity();
    auto numel = t.numel();
    std::vector<std::size_t> indices(t.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        double val = read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        if (val < min_val) min_val = val;
        advance_indices(indices.data(), t.shape());
    }

    auto out_dt = reduction_dtype(t.dtype());
    auto result = Tensor::full({}, min_val, out_dt);
    assert(result.has_value());
    return std::move(*result);
}

auto amax(const Tensor& t) -> Tensor {
    double max_val = -std::numeric_limits<double>::infinity();
    auto numel = t.numel();
    std::vector<std::size_t> indices(t.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        double val = read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        if (val > max_val) max_val = val;
        advance_indices(indices.data(), t.shape());
    }

    auto out_dt = reduction_dtype(t.dtype());
    auto result = Tensor::full({}, max_val, out_dt);
    assert(result.has_value());
    return std::move(*result);
}

// ---------------------------------------------------------------------------
// Reductions — along dimension
// ---------------------------------------------------------------------------

using ReduceOp = double(*)(double, double);

static auto dim_reduce(const Tensor& t, std::size_t dim, double init,
                       ReduceOp op, bool is_mean, DType out_dt)
    -> std::expected<Tensor, std::string> {
    if (dim >= t.ndim()) {
        return std::unexpected("reduction dimension out of range");
    }

    // Output shape: input shape with dim removed
    std::vector<std::size_t> out_shape;
    for (std::size_t i = 0; i < t.ndim(); ++i) {
        if (i != dim) out_shape.push_back(t.shape()[i]);
    }
    // If all dims removed, output is scalar
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = Tensor::zeros(out_shape, out_dt, t.layout());
    if (!result) return std::unexpected(result.error());
    auto& out = *result;

    auto out_numel = out.numel();
    std::vector<std::size_t> out_indices(out.ndim(), 0);

    for (std::size_t n = 0; n < out_numel; ++n) {
        // Map output indices to input indices
        std::vector<std::size_t> in_indices(t.ndim());
        std::size_t j = 0;
        for (std::size_t i = 0; i < t.ndim(); ++i) {
            if (i == dim) {
                in_indices[i] = 0;
            } else {
                in_indices[i] = out_indices[j++];
            }
        }

        double acc = init;
        for (std::size_t k = 0; k < t.shape()[dim]; ++k) {
            in_indices[dim] = k;
            auto off = compute_offset(t.strides(), in_indices.data(), t.ndim());
            double val = read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
            acc = op(acc, val);
        }

        if (is_mean && t.shape()[dim] > 0) {
            acc /= static_cast<double>(t.shape()[dim]);
        }

        auto off_o = compute_offset(out.strides(), out_indices.data(), out.ndim());
        write_from_double(static_cast<std::byte*>(out.data_ptr()) + off_o, out_dt, acc);

        advance_indices(out_indices.data(), out.shape());
    }
    return out;
}

auto sum(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string> {
    return dim_reduce(t, dim, 0.0,
        [](double a, double b) { return a + b; },
        false, reduction_dtype(t.dtype()));
}

auto mean(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string> {
    return dim_reduce(t, dim, 0.0,
        [](double a, double b) { return a + b; },
        true, DType::Float64);
}

auto amin(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string> {
    return dim_reduce(t, dim, std::numeric_limits<double>::infinity(),
        [](double a, double b) { return std::min(a, b); },
        false, reduction_dtype(t.dtype()));
}

auto amax(const Tensor& t, std::size_t dim) -> std::expected<Tensor, std::string> {
    return dim_reduce(t, dim, -std::numeric_limits<double>::infinity(),
        [](double a, double b) { return std::max(a, b); },
        false, reduction_dtype(t.dtype()));
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

auto Tensor::operator==(const Tensor& other) const -> bool {
    if (impl_->dtype != other.impl_->dtype) return false;
    if (impl_->shape != other.impl_->shape) return false;

    bool equal = true;
    auto elem_sz = dtype_size(impl_->dtype);

    std::vector<std::size_t> indices(ndim(), 0);
    auto numel_count = impl_->numel;

    for (std::size_t n = 0; n < numel_count && equal; ++n) {
        auto off_a = compute_offset(impl_->strides, indices.data(), ndim());
        auto off_b = compute_offset(other.impl_->strides, indices.data(), ndim());
        auto* a = static_cast<const std::byte*>(impl_->data) + off_a;
        auto* b = static_cast<const std::byte*>(other.impl_->data) + off_b;
        if (std::memcmp(a, b, elem_sz) != 0) {
            equal = false;
        }
        advance_indices(indices.data(), impl_->shape);
    }
    return equal;
}

auto Tensor::allclose(const Tensor& other, double rtol, double atol) const -> bool {
    if (impl_->dtype != other.impl_->dtype) return false;
    if (impl_->shape != other.impl_->shape) return false;

    bool close = true;
    std::vector<std::size_t> indices(ndim(), 0);

    auto check_close = [&]<typename T>() {
        for (std::size_t n = 0; n < impl_->numel && close; ++n) {
            auto off_a = compute_offset(impl_->strides, indices.data(), ndim());
            auto off_b = compute_offset(other.impl_->strides, indices.data(), ndim());
            auto* a_ptr = static_cast<const std::byte*>(impl_->data) + off_a;
            auto* b_ptr = static_cast<const std::byte*>(other.impl_->data) + off_b;
            auto a = static_cast<double>(*reinterpret_cast<const T*>(a_ptr));
            auto b = static_cast<double>(*reinterpret_cast<const T*>(b_ptr));
            if (std::abs(a - b) > atol + rtol * std::abs(b)) {
                close = false;
            }
            advance_indices(indices.data(), impl_->shape);
        }
    };

    dispatch_dtype(impl_->dtype, check_close);
    return close;
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

static void print_recursive(std::ostream& os, const void* data,
                            std::span<const std::size_t> shape,
                            std::span<const std::ptrdiff_t> strides,
                            DType dt, std::size_t depth) {
    if (shape.empty()) {
        // Scalar
        auto* p = static_cast<const std::byte*>(data);
        dispatch_dtype(dt, [&]<typename T>() {
            if constexpr (std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>) {
                os << static_cast<int>(*reinterpret_cast<const T*>(p));
            } else {
                os << *reinterpret_cast<const T*>(p);
            }
        });
        return;
    }

    os << "[";
    auto* base = static_cast<const std::byte*>(data);
    for (std::size_t i = 0; i < shape[0]; ++i) {
        if (i > 0) {
            os << ", ";
            if (shape.size() > 1) {
                os << "\n";
                for (std::size_t d = 0; d <= depth; ++d) os << " ";
            }
        }
        print_recursive(os, base + static_cast<std::ptrdiff_t>(i) * strides[0],
                         shape.subspan(1), strides.subspan(1), dt, depth + 1);
    }
    os << "]";
}

auto operator<<(std::ostream& os, const Tensor& t) -> std::ostream& {
    os << "Tensor(";
    print_recursive(os, t.data_ptr(), t.shape(), t.strides(), t.dtype(), 7);
    os << ", shape=[";
    for (std::size_t i = 0; i < t.ndim(); ++i) {
        if (i > 0) os << ", ";
        os << t.shape()[i];
    }
    os << "], dtype=" << dtype_name(t.dtype()) << ")";
    return os;
}

// ---------------------------------------------------------------------------
// arange
// ---------------------------------------------------------------------------

auto arange(double start, double stop, double step, DType dt)
    -> std::expected<Tensor, std::string> {
    if (step == 0.0) {
        return std::unexpected("arange: step cannot be zero");
    }
    if ((stop > start && step < 0.0) || (stop < start && step > 0.0)) {
        return std::unexpected("arange: step direction does not match start/stop");
    }
    std::size_t n = 0;
    if (stop != start) {
        n = static_cast<std::size_t>(std::ceil((stop - start) / step));
    }
    if (n == 0) {
        // Empty range: return a 1-element tensor? No — match NumPy: return empty.
        // But make_tensor rejects zero-dim shapes. Use n=0 case.
        // We need to handle this: NumPy returns an empty array for arange(5,5).
        // Our make_tensor rejects shape {0}, so create a special case.
        // Actually, let's just return an error for now or handle it differently.
        // NumPy: arange(5,5) returns array([], dtype=float64).
        // For simplicity, return error since our tensors don't support 0-size dims.
        return std::unexpected("arange: resulting range is empty");
    }
    auto result = Tensor::empty({n}, dt);
    if (!result) return std::unexpected(result.error());
    auto& out = *result;
    auto elem_sz = dtype_size(dt);
    auto* ptr = static_cast<std::byte*>(out.data_ptr());
    for (std::size_t i = 0; i < n; ++i) {
        write_from_double(ptr + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(elem_sz),
                          dt, start + static_cast<double>(i) * step);
    }
    return out;
}

auto arange(double stop, DType dt)
    -> std::expected<Tensor, std::string> {
    return arange(0.0, stop, 1.0, dt);
}

// ---------------------------------------------------------------------------
// linspace
// ---------------------------------------------------------------------------

auto linspace(double start, double stop, std::size_t n, DType dt)
    -> std::expected<Tensor, std::string> {
    if (n == 0) {
        return std::unexpected("linspace: n must be > 0");
    }
    auto result = Tensor::empty({n}, dt);
    if (!result) return std::unexpected(result.error());
    auto& out = *result;
    auto elem_sz = dtype_size(dt);
    auto* ptr = static_cast<std::byte*>(out.data_ptr());
    if (n == 1) {
        write_from_double(ptr, dt, start);
    } else {
        double step = (stop - start) / static_cast<double>(n - 1);
        for (std::size_t i = 0; i < n - 1; ++i) {
            write_from_double(ptr + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(elem_sz),
                              dt, start + static_cast<double>(i) * step);
        }
        // Last element is exactly stop (avoid floating-point drift)
        write_from_double(ptr + static_cast<std::ptrdiff_t>(n - 1) * static_cast<std::ptrdiff_t>(elem_sz),
                          dt, stop);
    }
    return out;
}

// ---------------------------------------------------------------------------
// where
// ---------------------------------------------------------------------------

auto where(const Tensor& condition, const Tensor& x, const Tensor& y)
    -> std::expected<Tensor, std::string> {
    if (condition.shape().size() != x.shape().size() ||
        condition.shape().size() != y.shape().size()) {
        return std::unexpected("where: condition, x, and y must have the same ndim");
    }
    for (std::size_t i = 0; i < condition.ndim(); ++i) {
        if (condition.shape()[i] != x.shape()[i] || condition.shape()[i] != y.shape()[i]) {
            return std::unexpected("where: condition, x, and y must have the same shape");
        }
    }
    if (x.dtype() != y.dtype()) {
        return std::unexpected("where: x and y must have the same dtype");
    }

    auto result = Tensor::empty(x.shape(), x.dtype(), x.layout());
    if (!result) return std::unexpected(result.error());
    auto& out = *result;

    auto numel = x.numel();
    auto elem_sz = dtype_size(x.dtype());
    std::vector<std::size_t> indices(x.ndim(), 0);

    for (std::size_t n = 0; n < numel; ++n) {
        auto off_c = compute_offset(condition.strides(), indices.data(), condition.ndim());
        auto off_x = compute_offset(x.strides(), indices.data(), x.ndim());
        auto off_y = compute_offset(y.strides(), indices.data(), y.ndim());
        auto off_o = compute_offset(out.strides(), indices.data(), out.ndim());

        double cval = read_as_double(
            static_cast<const std::byte*>(condition.data_ptr()) + off_c,
            condition.dtype());

        auto* src = (cval != 0.0)
            ? static_cast<const std::byte*>(x.data_ptr()) + off_x
            : static_cast<const std::byte*>(y.data_ptr()) + off_y;
        auto* dst = static_cast<std::byte*>(out.data_ptr()) + off_o;
        std::memcpy(dst, src, elem_sz);

        advance_indices(indices.data(), x.shape());
    }
    return out;
}

// ---------------------------------------------------------------------------
// meshgrid
// ---------------------------------------------------------------------------

auto meshgrid(std::span<const Tensor> tensors)
    -> std::expected<std::vector<Tensor>, std::string> {
    if (tensors.empty()) {
        return std::unexpected("meshgrid: need at least one input tensor");
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].ndim() != 1) {
            return std::unexpected("meshgrid: all input tensors must be 1-D");
        }
    }

    // Output shape: {tensors[0].numel(), tensors[1].numel(), ...}
    std::vector<std::size_t> out_shape;
    out_shape.reserve(tensors.size());
    for (auto& t : tensors) {
        out_shape.push_back(t.numel());
    }

    std::vector<Tensor> result;
    result.reserve(tensors.size());

    for (std::size_t ti = 0; ti < tensors.size(); ++ti) {
        auto& src = tensors[ti];
        auto out = Tensor::empty(out_shape, src.dtype());
        if (!out) return std::unexpected(out.error());

        auto total = out->numel();
        auto elem_sz = dtype_size(src.dtype());
        auto* out_ptr = static_cast<std::byte*>(out->data_ptr());
        auto* src_ptr = static_cast<const std::byte*>(src.data_ptr());
        auto src_stride = src.strides()[0];

        std::vector<std::size_t> indices(out_shape.size(), 0);

        for (std::size_t n = 0; n < total; ++n) {
            auto off_o = compute_offset(out->strides(), indices.data(), out->ndim());
            auto src_idx = indices[ti];
            auto off_s = static_cast<std::ptrdiff_t>(src_idx) * src_stride;
            std::memcpy(out_ptr + off_o, src_ptr + off_s, elem_sz);
            advance_indices(indices.data(), out_shape);
        }

        result.push_back(std::move(*out));
    }
    return result;
}

// ---------------------------------------------------------------------------
// eye
// ---------------------------------------------------------------------------

auto eye(std::size_t n, DType dt) -> std::expected<Tensor, std::string> {
    auto result = Tensor::zeros({n, n}, dt);
    if (!result) return result;
    auto* ptr = static_cast<std::byte*>(result->data_ptr());
    auto s0 = result->strides()[0];
    auto s1 = result->strides()[1];
    for (std::size_t i = 0; i < n; ++i) {
        write_from_double(ptr + static_cast<std::ptrdiff_t>(i) * s0 +
                              static_cast<std::ptrdiff_t>(i) * s1, dt, 1.0);
    }
    return result;
}

// ---------------------------------------------------------------------------
// concatenate
// ---------------------------------------------------------------------------

auto concatenate(std::span<const Tensor> tensors, std::size_t axis)
    -> std::expected<Tensor, std::string> {
    if (tensors.empty())
        return std::unexpected("concatenate: need at least one tensor");
    auto ndim = tensors[0].ndim();
    if (axis >= ndim)
        return std::unexpected("concatenate: axis out of range");
    auto dt = tensors[0].dtype();

    for (std::size_t t = 1; t < tensors.size(); ++t) {
        if (tensors[t].ndim() != ndim)
            return std::unexpected("concatenate: all tensors must have the same ndim");
        if (tensors[t].dtype() != dt)
            return std::unexpected("concatenate: all tensors must have the same dtype");
        for (std::size_t d = 0; d < ndim; ++d) {
            if (d != axis && tensors[t].shape()[d] != tensors[0].shape()[d])
                return std::unexpected("concatenate: shapes must match except along concat axis");
        }
    }

    std::vector<std::size_t> out_shape(tensors[0].shape().begin(), tensors[0].shape().end());
    for (std::size_t t = 1; t < tensors.size(); ++t)
        out_shape[axis] += tensors[t].shape()[axis];

    auto result = Tensor::empty(out_shape, dt);
    if (!result) return result;

    auto elem_sz = dtype_size(dt);
    std::size_t axis_offset = 0;
    for (auto& src : tensors) {
        auto src_numel = src.numel();
        std::vector<std::size_t> src_indices(ndim, 0);
        std::vector<std::size_t> dst_indices(ndim, 0);

        for (std::size_t n = 0; n < src_numel; ++n) {
            for (std::size_t d = 0; d < ndim; ++d)
                dst_indices[d] = src_indices[d] + (d == axis ? axis_offset : 0);

            auto off_src = compute_offset(src.strides(), src_indices.data(), ndim);
            auto off_dst = compute_offset(result->strides(), dst_indices.data(), ndim);
            std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_dst,
                        static_cast<const std::byte*>(src.data_ptr()) + off_src,
                        elem_sz);
            advance_indices(src_indices.data(), src.shape());
        }
        axis_offset += src.shape()[axis];
    }
    return result;
}

// ---------------------------------------------------------------------------
// stack
// ---------------------------------------------------------------------------

auto stack(std::span<const Tensor> tensors, std::size_t axis)
    -> std::expected<Tensor, std::string> {
    if (tensors.empty())
        return std::unexpected("stack: need at least one tensor");
    auto ndim = tensors[0].ndim();
    if (axis > ndim)
        return std::unexpected("stack: axis out of range");
    auto dt = tensors[0].dtype();

    for (std::size_t t = 1; t < tensors.size(); ++t) {
        if (tensors[t].ndim() != ndim)
            return std::unexpected("stack: all tensors must have the same ndim");
        if (tensors[t].dtype() != dt)
            return std::unexpected("stack: all tensors must have the same dtype");
        for (std::size_t d = 0; d < ndim; ++d) {
            if (tensors[t].shape()[d] != tensors[0].shape()[d])
                return std::unexpected("stack: all tensors must have the same shape");
        }
    }

    // Build output shape: insert tensors.size() at position axis
    std::vector<std::size_t> out_shape;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (d == axis) out_shape.push_back(tensors.size());
        out_shape.push_back(tensors[0].shape()[d]);
    }
    if (axis == ndim) out_shape.push_back(tensors.size());

    auto result = Tensor::empty(out_shape, dt);
    if (!result) return result;

    auto elem_sz = dtype_size(dt);
    for (std::size_t ti = 0; ti < tensors.size(); ++ti) {
        auto& src = tensors[ti];
        auto src_numel = src.numel();
        std::vector<std::size_t> src_indices(ndim, 0);
        std::vector<std::size_t> dst_indices(ndim + 1, 0);

        for (std::size_t n = 0; n < src_numel; ++n) {
            std::size_t si = 0;
            for (std::size_t d = 0; d <= ndim; ++d) {
                if (d == axis) dst_indices[d] = ti;
                else dst_indices[d] = src_indices[si++];
            }

            auto off_src = compute_offset(src.strides(), src_indices.data(), ndim);
            auto off_dst = compute_offset(result->strides(), dst_indices.data(), ndim + 1);
            std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_dst,
                        static_cast<const std::byte*>(src.data_ptr()) + off_src,
                        elem_sz);
            advance_indices(src_indices.data(), src.shape());
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

auto matmul(const Tensor& a, const Tensor& b)
    -> std::expected<Tensor, std::string> {
    auto out_dt = promote_dtype(a.dtype(), b.dtype());

    if (a.ndim() == 1 && b.ndim() == 1) {
        // Dot product: (K,) × (K,) → scalar
        if (a.numel() != b.numel())
            return std::unexpected("matmul: vectors must have same length for dot product");
        auto K = a.numel();
        double dot = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            double va = read_as_double(
                static_cast<const std::byte*>(a.data_ptr()) +
                static_cast<std::ptrdiff_t>(k) * a.strides()[0], a.dtype());
            double vb = read_as_double(
                static_cast<const std::byte*>(b.data_ptr()) +
                static_cast<std::ptrdiff_t>(k) * b.strides()[0], b.dtype());
            dot += va * vb;
        }
        return Tensor::full({}, dot, out_dt);
    }

    if (a.ndim() == 2 && b.ndim() == 1) {
        // Matrix-vector: (M,K) × (K,) → (M,)
        auto M = a.shape()[0];
        auto K = a.shape()[1];
        if (b.numel() != K)
            return std::unexpected("matmul: inner dimensions must match");
        auto result = Tensor::empty({M}, out_dt);
        if (!result) return result;
        for (std::size_t i = 0; i < M; ++i) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                std::size_t ai[] = {i, k};
                double va = read_as_double(
                    static_cast<const std::byte*>(a.data_ptr()) +
                    compute_offset(a.strides(), ai, 2), a.dtype());
                double vb = read_as_double(
                    static_cast<const std::byte*>(b.data_ptr()) +
                    static_cast<std::ptrdiff_t>(k) * b.strides()[0], b.dtype());
                sum += va * vb;
            }
            write_from_double(
                static_cast<std::byte*>(result->data_ptr()) +
                static_cast<std::ptrdiff_t>(i) * result->strides()[0], out_dt, sum);
        }
        return result;
    }

    if (a.ndim() == 1 && b.ndim() == 2) {
        // Vector-matrix: (K,) × (K,N) → (N,)
        auto K = b.shape()[0];
        auto N = b.shape()[1];
        if (a.numel() != K)
            return std::unexpected("matmul: inner dimensions must match");
        auto result = Tensor::empty({N}, out_dt);
        if (!result) return result;
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                std::size_t bj[] = {k, j};
                double va = read_as_double(
                    static_cast<const std::byte*>(a.data_ptr()) +
                    static_cast<std::ptrdiff_t>(k) * a.strides()[0], a.dtype());
                double vb = read_as_double(
                    static_cast<const std::byte*>(b.data_ptr()) +
                    compute_offset(b.strides(), bj, 2), b.dtype());
                sum += va * vb;
            }
            write_from_double(
                static_cast<std::byte*>(result->data_ptr()) +
                static_cast<std::ptrdiff_t>(j) * result->strides()[0], out_dt, sum);
        }
        return result;
    }

    if (a.ndim() == 2 && b.ndim() == 2) {
        // Matrix-matrix: (M,K) × (K,N) → (M,N)
        auto M = a.shape()[0];
        auto K = a.shape()[1];
        auto K2 = b.shape()[0];
        auto N = b.shape()[1];
        if (K != K2)
            return std::unexpected("matmul: inner dimensions must match (" +
                                   std::to_string(K) + " vs " + std::to_string(K2) + ")");
        auto result = Tensor::empty({M, N}, out_dt);
        if (!result) return result;
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    std::size_t ai[] = {i, k};
                    std::size_t bi[] = {k, j};
                    double va = read_as_double(
                        static_cast<const std::byte*>(a.data_ptr()) +
                        compute_offset(a.strides(), ai, 2), a.dtype());
                    double vb = read_as_double(
                        static_cast<const std::byte*>(b.data_ptr()) +
                        compute_offset(b.strides(), bi, 2), b.dtype());
                    sum += va * vb;
                }
                std::size_t oi[] = {i, j};
                write_from_double(
                    static_cast<std::byte*>(result->data_ptr()) +
                    compute_offset(result->strides(), oi, 2), out_dt, sum);
            }
        }
        return result;
    }

    return std::unexpected("matmul: inputs must be 1D or 2D");
}

// ---------------------------------------------------------------------------
// argmin / argmax — global
// ---------------------------------------------------------------------------

auto argmin(const Tensor& t) -> std::size_t {
    double best = std::numeric_limits<double>::infinity();
    std::size_t best_idx = 0;
    std::vector<std::size_t> indices(t.ndim(), 0);
    for (std::size_t n = 0; n < t.numel(); ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        double val = read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        if (val < best) { best = val; best_idx = n; }
        advance_indices(indices.data(), t.shape());
    }
    return best_idx;
}

auto argmax(const Tensor& t) -> std::size_t {
    double best = -std::numeric_limits<double>::infinity();
    std::size_t best_idx = 0;
    std::vector<std::size_t> indices(t.ndim(), 0);
    for (std::size_t n = 0; n < t.numel(); ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        double val = read_as_double(static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        if (val > best) { best = val; best_idx = n; }
        advance_indices(indices.data(), t.shape());
    }
    return best_idx;
}

// ---------------------------------------------------------------------------
// argmin / argmax — along dimension
// ---------------------------------------------------------------------------

static auto dim_argreduce(const Tensor& t, std::size_t dim, bool find_min)
    -> std::expected<Tensor, std::string> {
    if (dim >= t.ndim())
        return std::unexpected("argmin/argmax: dim out of range");

    std::vector<std::size_t> out_shape;
    for (std::size_t d = 0; d < t.ndim(); ++d)
        if (d != dim) out_shape.push_back(t.shape()[d]);
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = Tensor::empty(out_shape, DType::Int64);
    if (!result) return result;

    auto out_numel = result->numel();
    std::vector<std::size_t> out_indices(result->ndim(), 0);

    for (std::size_t n = 0; n < out_numel; ++n) {
        std::vector<std::size_t> in_indices(t.ndim());
        std::size_t j = 0;
        for (std::size_t d = 0; d < t.ndim(); ++d) {
            if (d == dim) in_indices[d] = 0;
            else in_indices[d] = out_indices[j++];
        }

        double best = find_min ? std::numeric_limits<double>::infinity()
                               : -std::numeric_limits<double>::infinity();
        std::int64_t best_k = 0;
        for (std::size_t k = 0; k < t.shape()[dim]; ++k) {
            in_indices[dim] = k;
            auto off = compute_offset(t.strides(), in_indices.data(), t.ndim());
            double val = read_as_double(
                static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
            if (find_min ? (val < best) : (val > best)) {
                best = val;
                best_k = static_cast<std::int64_t>(k);
            }
        }

        auto off_out = compute_offset(result->strides(), out_indices.data(), result->ndim());
        *reinterpret_cast<std::int64_t*>(
            static_cast<std::byte*>(result->data_ptr()) + off_out) = best_k;
        advance_indices(out_indices.data(), result->shape());
    }
    return result;
}

auto argmin(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string> {
    return dim_argreduce(t, dim, true);
}

auto argmax(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string> {
    return dim_argreduce(t, dim, false);
}

// ---------------------------------------------------------------------------
// any / all
// ---------------------------------------------------------------------------

auto any(const Tensor& t) -> bool {
    std::vector<std::size_t> indices(t.ndim(), 0);
    for (std::size_t n = 0; n < t.numel(); ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        double val = read_as_double(
            static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        if (val != 0.0) return true;
        advance_indices(indices.data(), t.shape());
    }
    return false;
}

auto all(const Tensor& t) -> bool {
    std::vector<std::size_t> indices(t.ndim(), 0);
    for (std::size_t n = 0; n < t.numel(); ++n) {
        auto off = compute_offset(t.strides(), indices.data(), t.ndim());
        double val = read_as_double(
            static_cast<const std::byte*>(t.data_ptr()) + off, t.dtype());
        if (val == 0.0) return false;
        advance_indices(indices.data(), t.shape());
    }
    return true;
}

// ---------------------------------------------------------------------------
// tile
// ---------------------------------------------------------------------------

auto tile(const Tensor& t, std::span<const std::size_t> reps)
    -> std::expected<Tensor, std::string> {
    if (reps.empty())
        return std::unexpected("tile: reps must not be empty");

    auto out_ndim = std::max(reps.size(), t.ndim());
    std::vector<std::size_t> full_reps(out_ndim, 1);
    std::vector<std::size_t> full_shape(out_ndim, 1);
    for (std::size_t i = 0; i < reps.size(); ++i)
        full_reps[out_ndim - reps.size() + i] = reps[i];
    for (std::size_t i = 0; i < t.ndim(); ++i)
        full_shape[out_ndim - t.ndim() + i] = t.shape()[i];

    std::vector<std::size_t> out_shape(out_ndim);
    for (std::size_t i = 0; i < out_ndim; ++i)
        out_shape[i] = full_shape[i] * full_reps[i];

    auto result = Tensor::empty(out_shape, t.dtype());
    if (!result) return result;

    auto src = t.contiguous();
    auto elem_sz = dtype_size(t.dtype());
    auto total = result->numel();
    std::vector<std::size_t> out_indices(out_ndim, 0);
    std::vector<std::size_t> src_indices(t.ndim());

    for (std::size_t n = 0; n < total; ++n) {
        for (std::size_t d = 0; d < t.ndim(); ++d) {
            auto out_d = d + (out_ndim - t.ndim());
            src_indices[d] = out_indices[out_d] % full_shape[out_d];
        }
        auto off_src = compute_offset(src.strides(), src_indices.data(), t.ndim());
        auto off_dst = compute_offset(result->strides(), out_indices.data(), out_ndim);
        std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_dst,
                    static_cast<const std::byte*>(src.data_ptr()) + off_src,
                    elem_sz);
        advance_indices(out_indices.data(), out_shape);
    }
    return result;
}

// ---------------------------------------------------------------------------
// flip
// ---------------------------------------------------------------------------

auto flip(const Tensor& t, std::size_t axis)
    -> std::expected<Tensor, std::string> {
    if (axis >= t.ndim())
        return std::unexpected("flip: axis out of range");

    auto result = Tensor::empty(t.shape(), t.dtype(), t.layout());
    if (!result) return result;

    auto elem_sz = dtype_size(t.dtype());
    auto axis_size = t.shape()[axis];
    std::vector<std::size_t> in_indices(t.ndim(), 0);
    std::vector<std::size_t> out_indices(t.ndim(), 0);

    for (std::size_t n = 0; n < t.numel(); ++n) {
        for (std::size_t d = 0; d < t.ndim(); ++d)
            out_indices[d] = (d == axis) ? (axis_size - 1 - in_indices[d]) : in_indices[d];

        auto off_in = compute_offset(t.strides(), in_indices.data(), t.ndim());
        auto off_out = compute_offset(result->strides(), out_indices.data(), result->ndim());
        std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_out,
                    static_cast<const std::byte*>(t.data_ptr()) + off_in,
                    elem_sz);
        advance_indices(in_indices.data(), t.shape());
    }
    return result;
}

// ---------------------------------------------------------------------------
// pad
// ---------------------------------------------------------------------------

auto pad(const Tensor& t, std::span<const std::size_t> widths, double value)
    -> std::expected<Tensor, std::string> {
    if (widths.size() != 2 * t.ndim())
        return std::unexpected("pad: widths must have 2 * ndim elements");

    std::vector<std::size_t> out_shape(t.ndim());
    for (std::size_t d = 0; d < t.ndim(); ++d)
        out_shape[d] = t.shape()[d] + widths[2 * d] + widths[2 * d + 1];

    auto result = Tensor::full(out_shape, value, t.dtype(), t.layout());
    if (!result) return result;

    auto elem_sz = dtype_size(t.dtype());
    std::vector<std::size_t> in_indices(t.ndim(), 0);
    std::vector<std::size_t> out_indices(t.ndim(), 0);

    for (std::size_t n = 0; n < t.numel(); ++n) {
        for (std::size_t d = 0; d < t.ndim(); ++d)
            out_indices[d] = in_indices[d] + widths[2 * d];

        auto off_in = compute_offset(t.strides(), in_indices.data(), t.ndim());
        auto off_out = compute_offset(result->strides(), out_indices.data(), result->ndim());
        std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_out,
                    static_cast<const std::byte*>(t.data_ptr()) + off_in,
                    elem_sz);
        advance_indices(in_indices.data(), t.shape());
    }
    return result;
}

// ---------------------------------------------------------------------------
// cumsum / cumprod
// ---------------------------------------------------------------------------

static auto cumulative_op(const Tensor& t, std::size_t dim, bool is_sum)
    -> std::expected<Tensor, std::string> {
    if (dim >= t.ndim())
        return std::unexpected(std::string(is_sum ? "cumsum" : "cumprod") + ": dim out of range");

    auto result = Tensor::empty(t.shape(), t.dtype(), t.layout());
    if (!result) return result;

    // Build outer shape (all dims except dim)
    std::vector<std::size_t> outer_shape;
    for (std::size_t d = 0; d < t.ndim(); ++d)
        if (d != dim) outer_shape.push_back(t.shape()[d]);

    std::size_t outer_numel = 1;
    for (auto s : outer_shape) outer_numel *= s;

    std::vector<std::size_t> outer_indices(outer_shape.size(), 0);
    std::vector<std::size_t> full_indices(t.ndim(), 0);

    for (std::size_t n = 0; n < outer_numel; ++n) {
        std::size_t j = 0;
        for (std::size_t d = 0; d < t.ndim(); ++d) {
            if (d != dim) full_indices[d] = outer_indices[j++];
        }

        double acc = is_sum ? 0.0 : 1.0;
        for (std::size_t k = 0; k < t.shape()[dim]; ++k) {
            full_indices[dim] = k;
            auto off_in = compute_offset(t.strides(), full_indices.data(), t.ndim());
            double val = read_as_double(
                static_cast<const std::byte*>(t.data_ptr()) + off_in, t.dtype());
            acc = is_sum ? (acc + val) : (acc * val);
            auto off_out = compute_offset(result->strides(), full_indices.data(), result->ndim());
            write_from_double(
                static_cast<std::byte*>(result->data_ptr()) + off_out, t.dtype(), acc);
        }

        if (!outer_shape.empty())
            advance_indices(outer_indices.data(), outer_shape);
    }
    return result;
}

auto cumsum(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string> {
    return cumulative_op(t, dim, true);
}

auto cumprod(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string> {
    return cumulative_op(t, dim, false);
}

// ---------------------------------------------------------------------------
// sort / argsort
// ---------------------------------------------------------------------------

auto sort(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string> {
    if (dim >= t.ndim())
        return std::unexpected("sort: dim out of range");

    auto out = t.clone();
    auto dim_size = t.shape()[dim];

    std::vector<std::size_t> outer_shape;
    for (std::size_t d = 0; d < t.ndim(); ++d)
        if (d != dim) outer_shape.push_back(t.shape()[d]);

    std::size_t outer_numel = 1;
    for (auto s : outer_shape) outer_numel *= s;

    std::vector<std::size_t> outer_indices(outer_shape.size(), 0);
    std::vector<std::size_t> full_indices(t.ndim(), 0);
    std::vector<double> fiber(dim_size);

    for (std::size_t n = 0; n < outer_numel; ++n) {
        std::size_t j = 0;
        for (std::size_t d = 0; d < t.ndim(); ++d) {
            if (d != dim) full_indices[d] = outer_indices[j++];
        }

        // Read fiber
        for (std::size_t k = 0; k < dim_size; ++k) {
            full_indices[dim] = k;
            auto off = compute_offset(out.strides(), full_indices.data(), out.ndim());
            fiber[k] = read_as_double(
                static_cast<const std::byte*>(out.data_ptr()) + off, out.dtype());
        }

        std::sort(fiber.begin(), fiber.end());

        // Write back
        for (std::size_t k = 0; k < dim_size; ++k) {
            full_indices[dim] = k;
            auto off = compute_offset(out.strides(), full_indices.data(), out.ndim());
            write_from_double(
                static_cast<std::byte*>(out.data_ptr()) + off, out.dtype(), fiber[k]);
        }

        if (!outer_shape.empty())
            advance_indices(outer_indices.data(), outer_shape);
    }
    return out;
}

auto argsort(const Tensor& t, std::size_t dim)
    -> std::expected<Tensor, std::string> {
    if (dim >= t.ndim())
        return std::unexpected("argsort: dim out of range");

    auto result = Tensor::empty(t.shape(), DType::Int64);
    if (!result) return result;

    auto src = t.contiguous();
    auto dim_size = t.shape()[dim];

    std::vector<std::size_t> outer_shape;
    for (std::size_t d = 0; d < t.ndim(); ++d)
        if (d != dim) outer_shape.push_back(t.shape()[d]);

    std::size_t outer_numel = 1;
    for (auto s : outer_shape) outer_numel *= s;

    std::vector<std::size_t> outer_indices(outer_shape.size(), 0);
    std::vector<std::size_t> full_indices(t.ndim(), 0);
    std::vector<double> fiber(dim_size);
    std::vector<std::int64_t> idx(dim_size);

    for (std::size_t n = 0; n < outer_numel; ++n) {
        std::size_t j = 0;
        for (std::size_t d = 0; d < t.ndim(); ++d) {
            if (d != dim) full_indices[d] = outer_indices[j++];
        }

        for (std::size_t k = 0; k < dim_size; ++k) {
            full_indices[dim] = k;
            auto off = compute_offset(src.strides(), full_indices.data(), src.ndim());
            fiber[k] = read_as_double(
                static_cast<const std::byte*>(src.data_ptr()) + off, src.dtype());
            idx[k] = static_cast<std::int64_t>(k);
        }

        std::sort(idx.begin(), idx.end(),
                  [&](std::int64_t a, std::int64_t b) {
                      return fiber[static_cast<std::size_t>(a)] <
                             fiber[static_cast<std::size_t>(b)];
                  });

        for (std::size_t k = 0; k < dim_size; ++k) {
            full_indices[dim] = k;
            auto off = compute_offset(result->strides(), full_indices.data(), result->ndim());
            *reinterpret_cast<std::int64_t*>(
                static_cast<std::byte*>(result->data_ptr()) + off) = idx[k];
        }

        if (!outer_shape.empty())
            advance_indices(outer_indices.data(), outer_shape);
    }
    return result;
}

// ---------------------------------------------------------------------------
// diag
// ---------------------------------------------------------------------------

auto diag(const Tensor& t, std::ptrdiff_t offset)
    -> std::expected<Tensor, std::string> {
    if (t.ndim() == 1) {
        // Create diagonal matrix from 1D input
        auto n = t.numel();
        auto abs_off = static_cast<std::size_t>(std::abs(offset));
        auto size = n + abs_off;
        auto result = Tensor::zeros({size, size}, t.dtype());
        if (!result) return result;

        auto elem_sz = dtype_size(t.dtype());
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t row = (offset >= 0) ? i : i + abs_off;
            std::size_t col = (offset >= 0) ? i + abs_off : i;
            std::size_t src_idx[] = {i};
            std::size_t dst_idx[] = {row, col};
            auto off_src = compute_offset(t.strides(), src_idx, 1);
            auto off_dst = compute_offset(result->strides(), dst_idx, 2);
            std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_dst,
                        static_cast<const std::byte*>(t.data_ptr()) + off_src,
                        elem_sz);
        }
        return result;
    }

    if (t.ndim() == 2) {
        // Extract diagonal from 2D input
        auto M = t.shape()[0];
        auto N = t.shape()[1];
        std::size_t diag_len;
        std::size_t start_row;
        std::size_t start_col;
        if (offset >= 0) {
            auto off = static_cast<std::size_t>(offset);
            if (off >= N)
                return std::unexpected("diag: offset out of range");
            diag_len = std::min(M, N - off);
            start_row = 0;
            start_col = off;
        } else {
            auto off = static_cast<std::size_t>(-offset);
            if (off >= M)
                return std::unexpected("diag: offset out of range");
            diag_len = std::min(M - off, N);
            start_row = off;
            start_col = 0;
        }

        auto result = Tensor::empty({diag_len}, t.dtype());
        if (!result) return result;

        auto elem_sz = dtype_size(t.dtype());
        for (std::size_t i = 0; i < diag_len; ++i) {
            std::size_t src_idx[] = {start_row + i, start_col + i};
            auto off_src = compute_offset(t.strides(), src_idx, 2);
            auto off_dst = static_cast<std::ptrdiff_t>(i) * result->strides()[0];
            std::memcpy(static_cast<std::byte*>(result->data_ptr()) + off_dst,
                        static_cast<const std::byte*>(t.data_ptr()) + off_src,
                        elem_sz);
        }
        return result;
    }

    return std::unexpected("diag: input must be 1D or 2D");
}

} // namespace utils
