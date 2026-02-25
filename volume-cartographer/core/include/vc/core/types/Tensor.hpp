#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vc {

enum class Layout { RowMajor, ColumnMajor };

// ---------- Range for view slicing ----------
struct Range {
    size_t start, stop;
};

inline Range range(size_t start, size_t stop) { return {start, stop}; }

// Forward declarations
template<typename T> class TensorView;
template<typename T> class TensorAdaptor;

// ---------- Tensor<T> — Owning N-D array ----------
template<typename T>
class Tensor {
public:
    using value_type = T;
    using shape_type = std::vector<size_t>;

    Tensor() = default;

    // Construct with shape and layout, uninitialized
    explicit Tensor(shape_type shape, Layout layout = Layout::RowMajor)
        : shape_(std::move(shape)), layout_(layout)
    {
        computeStrides();
        size_ = computeSize();
        data_ = std::make_unique<T[]>(size_);
    }

    // Construct with shape, layout, and fill value
    Tensor(shape_type shape, Layout layout, T fillValue)
        : shape_(std::move(shape)), layout_(layout)
    {
        computeStrides();
        size_ = computeSize();
        data_ = std::make_unique<T[]>(size_);
        std::fill_n(data_.get(), size_, fillValue);
    }

    // Copy
    Tensor(const Tensor& other)
        : shape_(other.shape_), strides_(other.strides_), layout_(other.layout_), size_(other.size_)
    {
        if (other.data_) {
            data_ = std::make_unique<T[]>(size_);
            std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(T));
        }
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            shape_ = other.shape_;
            strides_ = other.strides_;
            layout_ = other.layout_;
            size_ = other.size_;
            if (other.data_) {
                data_ = std::make_unique<T[]>(size_);
                std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(T));
            } else {
                data_.reset();
            }
        }
        return *this;
    }

    // Move
    Tensor(Tensor&& o) noexcept
        : data_(std::move(o.data_)), shape_(std::move(o.shape_)),
          strides_(std::move(o.strides_)), size_(o.size_), layout_(o.layout_)
    { o.size_ = 0; }
    Tensor& operator=(Tensor&& o) noexcept {
        data_ = std::move(o.data_); shape_ = std::move(o.shape_);
        strides_ = std::move(o.strides_); size_ = o.size_;
        layout_ = o.layout_; o.size_ = 0; return *this;
    }

    // Assign from TensorView (copy data from view into owning tensor)
    Tensor& operator=(const TensorView<T>& view);

    // Assign from const TensorView (copy data from const view into owning tensor)
    Tensor& operator=(const TensorView<const T>& view);

    // Element access — variadic for any dimensionality
    template<typename... Indices>
    T& operator()(Indices... indices) {
        const size_t idx[] = {static_cast<size_t>(indices)...};
        return data_[offset(idx, sizeof...(indices))];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        const size_t idx[] = {static_cast<size_t>(indices)...};
        return data_[offset(idx, sizeof...(indices))];
    }

    T* data() { return data_.get(); }
    const T* data() const { return data_.get(); }

    const shape_type& shape() const { return shape_; }
    size_t shape(size_t dim) const { return shape_[dim]; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    Layout layout() const { return layout_; }
    const shape_type& strides() const { return strides_; }

    void fill(T value) { std::fill_n(data_.get(), size_, value); }

    // Resize (reallocates, old data lost)
    void resize(shape_type newShape) {
        shape_ = std::move(newShape);
        computeStrides();
        size_ = computeSize();
        data_ = std::make_unique<T[]>(size_);
    }

private:
    shape_type shape_;
    shape_type strides_;
    Layout layout_ = Layout::RowMajor;
    size_t size_ = 0;
    std::unique_ptr<T[]> data_;

    void computeStrides() {
        const size_t nd = shape_.size();
        strides_.resize(nd);
        if (nd == 0) return;
        if (layout_ == Layout::ColumnMajor) {
            strides_[0] = 1;
            for (size_t i = 1; i < nd; i++)
                strides_[i] = strides_[i-1] * shape_[i-1];
        } else {
            strides_[nd-1] = 1;
            for (size_t i = nd-1; i > 0; i--)
                strides_[i-1] = strides_[i] * shape_[i];
        }
    }

    size_t computeSize() const {
        if (shape_.empty()) return 0;
        size_t s = 1;
        for (auto d : shape_) s *= d;
        return s;
    }

    size_t offset(const size_t* idx, size_t n) const {
        size_t off = 0;
        for (size_t i = 0; i < n; i++)
            off += idx[i] * strides_[i];
        return off;
    }

    // Friends
    template<typename U> friend class TensorView;
    template<typename U> friend class TensorAdaptor;
    template<typename U> friend TensorView<U> view(Tensor<U>& t, std::initializer_list<Range> ranges);
    template<typename U> friend TensorView<const U> view(const Tensor<U>& t, std::initializer_list<Range> ranges);
};

// ---------- TensorView<T> — Non-owning sub-view ----------
template<typename T>
class TensorView {
public:
    using value_type = T;
    using shape_type = std::vector<size_t>;

    TensorView(T* base, shape_type shape, shape_type strides, shape_type viewStrides)
        : base_(base), shape_(std::move(shape)), parentStrides_(std::move(strides)),
          viewShape_(shape_) // viewShape_ same as shape_ for simple case
    {
    }

    TensorView(T* base, shape_type shape, shape_type parentStrides, size_t baseOffset)
        : base_(base + baseOffset), shape_(std::move(shape)), parentStrides_(std::move(parentStrides))
    {
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        const size_t idx[] = {static_cast<size_t>(indices)...};
        size_t off = 0;
        for (size_t i = 0; i < sizeof...(indices); i++)
            off += idx[i] * parentStrides_[i];
        return base_[off];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        const size_t idx[] = {static_cast<size_t>(indices)...};
        size_t off = 0;
        for (size_t i = 0; i < sizeof...(indices); i++)
            off += idx[i] * parentStrides_[i];
        return base_[off];
    }

    const shape_type& shape() const { return shape_; }
    size_t shape(size_t dim) const { return shape_[dim]; }
    size_t size() const {
        size_t s = 1;
        for (auto d : shape_) s *= d;
        return s;
    }
    size_t ndim() const { return shape_.size(); }

    // Copy view data into a contiguous Tensor
    Tensor<T> toTensor(Layout layout = Layout::RowMajor) const {
        Tensor<T> result(shape_, layout);
        copyTo(result.data(), result.strides());
        return result;
    }

    T* data() { return base_; }
    const T* data() const { return base_; }
    const shape_type& parentStrides() const { return parentStrides_; }

private:
    void copyTo(T* dst, const shape_type& dstStrides) const {
        // Generic N-D copy
        const size_t nd = shape_.size();
        std::vector<size_t> indices(nd, 0);
        const size_t total = size();
        for (size_t i = 0; i < total; i++) {
            size_t srcOff = 0, dstOff = 0;
            for (size_t d = 0; d < nd; d++) {
                srcOff += indices[d] * parentStrides_[d];
                dstOff += indices[d] * dstStrides[d];
            }
            dst[dstOff] = base_[srcOff];
            // Increment indices
            for (size_t d = nd; d-- > 0;) {
                if (++indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }

    T* base_;
    shape_type shape_;
    shape_type parentStrides_;
    shape_type viewShape_;
};

// Tensor assignment from view
template<typename T>
Tensor<T>& Tensor<T>::operator=(const TensorView<T>& v) {
    shape_ = v.shape();
    layout_ = Layout::RowMajor;
    computeStrides();
    size_ = computeSize();
    data_ = std::make_unique<T[]>(size_);
    // Copy from view
    const size_t nd = shape_.size();
    std::vector<size_t> indices(nd, 0);
    for (size_t i = 0; i < size_; i++) {
        size_t srcOff = 0, dstOff = 0;
        for (size_t d = 0; d < nd; d++) {
            srcOff += indices[d] * v.parentStrides()[d];
            dstOff += indices[d] * strides_[d];
        }
        data_[dstOff] = v.data()[srcOff];
        for (size_t d = nd; d-- > 0;) {
            if (++indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }
    return *this;
}

// Tensor assignment from const view (e.g., viewing a const Tensor)
template<typename T>
Tensor<T>& Tensor<T>::operator=(const TensorView<const T>& v) {
    shape_ = v.shape();
    layout_ = Layout::RowMajor;
    computeStrides();
    size_ = computeSize();
    data_ = std::make_unique<T[]>(size_);
    const size_t nd = shape_.size();
    std::vector<size_t> indices(nd, 0);
    for (size_t i = 0; i < size_; i++) {
        size_t srcOff = 0, dstOff = 0;
        for (size_t d = 0; d < nd; d++) {
            srcOff += indices[d] * v.parentStrides()[d];
            dstOff += indices[d] * strides_[d];
        }
        data_[dstOff] = v.data()[srcOff];
        for (size_t d = nd; d-- > 0;) {
            if (++indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }
    return *this;
}

// ---------- TensorAdaptor<T> — Non-owning wrapper around raw pointer ----------
template<typename T>
class TensorAdaptor {
public:
    using value_type = T;
    using shape_type = std::vector<size_t>;

    TensorAdaptor(T* data, shape_type shape, Layout layout = Layout::RowMajor)
        : data_(data), shape_(std::move(shape)), layout_(layout)
    {
        computeStrides();
        size_ = computeSize();
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        const size_t idx[] = {static_cast<size_t>(indices)...};
        return data_[offset(idx, sizeof...(indices))];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        const size_t idx[] = {static_cast<size_t>(indices)...};
        return data_[offset(idx, sizeof...(indices))];
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    const shape_type& shape() const { return shape_; }
    size_t shape(size_t dim) const { return shape_[dim]; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    Layout layout() const { return layout_; }
    const shape_type& strides() const { return strides_; }

private:
    size_t computeSize() const {
        if (shape_.empty()) return 0;
        size_t s = 1;
        for (auto d : shape_) s *= d;
        return s;
    }

    void computeStrides() {
        const size_t nd = shape_.size();
        strides_.resize(nd);
        if (nd == 0) return;
        if (layout_ == Layout::ColumnMajor) {
            strides_[0] = 1;
            for (size_t i = 1; i < nd; i++)
                strides_[i] = strides_[i-1] * shape_[i-1];
        } else {
            strides_[nd-1] = 1;
            for (size_t i = nd-1; i > 0; i--)
                strides_[i-1] = strides_[i] * shape_[i];
        }
    }

    size_t offset(const size_t* idx, size_t n) const {
        size_t off = 0;
        for (size_t i = 0; i < n; i++)
            off += idx[i] * strides_[i];
        return off;
    }

    T* data_;
    shape_type shape_;
    shape_type strides_;
    Layout layout_;
    size_t size_ = 0;
};

// ---------- Factory functions ----------

template<typename T>
Tensor<T> empty(std::vector<size_t> shape, Layout layout = Layout::RowMajor) {
    return Tensor<T>(std::move(shape), layout);
}

// Overload accepting initializer_list (matches xt::empty<T>({z,y,x}) pattern)
template<typename T>
Tensor<T> empty(std::initializer_list<size_t> shape, Layout layout = Layout::RowMajor) {
    return Tensor<T>(std::vector<size_t>(shape), layout);
}

template<typename T>
Tensor<T> zeros(std::vector<size_t> shape, Layout layout = Layout::RowMajor) {
    return Tensor<T>(std::move(shape), layout, T(0));
}

template<typename T>
Tensor<T> zeros(std::initializer_list<size_t> shape, Layout layout = Layout::RowMajor) {
    return Tensor<T>(std::vector<size_t>(shape), layout, T(0));
}

template<typename T>
Tensor<T> full(std::vector<size_t> shape, T value, Layout layout = Layout::RowMajor) {
    return Tensor<T>(std::move(shape), layout, value);
}

template<typename T, typename U>
Tensor<T> full_like(const Tensor<T>& ref, U value) {
    return Tensor<T>(ref.shape(), ref.layout(), static_cast<T>(value));
}

template<typename T, typename U>
Tensor<T> full_like(const TensorView<T>& ref, U value) {
    // Create a new tensor with the same shape, default layout, filled with value
    return Tensor<T>(ref.shape(), Layout::RowMajor, static_cast<T>(value));
}

// ---------- View creation ----------

// View with Range slices
template<typename T>
TensorView<T> view(Tensor<T>& t, std::initializer_list<Range> ranges) {
    std::vector<Range> rv(ranges);
    const size_t nd = t.ndim();
    assert(rv.size() == nd);

    size_t baseOffset = 0;
    std::vector<size_t> newShape(nd);
    for (size_t i = 0; i < nd; i++) {
        baseOffset += rv[i].start * t.strides_[i];
        newShape[i] = rv[i].stop - rv[i].start;
    }
    return TensorView<T>(t.data(), std::move(newShape), t.strides_, baseOffset);
}

template<typename T>
TensorView<const T> view(const Tensor<T>& t, std::initializer_list<Range> ranges) {
    std::vector<Range> rv(ranges);
    const size_t nd = t.ndim();
    assert(rv.size() == nd);

    size_t baseOffset = 0;
    std::vector<size_t> newShape(nd);
    for (size_t i = 0; i < nd; i++) {
        baseOffset += rv[i].start * t.strides_[i];
        newShape[i] = rv[i].stop - rv[i].start;
    }
    return TensorView<const T>(t.data(), std::move(newShape), t.strides_, baseOffset);
}

// View from adaptor
template<typename T>
TensorView<T> view(TensorAdaptor<T>& t, std::initializer_list<Range> ranges) {
    std::vector<Range> rv(ranges);
    const size_t nd = t.ndim();
    assert(rv.size() == nd);

    size_t baseOffset = 0;
    std::vector<size_t> newShape(nd);
    for (size_t i = 0; i < nd; i++) {
        baseOffset += rv[i].start * t.strides()[i];
        newShape[i] = rv[i].stop - rv[i].start;
    }
    return TensorView<T>(t.data(), std::move(newShape), t.strides(), baseOffset);
}

// ---------- Adapt (wrap raw pointer) ----------

template<typename T>
TensorAdaptor<T> adapt(T* data, std::vector<size_t> shape, Layout layout = Layout::RowMajor) {
    return TensorAdaptor<T>(data, std::move(shape), layout);
}

template<typename T>
TensorAdaptor<T> adapt(T* data, std::initializer_list<size_t> shape, Layout layout = Layout::RowMajor) {
    return TensorAdaptor<T>(data, std::vector<size_t>(shape), layout);
}

} // namespace vc
