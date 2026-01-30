#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace volcart::zarr
{

/**
 * @brief Simple 3D tensor with row-major (C) memory layout.
 *
 * This class replaces xt::xtensor<T, 3> with a minimal implementation
 * that supports only the operations actually used in the volume-cartographer
 * codebase.
 *
 * Memory layout: Row-major (last index varies fastest)
 * Index calculation: i * shape[1] * shape[2] + j * shape[2] + k
 *
 * This matches zarr C-order storage where for shape [z,y,x], x varies fastest.
 *
 * @tparam T Element type (uint8_t, uint16_t, or float)
 */
template <typename T>
class Tensor3D
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::array<size_type, 3>;

    /** @brief Default constructor - creates empty tensor */
    Tensor3D() = default;

    /**
     * @brief Construct tensor with given shape (uninitialized data)
     * @param d0 Size of first dimension
     * @param d1 Size of second dimension
     * @param d2 Size of third dimension
     */
    Tensor3D(size_type d0, size_type d1, size_type d2);

    /**
     * @brief Construct tensor with given shape, filled with value
     * @param d0 Size of first dimension
     * @param d1 Size of second dimension
     * @param d2 Size of third dimension
     * @param fill Value to fill the tensor with
     */
    Tensor3D(size_type d0, size_type d1, size_type d2, T fill);

    /**
     * @brief Construct tensor from shape array (uninitialized data)
     * @param shape Array of dimension sizes {d0, d1, d2}
     */
    explicit Tensor3D(const shape_type& shape);

    /**
     * @brief Construct tensor from shape array with fill value
     * @param shape Array of dimension sizes {d0, d1, d2}
     * @param fill Value to fill the tensor with
     */
    Tensor3D(const shape_type& shape, T fill);

    /** @brief Copy constructor */
    Tensor3D(const Tensor3D& other) = default;

    /** @brief Move constructor */
    Tensor3D(Tensor3D&& other) noexcept = default;

    /** @brief Copy assignment */
    Tensor3D& operator=(const Tensor3D& other) = default;

    /** @brief Move assignment */
    Tensor3D& operator=(Tensor3D&& other) noexcept = default;

    ~Tensor3D() = default;

    /**
     * @brief Element access (mutable)
     * @param i First index
     * @param j Second index
     * @param k Third index
     * @return Reference to element at (i, j, k)
     */
    T& operator()(size_type i, size_type j, size_type k);

    /**
     * @brief Element access (const)
     * @param i First index
     * @param j Second index
     * @param k Third index
     * @return Const reference to element at (i, j, k)
     */
    const T& operator()(size_type i, size_type j, size_type k) const;

    /**
     * @brief Get pointer to raw data
     * @return Pointer to first element
     */
    T* data() noexcept { return data_.data(); }

    /**
     * @brief Get const pointer to raw data
     * @return Const pointer to first element
     */
    const T* data() const noexcept { return data_.data(); }

    /**
     * @brief Get tensor shape
     * @return Array of dimension sizes {d0, d1, d2}
     */
    const shape_type& shape() const noexcept { return shape_; }

    /**
     * @brief Get tensor strides (in elements, not bytes)
     * @return Array of strides for each dimension
     */
    shape_type strides() const noexcept;

    /**
     * @brief Get total number of elements
     * @return Product of all dimensions
     */
    size_type size() const noexcept
    {
        return shape_[0] * shape_[1] * shape_[2];
    }

    /**
     * @brief Get size in bytes
     * @return Total memory used by data
     */
    size_type nbytes() const noexcept { return size() * sizeof(T); }

    /**
     * @brief Check if tensor is empty
     * @return True if size is 0
     */
    bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Resize tensor (invalidates existing data)
     * @param d0 New size of first dimension
     * @param d1 New size of second dimension
     * @param d2 New size of third dimension
     */
    void resize(size_type d0, size_type d1, size_type d2);

    /**
     * @brief Fill all elements with value
     * @param value Value to fill with
     */
    void fill(T value);

private:
    std::vector<T> data_;
    shape_type shape_{0, 0, 0};
    size_type stride0_{0};  // shape_[1] * shape_[2]
    size_type stride1_{0};  // shape_[2]

    /** @brief Calculate linear index from 3D coordinates */
    size_type linearIndex(size_type i, size_type j, size_type k) const noexcept
    {
        return i * stride0_ + j * stride1_ + k;
    }
};

// Factory functions

/**
 * @brief Create uninitialized tensor
 * @tparam T Element type
 * @param d0 Size of first dimension
 * @param d1 Size of second dimension
 * @param d2 Size of third dimension
 * @return New tensor with uninitialized data
 */
template <typename T>
Tensor3D<T> empty(std::size_t d0, std::size_t d1, std::size_t d2);

/**
 * @brief Create zero-initialized tensor
 * @tparam T Element type
 * @param d0 Size of first dimension
 * @param d1 Size of second dimension
 * @param d2 Size of third dimension
 * @return New tensor filled with zeros
 */
template <typename T>
Tensor3D<T> zeros(std::size_t d0, std::size_t d1, std::size_t d2);

/**
 * @brief Create tensor filled with value
 * @tparam T Element type
 * @param d0 Size of first dimension
 * @param d1 Size of second dimension
 * @param d2 Size of third dimension
 * @param val Fill value
 * @return New tensor filled with val
 */
template <typename T>
Tensor3D<T> full(std::size_t d0, std::size_t d1, std::size_t d2, T val);

/**
 * @brief Create tensor filled with value, same shape as another
 * @tparam T Element type
 * @param other Tensor to copy shape from
 * @param val Fill value
 * @return New tensor with same shape as other, filled with val
 */
template <typename T>
Tensor3D<T> full_like(const Tensor3D<T>& other, T val);

/**
 * @brief Copy a view (subregion) from source to destination tensor
 *
 * Copies elements from src[z0:z1, y0:y1, x0:x1] to dst starting at (0,0,0).
 * Destination must be pre-allocated with appropriate size.
 *
 * @tparam T Element type
 * @param src Source tensor
 * @param dst Destination tensor (must be sized to hold the view)
 * @param z0 Start index for first dimension (inclusive)
 * @param z1 End index for first dimension (exclusive)
 * @param y0 Start index for second dimension (inclusive)
 * @param y1 End index for second dimension (exclusive)
 * @param x0 Start index for third dimension (inclusive)
 * @param x1 End index for third dimension (exclusive)
 */
template <typename T>
void copy_view(
    const Tensor3D<T>& src,
    Tensor3D<T>& dst,
    std::size_t z0,
    std::size_t z1,
    std::size_t y0,
    std::size_t y1,
    std::size_t x0,
    std::size_t x1);

// Explicit instantiation declarations for common types
extern template class Tensor3D<std::uint8_t>;
extern template class Tensor3D<std::uint16_t>;
extern template class Tensor3D<float>;

extern template Tensor3D<std::uint8_t> empty(std::size_t, std::size_t, std::size_t);
extern template Tensor3D<std::uint16_t> empty(std::size_t, std::size_t, std::size_t);
extern template Tensor3D<float> empty(std::size_t, std::size_t, std::size_t);

extern template Tensor3D<std::uint8_t> zeros(std::size_t, std::size_t, std::size_t);
extern template Tensor3D<std::uint16_t> zeros(std::size_t, std::size_t, std::size_t);
extern template Tensor3D<float> zeros(std::size_t, std::size_t, std::size_t);

extern template Tensor3D<std::uint8_t> full(std::size_t, std::size_t, std::size_t, std::uint8_t);
extern template Tensor3D<std::uint16_t> full(std::size_t, std::size_t, std::size_t, std::uint16_t);
extern template Tensor3D<float> full(std::size_t, std::size_t, std::size_t, float);

extern template Tensor3D<std::uint8_t> full_like(const Tensor3D<std::uint8_t>&, std::uint8_t);
extern template Tensor3D<std::uint16_t> full_like(const Tensor3D<std::uint16_t>&, std::uint16_t);
extern template Tensor3D<float> full_like(const Tensor3D<float>&, float);

extern template void copy_view(
    const Tensor3D<std::uint8_t>&, Tensor3D<std::uint8_t>&,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
extern template void copy_view(
    const Tensor3D<std::uint16_t>&, Tensor3D<std::uint16_t>&,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
extern template void copy_view(
    const Tensor3D<float>&, Tensor3D<float>&,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

}  // namespace volcart::zarr
