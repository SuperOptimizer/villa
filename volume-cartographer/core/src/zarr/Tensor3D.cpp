#include "vc/core/zarr/Tensor3D.hpp"

#include <algorithm>
#include <cstring>

namespace volcart::zarr
{

// Constructor implementations

template <typename T>
Tensor3D<T>::Tensor3D(size_type d0, size_type d1, size_type d2)
    : data_(d0 * d1 * d2), shape_{d0, d1, d2}
{
}

template <typename T>
Tensor3D<T>::Tensor3D(size_type d0, size_type d1, size_type d2, T fill)
    : data_(d0 * d1 * d2, fill), shape_{d0, d1, d2}
{
}

template <typename T>
Tensor3D<T>::Tensor3D(const shape_type& shape)
    : data_(shape[0] * shape[1] * shape[2]), shape_(shape)
{
}

template <typename T>
Tensor3D<T>::Tensor3D(const shape_type& shape, T fill)
    : data_(shape[0] * shape[1] * shape[2], fill), shape_(shape)
{
}

// Element access

template <typename T>
T& Tensor3D<T>::operator()(size_type i, size_type j, size_type k)
{
    return data_[linearIndex(i, j, k)];
}

template <typename T>
const T& Tensor3D<T>::operator()(size_type i, size_type j, size_type k) const
{
    return data_[linearIndex(i, j, k)];
}

// Strides calculation

template <typename T>
typename Tensor3D<T>::shape_type Tensor3D<T>::strides() const noexcept
{
    // Row-major (C-order) strides (in elements)
    return {shape_[1] * shape_[2], shape_[2], 1};
}

// Resize

template <typename T>
void Tensor3D<T>::resize(size_type d0, size_type d1, size_type d2)
{
    shape_ = {d0, d1, d2};
    data_.resize(d0 * d1 * d2);
}

// Fill

template <typename T>
void Tensor3D<T>::fill(T value)
{
    std::fill(data_.begin(), data_.end(), value);
}

// Factory function implementations

template <typename T>
Tensor3D<T> empty(std::size_t d0, std::size_t d1, std::size_t d2)
{
    return Tensor3D<T>(d0, d1, d2);
}

template <typename T>
Tensor3D<T> zeros(std::size_t d0, std::size_t d1, std::size_t d2)
{
    return Tensor3D<T>(d0, d1, d2, T{0});
}

template <typename T>
Tensor3D<T> full(std::size_t d0, std::size_t d1, std::size_t d2, T val)
{
    return Tensor3D<T>(d0, d1, d2, val);
}

template <typename T>
Tensor3D<T> full_like(const Tensor3D<T>& other, T val)
{
    const auto& s = other.shape();
    return Tensor3D<T>(s[0], s[1], s[2], val);
}

template <typename T>
void copy_view(
    const Tensor3D<T>& src,
    Tensor3D<T>& dst,
    std::size_t z0,
    std::size_t z1,
    std::size_t y0,
    std::size_t y1,
    std::size_t x0,
    std::size_t x1)
{
    const auto& srcShape = src.shape();
    const auto& dstShape = dst.shape();

    // Validate bounds
    if (z1 > srcShape[0] || y1 > srcShape[1] || x1 > srcShape[2]) {
        throw std::out_of_range("copy_view: source range exceeds tensor bounds");
    }

    std::size_t viewZ = z1 - z0;
    std::size_t viewY = y1 - y0;
    std::size_t viewX = x1 - x0;

    if (viewZ > dstShape[0] || viewY > dstShape[1] || viewX > dstShape[2]) {
        throw std::out_of_range(
            "copy_view: destination too small for requested view");
    }

    // Copy element by element (could be optimized for contiguous regions)
    // For row-major, contiguous regions are along the last dimension (x)
    for (std::size_t i = 0; i < viewZ; ++i) {
        for (std::size_t j = 0; j < viewY; ++j) {
            // Copy contiguous run along dimension 2 (x)
            const T* srcPtr = &src(z0 + i, y0 + j, x0);
            T* dstPtr = &dst(i, j, 0);
            std::memcpy(dstPtr, srcPtr, viewX * sizeof(T));
        }
    }
}

// Explicit instantiations

template class Tensor3D<std::uint8_t>;
template class Tensor3D<std::uint16_t>;
template class Tensor3D<float>;

template Tensor3D<std::uint8_t> empty(std::size_t, std::size_t, std::size_t);
template Tensor3D<std::uint16_t> empty(std::size_t, std::size_t, std::size_t);
template Tensor3D<float> empty(std::size_t, std::size_t, std::size_t);

template Tensor3D<std::uint8_t> zeros(std::size_t, std::size_t, std::size_t);
template Tensor3D<std::uint16_t> zeros(std::size_t, std::size_t, std::size_t);
template Tensor3D<float> zeros(std::size_t, std::size_t, std::size_t);

template Tensor3D<std::uint8_t> full(std::size_t, std::size_t, std::size_t, std::uint8_t);
template Tensor3D<std::uint16_t> full(std::size_t, std::size_t, std::size_t, std::uint16_t);
template Tensor3D<float> full(std::size_t, std::size_t, std::size_t, float);

template Tensor3D<std::uint8_t> full_like(const Tensor3D<std::uint8_t>&, std::uint8_t);
template Tensor3D<std::uint16_t> full_like(const Tensor3D<std::uint16_t>&, std::uint16_t);
template Tensor3D<float> full_like(const Tensor3D<float>&, float);

template void copy_view(
    const Tensor3D<std::uint8_t>&, Tensor3D<std::uint8_t>&,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
template void copy_view(
    const Tensor3D<std::uint16_t>&, Tensor3D<std::uint16_t>&,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);
template void copy_view(
    const Tensor3D<float>&, Tensor3D<float>&,
    std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

}  // namespace volcart::zarr
