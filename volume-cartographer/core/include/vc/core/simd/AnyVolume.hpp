#pragma once

#include "SparseVolume.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <variant>

namespace vc::simd {

// --------------------------------------------------------------------------
// AnyVolume<T> — type-erased runtime dispatch for volumes
// --------------------------------------------------------------------------
template <typename T>
class AnyVolume {
public:
    virtual ~AnyVolume() = default;
    virtual T at(int z, int y, int x) = 0;
    virtual std::array<int, 3> shape() const = 0;
    virtual int chunk_size() const = 0;
};

// Concrete implementation wrapping SparseVolume
template <typename T, int N>
class AnyVolumeImpl : public AnyVolume<T> {
public:
    AnyVolumeImpl(vc::zarr::Dataset* ds, ChunkCache<T>* cache)
        : vol_(ds, cache) {}

    T at(int z, int y, int x) override { return vol_.at(z, y, x); }
    std::array<int, 3> shape() const override { return vol_.shape(); }
    int chunk_size() const override { return N; }

    SparseVolume<T, N>& volume() { return vol_; }
    const SparseVolume<T, N>& volume() const { return vol_; }

private:
    SparseVolume<T, N> vol_;
};

// --------------------------------------------------------------------------
// VolumeVariant — std::variant dispatch (no virtual overhead)
// --------------------------------------------------------------------------
template <typename T>
using VolumeVariant = std::variant<
    SparseVolume<T, 8>,
    SparseVolume<T, 16>,
    SparseVolume<T, 32>,
    SparseVolume<T, 64>,
    SparseVolume<T, 128>>;

// --------------------------------------------------------------------------
// Factory functions (implemented in SparseVolume.cpp, need full zarr type)
// --------------------------------------------------------------------------

template <typename T>
VolumeVariant<T> make_volume_variant(vc::zarr::Dataset* ds, ChunkCache<T>* cache);

template <typename T>
std::unique_ptr<AnyVolume<T>> make_volume(vc::zarr::Dataset* ds, ChunkCache<T>* cache);

}  // namespace vc::simd
