#pragma once
#include <array>
#include <cstddef>
#include "vc/core/zarr/Dtype.hpp"

class IChunkSource {
public:
    virtual ~IChunkSource() = default;

    virtual std::array<size_t, 3> volShape() const = 0;
    virtual std::array<size_t, 3> volChunkShape() const = 0;
    virtual volcart::zarr::Dtype volDtype() const = 0;
    virtual size_t volChunkElements() const = 0;
    virtual bool volReadChunk(size_t cz, size_t cy, size_t cx, void* buf) const = 0;
};
