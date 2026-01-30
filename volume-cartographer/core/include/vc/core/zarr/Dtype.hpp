#pragma once

#include <cstddef>
#include <string>

namespace volcart::zarr
{

enum class Dtype { UInt8, UInt16, Float32, Unknown };

std::string dtypeToString(Dtype dtype);
Dtype dtypeFromString(const std::string& s);
std::size_t dtypeSize(Dtype dtype);

}  // namespace volcart::zarr
