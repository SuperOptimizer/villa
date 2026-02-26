#pragma once

#include <cstddef>
#include <cstdint>

namespace fnv {

// FNV-1a constants for 64-bit
inline constexpr uint64_t BASIS = 14695981039346656037ULL;
inline constexpr uint64_t PRIME = 1099511628211ULL;

// FNV-1a hash over raw bytes
inline uint64_t hash(const void* data, size_t len)
{
    uint64_t h = BASIS;
    for (size_t i = 0; i < len; i++)
        h = (h ^ static_cast<const uint8_t*>(data)[i]) * PRIME;
    return h;
}

// Mix a single size_t value into an FNV-1a hash state
inline void mix(uint64_t& h, uint64_t val)
{
    h ^= val;
    h *= PRIME;
}

}  // namespace fnv
