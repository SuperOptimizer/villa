#pragma once
/// vc4d::DiskStore — Persistent on-disk chunk cache.
///
/// Simple key→file mapping using the chunk coordinate as the filename.
/// Stores compressed chunks to minimize disk usage.

#include "vc4d/core/zarr.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace vc4d {

class DiskStore {
public:
    explicit DiskStore(std::filesystem::path root, size_t max_bytes = 100ULL << 30);

    // Store compressed chunk bytes on disk.
    void put(int scale_level, ChunkCoord coord, std::span<const uint8_t> data);

    // Retrieve compressed chunk bytes. Returns nullopt if not cached.
    [[nodiscard]] std::optional<std::vector<uint8_t>> get(int scale_level, ChunkCoord coord) const;

    // Check if chunk exists on disk.
    [[nodiscard]] bool contains(int scale_level, ChunkCoord coord) const;

    [[nodiscard]] size_t usage_bytes() const { return usage_bytes_; }
    void evict_to_budget();
    void clear();

private:
    [[nodiscard]] std::filesystem::path chunk_file(int scale_level, ChunkCoord coord) const;

    std::filesystem::path root_;
    size_t max_bytes_;
    size_t usage_bytes_{};
};

} // namespace vc4d
