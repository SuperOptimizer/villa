#pragma once
/// vc4d::ZarrDataset — Minimal OME-Zarr reader.
///
/// Replaces vc3d's dependency on z5 + xtensor for zarr access.
/// We only need to read chunked uint8/uint16 3D arrays with blosc
/// compression, which is ~200 lines of code, not two large libraries.
///
/// The zarr format is simple:
///   <root>/.zarray          — JSON metadata (shape, chunks, dtype, compressor)
///   <root>/0.0.0, 0.0.1 ... — compressed chunk files
///
/// This implementation handles:
///   • blosc-compressed chunks (zstd, lz4, zlib codecs)
///   • C-order (row-major) chunk layout
///   • fill_value = 0 for missing chunks

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc4d {

// ---------------------------------------------------------------------------
// ChunkCoord — identifies a chunk in the grid
// ---------------------------------------------------------------------------
struct ChunkCoord {
    int z, y, x;

    bool operator==(const ChunkCoord&) const = default;
};

struct ChunkCoordHash {
    size_t operator()(ChunkCoord c) const {
        // Simple spatial hash — good enough for chunk lookups.
        size_t h = static_cast<size_t>(c.z) * 73856093;
        h ^= static_cast<size_t>(c.y) * 19349663;
        h ^= static_cast<size_t>(c.x) * 83492791;
        return h;
    }
};

// ---------------------------------------------------------------------------
// ZarrDataset — one scale level of a zarr array
// ---------------------------------------------------------------------------
class ZarrDataset {
public:
    explicit ZarrDataset(std::filesystem::path root);

    [[nodiscard]] const std::array<int, 3>& shape()       const { return shape_; }
    [[nodiscard]] const std::array<int, 3>& chunk_shape() const { return chunk_shape_; }
    [[nodiscard]] const std::string&         dtype()      const { return dtype_; }
    [[nodiscard]] const std::filesystem::path& path()     const { return root_; }

    // Number of chunks along each axis.
    [[nodiscard]] std::array<int, 3> chunk_grid_shape() const;

    // Read a chunk into the output buffer. Returns false if chunk doesn't exist
    // (fill with zeros). Output buffer must be chunk_shape[0]*[1]*[2] bytes.
    [[nodiscard]] bool read_chunk(ChunkCoord coord, std::span<uint8_t> out) const;

    // Read raw compressed chunk bytes from disk.
    [[nodiscard]] std::vector<uint8_t> read_chunk_raw(ChunkCoord coord) const;

    // Decompress a raw chunk into output buffer.
    static bool decompress(std::span<const uint8_t> compressed,
                           std::span<uint8_t> out,
                           const nlohmann::json& compressor_meta);

    // Chunk file path for a given coordinate.
    [[nodiscard]] std::filesystem::path chunk_path(ChunkCoord coord) const;

private:
    std::filesystem::path root_;
    nlohmann::json zarray_;

    std::array<int, 3> shape_{};
    std::array<int, 3> chunk_shape_{};
    std::string dtype_;
    std::string separator_{"."};
    nlohmann::json compressor_;
};

} // namespace vc4d
