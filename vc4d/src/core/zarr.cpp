#include "vc4d/core/zarr.hpp"

#include <fstream>
#include <stdexcept>

namespace vc4d {

ZarrDataset::ZarrDataset(std::filesystem::path root)
    : root_(std::move(root))
{
    auto zarray_path = root_ / ".zarray";
    if (!std::filesystem::exists(zarray_path))
        throw std::runtime_error("Missing .zarray at " + root_.string());

    std::ifstream f(zarray_path);
    zarray_ = nlohmann::json::parse(f);

    // Parse shape (zarr stores as [z, y, x] for C-order)
    auto& shape_arr = zarray_["shape"];
    shape_ = {shape_arr[0].get<int>(), shape_arr[1].get<int>(), shape_arr[2].get<int>()};

    // Parse chunk shape
    auto& chunks_arr = zarray_["chunks"];
    chunk_shape_ = {chunks_arr[0].get<int>(), chunks_arr[1].get<int>(), chunks_arr[2].get<int>()};

    // Parse dtype
    dtype_ = zarray_.value("dtype", "|u1");

    // Parse chunk separator
    if (zarray_.contains("dimension_separator"))
        separator_ = zarray_["dimension_separator"].get<std::string>();

    // Parse compressor
    if (zarray_.contains("compressor") && !zarray_["compressor"].is_null())
        compressor_ = zarray_["compressor"];
}

std::array<int, 3> ZarrDataset::chunk_grid_shape() const {
    return {
        (shape_[0] + chunk_shape_[0] - 1) / chunk_shape_[0],
        (shape_[1] + chunk_shape_[1] - 1) / chunk_shape_[1],
        (shape_[2] + chunk_shape_[2] - 1) / chunk_shape_[2]
    };
}

std::filesystem::path ZarrDataset::chunk_path(ChunkCoord coord) const {
    return root_ / (std::to_string(coord.z) + separator_ +
                    std::to_string(coord.y) + separator_ +
                    std::to_string(coord.x));
}

std::vector<uint8_t> ZarrDataset::read_chunk_raw(ChunkCoord coord) const {
    auto path = chunk_path(coord);
    if (!std::filesystem::exists(path))
        return {};

    auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> data(size);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

bool ZarrDataset::read_chunk(ChunkCoord coord, std::span<uint8_t> out) const {
    auto raw = read_chunk_raw(coord);
    if (raw.empty()) {
        // Fill with zeros (zarr fill_value = 0)
        std::ranges::fill(out, uint8_t{0});
        return false;
    }

    if (compressor_.is_null() || compressor_.empty()) {
        // Uncompressed — direct copy
        auto n = std::min(raw.size(), out.size());
        std::copy_n(raw.begin(), n, out.begin());
        return true;
    }

    return decompress(raw, out, compressor_);
}

bool ZarrDataset::decompress(std::span<const uint8_t> compressed,
                              std::span<uint8_t> out,
                              const nlohmann::json& compressor_meta)
{
    auto codec = compressor_meta.value("id", "");

    if (codec == "blosc") {
        // TODO: Call blosc_decompress when blosc is linked.
        // For now, this is a stub — blosc decompression will be added
        // when we integrate the blosc2 library (the only C dependency
        // we plan to keep besides Qt).
        (void)compressed;
        std::ranges::fill(out, uint8_t{0});
        return false;
    }

    // Unknown compressor
    std::ranges::fill(out, uint8_t{0});
    return false;
}

} // namespace vc4d
