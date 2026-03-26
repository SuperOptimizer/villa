#include "vc4d/cache/disk_store.hpp"

#include <fstream>

namespace vc4d {

DiskStore::DiskStore(std::filesystem::path root, size_t max_bytes)
    : root_(std::move(root)), max_bytes_(max_bytes)
{
    std::filesystem::create_directories(root_);
}

std::filesystem::path DiskStore::chunk_file(int scale_level, ChunkCoord coord) const {
    auto dir = root_ / std::to_string(scale_level);
    return dir / (std::to_string(coord.z) + "." +
                  std::to_string(coord.y) + "." +
                  std::to_string(coord.x) + ".chunk");
}

void DiskStore::put(int scale_level, ChunkCoord coord, std::span<const uint8_t> data) {
    auto path = chunk_file(scale_level, coord);
    std::filesystem::create_directories(path.parent_path());

    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));

    usage_bytes_ += data.size();
    evict_to_budget();
}

std::optional<std::vector<uint8_t>> DiskStore::get(int scale_level, ChunkCoord coord) const {
    auto path = chunk_file(scale_level, coord);
    if (!std::filesystem::exists(path))
        return std::nullopt;

    auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> data(size);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

bool DiskStore::contains(int scale_level, ChunkCoord coord) const {
    return std::filesystem::exists(chunk_file(scale_level, coord));
}

void DiskStore::evict_to_budget() {
    // TODO: LRU eviction based on file modification times
}

void DiskStore::clear() {
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    usage_bytes_ = 0;
}

} // namespace vc4d
