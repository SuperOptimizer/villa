#include "vc/core/cache/DiskStore.hpp"

#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::cache {

DiskStore::DiskStore(Config config) : config_(std::move(config))
{
    if (!config_.root.empty()) {
        std::filesystem::create_directories(config_.root);
    }
}

DiskStore::~DiskStore()
{
    if (!config_.persistent && !config_.root.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(config_.root, ec);
    }
}

std::filesystem::path DiskStore::chunkPath(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    auto base = config_.directMode
                    ? config_.root
                    : config_.root / volumeId;
    const auto& d = config_.delimiter;
    std::string name = std::to_string(key.iz) + d +
                       std::to_string(key.iy) + d +
                       std::to_string(key.ix);
    return base / std::to_string(key.level) / name;
}

size_t DiskStore::lockIndex(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    size_t h = std::hash<std::string>()(volumeId);
    h ^= ChunkKeyHash()(key);
    return h % kLockPoolSize;
}

std::optional<std::vector<uint8_t>> DiskStore::get(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    auto path = chunkPath(volumeId, key);

    int fd = ::open(path.c_str(), O_RDONLY | O_NOATIME);
    if (fd < 0) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return std::nullopt;
    }

    struct stat sb;
    if (::fstat(fd, &sb) != 0) {
        ::close(fd);
        return std::nullopt;
    }

    auto fileSize = static_cast<size_t>(sb.st_size);
    if (fileSize == 0) {
        ::close(fd);
        return std::nullopt;
    }

    std::vector<uint8_t> buf(fileSize);
    size_t total = 0;
    while (total < fileSize) {
        ssize_t n = ::read(fd, buf.data() + total, fileSize - total);
        if (n <= 0) {
            ::close(fd);
            return std::nullopt;
        }
        total += static_cast<size_t>(n);
    }
    ::close(fd);

    // Touch mtime for LRU (best effort, ignore failures)
    ::utimensat(AT_FDCWD, path.c_str(), nullptr, 0);

    return buf;
}

void DiskStore::put(
    const std::string& volumeId,
    const ChunkKey& key,
    const uint8_t* data,
    size_t size)
{
    auto path = chunkPath(volumeId, key);

    // Serialize writes to same key
    std::lock_guard lock(lockPool_[lockIndex(volumeId, key)]);

    // Create parent directories
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return;

    // Write to temp file, then rename (atomic on same filesystem)
    auto tmpPath = path;
    tmpPath += ".tmp";

    int fd = ::open(tmpPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return;

    size_t written = 0;
    while (written < size) {
        ssize_t n = ::write(fd, data + written, size - written);
        if (n <= 0) {
            ::close(fd);
            ::unlink(tmpPath.c_str());
            return;
        }
        written += static_cast<size_t>(n);
    }
    ::close(fd);

    // Atomic rename
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) {
        ::unlink(tmpPath.c_str());
    }
}

bool DiskStore::has(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    return std::filesystem::exists(chunkPath(volumeId, key));
}

void DiskStore::remove(
    const std::string& volumeId,
    const ChunkKey& key)
{
    std::error_code ec;
    std::filesystem::remove(chunkPath(volumeId, key), ec);
}

void DiskStore::evictToSize(size_t targetBytes)
{
    if (config_.root.empty()) return;

    // Collect all chunk files with their sizes and modification times
    struct FileInfo {
        std::filesystem::path path;
        size_t bytes;
        std::filesystem::file_time_type mtime;
    };

    std::vector<FileInfo> files;
    size_t totalSize = 0;

    std::error_code ec;
    for (auto& entry :
         std::filesystem::recursive_directory_iterator(config_.root, ec)) {
        if (!entry.is_regular_file()) continue;
        auto name = entry.path().filename().string();
        if (name.starts_with(".") || name.ends_with(".tmp") ||
            name.ends_with(".json"))
            continue;

        auto sz = entry.file_size();
        totalSize += sz;
        files.push_back(
            {entry.path(), static_cast<size_t>(sz), entry.last_write_time()});
    }

    if (totalSize <= targetBytes) return;

    // Sort oldest first
    std::sort(files.begin(), files.end(),
              [](const FileInfo& a, const FileInfo& b) {
                  return a.mtime < b.mtime;
              });

    // Remove oldest files until under target
    for (auto& fi : files) {
        if (totalSize <= targetBytes) break;
        std::filesystem::remove(fi.path, ec);
        if (!ec) {
            totalSize -= fi.bytes;
        }
    }
}

size_t DiskStore::totalBytes() const
{
    if (config_.root.empty() || !std::filesystem::exists(config_.root))
        return 0;

    size_t total = 0;
    std::error_code ec;
    for (auto& entry :
         std::filesystem::recursive_directory_iterator(config_.root, ec)) {
        if (entry.is_regular_file()) {
            total += static_cast<size_t>(entry.file_size());
        }
    }
    return total;
}

void DiskStore::clearVolume(const std::string& volumeId)
{
    auto volDir = config_.root / volumeId;
    std::error_code ec;
    std::filesystem::remove_all(volDir, ec);
}

void DiskStore::clearAll()
{
    if (config_.root.empty()) return;
    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(config_.root, ec)) {
        std::filesystem::remove_all(entry.path(), ec);
    }
}

}  // namespace vc::cache
