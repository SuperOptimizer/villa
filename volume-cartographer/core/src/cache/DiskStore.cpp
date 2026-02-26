#include "vc/core/cache/DiskStore.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"
#include "vc/core/cache/CacheUtils.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::cache {

DiskStore::DiskStore(Config config) : config_(std::move(config))
{
    if (!config_.root.empty()) {
        std::filesystem::create_directories(config_.root);
        if (std::filesystem::exists(config_.root)) {
            initTotalBytes();
        }
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
    return base / std::to_string(key.level) / chunkFilename(key, config_.delimiter);
}

size_t DiskStore::lockIndex(
    const std::string& volumeId,
    const ChunkKey& key) const noexcept
{
    size_t h = std::hash<std::string>()(volumeId);
    h ^= ChunkKeyHash()(key);
    return h % kLockPoolSize;
}

std::optional<std::vector<uint8_t>> DiskStore::get(
    const std::string& volumeId,
    const ChunkKey& key) const
{
    return readFileToVector(chunkPath(volumeId, key));
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
    if (ec) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[DISK] put: create_directories FAILED for %s: %s\n",
                         path.parent_path().c_str(), ec.message().c_str());
        return;
    }

    // Write to temp file, then rename (atomic on same filesystem)
    auto tmpPath = path;
    tmpPath += ".tmp";

    int fd = ::open(tmpPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[DISK] put: open FAILED for %s errno=%d\n",
                         tmpPath.c_str(), errno);
        return;
    }

    size_t written = 0;
    while (written < size) {
        ssize_t n = ::write(fd, data + written, size - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            int err = errno;
            if (err == ENOSPC) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[DISK] put: DISK FULL writing %s\n", tmpPath.c_str());
            } else {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[DISK] put: write FAILED for %s errno=%d (%s)\n",
                                 tmpPath.c_str(), err, strerror(err));
            }
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
        if (auto* log = cacheDebugLog())
            std::fprintf(log, "[DISK] put: rename FAILED %s -> %s: %s\n",
                         tmpPath.c_str(), path.c_str(), ec.message().c_str());
        ::unlink(tmpPath.c_str());
    } else {
        totalBytes_.fetch_add(size, std::memory_order_relaxed);
    }
}

void DiskStore::remove(
    const std::string& volumeId,
    const ChunkKey& key)
{
    auto path = chunkPath(volumeId, key);
    std::error_code ec;

    // Get file size before removing so we can update the tracked total
    auto sz = std::filesystem::file_size(path, ec);
    if (ec) {
        // File doesn't exist or can't stat — nothing to subtract
        std::filesystem::remove(path, ec);
        return;
    }

    std::filesystem::remove(path, ec);
    if (!ec) {
        auto prev = totalBytes_.load(std::memory_order_relaxed);
        auto sub = static_cast<size_t>(sz);
        // Clamp to avoid underflow
        while (prev >= sub &&
               !totalBytes_.compare_exchange_weak(
                   prev, prev - sub, std::memory_order_relaxed))
            ;
    }
}

void DiskStore::evictToSize(size_t targetBytes)
{
    if (config_.root.empty()) return;

    // Fast path: if tracked total is already within budget, skip the scan
    size_t tracked = totalBytes_.load(std::memory_order_relaxed);
    if (tracked > 0 && tracked <= targetBytes) return;

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

    // Sync tracked total with the scanned value
    totalBytes_.store(totalSize, std::memory_order_relaxed);

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

    // Update tracked total after eviction
    totalBytes_.store(totalSize, std::memory_order_relaxed);
}

size_t DiskStore::totalBytes() const
{
    return totalBytes_.load(std::memory_order_relaxed);
}

void DiskStore::initTotalBytes()
{
    if (config_.root.empty() || !std::filesystem::exists(config_.root)) {
        totalBytes_.store(0, std::memory_order_relaxed);
        return;
    }

    size_t total = 0;
    std::error_code ec;
    for (auto& entry :
         std::filesystem::recursive_directory_iterator(config_.root, ec)) {
        if (entry.is_regular_file()) {
            total += static_cast<size_t>(entry.file_size());
        }
    }
    totalBytes_.store(total, std::memory_order_relaxed);
}

void DiskStore::clearVolume(const std::string& volumeId)
{
    auto volDir = config_.root / volumeId;
    std::error_code ec;
    std::filesystem::remove_all(volDir, ec);
    // Re-scan since we can't easily know how much was in this volume
    initTotalBytes();
}

void DiskStore::clearAll()
{
    if (config_.root.empty()) return;
    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(config_.root, ec)) {
        std::filesystem::remove_all(entry.path(), ec);
    }
    totalBytes_.store(0, std::memory_order_relaxed);
}

}  // namespace vc::cache
