#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "ChunkKey.hpp"

namespace vc::cache {

// Persistent on-disk cache for compressed chunk data.
// Directory layout:
//   <root>/<volumeId>/<level>/<iz>.<iy>.<ix>
//
// Thread-safe for concurrent read/write to different keys.
// Same-key writes are serialized via a lock pool.
class DiskStore {
public:
    struct Config {
        std::filesystem::path root;           // cache root directory
        size_t maxBytes = 100ULL << 30;       // 100 GB default
        bool persistent = true;               // if false, cleaned up on destruction
    };

    explicit DiskStore(Config config);
    ~DiskStore();

    DiskStore(const DiskStore&) = delete;
    DiskStore& operator=(const DiskStore&) = delete;

    // Read compressed chunk bytes from disk. Returns nullopt on miss.
    std::optional<std::vector<uint8_t>> get(
        const std::string& volumeId,
        const ChunkKey& key) const;

    // Write compressed chunk bytes to disk. Creates directories as needed.
    void put(
        const std::string& volumeId,
        const ChunkKey& key,
        const uint8_t* data,
        size_t size);

    void put(
        const std::string& volumeId,
        const ChunkKey& key,
        const std::vector<uint8_t>& data)
    {
        put(volumeId, key, data.data(), data.size());
    }

    // Check if a chunk file exists on disk.
    bool has(const std::string& volumeId, const ChunkKey& key) const;

    // Remove a specific chunk from disk.
    void remove(const std::string& volumeId, const ChunkKey& key);

    // Evict oldest files until total size <= targetBytes.
    // Uses file modification time for ordering.
    void evictToSize(size_t targetBytes);

    // Total bytes currently stored on disk (approximate, scanned on demand).
    size_t totalBytes() const;

    // Remove all cached data for a volume.
    void clearVolume(const std::string& volumeId);

    // Remove all cached data.
    void clearAll();

    const std::filesystem::path& root() const { return config_.root; }

private:
    std::filesystem::path chunkPath(
        const std::string& volumeId,
        const ChunkKey& key) const;

    // Per-key lock pool to serialize same-key writes
    static constexpr int kLockPoolSize = 32;
    mutable std::mutex lockPool_[kLockPoolSize];
    size_t lockIndex(const std::string& volumeId, const ChunkKey& key) const;

    Config config_;
};

}  // namespace vc::cache
