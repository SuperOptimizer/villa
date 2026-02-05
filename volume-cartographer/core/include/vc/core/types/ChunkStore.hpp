#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

// Forward declarations
template <typename T> class ChunkCache;

/**
 * @brief Unified chunk cache managing both uint8 and uint16 cache pools.
 *
 * Multiple Volume objects can share a single ChunkStore, giving a global
 * cache budget across all open volumes.  This replaces the pattern of
 * each consumer creating its own ChunkCache<T>.
 */
class ChunkStore final {
public:
    explicit ChunkStore(size_t maxBytes = 4ULL * 1024 * 1024 * 1024);
    ~ChunkStore();

    ChunkStore(const ChunkStore&) = delete;
    ChunkStore& operator=(const ChunkStore&) = delete;

    void setMaxBytes(size_t maxBytes);

    struct Stats {
        uint64_t hits{0};
        uint64_t misses{0};
        uint64_t evictions{0};
        uint64_t bytesRead{0};
    };
    [[nodiscard]] Stats stats() const;
    void resetStats();
    void clear();

    // Access the underlying typed caches (used internally by Volume)
    ChunkCache<uint8_t>& cache8();
    ChunkCache<uint16_t>& cache16();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
