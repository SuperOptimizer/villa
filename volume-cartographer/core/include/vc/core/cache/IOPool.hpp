#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ChunkKey.hpp"

namespace vc::cache {

// Background I/O thread pool for async chunk fetching.
// Tasks are prioritized: lower pyramid level = higher priority (coarse data first).
// Duplicate submissions for the same key are deduplicated.
class IOPool {
public:
    // Callback signature: called from a worker thread when a chunk is fetched.
    // (key, compressed bytes) — empty bytes means fetch failed.
    using CompletionCallback =
        std::function<void(const ChunkKey&, std::vector<uint8_t>&&)>;

    // FetchFunc: the actual data fetching function (wraps ChunkSource::fetch).
    // Called from worker threads. Must be thread-safe.
    using FetchFunc = std::function<std::vector<uint8_t>(const ChunkKey&)>;

    explicit IOPool(int numThreads = 4);
    ~IOPool();

    IOPool(const IOPool&) = delete;
    IOPool& operator=(const IOPool&) = delete;

    // Set the fetch function (typically wraps ChunkSource + DiskStore lookup).
    void setFetchFunc(FetchFunc fn);

    // Set the completion callback (called from worker thread).
    void setCompletionCallback(CompletionCallback cb);

    // Set the current epoch. Tasks from the current epoch get higher priority.
    void setCurrentEpoch(uint64_t epoch);

    // Submit a chunk for background fetching.
    // Deduplicates: if the key is already queued or in-flight, this is a no-op.
    void submit(const ChunkKey& key);

    // Submit multiple keys at once (batch, reduces lock contention).
    void submit(const std::vector<ChunkKey>& keys);

    // Cancel all pending (not in-flight) tasks.
    void cancelPending();

    // Number of pending + in-flight tasks.
    [[nodiscard]] size_t pendingCount() const;

    // Gracefully stop all workers. Blocks until all threads exit.
    void stop();

private:
    void workerLoop();

    struct Task {
        ChunkKey key;
        // Higher level = higher priority (coarse first for progressive rendering).
        // Level 0 = finest, level N = coarsest. We want coarse data loaded first
        // so tiles can show coarse previews while fine data loads.
        // Within same level, FIFO order via sequence number.
        uint64_t seq = 0;
        uint64_t epoch = 0;  // epoch when this task was submitted

        bool operator>(const Task& o) const noexcept
        {
            if (key.level != o.key.level) return key.level < o.key.level;
            // Within same level, prefer newer epoch (higher = newer)
            if (epoch != o.epoch) return epoch < o.epoch;
            return seq > o.seq;
        }
    };

    FetchFunc fetchFunc_;
    CompletionCallback onComplete_;

    // Priority queue: min-heap by (level, seq)
    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> queue_;
    std::unordered_set<ChunkKey, ChunkKeyHash> queued_;  // dedup set for queued tasks
    std::unordered_set<ChunkKey, ChunkKeyHash> inFlightKeys_;  // keys currently being fetched
    std::atomic<uint64_t> nextSeq_{0};
    std::atomic<uint64_t> currentEpoch_{0};
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::vector<std::thread> workers_;
    std::atomic<bool> stopped_{false};
    std::atomic<size_t> inFlight_{0};
};

}  // namespace vc::cache
