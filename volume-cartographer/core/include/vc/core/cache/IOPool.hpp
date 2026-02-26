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
// Tasks are prioritized: coarser pyramid levels are fetched first
// (higher level number = coarser = higher priority) for progressive rendering.
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
        uint64_t seq = 0;
        uint64_t epoch = 0;  // epoch when this task was submitted

        // Ordering for the priority queue (min-heap via std::greater<Task>).
        // A task is "greater" if it should be popped LATER (lower priority).
        // We want coarse levels (higher number) popped first, so a task with
        // lower level number is "greater" (deferred). Within the same level,
        // newer epochs (higher number) are preferred. Within the same epoch,
        // earlier submissions (lower seq) are preferred (FIFO).
        bool operator>(const Task& o) const noexcept
        {
            if (key.level != o.key.level) return key.level < o.key.level;
            if (epoch != o.epoch) return epoch < o.epoch;
            return seq > o.seq;
        }
    };

    FetchFunc fetchFunc_;
    CompletionCallback onComplete_;

    // Priority queue: min-heap — pops the "smallest" task first (coarsest level)
    std::priority_queue<Task, std::vector<Task>, std::greater<Task>> queue_;
    std::unordered_set<ChunkKey, ChunkKeyHash> queued_;      // dedup set
    std::unordered_set<ChunkKey, ChunkKeyHash> inFlightKeys_;
    std::atomic<uint64_t> nextSeq_{0};
    std::atomic<uint64_t> currentEpoch_{0};
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::vector<std::thread> workers_;
    std::atomic<bool> stopped_{false};
    std::atomic<size_t> inFlight_{0};
};

}  // namespace vc::cache
