#include "vc/core/cache/IOPool.hpp"
#include "vc/core/cache/CacheDebugLog.hpp"

#include <exception>

namespace vc::cache {

IOPool::IOPool(int numThreads)
{
    workers_.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
        workers_.emplace_back(&IOPool::workerLoop, this);
    }
}

IOPool::~IOPool() { stop(); }

void IOPool::setFetchFunc(FetchFunc fn)
{
    std::lock_guard lock(mutex_);
    fetchFunc_ = std::move(fn);
}

void IOPool::setCompletionCallback(CompletionCallback cb)
{
    std::lock_guard lock(mutex_);
    onComplete_ = std::move(cb);
}

void IOPool::setCurrentEpoch(uint64_t epoch)
{
    currentEpoch_.store(epoch, std::memory_order_relaxed);
}

void IOPool::submit(const ChunkKey& key)
{
    {
        std::lock_guard lock(mutex_);
        // Deduplicate: skip if already in-flight or already queued
        if (inFlightKeys_.count(key) || !queued_.insert(key).second) return;
        queue_.push(
            Task{key, nextSeq_.fetch_add(1, std::memory_order_relaxed),
                 currentEpoch_.load(std::memory_order_relaxed)});
    }
    cv_.notify_one();
}

void IOPool::submit(const std::vector<ChunkKey>& keys)
{
    {
        std::lock_guard lock(mutex_);
        uint64_t epoch = currentEpoch_.load(std::memory_order_relaxed);
        for (auto& key : keys) {
            if (!inFlightKeys_.count(key) && queued_.insert(key).second) {
                queue_.push(Task{
                    key, nextSeq_.fetch_add(1, std::memory_order_relaxed),
                    epoch});
            }
        }
    }
    cv_.notify_all();
}

void IOPool::cancelPending()
{
    std::lock_guard lock(mutex_);
    while (!queue_.empty()) queue_.pop();
    queued_.clear();
}

size_t IOPool::pendingCount() const
{
    std::lock_guard lock(mutex_);
    return queue_.size() + inFlight_.load(std::memory_order_relaxed);
}

void IOPool::stop()
{
    stopped_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void IOPool::workerLoop()
{
    while (true) {
        Task task;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] {
                return stopped_.load(std::memory_order_acquire) ||
                       !queue_.empty();
            });

            if (stopped_.load(std::memory_order_acquire) && queue_.empty())
                return;

            if (queue_.empty()) continue;

            task = queue_.top();
            queue_.pop();
            queued_.erase(task.key);
            inFlightKeys_.insert(task.key);
        }

        inFlight_.fetch_add(1, std::memory_order_relaxed);

        // fetchFunc_ and onComplete_ are set once before any tasks are
        // submitted, so we read them without holding the lock. The mutex
        // acquire in submit() provides the necessary happens-before.
        std::vector<uint8_t> data;

        if (fetchFunc_) {
            try {
                data = fetchFunc_(task.key);
            } catch (const std::exception& e) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[IOPOOL] fetch exception for lvl=%d pos=(%d,%d,%d): %s\n",
                                 task.key.level, task.key.iz, task.key.iy, task.key.ix, e.what());
            } catch (...) {
                if (auto* log = cacheDebugLog())
                    std::fprintf(log, "[IOPOOL] unknown fetch exception for lvl=%d pos=(%d,%d,%d)\n",
                                 task.key.level, task.key.iz, task.key.iy, task.key.ix);
            }
        }

        {
            std::lock_guard lock(mutex_);
            inFlightKeys_.erase(task.key);
        }
        inFlight_.fetch_sub(1, std::memory_order_relaxed);

        if (onComplete_) {
            onComplete_(task.key, std::move(data));
        }
    }
}

}  // namespace vc::cache
