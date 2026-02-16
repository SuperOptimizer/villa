#include "vc/core/cache/IOPool.hpp"

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

void IOPool::submit(const ChunkKey& key)
{
    {
        std::lock_guard lock(mutex_);
        // Deduplicate: skip if already queued
        if (!queued_.insert(key).second) return;
        queue_.push(
            Task{key, nextSeq_.fetch_add(1, std::memory_order_relaxed)});
    }
    cv_.notify_one();
}

void IOPool::submit(const std::vector<ChunkKey>& keys)
{
    {
        std::lock_guard lock(mutex_);
        for (auto& key : keys) {
            if (queued_.insert(key).second) {
                queue_.push(Task{
                    key, nextSeq_.fetch_add(1, std::memory_order_relaxed)});
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
        }

        inFlight_.fetch_add(1, std::memory_order_relaxed);

        // Fetch the chunk data
        std::vector<uint8_t> data;
        FetchFunc fetchFn;
        {
            std::lock_guard lock(mutex_);
            fetchFn = fetchFunc_;
        }

        if (fetchFn) {
            try {
                data = fetchFn(task.key);
            } catch (...) {
                // Fetch failed — empty data signals failure
            }
        }

        inFlight_.fetch_sub(1, std::memory_order_relaxed);

        // Notify completion
        CompletionCallback cb;
        {
            std::lock_guard lock(mutex_);
            cb = onComplete_;
        }
        if (cb) {
            cb(task.key, std::move(data));
        }
    }
}

}  // namespace vc::cache
