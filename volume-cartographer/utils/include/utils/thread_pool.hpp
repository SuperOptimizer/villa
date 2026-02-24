#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace utils {

/// A fixed-size thread pool that accepts arbitrary callables and returns
/// futures for their results.
///
/// Worker threads block on a condition variable until tasks are enqueued.
/// The destructor sets a stop flag and joins every thread, draining any
/// remaining tasks.
class ThreadPool {
public:
    /// Construct a pool with @p num_threads worker threads.
    /// A value of 0 (the default) selects std::thread::hardware_concurrency(),
    /// falling back to 1 if that returns 0.
    explicit ThreadPool(std::size_t num_threads = 0)
        : num_threads_{num_threads == 0
                           ? (std::thread::hardware_concurrency() > 0
                                  ? std::thread::hardware_concurrency()
                                  : 1u)
                           : num_threads} {
        workers_.reserve(num_threads_);
        for (std::size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    /// Destructor: signals all workers to stop and joins them.
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) {
                w.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    auto operator=(const ThreadPool&) -> ThreadPool& = delete;
    ThreadPool(ThreadPool&&) = delete;
    auto operator=(ThreadPool&&) -> ThreadPool& = delete;

    /// Submit a callable with arguments to the pool.
    /// Returns a future that will hold the result (or re-throw any exception
    /// thrown by @p f).
    template <typename F, typename... Args>
    [[nodiscard]] auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        auto future = task->get_future();

        {
            std::lock_guard<std::mutex> lock{mutex_};
            if (stop_) {
                throw std::runtime_error("submit() called on stopped ThreadPool");
            }
            tasks_.emplace([task = std::move(task)]() { (*task)(); });
            pending_.fetch_add(1, std::memory_order_relaxed);
        }
        cv_.notify_one();

        return future;
    }

    /// Number of worker threads in the pool.
    [[nodiscard]] auto num_threads() const noexcept -> std::size_t {
        return num_threads_;
    }

    /// Approximate number of tasks waiting in the queue (not yet picked up
    /// by a worker).
    [[nodiscard]] auto pending_tasks() const noexcept -> std::size_t {
        return pending_.load(std::memory_order_relaxed);
    }

private:
    auto worker_loop() -> void {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock{mutex_};
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
                pending_.fetch_sub(1, std::memory_order_relaxed);
            }
            task();
        }
    }

    std::size_t num_threads_;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    bool stop_{false};

    std::atomic<std::size_t> pending_{0};
};

// ---------------------------------------------------------------------------
// parallel_for
// ---------------------------------------------------------------------------

/// Execute @p fn(i) for every i in [begin, end), distributing work across
/// @p pool in chunks of @p chunk_size.  Blocks until all iterations complete.
template <typename Fn>
auto parallel_for(ThreadPool& pool, std::size_t begin, std::size_t end,
                  std::size_t chunk_size, Fn&& fn) -> void {
    if (begin >= end) {
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve((end - begin + chunk_size - 1) / chunk_size);

    for (std::size_t chunk_begin = begin; chunk_begin < end;
         chunk_begin += chunk_size) {
        std::size_t chunk_end = std::min(chunk_begin + chunk_size, end);
        futures.push_back(pool.submit(
            [&fn, chunk_begin, chunk_end]() {
                for (std::size_t i = chunk_begin; i < chunk_end; ++i) {
                    fn(i);
                }
            }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

/// Execute @p fn(i) for every i in [begin, end), distributing work across
/// @p pool.  The range is split into one chunk per worker thread.  Blocks
/// until all iterations complete.
template <typename Fn>
auto parallel_for(ThreadPool& pool, std::size_t begin, std::size_t end,
                  Fn&& fn) -> void {
    if (begin >= end) {
        return;
    }
    std::size_t n = end - begin;
    std::size_t chunk = (n + pool.num_threads() - 1) / pool.num_threads();
    parallel_for(pool, begin, end, chunk, std::forward<Fn>(fn));
}

/// Execute @p fn(i) for every i in [begin, end) using a process-wide static
/// thread pool (Meyer's singleton).  Blocks until all iterations complete.
template <typename Fn>
auto parallel_for(std::size_t begin, std::size_t end, Fn&& fn) -> void {
    static ThreadPool default_pool;
    parallel_for(default_pool, begin, end, std::forward<Fn>(fn));
}

}  // namespace utils
