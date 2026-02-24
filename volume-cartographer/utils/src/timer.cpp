#include "utils/timer.hpp"
#include "utils/logging.hpp"

#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <string>

namespace utils {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Timer::Impl {
    std::string name;
    TimePoint start_time;
    TimePoint lap_time;
    bool is_running{false};
    TimerStats stats{0, 0.0, std::numeric_limits<double>::max(), 0.0};
    std::mutex mtx;

    explicit Impl(std::string n) : name(std::move(n)) {}
};

Timer::Timer(std::string name) : impl_(std::make_unique<Impl>(std::move(name))) {}
Timer::~Timer() = default;
Timer::Timer(Timer&& other) noexcept = default;
auto Timer::operator=(Timer&& other) noexcept -> Timer& = default;

auto Timer::start() -> void {
    std::lock_guard lock(impl_->mtx);
    impl_->start_time = Clock::now();
    impl_->lap_time = impl_->start_time;
    impl_->is_running = true;
}

auto Timer::stop() -> double {
    auto now = Clock::now();
    std::lock_guard lock(impl_->mtx);
    if (!impl_->is_running) {
        return 0.0;
    }
    auto ms = std::chrono::duration<double, std::milli>(now - impl_->start_time).count();
    impl_->is_running = false;
    impl_->stats.count++;
    impl_->stats.total_ms += ms;
    if (ms < impl_->stats.min_ms) impl_->stats.min_ms = ms;
    if (ms > impl_->stats.max_ms) impl_->stats.max_ms = ms;
    return ms;
}

auto Timer::lap() -> double {
    auto now = Clock::now();
    std::lock_guard lock(impl_->mtx);
    if (!impl_->is_running) {
        return 0.0;
    }
    auto ms = std::chrono::duration<double, std::milli>(now - impl_->lap_time).count();
    impl_->lap_time = now;
    return ms;
}

auto Timer::reset() -> void {
    std::lock_guard lock(impl_->mtx);
    impl_->is_running = false;
    impl_->stats = {0, 0.0, std::numeric_limits<double>::max(), 0.0};
}

auto Timer::name() const noexcept -> std::string_view {
    return impl_->name;
}

auto Timer::running() const noexcept -> bool {
    return impl_->is_running;
}

auto Timer::elapsed_ms() const noexcept -> double {
    if (!impl_->is_running) {
        return 0.0;
    }
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - impl_->start_time).count();
}

auto Timer::stats() const noexcept -> TimerStats {
    return impl_->stats;
}

namespace {
struct GlobalRegistry {
    std::map<std::string, std::unique_ptr<Timer>, std::less<>> timers;
    std::mutex mtx;
};

auto registry() -> GlobalRegistry& {
    static GlobalRegistry reg;
    return reg;
}
} // namespace

auto Timer::global(std::string_view name) -> Timer& {
    auto& reg = registry();
    std::lock_guard lock(reg.mtx);
    auto it = reg.timers.find(name);
    if (it != reg.timers.end()) {
        return *it->second;
    }
    auto [inserted, _] = reg.timers.emplace(std::string(name),
                                             std::make_unique<Timer>(std::string(name)));
    return *inserted->second;
}

auto Timer::print_all() -> void {
    auto& reg = registry();
    std::lock_guard lock(reg.mtx);
    for (auto& [n, t] : reg.timers) {
        auto s = t->stats();
        if (s.count > 0) {
            log_info("[Timer] {}: count={}, total={:.3f}ms, mean={:.3f}ms, min={:.3f}ms, max={:.3f}ms",
                     n, s.count, s.total_ms, s.mean_ms(), s.min_ms, s.max_ms);
        }
    }
}

auto Timer::reset_all() -> void {
    auto& reg = registry();
    std::lock_guard lock(reg.mtx);
    for (auto& [n, t] : reg.timers) {
        t->reset();
    }
}

auto Timer::all_stats() -> std::map<std::string, TimerStats, std::less<>> {
    auto& reg = registry();
    std::lock_guard lock(reg.mtx);
    std::map<std::string, TimerStats, std::less<>> result;
    for (auto& [n, t] : reg.timers) {
        result.emplace(n, t->stats());
    }
    return result;
}

struct ScopedTimer::Impl {
    TimePoint start_time;
    std::string name;
    std::function<void(std::string_view, double)> callback;
    Timer* timer{nullptr};

    Impl() : start_time(Clock::now()) {}
};

ScopedTimer::ScopedTimer(std::string name,
                         std::function<void(std::string_view, double)> callback)
    : impl_(std::make_unique<Impl>()) {
    impl_->name = std::move(name);
    impl_->callback = std::move(callback);
}

ScopedTimer::ScopedTimer(Timer& timer)
    : impl_(std::make_unique<Impl>()) {
    impl_->timer = &timer;
    timer.start();
}

ScopedTimer::~ScopedTimer() {
    if (impl_->timer) {
        impl_->timer->stop();
    } else {
        auto ms = std::chrono::duration<double, std::milli>(
            Clock::now() - impl_->start_time).count();
        if (impl_->callback) {
            impl_->callback(impl_->name, ms);
        } else {
            log_info("[ScopedTimer] {}: {:.3f}ms", impl_->name, ms);
        }
    }
}

auto ScopedTimer::elapsed_ms() const noexcept -> double {
    if (impl_->timer) {
        return impl_->timer->elapsed_ms();
    }
    return std::chrono::duration<double, std::milli>(
        Clock::now() - impl_->start_time).count();
}

} // namespace utils
