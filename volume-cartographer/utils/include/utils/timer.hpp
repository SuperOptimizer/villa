#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace utils {

struct TimerStats {
    std::size_t count{0};
    double total_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};

    [[nodiscard]] auto mean_ms() const noexcept -> double {
        return count > 0 ? total_ms / static_cast<double>(count) : 0.0;
    }
};

class Timer {
public:
    explicit Timer(std::string name = "");
    ~Timer();

    Timer(Timer&& other) noexcept;
    auto operator=(Timer&& other) noexcept -> Timer&;

    Timer(const Timer&) = delete;
    auto operator=(const Timer&) -> Timer& = delete;

    auto start() -> void;
    auto stop() -> double;
    auto lap() -> double;
    auto reset() -> void;

    [[nodiscard]] auto name() const noexcept -> std::string_view;
    [[nodiscard]] auto running() const noexcept -> bool;
    [[nodiscard]] auto elapsed_ms() const noexcept -> double;
    [[nodiscard]] auto stats() const noexcept -> TimerStats;

    static auto global(std::string_view name) -> Timer&;
    static auto print_all() -> void;
    static auto reset_all() -> void;
    static auto all_stats() -> std::map<std::string, TimerStats, std::less<>>;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string name,
                         std::function<void(std::string_view, double)> callback = {});
    explicit ScopedTimer(Timer& timer);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    auto operator=(const ScopedTimer&) -> ScopedTimer& = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    auto operator=(ScopedTimer&&) -> ScopedTimer& = delete;

    [[nodiscard]] auto elapsed_ms() const noexcept -> double;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace utils
