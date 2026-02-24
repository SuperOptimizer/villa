#pragma once

#include <atomic>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace utils {

enum class LogLevel : std::uint8_t { Trace, Debug, Info, Warn, Error, Off };

[[nodiscard]] auto log_level_name(LogLevel level) noexcept -> const char*;

using LogSink = std::function<void(LogLevel, std::string_view)>;
[[nodiscard]] auto console_sink(bool color = true) -> LogSink;
[[nodiscard]] auto file_sink(const std::string& path) -> LogSink;

class Logger {
public:
    explicit Logger(std::string name = "");
    ~Logger();

    Logger(Logger&& other) noexcept;
    auto operator=(Logger&& other) noexcept -> Logger&;

    Logger(const Logger&) = delete;
    auto operator=(const Logger&) -> Logger& = delete;

    auto set_level(LogLevel level) noexcept -> void;
    [[nodiscard]] auto level() const noexcept -> LogLevel;
    [[nodiscard]] auto name() const noexcept -> std::string_view;

    auto add_sink(LogSink sink) -> void;
    auto clear_sinks() -> void;

    auto log(LogLevel level, std::string_view message) -> void;

    template <typename... Args>
    auto log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) -> void {
        if (level < level_.load(std::memory_order_relaxed)) {
            return;
        }
        log(level, std::format(fmt, std::forward<Args>(args)...));
    }

    template <typename... Args>
    auto trace(std::format_string<Args...> fmt, Args&&... args) -> void {
        log(LogLevel::Trace, fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto debug(std::format_string<Args...> fmt, Args&&... args) -> void {
        log(LogLevel::Debug, fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto info(std::format_string<Args...> fmt, Args&&... args) -> void {
        log(LogLevel::Info, fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto warn(std::format_string<Args...> fmt, Args&&... args) -> void {
        log(LogLevel::Warn, fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto error(std::format_string<Args...> fmt, Args&&... args) -> void {
        log(LogLevel::Error, fmt, std::forward<Args>(args)...);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<LogLevel> level_{LogLevel::Info};
};

auto default_logger() -> Logger&;
auto set_default_level(LogLevel level) -> void;
auto add_default_sink(LogSink sink) -> void;

template <typename... Args>
auto log_trace(std::format_string<Args...> fmt, Args&&... args) -> void {
    default_logger().trace(fmt, std::forward<Args>(args)...);
}
template <typename... Args>
auto log_debug(std::format_string<Args...> fmt, Args&&... args) -> void {
    default_logger().debug(fmt, std::forward<Args>(args)...);
}
template <typename... Args>
auto log_info(std::format_string<Args...> fmt, Args&&... args) -> void {
    default_logger().info(fmt, std::forward<Args>(args)...);
}
template <typename... Args>
auto log_warn(std::format_string<Args...> fmt, Args&&... args) -> void {
    default_logger().warn(fmt, std::forward<Args>(args)...);
}
template <typename... Args>
auto log_error(std::format_string<Args...> fmt, Args&&... args) -> void {
    default_logger().error(fmt, std::forward<Args>(args)...);
}

} // namespace utils
