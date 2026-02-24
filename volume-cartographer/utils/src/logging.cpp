#include "utils/logging.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <vector>

#include <unistd.h>

namespace utils {

auto log_level_name(LogLevel level) noexcept -> const char* {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERROR";
        case LogLevel::Off:   return "OFF";
    }
    return "UNKNOWN";
}

auto console_sink(bool color) -> LogSink {
    bool use_color = color && isatty(STDERR_FILENO);
    return [use_color](LogLevel level, std::string_view msg) {
        if (use_color) {
            const char* color_code = "";
            switch (level) {
                case LogLevel::Trace: color_code = "\033[90m"; break;   // gray
                case LogLevel::Debug: color_code = "\033[36m"; break;   // cyan
                case LogLevel::Info:  color_code = "\033[32m"; break;   // green
                case LogLevel::Warn:  color_code = "\033[33m"; break;   // yellow
                case LogLevel::Error: color_code = "\033[31m"; break;   // red
                case LogLevel::Off:   break;
            }
            std::fprintf(stderr, "%s%.*s\033[0m", color_code,
                         static_cast<int>(msg.size()), msg.data());
        } else {
            std::fprintf(stderr, "%.*s", static_cast<int>(msg.size()), msg.data());
        }
    };
}

auto file_sink(const std::string& path) -> LogSink {
    auto ofs = std::make_shared<std::ofstream>(path, std::ios::app);
    return [ofs](LogLevel /*level*/, std::string_view msg) {
        if (ofs->is_open()) {
            ofs->write(msg.data(), static_cast<std::streamsize>(msg.size()));
            ofs->flush();
        }
    };
}

struct Logger::Impl {
    std::string name;
    std::vector<LogSink> sinks;
    std::mutex mtx;

    explicit Impl(std::string n) : name(std::move(n)) {}
};

Logger::Logger(std::string name) : impl_(std::make_unique<Impl>(std::move(name))) {}
Logger::~Logger() = default;
Logger::Logger(Logger&& other) noexcept
    : impl_(std::move(other.impl_))
    , level_(other.level_.load(std::memory_order_relaxed)) {}

auto Logger::operator=(Logger&& other) noexcept -> Logger& {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        level_.store(other.level_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
}

auto Logger::set_level(LogLevel level) noexcept -> void {
    level_.store(level, std::memory_order_relaxed);
}

auto Logger::level() const noexcept -> LogLevel {
    return level_.load(std::memory_order_relaxed);
}

auto Logger::name() const noexcept -> std::string_view {
    return impl_->name;
}

auto Logger::add_sink(LogSink sink) -> void {
    std::lock_guard lock(impl_->mtx);
    impl_->sinks.push_back(std::move(sink));
}

auto Logger::clear_sinks() -> void {
    std::lock_guard lock(impl_->mtx);
    impl_->sinks.clear();
}

auto Logger::log(LogLevel level, std::string_view message) -> void {
    if (level < level_.load(std::memory_order_relaxed)) {
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm_buf{};
    localtime_r(&time_t, &tm_buf);

    char time_str[32];
    std::snprintf(time_str, sizeof(time_str), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
                  tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                  static_cast<int>(ms.count()));

    std::string formatted;
    if (impl_->name.empty()) {
        formatted = std::format("[{}] [{}] {}\n", time_str, log_level_name(level), message);
    } else {
        formatted = std::format("[{}] [{}] [{}] {}\n", time_str, log_level_name(level),
                                impl_->name, message);
    }

    std::lock_guard lock(impl_->mtx);
    for (auto& sink : impl_->sinks) {
        sink(level, formatted);
    }
}

auto default_logger() -> Logger& {
    static Logger logger("");
    static bool initialized = [&]() {
        logger.add_sink(console_sink());
        return true;
    }();
    (void)initialized;
    return logger;
}

auto set_default_level(LogLevel level) -> void {
    default_logger().set_level(level);
}

auto add_default_sink(LogSink sink) -> void {
    default_logger().add_sink(std::move(sink));
}

} // namespace utils
