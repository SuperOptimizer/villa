#pragma once

#include <format>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

enum class LogLevel {
    trace = 0,
    debug = 1,
    info = 2,
    warn = 3,
    error = 4,
    critical = 5,
    off = 6
};

class SimpleLogger {
public:
    SimpleLogger(const std::string& name) : name_(name) {}

    // Overloads for pre-formatted strings
    void trace(const std::string& msg) { log_string(LogLevel::trace, msg); }
    void debug(const std::string& msg) { log_string(LogLevel::debug, msg); }
    void info(const std::string& msg) { log_string(LogLevel::info, msg); }
    void warn(const std::string& msg) { log_string(LogLevel::warn, msg); }
    void error(const std::string& msg) { log_string(LogLevel::error, msg); }
    void critical(const std::string& msg) { log_string(LogLevel::critical, msg); }

    // Overloads for format strings with arguments
    template<typename... Args>
    void trace(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::trace, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::debug, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::info, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::warn, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::error, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void critical(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::critical, fmt, std::forward<Args>(args)...);
    }

    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    void add_file(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        file_stream_ = std::make_shared<std::ofstream>(path, std::ios::app);
    }

private:
    void log_string(LogLevel level, const std::string& message) {
        if (level < level_) return;

        std::string level_str = level_to_string(level);

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        timestamp << '.' << std::setfill('0') << std::setw(3) << ms.count();

        std::string full_message = std::format("[{}] [{}] [{}] {}\n",
            timestamp.str(), name_, level_str, message);

        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << full_message << std::flush;

        if (file_stream_ && file_stream_->is_open()) {
            *file_stream_ << full_message << std::flush;
        }
    }

    template<typename... Args>
    void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
        if (level < level_) return;
        std::string message = std::format(fmt, std::forward<Args>(args)...);
        log_string(level, message);
    }

    static std::string level_to_string(LogLevel level) {
        switch (level) {
            case LogLevel::trace: return "trace";
            case LogLevel::debug: return "debug";
            case LogLevel::info: return "info";
            case LogLevel::warn: return "warning";
            case LogLevel::error: return "error";
            case LogLevel::critical: return "critical";
            default: return "unknown";
        }
    }

    std::string name_;
    LogLevel level_ = LogLevel::info;
    std::shared_ptr<std::ofstream> file_stream_;
    std::mutex mutex_;
};

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
auto Logger() -> std::shared_ptr<SimpleLogger>;
