#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <sstream>
#include <cstdio>
#include <chrono>

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
    Off
};

class SimpleLogger {
public:
    SimpleLogger();
    ~SimpleLogger();

    // Overloads for std::string (no formatting)
    void trace(const std::string& msg) {
        if (LogLevel::Trace >= min_level_) {
            log_impl(LogLevel::Trace, msg);
        }
    }

    void debug(const std::string& msg) {
        if (LogLevel::Debug >= min_level_) {
            log_impl(LogLevel::Debug, msg);
        }
    }

    void info(const std::string& msg) {
        if (LogLevel::Info >= min_level_) {
            log_impl(LogLevel::Info, msg);
        }
    }

    void warn(const std::string& msg) {
        if (LogLevel::Warn >= min_level_) {
            log_impl(LogLevel::Warn, msg);
        }
    }

    void error(const std::string& msg) {
        if (LogLevel::Error >= min_level_) {
            log_impl(LogLevel::Error, msg);
        }
    }

    void critical(const std::string& msg) {
        if (LogLevel::Critical >= min_level_) {
            log_impl(LogLevel::Critical, msg);
        }
    }

    // Template overloads for formatted logging
    template<typename... Args>
    void trace(const char* fmt, Args&&... args) {
        log(LogLevel::Trace, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void debug(const char* fmt, Args&&... args) {
        log(LogLevel::Debug, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(const char* fmt, Args&&... args) {
        log(LogLevel::Info, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(const char* fmt, Args&&... args) {
        log(LogLevel::Warn, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(const char* fmt, Args&&... args) {
        log(LogLevel::Error, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void critical(const char* fmt, Args&&... args) {
        log(LogLevel::Critical, fmt, std::forward<Args>(args)...);
    }

    void set_level(LogLevel level);
    void add_file_sink(const std::filesystem::path& path);

private:
    template<typename... Args>
    void log(LogLevel level, const char* fmt, Args&&... args) {
        if (level < min_level_) {
            return;
        }
        std::string message = format_message(fmt, std::forward<Args>(args)...);
        log_impl(level, message);
    }

    void log_impl(LogLevel level, const std::string& message);

    // Base case: no more arguments
    inline std::string format_message(const char* fmt) {
        std::string result;
        while (*fmt) {
            if (*fmt == '{' && *(fmt + 1) == '}') {
                // Found {} but no more arguments - just copy it literally
                result += "{}";
                fmt += 2;
            } else {
                result += *fmt++;
            }
        }
        return result;
    }

    // Recursive case: process one argument at a time
    template<typename T, typename... Args>
    std::string format_message(const char* fmt, T&& first, Args&&... args) {
        std::string result;
        while (*fmt) {
            if (*fmt == '{' && *(fmt + 1) == '}') {
                // Found {} - replace with the next argument
                std::ostringstream oss;
                oss << std::forward<T>(first);
                result += oss.str();
                fmt += 2;
                // Continue with remaining arguments
                return result + format_message(fmt, std::forward<Args>(args)...);
            } else {
                result += *fmt++;
            }
        }
        return result;
    }

    LogLevel min_level_;
    FILE* file_sink_;
};

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
auto Logger() -> std::shared_ptr<SimpleLogger>;

// Timing helper for performance profiling
class ScopedTimer {
public:
    ScopedTimer(const char* name)
        : name_(name)
        , start_(std::chrono::high_resolution_clock::now())
    {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        Logger()->info("[TIMING] {} took {} μs", name_, duration.count());
    }

private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
