#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <mutex>
#include <sstream>
#include <ctime>
#include <iomanip>

// Minimal logger implementation to replace spdlog
class MinimalLogger {
public:
    enum class Level { Debug, Info, Warn, Error };

    MinimalLogger(const std::string& name) : name_(name) {}

    template<typename... Args>
    void info(const std::string& fmt, Args&&... args) {
        log(Level::Info, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(const std::string& fmt, Args&&... args) {
        log(Level::Warn, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(const std::string& fmt, Args&&... args) {
        log(Level::Error, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void debug(const std::string& fmt, Args&&... args) {
        log(Level::Debug, fmt, std::forward<Args>(args)...);
    }

    void set_level(Level level) { level_ = level; }
    void add_file(const std::filesystem::path& path);

private:
    template<typename... Args>
    void log(Level level, const std::string& fmt, Args&&... args) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);

        std::ostringstream oss;
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";
        oss << "[" << level_to_string(level) << "] ";
        oss << "[" << name_ << "] ";
        oss << format_message(fmt, std::forward<Args>(args)...);
        oss << "\n";

        std::string message = oss.str();
        std::cout << message;

        if (file_.is_open()) {
            file_ << message;
            file_.flush();
        }
    }

    std::string level_to_string(Level level) {
        switch(level) {
            case Level::Debug: return "DEBUG";
            case Level::Info: return "INFO";
            case Level::Warn: return "WARN";
            case Level::Error: return "ERROR";
            default: return "UNKNOWN";
        }
    }

    template<typename T>
    std::string to_string_helper(const T& value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }

    std::string to_string_helper(const std::string& value) {
        return value;
    }

    std::string to_string_helper(const char* value) {
        return std::string(value);
    }

    std::string format_message(const std::string& fmt) {
        return fmt;
    }

    template<typename T, typename... Args>
    std::string format_message(const std::string& fmt, T&& first, Args&&... rest) {
        size_t pos = fmt.find("{}");
        if (pos == std::string::npos) {
            return fmt;
        }
        std::string result = fmt.substr(0, pos) + to_string_helper(std::forward<T>(first));
        result += format_message(fmt.substr(pos + 2), std::forward<Args>(rest)...);
        return result;
    }

    std::string name_;
    Level level_ = Level::Info;
    std::mutex mutex_;
    std::ofstream file_;
};

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
auto Logger() -> std::shared_ptr<MinimalLogger>;
