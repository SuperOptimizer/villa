#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ctime>
#include <iomanip>

// Simple logger class to replace spdlog
class SimpleLogger {
public:
    enum class Level {
        DEBUG,
        INFO,
        WARN,
        ERROR
    };

    SimpleLogger() : level_(Level::INFO) {}

    void setLevel(Level level) { level_ = level; }

    void addFileSink(const std::filesystem::path& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        file_ = std::make_unique<std::ofstream>(path, std::ios::app);
        if (!file_->is_open()) {
            std::cerr << "Failed to open log file: " << path << std::endl;
            file_.reset();
        }
    }

    template<typename... Args>
    void debug(const std::string& fmt, Args&&... args) {
        log(Level::DEBUG, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(const std::string& fmt, Args&&... args) {
        log(Level::INFO, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warn(const std::string& fmt, Args&&... args) {
        log(Level::WARN, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(const std::string& fmt, Args&&... args) {
        log(Level::ERROR, fmt, std::forward<Args>(args)...);
    }

private:
    template<typename... Args>
    void log(Level level, const std::string& fmt, Args&&... args) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        std::string message = format(fmt, std::forward<Args>(args)...);
        std::string prefix = getPrefix(level);
        std::string output = prefix + message;

        std::cout << output << std::endl;

        if (file_ && file_->is_open()) {
            (*file_) << output << std::endl;
            file_->flush();
        }
    }

    template<typename T>
    std::string toString(T&& value) {
        std::ostringstream oss;
        oss << std::forward<T>(value);
        return oss.str();
    }

    // Base case: no more arguments
    std::string format(const std::string& fmt) {
        return fmt;
    }

    // Recursive case: replace first {} with first argument
    template<typename T, typename... Args>
    std::string format(const std::string& fmt, T&& first, Args&&... rest) {
        size_t pos = fmt.find("{}");
        if (pos == std::string::npos) {
            return fmt; // No more placeholders
        }

        std::string result = fmt.substr(0, pos) + toString(std::forward<T>(first)) + fmt.substr(pos + 2);
        return format(result, std::forward<Args>(rest)...);
    }

    std::string getPrefix(Level level) {
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);

        std::ostringstream oss;
        oss << "[" << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "] ";

        switch (level) {
            case Level::DEBUG: oss << "[DEBUG] "; break;
            case Level::INFO:  oss << "[INFO]  "; break;
            case Level::WARN:  oss << "[WARN]  "; break;
            case Level::ERROR: oss << "[ERROR] "; break;
        }

        return oss.str();
    }

    Level level_;
    std::unique_ptr<std::ofstream> file_;
    std::mutex mutex_;
};

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
std::shared_ptr<SimpleLogger> Logger();
