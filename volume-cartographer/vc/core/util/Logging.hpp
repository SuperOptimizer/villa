#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <mutex>
#include <vector>

// Minimal logger class to replace spdlog
class SimpleLogger {
public:
    enum class Level {
        Trace,
        Debug,
        Info,
        Warn,
        Error,
        Critical
    };

    SimpleLogger();
    ~SimpleLogger();

    void setLevel(Level level) { currentLevel_ = level; }
    void addFile(const std::filesystem::path& path);

    template<typename... Args>
    void trace(const std::string& fmt, Args&&... args) {
        log(Level::Trace, fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void debug(const std::string& fmt, Args&&... args) {
        log(Level::Debug, fmt, std::forward<Args>(args)...);
    }

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
    void critical(const std::string& fmt, Args&&... args) {
        log(Level::Critical, fmt, std::forward<Args>(args)...);
    }

private:
    template<typename... Args>
    void log(Level level, const std::string& fmt, Args&&... args) {
        if (level < currentLevel_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        std::string msg = formatMessage(fmt, std::forward<Args>(args)...);
        std::string prefix = levelPrefix(level);

        std::cerr << prefix << msg << std::endl;

        for (auto& file : files_) {
            if (file.is_open()) {
                file << prefix << msg << std::endl;
                file.flush();
            }
        }
    }

    template<typename T>
    std::string toString(T&& val) {
        std::ostringstream oss;
        oss << std::forward<T>(val);
        return oss.str();
    }

    template<typename T, typename... Args>
    std::string formatMessage(const std::string& fmt, T&& first, Args&&... rest) {
        std::string result = fmt;
        size_t pos = result.find("{}");
        if (pos != std::string::npos) {
            result.replace(pos, 2, toString(std::forward<T>(first)));
        }
        if constexpr (sizeof...(rest) > 0) {
            return formatMessage(result, std::forward<Args>(rest)...);
        }
        return result;
    }

    static std::string formatMessage(const std::string& fmt) {
        return fmt;
    }

    static std::string levelPrefix(Level level);

    Level currentLevel_ = Level::Info;
    std::mutex mutex_;
    std::vector<std::ofstream> files_;
};

void AddLogFile(const std::filesystem::path& path);
void SetLogLevel(const std::string& s);
std::shared_ptr<SimpleLogger> Logger();
