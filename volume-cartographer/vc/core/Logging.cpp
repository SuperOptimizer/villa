#include "vc/core/util/Logging.hpp"

#include <chrono>
#include <iomanip>

SimpleLogger::SimpleLogger() = default;

SimpleLogger::~SimpleLogger()
{
    for (auto& file : files_) {
        if (file.is_open()) {
            file.close();
        }
    }
}

void SimpleLogger::addFile(const std::filesystem::path& path)
{
    std::lock_guard<std::mutex> lock(mutex_);
    files_.emplace_back(path);
}

std::string SimpleLogger::levelPrefix(Level level)
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";

    switch (level) {
        case Level::Trace:    oss << "[TRACE] "; break;
        case Level::Debug:    oss << "[DEBUG] "; break;
        case Level::Info:     oss << "[INFO] "; break;
        case Level::Warn:     oss << "[WARN] "; break;
        case Level::Error:    oss << "[ERROR] "; break;
        case Level::Critical: oss << "[CRITICAL] "; break;
    }

    return oss.str();
}

std::shared_ptr<SimpleLogger> Logger()
{
    static auto logger = std::make_shared<SimpleLogger>();
    return logger;
}

void AddLogFile(const std::filesystem::path& path)
{
    Logger()->addFile(path);
}

void SetLogLevel(const std::string& s)
{
    auto logger = Logger();
    if (s == "trace") {
        logger->setLevel(SimpleLogger::Level::Trace);
    } else if (s == "debug") {
        logger->setLevel(SimpleLogger::Level::Debug);
    } else if (s == "info") {
        logger->setLevel(SimpleLogger::Level::Info);
    } else if (s == "warn" || s == "warning") {
        logger->setLevel(SimpleLogger::Level::Warn);
    } else if (s == "error" || s == "err") {
        logger->setLevel(SimpleLogger::Level::Error);
    } else if (s == "critical" || s == "crit") {
        logger->setLevel(SimpleLogger::Level::Critical);
    }
}
