#include "vc/core/util/Logging.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <algorithm>

SimpleLogger::SimpleLogger()
    : min_level_(LogLevel::Info)
    , file_sink_(nullptr)
{
}

SimpleLogger::~SimpleLogger()
{
    if (file_sink_) {
        fclose(file_sink_);
        file_sink_ = nullptr;
    }
}

void SimpleLogger::set_level(LogLevel level)
{
    min_level_ = level;
}

void SimpleLogger::add_file_sink(const std::filesystem::path& path)
{
    if (file_sink_) {
        fclose(file_sink_);
    }

    file_sink_ = fopen(path.string().c_str(), "a");
    if (!file_sink_) {
        std::cerr << "Failed to open log file: " << path << std::endl;
    }
}

void SimpleLogger::log_impl(LogLevel level, const std::string& message)
{
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    // Format timestamp
    std::tm tm_buf;
    localtime_r(&now_c, &tm_buf);

    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &tm_buf);

    // Get level string
    const char* level_str;
    switch (level) {
        case LogLevel::Trace:    level_str = "TRACE"; break;
        case LogLevel::Debug:    level_str = "DEBUG"; break;
        case LogLevel::Info:     level_str = "INFO"; break;
        case LogLevel::Warn:     level_str = "WARN"; break;
        case LogLevel::Error:    level_str = "ERROR"; break;
        case LogLevel::Critical: level_str = "CRITICAL"; break;
        default:                 level_str = "UNKNOWN"; break;
    }

    // Build log line
    char full_message[4096];
    snprintf(full_message, sizeof(full_message), "[%s.%03d] [%s] %s\n",
             time_str, static_cast<int>(now_ms.count()), level_str, message.c_str());

    // Output to stderr for warnings and errors, stdout otherwise
    if (level >= LogLevel::Warn) {
        std::cerr << full_message;
        std::cerr.flush();
    } else {
        std::cout << full_message;
        std::cout.flush();
    }

    // Output to file if configured
    if (file_sink_) {
        fprintf(file_sink_, "%s", full_message);
        fflush(file_sink_);
    }
}

auto Logger() -> std::shared_ptr<SimpleLogger>
{
    static auto logger = std::make_shared<SimpleLogger>();
    return logger;
}

void AddLogFile(const std::filesystem::path& path)
{
    Logger()->add_file_sink(path);
}

void SetLogLevel(const std::string& s)
{
    std::string level_str = s;
    std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::tolower);

    if (level_str == "trace") {
        Logger()->set_level(LogLevel::Trace);
    } else if (level_str == "debug") {
        Logger()->set_level(LogLevel::Debug);
    } else if (level_str == "info") {
        Logger()->set_level(LogLevel::Info);
    } else if (level_str == "warn" || level_str == "warning") {
        Logger()->set_level(LogLevel::Warn);
    } else if (level_str == "error" || level_str == "err") {
        Logger()->set_level(LogLevel::Error);
    } else if (level_str == "critical" || level_str == "crit") {
        Logger()->set_level(LogLevel::Critical);
    } else if (level_str == "off") {
        Logger()->set_level(LogLevel::Off);
    } else {
        std::cerr << "Unknown log level: " << s << std::endl;
    }
}
