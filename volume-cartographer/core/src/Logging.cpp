#include "vc/core/util/Logging.hpp"

#include <memory>
#include <algorithm>

void MinimalLogger::add_file(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.close();
    }
    file_.open(path, std::ios::out | std::ios::app);
}

auto Logger() -> std::shared_ptr<MinimalLogger>
{
    static auto logger = std::make_shared<MinimalLogger>("volcart");
    return logger;
}

void AddLogFile(const std::filesystem::path& path)
{
    Logger()->add_file(path);
}

void SetLogLevel(const std::string& s)
{
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "debug") {
        Logger()->set_level(MinimalLogger::Level::Debug);
    } else if (lower == "info") {
        Logger()->set_level(MinimalLogger::Level::Info);
    } else if (lower == "warn" || lower == "warning") {
        Logger()->set_level(MinimalLogger::Level::Warn);
    } else if (lower == "error") {
        Logger()->set_level(MinimalLogger::Level::Error);
    }
}