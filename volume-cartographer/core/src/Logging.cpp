#include "vc/core/util/Logging.hpp"

#include <memory>
#include <algorithm>
#include <cctype>

std::shared_ptr<SimpleLogger> Logger()
{
    static auto logger = std::make_shared<SimpleLogger>();
    return logger;
}

void AddLogFile(const std::filesystem::path& path)
{
    Logger()->addFileSink(path);
}

void SetLogLevel(const std::string& s)
{
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower == "debug") {
        Logger()->setLevel(SimpleLogger::Level::DEBUG);
    } else if (lower == "info") {
        Logger()->setLevel(SimpleLogger::Level::INFO);
    } else if (lower == "warn" || lower == "warning") {
        Logger()->setLevel(SimpleLogger::Level::WARN);
    } else if (lower == "error") {
        Logger()->setLevel(SimpleLogger::Level::ERROR);
    }
}
