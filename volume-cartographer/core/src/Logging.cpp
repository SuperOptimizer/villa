#include "vc/core/util/Logging.hpp"

#include <memory>
#include <unordered_map>

auto Logger() -> std::shared_ptr<SimpleLogger>
{
    static auto logger = std::make_shared<SimpleLogger>("volcart");
    return logger;
}

void AddLogFile(const std::filesystem::path& path)
{
    Logger()->add_file(path);
}

void SetLogLevel(const std::string& s)
{
    static const std::unordered_map<std::string, LogLevel> level_map = {
        {"trace", LogLevel::trace},
        {"debug", LogLevel::debug},
        {"info", LogLevel::info},
        {"warn", LogLevel::warn},
        {"warning", LogLevel::warn},
        {"error", LogLevel::error},
        {"critical", LogLevel::critical},
        {"off", LogLevel::off}
    };

    auto it = level_map.find(s);
    if (it != level_map.end()) {
        Logger()->set_level(it->second);
    }
}
