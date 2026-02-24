#pragma once

#include <charconv>
#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace utils {

/// Result of parsing command-line arguments.
struct ParsedArgs {
    /// Check whether a flag was set or an option was provided.
    [[nodiscard]] auto has(std::string_view name) const -> bool;

    /// Return the first value for an option, or fallback if absent.
    [[nodiscard]] auto value(std::string_view name,
                             std::string_view fallback = "") const -> std::string;

    /// Return all values for a multi-valued option.
    [[nodiscard]] auto values(std::string_view name) const
        -> std::span<const std::string>;

    /// Return the collected positional arguments.
    [[nodiscard]] auto positionals() const -> std::span<const std::string>;

    /// Return the number of positional arguments.
    [[nodiscard]] auto positional_count() const -> std::size_t;

    /// Type-safe value conversion.  Supports arithmetic types and std::string.
    template <typename T>
    [[nodiscard]] auto value_as(std::string_view name) const
        -> std::expected<T, std::string>
    {
        if constexpr (std::is_same_v<T, std::string>) {
            auto v = value(name);
            if (v.empty() && !has(name)) {
                return std::unexpected("option not found: " + std::string(name));
            }
            return v;
        } else if constexpr (std::is_arithmetic_v<T>) {
            auto v = value(name);
            if (v.empty() && !has(name)) {
                return std::unexpected("option not found: " + std::string(name));
            }
            T result{};
            auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), result);
            if (ec != std::errc{} || ptr != v.data() + v.size()) {
                return std::unexpected(
                    "cannot convert '" + std::string(v) + "' to requested type");
            }
            return result;
        } else {
            static_assert(std::is_arithmetic_v<T> || std::is_same_v<T, std::string>,
                          "value_as<T> only supports arithmetic types and std::string");
        }
    }

    /// Type-safe value conversion with a default fallback.
    template <typename T>
    [[nodiscard]] auto value_as(std::string_view name, T fallback) const -> T {
        auto result = value_as<T>(name);
        return result.has_value() ? *result : fallback;
    }

    // ---- internal storage (public for aggregate init in tests) ----
    struct Impl;
    std::unique_ptr<Impl> impl_;

    ParsedArgs();
    ~ParsedArgs();
    ParsedArgs(ParsedArgs&&) noexcept;
    auto operator=(ParsedArgs&&) noexcept -> ParsedArgs&;
    ParsedArgs(const ParsedArgs&) = delete;
    auto operator=(const ParsedArgs&) -> ParsedArgs& = delete;
};

/// Command-line argument parser.
///
/// Supports flags (boolean switches), options (key-value pairs with optional
/// type conversion), positional arguments, aliases (short/long forms), and
/// auto-generated help text.
class ArgParser {
public:
    ArgParser();
    ~ArgParser();

    ArgParser(ArgParser&&) noexcept;
    auto operator=(ArgParser&&) noexcept -> ArgParser&;

    ArgParser(const ArgParser&) = delete;
    auto operator=(const ArgParser&) -> ArgParser& = delete;

    /// Register an option that takes a value.
    auto add_option(std::string_view canonical,
                    std::initializer_list<std::string_view> aliases,
                    std::string_view help,
                    bool required = false) -> void;

    /// Register a boolean flag (no value).
    auto add_flag(std::string_view canonical,
                  std::initializer_list<std::string_view> aliases,
                  std::string_view help) -> void;

    /// Register a positional argument with a descriptive name for help text.
    auto add_positional(std::string_view name,
                        std::string_view help,
                        bool required = true) -> void;

    /// Set a program description shown before option listing.
    auto set_description(std::string_view description) -> void;

    /// Parse arguments.  Returns ParsedArgs on success, or an error string.
    [[nodiscard]] auto parse(int argc, char** argv) const
        -> std::expected<ParsedArgs, std::string>;

    /// Generate formatted help text.
    [[nodiscard]] auto help_text() const -> std::string;

    /// Generate formatted help text with a custom heading.
    [[nodiscard]] auto help_text(std::string_view heading) const -> std::string;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace utils
