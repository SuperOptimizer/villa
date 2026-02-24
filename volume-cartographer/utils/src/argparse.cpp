#include "utils/argparse.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace utils {

// =============================================================================
// ParsedArgs::Impl
// =============================================================================

struct ParsedArgs::Impl {
    std::unordered_map<std::string, std::vector<std::string>> values;
    std::unordered_set<std::string> flags;
    std::vector<std::string> positionals;
};

ParsedArgs::ParsedArgs() : impl_(std::make_unique<Impl>()) {}
ParsedArgs::~ParsedArgs() = default;
ParsedArgs::ParsedArgs(ParsedArgs&&) noexcept = default;
auto ParsedArgs::operator=(ParsedArgs&&) noexcept -> ParsedArgs& = default;

auto ParsedArgs::has(std::string_view name) const -> bool {
    auto key = std::string(name);
    return impl_->flags.contains(key) || impl_->values.contains(key);
}

auto ParsedArgs::value(std::string_view name, std::string_view fallback) const
    -> std::string {
    auto it = impl_->values.find(std::string(name));
    if (it == impl_->values.end() || it->second.empty()) {
        return std::string(fallback);
    }
    return it->second.front();
}

auto ParsedArgs::values(std::string_view name) const
    -> std::span<const std::string> {
    auto it = impl_->values.find(std::string(name));
    if (it == impl_->values.end()) {
        return {};
    }
    return it->second;
}

auto ParsedArgs::positionals() const -> std::span<const std::string> {
    return impl_->positionals;
}

auto ParsedArgs::positional_count() const -> std::size_t {
    return impl_->positionals.size();
}

// =============================================================================
// ArgParser::Impl
// =============================================================================

struct ArgParser::Impl {
    struct OptionSpec {
        std::string canonical;
        std::vector<std::string> names; // canonical + aliases
        bool takes_value{false};
        bool required{false};
        std::string help;
    };

    struct PositionalSpec {
        std::string name;
        std::string help;
        bool required{true};
    };

    std::string description;
    std::vector<OptionSpec> specs;
    std::vector<PositionalSpec> positionals;
    std::unordered_map<std::string, std::size_t> alias_to_index;

    auto add_spec(std::string_view canonical,
                  std::initializer_list<std::string_view> aliases,
                  bool takes_value,
                  bool required,
                  std::string_view help) -> void {
        OptionSpec spec;
        spec.canonical = std::string(canonical);
        spec.takes_value = takes_value;
        spec.required = required;
        spec.help = std::string(help);
        spec.names.emplace_back(canonical);
        for (auto alias : aliases) {
            if (!alias.empty()) {
                spec.names.emplace_back(alias);
            }
        }
        specs.push_back(std::move(spec));
        auto index = specs.size() - 1;
        for (const auto& name : specs.back().names) {
            alias_to_index[name] = index;
        }
    }

    static auto format_name(std::string_view name) -> std::string {
        if (name.size() == 1) {
            return std::string("-") + std::string(name);
        }
        return std::string("--") + std::string(name);
    }

    static auto format_names(const std::vector<std::string>& names) -> std::string {
        std::ostringstream out;
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << format_name(names[i]);
        }
        return out.str();
    }
};

ArgParser::ArgParser() : impl_(std::make_unique<Impl>()) {}
ArgParser::~ArgParser() = default;
ArgParser::ArgParser(ArgParser&&) noexcept = default;
auto ArgParser::operator=(ArgParser&&) noexcept -> ArgParser& = default;

auto ArgParser::add_option(std::string_view canonical,
                           std::initializer_list<std::string_view> aliases,
                           std::string_view help,
                           bool required) -> void {
    impl_->add_spec(canonical, aliases, /*takes_value=*/true, required, help);
}

auto ArgParser::add_flag(std::string_view canonical,
                         std::initializer_list<std::string_view> aliases,
                         std::string_view help) -> void {
    impl_->add_spec(canonical, aliases, /*takes_value=*/false, /*required=*/false, help);
}

auto ArgParser::add_positional(std::string_view name,
                               std::string_view help,
                               bool required) -> void {
    impl_->positionals.push_back({
        .name = std::string(name),
        .help = std::string(help),
        .required = required,
    });
}

auto ArgParser::set_description(std::string_view description) -> void {
    impl_->description = std::string(description);
}

auto ArgParser::parse(int argc, char** argv) const
    -> std::expected<ParsedArgs, std::string> {
    ParsedArgs out;
    bool collect_positionals = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view token(argv[i]);

        // After "--", everything is positional
        if (collect_positionals || token.empty() || token[0] != '-') {
            out.impl_->positionals.emplace_back(token);
            continue;
        }

        if (token == "--") {
            collect_positionals = true;
            continue;
        }

        // Strip leading dashes and extract name/value
        bool has_value = false;
        std::string name;
        std::string value;

        if (token.starts_with("--")) {
            name = token.substr(2);
        } else {
            name = token.substr(1);
        }

        auto eq = name.find('=');
        if (eq != std::string::npos) {
            value = name.substr(eq + 1);
            name = name.substr(0, eq);
            has_value = true;
        }

        auto it = impl_->alias_to_index.find(name);
        if (it == impl_->alias_to_index.end()) {
            return std::unexpected("unknown option: " + std::string(token));
        }

        const auto& spec = impl_->specs[it->second];
        if (spec.takes_value) {
            if (!has_value) {
                if (i + 1 >= argc) {
                    return std::unexpected(
                        "missing value for option: " +
                        Impl::format_name(spec.canonical));
                }
                value = argv[++i];
            }
            out.impl_->values[spec.canonical].push_back(std::move(value));
        } else {
            if (has_value) {
                return std::unexpected(
                    "flag does not take a value: " +
                    Impl::format_name(spec.canonical));
            }
            out.impl_->flags.insert(spec.canonical);
        }
    }

    // Check required options
    for (const auto& spec : impl_->specs) {
        if (spec.required && !out.impl_->values.contains(spec.canonical)) {
            return std::unexpected(
                "missing required option: " + Impl::format_name(spec.canonical));
        }
    }

    // Check required positionals
    for (std::size_t i = 0; i < impl_->positionals.size(); ++i) {
        if (impl_->positionals[i].required &&
            i >= out.impl_->positionals.size()) {
            return std::unexpected(
                "missing required positional argument: " +
                impl_->positionals[i].name);
        }
    }

    return out;
}

auto ArgParser::help_text() const -> std::string {
    return help_text("");
}

auto ArgParser::help_text(std::string_view heading) const -> std::string {
    std::ostringstream out;

    if (!heading.empty()) {
        out << heading << "\n";
    }

    if (!impl_->description.empty()) {
        out << impl_->description << "\n";
    }

    if (!impl_->positionals.empty()) {
        out << "\nPositional arguments:\n";
        for (const auto& pos : impl_->positionals) {
            out << "  " << pos.name;
            if (!pos.required) {
                out << " (optional)";
            }
            if (!pos.help.empty()) {
                out << "  " << pos.help;
            }
            out << "\n";
        }
    }

    if (!impl_->specs.empty()) {
        out << "\nOptions:\n";
        for (const auto& spec : impl_->specs) {
            out << "  " << Impl::format_names(spec.names);
            if (spec.takes_value) {
                out << " <value>";
            }
            if (spec.required) {
                out << " (required)";
            }
            if (!spec.help.empty()) {
                out << "  " << spec.help;
            }
            out << "\n";
        }
    }

    return out.str();
}

} // namespace utils
