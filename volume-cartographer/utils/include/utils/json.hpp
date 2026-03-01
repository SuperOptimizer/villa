#pragma once
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace utils {

// ---------------------------------------------------------------------------
// JSON value types
// ---------------------------------------------------------------------------

struct JsonValue;
using JsonObject = std::unordered_map<std::string, JsonValue>;
using JsonArray  = std::vector<JsonValue>;

struct JsonValue {
    std::variant<
        std::nullptr_t,
        bool,
        double,
        std::string,
        JsonArray,
        JsonObject
    > data = nullptr;

    // -- Construction helpers ------------------------------------------------

    JsonValue() = default;
    JsonValue(std::nullptr_t)        : data(nullptr) {}
    JsonValue(bool v)                : data(v) {}
    JsonValue(double v)              : data(v) {}
    JsonValue(int v)                 : data(static_cast<double>(v)) {}
    JsonValue(std::int64_t v)        : data(static_cast<double>(v)) {}
    JsonValue(std::size_t v)         : data(static_cast<double>(v)) {}
    JsonValue(const char* v)         : data(std::string(v)) {}
    JsonValue(std::string v)         : data(std::move(v)) {}
    JsonValue(std::string_view v)    : data(std::string(v)) {}
    JsonValue(JsonArray v)           : data(std::move(v)) {}
    JsonValue(JsonObject v)          : data(std::move(v)) {}

    // -- Type queries --------------------------------------------------------

    [[nodiscard]] bool is_null()   const noexcept { return std::holds_alternative<std::nullptr_t>(data); }
    [[nodiscard]] bool is_bool()   const noexcept { return std::holds_alternative<bool>(data); }
    [[nodiscard]] bool is_number() const noexcept { return std::holds_alternative<double>(data); }
    [[nodiscard]] bool is_string() const noexcept { return std::holds_alternative<std::string>(data); }
    [[nodiscard]] bool is_array()  const noexcept { return std::holds_alternative<JsonArray>(data); }
    [[nodiscard]] bool is_object() const noexcept { return std::holds_alternative<JsonObject>(data); }

    // -- Accessors (const) ---------------------------------------------------

    [[nodiscard]] bool               as_bool()   const { return std::get<bool>(data); }
    [[nodiscard]] double             as_number() const { return std::get<double>(data); }
    [[nodiscard]] const std::string& as_string() const { return std::get<std::string>(data); }
    [[nodiscard]] const JsonArray&   as_array()  const { return std::get<JsonArray>(data); }
    [[nodiscard]] const JsonObject&  as_object() const { return std::get<JsonObject>(data); }

    // -- Accessors (mutable) -------------------------------------------------

    [[nodiscard]] std::string& as_string() { return std::get<std::string>(data); }
    [[nodiscard]] JsonArray&   as_array()  { return std::get<JsonArray>(data); }
    [[nodiscard]] JsonObject&  as_object() { return std::get<JsonObject>(data); }

    // -- Convenience helpers -------------------------------------------------

    /// Get a number cast to an integer type.
    template <typename T = std::int64_t>
    [[nodiscard]] T as_int() const { return static_cast<T>(as_number()); }

    /// Look up a key in an object. Throws std::out_of_range if missing.
    [[nodiscard]] const JsonValue& operator[](const std::string& key) const {
        return as_object().at(key);
    }

    /// Look up an index in an array. Throws std::out_of_range if out of bounds.
    [[nodiscard]] const JsonValue& operator[](std::size_t index) const {
        return as_array().at(index);
    }

    /// Return pointer to value for key, or nullptr if not found / not an object.
    [[nodiscard]] const JsonValue* find(const std::string& key) const noexcept {
        if (!is_object()) return nullptr;
        auto it = std::get<JsonObject>(data).find(key);
        if (it == std::get<JsonObject>(data).end()) return nullptr;
        return &it->second;
    }

    /// Get value for key with a default if missing.
    [[nodiscard]] const JsonValue& get(const std::string& key, const JsonValue& fallback) const noexcept {
        if (auto* p = find(key)) return *p;
        return fallback;
    }

    /// Number of elements (array) or entries (object). Returns 0 for other types.
    [[nodiscard]] std::size_t size() const noexcept {
        if (is_array())  return as_array().size();
        if (is_object()) return as_object().size();
        return 0;
    }

    /// Check if array/object is empty, or if value is null.
    [[nodiscard]] bool empty() const noexcept {
        if (is_null()) return true;
        return size() == 0;
    }
};

// ---------------------------------------------------------------------------
// JSON parser  (minimal recursive-descent)
// ---------------------------------------------------------------------------

namespace json_detail {

inline void skip_ws(std::string_view& s) noexcept {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '\n' || s.front() == '\r'))
        s.remove_prefix(1);
}

inline JsonValue parse_value(std::string_view& s);

inline std::string parse_string(std::string_view& s) {
    if (s.empty() || s.front() != '"')
        throw std::runtime_error("json: expected '\"'");
    s.remove_prefix(1);
    std::string out;
    while (!s.empty() && s.front() != '"') {
        if (s.front() == '\\') {
            s.remove_prefix(1);
            if (s.empty()) throw std::runtime_error("json: unexpected end of string escape");
            switch (s.front()) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'b':  out += '\b'; break;
                case 'f':  out += '\f'; break;
                case 'n':  out += '\n'; break;
                case 'r':  out += '\r'; break;
                case 't':  out += '\t'; break;
                case 'u': {
                    // Basic \uXXXX support (BMP only, no surrogate pairs).
                    if (s.size() < 5)
                        throw std::runtime_error("json: incomplete \\u escape");
                    s.remove_prefix(1);
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        cp <<= 4;
                        char c = s[static_cast<std::size_t>(i)];
                        if (c >= '0' && c <= '9')      cp |= static_cast<unsigned>(c - '0');
                        else if (c >= 'a' && c <= 'f') cp |= static_cast<unsigned>(c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') cp |= static_cast<unsigned>(c - 'A' + 10);
                        else throw std::runtime_error("json: invalid \\u hex digit");
                    }
                    s.remove_prefix(3); // 4th char consumed by outer loop
                    // Encode as UTF-8.
                    if (cp < 0x80) {
                        out += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: out += s.front(); break;
            }
        } else {
            out += s.front();
        }
        s.remove_prefix(1);
    }
    if (s.empty()) throw std::runtime_error("json: unterminated string");
    s.remove_prefix(1); // closing '"'
    return out;
}

inline double parse_number(std::string_view& s) {
    std::size_t len = 0;
    if (!s.empty() && s[0] == '-') ++len;
    bool has_dot = false;
    bool in_exponent = false;
    while (len < s.size()) {
        char c = s[len];
        if (c >= '0' && c <= '9') {
            ++len;
        } else if (c == '.' && !has_dot && !in_exponent) {
            has_dot = true;
            ++len;
        } else if ((c == 'e' || c == 'E') && !in_exponent) {
            in_exponent = true;
            ++len;
        } else if ((c == '+' || c == '-') && len > 0 && (s[len - 1] == 'e' || s[len - 1] == 'E')) {
            ++len;
        } else {
            break;
        }
    }
    double val = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + len, val);
    if (ec != std::errc{})
        throw std::runtime_error("json: bad number");
    s.remove_prefix(static_cast<std::size_t>(ptr - s.data()));
    return val;
}

inline JsonValue parse_value(std::string_view& s) {
    skip_ws(s);
    if (s.empty()) throw std::runtime_error("json: unexpected end of input");

    JsonValue v;
    if (s.front() == '"') {
        v.data = parse_string(s);
    } else if (s.front() == '{') {
        s.remove_prefix(1);
        JsonObject obj;
        skip_ws(s);
        while (!s.empty() && s.front() != '}') {
            skip_ws(s);
            auto key = parse_string(s);
            skip_ws(s);
            if (s.empty() || s.front() != ':')
                throw std::runtime_error("json: expected ':' in object");
            s.remove_prefix(1);
            obj[std::move(key)] = parse_value(s);
            skip_ws(s);
            if (!s.empty() && s.front() == ',') s.remove_prefix(1);
        }
        if (s.empty()) throw std::runtime_error("json: unterminated object");
        s.remove_prefix(1); // '}'
        v.data = std::move(obj);
    } else if (s.front() == '[') {
        s.remove_prefix(1);
        JsonArray arr;
        skip_ws(s);
        while (!s.empty() && s.front() != ']') {
            arr.push_back(parse_value(s));
            skip_ws(s);
            if (!s.empty() && s.front() == ',') s.remove_prefix(1);
        }
        if (s.empty()) throw std::runtime_error("json: unterminated array");
        s.remove_prefix(1); // ']'
        v.data = std::move(arr);
    } else if (s.starts_with("true")) {
        v.data = true;
        s.remove_prefix(4);
    } else if (s.starts_with("false")) {
        v.data = false;
        s.remove_prefix(5);
    } else if (s.starts_with("null")) {
        v.data = nullptr;
        s.remove_prefix(4);
    } else {
        v.data = parse_number(s);
    }
    skip_ws(s);
    return v;
}

} // namespace json_detail

// ---------------------------------------------------------------------------
// Public parse API
// ---------------------------------------------------------------------------

/// Parse a JSON string into a JsonValue. Throws std::runtime_error on failure.
[[nodiscard]] inline JsonValue json_parse(std::string_view text) {
    return json_detail::parse_value(text);
}

// ---------------------------------------------------------------------------
// JSON serializer
// ---------------------------------------------------------------------------

namespace json_detail {

inline void escape_string(std::string& out, std::string_view sv) {
    out += '"';
    for (char c : sv) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Control character -- emit \u00XX.
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

inline void serialize_impl(std::string& out, const JsonValue& v,
                            int indent, int depth) {
    const bool pretty = indent > 0;
    auto pad = [&](int extra = 0) {
        if (pretty) out.append(static_cast<std::size_t>((depth + extra) * indent), ' ');
    };

    if (v.is_null()) {
        out += "null";
    } else if (v.is_bool()) {
        out += v.as_bool() ? "true" : "false";
    } else if (v.is_number()) {
        double d = v.as_number();
        // Emit integers without trailing decimals.
        if (std::isfinite(d) && d == static_cast<double>(static_cast<std::int64_t>(d))
            && std::abs(d) < 1e15) {
            out += std::to_string(static_cast<std::int64_t>(d));
        } else {
            char buf[64];
            auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), d);
            out.append(buf, static_cast<std::size_t>(ptr - buf));
        }
    } else if (v.is_string()) {
        escape_string(out, v.as_string());
    } else if (v.is_array()) {
        const auto& arr = v.as_array();
        if (arr.empty()) { out += "[]"; return; }
        out += '[';
        if (pretty) out += '\n';
        for (std::size_t i = 0; i < arr.size(); ++i) {
            pad(1);
            serialize_impl(out, arr[i], indent, depth + 1);
            if (i + 1 < arr.size()) out += ',';
            if (pretty) out += '\n';
        }
        pad();
        out += ']';
    } else if (v.is_object()) {
        const auto& obj = v.as_object();
        if (obj.empty()) { out += "{}"; return; }
        out += '{';
        if (pretty) out += '\n';
        std::size_t idx = 0;
        // Sort keys for deterministic output.
        std::vector<std::string> keys;
        keys.reserve(obj.size());
        for (const auto& [k, _] : obj) keys.push_back(k);
        std::sort(keys.begin(), keys.end());
        for (const auto& k : keys) {
            pad(1);
            escape_string(out, k);
            out += ':';
            if (pretty) out += ' ';
            serialize_impl(out, obj.at(k), indent, depth + 1);
            if (idx + 1 < obj.size()) out += ',';
            if (pretty) out += '\n';
            ++idx;
        }
        pad();
        out += '}';
    }
}

} // namespace json_detail

/// Serialize a JsonValue to a JSON string.
/// Set indent > 0 for pretty-printing (number of spaces per level).
/// Set indent = 0 for compact output.
[[nodiscard]] inline std::string json_serialize(const JsonValue& v, int indent = 0) {
    std::string out;
    json_detail::serialize_impl(out, v, indent, 0);
    return out;
}

// ---------------------------------------------------------------------------
// Builder helpers  -- construct JSON values ergonomically
// ---------------------------------------------------------------------------

/// Create a JSON object from an initializer list of key-value pairs.
[[nodiscard]] inline JsonValue json_object(std::initializer_list<std::pair<std::string, JsonValue>> pairs) {
    JsonObject obj;
    for (auto& [k, v] : pairs) obj.emplace(k, v);
    return JsonValue{std::move(obj)};
}

/// Create a JSON array from a vector of values.
[[nodiscard]] inline JsonValue json_array(std::initializer_list<JsonValue> values) {
    return JsonValue{JsonArray(values)};
}

} // namespace utils
