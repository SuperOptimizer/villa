#include "utils/json.hpp"

#include <cmath>
#include <cstdio>
#include <ostream>
#include <utility>

namespace utils {

namespace {

auto err(std::size_t pos, std::string_view msg) -> std::unexpected<std::string> {
    return std::unexpected("JSON parse error at byte " + std::to_string(pos) + ": " + std::string(msg));
}

void skip_ws(std::string_view sv, std::size_t& pos) {
    while (pos < sv.size() && (sv[pos] == ' ' || sv[pos] == '\t' || sv[pos] == '\n' || sv[pos] == '\r'))
        ++pos;
}

// Forward declaration
auto parse_value(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string>;

auto parse_string_raw(std::string_view sv, std::size_t& pos) -> std::expected<std::string, std::string> {
    if (pos >= sv.size() || sv[pos] != '"')
        return err(pos, "expected '\"'");
    ++pos;
    std::string result;
    while (pos < sv.size()) {
        char c = sv[pos++];
        if (c == '"') return result;
        if (c == '\\') {
            if (pos >= sv.size()) return err(pos, "unterminated escape sequence");
            char esc = sv[pos++];
            switch (esc) {
            case '"':  result += '"';  break;
            case '\\': result += '\\'; break;
            case '/':  result += '/';  break;
            case 'b':  result += '\b'; break;
            case 'f':  result += '\f'; break;
            case 'n':  result += '\n'; break;
            case 'r':  result += '\r'; break;
            case 't':  result += '\t'; break;
            case 'u': {
                if (pos + 4 > sv.size()) return err(pos, "truncated \\u escape");
                auto hex = sv.substr(pos, 4);
                pos += 4;
                unsigned cp = 0;
                for (char h : hex) {
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= static_cast<unsigned>(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned>(h - 'A' + 10);
                    else return err(pos - 4, "invalid hex digit in \\u escape");
                }
                // Handle surrogate pairs
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (pos + 6 <= sv.size() && sv[pos] == '\\' && sv[pos + 1] == 'u') {
                        pos += 2;
                        auto hex2 = sv.substr(pos, 4);
                        pos += 4;
                        unsigned low = 0;
                        for (char h : hex2) {
                            low <<= 4;
                            if (h >= '0' && h <= '9') low |= static_cast<unsigned>(h - '0');
                            else if (h >= 'a' && h <= 'f') low |= static_cast<unsigned>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') low |= static_cast<unsigned>(h - 'A' + 10);
                            else return err(pos - 4, "invalid hex digit in surrogate pair");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                }
                // UTF-8 encode
                if (cp < 0x80) {
                    result += static_cast<char>(cp);
                } else if (cp < 0x800) {
                    result += static_cast<char>(0xC0 | (cp >> 6));
                    result += static_cast<char>(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    result += static_cast<char>(0xE0 | (cp >> 12));
                    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    result += static_cast<char>(0x80 | (cp & 0x3F));
                } else {
                    result += static_cast<char>(0xF0 | (cp >> 18));
                    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    result += static_cast<char>(0x80 | (cp & 0x3F));
                }
                break;
            }
            default: result += esc;
            }
        } else {
            result += c;
        }
    }
    return err(pos, "unterminated string");
}

auto parse_number(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string> {
    std::size_t start = pos;
    if (pos < sv.size() && sv[pos] == '-') ++pos;
    if (pos >= sv.size() || sv[pos] < '0' || sv[pos] > '9')
        return err(start, "expected digit");
    while (pos < sv.size() && sv[pos] >= '0' && sv[pos] <= '9') ++pos;
    if (pos < sv.size() && sv[pos] == '.') {
        ++pos;
        while (pos < sv.size() && sv[pos] >= '0' && sv[pos] <= '9') ++pos;
    }
    if (pos < sv.size() && (sv[pos] == 'e' || sv[pos] == 'E')) {
        ++pos;
        if (pos < sv.size() && (sv[pos] == '+' || sv[pos] == '-')) ++pos;
        while (pos < sv.size() && sv[pos] >= '0' && sv[pos] <= '9') ++pos;
    }
    std::string numstr(sv.substr(start, pos - start));
    try {
        double val = std::stod(numstr);
        return Json::number(val);
    } catch (...) {
        return err(start, "invalid number");
    }
}

auto parse_bool(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string> {
    if (sv.substr(pos, 4) == "true") { pos += 4; return Json::boolean(true); }
    if (sv.substr(pos, 5) == "false") { pos += 5; return Json::boolean(false); }
    return err(pos, "expected 'true' or 'false'");
}

auto parse_null(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string> {
    if (sv.substr(pos, 4) == "null") { pos += 4; return Json{}; }
    return err(pos, "expected 'null'");
}

auto parse_array(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string> {
    ++pos; // skip '['
    Json::array_t result;
    skip_ws(sv, pos);
    if (pos < sv.size() && sv[pos] == ']') { ++pos; return Json::array({}); }
    while (pos < sv.size()) {
        auto val = parse_value(sv, pos);
        if (!val) return val;
        result.push_back(std::move(*val));
        skip_ws(sv, pos);
        if (pos < sv.size() && sv[pos] == ',') { ++pos; continue; }
        if (pos < sv.size() && sv[pos] == ']') { ++pos; return Json::array(std::move(result)); }
        return err(pos, "expected ',' or ']'");
    }
    return err(pos, "unterminated array");
}

auto parse_object(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string> {
    ++pos; // skip '{'
    Json::object_t result;
    skip_ws(sv, pos);
    if (pos < sv.size() && sv[pos] == '}') { ++pos; return Json::object({}); }
    while (pos < sv.size()) {
        skip_ws(sv, pos);
        if (pos >= sv.size() || sv[pos] != '"')
            return err(pos, "expected string key");
        auto key = parse_string_raw(sv, pos);
        if (!key) return std::unexpected(key.error());
        skip_ws(sv, pos);
        if (pos >= sv.size() || sv[pos] != ':')
            return err(pos, "expected ':'");
        ++pos;
        auto val = parse_value(sv, pos);
        if (!val) return val;
        result.emplace(std::move(*key), std::move(*val));
        skip_ws(sv, pos);
        if (pos < sv.size() && sv[pos] == ',') { ++pos; continue; }
        if (pos < sv.size() && sv[pos] == '}') { ++pos; return Json::object(std::move(result)); }
        return err(pos, "expected ',' or '}'");
    }
    return err(pos, "unterminated object");
}

auto parse_value(std::string_view sv, std::size_t& pos) -> std::expected<Json, std::string> {
    skip_ws(sv, pos);
    if (pos >= sv.size()) return err(pos, "unexpected end of input");
    char c = sv[pos];
    if (c == '"') {
        auto s = parse_string_raw(sv, pos);
        if (!s) return std::unexpected(s.error());
        return Json::string_val(std::move(*s));
    }
    if (c == '{') return parse_object(sv, pos);
    if (c == '[') return parse_array(sv, pos);
    if (c == 't' || c == 'f') return parse_bool(sv, pos);
    if (c == 'n') return parse_null(sv, pos);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(sv, pos);
    return err(pos, std::string("unexpected character '") + c + "'");
}

} // anonymous namespace

// ── Constructors ──────────────────────────────────────────────────────────────

Json::Json() = default;

Json::Json(std::nullptr_t) {}

Json::Json(bool v) : type_(Type::Bool), bool_val_(v) {}

Json::Json(double v) : type_(Type::Number), num_val_(v) {}

Json::Json(int64_t v) : type_(Type::Number), num_val_(static_cast<double>(v)) {}

Json::Json(int v) : type_(Type::Number), num_val_(static_cast<double>(v)) {}

Json::Json(std::string v) : type_(Type::String), str_val_(std::move(v)) {}

Json::Json(const char* v) : type_(Type::String), str_val_(v) {}

Json::Json(array_t v) : type_(Type::Array), arr_val_(std::move(v)) {}

Json::Json(object_t v) : type_(Type::Object), obj_val_(std::move(v)) {}

// ── Named factories ───────────────────────────────────────────────────────────

auto Json::null_val() -> Json { return {}; }

auto Json::boolean(bool v) -> Json { return Json(v); }

auto Json::number(double v) -> Json { return Json(v); }

auto Json::string_val(std::string v) -> Json { return Json(std::move(v)); }

auto Json::array(array_t v) -> Json { return Json(std::move(v)); }

auto Json::object(object_t v) -> Json { return Json(std::move(v)); }

// ── Type queries ──────────────────────────────────────────────────────────────

auto Json::type() const noexcept -> Type { return type_; }
auto Json::is_null() const noexcept -> bool { return type_ == Type::Null; }
auto Json::is_bool() const noexcept -> bool { return type_ == Type::Bool; }
auto Json::is_number() const noexcept -> bool { return type_ == Type::Number; }
auto Json::is_string() const noexcept -> bool { return type_ == Type::String; }
auto Json::is_array() const noexcept -> bool { return type_ == Type::Array; }
auto Json::is_object() const noexcept -> bool { return type_ == Type::Object; }

// ── Value accessors ───────────────────────────────────────────────────────────

auto Json::as_bool() const -> bool { return bool_val_; }
auto Json::as_number() const -> double { return num_val_; }
auto Json::as_int() const -> int64_t { return static_cast<int64_t>(num_val_); }
auto Json::as_string() const -> const std::string& { return str_val_; }
auto Json::as_array() const -> const array_t& { return arr_val_; }
auto Json::as_array_mut() -> array_t& { return arr_val_; }
auto Json::as_object() const -> const object_t& { return obj_val_; }
auto Json::as_object_mut() -> object_t& { return obj_val_; }

// ── Subscript ─────────────────────────────────────────────────────────────────

auto Json::operator[](std::string_view key) const -> const Json& {
    auto it = obj_val_.find(key);
    if (it != obj_val_.end()) return it->second;
    static const Json null_json;
    return null_json;
}

auto Json::operator[](std::string_view key) -> Json& {
    if (type_ == Type::Null) {
        type_ = Type::Object;
    }
    return obj_val_[std::string(key)];
}

auto Json::contains(std::string_view key) const -> bool {
    return obj_val_.find(key) != obj_val_.end();
}

auto Json::operator[](std::size_t idx) const -> const Json& { return arr_val_[idx]; }
auto Json::operator[](std::size_t idx) -> Json& { return arr_val_[idx]; }

auto Json::size() const noexcept -> std::size_t {
    if (type_ == Type::Array) return arr_val_.size();
    if (type_ == Type::Object) return obj_val_.size();
    return 0;
}

// ── Parse / serialize ─────────────────────────────────────────────────────────

auto Json::parse(std::string_view input) -> std::expected<Json, std::string> {
    std::size_t pos = 0;
    return parse_value(input, pos);
}

void Json::serialize_to(std::string& out) const {
    switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += bool_val_ ? "true" : "false"; break;
    case Type::Number: {
        double intpart;
        if (std::modf(num_val_, &intpart) == 0.0 && std::isfinite(num_val_)
            && num_val_ >= -1e15 && num_val_ <= 1e15) {
            auto i = static_cast<int64_t>(num_val_);
            out += std::to_string(i);
        } else if (std::isnan(num_val_)) {
            out += "\"NaN\"";
        } else if (std::isinf(num_val_)) {
            out += num_val_ > 0 ? "\"Infinity\"" : "\"-Infinity\"";
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.17g", num_val_);
            out += buf;
        }
        break;
    }
    case Type::String: {
        out += '"';
        for (char c : str_val_) {
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
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
            }
        }
        out += '"';
        break;
    }
    case Type::Array: {
        out += '[';
        for (std::size_t i = 0; i < arr_val_.size(); ++i) {
            if (i > 0) out += ',';
            arr_val_[i].serialize_to(out);
        }
        out += ']';
        break;
    }
    case Type::Object: {
        out += '{';
        bool first = true;
        for (const auto& [k, v] : obj_val_) {
            if (!first) out += ',';
            first = false;
            Json::string_val(k).serialize_to(out);
            out += ':';
            v.serialize_to(out);
        }
        out += '}';
        break;
    }
    }
}

auto Json::serialize() const -> std::string {
    std::string out;
    serialize_to(out);
    return out;
}

// ── Comparison / I/O ──────────────────────────────────────────────────────────

auto Json::operator==(const Json& other) const -> bool {
    if (type_ != other.type_) return false;
    switch (type_) {
    case Type::Null: return true;
    case Type::Bool: return bool_val_ == other.bool_val_;
    case Type::Number: return num_val_ == other.num_val_;
    case Type::String: return str_val_ == other.str_val_;
    case Type::Array: return arr_val_ == other.arr_val_;
    case Type::Object: return obj_val_ == other.obj_val_;
    }
    return false;
}

auto operator<<(std::ostream& os, const Json& j) -> std::ostream& {
    return os << j.serialize();
}

} // namespace utils
