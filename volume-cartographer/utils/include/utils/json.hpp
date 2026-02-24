#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <iosfwd>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace utils {

class Json {
public:
    enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };
    using array_t = std::vector<Json>;
    using object_t = std::map<std::string, Json, std::less<>>;

    // Default: null
    Json();

    // Implicit converting constructors
    Json(std::nullptr_t);
    Json(bool v);
    Json(double v);
    Json(int64_t v);
    Json(int v);
    Json(std::string v);
    Json(const char* v);
    Json(array_t v);
    Json(object_t v);

    // Named factories (kept for explicitness and backward compat)
    static auto null_val() -> Json;
    static auto boolean(bool v) -> Json;
    static auto number(double v) -> Json;
    static auto string_val(std::string v) -> Json;
    static auto array(array_t v = {}) -> Json;
    static auto object(object_t v = {}) -> Json;

    // Type queries
    [[nodiscard]] auto type() const noexcept -> Type;
    [[nodiscard]] auto is_null() const noexcept -> bool;
    [[nodiscard]] auto is_bool() const noexcept -> bool;
    [[nodiscard]] auto is_number() const noexcept -> bool;
    [[nodiscard]] auto is_string() const noexcept -> bool;
    [[nodiscard]] auto is_array() const noexcept -> bool;
    [[nodiscard]] auto is_object() const noexcept -> bool;

    // Value accessors
    [[nodiscard]] auto as_bool() const -> bool;
    [[nodiscard]] auto as_number() const -> double;
    [[nodiscard]] auto as_int() const -> int64_t;
    [[nodiscard]] auto as_string() const -> const std::string&;
    [[nodiscard]] auto as_array() const -> const array_t&;
    [[nodiscard]] auto as_array_mut() -> array_t&;
    [[nodiscard]] auto as_object() const -> const object_t&;
    [[nodiscard]] auto as_object_mut() -> object_t&;

    // Object subscript
    [[nodiscard]] auto operator[](std::string_view key) const -> const Json&;
    auto operator[](std::string_view key) -> Json&;  // auto-vivifies null to object
    [[nodiscard]] auto contains(std::string_view key) const -> bool;

    // Array subscript
    [[nodiscard]] auto operator[](std::size_t idx) const -> const Json&;
    auto operator[](std::size_t idx) -> Json&;

    // Size (array or object element count)
    [[nodiscard]] auto size() const noexcept -> std::size_t;

    // Parse / serialize
    [[nodiscard]] static auto parse(std::string_view input) -> std::expected<Json, std::string>;
    [[nodiscard]] auto serialize() const -> std::string;

    // Comparison / I/O
    [[nodiscard]] auto operator==(const Json& other) const -> bool;
    friend auto operator<<(std::ostream& os, const Json& j) -> std::ostream&;

private:
    Type type_ = Type::Null;
    bool bool_val_ = false;
    double num_val_ = 0.0;
    std::string str_val_;
    array_t arr_val_;
    object_t obj_val_;

    void serialize_to(std::string& out) const;
};

} // namespace utils
