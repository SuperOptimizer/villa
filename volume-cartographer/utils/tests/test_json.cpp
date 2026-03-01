#include <utils/json.hpp>
#include <utils/test.hpp>
#include <cmath>
#include <stdexcept>

using namespace utils;

// ===========================================================================
// Parsing primitives
// ===========================================================================

TEST_CASE("json: parse null") {
    auto v = json_parse("null");
    REQUIRE(v.is_null());
    REQUIRE(!v.is_bool());
    REQUIRE(!v.is_number());
    REQUIRE(!v.is_string());
    REQUIRE(!v.is_array());
    REQUIRE(!v.is_object());
}

TEST_CASE("json: parse true") {
    auto v = json_parse("true");
    REQUIRE(v.is_bool());
    REQUIRE_EQ(v.as_bool(), true);
}

TEST_CASE("json: parse false") {
    auto v = json_parse("false");
    REQUIRE(v.is_bool());
    REQUIRE_EQ(v.as_bool(), false);
}

TEST_CASE("json: parse integer") {
    auto v = json_parse("42");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 42.0, 1e-9);
    REQUIRE_EQ(v.as_int(), 42);
}

TEST_CASE("json: parse negative integer") {
    auto v = json_parse("-7");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), -7.0, 1e-9);
}

TEST_CASE("json: parse zero") {
    auto v = json_parse("0");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 0.0, 1e-9);
}

TEST_CASE("json: parse floating point") {
    auto v = json_parse("3.14");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 3.14, 1e-9);
}

TEST_CASE("json: parse negative float") {
    auto v = json_parse("-0.5");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), -0.5, 1e-9);
}

TEST_CASE("json: parse scientific notation") {
    auto v = json_parse("1.5e10");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 1.5e10, 1.0);
}

TEST_CASE("json: parse negative exponent") {
    auto v = json_parse("2.5E-3");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 0.0025, 1e-12);
}

TEST_CASE("json: parse simple string") {
    auto v = json_parse("\"hello\"");
    REQUIRE(v.is_string());
    REQUIRE_EQ(v.as_string(), std::string("hello"));
}

TEST_CASE("json: parse empty string") {
    auto v = json_parse("\"\"");
    REQUIRE(v.is_string());
    REQUIRE_EQ(v.as_string(), std::string(""));
}

// ===========================================================================
// String escape sequences
// ===========================================================================

TEST_CASE("json: string escape - quote") {
    auto v = json_parse("\"a\\\"b\"");
    REQUIRE_EQ(v.as_string(), std::string("a\"b"));
}

TEST_CASE("json: string escape - backslash") {
    auto v = json_parse("\"a\\\\b\"");
    REQUIRE_EQ(v.as_string(), std::string("a\\b"));
}

TEST_CASE("json: string escape - slash") {
    auto v = json_parse("\"a\\/b\"");
    REQUIRE_EQ(v.as_string(), std::string("a/b"));
}

TEST_CASE("json: string escape - backspace") {
    auto v = json_parse("\"a\\bb\"");
    REQUIRE_EQ(v.as_string(), std::string("a\bb"));
}

TEST_CASE("json: string escape - formfeed") {
    auto v = json_parse("\"a\\fb\"");
    REQUIRE_EQ(v.as_string(), std::string("a\fb"));
}

TEST_CASE("json: string escape - newline") {
    auto v = json_parse("\"a\\nb\"");
    REQUIRE_EQ(v.as_string(), std::string("a\nb"));
}

TEST_CASE("json: string escape - carriage return") {
    auto v = json_parse("\"a\\rb\"");
    REQUIRE_EQ(v.as_string(), std::string("a\rb"));
}

TEST_CASE("json: string escape - tab") {
    auto v = json_parse("\"a\\tb\"");
    REQUIRE_EQ(v.as_string(), std::string("a\tb"));
}

TEST_CASE("json: string escape - unicode BMP ascii") {
    // \u0041 is 'A'
    auto v = json_parse("\"\\u0041\"");
    REQUIRE_EQ(v.as_string(), std::string("A"));
}

TEST_CASE("json: string escape - unicode 2-byte") {
    // \u00E9 is 'e' with acute accent (UTF-8: 0xC3 0xA9)
    auto v = json_parse("\"caf\\u00E9\"");
    REQUIRE_EQ(v.as_string(), std::string("caf\xC3\xA9"));
}

TEST_CASE("json: string escape - unicode 3-byte") {
    // \u4E16 is a CJK character (UTF-8: 0xE4 0xB8 0x96)
    auto v = json_parse("\"\\u4E16\"");
    std::string expected;
    expected += static_cast<char>(0xE4);
    expected += static_cast<char>(0xB8);
    expected += static_cast<char>(0x96);
    REQUIRE_EQ(v.as_string(), expected);
}

TEST_CASE("json: string escape - unicode lowercase hex") {
    auto v = json_parse("\"\\u00e9\"");
    REQUIRE_EQ(v.as_string(), std::string("caf\xC3\xA9").substr(3));
}

TEST_CASE("json: string escape - multiple escapes in one string") {
    auto v = json_parse("\"line1\\nline2\\ttab\\\\backslash\"");
    REQUIRE_EQ(v.as_string(), std::string("line1\nline2\ttab\\backslash"));
}

// ===========================================================================
// Arrays
// ===========================================================================

TEST_CASE("json: parse empty array") {
    auto v = json_parse("[]");
    REQUIRE(v.is_array());
    REQUIRE_EQ(v.size(), std::size_t(0));
    REQUIRE(v.empty());
}

TEST_CASE("json: parse array of numbers") {
    auto v = json_parse("[1, 2, 3]");
    REQUIRE(v.is_array());
    REQUIRE_EQ(v.size(), std::size_t(3));
    REQUIRE_NEAR(v[0].as_number(), 1.0, 1e-9);
    REQUIRE_NEAR(v[1].as_number(), 2.0, 1e-9);
    REQUIRE_NEAR(v[2].as_number(), 3.0, 1e-9);
}

TEST_CASE("json: parse array of mixed types") {
    auto v = json_parse("[1, \"two\", true, null, 4.5]");
    REQUIRE_EQ(v.size(), std::size_t(5));
    REQUIRE(v[0].is_number());
    REQUIRE(v[1].is_string());
    REQUIRE(v[2].is_bool());
    REQUIRE(v[3].is_null());
    REQUIRE(v[4].is_number());
}

TEST_CASE("json: parse nested arrays") {
    auto v = json_parse("[[1, 2], [3, 4]]");
    REQUIRE_EQ(v.size(), std::size_t(2));
    REQUIRE(v[0].is_array());
    REQUIRE(v[1].is_array());
    REQUIRE_NEAR(v[0][0].as_number(), 1.0, 1e-9);
    REQUIRE_NEAR(v[1][1].as_number(), 4.0, 1e-9);
}

TEST_CASE("json: parse single element array") {
    auto v = json_parse("[42]");
    REQUIRE(v.is_array());
    REQUIRE_EQ(v.size(), std::size_t(1));
    REQUIRE_NEAR(v[0].as_number(), 42.0, 1e-9);
}

// ===========================================================================
// Objects
// ===========================================================================

TEST_CASE("json: parse empty object") {
    auto v = json_parse("{}");
    REQUIRE(v.is_object());
    REQUIRE_EQ(v.size(), std::size_t(0));
    REQUIRE(v.empty());
}

TEST_CASE("json: parse simple object") {
    auto v = json_parse("{\"name\": \"Alice\", \"age\": 30}");
    REQUIRE(v.is_object());
    REQUIRE_EQ(v.size(), std::size_t(2));
    REQUIRE_EQ(v["name"].as_string(), std::string("Alice"));
    REQUIRE_NEAR(v["age"].as_number(), 30.0, 1e-9);
}

TEST_CASE("json: parse nested object") {
    auto v = json_parse("{\"person\": {\"name\": \"Bob\", \"scores\": [90, 85, 92]}}");
    REQUIRE(v.is_object());
    const auto& person = v["person"];
    REQUIRE(person.is_object());
    REQUIRE_EQ(person["name"].as_string(), std::string("Bob"));
    REQUIRE(person["scores"].is_array());
    REQUIRE_EQ(person["scores"].size(), std::size_t(3));
    REQUIRE_NEAR(person["scores"][1].as_number(), 85.0, 1e-9);
}

TEST_CASE("json: parse object with all value types") {
    auto v = json_parse(R"({
        "s": "hello",
        "n": 3.14,
        "b": true,
        "f": false,
        "null_val": null,
        "arr": [1, 2],
        "obj": {"k": "v"}
    })");
    REQUIRE(v["s"].is_string());
    REQUIRE(v["n"].is_number());
    REQUIRE(v["b"].is_bool());
    REQUIRE(v["f"].is_bool());
    REQUIRE(v["null_val"].is_null());
    REQUIRE(v["arr"].is_array());
    REQUIRE(v["obj"].is_object());
}

TEST_CASE("json: deeply nested structure") {
    auto v = json_parse(R"({"a": {"b": {"c": {"d": 42}}}})");
    REQUIRE_NEAR(v["a"]["b"]["c"]["d"].as_number(), 42.0, 1e-9);
}

// ===========================================================================
// Whitespace handling
// ===========================================================================

TEST_CASE("json: leading whitespace") {
    auto v = json_parse("   42");
    REQUIRE_NEAR(v.as_number(), 42.0, 1e-9);
}

TEST_CASE("json: trailing whitespace") {
    auto v = json_parse("42   ");
    REQUIRE_NEAR(v.as_number(), 42.0, 1e-9);
}

TEST_CASE("json: multiline whitespace") {
    auto v = json_parse("{\n  \"a\" :\t1 ,\n  \"b\" :\r\n  2\n}");
    REQUIRE_NEAR(v["a"].as_number(), 1.0, 1e-9);
    REQUIRE_NEAR(v["b"].as_number(), 2.0, 1e-9);
}

// ===========================================================================
// Value access and helpers
// ===========================================================================

TEST_CASE("json: as_int conversion") {
    auto v = json_parse("42");
    REQUIRE_EQ(v.as_int(), std::int64_t(42));
    REQUIRE_EQ(v.as_int<int>(), 42);
    REQUIRE_EQ(v.as_int<unsigned>(), 42u);
}

TEST_CASE("json: operator[] for object") {
    auto v = json_parse("{\"key\": \"value\"}");
    REQUIRE_EQ(v["key"].as_string(), std::string("value"));
}

TEST_CASE("json: operator[] for array") {
    auto v = json_parse("[10, 20, 30]");
    REQUIRE_NEAR(v[std::size_t(1)].as_number(), 20.0, 1e-9);
}

TEST_CASE("json: operator[] out of range throws") {
    auto v = json_parse("[1, 2]");
    REQUIRE_THROWS(v[std::size_t(5)]);
}

TEST_CASE("json: operator[] missing key throws") {
    auto v = json_parse("{\"a\": 1}");
    REQUIRE_THROWS(v["missing"]);
}

TEST_CASE("json: find existing key") {
    auto v = json_parse("{\"a\": 1, \"b\": 2}");
    auto* p = v.find("a");
    REQUIRE(p != nullptr);
    REQUIRE_NEAR(p->as_number(), 1.0, 1e-9);
}

TEST_CASE("json: find missing key returns nullptr") {
    auto v = json_parse("{\"a\": 1}");
    REQUIRE(v.find("missing") == nullptr);
}

TEST_CASE("json: find on non-object returns nullptr") {
    auto v = json_parse("[1, 2]");
    REQUIRE(v.find("key") == nullptr);
}

TEST_CASE("json: get with fallback - key exists") {
    auto v = json_parse("{\"a\": 1}");
    JsonValue fallback(99);
    REQUIRE_NEAR(v.get("a", fallback).as_number(), 1.0, 1e-9);
}

TEST_CASE("json: get with fallback - key missing") {
    auto v = json_parse("{\"a\": 1}");
    JsonValue fallback(99);
    REQUIRE_NEAR(v.get("missing", fallback).as_number(), 99.0, 1e-9);
}

TEST_CASE("json: size of array") {
    auto v = json_parse("[1, 2, 3, 4]");
    REQUIRE_EQ(v.size(), std::size_t(4));
}

TEST_CASE("json: size of object") {
    auto v = json_parse("{\"a\": 1, \"b\": 2}");
    REQUIRE_EQ(v.size(), std::size_t(2));
}

TEST_CASE("json: size of non-container returns 0") {
    REQUIRE_EQ(json_parse("42").size(), std::size_t(0));
    REQUIRE_EQ(json_parse("\"hello\"").size(), std::size_t(0));
    REQUIRE_EQ(json_parse("true").size(), std::size_t(0));
    REQUIRE_EQ(json_parse("null").size(), std::size_t(0));
}

TEST_CASE("json: empty checks") {
    REQUIRE(json_parse("null").empty());
    REQUIRE(json_parse("[]").empty());
    REQUIRE(json_parse("{}").empty());
    REQUIRE(!json_parse("[1]").empty());
    REQUIRE(!json_parse("{\"a\":1}").empty());
}

TEST_CASE("json: mutable as_string") {
    auto v = json_parse("\"hello\"");
    v.as_string() = "world";
    REQUIRE_EQ(v.as_string(), std::string("world"));
}

TEST_CASE("json: mutable as_array") {
    auto v = json_parse("[1, 2]");
    v.as_array().push_back(JsonValue(3));
    REQUIRE_EQ(v.size(), std::size_t(3));
}

TEST_CASE("json: mutable as_object") {
    auto v = json_parse("{\"a\": 1}");
    v.as_object()["b"] = JsonValue(2);
    REQUIRE_EQ(v.size(), std::size_t(2));
}

// ===========================================================================
// Construction helpers
// ===========================================================================

TEST_CASE("json: default construction is null") {
    JsonValue v;
    REQUIRE(v.is_null());
}

TEST_CASE("json: construct from nullptr") {
    JsonValue v(nullptr);
    REQUIRE(v.is_null());
}

TEST_CASE("json: construct from bool") {
    JsonValue v(true);
    REQUIRE(v.is_bool());
    REQUIRE_EQ(v.as_bool(), true);
}

TEST_CASE("json: construct from double") {
    JsonValue v(3.14);
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 3.14, 1e-9);
}

TEST_CASE("json: construct from int") {
    JsonValue v(42);
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 42.0, 1e-9);
}

TEST_CASE("json: construct from int64") {
    JsonValue v(std::int64_t(100));
    REQUIRE(v.is_number());
}

TEST_CASE("json: construct from size_t") {
    JsonValue v(std::size_t(99));
    REQUIRE(v.is_number());
}

TEST_CASE("json: construct from const char*") {
    JsonValue v("hello");
    REQUIRE(v.is_string());
    REQUIRE_EQ(v.as_string(), std::string("hello"));
}

TEST_CASE("json: construct from std::string") {
    JsonValue v(std::string("world"));
    REQUIRE(v.is_string());
}

TEST_CASE("json: construct from string_view") {
    std::string_view sv = "test";
    JsonValue v(sv);
    REQUIRE(v.is_string());
    REQUIRE_EQ(v.as_string(), std::string("test"));
}

TEST_CASE("json: json_object builder") {
    auto v = json_object({{"name", "Alice"}, {"age", 30}});
    REQUIRE(v.is_object());
    REQUIRE_EQ(v["name"].as_string(), std::string("Alice"));
    REQUIRE_NEAR(v["age"].as_number(), 30.0, 1e-9);
}

TEST_CASE("json: json_array builder") {
    auto v = json_array({1, "two", true, nullptr});
    REQUIRE(v.is_array());
    REQUIRE_EQ(v.size(), std::size_t(4));
    REQUIRE(v[0].is_number());
    REQUIRE(v[1].is_string());
    REQUIRE(v[2].is_bool());
    REQUIRE(v[3].is_null());
}

// ===========================================================================
// Serialization
// ===========================================================================

TEST_CASE("json: serialize null") {
    REQUIRE_EQ(json_serialize(JsonValue(nullptr)), std::string("null"));
}

TEST_CASE("json: serialize true") {
    REQUIRE_EQ(json_serialize(JsonValue(true)), std::string("true"));
}

TEST_CASE("json: serialize false") {
    REQUIRE_EQ(json_serialize(JsonValue(false)), std::string("false"));
}

TEST_CASE("json: serialize integer number") {
    // Integers within range should serialize without decimals
    REQUIRE_EQ(json_serialize(JsonValue(42)), std::string("42"));
    REQUIRE_EQ(json_serialize(JsonValue(0)), std::string("0"));
    REQUIRE_EQ(json_serialize(JsonValue(-7)), std::string("-7"));
}

TEST_CASE("json: serialize float number") {
    auto s = json_serialize(JsonValue(3.14));
    // Re-parse to check round-trip accuracy
    auto v = json_parse(s);
    REQUIRE_NEAR(v.as_number(), 3.14, 1e-9);
}

TEST_CASE("json: serialize simple string") {
    REQUIRE_EQ(json_serialize(JsonValue("hello")), std::string("\"hello\""));
}

TEST_CASE("json: serialize string with escapes") {
    JsonValue v(std::string("a\"b\\c\n\t"));
    auto s = json_serialize(v);
    REQUIRE_EQ(s, std::string("\"a\\\"b\\\\c\\n\\t\""));
}

TEST_CASE("json: serialize string with control characters") {
    // Control char (e.g. 0x01) should serialize as \u0001
    std::string input;
    input += static_cast<char>(0x01);
    auto s = json_serialize(JsonValue(input));
    REQUIRE_EQ(s, std::string("\"\\u0001\""));
}

TEST_CASE("json: serialize empty array") {
    REQUIRE_EQ(json_serialize(JsonValue(JsonArray{})), std::string("[]"));
}

TEST_CASE("json: serialize empty object") {
    REQUIRE_EQ(json_serialize(JsonValue(JsonObject{})), std::string("{}"));
}

TEST_CASE("json: serialize array compact") {
    auto v = json_array({1, 2, 3});
    auto s = json_serialize(v, 0);
    REQUIRE_EQ(s, std::string("[1,2,3]"));
}

TEST_CASE("json: serialize object compact") {
    auto v = json_object({{"a", 1}, {"b", 2}});
    auto s = json_serialize(v, 0);
    // Keys are sorted for deterministic output
    REQUIRE_EQ(s, std::string("{\"a\":1,\"b\":2}"));
}

TEST_CASE("json: serialize pretty-printed") {
    auto v = json_object({{"a", 1}});
    auto s = json_serialize(v, 2);
    // Should contain newlines and indentation
    REQUIRE(s.find('\n') != std::string::npos);
    REQUIRE(s.find("  ") != std::string::npos);
}

TEST_CASE("json: serialize object keys sorted") {
    auto v = json_object({{"zebra", 1}, {"apple", 2}, {"mango", 3}});
    auto s = json_serialize(v, 0);
    // Keys should appear alphabetically
    auto pos_a = s.find("\"apple\"");
    auto pos_m = s.find("\"mango\"");
    auto pos_z = s.find("\"zebra\"");
    REQUIRE(pos_a < pos_m);
    REQUIRE(pos_m < pos_z);
}

// ===========================================================================
// Round-trip: parse -> serialize -> parse
// ===========================================================================

TEST_CASE("json: round-trip simple object") {
    std::string input = R"({"age":30,"name":"Alice"})";
    auto v1 = json_parse(input);
    auto serialized = json_serialize(v1, 0);
    auto v2 = json_parse(serialized);
    REQUIRE_EQ(v2["name"].as_string(), std::string("Alice"));
    REQUIRE_NEAR(v2["age"].as_number(), 30.0, 1e-9);
}

TEST_CASE("json: round-trip nested structure") {
    auto v1 = json_object({
        {"data", json_array({1, 2, 3})},
        {"nested", json_object({{"key", "value"}})}
    });
    auto serialized = json_serialize(v1, 0);
    auto v2 = json_parse(serialized);
    REQUIRE_EQ(v2["data"].size(), std::size_t(3));
    REQUIRE_EQ(v2["nested"]["key"].as_string(), std::string("value"));
}

TEST_CASE("json: round-trip string escapes") {
    JsonValue v1(std::string("line1\nline2\ttab\"quote\\backslash"));
    auto serialized = json_serialize(v1);
    auto v2 = json_parse(serialized);
    REQUIRE_EQ(v2.as_string(), v1.as_string());
}

TEST_CASE("json: round-trip all types") {
    auto v1 = json_array({nullptr, true, false, 42, 3.14, "hello"});
    auto serialized = json_serialize(v1, 0);
    auto v2 = json_parse(serialized);
    REQUIRE(v2[0].is_null());
    REQUIRE_EQ(v2[1].as_bool(), true);
    REQUIRE_EQ(v2[2].as_bool(), false);
    REQUIRE_NEAR(v2[3].as_number(), 42.0, 1e-9);
    REQUIRE_NEAR(v2[4].as_number(), 3.14, 1e-9);
    REQUIRE_EQ(v2[5].as_string(), std::string("hello"));
}

TEST_CASE("json: round-trip pretty print") {
    auto v1 = json_object({{"a", json_array({1, 2})}, {"b", "text"}});
    auto pretty = json_serialize(v1, 4);
    auto v2 = json_parse(pretty);
    REQUIRE_EQ(v2["a"].size(), std::size_t(2));
    REQUIRE_EQ(v2["b"].as_string(), std::string("text"));
}

// ===========================================================================
// Error handling
// ===========================================================================

TEST_CASE("json: error - empty input") {
    REQUIRE_THROWS(json_parse(""));
}

TEST_CASE("json: error - whitespace only") {
    REQUIRE_THROWS(json_parse("   "));
}

TEST_CASE("json: error - unterminated string") {
    REQUIRE_THROWS(json_parse("\"hello"));
}

TEST_CASE("json: error - unterminated object") {
    REQUIRE_THROWS(json_parse("{\"a\": 1"));
}

TEST_CASE("json: error - unterminated array") {
    REQUIRE_THROWS(json_parse("[1, 2"));
}

TEST_CASE("json: error - missing colon in object") {
    REQUIRE_THROWS(json_parse("{\"a\" 1}"));
}

TEST_CASE("json: error - bad number") {
    REQUIRE_THROWS(json_parse("abc"));
}

TEST_CASE("json: error - incomplete unicode escape") {
    REQUIRE_THROWS(json_parse("\"\\u00\""));
}

TEST_CASE("json: error - invalid unicode hex digit") {
    REQUIRE_THROWS(json_parse("\"\\u00GG\""));
}

TEST_CASE("json: error - escape at end of string") {
    REQUIRE_THROWS(json_parse("\"\\"));
}

TEST_CASE("json: error - wrong type access throws") {
    auto v = json_parse("42");
    REQUIRE_THROWS(v.as_string());
    REQUIRE_THROWS(v.as_bool());
    REQUIRE_THROWS(v.as_array());
    REQUIRE_THROWS(v.as_object());
}

TEST_CASE("json: error - string as_number throws") {
    auto v = json_parse("\"hello\"");
    REQUIRE_THROWS(v.as_number());
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST_CASE("json: large integer") {
    auto v = json_parse("9999999999999");
    REQUIRE(v.is_number());
    REQUIRE_NEAR(v.as_number(), 9999999999999.0, 1.0);
}

TEST_CASE("json: very small float") {
    auto v = json_parse("1e-300");
    REQUIRE(v.is_number());
    REQUIRE(v.as_number() > 0.0);
    REQUIRE(v.as_number() < 1e-299);
}

TEST_CASE("json: object with duplicate keys (last wins)") {
    // JSON spec says implementations vary; unordered_map will keep last inserted
    auto v = json_parse("{\"a\": 1, \"a\": 2}");
    REQUIRE(v.is_object());
    // The parser uses operator[] which overwrites, so last value should win
    REQUIRE_NEAR(v["a"].as_number(), 2.0, 1e-9);
}

TEST_CASE("json: array with trailing comma (lenient)") {
    // The parser is lenient about trailing commas
    auto v = json_parse("[1, 2, 3,]");
    REQUIRE(v.is_array());
    REQUIRE_EQ(v.size(), std::size_t(3));
}

TEST_CASE("json: object with trailing comma (lenient)") {
    auto v = json_parse("{\"a\": 1,}");
    REQUIRE(v.is_object());
    REQUIRE_EQ(v.size(), std::size_t(1));
}

TEST_CASE("json: string with unicode null \\u0000") {
    auto v = json_parse("\"a\\u0000b\"");
    // The null char should be embedded in the string
    REQUIRE_EQ(v.as_string().size(), std::size_t(3));
}

TEST_CASE("json: complex realistic JSON") {
    auto v = json_parse(R"({
        "name": "utils2",
        "version": "1.0.0",
        "dependencies": [],
        "config": {
            "debug": false,
            "threads": 8,
            "ratio": 0.75,
            "description": "A \"quoted\" value with\nnewlines"
        },
        "tags": ["c++", "json", "parser"],
        "metadata": null
    })");
    REQUIRE(v.is_object());
    REQUIRE_EQ(v["name"].as_string(), std::string("utils2"));
    REQUIRE(v["dependencies"].is_array());
    REQUIRE(v["dependencies"].empty());
    REQUIRE_EQ(v["config"]["debug"].as_bool(), false);
    REQUIRE_NEAR(v["config"]["threads"].as_number(), 8.0, 1e-9);
    REQUIRE_NEAR(v["config"]["ratio"].as_number(), 0.75, 1e-9);
    REQUIRE(v["config"]["description"].as_string().find("\"quoted\"") != std::string::npos);
    REQUIRE_EQ(v["tags"].size(), std::size_t(3));
    REQUIRE(v["metadata"].is_null());
}

UTILS_TEST_MAIN()
