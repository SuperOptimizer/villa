// LoadJson.hpp - JSON loading and validation utilities for VC
#pragma once

#include <nlohmann/json_fwd.hpp>
#include <filesystem>
#include <string>
#include <initializer_list>

namespace vc::json {

nlohmann::json load_json_file(const std::filesystem::path& path);

// ============ VALIDATION ============

/**
 * Ensure all required fields exist in a JSON object.
 * @param json The JSON object to validate
 * @param fields List of required field names
 * @param context Description for error messages (e.g., file path)
 * @throws std::runtime_error listing the first missing field
 */
void require_fields(
    const nlohmann::json& json,
    std::initializer_list<const char*> fields,
    const std::string& context);

/**
 * Ensure a field equals an expected string value.
 * @param json The JSON object
 * @param field Field name to check
 * @param expected Expected string value
 * @param context Description for error messages (e.g., file path)
 * @throws std::runtime_error if field missing or doesn't match expected value
 */
void require_type(
    const nlohmann::json& json,
    const char* field,
    const std::string& expected,
    const std::string& context);

// ============ SAFE ACCESS HELPERS ============

// Returns a number if present (float/int or string convertible), else def.
double number_or(const nlohmann::json* m, const char* key, double def);

// Returns a string if present and of string type, else def.
std::string string_or(const nlohmann::json* m, const char* key, const std::string& def);

// Returns tags object if present & object, else {} (by value).
nlohmann::json tags_or_empty(const nlohmann::json* m);

bool has_tag(const nlohmann::json* m, const char* tag);

// Ensure *p is non-null and an object {}.
void ensure_object(nlohmann::json*& p);

// Ensure *p is an object and (*p)["tags"] is an object; returns a ref.
nlohmann::json& ensure_tags(nlohmann::json*& p);

} // namespace vc::json
