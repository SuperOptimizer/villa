#include "vc/core/util/LoadJson.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace vc::json {

nlohmann::json load_json_file(const std::filesystem::path& path)
{
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("JSON file not found: " + path.string());
    }
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open JSON file: " + path.string());
    }
    try {
        return nlohmann::json::parse(file);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Invalid JSON in " + path.string() + ": " + e.what());
    }
}

void require_fields(
    const nlohmann::json& json,
    std::initializer_list<const char*> fields,
    const std::string& context)
{
    for (const char* field : fields) {
        if (!json.contains(field)) {
            throw std::runtime_error(context + " missing required field: " + field);
        }
    }
}

void require_type(
    const nlohmann::json& json,
    const char* field,
    const std::string& expected,
    const std::string& context)
{
    if (!json.contains(field)) {
        throw std::runtime_error(context + " missing required field: " + field);
    }
    std::string actual;
    try {
        actual = json[field].get<std::string>();
    } catch (const nlohmann::json::type_error&) {
        throw std::runtime_error(context + " field '" + std::string(field) + "' must be a string");
    }
    if (actual != expected) {
        throw std::runtime_error(context + " has type '" + actual + "', expected '" + expected + "'");
    }
}

double number_or(const nlohmann::json* m, const char* key, double def) {
    if (!m || !m->is_object()) return def;
    auto it = m->find(key);
    if (it == m->end()) return def;
    if (it->is_number_float())   return it->get<double>();
    if (it->is_number_integer()) return static_cast<double>(it->get<int64_t>());
    if (it->is_string()) {
        try { return std::stod(it->get<std::string>()); } catch (...) { return def; }
    }
    return def;
}

std::string string_or(const nlohmann::json* m, const char* key, const std::string& def) {
    if (!m || !m->is_object()) return def;
    auto it = m->find(key);
    if (it != m->end() && it->is_string()) return it->get<std::string>();
    return def;
}

nlohmann::json tags_or_empty(const nlohmann::json* m) {
    if (!m || !m->is_object()) return nlohmann::json::object();
    auto it = m->find("tags");
    if (it != m->end() && it->is_object()) return *it;
    return nlohmann::json::object();
}

bool has_tag(const nlohmann::json* m, const char* tag) {
    auto t = tags_or_empty(m);
    return t.contains(tag);
}

void ensure_object(nlohmann::json*& p) {
    if (!p) { p = new nlohmann::json(nlohmann::json::object()); return; }
    if (!p->is_object()) *p = nlohmann::json::object();
}

nlohmann::json& ensure_tags(nlohmann::json*& p) {
    ensure_object(p);
    nlohmann::json& root = *p;
    auto it = root.find("tags");
    if (it == root.end() || !it->is_object()) {
        root["tags"] = nlohmann::json::object();
    }
    return root["tags"];
}

} // namespace vc::json
