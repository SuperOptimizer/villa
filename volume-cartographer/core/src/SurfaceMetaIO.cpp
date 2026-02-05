#include "vc/core/util/SurfaceMetaIO.hpp"
#include "vc/core/util/SurfaceMeta.hpp"
#include <nlohmann/json.hpp>
#include <set>

using json = nlohmann::json;

namespace {

// Known typed fields — these are NOT stored in extras
static const std::set<std::string> KNOWN_KEYS = {
    "type", "uuid", "format", "scale", "bbox",
    "date_last_modified", "name", "area_vx2", "area_cm2", "area",
    "avg_cost", "max_gen", "volume", "scroll_source", "source",
    "elapsed_time_s", "seed", "seed_surface_id", "seed_surface_name",
    "grid_offset", "used_approved_segments", "tags"
};

SurfaceTagEntry parseTagEntry(const json& j) {
    SurfaceTagEntry e;
    if (j.contains("date") && j["date"].is_string())
        e.date = j["date"].get<std::string>();
    if (j.contains("user") && j["user"].is_string())
        e.user = j["user"].get<std::string>();
    if (j.contains("source") && j["source"].is_string())
        e.source = j["source"].get<std::string>();
    return e;
}

json tagEntryToJson(const SurfaceTagEntry& e) {
    json j = json::object();
    if (!e.date.empty()) j["date"] = e.date;
    if (!e.user.empty()) j["user"] = e.user;
    if (!e.source.empty()) j["source"] = e.source;
    return j;
}

SurfaceTags parseTags(const json& j) {
    SurfaceTags t;
    if (!j.is_object()) return t;
    if (j.contains("approved"))       t.approved = parseTagEntry(j["approved"]);
    if (j.contains("defective"))      t.defective = parseTagEntry(j["defective"]);
    if (j.contains("reviewed"))       t.reviewed = parseTagEntry(j["reviewed"]);
    if (j.contains("revisit"))        t.revisit = parseTagEntry(j["revisit"]);
    if (j.contains("inspect"))        t.inspect = parseTagEntry(j["inspect"]);
    if (j.contains("partial_review")) t.partial_review = parseTagEntry(j["partial_review"]);
    return t;
}

json tagsToJson(const SurfaceTags& t) {
    json j = json::object();
    if (t.approved)       j["approved"] = tagEntryToJson(*t.approved);
    if (t.defective)      j["defective"] = tagEntryToJson(*t.defective);
    if (t.reviewed)       j["reviewed"] = tagEntryToJson(*t.reviewed);
    if (t.revisit)        j["revisit"] = tagEntryToJson(*t.revisit);
    if (t.inspect)        j["inspect"] = tagEntryToJson(*t.inspect);
    if (t.partial_review) j["partial_review"] = tagEntryToJson(*t.partial_review);
    return j;
}

}  // namespace

namespace vc::meta {

SurfaceMeta parseFromJson(const json& j) {
    SurfaceMeta m;

    // Core identity
    if (j.contains("type") && j["type"].is_string())
        m.type = j["type"].get<std::string>();
    if (j.contains("uuid") && j["uuid"].is_string())
        m.uuid = j["uuid"].get<std::string>();
    if (j.contains("format") && j["format"].is_string())
        m.format = j["format"].get<std::string>();
    if (j.contains("scale") && j["scale"].is_array() && j["scale"].size() >= 2)
        m.scale = {j["scale"][0].get<float>(), j["scale"][1].get<float>()};
    if (j.contains("bbox") && j["bbox"].is_array() && j["bbox"].size() >= 2) {
        cv::Vec3f low = {j["bbox"][0][0].get<float>(), j["bbox"][0][1].get<float>(), j["bbox"][0][2].get<float>()};
        cv::Vec3f high = {j["bbox"][1][0].get<float>(), j["bbox"][1][1].get<float>(), j["bbox"][1][2].get<float>()};
        m.bbox = {low, high};
    }

    // Standard metadata
    if (j.contains("date_last_modified") && j["date_last_modified"].is_string())
        m.date_last_modified = j["date_last_modified"].get<std::string>();
    if (j.contains("name") && j["name"].is_string())
        m.name = j["name"].get<std::string>();
    if (j.contains("area_vx2") && j["area_vx2"].is_number())
        m.area_vx2 = j["area_vx2"].get<double>();
    if (j.contains("area_cm2") && j["area_cm2"].is_number())
        m.area_cm2 = j["area_cm2"].get<double>();
    if (j.contains("area") && j["area"].is_number())
        m.area = j["area"].get<double>();
    if (j.contains("avg_cost") && j["avg_cost"].is_number())
        m.avg_cost = j["avg_cost"].get<double>();
    if (j.contains("max_gen") && j["max_gen"].is_number())
        m.max_gen = j["max_gen"].get<int>();
    if (j.contains("volume") && j["volume"].is_string())
        m.volume = j["volume"].get<std::string>();
    if (j.contains("scroll_source") && j["scroll_source"].is_string())
        m.scroll_source = j["scroll_source"].get<std::string>();
    if (j.contains("source") && j["source"].is_string())
        m.source = j["source"].get<std::string>();
    if (j.contains("elapsed_time_s") && j["elapsed_time_s"].is_number())
        m.elapsed_time_s = j["elapsed_time_s"].get<double>();

    // Seed info
    if (j.contains("seed") && j["seed"].is_array() && j["seed"].size() >= 3)
        m.seed = cv::Vec3f{j["seed"][0].get<float>(), j["seed"][1].get<float>(), j["seed"][2].get<float>()};
    if (j.contains("seed_surface_id") && j["seed_surface_id"].is_string())
        m.seed_surface_id = j["seed_surface_id"].get<std::string>();
    if (j.contains("seed_surface_name") && j["seed_surface_name"].is_string())
        m.seed_surface_name = j["seed_surface_name"].get<std::string>();
    if (j.contains("grid_offset") && j["grid_offset"].is_array() && j["grid_offset"].size() >= 2)
        m.grid_offset = std::array<int,2>{j["grid_offset"][0].get<int>(), j["grid_offset"][1].get<int>()};

    // Used approved segments
    if (j.contains("used_approved_segments") && j["used_approved_segments"].is_array()) {
        for (auto& el : j["used_approved_segments"])
            if (el.is_string())
                m.used_approved_segments.push_back(el.get<std::string>());
    }

    // Tags
    if (j.contains("tags") && j["tags"].is_object())
        m.tags = parseTags(j["tags"]);

    // Extras: everything not in KNOWN_KEYS
    if (j.is_object()) {
        for (auto& [key, val] : j.items()) {
            if (!KNOWN_KEYS.contains(key)) {
                m.extras[key] = val.dump();
            }
        }
    }

    return m;
}

json toJson(const SurfaceMeta& m) {
    json j = json::object();

    // Core identity
    j["type"] = m.type;
    if (!m.uuid.empty()) j["uuid"] = m.uuid;
    j["format"] = m.format;
    j["scale"] = {m.scale[0], m.scale[1]};
    if (m.bbox) {
        j["bbox"] = {{(*m.bbox)[0][0], (*m.bbox)[0][1], (*m.bbox)[0][2]},
                      {(*m.bbox)[1][0], (*m.bbox)[1][1], (*m.bbox)[1][2]}};
    }

    // Standard metadata (only write if set / non-default)
    if (!m.date_last_modified.empty()) j["date_last_modified"] = m.date_last_modified;
    if (!m.name.empty()) j["name"] = m.name;
    if (m.area_vx2 >= 0) j["area_vx2"] = m.area_vx2;
    if (m.area_cm2 >= 0) j["area_cm2"] = m.area_cm2;
    if (m.area >= 0) j["area"] = m.area;
    if (m.avg_cost >= 0) j["avg_cost"] = m.avg_cost;
    if (m.max_gen >= 0) j["max_gen"] = m.max_gen;
    if (!m.volume.empty()) j["volume"] = m.volume;
    if (!m.scroll_source.empty()) j["scroll_source"] = m.scroll_source;
    if (!m.source.empty()) j["source"] = m.source;
    if (m.elapsed_time_s >= 0) j["elapsed_time_s"] = m.elapsed_time_s;

    // Seed info
    if (m.seed)
        j["seed"] = {(*m.seed)[0], (*m.seed)[1], (*m.seed)[2]};
    if (!m.seed_surface_id.empty()) j["seed_surface_id"] = m.seed_surface_id;
    if (!m.seed_surface_name.empty()) j["seed_surface_name"] = m.seed_surface_name;
    if (m.grid_offset)
        j["grid_offset"] = {(*m.grid_offset)[0], (*m.grid_offset)[1]};

    // Used approved segments
    if (!m.used_approved_segments.empty())
        j["used_approved_segments"] = m.used_approved_segments;

    // Tags
    json t = tagsToJson(m.tags);
    if (!t.empty()) j["tags"] = t;

    // Extras: parse raw JSON text back into json objects
    for (auto& [key, val] : m.extras) {
        j[key] = json::parse(val);
    }

    return j;
}

}  // namespace vc::meta
