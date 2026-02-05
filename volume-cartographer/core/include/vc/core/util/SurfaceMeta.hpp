#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <opencv2/core/matx.hpp>

struct SurfaceTagEntry {
    std::string date;
    std::string user;
    std::string source;  // Only used by partial_review
};

struct SurfaceTags {
    std::optional<SurfaceTagEntry> approved;
    std::optional<SurfaceTagEntry> defective;
    std::optional<SurfaceTagEntry> reviewed;
    std::optional<SurfaceTagEntry> revisit;
    std::optional<SurfaceTagEntry> inspect;
    std::optional<SurfaceTagEntry> partial_review;
};

struct SurfaceMeta {
    // Core identity (always set during save)
    std::string type = "seg";
    std::string uuid;
    std::string format = "tifxyz";
    cv::Vec2f scale = {1.0f, 1.0f};
    std::optional<std::array<cv::Vec3f, 2>> bbox;  // [0]=low, [1]=high

    // Standard optional fields
    std::string date_last_modified;
    std::string name;
    double area_vx2 = -1.0;
    double area_cm2 = -1.0;
    double area = -1.0;        // alternative area field (used by vc_tifxyz)
    double avg_cost = -1.0;
    int max_gen = -1;
    std::string volume;        // associated volume uuid
    std::string scroll_source;
    std::string source;        // tool that created this surface
    double elapsed_time_s = -1.0;

    // Seed info (set by tracer/growth)
    std::optional<cv::Vec3f> seed;
    std::string seed_surface_id;
    std::string seed_surface_name;
    std::optional<std::array<int, 2>> grid_offset;  // [col, row]

    // Approved segments used during growth
    std::vector<std::string> used_approved_segments;

    // Tags
    SurfaceTags tags;

    // Round-trip preservation: unknown top-level keys stored as raw JSON text
    std::map<std::string, std::string> extras;
};
