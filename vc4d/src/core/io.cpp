#include "vc4d/io/io.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vc4d::io {

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

nlohmann::json load_json(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open " + path.string());
    return nlohmann::json::parse(f);
}

void save_json(const nlohmann::json& j, const std::filesystem::path& path) {
    std::ofstream f(path);
    f << j.dump(2);
}

// ---------------------------------------------------------------------------
// Surface I/O
// ---------------------------------------------------------------------------

nlohmann::json load_surface_meta(const std::filesystem::path& dir) {
    return load_json(dir / "meta.json");
}

QuadSurface load_surface(const std::filesystem::path& dir) {
    auto meta = load_surface_meta(dir);

    // Load the point grid from area_*.tif files
    auto points = load_tifxyz(dir / "pointset.tif");

    Vec2f scale{1.f, 1.f};
    if (meta.contains("scale")) {
        auto& s = meta["scale"];
        scale = {s[0].get<float>(), s[1].get<float>()};
    }

    QuadSurface surf(std::move(points), scale);
    if (meta.contains("uuid"))
        surf.id = meta["uuid"].get<std::string>();
    if (meta.contains("name"))
        surf.name = meta["name"].get<std::string>();

    return surf;
}

void save_surface(const QuadSurface& surf, const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);

    save_tifxyz(surf.points(), dir / "pointset.tif");

    nlohmann::json meta;
    meta["uuid"] = surf.id;
    meta["name"] = surf.name;
    meta["scale"] = {surf.scale().x, surf.scale().y};
    meta["rows"] = surf.rows();
    meta["cols"] = surf.cols();
    meta["valid_points"] = surf.count_valid_points();

    save_json(meta, dir / "meta.json");

    // Save channels
    for (const auto& name : surf.channel_names()) {
        // TODO: Save each channel as a TIFF
    }
}

// ---------------------------------------------------------------------------
// TIF XYZ (stub — needs TIFF library integration)
// ---------------------------------------------------------------------------

Grid<Vec3f> load_tifxyz(const std::filesystem::path& /*path*/) {
    // TODO: Implement TIFF reading.
    // This needs either Qt's QImage (limited TIFF support) or libtiff.
    // For vc4d we'll evaluate using Qt's image I/O where possible
    // and libtiff only for float TIFFs.
    return {};
}

void save_tifxyz(const Grid<Vec3f>& /*grid*/, const std::filesystem::path& /*path*/) {
    // TODO: Implement TIFF writing.
}

// ---------------------------------------------------------------------------
// OBJ I/O
// ---------------------------------------------------------------------------

ObjMesh load_obj(const std::filesystem::path& path) {
    ObjMesh mesh;
    std::ifstream f(path);
    if (!f.is_open()) return mesh;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            Vec3f v;
            ss >> v.x >> v.y >> v.z;
            mesh.vertices.push_back(v);
        } else if (prefix == "vn") {
            Vec3f n;
            ss >> n.x >> n.y >> n.z;
            mesh.normals.push_back(n);
        } else if (prefix == "vt") {
            Vec2f t;
            ss >> t.x >> t.y;
            mesh.texcoords.push_back(t);
        } else if (prefix == "f") {
            // Parse face — handle "v", "v/vt", "v/vt/vn", "v//vn" formats
            std::array<int, 3> tri{};
            for (int i = 0; i < 3; ++i) {
                std::string token;
                ss >> token;
                // Extract vertex index (before first '/')
                auto slash = token.find('/');
                auto idx_str = (slash != std::string::npos) ? token.substr(0, slash) : token;
                tri[i] = std::stoi(idx_str) - 1;  // OBJ is 1-indexed
            }
            mesh.triangles.push_back(tri);
        }
    }
    return mesh;
}

void save_obj(const ObjMesh& mesh, const std::filesystem::path& path) {
    std::ofstream f(path);

    for (const auto& v : mesh.vertices)
        f << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';

    for (const auto& n : mesh.normals)
        f << "vn " << n.x << ' ' << n.y << ' ' << n.z << '\n';

    for (const auto& t : mesh.texcoords)
        f << "vt " << t.x << ' ' << t.y << '\n';

    for (const auto& tri : mesh.triangles) {
        f << "f " << (tri[0]+1) << ' ' << (tri[1]+1) << ' ' << (tri[2]+1) << '\n';
    }
}

ObjMesh surface_to_mesh(const QuadSurface& surf) {
    ObjMesh mesh;

    // Map (row, col) -> vertex index
    Grid<int> idx_map(surf.rows(), surf.cols());
    int vi = 0;

    for (auto [r, c, p] : surf.points().valid_points()) {
        mesh.vertices.push_back(p);
        idx_map.set(r, c, vi++);
    }

    // Generate triangles from valid quads (two triangles per quad)
    for (auto [r, c, p00, p01, p10, p11] : const_cast<Grid<Vec3f>&>(const_cast<QuadSurface&>(surf).points()).valid_quads()) {
        int i00 = *idx_map(r, c);
        int i01 = *idx_map(r, c + 1);
        int i10 = *idx_map(r + 1, c);
        int i11 = *idx_map(r + 1, c + 1);

        mesh.triangles.push_back({i00, i01, i10});
        mesh.triangles.push_back({i01, i11, i10});
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// VolumePkg helpers
// ---------------------------------------------------------------------------

nlohmann::json load_volpkg_config(const std::filesystem::path& volpkg_dir) {
    return load_json(volpkg_dir / "config.json");
}

std::vector<std::string> list_volumes(const std::filesystem::path& volpkg_dir) {
    std::vector<std::string> ids;
    auto vol_dir = volpkg_dir / "volumes";
    if (!std::filesystem::exists(vol_dir)) return ids;

    for (const auto& entry : std::filesystem::directory_iterator(vol_dir)) {
        if (entry.is_directory())
            ids.push_back(entry.path().filename().string());
    }
    std::ranges::sort(ids);
    return ids;
}

std::vector<std::string> list_segmentations(const std::filesystem::path& volpkg_dir) {
    std::vector<std::string> ids;
    auto seg_dir = volpkg_dir / "paths";
    if (!std::filesystem::exists(seg_dir)) return ids;

    for (const auto& entry : std::filesystem::directory_iterator(seg_dir)) {
        if (entry.is_directory())
            ids.push_back(entry.path().filename().string());
    }
    std::ranges::sort(ids);
    return ids;
}

} // namespace vc4d::io
