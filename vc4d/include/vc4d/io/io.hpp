#pragma once
/// vc4d I/O — Free functions for loading and saving data.
///
/// Key difference from vc3d: I/O is not mixed into data types.
/// vc3d's QuadSurface had save(), load(), ensureLoaded(), writeValidMask(),
/// writeDataToDirectory() etc. all as methods.  Here, data types are pure
/// value types and I/O is separate.
///
/// Supported formats:
///   • tifxyz  — TIFF-encoded XYZ point grids (vc3d's native surface format)
///   • OBJ     — Wavefront mesh (import/export)
///   • JSON    — metadata, volpkg config, zarr .zarray
///   • Zarr    — chunked volumetric data (see zarr.hpp)

#include "vc4d/core/grid.hpp"
#include "vc4d/core/math.hpp"
#include "vc4d/core/surface.hpp"

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace vc4d::io {

// ---------------------------------------------------------------------------
// Surface I/O
// ---------------------------------------------------------------------------

// Load a QuadSurface from a directory containing area_XXX.tif files and
// meta.json.  This is the vc3d/vc4d native format.
[[nodiscard]] QuadSurface load_surface(const std::filesystem::path& dir);

// Save a QuadSurface to a directory.
void save_surface(const QuadSurface& surf, const std::filesystem::path& dir);

// Load just the metadata (fast — doesn't read point data).
[[nodiscard]] nlohmann::json load_surface_meta(const std::filesystem::path& dir);

// ---------------------------------------------------------------------------
// TIF XYZ (3-channel TIFF encoding 3D coordinates)
// ---------------------------------------------------------------------------

// Load a 3-channel float TIFF as a Grid<Vec3f>.
// Invalid pixels (all zeros or all -1) become empty cells.
[[nodiscard]] Grid<Vec3f> load_tifxyz(const std::filesystem::path& path);

// Save a Grid<Vec3f> as a 3-channel float TIFF.
// Empty cells are written as (0, 0, 0).
void save_tifxyz(const Grid<Vec3f>& grid, const std::filesystem::path& path);

// ---------------------------------------------------------------------------
// OBJ mesh I/O
// ---------------------------------------------------------------------------

struct ObjMesh {
    std::vector<Vec3f> vertices;
    std::vector<std::array<int, 3>> triangles;  // vertex indices
    std::vector<Vec3f> normals;                  // per-vertex (optional)
    std::vector<Vec2f> texcoords;                // per-vertex (optional)
};

[[nodiscard]] ObjMesh load_obj(const std::filesystem::path& path);
void save_obj(const ObjMesh& mesh, const std::filesystem::path& path);

// Convert a QuadSurface to a triangle mesh.
[[nodiscard]] ObjMesh surface_to_mesh(const QuadSurface& surf);

// ---------------------------------------------------------------------------
// Volume package (volpkg) I/O
// ---------------------------------------------------------------------------

// Load volpkg config.json.
[[nodiscard]] nlohmann::json load_volpkg_config(const std::filesystem::path& volpkg_dir);

// List volume directories in a volpkg.
[[nodiscard]] std::vector<std::string> list_volumes(const std::filesystem::path& volpkg_dir);

// List segmentation directories in a volpkg.
[[nodiscard]] std::vector<std::string> list_segmentations(const std::filesystem::path& volpkg_dir);

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

[[nodiscard]] nlohmann::json load_json(const std::filesystem::path& path);
void save_json(const nlohmann::json& j, const std::filesystem::path& path);

} // namespace vc4d::io
