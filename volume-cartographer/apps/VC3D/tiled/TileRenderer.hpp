#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

#include "TiledViewerCamera.hpp"
#include "TileScene.hpp"
#include "vc/core/util/Compositing.hpp"

class Surface;
class PlaneSurface;
class QuadSurface;
class Volume;
namespace z5 { class Dataset; }

// Parameters for a single tile render call.
// Collected on the main thread, passed to the renderer.
struct TileRenderParams {
    WorldTileKey worldKey;
    bool isPlaneSurface = false;
    uint64_t epoch = 0;

    // Surface parameter space ROI for this tile (same for both surface types)
    cv::Rect2f surfaceROI;

    // Tile dimensions in pixels
    int tileW = 0;
    int tileH = 0;

    // Camera state snapshot
    float scale = 1.0f;
    float dsScale = 1.0f;
    int dsScaleIdx = 0;
    float zOff = 0.0f;

    // Rendering parameters
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    bool stretchValues = false;
    std::string colormapId;
    bool useFastInterpolation = false;
    CompositeRenderSettings compositeSettings;
    uint8_t isoCutoff = 0;
};

// Result from rendering a single tile
struct TileRenderResult {
    WorldTileKey worldKey;
    bool isPlaneSurface = false;
    cv::Mat image;       // BGR cv::Mat
    uint64_t epoch = 0;

    // Camera state snapshot for cache key reconstruction
    float scale = 1.0f;
    float zOff = 0.0f;
    int dsScaleIdx = 0;

    // Actual pyramid level used (may differ from requested if best-effort)
    int actualLevel = 0;
};

// Stateless tile renderer. Thread-safe (no Qt objects, no mutable state).
// Extracted from CVolumeViewer::render_area() logic.
class TileRenderer
{
public:
    // Render a single tile synchronously.
    // All inputs passed by value/pointer - no shared mutable state.
    // Volume owns its cache; no external ChunkCache needed.
    static TileRenderResult renderTile(
        const TileRenderParams& params,
        const std::shared_ptr<Surface>& surface,
        Volume* volume);

private:
    // Generate view coordinates for the tile
    static void generateTileCoords(
        cv::Mat_<cv::Vec3f>& coords,
        const TileRenderParams& params,
        const std::shared_ptr<Surface>& surface);
};
