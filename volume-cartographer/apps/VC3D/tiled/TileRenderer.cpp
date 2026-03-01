#include "TileRenderer.hpp"
#include "PostProcess.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"

#include <algorithm>
#include <cmath>

TileRenderResult TileRenderer::renderTile(
    const TileRenderParams& params,
    const std::shared_ptr<Surface>& surface,
    Volume* volume)
{
    TileRenderResult result;
    result.worldKey = params.worldKey;
    result.epoch = params.epoch;
    result.scale = params.scale;
    result.zOff = params.zOff;
    result.dsScaleIdx = params.dsScaleIdx;

    if (!surface || !volume) {
        return result;
    }

    if (!volume->zarrDataset(params.dsScaleIdx)) {
        return result;
    }

    // Generate coordinates for this tile
    cv::Mat_<cv::Vec3f> coords;
    generateTileCoords(coords, params, surface);

    if (coords.empty()) {
        return result;
    }

    // Quick reject: if the tile's world AABB doesn't intersect data bounds,
    // skip the entire sampling pass (avoids cache lookups and prefetch requests).
    const auto& db = volume->dataBounds();
    if (db.valid) {
        const cv::Vec3f& c00 = coords(0, 0);
        const cv::Vec3f& c01 = coords(0, coords.cols - 1);
        const cv::Vec3f& c10 = coords(coords.rows - 1, 0);
        const cv::Vec3f& c11 = coords(coords.rows - 1, coords.cols - 1);

        float tMinX = std::min({c00[0], c01[0], c10[0], c11[0]});
        float tMaxX = std::max({c00[0], c01[0], c10[0], c11[0]});
        float tMinY = std::min({c00[1], c01[1], c10[1], c11[1]});
        float tMaxY = std::max({c00[1], c01[1], c10[1], c11[1]});
        float tMinZ = std::min({c00[2], c01[2], c10[2], c11[2]});
        float tMaxZ = std::max({c00[2], c01[2], c10[2], c11[2]});

        // Conservative margin for interpolation + composite layers
        float margin = 2.0f;
        if (params.compositeSettings.enabled || params.compositeSettings.planeEnabled) {
            margin += static_cast<float>(std::max({
                params.compositeSettings.layersFront,
                params.compositeSettings.layersBehind,
                params.compositeSettings.planeLayersFront,
                params.compositeSettings.planeLayersBehind}));
        }

        if (tMaxX < db.minX - margin || tMinX > db.maxX + margin ||
            tMaxY < db.minY - margin || tMinY > db.maxY + margin ||
            tMaxZ < db.minZ - margin || tMinZ > db.maxZ + margin) {
            return result;
        }
    }

    // Sample volume data
    cv::Mat_<uint8_t> gray;

    // Check for composite rendering
    PlaneSurface* plane = dynamic_cast<PlaneSurface*>(surface.get());
    const bool useComposite = (params.compositeSettings.enabled &&
                               (params.compositeSettings.layersFront > 0 ||
                                params.compositeSettings.layersBehind > 0));
    const bool usePlaneComposite = (plane != nullptr &&
                                    params.compositeSettings.planeEnabled &&
                                    (params.compositeSettings.planeLayersFront > 0 ||
                                     params.compositeSettings.planeLayersBehind > 0));

    if (useComposite && !plane) {
        // QuadSurface composite: need normals
        cv::Mat_<cv::Vec3f> normals;

        surface->gen(&coords, &normals, cv::Size(params.tileW, params.tileH),
                     cv::Vec3f(0, 0, 0), params.scale,
                     {params.surfaceROI.x * params.scale,
                      params.surfaceROI.y * params.scale,
                      params.zOff});

        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.composite = params.compositeSettings.params;
        sp.zStart = params.compositeSettings.reverseDirection
                        ? -params.compositeSettings.layersBehind
                        : -params.compositeSettings.layersFront;
        sp.zEnd = params.compositeSettings.reverseDirection
                      ? params.compositeSettings.layersFront
                      : params.compositeSettings.layersBehind;

        result.actualLevel = volume->sampleCompositeBestEffort(
            gray, coords, normals, sp);
    } else if (usePlaneComposite) {
        cv::Vec3f planeNormal = plane->normal(cv::Vec3f(0, 0, 0));
        cv::Mat_<cv::Vec3f> normals(coords.size(), planeNormal);

        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.composite = params.compositeSettings.params;
        sp.zStart = params.compositeSettings.reverseDirection
                        ? -params.compositeSettings.planeLayersBehind
                        : -params.compositeSettings.planeLayersFront;
        sp.zEnd = params.compositeSettings.reverseDirection
                      ? params.compositeSettings.planeLayersFront
                      : params.compositeSettings.planeLayersBehind;

        result.actualLevel = volume->sampleCompositeBestEffort(
            gray, coords, normals, sp);
    } else {
        vc::SampleParams sp;
        sp.level = params.dsScaleIdx;
        sp.method = (params.useFastInterpolation || params.dsScaleIdx >= 3)
                        ? vc::Sampling::Nearest : vc::Sampling::Trilinear;

        result.actualLevel = volume->sampleBestEffort(gray, coords, sp);
    }

    // Post-process

    if (gray.empty()) {
        return result;
    }

    // Unified post-processing: produces QImage::Format_RGB32 directly,
    // bypassing all cvtColor conversions and RGB888→RGB32 expansion.
    PostProcessParams pp;
    pp.isoCutoff = params.compositeSettings.params.isoCutoff;
    pp.windowLow = params.windowLow;
    pp.windowHigh = params.windowHigh;
    pp.stretchValues = params.stretchValues;
    pp.colormapId = params.colormapId;
    pp.postStretchValues = params.compositeSettings.postStretchValues;
    pp.removeSmallComponents = params.compositeSettings.postRemoveSmallComponents;
    pp.minComponentSize = params.compositeSettings.postMinComponentSize;
    result.image = applyPostProcess(gray, pp);
    return result;
}

void TileRenderer::generateTileCoords(
    cv::Mat_<cv::Vec3f>& coords,
    const TileRenderParams& params,
    const std::shared_ptr<Surface>& surface)
{
    const cv::Size tileSize(params.tileW, params.tileH);

    // Both PlaneSurface and QuadSurface use the same gen() call:
    // ptr = (0,0,0), offset = (surfaceROI * scale, zOff)
    // For QuadSurface, this is equivalent to using surfacePtr with canvas-relative offsets
    // because surfacePtr cancels out: ptr + (ptr*scale + dx + px) / scale = ptr + ptr + (dx+px)/scale
    // Using ptr=0: (surfROI*scale + px) / scale = surfROI + px/scale (same surface param)
    surface->gen(&coords, nullptr, tileSize, cv::Vec3f(0, 0, 0), params.scale,
                 {params.surfaceROI.x * params.scale,
                  params.surfaceROI.y * params.scale,
                  params.zOff});
}

