#include "TileRenderer.hpp"
#include "PostProcess.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/Sampling.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

TileRenderResult TileRenderer::renderTile(
    const TileRenderParams& params,
    const std::shared_ptr<Surface>& surface,
    Volume* volume)
{
    TileRenderResult result;
    result.worldKey = params.worldKey;
    result.isPlaneSurface = params.isPlaneSurface;
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

        int z_start = params.compositeSettings.reverseDirection
                          ? -params.compositeSettings.layersBehind
                          : -params.compositeSettings.layersFront;
        int z_end = params.compositeSettings.reverseDirection
                        ? params.compositeSettings.layersFront
                        : params.compositeSettings.layersBehind;

        result.actualLevel = volume->read3dBestEffort(
            gray, coords, normals, z_start, z_end,
            params.compositeSettings.params, params.dsScaleIdx);
    } else if (usePlaneComposite) {
        int z_start = params.compositeSettings.reverseDirection
                          ? -params.compositeSettings.planeLayersBehind
                          : -params.compositeSettings.planeLayersFront;
        int z_end = params.compositeSettings.reverseDirection
                        ? params.compositeSettings.planeLayersFront
                        : params.compositeSettings.planeLayersBehind;

        cv::Vec3f planeNormal = plane->normal(cv::Vec3f(0, 0, 0));
        cv::Mat_<cv::Vec3f> normals(coords.size(), planeNormal);

        result.actualLevel = volume->read3dBestEffort(
            gray, coords, normals, z_start, z_end,
            params.compositeSettings.params, params.dsScaleIdx);
    } else {
        vc::Sampling method = params.useFastInterpolation
            ? vc::Sampling::Nearest : vc::Sampling::Trilinear;
        result.actualLevel = volume->read2dBestEffort(
            gray, coords, params.dsScaleIdx, method);
    }

    // Post-process

    if (gray.empty()) {
        return result;
    }

    // Unified post-processing
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

