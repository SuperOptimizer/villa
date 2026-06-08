#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Sampling.hpp"

#include <opencv2/core/mat.hpp>

#include <cstdint>

namespace vc::render {

class ChunkedPlaneSampler {
public:
    struct Options {
        Options()
            : sampling(vc::Sampling::Nearest)
            , tileSize(32)
        {
        }
        Options(vc::Sampling sampling_, int tileSize_)
            : sampling(sampling_)
            , tileSize(tileSize_)
        {
        }

        vc::Sampling sampling;
        int tileSize;
        // Composite MAX early-out ceiling. The composite kernel maxes raw voxel
        // bytes; the viewer later windows the result through a LUT, so ANY voxel
        // >= the window's high end maps to the same (clamped) output pixel. Once the
        // running max reaches this, no later layer can change the result -> stop the
        // depth walk. 255 = no early-out (default; safe for any LUT). Set to the
        // active windowHigh (rounded up) to skip the rest of a column the instant it
        // saturates -- lossless, and on bright/dense material it skips most of 64
        // layers. Only consulted on the Composite path.
        int compositeSaturationValue = 255;
    };

    struct Stats {
        int coveredPixels = 0;
        int requestedChunks = 0;
        int errorChunks = 0;
        // tick/settle: the chunk keys this render missed (resident lookup returned
        // MissQueued). The viewer hands these to ChunkCache::requestChunks() so the
        // next tick fetches them. Empty for the non-tick callers.
        std::vector<ChunkKey> missedKeys;
    };

    // Samples one pyramid level into `out` for pixels not already marked in
    // `coverage`. Coordinates are logical level-0 XYZ voxel coordinates.
    static Stats samplePlaneLevel(IChunkedArray& array,
                                  int level,
                                  const cv::Vec3f& origin,
                                  const cv::Vec3f& vxStep,
                                  const cv::Vec3f& vyStep,
                                  cv::Mat_<uint8_t>& out,
                                  cv::Mat_<uint8_t>& coverage,
                                  const Options& options = Options());

    static Stats sampleCoordsLevel(IChunkedArray& array,
                                   int level,
                                   const cv::Mat_<cv::Vec3f>& coords,
                                   cv::Mat_<uint8_t>& out,
                                   cv::Mat_<uint8_t>& coverage,
                                   const Options& options = Options());

    // Fused max-composite: for each pixel, samples numLayers depths along the
    // surface normal (coord + normal*(layerStart+i)*layerStep, Nearest) and
    // writes the MAX over the covered samples. Folds the per-layer offset,
    // sampling and reduction into ONE pass so the chunk lookup + index math is
    // shared across the depths of a pixel (they almost always fall in the same
    // chunk) instead of repeated per layer. Output is identical to sampling each
    // layer separately and taking composite_max. A pixel with no qualifying
    // sample is left uncovered.
    static Stats sampleCoordsMaxComposite(IChunkedArray& array,
                                          int level,
                                          const cv::Mat_<cv::Vec3f>& coords,
                                          const cv::Mat_<cv::Vec3f>& normals,
                                          int layerStart,
                                          int numLayers,
                                          float layerStep,
                                          cv::Mat_<uint8_t>& out,
                                          cv::Mat_<uint8_t>& coverage,
                                          const Options& options = Options());

};

} // namespace vc::render
