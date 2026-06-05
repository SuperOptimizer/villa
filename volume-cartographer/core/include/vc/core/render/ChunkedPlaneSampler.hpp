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
    // writes the MAX over the covered samples whose value >= isoCutoff. Folds
    // the per-layer offset, sampling and reduction into ONE pass so the chunk
    // lookup + index math is shared across the depths of a pixel (they almost
    // always fall in the same chunk) instead of repeated per layer. Output is
    // identical to sampling each layer separately and taking composite_max.
    // A pixel with no qualifying sample is left uncovered.
    static Stats sampleCoordsMaxComposite(IChunkedArray& array,
                                          int level,
                                          const cv::Mat_<cv::Vec3f>& coords,
                                          const cv::Mat_<cv::Vec3f>& normals,
                                          int layerStart,
                                          int numLayers,
                                          float layerStep,
                                          float isoCutoff,
                                          cv::Mat_<uint8_t>& out,
                                          cv::Mat_<uint8_t>& coverage,
                                          const Options& options = Options());

};

} // namespace vc::render
