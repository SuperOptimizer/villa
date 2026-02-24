#pragma once

#include <cstdint>
#include <optional>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/util/Compositing.hpp"

namespace vc {

// Unified sampling parameters for Volume methods.
// Default-constructed gives level-0 trilinear with no composite/postprocess.
struct SampleParams {
    int level = 0;
    Sampling method = Sampling::Trilinear;

    // Compositing (optional). When set, caller must provide normals.
    std::optional<CompositeParams> composite;
    int zStart = 0;
    int zEnd = 0;

    // Post-processing (optional). Applied after sampling.
    // Grayscale pipeline only (no colormap — that lives in the app layer).
    struct PostProcess {
        float windowLow = 0.0f;
        float windowHigh = 255.0f;
        bool stretchValues = false;
        bool postStretchValues = false;
        bool removeSmallComponents = false;
        int minComponentSize = 50;
        uint8_t isoCutoff = 0;
    };
    std::optional<PostProcess> postProcess;
};

}  // namespace vc
