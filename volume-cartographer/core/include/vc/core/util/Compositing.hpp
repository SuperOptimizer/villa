#pragma once

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <numbers>

// Compositing parameters. ALL compositing is done by matter-compressor
// (mc_render); these fields map 1:1 onto mc_render_params. The mc reduction modes
// are: min, mean, max, alpha, stddev, shaded, percentile, depth. "shaded" is mc's
// emission-absorption relief render (gradient-normal lighting + curvature + raking
// via the light direction) -- it subsumes VC3D's old beerLambert/lighting/raking/
// volume-gradient C++ passes, which are deleted.
struct CompositeParams {
    // mc reduction method along the normal.
    std::string method = "mean";

    // ALPHA / SHADED: value threshold + per-sample opacity (mc alpha_min/opacity).
    float alphaMin = 0.0f;
    float alphaOpacity = 1.0f;

    // MC_COMP_PERCENTILE rank in (0,1].
    float percentile = 0.9f;

    // MC_COMP_SHADED knobs (ignored by other modes) -> mc_render_params.
    // light = direction toward the light (z,y,x); tilt for raking relief.
    float lightZ = 0.0f, lightY = 0.0f, lightX = 0.0f;
    float ambient = 0.25f;
    float diffuse = 0.75f;
    float specular = 0.20f;
    float shininess = 24.0f;
    float absorption = 1.0f;
    float shadow = 0.0f;
    float sss = 0.0f;
    float curvature = 0.0f;

    bool operator==(const CompositeParams&) const = default;
};

// Consolidated rendering settings for composite mode (Qt-free). The layer counts
// become the mc [t0,t1] slab; everything else is in CompositeParams.
struct CompositeRenderSettings {
    bool enabled = false;
    int layersFront = 8;
    int layersBehind = 0;
    bool reverseDirection = false;

    bool planeEnabled = false;
    int planeLayersFront = 4;
    int planeLayersBehind = 4;

    CompositeParams params;

    bool operator==(const CompositeRenderSettings&) const = default;
};
