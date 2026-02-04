// Composite and lighting settings for CVolumeViewer, extracted for parallel compilation.

#include "CVolumeViewer.hpp"

#include <algorithm>
#include <unordered_set>

void CVolumeViewer::setCompositeEnabled(bool enabled)
{
    if (_composite_enabled != enabled) {
        _composite_enabled = enabled;
        renderVisible(true);
        updateStatusLabel();
    }
}

void CVolumeViewer::setCompositeLayersInFront(int layers)
{
    if (layers >= 0 && layers <= 100 && layers != _composite_layers_front) {
        _composite_layers_front = layers;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeLayersBehind(int layers)
{
    if (layers >= 0 && layers <= 100 && layers != _composite_layers_behind) {
        _composite_layers_behind = layers;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaMin(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_alpha_min) {
        _composite_alpha_min = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaMax(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_alpha_max) {
        _composite_alpha_max = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeAlphaThreshold(int value)
{
    if (value >= 0 && value <= 10000 && value != _composite_alpha_threshold) {
        _composite_alpha_threshold = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeMaterial(int value)
{
    if (value >= 0 && value <= 255 && value != _composite_material) {
        _composite_material = value;
        if (_composite_enabled && _composite_method == "alpha") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeReverseDirection(bool reverse)
{
    if (reverse != _composite_reverse_direction) {
        _composite_reverse_direction = reverse;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeBLExtinction(float value)
{
    if (value != _composite_bl_extinction) {
        _composite_bl_extinction = value;
        if (_composite_enabled && _composite_method == "beerLambert") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeBLEmission(float value)
{
    if (value != _composite_bl_emission) {
        _composite_bl_emission = value;
        if (_composite_enabled && _composite_method == "beerLambert") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeBLAmbient(float value)
{
    if (value != _composite_bl_ambient) {
        _composite_bl_ambient = value;
        if (_composite_enabled && _composite_method == "beerLambert") {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setLightingEnabled(bool enabled)
{
    if (enabled != _lighting_enabled) {
        _lighting_enabled = enabled;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setLightAzimuth(float degrees)
{
    if (degrees != _light_azimuth) {
        _light_azimuth = degrees;
        if (_composite_enabled && _lighting_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setLightElevation(float degrees)
{
    if (degrees != _light_elevation) {
        _light_elevation = degrees;
        if (_composite_enabled && _lighting_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setLightDiffuse(float value)
{
    if (value != _light_diffuse) {
        _light_diffuse = value;
        if (_composite_enabled && _lighting_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setLightAmbient(float value)
{
    if (value != _light_ambient) {
        _light_ambient = value;
        if (_composite_enabled && _lighting_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setUseVolumeGradients(bool enabled)
{
    if (enabled != _use_volume_gradients) {
        _use_volume_gradients = enabled;
        // Don't invalidate cache - gradients are still valid, just not being used
        if (_composite_enabled && _lighting_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setIsoCutoff(int value)
{
    value = std::clamp(value, 0, 255);
    if (value != _iso_cutoff) {
        _iso_cutoff = value;
        renderVisible(true);
    }
}

void CVolumeViewer::setPostStretchValues(bool enabled)
{
    if (enabled != _postStretchValues) {
        _postStretchValues = enabled;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setPostRemoveSmallComponents(bool enabled)
{
    if (enabled != _postRemoveSmallComponents) {
        _postRemoveSmallComponents = enabled;
        if (_composite_enabled) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setPostMinComponentSize(int size)
{
    size = std::clamp(size, 1, 100000);
    if (size != _postMinComponentSize) {
        _postMinComponentSize = size;
        if (_composite_enabled && _postRemoveSmallComponents) {
            renderVisible(true);
        }
    }
}

void CVolumeViewer::setCompositeMethod(const std::string& method)
{
    // Validate method is one of the supported methods
    static const std::unordered_set<std::string> validMethods = {
        "max", "mean", "min", "alpha", "beerLambert"
    };

    if (method != _composite_method && validMethods.count(method) > 0) {
        _composite_method = method;
        if (_composite_enabled) {
            renderVisible(true);
            updateStatusLabel();
        }
    }
}
