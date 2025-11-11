#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>



CSurfaceCollection::~CSurfaceCollection()
{
    for (auto& pair : _surfs) {
        if (pair.second.owns && pair.second.ptr) {
            delete pair.second.ptr;
        }
    }

    for (auto& pair : _pois) {
        delete pair.second;
    }

    for (auto& pair : _intersections) {
        delete pair.second;
    }
}

void CSurfaceCollection::setSurface(const std::string &name, Surface* surf, bool noSignalSend, bool takeOwnership)
{
    auto it = _surfs.find(name);
    if (it != _surfs.end()) {
        if (it->second.owns && it->second.ptr && it->second.ptr != surf) {
            delete it->second.ptr;
        }
        it->second.ptr = surf;
        it->second.owns = takeOwnership;
    } else {
        _surfs[name] = {surf, takeOwnership};
    }
    if (!noSignalSend) {
        sendSurfaceChanged(name, surf);
    }
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

Surface* CSurfaceCollection::surface(const std::string &name)
{
    auto it = _surfs.find(name);
    if (it == _surfs.end())
        return nullptr;
    return it->second.ptr;
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.count(name))
        return nullptr;
    return _pois[name];
}

std::vector<Surface*> CSurfaceCollection::surfaces()
{
    std::vector<Surface*> surfaces;
    surfaces.reserve(_surfs.size());

    for(auto surface : _surfs) {
        surfaces.push_back(surface.second.ptr);
    } 

    return surfaces;
}

std::vector<POI*> CSurfaceCollection::pois()
{
    std::vector<POI*> pois;
    pois.reserve(_pois.size());

    for(auto poi : _pois) {
        pois.push_back(poi.second);  
    } 

    return pois;
}

std::vector<std::string> CSurfaceCollection::surfaceNames()
{
    std::vector<std::string> keys;
    for(auto &it : _surfs)
        keys.push_back(it.first);
    
    return keys;
}

std::vector<std::string> CSurfaceCollection::poiNames()
{
    std::vector<std::string> keys;
    for(auto &it : _pois)
        keys.push_back(it.first);

    return keys;
}

void CSurfaceCollection::setIntersection(const std::string &a, const std::string &b, Intersection *intersect)
{
    auto key = std::make_pair(a, b);
    if (_intersections.count(key)) {
        delete _intersections[key];  // Delete old before overwriting
    }
    _intersections[key] = intersect;
    sendIntersectionChanged(a, b, intersect);
}

Intersection *CSurfaceCollection::intersection(const std::string &a, const std::string &b)
{
    if (_intersections.count({a,b}))
        return _intersections[{a,b}];
        
    if (_intersections.count({b,a}))
        return _intersections[{b,a}];
    
    return nullptr;
}

std::vector<std::pair<std::string,std::string>> CSurfaceCollection::intersections(const std::string &a)
{
    std::vector<std::pair<std::string,std::string>> res;

    if (!a.size()) {
        for(auto item : _intersections)
            res.push_back(item.first);
    }
    else
        for(auto item : _intersections) {
            if (item.first.first == a)
                res.push_back(item.first);
            else if (item.first.second == a)
                res.push_back(item.first);
        }
    return res;
}

void CSurfaceCollection::rebuildSpatialIndex(const cv::Vec3f& volume_dimensions)
{
    _segment_indices.clear();
    _next_segment_index = 0;

    // Collect all QuadSurfaces
    std::vector<std::pair<std::string, QuadSurface*>> quads;
    for (auto& pair : _surfs) {
        QuadSurface* quad = dynamic_cast<QuadSurface*>(pair.second.ptr);
        if (quad) {
            quads.push_back({pair.first, quad});
        }
    }

    if (quads.empty()) {
        std::cout << "Spatial index rebuilt: indexed 0 segments" << std::endl;
        return;
    }

    // Use volume dimensions if provided, otherwise compute from segment bboxes
    cv::Vec3f global_min, global_max;
    float cell_size = 50.0f;

    if (volume_dimensions[0] > 0 && volume_dimensions[1] > 0 && volume_dimensions[2] > 0) {
        // Use volume dimensions to set grid bounds
        global_min = cv::Vec3f(0, 0, 0);
        global_max = volume_dimensions;
        std::cout << "Spatial index: using volume dimensions ("
                  << volume_dimensions[0] << "x" << volume_dimensions[1] << "x" << volume_dimensions[2] << ")" << std::endl;
    } else {
        // Fallback: compute bounds from segment bounding boxes
        global_min = cv::Vec3f(std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max(),
                               std::numeric_limits<float>::max());
        global_max = cv::Vec3f(std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::lowest());

        for (auto& [name, quad] : quads) {
            Rect3D bbox = quad->bbox();
            for (int i = 0; i < 3; i++) {
                global_min[i] = std::min(global_min[i], bbox.low[i]);
                global_max[i] = std::max(global_max[i], bbox.high[i]);
            }
        }

        // Add padding
        float padding = cell_size * 5;
        global_min = global_min - cv::Vec3f(padding, padding, padding);
        global_max = global_max + cv::Vec3f(padding, padding, padding);
        std::cout << "Spatial index: computed from segments" << std::endl;
    }

    _spatial_index = std::make_unique<MultiSurfaceIndex>(cell_size, global_min, global_max);

    // Add all segments to the index
    for (auto& [name, quad] : quads) {
        int idx = _next_segment_index++;
        _segment_indices[name] = idx;
        _spatial_index->addPatch(idx, quad);
    }

    std::cout << "Spatial index rebuilt: indexed " << quads.size() << " segments" << std::endl;
}

void CSurfaceCollection::updateSegmentInSpatialIndex(const std::string& name)
{
    if (!_spatial_index) {
        // Index doesn't exist yet, create it
        rebuildSpatialIndex();
        return;
    }

    QuadSurface* quad = dynamic_cast<QuadSurface*>(surface(name));
    if (!quad) {
        return;
    }

    // Check if this segment already has an index
    auto it = _segment_indices.find(name);
    if (it != _segment_indices.end()) {
        // Update existing segment
        _spatial_index->updatePatch(it->second, quad);
    } else {
        // Add new segment
        int idx = _next_segment_index++;
        _segment_indices[name] = idx;
        _spatial_index->addPatch(idx, quad);
    }
}

std::vector<std::string> CSurfaceCollection::getSegmentsInRegion(const cv::Vec3f& center, float radius) const
{
    if (!_spatial_index) {
        return {};
    }

    // Query spatial index for segments near this point
    std::vector<int> candidate_indices = _spatial_index->getCandidatePatches(center, radius);

    // Build reverse lookup cache if needed (only once)
    static std::vector<std::string> index_to_name;
    if (index_to_name.size() != _segment_indices.size()) {
        index_to_name.resize(_segment_indices.size());
        for (const auto& pair : _segment_indices) {
            if (pair.second < index_to_name.size()) {
                index_to_name[pair.second] = pair.first;
            }
        }
    }

    // Convert indices back to segment names using fast array lookup
    std::vector<std::string> result;
    result.reserve(candidate_indices.size());
    for (int idx : candidate_indices) {
        if (idx >= 0 && idx < index_to_name.size() && !index_to_name[idx].empty()) {
            result.push_back(index_to_name[idx]);
        }
    }

    return result;
}

std::vector<std::string> CSurfaceCollection::getSegmentsInBoundingBox(const cv::Vec3f& min_bound, const cv::Vec3f& max_bound) const
{
    if (!_spatial_index) {
        return {};
    }

    // Query spatial index for segments in this bounding box
    std::vector<int> candidate_indices = _spatial_index->getCandidatePatchesByRegion(min_bound, max_bound);

    // Build reverse lookup cache if needed (only once)
    static std::vector<std::string> index_to_name;
    if (index_to_name.size() != _segment_indices.size()) {
        index_to_name.resize(_segment_indices.size());
        for (const auto& pair : _segment_indices) {
            if (pair.second < index_to_name.size()) {
                index_to_name[pair.second] = pair.first;
            }
        }
    }

    // Convert indices back to segment names using fast array lookup
    std::vector<std::string> result;
    result.reserve(candidate_indices.size());
    for (int idx : candidate_indices) {
        if (idx >= 0 && idx < index_to_name.size() && !index_to_name[idx].empty()) {
            result.push_back(index_to_name[idx]);
        }
    }

    return result;
}

std::vector<std::string> CSurfaceCollection::getSegmentsInYZPlane(float y_min, float y_max, float z_min, float z_max) const
{
    if (!_spatial_index) {
        return {};
    }
    auto result = _spatial_index->getCandidatePatchesInYZPlane(y_min, y_max, z_min, z_max);

    // Build reverse lookup cache if needed (only once)
    static std::vector<std::string> index_to_name;
    if (index_to_name.size() != _segment_indices.size()) {
        index_to_name.resize(_segment_indices.size());
        for (const auto& pair : _segment_indices) {
            if (pair.second < index_to_name.size()) {
                index_to_name[pair.second] = pair.first;
            }
        }
    }

    // Convert indices to names
    std::vector<std::string> names;
    for (int idx : result) {
        if (idx >= 0 && idx < index_to_name.size() && !index_to_name[idx].empty()) {
            names.push_back(index_to_name[idx]);
        }
    }
    return names;
}

std::vector<std::string> CSurfaceCollection::getSegmentsInXZPlane(float x_min, float x_max, float z_min, float z_max) const
{
    if (!_spatial_index) {
        return {};
    }
    auto result = _spatial_index->getCandidatePatchesInXZPlane(x_min, x_max, z_min, z_max);

    // Build reverse lookup cache if needed (only once)
    static std::vector<std::string> index_to_name;
    if (index_to_name.size() != _segment_indices.size()) {
        index_to_name.resize(_segment_indices.size());
        for (const auto& pair : _segment_indices) {
            if (pair.second < index_to_name.size()) {
                index_to_name[pair.second] = pair.first;
            }
        }
    }

    // Convert indices to names
    std::vector<std::string> names;
    for (int idx : result) {
        if (idx >= 0 && idx < index_to_name.size() && !index_to_name[idx].empty()) {
            names.push_back(index_to_name[idx]);
        }
    }
    return names;
}

std::vector<std::string> CSurfaceCollection::getSegmentsInXYPlane(float x_min, float x_max, float y_min, float y_max) const
{
    if (!_spatial_index) {
        return {};
    }
    auto result = _spatial_index->getCandidatePatchesInXYPlane(x_min, x_max, y_min, y_max);

    // Build reverse lookup cache if needed (only once)
    static std::vector<std::string> index_to_name;
    if (index_to_name.size() != _segment_indices.size()) {
        index_to_name.resize(_segment_indices.size());
        for (const auto& pair : _segment_indices) {
            if (pair.second < index_to_name.size()) {
                index_to_name[pair.second] = pair.first;
            }
        }
    }

    // Convert indices to names
    std::vector<std::string> names;
    for (int idx : result) {
        if (idx >= 0 && idx < index_to_name.size() && !index_to_name[idx].empty()) {
            names.push_back(index_to_name[idx]);
        }
    }
    return names;
}
