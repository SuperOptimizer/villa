#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;

void CSurfaceCollection::setSlice(const std::string &name, CoordGenerator* slice)
{
    _slices[name] = slice;
    sendSliceChanged(name, slice);
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

CoordGenerator* CSurfaceCollection::slice(const std::string &name)
{
    if (!_slices.count(name))
        return nullptr;
    return _slices[name];
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.count(name))
        return nullptr;
    return _pois[name];
}

std::vector<std::string> CSurfaceCollection::slices()
{
    std::vector<std::string> keys;
    for(auto &it : _slices)
        keys.push_back(it.first);
    
    return keys;
}

std::vector<std::string> CSurfaceCollection::pois()
{
    std::vector<std::string> keys;
    for(auto &it : _pois)
        keys.push_back(it.first);

    return keys;
}
