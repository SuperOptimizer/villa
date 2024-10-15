#include "CSurfaceCollection.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;

void CSurfaceCollection::setSurface(const std::string &name, Surface* surf)
{
    _surfs[name] = surf;
    sendSurfaceChanged(name, surf);
}

void CSurfaceCollection::setPOI(const std::string &name, POI *poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

Surface* CSurfaceCollection::surface(const std::string &name)
{
    if (!_surfs.count(name))
        return nullptr;
    return _surfs[name];
}

POI *CSurfaceCollection::poi(const std::string &name)
{
    if (!_pois.count(name))
        return nullptr;
    return _pois[name];
}

std::vector<std::string> CSurfaceCollection::surfaces()
{
    std::vector<std::string> keys;
    for(auto &it : _surfs)
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
