#include "CSliceCollection.hpp"

#include "vc/core/util/Slicing.hpp"

using namespace ChaoVis;

void CSliceCollection::setSlice(const std::string &name, CoordGenerator* slice)
{
    _slices[name] = slice;
    sendSliceChanged(name, slice);
}

void CSliceCollection::setSegmentator(const std::string &name, ControlPointSegmentator* seg)
{
    _segmentators[name] = seg;
    sendSegmentatorChanged(name, seg);
}

void CSliceCollection::setPoi(const std::string &name, cv::Vec3f poi)
{
    _pois[name] = poi;
    sendPOIChanged(name, poi);
}

CoordGenerator* CSliceCollection::getSlice(const std::string &name)
{
    if (!_slices.count(name))
        return nullptr;
    return _slices[name];
}

ControlPointSegmentator* CSliceCollection::getSegmentator(const std::string &name)
{
    if (!_segmentators.count(name))
        return nullptr;
    return _segmentators[name];
}

cv::Vec3f CSliceCollection::getPoi(const std::string &name)
{
    if (!_pois.count(name))
        return {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::quiet_NaN()};
    return _pois[name];
}

std::vector<std::string> CSliceCollection::slices()
{
    std::vector<std::string> keys;
    for(auto &it : _slices)
        keys.push_back(it.first);
    
    return keys;
}

std::vector<std::string> CSliceCollection::segmentators()
{
    std::vector<std::string> keys;
    for(auto &it : _segmentators)
        keys.push_back(it.first);
    
    return keys;
}

std::vector<std::string> CSliceCollection::pois()
{
    std::vector<std::string> keys;
    for(auto &it : _pois)
        keys.push_back(it.first);

    return keys;
}