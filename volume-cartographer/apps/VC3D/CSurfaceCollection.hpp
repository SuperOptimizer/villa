#pragma once

#include <QObject>
#include <opencv2/core.hpp>

class CoordGenerator;
class ControlPointSegmentator;

namespace ChaoVis
{
    
struct POI
{
    cv::Vec3f p = {0,0,0};
    CoordGenerator *src = nullptr;
    cv::Vec3f n = {0,0,0};
};

//this class shall handle all the (gui) interactions for its stored objects but does not itself provide the gui
//slices: all the defined slices of all kinds
//Segmentators: segmentations and interactions with segments
//POIs : e.g. active constrol points or slicing focus points
class CSurfaceCollection : public QObject
{
    Q_OBJECT
    
public:
    void setSlice(const std::string &name, CoordGenerator*);
    void setSegmentator(const std::string &name, ControlPointSegmentator*);
    void setPOI(const std::string &name, POI *poi);
    CoordGenerator*slice(const std::string &name);
    ControlPointSegmentator* segmentator(const std::string &name);
    POI *poi(const std::string &name);
    std::vector<std::string> slices();
    std::vector<std::string> segmentators();
    std::vector<std::string> pois();
    
signals:
    void sendSliceChanged(std::string, CoordGenerator*);
    void sendSegmentatorChanged(std::string, ControlPointSegmentator*);
    void sendPOIChanged(std::string, POI*);
    
protected:
    bool _regular_pan = false;
    std::unordered_map<std::string, CoordGenerator*> _slices;
    std::unordered_map<std::string, ControlPointSegmentator*> _segmentators;
    std::unordered_map<std::string, POI*> _pois;
};

}
