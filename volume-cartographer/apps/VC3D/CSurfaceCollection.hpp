#pragma once

#include <QObject>
#include <opencv2/core.hpp>

class Surface;

namespace ChaoVis
{
    
struct POI
{
    cv::Vec3f p = {0,0,0};
    Surface *src = nullptr;
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
    void setSurface(const std::string &name, Surface*);
    void setPOI(const std::string &name, POI *poi);
    Surface *surface(const std::string &name);
    POI *poi(const std::string &name);
    std::vector<std::string> surfaces();
    std::vector<std::string> pois();
    
signals:
    void sendSurfaceChanged(std::string, Surface*);
    void sendPOIChanged(std::string, POI*);
    
protected:
    bool _regular_pan = false;
    std::unordered_map<std::string, Surface*> _surfs;
    std::unordered_map<std::string, POI*> _pois;
};

}
