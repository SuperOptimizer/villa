#pragma once

#include <QObject>
#include <opencv2/core.hpp>

#include <vc/core/util/HashFunctions.hpp>

#include "vc/core/util/Surface.hpp"


struct POI
{
    cv::Vec3f p = {0,0,0};
    Surface *src = nullptr;
    cv::Vec3f n = {0,0,0};
};

struct Intersection
{
    std::vector<std::vector<cv::Vec3f>> lines;
};



// This class shall handle all the (GUI) interactions for its stored objects but does not itself provide the GUI
// Slices: all the defined slices of all kinds
// Segmentators: segmentations and interactions with segments
// POIs : e.g. active constrol points or slicing focus points
class CSurfaceCollection : public QObject
{
    Q_OBJECT
    
public:
    ~CSurfaceCollection();
    void setSurface(const std::string &name, Surface*, bool noSignalSend = false, bool takeOwnership = true);
    void setPOI(const std::string &name, POI *poi);
    void setIntersection(const std::string &a, const std::string &b, Intersection *intersect);
    Surface *surface(const std::string &name);
    Intersection *intersection(const std::string &a, const std::string &b);
    POI *poi(const std::string &name);
    std::vector<Surface*> surfaces();
    std::vector<POI*> pois();
    std::vector<std::string> surfaceNames();
    std::vector<std::string> poiNames();
    std::vector<std::pair<std::string,std::string>> intersections(const std::string &a = "");

    // Spatial index for accelerating intersection queries
    void rebuildSpatialIndex();
    void updateSegmentInSpatialIndex(const std::string& name);
    std::vector<std::string> getSegmentsInRegion(const cv::Vec3f& center, float radius) const;
    std::vector<std::string> getSegmentsInBoundingBox(const cv::Vec3f& min_bound, const cv::Vec3f& max_bound) const;
    // Plane-aware queries: filter by 2D coordinates only (much faster for axis-aligned planes)
    std::vector<std::string> getSegmentsInYZPlane(float y_min, float y_max, float z_min, float z_max) const;
    std::vector<std::string> getSegmentsInXZPlane(float x_min, float x_max, float z_min, float z_max) const;
    std::vector<std::string> getSegmentsInXYPlane(float x_min, float x_max, float y_min, float y_max) const;
    MultiSurfaceIndex* spatialIndex() { return _spatial_index.get(); }
    
signals:
    void sendSurfaceChanged(std::string, Surface*);
    void sendPOIChanged(std::string, POI*);
    void sendIntersectionChanged(std::string, std::string, Intersection*);
    
protected:
    struct SurfaceEntry {
        Surface* ptr = nullptr;
        bool owns = true;
    };

    bool _regular_pan = false;
    std::unordered_map<std::string, SurfaceEntry> _surfs;
    std::unordered_map<std::string, POI*> _pois;
    std::unordered_map<std::pair<std::string,std::string>, Intersection*, string_pair_hash> _intersections;

    // Spatial index mapping segment names to indices and accelerating spatial queries
    std::unique_ptr<MultiSurfaceIndex> _spatial_index;
    std::unordered_map<std::string, int> _segment_indices;  // Maps segment name -> index
    int _next_segment_index = 0;  // Next available index for new segments
};
