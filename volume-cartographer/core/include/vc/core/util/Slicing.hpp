#pragma once

#include <xtensor/xarray.hpp>
#include <opencv2/core.hpp>

namespace z5
{
    class Dataset;
}

//TODO replace coord-generator with surface!
class CoordGenerator
{
public:
    //given input volume shape, fill a coord slice
    void gen_coords(xt::xarray<float> &coords, int w, int h);
    void gen_coords(xt::xarray<float> &coords, const cv::Rect &roi, float render_scale = 1.0, float coord_scale = 1.0);
    virtual void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) = 0;
    // virtual void gen_normals(xt::xarray<float> &normals, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) = 0;
    virtual void setOffsetZ(float off) { _z_off = off; };
    virtual float offsetZ() { return _z_off; };
    // virtual cv::Vec3f offset() = 0;
    [[deprecated]]
    virtual cv::Vec3f normal_legacy(const cv::Vec3f &loc = {0,0,0}) = 0;
    [[deprecated]]
    virtual cv::Vec3f coord_legacy(const cv::Vec3f &loc = {0,0,0});
protected:
    float _z_off = 0;
};

//FIXME make normal private and keep normalized
class PlaneCoords : public CoordGenerator
{
public:
    PlaneCoords() {};
    PlaneCoords(cv::Vec3f origin_, cv::Vec3f normal_);
    virtual float pointDist(cv::Vec3f wp);
    virtual cv::Vec3f project(cv::Vec3f wp, float render_scale = 1.0, float coord_scale = 1.0);
    void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    // void gen_normals(xt::xarray<float> &normals, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    // float plane_mul() const;
    virtual float scalarp(cv::Vec3f point) const;
    virtual float height(cv::Vec3f point) const { return 0.0; };
    // virtual void gen_coords(float i, float j, int x, int y, float render_scale, float coord_scale) const = 0;
    using CoordGenerator::gen_coords;
    // cv::Vec3f offset() override { abort(); };
    void setNormal(cv::Vec3f normal);
    cv::Vec3f normal_legacy(const cv::Vec3f &loc = {0,0,0}) override { return _normal; };
    cv::Vec3f origin = {0,0,0};
protected:
    cv::Vec3f _normal = {0,0,1};
};

//FIXME make normal private and keep normalized
class GridCoords : public CoordGenerator
{
public:
    GridCoords() {};
    GridCoords(cv::Mat_<cv::Vec3f> &points, float sx = 1.0, float sy = 1.0, const cv::Vec3f &offset = {0,0,0}) : _points(points), _sx(sx), _sy(sy), _offset(offset) {};
    void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    // void gen_normals(xt::xarray<float> &normals, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    cv::Vec3f normal_legacy(const cv::Vec3f &loc = {0,0,0}) override;
    // cv::Vec3f offset() override;
    using CoordGenerator::gen_coords;
    cv::Mat_<cv::Vec3f> _points;
    float _sx = 1.0;
    float _sy = 1.0;
    cv::Vec3f _offset;
};

//TODO constrain to visible area? or add visiable area disaplay?
void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneCoords *plane, const cv::Rect &plane_roi, float step);

float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneCoords *plane, float init_step = 16.0, float min_step = 0.125);

//TODO generation overrun
//TODO groupkey overrun
class ChunkCache
{
public:
    ChunkCache(size_t size) : _size(size) {};
    ~ChunkCache();
    
    //get key for a subvolume - should be uniqueley identified between all groups and volumes that use this cache.
    //for example by using path + group name
    uint64_t groupKey(std::string name);
    
    //key should be unique for chunk and contain groupkey (groupkey sets highest 16bits of uint64_t)
    void put(uint64_t key, xt::xarray<uint8_t> *ar);
    xt::xarray<uint8_t> *get(uint64_t key);
    bool has(uint64_t key);
private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<uint64_t,xt::xarray<uint8_t>*> _store;
    //store generation number
    std::unordered_map<uint64_t,uint64_t> _gen_store;
    //store group keys
    std::unordered_map<std::string,uint64_t> _group_store;
};

//NOTE depending on request this might load a lot (the whole array) into RAM
void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache = nullptr);
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points);
cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);
