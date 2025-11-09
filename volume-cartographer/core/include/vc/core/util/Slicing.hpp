#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include <opencv2/core.hpp>

#include <vc/core/util/HashFunctions.hpp>

#include <shared_mutex>
#include <z5/dataset.hxx>

//TODO generation overrun
//TODO groupkey overrun
class ChunkCache
{
public:
    ChunkCache(size_t size) : _size(size) {};
    ~ChunkCache();
    
    //get key for a subvolume - should be uniqueley identified between all groups and volumes that use this cache.
    //for example by using path + group name
    int groupIdx(const std::string &name);
    
    //key should be unique for chunk and contain groupkey (groupkey sets highest 16bits of uint64_t)
    void put(const cv::Vec4i &key, xt::xarray<uint8_t> *ar);
    std::shared_ptr<xt::xarray<uint8_t>> get(const cv::Vec4i &key);
    void reset();
    bool has(const cv::Vec4i &idx);

    // 16-bit lane
    void put16(const cv::Vec4i &key, xt::xarray<uint16_t> *ar);
    std::shared_ptr<xt::xarray<uint16_t>> get16(const cv::Vec4i &key);
    bool has16(const cv::Vec4i &idx);

    std::shared_mutex mutex;
private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint8_t>>,vec4i_hash> _store;
    //store generation number
    std::unordered_map<cv::Vec4i,uint64_t,vec4i_hash> _gen_store;
    // 16-bit storage and generations
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<uint16_t>>,vec4i_hash> _store16;
    std::unordered_map<cv::Vec4i,uint64_t,vec4i_hash> _gen_store16;
    //store group keys
    std::unordered_map<std::string,int> _group_store;

    std::shared_mutex _mutex;
};

//NOTE depending on request this might load a lot (the whole array) into RAM
// void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache = nullptr, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache = nullptr, bool nearest_neighbor=false);
template <typename T>
void readArea3D(xt::xtensor<T,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache *cache) { throw std::runtime_error("missing implementation"); }
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache *cache);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i offset, z5::Dataset *ds, ChunkCache *cache);
cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);

cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p);
cv::Vec3d at_int(const cv::Mat_<cv::Vec3d> &points, cv::Vec2f p);
float at_int(const cv::Mat_<float> &points, cv::Vec2f p);
bool loc_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l);
bool loc_valid(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l);
bool loc_valid_xy(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l);
bool loc_valid_xy(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l);