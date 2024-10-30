#pragma once

#include <xtensor/xarray.hpp>
#include <opencv2/core.hpp>

#include <shared_mutex>

namespace z5
{
    class Dataset;
}

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
    std::shared_ptr<xt::xarray<uint8_t>> get(uint64_t key);
    void reset();
    bool has(uint64_t key);
    std::shared_mutex mutex;
private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;
    std::unordered_map<uint64_t,std::shared_ptr<xt::xarray<uint8_t>>> _store;
    //store generation number
    std::unordered_map<uint64_t,uint64_t> _gen_store;
    //store group keys
    std::unordered_map<std::string,uint64_t> _group_store;

    std::shared_mutex _mutex;
};

//NOTE depending on request this might load a lot (the whole array) into RAM
void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache *cache = nullptr);
cv::Mat_<cv::Vec3f> smooth_vc_segmentation(const cv::Mat_<cv::Vec3f> &points);
cv::Mat_<cv::Vec3f> vc_segmentation_calc_normals(const cv::Mat_<cv::Vec3f> &points);
void vc_segmentation_scales(cv::Mat_<cv::Vec3f> points, double &sx, double &sy);
cv::Vec3f grid_normal(const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &loc);
