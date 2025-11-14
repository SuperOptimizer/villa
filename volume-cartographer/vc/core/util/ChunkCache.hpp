#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include <opencv2/core.hpp>

#include <vc/core/util/HashFunctions.hpp>

#include <shared_mutex>
#include <unordered_map>
#include <memory>
#include <string>

//TODO generation overrun
//TODO groupkey overrun
template<typename T>
class ChunkCache
{
public:
    ChunkCache(size_t size) : _size(size) {};
    ~ChunkCache();

    //get key for a subvolume - should be uniqueley identified between all groups and volumes that use this cache.
    //for example by using path + group name
    int groupIdx(const std::string &name);

    void put(const cv::Vec4i &key, xt::xarray<T> *ar);
    std::shared_ptr<xt::xarray<T>> get(const cv::Vec4i &key);
    bool has(const cv::Vec4i &idx);

    void reset();

    std::shared_mutex mutex;

private:
    uint64_t _generation = 0;
    size_t _size = 0;
    size_t _stored = 0;

    // Storage for type T
    std::unordered_map<cv::Vec4i,std::shared_ptr<xt::xarray<T>>,vec4i_hash> _store;
    std::unordered_map<cv::Vec4i,uint64_t,vec4i_hash> _gen_store;

    //store group keys
    std::unordered_map<std::string,int> _group_store;

    std::shared_mutex _mutex;
};
