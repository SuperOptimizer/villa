#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"


// struct passTroughComputor
// {
//     size_t border = 0;
//     void compute(const xt::xarray<uint8_t> &large, xt::xarray<uint8_t> &small)
//     {
//         small = large;
//     }
// };

struct vec3i_hash {
    size_t operator()(cv::Vec3i p) const
    {
        size_t hash1 = std::hash<int>{}(p[0]);
        size_t hash2 = std::hash<int>{}(p[1]);
        size_t hash3 = std::hash<int>{}(p[2]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache) {};
    T &operator()(const cv::Vec3i &p)
    {
        cv::Vec3i id = {p[0]/_chunk_size[0],p[1]/_chunk_size[1],p[2]/_chunk_size[2]};

        if (!_chunks.count(id)) {
            //readchunk
            xt::xarray<T> large;
            xt::xarray<T> *small;

            //read chunk with extra border

            _compute_f(large, *small);

            _chunks[id] = small;
        }

        return _chunks[id]->operator()(p[0]-id[0],p[1]-id[1],p[2]-id[2]);
    }
    T &operator()(int x, int y, int z)
    {
        return operator()({x,y,z});
    }

    size_t w,h,d;
    std::unordered_map<cv::Vec3i,xt::xarray<T>*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    cv::Vec3i _chunk_size;
    size_t _border;
    C &_compute_f;
};
