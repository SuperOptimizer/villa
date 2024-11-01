#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include <xtensor/xtensor.hpp>

#include "z5/multiarray/xtensor_access.hxx"


std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

template <size_t N>
std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

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
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        auto shape = ds->chunking().blockShape();
        _border = compute_f.BORDER;
    };
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id)) {
            CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);
            CHUNKT *small = new CHUNKT();
            *small = xt::empty<T>({s,s,s});

            cv::Vec3i offset =
            {id[0]*s-_border,
                id[1]*s-_border,
                id[2]*s-_border};

            readArea3D(large, offset, _ds, _cache);

            _compute_f.compute(large, *small);

            _chunks[id] = small;
        }
        return _chunks[id]->operator()(p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s);
    }
    T &operator()(int x, int y, int z)
    {
        return operator()({x,y,z});
    }

    std::unordered_map<cv::Vec3i,CHUNKT*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    size_t _border;
    C &_compute_f;
};
