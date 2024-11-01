#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include <xtensor/xtensor.hpp>

#include "z5/multiarray/xtensor_access.hxx"


// static std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }
//
// template <size_t N>
// static std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }

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

template <typename T, typename C> class Chunked3dAccessor;

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;
    // friend Chunked3dAccessor;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        auto shape = ds->chunking().blockShape();
        _border = compute_f.BORDER;
    };
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id))
            cache_chunk(id);
        return _chunks[id]->operator()(p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s);
    }
    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }
    T &safe_at(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        CHUNKT *chunk = nullptr;

        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk->operator()(p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s);
    }
    T &safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    CHUNKT *cache_chunk_safe(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);
        CHUNKT *small = new CHUNKT();
        *small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

            readArea3D(large, offset, _ds, _cache);

            _compute_f.template compute<CHUNKT,T>(large, *small);

            _mutex.lock();
            if (!_chunks.count(id))
                _chunks[id] = small;
            _mutex.unlock();

        return small;
    }

    CHUNKT *cache_chunk(const cv::Vec3i &id) {
        auto s = C::CHUNK_SIZE;
        CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);
        CHUNKT *small = new CHUNKT();
        *small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {id[0]*s-_border,
            id[1]*s-_border,
            id[2]*s-_border};

            readArea3D(large, offset, _ds, _cache);

            _compute_f.template compute<CHUNKT,T>(large, *small);

            _chunks[id] = small;

        return small;
    }

    CHUNKT *chunk(const cv::Vec3i &id) {
        if (!_chunks.count(id))
            return cache_chunk(id);
        return _chunks[id];
    }

    CHUNKT *chunk_safe(const cv::Vec3i &id) {
        CHUNKT *chunk = nullptr;
        _mutex.lock_shared();
        if (_chunks.count(id)) {
            chunk = _chunks[id];
            _mutex.unlock();
        }
        else {
            _mutex.unlock();
            chunk = cache_chunk_safe(id);
        }

        return chunk;
    }

    std::unordered_map<cv::Vec3i,CHUNKT*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache *_cache;
    size_t _border;
    C &_compute_f;
    std::shared_mutex _mutex;
};


template <typename T, typename C>
class Chunked3dAccessor
{
public:
    using CHUNKT = typename Chunked3d<T,C>::CHUNKT;

    Chunked3dAccessor(Chunked3d<T,C> &ar) : _ar(ar) {};

    static Chunked3dAccessor &create(Chunked3d<T,C> &ar)
    {
        return Chunked3dAccessor(ar);
    }

    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk(p);
        }

        return _chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]);
    }



    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }

    T& safe_at(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;

        if (_corner[0] == -1)
            get_chunk_safe(p);
        else {
            bool miss = false;
            for(int i=0;i<3;i++)
                if (p[i] < _corner[i])
                    miss = true;
            for(int i=0;i<3;i++)
                if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                    miss = true;
            if (miss)
                get_chunk_safe(p);
        }

        return _chunk->operator()(p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]);
    }

    T& safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    void get_chunk(const cv::Vec3i &p)
    {
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk(id);
        _corner = id*C::CHUNK_SIZE;
    }

    void get_chunk_safe(const cv::Vec3i &p)
    {
        cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
        _chunk = _ar.chunk_safe(id);
        _corner = id*C::CHUNK_SIZE;
    }

protected:
    CHUNKT *_chunk;
    Chunked3d<T,C> &_ar;
    cv::Vec3i _corner = {-1,-1,-1};
};

