#pragma once

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/matx.inl.hpp>
#include <opencv2/core/saturate.hpp>
#include <xsimd/memory/xsimd_aligned_allocator.hpp>
#include <xsimd/types/xsimd_api.hpp>
#include <xtensor/core/xlayout.hpp>
#include <xtensor/core/xstrides.hpp>
#include <xtensor/core/xtensor_forward.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/utils/xutils.hpp>
#include <xtensor/views/xslice.hpp>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "z5/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"

template <typename T> class ChunkCache;

// Max number of ChunkedCachedInterpolator entries per thread
inline constexpr std::size_t CCI_TLS_MAX = 256;



struct passTroughComputor
{
    static constexpr int BORDER = 0;
    static constexpr int CHUNK_SIZE = 32;
    static constexpr int FILL_V = 0;
    const std::string UNIQUE_ID_STRING = "";
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        constexpr int low = BORDER;
        constexpr int high = BORDER + CHUNK_SIZE;
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

template <typename T, typename C> class Chunked3dAccessor;

std::filesystem::path resolve_chunk_cache_dir(
    const std::filesystem::path& cache_root,
    const std::filesystem::path& root,
    bool persistent,
    const std::filesystem::path& ds_path);

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d final {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache<T> *cache) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
    };
    ~Chunked3d()
    {
        if (!_persistent)
            remove_all(_cache_dir);
    };
    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache<T> *cache, const std::filesystem::path &cache_root) : _compute_f(compute_f), _ds(ds), _cache(cache)
    {
        _border = compute_f.BORDER;
        
        if (_ds)
            _shape = {static_cast<int>(_ds->shape()[0]), static_cast<int>(_ds->shape()[1]), static_cast<int>(_ds->shape()[2])};
        
        if (cache_root.empty())
            return;

        if (!_compute_f.UNIQUE_ID_STRING.size())
            throw std::runtime_error("requested fs cache for compute function without identifier");        
        
        std::filesystem::path root = cache_root/_compute_f.UNIQUE_ID_STRING;
        
        std::filesystem::create_directories(root);
        
        if (!_ds)
            _persistent = false;
        
        _cache_dir = resolve_chunk_cache_dir(cache_root, root, _persistent,
                                              _ds ? _ds->path() : std::filesystem::path{});
        
        if (_cache_dir.empty())
            throw std::runtime_error("could not create cache dir - maybe too many caches in cache root (max 1000!)");
        
    };
    [[gnu::always_inline]] constexpr size_t calc_off(const cv::Vec3i &p) const noexcept
    {
        constexpr auto s = C::CHUNK_SIZE;
        return static_cast<size_t>(p[0]) + static_cast<size_t>(p[1])*s + static_cast<size_t>(p[2])*s*s;
    }
    T &operator()(const cv::Vec3i &p)
    {
        auto s = C::CHUNK_SIZE;
        cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

        if (!_chunks.count(id))
            cache_chunk(id);

        return _chunks[id][calc_off({p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s})];
    }
    T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }
    T &safe_at(const cv::Vec3i &p)
    {
        const auto s = C::CHUNK_SIZE;
        const cv::Vec3i id{ p[0]/s, p[1]/s, p[2]/s };

        {
            std::shared_lock<std::shared_mutex> rlock(_mutex);
            if (auto it = _chunks.find(id); it != _chunks.end()) {
                T* chunk = it->second;
                return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
            }
        }
        // compute/load outside shared lock
        T* chunk = cache_chunk_safe(id);
        return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
    }
    T &safe_at(int z, int y, int x)
    {
        return safe_at({z,y,x});
    }

    std::filesystem::path id_path(const std::filesystem::path &dir, const cv::Vec3i &id)
    {
        return dir / (std::to_string(id[0]) + "." + std::to_string(id[1]) + "." + std::to_string(id[2]));
    }

    T *cache_chunk_safe_mmap(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;

        std::filesystem::path tgt_path = id_path(_cache_dir, id);
        size_t len = s*s*s;
        size_t len_bytes = len*sizeof(T);

        if (std::filesystem::exists(tgt_path)) {
            int fd = open(tgt_path.string().c_str(), O_RDWR);
            T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);

            _mutex.lock();
            if (!_chunks.count(id)) {
                _chunks[id] = chunk;
            }
            else {
                munmap(chunk, len_bytes);
                chunk = _chunks[id];
            }
            _mutex.unlock();

            return chunk;
        }

        std::filesystem::path tmp_path;
        _mutex.lock();
        std::stringstream ss;
        ss << this << "_" << std::this_thread::get_id() << "_" << _tmp_counter++;
        tmp_path = std::filesystem::path(_cache_dir) / ss.str();
        _mutex.unlock();
        int fd = open(tmp_path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
        if (fd == -1) {
            const int err = errno;
            std::stringstream msg;
            msg << "Chunked3d: open failed for " << tmp_path << ": " << std::strerror(err);
            throw std::runtime_error(msg.str());
        }
        int ret = ftruncate(fd, len_bytes);
        if (ret != 0) {
            const int err = errno;
            close(fd);
            std::stringstream msg;
            msg << "Chunked3d: ftruncate failed for " << tmp_path
                << " (" << len_bytes << " bytes): " << std::strerror(err);
            throw std::runtime_error(msg.str());
        }
        T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        cv::Vec3i offset =
        {static_cast<int>(id[0]*s-_border),
            static_cast<int>(id[1]*s-_border),
            static_cast<int>(id[2]*s-_border)};

        CHUNKT small = xt::empty<T>({s,s,s});
        CHUNKT large;
        if (_ds) {
            large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);

            readArea3D(large, offset, _ds, _cache);
        }

        _compute_f.template compute<CHUNKT,T>(large, small, offset);

        for(int i=0;i<len;i++)
            chunk[i] = (&small(0,0,0))[i];

        _mutex.lock();
        if (!_chunks.count(id)) {
            _chunks[id] = chunk;
            int ret = rename(tmp_path.string().c_str(), tgt_path.string().c_str());

            if (ret)
                throw std::runtime_error("oops rename failed!");
        }
        else {
            munmap(chunk, len_bytes);
            unlink(tmp_path.string().c_str());
            chunk = _chunks[id];
        }
        _mutex.unlock();

        return chunk;
    }


    T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    {
        auto s = C::CHUNK_SIZE;
        CHUNKT small = xt::empty<T>({s,s,s});

        cv::Vec3i offset =
        {static_cast<int>(id[0]*s-_border),
            static_cast<int>(id[1]*s-_border),
            static_cast<int>(id[2]*s-_border)};
            
        CHUNKT large;
        if (_ds) {
            large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
            large = xt::full_like(large, C::FILL_V);
            
            readArea3D(large, offset, _ds, _cache);
        }

        _compute_f.template compute<CHUNKT,T>(large, small, offset);

        T *chunk = nullptr;

        _mutex.lock();
        if (!_chunks.count(id)) {
            chunk = (T*)malloc(s*s*s*sizeof(T));
            memcpy(chunk, &small(0,0,0), s*s*s*sizeof(T));
            _chunks[id] = chunk;
        }
        else {
            chunk = _chunks[id];
        }
        _mutex.unlock();

        return chunk;
    }

    // T *cache_chunk_safe_alloc(const cv::Vec3i &id)
    // {
    //     auto s = C::CHUNK_SIZE;
    //     CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
    //     large = xt::full_like(large, C::FILL_V);
    //     CHUNKT small = xt::empty<T>({s,s,s});
    //
    //     cv::Vec3i offset =
    //     {id[0]*s-_border,
    //         id[1]*s-_border,
    //         id[2]*s-_border};
    //
    //     readArea3D(large, offset, _ds, _cache);
    //
    //     _compute_f.template compute<CHUNKT,T>(large, small);
    //
    //     _mutex.lock();
    //     if (!_chunks.count(id))
    //         _chunks[id] = small;
    //     else {
    //         #pragma omp atomic
    //         chunk_compute_collisions++;
    //         delete small;
    //         small = _chunks[id];
    //     }
    //     #pragma omp atomic
    //     chunk_compute_total++;
    //     _mutex.unlock();
    //
    //     return small;
    // }

    T *cache_chunk_safe(const cv::Vec3i &id)
    {
        if (_cache_dir.empty())
            return cache_chunk_safe_alloc(id);
        else
            return cache_chunk_safe_mmap(id);
    }

    T *cache_chunk(const cv::Vec3i &id) {
        return cache_chunk_safe(id);
        // auto s = C::CHUNK_SIZE;
        // CHUNKT large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        // large = xt::full_like(large, C::FILL_V);
        // CHUNKT *small = new CHUNKT();
        // *small = xt::empty<T>({s,s,s});
        //
        // cv::Vec3i offset =
        // {id[0]*s-_border,
        //     id[1]*s-_border,
        //     id[2]*s-_border};
        //
        //     readArea3D(large, offset, _ds, _cache);
        //
        //     _compute_f.template compute<CHUNKT,T>(large, *small);
        //
        //     _chunks[id] = small;
        //
        // return small;
    }

    T *chunk(const cv::Vec3i &id) {
        if (!_chunks.count(id))
            return cache_chunk(id);
        return _chunks[id];
    }

    T *chunk_safe(const cv::Vec3i &id) {
        T *chunk = nullptr;
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
    
    std::vector<int> shape() {
        return _shape;
    }

    std::unordered_map<cv::Vec3i,T*,vec3i_hash> _chunks;
    z5::Dataset *_ds;
    ChunkCache<T> *_cache;
    size_t _border;
    C &_compute_f;
    std::shared_mutex _mutex;
    uint64_t _tmp_counter = 0;
    std::filesystem::path _cache_dir;
    bool _persistent = true;
    std::vector<int> _shape;
};

template <typename T, typename C>
class Chunked3dAccessor final
{
public:
    using CHUNKT = typename Chunked3d<T,C>::CHUNKT;

    Chunked3dAccessor(Chunked3d<T,C> &ar) : _ar(ar) {};

    static Chunked3dAccessor create(Chunked3d<T,C> &ar)
    {
        return Chunked3dAccessor(ar);
    }

    [[gnu::always_inline]] T &operator()(const cv::Vec3i &p)
    {
        constexpr auto s = C::CHUNK_SIZE;

        // Fast path: check if point is in current chunk
        if (_corner[0] != -1) [[likely]] {
            const bool in_chunk =
                (static_cast<unsigned>(p[0] - _corner[0]) < s) &&
                (static_cast<unsigned>(p[1] - _corner[1]) < s) &&
                (static_cast<unsigned>(p[2] - _corner[2]) < s);
            if (!in_chunk) [[unlikely]]
                get_chunk(p);
        } else [[unlikely]] {
            get_chunk(p);
        }

        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }

    [[gnu::always_inline]] T &operator()(int z, int y, int x)
    {
        return operator()({z,y,x});
    }

    [[gnu::always_inline]] T& safe_at(const cv::Vec3i &p)
    {
        constexpr auto s = C::CHUNK_SIZE;

        // Fast path: check if point is in current chunk
        if (_corner[0] != -1) [[likely]] {
            const bool in_chunk =
                (static_cast<unsigned>(p[0] - _corner[0]) < s) &&
                (static_cast<unsigned>(p[1] - _corner[1]) < s) &&
                (static_cast<unsigned>(p[2] - _corner[2]) < s);
            if (!in_chunk) [[unlikely]]
                get_chunk_safe(p);
        } else [[unlikely]] {
            get_chunk_safe(p);
        }

        return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
    }

    [[gnu::always_inline]] T& safe_at(int z, int y, int x)
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

    Chunked3d<T,C> &_ar;
protected:
    T *_chunk;
    cv::Vec3i _corner = {-1,-1,-1};
};

// ────────────────────────────────────────────────────────────────────────────────
//  CachedChunked3dInterpolator – thread-safe version
// ────────────────────────────────────────────────────────────────────────────────

template <typename T, typename C>
class CachedChunked3dInterpolator final
{
public:
    using Acc   = Chunked3dAccessor<T, C>;
    using ArRef = Chunked3d<T, C>&;

    explicit CachedChunked3dInterpolator(ArRef ar)
        : _ar(ar), _shape(ar.shape())
    {}

    CachedChunked3dInterpolator(const CachedChunked3dInterpolator&)            = delete;
    CachedChunked3dInterpolator& operator=(const CachedChunked3dInterpolator&) = delete;

    /** Trilinear interpolation. */
    template <typename V>
    inline void Evaluate(const V& z, const V& y, const V& x, V* out) const
    {
        // ---- 1. get *this* thread’s private accessor ------------------------
        Acc& a = local_accessor();

        // ---- 2. fast trilinear interpolation --------------------
        cv::Vec3d f { val(z), val(y), val(x) };

        cv::Vec3i corner { static_cast<int>(std::floor(f[0])),
                           static_cast<int>(std::floor(f[1])),
                           static_cast<int>(std::floor(f[2])) };

        for (int i=0; i<3; ++i) {
            corner[i] = std::max(corner[i], 0);
            if (!_shape.empty())
                corner[i] = std::min(corner[i], _shape[i]-2);
        }

        const V fx = z - V(corner[0]);
        const V fy = y - V(corner[1]);
        const V fz = x - V(corner[2]);

        // clamp only once – cheaper than three branches per component
        const V cx = std::clamp(fx, V(0), V(1));
        const V cy = std::clamp(fy, V(0), V(1));
        const V cz = std::clamp(fz, V(0), V(1));

        // fetch the eight lattice points
        const V c000 = V(a.safe_at(corner));
        const V c100 = V(a.safe_at(corner + cv::Vec3i(1,0,0)));
        const V c010 = V(a.safe_at(corner + cv::Vec3i(0,1,0)));
        const V c110 = V(a.safe_at(corner + cv::Vec3i(1,1,0)));
        const V c001 = V(a.safe_at(corner + cv::Vec3i(0,0,1)));
        const V c101 = V(a.safe_at(corner + cv::Vec3i(1,0,1)));
        const V c011 = V(a.safe_at(corner + cv::Vec3i(0,1,1)));
        const V c111 = V(a.safe_at(corner + cv::Vec3i(1,1,1)));

        // interpolate using FMA-friendly form: a + t*(b-a) instead of (1-t)*a + t*b
        const V c00 = c000 + cz*(c001 - c000);
        const V c01 = c010 + cz*(c011 - c010);
        const V c10 = c100 + cz*(c101 - c100);
        const V c11 = c110 + cz*(c111 - c110);

        const V c0  = c00 + cy*(c01 - c00);
        const V c1  = c10 + cy*(c11 - c10);

        *out = c0 + cx*(c1 - c0);
    }

    // -------------------------------------------------------------------------
    double val(const double& v) const { return v; }
    template <typename JetT> double val(const JetT& v) const { return v.a; }

private:
    /** Return the accessor that is *unique to this combination of
     *  (interpolator instance, thread)*. */
    Acc& local_accessor() const
    {
        // Per-thread, bounded cache keyed by the underlying array address.
        // (Multiple interpolators over the same array share one accessor.)
        struct TLS {
            std::unordered_map<const void*, Acc> map;
            std::deque<const void*> order;          // FIFO ~ LRU-ish
        };
        thread_local TLS tls;

        constexpr std::size_t kMax = CCI_TLS_MAX;

        const void* key = static_cast<const void*>(&_ar);

        if (auto it = tls.map.find(key); it != tls.map.end()) {
            return it->second;
        }

        // Evict oldest if at capacity
        if (tls.map.size() >= kMax && !tls.order.empty()) {
            const void* old = tls.order.front();
            tls.order.pop_front();
            tls.map.erase(old);
        }

        auto [it2, inserted] = tls.map.emplace(key, Acc{_ar});
        if (inserted) tls.order.push_back(key);
        return it2->second;
    }

    ArRef               _ar;
    std::vector<int>    _shape;
};

struct Chunked3dFloatFromUint8
{
    Chunked3dFloatFromUint8(std::unique_ptr<z5::Dataset> &&ds, float scale, ChunkCache<uint8_t> *cache, std::string const &cache_root, std::string const &unique_id) :
        _passthrough{unique_id},
        _x(_passthrough, ds.get(), cache, cache_root),
        _scale(scale),  // multiplying by this maps indices of the 'canonical' volume to indices of our dataset
        _ds(std::move(ds))  // take ownership of the dataset, as Chunked3d accesses it through a bare pointer
    {
    }

    float operator()(cv::Vec3d p)
    {
        // p has zyx ordering!
        p *= _scale;
        cv::Vec3i i{static_cast<int>(lround(p[0])), static_cast<int>(lround(p[1])), static_cast<int>(lround(p[2]))};
        uint8_t x = _x.safe_at(i);
        return static_cast<float>(x) / 255.f;
    }

    float operator()(double z, double y, double x)
    {
        return operator()({z,y,x});
    }

    passTroughComputor _passthrough;
    Chunked3d<uint8_t, passTroughComputor> _x;
    float _scale;
    std::unique_ptr<z5::Dataset> _ds;
};

struct Chunked3dVec3fFromUint8
{
    Chunked3dVec3fFromUint8(std::vector<std::unique_ptr<z5::Dataset>> &&dss, float scale, ChunkCache<uint8_t> *cache, std::string const &cache_root, std::string const &unique_id) :
        _passthrough_x{unique_id + "_x"},
        _passthrough_y{unique_id + "_y"},
        _passthrough_z{unique_id + "_z"},
        _x(_passthrough_x, dss[0].get(), cache, cache_root),
        _y(_passthrough_y, dss[1].get(), cache, cache_root),
        _z(_passthrough_z, dss[2].get(), cache, cache_root),
        _scale(scale),  // multiplying by this maps indices of the 'canonical' volume to indices of our three volumes
        _dss(std::move(dss))  // take ownership of the datasets here, as Chunked3d accesses them through a bare pointer
    {
    }

    cv::Vec3f operator()(cv::Vec3d p)
    {
        // Both p and returned vector have zyx ordering!
        p *= _scale;
        cv::Vec3i i{static_cast<int>(lround(p[0])), static_cast<int>(lround(p[1])), static_cast<int>(lround(p[2]))};
        uint8_t x = _x.safe_at(i);
        uint8_t y = _y.safe_at(i);
        uint8_t z = _z.safe_at(i);
        return (cv::Vec3f{static_cast<float>(z), static_cast<float>(y), static_cast<float>(x)} - cv::Vec3f{128.f, 128.f, 128.f}) / 127.f;
    }

    cv::Vec3f operator()(double z, double y, double x)
    {
        return operator()({z,y,x});
    }

    passTroughComputor _passthrough_x, _passthrough_y, _passthrough_z;
    Chunked3d<uint8_t, passTroughComputor> _x, _y, _z;
    float _scale;
    std::vector<std::unique_ptr<z5::Dataset>> _dss;
};

// Suppress implicit instantiation in every TU – the explicit instantiation
// lives in ChunkedTensor.cpp.  This avoids duplicating the heavy method bodies
// (cache_chunk_safe_mmap, cache_chunk_safe_alloc, etc.) across ~12 translation units.
extern template class Chunked3d<uint8_t, passTroughComputor>;
extern template class Chunked3dAccessor<uint8_t, passTroughComputor>;
extern template class CachedChunked3dInterpolator<uint8_t, passTroughComputor>;
