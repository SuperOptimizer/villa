#pragma once

#include "vc/core/util/Slicing.hpp"

#include <opencv2/core.hpp>
#include "z5/dataset.hxx"

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xtensor.hpp)
#include XTENSORINCLUDE(containers, xadapt.hpp)
#include XTENSORINCLUDE(views, xview.hpp)

#include "z5/multiarray/xtensor_access.hxx"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#ifndef CCI_TLS_MAX // Max number for ChunkedCachedInterpolator
#define CCI_TLS_MAX 256
#endif



struct passTroughComputor
{
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    const std::string UNIQUE_ID_STRING = "";
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

static uint64_t miss = 0;
static uint64_t total = 0;
static uint64_t chunk_compute_collisions = 0;
static uint64_t chunk_compute_total = 0;

template <typename T, typename C> class Chunked3dAccessor;

static std::string tmp_name_proc_thread()
{
    std::stringstream ss;
    ss << "tmp_" << getpid() << "_" << std::this_thread::get_id();
    return ss.str();
}

//chunked 3d tensor for on-demand computation from a zarr dataset ... could as some point be file backed ...
template <typename T, typename C>
class Chunked3d {
public:
    using CHUNKT = xt::xtensor<T,3,xt::layout_type::column_major>;

    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache<T> *cache);
    ~Chunked3d();
    Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache<T> *cache, const std::filesystem::path &cache_root);

    size_t calc_off(const cv::Vec3i &p);
    T &operator()(const cv::Vec3i &p);
    T &operator()(int z, int y, int x);
    T &safe_at(const cv::Vec3i &p);
    T &safe_at(int z, int y, int x);

    std::filesystem::path id_path(const std::filesystem::path &dir, const cv::Vec3i &id);

    T *cache_chunk_safe_mmap(const cv::Vec3i &id);
    T *cache_chunk_safe_alloc(const cv::Vec3i &id);
    T *cache_chunk_safe(const cv::Vec3i &id);
    T *cache_chunk(const cv::Vec3i &id);

    T *chunk(const cv::Vec3i &id);
    T *chunk_safe(const cv::Vec3i &id);

    std::vector<int> shape();

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

void print_accessor_stats();

template <typename T, typename C>
class Chunked3dAccessor
{
public:
    using CHUNKT = typename Chunked3d<T,C>::CHUNKT;

    Chunked3dAccessor(Chunked3d<T,C> &ar);

    static Chunked3dAccessor create(Chunked3d<T,C> &ar);

    T &operator()(const cv::Vec3i &p);
    T &operator()(int z, int y, int x);

    T& safe_at(const cv::Vec3i &p);
    T& safe_at(int z, int y, int x);

    void get_chunk(const cv::Vec3i &p);
    void get_chunk_safe(const cv::Vec3i &p);

    Chunked3d<T,C> &_ar;
protected:
    T *_chunk;
    cv::Vec3i _corner = {-1,-1,-1};
};

// ────────────────────────────────────────────────────────────────────────────────
//  CachedChunked3dInterpolator – thread-safe version
// ────────────────────────────────────────────────────────────────────────────────

template <typename T, typename C>
class CachedChunked3dInterpolator
{
public:
    using Acc   = Chunked3dAccessor<T, C>;
    using ArRef = Chunked3d<T, C>&;

    explicit CachedChunked3dInterpolator(ArRef ar);

    CachedChunked3dInterpolator(const CachedChunked3dInterpolator&)            = delete;
    CachedChunked3dInterpolator& operator=(const CachedChunked3dInterpolator&) = delete;

    /** Trilinear interpolation. */
    template <typename V>
    inline void Evaluate(const V& z, const V& y, const V& x, V* out) const
    {
        // ---- 1. get *this* thread's private accessor ------------------------
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

        // interpolate
        const V c00 = (V(1)-cz)*c000 + cz*c001;
        const V c01 = (V(1)-cz)*c010 + cz*c011;
        const V c10 = (V(1)-cz)*c100 + cz*c101;
        const V c11 = (V(1)-cz)*c110 + cz*c111;

        const V c0  = (V(1)-cy)*c00 + cy*c01;
        const V c1  = (V(1)-cy)*c10 + cy*c11;

        *out = (V(1)-cx)*c0 + cx*c1;
    }

    // -------------------------------------------------------------------------
    double val(const double& v) const;
    template <typename JetT>
    double val(const JetT& v) const {
        if constexpr (std::is_arithmetic_v<JetT>) {
            return static_cast<double>(v);
        } else {
            return v.a;
        }
    }

private:
    /** Return the accessor that is *unique to this combination of
     *  (interpolator instance, thread)*. */
    Acc& local_accessor() const;

    ArRef               _ar;
    std::vector<int>    _shape;
};

struct Chunked3dFloatFromUint8
{
    Chunked3dFloatFromUint8(std::unique_ptr<z5::Dataset> &&ds, float scale, ChunkCache<uint8_t>*cache, std::string const &cache_root, std::string const &unique_id) :
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
        cv::Vec3i i{lround(p[0]), lround(p[1]), lround(p[2])};
        uint8_t x = _x.safe_at(i);
        return float{x} / 255.f;
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
    Chunked3dVec3fFromUint8(std::vector<std::unique_ptr<z5::Dataset>> &&dss, float scale, ChunkCache<uint8_t>*cache, std::string const &cache_root, std::string const &unique_id) :
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
        cv::Vec3i i{lround(p[0]), lround(p[1]), lround(p[2])};
        uint8_t x = _x.safe_at(i);
        uint8_t y = _y.safe_at(i);
        uint8_t z = _z.safe_at(i);
        return (cv::Vec3f{z, y, x} - cv::Vec3f{128.f, 128.f, 128.f}) / 127.f;
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
