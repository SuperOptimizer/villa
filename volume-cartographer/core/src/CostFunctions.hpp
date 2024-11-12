#pragma once

#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/core.hpp>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

#include "cubic_interpolation_acc.h"

struct CeresGrid2DcvMat3f {
    enum { DATA_DIMENSION = 3 };
    void GetValue(int row, int col, double* f) const
    {
        if (col >= _m.cols) col = _m.cols-1;
        if (row >= _m.rows) row = _m.rows-1;
        if (col <= 0) col = 0;
        if (row <= 0) row = 0;
        cv::Vec3f v = _m(row, col);
        f[0] = v[0];
        f[1] = v[1];
        f[2] = v[2];
    }
    const cv::Mat_<cv::Vec3f> _m;
};

struct CeresGrid2DcvMat1f {
    enum { DATA_DIMENSION = 1 };
    void GetValue(int row, int col, double* f) const
    {
        if (col >= _m.cols) col = _m.cols-1;
        if (row >= _m.rows) row = _m.rows-1;
        if (col <= 0) col = 0;
        if (row <= 0) row = 0;
        cv::Vec3f v = _m(row, col);
        f[0] = v[0];
    }
    const cv::Mat_<float> _m;
};

template <typename B, int N>
struct CeresGrid2DcvMat_ {
    using V = cv::Vec<B,N>;
    enum { DATA_DIMENSION = N };
    void GetValue(int row, int col, double* f) const
    {
        if (col >= _m.cols) col = _m.cols-1;
        if (row >= _m.rows) row = _m.rows-1;
        if (col <= 0) col = 0;
        if (row <= 0) row = 0;
        const V &v = _m(row, col);
        for(int i=0;i<N;i++)
            f[i] = v[i];
    }
    const cv::Mat_<V> _m;
};


//cost functions for physical paper
struct DistLoss {
    DistLoss(float dist, float w) : _d(dist), _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        //FIXME where are invalid coords coming from?
        if (a[0] == -1 && a[1] == -1 && a[2] == -1) {
            residual[0] = T(0);
            std::cout << "invalidn CORNER" << std::endl;
            return true;
        }
        //FIXME where are invalid coords coming from?
        if (b[0] == -1 && b[1] == -1 && b[2] == -1) {
            residual[0] = T(0);
            std::cout << "invalidn CORNER" << std::endl;
            return true;
        }

        T d[3];
        d[0] = a[0] - b[0];
        d[1] = a[1] - b[1];
        d[2] = a[2] - b[2];

        T dist = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);

        if (dist < _d)
            residual[0] = T(_w)*(T(_d)/dist - T(1));
        else
            residual[0] = T(_w)*(dist/T(_d) - T(1));

        return true;
    }

    double _d;
    double _w;

    static ceres::CostFunction* Create(float d, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<DistLoss, 1, 3, 3>(new DistLoss(d, w));
    }
};

//cost functions for physical paper
struct DistLoss2D {
    DistLoss2D(float dist, float w) : _d(dist), _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        if (a[0] == -1 && a[1] == -1 && a[2] == -1) {
            residual[0] = T(0);
            std::cout << "invalid CORNER" << std::endl;
            return true;
        }
        if (b[0] == -1 && b[1] == -1 && b[2] == -1) {
            residual[0] = T(0);
            std::cout << "invalid CORNER" << std::endl;
            return true;
        }

        T d[2];
        d[0] = a[0] - b[0];
        d[1] = a[1] - b[1];

        T dist = sqrt(d[0]*d[0] + d[1]*d[1]);

        if (dist <= 0) {
            residual[0] = T(_w)*(dist - T(1));
            std::cout << "uhohh" << std::endl;
        }
        else {
            if (dist < _d)
                residual[0] = T(_w)*(T(_d)/(dist+T(1e-2)) - T(1));
            else
                residual[0] = T(_w)*(dist/T(_d) - T(1));
        }

        return true;
    }

    double _d;
    double _w;

    static ceres::CostFunction* Create(float d, float w = 1.0)
    {
        if (d == 0)
            throw std::runtime_error("dist can't be zero for DistLoss2D");
        return new ceres::AutoDiffCostFunction<DistLoss2D, 1, 2, 2>(new DistLoss2D(d, w));
    }
};

//cost functions for physical paper
struct LocMinDistLoss {
    // LocMinDistLoss(const cv::Vec2f &scale, float mindist, float w) : _scale(scale), {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        //FIXME showhow we sometimes feed invalid loc (-1/-1) probably because we encountered the edge?
        // if (a[0] < 0 || a[1] < 0 || b[0] < 0 || b[1] < 0) {
        //     residual[0] = T(0);
        //     return true;
        // }

        T as[2];
        T bs[2];

        T d[2];
        d[0] = (a[0]-b[0])/T(_scale[0]);
        d[1] = (a[1]-b[1])/T(_scale[1]);

        T d2 = d[0]*d[0] + d[1]*d[1];

        if (d2 < T(_mindist*_mindist))
            residual[0] = T(_mindist) - sqrt(d2);
        else
            residual[0] = T(0);

        return true;
    }

    cv::Vec2f _scale;
    float _mindist;
    float _w;

    static ceres::CostFunction* Create(const cv::Vec2f &scale, float mindist, float w)
    {
        return new ceres::AutoDiffCostFunction<LocMinDistLoss, 1, 2, 2>(new LocMinDistLoss({scale, mindist, w}));
    }
};

//cost functions for physical paper
struct StraightLoss {
    StraightLoss(float w) : _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        T d1[3], d2[3];
        d1[0] = b[0] - a[0];
        d1[1] = b[1] - a[1];
        d1[2] = b[2] - a[2];

        d2[0] = c[0] - b[0];
        d2[1] = c[1] - b[1];
        d2[2] = c[2] - b[2];

        T l1 = sqrt(d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2]);
        T l2 = sqrt(d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]);

        T dot = (d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2])/(l1*l2);

        residual[0] = T(_w)*(T(1)-dot);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<StraightLoss, 1, 3, 3, 3>(new StraightLoss(w));
    }
};

//cost functions for physical paper
struct StraightLoss2D {
    StraightLoss2D(float w) : _w(w) {};
    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const {
        T d1[2], d2[2];
        d1[0] = b[0] - a[0];
        d1[1] = b[1] - a[1];

        d2[0] = c[0] - b[0];
        d2[1] = c[1] - b[1];

        T l1 = sqrt(d1[0]*d1[0] + d1[1]*d1[1]);
        T l2 = sqrt(d2[0]*d2[0] + d2[1]*d2[1]);

        T dot = (d1[0]*d2[0] + d1[1]*d2[1])/(l1*l2);

        residual[0] = T(_w)*(T(1)-dot);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<StraightLoss2D, 1, 2, 2, 2>(new StraightLoss2D(w));
    }
};

//cost functions for physical paper
struct SurfaceLoss {
    SurfaceLoss(const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, float w) : _interpolator(interp), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, const T* const l, T* residual) const {
        T v[3];

        _interpolator.Evaluate(l[1], l[0], v);

        residual[0] = T(_w)*(v[0] - p[0]);
        residual[1] = T(_w)*(v[1] - p[1]);
        residual[2] = T(_w)*(v[2] - p[2]);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, float w = 1.0)
    {
        // auto l = new SurfaceLoss(grid);
        // std::cout << l->_interpolator.grid_._m.size() << std::endl;
        // auto g = CeresGrid2DcvMat3f({grid});
        // std::cout << g._m.size() << std::endl;
        // std::cout << l->_interpolator.grid_._m.size() << std::endl;
        return new ceres::AutoDiffCostFunction<SurfaceLoss, 3, 3, 2>(new SurfaceLoss(interp, w));
    }

    const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &_interpolator;
};


// CeresGrid2DcvMat3f grid({points});
// ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> interp(grid);

//cost functions for physical paper
struct SurfaceLossD {
    //NOTE we expect loc to be [y, x]
    SurfaceLossD(const cv::Mat_<cv::Vec3d> &m, float w) : _m(m), _grid({_m}), _interpolator(_grid), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, const T* const l, T* residual) const {
        T v[3];

        if (!loc_valid<double,3>(_m, {val(l[0]), val(l[1])})) {
            residual[0] = T(0);
            residual[1] = T(0);
            residual[2] = T(0);
            return true;
        }

        _interpolator.Evaluate(l[0], l[1], v);

        residual[0] = T(_w)*(v[0] - p[0]);
        residual[1] = T(_w)*(v[1] - p[1]);
        residual[2] = T(_w)*(v[2] - p[2]);

        return true;
    }

    const cv::Mat_<cv::Vec<double,3>> _m;
    CeresGrid2DcvMat_<double,3> _grid;
    const ceres::BiCubicInterpolator<CeresGrid2DcvMat_<double,3>> _interpolator;
    float _w;

    int  val(const double &v) const { return v; }
    template< typename JetT>
    int  val(const JetT &v) const { return v.a; }

    static ceres::CostFunction* Create(const cv::Mat_<cv::Vec3d> &m, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<SurfaceLossD, 3, 3, 2>(new SurfaceLossD(m, w));
    }

};

//loss on tgt dist in 3d of lookup up 2d locations
struct DistLossLoc3D {
    DistLossLoc3D(const cv::Mat_<cv::Vec3d> &m, float dist, float w) : _d(dist), _m(m), _grid({_m}), _interpolator(_grid), _w(w) {};
    template <typename T>
    bool operator()(const T* const la, const T* const lb, T* residual) const {

        T a[3], b[3];

        if (!loc_valid<double,3>(_m, {val(la[0]), val(la[1])}) || !loc_valid<double,3>(_m, {val(lb[0]), val(lb[1])})) {
            residual[0] = T(0);
            return true;
        }

        _interpolator.Evaluate(la[0], la[1], a);
        _interpolator.Evaluate(lb[0], lb[1], a);

        T d[3];
        d[0] = a[0] - b[0];
        d[1] = a[1] - b[1];
        d[2] = a[2] - b[2];

        T dist = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);

        if (dist < _d)
            residual[0] = T(_w)*(T(_d)/dist - T(1));
        else
            residual[0] = T(_w)*(dist/T(_d) - T(1));

        return true;
    }

    double _d;
    const cv::Mat_<cv::Vec<double,3>> _m;
    CeresGrid2DcvMat_<double,3> _grid;
    const ceres::BiCubicInterpolator<CeresGrid2DcvMat_<double,3>> _interpolator;
    float _w;

    static ceres::CostFunction* Create(const cv::Mat_<cv::Vec3d> &m, float dist, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<DistLoss, 1, 2, 2>(new DistLoss(dist, w));
    }
};

//cost functions for physical paper
struct UsedSurfaceLoss {
    UsedSurfaceLoss(const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &interp, float w) : _interpolator(interp), _w(w) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T v[1];

        _interpolator.Evaluate(l[1], l[0], v);

        residual[0] = T(_w)*(v[0]);

        return true;
    }

    float _w;

    static ceres::CostFunction* Create(const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &interp, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<UsedSurfaceLoss, 3, 2>(new UsedSurfaceLoss(interp, w));
    }

    const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &_interpolator;
};


struct VXYCost {
    VXYCost(const std::pair<cv::Vec2i, cv::Vec3f> &p1, const std::pair<cv::Vec2i, cv::Vec3f> &p2) : _p1(p1.second), _p2(p2.second)
    {
        _d = p2.first-p1.first;
    };
    template <typename T>
    bool operator()(const T* const vx, const T* const vy, T* residual) const {
        T p1[3] = {T(_p1[0]),T(_p1[1]),T(_p1[2])};
        T p2[3];

        p2[0] = p1[0] + T(_d[0])*vx[0] + T(_d[1])*vy[0];
        p2[1] = p1[1] + T(_d[0])*vx[1] + T(_d[1])*vy[1];
        p2[2] = p1[2] + T(_d[0])*vx[2] + T(_d[1])*vy[2];

        residual[0] = p2[0] - T(_p2[0]);
        residual[1] = p2[1] - T(_p2[1]);
        residual[2] = p2[2] - T(_p2[2]);

        return true;
    }

    cv::Vec3f _p1, _p2;
    cv::Vec2i _d;

    static ceres::CostFunction* Create(const std::pair<cv::Vec2i, cv::Vec3f> &p1, const std::pair<cv::Vec2i, cv::Vec3f> &p2)
    {
        return new ceres::AutoDiffCostFunction<VXYCost, 3, 3, 3>(new VXYCost(p1, p2));
    }
};

struct OrthogonalLoss {
    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        T dot;
        dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

        T la = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        T lb = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2]);

        residual[0] = dot/(la*lb);

        return true;
    }

    static ceres::CostFunction* Create()
    {
        return new ceres::AutoDiffCostFunction<OrthogonalLoss, 1, 3, 3>(new OrthogonalLoss());
    }
};

struct ParallelLoss {
    ParallelLoss(const cv::Vec3f &ref, float w) : _w(w)
    {
        cv::normalize(ref, _ref);
    }

    template <typename T>
    bool operator()(const T* const a, T* residual) const {
        T dot;
        dot = a[0]*T(_ref[0]) + a[1]*T(_ref[1]) + a[2]*T(_ref[2]);

        T la = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);

        residual[0] = T(1)-T(_w)*dot/la;

        return true;
    }

    cv::Vec3f _ref;
    float _w;

    static ceres::CostFunction* Create(const cv::Vec3f &ref, const float &w)
    {
        return new ceres::AutoDiffCostFunction<ParallelLoss, 1, 3>(new ParallelLoss(ref, w));
    }
};

//can't decide between C++ tensor libs, xtensor wasn't great and few algos supported anyways...
template <typename T>
class StupidTensor
{
public:
    StupidTensor() {};
    template <typename O> StupidTensor(const StupidTensor<O> &other) { create(other.planes[0].size(), other.planes.size()); };
    StupidTensor(const cv::Size &size, int d) { create(size, d); };
    void create(const cv::Size &size, int d)
    {
        planes.resize(d);
        for(auto &p : planes)
            p.create(size);
    }
    void setTo(const T &v)
    {
        for(auto &p : planes)
            p.setTo(v);
    }
    template <typename O> void convertTo(O &out, int code) const
    {
        out.create(planes[0].size(), planes.size());
        for(int z=0;z<planes.size();z++)
            planes[z].convertTo(out.planes[z], code);
    }
    T &at(int z, int y, int x) { return planes[z](y,x); }
    T &operator()(int z, int y, int x) { return at(z,y,x); }
    std::vector<cv::Mat_<T>> planes;
};

using st_u = StupidTensor<uint8_t>;
using st_f = StupidTensor<cv::Vec<float,1>>;
using st_1u = StupidTensor<cv::Vec<uint8_t,1>>;
using st_3f = StupidTensor<cv::Vec3f>;

template <typename B, int N>
class StupidTensorInterpolator
{
using V = cv::Vec<B,N>;
public:
    using GRID = CeresGrid2DcvMat_<B,N>;
    StupidTensorInterpolator() {};
    StupidTensorInterpolator(StupidTensor<V> &t)
    {
        _d = t.planes.size();
        for(auto &p : t.planes)
            grid_planes.push_back(GRID({p}));
        for(auto &p : grid_planes)
            interp_planes.push_back(ceres::BiCubicInterpolator<GRID>(p));
    };
    std::vector<GRID> grid_planes;
    std::vector<ceres::BiCubicInterpolator<GRID>> interp_planes;
    template <typename T> void Evaluate(const T &z, const T &y, const T &x, T *out) const
    {
        //FIXME linear interpolate along z
        if (z < 0.0)
            interp_planes[0].Evaluate(y, x, out);
        else if (z >= _d-2)
            return interp_planes[_d-1].Evaluate(y, x, out);
        else {
            T m = z-floor(z);
            int zi = idx(z);
            T low[N];
            T high[N];
            interp_planes[zi].Evaluate(y, x, low);
            interp_planes[zi+1].Evaluate(y, x, high);
            for(int i=0;i<N;i++)
                out[i] = (T(1)-m)*low[i] + m*high[i];
        }
    }
    int  idx(const double &v) const { return v; }
    template< typename JetT>
    int  idx(const JetT &v) const { return v.a; }
    int _d = 0;
};


template <typename T, typename C>
struct CeresGridChunked3DTensor {
    enum { DATA_DIMENSION = 1 };
    void GetValue(int row, int col, double* f) const
    {
        f[0] = _t(_z, row, col);
    }
    Chunked3d<T,C> &_t;
    int _z;
};

template <typename T, typename C>
struct CeresGridChunked3DTensor3D {
    CeresGridChunked3DTensor3D(Chunked3dAccessor<T,C> &a) : _a(a) {};
    enum { DATA_DIMENSION = 1 };
    void GetValue(int z, int row, int col, double* f)
    {
        f[0] = _a.safe_at(z, row, col);
    }
    Chunked3dAccessor<T,C> &_a;
};

template <typename T, typename C>
class CachedChunked3dInterpolatorMixed
{
public:
    CachedChunked3dInterpolatorMixed(Chunked3d<T,C> &t) : _low_a(t), _high_a(t), _low_g(_low_a), _high_g(_high_a), _low_i(_low_g), _high_i(_high_g)
    {
        _d = t.shape()[0];

        // std::cout << "CachedChunked3dInterpolator acc" << &_low_a << " " << &_low_a._ar << std::endl;
        // std::cout << "CachedChunked3dInterpolator acc" << &_low_i.grid_._a << std::endl;
    };

    // CachedChunked3dInterpolator(Chunked3d<T,C> &t)
    // {
    //     _low_a = {t};
    //     _high_a = (t);
    //     _low_g = {_low_a};
    //     _high_g = {_high_a};
    //     _low_i = {_low_g};
    //     _high_i = {_high_g};
    //
    //     _d = t.shape()[0];
    // };

    Chunked3dAccessor<T,C> _low_a;
    Chunked3dAccessor<T,C> _high_a;

    CeresGridChunked3DTensor3D<T,C> _low_g;
    CeresGridChunked3DTensor3D<T,C> _high_g;

    ceres::LinxBiCubicInterpolator<CeresGridChunked3DTensor3D<T,C>> _low_i;
    ceres::LinxBiCubicInterpolator<CeresGridChunked3DTensor3D<T,C>> _high_i;

    template <typename V> void Evaluate(const V &z, const V &y, const V &x, V *out)
    {
        double zv = val(z);
        //FIXME linear interpolate along z
        if (zv < 0.0) {
            _low_i.set_z(zv);
            _low_i.Evaluate(y, x, out);
        }
        else if (int(zv)+1 >= _d) {
            _high_i.set_z(_d-1);
            _high_i.Evaluate(y, x, out);
        }
        else {
            V m = z-floor(z);
            int zi = zv;
            V low;
            V high;
            _low_i.set_z(zi);
            _high_i.set_z(zi+1);
            _low_i.Evaluate(y, x, &low);
            _high_i.Evaluate(y, x, &high);
            *out = (V(1)-m)*low + m*high;
        }
    }
    double  val(const double &v) const { return v; }
    template< typename JetT>
    double  val(const JetT &v) const { return v.a; }
    int _d = 0;
};



template <typename T, typename C>
class Chunked3dInterpolator
{
public:
    Chunked3dInterpolator(Chunked3d<T,C> &t, int depth)
    {
        for(int z=0;z<depth;z++)
            interp_grids.push_back(CeresGridChunked3DTensor<T,C>({t,z}));
        for(auto &grid : interp_grids)
            interp_planes.push_back(ceres::BiCubicInterpolator<CeresGridChunked3DTensor<T,C>>(grid));
    };
    std::vector<CeresGridChunked3DTensor<T,C>> interp_grids;
    std::vector<ceres::BiCubicInterpolator<CeresGridChunked3DTensor<T,C>>> interp_planes;

    template <typename V> void Evaluate(const V &z, const V &y, const V &x, V *out) const
    {
        //FIXME linear interpolate along z
        if (z < 0.0)
            interp_planes[0].Evaluate(y, x, out);
        else if (z >= interp_planes.size())
            return interp_planes[interp_planes.size()-1].Evaluate(y, x, out);
        else {
            V m = z-floor(z);
            int zi = idx(z);
            V low;
            V high;
            interp_planes[zi].Evaluate(y, x, &low);
            interp_planes[zi+1].Evaluate(y, x, &high);
            *out = (V(1)-m)*low + m*high;
        }
    }
    int  idx(const double &v) const { return v; }
    template< typename JetT>
    int  idx(const JetT &v) const { return v.a; }
    int _d = 0;
};

//cost functions for physical paper
template <typename I>
struct EmptySpaceLoss {
    EmptySpaceLoss(const I &interp, float w) : _interpolator(interp), _w(w) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T v;

        _interpolator.template Evaluate<T>(l[2], l[1], l[0], &v);

        residual[0] = T(_w)*v;

        return true;
    }

    float _w;
    const I &_interpolator;

    static ceres::CostFunction* Create(const I &interp, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<EmptySpaceLoss, 1, 3>(new EmptySpaceLoss(interp, w));
    }

};

template <typename T, typename C>
struct EmptySpaceLossAcc {
    EmptySpaceLossAcc(Chunked3d<T,C> &t, float w) : _interpolator(std::make_unique<CachedChunked3dInterpolator<T,C>>(t)), _w(w) {};
    template <typename E>
    bool operator()(const E* const l, E* residual) const {
        E v;

        _interpolator->template Evaluate<E>(l[2], l[1], l[0], &v);

        residual[0] = E(_w)*v;

        return true;
    }

    float _w;
    std::unique_ptr<CachedChunked3dInterpolator<T,C>> _interpolator;

    static ceres::CostFunction* Create(Chunked3d<T,C> &t, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<EmptySpaceLossAcc<T,C>, 1, 3>(new EmptySpaceLossAcc<T,C>(t, w));
    }

};

//cost functions for physical paper
template <typename I>
struct EmptySpaceLineLoss {
    // EmptySpaceMultiLoss(const StupidTensorInterpolator<uint8_t,1> &interp, int steps, uint8_t *state_a, uint8_t *state_b,  w) : _interpolator(interp), _steps(steps), _w(w) {};
    template <typename T>
    bool operator()(const T* const la, const T* const lb, T* residual) const {
        T v;
        T sum = T(0);

        for(int i=1;i<_steps;i++) {
            T f2 = T(float(i)/_steps);
            T f1 = T(1.0f-float(i)/_steps);
            _interpolator.template Evaluate<T>(f1*la[2]+f2*lb[2], f1*la[1]+f2*lb[1], f1*la[0]+f2*lb[0], &v);
            sum += T(_w)*v;
        }

        residual[0] = sum/T(_steps-1);

        return true;
    }

    const I &_interpolator;
    int _steps;
    float _w;

    static ceres::CostFunction* Create(const I &interp, int steps, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<EmptySpaceLineLoss, 1, 3, 3>(new EmptySpaceLineLoss({interp, steps, w}));
    }

};

template <typename E, typename C>
struct AnchorLoss {
    AnchorLoss(Chunked3d<E,C> &t, float w) : _interp(new CachedChunked3dInterpolator<E,C>(t)), _w(w) {};
    template <typename T>
    bool operator()(const T* const p, const T* const anchor, T* residual) const {
        T v;
        T sum = T(0);

        //anchor must very much be in low range! (TODO could also make anchor similar to emptylineloss at intervals from original anchor position!)
        _interp->template Evaluate<T>(anchor[0], anchor[1], anchor[2], &v);

        T d[3] = {p[0]-anchor[0], p[1]-anchor[1], p[2]-anchor[2]};

        v = v - T(1);

        if (v < 0)
            v = T(0);

        residual[0] = T(_w)*v*v;
        residual[1] = T(_w)*sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);

        return true;
    }

    std::unique_ptr<CachedChunked3dInterpolator<E,C>> _interp;
    float _w;

    static ceres::CostFunction* Create(Chunked3d<E,C> &t, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<AnchorLoss, 2, 3, 3>(new AnchorLoss({t, w}));
    }

};

//cost functions for physical paper
template <typename T, typename C>
struct EmptySpaceLineLossAcc {
    EmptySpaceLineLossAcc(Chunked3d<T,C> &t, int steps, float w) : _steps(steps), _w(w)
    {
        // _interpolator = std::make_unique<std::vector<CachedChunked3dInterpolator<T,C>>>();
        // for(int i=1;i<_steps;i++)
        //     _interpolator->push_back(t);
        // _interpolator.reset(new CachedChunked3dInterpolator<T,C>(t));
        _interpolator.resize(_steps-1);
        for(int i=1;i<_steps;i++)
            _interpolator[i-1].reset(new CachedChunked3dInterpolator<T,C>(t));
    };
    template <typename E>
    bool operator()(const E* const la, const E* const lb, E* residual) const {
        E v;
        E sum = E(0);

        bool ign = false;

        for(int i=1;i<_steps;i++) {
            E f2 = E(float(i)/_steps);
            E f1 = E(1.0f-float(i)/_steps);
            // (*_interpolator)[i-1].template Evaluate<E>(f1*la[2]+f2*lb[2], f1*la[1]+f2*lb[1], f1*la[0]+f2*lb[0], &v);
            _interpolator[i-1].get()->template Evaluate<E>(f1*la[2]+f2*lb[2], f1*la[1]+f2*lb[1], f1*la[0]+f2*lb[0], &v);
            sum += E(_w)*v;
        }

        residual[0] = sum/E(_steps-1);

        return true;
    }

    // std::unique_ptr<std::vector<CachedChunked3dInterpolator<T,C>>> _interpolator;
    std::vector<std::unique_ptr<CachedChunked3dInterpolator<T,C>>> _interpolator;
    // std::unique_ptr<CachedChunked3dInterpolator<T,C>> _interpolator;
    int _steps;
    float _w;

    static ceres::CostFunction* Create(Chunked3d<T,C> &t, int steps, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<EmptySpaceLineLossAcc, 1, 3, 3>(new EmptySpaceLineLossAcc(t, steps, w));
    }

};
