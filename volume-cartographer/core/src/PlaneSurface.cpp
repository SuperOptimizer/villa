#include "vc/core/util/Surface.hpp"

#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/calib3d.hpp>
#include <cmath>
#include <limits>
#include <vector>


void PlaneSurface::update()
{
    cv::Vec3f vx, vy;

    vxy_from_normal(_origin,_normal,vx,vy);

    if (std::abs(_inPlaneRotation) > std::numeric_limits<float>::epsilon()) {
        vx = rotateAroundAxis(vx, _normal, _inPlaneRotation);
        vy = rotateAroundAxis(vy, _normal, _inPlaneRotation);
    }

    cv::normalize(vx, vx, 1, 0, cv::NORM_L2);
    cv::normalize(vy, vy, 1, 0, cv::NORM_L2);

    _vx = vx;
    _vy = vy;

    std::vector <cv::Vec3f> src = {_origin,_origin+_normal,_origin+_vx,_origin+_vy};
    std::vector <cv::Vec3f> tgt = {{0,0,0},{0,0,1},{1,0,0},{0,1,0}};
    cv::Mat transf;
    cv::Mat inliers;

    cv::estimateAffine3D(src, tgt, transf, inliers, 0.1, 0.99);

    _M = transf({0,0,3,3});
    _T = transf({3,0,1,3});
}

cv::Vec3f PlaneSurface::project(cv::Vec3f wp, float render_scale, float coord_scale)
{
    cv::Vec3d res = _M*cv::Vec3d(wp)+_T;
    res *= render_scale*coord_scale;

    return {res(0), res(1), res(2)};
}

float PlaneSurface::scalarp(cv::Vec3f point) const
{
    return point.dot(_normal) - _origin.dot(_normal);
}



void PlaneSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    bool create_normals = normals || offset[2] || ptr[2];
    cv::Vec3f total_offset = internal_loc(offset/scale, ptr, {1,1});

    int w = size.width;
    int h = size.height;

    cv::Mat_<cv::Vec3f> _coords_header;
    cv::Mat_<cv::Vec3f> _normals_header;

    if (!coords)
        coords = &_coords_header;
    if (!normals)
        normals = &_normals_header;

    coords->create(size);

    if (create_normals)
        normals->create(size);

    const cv::Vec3f vx = _vx;
    const cv::Vec3f vy = _vy;

    float m = 1/scale;
    cv::Vec3f use_origin = _origin + _normal*total_offset[2];

#pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            (*coords)(j,i) = vx*(i*m+total_offset[0]) + vy*(j*m+total_offset[1]) + use_origin;
        }
}

cv::Vec3f PlaneSurface::pointer()
{
    return cv::Vec3f(0, 0, 0);
}

void PlaneSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    ptr += offset;
}

cv::Vec3f PlaneSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return ptr + offset;
}

cv::Vec3f PlaneSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> coords;
    gen(&coords, nullptr, {1,1}, ptr, 1.0, offset);
    return coords(0,0);
}

cv::Vec3f PlaneSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _normal;
}


PlaneSurface::PlaneSurface(cv::Vec3f origin_, cv::Vec3f normal) : _origin(origin_)
{
    cv::normalize(normal, _normal);
    update();
};

void PlaneSurface::setNormal(cv::Vec3f normal)
{
    cv::normalize(normal, _normal);
    update();
}

void PlaneSurface::setOrigin(cv::Vec3f origin)
{
    _origin = origin;
    update();
}

cv::Vec3f PlaneSurface::origin()
{
    return _origin;
}

float PlaneSurface::pointDist(cv::Vec3f wp)
{
    float plane_off = _origin.dot(_normal);
    float scalarp = wp.dot(_normal) - plane_off /*- _z_off*/;

    return abs(scalarp);
}

void PlaneSurface::setInPlaneRotation(float radians)
{
    _inPlaneRotation = radians;
    update();
}

void PlaneSurface::setAxisAlignedRotationKey(int key)
{
    _axisAlignedRotationKey = key;
}