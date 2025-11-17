#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//TODO remove
#include <opencv2/highgui.hpp>

#include <unordered_map>
#include <nlohmann/json.hpp>
#include <system_error>
#include <cmath>
#include <limits>
#include <cerrno>
#include <algorithm>
#include <vector>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <tiffio.h>

namespace {

void normalizeMaskChannel(cv::Mat& mask)
{
    if (mask.empty()) {
        return;
    }

    cv::Mat singleChannel;
    if (mask.channels() == 1) {
        singleChannel = mask;
    } else if (mask.channels() == 3) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGR2GRAY);
    } else if (mask.channels() == 4) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGRA2GRAY);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(mask, channels);
        if (!channels.empty()) {
            singleChannel = channels[0];
        } else {
            singleChannel = mask;
        }
    }

    if (singleChannel.depth() != CV_8U) {
        cv::Mat converted;
        singleChannel.convertTo(converted, CV_8U);
        singleChannel = converted;
    }

    mask = singleChannel;
}

} // namespace

void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names) {
    nlohmann::json overlap_json;
    overlap_json["overlapping"] = std::vector<std::string>(overlapping_names.begin(), overlapping_names.end());

    std::ofstream o(seg_path / "overlapping.json");
    o << std::setw(4) << overlap_json << std::endl;
}

std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path) {
    std::set<std::string> overlapping;
    std::filesystem::path json_path = seg_path / "overlapping.json";

    if (std::filesystem::exists(json_path)) {
        std::ifstream i(json_path);
        nlohmann::json overlap_json;
        i >> overlap_json;

        if (overlap_json.contains("overlapping")) {
            for (const auto& name : overlap_json["overlapping"]) {
                overlapping.insert(name.get<std::string>());
            }
        }
    }

    return overlapping;
}



//NOTE we have 3 coordiante systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}

Surface::~Surface()
{
    if (meta) {
        delete meta;
    }
}

SurfaceControlPoint::SurfaceControlPoint(Surface *base, const cv::Vec3f &ptr_, const cv::Vec3f &control)
{
    ptr = ptr_;
    orig_wp = base->coord(ptr_);
    normal = base->normal(ptr_);
    control_point = control;
}

DeltaSurface::DeltaSurface(Surface *base) : _base(base)
{

}

void DeltaSurface::setBase(Surface *base)
{
    _base = base;
}

cv::Vec3f DeltaSurface::pointer()
{
    return _base->pointer();
}

void DeltaSurface::move(cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    _base->move(ptr, offset);
}

bool DeltaSurface::valid(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->valid(ptr, offset);
}

cv::Vec3f DeltaSurface::loc(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->loc(ptr, offset);
}

cv::Vec3f DeltaSurface::coord(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->coord(ptr, offset);
}

cv::Vec3f DeltaSurface::normal(const cv::Vec3f &ptr, const cv::Vec3f &offset)
{
    return _base->normal(ptr, offset);
}

float DeltaSurface::pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters)
{
    return _base->pointTo(ptr, tgt, th, max_iters);
}

void ControlPointSurface::addControlPoint(const cv::Vec3f &base_ptr, cv::Vec3f control_point)
{
    _controls.push_back(SurfaceControlPoint(this, base_ptr, control_point));
}

void ControlPointSurface::gen(cv::Mat_<cv::Vec3f> *coords_, cv::Mat_<cv::Vec3f> *normals_, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    std::cout << "corr gen " << _controls.size() << std::endl;
    cv::Mat_<cv::Vec3f> _coords_local;

    cv::Mat_<cv::Vec3f> *coords = coords_;

    if (!coords)
        coords = &_coords_local;

    _base->gen(coords, normals_, size, ptr, scale, offset);

    int w = size.width;
    int h = size.height;
    cv::Rect bounds(0,0,w,h);

    cv::Vec3f upper_left_nominal = nominal_loc(offset/scale, ptr, dynamic_cast<QuadSurface*>(_base)->_scale);

    float z_offset = upper_left_nominal[2];
    upper_left_nominal[2] = 0;

    auto sdist = [](const cv::Vec3f &a, const cv::Vec3f &b) {
        cv::Vec3f d = a-b;
        return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
    };

    for(auto p : _controls) {
        cv::Vec3f p_loc = nominal_loc(loc(p.ptr), ptr, dynamic_cast<QuadSurface*>(_base)->_scale) - upper_left_nominal;
        std::cout << p_loc << p_loc*scale << loc(p.ptr) << ptr << std::endl;
        p_loc *= scale;
        cv::Rect roi(p_loc[0]-40, p_loc[1]-40, 80, 80);
        cv::Rect area = roi & bounds;

        PlaneSurface plane(p.control_point, p.normal);
        float delta = plane.scalarp(coord(p.ptr));
        cv::Vec3f move = delta*p.normal;

        std::cout << area << roi << bounds << move << p.control_point << p.normal << coord(p.ptr) << std::endl;

        for(int j=area.y; j<area.y+area.height; j++)
            for(int i=area.x; i<area.x+area.width; i++) {
                float w = sdist(p_loc, cv::Vec3f(i,j,0));
                w = exp(-w/(20*20));
                (*coords)(j,i) += w*move;
            }
    }
}

void ControlPointSurface::setBase(Surface *base)
{
    DeltaSurface::setBase(base);

    assert(dynamic_cast<QuadSurface*>(base));

    //FIXME reset control points?
    std::cout << "ERROR implement search for ControlPointSurface::setBase()" << std::endl;
}

RefineCompSurface::RefineCompSurface(z5::Dataset *ds, ChunkCache *cache, QuadSurface *base)
: DeltaSurface(base)
{
    _ds = ds;
    _cache = cache;
}

void RefineCompSurface::gen(cv::Mat_<cv::Vec3f> *coords_, cv::Mat_<cv::Vec3f> *normals_, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> _coords_local;
    cv::Mat_<cv::Vec3f> _normals_local;

    cv::Mat_<cv::Vec3f> *coords = coords_;
    cv::Mat_<cv::Vec3f> *normals = normals_;

    if (!coords)
        coords = &_coords_local;
    if (!normals)
        normals = &_normals_local;

    _base->gen(coords, normals, size, ptr, scale, offset);

    cv::Mat_<cv::Vec3f> res;
    cv::Mat_<float> transparent(size, 1);
    cv::Mat_<float> blur(size, 0);
    cv::Mat_<float> integ_z(size, 0);

    if (stop < start)
        step = -abs(step);

    for(int n=0; n<=(stop-start)/step; n++) {
        cv::Mat_<uint8_t> slice;
        float off = start + step*n;
        readInterpolated3D(slice, _ds, (*coords+*normals*off)*scale, _cache);

        cv::Mat floatslice;
        slice.convertTo(floatslice, CV_32F, 1/255.0);

        cv::GaussianBlur(floatslice, blur, {7,7}, 0);
        cv::Mat opaq_slice = blur;

        opaq_slice = (opaq_slice-low)/(high-low);
        opaq_slice = cv::min(opaq_slice,1);
        opaq_slice = cv::max(opaq_slice,0);

        cv::Mat joint = transparent.mul(opaq_slice);
        integ_z += joint * off * scale;
        transparent = transparent-joint;
    }

    integ_z /= (1-transparent);

    cv::Mat mul;
    cv::cvtColor(integ_z, mul, cv::COLOR_GRAY2BGR);
    *coords += (*normals).mul(mul+1+offset[2]);
}

struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};



Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p)
{
    Rect3D res = a;
    for(int d=0;d<3;d++) {
        res.low[d] = std::min(res.low[d], p[d]);
        res.high[d] = std::max(res.high[d], p[d]);
    }

    return res;
}


bool intersect(const Rect3D &a, const Rect3D &b)
{
    for(int d=0;d<3;d++) {
        if (a.high[d] < b.low[d])
            return false;
        if (a.low[d] > b.high[d])
            return false;
    }

    return true;
}


bool overlap(SurfaceMeta &a, SurfaceMeta &b, int max_iters)
{
    if (!intersect(a.bbox, b.bbox))
        return false;

    cv::Mat_<cv::Vec3f> points = a.surface()->rawPoints();
    for(int r=0; r<std::max(10, max_iters/10); r++) {
        cv::Vec2f p = {rand() % points.cols, rand() % points.rows};
        cv::Vec3f loc = points(p[1], p[0]);
        if (loc[0] == -1)
            continue;

        cv::Vec3f ptr = b.surface()->pointer();
        if (b.surface()->pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
            return true;
        }
    }
    return false;
}


bool contains(SurfaceMeta &a, const cv::Vec3f &loc, int max_iters)
{
    if (!intersect(a.bbox, {loc,loc}))
        return false;

    cv::Vec3f ptr = a.surface()->pointer();
    if (a.surface()->pointTo(ptr, loc, 2.0, max_iters) <= 2.0) {
        return true;
    }
    return false;
}

bool contains(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs)
{
    for(auto &p : locs)
        if (!contains(a, p))
            return false;

    return true;
}

bool contains_any(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs)
{
    for(auto &p : locs)
        if (contains(a, p))
            return true;

    return false;
}


void generate_mask(QuadSurface* surf,
                            cv::Mat_<uint8_t>& mask,
                            cv::Mat_<uint8_t>& img,
                            z5::Dataset* ds_high,
                            z5::Dataset* ds_low,
                            ChunkCache* cache) {
    cv::Mat_<cv::Vec3f> points = surf->rawPoints();

    // Choose resolution based on surface size
    if (points.cols >= 4000) {
        // Large surface: work at 0.25x scale
        if (ds_low && cache) {
            readInterpolated3D(img, ds_low, points * 0.25, cache);
        } else {
            img.create(points.size());
            img.setTo(0);
        }

        mask.create(img.size());
        for(int j = 0; j < img.rows; j++) {
            for(int i = 0; i < img.cols; i++) {
                mask(j,i) = (points(j,i)[0] == -1) ? 0 : 255;
            }
        }
    } else {
        // Small surface: resize and downsample
        cv::Mat_<cv::Vec3f> scaled;
        cv::Vec2f scale = surf->scale();
        cv::resize(points, scaled, {0,0}, 1.0/scale[0], 1.0/scale[1], cv::INTER_CUBIC);

        if (ds_high && cache) {
            readInterpolated3D(img, ds_high, scaled, cache);
            cv::resize(img, img, {0,0}, 0.25, 0.25, cv::INTER_CUBIC);
        } else {
            img.create(cv::Size(points.cols/4.0, points.rows/4.0));
            img.setTo(0);
        }

        mask.create(img.size());
        for(int j = 0; j < img.rows; j++) {
            for(int i = 0; i < img.cols; i++) {
                int orig_j = j * 4 * scale[1];
                int orig_i = i * 4 * scale[0];
                mask(j,i) = (points(orig_j, orig_i)[0] == -1) ? 0 : 255;
            }
        }
    }
}

