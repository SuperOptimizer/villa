#include "vc/core/util/Surface.hpp"

#include "vc/core/io/PointSetIO.hpp"
#include "vc/core/util/Slicing.hpp"

#include "SurfaceHelpers.hpp"

class SurfacePointer
{
    
};

class TrivialSurfacePointer : public SurfacePointer
{
public:
    TrivialSurfacePointer(cv::Vec3f point_) : point(point_) {};
    cv::Vec3f point;
};

cv::Vec2f offsetPoint2d(TrivialSurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = ptr->point + offset;
    return {p[0], p[1]};
}

class IndirectSurfacePointer : SurfacePointer
{
public:
    IndirectSurfacePointer(Surface *surface_, cv::Vec3f point_) : surf(surface_), point(point_) {};
    Surface *surf;
    cv::Vec3f point;
};

class FusionSurfacePointer : SurfacePointer
{
public:
    FusionSurfacePointer();
protected:
    std::vector<IndirectSurfacePointer*> _surfaces;
};

//TODO add non-cloning variant?
QuadSurface::QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale)
{
    _points = points.clone();
    //-1 as many times we read with linear interpolation and access +1 locations
    _bounds = {0,0,points.cols-1,points.rows-1};
    _scale = scale;
    _center = {points.cols/2,points.rows/2,0};
}

SurfacePointer *QuadSurface::pointer()
{
    return new TrivialSurfacePointer({0,0,0});
}


void QuadSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    ((TrivialSurfacePointer*)ptr)->point += offset;
}

bool QuadSurface::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = ((TrivialSurfacePointer*)ptr)->point + offset + _center;
    return _bounds.contains({p[0],p[1]});
}

static cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    cv::Vec3f p00 = points(y,x);
    cv::Vec3f p01 = points(y,x+1);
    cv::Vec3f p10 = points(y+1,x);
    cv::Vec3f p11 = points(y+1,x+1);
    
    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

cv::Vec3f QuadSurface::coord(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return at_int(_points, offsetPoint2d((TrivialSurfacePointer*)ptr,offset+_center));
}


QuadSurface *load_quad_from_vcps(const std::string &path)
{    
    volcart::OrderedPointSet<cv::Vec3d> segment_raw = volcart::PointSetIO<cv::Vec3d>::ReadOrderedPointSet(path);
    
    cv::Mat src(segment_raw.height(), segment_raw.width(), CV_64FC3, (void*)const_cast<cv::Vec3d*>(&segment_raw[0]));
    cv::Mat_<cv::Vec3f> points;
    src.convertTo(points, CV_32F);    
    
    double sx, sy;
    
    vc_segmentation_scales(points, sx, sy);
    
    return new QuadSurface(points, {sx,sy});
}

CoordGenerator *QuadSurface::generator(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec3f total_offset = offset+_center;
    
    //without pointer we just use the center == default pointer
    if (ptr)
        total_offset += ((TrivialSurfacePointer*)ptr)->point;
    
    //FIXME implement & use offset for gridcoords    
    return new GridCoords(&_points, _scale[0], _scale[1], total_offset);
}


QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out)
{
    cv::Mat_<cv::Vec3f> points;
    
    TrivialSurfacePointer *trivial_ptr = (TrivialSurfacePointer*)ptr;
    
    std::cout << "ptr" << trivial_ptr->point << std::endl;
    
    points = derive_regular_region_largesteps(src->_points, trivial_ptr->point[0]+src->_center[0], trivial_ptr->point[1]+src->_center[1], step_search, w*step_out/step_search, h*step_out/step_search);
    points = upsample_with_grounding(points, {w,h}, src->_points, src->_scale[0], src->_scale[1]);
    
    return new QuadSurface(points, {1.0/step_out, 1.0/step_out});
}