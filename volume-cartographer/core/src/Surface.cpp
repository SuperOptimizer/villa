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

// class PairedSurfacePointer : public SurfacePointer
// {
// public:
//     PairedSurfacePointer(const cv::Vec3f &surface_point, const cv::Vec3f &world_point) : sp(surface_point), wp(world_point) {};
//     cv::Vec3f wp;
//     cv::Vec3f sp;
// };

cv::Vec2f offsetPoint2d(TrivialSurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = ptr->point + offset;
    return {p[0], p[1]};
}

// class IndirectSurfacePointer : SurfacePointer
// {
// public:
//     IndirectSurfacePointer(Surface *surface_, cv::Vec3f point_) : surf(surface_), point(point_) {};
//     Surface *surf;
//     cv::Vec3f point;
// };

// class FusionSurfacePointer : SurfacePointer
// {
// public:
//     FusionSurfacePointer();
// protected:
//     std::vector<IndirectSurfacePointer*> _surfaces;
// };

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

cv::Vec3f QuadSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec2f loc = offsetPoint2d((TrivialSurfacePointer*)ptr,offset+_center);
    
    return grid_normal(_points, {loc[0],loc[1],0});
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

//just forwards everything but gen_coords ... can we make this more elegant without having to call the specific methods?
class ControlPointCoords : public CoordGenerator
{
public:
    CoordGenerator *_base = nullptr;
    ControlPointCoords(CoordGenerator *base) : _base(base) {};
    void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override
    {
        _base->gen_coords(coords,x,y,w,h,render_scale,coord_scale);
        
        std::cout << "call mod gen_coords" << std::endl;
        
        for(int j=0;j<h;j++)
            for(int i=0;i<h;i++) {
                float ox = 0.1*(i-w/2);
                float oy = 0.1*(j-h/2);
                float sd = ox*ox + oy*oy;
                coords(j,i,0) += 100.0/std::max(1.0f,sd);
            }
    }
    void setOffsetZ(float off) override { _base->setOffsetZ(off); };
    float offsetZ() override { return _base->offsetZ(); };
    cv::Vec3f normal(const cv::Vec3f &loc = {0,0,0}) override { return _base->normal(loc); };
    cv::Vec3f coord(const cv::Vec3f &loc = {0,0,0}) override { return _base->coord(loc); };
    protected:
        float _z_off = 0;
    
};

ControlPointSurface::ControlPointSurface(Surface *base, SurfacePointer *base_ptr, cv::Vec3f control_point)
{
    _base = base;
    _ptr = base_ptr;
    _orig_wp = base->coord(_ptr);
    _normal = base->normal(_ptr);
    _control_point = control_point;
    
    //TODO should we check consistency of these values? Or shoud we calc normal from base & control point?
    //we could also use control point as a plane with the normal of the surface normal and calc dist along projected 3d point onto the normal - this way its independent of the base xy layout (better not that will probably mess up very detailed complex 3d surfaces that do not follow the normal). But we could use the normal of base + projected distance ...
    //these could just be different options for how to apply the control point ...
}

SurfacePointer *ControlPointSurface::pointer()
{
    return _base->pointer();
}

void ControlPointSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    _base->move(ptr, offset);
}
bool ControlPointSurface::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    //FIXME check ROI! + base surface valid
    return true;
}

cv::Vec3f ControlPointSurface::coord(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    //FIXME should actually check distance between control point and surface ...
    // return _base(ptr) + _normal*10;
    std::cout << "FIXME: implement ControlPointSurface::coord()" << std::endl;
    cv::Vec3f(-1,-1,-1);
}

cv::Vec3f ControlPointSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    std::cout << "FIXME: implement ControlPointSurface::normal()" << std::endl;
    cv::Vec3f(-1,-1,-1);
}

CoordGenerator *ControlPointSurface::generator(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return new ControlPointCoords(_base->generator(ptr, offset));
}

void ControlPointSurface::setBase(QuadSurface *base)
{
    _base = base;
    
    abort();
    //FIXME refresh the cursor!
    //should still be the same? cursors should stay at a 3d position?!?
}