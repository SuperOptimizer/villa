#include "vc/core/util/Surface.hpp"

#include "vc/core/io/PointSetIO.hpp"
#include "vc/core/util/Slicing.hpp"

#include "SurfaceHelpers.hpp"

class SurfacePointer
{
public:
    virtual SurfacePointer *clone() const = 0;
};

class TrivialSurfacePointer : public SurfacePointer
{
public:
    TrivialSurfacePointer(cv::Vec3f loc_) : loc(loc_) {}
    SurfacePointer *clone() const override { return new TrivialSurfacePointer(*this); }
    cv::Vec3f loc;
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
    cv::Vec3f p = ptr->loc + offset;
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


static cv::Vec3f offset3(const cv::Vec3f &loc, const cv::Vec3f &offset_scaled, const cv::Vec2f &scale, const cv::Vec3f &offset_unscaled)
{
    // std::cout << "offset3" << loc << offset_scaled << scale << offset_unscaled << std::endl;
    return loc + cv::Vec3f(offset_scaled[0]*scale[0]+offset_unscaled[0], offset_scaled[1]*scale[1]+offset_unscaled[1], offset_scaled[2]+offset_unscaled[2]);
}

static cv::Vec2f offset2(const cv::Vec3f &loc, const cv::Vec3f &offset_scaled, const cv::Vec2f &scale, const cv::Vec3f &offset_unscaled)
{
    return cv::Vec2f(loc[0],loc[1]) + cv::Vec2f(offset_scaled[0]*scale[0]+offset_unscaled[0], offset_scaled[1]*scale[1]+offset_unscaled[1]);
}

void QuadSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    ptr_inst->loc = offset3(ptr_inst->loc, offset, _scale, {0,0,0});
    // std::cout << "moved " << ptr << ptr_inst->loc << offset << std::endl;
}

bool QuadSurface::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec2i p = offset2(ptr_inst->loc, offset, _scale, _center);
    
    return _bounds.contains(p);
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
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec2f p = offset2(ptr_inst->loc, offset, _scale, _center);
    
    return at_int(_points, p);
}

cv::Vec3f QuadSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec3f p = offset3(ptr_inst->loc, offset, _scale, _center);
    p[0] /= _scale[0];
    p[1] /= _scale[1];
    
    return p;
}

cv::Vec3f QuadSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec3f p = offset3(ptr_inst->loc, offset, _scale, _center);
    
    return grid_normal(_points, p);
}

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

static float search_min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);
    float res;
    
    //TODO check maybe add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2f step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            //just skip if out of bounds
            if (!boundary.contains({cand[0],cand[1]}))
                continue;
            
            val = at_int(points, cand);
            res = sdist(val, tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_x)
            break;
    }

    return sqrt(best);
}

//search the surface point that is closest to th tgt coord
float QuadSurface::pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th)
{
    TrivialSurfacePointer *tgt_ptr = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(tgt_ptr);

    cv::Vec2f loc = {tgt_ptr->loc[0],tgt_ptr->loc[1]};
    cv::Vec3f _out;
    
    cv::Vec2f step_small = {std::max(1.0f,_scale[0]),std::max(1.0f,_scale[1])};
    float min_mul = std::min(0.1*_points.cols/_scale[0],0.1*_points.rows/_scale[1]);
    cv::Vec2f step_large = {min_mul*_scale[0],min_mul*_scale[1]};

    float dist = search_min_loc(_points, loc, _out, tgt, step_small, _scale[0]*0.01);
    
    if (dist < th && dist >= 0) {
        tgt_ptr->loc = {loc[0],loc[1]};
        return dist;
    }
    
    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(_points.cols/_scale[0]+_points.rows/_scale[1]);
    
    //FIXME is this excessive?
    for(int r=0;r<1000;r++) {
        loc = {1 + (rand() % _points.cols-3), 1 + (rand() % _points.rows-3)};
        
        float dist = search_min_loc(_points, loc, _out, tgt, step_large, _scale[0]*0.01);
        
        if (dist < th && dist >= 0) {
            tgt_ptr->loc = {loc[0],loc[1]};
            return dist;
        } else if (dist >= 0 && dist < min_dist) {
            min_loc = loc;
            min_dist = dist;
        }
    }
    
    tgt_ptr->loc = {min_loc[0],min_loc[1]};
    return min_dist;
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
    //without pointer we just use the center == default pointer
    cv::Vec3f total_offset = offset3(0, offset, _scale, 0);
    
    if (ptr) {
        TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
        assert(ptr_inst);

        total_offset += ptr_inst->loc;
    }
    
    //FIXME implement & use offset for gridcoords    
    return new GridCoords(&_points, _scale[0], _scale[1], total_offset);
}


QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out)
{
    cv::Mat_<cv::Vec3f> points;
    
    TrivialSurfacePointer *trivial_ptr = (TrivialSurfacePointer*)ptr;
    
    std::cout << "ptr" << trivial_ptr->loc << std::endl;
    
    points = derive_regular_region_largesteps(src->_points, trivial_ptr->loc[0]+src->_center[0], trivial_ptr->loc[1]+src->_center[1], step_search, w*step_out/step_search, h*step_out/step_search);
    points = upsample_with_grounding(points, {w,h}, src->_points, src->_scale[0], src->_scale[1]);
    
    return new QuadSurface(points, {1.0/step_out, 1.0/step_out});
}

//just forwards everything but gen_coords ... can we make this more elegant without having to call the specific methods?
class ControlPointCoords : public CoordGenerator
{
public:
    ControlPointCoords(TrivialSurfacePointer *gen_ptr, ControlPointSurface *surf);
    void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    void setOffsetZ(float off) override { _base_gen->setOffsetZ(off); };
    float offsetZ() override { return _base_gen->offsetZ(); };
    cv::Vec3f normal(const cv::Vec3f &loc = {0,0,0}) override { return _base_gen->normal(loc); };
    cv::Vec3f coord(const cv::Vec3f &loc = {0,0,0}) override { return _base_gen->coord(loc); };
protected:
    TrivialSurfacePointer *_gen_ptr;
    ControlPointSurface *_surf;
    CoordGenerator *_base_gen;
};

ControlPointCoords::ControlPointCoords(TrivialSurfacePointer *gen_ptr, ControlPointSurface *surf)
{
    _gen_ptr = gen_ptr;
    _surf = surf;
    _base_gen = _surf->_base->generator(_gen_ptr);
}

void ControlPointCoords::gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale, float coord_scale)
{
    //FIXME does generator create a new generator? need at some point check ownerships of these apis ...
    _base_gen->gen_coords(coords,x,y,w,h,render_scale,coord_scale);
    
    std::cout << "call mod gen_coords " << x << " " << y << _surf->_base->loc(_gen_ptr)*coord_scale << _surf->_base->loc(_surf->_controls[0].ptr)*coord_scale << std::endl;
    
    for(int j=0;j<h;j++)
        for(int i=0;i<h;i++) {
            float ox = 0.1*(i-w/2);
            float oy = 0.1*(j-h/2);
            float sd = ox*ox + oy*oy;
            coords(j,i,0) += 100.0/std::max(1.0f,sd);
        }
}


ControlPointSurface::ControlPointSurface(Surface *base)
{
    _base = base;
}

SurfaceControlPoint::SurfaceControlPoint(Surface *base, SurfacePointer *ptr_, const cv::Vec3f &control)
{
    ptr = ptr_->clone();
    orig_wp = base->coord(ptr_);
    normal = base->normal(ptr_);
    control_point = control;
}

void ControlPointSurface::addControlPoint(SurfacePointer *base_ptr, cv::Vec3f control_point)
{
    _controls.push_back(SurfaceControlPoint(_base, base_ptr, control_point));
    
    std::cout << "add control " << control_point << _base->loc(base_ptr) << _base->loc(_controls[0].ptr) << std::endl;
    
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
    _base->valid(ptr, offset);
}

cv::Vec3f ControlPointSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    std::cout << "FIXME: implement ControlPointSurface::loc()" << std::endl;
    cv::Vec3f(-1,-1,-1);
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
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    TrivialSurfacePointer *gen_ptr = new TrivialSurfacePointer(ptr_inst->loc);
    _base->move(gen_ptr, offset);
    
    return new ControlPointCoords(gen_ptr, this);
}

float ControlPointSurface::pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th)
{
    //surfacepointer is supposed to always stay in the same nominal coordinates - so always refer down to the most lowest level / input / source
    _base->pointTo(ptr, tgt, th);
}
void ControlPointSurface::setBase(QuadSurface *base)
{
    _base = base;
    
    //FIXME reset control points?
}