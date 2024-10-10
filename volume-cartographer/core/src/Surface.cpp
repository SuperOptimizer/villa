#include "vc/core/util/Surface.hpp"

#include "vc/core/io/PointSetIO.hpp"
#include "vc/core/util/Slicing.hpp"

#include "SurfaceHelpers.hpp"

#include <opencv2/imgproc.hpp>

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

cv::Vec2f offsetPoint2d(TrivialSurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = ptr->loc + offset;
    return {p[0], p[1]};
}

//TODO add non-cloning variant?
QuadSurface::QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale)
{
    _points = points.clone();
    //-1 as many times we read with linear interpolation and access +1 locations
    _bounds = {0,0,points.cols-1,points.rows-1};
    _scale = scale;
    _center = {points.cols/2.0/_scale[0],points.rows/2.0/_scale[1],0};
    std::cout << "quad center scale" << _center << _scale <<  _points.size() <<  std::endl;
}

SurfacePointer *QuadSurface::pointer()
{
    return new TrivialSurfacePointer({0,0,0});
}

//there are two coords, outside (nominal) and inside
static cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

static cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}

//FIXME loc & offset_unscaled are redundant!
// static cv::Vec3f offset3(const cv::Vec3f &loc, const cv::Vec3f &offset_scaled, const cv::Vec2f &scale, const cv::Vec3f &offset_unscaled)
// {
//     return loc + cv::Vec3f(offset_scaled[0]*scale[0]+offset_unscaled[0], offset_scaled[1]*scale[1]+offset_unscaled[1], offset_scaled[2]+offset_unscaled[2]);
// }

// static cv::Vec2f offset2(const cv::Vec3f &loc, const cv::Vec3f &offset_scaled, const cv::Vec2f &scale, const cv::Vec3f &offset_unscaled)
// {
//     return cv::Vec2f(loc[0],loc[1]) + cv::Vec2f(offset_scaled[0]*scale[0]+offset_unscaled[0], offset_scaled[1]*scale[1]+offset_unscaled[1]);
// }

void QuadSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    ptr_inst->loc += cv::Vec3f(offset[0]*_scale[0],offset[1]*_scale[1],offset[2]);
}

bool QuadSurface::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec3f p = internal_loc(offset+_center, ptr_inst->loc, _scale);
    
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
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec3f p = internal_loc(offset+_center, ptr_inst->loc, _scale);
    
    return at_int(_points, {p[0],p[1]});
}

cv::Vec3f QuadSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    return nominal_loc(offset+_center, ptr_inst->loc, _scale);
}

cv::Vec3f QuadSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    cv::Vec3f p = internal_loc(offset+_center, ptr_inst->loc, _scale);
    
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

/*void GridCoords::gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale, float coord_scale)
{
    if (render_scale > 1.0 || render_scale < 0.5) {
        std::cout << "FIXME: support wider render scale for GridCoords::gen_coords()" << std::endl;
        return;
    }
    
    coords = xt::zeros<float>({h,w,3});
    cv::Mat_<cv::Vec3f> warped;
    
    float s = 1/coord_scale;
    std::vector<cv::Vec2f> dst = {{0,0},{w,0},{0,h}};
    cv::Vec2f off2d = {_offset[0]*_sx,_offset[1]*_sy};
    std::cout << "using off2d " << off2d << _offset << std::endl;
    std::vector<cv::Vec2f> src = {cv::Vec2f(x*_sx,y*_sy)*s+off2d,cv::Vec2f((x+w)*_sx,y*_sy)*s+off2d,cv::Vec2f(x*_sx,(y+h)*_sy)*s+off2d};
    
    cv::Mat affine = cv::getAffineTransform(src, dst);
    
    cv::warpAffine(*_points, warped, affine, {w,h});
    
    if (_z_off || _offset[2]) {
        // std::cout << "FIX offset for GridCoords::gen_coords!" << std::endl;
        
        cv::Mat_<cv::Vec3f> normals(warped.size());
        for(int j=0;j<h;j++)
            for(int i=0;i<w;i++)
                normals(j, i) = grid_normal(warped, {i,j});
        
        warped += normals*(_z_off+_offset[2]);
    }
    
    #pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            cv::Vec3f point = warped(j,i);
            coords(j,i,0) = point[2]*coord_scale;
            coords(j,i,1) = point[1]*coord_scale;
            coords(j,i,2) = point[0]*coord_scale;
        }
}*/

void QuadSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset)
{
    TrivialSurfacePointer _ptr({0,0,0});
    if (!ptr)
        ptr = &_ptr;
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    
    bool create_normals = normals || offset[2] || ptr_inst->loc[2];
    
    cv::Vec3f offset_internal = internal_loc(offset/scale+_center, ptr_inst->loc, _scale);
    std::cout << "upper left" << _points.size() << offset_internal << ptr_inst->loc << _center << offset << scale << _scale << std::endl;
    
    int w = size.width;
    int h = size.height;

    cv::Mat_<cv::Vec3f> _coords_header;
    cv::Mat_<cv::Vec3f> _normals_header;
    
    if (!coords)
        coords = &_coords_header;
    if (!normals)
        normals = &_normals_header;
    
    coords->create(size);
    cv::Mat_<cv::Vec3f> warped;
    
    std::vector<cv::Vec2f> dst = {{0,0},{w,0},{0,h}};
    cv::Vec2f off2d = {offset_internal[0],offset_internal[1]};
    std::cout << "using off2d " << off2d << std::endl;
    std::vector<cv::Vec2f> src = {off2d,off2d+cv::Vec2f(w*_scale[0]/scale,0),off2d+cv::Vec2f(0,h*_scale[1]/scale)};
    
    cv::Mat affine = cv::getAffineTransform(src, dst);
    
    cv::warpAffine(_points, *coords, affine, size);
    
    if (create_normals) {
        // std::cout << "FIX offset for GridCoords::gen_coords!" << std::endl;
        
        normals->create(size);
        for(int j=0;j<h;j++)
            for(int i=0;i<w;i++)
                (*normals)(j, i) = grid_normal(*coords, {i,j});
        
        warped += (*normals)*offset_internal[2];
    }
    
    (*coords) *= scale;
}
/*CoordGenerator *QuadSurface::generator(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    //without pointer we just use the center == default pointer
    cv::Vec3f total_offset = offset3(0, offset+_center, _scale, {0,0,0});
    
    if (ptr) {
        TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
        assert(ptr_inst);

        total_offset += ptr_inst->loc;
    }
    
    //FIXME implement & use offset for gridcoords
    return new GridCoords(&_points, _scale[0], _scale[1], {total_offset[0]/_scale[0],total_offset[1]/_scale[1],total_offset[2]});
}*/


QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out)
{
    cv::Mat_<cv::Vec3f> points;
    
    TrivialSurfacePointer *trivial_ptr = (TrivialSurfacePointer*)ptr;
    
    std::cout << "ptr" << trivial_ptr->loc << std::endl;
    
    points = derive_regular_region_largesteps(src->_points, trivial_ptr->loc[0]+src->_center[0]*src->_scale[0], trivial_ptr->loc[1]+src->_center[1]*src->_scale[1], step_search, w*step_out/step_search, h*step_out/step_search);
    points = upsample_with_grounding(points, {w,h}, src->_points, src->_scale[0], src->_scale[1]);
    
    return new QuadSurface(points, {1.0/step_out, 1.0/step_out});
}

//just forwards everything but gen_coords ... can we make this more elegant without having to call the specific methods?
/*class ControlPointCoords : public CoordGenerator
{
public:
    ControlPointCoords(TrivialSurfacePointer *gen_ptr, ControlPointSurface *surf);
    void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    void setOffsetZ(float off) override { _base_gen->setOffsetZ(off); };
    float offsetZ() override { return _base_gen->offsetZ(); };
    cv::Vec3f offset() override { return _base_gen->offset(); };
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
    
    cv::Rect bounds(0,0,w,h);
    
    // cv::Vec2f off2d = {_offset[0]*_sx,_offset[1]*_sy};
    // 0 ~ x*_sx*s+off2d.x
    
    for(auto p : _surf->_controls) {
        cv::Vec3f loc = _surf->_base->loc(p.ptr) - cv::Vec3f(_base_gen->offset()[0],_base_gen->offset()[1],0);
        loc *= 1/coord_scale;
        loc -= cv::Vec3f(x,y,0);
        cv::Rect roi(loc[0]-40,loc[1]-40,80,80);
        cv::Rect area = roi & bounds;
        
        PlaneCoords plane(p.control_point, p.normal);
        float delta = plane.scalarp(_surf->_base->coord(p.ptr));
        cv::Vec3f move = delta*p.normal;

        for(int j=area.y;j<area.y+area.height;j++)
            for(int i=area.x;i<area.x+area.width;i++) {
                float w = sdist(loc, cv::Vec3f(i,j,0));
                w = exp(-w/(20*20));
                coords(j,i,2) += w*move[0];
                coords(j,i,1) += w*move[1];
                coords(j,i,0) += w*move[2];
            }
    }
}*/

SurfaceControlPoint::SurfaceControlPoint(Surface *base, SurfacePointer *ptr_, const cv::Vec3f &control)
{
    ptr = ptr_->clone();
    orig_wp = base->coord(ptr_);
    normal = base->normal(ptr_);
    control_point = control;
}

ControlPointSurface::ControlPointSurface(QuadSurface *base)
{
    _base = base;
}

void ControlPointSurface::addControlPoint(SurfacePointer *base_ptr, cv::Vec3f control_point)
{
    _controls.push_back(SurfaceControlPoint(_base, base_ptr, control_point));
    
    std::cout << "add control " << control_point << _base->loc(base_ptr) << _base->loc(_controls[0].ptr) << _base->_center << _base->loc(_base->pointer()) << std::endl;
    
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

/*CoordGenerator *ControlPointSurface::generator(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    TrivialSurfacePointer *gen_ptr = new TrivialSurfacePointer(ptr_inst->loc);
    _base->move(gen_ptr, offset);
    
    return new ControlPointCoords(gen_ptr, this);
}*/

void ControlPointSurface::gen(cv::Mat_<cv::Vec3f> *coords_, cv::Mat_<cv::Vec3f> *normals_, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset)
{
    std::cout << "corr gen " << _controls.size() << std::endl;
    cv::Mat_<cv::Vec3f> _coords_local;
    // cv::Mat_<cv::Vec3f> _normals_local;
    
    cv::Mat_<cv::Vec3f> *coords = coords_;
    // cv::Mat_<cv::Vec3f> *normals = normals_;
    
    if (!coords)
        coords = &_coords_local;
    // if (!normals)
        // normals = &_normals_local;
    
    TrivialSurfacePointer _ptr_local({0,0,0});
    if (!ptr)
        ptr = &_ptr_local;
    
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);

    _base->gen(coords, normals_, size, ptr, scale, offset);
    
    int w = size.width;
    int h = size.height;
    cv::Rect bounds(0,0,w,h);
    
    cv::Vec3f upper_left = nominal_loc(offset/scale+_base->_center, ptr_inst->loc, _base->_scale);
    
    float z_offset = upper_left[2];
    upper_left[2] = 0;
    
    //FIXME implement z_offset
    
    std::cout << "offset" << upper_left << _base->_center << ptr_inst->loc << std::endl;
    
    for(auto p : _controls) {
        cv::Vec3f p_loc = nominal_loc(_base->loc(p.ptr) + offset/scale + _base->_center, ptr_inst->loc, _base->_scale)  - upper_left;
        p_loc *= scale;
        std::cout << p_loc << _base->loc(p.ptr) << ptr_inst->loc << std::endl;
        cv::Rect roi(p_loc[0]-40,p_loc[1]-40,80,80);
        cv::Rect area = roi & bounds;
        
        PlaneCoords plane(p.control_point, p.normal);
        float delta = plane.scalarp(_base->coord(p.ptr));
        cv::Vec3f move = delta*p.normal;
        
        std::cout << area << roi << bounds << std::endl;
        
        for(int j=area.y;j<area.y+area.height;j++)
            for(int i=area.x;i<area.x+area.width;i++) {
                //TODO correct by scale!
                float w = sdist(p_loc, cv::Vec3f(i,j,0));
                w = exp(-w/(20*20));
                (*coords)(j,i) += w*move;
            }
    }
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
    std::cout << "ERROR implement ControlPointSurface::setBase()" << std::endl;
}

/*cv::Mat_<cv::Vec3f> surf_alpha_integ_dbg(z5::Dataset *ds, ChunkCache *chunk_cache, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<cv::Vec3f> &normals)
{
    cv::Mat_<cv::Vec3f> res;
    
    cv::Mat_<float> integ(points.size(), 0);
    cv::Mat_<float> integ_blur(points.size(), 0);
    cv::Mat_<float> transparent(points.size(), 1);
    cv::Mat_<float> blur(points.size(), 0);
    cv::Mat_<float> integ_z(points.size(), 0);
    
    for(int n=0;n<21;n++) {
        xt::xarray<uint8_t> raw_extract;
        // coords = points_reg*2.0;
        float off = (n-5)*0.5;
        readInterpolated3D(raw_extract, ds, xt_from_mat((points+normals*off)*0.5), chunk_cache);
        cv::Mat_<uint8_t> slice = cv::Mat(raw_extract.shape(0), raw_extract.shape(1), CV_8U, raw_extract.data());
        
        // char buf[64];
        // sprintf(buf, "slice%02d.tif", n);
        // cv::imwrite(buf, slice);
        
        cv::Mat floatslice;
        slice.convertTo(floatslice, CV_32F, 1/255.0);
        
        cv::GaussianBlur(floatslice, blur, {7,7}, 0);
        cv::Mat opaq_slice = blur;
        
        float low = 0.0; //map to 0
        float up = 1.0; //map to 1
        opaq_slice = (opaq_slice-low)/(up-low);
        opaq_slice = cv::min(opaq_slice,1);
        opaq_slice = cv::max(opaq_slice,0);
        
        // sprintf(buf, "opaq%02d.tif", n);
        // cv::imwrite(buf, opaq_slice);
        
        printf("vals %d i t o b v: %f %f %f %f\n", n, integ.at<float>(500,600), transparent.at<float>(500,600), opaq_slice.at<float>(500,600), blur.at<float>(500,600), floatslice.at<float>(500,600));
        
        cv::Mat joint = transparent.mul(opaq_slice);
        integ += joint.mul(floatslice);
        integ_blur += joint.mul(blur);
        integ_z += joint * off;
        transparent = transparent-joint;
        
        // sprintf(buf, "transp%02d.tif", n);
        // cv::imwrite(buf, transparent);
        // 
        // sprintf(buf, "opaq2%02d.tif", n);
        // cv::imwrite(buf, opaq_slice);
        
        printf("res %d i t: %f %f\n", n, integ.at<float>(500,600), transparent.at<float>(500,600));
        
        // avgimg = avgimg + floatslice;
        // cv::imwrite(buf, avgimg/(n+1));
        
        // slices.push_back(slice);
        // for(int j=0;j<points.rows;j++)
        //     for(int i=0;i<points.cols;i++) {
        //         //found == 0: still searching for first time < 50!
        //         //found == 1: record < 50 start looking for >= 50 to stop
        //         //found == 2: done, found border
        //         if (slice(j,i) < 40 && found(j,i) <= 1) {
        //             height(j,i) = n+1;
        //             found(j,i) = 1;
        //         }
        //         else if (slice(j,i) >= 40 && found(j,i) == 1) {
        //             found(j,i) = 2;
        //         }
        //     }
    }        // slices.push_back(slice);
    
    integ /= (1-transparent);
    integ_blur /= (1-transparent);
    integ_z /= (1-transparent);
    
    cv::imwrite("blended.tif", integ);
    cv::imwrite("blended_blur.tif", integ_blur);
    cv::imwrite("blended_comp1.tif", integ/(integ_blur+0.5));
    cv::imwrite("blended_comp3.tif", integ-integ_blur+0.5);
    cv::imwrite("blended_comp2.tif", integ/(integ_blur+0.01));
    cv::imwrite("tranparency.tif", transparent);
    
    // for(int j=0;j<points.rows;j++)
    //     for(int i=0;i<points.cols;i++)
    //         if (found(j,i) == 1)
    //             height(j,i) = 0;
    
    //never change opencv, never change ...
    
    // cv::cvtColor(height, mul, cv::COLOR_GRAY2BGR);
    // cv::imwrite("max.tif", maximg);
    
    cv::Mat mul;
    cv::cvtColor(integ_z, mul, cv::COLOR_GRAY2BGR);
    cv::Mat_<cv::Vec3f> new_surf = points + normals.mul(mul);
    cv::Mat_<cv::Vec3f> new_surf_1 = new_surf + normals;
    cv::Mat_<cv::Vec3f> new_surf_n1 = new_surf - normals;
    //     
    xt::xarray<uint8_t> img;
    readInterpolated3D(img, ds, xt_from_mat(new_surf*0.5), chunk_cache);
    cv::Mat_<uint8_t> slice = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    //     
    printf("writ slice!\n");
    cv::imwrite("new_surf.tif", slice);
    
    readInterpolated3D(img, ds, xt_from_mat(new_surf_1*0.5), chunk_cache);
    slice = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    cv::imwrite("new_surf1.tif", slice);
    
    readInterpolated3D(img, ds, xt_from_mat(new_surf_n1*0.5), chunk_cache);
    slice = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    cv::imwrite("new_surf-1.tif", slice);
    
    // cv::Mat_<float> height_vis = height/21;
    // height_vis = cv::min(height_vis,1-height_vis)*2;
    // cv::imwrite("off.tif", height_vis);
    
    //now big question: how far away from average is the new surf!
    
    //     cv::Mat avg_surf;
    //     cv::GaussianBlur(new_surf, avg_surf, {7,7}, 0);
    //     
    //     readInterpolated3D(img, ds, xt_from_mat(avg_surf*0.5), chunk_cache);
    //     slice = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    //     
    //     cv::imwrite("avg_surf.tif", slice);
    //     
    //     
    //     cv::Mat_<float> rel_height(points.size(), 0);
    //     
    //     cv::Mat_<cv::Vec3f> dist = avg_surf-new_surf;
    //     
    //     #pragma omp parallel for
    //     for(int j=0;j<points.rows;j++)
    //         for(int i=0;i<points.cols;i++) {
    //             rel_height(j,i) = cv::norm(dist(j,i));
    //         }
    //         
    //         cv::imwrite("rel_height.tif", rel_height);
    
    return new_surf;
}*/

//just forwards everything but gen_coords ... can we make this more elegant without having to call the specific methods?
/*class RefineCompCoords : public CoordGenerator
{
public:
    RefineCompCoords(TrivialSurfacePointer *gen_ptr, RefineCompSurface *surf);
    void gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale = 1.0, float coord_scale = 1.0) override;
    void setOffsetZ(float off) override { _base_gen->setOffsetZ(off); };
    float offsetZ() override { return _base_gen->offsetZ(); };
    cv::Vec3f offset() override { return _base_gen->offset(); };
    cv::Vec3f normal(const cv::Vec3f &loc = {0,0,0}) override { return _base_gen->normal(loc); };
    cv::Vec3f coord(const cv::Vec3f &loc = {0,0,0}) override { return _base_gen->coord(loc); };
protected:
    TrivialSurfacePointer *_gen_ptr;
    RefineCompSurface *_surf;
    // CoordGenerator *_base_gen;
};

RefineCompCoords::RefineCompCoords(TrivialSurfacePointer *gen_ptr, RefineCompSurface *surf)
{
    _gen_ptr = gen_ptr;
    _surf = surf;
    _base_gen = _surf->_base->generator(_gen_ptr);
}

void RefineCompCoords::gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale, float coord_scale)
{
    //FIXME does generator create a new generator? need at some point check ownerships of these apis ...
    _base_gen->gen_coords(coords,x,y,w,h,render_scale,coord_scale);
    
//     cv::Rect bounds(0,0,w,h);
//     
//     // cv::Vec2f off2d = {_offset[0]*_sx,_offset[1]*_sy};
//     // 0 ~ x*_sx*s+off2d.x
//     
//     for(auto p : _surf->_controls) {
//         cv::Vec3f loc = _surf->_base->loc(p.ptr) - cv::Vec3f(_base_gen->offset()[0],_base_gen->offset()[1],0);
//         loc *= 1/coord_scale;
//         loc -= cv::Vec3f(x,y,0);
//         cv::Rect roi(loc[0]-40,loc[1]-40,80,80);
//         cv::Rect area = roi & bounds;
//         
//         PlaneCoords plane(p.control_point, p.normal);
//         float delta = plane.scalarp(_surf->_base->coord(p.ptr));
//         cv::Vec3f move = delta*p.normal;
//         
//         for(int j=area.y;j<area.y+area.height;j++)
//             for(int i=area.x;i<area.x+area.width;i++) {
//                 float w = sdist(loc, cv::Vec3f(i,j,0));
//                 w = exp(-w/(20*20));
//                 coords(j,i,2) += w*move[0];
//                 coords(j,i,1) += w*move[1];
//                 coords(j,i,0) += w*move[2];
//             }
//     }
}*/

RefineCompSurface::RefineCompSurface(Surface *base)
{
    _base = base;
}

SurfacePointer *RefineCompSurface::pointer()
{
    return _base->pointer();
}

void RefineCompSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    _base->move(ptr, offset);
}
bool RefineCompSurface::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    _base->valid(ptr, offset);
}

cv::Vec3f RefineCompSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    std::cout << "FIXME: implement RefineCompSurface::loc()" << std::endl;
    cv::Vec3f(-1,-1,-1);
}

cv::Vec3f RefineCompSurface::coord(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    //FIXME should actually check distance between control point and surface ...
    // return _base(ptr) + _normal*10;
    std::cout << "FIXME: implement RefineCompSurface::coord()" << std::endl;
    cv::Vec3f(-1,-1,-1);
}

cv::Vec3f RefineCompSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    std::cout << "FIXME: implement RefineCompSurface::normal()" << std::endl;
    cv::Vec3f(-1,-1,-1);
}

// CoordGenerator *RefineCompSurface::generator(SurfacePointer *ptr, const cv::Vec3f &offset)
// {
//     TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
//     assert(ptr_inst);
//     
//     TrivialSurfacePointer *gen_ptr = new TrivialSurfacePointer(ptr_inst->loc);
//     _base->move(gen_ptr, offset);
//     
//     return new RefineCompCoords(gen_ptr, this);
// }

float RefineCompSurface::pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th)
{
    //surfacepointer is supposed to always stay in the same nominal coordinates - so always refer down to the most lowest level / input / source
    _base->pointTo(ptr, tgt, th);
}

void RefineCompSurface::setBase(QuadSurface *base)
{
    _base = base;
}