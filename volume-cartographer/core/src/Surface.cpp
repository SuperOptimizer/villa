#include "vc/core/util/Surface.hpp"

#include "vc/core/io/PointSetIO.hpp"
#include "vc/core/util/Slicing.hpp"

#include "SurfaceHelpers.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//TODO remove
#include <opencv2/highgui.hpp>

#include <unordered_map>

cv::Vec2f offsetPoint2d(TrivialSurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Vec3f p = ptr->loc + offset;
    return {p[0], p[1]};
}

//NOTE we have 3 coordiante systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
static cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

static cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
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

//given origin and normal, return the normalized vector v which describes a point : origin + v which lies in the plane and maximizes v.x at the cost of v.y,v.z
static cv::Vec3f vx_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    //impossible
    if (n[1] == 0 && n[2] == 0)
        return {0,0,0};

    //also trivial
    if (n[0] == 0)
        return {1,0,0};

    cv::Vec3f v = {1,0,0};

    if (n[1] == 0) {
        v[1] = 0;
        //either n1 or n2 must be != 0, see first edge case
        v[2] = -n[0]/n[2];
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    if (n[2] == 0) {
        //either n1 or n2 must be != 0, see first edge case
        v[1] = -n[0]/n[1];
        v[2] = 0;
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    v[1] = -n[0]/(n[1]+n[2]);
    v[2] = v[1];
    cv::normalize(v, v, 1,0, cv::NORM_L2);

    return v;
}

static cv::Vec3f vy_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    cv::Vec3f v = vx_from_orig_norm({o[1],o[0],o[2]}, {n[1],n[0],n[2]});
    return {v[1],v[0],v[2]};
}

static void vxy_from_normal(cv::Vec3f orig, cv::Vec3f normal, cv::Vec3f &vx, cv::Vec3f &vy)
{
    vx = vx_from_orig_norm(orig, normal);
    vy = vy_from_orig_norm(orig, normal);

    //TODO will there be a jump around the midpoint?
    if (abs(vx[0]) >= abs(vy[1]))
        vy = cv::Mat(normal).cross(cv::Mat(vx));
    else
        vx = cv::Mat(normal).cross(cv::Mat(vy));

    //FIXME probably not the right way to normalize the direction?
    if (vx[0] < 0)
        vx *= -1;
    if (vy[1] < 0)
        vy *= -1;
}

void PlaneSurface::update()
{
    cv::Vec3f vx, vy;

    vxy_from_normal(_origin,_normal,vx,vy);

    std::vector <cv::Vec3f> src = {_origin,_origin+_normal,_origin+vx,_origin+vy};
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

void PlaneSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset)
{
    TrivialSurfacePointer _ptr({0,0,0});
    if (!ptr)
        ptr = &_ptr;
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);

    bool create_normals = normals || offset[2] || ptr_inst->loc[2];

    cv::Vec3f total_offset = internal_loc(offset/scale, ptr_inst->loc, {1,1});
    // std::cout << "PlaneCoords::gen upper left" << upper_left_actual /*<< ptr_inst->loc*/ << origin << offset << scale << std::endl;

    int w = size.width;
    int h = size.height;

    cv::Mat_<cv::Vec3f> _coords_header;
    cv::Mat_<cv::Vec3f> _normals_header;

    if (!coords)
        coords = &_coords_header;
    if (!normals)
        normals = &_normals_header;

    coords->create(size);

    if (create_normals) {
        // std::cout << "FIX offset for GridCoords::gen_coords!" << std::endl;

        normals->create(size);
        // for(int j=0;j<h;j++)
        //     for(int i=0;i<w;i++)
        //         (*normals)(j, i) = grid_normal(*coords, {i,j});
        //
        // *coords += (*normals)*upper_left_actual[2];
    }

    cv::Vec3f vx, vy;
    vxy_from_normal(_origin,_normal,vx,vy);

    float m = 1/scale;

    cv::Vec3f use_origin = _origin + _normal*total_offset[2];

#pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            // cv::Vec3f p = vx*(i*m+total_offset[0]) + vy*(j*m+total_offset[1]) + _normal*total_offset[2] + origin;
            // (*coords)(j,i)[0] = p[2];
            // (*coords)(j,i)[1] = p[1];
            // (*coords)(j,i)[2] = p[0];
            (*coords)(j,i) = vx*(i*m+total_offset[0]) + vy*(j*m+total_offset[1]) + use_origin;
        }
}

SurfacePointer *PlaneSurface::pointer()
{
    return new TrivialSurfacePointer({0,0,0});
}

void PlaneSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);

    ptr_inst->loc += offset;
}

cv::Vec3f PlaneSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);

    return ptr_inst->loc+offset;
}

cv::Vec3f PlaneSurface::coord(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> coords;

    gen(&coords, nullptr, {1,1}, ptr, 1.0, offset);

    return coords(0,0);
}

cv::Vec3f PlaneSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _normal;
}

//TODO add non-cloning variant?
QuadSurface::QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale)
{
    _points = points.clone();
    //-1 as many times we read with linear interpolation and access +1 locations
    _bounds = {0,0,points.cols-1,points.rows-1};
    _scale = scale;
    _center = {points.cols/2.0/_scale[0],points.rows/2.0/_scale[1],0};
}

QuadSurface *smooth_vc_segmentation(QuadSurface *src)
{
    cv::Mat_<cv::Vec3f> points = smooth_vc_segmentation(src->_points);
    
    double sx, sy;
    vc_segmentation_scales(points, sx, sy);
    
    return new QuadSurface(points, {sx,sy});
}

SurfacePointer *QuadSurface::pointer()
{
    return new TrivialSurfacePointer({0,0,0});
}

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

    cv::Rect bounds = {0,0,_points.cols-2,_points.rows-2};
    if (!bounds.contains({p[0],p[1]}))
        return {-1,-1,-1};
        
    return at_int(_points, {p[0],p[1]});
}

cv::Vec3f QuadSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    return nominal_loc(offset, ptr_inst->loc, _scale);
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

static float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrt(d.dot(d));

    return abs(l-t_dist);
}

static float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds)
{
    float sum = 0;
    for(int i=0;i<tgts.size();i++) {
        float d = tdist(v, tgts[i], tds[i]);
        sum += d*d;
    }

    return sum;
}

//search location in points where we minimize error to multiple objectives using iterated local search
//tgts,tds -> distance to some POIs
//plane -> stay on plane
float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, float init_step, float min_step)
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return -1;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    if (plane) {
        float d = plane->pointDist(val);
        best += d*d;
    }
    float res;

    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;


    // std::cout << "init " << best << tgts[0] << val << loc << "\n";


    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;

            if (!boundary.contains({cand[0],cand[1]})) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return -1;
            }


            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds);
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                // std::cout << res << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
            // std::cout << "(" << res << val << step << cand << "\n";
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step < min_step)
            break;
    }

    // std::cout << "best" << best << out << "\n" <<  std::endl;
    return best;
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
float QuadSurface::pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th, int max_iters)
{
    TrivialSurfacePointer *tgt_ptr = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(tgt_ptr);

    cv::Vec2f loc = cv::Vec2f(tgt_ptr->loc[0],tgt_ptr->loc[1]) + cv::Vec2f(_center[0]*_scale[0],_center[1]*_scale[1]);
    cv::Vec3f _out;
    
    cv::Vec2f step_small = {std::max(1.0f,_scale[0]),std::max(1.0f,_scale[1])};
    float min_mul = std::min(0.1*_points.cols/_scale[0],0.1*_points.rows/_scale[1]);
    cv::Vec2f step_large = {min_mul*_scale[0],min_mul*_scale[1]};

    float dist = search_min_loc(_points, loc, _out, tgt, step_small, _scale[0]*0.01);
    
    if (dist < th && dist >= 0) {
        tgt_ptr->loc = cv::Vec3f(loc[0],loc[1],0) - cv::Vec3f(_center[0]*_scale[0],_center[1]*_scale[1],0);
        return dist;
    }
    
    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(_points.cols/_scale[0]+_points.rows/_scale[1]);
    
    //FIXME is this excessive?
    for(int r=0;r<max_iters;r++) {
        loc = {1 + (rand() % _points.cols-3), 1 + (rand() % _points.rows-3)};
        
        float dist = search_min_loc(_points, loc, _out, tgt, step_large, _scale[0]*0.01);
        
        if (dist < th && dist >= 0) {
            tgt_ptr->loc = cv::Vec3f(loc[0],loc[1],0) - cv::Vec3f(_center[0]*_scale[0],_center[1]*_scale[1],0);
            return dist;
        } else if (dist >= 0 && dist < min_dist) {
            min_loc = loc;
            min_dist = dist;
        }
    }
    
    tgt_ptr->loc = cv::Vec3f(min_loc[0],min_loc[1],0) - cv::Vec3f(_center[0]*_scale[0],_center[1]*_scale[1],0);
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

bool face_contains_vertex(cv::Vec3i face, int vertex)
{
    if (face[0] == vertex)
        return true;
    if (face[1] == vertex)
        return true;
    if (face[2] == vertex)
        return true;
    return false;
}

//try to interpret triangle surface as quads - only works in cases where an actual quad surface is stored as a triangle surface
QuadSurface *load_quad_from_obj(const std::string &path)
{
    //triangle ID to 3d location
    std::unordered_map<int,cv::Vec3f> vertices;
    //any face corner vertex id to the actual face id
    std::unordered_multimap<int,cv::Vec3i> faces;

    std::ifstream obj(path);
    std::string line;
    while (std::getline(obj, line))
    {
        if (line.size() <  7)
            continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream iss(line);
            float x, y, z;
            char v;
            if (!(iss >> v >> x >> y >> z)) {
                //something went wrong
                return nullptr;
            }
            vertices[vertices.size()] = {x,y,z};
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream iss(line);
            std::string idstring;
            int a, b, c;
            char f;
            if (!(iss >> f >> idstring)) {
                //something went wrong
                return nullptr;
            }
            std::replace(idstring.begin(), idstring.end(), '/', ' ');
            iss.str(idstring);
            if (!(iss >> a >> b >> c)) {
                //something went wrong
                return nullptr;
            }
            cv::Vec3i face = {a,b,c};
            faces.insert({{a,face},{b,face},{c,face}});
        }
    }

    return nullptr;
}

void QuadSurface::gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset)
{
    TrivialSurfacePointer _ptr({0,0,0});
    if (!ptr)
        ptr = &_ptr;
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    
    bool create_normals = normals || offset[2] || ptr_inst->loc[2];
    
    cv::Vec3f upper_left_actual = internal_loc(offset/scale+_center, ptr_inst->loc, _scale);
    
    int w = size.width;
    int h = size.height;

    cv::Mat_<cv::Vec3f> _coords_header;
    cv::Mat_<cv::Vec3f> _normals_header;
    
    if (!coords)
        coords = &_coords_header;
    if (!normals)
        normals = &_normals_header;
    
    coords->create(size);
    
    std::vector<cv::Vec2f> dst = {{0,0},{w,0},{0,h}};
    cv::Vec2f off2d = {upper_left_actual[0],upper_left_actual[1]};
    std::vector<cv::Vec2f> src = {off2d,off2d+cv::Vec2f(w*_scale[0]/scale,0),off2d+cv::Vec2f(0,h*_scale[1]/scale)};
    
    cv::Mat affine = cv::getAffineTransform(src, dst);
    
    cv::warpAffine(_points, *coords, affine, size);
    
    if (create_normals) {
        // std::cout << "FIX offset for GridCoords::gen_coords!" << std::endl;
        
        normals->create(size);
        for(int j=0;j<h;j++)
            for(int i=0;i<w;i++)
                (*normals)(j, i) = grid_normal(*coords, {i,j});
        
        *coords += (*normals)*upper_left_actual[2];
    }
}

QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out)
{
    cv::Mat_<cv::Vec3f> points;
    
    TrivialSurfacePointer *trivial_ptr = (TrivialSurfacePointer*)ptr;
    
    std::cout << "ptr" << trivial_ptr->loc << std::endl;

    cv::Mat_<cv::Vec2f> locs;
    
    points = derive_regular_region_largesteps(src->_points, locs, trivial_ptr->loc[0]+src->_center[0]*src->_scale[0], trivial_ptr->loc[1]+src->_center[1]*src->_scale[1], step_search, w*step_out/step_search, h*step_out/step_search);
    points = upsample_with_grounding(points, locs, {w,h}, src->_points, src->_scale[0], src->_scale[1]);
    
    return new QuadSurface(points, {1.0/step_out, 1.0/step_out});
}

SurfaceControlPoint::SurfaceControlPoint(Surface *base, SurfacePointer *ptr_, const cv::Vec3f &control)
{
    ptr = ptr_->clone();
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

SurfacePointer *DeltaSurface::pointer()
{
    return _base->pointer();
}


void DeltaSurface::move(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    _base->move(ptr, offset);
}

bool DeltaSurface::valid(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _base->valid(ptr, offset);
}

cv::Vec3f DeltaSurface::loc(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _base->loc(ptr, offset);
}

cv::Vec3f DeltaSurface::coord(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _base->coord(ptr, offset);
}

cv::Vec3f DeltaSurface::normal(SurfacePointer *ptr, const cv::Vec3f &offset)
{
    return _base->normal(ptr, offset);
}

float DeltaSurface::pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th, int max_iters)
{
    return _base->pointTo(ptr, tgt, th, max_iters);
}


void ControlPointSurface::addControlPoint(SurfacePointer *base_ptr, cv::Vec3f control_point)
{
    _controls.push_back(SurfaceControlPoint(this, base_ptr, control_point));
    
}

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
    
    //FIXME can we do this without assuming quad base? I think so ...
    
    cv::Vec3f upper_left_nominal = nominal_loc(offset/scale, ptr_inst->loc, dynamic_cast<QuadSurface*>(_base)->_scale);
    
    float z_offset = upper_left_nominal[2];
    upper_left_nominal[2] = 0;
    
    //FIXME implement z_offset
    
    for(auto p : _controls) {
        cv::Vec3f p_loc = nominal_loc(loc(p.ptr), ptr_inst->loc, dynamic_cast<QuadSurface*>(_base)->_scale)  - upper_left_nominal;
        std::cout << p_loc << p_loc*scale <<  loc(p.ptr) << ptr_inst->loc << std::endl;
        p_loc *= scale;
        cv::Rect roi(p_loc[0]-40,p_loc[1]-40,80,80);
        cv::Rect area = roi & bounds;
        
        PlaneSurface plane(p.control_point, p.normal);
        float delta = plane.scalarp(coord(p.ptr));
        cv::Vec3f move = delta*p.normal;
        
        std::cout << area << roi << bounds << move << p.control_point << p.normal << coord(p.ptr) << std::endl;
        
        for(int j=area.y;j<area.y+area.height;j++)
            for(int i=area.x;i<area.x+area.width;i++) {
                //TODO correct by scale!
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

void RefineCompSurface::gen(cv::Mat_<cv::Vec3f> *coords_, cv::Mat_<cv::Vec3f> *normals_, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset)
{
    cv::Mat_<cv::Vec3f> _coords_local;
    cv::Mat_<cv::Vec3f> _normals_local;
    
    cv::Mat_<cv::Vec3f> *coords = coords_;
    cv::Mat_<cv::Vec3f> *normals = normals_;
    
    if (!coords)
        coords = &_coords_local;
    if (!normals)
    normals = &_normals_local;
    
    TrivialSurfacePointer _ptr_local({0,0,0});
    if (!ptr)
        ptr = &_ptr_local;
    
    TrivialSurfacePointer *ptr_inst = dynamic_cast<TrivialSurfacePointer*>(ptr);
    assert(ptr_inst);
    
    _base->gen(coords, normals, size, ptr, scale, offset);
    
    cv::Mat_<cv::Vec3f> res;
    
    // cv::Mat_<float> integ(size, 0);
    // cv::Mat_<float> integ_blur(size, 0);
    cv::Mat_<float> transparent(size, 1);
    cv::Mat_<float> blur(size, 0);
    cv::Mat_<float> integ_z(size, 0);
    
    for(int n=0;n<21;n++) {
        cv::Mat_<uint8_t> slice;
        float off = (n-5);
        readInterpolated3D(slice, _ds, (*coords+*normals*off)*scale, _cache);
        
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
        
        // printf("vals %d i t o b v: %f %f %f %f\n", n, integ.at<float>(500,600), transparent.at<float>(500,600), opaq_slice.at<float>(500,600), blur.at<float>(500,600), floatslice.at<float>(500,600));
        
        cv::Mat joint = transparent.mul(opaq_slice);
        // integ += joint.mul(floatslice);
        // integ_blur += joint.mul(blur);
        integ_z += joint * off * scale;
        transparent = transparent-joint;
    }
    
    integ_z /= (1-transparent);
    
    //NOTE could be used as an additional output layer to improv visualization!
    // integ /= (1-transparent);
    // integ_blur /= (1-transparent);
    // cv::imwrite("blended_comp1.tif", integ/(integ_blur+0.5));

    cv::Mat mul;
    cv::cvtColor(integ_z, mul, cv::COLOR_GRAY2BGR);
    *coords += (*normals).mul(mul+1+offset[2]);
}

//TODO check if this actually works?!
void set_block(cv::Mat_<uint8_t> &block, const cv::Vec3f &last_loc, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x1 = (loc[0]-roi.x)/step;
    int y1 = (loc[1]-roi.y)/step;
    int x2 = (last_loc[0]-roi.x)/step;
    int y2 = (last_loc[1]-roi.y)/step;

    if (x1 < 0 || y1 < 0 || x1 >= block.cols || y1 >= block.rows)
        return;
    if (x2 < 0 || y2 < 0 || x2 >= block.cols || y2 >= block.rows)
        return;

    if (x1 == x2 && y1 == y2)
        block(y1, x1) = 1;
    else
        cv::line(block, {x1,y1},{x2,y2}, 3);
}

uint8_t get_block(const cv::Mat_<uint8_t> &block, const cv::Vec3f &loc, const cv::Rect &roi, float step)
{
    int x = (loc[0]-roi.x)/step;
    int y = (loc[1]-roi.y)/step;

    if (x < 0 || y < 0 || x >= block.cols || y >= block.rows)
        return 1;

    return block(y, x);
}

void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step)
{
    //start with random points and search for a plane intersection

    float block_step = 0.5*step;

    cv::Mat_<uint8_t> block(cv::Size(plane_roi.width/block_step, plane_roi.height/block_step), 0);

    cv::Rect grid_bounds(1,1,points.cols-2,points.rows-2);

    for(int r=0;r<100;r++) {
        std::vector<cv::Vec3f> seg;
        std::vector<cv::Vec2f> seg_loc;
        std::vector<cv::Vec3f> seg2;
        std::vector<cv::Vec2f> seg_loc2;
        cv::Vec2f loc;
        cv::Vec2f loc2;
        cv::Vec2f loc3;
        cv::Vec3f point;
        cv::Vec3f point2;
        cv::Vec3f point3;
        cv::Vec3f plane_loc;
        cv::Vec3f last_plane_loc;
        float dist = -1;


        //initial points
        for(int i=0;i<100;i++) {
            loc = {std::rand() % (points.cols-1), std::rand() % (points.rows-1)};
            point = at_int(points, loc);

            plane_loc = plane->project(point);
            if (!plane_roi.contains({plane_loc[0],plane_loc[1]}))
                continue;

                dist = min_loc(points, loc, point, {}, {}, plane);

                plane_loc = plane->project(point);
                if (!plane_roi.contains({plane_loc[0],plane_loc[1]}))
                    dist = -1;

                    if (get_block(block, plane_loc, plane_roi, block_step))
                        dist = -1;

            if (dist >= 0 && dist <= 1)
                break;
        }

        // std::cout << loc << " init at dist " << dist << std::endl;

        if (dist < 0 || dist > 1)
            continue;

        seg.push_back(point);
        seg_loc.push_back(loc);

        //point2
        loc2 = loc;
        //search point at distance of 1 to init point
        dist = min_loc(points, loc2, point2, {point}, {1}, plane);

        if (dist < 0 || dist > 1)
            continue;

        seg.push_back(point2);
        seg_loc.push_back(loc2);

        last_plane_loc = plane->project(point);
        plane_loc = plane->project(point2);
        set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
        last_plane_loc = plane_loc;

        //go one direction
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains({loc3[0],loc3[1]}))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.5);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.5);

                if (dist < 0 || dist > 1)
                    break;

            seg.push_back(point3);
            seg_loc.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);
            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        //now the other direction
        loc2 = seg_loc[0];
        loc = seg_loc[1];
        point2 = seg[0];
        point = seg[1];

        last_plane_loc = plane->project(point2);

        //FIXME repeat by not copying code ...
        for(int n=0;n<100;n++) {
            //now search following points
            cv::Vec2f loc3 = loc2+loc2-loc;

            if (!grid_bounds.contains({loc3[0],loc3[1]}))
                break;

                point3 = at_int(points, loc3);

                //search point close to prediction + dist 1 to last point
                dist = min_loc(points, loc3, point3, {point,point2,point3}, {2*step,step,0}, plane, 0.5);

                //then refine
                dist = min_loc(points, loc3, point3, {point2}, {step}, plane, 0.5);

                if (dist < 0 || dist > 1)
                    break;

            seg2.push_back(point3);
            seg_loc2.push_back(loc3);
            point = point2;
            point2 = point3;
            loc = loc2;
            loc2 = loc3;

            plane_loc = plane->project(point3);
            if (get_block(block, plane_loc, plane_roi, block_step))
                break;

            set_block(block, last_plane_loc, plane_loc, plane_roi, block_step);
            last_plane_loc = plane_loc;
        }

        std::reverse(seg2.begin(), seg2.end());
        std::reverse(seg_loc2.begin(), seg_loc2.end());

        seg2.insert(seg2.end(), seg.begin(), seg.end());
        seg_loc2.insert(seg_loc2.end(), seg_loc.begin(), seg_loc.end());


        seg_vol.push_back(seg2);
        seg_grid.push_back(seg_loc2);
    }
}
