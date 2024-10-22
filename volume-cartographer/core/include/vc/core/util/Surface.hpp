#pragma once

#include <opencv2/core.hpp> 

class QuadSurface;
class ChunkCache;

namespace z5 {
    class Dataset;
}


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

QuadSurface *load_quad_from_vcps(const std::string &path);
QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search = 100, int step_out = 5);
QuadSurface *smooth_vc_segmentation(QuadSurface *src);

//base surface class
class Surface
{
public:    
    // a pointer in some central location
    virtual SurfacePointer *pointer() = 0;
    
    //move pointer within internal coordinate system
    virtual void move(SurfacePointer *ptr, const cv::Vec3f &offset) = 0;
    //does the pointer location contain valid surface data
    virtual bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //nominal pointer coordinates (in "output" coordinates)
    virtual cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //read coord at pointer location, potentially with (3) offset
    virtual cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual float pointTo(SurfacePointer *ptr, const cv::Vec3f &coord, float th) = 0;
    //coordgenerator relative to ptr&offset
    //needs to be deleted after use
    virtual void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset) = 0;
    //not yet
    // virtual void normal(SurfacePointer *ptr, cv::Vec3f offset);
};

class PlaneSurface : public Surface
{
public:
    //Surface API FIXME
    SurfacePointer *pointer() override;
    void move(SurfacePointer *ptr, const cv::Vec3f &offset);
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override { return true; };
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &coord, float th) override { abort(); };

    PlaneSurface() {};
    PlaneSurface(cv::Vec3f origin_, cv::Vec3f normal_);

    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset) override;

    float pointDist(cv::Vec3f wp);
    cv::Vec3f project(cv::Vec3f wp, float render_scale = 1.0, float coord_scale = 1.0);
    void setNormal(cv::Vec3f normal);
    void setOrigin(cv::Vec3f origin);
    cv::Vec3f origin();
    float scalarp(cv::Vec3f point) const;
protected:
    void update();
    cv::Vec3f _normal = {0,0,1};
    cv::Vec3f _origin = {0,0,0};
    cv::Mat _M;
    cv::Mat _T;
};

//quads based surface class with a pointer implementing a nominal scale of 1 voxel
class QuadSurface : public Surface
{
public:
    SurfacePointer *pointer();
    QuadSurface() {};
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset) override;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th) override;

    virtual cv::Mat_<cv::Vec3f> rawPoints() { return _points; }

    friend QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out);
    friend QuadSurface *smooth_vc_segmentation(QuadSurface *src);
    friend class ControlPointSurface;
protected:
    cv::Mat_<cv::Vec3f> _points;
    cv::Rect _bounds;
    cv::Vec2f _scale;
    cv::Vec3f _center;
};


//surface representing some operation on top of a base surface
//by default all ops but gen() are forwarded to the base
class DeltaQuadSurface : public Surface
{
public:
    //default - just assign base ptr, override if additional processing necessary
    //like relocate ctrl points, mark as dirty, ...
    virtual void setBase(Surface *base);
    DeltaQuadSurface(Surface *base);
    
    virtual SurfacePointer *pointer() override;
    
    void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset) override = 0;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th) override;

protected:
    Surface *_base = nullptr;
};

//might in the future have more properties! or those props are handled in whatever class manages a set of control points ...
class SurfaceControlPoint
{
public:
    SurfaceControlPoint(Surface *base, SurfacePointer *ptr_, const cv::Vec3f &control);
    SurfacePointer *ptr; //ptr to control point in base surface
    cv::Vec3f orig_wp; //the original 3d location where the control point was created
    cv::Vec3f normal; //original normal
    cv::Vec3f control_point; //actual control point location - should be in line with _orig_wp along the normal, but could change if the underlaying surface changes!
};

class ControlPointSurface : public DeltaQuadSurface
{
public:
    ControlPointSurface(Surface *base) : DeltaQuadSurface(base) {};
    void addControlPoint(SurfacePointer *base_ptr, cv::Vec3f control_point);
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset) override;

    void setBase(Surface *base);

protected:
    std::vector<SurfaceControlPoint> _controls;
};

class RefineCompSurface : public DeltaQuadSurface
{
public:
    RefineCompSurface(z5::Dataset *ds, ChunkCache *cache, QuadSurface *base = nullptr);
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset) override;
    
protected:
    z5::Dataset *_ds;
    ChunkCache *_cache;
};

//TODO constrain to visible area? or add visiable area disaplay?
void find_intersect_segments(std::vector<std::vector<cv::Vec3f>> &seg_vol, std::vector<std::vector<cv::Vec2f>> &seg_grid, const cv::Mat_<cv::Vec3f> &points, PlaneSurface *plane, const cv::Rect &plane_roi, float step);

float min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, float init_step = 16.0, float min_step = 0.125);
