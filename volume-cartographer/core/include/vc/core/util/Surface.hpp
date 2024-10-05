#pragma once

#include <opencv2/core.hpp> 

class SurfacePointer;
class CoordGenerator;
class QuadSurface;
class TrivialSurfacePointer;


QuadSurface *load_quad_from_vcps(const std::string &path);
QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search = 100, int step_out = 5);

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
    virtual CoordGenerator *generator(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //not yet
    // virtual void normal(SurfacePointer *ptr, cv::Vec3f offset);
};

//quads based surface class with a pointer of nominal scale 1
class QuadSurface : public Surface
{
public:
    SurfacePointer *pointer();
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    CoordGenerator *generator(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th) override;
    
    friend QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out);
protected:
    cv::Mat_<cv::Vec3f> _points;
    cv::Rect _bounds;
    cv::Vec2f _scale;
    cv::Vec3f _center;
};

//everything shall be exactly the same as a parent quad-surface, apart fromt the actual output coords around the normals
//and we might want an alpha map at some point but here is not required
class ControlPointSurface : public Surface
{
public:
    ControlPointSurface(Surface *base, SurfacePointer *base_ptr, cv::Vec3f control_point);
    SurfacePointer *pointer() override;
    void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    CoordGenerator *generator(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &tgt, float th) override;
    //TODO make derivative/dependencies generic/common interface
    void setBase(QuadSurface *base);

protected:
    Surface *_base; //base surface
    TrivialSurfacePointer *_ptr; //ptr to control point in base surface
    cv::Vec3f _orig_wp; //the original 3d location where the control point was created
    cv::Vec3f _normal; //original normal
    cv::Vec3f _control_point; //actual control point location - should be in line with _orig_wp along the normal, but could change if the underlaying surface changes!
};


//quads based surface class with a pointer of nominal scale 1
// class ControlPointOffset : public QuadSurface
// {
// public:
//     SurfacePointer *pointer();
//     QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
//     void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
//     bool valid(SurfacePointer *ptr, const cv::Vec3f &offset) override;
//     cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset) override;
//     CoordGenerator *generator(SurfacePointer *ptr, const cv::Vec3f &offset) override;
//     
//     friend QuadSurface *regularized_local_quad(QuadSurface *src, SurfacePointer *ptr, int w, int h, int step_search, int step_out);
// protected:
//     cv::Mat_<cv::Vec3f> _points;
//     cv::Rect _bounds;
//     cv::Vec2f _scale;
//     cv::Vec3f _center;
// };