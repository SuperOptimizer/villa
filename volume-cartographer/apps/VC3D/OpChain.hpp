#pragma once

#include "vc/core/util/Surface.hpp"

class QuadSurface;
class DeltaQuadSurface;
class SurfacePointer;
class ChunkCache;

namespace z5 {
    class Dataset;
}

enum class OpChainSourceMode: int
{
    RAW = 0,
    DEFAULT_MESHING
};

//special "windowed" surface that represents a set of delta surfaces on top of a base QuadSurface
//caches the generated coords to base surface method on this cached representation
class OpChain : public Surface {
public:
    OpChain(QuadSurface *src) : _src(src) {};
    // cv::Mat render(SurfacePointer *ptr, const cv::Size &size, float z, float scale, ChunkCache *cache, z5::Dataset *ds);
    QuadSurface *surf(SurfacePointer *ptr, const cv::Size &size, float z, float scale, ChunkCache *cache, z5::Dataset *ds);
    void append(DeltaQuadSurface *op);

    SurfacePointer *pointer() override;
    void move(SurfacePointer *ptr, const cv::Vec3f &offset) override;
    bool valid(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(SurfacePointer *ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(SurfacePointer *ptr, const cv::Vec3f &coord, float th)  override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, SurfacePointer *ptr, float scale, const cv::Vec3f &offset);


protected:
    OpChainSourceMode _src_mode = OpChainSourceMode::DEFAULT_MESHING;
    std::vector<DeltaQuadSurface*> _ops;
    QuadSurface *_src = nullptr;
    QuadSurface *_crop = nullptr;

    // float _last_scale = 0;
    // cv::Vec3f _last_offset = {0,0,0};
    // TrivialSurfacePointer *_last_ptr = nullptr;
    // cv::Mat_<cv::Vec3f> _last_gen;
};
