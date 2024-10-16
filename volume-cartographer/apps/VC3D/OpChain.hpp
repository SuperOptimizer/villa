#pragma once

#include <opencv2/core/core.hpp>

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


class OpChain {
public:
    OpChain(QuadSurface *src) : _src(src) {};
    cv::Mat render(SurfacePointer *ptr, const cv::Rect &roi, float z, float scale, ChunkCache *cache, z5::Dataset *ds);
    QuadSurface *surf(SurfacePointer *ptr, const cv::Rect &roi, float z, float scale, ChunkCache *cache, z5::Dataset *ds);
    void append(DeltaQuadSurface *op);

protected:
    OpChainSourceMode _src_mode = OpChainSourceMode::RAW;
    std::vector<DeltaQuadSurface*> _ops;
    QuadSurface *_src = nullptr;
    QuadSurface *_crop = nullptr;
};
