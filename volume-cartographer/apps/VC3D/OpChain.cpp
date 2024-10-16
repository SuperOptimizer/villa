#include "OpChain.hpp"

#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

QuadSurface *OpChain::surf(SurfacePointer *ptr, const cv::Rect &roi, float z, float scale, ChunkCache *cache, z5::Dataset *ds)
{
    QuadSurface *last = nullptr;

    if (_src_mode == OpChainSourceMode::RAW) {
        last = _src;
    }
    else if (_src_mode == OpChainSourceMode::DEFAULT_MESHING)
    {
        //re-parametrize base then go trough surfaces ...
        if (_crop) {
            delete _crop;
            _crop = nullptr;
        }

        int search_step = 40;
        int mesh_step = 20;

        _crop = regularized_local_quad(_src, ptr, roi.width/mesh_step/scale, roi.height/mesh_step/scale, search_step, mesh_step);
    }

    //reset op chain
    for(auto s : _ops) {
        s->setBase(last);
        last = s;
    }

    return last;
}

cv::Mat OpChain::render(SurfacePointer *ptr, const cv::Rect &roi, float z, float scale, ChunkCache *cache, z5::Dataset *ds)
{
    QuadSurface *last = surf(ptr, roi, z, scale, cache, ds);

    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<uint8_t> img;
    if (_src_mode == OpChainSourceMode::RAW)
        last->gen(&coords, nullptr, roi.size(), ptr, scale, {-roi.width/2, -roi.height/2, z});
    else
        last->gen(&coords, nullptr, roi.size(), nullptr, scale, {-roi.width/2, -roi.height/2, z});
    readInterpolated3D(img, ds, coords*scale, cache);

    return img;
}

void OpChain::append(DeltaQuadSurface *op)
{
    _ops.push_back(op);
}
