#include "vc/core/util/Surface.hpp"
#include "vc/core/util/Slicing.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <vector>

// Intersection detection code
// Note: find_intersect_segments() is implemented in Surface.cpp

bool intersect(const Rect3D &a, const Rect3D &b)
{
    for(int d=0;d<3;d++) {
        if (a.high[d] < b.low[d])
            return false;
        if (a.low[d] > b.high[d])
            return false;
    }

    return true;
}

Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p)
{
    Rect3D res = a;
    for(int d=0;d<3;d++) {
        res.low[d] = std::min(res.low[d], p[d]);
        res.high[d] = std::max(res.high[d], p[d]);
    }

    return res;
}
