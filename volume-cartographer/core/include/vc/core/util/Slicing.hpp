#pragma once

#include <xtensor/xarray.hpp>
#include <opencv2/core.hpp>

namespace z5
{
    class Dataset;
}

class CoordGenerator
{
public:
    //given input volume shape, fill a coord slice
    virtual void gen_coords(xt::xarray<float> &coords, int w, int h) = 0;
};

class PlaneCoords : public CoordGenerator
{
public:
    PlaneCoords(cv::Vec3f origin_, cv::Vec3f normal_);
    virtual void gen_coords(xt::xarray<float> &coords, int w, int h);
    cv::Vec3f origin = {0,0,0};
    cv::Vec3f normal = {1,1,1};
};


//NOTE depending on request this might load a lot (the whole array) into RAM
void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords);

//TODO make the chunking more intelligent and efficient - for now this is probably good enough ...
//this method will chunk over the second and third last dim of coords (which should probably be x and y)
void readInterpolated3DChunked(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, size_t chunk_size);