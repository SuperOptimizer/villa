#include "vc/core/util/Slicing.hpp"

#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xview.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <shared_mutex>

using shape = z5::types::ShapeType;
using namespace xt::placeholders;


static std::ostream& operator<< (std::ostream& out, const std::vector<int> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

template <size_t N>
static std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

static std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

namespace z5 {
    namespace multiarray {

        template<typename T>
        inline xt::xarray<T> *readChunk(const Dataset & ds,
                            types::ShapeType chunkId)
        {
            if (!ds.chunkExists(chunkId)) {
                return nullptr;
            }

            assert(ds.isZarr());
            
            types::ShapeType chunkShape;
            // size_t chunkSize;
            ds.getChunkShape(chunkId, chunkShape);
            // get the shape of the chunk (as stored it is stored)
            //for ZARR also edge chunks are always full size!
            const std::size_t maxChunkSize = ds.defaultChunkSize();
            const auto & maxChunkShape = ds.defaultChunkShape();
            
            // chunkSize = std::accumulate(chunkShape.begin(), chunkShape.end(), 1, std::multiplies<std::size_t>());
            
            xt::xarray<T> *out = new xt::xarray<T>();
            *out = xt::empty<T>(maxChunkShape);
            
            // read the data from storage
            std::vector<char> dataBuffer;
            ds.readRawChunk(chunkId, dataBuffer);
            
            // decompress the data
            ds.decompress(dataBuffer, out->data(), maxChunkSize);
            
            return out;
        }
    }
}

// shape chunkId(const std::unique_ptr<z5::Dataset> &ds, shape coord)
// {
//     shape div = ds->chunking().blockShape();
//     shape id = coord;
//     for(int i=0;i<id.size();i++)
//         id[i] /= div[i];
//     return id;
// }
// 
// shape idCoord(const std::unique_ptr<z5::Dataset> &ds, shape id)
// {
//     shape mul = ds->chunking().blockShape();
//     shape coord = id;
//     for(int i=0;i<coord.size();i++)
//         coord[i] *= mul[i];
//     return coord;
// }

//NOTE depending on request this might load a lot (the whole array) into RAM
// template <typename T> void readInterpolated3D(T out, z5::Dataset *ds, const xt::xarray<float> &coords)
void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords)
{
    // auto dims = xt::range(_,coords.shape().size()-2);
    // std::vector<long int> da = dims;
    std::vector<int> dims;
    for(int i=0;i<coords.shape().size()-1;i++)
        dims.push_back(i);
    xt::xarray<float> upper = xt::amax(coords, dims);
    xt::xarray<float> lower = xt::amin(coords, dims);
    
    // std::cout << "lwo/high" << lower << upper << std::endl;
    upper(0) = std::min(upper(0),float(ds->shape(0)-1));
    upper(1) = std::min(upper(1),float(ds->shape(1)-1));
    upper(2) = std::min(upper(2),float(ds->shape(2)-1));
    lower(0) = std::max(lower(0),0.0f);
    lower(1) = std::max(lower(1),0.0f);
    lower(2) = std::max(lower(2),0.0f);
    // lower = xt::amax(lower, {0,0,0});
    // std::cout << "lwo/high" << lower << upper << std::endl;
    
    for (int i=0;i<3;i++)
        if (lower(i) > upper(i))
            return;
    
    // std::cout << "maxshape" << .shape() << std::endl;
    
    shape offset(3);
    shape size(3);
    for(int i=0;i<3;i++) {
        offset[i] = lower[i];
        size[i] = ceil(std::max(upper[i] - offset[i]+1,1.0f));
    }
    // std::cout << "offset" << offset << std::endl;
    // std::cout << "size" << size << std::endl;
    
    if (!size[0] || !size[1] || !size[2])
        return;
    
    xt::xarray<uint8_t> buf(size);
    
    // z5::multiarray::readSubarray<uint8_t>(*ds, buf, offset.begin(), std::thread::hardware_concurrency());
    // std::cout << buf.shape() << "\n";
    z5::multiarray::readSubarray<uint8_t>(*ds, buf, offset.begin(), 1);
    
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    if (out.shape() != out_shape) {
        // std::cout << "wtf allocating out as its wrong size!" << std::endl;
        out = xt::zeros<uint8_t>(out_shape);
    }
    
    // std::cout << out_shape << std::endl;
    
    auto iter_coords = xt::axis_slice_begin(coords, 2);
    auto iter_out = xt::axis_slice_begin(out, 2);
    auto end_coords = xt::axis_slice_end(coords, 2);
    auto end_out = xt::axis_slice_end(out, 2);
    size_t inb_count = 0;
    size_t total = 0;
    while(iter_coords != end_coords)
    {
        total++;
        std::vector<int> idx = {int((*iter_coords)(0)-offset[0]),int((*iter_coords)(1)-offset[1]),int((*iter_coords)(2)-offset[2])};
        // if (total % 1000 == 0)
        // std::cout << idx << *iter_coords << offset << buf.in_bounds(idx[0],idx[1],idx[2]) << buf.shape() << "\n";
        if (buf.in_bounds(idx[0],idx[1],idx[2])) {
            *iter_out = buf[idx];
            inb_count++;
        }
        else
            *iter_out = 0;
        iter_coords++;
        iter_out++;
    }
    // printf("inb sum: %d %d %.f\n", inb_count, total, float(inb_count)/total);
    
    // std::cout << "read to buf" << std::endl;
}

//TODO make the chunking more intelligent and efficient - for now this is probably good enough ...
//this method will chunk over the second and third last dim of coords (which should probably be x and y)
void readInterpolated3DChunked(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, size_t chunk_size)
{
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    out = xt::zeros<uint8_t>(out_shape);
    
    // std::cout << out_shape << " " << coords.shape() << "\n";
    
    //FIXME assert dims
    
    int xdim = coords.shape().size()-2;
    int ydim = coords.shape().size()-3;
    
    std::cout << coords.shape(ydim) << " " << coords.shape(xdim) << "\n";
    
#pragma omp parallel for
    for(size_t y = 0;y<coords.shape(ydim);y+=chunk_size)
        for(size_t x = 0;x<coords.shape(xdim);x+=chunk_size) {
            // xt::xarray<uint8_t> out_view = xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()});
            auto coord_view = xt::strided_view(coords, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()});
            
            // std::cout << out_view.shape() << " " << x << "x" << y << std::endl;
            // return;
            xt::xarray<uint8_t> tmp;
            readInterpolated3D(tmp, ds, coord_view);
            //FIXME figure out xtensor copy/reference dynamics ...
            xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()}) = tmp;
            // readInterpolated3D(xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()}), ds, coord_view);
        }
        
}

static std::unordered_map<uint64_t,xt::xarray<uint8_t>*> cache;

//algorithm 2: do interpolation on basis of individual chunks
void readInterpolated3D_a2(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords)
{
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    out = xt::zeros<uint8_t>(out_shape);
    
    // std::cout  << "out shape" << out_shape << " " << coords.shape() << "\n";
    
    //FIXME assert dims
    
    int xdim = coords.shape().size()-2;
    int ydim = coords.shape().size()-3;
    
    // std::cout << coords.shape(ydim) << " " << coords.shape(xdim) << std::endl;
    
    //10bit per dim would actually be fine until chunksize of 16 @ dim size of 16384
    //using 16 bits is pretty safe

    xt::xarray<uint16_t> chunk_ids = xt::empty<uint16_t>(coords.shape());
    auto chunk_size = xt::adapt(ds->chunking().blockShape(),{1,1,3});
    // std::cout << coords.shape() << chunk_size.shape() << std::endl;
    chunk_ids = coords/chunk_size;
    
    xt::xarray<int16_t> local_coords = xt::clip(coords - (chunk_ids*xt::xarray<float>(chunk_size)),-1,32767);
    
    // xt::xarray<uint8_t> valid = xt::amin(local_coords, {2}) >= 0;
    
    // std::cout << "local coords" << local_coords.shape() << "\n";
    
    std::shared_mutex mutex;
    
    //FIXME need to iterate all dims e.g. could have z or more ... (maybe just flatten ... so we only have z at most)
#pragma omp parallel for
    for(size_t y = 0;y<coords.shape(ydim);y++) {
        // xt::xarray<uint16_t> last_id;
        uint64_t last_key = -1;
        xt::xarray<uint8_t> *chunk = nullptr;
        for(size_t x = 0;x<coords.shape(xdim);x++) {
            auto id = xt::strided_view(chunk_ids, {y, x, xt::all()});
            
            
            uint64_t key = (id[0]) ^ (uint64_t(id[1])<<20) ^ (uint64_t(id[2])<<40);

            //TODO compare keys
            if (key != last_key) {
            // if (id != last_id) {
                
                // last_id = id;
                last_key = key;
                
                
                //TODO replace with precomputed valid value
                if (local_coords(y,x,0) < 0 || local_coords(y,x,1) < 0 || local_coords(y,x,2) < 0) {
                    chunk = nullptr;
                    continue;
                }
                
                
                mutex.lock_shared();
                
                if (cache.count(key))
                    chunk = cache[key];
                else {
                    xt::xarray<size_t> id_t = id;
                    std::vector<size_t> localid = std::vector<size_t>(id_t.begin(),id_t.end());
                    chunk = z5::multiarray::readChunk<uint8_t>(*ds, localid);
                    mutex.unlock();
                    mutex.lock();
                    cache[key] = chunk;
                }
                mutex.unlock();
            }
            
            if (chunk) {
                // if (local_coords(y,x,0) >= chunk_size(0,0,0) || local_coords(y,x,1) >= chunk_size(0,0,1) || local_coords(y,x,2) >= chunk_size(0,0,2)) {
                //     std::cout << "coord error!" << local_coords(y,x,0) << "x" << local_coords(y,x,1)<< "x" <<local_coords(y,x,2) << ds->chunking().blockShape() << "\n";
                //     continue;
                // }
                // if (local_coords(y,x,0) < 0 || local_coords(y,x,1) < 0 || local_coords(y,x,2) < 0) {
                //     std::cout << "coord error!" << local_coords(y,x,0) << "x" << local_coords(y,x,1)<< "x" <<local_coords(y,x,2) << ds->chunking().blockShape() << "\n";
                //     continue;
                // }
                auto tmp = chunk->operator()(local_coords(y,x,0),local_coords(y,x,1),local_coords(y,x,2));
                // std::cout << local_coords(y,x) << y << "x" << x << " : " << int(tmp) << std::endl;
                out(y,x,0) = tmp;
            }
            
            // return;
            // xt::xarray<uint8_t> tmp;
            // readInterpolated3D(tmp, ds, coord_view);
            //FIXME figure out xtensor copy/reference dynamics ...
            // xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()}) = tmp;
            // readInterpolated3D(xt::strided_view(out, {xt::ellipsis(), xt::range(y, y+chunk_size), xt::range(x, x+chunk_size), xt::all()}), ds, coord_view);
        }
    }
        
}


PlaneCoords::PlaneCoords(cv::Vec3f origin_, cv::Vec3f normal_) : origin(origin_)
{
    cv::normalize(normal_, normal, 1,0, cv::NORM_L2);
    
};

void PlaneCoords::gen_coords(xt::xarray<float> &coords, int w, int h)
{
    // auto grid = xt::meshgrid(xt::arange<float>(0,h),xt::arange<float>(0,w));
    
    cv::Vec3f vx,vy;
    //TODO will there be a jump around the midpoint?
    //FIXME how to decide direction of cross vector?
    if (abs(normal[0]) >= abs(normal[1])) {
        vx = cv::Vec3f(1,0,-normal[0]/normal[2]);
        cv::normalize(vx, vx, 1,0, cv::NORM_L2);
        vy = cv::Mat(normal).cross(cv::Mat(vx));
    }
    else {
        vy = cv::Vec3f(0,1,-normal[1]/normal[2]);
        cv::normalize(vy, vy, 1,0, cv::NORM_L2);
        vx = cv::Mat(normal).cross(cv::Mat(vy));
    }
    if (vx[0] < 0)
        vx *= -1;
    if (vy[1] < 0)
        vy *= -1;
    
    std::cout << "vecs" << normal << vx << vy << "\n";
    
    xt::xarray<float> vx_t{{{vx[2],vx[1],vx[0]}}};
    xt::xarray<float> vy_t{{{vy[2],vy[1],vy[0]}}};
    xt::xarray<float> origin_t{{{origin[2],origin[1],origin[0]}}};
    
    xt::xarray<float> xrange = xt::arange<float>(-w/2,w/2).reshape({1, -1, 1});
    xt::xarray<float> yrange = xt::arange<float>(-h/2,h/2).reshape({-1, 1, 1});
    
    // xrange = xrange.reshape(-1, 1, 1);
    
    // std::cout << xrange.shape() << vx_t.shape() <<  std::endl;
    
    coords = vx_t*xrange + vy_t*yrange+origin_t;
    
    
    // auto res = xt::stack(xt::xtuple(std::get<0>(grid),std::get<1>(grid)), 2);
    // coords = res;
    // // auto sh = ;
    // std::cout << coords.shape() << std::endl;
//     std::cout << xt::strided_view(coords,{h/2,w/2,xt::all()}) << std::endl;
//     std::cout << xt::strided_view(coords,{h/2,w/2+1,xt::all()}) << std::endl;
//     std::cout << xt::strided_view(coords,{h/2+1,w/2,xt::all()}) << std::endl;
//     
//     
//     std::cout << xt::strided_view(coords,{0,0,xt::all()}) << std::endl;
//     std::cout << xt::strided_view(coords,{0,w-1,xt::all()}) << std::endl;
//     std::cout << xt::strided_view(coords,{h-1,0,xt::all()}) << std::endl;
//     std::cout << xt::strided_view(coords,{h-1,w-1,xt::all()}) << std::endl;
}
