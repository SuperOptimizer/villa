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
#include <opencv2/calib3d.hpp>
#include <shared_mutex>

#include <algorithm>
#include <random>

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



uint64_t ChunkCache::groupKey(std::string name)
{
    if (!_group_store.count(name))
        _group_store[name] = _group_store.size()+1;
    
     return _group_store[name] << 48;
}
    
void ChunkCache::put(uint64_t key, xt::xarray<uint8_t> *ar)
{
    // if (!ar)
    //     return;
    
    if (ar)
        _stored += ar->size();
    
    if (_stored >= _size) {
        printf("cache reduce %f\n",float(_stored)/1024/1024);
        //stores (key,generation)
        using KP = std::pair<uint64_t, uint64_t>;
        std::vector<KP> gen_list(_gen_store.begin(), _gen_store.end());
        std::sort(gen_list.begin(), gen_list.end(), [](KP &a, KP &b){ return a.second < b.second; });
        for(auto it : gen_list) {
            xt::xarray<uint8_t> *ar = _store[it.first];
            //TODO we could remove this with lower probability so we dont store infiniteyl empty blocks but also keep more of them as they are cheap
            if (ar) {
                size_t size = ar->storage().size();
                delete ar;
                _stored -= size;
            }
            
            _store.erase(it.first);
            _gen_store.erase(it.first);

            //we delete 10% of cache content to amortize sorting costs
            if (_stored < 0.9*_size)
                break;
        }
        printf("cache reduce done %f\n",float(_stored)/1024/1024);
    }
    
    _store[key] = ar;
    _generation++;
    _gen_store[key] = _generation;
}


ChunkCache::~ChunkCache()
{
    for(auto &it : _store)
        delete it.second;
}

xt::xarray<uint8_t> *ChunkCache::get(uint64_t key)
{
    auto res = _store.find(key);
    if (res == _store.end())
        return nullptr;

    _generation++;
    _gen_store[key] = _generation;
    
    return res->second;
}

bool ChunkCache::has(uint64_t key)
{
    return _store.count(key);
}

void readInterpolated3D_a2(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache);

//NOTE depending on request this might load a lot (the whole array) into RAM
// template <typename T> void readInterpolated3D(T out, z5::Dataset *ds, const xt::xarray<float> &coords)
void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache) {
    readInterpolated3D_a2(out, ds, coords, cache);
}

void readInterpolated3D_plain(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords)
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
void readInterpolated3D_a2(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache)
{
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    out = xt::zeros<uint8_t>(out_shape);
    
    ChunkCache local_cache(1e9);
    
    if (!cache) {
        std::cout << "WARNING should use a shared chunk cache!" << std::endl;
        cache = &local_cache;
    }
    
    // std::cout  << "out shape" << out_shape << " " << coords.shape() << "\n";
    
    //FIXME assert dims
    
    int xdim = coords.shape().size()-2;
    int ydim = coords.shape().size()-3;
    
    // std::cout << coords.shape(ydim) << " " << coords.shape(xdim) << std::endl;
    
    //10bit per dim would actually be fine until chunksize of 16 @ dim size of 16384
    //using 16 bits is pretty safe

    //these three lines are 0.12s of 0.75s (and not threaded) - if processed here to a xtensor
    // auto chunk_ids = xt::empty<uint16_t>(coords.shape());
    // auto chunk_size = xt::adapt(ds->chunking().blockShape(),{1,1,3});
    // auto chunk_ids = coords/chunk_size;
    
    auto cw = ds->chunking().blockShape()[0];
    auto ch = ds->chunking().blockShape()[1];
    auto cd = ds->chunking().blockShape()[2];
    
    //this is 0.35 of 0.75s (and not threaded!) - if processed here to a xtensor
    // auto local_coords = xt::clip(coords - (xt::floor(chunk_ids)*xt::xarray<float>(chunk_size)),-1,32767);
    
    // xt::xarray<uint8_t> valid = xt::amin(local_coords, {2}) >= 0;
    
    // std::cout << "local coords" << local_coords.shape() << "\n";
    
    std::shared_mutex mutex;
    
    uint64_t key_base = cache->groupKey(ds->path());
    
    // std::cout << "cache using z5 path" << ds->path() << std::endl;
    
    //FIXME based on key math we should check bounds here using volume and chunk size
    
    // return;
    
    //FIXME need to iterate all dims e.g. could have z or more ... (maybe just flatten ... so we only have z at most)
    //the whole loop is 0.29s of 0.75s (if threaded)
    // #pragma omp parallel for schedule(dynamic, 512) collapse(2)
#pragma omp parallel for
    for(size_t y = 0;y<coords.shape(ydim);y++) {
        // xt::xarray<uint16_t> last_id;
        uint64_t last_key = -1;
        xt::xarray<uint8_t> *chunk = nullptr;
        for(size_t x = 0;x<coords.shape(xdim);x++) {
            // auto id = xt::strided_view(chunk_ids, {y, x, xt::all()});
            
            float ox = coords(y,x,0);
            float oy = coords(y,x,1);
            float oz = coords(y,x,2);
            
            int ix = int(ox)/cw;
            int iy = int(oy)/ch;
            int iz = int(oz)/cd;
            
            
            // uint64_t key = (id[0]) ^ (uint64_t(id[1])<<20) ^ (uint64_t(id[2])<<40);
            uint64_t key = key_base ^ uint64_t(ix) ^ (uint64_t(iy)<<16) ^ (uint64_t(iz)<<32);

            //TODO compare keys
            if (key != last_key) {
            // if (id != last_id) {
                
                // last_id = id;
                last_key = key;
                
                mutex.lock_shared();
                
                if (!cache->has(key)) {
                    mutex.unlock();
                    chunk = z5::multiarray::readChunk<uint8_t>(*ds, {size_t(ix),size_t(iy),size_t(iz)});
                    mutex.lock();
                    cache->put(key, chunk);
                }
                else
                    chunk = cache->get(key);
                mutex.unlock();
            }
            
            //this is slow
            //chunk->in_bounds(local_coords(y,x,0), local_coords(y,x,1), local_coords(y,x,2))
            
            if (chunk) {
                int lx = ox-ix*cw;
                int ly = oy-iy*ch;
                int lz = oz-iz*cd;
                if (lx < 0 || ly < 0 || lz < 0 || lx >= cw || ly >= ch || lz >= cd)
                    continue;
                out(y,x,0) = chunk->operator()(lx,ly,lz);
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

//given origin and normal, return the normalized vector v which describes a point : origin + v which lies in the plane and maximizes v.x at the cost of v.y,v.z
cv::Vec3f vx_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
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

cv::Vec3f vy_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    cv::Vec3f v = vx_from_orig_norm({o[1],o[0],o[2]}, {n[1],n[0],n[2]});
    return {v[1],v[0],v[2]};
}

void CoordGenerator::gen_coords(xt::xarray<float> &coords, int w, int h) const
{
    return gen_coords(coords, -w/2, -h/2, w, h, 1.0, 1.0);
}

void CoordGenerator::gen_coords(xt::xarray<float> &coords, const cv::Rect &roi, float render_scale, float coord_scale) const
{
    return gen_coords(coords, roi.x, roi.y, roi.width, roi.height, render_scale, coord_scale);
}

void vxy_from_normal(cv::Vec3f orig, cv::Vec3f normal, cv::Vec3f &vx, cv::Vec3f &vy)
{
    cv::Vec3f n;
    cv::normalize(normal, n, 1,0, cv::NORM_L2);
    
    vx = vx_from_orig_norm(orig, n);
    vy = vy_from_orig_norm(orig, n);
    
    //TODO will there be a jump around the midpoint?
    if (abs(vx[0]) >= abs(vy[1]))
        vy = cv::Mat(n).cross(cv::Mat(vx));
    else
        vx = cv::Mat(n).cross(cv::Mat(vy));
    
    //FIXME probably not the right way to normalize the direction?
    if (vx[0] < 0)
        vx *= -1;
    if (vy[1] < 0)
        vy *= -1;
}

void PlaneCoords::gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale, float coord_scale) const
{
    // auto grid = xt::meshgrid(xt::arange<float>(0,h),xt::arange<float>(0,w));
    cv::Vec3f vx, vy;
    vxy_from_normal(origin,normal,vx,vy);
    
    //why bother if xtensor is soo slow (around 10x of manual loop below even w/o threading)
    //     xt::xarray<float> vx_t{{{vx[2],vx[1],vx[0]}}};
    //     xt::xarray<float> vy_t{{{vy[2],vy[1],vy[0]}}};
    //     xt::xarray<float> origin_t{{{origin[2],origin[1],origin[0]}}};
    //     
    //     xt::xarray<float> xrange = xt::arange<float>(-w/2,w/2).reshape({1, -1, 1});
    //     xt::xarray<float> yrange = xt::arange<float>(-h/2,h/2).reshape({-1, 1, 1});
    //     
    //     coords = vx_t*xrange + vy_t*yrange+origin_t;
    
    coords = xt::empty<float>({h,w,3});
    
    float m = 1/render_scale;
    
    #pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            coords(j,i,0) = vx[2]*(i*m+x) + vy[2]*(j*m+y) + origin[2]*coord_scale;
            coords(j,i,1) = vx[1]*(i*m+x) + vy[1]*(j*m+y) + origin[1]*coord_scale;
            coords(j,i,2) = vx[0]*(i*m+x) + vy[0]*(j*m+y) + origin[0]*coord_scale;
        }
}

void IDWHeightPlaneCoords::gen_coords(xt::xarray<float> &coords, int x, int y, int w, int h, float render_scale, float coord_scale) const
{
    printf("idw gen coords\n");
    // auto grid = xt::meshgrid(xt::arange<float>(0,h),xt::arange<float>(0,w));
    cv::Vec3f vx, vy;
    vxy_from_normal(origin,normal,vx,vy);
    
    cv::Vec3f n;
    cv::normalize(normal, n);
    
    coords = xt::empty<float>({h,w,3});
    
    float m = 1/render_scale;
    
#pragma omp parallel for
    for(int j=0;j<h;j++)
        for(int i=0;i<w;i++) {
            float px = vx[0]*(i*m+x) + vy[0]*(j*m+y) + origin[0]*coord_scale;
            float py = vx[1]*(i*m+x) + vy[1]*(j*m+y) + origin[1]*coord_scale;
            float pz = vx[2]*(i*m+x) + vy[2]*(j*m+y) + origin[2]*coord_scale;
            cv::Vec3f p = {px,py,pz};
            p += height({px/coord_scale,py/coord_scale,pz/coord_scale})*coord_scale*n;
            coords(j,i,0) = p[2];
            coords(j,i,1) = p[1];
            coords(j,i,2) = p[0];
        }
}

/*cv::Point3f PlaneCoords::gen_coords(float i, float j, int x, int y, float render_scale, float coord_scale) const
{    
    float m = 1/render_scale;

    float cz = vx[2]*(i*m+x) + vy[2]*(j*m+y) + origin[2]*coord_scale;
    float cy = vx[1]*(i*m+x) + vy[1]*(j*m+y) + origin[1]*coord_scale;
    float cx = vx[0]*(i*m+x) + vy[0]*(j*m+y) + origin[0]*coord_scale;
    
    return {cx,cy,cz};
}*/

// bool plane_side(const cv::Point3f &point, const cv::Point3f &normal, float plane_off)
// {
//     return point.dot(normal) >= plane_off;
// }

float plane_mul(cv::Vec3f n)
{
    return 1.0/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
}

float PlaneCoords::scalarp(cv::Vec3f point) const
{
    cv::Vec3f n;
    cv::normalize(normal, n);
    return point.dot(n) - origin.dot(n);
}

float IDWHeightPlaneCoords::height(cv::Vec3f point) const
{
    if (control_points->size() < 4)
        return 0;
    
    cv::Vec3f n;
    cv::normalize(normal, n);
    float m = plane_mul(n);
    cv::Point3f projected_ref = point - (point.dot(n) - origin.dot(n))*m*n;
    
    // std::cout << point << "\n";
    
    double sum = 0;
    double weights = 0;
    for(auto &control : *control_points) {
        double h = (control.dot(n) - origin.dot(n))*m;
        cv::Point3f projected_control = control - h*n;
        // std::cout << control << projected_control << std::endl;
        double dist = cv::norm(projected_control-projected_ref);
        // printf("controlh %f dist %f\n", h, dist);
        double w = 1/(dist*dist);
        sum += w*h;
        weights += w;
        // printf(" %f  %f\n", sum, weights);
    }
    
    // printf("height %f\n", sum/weights);
    
    return sum/weights;
}


float IDWHeightPlaneCoords::scalarp(cv::Vec3f point) const
{
    cv::Vec3f n;
    cv::normalize(normal, n);
    // cv::Point3f projected_point = point - (point.dot(n) - origin.dot(n))*plane_mul(n)*n;
    // 
    // printf("projected point height %f\n", (projected_point.dot(n) - origin.dot(n))*plane_mul(n));
    // printf("original point height %f\n", (point.dot(n) - origin.dot(n))*plane_mul(n));
    // printf("mult %f\n", (projected_point.dot(n) - origin.dot(n))*plane_mul(n)/((point.dot(n) - origin.dot(n))*plane_mul(n)));
    // printf("\n");
    
    return point.dot(n) - origin.dot(n) - height(point);
}

void find_intersect_segments(std::vector<std::vector<cv::Point2f>> &segments_roi, const PlaneCoords *other, const CoordGenerator *roi_gen, const cv::Rect roi, float render_scale, float coord_scale)
{    
    xt::xarray<float> coords;
    
    //FIXME make generators more flexible so we can generate more sparse data
    printf("gen coord from within intersect\n");
    roi_gen->gen_coords(coords, roi, render_scale, coord_scale);
    
    std::vector<std::tuple<cv::Point,cv::Point3f,float>> upper;
    std::vector<std::tuple<cv::Point,cv::Point3f,float>> lower;
    std::vector<cv::Point2f> seg_points;
    
    // float plane_off = other->origin.dot(other->normal);
    cv::Vec3f n;
    cv::normalize(other->normal, n);
    float pmul = plane_mul(n);
    
    // std::cout << "other" << other->origin << other->normal << other->origin.dot(other->normal) << "\n";
    // std::cout << "roi" << ((PlaneCoords*)roi_gen)->origin << ((PlaneCoords*)roi_gen)->normal <<  "\n";
    
    for(int c=0;c<1000;c++) {
        int x = std::rand() % roi.width;
        int y = std::rand() % roi.height;
        
        
        cv::Point3f point = {coords(y,x,2),coords(y,x,1),coords(y,x,0)};
        point /= coord_scale;
        
        cv::Point2f img_point = {x,y};
        
        float scalarp = other->scalarp(point);
        
        // std::cout << point << " distsqs " << scalarp+ plane_off << " bias " << scalarp  << " loc " <<  x << "x" << y << "\n";
        
        if (scalarp > 0)
            upper.push_back({img_point,point,scalarp*pmul});
        else if(scalarp < 0)
            lower.push_back({img_point,point,-scalarp*pmul});
    }
    
    auto rng = std::default_random_engine {};
    std::shuffle(upper.begin(), upper.end(), rng);
    std::shuffle(lower.begin(), lower.end(), rng);
    
    
    std::vector<cv::Point2f> intersects;
    
    //brute force cause I'm lazy
    //FIXME if we have very vew points in uppper/lower: regenerate more points around there or reuse points
    for(int r=0;r<std::min<int>(100,std::min(upper.size(),lower.size()));r++) {
        float d_up = std::get<2>(upper[r]);
        float d_low = std::get<2>(lower[r]);
        cv::Point2f ip_up = std::get<0>(upper[r]);
        cv::Point2f ip_low = std::get<0>(lower[r]);
        
        cv::Point2f res = d_low/(d_up+d_low) * ip_up + d_up/(d_up+d_low) * ip_low;
        
        for(int s=0;s<5;s++) {
            assert(coords.in_bounds(round(res.y),round(res.x),2));
            //FIXME interpolate point and use lower acceptance threshold
            cv::Point3f point = {coords(round(res.y),round(res.x),2),coords(round(res.y),round(res.x),1),coords(round(res.y),round(res.x),0)};
            point /= coord_scale;
            float sdist = other->scalarp(point)*pmul;
            if (abs(sdist) < 0.5/coord_scale)
                break;

            if (sdist > 0) {
                d_up = sdist;
                ip_up = res;
            }
            else if(sdist < 0) {
                d_low = -sdist;
                ip_low = res;
            }
            
            res = d_low/(d_up+d_low) * ip_up + d_up/(d_up+d_low) * ip_low;
        }
        
        
        // cv::Point2f img_point = {x/render_scale+roi.x,y/render_scale+roi.y};
        intersects.push_back({res.x+roi.x*render_scale,res.y+roi.y*render_scale});
    }
    
    //this will only work if we have straight line!
    std::sort(intersects.begin(),
              intersects.end(),
              [](const cv::Point2f &a, const cv::Point2f &b){if (a.y != b.y) return a.y < b.y; return a.x < b.x; }
        );
    
    segments_roi.push_back(intersects);
}

void PlaneIDWSegmentator::add(cv::Vec3f wp, cv::Vec3f normal)
{
    std::cout << "added point" << _points.size()+1 << wp << normal << std::endl;
    if (_points.size() < 2) {
        _points.push_back({{0,0},wp});
        control_points.push_back(wp);
        return;
    }
    
    //FIXME check if points are on the same line!
    if (_points.size() == 2) {
        _points.push_back({{0,0},wp});
        control_points.push_back(wp);
        
        //FIXME calc 2d pints
        
        _generator->origin = _points[0].second;
        cv::Vec3f vx, vy;
        cv::normalize(_points[1].second-_points[0].second, vx, 1,0, cv::NORM_L2);
        cv::normalize(_points[2].second-_points[0].second, vy, 1,0, cv::NORM_L2);
        _generator->normal = vx.cross(vy);
        
        std::cout << "updated plane to " << _generator->origin << _generator->normal << std::endl;
        
        return;
    }

    _points.push_back({{0,0},wp});
    control_points.push_back(wp);
}

PlaneCoords *PlaneIDWSegmentator::generator() const
{
    return _generator;
}

PlaneIDWSegmentator::PlaneIDWSegmentator()
{
    _generator = new IDWHeightPlaneCoords(&control_points);
}


float PlaneCoords::pointDist(cv::Vec3f wp)
{
    cv::Vec3f n;
    cv::normalize(normal, n, 1,0, cv::NORM_L2);
    float plane_off = origin.dot(n);
    float plane_mul = 1.0/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    float scalarp = wp.dot(n) - plane_off;
        
    return abs(scalarp)*plane_mul;
}

cv::Vec3f PlaneCoords::project(cv::Vec3f wp, const cv::Rect &roi, float render_scale, float coord_scale)
{
    cv::Vec3f vx, vy;
    cv::Vec3f n;
    cv::normalize(normal, n, 1,0, cv::NORM_L2);
    
    vxy_from_normal(origin,n,vx,vy);
    
    vx = vx/render_scale/coord_scale;
    vy = vy/render_scale/coord_scale;
    
    std::vector <cv::Vec3f> src = {origin,origin+n,origin+vx,origin+vy};
    std::vector <cv::Vec3f> tgt = {{0,0,0},{0,0,1},{1,0,0},{0,1,0}};
    cv::Mat transf;
    cv::Mat inliers;
    
    cv::estimateAffine3D(src, tgt, transf, inliers, 0.1, 0.99);
    
    cv::Mat M = transf({0,0,3,3});
    cv::Mat T = transf({3,0,1,3});
    
    cv::Mat_<double> res = M*cv::Vec3d(wp)+T;
    
    return {res(0,0), res(0,1), res(0,2)};
}