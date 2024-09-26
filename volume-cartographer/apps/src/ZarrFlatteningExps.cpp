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
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Slicing.hpp"

#include <unordered_map>

using shape = z5::types::ShapeType;
using namespace xt::placeholders;

std::ostream& operator<< (std::ostream& out, const std::vector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const std::vector<int> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

template <size_t N>
std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}


shape chunkId(const std::unique_ptr<z5::Dataset> &ds, shape coord)
{
    shape div = ds->chunking().blockShape();
    shape id = coord;
    for(int i=0;i<id.size();i++)
        id[i] /= div[i];
    return id;
}

shape idCoord(const std::unique_ptr<z5::Dataset> &ds, shape id)
{
    shape mul = ds->chunking().blockShape();
    shape coord = id;
    for(int i=0;i<coord.size();i++)
        coord[i] *= mul[i];
    return coord;
}

void timed_plane_slice(PlaneCoords &plane, z5::Dataset *ds, size_t size, ChunkCache *cache, std::string msg)
{
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;
    
    auto start = std::chrono::high_resolution_clock::now();
    plane.gen_coords(coords, size, size);
    auto end = std::chrono::high_resolution_clock::now();
    // std::cout << std::chrono::duration<double>(end-start).count() << "s gen_coords() " << msg << std::endl;
    start = std::chrono::high_resolution_clock::now();
    readInterpolated3D(img, ds, coords, cache);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing " << msg << std::endl; 
}

cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
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

float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, bool z_search = true)
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        // loc = {-1,-1};
        // printf("init fail %d %d\n", loc[0], loc[1]);
        return;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    // std::cout << "at" << loc << val << std::endl;
    out = val;
    float best = sdist(val, tgt);
    // printf("init dist %f\n", best);
    float res;
    
    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};

    float step = 1.0;
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!boundary.contains({cand[0],cand[1]})) {
                out = {-1,-1,-1};
                // loc = {-1,-1};
                return;
            }
                
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = sdist(val,tgt);
            if (res < best) {
                // std::cout << res << tgt << val << step << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << tgt << val << step << "\n";
        }
        
        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
    
    // std::cout << "best" << best << tgt << out << "\n" <<  std::endl;
}

float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrt(d.dot(d));
    
    return abs(l-t_dist);
}

float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds)
{
    float sum = 0;
    for(int i=0;i<tgts.size();i++) {
        float d = tdist(v, tgts[i], tds[i]);
        sum += d*d;
    }
    
    return sum;
}

void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, bool z_search = true)
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-1,points.rows-1);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds);
    float res;
    
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search;
    if (z_search)
        search = {{0,-1},{0,1},{-1,0},{1,0}};
    else
        search = {{1,0},{-1,0}};
    float step = 16.0;
    
    
    // std::cout << "init " << best << tgts[0] << val << loc << "\n";
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!boundary.contains({cand[0],cand[1]})) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return;
            }
            
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds);
            if (res < best) {
                // std::cout << res << tgts[0] << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
                // std::cout << "(" << res << tgts[0] << val << step << cand << "\n";
        }
        
        if (!changed && step > 0.125) {
            step *= 0.5;
            changed = true;
        }
    }
    
    // std::cout << "best" << best << tgts[0] << out << "\n" <<  std::endl;
}

cv::Vec3f pred(const cv::Mat_<cv::Vec3f> &points, int x, int y, int x1,int y1, int x2, int y2, float mul)
{
    cv::Vec3f from = points(y+y1, x+x1);
    cv::Vec3f ref = points(y+y2, x+x2);
    cv::Vec3f dir = (from-ref)*mul;
    
    return from+dir;
}

//this works surprisingly well, though some artifacts where original there was a lot of skew
cv::Mat_<cv::Vec3f> derive_regular_region_stupid_gauss(cv::Mat_<cv::Vec3f> points)
{
    cv::Mat_<cv::Vec3f> out = points.clone();
        cv::Mat_<cv::Vec3f> blur = points.clone();
    cv::Mat_<cv::Vec2f> locs(points.size());
    
    cv::GaussianBlur(out, blur, {1,255}, 0);
    
#pragma omp parallel for
    for(int j=1;j<points.rows;j++)
        for(int i=1;i<points.cols-1;i++) {
            // min_loc(points, {i,j}, out(j,i), {out(j,i)[0],out(j,i)[1],out(j,i)[2]});
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), blur(j,i), false);
        }
        
    return out;
}

//instead of optimizing the distance to a (blurred) location which might not exist on the plane,
//optimize for a location which intersects neighoring spheres
//(basically try to ignore surface normal without knowing what the normal actually is)
cv::Mat_<cv::Vec3f> derive_regular_region_stupid_gauss_indirect(cv::Mat_<cv::Vec3f> points)
{
    cv::Mat_<cv::Vec3f> out = points.clone();
    cv::Mat_<cv::Vec3f> blur = points.clone();
    cv::Mat_<cv::Vec2f> locs(points.size());
    
    cv::GaussianBlur(points, blur, {1,255}, 0);
    
    int dist = 20;
    
    #pragma omp parallel for
    for(int j=2*dist;j<points.rows-2*dist;j++)
        for(int i=2*dist;i<points.cols-2*dist;i++) {
            std::vector<cv::Vec3f> tgts = {blur(j-dist,i),blur(j+dist,i),blur(j,i+dist/5),blur(j,i-dist/5)};
            std::vector<float> dists = {sqrt(sdist(tgts[0],blur(j,i))),sqrt(sdist(tgts[1],blur(j,i))),sqrt(sdist(tgts[2],blur(j,i))),sqrt(sdist(tgts[3],blur(j,i)))};
            // printf("%f %f %f %f\n", dists[0], dists[1], dists[2], dists[3]);
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), tgts, dists, false);
        }
        
        return out;
}

//try to ignore the local surface normal in error calculation
cv::Mat_<cv::Vec3f> derive_regular_region_stupid_gauss_normalcomp(cv::Mat_<cv::Vec3f> points)
{
    //TODO calc local normal, blurr it pass through to minimzation
}

//given an input image 
cv::Mat_<cv::Vec3f> derive_regular_region(cv::Mat_<cv::Vec3f> points)
{
    cv::Mat_<cv::Vec3f> out = points.clone();
    cv::Mat_<cv::Vec2f> locs(points.size());

    cv::GaussianBlur(out, out, {1,255}, 0);
    
#pragma omp parallel for
    for(int j=1;j<points.rows;j++)
        for(int i=1;i<points.cols-1;i++) {
            // min_loc(points, {i,j}, out(j,i), {out(j,i)[0],out(j,i)[1],out(j,i)[2]});
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), out(j,i));
        }
    
    // cv::Mat_<cv::Vec3f> global;
    // cv::reduce(points, global, 0, cv::REDUCE_AVG);
    //     for(int i=0;i<points.cols;i++) {
    //         for(int j=0;j<points.rows;j++) {
    //             min_loc(points, loc(j,i), out(j,i), {global(i)[0],global(i)[1],points(j,i)[2]});
    //         }
    //     }
    
//    /* // for(int j=499;j<500;j++) {
//     int j = 499;
//         //construct a line intersecting fixed z with spacing of 1
//         cv::Vec3f last_p = points(j,99);
//         cv::Vec2f last_loc = {j,99};
//         for(int i=100;i<points.cols-100;i++) {
//             std::vector<cv::Vec3f> tgts = {last_p};
//             auto dists = {1.0f};
//             min_loc(points, last_loc, out(j,i), tgts, dists, false);
//             last_p = out(j, i);
//             locs(j,i) = {i,j};
//         }
//         // break;
//     // }
//     
//     float sqd = sqrt(2);
//     
// //     // float dist
// // // #pragma omp parallel for
//     for(int j=500;j<1000;j++)
// #pragma omp parallel for
//         for(int i=200;i<800;i++) {
//             std::vector<cv::Vec3f> tgts = {out(j-1,i-1),out(j-1,i),out(j-1,i+1)};
//             // std::vector<cv::Vec3f> tgts = {out(j-1,i-1),out(j-1,i),out(j-1,i+1),out(j-1,i-2),out(j-1,i+2)};
//             auto dists = {sqd,1.0f,sqd};
//             // auto dists = {5.0f,1.0f,5.0f,10.0f,10.0f};
//             // std::cout << sqrt(sdist(tgts[0],points(j,i))) << " " << sqrt(sdist(tgts[1],points(j,i))) << " "  << sqrt(sdist(tgts[2],points(j,i)))  << std::endl;
//             locs(j,i) = locs(j-1,i);
//             min_loc(points, locs(j,i), out(j,i), tgts, dists);
//         }
//         
//     for(int j=510;j<520;j++)
//         // #pragma omp parallel for
//         for(int i=200;i<800;i++) {
//             out(j,i) = {-1,-1,-1};
//         }*/

    return out;
}

cv::Mat_<cv::Vec3f> calc_normals(const cv::Mat_<cv::Vec3f> &points) {
    int n_step = 1;
    cv::Mat_<cv::Vec3f> blur;
    cv::GaussianBlur(points, blur, {21,21}, 0);
    cv::Mat_<cv::Vec3f> normals(points.size());
#pragma omp parallel for
    for(int j=n_step;j<points.rows-n_step;j++)
        for(int i=n_step;i<points.cols-n_step;i++) {
            cv::Vec3f xv = blur(j,i+n_step)-blur(j,i-n_step);
            cv::Vec3f yv = blur(j+n_step,i)-blur(j-n_step,i);
            
            cv::normalize(xv,xv);
            cv::normalize(yv,yv);
            
            cv::Vec3f n = yv.cross(xv);
            n = n/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            
            normals(j,i) = n;
        }
    cv::GaussianBlur(normals, normals, {21,21}, 0);
    for(int j=n_step;j<points.rows-n_step;j++)
        for(int i=n_step;i<points.cols-n_step;i++) {
            cv::normalize(normals(j,i), normals(j,i));
        }
        
    return normals;
}

xt::xarray<float> xt_from_mat(const cv::Mat_<cv::Vec3f> &m)
{
    xt::xarray<float> t = xt::empty<float>({m.rows, m.cols, 3});
    
    for(int j=0;j<m.rows;j++) 
        for(int i=0;i<m.cols;i++) {
            cv::Vec3f v = m(j,i);
            t(j,i,0) = v[2];
            t(j,i,1) = v[1];
            t(j,i,2) = v[0];
        }
        
        return t;
}

cv::Mat_<cv::Vec3f> surf_normal_search(z5::Dataset *ds, ChunkCache *chunk_cache, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<cv::Vec3f> &normals)
{
    cv::Mat_<cv::Vec3f> res;
    
    // std::Vector<cv::Mat_<uint8_t>> slices;
    cv::Mat_<uint8_t> found(points.size(), 0);
    cv::Mat_<uint8_t> maximg(points.size(), 0);
    cv::Mat_<float> avgimg(points.size(), 0);
    cv::Mat_<float> height(points.size(), 0);
    
    for(int n=0;n<21;n++) {
        xt::xarray<uint8_t> raw_extract;
        // coords = points_reg*2.0;
        float off = (n-10)*0.5;
        readInterpolated3D(raw_extract, ds, xt_from_mat((points+normals*off)*0.5), chunk_cache);
        cv::Mat_<uint8_t> slice = cv::Mat(raw_extract.shape(0), raw_extract.shape(1), CV_8U, raw_extract.data());
        
        char buf[64];
        sprintf(buf, "slice%02d.tif", n);
        cv::imwrite(buf, slice);
        
        maximg = cv::max(slice, maximg);
        
        sprintf(buf, "max%02d.tif", n);
        maximg = cv::max(slice, maximg);
        cv::imwrite(buf, maximg);
        
        sprintf(buf, "avg%02d.tif", n);
        cv::Mat floatslice;
        slice.convertTo(floatslice, CV_32F);
        avgimg = avgimg + floatslice;
        cv::imwrite(buf, avgimg/(n+1));
        
        // slices.push_back(slice);
        for(int j=0;j<points.rows;j++)
            for(int i=0;i<points.cols;i++) {
                //found == 0: still searching for first time < 50!
                //found == 1: record < 50 start looking for >= 50 to stop
                //found == 2: done, found border
                if (slice(j,i) < 40 && found(j,i) <= 1) {
                    height(j,i) = n+1;
                    found(j,i) = 1;
                }
                else if (slice(j,i) >= 40 && found(j,i) == 1) {
                    found(j,i) = 2;
                }
            }
    }        // slices.push_back(slice);
    
    for(int j=0;j<points.rows;j++)
        for(int i=0;i<points.cols;i++)
            if (found(j,i) == 1)
                height(j,i) = 0;
    
    //never change opencv, never change ...
    cv::Mat mul;
    cv::cvtColor(height, mul, cv::COLOR_GRAY2BGR);
    cv::imwrite("max.tif", maximg);
    
    cv::Mat_<cv::Vec3f> new_surf = points + normals.mul(mul);
    
    xt::xarray<uint8_t> img;
    readInterpolated3D(img, ds, xt_from_mat(new_surf*0.5), chunk_cache);
    cv::Mat_<uint8_t> slice = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());

    cv::imwrite("surf.tif", slice);
    
    cv::Mat_<float> height_vis = height/21;
    height_vis = cv::min(height_vis,1-height_vis)*2;
    cv::imwrite("off.tif", height_vis);
    
    //now big question: how far away from average is the new surf!
    
    cv::Mat avg_surf;
    cv::GaussianBlur(new_surf, avg_surf, {7,7}, 0);
    
    readInterpolated3D(img, ds, xt_from_mat(avg_surf*0.5), chunk_cache);
    slice = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
    
    cv::imwrite("avg_surf.tif", slice);
    
    
    cv::Mat_<float> rel_height(points.size(), 0);
    
    cv::Mat_<cv::Vec3f> dist = avg_surf-new_surf;
    
#pragma omp parallel for
    for(int j=0;j<points.rows;j++)
        for(int i=0;i<points.cols;i++) {
            rel_height(j,i) = cv::norm(dist(j,i));
        }
    
    cv::imwrite("rel_height.tif", rel_height);
    
    return new_surf;
}


int main(int argc, char *argv[])
{
  assert(argc == 2);
  // z5::filesystem::handle::File f(argv[1]);
  z5::filesystem::handle::Group group(argv[1], z5::FileMode::FileMode::r);
  z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
  std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

  std::cout << "ds shape " << ds->shape() << std::endl;
  std::cout << "ds shape via chunk " << ds->chunking().shape() << std::endl;
  std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

  xt::xarray<float> coords = xt::zeros<float>({500,500,3});
  xt::xarray<uint8_t> img;
  
  std::vector<cv::Mat> chs;
  cv::imreadmulti("../grid_slice_coords.tif", chs, cv::IMREAD_UNCHANGED);
  cv::Mat_<cv::Vec3f> points;
  cv::merge(chs, points);
  
  //ok so given the image points which gives a mapping 2d->3d
  //we want to generate another mapping 2d->3d where
  //distance between two neighboring points in 2d is == 1 in 3d
  //where we minimize the distance between
  
  std::cout << "src" << points.size() << std::endl;
  
  std::vector<cv::Mat> chs_norm(3);
  cv::Rect roi(0,0,2000,2000);
  for(auto &m : chs) {
      double min,max;
      cv::minMaxLoc(m(roi),&min,&max);
      printf("minmax %f %f\n",min,max);
    }
      
  // chs_norm[0] = chs[2](roi)/1000;
  // chs_norm[1] = (chs[1](roi)-2000)/1000;
  // chs_norm[2] = (chs[0](roi)-3000)/2000;
  // cv::imwrite("x_src.tif", chs_norm[0]);
  // cv::imwrite("y_src.tif", chs_norm[1]);
  // cv::imwrite("z_src.tif", chs_norm[2]);
  
  
  roi = {0,0,800,4000};
  points = points(roi);
  
  cv::Mat_<cv::Vec3f> points_reg = derive_regular_region_stupid_gauss(points);
  
  cv::Mat_<cv::Vec3f> normals = calc_normals(points_reg);
  std::vector<cv::Mat> chs_normals(3);
  cv::split(normals, chs_normals);
  cv::imwrite("x_n.tif", chs_normals[0]);
  cv::imwrite("y_n.tif", chs_normals[1]);
  cv::imwrite("z_n.tif", chs_normals[2]);
  
  // std::cout << points(500,500) << points(500,501) << points(501,500) << std::endl;
  
  // GridCoords gen_grid(&points);
  
  // gen_plane.gen_coords(coords, 1000, 1000);
  // gen_grid.gen_coords(coords, 0, 0, 1000, 1000, 1.0, 0.5);
  
  // cv::Mat height = height_map_search(gen_rid);
  cv::Mat m;
  
  ChunkCache chunk_cache(10e9);
  
  GridCoords gen_grid(&points_reg);
  gen_grid.gen_coords(coords, 0, 0, 4000, 4000, 1.0, 0.5);
  readInterpolated3D(img,ds.get(),coords, &chunk_cache);
  m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  cv::imwrite("plane.tif", m);
    
  xt::xarray<uint8_t> raw_extract;
  // coords = points_reg*2.0;
  readInterpolated3D(raw_extract, ds.get(), xt_from_mat(points_reg*0.5), &chunk_cache);
  m = cv::Mat(raw_extract.shape(0), raw_extract.shape(1), CV_8U, raw_extract.data());
  cv::imwrite("raw.tif", m);
  
  roi = {0,0,800,4000};
  points_reg = points_reg(roi);
  normals = normals(roi);
  
  cv::resize(points_reg, points_reg, {0,0}, 5, 1);
  cv::resize(normals, normals, {0,0}, 5, 1);
  
  points_reg = surf_normal_search(ds.get(), &chunk_cache, points_reg, normals);
  
  // GridCoords gen_grid(&points_reg);
  // gen_grid.gen_coords(coords, 0, 0, 1000, 1000, 1.0, 0.5);
  // readInterpolated3D(img,ds.get(),coords, &chunk_cache);
  // m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  // cv::imwrite("surf.tif", m);

//   
//   auto start = std::chrono::high_resolution_clock::now();
//   readInterpolated3D(img,ds.get(),coords, &chunk_cache);
//   auto end = std::chrono::high_resolution_clock::now();
//   std::cout << std::chrono::duration<double>(end-start).count() << "s cold" << std::endl;
//   
//   cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
//   cv::imwrite("plane.tif", m);
//   
//   
//   GridCoords gen_grid_ref(&points);
//   gen_grid_ref.gen_coords(coords, 0, 00, 1000, 1000, 1.0, 0.5);
//   readInterpolated3D(img,ds.get(),coords, &chunk_cache);
//   m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
//   cv::imwrite("ref_plane.tif", m);
  
  return 0;
}
