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
#include "vc/core/io/PointSetIO.hpp"

#include <unordered_map>
#include <filesystem>

using shape = z5::types::ShapeType;
using namespace xt::placeholders;
namespace fs = std::filesystem; 

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

float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, const std::vector<float> ws = {})
{
    if (!ws.size()) {
        float sum = 0;
        for(int i=0;i<tgts.size();i++) {
            float d = tdist(v, tgts[i], tds[i]);
            sum += d*d;///tds[i]*20;
        }
        
        return sum;
    }
    else {
        float sum = 0;
        for(int i=0;i<tgts.size();i++) {
            float d = tdist(v, tgts[i], tds[i]);
            sum += d*d*ws[i];
        }
        
        return sum;
    }
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
    cv::Mat_<cv::Vec3f> blur(points.cols, points.rows);
    cv::Mat_<cv::Vec2f> locs(points.size());
    
    cv::Mat trans = out.t();
    
#pragma omp parallel for
    for(int j=0;j<trans.rows;j++) 
        cv::GaussianBlur(trans({0,j,trans.cols,1}), blur({0,j,trans.cols,1}), {255,1}, 0);

    blur = blur.t();
    
#pragma omp parallel for
    for(int j=1;j<points.rows;j++)
        for(int i=1;i<points.cols-1;i++) {
            // min_loc(points, {i,j}, out(j,i), {out(j,i)[0],out(j,i)[1],out(j,i)[2]});
            cv::Vec2f loc = {i,j};
            min_loc(points, loc, out(j,i), blur(j,i), false);
        }
        
    return out;
}

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

float min_loc_dbg(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneCoords *plane, cv::Vec2f init_step, float min_step_f, const std::vector<float> &ws = {})
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = tdist_sum(val, tgts, tds, ws);
    if (plane) {
        float d = plane->pointDist(val);
        best += d*d;
    }
    float res;
    
    //TODO add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2f step = init_step;
    
    
    // std::cout << "init " << best << tgts[0] << val << loc << "\n";
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            if (!boundary.contains({cand[0],cand[1]})) {
                out = {-1,-1,-1};
                loc = {-1,-1};
                return -1;
            }
            
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds, ws);
            if (plane) {
                float d = plane->pointDist(val);
                res += d*d;
            }
            if (res < best) {
                // std::cout << res << val << step << cand << "\n";
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
            // else
            // std::cout << "(" << res << val << step << cand << "\n";
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_f*init_step[0])
            break;
    }
    
    // std::cout << "best" << best << out <<  std::endl;
    return sqrt(best/tgts.size());
}

template<typename T> std::vector<T> join(const std::vector<T> &a, const std::vector<T> &b)
{
    std::vector<T> c = a;
    c.insert(c.end(), b.begin(), b.end());
    
    return c;
}

float multi_step_search(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneCoords *plane, cv::Vec2f init_step, const std::vector<cv::Vec3f> &opt_t, const std::vector<float> &opt_d, int &failstate, const std::vector<float> &ws)
{
    // std::cout << init << loc << std::endl;
    failstate = 0;
    cv::Vec2f init_loc = loc;
    
    std::vector<cv::Vec3f> t2 = join(tgts, opt_t);
    std::vector<float> d2 = join(tds, opt_d);
    
    float res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step, 0.1, ws);
    float res2 = 0;
    // float res2 = min_loc_dbg(points, loc, out, join(tgts,{out}), join(tds,{0}), plane, init_step*0.1, 0.1);
    float res3 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step*0.1, 0.1, ws);
    
    // printf("%f (%f %f)\n", res3, res2, res1);
    
    // return res3;
    
    float th = 5.0;
    
    if (res3 < th)
        return res3;

    // printf("start it %f\n", res3);
    
    for(int i=0;i<100;i++)
    {
        cv::Vec2f off = {rand()%100,rand()%100};
        off -= cv::Vec2f(50,50);
        off = mul(off, init_step)*100/50;
        loc = init_loc + off;
        
        res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step*10, 0.01, ws);
        // res2 = min_loc_dbg(points, loc, out, join(tgts,{out}), join(tds,{0}), plane, init_step*0.1, 0.1);
        res3 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws);
        
        // printf("   %f %f\n", res3, res1);
        
        if (res3 < th) {
            // printf("  it %f (%f)\n", res3, res1);
            return res3;
        }
    }
    
    loc = init_loc;
    res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step, 0.1, ws);
    // res2 = min_loc_dbg(points, loc, out, join(tgts,{out}), join(tds,{0}), plane, init_step*0.1, 0.1);
    res3 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step*0.1, 0.1, ws);
    // printf("  fallback %f (%f %f)\n", res3, res2, res1);
    failstate = 1;
    
    return -abs(res3);
}

float multi_step_search2(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneCoords *plane, cv::Vec2f init_step, const std::vector<cv::Vec3f> &opt_t, const std::vector<float> &opt_d)
{
    // std::cout << init << loc << std::endl;
    cv::Vec2f init_loc = loc;
    
    std::vector<cv::Vec3f> t2 = join(tgts, opt_t);
    std::vector<float> d2 = join(tds, opt_d);
    
    float res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step, 0.01);
    
    // printf("%f (%f %f)\n", res3, res2, res1);
    printf("%f\n", res1);
    
    if (res1 < 5.0)
        return res1;
    
    for(int i=0;i<100;i++)
    {
        cv::Vec2f off = {rand()%100,rand()%100};
        off -= cv::Vec2f(50,50);
        off = mul(off, init_step);
        loc = init_loc + off;
        
        res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step, 0.01);
        
        if (res1 < 5.0) {
            printf("  it %f \n", res1);
            return res1;
        }
    }
    
    loc = init_loc;
    res1 = min_loc_dbg(points, loc, out, t2, d2, plane, init_step, 0.01);
    printf("  fallback %f\n", res1);
    
    return res1;
}


//lets try again
//FIXME mark covered regions as not available so we can't repeat them'
cv::Mat_<cv::Vec3f> derive_regular_region_largesteps(cv::Mat_<cv::Vec3f> points)
{
    double sx, sy;
    vc_segmentation_scales(points, sx, sy);
    
    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;
    std::vector<cv::Vec2i> setfail;
    
    std::cout << "input avg step " << sx << " " << sy << points.size() << std::endl;
    
    //TODO use scaling and average diffeence vecs for init?
    float D = sqrt(2);
    
    float T = 20;
    int w = 500;
    int h = 500;
    
    int r = 3;
    
    cv::Vec2f step = {sx*T/10, sy*T/10};
    
    cv::Mat_<cv::Vec3f> out(h,w);
    cv::Mat_<cv::Vec2f> locs(h,w);
    cv::Mat_<uint8_t> state(h,w);
    cv::Mat_<float> dbg(h,w);
    out.setTo(-1);
    state.setTo(0);
    dbg.setTo(0);
    
    //FIXME the init locations are probably very important!
    
    //FIXME local search can be affected by noise/artefacts in data, add some re-init random initilizations if we see failures?
    
    int x0 = w/2;
    int y0 = h/2;

    locs(y0,x0) = {200, 1000};
    // locs(y0,x0) = {600, 1000};
    out(y0,x0) = at_int(points, locs(y0,x0));
    
    float res;
    
    
    //first point to the right
    locs(y0,x0+1) = locs(y0,x0)+cv::Vec2f(1,0);
    res = min_loc_dbg(points, locs(y0,x0+1), out(y0,x0+1), {out(y0,x0)}, {T}, nullptr, step, 0.01);
    std::cout << res << std::endl;
    
    //bottom left
    locs(y0+1,x0) = locs(y0,x0)+cv::Vec2f(0,1);
    res = min_loc_dbg(points, locs(y0+1,x0), out(y0+1,x0), {out(y0,x0),out(y0,x0+1)}, {T,D*T}, nullptr, step, 0.01);
    std::cout << res << std::endl;
    
    //bottom right
    locs(y0+1,x0+1) = locs(y0,x0)+cv::Vec2f(1,1);
    res = min_loc_dbg(points, locs(y0+1,x0+1), out(y0+1,x0+1), {out(y0,x0),out(y0,x0+1),out(y0+1,x0)}, {D*T,T,T}, nullptr, step, 0.01);
    std::cout << res << std::endl;
    
    std::cout << out(y0,x0) << out(y0,x0+1) << std::endl;
    std::cout << out(y0+1,x0) << out(y0+1,x0+1) << std::endl;
    
    // locs(j,i) = locs(j-1,i);
    
    std::vector<cv::Vec3f> refs;
    std::vector<float> dists;
    
    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};
    
    cv::Rect bounds(0,0,h-1,w-1);
    
    state(y0,x0) = 1;
    state(y0+1,x0) = 1;
    state(y0,x0+1) = 1;
    state(y0+1,x0+1) = 1;
    
    fringe.push_back({y0,x0});
    fringe.push_back({y0+1,x0});
    fringe.push_back({y0,x0+1});
    fringe.push_back({y0+1,x0+1});
    
    int succ = 0;
    int total_fail = 0;
    
    while (fringe.size()) {
        for(auto p : fringe)
        {
            if (state(p) != 1)
                continue;
            
            // std::cout << "check " << p << std::endl;
            
            for(auto n : neighs)
                if (bounds.contains(p+n) && state(p+n) == 0) {
                    state(p+n) = 2;
                    cands.push_back(p+n);
                    // std::cout << "cand  " << p+n << std::endl;
                }
        }
        
        
        fringe.resize(0);
        
        printf("have %d cands %d %d\n", cands.size(), succ, total_fail);
        
        for(auto p : cands) {
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            std::vector<float> ws;
            cv::Vec2f loc_sum = 0;
            int fail = 0;
            
            for(int oy=std::max(p[0]-r,0);oy<=p[0]+r;oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,out.cols-1);ox++)
                    if (state(oy,ox) == 1) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-p[0];
                        int dx = ox-p[1];
                        float d = sqrt(dy*dy+dx*dx);
                        dists.push_back(T*d);
                        loc_sum += locs(oy,ox);
                        // float w = 1.0/(std::max(abs(dbg(oy,ox)),1.0f));
                        // ws.push_back(w);
                    }
                    else if (state(oy,ox) == 10)
                        fail++;
                    // else if (state(oy,ox) == 10)
                        // fail++;
                    
            locs(p) = loc_sum*(1.0/dists.size());
            
            if (fail >= 2) {
                setfail.push_back(p);
                continue;
            }
            
            if (succ > 200 && dists.size()-4*fail <= 12) {
                continue;
            }
                    
            int failstate = 0;
            res = multi_step_search(points, locs(p), out(p), refs, dists, nullptr, step, {}, {}, failstate, ws);
            
            // printf("%f\n", res);

            dbg(p) = -res;
                    
            if (failstate) {
                //no good minimum found
                // state(p) = 10;
                // out(p) = -1;
                // succ++;
                setfail.push_back(p);
                total_fail++;
                // printf("%f\n", res);
                // state(p) = 1;
                // fringe.push_back(p);
            }
            else if (res < 0) {
                //image edge encountered
                state(p) = 11;
                out(p) = -1;
            }
            else {
                succ++;
                state(p) = 1;
                fringe.push_back(p);
            }
            
        }
        cands.resize(0);
        
        for(auto p : setfail) {
            dbg(p) = -1;
            state(p) = 10;
            out(p) = -1;
        }
        setfail.resize(0);
    }
    
    cv::resize(dbg, dbg, {0,0}, 10.0, 10.0, cv::INTER_NEAREST);
    cv::imwrite("dbg.tif", dbg);
    
    
    //now lets expand a whole row
    /*for(int i=2;i<w;i++) {
//         float res;
//         //predict upper loc
//         // locs(0,i) = 2*locs(0,i-1)-locs(0,i-2);
//         locs(0,i) = locs(0,i-1);
//         
//         //FIXME this may take a curve?
//         res = multi_step_search(points, locs(0,i), out(0,i), {out(0,i-1),out(1,i-1)}, {T,D*T}, nullptr, step, {out(0,i-2)}, {2*T});
//         
//         if (res == -1)
//             break;
//         
// //         res = min_loc_dbg(points, locs(0,i), out(0,i), {out(0,i-2),out(0,i-1),out(1,i-1)}, {2*T,T,D*T}, nullptr, step, 0.01);
// //         
// //         std::cout << res << std::endl;
// // 
// //         //predict lower loc
//         // locs(1,i) = 2*locs(1,i-1)-locs(1,i-2);
//         locs(1,i) = locs(1,i-1);
//         //         res = min_loc_dbg(points, locs(1,i), out(1,i), {out(0,i),out(0,i-1),out(1,i-1)}, {T,D*T,T}, nullptr, step, 0.01);
//         res = multi_step_search(points, locs(1,i), out(1,i), {out(0,i),out(0,i-1),out(1,i-1)}, {T,D*T,T}, nullptr, step, {out(1,i-2)}, {2*T});
//         //         std::cout << res << std::endl;
//         if (res == -1)
//             break;
        {
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            
            int j = 0;
            
            for(int oy=std::max(j-r,0);oy<=2;oy++)
                for(int ox=std::max(i-r,0);ox<=std::min(i+r,out.cols-1);ox++)
                    if (out(oy,ox)[0] != -1 && (ox != i || oy != j)) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-j;
                        int dx = ox-i;
                        dists.push_back(T*sqrt(dy*dy+dx*dx));
                    }
                    
            locs(j,i) = locs(j,i-1);
            // if (dists.size() < 4) {
            //     out(j,i) = 0;
            // }
            // else {
                res = multi_step_search(points, locs(j,i), out(j,i), refs, dists, nullptr, step, {}, {});
                
            //     if (res == -1)
            //         out(j,i) = -1;
            // }
        }
        
        {
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            
            int j = 1;
            
            for(int oy=std::max(j-r,0);oy<=2;oy++)
                for(int ox=std::max(i-r,0);ox<=std::min(i+r,out.cols-1);ox++)
                    if (out(oy,ox)[0] != -1 && (ox != i || oy != j)) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-j;
                        int dx = ox-i;
                        dists.push_back(T*sqrt(dy*dy+dx*dx));
                    }
                    
            locs(j,i) = locs(j,i-1);
            // if (dists.size() < 4) {
            //     out(j,i) = 0;
            // }
            // else {
                res = multi_step_search(points, locs(j,i), out(j,i), refs, dists, nullptr, step, {}, {});
                
            //     if (res == -1)
            //         out(j,i) = -1;
            // }
        }
        
    }
    
    //now lets expand the rest
    for(int j=2;j<h;j++) {
        float res;
        
        for(int i=0;i<w;i++) {
            locs(j,i) = locs(j-1,i);
            
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            
            
            for(int oy=std::max(j-r,0);oy<=j;oy++)
                for(int ox=std::max(i-r,0);ox<=std::min(i+r,out.cols-1);ox++)
                    if (out(oy,ox)[0] != -1 && (ox != i || oy != j)) {
                        refs.push_back(out(oy,ox));
                        int dy = oy-j;
                        int dx = ox-i;
                        dists.push_back(T*sqrt(dy*dy+dx*dx));
                    }
                    
            // if (dists.size() < 4) {
            //     out(j,i) = -1;
            //     continue;
            // }
                    
            res = multi_step_search(points, locs(j,i), out(j,i), refs, dists, nullptr, step, {}, {});
            
            // res = multi_step_search(points, locs(j,i), out(j,i), {out(j-1,i),out(j-1,i-1),out(j,i-1)}, {T,D*T,T}, &plane, step, {out(j-2,i),out(j,i-2)}, {2*T,2*T});
            
            printf("%f\n", res);
            
            // if (res == -1) {
            //     out(j,i) = -1;
            //     // locs(j,i) = locs(j-1,i);
            //     // return out;
            //     continue;
            // }
            
        }
    }*/
    
    
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

// void parallel_equal_gauss_blur(cv::Mat src, cv::Mat tgt, int size)
// {
//     //for large sigmas a tranpose would be useful! this is actually slower than openmo snge thread, probobably need larger chunks
//         
//     cv::Mat tmp(src.size(), src.type());
// 
// #pragma omp parallel for
//     for(int j=0;j<src.rows;j++) 
//         cv::GaussianBlur(src({0,j,src.cols,1}), tmp({0,j,src.cols,1}), {size,1}, 0);
// #pragma omp parallel for
//     for(int i=0;i<src.cols;i++) 
//         cv::GaussianBlur(tmp({i,0,1, src.rows}), tgt({i,0,1, src.rows}), {1,size}, 0);
// }

//somehow opencvs functions are pretty slow 
static inline cv::Vec3f normed(const cv::Vec3f v)
{
    return v/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

class MeasureLife
{
public:
    MeasureLife(std::string msg)
    {
        std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    ~MeasureLife()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

int main(int argc, char *argv[])
{
    assert(argc >= 4 && argc <= 5);
    
    const char *vol_path = argv[1];
    const char *segment_path = argv[2];
    const char *outdir_path = argv[3];
    
    int min_slice = 0;
    int max_slice = 65;
    
    if (!fs::exists(outdir_path)) {
        fs::create_directory(outdir_path);
    }
    else if (!fs::is_directory(outdir_path)) {
        printf("ERROR: target path %s is not a directory\n", outdir_path);
        return EXIT_FAILURE;
    }

    if (argc == 5) {
        min_slice = atoi(argv[4]);
        max_slice = min_slice;
    }
    else if (!fs::is_empty(outdir_path)) {
        printf("ERROR: target path %s is not empty\n", outdir_path);
        return EXIT_FAILURE;
    }

    else if (!fs::is_directory(outdir_path) || !fs::is_empty(outdir_path)) {
        printf("ERROR: target path %s is not an empty directory\n", outdir_path);
        return EXIT_FAILURE;
    }
  
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << "zarr dataset size for scale group 1 " << ds->shape() << std::endl;
    std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

    auto timer = new MeasureLife("reading segment ...");
    volcart::OrderedPointSet<cv::Vec3d> segment_raw = volcart::PointSetIO<cv::Vec3d>::ReadOrderedPointSet(segment_path);
    delete timer;
    
    timer = new MeasureLife("smoothing segment ...");
    cv::Mat src(segment_raw.height(), segment_raw.width(), CV_64FC3, (void*)const_cast<cv::Vec3d*>(&segment_raw[0]));
    
    cv::Mat_<cv::Vec3f> points;
    src.convertTo(points, CV_32F);
    
    // points = smooth_vc_segmentation(points);
    points = derive_regular_region_largesteps(points);
    
    // cv::resize(points, points, {0,0}, 10.0, 10.0);

    double sx, sy;
    sx = 0.05;
    sy = 0.05;
    // vc_segmentation_scales(points, sx, sy);
    delete timer;
    
    GridCoords generator(&points, sx, sy);
    
    xt::xarray<float> coords;
    xt::xarray<uint8_t> img;
    
    std::cout << points.size() << sx << " " << sy << "\n";
    
    // return EXIT_SUCCESS;
    float ds_scale = 0.5;
    float output_scale = 0.5;
    
    timer = new MeasureLife("rendering ...\n");
    for(int off=min_slice;off<=max_slice;off++) {
        generator.setOffsetZ(off-32);
        MeasureLife time_slice("slice "+std::to_string(off)+" ... ");
        generator.gen_coords(coords, 0, 0, points.cols/sx*output_scale, points.rows/sy*output_scale, 1.0, output_scale);
        
        //we read from scale 1
        coords *= ds_scale/output_scale;
        
        ChunkCache chunk_cache(10e9);
        
        readInterpolated3D(img, ds.get(), coords, &chunk_cache);
        cv::Mat m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
        
        std::stringstream ss;
        ss << outdir_path << std::setw(2) << std::setfill('0') << off << ".tif";
        cv::imwrite(ss.str(), m);
    }
    std::cout << "rendering ";
    delete timer;
    
    return 0;
}
