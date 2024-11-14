#include "SurfaceHelpers.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "CostFunctions.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"


#include <xtensor/xview.hpp>

#include <fstream>

// static std::ostream& operator<< (std::ostream& out, const std::vector<size_t> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }
//
// static std::ostream& operator<< (std::ostream& out, const std::vector<int> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }
//
// template <size_t N>
// static std::ostream& operator<< (std::ostream& out, const std::array<size_t,N> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }
//
// static std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }

class ALifeTime
{
public:
    ALifeTime(const std::string &msg = "")
    {
        if (msg.size())
            std::cout << msg << std::flush;
        start = std::chrono::high_resolution_clock::now();
    }
    double unit = 0;
    std::string del_msg;
    std::string unit_string;
    ~ALifeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        if (del_msg.size())
            std::cout << del_msg << std::chrono::duration<double>(end-start).count() << " s";
        else
            std::cout << " took " << std::chrono::duration<double>(end-start).count() << " s";

        if (unit)
            std::cout << " " << unit/std::chrono::duration<double>(end-start).count() << unit_string << "/s" << std::endl;
        else
            std::cout << std::endl;

    }
    double seconds()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end-start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

static cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
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

static cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, cv::Vec2d p)
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

static cv::Vec3f at_int_inv(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f p)
{
    int x = p[1];
    int y = p[0];
    float fx = p[1]-x;
    float fy = p[0]-y;

    cv::Vec3f p00 = points(y,x);
    cv::Vec3f p01 = points(y,x+1);
    cv::Vec3f p10 = points(y+1,x);
    cv::Vec3f p11 = points(y+1,x+1);

    cv::Vec3f p0 = (1-fx)*p00 + fx*p01;
    cv::Vec3f p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

static float atf_int(const cv::Mat_<float> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;

    float p00 = points(y,x);
    float p01 = points(y,x+1);
    float p10 = points(y+1,x);
    float p11 = points(y+1,x+1);

    float p0 = (1-fx)*p00 + fx*p01;
    float p1 = (1-fx)*p10 + fx*p11;

    return (1-fy)*p0 + fy*p1;
}

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

static void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, bool z_search = true)
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
    printf("init dist %f\n", best);
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
    
    std::cout << "best" << best << tgt << out << "\n" <<  std::endl;
}

static float tdist(const cv::Vec3f &a, const cv::Vec3f &b, float t_dist)
{
    cv::Vec3f d = a-b;
    float l = sqrt(d.dot(d));
    
    return abs(l-t_dist);
}

static float tdist_sum(const cv::Vec3f &v, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, const std::vector<float> ws = {})
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

static void min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, bool z_search = true)
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

//this works surprisingly well, though some artifacts where original there was a lot of skew
//FIXME add to api ...
static cv::Mat_<cv::Vec3f> derive_regular_region_stupid_gauss(cv::Mat_<cv::Vec3f> points)
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

static inline cv::Vec2d mul(const cv::Vec2d &a, const cv::Vec2d &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

static float min_loc_dbg(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, cv::Vec2f init_step, float min_step_f, const std::vector<float> &ws = {}, bool robust_edge = false, const cv::Mat_<float> &used = cv::Mat())
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
    if (!used.empty())
        best += atf_int(used, loc)*100.0;
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
                if (!robust_edge || (step[0] < min_step_f*init_step[0])) {
                    out = {-1,-1,-1};
                    loc = {-1,-1};
                    return -1;
                }
                else
                    //skip to next scale
                    break;
            }
            
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds, ws);
            if (!used.empty())
                res += atf_int(used, loc)*100.0;
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
    
    // std::cout << "best" << best << out << "\n" <<  std::endl;
    return sqrt(best/tgts.size());
}

static float min_loc_dbgd(const cv::Mat_<cv::Vec3f> &points, cv::Vec2d &loc, cv::Vec3d &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds, PlaneSurface *plane, cv::Vec2d init_step, float min_step_f, const std::vector<float> &ws = {}, bool robust_edge = false, const cv::Mat_<float> &used = cv::Mat())
{
    // std::cout << "start minlo" << loc << std::endl;
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        loc = {-1,-1};
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
    if (!used.empty())
        best += atf_int(used, loc)*100.0;
    float res;

    //TODO add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2d> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2d step = init_step;


    // std::cout << "init " << best << tgts[0] << val << loc << "\n";


    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);

            if (!boundary.contains({cand[0],cand[1]})) {
                if (!robust_edge || (step[0] < min_step_f*init_step[0])) {
                    out = {-1,-1,-1};
                    loc = {-1,-1};
                    return -1;
                }
                else
                    //skip to next scale
                    break;
            }


            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = tdist_sum(val, tgts, tds, ws);
            if (!used.empty())
                res += atf_int(used, loc)*100.0;
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

    // std::cout << "best" << best << out << "\n" <<  std::endl;
    return sqrt(best/tgts.size());
}

template<typename T> std::vector<T> join(const std::vector<T> &a, const std::vector<T> &b)
{
    std::vector<T> c = a;
    c.insert(c.end(), b.begin(), b.end());
    
    return c;
}

static float multi_step_search(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds_, PlaneSurface *plane, cv::Vec2f init_step, const std::vector<cv::Vec3f> &opt_t, const std::vector<float> &opt_d, int &failstate, const std::vector<float> &ws, float th, const cv::Mat_<float> &used = cv::Mat())
{
    failstate = 0;
    cv::Vec2f init_loc = loc;
    cv::Vec3f first_res;
    
    std::vector<float> w2 = ws;
    std::vector<float> tds = tds_;
    float w_sum = 0;
    for(auto &w : w2) {
        w *= w;
        w_sum += w;
    }
    
    float res1 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
    float res3;
    cv::Vec3f val;
    if (res1 >= 0) {
        val = at_int(points, loc);
        res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
    }
    else
        res3 = res1;
    
    if (res3 < th)
        return res3;
    
    float best_res = res3;
    cv::Vec2f best_loc = loc;
    for(int i=0;i<10;i++)
    {
        cv::Vec2f off = {rand()%100,rand()%100};
        off -= cv::Vec2f(50,50);
        off = mul(off, init_step)*100/50;
        loc = init_loc + off;
        
        res1 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
        if (res1 >= 0) {
            val = at_int(points, loc);
            res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
            if (res1 < best_res) {
                best_res = res1;
                best_loc = loc;
            }
        }
        else
            res3 = res1;
        
        if (res3 < th) {
            return res3;
        }
    }
    
    loc = best_loc;
    res1 = min_loc_dbg(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
    if (res1 >= 0) {
        val = at_int(points, loc);
        res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
    }
    else
        res3 = res1;
    
    failstate = 1;
    
    return -abs(res3);
}

static float multi_step_searchd(const cv::Mat_<cv::Vec3d> &points, cv::Vec2d &loc, cv::Vec3d &out, const std::vector<cv::Vec3f> &tgts, const std::vector<float> &tds_, PlaneSurface *plane, cv::Vec2d init_step, const std::vector<cv::Vec3f> &opt_t, const std::vector<float> &opt_d, int &failstate, const std::vector<float> &ws, float th, const cv::Mat_<float> &used = cv::Mat())
{
    failstate = 0;
    cv::Vec2d init_loc = loc;
    cv::Vec3f first_res;

    std::vector<float> w2 = ws;
    std::vector<float> tds = tds_;
    float w_sum = 0;
    for(auto &w : w2) {
        w *= w;
        w_sum += w;
    }

    float res1 = min_loc_dbgd(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
    float res3;
    cv::Vec3f val;
    if (res1 >= 0) {
        val = at_int(points, loc);
        res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
    }
    else
        res3 = res1;

    if (res3 < th)
        return res3;

    float best_res = res3;
    cv::Vec2f best_loc = loc;
    for(int i=0;i<10;i++)
    {
        cv::Vec2d off = {rand()%100,rand()%100};
        off -= cv::Vec2d(50,50);
        off = mul(off, init_step)*100/50;
        loc = init_loc + off;

        res1 = min_loc_dbgd(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
        if (res1 >= 0) {
            val = at_int(points, loc);
            res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
            if (res1 < best_res) {
                best_res = res1;
                best_loc = loc;
            }
        }
        else
            res3 = res1;

        if (res3 < th) {
            return res3;
        }
    }

    loc = best_loc;
    res1 = min_loc_dbgd(points, loc, out, tgts, tds, plane, init_step, 0.01, ws, false, used);
    if (res1 >= 0) {
        val = at_int(points, loc);
        res3 = sqrt(tdist_sum(val, tgts, tds, w2)/w_sum);
    }
    else
        res3 = res1;


    failstate = 1;

    return -abs(res3);
}

void write_ply(std::string path, const std::vector<cv::Vec3f> &points)
{
    std::ofstream ply;
    ply.open(path);

    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << points.size() << "\n";
    ply << "property float x\n";
    ply << "property float y\n";
    ply << "property float z\n";
    ply << "end_header\n";

    for(auto p : points)
        ply << p[0] << " " << p[1] << " " << p[2] << "\n";
}


cv::Mat_<cv::Vec3f> derive_regular_region_largesteps(const cv::Mat_<cv::Vec3f> &points, cv::Mat_<cv::Vec2f> &locs, int seed_x, int seed_y, float step_size, int w, int h)
{
    double sx, sy;
    vc_segmentation_scales(points, sx, sy);
    
    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;
    std::vector<cv::Vec2i> setfail;
    std::vector<cv::Vec2i> collected_failures;
    std::vector<cv::Vec2i> skipped;
    
    std::cout << "input avg step " << sx << " " << sy << points.size() << std::endl;
    
    //TODO use scaling and average diffeence vecs for init?
    float D = sqrt(2);
    
    float T = step_size;
    
    float th = T/4;
    
    int r = 4;
    
    cv::Vec2f step = {sx*T/10, sy*T/10};
    
    cv::Mat_<cv::Vec3f> out(h,w);
    locs.create(h,w);
    cv::Mat_<uint8_t> state(h,w);
    cv::Mat_<float> dbg(h,w);
    cv::Mat_<float> x_curv(h,w);
    cv::Mat_<float> y_curv(h,w);
    cv::Mat_<float> used(points.size());
    out.setTo(-1);
    used.setTo(0);
    state.setTo(0);
    dbg.setTo(0);
    x_curv.setTo(1);
    y_curv.setTo(1);
    
    cv::Rect src_bounds(0,0,points.cols-3,points.rows-3);
    if (!src_bounds.contains({seed_x,seed_y}))
        return out;

    //FIXME the init locations are probably very important!
    
    //FIXME local search can be affected by noise/artefacts in data, add some re-init random initilizations if we see failures?
    
    int x0 = w/2;
    int y0 = h/2;

    cv::Rect used_area(x0,y0,1,1);
    locs(y0,x0) = {seed_x, seed_y};
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
    
    // std::vector<cv::Vec3f> refs;
    // std::vector<float> dists;
    
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
    int last_round_updated = 4;
    
    std::vector<cv::Vec2f> all_locs;
    bool skipped_from_skipped = false;
    int generation = 0;
    int stop_gen = -1;
    bool ignore_failures = false;
    
    while (fringe.size() || (!skipped_from_skipped && skipped.size()) || collected_failures.size()) {
        generation++;
        if (generation == stop_gen)
            break;
        last_round_updated = 0;

        //first: regular fring
        if (fringe.size()) {
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
            for(auto p : cands)
                state(p) = 0;
            printf("gen %d processing %d fringe cands (total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", generation, cands.size(), succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
            fringe.resize(0);
            skipped_from_skipped = false;
        }
        else if ((last_round_updated && skipped.size()) || collected_failures.size()) {
            //skipped && failed points are processed at the same time so we expand smoothly
            for(auto p : skipped)
                state(p) = 0;
            cands = skipped;
            printf("gen %d processing %d skipped cands (total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", generation, cands.size(), succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
            skipped.resize(0);
            skipped_from_skipped = true;

            for(auto p : collected_failures) {
                state(p) = 0;
                cands.push_back(p);
            }
            printf("gen %d processing %d fail cands (total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", generation, cands.size(), succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
            collected_failures.resize(0);
            ignore_failures = true;
            skipped_from_skipped = false;
        }
        else
            break;

        cv::Mat_<cv::Vec3f> curv_data(2*r+1,2*r+1);
        cv::Mat_<uint8_t> curv_valid(2*r+1,2*r+1);
        
        for(auto p : cands) {
            if (state(p))
                continue;
            
            std::vector<cv::Vec3f> refs;
            std::vector<float> dists;
            std::vector<float> ws;
            std::vector<cv::Vec2i> dbg_outp;
            cv::Vec2f loc_sum = 0;
            int fail = 0;
            
            curv_valid.setTo(0);
            
            // printf("run %d %d\n",p[1],p[0]);
            
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,out.rows-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,out.cols-1);ox++)
                    if (state(oy,ox) == 1) {
                        int dy = oy-p[0];
                        int dx = ox-p[1];
                        curv_valid(dy+r,dx+r) = 1;
                        curv_data(dy+r,dx+r) = out(oy,ox);
                    }
            
            float x_curve_sum = 0;
            int x_curve_count = 0;
            for(int j=0;j<2*r+1;j++)
                for(int i=0;i<2*r+1-2;i++) {
                    if (curv_valid(j,i) && curv_valid(j,i+1) && curv_valid(j,i+2)) {
                        x_curve_sum += sqrt(sdist(curv_data(j,i),curv_data(j,i+2)))/(2*T);
                        x_curve_count++;
                        // printf("%f\n",sqrt(sdist(curv_data(j,i),curv_data(j,i+2)))/(2*T));
                    }
                }
            if (x_curve_count)
                x_curv(p) = sqrt(std::min(1.0f,x_curve_sum/x_curve_count));
            
            float y_curve_sum = 0;
            int y_curve_count = 0;
            for(int j=0;j<2*r+1-2;j++)
                for(int i=0;i<2*r+1;i++) {
                    if (curv_valid(j,i) && curv_valid(j+1,i) && curv_valid(j+2,i)) {
                        y_curve_sum += sqrt(sdist(curv_data(j,i),curv_data(j+2,i)))/(2*T);
                        y_curve_count++;
                        // printf("%f\n",sqrt(sdist(curv_data(j,i),curv_data(j,i+2)))/(2*T));
                    }
                }
                if (y_curve_count)
                    y_curv(p) = sqrt(std::min(1.0f,y_curve_sum/y_curve_count));
            
            // printf("avg curv xy %f %f\n",x_curv(p),y_curv(p));
            
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,out.rows-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,out.cols-1);ox++)
                    if (state(oy,ox) == 1) {
                        refs.push_back(out(oy,ox));
                        float curv_pow_x = pow(x_curv(p),abs(ox-p[1]));
                        float curv_pow_y = pow(y_curv(p),abs(oy-p[0]));
                        float dy = abs(oy-p[0])*curv_pow_x;
                        float dx = abs(ox-p[1])*curv_pow_y;
                        // float dy = (oy-p[0])*y_curv(p);
                        // float dx = (ox-p[1])*x_curv(p);
                        // float dy = (oy-p[0]);
                        // float dx = (ox-p[1]);
                        float d = sqrt(dy*dy+dx*dx);
                        dists.push_back(T*d);
                        loc_sum += locs(oy,ox);
                        // float w = 1.0/(std::max(abs(dbg(oy,ox)),1.0f));
                        float w = 1*curv_pow_x*curv_pow_y/d;
                        ws.push_back(w);
                        dbg_outp.push_back({dy,dx});
                    }
                    else if (state(oy,ox) == 10)
                        fail++;
            // else if (state(oy,ox) == 10)
            // fail++;
                    
            locs(p) = loc_sum*(1.0/dists.size());
            
            if (fail >= 2 && !ignore_failures) {
                setfail.push_back(p);
                continue;
            }
            
            if (!ignore_failures && succ > 200 && dists.size()-4*fail <= 12) {
                skipped.push_back(p);
                continue;
            }
                    
            int failstate = 0;
            res = multi_step_search(points, locs(p), out(p), refs, dists, nullptr, step, {}, {}, failstate, ws, th, used);
            all_locs.push_back(locs(p));
            
            // printf("%f\n", res);

            dbg(p) = -res;
                    
            if (failstate && !ignore_failures) {
                printf("fail %f %d %d\n", res, p[1]*5, p[0]*5);
                setfail.push_back(p);
                total_fail++;
                
                // std::vector<cv::Vec3f> succ_ps;
                // for(int j=0;j<10;j++)
                //     for(int i=0;i<10;i++) {
                //         cv::Vec2i l = p+cv::Vec2i(j-5,i-5);
                //         if (state(l) == 1)
                //             succ_ps.push_back(out(l));
                //     }
                    
                /*std::vector<cv::Vec3f> input_ps;
                std::vector<cv::Vec3f> rounded_ps;
                cv::Vec2i ref_loc = loc_sum*(1.0/dists.size());
                // cv::Vec2i ref_loc = locs(some_point);
                for(int j=0;j<50;j++)
                    for(int i=0;i<50;i++) {
                        cv::Vec2i l = ref_loc+cv::Vec2i(j-25,i-25);
                        input_ps.push_back(points(l[1],l[0]));
                    }
                // for(auto l : dbg_outp)
                    // input_ps.push_back(points(locs(l)[0],locs(l)[1]));
                // for(auto l : all_locs) 
                //     input_ps.push_back(at_int(points,l));
                // for(auto l : all_locs) 
                //     rounded_ps.push_back(points(l[1],l[0]));
                
                for(int n=0;n<refs.size();n++) {
                    std::cout << refs[n] << dbg_outp[0] << " " << dbg_outp[1] << std::endl;
                }
                
                write_ply("surf.ply", refs);
                write_ply("points.ply", input_ps);
                write_ply("res.ply", {out(p)});*/
                // write_ply("points_nearest.ply", rounded_ps);
                
                // cv::imwrite("xcurv.tif",x_curv);
                // cv::imwrite("ycurv.tif",y_curv);
                // return out;
            }
            else if (res < 0 && !failstate) {
                //image edge encountered
                state(p) = 11;
                out(p) = -1;
            }
            else {
                last_round_updated++;
                succ++;
                state(p) = 1;
                fringe.push_back(p);
                if (!used_area.contains({p[1],p[0]})) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                }
                cv::Rect roi = {locs(p)[0]-80,locs(p)[1]-80,160,160};
                roi = roi & src_bounds;
                for(int j=roi.y;j<roi.br().y;j++)
                    for(int i=roi.x;i<roi.br().x;i++) {
                        // used(j,i) = std::max(used(j,i), float(1.0-1.0/T*sqrt(sdist(points(locs(p)[1],locs(p)[0]), points(j,i))+1e-2)));
                        used(j,i) = std::min(1.0f, used(j,i) + std::max(0.0f, float(1.0-1.0/T*sqrt(sdist(points(locs(p)[1],locs(p)[0]), points(j,i))+1e-2))));
                    }
            }
            
        }
        cands.resize(0);
  
        
        for(auto p : setfail) {
            dbg(p) = -1;
            state(p) = 10;
            out(p) = -1;
            collected_failures.push_back(p);
        }
        setfail.resize(0);
        
        // if (ignore_failures)
        //     stop_gen = generation+10;
        
        printf("-> total succ/fail %d/%d fringe: %d skipped: %d failures: %d\n", succ, total_fail, fringe.size(), skipped.size(), collected_failures.size());
    }

    out = out(used_area).clone();
    state = state(used_area);

    std::vector<cv::Vec3f> valid_ps;
    for(int j=0;j<out.rows;j++)
        for(int i=0;i<out.cols;i++)
            if (state(j, i)== 1)
                valid_ps.push_back(out(j,i));

    write_ply("points.ply", valid_ps);
    
    return out;
}

#define OPTIMIZE_ALL 1
#define SURF_LOSS 2
#define SPACE_LOSS 2 //SURF and SPACE are never used together
#define LOSS_3D_INDIRECT 4

#define STATE_UNUSED 0
#define STATE_LOC_VALID 1
#define STATE_PROCESSING 2
#define STATE_COORD_VALID 4
#define STATE_FAIL 8
#define STATE_PHYS_ONLY 16

bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

//gen straigt loss given point and 3 offsets
int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w = 0.5)
{
    if (!coord_valid(state(p+o1)))
        return 0;
    if (!coord_valid(state(p+o2)))
        return 0;
    if (!coord_valid(state(p+o3)))
        return 0;

    problem.AddResidualBlock(StraightLoss::Create(w), nullptr, &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);

    if (!optimize_all) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&dpoints(p+o3)[0]);
    }

    return 1;
}

//gen straigt loss given point and 3 offsets
int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, float unit, bool optimize_all, ceres::ResidualBlockId *res, float w = 1.0)
{
    if (!coord_valid(state(p)))
        return 0;
    if (!coord_valid(state(p+off)))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &dpoints(p)[0], &dpoints(p+off)[0]);

    if (res)
        *res = tmp;

    if (!optimize_all)
        problem.SetParameterBlockConstant(&dpoints(p+off)[0]);

    return 1;
}

//gen straigt loss given point and 3 offsets
int gen_surf_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, cv::Mat_<cv::Vec2d> &loc, float w = 1.0)
{
    if (!loc_valid(state(p)))
        return 0;

    problem.AddResidualBlock(SurfaceLoss::Create(interp, w), nullptr, &out(p)[0], &loc(p)[0]);

    return 1;
}

//gen straigt loss given point and 3 offsets
int gen_used_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, const ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> &interp, cv::Mat_<cv::Vec2d> &loc, float w = 1.0)
{
    if (!loc_valid(state(p)))
        return 0;

    problem.AddResidualBlock(UsedSurfaceLoss::Create(interp, w), nullptr, &loc(p)[0]);

    return 1;
}

//gen straigt loss given point and 3 offsets
ceres::ResidualBlockId gen_loc_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec2d> &loc, cv::Vec2f loc_scale, float mindist, float w = 0.1)
{
    if ((!off[0] && !off [1]) || !loc_valid(state(p+off)))
        return nullptr;

    // cv::Vec2d d = loc(p) - loc(p+off);
    // d[0] /= loc_scale[0];
    // d[1] /= loc_scale[1];
    // std::cout << off << cv::norm(d)/cv::norm(off) << loc_scale << std::endl;


    return problem.AddResidualBlock(LocMinDistLoss::Create(loc_scale, cv::norm(off)*mindist, w), nullptr, &loc(p)[0], &loc(p+off)[0]);
}

//create all valid losses for this point
int create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, cv::Mat_<cv::Vec2d> &loc, float unit, int flags = 0)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, state, out, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, state, out, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, state, out, flags & OPTIMIZE_ALL);

    //vertical
    count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, state, out, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, state, out, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, state, out, flags & OPTIMIZE_ALL);

    float dist_w = 1.0;

    //direct neighboars
    count += gen_dist_loss(problem, p, {0,-1}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss(problem, p, {0,1}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss(problem, p, {-1,0}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss(problem, p, {1,0}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);

    //diagonal neighbors
    count += gen_dist_loss(problem, p, {1,-1}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss(problem, p, {-1,1}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss(problem, p, {1,1}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss(problem, p, {-1,-1}, state, out, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);

    if (flags & SURF_LOSS)
        count += gen_surf_loss(problem, p, state, out, interp, loc);

    return count;
}

cv::Vec2i lower_p(const cv::Vec2i &point, const cv::Vec2i &offset)
{
    if (offset[0] == 0) {
        if (offset[1] < 0)
            return point+offset;
        else
            return point;
    }
    if (offset[0] < 0)
        return point+offset;
    else
        return point;
}

bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status)
{
    return loss_status(lower_p(p, off)) & (1 << bit);
}

int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set)
{
    if (set)
        loss_status(lower_p(p, off)) |= (1 << bit);
    return set;
}

int conditional_dist_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, float unit, bool optimize_all, float w = 1.0)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_dist_loss(problem, p, off, state, out, unit, optimize_all, nullptr, w));
    return set;
};

int conditional_straight_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, bool optimize_all)
{
    int set = 0;
    if (!loss_mask(bit, p, o2, loss_status))
        set += set_loss_mask(bit, p, o2, loss_status, gen_straight_loss(problem, p, o1, o2, o3, state, out, optimize_all));
    return set;
};


struct vec2i_hash {
    size_t operator()(cv::Vec2i p) const
    {
        size_t hash1 = std::hash<float>{}(p[0]);
        size_t hash2 = std::hash<float>{}(p[1]);

        //magic numbers from boost. should be good enough
        return hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
};

//create only missing losses so we can optimize the whole problem
int create_missing_centered_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, const ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> &interp, cv::Mat_<cv::Vec2d> &loc, float unit, const cv::Vec2f &loc_scale, std::unordered_map<cv::Vec2i,std::vector<ceres::ResidualBlockId>,vec2i_hash> &foldLossIds, int flags = SURF_LOSS | OPTIMIZE_ALL)
{
    //generate losses for point p
    uint8_t old_state = state(p);
    state(p) = 1;

    int count = 0;

    //horizontal
    count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, state, out, flags);
    count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, state, out, flags);
    count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, state, out, flags);

    //vertical
    count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, state, out, flags);
    count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, state, out, flags);
    count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, state, out, flags);

    float dist_w = 1.0;

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, state, out, unit, flags, dist_w);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, state, out, unit, flags, dist_w);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, state, out, unit, flags, dist_w);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, state, out, unit, flags, dist_w);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, state, out, unit, flags, dist_w);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, state, out, unit, flags, dist_w);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, state, out, unit, flags, dist_w);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, state, out, unit, flags, dist_w);

    if (flags & SURF_LOSS && !loss_mask(6, p, {0,0}, loss_status)) {
        count += set_loss_mask(6, p, {0,0}, loss_status, gen_surf_loss(problem, p, state, out, interp, loc));

        int r = 4;
        count += set_loss_mask(6, p, {0,0}, loss_status, gen_surf_loss(problem, p, state, out, interp, loc));

        for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,state.rows-1);oy++)
            for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,state.cols-1);ox++) {
                cv::Vec2i off = {oy-p[0],ox-p[1]};
                ceres::ResidualBlockId id = gen_loc_dist_loss(problem, p, off, state, loc, loc_scale, 1.0*unit);
                if (id)
                    foldLossIds[p].push_back(id);
            }
    }

    state(p) = old_state;

    return count;
}

void freeze_inner_params(ceres::Problem &problem, int edge_dist, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, cv::Mat_<cv::Vec2d> &loc, cv::Mat_<uint16_t> &loss_status, int inner_flags)
{
    cv::Mat_<float> dist(state.size());

    edge_dist = std::min(edge_dist,254);


    cv::Mat_<uint8_t> masked;
    bitwise_and(masked, (uint8_t)inner_flags, masked);


    cv::distanceTransform(masked, dist, cv::DIST_L1, cv::DIST_MASK_3);


    // cv::imwrite("dists.tif",dist);

    for(int j=0;j<dist.rows;j++)
        for(int i=0;i<dist.cols;i++) {
            if (dist(j,i) >= edge_dist && !loss_mask(7, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.SetParameterBlockConstant(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.SetParameterBlockConstant(&loc(j,i)[0]);
                set_loss_mask(7, {j,i}, {0,0}, loss_status, 1);
            }
            if (dist(j,i) >= edge_dist+1 && !loss_mask(8, {j,i}, {0,0}, loss_status)) {
                if (problem.HasParameterBlock(&out(j,i)[0]))
                    problem.RemoveParameterBlock(&out(j,i)[0]);
                if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
                    problem.RemoveParameterBlock(&loc(j,i)[0]);
                set_loss_mask(8, {j,i}, {0,0}, loss_status, 1);
            }
        }
}

void remove_inner_foldlosses(ceres::Problem &problem, int edge_dist, cv::Mat_<uint8_t> &state,  std::unordered_map<cv::Vec2i,std::vector<ceres::ResidualBlockId>,vec2i_hash> &foldLossIds)
{
    cv::Mat_<float> dist(state.size());

    edge_dist = std::min(edge_dist,254);

    std::vector<ceres::ResidualBlockId> residual_blocks;
    problem.GetResidualBlocks(&residual_blocks);
    std::set<ceres::ResidualBlockId> set;
    for(auto &id : residual_blocks)
        set.insert(id);

    cv::distanceTransform(state, dist, cv::DIST_L1, cv::DIST_MASK_3);

    for(int j=0;j<dist.rows;j++)
        for(int i=0;i<dist.cols;i++) {
            if (dist(j,i) >= edge_dist && foldLossIds.count({j,i})) {
                cv::Vec2i p = {j,i};
                for(auto &id : foldLossIds[p])
                    if (set.count(id))
                        problem.RemoveResidualBlock(id);
                foldLossIds[p].resize(0);
            }
        }
}

//use a physical paper model
//first predict a position from just the physical model (keeping everything else constant)
//then refine with physical model and bicubic interpolation of surface location
//later refine whole model
cv::Mat_<cv::Vec3f> derive_regular_region_largesteps_phys(const cv::Mat_<cv::Vec3f> &points, cv::Mat_<cv::Vec2f> &locs, int seed_x, int seed_y, float step_size, int w, int h)
{
    double sx, sy;
    vc_segmentation_scales(points, sx, sy);

    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;

    std::cout << "input avg step " << sx << " " << sy << points.size() << std::endl;

    //TODO use scaling and average diffeence vecs for init?
    float D = sqrt(2);

    float T = step_size;

    float th = T/4;

    int r = 2;

    cv::Vec2f step = {sx*T/10, sy*T/10};

    cv::Mat_<cv::Vec3d> out(h,w);
    cv::Mat_<cv::Vec2d> locd(h,w);
    locs.create(h,w);
    cv::Mat_<uint8_t> state(h,w);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);
    cv::Mat_<float> cost_init(cv::Size(w,h),0);
    cv::Mat_<float> search_init(cv::Size(w,h),0);
    cv::Mat_<float> dbg(h,w);
    cv::Mat_<float> x_curv(h,w);
    cv::Mat_<float> y_curv(h,w);
    cv::Mat_<float> used(points.size());
    std::unordered_map<cv::Vec2i,std::vector<ceres::ResidualBlockId>,vec2i_hash> foldLossIds;
    out.setTo(-1);
    used.setTo(0);
    state.setTo(0);
    dbg.setTo(0);
    x_curv.setTo(1);
    y_curv.setTo(1);

    cv::Rect src_bounds(0,0,points.cols-3,points.rows-3);
    if (!src_bounds.contains({seed_x,seed_y}))
        return out;

    int x0 = w/2;
    int y0 = h/2;

    cv::Rect used_area(x0,y0,2,2);
    locd(y0,x0) = {seed_x, seed_y};
    out(y0,x0) = at_int(points, locd(y0,x0));

    float res;


    //first point to the right
    locd(y0,x0+1) = locd(y0,x0)+cv::Vec2d(1,0);
    res = min_loc_dbgd(points, locd(y0,x0+1), out(y0,x0+1), {out(y0,x0)}, {T}, nullptr, step, 0.01);
    std::cout << res << std::endl;

    //bottom left
    locd(y0+1,x0) = locd(y0,x0)+cv::Vec2d(0,1);
    res = min_loc_dbgd(points, locd(y0+1,x0), out(y0+1,x0), {out(y0,x0),out(y0,x0+1)}, {T,D*T}, nullptr, step, 0.01);
    std::cout << res << std::endl;

    //bottom right
    locd(y0+1,x0+1) = locd(y0,x0)+cv::Vec2d(1,1);
    res = min_loc_dbgd(points, locd(y0+1,x0+1), out(y0+1,x0+1), {out(y0,x0),out(y0,x0+1),out(y0+1,x0)}, {D*T,T,T}, nullptr, step, 0.01);
    std::cout << res << std::endl;

    std::cout << out(y0,x0) << out(y0,x0+1) << std::endl;
    std::cout << out(y0+1,x0) << out(y0+1,x0+1) << std::endl;

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};
    cv::Rect bounds(2,2,h-3,w-3);

    state(y0,x0) = 1;
    state(y0+1,x0) = 1;
    state(y0,x0+1) = 1;
    state(y0+1,x0+1) = 1;

    ceres::Problem big_problem;

    CeresGrid2DcvMat3f grid({points});
    ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> interp(grid);

    CeresGrid2DcvMat1f grid_used({points});
    ceres::BiCubicInterpolator<CeresGrid2DcvMat1f> interp_used(grid_used);

    int loss_count;
    loss_count += create_missing_centered_losses(big_problem, loss_status, {y0,x0}, state, out, interp, locd, step_size, {sx,sy}, foldLossIds);
    loss_count += create_missing_centered_losses(big_problem, loss_status, {y0+1,x0}, state, out, interp, locd, step_size, {sx,sy}, foldLossIds);
    loss_count += create_missing_centered_losses(big_problem, loss_status, {y0,x0+1}, state, out, interp, locd, step_size, {sx,sy}, foldLossIds);
    loss_count += create_missing_centered_losses(big_problem, loss_status, {y0+1,x0+1}, state, out, interp, locd, step_size, {sx,sy}, foldLossIds);

    big_problem.SetParameterBlockConstant(&locd(y0,x0)[0]);
    big_problem.SetParameterBlockConstant(&out(y0,x0)[0]);

    std::cout << "init loss count " << loss_count << std::endl;

    ceres::Solver::Options options_big;
    // options_big.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_big.linear_solver_type = ceres::SPARSE_SCHUR;
    // options_big.linear_solver_type = ceres::DENSE_QR;
    // options_big.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    // options_big.dense_linear_algebra_library_type = ceres::CUDA;
    // options_big.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options_big.minimizer_progress_to_stdout = false;
    //TODO check for update ...
    // options_big.enable_fast_removal = true;
    // options_big.num_threads = omp_get_max_threads();
    options_big.max_num_iterations = 10000;

    ceres::Solver::Summary summary;
    ceres::Solve(options_big, &big_problem, &summary);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.dense_linear_algebra_library_type = ceres::CUDA;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;


    std::cout << summary.BriefReport() << "\n";

    big_problem.SetParameterBlockConstant(&locd(y0+1,x0)[0]);
    big_problem.SetParameterBlockConstant(&locd(y0,x0+1)[0]);
    big_problem.SetParameterBlockConstant(&locd(y0+1,x0+1)[0]);

    fringe.push_back({y0,x0});
    fringe.push_back({y0+1,x0});
    fringe.push_back({y0,x0+1});
    fringe.push_back({y0+1,x0+1});

    int succ = 0;
    int total_fail = 0;
    int last_round_updated = 4;

    int generation = 0;
    int stop_gen = 0;
    bool ignore_failures = false;

    std::cout << "go " << fringe.size() << std::endl;

    while (fringe.size()) {
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        for(auto p : fringe)
        {
            //TODO maybe should also check neighs of cood_valid?
            if (!loc_valid(state(p)))
                continue;

            for(auto n : neighs)
                if (bounds.contains(p+n) && state(p+n) == 0) {
                    state(p+n) = STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        for(auto p : cands)
            state(p) = STATE_UNUSED;
        printf("gen %d processing %d fringe cands (total done %d fringe: %d\n", generation, cands.size(), succ, fringe.size());
        fringe.resize(0);

        std::cout << "cands " << cands.size() << std::endl;
        for(auto p : cands) {
            if (state(p) != STATE_UNUSED)
                continue;

            int ref_count = 0;
            cv::Vec3d avg = {0,0,0};
            cv::Vec2d avgl = {0,0};
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,out.rows-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,out.cols-1);ox++)
                    if (loc_valid(state(oy,ox))) {
                        ref_count++;
                        avg += out(oy,ox);
                        avgl += locd(oy,ox);
                    }

            if (ref_count < 4)
                continue;

            avg /= ref_count;
            avgl /= ref_count;
            out(p) = avg;
            locd(p) = avgl;

            ceres::Problem problem;

            state(p) = STATE_COORD_VALID;
            int local_loss_count = create_centered_losses(problem, p, state, out, interp, locd, step_size);

            // std::cout << "loss count " << local_loss_count << std::endl;

            //FIXME need to handle the edge of the input definition!!

            // ceres::Solver::Options options;
            // options.linear_solver_type = ceres::DENSE_QR;
            // options.max_num_iterations = 1000;
            // options.minimizer_progress_to_stdout = false;
            // ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cost_init(p) = summary.final_cost;

            float res = min_loc_dbgd(points, locd(p), out(p), {out(p)}, {0}, nullptr, step, 0.01, {}, false, used);


            //FIXME sometimes we get a random failure not at edge!
            int flags = OPTIMIZE_ALL;
            if (res < 0) {
                state(p) = STATE_COORD_VALID;
                out(p) = avg;
                // locd(p) = avgl;
                // encountered points edge?
                // TODO we could include it (without surface) into the problem, so we get a surface over the edge of the defined input? just include it but use special state, so its not used to count further refs against ...
                // state(p) = 10;
                // continue;
            }
            else {
                state(p) = STATE_LOC_VALID;
                flags |= SURF_LOSS;
            }

            if (loc_valid(state(p))) {
                int used_r = 4;
                for(int oy=std::max(p[0]-used_r,0);oy<=std::min(p[0]+used_r,state.rows-1);oy++)
                    for(int ox=std::max(p[1]-used_r,0);ox<=std::min(p[1]+used_r,state.cols-1);ox++) {
                        cv::Vec2i off = {oy-p[0],ox-p[1]};
                        if (gen_loc_dist_loss(problem, p, off, state, locd, {sx,sy}, 1.0*step_size))
                            problem.SetParameterBlockConstant(&locd(p+off)[0]);
                    }

                gen_surf_loss(problem, p, state, out, interp, locd);
                // gen_used_loss(problem, p, state, out, interp_used, locd, 100.0);

                // int used_r = 4;
                //
                // for(int oy=std::max(p[0]-used_r,0);oy<=std::min(p[0]+used_r,state.rows-1);oy++)
                //     for(int ox=std::max(p[1]-used_r,0);ox<=std::min(p[1]+used_r,state.cols-1);ox++) {
                //         cv::Vec2i off = {oy-p[0],ox-p[1]};
                //         gen_loc_dist_loss(problem, p, off, state, locd, {sx,sy}, 1.0*step_size);
                //     }

                ceres::Solve(options, &problem, &summary);
            }
            cost_init(p) = summary.final_cost;

            search_init(p) = res;

            loss_count += create_missing_centered_losses(big_problem, loss_status, p, state, out, interp, locd, step_size, {sx,sy}, foldLossIds, flags);

            last_round_updated++;
            succ++;
            fringe.push_back(p);
            if (!used_area.contains({p[1],p[0]})) {
                used_area = used_area | cv::Rect(p[1],p[0],1,1);
            }

            //FIXME better failure/out-of-bounds handling?
            if (loc_valid(state(p))) {
                cv::Rect roi = {locd(p)[0]-T,locd(p)[1]-T,2*T,2*T};
                roi = roi & bounds;
                for(int j=roi.y;j<roi.br().y;j++)
                    for(int i=roi.x;i<roi.br().x;i++) {
                        used(j,i) = std::min(1.0f, used(j,i) + std::max(0.0f, float(1.0-1.0/T*sqrt(sdist(points(locd(p)[1],locd(p)[0]), points(j,i))+1e-2))));
                    }
            }

            if (summary.termination_type == ceres::NO_CONVERGENCE) {
                // std::cout << summary.BriefReport() << "\n";
                stop_gen = generation+1;
                break;
            }
        }

        if (generation > 3) {
            remove_inner_foldlosses(big_problem, 4, state, foldLossIds);
            freeze_inner_params(big_problem, 10, state, out, locd, loss_status, STATE_LOC_VALID);

            options_big.max_num_iterations = 10;
        }
        std::cout << "running big solve" << std::endl;
        ceres::Solve(options_big, &big_problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        cands.resize(0);

        printf("-> total done %d/ fringe: %d\n", succ, fringe.size());
    }

    cv::Mat_<cv::Vec3f> outf;
    out(used_area).convertTo(outf, CV_32F);
    state = state(used_area);

    // cv::imwrite("cost_init.tif", cost_init(used_area));
    // cv::imwrite("search_init.tif", search_init(used_area));
    // cv::imwrite("used.tif", used);
    //
    // std::vector<cv::Vec3f> valid_ps;
    // for(int j=0;j<outf.rows;j++)
    //     for(int i=0;i<outf.cols;i++)
    //         if (state(j, i)== 1)
    //             valid_ps.push_back(outf(j,i));
    //
    // write_ply("points_solve.ply", valid_ps);

    locd(used_area).convertTo(locs, CV_32F);

    cv::Rect out_bounds(0,0,points.cols-2,points.rows-2);
    for(int j=0;j<outf.rows;j++)
        for(int i=0;i<outf.cols;i++)
            if (state(j,i) == 1) {
                cv::Vec2i l = locs(j,i);
                if (out_bounds.contains(l))
                    outf(j, i) = at_int(points, locs(j,i));
                else
                    outf(j, i) = {-1,-1,-1};
            }

    // valid_ps.resize(0);
    // for(int j=0;j<outf.rows;j++)
    //     for(int i=0;i<outf.cols;i++)
    //         if (state(j, i)== 1)
    //             valid_ps.push_back(outf(j,i));
    //
    // write_ply("points_surf.ply", valid_ps);
    //
    // valid_ps.resize(0);
    // std::cout << points.size << std::endl;
    // cv::Mat_<cv::Vec3f> pcrop = points(cv::Rect(seed_x-100,seed_y-400,200,800));
    // for(int j=0;j<pcrop.rows;j++)
    //     for(int i=0;i<pcrop.cols;i++)
    //         valid_ps.push_back(pcrop(j,i));
    //
    // write_ply("input.ply", valid_ps);


    //FIXME wtf?
    // out(used_area).convertTo(outf, CV_32F);

    return outf;
}

cv::Mat_<cv::Vec3f> upsample_with_grounding_simple(const cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, const cv::Size &tgt_size, const cv::Mat_<cv::Vec3f> &points, double sx, double sy)
{
    std::cout << "upsample with simple interpolation " << small.size() << " -> " << tgt_size << std::endl; 
    cv::Mat_<cv::Vec3f> large;
    // cv::Mat_<cv::Vec2f> locs(small.size());
    // large = small;
    // cv::resize(small, large, small.size()*2, cv::INTER_CUBIC);
    cv::resize(small, large, tgt_size, cv::INTER_CUBIC);
    
    cv::Vec2f step = {sx*10, sy*10};
    cv::resize(locs, locs, large.size(), cv::INTER_CUBIC);

#pragma omp parallel for
    for(int j=0;j<large.rows;j++) {
        for(int i=0;i<large.cols;i++) {
            cv::Vec3f tgt = large(j,i);
            if (tgt[0] == -1)
                continue;
            float res;
            
            res = min_loc_dbg(points, locs(j,i), large(j,i), {tgt}, {0}, nullptr, step, 0.001, {}, true);
        }
    }
    
    return large;
}

float dist2(int x, int y)
{
    return sqrt(x*x+y*y);
}

float dist2(const cv::Vec2f &v)
{
    return sqrt(v[0]*v[0]+v[1]*v[1]);
}

cv::Mat_<cv::Vec3f> upsample_with_grounding_skip(const cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, int scale, const cv::Mat_<cv::Vec3f> &points, double sx, double sy)
{
    std::cout << "upsample with interpolation & search " << small.size() << " x " << scale << std::endl; 
    // cv::Mat_<cv::Vec2f> locs(small.size());
    
    cv::Vec2f step_large = {sx*128, sy*128};
    cv::Vec2f step = {sx*10, sy*10};
    
    int rdone = 0;
    
    cv::Size tgt_size = small.size() * scale;
    cv::Mat_<cv::Vec3f> large(tgt_size);
    
    cv::resize(locs, locs, large.size(), cv::INTER_CUBIC);
    
    large.setTo(-1);
    
#pragma omp parallel for
    for(int j=0;j<small.rows-1;j++)
        for(int i=0;i<small.cols-1;i++) {
            cv::Vec3f tgt1 = small(j,i);
            cv::Vec3f tgt2 = small(j,i+1);
            cv::Vec3f tgt3 = small(j+1,i);
            cv::Vec3f tgt4 = small(j+1,i+1);
            // dx /= scale;
            // dy /= scale;
            //TODO same for the others
            if (tgt1[0] == -1)
                continue;
            if (tgt2[0] == -1)
                continue;
            if (tgt3[0] == -1)
                continue;
            if (tgt4[0] == -1)
                continue;
            
            float dx1 = sqrt(sdist(small(j,i+1),small(j,i)))/scale;
            float dx2 = sqrt(sdist(small(j+1,i+1),small(j+1,i)))/scale;
            float dy1 = sqrt(sdist(small(j,i),small(j+1,i)))/scale;
            float dy2 = sqrt(sdist(small(j,i+1),small(j+1,i+1)))/scale;

            //TODO had a neat idea of using perspective transform to get good target distances ... not quite sure which coords to use though so until then use linear interpolation for the target distance
            //we are only interested in distances, so regard the quad as the length of four sides!
            // std::vector<cv::Vec2f> dist_coords = {{0,0},{0,scale},{scale,0},{scale,scale}};
            // std::vector<cv::Vec3f> dist_loc = {tgt1,tgt2,tgt3,tgt4};
            // cv::Mat m = cv::findHomography(dist_coords,dist_loc);
            
            //TODO this will overwrite points on the border between two blocks ... maybe average those?
            //TODO ideally we would do a local fit ... and that do that anyways also already on the initial search whe creating the mesh ...
            for(int ly=0;ly<scale;ly++)
                for(int lx=0;lx<scale;lx++) {
                    // large(j*scale+ly,i*scale+lx) = at_int(points, locs(j*scale+ly,i*scale+lx));
                    // continue;
                    if (!lx && !ly) {
                        large(j*scale,i*scale) = small(j,i);
                        continue;
                    }
                    int nx = scale-lx;
                    int ny = scale-ly;
                    
                    // std::vector<cv::Vec2f> dist_loc;
                    // dist_coords = {{lx,ly}, {nx,ly},{lx,ny},{nx,ny}};
                    
                    // cv::perspectiveTransform(dist_coords, dist_loc, m);
                    
                    // std::vector<float> dists = {dist2(dist_loc[0]),dist2(dist_loc[1]),dist2(dist_loc[3]),dist2(dist_loc[3])};
                    // std::vector<float> dists = {1,0,0,0};
                    float fy = float(ly)/scale;
                    float fx = float(lx)/scale;
                
                    float dx = (1-fy)*dx1+fy*dx2;
                    float dy = (1-fx)*dy1+fx*dy2;
                    std::vector<float> dists = {dist2(lx*dx,ly*dy),dist2(nx*dx,ly*dy),dist2(lx*dx,ny*dy),dist2(nx*dx,ny*dy)};
                    std::vector<float> ws = {1/dists[0], 1/dists[1],1/dists[2],1/dists[3]};
                    float res = min_loc_dbg(points, locs(j*scale+ly,i*scale+lx), large(j*scale+ly,i*scale+lx), {tgt1,tgt2,tgt3,tgt4}, dists, nullptr, step, 0.001, ws, true);
                }
            
        }
        
    return large;
}

cv::Mat_<cv::Vec3f> upsample_with_grounding(cv::Mat_<cv::Vec3f> &small, cv::Mat_<cv::Vec2f> &locs, const cv::Size &tgt_size, const cv::Mat_<cv::Vec3f> &points, double sx, double sy)
{
    int scale = std::max(tgt_size.width/small.cols, tgt_size.height/small.rows);
    cv::Size int_tgt = tgt_size*scale;
    cv::Mat_<cv::Vec3f> large = small;

    while (scale >= 2) {
        scale = 2;
        int_tgt = tgt_size*scale;

        if (small.size() != int_tgt)
            large = upsample_with_grounding_skip(small, locs, scale, points, sx, sy);

        small = large;
        scale = std::max(tgt_size.width/small.cols, tgt_size.height/small.rows);
    }

    if (int_tgt != tgt_size)
        large = upsample_with_grounding_simple(large, locs, tgt_size, points, sx, sy);

    return large;
}

void refine_normal(const std::vector<std::pair<cv::Vec2i,cv::Vec3f>> &refs, cv::Vec3f &point, cv::Vec3f &normal, cv::Vec3f &vx, cv::Vec3f &vy, const std::vector<float> &ws)
{
    //losses are
    //points all should be in plane defined by normal && point
    //vx, vy should explain relative positions between points
    //vx,vy,normal should be orthogonal (?) - for now secondary?

    //things can never be normal enough
    cv::normalize(vx,vx);
    cv::normalize(vy,vy);
    cv::normalize(normal,normal);

    ceres::Problem problem;
    double vxd[3] = {vx[0],vx[1],vx[2]};
    double vyd[3] = {vy[0],vy[1],vy[2]};
    double nd[3] = {normal[0],normal[1],normal[2]};

    for(int j=0;j<refs.size();j++) {
        auto a = refs[j];
        for(int i=0;i<refs.size();i++) {
            auto b = refs[i];
            float w = std::max(ws[j],ws[i]);
            //add vx, vy losses
            if (a.first[0] == b.first[0])
                problem.AddResidualBlock(ParallelLoss::Create(b.second-a.second,w), nullptr, vyd);
            else if (a.first[1] == b.first[1])
                problem.AddResidualBlock(ParallelLoss::Create(b.second-a.second,w), nullptr, vxd);
        }
    }

    problem.AddResidualBlock(OrthogonalLoss::Create(), nullptr, vxd, vyd);
    problem.AddResidualBlock(OrthogonalLoss::Create(), nullptr, vyd, nd);
    problem.AddResidualBlock(OrthogonalLoss::Create(), nullptr, vxd, nd);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << "\n";

    // std::cout << vx << vy << normal << std::endl;

    for(int c=0;c<3;c++) {
        vx[c] = vxd[c];
        vy[c] = vyd[c];
        normal[c] = nd[c];
    }

    cv::normalize(vx,vx);
    cv::normalize(vy,vy);
    cv::normalize(normal,normal);

    // std::cout << vx << vy << normal << std::endl << std::endl;

    return;
}

struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};

uint8_t max_d_ign(uint8_t a, uint8_t b)
{
    if (a == 255)
        return b;
    if (b == 255)
        return a;
    return std::max(a,b);
}

void distanceTransform(const st_1u &src, st_1u &dist, int steps)
{
    st_1u a;
    st_1u b(src);

    for(auto m : src.planes)
        a.planes.push_back(m.clone());

    src.convertTo(a, CV_8UC1);

    st_1u *p1 = &a;
    st_1u *p2 = &b;

    int w = src.planes[0].cols;
    int h = src.planes[0].rows;
    int d = src.planes.size();

#pragma omp parallel for
    for(int k=0;k<d;k++)
        for(int j=0;j<h;j++)
            for(int i=0;i<w;i++)
                if (p1->at(k,j,i)[0] != 0)
                    p1->at(k,j,i)[0] = 255;

    int n_set = 1;
    for(int n=0;n<steps;n++) {
        std::cout << "step " << n << " of " << steps << std::endl;
        n_set = 0;
#pragma omp parallel for
        for(int k=0;k<d;k++)
            for(int j=0;j<h;j++)
                for(int i=0;i<w;i++) {
                    uint8_t dist = p1->at(k,j,i)[0];
                    if (dist == 255) {
                        n_set++;
                        if (k) dist = max_d_ign(dist, p1->at(k-1,j,i)[0]);
                        if (k < d-1) dist = max_d_ign(dist, p1->at(k+1,j,i)[0]);
                        if (j) dist = max_d_ign(dist, p1->at(k,j-1,i)[0]);
                        if (j < h-1) dist = max_d_ign(dist, p1->at(k,j+1,i)[0]);
                        if (i) dist = max_d_ign(dist, p1->at(k,j,i-1)[0]);
                        if (i < w-1) dist = max_d_ign(dist, p1->at(k,j,i+1)[0]);
                        if (dist != 255)
                            p2->at(k,j,i) = dist+1;
                        else
                            p2->at(k,j,i) = dist;
                    }
                    else
                        p2->at(k,j,i) = dist;

                }
        st_1u *tmp = p1;
        p1 = p2;
        p2 = tmp;
    }

    dist = *p1;

#pragma omp parallel for
    for(int k=0;k<d;k++)
        for(int j=0;j<h;j++)
            for(int i=0;i<w;i++)
                if (dist.at(k,j,i)[0] == 255)
                    dist.at(k,j,i)[0] = steps;
}

template <typename T, typename C>
int gen_space_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, float w = 0.1)
{
    if (!loc_valid(state(p)))
        return 0;

    problem.AddResidualBlock(EmptySpaceLossAcc<T,C>::Create(t, w), nullptr, &loc(p)[0]);

    return 1;
}

template <typename I>
int gen_space_line_loss_slow(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, const I &interp, int steps, float w = 0.1)
{
    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    //TODO this will always succeed, but costfunction might not actually work, maybe actually check if it can be added?
    problem.AddResidualBlock(EmptySpaceLineLoss<I>::Create(interp, steps/2, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

template <typename T, typename C>
int gen_space_line_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps, float w = 0.1, float dist_th = 2)
{
    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    // Chunked3dAccessor<T,C> a(t);

    // float len = cv::norm(loc(p)-loc(p+off));
    //
    // double dist = 0;
    // for(int i=0;i<=len;i++) {
    //     float f1 = float(i)/len;
    //     float f2 = 1-f1;
    //     cv::Vec3d l = loc(p)*f1 + loc(p+off)*f2;
    //     double d2 = a.safe_at(l);
    //     dist = std::max(dist, d2);
    // }
    //
    // if (dist >= dist_th)
    //     return 0;

    //TODO this will always succeed, but costfunction might not actually work, maybe actually check if it can be added?
    problem.AddResidualBlock(EmptySpaceLineLossAcc<T,C>::Create(t, steps, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

template <typename T, typename C>
int gen_anchor_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, double *anchor, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps, float w = 0.001)
{
    return 0;

    if (!loc_valid(state(p)))
        return 0;
    if (!loc_valid(state(p+off)))
        return 0;

    anchor[0] = loc(p+off)[0];
    anchor[1] = loc(p+off)[1];
    anchor[2] = loc(p+off)[2];

    //TODO this will always succeed, but costfunction might not actually work, maybe actually check if it can be added?
    problem.AddResidualBlock(AnchorLoss<T,C>::Create(t, w), nullptr, &loc(p)[0], &loc(p+off)[0]);

    return 1;
}

float space_trace_dist_w = 1.0;

//create all valid losses for this point
template <typename I, typename T, typename C>
int emptytrace_create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, const I &interp, Chunked3d<T,C> &t, float unit, int flags = 0)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, state, loc, flags & OPTIMIZE_ALL);

    //vertical
    count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, state, loc, flags & OPTIMIZE_ALL);
    count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, state, loc, flags & OPTIMIZE_ALL);

    //direct neighboars
    count += gen_dist_loss(problem, p, {0,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {0,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,0}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    //diagonal neighbors
    count += gen_dist_loss(problem, p, {1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {1,1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);
    count += gen_dist_loss(problem, p, {-1,-1}, state, loc, unit, flags & OPTIMIZE_ALL, nullptr, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        count += gen_space_loss(problem, p, state, loc, t);

        count += gen_space_line_loss(problem, p, {1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {-1,0}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,1}, state, loc, t, unit);
        count += gen_space_line_loss(problem, p, {0,-1}, state, loc, t, unit);
    }

    return count;
}

template <typename T, typename C>
int conditional_spaceline_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_space_line_loss(problem, p, off, state, loc, t, steps));
    return set;
};


template <typename T, typename C>
int conditional_anchor_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, double *anchor, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, Chunked3d<T,C> &t, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_anchor_loss(problem, p, off, anchor, state, loc, t, steps));
    return set;
};

template <typename I>
int conditional_spaceline_loss_slow(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, const I &interp, int steps)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_space_line_loss_slow(problem, p, off, state, loc, interp, steps));
    return set;
};

//create only missing losses so we can optimize the whole problem
template <typename I, typename T, typename C>
int emptytrace_create_missing_centered_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, const I &interp, Chunked3d<T,C> &t, float unit, int flags = SPACE_LOSS | OPTIMIZE_ALL)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    // if (flags & SPACE_LOSS) {
        count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, state, loc, flags);

        //vertical
        count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, state, loc, flags);
        count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, state, loc, flags);
    // }

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, state, loc, unit, flags, space_trace_dist_w);

    if (flags & SPACE_LOSS) {
        if (!loss_mask(6, p, {0,0}, loss_status))
            count += set_loss_mask(6, p, {0,0}, loss_status, gen_space_loss(problem, p, state, loc, t));

        count += conditional_spaceline_loss(7, p, {1,0}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(7, p, {-1,0}, loss_status, problem, state, loc, t, unit);

        count += conditional_spaceline_loss(8, p, {0,1}, loss_status, problem, state, loc, t, unit);
        count += conditional_spaceline_loss(8, p, {0,-1}, loss_status, problem, state, loc, t, unit);

        //FIXME should anchor loss be from last good position (potentially of the anchors anchor?)
        count += conditional_anchor_loss(9, p, {1,0}, &a1(p)[0], loss_status, problem, state, loc, t, unit);
        count += conditional_anchor_loss(10, p, {-1,0}, &a2(p)[0], loss_status, problem, state, loc, t, unit);

        count += conditional_anchor_loss(11, p, {0,1}, &a3(p)[0], loss_status, problem, state, loc, t, unit);
        count += conditional_anchor_loss(12, p, {0,-1}, &a4(p)[0], loss_status, problem, state, loc, t, unit);
    }

    return count;
}

//optimize within a radius, setting edge points to constant
template <typename I, typename T, typename C>
float local_optimization(int radius, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, const I &interp, Chunked3d<T,C> &t, float unit, bool quiet = false)
{
    ceres::Problem problem;
    cv::Mat_<uint16_t> loss_status(state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius)
                emptytrace_create_missing_centered_losses(problem, loss_status, op, state, locs, a1,a2,a3,a4, interp, t, unit);
        }
    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) > radius && problem.HasParameterBlock(&locs(op)[0]))
                problem.SetParameterBlockConstant(&locs(op)[0]);
        }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    options.function_tolerance = 1e-4;
    // options.num_threads = 1;
    // options.num_threads = omp_get_max_threads();

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!quiet)
        std::cout << "local solve radius" << radius << " " << summary.BriefReport() << std::endl;

    return sqrt(summary.final_cost/summary.num_residual_blocks);
}

//optimize within a radius, setting edge points to constant
template <typename I, typename T, typename C>
float local_inpaint_optimization(int radius, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, const I &interp, Chunked3d<T,C> &t, float unit, bool quiet = false)
{
    ceres::Problem problem;
    //FIXME I think this could be way faster!
    cv::Mat_<uint16_t> loss_status(state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,locs.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,locs.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius)
                emptytrace_create_missing_centered_losses(problem, loss_status, op, state, locs, a1,a2,a3,a4, interp, t, unit);
        }
        for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,locs.rows-1);oy++)
            for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,locs.cols-1);ox++) {
                cv::Vec2i op = {oy, ox};
                if ((cv::norm(p-op) > radius || (state(op) & STATE_LOC_VALID)) && problem.HasParameterBlock(&locs(op)[0]))
                    problem.SetParameterBlockConstant(&locs(op)[0]);
            }


        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 10000;
        options.function_tolerance = 1e-4;
        // options.num_threads = 1;
        // options.num_threads = omp_get_max_threads();

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (!quiet)
            std::cout << "local solve radius" << radius << " " << summary.BriefReport() << std::endl;

    return sqrt(summary.final_cost/summary.num_residual_blocks);
}


//use closing operation to add inner points, TODO should probably also implement fringe based extension ...
template <typename I, typename T, typename C>
void add_phy_losses_closing(ceres::Problem &big_problem, int radius, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, cv::Mat_<uint16_t> &loss_status, const std::vector<cv::Vec2i> &cands, float unit, float phys_fail_th, const
I &interp, Chunked3d<T,C> &t, std::vector<cv::Vec2i> &added)
{
    cv::Mat_<float> dist(state.size());

    cv::Mat_<uint8_t> masked;
    bitwise_and(state, (uint8_t)STATE_LOC_VALID, masked);

    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});

    cv::dilate(masked, masked, m, {-1,-1}, radius);
    cv::erode(masked, masked, m, {-1,-1}, radius);

    int r2 = 1;

    cv::Mat_<cv::Vec3d> _empty;

    //FIXME use fringe-like approach for better ordering!
    for(int j=0;j<locs.rows;j++)
        for(int i=0;i<locs.cols;i++) {
            cv::Vec2i p = {j,i};
            if (!masked(p))
                continue;

            // if (state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) {
            //     //just fill in ... should not be necessary?
            //     emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, _empty, _empty, _empty, _empty, interp, t, unit, OPTIMIZE_ALL);
            //     continue;
            // }

            if ((state(p) & (STATE_COORD_VALID | STATE_LOC_VALID)) == 0) {
                int ref_count = 0;
                cv::Vec3d avg = {0,0,0};
                for(int oy=std::max(p[0]-r2,0);oy<=std::min(p[0]+r2,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r2,0);ox<=std::min(p[1]+r2,locs.cols-1);ox++)
                        if (state(oy,ox) & STATE_COORD_VALID) {
                            avg += locs(oy,ox);
                            ref_count++;
                        }
                avg /= ref_count;

                if (ref_count < 2)
                    continue;

                locs(p) = avg;
                state(p) |= STATE_COORD_VALID;

                ceres::Problem problem;

                int local_loss_count = emptytrace_create_centered_losses(problem, p, state, locs, interp, t, unit);

                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.max_num_iterations = 1000;
                // options.num_threads = 1;
                // options.num_threads = omp_get_max_threads();
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                // local_inpaint_optimization(4, p, state, locs, a1, a2, a3, a4, interp, t, unit);

                //
                // double loss1 = summary.final_cost;

                // std::cout << loss1 << std::endl;

                // if (loss1 > phys_fail_th) {
                //     float err = 0;
                //     for(int range = 1; range<=16;range++) {
                //         err = local_optimization(range, p, state, locs, a1, a2, a3, a4, interp, t, unit);
                //         if (err <= phys_fail_th)
                //             break;
                //     }
                //
                //     if (err > phys_fail_th)
                //         std::cout << std::endl << "WARNING WARNING WARNING" << std::endl << "fix phys inpaint init! " << loss1 << std::endl << std::endl;
                // }
/*
                if (loss1 > phys_fail_th) {
                    std::cout << "fix phys inpaint init! " << loss1 << std::endl;
                }
                else*/
                emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, _empty, _empty, _empty, _empty, interp, t, unit, OPTIMIZE_ALL);
                added.push_back(p);
            }

    }
}
template <typename I, typename T, typename C>
void area_wrap_phy_losses_closing_list(const cv::Rect &roi, ceres::Problem &big_problem, int radius, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, cv::Mat_<uint16_t> &loss_status, std::vector<cv::Vec2i> cands, float unit, float phys_fail_th, const I &interp, Chunked3d<T,C> &t, std::vector<cv::Vec2i> &added, bool global_opt, bool use_area)
{
    for(auto &p : cands)
        p -= cv::Vec2i(roi.y,roi.x);

    std::vector<cv::Vec2i> tmp_added;

    cv::Mat_<uint8_t> state_view = state(roi);
    cv::Mat_<cv::Vec3d> locs_view = locs(roi);
    cv::Mat_<cv::Vec3d> a1_view = a1(roi);
    cv::Mat_<cv::Vec3d> a2_view = a2(roi);
    cv::Mat_<cv::Vec3d> a3_view = a3(roi);
    cv::Mat_<cv::Vec3d> a4_view = a4(roi);
    cv::Mat_<uint16_t> loss_status_view = loss_status(roi);

    add_phy_losses_closing_list(big_problem, radius, state_view, locs_view, a1_view, a2_view, a3_view, a4_view, loss_status_view, cands, unit, phys_fail_th, interp, t, tmp_added, global_opt, use_area);

    for(auto &p : tmp_added)
        added.push_back(p+cv::Vec2i(roi.y,roi.x));
}


static float min_dist(const cv::Vec2i &p, const std::vector<cv::Vec2i> &list)
{
    double dist = 10000000000;
    for(auto &o : list) {
        if (o[0] == -1 || o == p)
            continue;
        dist = std::min(cv::norm(o-p), dist);
    }

    return dist;
}

static cv::Point2i extract_point_min_dist(std::vector<cv::Vec2i> &cands, std::vector<cv::Vec2i> &blocked, int &idx, float dist)
{
    for(int i=0;i<cands.size();i++) {
        cv::Vec2i p = cands[(i + idx) % cands.size()];

        if (p[0] == -1)
            continue;

        if (min_dist(p, blocked) >= dist) {
            cands[(i + idx) % cands.size()] = {-1,-1};
            idx = (i + idx + 1) % cands.size();

            return p;
        }
    }

    return {-1,-1};
}

//collection of points which can be retrieved with minimum distance requirement
class OmpThreadPointCol
{
public:
    OmpThreadPointCol(float dist, const std::vector<cv::Vec2i> &src) :
        _thread_count(omp_get_max_threads()),
        _dist(dist),
        _points(src),
        _thread_points(_thread_count,{-1,-1}),
        _thread_idx(_thread_count, -1) {};
    cv::Point2i next()
    {
        int t_id = omp_get_thread_num();
        if (_thread_idx[t_id] == -1)
            _thread_idx[t_id] = rand() % _thread_count;
        _thread_points[t_id] = {-1,-1};
#pragma omp critical
        _thread_points[t_id] = extract_point_min_dist(_points, _thread_points, _thread_idx[t_id], _dist);
        return _thread_points[t_id];
    }
protected:
    int _thread_count;
    float _dist;
    std::vector<cv::Vec2i> _points;
    std::vector<cv::Vec2i> _thread_points;
    std::vector<int> _thread_idx;
};

static int gen_dbg = 0;

//use closing operation to add inner points, TODO should probably also implement fringe based extension ...
template <typename I, typename T, typename C>
void add_phy_losses_closing_list(ceres::Problem &big_problem, int radius, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &locs, cv::Mat_<cv::Vec3d> &a1, cv::Mat_<cv::Vec3d> &a2, cv::Mat_<cv::Vec3d> &a3, cv::Mat_<cv::Vec3d> &a4, cv::Mat_<uint16_t> &loss_status, const std::vector<cv::Vec2i> &cands, float unit, float phys_fail_th, const
I &interp, Chunked3d<T,C> &t, std::vector<cv::Vec2i> &added, bool global_opt, bool use_area)
{
    cv::Mat_<float> dist(state.size());

    cv::Mat_<uint8_t> masked;
    bitwise_and(state, (uint8_t)(STATE_LOC_VALID | STATE_COORD_VALID), masked);

    cv::Mat m = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5,5});

    cv::imwrite("gen_dbg_src_"+std::to_string(gen_dbg)+".tif", masked);

    cv::dilate(masked, masked, m, {-1,-1}, radius/2);
    cv::erode(masked, masked, m, {-1,-1}, radius/2);

    // cv::imwrite("gen_dbg_mask_"+std::to_string(gen_dbg)+".tif", masked);
    // cv::Mat_<uint8_t> dbg_ref(masked.size(),0);
    // cv::Mat_<uint8_t> dbg_ref2(masked.size(),0);


    int r = 1;
    int r2 = 3;

    cv::Mat_<cv::Vec3d> _empty;

    std::vector<cv::Vec2i> use_cands;

    if (use_area) {
        for(int j=0;j<locs.rows;j++)
            for(int i=0;i<locs.cols;i++) {
                cv::Vec2i p = {j,i};
                if (!masked(p))
                    continue;

                if (state(p) & (STATE_COORD_VALID | STATE_LOC_VALID))
                    continue;

                use_cands.push_back(p);
            }
    }
    else
        use_cands = cands;

    int changed = 10;
    while (changed >= 10) {
        changed = 0;

        OmpThreadPointCol threadcol(9, use_cands);

#pragma omp parallel
        while (true)
        {
            cv::Vec2i p = threadcol.next();

            if (p[0] == -1)
                break;

            if (!masked(p))
                continue;

            if (state(p) & (STATE_COORD_VALID | STATE_LOC_VALID))
                continue;

            int ref_count = 0;
            cv::Vec3d avg = {0,0,0};
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,locs.rows-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,locs.cols-1);ox++)
                    if (state(oy,ox) & (STATE_COORD_VALID | STATE_LOC_VALID)) {
                        avg += locs(oy,ox);
                        ref_count++;
                    }
            avg /= ref_count;

            int ref_count2 = 0;
            for(int oy=std::max(p[0]-r2,0);oy<=std::min(p[0]+r2,locs.rows-1);oy++)
                for(int ox=std::max(p[1]-r2,0);ox<=std::min(p[1]+r2,locs.cols-1);ox++)
                    if (state(oy,ox) & (STATE_COORD_VALID | STATE_LOC_VALID)) {
                        ref_count2++;
                    }


            // std::cout << "try inpaint " << ref_count << std::endl;

            // dbg_ref(p) = ref_count;
            // dbg_ref2(p) = ref_count2;

            if (ref_count < 4)
                continue;

            if (ref_count2 < 25)
                continue;

#pragma omp atomic
            changed++;

            locs(p) = avg;
            state(p) |= STATE_COORD_VALID;

            ceres::Problem problem;

            int local_loss_count = emptytrace_create_centered_losses(problem, p, state, locs, interp, t, unit);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 1000;
            // options.num_threads = 1;
            // options.num_threads = omp_get_max_threads();
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // local_inpaint_optimization(4, p, state, locs, a1, a2, a3, a4, interp, t, unit);

            // local_inpaint_optimization(2, p, state, locs, a1, a2, a3, a4, interp, t, unit, true);
            // local_optimization(4, p, state, locs, a1, a2, a3, a4, interp, t, unit, true);

            //

            local_inpaint_optimization(2, p, state, locs, a1, a2, a3, a4, interp, t, unit, true);
            local_optimization(4, p, state, locs, a1, a2, a3, a4, interp, t, unit, true);

            ceres::Solve(options, &problem, &summary);
            double loss1 = summary.final_cost;

            // std::cout << loss1 << std::endl;

            // if (loss1 > phys_fail_th) {
            //     float err = 0;
            //     for(int range = 1; range<=16;range++) {
            //         err = local_optimization(range, p, state, locs, a1, a2, a3, a4, interp, t, unit);
            //         if (err <= phys_fail_th)
            //             break;
            //     }
            //
            //     if (err > phys_fail_th)
            //         std::cout << std::endl << "WARNING WARNING WARNING" << std::endl << "fix phys inpaint init! " << loss1 << std::endl << std::endl;
            // }

            if (loss1 > phys_fail_th) {
                std::cout << "fix phys inpaint init! " << loss1 << std::endl;
                // local_optimization(4, p, state, locs, a1, a2, a3, a4, interp, t, unit);
                // state(p) = 0;
            }
            // else {
                if (global_opt) {
    #pragma omp critical
                    emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, _empty, _empty, _empty, _empty, interp, t, unit, OPTIMIZE_ALL);
                }
    #pragma omp critical
                added.push_back(p);
            // }
        }
        // std::cout << "iter changed " << changed << std::endl;
    }

    // cv::imwrite("gen_dbg_ref_"+std::to_string(gen_dbg)+".tif", dbg_ref);
    // cv::imwrite("gen_dbg_ref2_"+std::to_string(gen_dbg)+".tif", dbg_ref2);
    // gen_dbg++;
}


template <typename E>
E _max_d_ign(const E &a, const E &b)
{
    if (a == E(-1))
        return b;
    if (b == E(-1))
        return a;
    return std::max(a,b);
}

template <typename T, typename E>
void _dist_iteration(T &from, T &to, int s)
{
    E magic = -1;
#pragma omp parallel for
    for(int k=0;k<s;k++)
        for(int j=0;j<s;j++)
            for(int i=0;i<s;i++) {
                E dist = from(k,j,i);
                if (dist == magic) {
                    if (k) dist = _max_d_ign(dist, from(k-1,j,i));
                    if (k < s-1) dist = _max_d_ign(dist, from(k+1,j,i));
                    if (j) dist = _max_d_ign(dist, from(k,j-1,i));
                    if (j < s-1) dist = _max_d_ign(dist, from(k,j+1,i));
                    if (i) dist = _max_d_ign(dist, from(k,j,i-1));
                    if (i < s-1) dist = _max_d_ign(dist, from(k,j,i+1));
                    if (dist != magic)
                        to(k,j,i) = dist+1;
                    else
                        to(k,j,i) = dist;
                }
                else
                    to(k,j,i) = dist;

            }
}

template <typename T, typename E>
T distance_transform(const T &chunk, int steps, int size)
{
    T c1 = xt::empty<E>(chunk.shape());
    T c2 = xt::empty<E>(chunk.shape());

    c1 = chunk;

    E magic = -1;

    for(int n=0;n<steps/2;n++) {
        _dist_iteration<T,E>(c1,c2,size);
        _dist_iteration<T,E>(c2,c1,size);
    }

#pragma omp parallel for
    for(int z=0;z<size;z++)
        for(int y=0;y<size;y++)
            for(int x=0;x<size;x++)
                if (c1(z,y,x) == magic)
                    c1(z,y,x) = steps;

    return c1;
}

struct thresholdedDistance
{
    enum {BORDER = 16};
    enum {CHUNK_SIZE = 64};
    const std::string CHUNK_DIR = "/home/hendrik/.cache/vcchunksdist_16_64";
    enum {FILL_V = 0};
    template <typename T, typename E> void compute(const T &large, T &small)
    {
        T outer = xt::empty<E>(large.shape());

        int s = CHUNK_SIZE+2*BORDER;
        E magic = -1;

        int good_count = 0;

#pragma omp parallel for
        for(int z=0;z<s;z++)
            for(int y=0;y<s;y++)
                for(int x=0;x<s;x++)
                    if (large(z,y,x) < 50)
                        outer(z,y,x) = magic;
        else {
            good_count++;
            outer(z,y,x) = 0;
        }

        outer = distance_transform<T,E>(outer, 15, s);

        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);

        auto crop_outer = view(outer, xt::range(low,high),xt::range(low,high),xt::range(low,high));

        small = crop_outer;
    }

};

float dist_th = 2.0;

QuadSurface *empty_space_tracing_quad_phys(z5::Dataset *ds, float scale, ChunkCache *cache, cv::Vec3f origin, float step)
{
    ALifeTime f_timer("empty space tracing\n");
    DSReader reader = {ds,scale,cache};

    int stop_gen = 50;

    //FIXME show and handle area edge!
    int w = 2*stop_gen+50;
    int h = w;
    int z = w;
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);
    // cv::normalize(normal, normal);

    int x0 = w/2;
    int y0 = h/2;

    // PlaneSurface plane(origin, normal);
    // cv::Mat_<cv::Vec3f> coords(h,w);
    // plane.gen(&coords, nullptr, size, nullptr, reader.scale, {-w/2,-h/2,0});

    // coords *= reader.scale;
    // double off = 0;
    // cv::Mat_<uint8_t> slice(h,w);
    //
    // st_1u vol;
    // st_1u voldist;
    //
    // std::cout << " reading " << double(w)*h*z/1000/1000 << "M voxels" << std::endl;
    // ALifeTime *timer = new ALifeTime("reading...");
    // for(int zi=0;zi<z;zi++) {
    //     double off = zi - z/2;
    //     cv::Mat_<cv::Vec3f> offmat(size, normal*off);
    //     readInterpolated3D(slice, reader.ds, coords+offmat, reader.cache);
    //     vol.planes.push_back(slice.clone());
    // }
    // delete timer;

//     timer = new ALifeTime("thresholding...");
// #pragma omp parallel for
//     for(auto &p : vol.planes)
//         cv::threshold(p, p, 1, 1, cv::THRESH_BINARY);
//     delete timer;
//
//     timer = new ALifeTime("distancestransform...");
//     distanceTransform(vol, voldist, 40);
//     delete timer;


    thresholdedDistance compute;

    Chunked3d<uint8_t,thresholdedDistance> proc_tensor(compute, ds, cache);


    passTroughComputor pass;

    Chunked3d<uint8_t,passTroughComputor> dbg_tensor(pass, ds, cache);

    std::cout << "seed val " << origin << " " <<
    (int)dbg_tensor(origin[2],origin[1],origin[0]) << std::endl;

    ALifeTime *timer = new ALifeTime("search & optimization ...");

    //start of empty space tracing
    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp_global(proc_tensor);

    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;

    //TODO use scaling and average diffeence vecs for init?
    float D = sqrt(2);

    float T = step;
    float Ts = step*reader.scale;

    int r = 1;
    int r2 = 3;


    cv::Mat_<cv::Vec3d> a1(size);
    cv::Mat_<cv::Vec3d> a2(size);
    cv::Mat_<cv::Vec3d> a3(size);
    cv::Mat_<cv::Vec3d> a4(size);

    cv::Mat_<cv::Vec3d> locs(size,cv::Vec3f(-1,-1,-1));
    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint8_t> phys_fail(size,0);
    cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);

    cv::Vec3f vx = {1,0,0} ;//vx_from_orig_norm(origin, normal);
    cv::Vec3f vy = {0,1,0} ;//vy_from_orig_norm(origin, normal);

    cv::Rect used_area(x0,y0,2,2);
    //these are locations in the local volume!
    locs(y0,x0) = origin;
    locs(y0,x0+1) = origin+vx*0.1;
    locs(y0+1,x0) = origin+vy*0.1;
    locs(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

    // cv::Mat_<uint8_t> dists(500,500);
    // for(int j=0;j<500;j++)
    //     for(int i=0;i<500;i++) {
    //         locs(y0+j,x0+i) = origin+vx*i + vy*j;
    //         cv::Vec3i l = locs(y0+j,x0+i);
    //         dists(j,i) = dbg_tensor(l[2],l[1],l[0]);
    //     }
    //
    // cv::imwrite("dbg.tif", dists);
    // return new QuadSurface(locs(cv::Rect(x0,y0,500,500)),{1,1});

    ceres::Problem big_problem;

    int loss_count;
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0}, state, locs, a1,a2,a3,a4,interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0}, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0,x0+1}, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);
    loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, {y0+1,x0+1}, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);

    //TODO only fix later
    // big_problem.SetParameterBlockConstant(&locs(y0,x0)[0]);
    // big_problem.SetParameterBlockConstant(&locs(y0+1,x0)[0]);
    // big_problem.SetParameterBlockConstant(&locs(y0,x0+1)[0]);
    // big_problem.SetParameterBlockConstant(&locs(y0+1,x0+1)[0]);

    std::cout << "init loss count " << loss_count << std::endl;

    ceres::Solver::Options options_big;
    // options_big.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_big.linear_solver_type = ceres::SPARSE_SCHUR;
    // options_big.linear_solver_type = ceres::DENSE_QR;
    // options_big.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    // options_big.dense_linear_algebra_library_type = ceres::CUDA;
    // options_big.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options_big.minimizer_progress_to_stdout = false;
    //TODO check for update ...
    // options_big.enable_fast_removal = true;
    // options_big.num_threads = omp_get_max_threads();
    // options_big.num_threads = 1;
    options_big.max_num_iterations = 10000;
    // options_big.function_tolerance = 1e-4;


    ceres::Solver::Summary big_summary;
    ceres::Solve(options_big, &big_problem, &big_summary);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.dense_linear_algebra_library_type = ceres::CUDA;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-3;
    // options.num_threads = omp_get_max_threads();
    // options.num_threads = 1;


    std::cout << big_summary.BriefReport() << "\n";

    fringe.push_back({y0,x0});
    fringe.push_back({y0+1,x0});
    fringe.push_back({y0,x0+1});
    fringe.push_back({y0+1,x0+1});

    // for(auto &p : fringe) {
    //     cv::Vec3d l = locs(p);
    //     std::cout << l << std::endl;
        // double dist;
        // interp_global.Evaluate(l[2],l[1],l[0], &dist);
        // // std::cout << "init dist1 " << dist << std::endl;
        // // interp_global.Evaluate(l[0],l[1],l[2], &dist);
        // std::cout << "init dist2 " << dist << std::endl;
        // std::cout << "init dist3 " << (int)proc_tensor(l[2],l[1],l[0]) << std::endl;
        // std::cout << "init dist4 " << (int)dbg_tensor(l[2],l[1],l[0]) << std::endl;
    // }

    ceres::Solve(options_big, &big_problem, &big_summary);


    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    int succ = 0;

    int generation = 0;
    int phys_fail_count = 0;
    double phys_fail_th = 0.1;

    int max_local_opt_r = 4;

    omp_set_num_threads(1);

    std::vector<float> gen_max_cost;
    std::vector<float> gen_avg_cost;

    while (fringe.size()) {
        bool global_opt = generation <= 20;

        ALifeTime timer_gen;
        timer_gen.del_msg = "time per generation ";

        int phys_fail_count_gen = 0;
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        std::vector<cv::Vec2i> rest_ps;

        for(auto p : fringe)
        {
            if ((state(p) & STATE_LOC_VALID) == 0) {
                if (state(p) & STATE_COORD_VALID)
                    for(auto n : neighs)
                        if (bounds.contains(p+n)
                            && (state(p+n) & (STATE_PROCESSING | STATE_LOC_VALID | STATE_COORD_VALID)) == 0) {
                            rest_ps.push_back(p+n);
                            }
                continue;
            }

            for(auto n : neighs)
                if (bounds.contains(p+n)
                    && (state(p+n) & STATE_PROCESSING) == 0
                    && (state(p+n) & STATE_LOC_VALID) == 0) {
                    state(p+n) |= STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        printf("gen %d processing %d fringe cands (total done %d fringe: %d\n", generation, cands.size(), succ, fringe.size());
        fringe.resize(0);

        std::cout << "cands " << cands.size() << std::endl;

        if (!cands.size())
            continue;

        // auto rng = std::default_random_engine {};
        // std::shuffle(std::begin(cands), std::end(cands), rng);

        int succ_gen = 0;
        std::vector<cv::Vec2i> succ_gen_ps;

        std::vector<cv::Vec2i> thread_ps(omp_get_max_threads());

        // for(auto &p : thread_ps)
            // p = {-1,-1};

        OmpThreadPointCol cands_threadcol(max_local_opt_r*2+1, cands);

#pragma omp parallel
        {
            CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
//             int idx = rand() % cands.size();
            while (true) {
                cv::Vec2i p = cands_threadcol.next();
                if (p[0] == -1)
                    break;


//                 int parallism = 0;
// #pragma omp critical
//                 for(auto &o : thread_ps)
//                     if (o[0] != -1)
//                         parallism++;
//
//                 std::cout << "threads active: " << parallism << std::endl;

                if (state(p) & STATE_LOC_VALID)
                    continue;

                int ref_count = 0;
                cv::Vec3d avg = {0,0,0};
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,locs.rows-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,locs.cols-1);ox++)
                        if (state(oy,ox) & STATE_LOC_VALID) {
                            ref_count++;
                            avg += locs(oy,ox);
                        }

                // int ref_count2 = 0;
                // for(int oy=std::max(p[0]-r2,0);oy<=std::min(p[0]+r2,locs.rows-1);oy++)
                //     for(int ox=std::max(p[1]-r2,0);ox<=std::min(p[1]+r2,locs.cols-1);ox++)
                //         if (state(oy,ox) & (STATE_LOC_VALID | STATE_COORD_VALID)) {
                //             ref_count2++;
                //         }

                if (ref_count < 2 /*|| (generation > 3 && ref_count2 < 14)*/) {
                    state(p) &= ~STATE_PROCESSING;
#pragma omp critical
                    rest_ps.push_back(p);
                    continue;
                }

                avg /= ref_count;
                locs(p) = avg;


                //TODO don't reinit if we are running on exist cood!

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                int local_loss_count = emptytrace_create_centered_losses(problem, p, state, locs, interp, proc_tensor, Ts);

                //FIXME need to handle the edge of the input definition!!
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                double loss1 = summary.final_cost;

                // std::cout << loss1 << std::endl;

                if (loss1 > phys_fail_th) {
                    cv::Vec3d best_loc = locs(p);
                    double best_loss = loss1;
                    for (int n=0;n<100;n++) {
                        int range = step*10;
                        locs(p) = avg + cv::Vec3d((rand()%(range*2))-range,(rand()%(range*2))-range,(rand()%(range*2))-range);
                        ceres::Solve(options, &problem, &summary);
                        loss1 = summary.final_cost;
                        if (loss1 < best_loss) {
                            best_loss = loss1;
                            best_loc = locs(p);
                        }
                        if (loss1 < phys_fail_th)
                            break;
                    }
                    loss1 = best_loss;
                    locs(p) = best_loc;
                }

                cv::Vec3d phys_only_loc = locs(p);

                gen_space_loss(problem, p, state, locs, proc_tensor);

                gen_space_line_loss(problem, p, {1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {-1,0}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,1}, state, locs, proc_tensor, T, 0.1, 100);
                gen_space_line_loss(problem, p, {0,-1}, state, locs, proc_tensor, T, 0.1, 100);

                double anchors[12];
                gen_anchor_loss(problem, p, {1,0}, anchors, state, locs, proc_tensor, T);
                gen_anchor_loss(problem, p, {-1,0}, anchors+3, state, locs, proc_tensor, T);
                gen_anchor_loss(problem, p, {0,1}, anchors+6, state, locs, proc_tensor, T);
                gen_anchor_loss(problem, p, {0,-1}, anchors+9, state, locs, proc_tensor, T);

                ceres::Solve(options, &problem, &summary);
                // std::cout << summary.BriefReport() << "\n";
                // local_optimization(1, p, state, locs, interp, proc_tensor, Ts, true);

                double dist;
                //check steps
                interp.Evaluate(locs(p)[2],locs(p)[1],locs(p)[0], &dist);
                int count = 0;
                for (auto &off : neighs) {
                    if (state(p+off) & STATE_LOC_VALID) {
                        for(int i=1;i<T;i++) {
                            float f1 = float(i)/T;
                            float f2 = 1-f1;
                            cv::Vec3d l = locs(p)*f1 + locs(p+off)*f2;
                            double d2;
                            interp.Evaluate(l[2],l[1],l[0], &d2);
                            // dist += d2;
                            dist = std::max(dist, d2);
                            count++;
                        }
                    }
                }

                // dist /= count;

                init_dist(p) = dist;

                // std::cout << "dist " << dist << " cost " << summary.final_cost << std::endl;

                //FIXME revisit dists after (every?) iteration?
                if (dist >= dist_th || summary.final_cost >= 0.1) {
                    locs(p) = phys_only_loc;
                    state(p) = STATE_COORD_VALID;
                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts, OPTIMIZE_ALL);
                    }
                    if (loss1 > phys_fail_th) {
                        phys_fail(p) = 1;

                        float err = 0;
                        for(int range = 1; range<=max_local_opt_r;range++) {
                            err = local_optimization(range, p, state, locs, a1,a2,a3,a4, interp, proc_tensor, Ts);
                            if (err <= phys_fail_th)
                                break;
                        }
                        if (err > phys_fail_th) {
                            std::cout << "local phys fail! " << err << std::endl;
#pragma omp atomic
                            phys_fail_count++;
#pragma omp atomic
                            phys_fail_count_gen++;
                        }
                    }
                }
                else {
                    //FIXMe still add (some?) material losses for empty points so we get valid surface structure!

                    if (global_opt) {
#pragma omp critical
                        loss_count += emptytrace_create_missing_centered_losses(big_problem, loss_status, p, state, locs, a1,a2,a3,a4, interp_global, proc_tensor, Ts);
                    }
#pragma omp atomic
                    succ++;
#pragma omp atomic
                    succ_gen++;
#pragma omp critical
                    {
                        if (!used_area.contains({p[1],p[0]})) {
                            used_area = used_area | cv::Rect(p[1],p[0],1,1);
                        }
                    }
                }
#pragma omp critical
                {
                    fringe.push_back(p);
                    succ_gen_ps.push_back(p);
                }
            }
        }

        for(auto p: fringe)
            if (locs(p)[0] == -1)
                std::cout << "impossible!" << p << cv::Vec2i(y0,x0) << std::endl;

        std::vector<cv::Vec2i> added;
        // add_phy_losses_closing_list(big_problem, 20, state, locs, a1,a2,a3,a4, loss_status, cands, Ts, phys_fail_th, interp_global, proc_tensor, added);
        // add_phy_losses_closing_list(big_problem, 20, state, locs, a1,a2,a3,a4, loss_status, rest_ps, Ts, phys_fail_th, interp_global, proc_tensor, added);
        // add_phy_losses_closing_list(big_problem, 20, state, locs, a1,a2,a3,a4, loss_status, cands, Ts, phys_fail_th, interp_global, proc_tensor, added);
        // add_phy_losses_closing_list(big_problem, 20, state, locs, a1,a2,a3,a4, loss_status, rest_ps, Ts, phys_fail_th, interp_global, proc_tensor, added);
        // for(auto &p : added) {
        //     succ_gen_ps.push_back(p);
        //     fringe.push_back(p);
        // }

        if (generation >= 3) {
            options_big.max_num_iterations = 10;
        }

        //this actually did work (but was slow ...)
        if (phys_fail_count_gen) {
            options_big.minimizer_progress_to_stdout = true;
            options_big.max_num_iterations = 100;
        }
        else
            options_big.minimizer_progress_to_stdout = false;

        if (!global_opt) {
            std::vector<cv::Vec2i> opt_local;
            for(auto p : succ_gen_ps)
                if (p[0] % 4 == 0 && p[1] % 4 == 0)
                    opt_local.push_back(p);

            if (opt_local.size()) {
                OmpThreadPointCol opt_local_threadcol(17, opt_local);

#pragma omp parallel
                while (true)
                {
                    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(8, p, state, locs, a1,a2,a3,a4, interp, proc_tensor, Ts, true);
                }
            }
        }
        else {
            std::cout << "running big solve" << std::endl;
            ceres::Solve(options_big, &big_problem, &big_summary);
            std::cout << big_summary.BriefReport() << "\n";
            std::cout << "avg err:" << sqrt(big_summary.final_cost/big_summary.num_residual_blocks) << std::endl;
        }


        //FIXME maybe not remove all?
        // for(int j=0;j<locs.rows;j++)
        //     for(int i=0;i<locs.cols;i++) {
        //         cv::Vec2i p = {j,i};
        //         if (state(p) & STATE_LOC_VALID) {
        //             for (auto &off : neighs) {
        //                 if (state(p+off) & STATE_LOC_VALID) {
        //                     double dist = 0;
        //                     for(int i=0;i<=T;i++) {
        //                         float f1 = float(i)/T;
        //                         float f2 = 1-f1;
        //                         cv::Vec3d l = locs(p)*f1 + locs(p+off)*f2;
        //                         double d2;
        //                         interp_global.Evaluate(l[2],l[1],l[0], &d2);
        //                         dist = std::max(dist, d2);
        //                     }
        //
        //                     if (dist >= 2) {
        //                         if (generation < 20) {
        //                             if (big_problem.HasParameterBlock(&locs(p)[0])) {
        //                                 big_problem.RemoveParameterBlock(&locs(p)[0]);
        //                                 loss_status(p) = 0;
        //                             }
        //                         }
        //                         else
        //                             state(p) = STATE_COORD_VALID;
        //                     }
        //                 }
        //             }
        //         }
        //     }

        cv::Rect used_plus = {used_area.x-8,used_area.y-8,used_area.width+16,used_area.height+16};
        // area_wrap_phy_losses_closing_list(used_plus, big_problem, generation/2, state, locs, a1,a2,a3,a4, loss_status, rest_ps, Ts, phys_fail_th, interp_global, proc_tensor, added, global_opt, false);
        area_wrap_phy_losses_closing_list(used_plus, big_problem, generation/2, state, locs, a1,a2,a3,a4, loss_status, rest_ps, Ts, phys_fail_th, interp_global, proc_tensor, added, global_opt, false);
        // add_phy_losses_closing_list(big_problem, 20, state, locs, a1,a2,a3,a4, loss_status, rest_ps, Ts, phys_fail_th, interp_global, proc_tensor, added, global_opt);
        for(auto &p : added) {
            // succ_gen_ps.push_back(p);
            fringe.push_back(p);
        }

        if (generation > 10 && global_opt) {
            cv::Mat_<cv::Vec2d> _empty;
            freeze_inner_params(big_problem, 10, state, locs, _empty, loss_status, STATE_LOC_VALID | STATE_COORD_VALID);
        }

        cands.resize(0);

        cv::Mat_<cv::Vec3d> locs_crop = locs(used_area);
        cv::Mat_<uint8_t> state_crop = state(used_area);
        double max_cost = 0;
        double avg_cost = 0;
        int cost_count = 0;
        for(int j=0;j<locs_crop.rows;j++)
            for(int i=0;i<locs_crop.cols;i++) {
                ceres::Problem problem;
                emptytrace_create_centered_losses(problem, {j,i}, state_crop, locs_crop, interp_global, proc_tensor, Ts);
                double cost = 0.0;
                problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
                max_cost = std::max(max_cost, cost);
                avg_cost += cost;
                cost_count++;
            }
        gen_avg_cost.push_back(avg_cost/cost_count);
        gen_max_cost.push_back(max_cost);

        printf("-> total done %d/ fringe: %d surf: %fmm^2\n", succ, fringe.size(), succ*step*step*0.008*0.008);

        timer_gen.unit = succ_gen*0.008*0.008*T*T;
        timer_gen.unit_string = "mm^2";
        print_accessor_stats();

    }
    delete timer;

    // loss_status = loss_status(used_area);
    // cv::imwrite("loss_status.tif", loss_status);

    // CeresGrid2DcvMat3f grid_coords({coords});
    // ceres::BiCubicInterpolator<CeresGrid2DcvMat3f> interp_coords(grid_coords);

    locs = locs(used_area);
    state = state(used_area);
    // init_dist = init_dist(used_area);

    // cv::imwrite("init_dist.tif", init_dist);
    // cv::imwrite("state.tif", state);

    // cv::Mat_<uint8_t> classification(locs.size(), 0);
    // for(int j=0;j<locs.rows;j++)
    //     for(int i=0;i<locs.cols;i++)
    //         classification(j,i) = dbg_tensor(locs(j,i)[2],locs(j,i)[1],locs(j,i)[0]);
    // cv::imwrite("class.tif", classification);

    // cv::Mat_<uint8_t> dists(locs.size(), 0);
    // for(int j=0;j<locs.rows;j++)
    //     for(int i=0;i<locs.cols;i++)
    //         dists(j,i) = proc_tensor(locs(j,i)[2],locs(j,i)[1],locs(j,i)[0]);
    // cv::imwrite("dists.tif", dists);

    // cv::Mat_<cv::Vec3f> points(locs.size(), cv::Vec3f(-1,-1,-1));
    // cv::Mat_<float> dists(points.size(), 1000);
    // for(int j=0;j<locs.rows;j++)
    //     for(int i=0;i<locs.cols;i++) {
    //         if (!loc_valid(state(j,i)))
    //             continue;
    //         cv::Vec3d l = locs(j,i);
    //         double p[3];
    //
    //         interp_coords.Evaluate(l[1],l[0], p);
    //         cv::Vec3f coord = {p[0],p[1],p[2]};
    //         coord += (l[2]-z/2) * normal;
    //         points(j,i) = coord;
    //
    //         double d;
    //         interp.Evaluate<double>(l[2],l[1],l[0], &d);
    //         dists(j,i) = d;
    //     }


    // cv::imwrite("phys_fail.tif", phys_fail*255);

    // cv::imwrite("dists.tif", dists);
    // points *= 1/reader.scale;

    double max_cost = 0;
    double avg_cost = 0;
    int count = 0;
    for(int j=0;j<locs.rows;j++)
        for(int i=0;i<locs.cols;i++) {
            ceres::Problem problem;
            emptytrace_create_centered_losses(problem, {j,i}, state, locs, interp_global, proc_tensor, Ts);
            double cost = 0.0;
            problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
            max_cost = std::max(max_cost, cost);
            avg_cost += cost;
            count++;
        }
    avg_cost /= count;


    // printf("generated approximate surface %fkvx^2\n", succ*step*step/1000000.0);
    printf("generated approximate surface %fmm^2\n", succ*step*step*0.008*0.008);

    QuadSurface *surf = new QuadSurface(locs, {1/T, 1/T});

    surf->meta = new nlohmann::json;
    (*surf->meta)["area_cm"] = succ*step*step*0.008*0.008/100;
    (*surf->meta)["max_cost"] = max_cost;
    (*surf->meta)["avg_cost"] = avg_cost;
    (*surf->meta)["max_gen"] = generation;
    (*surf->meta)["gen_avg_cost"] = gen_avg_cost;
    (*surf->meta)["gen_max_cost"] = gen_max_cost;
    (*surf->meta)["seed"] = {origin[0],origin[1],origin[2]};
    (*surf->meta)["elapsed_time_s"] = f_timer.seconds();

    return surf;
}

using SurfPoint = std::pair<SurfaceMeta*,cv::Vec2i>;

class resId_t
{
public:
    resId_t() {};
    resId_t(int type, SurfaceMeta* sm, cv::Vec2i p) : _type(type), _sm(sm), _p(p) {};
    resId_t(int type, SurfaceMeta* sm, const cv::Vec2i &a, const cv::Vec2i &b) : _type(type), _sm(sm)
    {
        if (a[0] == b[0]) {
            if (a[1] <= b[1])
                _p = a;
            else
                _p = b;
        }
        else if (a[0] < b[0])
            _p = a;
        else
            _p = b;

    }
    bool operator==(const resId_t &o) const
    {
        if (_type != o._type)
            return false;
        if (_sm != o._sm)
            return false;
        if (_p != o._p)
            return false;
        return true;
    }

    int _type;
    SurfaceMeta* _sm;
    cv::Vec2i _p;
};

struct resId_hash {
    size_t operator()(resId_t id) const
    {
        size_t hash1 = std::hash<int>{}(id._type);
        size_t hash2 = std::hash<void*>{}(id._sm);
        size_t hash3 = std::hash<int>{}(id._p[0]);
        size_t hash4 = std::hash<int>{}(id._p[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};


struct SurfPoint_hash {
    size_t operator()(SurfPoint p) const
    {
        size_t hash1 = std::hash<void*>{}(p.first);
        size_t hash2 = std::hash<int>{}(p.second[0]);
        size_t hash3 = std::hash<int>{}(p.second[1]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

//Surface tracking data for loss functions
class SurfTrackerData
{

public:
    cv::Vec2d &loc(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        return _data[{sm,loc}];
    }
    ceres::ResidualBlockId &resId(const resId_t &id)
    {
        return _res_blocks[id];
    }
    bool hasResId(const resId_t &id)
    {
        // std::cout << "check hasResId " << id._sm << " " << id._type << " " << id._p << std::endl;
        return _res_blocks.count(id);
    }
    bool has(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        return _data.count({sm,loc});
    }
    void erase(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        _data.erase({sm,loc});
    }
    void eraseSurf(SurfaceMeta *sm, const cv::Vec2i &loc)
    {
        _surfs[loc].erase(sm);
    }
    std::set<SurfaceMeta*> &surfs(const cv::Vec2i &loc)
    {
        return _surfs[loc];
    }
    cv::Vec3d lookup_int(SurfaceMeta *sm, const cv::Vec2i &p)
    {
        auto id = std::make_pair(sm,p);
        if (!_data.count(id))
            throw std::runtime_error("error, lookup failed!");
        cv::Vec2d l = loc(sm, p);
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, sm->surf()->rawPoints().rows-2,sm->surf()->rawPoints().cols-2};
            cv::Vec2i li = {floor(l[0]),floor(l[1])};
            if (bounds.contains(li))
                return at_int_inv(sm->surf()->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
    }
    bool valid_int(SurfaceMeta *sm, const cv::Vec2i &p)
    {
        auto id = std::make_pair(sm,p);
        if (!_data.count(id))
            return false;
        cv::Vec2d l = loc(sm, p);
        if (l[0] == -1)
            return false;
        else {
            cv::Rect bounds = {0, 0, sm->surf()->rawPoints().rows-2,sm->surf()->rawPoints().cols-2};
            cv::Vec2i li = {floor(l[0]),floor(l[1])};
            if (bounds.contains(li))
            {
                if (sm->surf()->rawPoints()(li[0],li[1])[0] == -1)
                    return false;
                if (sm->surf()->rawPoints()(li[0]+1,li[1])[0] == -1)
                    return false;
                if (sm->surf()->rawPoints()(li[0],li[1]+1)[0] == -1)
                    return false;
                if (sm->surf()->rawPoints()(li[0]+1,li[1]+1)[0] == -1)
                    return false;
                return true;
            }
            else
                return false;
        }
    }
    cv::Vec3d lookup_int_loc(SurfaceMeta *sm, const cv::Vec2f &l)
    {
        if (l[0] == -1)
            return {-1,-1,-1};
        else {
            cv::Rect bounds = {0, 0, sm->surf()->rawPoints().rows-2,sm->surf()->rawPoints().cols-2};
            if (bounds.contains({l[0],l[1]}))
                return at_int_inv(sm->surf()->rawPoints(), l);
            else
                return {-1,-1,-1};
        }
    }
// protected:
    std::unordered_map<SurfPoint,cv::Vec2d,SurfPoint_hash> _data;
    std::unordered_map<resId_t,ceres::ResidualBlockId,resId_hash> _res_blocks;
    std::unordered_map<cv::Vec2i,std::set<SurfaceMeta*>,vec2i_hash> _surfs;
};

int add_surftrack_distloss(SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 1.0)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.has(sm, p))
        return 0;
    if ((state(p+off) & STATE_LOC_VALID) == 0 || !data.has(sm, p+off))
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss2D::Create(unit*cv::norm(off),w), nullptr, &data.loc(sm, p)[0], &data.loc(sm, p+off)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&data.loc(sm, p+off)[0]);

    return 1;
}


int add_surftrack_distloss_3D(cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 1.0)
{
    if ((state(p) & (STATE_COORD_VALID | STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_COORD_VALID | STATE_COORD_VALID)) == 0)
        return 0;

    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &points(p)[0], &points(p+off)[0]);

    // std::cout << cv::norm(points(p)-points(p+off)) << " tgt " << unit << points(p) << points(p+off) << std::endl;
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0)
        problem.SetParameterBlockConstant(&points(p+off)[0]);

    return 1;
}

int cond_surftrack_distloss_3D(int type, SurfaceMeta *sm, cv::Mat_<cv::Vec3d> &points, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_distloss_3D(points, p, off, problem, state, unit, flags, &res);

    data.resId(id) = res;

    return count;
}

int cond_surftrack_distloss(int type, SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
{
    resId_t id(type, sm, p, p+off);
    if (data.hasResId(id))
        return 0;

    add_surftrack_distloss(sm, p, off, data, problem, state, unit, flags, &data.resId(id));

    return 1;
}

// int cond_surftrack_distloss_3d(int type, SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &off, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, float unit, int flags = 0)
// {
//     resId_t id(type, sm, p, p+off);
//
//     if (data.hasResId(id))
//         return 0;
//
//     ceres::ResidualBlockId res;
//     int count = add_surftrack_distloss_3D(sm, p, off, data, problem, state, unit, flags, &res);
//
//     if (count)
//         data.resId(id) = res;
//
//     return count;
// }


int add_surftrack_straightloss(SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, float w = 1.0)
{
    if ((state(p+o1) & STATE_LOC_VALID) == 0 || !data.has(sm, p+o1))
        return 0;
    if ((state(p+o2) & STATE_LOC_VALID) == 0 || !data.has(sm, p+o2))
        return 0;
    if ((state(p+o3) & STATE_LOC_VALID) == 0 || !data.has(sm, p+o3))
        return 0;

    // std::cout << "add straight " << sm << p << o1 << o2 << o3 << std::endl;
    problem.AddResidualBlock(StraightLoss2D::Create(w), nullptr, &data.loc(sm, p+o1)[0], &data.loc(sm, p+o2)[0], &data.loc(sm, p+o3)[0]);

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(sm, p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(sm, p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&data.loc(sm, p+o3)[0]);
    }

    return 1;
}

int add_surftrack_straightloss_3D(const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<cv::Vec3d> &points ,ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, ceres::ResidualBlockId *res = nullptr, float w = 1.0)
{
    if ((state(p+o1) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o2) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+o3) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;

    // std::cout << "add straight " << sm << p << o1 << o2 << o3 << std::endl;
    ceres::ResidualBlockId tmp =
    problem.AddResidualBlock(StraightLoss::Create(w), nullptr, &points(p+o1)[0], &points(p+o2)[0], &points(p+o3)[0]);
    if (res)
        *res = tmp;

    if ((flags & OPTIMIZE_ALL) == 0) {
        if (o1 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o1)[0]);
        if (o2 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o2)[0]);
        if (o3 != cv::Vec2i(0,0))
            problem.SetParameterBlockConstant(&points(p+o3)[0]);
    }

    return 1;
}

int cond_surftrack_straightloss_3D(int type, SurfaceMeta *sm, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<cv::Vec3d> &points, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, int flags = 0, float w = 1.0)
{
    resId_t id(type, sm, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_straightloss_3D(p, o1, o2, o3, points ,problem, state, flags, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

//gen straigt loss given point and 3 offsets
int add_surftrack_surfloss(SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, ceres::ResidualBlockId *res = nullptr, float w = 0.1)
{
    if ((state(p) & STATE_LOC_VALID) == 0 || !data.valid_int(sm, p))
        return 0;

    // std::cout << "add_surftrack_surfloss() " << sm << p << points(p) << " vs " << interp_lin_2d(sm->surf()->rawPoints(), data.loc(sm, p)) << std::endl;

    ceres::ResidualBlockId tmp;
    tmp = problem.AddResidualBlock(SurfaceLossD::Create(sm->surf()->rawPoints(), w), nullptr, &points(p)[0], &data.loc(sm, p)[0]);
    if (res)
        *res = tmp;

    return 1;
}

//gen straigt loss given point and 3 offsets
int cond_surftrack_surfloss(int type, SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points)
{
    resId_t id(type, sm, p);
    if (data.hasResId(id))
        return 0;

    ceres::ResidualBlockId res;
    int count = add_surftrack_surfloss(sm, p, data, problem, state, points, &res);

    if (count)
        data.resId(id) = res;

    return count;
}

//will optimize only the center point
int surftrack_add_local(SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, int flags = 0)
{
    int count = 0;
    //direct
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += add_surftrack_distloss_3D(points, p, {0,1}, problem, state, step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,0}, problem, state, step, flags);
        count += add_surftrack_distloss_3D(points, p, {0,-1}, problem, state, step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,0}, problem, state, step, flags);

        //v
        count += add_surftrack_distloss_3D(points, p, {1,1}, problem, state, step, flags);
        count += add_surftrack_distloss_3D(points, p, {1,-1}, problem, state, step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,1}, problem, state, step, flags);
        count += add_surftrack_distloss_3D(points, p, {-1,-1}, problem, state, step, flags);

        //horizontal
        count += add_surftrack_straightloss_3D(p, {0,-2},{0,-1},{0,0}, points, problem, state);
        count += add_surftrack_straightloss_3D(p, {0,-1},{0,0},{0,1}, points, problem, state);
        count += add_surftrack_straightloss_3D(p, {0,0},{0,1},{0,2}, points, problem, state);

        //vertical
        count += add_surftrack_straightloss_3D(p, {-2,0},{-1,0},{0,0}, points, problem, state);
        count += add_surftrack_straightloss_3D(p, {-1,0},{0,0},{1,0}, points, problem, state);
        count += add_surftrack_straightloss_3D(p, {0,0},{1,0},{2,0}, points, problem, state);
    }
    else {
        count += add_surftrack_distloss(sm, p, {0,1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {1,0}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {0,-1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {-1,0}, data, problem, state, step);

        //diagonal
        count += add_surftrack_distloss(sm, p, {1,1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {1,-1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {-1,1}, data, problem, state, step);
        count += add_surftrack_distloss(sm, p, {-1,-1}, data, problem, state, step);

        //horizontal
        count += add_surftrack_straightloss(sm, p, {0,-2},{0,-1},{0,0}, data, problem, state);
        count += add_surftrack_straightloss(sm, p, {0,-1},{0,0},{0,1}, data, problem, state);
        count += add_surftrack_straightloss(sm, p, {0,0},{0,1},{0,2}, data, problem, state);

        //vertical
        count += add_surftrack_straightloss(sm, p, {-2,0},{-1,0},{0,0}, data, problem, state);
        count += add_surftrack_straightloss(sm, p, {-1,0},{0,0},{1,0}, data, problem, state);
        count += add_surftrack_straightloss(sm, p, {0,0},{1,0},{2,0}, data, problem, state);
    }

    if (flags & SURF_LOSS) {
        count += add_surftrack_surfloss(sm, p, data, problem, state, points);
    }

    return count;
}

//will optimize only the center point
int surftrack_add_global(SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, int flags = 0)
{
    if ((state(p) & (STATE_LOC_VALID | STATE_COORD_VALID)) == 0)
        return 0;

    int count = 0;
    //losses are defind in 3D
    if (flags & LOSS_3D_INDIRECT) {
        // h
        count += cond_surftrack_distloss_3D(0, sm, points, p, {0,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(0, sm, points, p, {1,0}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, sm, points, p, {0,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(1, sm, points, p, {-1,0}, data, problem, state, step, flags);

        //v
        count += cond_surftrack_distloss_3D(2, sm, points, p, {1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(2, sm, points, p, {1,-1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, sm, points, p, {-1,1}, data, problem, state, step, flags);
        count += cond_surftrack_distloss_3D(3, sm, points, p, {-1,-1}, data, problem, state, step, flags);

        //horizontal
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,-2},{0,-1},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,-1},{0,0},{0,1}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(4, sm, p, {0,0},{0,1},{0,2}, points, data, problem, state, flags);

        //vertical
        count += cond_surftrack_straightloss_3D(5, sm, p, {-2,0},{-1,0},{0,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, sm, p, {-1,0},{0,0},{1,0}, points, data, problem, state, flags);
        count += cond_surftrack_straightloss_3D(5, sm, p, {0,0},{1,0},{2,0}, points, data, problem, state, flags);
    }
    //losses are defined on surface
    else {
        //FIXME implement if needed
        // //direct
        // count += add_surftrack_distloss(sm, p, {0,1}, data, problem, state, step);
        // count += add_surftrack_distloss(sm, p, {1,0}, data, problem, state, step);
        // count += add_surftrack_distloss(sm, p, {0,-1}, data, problem, state, step);
        // count += add_surftrack_distloss(sm, p, {-1,0}, data, problem, state, step);
        //
        // //diagonal
        // count += add_surftrack_distloss(sm, p, {1,1}, data, problem, state, step);
        // count += add_surftrack_distloss(sm, p, {1,-1}, data, problem, state, step);
        // count += add_surftrack_distloss(sm, p, {-1,1}, data, problem, state, step);
        // count += add_surftrack_distloss(sm, p, {-1,-1}, data, problem, state, step);
        //
        // //horizontal
        // count += add_surftrack_straightloss(sm, p, {0,-2},{0,-1},{0,0}, data, problem, state);
        // count += add_surftrack_straightloss(sm, p, {0,-1},{0,0},{0,1}, data, problem, state);
        // count += add_surftrack_straightloss(sm, p, {0,0},{0,1},{0,2}, data, problem, state);
        //
        // //vertical
        // count += add_surftrack_straightloss(sm, p, {-2,0},{-1,0},{0,0}, data, problem, state);
        // count += add_surftrack_straightloss(sm, p, {-1,0},{0,0},{1,0}, data, problem, state);
        // count += add_surftrack_straightloss(sm, p, {0,0},{1,0},{2,0}, data, problem, state);
    }

    if (flags & SURF_LOSS && state(p) & STATE_LOC_VALID)
        count += cond_surftrack_surfloss(6, sm, p, data, problem, state, points);

    return count;
}

double local_cost_destructive(SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, cv::Vec3f loc, int *ref_count = nullptr)
{
    assert(!data.has(sm, p));
    uint8_t state_old = state(p);
    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
    int count;
    double test_loss = 0.0;
    {
        ceres::Problem problem_test;

        data.loc(sm, p) = {loc[1], loc[0]};

        count = surftrack_add_local(sm, p, data, problem_test, state, points, step);
        if (ref_count)
            *ref_count = count;

        problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);

        // std::cout << "surf test loss" << test_loss << std::endl;

        // if (test_loss < 1e-3)
            // inliers++;
    } //destroy problme before data
    data.erase(sm, p);
    state(p) = state_old;

    //FIXME normalize with number of res blocks!
    if (!count)
        //FIXME why is this useful?!
        return 0;
    else
        return sqrt(test_loss/count);
}


double local_cost(SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, int *ref_count = nullptr)
{
    assert(!data.has(sm, p));
    int count;
    double test_loss = 0.0;
        ceres::Problem problem_test;

    count = surftrack_add_local(sm, p, data, problem_test, state, points, step);
    if (ref_count)
        *ref_count = count;

    problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);

    //FIXME normalize with number of res blocks!
    if (!count)
        //FIXME why is this useful?!
        return 0;
    else
        return sqrt(test_loss/count);
}

double local_solve(SurfaceMeta *sm, const cv::Vec2i p, SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, float step, int flags)
{
    ceres::Problem problem;

    surftrack_add_local(sm, p, data, problem, state, points, step, flags);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    ceres::Solver::Summary summary;
    //FIXME solver can run out of valid area!
    ceres::Solve(options, &problem, &summary);

    // std::cout << summary.BriefReport() << std::endl;
    // std::cout << "numres " << summary.num_residual_blocks << std::endl;

    if (summary.num_residual_blocks < 3)
        return 10000;

    // std::cout << summary.final_cost/summary.num_residual_blocks << std::endl;
    return summary.final_cost/summary.num_residual_blocks;
}


cv::Mat_<cv::Vec3d> surftrack_genpoints_hr(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Rect &used_area, float step, float step_src)
{

    cv::Mat_<cv::Vec3f> points_hr(state.rows*step, state.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(state.rows*step, state.cols*step, 0);
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i) & (STATE_LOC_VALID|STATE_COORD_VALID)
                && state(j+1,i+1) & (STATE_LOC_VALID|STATE_COORD_VALID))
            {
            for(auto &sm : data.surfs({j,i})) {
                if (data.valid_int(sm,{j,i})
                    && data.valid_int(sm,{j,i+1})
                    && data.valid_int(sm,{j+1,i})
                    && data.valid_int(sm,{j+1,i+1}))
                {
                    cv::Vec2f l00 = data.loc(sm,{j,i});
                    cv::Vec2f l01 = data.loc(sm,{j,i+1});
                    cv::Vec2f l10 = data.loc(sm,{j+1,i});
                    cv::Vec2f l11 = data.loc(sm,{j+1,i+1});

                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                            cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                            cv::Vec2f l = (1-fy)*l0 + fy*l1;
                            points_hr(j*step+sy,i*step+sx) += data.lookup_int_loc(sm,l);
                            counts_hr(j*step+sy,i*step+sx) += 1;
                        }
                }
            }
            if (!counts_hr(j*step+1,i*step+1)) {
                int src_loc_valid_count = 0;
                if (state(j,i) & STATE_LOC_VALID)
                    src_loc_valid_count++;
                if (state(j,i+1) & STATE_LOC_VALID)
                    src_loc_valid_count++;
                if (state(j+1,i) & STATE_LOC_VALID)
                    src_loc_valid_count++;
                if (state(j+1,i+1) & STATE_LOC_VALID)
                    src_loc_valid_count++;

                if (src_loc_valid_count >= 2)
                    continue;

                cv::Vec3d c00 = points(j,i);
                cv::Vec3d c01 = points(j,i+1);
                cv::Vec3d c10 = points(j+1,i);
                cv::Vec3d c11 = points(j+1,i+1);

                for(int sy=0;sy<=step;sy++)
                    for(int sx=0;sx<=step;sx++) {
                        if (!counts_hr(j*step+sy,i*step+sx)) {
                            float fx = sx/step;
                            float fy = sy/step;
                            cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                            cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                            cv::Vec3d c = (1-fy)*c0 + fy*c1;
                            points_hr(j*step+sy,i*step+sx) = c;
                            counts_hr(j*step+sy,i*step+sx) = 1;
                        }
                    }
            }
        }
    }
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
            else
                points_hr(j,i) = {-1,-1,-1};

    return points_hr;
}

int static dbg_counter = 0;
float local_cost_inl_th = 0.1;

//try flattening the current surface mapping assuming direct 3d distances
//TODO use accurate distance metrics by building local surfaces?
//TODO just (fixed) downweighting of areas where distances are more off?
//TODO sample adjustment factors just like the actual 3d coords are sampled from a precomputed set?
//this is basically just a reparametrization!
void optimize_surface_mapping(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Rect &used_area, float step, float step_src, const cv::Vec2i &seed, int closing_r, bool keep_inpainted = false)
{
    cv::Mat_<cv::Vec3d> points_new = points.clone();
    SurfaceMeta sm;
    sm._surf = new QuadSurface(points, {1,1});

    SurfTrackerData data_new;
    data_new._data = data._data;

    ceres::Problem problem;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID) {
                data_new.surfs({j,i}).insert(&sm);
                data_new.loc(&sm, {j,i}) = {j,i};
            }

    cv::Mat_<uint8_t> new_state = state.clone();

    //generate closed version of state
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});

    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    int res_count = 0;
    //slowly inpaint physics only points
    for(int r=0;r<closing_r;r++) {
        cv::Mat_<uint8_t> masked;
        bitwise_and(state, STATE_VALID, masked);
        cv::dilate(masked, masked, m, {-1,-1}, r);
        cv::erode(masked, masked, m, {-1,-1}, r);
        cv::imwrite("masked.tif", masked);

        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if ((masked(j,i) & STATE_VALID) && (~new_state(j,i) & STATE_VALID)) {
                    new_state(j, i) = STATE_COORD_VALID;
                    points_new(j,i) = {1,2,3};
                    //TODO add local area solve
                    double err = local_solve(&sm, {j,i}, data_new, new_state, points_new, step*step_src, LOSS_3D_INDIRECT | SURF_LOSS);
                    res_count += surftrack_add_global(&sm, {j,i}, data_new, problem, new_state, points_new, step*step_src, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
                    // if (err > 0.1) {
                    //     new_state(j, i) = 0;
                    //     points_new(j,i) = {-1,-1,-1};
                    // }
                }
    }

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID)
                if (problem.HasParameterBlock(&points_new(j,i)[0]))
                    problem.SetParameterBlockConstant(&points_new(j,i)[0]);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 1000;

    ceres::Solver::Summary summary;
    //FIXME solver can run out of valid area!
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    cv::Mat_<cv::Vec3d> points_inpainted = points_new.clone();
    cv::Mat_<uint8_t> state_inpainted = new_state.clone();


    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (state(j,i) & STATE_LOC_VALID)
                if (problem.HasParameterBlock(&points_new(j,i)[0]))
                    problem.SetParameterBlockVariable(&points_new(j,i)[0]);

    cv::imwrite("state_old"+std::to_string(dbg_counter)+".tif", state);
    cv::imwrite("state_new"+std::to_string(dbg_counter)+".tif", new_state);

    std::cout << "using " << used_area.tl() << used_area.br() << std::endl;

    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            res_count += surftrack_add_global(&sm, {j,i}, data_new, problem, new_state, points_new, step*step_src, LOSS_3D_INDIRECT | SURF_LOSS | OPTIMIZE_ALL);

    std::cout << "optimizing " << res_count << " residuals, seed " << seed << std::endl;

    if (problem.HasParameterBlock(&data_new.loc(&sm, seed)[0]))
        problem.SetParameterBlockConstant(&data_new.loc(&sm, seed)[0]);
    if (problem.HasParameterBlock(&data_new.loc(&sm, seed+cv::Vec2i(0,1))[0]))
        problem.SetParameterBlockConstant(&data_new.loc(&sm, seed+cv::Vec2i(0,1))[0]);
    if (problem.HasParameterBlock(&data_new.loc(&sm, seed+cv::Vec2i(1,0))[0]))
        problem.SetParameterBlockConstant(&data_new.loc(&sm, seed+cv::Vec2i(1,0))[0]);
    if (problem.HasParameterBlock(&data_new.loc(&sm, seed+cv::Vec2i(1,1))[0]))
        problem.SetParameterBlockConstant(&data_new.loc(&sm, seed+cv::Vec2i(1,1))[0]);


    //FIXME solver can run out of valid area!
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << std::endl;

    //interpolate from the original set
    cv::Mat_<cv::Vec3d> points_hr = surftrack_genpoints_hr(data, state_inpainted, points_inpainted, used_area, step, step_src);

    cv::imwrite("points_lr"+std::to_string(dbg_counter)+".tif", points);
    cv::imwrite("points_hr"+std::to_string(dbg_counter)+".tif", points_hr);

    //FIXME we still produce holes
    SurfTrackerData data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (new_state(j,i) & STATE_VALID) {
            // if (data_new.has(&sm, {j,i})) {
                cv::Vec2d l = data_new.loc(&sm ,{j,i});
                int y = l[0];
                int x = l[1];
                l *= step;
                if (loc_valid(points_hr, l)) {
                    points_out(j, i) = interp_lin_2d(points_hr, l);
                    state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;

                    std::set<SurfaceMeta*> surfs;
                    surfs.insert(data.surfs({y,x}).begin(), data.surfs({y,x}).end());
                    surfs.insert(data.surfs({y,x+1}).begin(), data.surfs({y,x+1}).end());
                    surfs.insert(data.surfs({y+1,x}).begin(), data.surfs({y+1,x}).end());
                    surfs.insert(data.surfs({y+1,x+1}).begin(), data.surfs({y+1,x+1}).end());

                    for(auto &s : surfs) {
                        SurfacePointer *ptr = s->surf()->pointer();
                        float res = s->surf()->pointTo(ptr, points_out(j, i), 2.0, 4);
                        if (res <= 2.0) {
                            data_out.surfs({j,i}).insert(s);
                            cv::Vec3f loc = s->surf()->loc_raw(ptr);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                        }
                    }
                }
            }

    //now filter by consistency
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (state_out(j,i) & STATE_VALID) {
                std::set<SurfaceMeta*> surf_src = data_out.surfs({j,i});
                for (auto s : surf_src) {
                    int count;
                    float cost = local_cost(s, {j,i}, data_out, state_out, points_out, step, &count);
                    if (cost >= local_cost_inl_th) {
                        std::cout << cost << " count " << count << std::endl;
                        data_out.erase(s, {j,i});
                        data_out.eraseSurf(s, {j,i});
                    }
                    //disasble if less than 1 surfs?
                }
                // std::cout << "remain " << data_out.surfs({j,i}).size() << " of " << surf_src.size() << std::endl;
                if (data_out.surfs({j,i}).size() < 2) {
                    state_out(j,i) = 0;
                    points_out(j, i) = {-1,-1,-1};
                }
            }

    // // //reinterpolate from existing points
    // // //FIXME this is still no end2end global optimization, so could over time diverge ...
    // // SurfTrackerData data_out;
    // // cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    // // cv::Mat_<int> count_out(points.size(), 0);
    // // cv::Mat_<uint8_t> state_out(state.size(), 0);
    // // for(int j=used_area.y;j<used_area.br().y-1;j++)
    // //     for(int i=used_area.x;i<used_area.br().x-1;i++)
    // //         if (new_state(j,i) & STATE_VALID) {
    // //             points_out(j,i) = {0,0,0};
    // //             cv::Vec2f l = data_new.loc(&sm ,{j,i});
    // //             int y = l[0];
    // //             int x = l[1];
    // //             for(auto &s : data.surfs({y,x})) {
    // //                 if (state(y,x) & STATE_LOC_VALID && data.valid_int(s,{y,x})
    // //                     && state(y,x+1) & STATE_LOC_VALID && data.valid_int(s,{y,x+1})
    // //                     && state(y+1,x) & STATE_LOC_VALID && data.valid_int(s,{y+1,x})
    // //                     && state(y+1,x+1) & STATE_LOC_VALID && data.valid_int(s,{y+1,x+1}))
    // //                 {
    // //                     cv::Vec2f l00 = data.loc(s,{y,x});
    // //                     cv::Vec2f l01 = data.loc(s,{y,x+1});
    // //                     cv::Vec2f l10 = data.loc(s,{y+1,x});
    // //                     cv::Vec2f l11 = data.loc(s,{y+1,x+1});
    // //
    // //                     float fx = l[1]-x;
    // //                     float fy = l[0]-y;
    // //                     cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
    // //                     cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
    // //                     cv::Vec2f l = (1-fy)*l0 + fy*l1;
    // //                     points_out(j,i) += data.lookup_int_loc(s,l);
    // //                     count_out(j,i) += 1;
    // //                     data_out.surfs({j,i}).insert(s);
    // //                     data_out.loc(s, {j,i}) = l;
    // //                     state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;
    // //                 }
    // //         }
    // //         if (count_out(j,i)) {
    // //             points_out(j,i) /= count_out(j,i);
    // //             // std::cout << points(j,i) << " vs " << points_out(j,i) << std::endl;
    // //         }
    // //         else if (keep_inpainted) {
    // //             state_out(j, i) = STATE_COORD_VALID;
    // //             points_out(j,i) = points_new(j, i);
    // //         }
    // //         else
    // //             points_out(j,i) = {-1,-1,-1};
    // //     }

    points = points_out;
    state = state_out;
    data = data_out;

    cv::imwrite("state_out"+std::to_string(dbg_counter)+".tif", state_out);
    dbg_counter++;
}

QuadSurface *grow_surf_from_surfs(SurfaceMeta *seed, const std::vector<SurfaceMeta*> &surfs_v, float step)
{
    std::unordered_map<std::string,SurfaceMeta*> surfs;
    float src_step = 20;

    for(auto &sm : surfs_v)
        surfs[sm->name()] = sm;

    for(auto &sm : surfs_v)
        for(auto name : sm->overlapping_str)
            sm->overlapping.insert(surfs[name]);

    std::cout << "total number of surfs:" << surfs.size() << std::endl;
    std::cout << seed << "name" << seed->name() << " seed overlapping:" << seed->overlapping.size() << "/" << seed->overlapping_str.size() << std::endl;

    cv::Mat_<cv::Vec3f> seed_points = seed->surf()->rawPoints();

//     cv::Mat_<uint8_t> counts(seed_points.size(), 0);
// #pragma omp parallel for
//     for(int j=0;j<counts.rows;j++)
//         for(int i=0;i<counts.rows;i++)
//             for(auto &sm : seed->overlapping)
//                 if (seed_points(j,i)[0] != -1) {
//                     SurfacePointer *ptr = sm->surf()->pointer();
//                     float dist = sm->surf()->pointTo(ptr, seed_points(j,i), 2.0, 4);
//                     if (dist <= 2.0)
//                         counts(j,i)++;
//                     std::cout << dist << " " << j << " " << i << std::endl;
//                 }
//     cv::imwrite("counts.tif", counts);

    //FIXME shouldn change start of opt but does?! (32-good, 64 bad, 50 good?)
    int stop_gen = 32;
    int opt_map_every = 6;
    int closing_r = 20;
    int w = stop_gen*2*1.1+5+2*closing_r;
    int h = stop_gen*2*1.1+5+2*closing_r;
    cv::Size  size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);

    int x0 = w/2;
    int y0 = h/2;
    int r = 1;

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};

    std::unordered_set<cv::Vec2i,vec2i_hash> fringe;
    std::unordered_set<cv::Vec2i,vec2i_hash> cands;

    // std::unordered_map<std::tuple<SurfaceMeta,cv::Vec2f>>

    // cv::Mat_<cv::Vec3d> locs(size,cv::Vec3f(-1,-1,-1));
    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<cv::Vec3d> points(size,{-1,-1,-1});

    cv::Rect used_area(x0,y0,2,2);

    SurfTrackerData data;

    //start in the center
    data.loc(seed,{y0,x0}) = {seed_points.rows/2, seed_points.cols/2 };
    data.loc(seed,{y0,x0+1}) = {seed_points.rows/2, seed_points.cols/2+step };
    data.loc(seed,{y0+1,x0}) = {seed_points.rows/2+step, seed_points.cols/2 };
    data.loc(seed,{y0+1,x0+1}) = {seed_points.rows/2+step, seed_points.cols/2+step };

    // std::cout << "init " << data.loc(seed,{y0,x0}) << data.loc(seed,{y0,x0+1}) << data.loc(seed,{y0+1,x0}) << data.loc(seed,{y0+1,x0+1}) << std::endl;

    data.surfs({y0,x0}).insert(seed);
    data.surfs({y0,x0+1}).insert(seed);
    data.surfs({y0+1,x0}).insert(seed);
    data.surfs({y0+1,x0+1}).insert(seed);

    points({y0,x0}) = data.lookup_int(seed,{y0,x0});
    //FIXME WTF?!!!
    points({y0,x0+1}) = data.lookup_int(seed,{y0+1,x0});
    points({y0+1,x0}) = data.lookup_int(seed,{y0,x0+1});
    points({y0+1,x0+1}) = data.lookup_int(seed,{y0+1,x0+1});

    state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
    state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

    fringe.insert(cv::Vec2i(y0,x0));
    fringe.insert(cv::Vec2i(y0+1,x0));
    fringe.insert(cv::Vec2i(y0,x0+1));
    fringe.insert(cv::Vec2i(y0+1,x0+1));

    //insert initial surfs per location
    for(auto p: fringe) {
        data.surfs(p).insert(seed);
        cv::Vec3f coord = points(p);
        std::cout << "testing " << p << " from cands: " << seed->overlapping.size() << coord << std::endl;
        for(auto s : seed->overlapping) {
            SurfacePointer *ptr = s->surf()->pointer();
            if (s->surf()->pointTo(ptr, coord, 2.0) <= 2.0) {
                cv::Vec3f loc = s->surf()->loc_raw(ptr);
                // std::cout << loc << std::endl;
                data.surfs(p).insert(s);
                data.loc(s, p) = {loc[1], loc[0]};
            }
        }
        std::cout << "fringe point " << p << " surfcount " << data.surfs(p).size() << " init " << data.loc(seed, p) << data.lookup_int(seed, p) << std::endl;
    }

    std::cout << "starting from " << x0 << " " << y0 << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;

    int succ = 0;
    for(int generation=0;generation<stop_gen;generation++) {
        for(auto p : fringe)
        {
            if ((state(p) & STATE_LOC_VALID) == 0)
                continue;

            for(auto n : neighs)
                if (bounds.contains(p+n)
                    && (state(p+n) & STATE_PROCESSING) == 0
                    && (state(p+n) & STATE_LOC_VALID) == 0)
                {
                    state(p+n) |= STATE_PROCESSING;
                    cands.insert(p+n);
                }
        }
        fringe.clear();

        std::cout << "go with cands " << cands.size() << std::endl;

        for(auto &p : cands) {
            if (state(p) & STATE_LOC_VALID)
                continue;

            std::set<SurfaceMeta*> local_surfs = {seed};

            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data.surfs({oy,ox});
                        local_surfs.insert(p_surfs.begin(), p_surfs.end());
                    }

            // std::cout << "adding with local surf count " << local_surfs.size() << p << std::endl;

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            SurfaceMeta *best_surf = nullptr;
            cv::Vec2d best_loc = {-1,-1};
            //FIXME also store best loc or re-optimize it!
            // std::set<SurfaceMeta*> seed_surfs = {seed};
            // for(auto ref_surf : seed_surfs) {
            for(auto ref_surf : local_surfs) {
                int ref_count = 0;
                cv::Vec2d avg = {0,0};
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                        //FIXME need to also check for -1/-1 entries at multiple locations? (maybe whereever we insert loc)
                        if ((state(oy,ox) & STATE_LOC_VALID) && data.valid_int(ref_surf,{oy,ox})) {
                            ref_count++;
                            avg += data.loc(ref_surf,{oy,ox});
                        }

                if (ref_count < 2)
                    continue;

                avg /= ref_count;
                data.loc(ref_surf,p) = avg;

                // std::cout << "avg " << avg << std::endl;
                // std::cout << "is " << data.lookup_int(ref_surf,p) << std::endl;

                ceres::Problem problem;

                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                surftrack_add_local(ref_surf, p, data, problem, state, points, step);

                ceres::Solver::Summary summary;
                //FIXME solver can run out of valid area!
                ceres::Solve(options, &problem, &summary);
                float loss = summary.final_cost;

                cv::Vec2d ref_loc = data.loc(ref_surf,p);

                // std::cout << "res " << data.loc(ref_surf,p) << " loss " << loss << std::endl;

                // std::cout << summary.FullReport() << std::endl;

                if (!data.valid_int(ref_surf,p)) {
                    // std::cout << "went off ref surf valid area " << std::endl;
                    continue;
                }

                //FIXME p can point to partially invalid part of ref_surf!
                cv::Vec3d coord = data.lookup_int(ref_surf,p);
                // std::cout << "output " << coord << p << data.loc(ref_surf,p) << std::endl;
                if (coord[0] == -1)
                    continue;

                data.erase(ref_surf,p);
                state(p) = 0;

                // int inliers = 0;
                int inliers_sum = 0;
                for(auto test_surf : local_surfs) {
                    // if (test_surf == ref_surf)
                        // continue;

                    SurfacePointer *ptr = test_surf->surf()->pointer();
                    //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                    if (test_surf->surf()->pointTo(ptr, coord, 2.0, 4) <= 2.0) {
                        int count = 0;
                        float cost = local_cost_destructive(test_surf, p, data, state, points, step, test_surf->surf()->loc_raw(ptr), &count);
                        if (cost < local_cost_inl_th) {
                            // inliers++;
                            inliers_sum += count;
                        }
                        // std::cout << "cost " << cost << std::endl;

                    }


                    //     {
                    //         ceres::Problem problem_test;
                    //
                    //         cv::Vec3f loc = test_surf->surf()->loc_raw(ptr);
                    //         data.loc(test_surf, p) = {loc[1], loc[0]};
                    //
                    //         state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                    //         surftrack_add_local(test_surf, p, data, problem_test, state, step);
                    //         state(p) = 0;
                    //
                    //         double test_loss = 0.0;
                    //         problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);
                    //
                    //         std::cout << "surf test loss" << test_loss <<  " blocks " << summary.num_residual_blocks << std::endl;
                    //
                    //         ifê (test_loss < 0.1)
                    //             inliers++;
                    //     } //destroy problme before data
                    //     data.erase(test_surf, p);
                    //     // return nullptr;
                    // }

                    // ceres::Problem problem_test;
                    // surftrack_add_local(test_surf, p, data, problem_test, state, step);
                    //
                    // double test_loss = 0;
                    // problem_test.Evaluate(ceres::Problem::EvaluateOptions(), &test_loss, nullptr, nullptr, nullptr);
                    // std::cout << test_surf << " " << test_loss << " res c " << summary.num_residual_blocks << std::endl;
                }
                if (inliers_sum > best_inliers) {
                    best_inliers = inliers_sum;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                }
                // std::cout << "inliers " << inliers << "/" << local_surfs.size() << std::endl;
            }

            cv::imwrite("points"+std::to_string(generation)+".tif", points);

            //FIXME if fail reset data.loc(ref_surf,p)
            // std::cout << "done " << best_inliers <<  best_coord << std::endl;

            //FIXME can check max dist between best_coord and any ref/avg init?


            if (best_inliers >= 10 /*&& best_inliers >= local_surfs.size()/2*/) {
                if (best_coord[0] == -1)
                    throw std::runtime_error("WTF1");
                points(p) = best_coord;
                data.surfs(p).insert(best_surf);
                data.loc(best_surf, p) = best_loc;
                state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                ceres::Problem problem;
                surftrack_add_local(best_surf, p, data, problem, state, points, step, SURF_LOSS);

                std::set<SurfaceMeta*> more_local_surfs;
                for(auto test_surf : local_surfs) {
                    if (test_surf == best_surf)
                        continue;

                    SurfacePointer *ptr = test_surf->surf()->pointer();
                    if (test_surf->surf()->pointTo(ptr, best_coord, 2.0, 4) <= 2.0) {
                        cv::Vec3f loc = test_surf->surf()->loc_raw(ptr);
                        if (local_cost_destructive(test_surf, p, data, state, points, step, loc) < local_cost_inl_th)
                            data.surfs(p).insert(test_surf);
                            data.loc(test_surf, p) = {loc[1], loc[0]};
                            surftrack_add_local(test_surf, p, data, problem, state, points, step, SURF_LOSS);

                            for(auto s : test_surf->overlapping)
                                if (!data.has(s, p))
                                    more_local_surfs.insert(s);
                    }
                }

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                // std::cout << summary.BriefReport() << std::endl;


                //FIXME could also iterate this until we don't add new sufs... might even be cleaner code
                //add agreeing surfs to location
                // for(auto test_surf : local_surfs) {
                //     if (data.has(test_surf, p))
                //         continue;
                //
                //     if (test_surf == seed)
                //         std::cout << "   cmp seed!" << std::endl;
                //
                //     SurfacePointer *ptr = test_surf->surf()->pointer();
                //     float res = test_surf->surf()->pointTo(ptr, best_coord, 2.0, 4);
                //     if (res <= 2.0) {
                //         if (best_coord[0] == -1)
                //             throw std::runtime_error("WTF2");
                //         cv::Vec3f loc = test_surf->surf()->loc_raw(ptr);
                //         std::cout << "add surf " << res << loc << best_coord << std::endl;
                //         cv::Vec3f coord = data.lookup_int_loc(test_surf, {loc[1], loc[0]});
                //         if (coord[0] == -1) {
                //             std::cout << "WTF3" << std::endl;
                //             continue;
                //             // throw std::runtime_error("WTF");
                //         }
                //         if (local_cost_destructive(test_surf, p, data, state, step, loc) < 1e-2) {
                //             // std::cout << res << " " << best_coord << loc << std::endl;
                //             data.loc(test_surf, p) = {loc[1], loc[0]};
                //             data.surfs(p).insert(test_surf);
                //             for(auto s : test_surf->overlapping)
                //                 if (!data.has(s, p))
                //                     more_local_surfs.insert(s);
                //         }
                //     }
                // }
                //check for more agreeing surfs by checking agreeings surfs neigbors
                for(auto test_surf : more_local_surfs) {
                    SurfacePointer *ptr = test_surf->surf()->pointer();
                    float res = test_surf->surf()->pointTo(ptr, best_coord, 2.0, 4);
                    if (res <= 2.0) {
                        cv::Vec3f loc = test_surf->surf()->loc_raw(ptr);
                        // std::cout << "more:" << res << best_coord << loc << test_surf->surf()->rawPoints().size() << std::endl;

                        cv::Vec3f coord = data.lookup_int_loc(test_surf, {loc[1], loc[0]});
                        if (coord[0] == -1) {
                            // std::cout << "WTF3" << std::endl;
                            continue;
                        }
                        if (local_cost_destructive(test_surf, p, data, state, points, step, loc) < local_cost_inl_th) {
                            data.loc(test_surf, p) = {loc[1], loc[0]};
                            data.surfs(p).insert(test_surf);
                        }
                    }
                }
                // std::cout << "p surf count" << p << " " << data.surfs(p).size() << best_coord << std::endl;
                succ++;
                if (!used_area.contains({p[1],p[0]})) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                }
                fringe.insert(p);

                // std::cout << std::endl << "checking against seed " << p << std::endl;
                // if (data.has(seed, p))
                    // std::cout << "seed diff " << points(p) << best_coord << data.lookup_int(seed,p) << cv::norm(best_coord - data.lookup_int(seed,p)) << std::endl << std::endl;;
            }
            else if (best_inliers == -1) {
                //just try again some other time
                state(p) = 0;
                points(p) = {-1,-1,-1};
                //FIXME reset data.loc(ref_surf,p)
                // std::cout << std::endl << "FAIL1 " << best_inliers << std::endl;
            }
            else {
                // state(p) = STATE_FAIL;
                state(p) = 0;
                points(p) = {-1,-1,-1};
                //FIXME reset data.loc(ref_surf,p)
                // std::cout << std::endl << "FAIL2 " << best_inliers << std::endl;
            }
            // std::cout << std::endl;

            // if (succ >= 1)
            // return nullptr;
            // break;

        }

        //lets just see what happens
        if (generation && /*generation+1 != stop_gen &&*/ (generation % opt_map_every) == 0) {
            optimize_surface_mapping(data, state, points, used_area, step, src_step, {y0,x0}, closing_r, true   );

            //recalc fringe after surface optimization (which often shrinks the surf)
            fringe.clear();
            for(int j=used_area.y-1;j<=used_area.br().y;j++)
                for(int i=used_area.x-1;i<=used_area.br().x;i++)
                    if (state(j,i) & STATE_LOC_VALID)
                        fringe.insert({j,i});
        }

        printf("gen %d processing %d fringe cands (total done %d fringe: %d)\n", generation, cands.size(), succ, fringe.size());
        if (!fringe.size())
            break;
    }

    // optimize_surface_mapping(data, state, points, used_area, step, src_step, {y0,x0}, closing_r, true);

    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    // cv::Mat_<float> locx(points.size(),{-1});
    // cv::Mat_<float> locy(points.size(),{-1});
    // for(int j=used_area.y;j<used_area.br().y;j++)
    //     for(int i=used_area.x;i<used_area.br().x;i++)
    //         if (state(j,i) & STATE_LOC_VALID) {
    //             locx(j,i) = data.loc(seed,{j,i})[1];
    //             locy(j,i) = data.loc(seed,{j,i})[0];
    //         }
    // cv::imwrite("locx.tif", locx);
    // cv::imwrite("locy.tif", locy);
    //
    // cv::imwrite("state.tif", state);

    // std::cout << "used area " << std::endl;

    // points = points(used_area);
    // cv::Mat_<cv::Vec3d> points_hr_int;
    // cv::resize(points, points_hr_int, {0,0}, step, step, cv::INTER_CUBIC);
    // QuadSurface *surf = new QuadSurface(points, {1/src_step,1/src_step});

    // QuadSurface *surf = new QuadSurface(points(used_area), {1/src_step/step,1/src_step/step});

    int loc_valid_count = 0;

    // upsampling surface from segments
    cv::Mat_<cv::Vec3f> points_hr(points.rows*step, points.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(points.rows*step, points.cols*step, 0);
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++) {
            if (state(j,i) & STATE_VALID
                && state(j,i+1) & STATE_VALID
                && state(j+1,i) & STATE_VALID
                && state(j+1,i+1) & STATE_VALID)
            {
                loc_valid_count++;
                for(auto &sm : data.surfs({j,i})) {
                    if (state(j,i) & STATE_LOC_VALID && data.valid_int(sm,{j,i})
                        && state(j,i+1) & STATE_LOC_VALID && data.valid_int(sm,{j,i+1})
                        && state(j+1,i) & STATE_LOC_VALID && data.valid_int(sm,{j+1,i})
                        && state(j+1,i+1) & STATE_LOC_VALID && data.valid_int(sm,{j+1,i+1}))
                    {
                        cv::Vec2f l00 = data.loc(sm,{j,i});
                        cv::Vec2f l01 = data.loc(sm,{j,i+1});
                        cv::Vec2f l10 = data.loc(sm,{j+1,i});
                        cv::Vec2f l11 = data.loc(sm,{j+1,i+1});

                            for(int sy=0;sy<=step;sy++)
                                for(int sx=0;sx<=step;sx++) {
                                    float fx = sx/step;
                                    float fy = sy/step;
                                    cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                                    cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                                    cv::Vec2f l = (1-fy)*l0 + fy*l1;
                                    points_hr(j*step+sy,i*step+sx) += data.lookup_int_loc(sm,l);
                                    counts_hr(j*step+sy,i*step+sx) += 1;
                                    // if (sx == 0 && sy == 0)
                                    //     std::cout << "diff " << cv::Vec2i(j,i) << data.lookup_int_loc(sm,l) << points(j,i) << cv::norm(data.lookup_int_loc(sm,l) - points(j,i)) << std::endl;
                                }
                    }

                }
                if (!counts_hr(j*step+1,i*step+1)) {
                    cv::Vec3d c00 = points(j,i);
                    cv::Vec3d c01 = points(j,i+1);
                    cv::Vec3d c10 = points(j+1,i);
                    cv::Vec3d c11 = points(j+1,i+1);

                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            if (!counts_hr(j*step+sy,i*step+sx)) {
                                float fx = sx/step;
                                float fy = sy/step;
                                cv::Vec3d c0 = (1-fx)*c00 + fx*c01;
                                cv::Vec3d c1 = (1-fx)*c10 + fx*c11;
                                cv::Vec3d c = (1-fy)*c0 + fy*c1;
                                points_hr(j*step+sy,i*step+sx) = c;
                                counts_hr(j*step+sy,i*step+sx) = 1;
                            }
                        }
                }
            }
        }
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
            else
                points_hr(j,i) = {-1,-1,-1};
                // points_hr(j,i) = points_hr_int(j, i);

    // points = points(used_area);
    // state = state(used_area);

    std::cout << "area est: " << loc_valid_count*0.008*0.008*src_step*src_step*step*step << "mm^2" << std::endl;

    // std::cout << points.size() << std::endl;

    QuadSurface *surf = new QuadSurface(points_hr, {1/src_step,1/src_step});

    surf->meta = new nlohmann::json;

    return surf;
}
