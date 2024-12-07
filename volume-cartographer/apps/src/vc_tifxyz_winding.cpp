#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

namespace fs = std::filesystem;

using json = nlohmann::json;

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

template <typename E>
static inline E at_int(const cv::Mat_<E> &points, cv::Vec2f p)
{
    int x = p[0];
    int y = p[1];
    float fx = p[0]-x;
    float fy = p[1]-y;
    
    E p00 = points(y,x);
    E p01 = points(y,x+1);
    E p10 = points(y+1,x);
    E p11 = points(y+1,x+1);
    
    E p0 = (1-fx)*p00 + fx*p01;
    E p1 = (1-fx)*p10 + fx*p11;
    
    return (1-fy)*p0 + fy*p1;
}

static float dot_s(const cv::Vec3f &p)
{
    return p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
}

template <typename E>
float ldist(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return cv::norm((p-tgt_o).cross(p-tgt_o-tgt_v))/cv::norm(tgt_v);
}

template <typename E>
static float search_min_line(const cv::Mat_<E> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt_o, cv::Vec3f tgt_v, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains({loc[0],loc[1]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    E val = at_int(points, loc);
    out = val;
    float best = ldist(val, tgt_o, tgt_v);
    float res;
    
    //TODO check maybe add more search patterns, compare motion estimatino for video compression, x264/x265, ...
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    cv::Vec2f step = init_step;
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);
            
            //just skip if out of bounds
            if (!boundary.contains({cand[0],cand[1]}))
                continue;
                
                val = at_int(points, cand);
                res = ldist(val, tgt_o, tgt_v);
                if (res < best) {
                    changed = true;
                    best = res;
                    loc = cand;
                    out = val;
                }
        }
        
        if (changed)
            continue;
        
        step *= 0.5;
        changed = true;
        
        if (step[0] < min_step_x)
            break;
    }
    
    return best;
}

template <typename E>
float line_off(const E &p, const cv::Vec3f &tgt_o, const cv::Vec3f &tgt_v)
{
    return (tgt_o-p).dot(tgt_v)/dot_s(tgt_v);
}

using IntersectVec = std::vector<std::pair<float,cv::Vec2f>>;

IntersectVec getIntersects(const cv::Vec2i &seed, const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &step)
{
    cv::Vec3f o = points(seed[1],seed[0]);
    cv::Vec3f n = grid_normal(points, {seed[0],seed[1],seed[2]});
    std::vector<cv::Vec2f> locs = {seed};
    for(int i=0;i<1000;i++)
    {
        cv::Vec2f loc = {rand() % points.cols, seed[1] - 50 + (rand() % 100)};
        cv::Vec3f res;
        float dist = search_min_line(points, loc, res, o, n, step, 0.01);
        
        if (dist > 2 || dist < 0)
            continue;
        
        if (!loc_valid_xy(points,loc))
            continue;
        
        // std::cout << dist << res << loc << std::endl;
        
        bool found = false;
        for(auto l : locs) {
            if (cv::norm(loc, l) <= 2) {
                found = true;
                break;
            }
        }
        if (!found)
            locs.push_back(loc);
    }

    IntersectVec dist_locs;
    for(auto l : locs)
        dist_locs.push_back({line_off(at_int(points,l),o,n), l});
    // std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.first > b.first; });
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.second[0] > b.second[0]; });
    
    //we could have two groups (other part of the scroll), in that case the x locations should be between the ones of the first group!

    // for(auto p : dist_locs)
        // std::cout << p.first << p.second << std::endl;
    
    bool two_halves = false;
    for(int i=1;i<dist_locs.size()-1;i++) 
        if (abs(dist_locs[i-1].first - dist_locs[i+1].first) < std::min(abs(dist_locs[i-1].first - dist_locs[i].first),abs(dist_locs[i].first - dist_locs[i+1].first))) {
            two_halves = true;
        }
        
    // std::cout << "two " << two_halves << std::endl;
    
    if (!two_halves) {
        std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.second[0] < b.second[0]; });
        // std::cout << std::endl;
        return dist_locs;
    }

    IntersectVec a, b;
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.first < b.first; });
    a.push_back(dist_locs[0]);
    b.push_back(dist_locs.back());
    dist_locs.erase(dist_locs.begin());
    dist_locs.erase(dist_locs.begin()+dist_locs.size()-1);
    
    bool seed_in_a = (a.back().first == 0.01);
    
    for(auto pair : dist_locs) {
        // std::cout << pair.first << std::endl;
        if (abs(pair.first - a.back().first) < abs(pair.first - b.back().first)) {
            a.push_back(pair);
            if (pair.first == 0.01)
                seed_in_a = true;
        }
        else
            b.push_back(pair);
    }
    
    // std::cout << "seed " << seed_in_a << std::endl;
//     
    if (seed_in_a) {
        dist_locs = a;
    }
    else
        dist_locs = b;
    
    std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.second[0] < b.second[0]; });
    // std::cout << "out" << std::endl;
    // for(auto p : dist_locs)
        // std::cout << p.first << p.second << std::endl;
        
    // std::cout << std::endl;
    return dist_locs;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <tiffxyz> <out>" << std::endl;
        return EXIT_SUCCESS;
    }
    
    fs::path seg_path = argv[1];
    fs::path obj_path = argv[2];
    
    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cout << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f> points = surf->rawPoints();
    
    cv::Mat_<cv::Vec3b> img(points.size(), 0);
    
    // cv::Vec2i seed = {1145, 168};
    
    int num_conn = 500;
    
    std::vector<IntersectVec> intersects(num_conn);
    
#pragma omp parallel
    {
        unsigned int sr = omp_get_thread_num();
#pragma omp for
        for(int i=0;i<num_conn;i++) {
            
            
            cv::Vec2i seed = {rand_r(&sr) % points.cols, rand_r(&sr) % points.rows};
            while (points(seed[1],seed[0])[0] == -1)
                seed = {rand_r(&sr) % points.cols, rand_r(&sr) % points.rows};

            intersects[i] = getIntersects(seed, points, surf->_scale);
        }
    }
    
    // for(auto pair : intersects) {
    //     std::cout << pair.first << pair.second << std::endl;
    // }
    
    for(auto &iv : intersects) {
        cv::Vec3b col = {50+rand() % 155,50+rand() % 155,50+rand() % 155};
        for(auto &pair : iv) {
            // img(pair.second[1],pair.second[0]) = col;
            std::cout << pair.first << pair.second << std::endl;
            cv::circle(img, cv::Point(pair.second), 3, col, -1);
        }
        std::cout << std::endl;
    }
    cv::imwrite("dbg.tif", img);
    
    std::vector<std::vector<int>> wind_dists_x(points.cols/100+1);
    
    std::cout << "x size " << wind_dists_x.size() << std::endl;
    
    for(auto &iv : intersects) {
        if (iv.size() > 4)
            for(int n=0;n<iv.size()-1;n++) {
                int x1 = iv[n].second[0];
                int x2 = iv[n+1].second[0];
                float dist = x2-x1;
                std::cout << dist << std::endl;
                wind_dists_x[x1/100].push_back(dist);
                wind_dists_x[x2/100].push_back(dist);
            }
    }
    
    for(auto &dists : wind_dists_x)
        std::sort(dists.begin(), dists.end());
    
    std::cout << "final out " << wind_dists_x.size() << std::endl;
    for(auto &dists : wind_dists_x)
        if (dists.size())
            std::cout << dists[dists.size()/2] << std::endl;
    
    return EXIT_SUCCESS;
}
