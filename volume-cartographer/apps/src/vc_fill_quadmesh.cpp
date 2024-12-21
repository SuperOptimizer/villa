#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/factory.hxx"
#include <nlohmann/json.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <omp.h>

namespace fs = std::filesystem;

using json = nlohmann::json;


static float dist_w = 1.0;
static float straight_w = 0.02;
static float surf_w = 0.1;
static float z_loc_loss_w = 0.002;
static float wind_w = 10.0;

static inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
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

float surf_th = 0.5;

IntersectVec getIntersects(const cv::Vec2i &seed, const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &step)
{
    cv::Vec3f o = points(seed[1],seed[0]);
    cv::Vec3f n = grid_normal(points, {seed[0],seed[1],seed[2]});
    if (std::isnan(n[0]))
        return {};
    std::vector<cv::Vec2f> locs = {seed};
    for(int i=0;i<1000;i++)
    {
        cv::Vec2f loc = {rand() % points.cols, seed[1] - 50 + (rand() % 100)};
        cv::Vec3f res;
        float dist = search_min_line(points, loc, res, o, n, step, 0.01);
        
        if (dist > 0.5 || dist < 0)
            continue;
        
        if (!loc_valid_xy(points,loc))
            continue;
        
        // std::cout << dist << res << loc << std::endl;
        
        bool found = false;
        for(auto l : locs) {
            if (cv::norm(loc, l) <= 4) {
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
    
    //just sort by normal position and let median take care of the rest
    // std::sort(dist_locs.begin(), dist_locs.end(), [](auto a, auto b) {return a.first < b.first; });
    // return dist_locs;    
    
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

//l is [y, x]!
bool loc_valid_nan(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    if (std::isnan(l[0]))
        return false;
    
    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {floor(l[0]),floor(l[1])};
    
    if (!bounds.contains(li))
        return false;
    
    if (std::isnan(m(li[0],li[1])))
        return false;
    if (std::isnan(m(li[0]+1,li[1])))
        return false;
    if (std::isnan(m(li[0],li[1]+1)))
        return false;
    if (std::isnan(m(li[0]+1,li[1]+1)))
        return false;
    return true;
}

bool loc_valid_nan_xy(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    return loc_valid_nan(m, {l[1],l[0]});
}

float find_wind_x(cv::Mat_<float> &winding, cv::Vec2f &loc, float tgt_wind)
{
    if (!loc_valid_nan_xy(winding, loc))
        return -1;
    
    float best_diff = abs(at_int(winding,loc)-tgt_wind);
    
    std::vector<cv::Vec2f> neighs = {{1,0},{-1,0}};
    
    float step = 16.0;
    bool updated = true;
    while (updated)
    {
        updated = false;
        for (auto n : neighs) {
            cv::Vec2f cand = loc + step*n;
            if (!loc_valid_nan_xy(winding, cand))
                continue;
            float diff = abs(at_int(winding,cand)-tgt_wind);
            if (diff < best_diff) {
                best_diff = diff;
                loc = cand;
                updated = true;
                break;
            }
        }
        
        if (!updated) {
            if (step <= 0.001)
                break;
            step /= 2;
            updated = true;
        }
    }

    return best_diff;
}

std::string time_str()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);
    
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << ms.count();
    
    return oss.str();
}

int gen_surfloss(const cv::Vec2i p, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float w = 0.1)
{
    if ((state(p) & STATE_LOC_VALID) == 0)
        return 0;
    
    problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), new ceres::HuberLoss(1.0), &points(p)[0], &locs(p)[0]);
    // problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), new ceres::TukeyLoss(2.0), &points(p)[0], &locs(p)[0]);
    // problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), nullptr, &points(p)[0], &locs(p)[0]);

    return 1;
}

//gen straigt loss given point and 3 offsets
int gen_dist_loss_fill(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, float unit, bool optimize_all, ceres::ResidualBlockId *res, float w)
{
    if ((state(p) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    if ((state(p+off) & (STATE_LOC_VALID|STATE_COORD_VALID)) == 0)
        return 0;
    
    // ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), new ceres::HuberLoss(1.0), &dpoints(p)[0], &dpoints(p+off)[0]);
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &dpoints(p)[0], &dpoints(p+off)[0]);
    
    if (res)
        *res = tmp;
    
    if (!optimize_all)
        problem.SetParameterBlockConstant(&dpoints(p+off)[0]);
    
    return 1;
}

static bool loc_valid(int state)
{
    return state & STATE_LOC_VALID;
}

static bool coord_valid(int state)
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

//gen straigt loss given point and 3 offsets
int gen_straight_loss2(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, bool optimize_all, float w)
{
    if (!coord_valid(state(p+o1)))
        return 0;
    if (!coord_valid(state(p+o2)))
        return 0;
    if (!coord_valid(state(p+o3)))
        return 0;
    
    problem.AddResidualBlock(StraightLoss2::Create(w), nullptr, &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);
    
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

int create_centered_losses_left(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float unit, int flags = 0)
{
    if (!coord_valid(state(p)))
        return 0;
    
    //generate losses for point p
    int count = 0;
    
    //horizontal
    count += gen_straight_loss2(problem, p, {0,-2},{0,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {0,-1},{0,0},{0,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {0,0},{0,1},{0,2}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //vertical
    // count += gen_straight_loss2(problem, p, {-2,0},{-1,0},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {-1,0},{0,0},{1,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {0,0},{1,0},{2,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //further and diag!
    count += gen_straight_loss2(problem, p, {-2,-2},{-1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {2,-2},{1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    
    count += gen_straight_loss2(problem, p, {-4,-2},{-2,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
    count += gen_straight_loss2(problem, p, {4,-2},{2,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
//     
    count += gen_straight_loss2(problem, p, {0,-4},{0,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {2,-4},{1,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {-2,-4},{-1,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {4,-4},{2,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {-4,-4},{-2,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    
//     count += gen_straight_loss2(problem, p, {0,-6},{0,-3},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
//     count += gen_straight_loss2(problem, p, {2,-6},{1,-3},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
//     count += gen_straight_loss2(problem, p, {-2,-6},{-1,-3},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
//     
//     count += gen_straight_loss2(problem, p, {0,-8},{0,-4},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
//     count += gen_straight_loss2(problem, p, {4,-8},{2,-4},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
//     count += gen_straight_loss2(problem, p, {-4,-8},{-2,-4},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    //diagonal neighbors
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.7*dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.7*dist_w);
    
    
    count += gen_dist_loss_fill(problem, p, {2,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-2,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    
    //far left
    count += gen_dist_loss_fill(problem, p, {0,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {1,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w);

    return count;
}

int create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float unit, int flags = 0)
{
    if (!coord_valid(state(p)))
        return 0;

    //generate losses for point p
    int count = 0;
    
    //horizontal
    // count += gen_straight_loss2(problem, p, {0,-2},{0,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {0,-1},{0,0},{0,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {0,0},{0,1},{0,2}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //vertical
    // count += gen_straight_loss2(problem, p, {-2,0},{-1,0},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {-1,0},{0,0},{1,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {0,0},{1,0},{2,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //diag
    count += gen_straight_loss2(problem, p, {-1,-1},{0,0},{1,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {-1,1},{0,0},{1,-1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {1,-1},{0,0},{-1,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {1,1},{0,0},{-1,-1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {0,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {1,-0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    //diag neighboars
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w);
    
    return count;
}

float find_loc_wind(cv::Vec2f &loc, float tgt_wind, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, const cv::Vec3f &tgt, float th)
{
    float best_res = -1;
    for(int r=0;r<100;r++) {
        cv::Vec2f cand = loc;
        
        if (r)
            cand = {rand() % points.cols, rand() % points.rows};
        
        if (abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        cv::Vec3f out_;
        float res = min_loc(points, cand, out_, {tgt}, {0}, nullptr, 4.0, 0.01);
        
        if (res < 0)
            continue;
        
        if (abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        if (res < th) {
            loc = cand;
            return res;
        }
        
        if (best_res == -1 || res < best_res) {
            loc = cand;
            best_res = res;
        }
    }
    
    return sqrt(best_res);
}

template<typename T, typename E>
void interp_lin_2d(const cv::Mat_<E> &m, T y, T x, T *v) {
    if (y < 0)
        y = T(0);
    if (y > m.rows-2)
        y = T(m.rows-2);
    
    if (x < 0)
        x = T(0);
    if (x > m.cols-2)
        x = T(m.cols-2);
    
    int yi = val(y);
    int xi = val(x);
    
    T fx = x - T(xi);
    T fy = y - T(yi);
    
    E c00 = m(yi,xi);
    E c01 = m(yi,xi+1);
    E c10 = m(yi+1,xi);
    E c11 = m(yi+1,xi+1);
    
    T c0 = (T(1)-fx)*T(c00) + fx*T(c01);
    T c1 = (T(1)-fx)*T(c10) + fx*T(c11);
    v[0] = (T(1)-fy)*c0 + fy*c1;
}

static bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l)
{
    if (l[0] == -1)
        return false;
    
    cv::Rect bounds = {0, 0, m.rows-2,m.cols-2};
    cv::Vec2i li = {floor(l[0]),floor(l[1])};
    
    if (!bounds.contains(li))
        return false;
    
    if (std::isnan(m(li[0],li[1])))
        return false;
    if (std::isnan(m(li[0]+1,li[1])))
        return false;
    if (std::isnan(m(li[0],li[1]+1)))
        return false;
    if (std::isnan(m(li[0]+1,li[1]+1)))
        return false;
    return true;
}

template<typename E>
//cost functions for physical paper
struct Interp2DLoss {
    //NOTE we expect loc to be [y, x]
    Interp2DLoss(const cv::Mat_<E> &m, E tgt, float w) : _m(m), _w(w), _tgt(tgt) {};
    template <typename T>
    bool operator()(const T* const l, T* residual) const {
        T v[3];
        
        if (!loc_valid(_m, {val(l[0]), val(l[1])})) {
            residual[0] = T(0);
            return true;
        }
        
        interp_lin_2d(_m, l[0], l[1], v);
        
        residual[0] = T(_w)*(v[0] - T(_tgt));
        
        return true;
    }
    
    const cv::Mat_<E> &_m;
    float _w;
    E _tgt;
    
    static ceres::CostFunction* Create(const cv::Mat_<E> &m, E tgt, float w = 1.0)
    {
        return new ceres::AutoDiffCostFunction<Interp2DLoss, 1, 2>(new Interp2DLoss(m, tgt, w));
    }
    
};

float wind_th = 0.1;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " <tiffxyz> <winding>" << std::endl;
        return EXIT_SUCCESS;
    }
    
    fs::path seg_path = argv[1];
    fs::path wind_path = argv[2];
    
    QuadSurface *surf = nullptr;
    try {
        surf = load_quad_from_tifxyz(seg_path);
    }
    catch (...) {
        std::cout << "error when loading: " << seg_path << std::endl;
        return EXIT_FAILURE;
    }
    
    cv::Mat_<float> winding_in = cv::imread(wind_path, cv::IMREAD_UNCHANGED);
    cv::Mat_<cv::Vec3f> points_in = surf->rawPoints();
    
    // cv::Rect bbox_src(10,10,points_in.cols-20,points_in.rows-20);
    // cv::Rect bbox_src(10,60,points_in.cols-20,240);
    // cv::Rect bbox_src(80,110,1000,80);
    cv::Rect bbox_src(64,50,1000,160);
    // cv::Rect bbox_src(10,10,1000,points_in.rows-20);
    
    float src_step = 20;
    int trace_mul = 1;
    float step = src_step*trace_mul;
    
    cv::Size size = {points_in.cols/trace_mul, points_in.rows/trace_mul};
    cv::Rect bbox = {bbox_src.x/trace_mul, bbox_src.y/trace_mul, bbox_src.width/trace_mul, bbox_src.height/trace_mul};

    cv::Mat_<uint8_t> state(size, 0);
    cv::Mat_<uint8_t> init_state(size, 0);
    cv::Mat_<cv::Vec3d> points(size, {-1,-1,-1});
    cv::Mat_<cv::Vec2d> locs(size, {-1,-1});
    cv::Mat_<float> winding(size, NAN);
    
    cv::Mat_<uint8_t> fail_code(size, 0);
    
    // cv::Rect bbox(294,34,1000,364);
    // cv::Rect bbox(294,34+250,200,114);
    // cv::Rect bbox(333,20,1000,410);
    
    //large
    // cv::Rect bbox(333,10,1000,700);
    
    
    int opt_w = 4;
    
    cv::Rect first_col = {bbox.x,bbox.y,opt_w,bbox.height};
    // points_in(first_col).copyTo(points(first_col));
    // winding_in(first_col).copyTo(winding(first_col));
    
    // std::cout << winding_in(bbox.y, bbox.x) << std::endl;
    
    std::cout << first_col.tl() << first_col.br() << std::endl;
    
    for(int j=first_col.y;j<first_col.br().y;j++)
        for(int i=first_col.x;i<first_col.br().x;i++) {
            points(j,i) = points_in(j*trace_mul, i*trace_mul);
            winding(j,i) = winding_in(j*trace_mul, i*trace_mul);
            // if (points_in(j,i)[0] == -1)
                // continue;
            locs(j,i) = {j*trace_mul,i*trace_mul};
            state(j,i) = STATE_LOC_VALID | STATE_COORD_VALID;
        }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    
    
    ceres::Solver::Options options_col;
    // options.linear_solver_type = ceres::DENSE_QR;
    options_col.linear_solver_type = ceres::SPARSE_SCHUR;
    options_col.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options_col.minimizer_progress_to_stdout = false;
    options_col.max_num_iterations = 10000;
    
    cv::Vec3f seed_coord = {-1,-1,-1};
    cv::Vec2f seed_loc = {-1,-1};
    
    std::vector<float> avg_wind(size.width);
    std::vector<int> wind_counts(size.width);
    std::vector<float> tgt_wind(size.width);
    
    {
        ceres::Problem problem_init;
        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++)
                if (points_in(j*trace_mul,i*trace_mul)[0] == -1) {
                    points(j, i) = {rand()%1000,rand()%1000,rand()%1000};
                    create_centered_losses(problem_init, {j,i}, state, points_in, points, locs, step, 0);
                }
                else {
                    avg_wind[i] += winding_in(j*trace_mul,i*trace_mul);
                    wind_counts[i]++;

                    if (seed_loc[0] == -1) {
                        seed_loc = {j,i};
                        seed_coord = points_in(j*trace_mul,i*trace_mul);
                    }
                }

        for(int i=first_col.x;i<first_col.br().x;i++) {
            std::cout << "init wind col " << i << " " << avg_wind[i] / wind_counts[i] << " " << wind_counts[i] << std::endl;
            avg_wind[i] /= wind_counts[i];
            wind_counts[i] = 1;
        }
        
        for(int i=bbox.x;i<bbox.x+opt_w;i++)
            tgt_wind[i] = avg_wind[i];                

        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++)
                if (points_in(j*trace_mul,i*trace_mul)[0] == -1) {
                    points(j, i) = {rand()%1000,rand()%1000,rand()%1000};
                    cv::Vec2i p = {j,i};
                    if (problem_init.HasParameterBlock(&locs(p)[0]))
                        problem_init.SetParameterBlockVariable(&locs(p)[0]);
                    if (problem_init.HasParameterBlock(&points(p)[0]))
                        problem_init.SetParameterBlockVariable(&points(p)[0]);
                }
                    
        
        ceres::Solver::Summary summary;
        ceres::Solve(options_col, &problem_init, &summary);
        std::cout << summary.BriefReport() << std::endl;
    }
    
    std::vector<cv::Vec2i> neighs = {{0,-1},{-1,-1},{1,-1},{-2,-1},{2,-1},{2,-2},{1,-2},{0,-2},{-1,-2},{-2,-2},{-3,-2},{3,-2},{-4,-2},{4,-2}};
    // std::vector<cv::Vec2i> neighs = {{0,-1},{-1,-1},{1,-1},{2,-2},{1,-2},{0,-2},{-1,-2},{-2,-2}};
    
    
    cv::Mat_<float> surf_dist(points.size(), 0);

    for(int i=bbox.x+opt_w;i<bbox.br().x;i++) {
        tgt_wind[i] = 2*avg_wind[i-1]-avg_wind[i-2];
        
        std::cout << "wind tgt: " << tgt_wind[i] << " " << avg_wind[i-1] << " " << avg_wind[i-2] << std::endl;
        
        std::cout << "proc row " << i << std::endl;
        ceres::Problem problem_col;
#pragma omp parallel for
        for(int j=bbox.y;j<bbox.br().y;j++) {
            cv::Vec2i p = {j,i};
            
            locs(p) = {-1,-1}; //locs(j,i-1)+cv::Vec2d(0,1/step);
            points(p) = {-1,-1, -1}; //points(j,i-1)+cv::Vec3d(0.1,0.1,0.1);
            
            state(p) = 0;
            
            //FIXME
            for(auto n :neighs) {
                cv::Vec2d cand = locs(p+n)+cv::Vec2d(0,1/step);
                if (loc_valid(points_in,cand)) {
                    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                    locs(p) = cand;
                    points(p) = at_int(points_in, {cand[1],cand[0]});
                    break;
                }
                if (coord_valid(state(p+n))) {
                    points(p) = points(p+n);
                    state(p) = STATE_COORD_VALID;
                }
            }
            
            init_state(p) = state(p);
            
            ceres::Solver::Summary summary;
            {
                ceres::Problem problem;
                create_centered_losses_left(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
                problem.AddResidualBlock(Interp2DLoss<float>::Create(winding, tgt_wind[i], wind_w), nullptr, &locs(p)[0]);
                
                // create_centered_losses(problem, p, state, points_in, points, locs, step, 0);
                
                ceres::Solve(options, &problem, &summary);
                // std::cout << summary.BriefReport() << "\n";
                // std::cout << sqrt(summary.final_cost/summary.num_residuals) << std::endl;
                
                // if (loc_valid(points, locs(p)))
                //     std::cout << sqrt(summary.final_cost/summary.num_residuals) << " "<< cv::norm(cv::Vec3f(points(p))-at_int(points_in, {locs(p)[1],locs(p)[0]})) << std::endl;
                // else
                //     std::cout << sqrt(summary.final_cost/summary.num_residuals) << std::endl;
            }
            
            if (loc_valid(points_in, locs(p)))
                surf_dist(p) = cv::norm(cv::Vec3f(points(p))-at_int(points_in, {locs(p)[1],locs(p)[0]}));
            else
                surf_dist(p) = -1;
            
//             if (surf_dist(p) > 10 || surf_dist(p) < 0) {
//                 ceres::Problem problem;
//                 create_centered_losses(problem, p, state, points_in, points, locs, step, 0);
//                 // create_centered_losses(problem, p, state, points_in, points, locs, step, 0);
//                 
//                 ceres::Solve(options, &problem, &summary);
//                 
//                 create_centered_losses(*problem1, p, state, points_in, points, locs, step, 0);
//                 create_centered_losses(*problem2, p, state, points_in, points, locs, step, 0);
//             }
//             else {
                // create_centered_losses(problem_col, p, state, points_in, points, locs, step, LOSS_ON_SURF);
                // create_centered_losses(problem_col, p+cv::Vec2i(0,-1), state, points_in, points, locs, step, LOSS_ON_SURF);
                // create_centered_losses_left(problem_col, p, state, points_in, points, locs, step, LOSS_ON_SURF);
            // }
            
            continue;
            
//             /*cv::Vec2f loc = {locs(p)[1], locs(p)[0]};
//              
//             //FIXME try multipl locs and keep the best ...
//             float res = find_loc_wind(loc, winding(p), points_in, winding_in, points(p), 2.0);
//             loc = {loc[1], loc[0]};
//             
//             // std::cout << res << std::endl;
//             
//             if (res < 0) {
//                 fail_code(p) = 1;
//                 continue;
//             }
//             
//             if (res >= 10) {
//                 fail_code(p) = 2;
//                 continue;
//             }
//             
//             if (abs(winding_in(loc[0],loc[1]) - winding(p)) >= 0.3) {
//                 fail_code(p) = 3;
//                 continue;
//             }
//             
//             locs(p) = loc;
//             
//             // std::cout << res << std::endl;
//             
//             {
//                 ceres::Problem problem;
//                 // create_centered_losses(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
//                 create_centered_losses(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
//                 
//                 ceres::Solver::Summary summary;
//                 ceres::Solve(options, &problem, &summary);
//                 // std::cout << summary.BriefReport() << "\n";
//                 // std::cout << sqrt(summary.final_cost/summary.num_residuals) << std::endl;
//             }
//             
//             if (loc_valid(points,locs(p)))
//                 winding(p) = winding_in(locs(p)[0],locs(p)[1]);*/
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            cv::Vec2i p = {j,i};
            
            for(int o=0;o<=opt_w;o++) {
                cv::Vec2i po = {j,i-o};
                // state(p) = 0;
                if (!loc_valid(points_in,locs(po))) {
                    state(p) &= ~STATE_LOC_VALID;
                    // points(po) = at_int(points_in,{locs(po)[1],locs(po)[0]});
                    // state(p) = STATE_COORD_VALID | STATE_LOC_VALID;
                }
            }
        }
        
        cv::Mat_<uint8_t> state_inpaint = state.clone();
        cv::Mat_<uint8_t> mask;
        bitwise_and(state, (uint8_t)STATE_LOC_VALID, mask);
        cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
        cv::dilate(mask, mask, m, {-1,-1}, 20);
        
        //also fill the mask in y dir
        for(int x=first_col.x;x<i;x++) {
            int col_first = first_col.height;
            int col_last = -1;
            for(int j=0;j<state_inpaint.rows;j++) {
                if (mask(j,x)) {
                    col_first = std::min(col_first, j);
                    col_last = std::max(col_first, j);
                }
            }
            for(int j=col_first;j<col_first;j++)
                mask(j,x) = 1;
        }
        
        for(int j=0;j<state_inpaint.rows;j++)
            for(int x=first_col.x;x<i;x++) {
                state_inpaint(j,x) = 0;
                if (loc_valid(state(j,x))) {
                    if (points(j,x)[0] == -1)
                        throw std::runtime_error("need points 3!");
                    state_inpaint(j,x) = STATE_COORD_VALID | STATE_LOC_VALID;
                }
                else if (mask(j,x)) {
                    // std::cout << "inpaint only! " << cv::Vec2i(x,j) << std::endl;
                    if (points(j,x)[0] != -1)
                        state_inpaint(j,x) = STATE_COORD_VALID;
                    // else {
                        //FIXME still not sure shy this happens?
                        // std::cout << "no valid coord! " << cv::Vec2i(x,j) << std::endl;
                        // state_inpaint(j,x) = STATE_COORD_VALID;
                        // points(j, x) = {rand() % 100 +1,rand() % 100 +1,rand() % 100 +1 };
                    // }
                }
            }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            cv::Vec2i p = {j,i};
            
            for(int o=0;o<=opt_w;o++)
                create_centered_losses(problem_col, p+cv::Vec2i(0,-o), state_inpaint, points_in, points, locs, step, LOSS_ON_SURF);
            // create_centered_losses_left(problem_col, p, state_inpaint, points_in, points, locs, step, LOSS_ON_SURF);
            
            // for(int o=0;o<=opt_w;o++)
                // create_centered_losses(problem_col, p+cv::Vec2i(0,-o), state, points_in, points, locs, step, 0);
            // create_centered_losses_left(problem_col, p, state, points_in, points, locs, step, 0);
            
            if (state_inpaint(p) & STATE_LOC_VALID)
                problem_col.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(points_in, seed_coord[2] - (p[0]-seed_loc[0])*step, z_loc_loss_w), nullptr, &locs(p)[0]);
            
            for(int o=0;o<opt_w;o++)
                if (state_inpaint(j,i-o) & STATE_LOC_VALID)
                    problem_col.AddResidualBlock(Interp2DLoss<float>::Create(winding, tgt_wind[i-o], wind_w), nullptr, &locs(p+cv::Vec2i(0,-o))[0]);
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-20);o<i-opt_w;o++)
                if (!loc_valid(state(j,o)) && coord_valid(state(j, o)))
                create_centered_losses(problem_col, state_inpaint(j, o), state_inpaint, points_in, points, locs, step, 0);
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-20);o<i-opt_w;o++)
                if (!loc_valid(state(j,o)) && coord_valid(state(j, o)))
                    if (problem_col.HasParameterBlock(&points(j,o)[0]))
                        problem_col.SetParameterBlockVariable(&points(j,o)[0]);
        }
        
        for(int x=i-opt_w+1;x<=i;x++)
            for(int j=bbox.y;j<bbox.br().y;j++) {
                cv::Vec2i p = {j,x};
                
                if (loc_valid(state_inpaint(j,x))) {
                    if (problem_col.HasParameterBlock(&locs(p)[0]))
                        problem_col.SetParameterBlockConstant(&locs(p)[0]);
                    if (problem_col.HasParameterBlock(&points(p)[0]))
                        problem_col.SetParameterBlockConstant(&points(p)[0]);
                }
                else {
                    if (problem_col.HasParameterBlock(&locs(p)[0]))
                        problem_col.SetParameterBlockVariable(&locs(p)[0]);
                    if (problem_col.HasParameterBlock(&points(p)[0]))
                        problem_col.SetParameterBlockVariable(&points(p)[0]);
                }
            }
            
        ceres::Solver::Summary summary;
        ceres::Solve(options_col, &problem_col, &summary);
        
        // std::cout << summary.FullReport() << std::endl;
        std::cout << summary.BriefReport() << std::endl;
        
        for(int x=i-opt_w+1;x<=i;x++)
            for(int j=bbox.y;j<bbox.br().y;j++) {
                cv::Vec2i p = {j,x};

                if (problem_col.HasParameterBlock(&locs(p)[0]))
                    problem_col.SetParameterBlockVariable(&locs(p)[0]);
                if (problem_col.HasParameterBlock(&points(p)[0]))
                    problem_col.SetParameterBlockVariable(&points(p)[0]);
            }
            
            
        ceres::Solve(options_col, &problem_col, &summary);
        
        // std::cout << summary.FullReport() << std::endl;
        std::cout << summary.BriefReport() << std::endl;
        
        //TODO do something with the winding if its off (e.g. rerun and ignor?)
        
        if (i % 10 == 0) {
            cv::imwrite("state_inpaint_pref.tif",state_inpaint(bbox)*20);
        }
        
        avg_wind[i] = 0;
        wind_counts[i] = 0;
        for(int x=std::max(i-opt_w,bbox.x+opt_w);x<=i;x++)
            for(int j=bbox.y;j<bbox.br().y;j++) {
                if (!loc_valid(points_in,locs(j,x))) {
                    locs(j,x) = {-1,-1};
                    winding(j, x) = NAN;
                    state_inpaint(j,x) &= ~STATE_LOC_VALID;
                }
                else {
                    winding(j, x) = at_int(winding_in, {locs(j,x)[1],locs(j,x)[0]});
                    
                    if (x == i) {
                        if (abs(winding(j, x)-tgt_wind[i]) <= wind_th) {
                            avg_wind[i] += winding(j, x);
                            wind_counts[i] ++;
                        }
                    }
                }

                if (!coord_valid(state_inpaint(j,x))) {
                    points(j,x) = {-1,-1,-1};
                    state(j,x) = 0;
                }
                else {
                    state(j,x) = state_inpaint(j, x);
                    if (points(j,x)[0] == -1)
                        throw std::runtime_error("need points 2!");
                }
            }
            
        if (i % 10 == 0) {
            std::vector<cv::Mat> chs;
            cv::split(points, chs);
            cv::imwrite("newx.tif",chs[0](bbox));
            cv::imwrite("surf_dist.tif",surf_dist(bbox));
            cv::imwrite("winding_out.tif",winding(bbox)+3);
            cv::imwrite("state.tif",state(bbox)*20);
            cv::imwrite("state_inpaint.tif",state_inpaint(bbox)*20);
            cv::imwrite("init_state.tif",init_state(bbox)*20);
        }
            
        if (!wind_counts[i]) {
            std::cout << "stopping as zero valid locations found!";
            break;
        }

        avg_wind[i] /= wind_counts[i];
        
        std::cout << "avg wind number for col " << i << " : " << avg_wind[i] << " ( tgt was " << tgt_wind[i] << " ) using #" << wind_counts[i] << std::endl;
        wind_counts[i] = 1;
    }
    
    {
    std::vector<cv::Mat> chs;
    cv::split(points, chs);
    cv::imwrite("newx.tif",chs[0]);
    }
    
    cv::imwrite("surf_dist.tif",surf_dist);
    cv::imwrite("fail.tif",fail_code);
    cv::imwrite("winding_out.tif",winding+3);
    
    {
        std::vector<cv::Mat> chs;
        cv::split(locs, chs);
        cv::imwrite("locx.tif",chs[0]);
        cv::imwrite("locy.tif",chs[1]);
    }
    
    QuadSurface *surf_full = new QuadSurface(points(bbox), surf->_scale/trace_mul);
    
    fs::path tgt_dir = "/home/hendrik/data/ml_datasets/vesuvius/manual_wget/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/";
    std::string name_prefix = "testing_fill_";
    std::string uuid = name_prefix + time_str();
    fs::path seg_dir = tgt_dir / uuid;
    std::cout << "saving " << seg_dir << std::endl;
    surf_full->save(seg_dir, uuid);
    
    return EXIT_SUCCESS;
}
