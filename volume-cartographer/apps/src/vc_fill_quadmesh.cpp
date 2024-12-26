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

static int trace_mul;
static float dist_w;
static float straight_w;
static float surf_w;
static float z_loc_loss_w;
static float wind_w;
static float wind_th;
static int inpaint_back_range;
// static int far_dist = 2;

static int layer_reg_range = 15;
static float layer_reg_range_vx = 500.0;


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

int gen_surfloss(const cv::Vec2i p, ceres::Problem &problem, const cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float w = 0.1, int flags = 0)
{
    if ((state(p) & STATE_LOC_VALID) == 0)
        return 0;
    
    // problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), new ceres::HuberLoss(1.0), &points(p)[0], &locs(p)[0]);
    // problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), new ceres::TukeyLoss(2.0), &points(p)[0], &locs(p)[0]);
    problem.AddResidualBlock(SurfaceLossD::Create(points_in, w), nullptr, &points(p)[0], &locs(p)[0]);

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
    // problem.AddResidualBlock(StraightLoss2::Create(w), new ceres::CauchyLoss(1.0), &dpoints(p+o1)[0], &dpoints(p+o2)[0], &dpoints(p+o3)[0]);
    
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

int create_centered_losses_left_large(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float unit, int flags = 0)
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
    
    count += gen_straight_loss2(problem, p, {-6,-2},{-2,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
    count += gen_straight_loss2(problem, p, {6,-2},{3,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.5*straight_w);
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
    count += gen_dist_loss_fill(problem, p, {4,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    count += gen_dist_loss_fill(problem, p, {-4,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.5*dist_w);
    
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

int create_centered_losses_left(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float unit, int flags = 0)
{
    if (!coord_valid(state(p)))
        return 0;
    
    //generate losses for point p
    int count = 0;
    
    //horizontal
    count += gen_straight_loss2(problem, p, {0,-2},{0,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //further and diag!
    count += gen_straight_loss2(problem, p, {-2,-2},{-1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    count += gen_straight_loss2(problem, p, {2,-2},{1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, 0.7*straight_w);
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    //diagonal neighbors
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.7*dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, 0.7*dist_w);
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w);
    
    return count;
}

int create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float unit, int flags = 0, float w_mul = 1.0)
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
    //far dist
    // if (far_dist != 1)
        // count += gen_straight_loss2(problem, p, {-far_dist,0},{0,0},{far_dist,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss2(problem, p, {0,0},{1,0},{2,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //diag
    count += gen_straight_loss2(problem, p, {-1,-1},{0,0},{1,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {-1,1},{0,0},{1,-1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {1,-1},{0,0},{-1,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss2(problem, p, {1,1},{0,0},{-1,-1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {0,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-1,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {1,-0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    //diag neighboars
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    //+1
    count += gen_dist_loss_fill(problem, p, {-2,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {2,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {0,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {0,2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    
    count += gen_dist_loss_fill(problem, p, {2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {2,2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    count += gen_dist_loss_fill(problem, p, {-2,2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w*w_mul);
    
    
    //+1
//     count += gen_dist_loss_fill(problem, p, {-5,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     count += gen_dist_loss_fill(problem, p, {5,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     count += gen_dist_loss_fill(problem, p, {0,-5}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     count += gen_dist_loss_fill(problem, p, {0,5}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     
//     
//     count += gen_dist_loss_fill(problem, p, {5,-5}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     count += gen_dist_loss_fill(problem, p, {5,5}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     count += gen_dist_loss_fill(problem, p, {-5,-5}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
//     count += gen_dist_loss_fill(problem, p, {-5,5}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w);
    
    return count;
}

float find_loc_wind_slow(cv::Vec2f &loc, float tgt_wind, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, const cv::Vec3f &tgt, float th)
{
    float best_res = -1;
    uint32_t sr = loc[0]+loc[1];
    for(int r=0;r<1000;r++) {
        cv::Vec2f cand = loc;
        
        if (r)
            cand = {rand_r(&sr) % points.cols, rand_r(&sr) % points.rows};
        
        // std::cout << "cand " << cand << winding(cand[1],cand[0]) << " " << tgt_wind << std::endl;
        
        if (std::isnan(winding(cand[1],cand[0])) || abs(winding(cand[1],cand[0])-tgt_wind) > 0.5)
            continue;
        
        cv::Vec3f out_;
        float res = min_loc(points, cand, out_, {tgt}, {0}, nullptr, 4.0, 0.01);
        
        // std::cout << res << std::endl;
        
        if (res < 0)
            continue;
        
        if (std::isnan(winding(cand[1],cand[0])) || abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
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

static float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d.dot(d);
}

float min_loc_wind(const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, cv::Vec2f &loc, cv::Vec3f &out, float tgt_wind, const cv::Vec3f &tgt, float init_step, float min_step, bool avoid_edges = true)
{
    if (!loc_valid(points, {loc[1],loc[0]})) {
        out = {-1,-1,-1};
        return -1;
    }
    
    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = abs(at_int(winding, loc)-tgt_wind)*10 + sdist(val,tgt);
    float res;
    
    // std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,0},{1,0}};
    float step = init_step;
    
    
    
    while (changed) {
        changed = false;
        
        for(auto &off : search) {
            cv::Vec2f cand = loc+off*step;
            
            if (!loc_valid(points, {cand[1],cand[0]})) {
                if (avoid_edges)
                    continue;
                else if (step < min_step*2) 
                {
                    out = {-1,-1,-1};
                    return -1;
                }
            }
            
            val = at_int(points, cand);
            // std::cout << "at" << cand << val << std::endl;
            res = abs(at_int(winding, cand)-tgt_wind)*10;
            res += sdist(val,tgt);
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
        
        if (step < min_step)
            break;
    }
    
    // std::cout << "best" << best << out << "\n" <<  std::endl;
    return best;
}

float find_loc_wind(cv::Vec2f &loc, float tgt_wind, const cv::Mat_<cv::Vec3f> &points, const cv::Mat_<float> &winding, const cv::Vec3f &tgt, float th, bool avoid_edges = true)
{
    float best_res = -1;
    uint32_t sr = loc[0]+loc[1];
    for(int r=0,full_r=0;r<1000,full_r<10000;full_r++) {
        cv::Vec2f cand = loc;
        
        if (full_r || !loc_valid_nan_xy(winding, cand))
            cand = {rand_r(&sr) % points.cols, rand_r(&sr) % points.rows};
        
        if (abs(winding(cand[1],cand[0])-tgt_wind) > 0.3)
            continue;
        
        r++;
        
        cv::Vec3f out_;
        float res = min_loc_wind(points, winding, cand, out_, tgt_wind, tgt, 4.0, 0.001, avoid_edges);
        
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
    
    return best_res;
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

static void write_ply(std::string path, const std::vector<cv::Vec3f> &points)
{
    std::cout << "ply " << points.size() << std::endl; 
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

int find_neighbors(cv::Mat_<cv::Vec3f> const &points, cv::Mat_<float> const &winding, const cv::Vec3f &ref_point, float ref_wind, std::vector<cv::Vec2d> &locs_out, std::vector<int> &wind_idx_out)
{
    for(int wf=-layer_reg_range;wf<=layer_reg_range;wf++) {
        cv::Vec2f loc = {0,0};
        float layer_tgt_w = ref_wind + wf;
        float res = find_loc_wind(loc, layer_tgt_w, points, winding, ref_point, 10.0);
        loc = {loc[1],loc[0]};
        cv::Vec3f found_p = at_int(points, {loc[1],loc[0]});
        float found_wind = at_int(winding, {loc[1],loc[0]});
        if (res >= 0 &&
            cv::norm(found_p - ref_point) <= layer_reg_range_vx
            && abs(found_wind - layer_tgt_w) <= 0.3)
        {
            locs_out.push_back(loc);
            wind_idx_out.push_back(wf);
            // std::cout << "potential neighbor at dist " << cv::norm(at_int(points, {loc[1],loc[0]}) - cv::Vec3f(points(p))) << " " << at_int(winding_in, {loc[1],loc[0]}) << " " << layer_tgt_w << std::endl;
        }
    }
    
    return locs_out.size();
}


cv::Mat_<cv::Vec3f> points_hr_grounding(const cv::Mat_<uint8_t> &state, std::vector<float> tgt_wind, const cv::Mat_<float> &winding, const cv::Mat_<cv::Vec3f> &points_tgt_in, const cv::Mat_<cv::Vec3f> &points_src, int step)
{
    cv::Mat_<cv::Vec3f> points_hr(points_tgt_in.rows*step, points_tgt_in.cols*step, {0,0,0});
    cv::Mat_<int> counts_hr(points_tgt_in.rows*step, points_tgt_in.cols*step, 0);
    
    cv::Mat_<cv::Vec3f> points_tgt = points_tgt_in.clone();

    for(int j=0;j<points_tgt.rows-1;j++)
        for(int i=0;i<points_tgt.cols-1;i++) {
            if (points_tgt(j,i)[0] == -1)
                continue;
            if (points_tgt(j,i+1)[0] == -1)
                continue;
            if (points_tgt(j+1,i)[0] == -1)
                continue;
            if (points_tgt(j+1,i+1)[0] == -1)
                continue;
            
            // std::cout << "try find corners" << std::endl;
            
            cv::Vec2f l00, l01, l10, l11;
            
            float hr_th = 20.0;
            float res;
            cv::Vec3f out_;

            res = find_loc_wind_slow(l00, tgt_wind[i], points_src, winding, points_tgt(j,i), hr_th*hr_th);
            if (res < 0 || res > hr_th*hr_th)
                continue;

            l01 = l00;
            res = min_loc(points_src, l01, out_, {points_tgt(j,i+1)}, {0}, nullptr, 1.0, 0.01);
            if (res < 0 || res > hr_th*hr_th)
                continue;
            l10 = l00;
            res = min_loc(points_src, l10, out_, {points_tgt(j+1,i)}, {0}, nullptr, 1.0, 0.01);
            if (res < 0 || res > hr_th*hr_th)
                continue;
            l11 = l00;
            res = min_loc(points_src, l11, out_, {points_tgt(j+1,i+1)}, {0}, nullptr, 1.0, 0.01);
            if (res < 0 || res > hr_th*hr_th)
                continue;
            
            //FIXME should also re-use already found corners for interpolation!
            points_tgt(j, i) = at_int(points_src, l00);
            points_tgt(j, i+1) = at_int(points_src, l01);
            points_tgt(j+1, i) = at_int(points_src, l10);
            points_tgt(j+1, i+1) = at_int(points_src, l11);
            
            l00 = {l00[1],l00[0]};
            l01 = {l01[1],l01[0]};
            l10 = {l10[1],l10[0]};
            l11 = {l11[1],l11[0]};
            
            
            // std::cout << "succ!" << res << cv::Vec2i(i,j) << l00 << l01 << points_tgt(j,i) << std::endl;
            
            for(int sy=0;sy<=step;sy++)
                for(int sx=0;sx<=step;sx++) {
                    float fx = float(sx)/step;
                    float fy = float(sy)/step;
                    cv::Vec2f l0 = (1-fx)*l00 + fx*l01;
                    cv::Vec2f l1 = (1-fx)*l10 + fx*l11;
                    cv::Vec2f l = (1-fy)*l0 + fy*l1;
                    // std::cout << l << at_int(points_src, {l[1],l[0]}) << std::endl;
                    points_hr(j*step+sy,i*step+sx) += at_int(points_src, {l[1],l[0]});
                    counts_hr(j*step+sy,i*step+sx) += 1;
                }
        }

        for(int j=0;j<points_tgt.rows-1;j++)
            for(int i=0;i<points_tgt.cols-1;i++) {
                if (!counts_hr(j*step+1,i*step+1)) {
                    for(int sy=0;sy<=step;sy++)
                        for(int sx=0;sx<=step;sx++) {
                            float fx = float(sx)/step;
                            float fy = float(sy)/step;
                            cv::Vec3f c0 = (1-fx)*points_tgt(j,i) + fx*points_tgt(j,i+1);
                            cv::Vec3f c1 = (1-fx)*points_tgt(j+1,i) + fx*points_tgt(j+1,i+1);
                            cv::Vec3f c = (1-fy)*c0 + fy*c1;
                            // std::cout << l << at_int(points_src, {l[1],l[0]}) << std::endl;
                            points_hr(j*step+sy,i*step+sx) += c;
                            counts_hr(j*step+sy,i*step+sx) += 1;
                        }
                    }
            }

#pragma omp parallel for
    for(int j=0;j<points_hr.rows;j++)
        for(int i=0;i<points_hr.cols;i++)
            if (counts_hr(j,i))
                points_hr(j,i) /= counts_hr(j,i);
    else
        points_hr(j,i) = {-1,-1,-1};
    
    return points_hr;
}

template <typename E> cv::Mat_<E> pad(const cv::Mat_<E> &src, int amount, const E &val)
{
    cv::Mat_<E> m(src.rows+2*amount, src.cols+2*amount, val);
    
    src.copyTo(m(cv::Rect(amount, amount, src.cols, src.rows)));
    
    std::cout << "pad " << src.size() << m.size() << std::endl;
    
    return m;
}

class diffuseWindings3D
{
public:
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 64};
    enum {FILL_V = 0};
    enum {SD = 4};
    const std::string UNIQUE_ID_STRING = "qt8m2n1tdy_"+std::to_string(BORDER)+"_"+std::to_string(CHUNK_SIZE)+"_"+std::to_string(FILL_V)+"_"+std::to_string(SD);
    std::vector<cv::Mat_<float>> _winds;
    std::vector<cv::Mat_<cv::Vec3f>> _points;
    
    diffuseWindings3D(const std::vector<cv::Mat_<float>> &winds, const std::vector<cv::Mat_<cv::Vec3f>> &points) : _winds(winds), _points(points)
    {
        for(auto &m : _points) {
            cv::Mat_<cv::Vec3f> dup = m.clone();
            dup.forEach([](cv::Vec3f &p, const int * position) -> void {
                if (p[0] == -1)
                    p = {NAN,NAN,NAN};
            });
            cv::resize(dup, m, {0,0}, 4, 4);
        }
        for(auto &m : _winds) {
            cv::resize(m, m, {0,0}, 4, 4);
        }
    };
    
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        T winds_large = xt::empty<E>({CHUNK_SIZE+2*BORDER,CHUNK_SIZE+2*BORDER,CHUNK_SIZE+2*BORDER});
        winds_large.fill(NAN);

        std::cout << "compute chunk" << offset_large << std::endl;
        
        for(int s=0;s<_points.size();s++)
#pragma omp parallel for collapse(2)
            for(int j=0;j<_points[s].rows;j++)
                for(int i=0;i<_points[s].cols;i++) {
                    cv::Vec3i p = _points[s](j,i)/SD;
                    
                    if (std::isnan(p[0]))
                        continue;
                    
                    p = {p[2],p[1],p[0]};
                    
                    // std::cout << p << offset_large << std::endl;
                    
                    bool inside = true;
                    for(int d=0;d<3;d++)
                        if (p[d] < offset_large[d]) {
                            inside = false;
                            break;
                        }
                        else if (p[d] >= offset_large[d]+CHUNK_SIZE+2*BORDER){
                            inside = false;
                            break;
                        }
                        
                    if (!inside)
                        continue;
                            
                    winds_large(p[0]-offset_large[0], p[1]-offset_large[1], p[2]-offset_large[2]) = _winds[s](j,i);
                }
        
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        
        auto crop_winds = view(winds_large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
        
        small = crop_winds;
    }
    
};

int main(int argc, char *argv[])
{
    if (argc != 5 && (argc-2) % 3 != 0)  {
        std::cout << "usage: " << argv[0] << " <params.json> <tiffxyz> <winding> <weight> ..." << std::endl;
        std::cout << "  multiple triplets of <tiffxyz> <winding> <weight> can be used for a joint optimization" << std::endl;
        return EXIT_SUCCESS;
    }
    
    std::vector<QuadSurface*> surfs;
    std::vector<cv::Mat_<cv::Vec3f>> surf_points;
    std::vector<cv::Mat_<float>> winds;
    std::vector<cv::Mat_<uint8_t>> supports;
    std::vector<float> weights;
    std::vector<cv::Mat_<cv::Vec2d>> surf_locs;

    std::ifstream params_f(argv[1]);
    json params = json::parse(params_f);
    
    trace_mul = params.value("trace_mul", 1);
    dist_w = params.value("dist_w", 0.3);
    straight_w = params.value("straight_w", 0.02);
    surf_w = params.value("surf_w", 0.1);
    z_loc_loss_w = params.value("z_loc_loss_w", 0.0005);
    wind_w = params.value("wind_w", 100.0);
    wind_th = params.value("wind_th", 0.3);
    inpaint_back_range = params.value("inpaint_back_range", 40);
    int opt_w_short = params.value("opt_w", 4);
    
    int opt_w = opt_w_short;
    int large_opt_w = 32;
    int large_opt_every = 8;
    
    for(int n=0;n<argc/3;n++) {
        QuadSurface *surf = load_quad_from_tifxyz(argv[n*3+2]);
        
        cv::Mat_<float> wind = cv::imread(argv[n*3+3], cv::IMREAD_UNCHANGED);
        
        surfs.push_back(surf);
        winds.push_back(wind);
        surf_points.push_back(surf->rawPoints());
        weights.push_back(atof(argv[n*3+4]));
        supports.push_back(cv::Mat_<uint8_t>(winds[0].size(), 0));
        surf_locs.push_back(cv::Mat_<cv::Vec2d>(winds[0].size(), {-1,-1}));
    }
    
    //
    if (surfs.size() > 1) {
        for(int i=1;i<surfs.size();i++) {
            //try to find random matches between the surfaces, always coming from surf 0 for now
            std::vector<float> offsets;
            
            for(int r=0;r<1000;r++) {
                cv::Vec2i p = {rand() % surf_points[0].rows, rand() % surf_points[0].cols};
                if (surf_points[0](p)[0] == -1)
                    continue;
                
                
                SurfacePointer *ptr = surfs[i]->pointer();
                float res = surfs[i]->pointTo(ptr, surf_points[0](p), 2.0);
                
                if (res < 0 || res >= 2)
                    continue;
                
                cv::Vec3f loc = surfs[i]->loc_raw(ptr);
                
                // std::cout << loc << std::endl;
                // std::cout << winds[0](p) - at_int(winds[i], {loc[0],loc[1]}) << std::endl;
                offsets.push_back(winds[0](p) - at_int(winds[i], {loc[0],loc[1]}));
            }
            
            std::sort(offsets.begin(), offsets.end());
            std::cout << "off 0.1 " << offsets[offsets.size()*0.1] << std::endl;
            std::cout << "off 0.5 " << offsets[offsets.size()*0.5] << std::endl;
            std::cout << "off 0.9 " << offsets[offsets.size()*0.9] << std::endl;
            
            winds[i] += offsets[offsets.size()*0.5];
        }
    }
    
    for(int s=0;s<surf_points.size();s++) {
        surf_points[s] = pad(surf_points[s], 20, {-1,-1,-1});
        winds[s] = pad(winds[s], 20, NAN);
    }
    
    cv::Mat_<cv::Vec3f> points_in = surf_points[0];
    cv::Mat_<float> winding_in = winds[0].clone();
    
    if (!params.contains("cache_root")) {
        std::cout << "need cache_root, via .json parameters" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << params["cache_root"] << std::endl;
    
    diffuseWindings3D compute(winds, surf_points);
    Chunked3d<float,diffuseWindings3D> proc_tensor(compute, nullptr, nullptr, params["cache_root"]); 
    
    cv::Mat_<float> wind_dbg(500,500, NAN);
    
    for(int j=0;j<wind_dbg.rows;j++)
        for(int i=0;i<wind_dbg.cols;i++) {
            wind_dbg(j,i) = proc_tensor(1250,j+500,i+700);
        }
    
//     cv::Vec3i offset_large = {5000,2000,3000};
//     cv::Vec3i offset_size = {1,2000,3000};
//     for(int s=0;s<surf_points.size();s++)
// #pragma omp parallel for collapse(2)
//         for(int j=0;j<surf_points[s].rows;j++)
//             for(int i=0;i<surf_points[s].cols;i++) {
//                 cv::Vec3i p = surf_points[s](j,i);
//                     
//                 if (p[0] == -1)
//                     continue;
//                 
//                 p = {p[2],p[1],p[0]};
//                 
//                 // std::cout << p << offset_large << std::endl;
//                 
//                 bool inside = true;
//                 for(int d=0;d<3;d++)
//                 if (p[d] < offset_large[d]) {
//                     inside = false;
//                     break;
//                 }
//                 else if (p[d] >= offset_large[d]+offset_size[d]){
//                     inside = false;
//                     break;
//                 }
//                     
//                 if (!inside)
//                     continue;
// 
//                 wind_dbg(p[1]-offset_large[1],p[2]-offset_large[2]) = winds[s](j,i);
//             }
    
    cv::imwrite("wind_inp.tif", wind_dbg);
    return EXIT_SUCCESS;
    
    
    // cv::Rect bbox_src(10,10,points_in.cols-20,points_in.rows-20);
    cv::Rect bbox_src(70,10,points_in.cols-10-70,points_in.rows-20);
    // cv::Rect bbox_src(3300,10,1000,points_in.rows-20);
    
    float src_step = 20;
    float step = src_step*trace_mul;
    
    cv::Size size = {points_in.cols/trace_mul, points_in.rows/trace_mul};
    cv::Rect bbox = {bbox_src.x/trace_mul, bbox_src.y/trace_mul, bbox_src.width/trace_mul, bbox_src.height/trace_mul};

    cv::Mat_<uint8_t> state(size, 0);
    cv::Mat_<uint8_t> init_state(size, 0);
    cv::Mat_<cv::Vec3d> points(size, {-1,-1,-1});
    cv::Mat_<cv::Vec2d> locs(size, {-1,-1});
    cv::Mat_<float> winding(size, NAN);
    cv::Mat_<float> init_errs(size, -1);
    
    cv::Mat_<uint8_t> fail_code(size, 0);
    
    cv::Rect first_col = {bbox.x,bbox.y,opt_w,bbox.height};
    
    int last_miny, last_maxy;
    
    //FIXME use joint first col for whole start?
    
    for(int i=first_col.x;i<first_col.br().x;i++) {
        int col_first = first_col.height;
        int col_last = -1;
        for(int j=first_col.y;j<first_col.br().y;j++) {            
            if (points_in(j*trace_mul,i*trace_mul)[0] != -1) {
                col_first = std::min(col_first, j);
                col_last = std::max(col_last, j);
            }
            else
                //for now only take the first contiguous block!
                if (col_first != first_col.height)
                    break;
        }
        
        std::cout << " i " << i << " " << col_first << " " << col_last << std::endl;
        
        last_miny = col_first;
        last_maxy = col_last;
        
        for(int j=col_first;j<=col_last;j++) {
            points(j,i) = points_in(j*trace_mul, i*trace_mul);
            winding(j,i) = winding_in(j*trace_mul, i*trace_mul);
            if (points(j,i)[0] != -1) {
                state(j,i) = STATE_LOC_VALID | STATE_COORD_VALID;
                locs(j,i) = {j*trace_mul,i*trace_mul};
            }
            else {
                if (points(j-1,i)[0] != -1) 
                    points(j, i) = points(j-1,i) + cv::Vec3d(0.1,0.1,0.1);
                else
                    points(j, i) = {rand()%1000,rand()%1000,rand()%1000};
                state(j,i) = STATE_COORD_VALID;
            }
                
        }
    }
    
    cv::imwrite("state.tif", state*20);
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    
    
    ceres::Solver::Options options_col;
    // options.linear_solver_type = ceres::DENSE_QR;
    options_col.linear_solver_type = ceres::SPARSE_SCHUR;
    options_col.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    options_col.num_threads = 32;
    options_col.minimizer_progress_to_stdout = false;
    options_col.max_num_iterations = 10000;
    
    cv::Vec3f seed_coord = {-1,-1,-1};
    cv::Vec2i seed_loc = {-1,-1};
    
    std::vector<float> avg_wind(size.width);
    std::vector<int> wind_counts(size.width);
    std::vector<float> tgt_wind(size.width);
    
    {
        ceres::Problem problem_init;
        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++) {
                create_centered_losses(problem_init, {j,i}, state, points_in, points, locs, step, 0);

                if (loc_valid(state(j,i))) {
                    avg_wind[i] += winding_in(j*trace_mul,i*trace_mul);
                    wind_counts[i]++;

                    if (seed_loc[0] == -1) {
                        seed_loc = {j,i};
                        seed_coord = points_in(j*trace_mul,i*trace_mul);
                    }
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
            for(int i=first_col.x;i<first_col.br().x;i++) {
                cv::Vec2i p = {j,i};
                if (!loc_valid(state(p)) && coord_valid(state(p))) {
                    if (problem_init.HasParameterBlock(&locs(p)[0]))
                        problem_init.SetParameterBlockVariable(&locs(p)[0]);
                    if (problem_init.HasParameterBlock(&points(p)[0]))
                        problem_init.SetParameterBlockVariable(&points(p)[0]);
                }
                else {
                    if (problem_init.HasParameterBlock(&locs(p)[0]))
                        problem_init.SetParameterBlockConstant(&locs(p)[0]);
                    if (problem_init.HasParameterBlock(&points(p)[0]))
                        problem_init.SetParameterBlockConstant(&points(p)[0]);
                }
                    
            }
                    
        
        ceres::Solver::Summary summary;
        // ceres::Solve(options_col, &problem_init, &summary);
        // std::cout << summary.BriefReport() << std::endl;
        
        for(int j=first_col.y;j<first_col.br().y;j++) {
            for(int i=first_col.x;i<first_col.br().x;i++) {
                if (problem_init.HasParameterBlock(&locs(j,i)[0]))
                    problem_init.SetParameterBlockVariable(&locs(j,i)[0]);
                if (problem_init.HasParameterBlock(&points(j,i)[0]))
                    problem_init.SetParameterBlockVariable(&points(j,i)[0]);
                }
            }

        for(int j=first_col.y;j<first_col.br().y;j++)
            for(int i=first_col.x;i<first_col.br().x;i++) {
                if (loc_valid(state(j,i))) {
                    problem_init.AddResidualBlock(Interp2DLoss<float>::Create(winding_in, avg_wind[i], wind_w), nullptr, &locs(j,i)[0]);
                    problem_init.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(points_in, seed_coord[2] - (j-seed_loc[0])*step, z_loc_loss_w), nullptr, &locs(j,i)[0]);
                }
            }
            
        problem_init.SetParameterBlockConstant(&points(seed_loc)[0]);
        problem_init.SetParameterBlockConstant(&locs(seed_loc)[0]);
        
        // ceres::Solve(options_col, &problem_init, &summary);
        // std::cout << summary.BriefReport() << std::endl;
        
    }
    
    int expand_rate = 2;
    int shrink_inv_rate = 2;
    int dilate = 20;
    
    int lower_bound = last_miny;
    int upper_bound = last_maxy;
    int lower_loc = lower_bound;
    int upper_loc = upper_bound;
    
    // std::vector<cv::Vec2i> neighs = {{0,-1},{-1,-1},{1,-1},{-2,-1},{2,-1},{2,-2},{1,-2},{0,-2},{-1,-2},{-2,-2},{-3,-2},{3,-2},{-4,-2},{4,-2}};
    std::vector<cv::Vec2i> neighs = {{0,-1},{-1,-1},{1,-1},{-2,-1},{2,-1}};
    
    cv::Mat_<float> surf_dist(points.size(), 0);
    
    // std::vector<cv::Vec3f> layer_neighs, layer_neighs_inp;
    
    int runaways = 0;
    
    for(int i=bbox.x+opt_w;i<bbox.br().x;i++) {
        if (i % large_opt_every == 0 && i-bbox.x > large_opt_w)
            opt_w = large_opt_w;
        else
            opt_w = opt_w_short;
        
        tgt_wind[i] = 2*avg_wind[i-1]-avg_wind[i-2];
        
        std::cout << "wind tgt: " << tgt_wind[i] << " " << avg_wind[i-1] << " " << avg_wind[i-2] << std::endl;
        
        if (lower_bound > lower_loc-dilate)
            lower_bound = std::max(lower_loc-dilate,lower_bound-expand_rate);
        else if (lower_bound < lower_loc-dilate)
            lower_bound = std::min(lower_loc-dilate,lower_bound + (i % shrink_inv_rate));
            
        if (upper_bound < upper_loc+dilate)
            upper_bound = std::min(upper_loc+dilate,upper_bound+expand_rate);
        else if (upper_bound > upper_loc+dilate)
            upper_bound = std::max(upper_loc+dilate,upper_bound - (i % shrink_inv_rate));
        
        std::cout << "proc col " << i << std::endl;
        ceres::Problem problem_col;
#pragma omp parallel for
        for(int j=std::max(bbox.y,lower_bound);j<std::min(bbox.br().y,upper_bound+1);j++) {
            cv::Vec2i p = {j,i};
            
            locs(p) = {-1,-1};
            points(p) = {-1,-1, -1};
            
            state(p) = 0;
            
            //FIXME
            for(auto n : neighs) {
                // cv::Vec2d cand = locs(p+n) + cv::Vec2d(0,1/step);
                // if (loc_valid(points_in,cand) && loc_valid(state(p+n))) {
                //     state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                //     locs(p) = cand;
                //     points(p) = at_int(points_in, {cand[1],cand[0]})+cv::Vec3f(((j+i)%10)*0.01, ((j+i+1)%10)*0.01,((j+i+2)%10)*0.01);
                //     break;
                // }
                if (!state(p) && coord_valid(state(p+n))) {
                    points(p) = points(p+n)+cv::Vec3d(((j+i)%10)*0.01, ((j+i+1)%10)*0.01,((j+i+2)%10)*0.01);
                    state(p) = STATE_COORD_VALID;
                    break;
                }
            }
            
            // init_state(p) = state(p);
            
            if (points(p)[0] == -1)
                continue;
            
            //TODO check wind!cv::norm(at_int(winding_in, {loc[1],loc[0]}) - tgt_wind[i]) <= wind_th
            
            {
                ceres::Solver::Summary summary;
                ceres::Problem problem;
                create_centered_losses_left_large(problem, p, state, points_in, points, locs, step, 0);
                // problem.AddResidualBlock(Interp2DLoss<float>::Create(winding, tgt_wind[i], wind_w), nullptr, &locs(p)[0]);
                // problem.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(points_in, seed_coord[2] - (p[0]-seed_loc[0])*step, z_loc_loss_w), nullptr, &locs(p)[0]);
                
                ceres::Solve(options, &problem, &summary);
                
                init_errs(p) = sqrt(summary.final_cost/summary.num_residual_blocks);
            }
            
            // if (!loc_valid(winding_in, locs(p)) || std::abs(at_int(winding_in, {locs(p)[1],locs(p)[0]}) - tgt_wind[i]) > wind_th) {
            //     state(p) &= ~STATE_LOC_VALID;
            //     locs(p) = {-1,-1};
            // }
            
            //TODO running this unconditionally should not break anything!
// //             if (!loc_valid(points_in, locs(p))) {
// //                 // bool valid_neigh = false;
// //                 // std::vector<cv::Vec2i> direct_neighs = {{0,-1},{-1,-1},{1,-1}};
// //                 // for (auto n : direct_neighs)
// //                 //     if (loc_valid(state(p+n)))
// //                 //         valid_neigh = true;
// //                 
// //                 // if (valid_neigh) {
// //                     cv::Vec2f loc = {0,0};
// //                     float res = find_loc_wind(loc, tgt_wind[i], points_in, winding_in, points(p), 10.0, false);
// //                     loc = {loc[1],loc[0]};
// //                     if (res >= 0 &&
// //                         cv::norm(at_int(points_in, {loc[1],loc[0]}) - cv::Vec3f(points(p))) <= 100
// //                         && cv::norm(at_int(winding_in, {loc[1],loc[0]}) - tgt_wind[i]) <= wind_th) {
// //                             locs(p) = loc;
// //                             state(p) = STATE_COORD_VALID | STATE_LOC_VALID;
// //                             // std::cout << res << " " << cv::norm(at_int(points_in, {loc[1],loc[0]}) - cv::Vec3f(points(p))) << " " << cv::norm(at_int(winding_in, {loc[1],loc[0]}) - tgt_wind[i]) << std::endl;
// //                         }
// //                 // }
// //             }
                    

//             std::vector<cv::Vec2d> locs_layers;
//             std::vector<int> idxs_layers;
//             find_neighbors(points_in, winding_in, points(p), tgt_wind[i], locs_layers, idxs_layers);
//             
//             if (loc_valid(points_in, locs(p)))
// #pragma omp critical
//                 for(auto & l : locs_layers)
//                     layer_neighs.push_back(at_int(points_in, {l[1],l[0]}));
//             else
// #pragma omp critical
//                 for(auto & l : locs_layers)
//                     layer_neighs_inp.push_back(at_int(points_in, {l[1],l[0]}));
        }
        
        last_miny--;
        last_maxy++;
        
        // if (i % 10 == 0) {
        //     write_ply("col_layer_neighs.ply", layer_neighs);
        //     write_ply("col_layer_neighs_inp.ply", layer_neighs_inp);
        // }
        
// //         for(int j=bbox.y;j<bbox.br().y;j++) {
// //             cv::Vec2i p = {j,i};
// //             
// //             for(int o=0;o<=opt_w;o++) {
// //                 cv::Vec2i po = {j,i-o};
// //                 if (!loc_valid(points_in,locs(po)) || std::abs(at_int(winding_in, {locs(po)[1],locs(po)[0]}) - tgt_wind[i-o]) > wind_th) {
// //                     state(po) &= ~STATE_LOC_VALID;
// //                     locs(po) = {-1,-1};
// //                 }
// //             }
// //         }
        
        cv::Mat_<uint8_t> state_inpaint = state.clone();
//         cv::Mat_<uint8_t> mask;
//         bitwise_and(state, (uint8_t)STATE_LOC_VALID, mask);
//         cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
//         cv::dilate(mask, mask, m, {-1,-1}, 80/trace_mul);
//         
//         //also fill the mask in y dir
//         for(int x=std::max(bbox.x,i-opt_w);x<=i;x++) {
//             int col_first = first_col.height;
//             int col_last = -1;
//             for(int j=0;j<state_inpaint.rows;j++) {
//                 if (mask(j,x)) {
//                     col_first = std::min(col_first, j);
//                     col_last = std::max(col_last, j);
//                 }
//             }
//             for(int j=col_first;j<=col_last;j++)
//                 mask(j,x) = 1;
//             
//             if (x == i) {
//                 last_miny = std::max(last_miny-1, col_first);
//                 last_maxy = std::min(last_maxy+1, col_last);
//             }
//         }
//         
//         //FIXME where do we add not yet used points?
//         for(int j=0;j<state_inpaint.rows;j++)
//             for(int x=std::max(bbox.x,i-opt_w);x<=i;x++)
//                 if (!mask(j,x) && coord_valid(state(j,x))) {
//                     state(j,x) = 0;
//                     state_inpaint(j,x) = 0;
//                     points(j,x) = {-1,-1, -1};
//                     //TODO should we also reset all surf locs?
//                 }
                    
        
        //FIXME check everwrite better with multi-surf!
        // for(int j=0;j<state_inpaint.rows;j++)
        //     for(int x=first_col.x;x<=i;x++) {
        //         if (loc_valid(state(j,x))) {
        //             if (points(j,x)[0] == -1)
        //                 throw std::runtime_error("need points 3!");
        //             state_inpaint(j,x) = STATE_COORD_VALID | STATE_LOC_VALID;
        //         }
        //         else if (mask(j,x) && coord_valid(state(j,x))) {
        //             // std::cout << "inpaint only! " << cv::Vec2i(x,j) << std::endl;
        //             if (points(j,x)[0] != -1) {
        //                 state_inpaint(j,x) = STATE_COORD_VALID;
        //                 locs(j,x) = {-1,-1};
        //             }
        //             else {
        //                 state_inpaint(j,x) = 0;
        //                 points(j,x) = {-1,-1,-1};
        //                 locs(j,x) = {-1,-1};
        //                 //TODO still not sure shy this happens? is it still happening?
        //                 // std::cout << "no valid coord! " << cv::Vec2i(x,j) << std::endl;
        //             }
        //         }
        //         else {
        //             state_inpaint(j,x) = 0;
        //             points(j,x) = {-1,-1,-1};
        //             locs(j,x) = {-1,-1};
        //         }
        //     }
            
//         for(int j=bbox.y;j<bbox.br().y;j++) {
//             cv::Vec2i p = {j,i};
//             
//             for(int o=0;o<=opt_w;o++) {
//                 cv::Vec2i po = {j,i-o};
//                 create_centered_losses(problem_col, po, state_inpaint, points_in, points, locs, step, 0);
//                 
//                 // gen_surfloss(po, problem_col, state_inpaint, points_in, points, locs, surf_w*weights[0]);
//             }
//             
// //             if (state_inpaint(p) & STATE_LOC_VALID)
// //                 problem_col.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(points_in, seed_coord[2] - (p[0]-seed_loc[0])*step, z_loc_loss_w), nullptr, &locs(p)[0]);
// //             
// //             for(int o=0;o<opt_w;o++)
// //                 if (state_inpaint(j,i-o) & STATE_LOC_VALID)
// //                     problem_col.AddResidualBlock(Interp2DLoss<float>::Create(winding_in, tgt_wind[i-o], wind_w), nullptr, &locs(j,i-o)[0]);
//         }
        
        // std::vector<cv::Vec2d> add_locs;
        std::vector<cv::Vec2i> add_ps;
        std::vector<int> add_idxs;
        
        
        //FIXME re-use existing locs if there!
#pragma omp parallel for
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=0;o<=opt_w;o++) {
                //only add in area where we also inpaint
                if (!coord_valid(state(j,i-o)))
                    continue;
                
                cv::Vec2i po = {j,i-o};
                // if (surfs.size() > 1 && coord_valid(state_inpaint(po)))
                    for(int s=0;s<surf_points.size();s++)
                    {
                        cv::Vec2f loc = {0,0};
                        float res = find_loc_wind(loc, tgt_wind[i-o], surf_points[s], winds[s], points(po), 1.0, false);
                        loc = {loc[1],loc[0]};
                        if (res >= 0 &&
                            cv::norm(at_int(surf_points[s], {loc[1],loc[0]}) - cv::Vec3f(points(po))) <= 100
                            && cv::norm(at_int(winds[s], {loc[1],loc[0]}) - tgt_wind[i-o]) <= wind_th)
#pragma omp critical
                            {
                                // std::cout << "adding " << cv::norm(at_int(surf_points[s], {loc[1],loc[0]}) - cv::Vec3f(points(po))) << " " << cv::norm(at_int(winds[s], {loc[1],loc[0]}) - tgt_wind[i-o]) << at_int(surf_points[s], {loc[1],loc[0]}) << loc << po << " " << s << std::endl;
                                // add_locs.push_back(loc);
                                surf_locs[s](po) = loc;
                                add_idxs.push_back(s);
                                add_ps.push_back(po);
                                supports[s](po) = 1;
                            }
                        else
                            supports[s](po) = 0;
                    }
                
            }
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++)
            for(int o=0;o<=opt_w;o++) {
                cv::Vec2i po = {j,i-o};

                int sup_count = 0;
                for(int s=0;s<surf_points.size();s++)
                    sup_count += supports[s](po);
                if (sup_count)
                    state(po) = STATE_LOC_VALID | STATE_COORD_VALID;
                else
                    state(po) &= ~STATE_LOC_VALID;
            }
        
        for(int n=0;n<add_idxs.size();n++) {
            int idx = add_idxs[n];
            cv::Vec2i p = add_ps[n];
            //FIXME THESE POINTS ARE HANDLED AS INPAINT AREA IN LATER STEPS!!!
            problem_col.AddResidualBlock(SurfaceLossD::Create(surf_points[idx], surf_w*weights[idx]), nullptr, &points(p)[0], &surf_locs[idx](p)[0]);
            problem_col.AddResidualBlock(Interp2DLoss<float>::Create(winds[idx], tgt_wind[p[1]], wind_w*weights[idx]), nullptr, &surf_locs[idx](p)[0]);
            // problem_col.AddResidualBlock(ZLocationLoss<cv::Vec3f>::Create(surf_points[idx], seed_coord[2] - (p[0]-seed_loc[0])*step, z_loc_loss_w*weights[idx]), nullptr, &surf_locs[idx](p)[0]);
        }
        
        //FIXME add centerd losses whe
        for(int j=bbox.y;j<bbox.br().y;j++)
            for(int o=std::max(bbox.x,i-inpaint_back_range);o<=i;o++)
                if (coord_valid(state(j, o))) {
                    float w_mul = 1.0;
                    if (!coord_valid(state(j, o)))
                        w_mul = 0.3;
                    create_centered_losses(problem_col, {j, o}, state_inpaint, points_in, points, locs, step, 0, w_mul);
                    problem_col.AddResidualBlock(ZCoordLoss::Create(seed_coord[2] - (j-seed_loc[0])*step, z_loc_loss_w), nullptr, &points(j,o)[0]);
                }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-inpaint_back_range);o<=i;o++) {
                if (problem_col.HasParameterBlock(&locs(j, o)[0]))
                    problem_col.SetParameterBlockVariable(&locs(j, o)[0]);
                if (problem_col.HasParameterBlock(&points(j, o)[0]))
                    problem_col.SetParameterBlockVariable(&points(j, o)[0]);
            }
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-inpaint_back_range-2);o<=i-inpaint_back_range;o++)
                if (loc_valid(state(j, o)))
                    if (problem_col.HasParameterBlock(&points(j,o)[0]))
                        problem_col.SetParameterBlockConstant(&points(j,o)[0]);
        }
        
        for(int j=bbox.y;j<bbox.br().y;j++) {
            for(int o=std::max(bbox.x,i-inpaint_back_range-2);o<=i;o++)
                if (loc_valid(state(j, o)))
                    if (problem_col.HasParameterBlock(&points(j,o)[0]))
                        problem_col.SetParameterBlockConstant(&points(j,o)[0]);
        }
            
        std::cout << "start solve with blocks " << problem_col.NumResidualBlocks() << std::endl;
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
        
        if (i % 10 == 0) {
            cv::imwrite("state_inpaint_pref.tif",state_inpaint(bbox)*20);
        }

        bool stop = false;
        float min_w = 0, max_w = 0;
        lower_loc = bbox.br().x+1;
        upper_loc = -1;
        for(int x=std::max(i-opt_w,bbox.x);x<=i;x++) {
//             avg_wind[x] = 0;
//             wind_counts[x] = 0;
//             for(int j=bbox.y;j<bbox.br().y;j++) {
//                 cv::Vec2i p = {j,x};
//                 
//                 if (loc_valid(points_in, locs(p)))
//                     surf_dist(p) = cv::norm(cv::Vec3f(points(p))-at_int(points_in, {locs(p)[1],locs(p)[0]}));
//                 else
//                     surf_dist(p) = -1;
//                 
//                 if (!loc_valid(points_in,locs(j,x))) {
//                     locs(j,x) = {-1,-1};
//                     winding(j, x) = NAN;
//                     state_inpaint(j,x) &= ~STATE_LOC_VALID;
//                 }
//                 else {
//                     winding(j, x) = at_int(winding_in, {locs(j,x)[1],locs(j,x)[0]});
//                     
//                     if (abs(winding(j, x)-tgt_wind[x]) <= wind_th) {
//                         avg_wind[x] += winding(j, x);
//                         wind_counts[x] ++;
//                     }
//                     else
//                         runaways++;
//                     
//                     min_w = std::min(min_w,winding(j, x)-tgt_wind[x]);
//                     max_w = std::max(max_w,winding(j, x)-tgt_wind[x]);
//                 }
// 
//                 if (!coord_valid(state_inpaint(j,x))) {
//                     points(j,x) = {-1,-1,-1};
//                     state(j,x) = 0;
//                 }
//                 else {
//                     state(j,x) = state_inpaint(j, x);
//                     if (points(j,x)[0] == -1)
//                         throw std::runtime_error("need points 2!");
//                 }
            
            //FIXME do this but check for tboht suport and loc valid!
//             for(int j=bbox.y;j<bbox.br().y;j++)
//                 for(int o=0;o<=opt_w;o++) {
//                     cv::Vec2i po = {j,i-o};
//                     
//                     int sup_count = 0;
//                     for(int s=0;s<surfs.size();s++)
//                         sup_count += supports[s](po);
//                     if (sup_count)
//                         state(po) = STATE_LOC_VALID;
//                     else
//                         state(po) &= ~STATE_LOC_VALID;
//                 }
            
            avg_wind[x] = 0;
            wind_counts[x] = 0;
            for(int j=bbox.y;j<bbox.br().y;j++) {
                
                cv::Vec2i p = {j,x};
                for (int s=0;s<surf_points.size();s++) {
                    if (supports[s](p)) {
                        if (loc_valid(surf_points[s], surf_locs[s](p))) {
                            if (abs(at_int(winds[s], {surf_locs[s](p)[1],surf_locs[s](p)[0]}) - tgt_wind[x]) <= wind_th) {
                                //FIXME check wind + support + loc avlid
                                avg_wind[x] += at_int(winds[s], {surf_locs[s](p)[1],surf_locs[s](p)[0]});
                                wind_counts[x]++;
                                if (i == x) {
                                    upper_loc = std::max(upper_loc, j);
                                    lower_loc = std::min(lower_loc, j);
                                }
                                    
                                // std::cout << "got wind " << x << std::endl;
                                
                            }
                            else
                                std::cout << "wind th " << abs(at_int(winds[s], {surf_locs[s](p)[1],surf_locs[s](p)[0]}) - tgt_wind[x]) << " " << at_int(winds[s], {surf_locs[s](p)[1],surf_locs[s](p)[0]}) << tgt_wind[x] << " " << std::endl;
                        }
                        else
                        {
                            std::cout << "lost a point! " << p << " " << s << " " << surf_locs[s](p) << std::endl;
                        }
                    }
                }
            }
            if (wind_counts[x])
                avg_wind[x] /= wind_counts[x];
            else {
                stop = true;
            }
        }
            
        if (avg_wind[i-1] < avg_wind[i-2] - wind_th/2) {
            stop = true;
            std::cout << "stopping wind is wrong! " << avg_wind[i-2] << " " << avg_wind[i-1]  << std::endl;
        }
            
        if (i % 10 == 0 || !wind_counts[i] || i == bbox.br().x-1 || stop) {
            std::vector<cv::Mat> chs;
            cv::split(points, chs);
            cv::imwrite("newx.tif",chs[0](bbox));
            cv::imwrite("newz.tif",chs[2](bbox));
            cv::imwrite("surf_dist.tif",surf_dist(bbox));
            cv::imwrite("winding_out.tif",winding(bbox)+3);
            cv::imwrite("state.tif",state*20);
            cv::imwrite("state_inpaint.tif",state_inpaint(bbox)*20);
            cv::imwrite("init_state.tif",init_state*20);
            cv::imwrite("init_errs.tif",init_errs(bbox));
            
            for(int s=0;s<supports.size();s++)
                cv::imwrite("supports"+std::to_string(s)+".tif",supports[s](bbox)*255);
                
        }
            
        if (!wind_counts[i]) {
            std::cout << "stopping as zero valid locations found!" << i << std::endl;
            break;
        }
        
        if (stop)
            break;
        
        
        std::cout << "avg wind number for col " << i << " : " << avg_wind[i] << " ( tgt was " << tgt_wind[i] << " ) using #" << wind_counts[i]  << " spread " << min_w << " - " << max_w << " runaway: " << runaways << std::endl;
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
    
    {
        QuadSurface *surf_full = new QuadSurface(points(bbox), surfs[0]->_scale/trace_mul);
        fs::path tgt_dir = "/home/hendrik/data/ml_datasets/vesuvius/manual_wget/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/";
        std::string name_prefix = "testing_fill_";
        std::string uuid = name_prefix + time_str();
        fs::path seg_dir = tgt_dir / uuid;
        std::cout << "saving " << seg_dir << std::endl;
        surf_full->save(seg_dir, uuid);
        
        cv::Mat_<float> winding_ideal(winding.size(), NAN);
        for(int i=0;i<tgt_wind.size();i++)
            winding_ideal(cv::Rect(i,0,1,winding.rows)).setTo(tgt_wind[i]);

        cv::imwrite(seg_dir/"winding_exact.tif",winding);
        cv::imwrite(seg_dir/"winding.tif",winding_ideal);
    }
    
    {
        cv::Mat_<cv::Vec3f> points_hr = points_hr_grounding(state, tgt_wind, winding_in, points, points_in, trace_mul);
        QuadSurface *surf_hr = new QuadSurface(points_hr, surfs[0]->_scale);
        fs::path tgt_dir = "/home/hendrik/data/ml_datasets/vesuvius/manual_wget/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/";
        std::string name_prefix = "testing_fill_hr_";
        std::string uuid = name_prefix + time_str();
        fs::path seg_dir = tgt_dir / uuid;
        std::cout << "saving " << seg_dir << std::endl;
        surf_hr->save(seg_dir, uuid);
    }
    
    return EXIT_SUCCESS;
}
