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
    
    ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), new ceres::HuberLoss(1.0), &dpoints(p)[0], &dpoints(p+off)[0]);
    // ceres::ResidualBlockId tmp = problem.AddResidualBlock(DistLoss::Create(unit*cv::norm(off),w), nullptr, &dpoints(p)[0], &dpoints(p+off)[0]);
    
    if (res)
        *res = tmp;
    
    if (!optimize_all)
        problem.SetParameterBlockConstant(&dpoints(p+off)[0]);
    
    return 1;
}

static float dist_w = 1.0;
static float straight_w = 1.0;
static float surf_w = 0.1;

int create_centered_losses(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, const cv::Mat_<cv::Vec3f> &points_in, cv::Mat_<cv::Vec3d> &points, cv::Mat_<cv::Vec2d> &locs, float unit, int flags = 0)
{
    //generate losses for point p
    int count = 0;
    
    //horizontal
    count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //vertical
    // count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    // count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //furtehr and diag!
    count += gen_straight_loss(problem, p, {-2,-2},{-1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss(problem, p, {2,-2},{1,-1},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    count += gen_straight_loss(problem, p, {0,-4},{0,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss(problem, p, {2,-4},{1,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss(problem, p, {-2,-4},{-1,-2},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    count += gen_straight_loss(problem, p, {0,-8},{0,-4},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss(problem, p, {4,-8},{2,-4},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    count += gen_straight_loss(problem, p, {-4,-8},{-2,-4},{0,0}, state, points, flags & OPTIMIZE_ALL, straight_w);
    
    //direct neighboars
    count += gen_dist_loss_fill(problem, p, {0,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    // count += gen_dist_loss(problem, p, {0,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    // count += gen_dist_loss(problem, p, {-1,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    // count += gen_dist_loss(problem, p, {1,0}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    //diagonal neighbors
    count += gen_dist_loss_fill(problem, p, {1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    // count += gen_dist_loss(problem, p, {-1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    // count += gen_dist_loss(problem, p, {1,1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-1}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    //far left
    count += gen_dist_loss_fill(problem, p, {0,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {1,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {-1,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    count += gen_dist_loss_fill(problem, p, {-2,-2}, state, points, unit, flags & OPTIMIZE_ALL, nullptr, dist_w);
    
    if (flags & LOSS_ON_SURF)
        gen_surfloss(p, problem, state, points_in, points, locs, surf_w);

    return count;
}

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

    cv::Mat_<uint8_t> state(points_in.size(), 0);
    cv::Mat_<cv::Vec3d> points(points_in.size(), {-1,-1,-1});
    cv::Mat_<cv::Vec2d> locs(points_in.size(), {-1,-1});
    cv::Mat_<float> winding(points_in.size(), NAN);
    
    cv::Mat_<uint8_t> fail_code(points_in.size(), 0);
    
    cv::Rect bbox(294,34,200,364);
    
    float step = 20;
    
    cv::Rect first_col = {bbox.x,bbox.y,2,bbox.height};
    points_in(first_col).copyTo(points(first_col));
    winding_in(first_col).copyTo(winding(first_col));
    
    std::cout << first_col.tl() << first_col.br() << std::endl;
    
    for(int j=first_col.y;j<first_col.br().y;j++)
        for(int i=first_col.x;i<first_col.br().x;i++) {
            locs(j,i) = {j,i};
            state(j,i) = STATE_LOC_VALID;
        }
    
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 10000;
    
    std::vector<cv::Vec2d> neighs = {{0,-1},{-1,-1},{1,-1}};
    
    for(int i=bbox.x+2;i<bbox.br().x;i++) {
        std::cout << "proc row " << i << std::endl;
#pragma omp parallel for
        for(int j=bbox.y;j<bbox.br().y;j++) {
            cv::Vec2i p = {j,i};
            
            locs(p) = locs(j,i-1)+cv::Vec2d(0,step);
            points(p) = points(j,i-1)+cv::Vec3d(0.1,0.1,0.1);
            winding(p) = winding(j,i-1);
            
            for(auto n :neighs) {
                cv::Vec2d cand = cv::Vec2d(p)+n+cv::Vec2d(0,step);
                if (loc_valid(points_in,cand)) {
                    locs(p) = cand;
                    points(p) = at_int(points_in, {cand[1],cand[0]});
                    break;
                }
            }
            
            
            state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
            
            {
                ceres::Problem problem;
                // create_centered_losses(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
                create_centered_losses(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
                
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                // std::cout << summary.BriefReport() << "\n";
                // std::cout << sqrt(summary.final_cost/summary.num_residuals) << std::endl;
            }
            
            cv::Vec2f loc = {locs(p)[1], locs(p)[0]};
            
            float res;
            for(int r=0;r<10;r++) {
                
                //FIXME pointot alt which includes winding number! 
                //test random inits for pointto!
                res = pointTo(loc, points_in, points(p), 2.0, 1000, surf->_scale[0]);
                loc = {loc[1], loc[0]};
                
                if (res < 0)
                    continue;
            
                if (res > 100)
                    continue;
                
                if (abs(winding_in(loc[0],loc[1]) - winding(p)) < 0.3)
                    break;
            }
            
            if (res < 0) {
                fail_code(p) = 1;
                continue;
            }
            
            if (res >= 10) {
                fail_code(p) = 2;
                continue;
            }
            
            if (abs(winding_in(loc[0],loc[1]) - winding(p)) >= 0.3) {
                fail_code(p) = 3;
                continue;
            }
            
            locs(p) = loc;
            
            // std::cout << res << std::endl;
            
            {
                ceres::Problem problem;
                // create_centered_losses(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
                create_centered_losses(problem, p, state, points_in, points, locs, step, LOSS_ON_SURF);
                
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                // std::cout << summary.BriefReport() << "\n";
                // std::cout << sqrt(summary.final_cost/summary.num_residuals) << std::endl;
            }
            
            if (loc_valid(points,locs(p)))
                winding(p) = winding_in(loc[0],loc[1]);
        }
    }
    
    std::vector<cv::Mat> chs;
    cv::split(points, chs);
    
    cv::imwrite("newx.tif",chs[0]);
    cv::imwrite("fail.tif",fail_code);
    cv::imwrite("winding_out.tif",winding);
    
    QuadSurface *surf_full = new QuadSurface(points(bbox), surf->_scale);
    
    fs::path tgt_dir = "/home/hendrik/data/ml_datasets/vesuvius/manual_wget/dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/";
    std::string name_prefix = "testing_fill_";
    std::string uuid = name_prefix + time_str();
    fs::path seg_dir = tgt_dir / uuid;
    std::cout << "saving " << seg_dir << std::endl;
    surf_full->save(seg_dir, uuid);
    
    return EXIT_SUCCESS;
}
