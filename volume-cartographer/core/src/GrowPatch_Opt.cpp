// GrowPatch_Opt.cpp - Loss generation, optimization, and inpaint functions
// Split from GrowPatch.cpp to reduce per-TU compile time (Ceres autodiff instantiation)

#include "vc/tracer/GrowPatch_Internal.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/tracer/CostFunctions.hpp"
#include "vc/tracer/Tracer.hpp"

#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

// Out-of-line special members for TraceData (unique_ptr<incomplete type> in header)
TraceData::TraceData(const std::vector<DirectionField> &direction_fields) : direction_fields(direction_fields) {}
TraceData::~TraceData() = default;
TraceData::TraceData(TraceData&&) noexcept = default;

// Out-of-line: uses TracerParams struct instead of JSON
#include "vc/tracer/TracerParams.hpp"

void LossSettings::applyWeights(const TracerParams& params) {
    auto maybe_set = [&](float val, LossType type) {
        if (val >= 0.0f) w[type] = val;
    };
    maybe_set(params.snap_weight, LossType::SNAP);
    maybe_set(params.normal_weight, LossType::NORMAL);
    maybe_set(params.normal3dline_weight, LossType::NORMAL3DLINE);
    maybe_set(params.straight_weight_loss, LossType::STRAIGHT);
    maybe_set(params.dist_weight, LossType::DIST);
    maybe_set(params.direction_weight, LossType::DIRECTION);
    maybe_set(params.sdir_weight, LossType::SDIR);
    maybe_set(params.correction_weight, LossType::CORRECTION);
    maybe_set(params.reference_ray_weight, LossType::REFERENCE_RAY);
}

class ReferenceClearanceCost {
public:
    ReferenceClearanceCost(const cv::Vec3d& target,
                           double min_clearance,
                           double weight)
        : target_(target),
          min_clearance_(min_clearance),
          weight_(weight) {}

    bool operator()(const double* candidate, double* residual) const {
        if (weight_ <= 0.0 || min_clearance_ <= 0.0) [[unlikely]] {
            residual[0] = 0.0;
            return true;
        }

        const cv::Vec3d diff{candidate[0] - target_[0],
                             candidate[1] - target_[1],
                             candidate[2] - target_[2]};
        const double dist = cv::norm(diff);
        if (dist >= min_clearance_) {
            residual[0] = 0.0;
        } else {
            residual[0] = weight_ * (min_clearance_ - dist);
        }
        return true;
    }

private:
    cv::Vec3d target_;
    double min_clearance_;
    double weight_;
};

class ReferenceRayOcclusionCost {
public:
    ReferenceRayOcclusionCost(Chunked3d<uint8_t, passTroughComputor>* volume,
                              const cv::Vec3d& target,
                              double threshold,
                              double weight,
                              double step,
                              double max_distance) :
        volume_(volume),
        target_(target),
        threshold_(threshold),
        weight_(weight),
        step_(step > 0.0 ? step : 1.0),
        max_distance_(max_distance)
    {}

    bool operator()(const double* candidate, double* residual) const {
        if (!volume_ || weight_ <= 0.0) [[unlikely]] {
            residual[0] = 0.0;
            return true;
        }

        const cv::Vec3d start{candidate[0], candidate[1], candidate[2]};
        const double distance = cv::norm(target_ - start);
        if (distance <= 1e-6) {
            residual[0] = 0.0;
            return true;
        }

        if (max_distance_ > 0.0 && distance > max_distance_) {
            residual[0] = 0.0;
            return true;
        }

        const int steps = std::max(1, static_cast<int>(std::ceil(distance / step_)));
        const cv::Vec3d delta = (target_ - start) / static_cast<double>(steps + 1);

        double max_value = std::numeric_limits<double>::lowest();
        bool hit_threshold = false;
        cv::Vec3d current = start;

        const double start_value = sample(start);
        int begin_step = 1;
        if (std::isfinite(start_value) && start_value >= threshold_) {
            bool exited_material = false;
            for (; begin_step <= steps; ++begin_step) {
                current += delta;
                const double value = sample(current);
                if (!std::isfinite(value)) {
                    continue;
                }
                if (value < threshold_) {
                    exited_material = true;
                    ++begin_step;  // start checking one step beyond the exit.
                    break;
                }
            }
            if (!exited_material) {
                residual[0] = 0.0;
                return true;
            }
        } else {
            current = start;
        }

        for (int i = begin_step; i <= steps; ++i) {
            current += delta;
            const double value = sample(current);
            if (!std::isfinite(value)) {
                continue;
            }

            max_value = std::max(max_value, value);
            if (value >= threshold_) {
                hit_threshold = true;
                break;
            }
        }

        if (!hit_threshold) {
            residual[0] = 0.0;
            return true;
        }

        if (!std::isfinite(max_value) || max_value < 0.0) {
            max_value = 0.0;
        }

        const double diff = std::max(0.0, threshold_ - max_value);
        residual[0] = weight_ * diff;
        return true;
    }

private:
    double sample(const cv::Vec3d& xyz) const {
        if (!interp_) {
            interp_ = std::make_unique<CachedChunked3dInterpolator<uint8_t, passTroughComputor>>(*volume_);
        }
        double value = 0.0;
        interp_->Evaluate(xyz[2], xyz[1], xyz[0], &value);
        return value;
    }

    Chunked3d<uint8_t, passTroughComputor>* volume_;
    cv::Vec3d target_;
    double threshold_;
    double weight_;
    double step_;
    double max_distance_;
    mutable std::unique_ptr<CachedChunked3dInterpolator<uint8_t, passTroughComputor>> interp_;
};

// TraceParameters, LossType, LossSettings defined in GrowPatch_Internal.hpp

// Forward declarations for loss helpers (defined further down in this file)
static bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status) noexcept;
static int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set) noexcept;

//gen straigt loss given point and 3 offsets
static int gen_straight_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2,
    const cv::Vec2i &o3, TraceParameters &params, const LossSettings &settings)
{
    if (!coord_valid(params.state(p+o1))) [[unlikely]]
        return 0;
    if (!coord_valid(params.state(p+o2))) [[unlikely]]
        return 0;
    if (!coord_valid(params.state(p+o3))) [[unlikely]]
        return 0;

    problem.AddResidualBlock(StraightLoss::Create(settings(LossType::STRAIGHT, p)), nullptr, &params.dpoints(p+o1)[0], &params.dpoints(p+o2)[0], &params.dpoints(p+o3)[0]);

    return 1;
}

static int gen_dist_loss(ceres::Problem &problem, const cv::Vec2i &p, const cv::Vec2i &off, TraceParameters &params,
    const LossSettings &settings)
{
    // Add a loss saying that dpoints(p) and dpoints(p+off) should themselves be distance |off| apart
    // Here dpoints is a 2D grid mapping surface-space points to 3D volume space
    // So this says that distances should be preserved from volume to surface

    if (!coord_valid(params.state(p))) [[unlikely]]
        return 0;
    if (!coord_valid(params.state(p+off))) [[unlikely]]
        return 0;

    if (params.dpoints(p)[0] == -1) [[unlikely]]
        throw std::runtime_error("invalid loc passed as valid!");

    problem.AddResidualBlock(DistLoss::Create(params.unit*cv::norm(off),settings(LossType::DIST, p)), nullptr, &params.dpoints(p)[0], &params.dpoints(p+off)[0]);

    return 1;
}

static int gen_reference_ray_loss(ceres::Problem &problem, const cv::Vec2i &p,
                                  TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!settings.reference_raycast_enabled())
        return 0;
    if (!trace_data.raw_volume)
        return 0;
    if (!(params.state(p) & STATE_LOC_VALID))
        return 0;

    const float w = settings(LossType::REFERENCE_RAY, p);
    if (w <= 0.0f)
        return 0;

    const cv::Vec3d& candidate = params.dpoints(p);
    if (candidate[0] == -1.0 && candidate[1] == -1.0 && candidate[2] == -1.0)
        return 0;

    cv::Vec3f ptr = settings.reference_raycast.surface->pointer();
    const cv::Vec3f candidate_f(static_cast<float>(candidate[0]),
                                static_cast<float>(candidate[1]),
                                static_cast<float>(candidate[2]));

    float dist = settings.reference_raycast.surface->pointTo(
        ptr,
        candidate_f,
        std::numeric_limits<float>::max(),
        1000);
    if (dist < 0.0f)
        return 0;
    if (settings.reference_raycast.max_distance > 0.0 && dist > settings.reference_raycast.max_distance)
        return 0;

    const cv::Vec3f nearest = settings.reference_raycast.surface->coord(ptr);
    const cv::Vec3d target{nearest[0], nearest[1], nearest[2]};

    {
        auto* functor = new ReferenceRayOcclusionCost(trace_data.raw_volume,
                                                      target,
                                                      settings.reference_raycast.voxel_threshold,
                                                      static_cast<double>(w),
                                                      settings.reference_raycast.sample_step,
                                                      settings.reference_raycast.max_distance);

        auto* cost = new ceres::NumericDiffCostFunction<ReferenceRayOcclusionCost, ceres::CENTRAL, 1, 3>(functor);
        problem.AddResidualBlock(cost, nullptr, &params.dpoints(p)[0]);
    }

    if (settings.reference_raycast.clearance_weight > 0.0 &&
        settings.reference_raycast.min_clearance > 0.0) {
        auto* clearance_functor = new ReferenceClearanceCost(target,
                                                             settings.reference_raycast.min_clearance,
                                                             settings.reference_raycast.clearance_weight * static_cast<double>(w));
        auto* clearance_cost = new ceres::NumericDiffCostFunction<ReferenceClearanceCost, ceres::CENTRAL, 1, 3>(clearance_functor);
        problem.AddResidualBlock(clearance_cost, nullptr, &params.dpoints(p)[0]);
    }

    return 1;
}

static int conditional_reference_ray_loss(int bit,
                                          const cv::Vec2i &p,
                                          cv::Mat_<uint16_t> &loss_status,
                                          ceres::Problem &problem,
                                          TraceParameters &params,
                                          const TraceData &trace_data,
                                          const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, {0, 0}, loss_status)) {
        set = set_loss_mask(bit, p, {0, 0}, loss_status, gen_reference_ray_loss(problem, p, params, trace_data, settings));
    }
    return set;
}

// -------------------------
// helpers used by conditionals (must be before they’re used)
// -------------------------
[[gnu::always_inline]] static inline cv::Vec2i lower_p(const cv::Vec2i &point, const cv::Vec2i &offset) noexcept
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

// Order two points along the axis implied by their grid relation.
// - For horizontal edges (same row), order by column.
// - For vertical edges (same col), order by row.
// Falls back to lexicographic (row,col) if neither axis matches.
[[gnu::always_inline]] static inline std::pair<cv::Vec2i, cv::Vec2i> order_p(const cv::Vec2i& p, const cv::Vec2i& q) noexcept
{
    if (p[0] == q[0]) { // same row => horizontal edge
        return (p[1] <= q[1]) ? std::make_pair(p, q) : std::make_pair(q, p);
    }
    if (p[1] == q[1]) { // same col => vertical edge
        return (p[0] <= q[0]) ? std::make_pair(p, q) : std::make_pair(q, p);
    }
    // fallback
    return (p[0] < q[0] || (p[0] == q[0] && p[1] <= q[1])) ? std::make_pair(p, q) : std::make_pair(q, p);
}

[[gnu::always_inline]] static inline bool loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status) noexcept
{
    return loss_status(lower_p(p, off)) & (1 << bit);
}

[[gnu::always_inline]] static inline int set_loss_mask(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status, int set) noexcept
{
    if (set)
        loss_status(lower_p(p, off)) |= (1 << bit);
    return set;
}

// -------------------------
// Symmetric Dirichlet losses (definitions)
// -------------------------
static int gen_sdirichlet_loss(ceres::Problem &problem,
                               const cv::Vec2i &p,
                               TraceParameters &params,
                               const LossSettings &settings,
                               double sdir_eps_abs,
                               double sdir_eps_rel)
{
    // Need p, p+u, p+v inside the image; treat (p) as the lower-left of a cell
    const int rows = params.state.rows;
    const int cols = params.state.cols;
    if (p[0] < 0 || p[1] < 0 || p[0] >= rows - 1 || p[1] >= cols - 1) {
        return 0;
    }

    const cv::Vec2i pu = p + cv::Vec2i(0, 1);  // (i, j+1)  u-direction
    const cv::Vec2i pv = p + cv::Vec2i(1, 0);  // (i+1, j)  v-direction

    // All three parameter blocks must be present/valid
    if (!coord_valid(params.state(p)) ||
        !coord_valid(params.state(pu)) ||
        !coord_valid(params.state(pv))) [[unlikely]] {
        return 0;
    }

    const float w = settings(LossType::SDIR, p);
    if (w <= 0.0f) {
        return 0;
    }

    ceres::LossFunction* robust = nullptr;
    robust = new ceres::CauchyLoss(1.0); // cauchy scale = 1.0

    problem.AddResidualBlock(
        SymmetricDirichletLoss::Create(/*unit*/ params.unit,
                                       /*w       */ w,
                                       /*eps_abs */ sdir_eps_abs,
                                       /*eps_rel */ sdir_eps_rel),
        /*loss*/ robust, // Cauchy Loss
        &params.dpoints(p)[0],
        &params.dpoints(pu)[0],
        &params.dpoints(pv)[0]);

    return 1;
}

static int conditional_sdirichlet_loss(int bit,
                                       const cv::Vec2i &p,
                                       cv::Mat_<uint16_t> &loss_status,
                                       ceres::Problem &problem,
                                       TraceParameters &params,
                                       const LossSettings &settings,
                                       double sdir_eps_abs,
                                       double sdir_eps_rel)
{
    int set = 0;
    // One SD term per cell (keyed at p itself)
    if (!loss_mask(bit, p, {0, 0}, loss_status)) {
        set = set_loss_mask(bit, p, {0, 0}, loss_status,
                            gen_sdirichlet_loss(problem, p, params, settings, sdir_eps_abs, sdir_eps_rel));
    }
    return set;
}

static int conditional_dist_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &off, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, TraceParameters &params, const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_dist_loss(problem, p, off, params, settings));
    return set;
};

static int gen_3d_normal_line_loss(ceres::Problem &problem,
                                  const cv::Vec2i &p,
                                  const cv::Vec2i &off,
                                  TraceParameters &params,
                                  const TraceData &trace_data,
                                  const LossSettings &settings)
{
    if (!trace_data.normal3d_field) return 0;

    const float w = settings(LossType::NORMAL3DLINE, p);
    if (w <= 0.0f) return 0;

    // We enforce the 90° constraint on the directed segment p_base -> p_off.
    // For consistent disambiguation of the directed normal field, provide a third point:
    // the next quad corner in clockwise direction when walking from base towards off.
    // This third point is treated as non-differentiable inside the loss.
    const cv::Vec2i& base = p;
    const cv::Vec2i off_p = p + off;

    // Determine clockwise neighbor in (row,col) grid coordinates.
    // Mapping assumes u = +col (right), v = +row (down).
    // Clockwise around a cell: right -> down -> left -> up.
    cv::Vec2i cw_off;
    if (off[0] == 0 && off[1] == 1) {          // right
        cw_off = cv::Vec2i(1, 0);              // down
    } else if (off[0] == 1 && off[1] == 0) {   // down
        cw_off = cv::Vec2i(0, -1);             // left
    } else if (off[0] == 0 && off[1] == -1) {  // left
        cw_off = cv::Vec2i(-1, 0);             // up
    } else if (off[0] == -1 && off[1] == 0) {  // up
        cw_off = cv::Vec2i(0, 1);              // right
    } else {
        // Only defined for direct 4-neighborhood edges.
        return 0;
    }
    const cv::Vec2i cw_p = base + cw_off;

    if (!coord_valid(params.state(base)) || !coord_valid(params.state(off_p)) || !coord_valid(params.state(cw_p))) [[unlikely]] {
        return 0;
    }

    problem.AddResidualBlock(
        Normal3DLineLoss::Create(*trace_data.normal3d_field, trace_data.normal3d_fit_quality.get(), w),
        nullptr,
        &params.dpoints(base)[0],
        &params.dpoints(off_p)[0],
        &params.dpoints(cw_p)[0]);

    return 1;
}

static int conditional_3d_normal_line_loss(int bit,
                                          const cv::Vec2i &p,
                                          const cv::Vec2i &off,
                                          cv::Mat_<uint16_t> &loss_status,
                                          ceres::Problem &problem,
                                          TraceParameters &params,
                                          const TraceData &trace_data,
                                          const LossSettings &settings)
{
    if (!trace_data.normal3d_field) return 0;
    int set = 0;
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_3d_normal_line_loss(problem, p, off, params, trace_data, settings));
    return set;
}

static int conditional_straight_loss(int bit, const cv::Vec2i &p, const cv::Vec2i &o1, const cv::Vec2i &o2, const cv::Vec2i &o3,
    cv::Mat_<uint16_t> &loss_status, ceres::Problem &problem, TraceParameters &params, const LossSettings &settings)
{
    int set = 0;
    if (!loss_mask(bit, p, o2, loss_status))
        set += set_loss_mask(bit, p, o2, loss_status, gen_straight_loss(problem, p, o1, o2, o3, params, settings));
    return set;
};

static int gen_normal_loss(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!trace_data.ngv) [[unlikely]] return 0;

    cv::Vec2i p_br = p + cv::Vec2i(1,1);
    if (!coord_valid(params.state(p)) || !coord_valid(params.state(p[0], p_br[1])) || !coord_valid(params.state(p_br[0], p[1])) || !coord_valid(params.state(p_br))) [[unlikely]] {
        return 0;
    }

    cv::Vec2i p_tr = {p[0], p[1] + 1};
    cv::Vec2i p_bl = {p[0] + 1, p[1]};

    // Points for the quad: A, B1, B2, C
    double* pA = &params.dpoints(p)[0];
    double* pB1 = &params.dpoints(p_tr)[0];
    double* pB2 = &params.dpoints(p_bl)[0];
    double* pC = &params.dpoints(p_br)[0];
    
    int count = 0;
    // int i = 1;
    for (int i = 0; i < 3; ++i) { // For each plane
        // bool direction_aware = (i == 0); // XY plane

        bool direction_aware = false; // this is not that simple ...
        // Loss with p as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pA, pB1, pB2, pC);
        // Loss with p_br as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pC, pB2, pB1, pA);
        // Loss with p_tr as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pB1, pC, pA, pB2);
        // Loss with p_bl as base point A
        problem.AddResidualBlock(NormalConstraintPlane::Create(*trace_data.ngv, i, settings(LossType::NORMAL, p), settings(LossType::SNAP, p), trace_data.normal3d_fit_quality.get(), direction_aware, settings.z_min, settings.z_max), nullptr, pB2, pA, pC, pB1);
        count += 4;
    }

    //FIXME make params constant if not optimize-all is set

    return count;
}

static int conditional_normal_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, TraceParameters &params, const TraceData &trace_data, const LossSettings &settings)
{
    if (!trace_data.ngv) return 0;
    int set = 0;
    if (!loss_mask(bit, p, {0,0}, loss_status))
        set = set_loss_mask(bit, p, {0,0}, loss_status, gen_normal_loss(problem, p, params, trace_data, settings));
    return set;
};

// static void freeze_inner_params(ceres::Problem &problem, int edge_dist, const cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out,
//     cv::Mat_<cv::Vec2d> &loc, cv::Mat_<uint16_t> &loss_status, int inner_flags)
// {
//     cv::Mat_<float> dist(state.size());
//
//     edge_dist = std::min(edge_dist,254);
//
//
//     cv::Mat_<uint8_t> masked;
//     bitwise_and(state, cv::Scalar(inner_flags), masked);
//
//
//     cv::distanceTransform(masked, dist, cv::DIST_L1, cv::DIST_MASK_3);
//
//     for(int j=0;j<dist.rows;j++)
//         for(int i=0;i<dist.cols;i++) {
//             if (dist(j,i) >= edge_dist && !loss_mask(7, {j,i}, {0,0}, loss_status)) {
//                 if (problem.HasParameterBlock(&out(j,i)[0]))
//                     problem.SetParameterBlockConstant(&out(j,i)[0]);
//                 if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
//                     problem.SetParameterBlockConstant(&loc(j,i)[0]);
//                 // set_loss_mask(7, {j,i}, {0,0}, loss_status, 1);
//             }
//             if (dist(j,i) >= edge_dist+2 && !loss_mask(8, {j,i}, {0,0}, loss_status)) {
//                 if (problem.HasParameterBlock(&out(j,i)[0]))
//                     problem.RemoveParameterBlock(&out(j,i)[0]);
//                 if (!loc.empty() && problem.HasParameterBlock(&loc(j,i)[0]))
//                     problem.RemoveParameterBlock(&loc(j,i)[0]);
//                 // set_loss_mask(8, {j,i}, {0,0}, loss_status, 1);
//             }
//         }
// }


// struct DSReader
// {
//     z5::Dataset *ds;
//     float scale;
//     ChunkCache *cache;
// };



int gen_direction_loss(ceres::Problem &problem,
    const cv::Vec2i &p,
    const int off_dist,
    cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc,
    std::vector<DirectionField> const &direction_fields,
    const LossSettings& settings)
{
    // Add losses saying that the local basis vectors of the patch at loc(p) should match those of the given fields

    if (!loc_valid(state(p)))
        return 0;

    cv::Vec2i const p_off_horz{p[0], p[1] + off_dist};
    cv::Vec2i const p_off_vert{p[0] + off_dist, p[1]};

    const float baseWeight = settings(LossType::DIRECTION, p);

    int count = 0;
    for (const auto &field: direction_fields) {
        const float totalWeight = baseWeight * field.weight;
        if (totalWeight <= 0.0f) {
            continue;
        }
        if (field.direction == "horizontal") {
            if (!loc_valid(state(p_off_horz)))
                continue;
            problem.AddResidualBlock(FiberDirectionLoss::Create(*field.field_ptr, field.weight_ptr.get(), totalWeight), nullptr, &loc(p)[0], &loc(p_off_horz)[0]);
        } else if (field.direction == "vertical") {
            if (!loc_valid(state(p_off_vert)))
                continue;
            problem.AddResidualBlock(FiberDirectionLoss::Create(*field.field_ptr, field.weight_ptr.get(), totalWeight), nullptr, &loc(p)[0], &loc(p_off_vert)[0]);
        } else if (field.direction == "normal") {
            if (!loc_valid(state(p_off_horz)) || !loc_valid(state(p_off_vert)))
                continue;
            problem.AddResidualBlock(NormalDirectionLoss::Create(*field.field_ptr, field.weight_ptr.get(), totalWeight), nullptr, &loc(p)[0], &loc(p_off_horz)[0], &loc(p_off_vert)[0]);
        } else {
            assert(false);
        }
        ++count;
    }

    return count;
}

//create all valid losses for this point
// Forward declarations
static int gen_corr_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, TraceData& trace_data, const LossSettings &settings);
static int conditional_corr_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
                                 ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &loc, TraceData& trace_data, const LossSettings &settings);

int add_losses(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params,
    const TraceData &trace_data, const LossSettings &settings, int flags)
{
    //generate losses for point p
    int count = 0;

    if (p[0] < 2 || p[1] < 2 || p[1] >= params.state.cols-2 || p[0] >= params.state.rows-2)
        throw std::runtime_error("point too close to problem border!");

    if (flags & LOSS_STRAIGHT) {
        //horizontal
        count += gen_straight_loss(problem, p, {0,-2},{0,-1},{0,0}, params, settings);
        count += gen_straight_loss(problem, p, {0,-1},{0,0},{0,1}, params, settings);
        count += gen_straight_loss(problem, p, {0,0},{0,1},{0,2}, params, settings);

        //vertical
        count += gen_straight_loss(problem, p, {-2,0},{-1,0},{0,0}, params, settings);
        count += gen_straight_loss(problem, p, {-1,0},{0,0},{1,0}, params, settings);
        count += gen_straight_loss(problem, p, {0,0},{1,0},{2,0}, params, settings);

        //diag1
        count += gen_straight_loss(problem, p, {-2,-2},{-1,-1},{0,0}, params, settings);
        count += gen_straight_loss(problem, p, {-1,-1},{0,0},{1,1}, params, settings);
        count += gen_straight_loss(problem, p, {0,0},{1,1},{2,2}, params, settings);

        //diag2
        count += gen_straight_loss(problem, p, {-2,2},{-1,1},{0,0}, params, settings);
        count += gen_straight_loss(problem, p, {-1,1},{0,0},{1,-1}, params, settings);
        count += gen_straight_loss(problem, p, {0,0},{1,-1},{2,-2}, params, settings);
    }

    if (flags & LOSS_DIST) {
        //direct neighboars
        count += gen_dist_loss(problem, p, {0,-1}, params, settings);
        count += gen_dist_loss(problem, p, {0,1}, params, settings);
        count += gen_dist_loss(problem, p, {-1,0}, params, settings);
        count += gen_dist_loss(problem, p, {1,0}, params, settings);

        //diagonal neighbors
        count += gen_dist_loss(problem, p, {1,-1}, params, settings);
        count += gen_dist_loss(problem, p, {-1,1}, params, settings);
        count += gen_dist_loss(problem, p, {1,1}, params, settings);
        count += gen_dist_loss(problem, p, {-1,-1}, params, settings);
    }

    if (flags & LOSS_NORMALSNAP) {
        //gridstore normals
        count += gen_normal_loss(problem, p                   , params, trace_data, settings);
        count += gen_normal_loss(problem, p + cv::Vec2i(-1,-1), params, trace_data, settings);
        count += gen_normal_loss(problem, p + cv::Vec2i( 0,-1), params, trace_data, settings);
        count += gen_normal_loss(problem, p + cv::Vec2i(-1, 0), params, trace_data, settings);
    }

    if (flags & LOSS_SDIR) {
        //symmetric dirichlet
        count += gen_sdirichlet_loss(problem, p, params, settings, /*sdir_eps_abs=*/1e-8, /*sdir_eps_rel=*/1e-2);
        count += gen_sdirichlet_loss(problem, p + cv::Vec2i(-1, 0), params, settings, 1e-8, 1e-2);
        count += gen_sdirichlet_loss(problem, p + cv::Vec2i( 0,-1), params, settings, 1e-8, 1e-2);
    }

    if (flags & LOSS_3DNORMALLINE) {
        // one per edge (use +u and +v edges only)
        count += gen_3d_normal_line_loss(problem, p, cv::Vec2i(0, 1), params, trace_data, settings);
        count += gen_3d_normal_line_loss(problem, p, cv::Vec2i(1, 0), params, trace_data, settings);
    }

    count += gen_reference_ray_loss(problem, p, params, trace_data, settings);

    return count;
}

static int conditional_direction_loss(int bit,
    const cv::Vec2i &p,
    const int u_off,
    cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem,
    cv::Mat_<uint8_t> &state,
    cv::Mat_<cv::Vec3d> &loc,
    const LossSettings& settings,
    std::vector<DirectionField> const &direction_fields)
{
    if (!direction_fields.size())
        return 0;

    int set = 0;
    cv::Vec2i const off{0, u_off};
    if (!loss_mask(bit, p, off, loss_status))
        set = set_loss_mask(bit, p, off, loss_status, gen_direction_loss(problem, p, u_off, state, loc, direction_fields, settings));
    return set;
};

//create only missing losses so we can optimize the whole problem
static int gen_corr_loss(ceres::Problem &problem, const cv::Vec2i &p, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &dpoints, TraceData& trace_data, const LossSettings &settings)
{
    const float weight = settings(LossType::CORRECTION, p);
    if (!trace_data.point_correction.isValid() || weight <= 0.0f) {
        return 0;
    }

    const auto& pc = trace_data.point_correction;

    const auto& all_grid_locs = pc.all_grid_locs();
    if (all_grid_locs.empty()) {
        return 0;
    }

    cv::Vec2i p_br = p + cv::Vec2i(1,1);
    if (!coord_valid(state(p)) || !coord_valid(state(p[0], p_br[1])) || !coord_valid(state(p_br[0], p[1])) || !coord_valid(state(p_br))) {
        return 0;
    }

    const auto& all_tgts = pc.all_tgts();

    std::vector<cv::Vec3f> filtered_tgts;
    std::vector<cv::Vec2f> filtered_grid_locs;
    filtered_tgts.reserve(all_tgts.size());
    filtered_grid_locs.reserve(all_tgts.size());
    cv::Vec2i quad_loc_int = {p[1], p[0]};

    for (size_t i = 0; i < all_tgts.size(); ++i) {
        const auto& grid_loc = all_grid_locs[i];
        float dx = grid_loc[0] - quad_loc_int[0];
        float dy = grid_loc[1] - quad_loc_int[1];
        if (dx * dx + dy * dy <= 4.0 * 4.0) {
            filtered_tgts.push_back(all_tgts[i]);
            filtered_grid_locs.push_back(all_grid_locs[i]);
        }
    }

    if (filtered_tgts.empty()) {
        return 0;
    }

    auto points_correction_loss = new PointsCorrectionLoss(filtered_tgts, filtered_grid_locs, quad_loc_int, weight);
    auto cost_function = new ceres::DynamicAutoDiffCostFunction<PointsCorrectionLoss>(
        points_correction_loss
    );

    std::vector<double*> parameter_blocks;
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p)[0]);
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p + cv::Vec2i(0, 1))[0]);
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p + cv::Vec2i(1, 0))[0]);
    cost_function->AddParameterBlock(3);
    parameter_blocks.push_back(&dpoints(p + cv::Vec2i(1, 1))[0]);

    cost_function->SetNumResiduals(1);

    problem.AddResidualBlock(cost_function, nullptr, parameter_blocks);

    return 1;
}

static int conditional_corr_loss(int bit, const cv::Vec2i &p, cv::Mat_<uint16_t> &loss_status,
    ceres::Problem &problem, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &out, TraceData& trace_data, const LossSettings &settings)
{
    if (!trace_data.point_correction.isValid()) return 0;
    int set = 0;
    if (!loss_mask(bit, p, {0,0}, loss_status))
        set = set_loss_mask(bit, p, {0,0}, loss_status, gen_corr_loss(problem, p, state, out, trace_data, settings));
    return set;
};

// Compute local surface normal from neighboring 3D points using cross product of tangent vectors.
// Returns normalized normal vector, or zero vector if insufficient neighbors.
// If surface_normals is provided and has valid neighbors, orients the result to be consistent.
cv::Vec3d compute_surface_normal_at(
    const cv::Vec2i& p,
    const cv::Mat_<cv::Vec3d>& dpoints,
    const cv::Mat_<uchar>& state,
    const cv::Mat_<cv::Vec3d>* surface_normals)
{
    auto is_valid = [&](const cv::Vec2i& pt) {
        return pt[0] >= 0 && pt[0] < dpoints.rows &&
               pt[1] >= 0 && pt[1] < dpoints.cols &&
               (state(pt) & STATE_LOC_VALID);
    };

    // Try to get horizontal and vertical tangents
    cv::Vec3d tangent_h(0,0,0), tangent_v(0,0,0);
    bool has_h = false, has_v = false;

    // Horizontal tangent
    cv::Vec2i left = {p[0], p[1] - 1};
    cv::Vec2i right = {p[0], p[1] + 1};
    if (is_valid(left) && is_valid(right)) {
        tangent_h = dpoints(right) - dpoints(left);
        has_h = true;
    }

    // Vertical tangent
    cv::Vec2i up = {p[0] - 1, p[1]};
    cv::Vec2i down = {p[0] + 1, p[1]};
    if (is_valid(up) && is_valid(down)) {
        tangent_v = dpoints(down) - dpoints(up);
        has_v = true;
    }

    if (!has_h || !has_v) {
        return cv::Vec3d(0,0,0);
    }

    cv::Vec3d normal = tangent_h.cross(tangent_v);
    double len = cv::norm(normal);
    if (len < 1e-9) {
        return cv::Vec3d(0,0,0);
    }
    normal /= len;

    // Orient consistently with neighbors if surface_normals provided
    if (surface_normals) {
        static const cv::Vec2i neighbor_offsets[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (const auto& off : neighbor_offsets) {
            cv::Vec2i neighbor = p + off;
            if (is_valid(neighbor)) {
                const cv::Vec3d& neighbor_normal = (*surface_normals)(neighbor);
                if (cv::norm(neighbor_normal) > 0.5) {
                    // Flip if pointing opposite to neighbor
                    if (normal.dot(neighbor_normal) < 0) {
                        normal = -normal;
                    }
                    break;  // Use first valid neighbor for orientation
                }
            }
        }
    }

    return normal;
}

// Compute and store oriented surface normal for a point
// Uses existing neighbor normals for consistent orientation
void update_surface_normal(
    const cv::Vec2i& p,
    const cv::Mat_<cv::Vec3d>& dpoints,
    const cv::Mat_<uchar>& state,
    cv::Mat_<cv::Vec3d>& surface_normals)
{
    cv::Vec3d normal = compute_surface_normal_at(p, dpoints, state, &surface_normals);
    if (cv::norm(normal) > 0.5) {
        surface_normals(p) = normal;
    }
}

int add_missing_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p,
    TraceParameters &params, TraceData& trace_data,
    const LossSettings &settings)
{
    //generate losses for point p
    int count = 0;

    //horizontal
    count += conditional_straight_loss(0, p, {0,-2},{0,-1},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {0,-1},{0,0},{0,1}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {0,0},{0,1},{0,2}, loss_status, problem, params, settings);

    //vertical
    count += conditional_straight_loss(1, p, {-2,0},{-1,0},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {-1,0},{0,0},{1,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {0,0},{1,0},{2,0}, loss_status, problem, params, settings);

    //diag1
    count += conditional_straight_loss(0, p, {-2,-2},{-1,-1},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {-1,-1},{0,0},{1,1}, loss_status, problem, params, settings);
    count += conditional_straight_loss(0, p, {0,0},{1,1},{2,2}, loss_status, problem, params, settings);

    //diag2
    count += conditional_straight_loss(1, p, {-2,2},{-1,1},{0,0}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {-1,1},{0,0},{1,-1}, loss_status, problem, params, settings);
    count += conditional_straight_loss(1, p, {0,0},{1,-1},{2,-2}, loss_status, problem, params, settings);

    //direct neighboars h
    count += conditional_dist_loss(2, p, {0,-1}, loss_status, problem, params, settings);
    count += conditional_dist_loss(2, p, {0,1}, loss_status, problem, params, settings);

    //direct neighbors v
    count += conditional_dist_loss(3, p, {-1,0}, loss_status, problem, params, settings);
    count += conditional_dist_loss(3, p, {1,0}, loss_status, problem, params, settings);

    //diagonal neighbors
    count += conditional_dist_loss(4, p, {1,-1}, loss_status, problem, params, settings);
    count += conditional_dist_loss(4, p, {-1,1}, loss_status, problem, params, settings);

    count += conditional_dist_loss(5, p, {1,1}, loss_status, problem, params, settings);
    count += conditional_dist_loss(5, p, {-1,-1}, loss_status, problem, params, settings);

    //symmetric dirichlet
    count += conditional_sdirichlet_loss(6, p,                    loss_status, problem, params, settings, /*sdir_eps_abs=*/1e-8, /*sdir_eps_rel=*/1e-2);
    count += conditional_sdirichlet_loss(6, p + cv::Vec2i(-1, 0), loss_status, problem, params, settings, 1e-8, 1e-2);
    count += conditional_sdirichlet_loss(6, p + cv::Vec2i( 0,-1), loss_status, problem, params, settings, 1e-8, 1e-2);

    //normal field
    count += conditional_direction_loss(9, p, 1, loss_status, problem, params.state, params.dpoints, settings, trace_data.direction_fields);
    count += conditional_direction_loss(9, p, -1, loss_status, problem, params.state, params.dpoints, settings, trace_data.direction_fields);

    //gridstore normals
    count += conditional_normal_loss(10, p                   , loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i(-1,-1), loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i( 0,-1), loss_status, problem, params, trace_data, settings);
    count += conditional_normal_loss(10, p + cv::Vec2i(-1, 0), loss_status, problem, params, trace_data, settings);

    // fitted 3D normals: edge-perpendicularity (once per edge)
    // Add for all edges touching p; the mask makes sure each undirected edge is only added once.
    count += conditional_3d_normal_line_loss(13, p, cv::Vec2i(0, 1),  loss_status, problem, params, trace_data, settings);
    count += conditional_3d_normal_line_loss(13, p, cv::Vec2i(0, -1), loss_status, problem, params, trace_data, settings);
    count += conditional_3d_normal_line_loss(14, p, cv::Vec2i(1, 0),  loss_status, problem, params, trace_data, settings);
    count += conditional_3d_normal_line_loss(14, p, cv::Vec2i(-1, 0), loss_status, problem, params, trace_data, settings);

    //snapping
    count += conditional_corr_loss(11, p,                    loss_status, problem, params.state, params.dpoints, trace_data, settings);
    count += conditional_corr_loss(11, p + cv::Vec2i(-1,-1), loss_status, problem, params.state, params.dpoints, trace_data, settings);
    count += conditional_corr_loss(11, p + cv::Vec2i( 0,-1), loss_status, problem, params.state, params.dpoints, trace_data, settings);
    count += conditional_corr_loss(11, p + cv::Vec2i(-1, 0), loss_status, problem, params.state, params.dpoints, trace_data, settings);

    count += conditional_reference_ray_loss(15, p, loss_status, problem, params, trace_data, settings);

    return count;
}


template <typename T>
void masked_blur(cv::Mat_<T>& img, const cv::Mat_<uchar>& mask) {
    cv::Mat_<T> fw_pass = img.clone();
    cv::Mat_<T> bw_pass = img.clone();

    // Initial forward pass (top-left neighbors)
    for (int y = 1; y < img.rows; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            if (mask(y, x) == 0) {
                T val = (fw_pass(y - 1, x) + fw_pass(y, x - 1) + fw_pass(y - 1, x - 1) + fw_pass(y - 1, x + 1)) * 0.25;
                fw_pass(y, x) = val;
            }
        }
    }

    // Initial backward pass (bottom-right neighbors)
    for (int y = img.rows - 2; y >= 0; --y) {
        for (int x = img.cols - 2; x >= 1; --x) {
            if (mask(y, x) == 0) {
                T val = (bw_pass(y + 1, x) + bw_pass(y, x + 1) + bw_pass(y + 1, x + 1) + bw_pass(y + 1, x - 1)) * 0.25;
                bw_pass(y, x) = val;
            }
        }
    }

    // Average initial passes
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (mask(y, x) == 0) {
                img(y, x) = (fw_pass(y, x) + bw_pass(y, x)) * 0.5;
            }
        }
    }

    // 4 additional passes
    for (int i = 0; i < 4; ++i) {
        // Forward pass
        for (int y = 1; y < img.rows; ++y) {
            for (int x = 1; x < img.cols - 1; ++x) {
                if (mask(y, x) == 0) {
                    T val = (img(y, x) + img(y - 1, x) + img(y, x - 1) + img(y - 1, x - 1) + img(y - 1, x + 1)) * 0.2;
                    img(y, x) = val;
                }
            }
        }
        // Backward pass
        for (int y = img.rows - 2; y >= 0; --y) {
            for (int x = img.cols - 2; x >= 1; --x) {
                if (mask(y, x) == 0) {
                    T val = (img(y, x) + img(y + 1, x) + img(y, x + 1) + img(y + 1, x + 1) + img(y + 1, x - 1)) * 0.2;
                    img(y, x) = val;
                }
            }
        }
    }
}

static void local_optimization_roi(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data, LossSettings &settings, int flags);

//optimize within a radius, setting edge points to constant
bool inpaint(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data)
{
    // check that a two pixel border is 1
    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            if (y < 2 || y >= roi.height - 2 || x < 2 || x >= roi.width - 2) {
                if (mask(y, x) == 0) {
                    return false;
                }
            }
        }
    }

    cv::Mat_<cv::Vec3d> dpoints_roi = params.dpoints(roi);
    masked_blur(dpoints_roi, mask);

    ceres::Problem problem;
    LossSettings settings;
    settings[DIST] = 1.0;
    settings[STRAIGHT] = 1.0;

    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            if (!mask(y, x)) {
                params.state(roi.y + y, roi.x + x) = STATE_LOC_VALID | STATE_COORD_VALID;
            }
        }
    }

    LossSettings base;
    base[NORMAL] = 10.0;
    LossSettings nosnap = base;
    nosnap[SNAP] = 0;
    local_optimization_roi(roi, mask, params, trace_data, nosnap, LOSS_DIST | LOSS_STRAIGHT);
    local_optimization_roi(roi, mask, params, trace_data, nosnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);

    LossSettings lowsnap = base;
    lowsnap[SNAP] = 0.01*base[SNAP];
    local_optimization_roi(roi, mask, params, trace_data, lowsnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);
    lowsnap[SNAP] = 0.1*base[SNAP];
    local_optimization_roi(roi, mask, params, trace_data, lowsnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);
    lowsnap[SNAP] = base[SNAP];
    local_optimization_roi(roi, mask, params, trace_data, lowsnap, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);
    LossSettings default_settings;
    local_optimization_roi(roi, mask, params, trace_data, default_settings, LOSS_DIST | LOSS_STRAIGHT | LOSS_NORMALSNAP);

    return true;
}


//optimize within a radius, setting edge points to constant
static void local_optimization_roi(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data, LossSettings &settings, int flags)
{
    ceres::Problem problem;

    for (int y = 2; y < roi.height - 2; ++y) {
        for (int x = 2; x < roi.width - 2; ++x) {
            // if (!mask(y, x)) {
                add_losses(problem, {roi.y + y, roi.x + x}, params, trace_data, settings, flags);
            // }
        }
    }

    for (int y = 0; y < roi.height; ++y) {
        for (int x = 0; x < roi.width; ++x) {
            if (mask(y, x) && problem.HasParameterBlock(&params.dpoints.at<cv::Vec3d>(roi.y + y, roi.x + x)[0])) {
                problem.SetParameterBlockConstant(&params.dpoints.at<cv::Vec3d>(roi.y + y, roi.x + x)[0]);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 10000;
    // options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // std::cout << "inpaint solve " << summary.BriefReport() << "\n";

    // cv::imwrite("opt_mask.tif", mask);
}

// LocalOptimizationConfig, AntiFlipbackConfig defined in GrowPatch_Internal.hpp

float local_optimization(int radius, const cv::Vec2i &p, TraceParameters &params,
    TraceData& trace_data, LossSettings &settings, bool quiet, bool parallel,
    const LocalOptimizationConfig* solver_config,
    const AntiFlipbackConfig* flipback_config)
{
    // This Ceres problem is parameterised by locs; residuals are progressively added as the patch grows enforcing that
    // all points in the patch are correct distance in 2D vs 3D space, not too high curvature, near surface prediction, etc.
    ceres::Problem problem;
    cv::Mat_<uint16_t> loss_status(params.state.size());

    int r_outer = radius+3;

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,params.dpoints.cols-1);ox++)
            loss_status(oy,ox) = 0;

    for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,params.dpoints.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) <= radius) {
                add_missing_losses(problem, loss_status, op, params, trace_data, settings);
            }
        }

    // Add anti-flipback loss if configured
    // This penalizes points that move too far in the inward (negative normal) direction
    if (flipback_config && flipback_config->anchors && flipback_config->surface_normals) {
        for(int oy=std::max(p[0]-radius,0);oy<=std::min(p[0]+radius,params.dpoints.rows-1);oy++)
            for(int ox=std::max(p[1]-radius,0);ox<=std::min(p[1]+radius,params.dpoints.cols-1);ox++) {
                cv::Vec2i op = {oy, ox};
                if (cv::norm(p-op) <= radius && (params.state(op) & STATE_LOC_VALID)) {
                    cv::Vec3d anchor = (*flipback_config->anchors)(op);
                    cv::Vec3d normal = (*flipback_config->surface_normals)(op);
                    // Only add loss if we have valid anchor and normal
                    if (cv::norm(normal) > 0.5 && anchor[0] >= 0) {
                        problem.AddResidualBlock(
                            AntiFlipbackLoss::Create(anchor, normal, flipback_config->threshold, flipback_config->weight),
                            nullptr,
                            &params.dpoints(op)[0]);
                    }
                }
            }
    }

    for(int oy=std::max(p[0]-r_outer,0);oy<=std::min(p[0]+r_outer,params.dpoints.rows-1);oy++)
        for(int ox=std::max(p[1]-r_outer,0);ox<=std::min(p[1]+r_outer,params.dpoints.cols-1);ox++) {
            cv::Vec2i op = {oy, ox};
            if (cv::norm(p-op) > radius && problem.HasParameterBlock(&params.dpoints(op)[0]))
                problem.SetParameterBlockConstant(&params.dpoints(op)[0]);
        }

    ceres::Solver::Options options;
    if (solver_config && solver_config->use_dense_qr) {
        options.linear_solver_type = ceres::DENSE_QR;
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = solver_config ? solver_config->max_iterations : 1000;

    // options.function_tolerance = 1e-4;
    // options.use_nonmonotonic_steps = true;
    // options.use_mixed_precision_solves = true;
    // options.max_num_refinement_iterations = 3;
    // options.use_inner_iterations = true;

    if (parallel)
        options.num_threads = omp_get_max_threads();

//    if (problem.NumParameterBlocks() > 1) {
//        options.use_inner_iterations = true;
//    }
// // NOTE currently CPU seems always faster (40x , AMD 5800H vs RTX 3080 mobile, even a 5090 would probably be slower?)
// #ifdef VC_USE_CUDA_SPARSE
//     // Check if Ceres was actually built with CUDA sparse support
//     if (g_use_cuda) {
//         if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
//             options.linear_solver_type = ceres::SPARSE_SCHUR;
//             // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//             options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
//
//             // if (options.linear_solver_type == ceres::SPARSE_SCHUR) {
//                 options.use_mixed_precision_solves = true;
//             // }
//         } else {
//             std::cerr << "Warning: use_cuda=true but Ceres was not built with CUDA sparse support. Falling back to CPU sparse." << "\n";
//         }
//     }
// #endif

    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);

    if (!quiet)
        std::cout << "local solve radius " << radius << " " << summary.BriefReport() << "\n";

    return sqrt(summary.final_cost/summary.num_residual_blocks);
}
