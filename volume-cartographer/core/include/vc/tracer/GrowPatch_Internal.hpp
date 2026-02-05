#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include <optional>
#include <limits>

#include "vc/core/util/SurfaceModeling.hpp"  // STATE_LOC_VALID etc.
#include "vc/core/types/ChunkedTensor.hpp"   // Chunked3d, Chunked3dVec3fFromUint8, etc. (needed for unique_ptr members)

// Forward declarations
namespace ceres { class Problem; }
namespace vc::core::util { class NormalGridVolume; }
class QuadSurface;
class VCCollection;
struct DirectionField;
struct NormalFitQualityWeightField;
struct TracerParams;

// ---------------------------------------------------------------------------
// Loss bit constants
// ---------------------------------------------------------------------------
inline constexpr int LOSS_STRAIGHT    = 1;
inline constexpr int LOSS_DIST        = 2;
inline constexpr int LOSS_NORMALSNAP  = 4;
inline constexpr int LOSS_SDIR        = 8;
inline constexpr int LOSS_3DNORMALLINE = 32;

// ---------------------------------------------------------------------------
// Small inline helpers
// ---------------------------------------------------------------------------
template <typename T>
[[gnu::always_inline]] static constexpr bool point_in_bounds(const cv::Mat_<T>& mat, const cv::Vec2i& p) noexcept
{
    return p[0] >= 0 && p[0] < mat.rows && p[1] >= 0 && p[1] < mat.cols;
}

[[gnu::always_inline]] static constexpr bool loc_valid(int state) noexcept
{
    return state & STATE_LOC_VALID;
}

[[gnu::always_inline]] static constexpr bool coord_valid(int state) noexcept
{
    return (state & STATE_COORD_VALID) || (state & STATE_LOC_VALID);
}

// ---------------------------------------------------------------------------
// PointCorrection
// ---------------------------------------------------------------------------
class PointCorrection {
public:
    struct CorrectionCollection {
        std::vector<cv::Vec3f> tgts_;
        std::vector<cv::Vec2f> grid_locs_;
        std::optional<cv::Vec2f> anchor2d_;  // If set, use this as the 2D grid anchor instead of searching from first point
    };

    PointCorrection() = default;
    PointCorrection(const VCCollection& corrections);  // defined in GrowPatch.cpp

    void init(const cv::Mat_<cv::Vec3f> &points);      // defined in GrowPatch.cpp

    bool isValid() const { return is_valid_; }

    const std::vector<CorrectionCollection>& collections() const { return collections_; }

    std::vector<cv::Vec3f> all_tgts() const {
        std::vector<cv::Vec3f> flat_tgts;
        for (const auto& collection : collections_) {
            flat_tgts.insert(flat_tgts.end(), collection.tgts_.begin(), collection.tgts_.end());
        }
        return flat_tgts;
    }

    std::vector<cv::Vec2f> all_grid_locs() const {
        std::vector<cv::Vec2f> flat_locs;
        for (const auto& collection : collections_) {
            flat_locs.insert(flat_locs.end(), collection.grid_locs_.begin(), collection.grid_locs_.end());
        }
        return flat_locs;
    }

private:
    bool is_valid_ = false;
    std::vector<CorrectionCollection> collections_;
};

// ---------------------------------------------------------------------------
// TraceData
// ---------------------------------------------------------------------------
struct TraceData {
    TraceData(const std::vector<DirectionField> &direction_fields);  // Out-of-line (GrowPatch_Opt.cpp)
    ~TraceData();  // Out-of-line destructor (GrowPatch_Opt.cpp) for unique_ptr<incomplete type>
    TraceData(TraceData&&) noexcept;

    PointCorrection point_correction;
    const vc::core::util::NormalGridVolume *ngv = nullptr;
    const std::vector<DirectionField> &direction_fields;

    // Optional fitted-3D normals direction-field (zarr root with x/<scale>,y/<scale>,z/<scale> datasets)
    std::unique_ptr<Chunked3dVec3fFromUint8> normal3d_field;
    std::unique_ptr<NormalFitQualityWeightField> normal3d_fit_quality;

    Chunked3d<uint8_t, passTroughComputor>* raw_volume = nullptr;
};

// ---------------------------------------------------------------------------
// TraceParameters
// ---------------------------------------------------------------------------
struct TraceParameters {
    cv::Mat_<uint8_t> state;
    cv::Mat_<cv::Vec3d> dpoints;
    float unit;
};

// ---------------------------------------------------------------------------
// LossType enum + LossSettings struct
// ---------------------------------------------------------------------------
enum LossType {
    DIST,
    STRAIGHT,
    DIRECTION,
    SNAP,
    NORMAL,
    NORMAL3DLINE,
    SDIR,
    CORRECTION,
    REFERENCE_RAY,
    COUNT
};

struct LossSettings {
    std::vector<float> w = std::vector<float>(LossType::COUNT, 0.0f);
    std::vector<cv::Mat_<float>> w_mats = std::vector<cv::Mat_<float>>(LossType::COUNT);

    LossSettings() {
        w[LossType::SNAP] = 0.1f;
        w[LossType::NORMAL] = 10.0f;
        w[LossType::NORMAL3DLINE] = 0.0f;
        w[LossType::STRAIGHT] = 0.2;
        w[LossType::DIST] = 1.0f;
        w[LossType::DIRECTION] = 1.0f;
        w[LossType::SDIR] = 1.0f;
        w[LossType::CORRECTION] = 1.0f;
        w[LossType::REFERENCE_RAY] = 0.5f;
    }

    struct ReferenceRaycastSettings {
        QuadSurface* surface = nullptr;
        double voxel_threshold = 1.0;
        double sample_step = 1.0;
        double max_distance = 250.0;
        double min_clearance = 4.0;
        double clearance_weight = 1.0;
    } reference_raycast;

    bool reference_raycast_enabled() const {
        return reference_raycast.surface && w[LossType::REFERENCE_RAY] > 0.0f;
    }

    void applyWeights(const TracerParams& params);

    float operator()(LossType type, const cv::Vec2i& p) const {
        if (!w_mats[type].empty()) {
            return w_mats[type](p);
        }
        return w[type];
    }

    float& operator[](LossType type) {
        return w[type];
    }

    int z_min = -1;
    int z_max = std::numeric_limits<int>::max();
    int y_min = -1;
    int y_max = std::numeric_limits<int>::max();
    int x_min = -1;
    int x_max = std::numeric_limits<int>::max();
    // Anti-flipback constraint settings
    float flipback_threshold = 5.0f;  // Allow up to this much inward movement (voxels) before penalty
    float flipback_weight = 1.0f;     // Weight of the anti-flipback loss (0 = disabled)
};

// ---------------------------------------------------------------------------
// LocalOptimizationConfig and AntiFlipbackConfig
// ---------------------------------------------------------------------------
struct LocalOptimizationConfig {
    int max_iterations = 1000;
    bool use_dense_qr = false;
};

// Configuration for anti-flipback constraint
// This prevents the surface from flipping back through itself during optimization
struct AntiFlipbackConfig {
    const cv::Mat_<cv::Vec3d>* anchors = nullptr;        // Positions before optimization
    const cv::Mat_<cv::Vec3d>* surface_normals = nullptr; // Consistently oriented surface normals
    double threshold = 5.0;   // Allow up to this much inward movement (voxels) before penalty kicks in
    double weight = 1.0;       // Weight of the anti-flipback loss when activated
};

// ---------------------------------------------------------------------------
// Function declarations for cross-file calls (defined in GrowPatch_Opt.cpp)
// ---------------------------------------------------------------------------
int add_losses(ceres::Problem &problem, const cv::Vec2i &p, TraceParameters &params,
    const TraceData &trace_data, const LossSettings &settings, int flags = LOSS_STRAIGHT | LOSS_DIST);

int add_missing_losses(ceres::Problem &problem, cv::Mat_<uint16_t> &loss_status, const cv::Vec2i &p,
    TraceParameters &params, TraceData& trace_data,
    const LossSettings &settings);

float local_optimization(int radius, const cv::Vec2i &p, TraceParameters &params,
    TraceData& trace_data, LossSettings &settings, bool quiet = false, bool parallel = false,
    const LocalOptimizationConfig* solver_config = nullptr,
    const AntiFlipbackConfig* flipback_config = nullptr);

bool inpaint(const cv::Rect &roi, const cv::Mat_<uchar> &mask, TraceParameters &params, const TraceData &trace_data);

void update_surface_normal(
    const cv::Vec2i& p,
    const cv::Mat_<cv::Vec3d>& dpoints,
    const cv::Mat_<uchar>& state,
    cv::Mat_<cv::Vec3d>& surface_normals);

cv::Vec3d compute_surface_normal_at(
    const cv::Vec2i& p,
    const cv::Mat_<cv::Vec3d>& dpoints,
    const cv::Mat_<uchar>& state,
    const cv::Mat_<cv::Vec3d>* surface_normals = nullptr);
