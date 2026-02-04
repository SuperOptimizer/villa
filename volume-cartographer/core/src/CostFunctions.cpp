// Explicit instantiations for Ceres AutoDiffCostFunction to reduce template bloat.
// The cost functor structs remain in the header (required for Ceres autodiff),
// but the Create() factory methods are defined here to consolidate instantiation.

#include "vc/core/util/CostFunctions.hpp"

// ============================================================================
// Non-template cost functor Create() implementations
// ============================================================================

ceres::CostFunction* DistLoss::Create(float d, float w) {
    return new ceres::AutoDiffCostFunction<DistLoss, 1, 3, 3>(new DistLoss(d, w));
}

ceres::CostFunction* DistLoss2D::Create(float d, float w) {
    if (d == 0)
        throw std::runtime_error("dist can't be zero for DistLoss2D");
    return new ceres::AutoDiffCostFunction<DistLoss2D, 1, 2, 2>(new DistLoss2D(d, w));
}

ceres::CostFunction* StraightLoss::Create(float w) {
    return new ceres::AutoDiffCostFunction<StraightLoss, 1, 3, 3, 3>(new StraightLoss(w));
}

ceres::CostFunction* StraightLoss2::Create(float w) {
    return new ceres::AutoDiffCostFunction<StraightLoss2, 3, 3, 3, 3>(new StraightLoss2(w));
}

ceres::CostFunction* StraightLoss2D::Create(float w) {
    return new ceres::AutoDiffCostFunction<StraightLoss2D, 1, 2, 2, 2>(new StraightLoss2D(w));
}

ceres::CostFunction* SurfaceLossD::Create(const cv::Mat_<cv::Vec3f>& m, float w) {
    return new ceres::AutoDiffCostFunction<SurfaceLossD, 3, 3, 2>(new SurfaceLossD(m, w));
}

ceres::CostFunction* LinChkDistLoss::Create(const cv::Vec2d& p, float w) {
    return new ceres::AutoDiffCostFunction<LinChkDistLoss, 2, 2>(new LinChkDistLoss(p, w));
}

ceres::CostFunction* ZCoordLoss::Create(float z, float w) {
    return new ceres::AutoDiffCostFunction<ZCoordLoss, 1, 3>(new ZCoordLoss(z, w));
}

ceres::CostFunction* FiberDirectionLoss::Create(Chunked3dVec3fFromUint8& fiber_dirs,
                                                 Chunked3dFloatFromUint8* maybe_weights,
                                                 float w) {
    return new ceres::AutoDiffCostFunction<FiberDirectionLoss, 1, 3, 3>(
        new FiberDirectionLoss(fiber_dirs, maybe_weights, w));
}

ceres::CostFunction* NormalDirectionLoss::Create(Chunked3dVec3fFromUint8& normal_dirs,
                                                  Chunked3dFloatFromUint8* maybe_weights,
                                                  float w) {
    return new ceres::AutoDiffCostFunction<NormalDirectionLoss, 1, 3, 3, 3>(
        new NormalDirectionLoss(normal_dirs, maybe_weights, w));
}

ceres::CostFunction* Normal3DLineLoss::Create(Chunked3dVec3fFromUint8& normal_dirs,
                                               const NormalFitQualityWeightField* maybe_fit_quality,
                                               float w) {
    return new ceres::AutoDiffCostFunction<Normal3DLineLoss, 1, 3, 3, 3>(
        new Normal3DLineLoss(normal_dirs, maybe_fit_quality, w));
}

ceres::CostFunction* NormalConstraintPlane::Create(const vc::core::util::NormalGridVolume& normal_grid_volume,
                                                    int plane_idx,
                                                    double w_normal,
                                                    double w_snap,
                                                    const NormalFitQualityWeightField* maybe_fit_quality,
                                                    bool direction_aware,
                                                    int z_min,
                                                    int z_max,
                                                    bool invert_dir) {
    return new ceres::AutoDiffCostFunction<NormalConstraintPlane, 1, 3, 3, 3, 3>(
        new NormalConstraintPlane(normal_grid_volume, plane_idx, w_normal, w_snap, maybe_fit_quality, direction_aware, z_min, z_max, invert_dir));
}

ceres::CostFunction* PointCorrectionLoss2P::Create(const cv::Vec3f& correction_src,
                                                    const cv::Vec3f& correction_tgt,
                                                    const cv::Vec2i& grid_loc_int) {
    return new ceres::AutoDiffCostFunction<PointCorrectionLoss2P, 2, 3, 3, 3, 3, 2>(
        new PointCorrectionLoss2P(correction_src, correction_tgt, grid_loc_int));
}

ceres::CostFunction* PointCorrectionLoss::Create(const cv::Vec3f& correction_src,
                                                  const cv::Vec3f& correction_tgt,
                                                  const cv::Vec2i& grid_loc_int) {
    return new ceres::AutoDiffCostFunction<PointCorrectionLoss, 1, 3, 3, 3, 3, 2>(
        new PointCorrectionLoss(correction_src, correction_tgt, grid_loc_int));
}

ceres::CostFunction* SymmetricDirichletLoss::Create(double unit, double w, double eps_abs, double eps_rel) {
    return new ceres::AutoDiffCostFunction<SymmetricDirichletLoss, 1, 3, 3, 3>(
        new SymmetricDirichletLoss(unit, w, eps_abs, eps_rel));
}

ceres::CostFunction* AntiFlipbackLoss::Create(cv::Vec3d anchor, cv::Vec3d normal, double threshold, double w) {
    return new ceres::AutoDiffCostFunction<AntiFlipbackLoss, 1, 3>(
        new AntiFlipbackLoss(anchor, normal, threshold, w));
}
