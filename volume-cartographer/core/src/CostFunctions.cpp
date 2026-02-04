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
