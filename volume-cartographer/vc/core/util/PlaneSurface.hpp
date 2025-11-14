#pragma once

#include <opencv2/core.hpp>

#include "Surface.hpp"
#include "Slicing.hpp"

class PlaneSurface : public Surface
{
public:
    //Surface API
    cv::Vec3f pointer() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override { return true; }
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) override { abort(); }

    PlaneSurface() = default;
    PlaneSurface(const cv::Vec3f &origin_, const cv::Vec3f &normal_);

    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    float pointDist(const cv::Vec3f &wp) const;
    cv::Vec3f project(const cv::Vec3f &wp, float render_scale = 1.0, float coord_scale = 1.0) const;
    void setNormal(const cv::Vec3f &normal);
    void setOrigin(const cv::Vec3f &origin);
    cv::Vec3f origin();
    float scalarp(const cv::Vec3f &point) const;
    void setInPlaneRotation(float radians);
    float inPlaneRotation() const { return _inPlaneRotation; }
    cv::Vec3f basisX() const { return _vx; }
    cv::Vec3f basisY() const { return _vy; }
    void setAxisAlignedRotationKey(int key);
    int axisAlignedRotationKey() const { return _axisAlignedRotationKey; }

protected:
    void update();
    cv::Vec3f _normal = {0,0,1};
    cv::Vec3f _origin = {0,0,0};
    cv::Vec3f _vx = {1,0,0};
    cv::Vec3f _vy = {0,1,0};
    float _inPlaneRotation = 0.0f;
    cv::Matx33d _M;
    cv::Vec3d _T;
    int _axisAlignedRotationKey = -1;
};
