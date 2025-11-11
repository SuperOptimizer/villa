#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class QuadSurface;

class SurfacePatchIndex {
public:
    struct LookupResult {
        QuadSurface* surface = nullptr;
        cv::Vec3f ptr = {0, 0, 0};
        float distance = -1.0f;
    };

    SurfacePatchIndex();
    ~SurfacePatchIndex();

    SurfacePatchIndex(SurfacePatchIndex&&) noexcept;
    SurfacePatchIndex& operator=(SurfacePatchIndex&&) noexcept;

    SurfacePatchIndex(const SurfacePatchIndex&) = delete;
    SurfacePatchIndex& operator=(const SurfacePatchIndex&) = delete;

    void rebuild(const std::vector<QuadSurface*>& surfaces, float bboxPadding = 0.0f);
    void clear();
    bool empty() const;

    std::optional<LookupResult> locate(const cv::Vec3f& worldPoint,
                                       float tolerance,
                                       QuadSurface* targetSurface = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
