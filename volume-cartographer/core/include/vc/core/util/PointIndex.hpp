#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

class PointIndex
{
public:
    struct QueryResult {
        uint64_t id = 0;
        uint64_t collectionId = 0;
        cv::Vec3f position{0, 0, 0};
        float distanceSq = 0.0f;
    };

    PointIndex();
    ~PointIndex();

    PointIndex(const PointIndex&) = delete;
    PointIndex& operator=(const PointIndex&) = delete;

    void clear();
    bool empty() const;
    size_t size() const;

    void buildFromMat(const cv::Mat_<cv::Vec3f>& points, uint64_t collectionId = 0);

    std::vector<QueryResult> queryRadius(
        const cv::Vec3f& center,
        float radius) const;

    std::optional<QueryResult> nearest(
        const cv::Vec3f& position,
        float maxDistance = std::numeric_limits<float>::max()) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
