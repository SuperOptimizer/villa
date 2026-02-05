#pragma once

#include <opencv2/core/types.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vc::core::util {

struct GridMeta {
    float umbilicus_x = 0;
    float umbilicus_y = 0;
    bool aligned = false;
};

class GridStore final {
public:
    GridStore(const cv::Rect& bounds, int cell_size);
    explicit GridStore(const std::string& path);
    ~GridStore();

    void add(const std::vector<cv::Point>& points);
    [[nodiscard]] std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Rect& query_rect) const;
    [[nodiscard]] std::vector<std::shared_ptr<std::vector<cv::Point>>> get(const cv::Point2f& center, float radius) const;
    [[nodiscard]] std::vector<std::shared_ptr<std::vector<cv::Point>>> get_all() const;
    [[nodiscard]] cv::Size size() const noexcept;
    [[nodiscard]] size_t get_memory_usage() const noexcept;
    [[nodiscard]] size_t numSegments() const noexcept;
    [[nodiscard]] size_t numNonEmptyBuckets() const noexcept;

    [[nodiscard]] GridMeta& meta() noexcept { return meta_; }
    [[nodiscard]] const GridMeta& meta() const noexcept { return meta_; }

    void save(const std::string& path) const;
    void load_mmap(const std::string& path);

private:
    friend class LineSegList;
    class GridStoreImpl;

    std::unique_ptr<GridStoreImpl> pimpl_;
    GridMeta meta_;
};

}
