#pragma once

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/types.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace vc::core::util {

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

    // Accessor for metadata (stored internally as unique_ptr)
    [[nodiscard]] nlohmann::json& meta();
    [[nodiscard]] const nlohmann::json& meta() const;

    void save(const std::string& path) const;
    void load_mmap(const std::string& path);

private:
    friend class LineSegList;
    class GridStoreImpl;

    std::unique_ptr<GridStoreImpl> pimpl_;
    std::unique_ptr<nlohmann::json> meta_;
};

}
