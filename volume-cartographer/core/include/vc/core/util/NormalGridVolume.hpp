#pragma once

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <memory>
#include <optional>

// Forward declarations
namespace vc::core::util { class GridStore; }

namespace vc::core::util {

    class NormalGridVolume final {
    public:
        explicit NormalGridVolume(const std::string& path);
        ~NormalGridVolume();
        NormalGridVolume(NormalGridVolume&&) noexcept;
        NormalGridVolume& operator=(NormalGridVolume&&) noexcept;

        struct GridQueryResult {
            std::shared_ptr<const GridStore> grid1;
            std::shared_ptr<const GridStore> grid2;
            double weight;
        };

        std::optional<GridQueryResult> query(const cv::Point3f& point, int plane_idx) const;
        std::shared_ptr<const GridStore> query_nearest(const cv::Point3f& point, int plane_idx) const;

    public:
        const nlohmann::json& metadata() const;

    private:
        struct pimpl;

        std::unique_ptr<pimpl> pimpl_;
    };

} // namespace vc::core::util
