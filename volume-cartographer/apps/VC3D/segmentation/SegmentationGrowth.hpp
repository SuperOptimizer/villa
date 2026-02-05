#pragma once

#include "SegmentationGrowthEnums.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/matx.hpp>

#include "vc/ui/VCCollectionTypes.hpp"

class QuadSurface;
class Volume;

struct SegmentationDirectionFieldConfig {
    QString path;
    SegmentationDirectionFieldOrientation orientation{SegmentationDirectionFieldOrientation::Normal};
    int scale{0};
    double weight{1.0};

    [[nodiscard]] bool isValid() const noexcept { return !path.isEmpty(); }
};

struct SegmentationCorrectionsPayload {
    struct Collection {
        uint64_t id{0};
        std::string name;
        std::vector<ColPoint> points;
        CollectionMetadata metadata;
        cv::Vec3f color{0.0f, 0.0f, 0.0f};
        std::optional<cv::Vec2f> anchor2d;  // 2D grid anchor for persistent corrections
    };

    std::vector<Collection> collections;

    [[nodiscard]] bool empty() const noexcept { return collections.empty(); }
};

struct SegmentationGrowthRequest {
    SegmentationGrowthMethod method{SegmentationGrowthMethod::Tracer};
    SegmentationGrowthDirection direction{SegmentationGrowthDirection::All};
    int steps{0};
    std::vector<SegmentationGrowthDirection> allowedDirections;
    SegmentationCorrectionsPayload corrections;
    std::optional<std::pair<int, int>> correctionsZRange;
    std::vector<SegmentationDirectionFieldConfig> directionFields;
    std::optional<std::string> customParamsJson;  // raw JSON text, parsed at call site
    bool inpaintOnly{false};
    // Extrapolation parameters
    int extrapolationPointCount{7};
    ExtrapolationType extrapolationType{ExtrapolationType::Linear};
};

struct TracerGrowthContext {
    QuadSurface* resumeSurface{nullptr};
    class Volume* volume{nullptr};
    QString cacheRoot;
    double voxelSize{1.0};
    QString normalGridPath;
    QString normal3dZarrPath;
    // For corrections annotation saving
    std::filesystem::path volpkgRoot;
    std::vector<std::string> volumeIds;
    std::string growthVolumeId;
};

struct TracerGrowthResult {
    QuadSurface* surface{nullptr};
    QString error;
    QString statusMessage;
};

TracerGrowthResult runTracerGrowth(const SegmentationGrowthRequest& request,
                                   const TracerGrowthContext& context);

void updateSegmentationSurfaceMetadata(QuadSurface* surface,
                                       double voxelSize);
