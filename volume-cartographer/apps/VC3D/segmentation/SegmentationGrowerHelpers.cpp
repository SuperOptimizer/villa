/**
 * @file SegmentationGrowerHelpers.cpp
 * @brief Helper functions extracted from SegmentationGrower
 *
 * This file contains utility functions used by SegmentationGrower.
 * Extracted from SegmentationGrower.cpp to improve parallel compilation.
 */

#include "SegmentationGrowerHelpers.hpp"

#include "CVolumeViewer.hpp"
#include "ViewerManager.hpp"
#include "SurfacePanelController.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <QDir>
#include <QLoggingCategory>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>

Q_DECLARE_LOGGING_CATEGORY(lcSegGrowth);

namespace
{

bool isInvalidPointImpl(const cv::Vec3f& value)
{
    return !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ||
           (value[0] == -1.0f && value[1] == -1.0f && value[2] == -1.0f);
}

} // namespace

QString cacheRootForVolumePkg(const std::shared_ptr<VolumePkg>& pkg)
{
    if (!pkg) {
        return QString();
    }

    const QString base = QString::fromStdString(pkg->getVolpkgDirectory());
    return QDir(base).filePath(QStringLiteral("cache"));
}

void ensureGenerationsChannel(QuadSurface* surface)
{
    if (!surface) {
        return;
    }

    cv::Mat generations = surface->channel("generations");
    if (!generations.empty()) {
        return;
    }

    cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return;
    }

    cv::Mat_<uint16_t> seeded(points->rows, points->cols, static_cast<uint16_t>(1));
    surface->setChannel("generations", seeded);
}

void ensureSurfaceMetaObject(QuadSurface* surface)
{
    // No-op: meta is now always a valid SurfaceMeta value
    (void)surface;
}

bool isInvalidPoint(const cv::Vec3f& value)
{
    return isInvalidPointImpl(value);
}

std::optional<std::pair<int, int>> worldToGridIndexApprox(QuadSurface* surface,
                                                          const cv::Vec3f& worldPos,
                                                          cv::Vec3f& pointerSeed,
                                                          bool& pointerSeedValid,
                                                          SurfacePatchIndex* patchIndex)
{
    if (!surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    if (!pointerSeedValid) {
        pointerSeed = surface->pointer();
        pointerSeedValid = true;
    }

    surface->pointTo(pointerSeed, worldPos, std::numeric_limits<float>::max(), 400, patchIndex);
    cv::Vec3f raw = surface->loc_raw(pointerSeed);

    const int rows = points->rows;
    const int cols = points->cols;
    if (rows <= 0 || cols <= 0) {
        return std::nullopt;
    }

    int approxCol = static_cast<int>(std::lround(raw[0]));
    int approxRow = static_cast<int>(std::lround(raw[1]));
    approxRow = std::clamp(approxRow, 0, rows - 1);
    approxCol = std::clamp(approxCol, 0, cols - 1);

    if (isInvalidPointImpl((*points)(approxRow, approxCol))) {
        constexpr int kMaxRadius = 12;
        float bestDistSq = std::numeric_limits<float>::max();
        int bestRow = -1;
        int bestCol = -1;

        auto accumulateCandidate = [&](int r, int c) {
            const cv::Vec3f& candidate = (*points)(r, c);
            if (isInvalidPointImpl(candidate)) {
                return;
            }
            const cv::Vec3f diff = candidate - worldPos;
            const float distSq = diff.dot(diff);
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestRow = r;
                bestCol = c;
            }
        };

        for (int radius = 1; radius <= kMaxRadius; ++radius) {
            const int rowStart = std::max(0, approxRow - radius);
            const int rowEnd = std::min(rows - 1, approxRow + radius);
            const int colStart = std::max(0, approxCol - radius);
            const int colEnd = std::min(cols - 1, approxCol + radius);
            for (int r = rowStart; r <= rowEnd; ++r) {
                for (int c = colStart; c <= colEnd; ++c) {
                    accumulateCandidate(r, c);
                }
            }
            if (bestRow != -1) {
                approxRow = bestRow;
                approxCol = bestCol;
                break;
            }
        }

        if (bestRow == -1) {
            return std::nullopt;
        }
    }

    return std::make_pair(approxRow, approxCol);
}

std::optional<std::pair<int, int>> locateGridIndexWithPatchIndex(QuadSurface* surface,
                                                                 SurfacePatchIndex* patchIndex,
                                                                 const cv::Vec3f& worldPos,
                                                                 cv::Vec3f& pointerSeed,
                                                                 bool& pointerSeedValid)
{
    if (!surface) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    if (patchIndex) {
        cv::Vec2f gridScale = surface->scale();
        float scale = std::max(gridScale[0], gridScale[1]);
        if (!std::isfinite(scale) || scale <= 0.0f) {
            scale = 1.0f;
        }
        const float tolerance = std::max(8.0f, scale * 8.0f);
        // Query without surface filter, then verify result matches our surface
        if (auto hit = patchIndex->locate(worldPos, tolerance)) {
            if (hit->surface.get() == surface) {
                const int col = std::clamp(static_cast<int>(std::round(hit->ptr[0])),
                                           0,
                                           points->cols - 1);
                const int row = std::clamp(static_cast<int>(std::round(hit->ptr[1])),
                                           0,
                                           points->rows - 1);
                return std::make_pair(row, col);
            }
        }
    }

    return worldToGridIndexApprox(surface, worldPos, pointerSeed, pointerSeedValid, patchIndex);
}

std::optional<cv::Rect> computeCorrectionsAffectedBounds(QuadSurface* surface,
                                                      const SegmentationCorrectionsPayload& corrections,
                                                      ViewerManager* viewerManager)
{
    if (!surface || corrections.empty()) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    SurfacePatchIndex* patchIndex = viewerManager ? viewerManager->surfacePatchIndex() : nullptr;

    cv::Vec3f pointerSeed{0.0f, 0.0f, 0.0f};
    bool pointerSeedValid = false;

    const int rows = points->rows;
    const int cols = points->cols;

    constexpr int kPaddingCells = 12;

    bool anyMapped = false;
    int unionRowStart = rows;
    int unionRowEnd = 0;
    int unionColStart = cols;
    int unionColEnd = 0;

    for (const auto& collection : corrections.collections) {
        if (collection.points.empty()) {
            continue;
        }

        int collectionMinRow = rows;
        int collectionMaxRow = -1;
        int collectionMinCol = cols;
        int collectionMaxCol = -1;

        for (const auto& colPoint : collection.points) {
            auto gridIndex = locateGridIndexWithPatchIndex(surface,
                                                           patchIndex,
                                                           colPoint.p,
                                                           pointerSeed,
                                                           pointerSeedValid);
            if (!gridIndex) {
                continue;
            }
            const auto [row, col] = *gridIndex;
            collectionMinRow = std::min(collectionMinRow, row);
            collectionMaxRow = std::max(collectionMaxRow, row);
            collectionMinCol = std::min(collectionMinCol, col);
            collectionMaxCol = std::max(collectionMaxCol, col);
        }

        if (collectionMinRow > collectionMaxRow || collectionMinCol > collectionMaxCol) {
            continue;
        }

        anyMapped = true;
        const int rowStart = std::max(0, collectionMinRow - kPaddingCells);
        const int rowEndExclusive = std::min(rows, collectionMaxRow + kPaddingCells + 1);
        const int colStart = std::max(0, collectionMinCol - kPaddingCells);
        const int colEndExclusive = std::min(cols, collectionMaxCol + kPaddingCells + 1);

        unionRowStart = std::min(unionRowStart, rowStart);
        unionRowEnd = std::max(unionRowEnd, rowEndExclusive);
        unionColStart = std::min(unionColStart, colStart);
        unionColEnd = std::max(unionColEnd, colEndExclusive);
    }

    if (!anyMapped) {
        return cv::Rect(0, 0, cols, rows);
    }

    const int width = std::max(0, unionColEnd - unionColStart);
    const int height = std::max(0, unionRowEnd - unionRowStart);
    if (width == 0 || height == 0) {
        return cv::Rect(0, 0, cols, rows);
    }

    return cv::Rect(unionColStart, unionRowStart, width, height);
}

std::optional<CorrectionsBounds> computeCorrectionsBounds(
    const SegmentationCorrectionsPayload& corrections,
    QuadSurface* surface,
    float minWorldSize)
{
    if (!surface || corrections.empty()) {
        return std::nullopt;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return std::nullopt;
    }

    // Find min/max of all correction point positions (3D world coords)
    cv::Vec3f worldMin(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
    cv::Vec3f worldMax(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());

    bool hasPoints = false;
    for (const auto& collection : corrections.collections) {
        for (const auto& colPoint : collection.points) {
            hasPoints = true;
            worldMin[0] = std::min(worldMin[0], colPoint.p[0]);
            worldMin[1] = std::min(worldMin[1], colPoint.p[1]);
            worldMin[2] = std::min(worldMin[2], colPoint.p[2]);
            worldMax[0] = std::max(worldMax[0], colPoint.p[0]);
            worldMax[1] = std::max(worldMax[1], colPoint.p[1]);
            worldMax[2] = std::max(worldMax[2], colPoint.p[2]);
        }
    }

    if (!hasPoints) {
        return std::nullopt;
    }

    // Compute center and expand to at least minWorldSize in each dimension
    cv::Vec3f center = (worldMin + worldMax) * 0.5f;
    cv::Vec3f halfSize;
    for (int i = 0; i < 3; ++i) {
        float extent = worldMax[i] - worldMin[i];
        halfSize[i] = std::max(extent * 0.5f, minWorldSize * 0.5f);
    }

    worldMin = center - halfSize;
    worldMax = center + halfSize;

    // Find all grid cells whose 3D positions fall within this world-space box
    const int rows = points->rows;
    const int cols = points->cols;

    int gridRowMin = rows;
    int gridRowMax = -1;
    int gridColMin = cols;
    int gridColMax = -1;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const cv::Vec3f& pt = (*points)(r, c);
            if (isInvalidPointImpl(pt)) {
                continue;
            }

            // Check if point is within the world-space bounding box
            if (pt[0] >= worldMin[0] && pt[0] <= worldMax[0] &&
                pt[1] >= worldMin[1] && pt[1] <= worldMax[1] &&
                pt[2] >= worldMin[2] && pt[2] <= worldMax[2]) {
                gridRowMin = std::min(gridRowMin, r);
                gridRowMax = std::max(gridRowMax, r);
                gridColMin = std::min(gridColMin, c);
                gridColMax = std::max(gridColMax, c);
            }
        }
    }

    if (gridRowMin > gridRowMax || gridColMin > gridColMax) {
        // No grid cells found within bounds, return full grid
        return CorrectionsBounds{worldMin, worldMax, cv::Rect(0, 0, cols, rows)};
    }

    // Clip to surface grid bounds (already implicitly done by loop bounds)
    int width = gridColMax - gridColMin + 1;
    int height = gridRowMax - gridRowMin + 1;

    CorrectionsBounds bounds;
    bounds.worldMin = worldMin;
    bounds.worldMax = worldMax;
    bounds.gridRegion = cv::Rect(gridColMin, gridRowMin, width, height);

    return bounds;
}

std::unique_ptr<QuadSurface> cropSurfaceToGridRegion(
    const QuadSurface* surface,
    const cv::Rect& gridRegion)
{
    if (!surface) {
        return nullptr;
    }

    const cv::Mat_<cv::Vec3f>* points = surface->rawPointsPtr();
    if (!points || points->empty()) {
        return nullptr;
    }

    const int rows = points->rows;
    const int cols = points->cols;

    // Clamp region to valid bounds
    int x0 = std::clamp(gridRegion.x, 0, cols - 1);
    int y0 = std::clamp(gridRegion.y, 0, rows - 1);
    int x1 = std::clamp(gridRegion.x + gridRegion.width, 1, cols);
    int y1 = std::clamp(gridRegion.y + gridRegion.height, 1, rows);

    if (x1 <= x0 || y1 <= y0) {
        return nullptr;
    }

    // Extract ROI
    cv::Mat_<cv::Vec3f> roi(*points, cv::Range(y0, y1), cv::Range(x0, x1));
    cv::Mat_<cv::Vec3f> roiClone = roi.clone();

    // Create new QuadSurface with the cropped data
    auto cropped = std::make_unique<QuadSurface>(roiClone, surface->scale());

    return cropped;
}

std::string generateTimestampString()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&time, &tm);

    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H-%M-%S", &tm);
    return std::string(buffer);
}

void saveCorrectionsAnnotation(
    const std::filesystem::path& volpkgRoot,
    const std::string& segmentId,
    const QuadSurface* beforeCrop,
    const QuadSurface* afterCrop,
    const SegmentationCorrectionsPayload& corrections,
    const std::vector<std::string>& volumeIds,
    const std::string& growthVolumeId,
    const CorrectionsBounds& bounds)
{
    if (!beforeCrop || !afterCrop || corrections.empty()) {
        return;
    }

    // Create timestamped folder
    std::string timestamp = generateTimestampString();
    std::filesystem::path correctionsDir = volpkgRoot / "corrections" / timestamp;

    std::error_code ec;
    std::filesystem::create_directories(correctionsDir, ec);
    if (ec) {
        qCWarning(lcSegGrowth) << "Failed to create corrections directory:" << ec.message().c_str();
        return;
    }

    // Save before tifxyz
    std::filesystem::path beforePath = correctionsDir / "before";
    try {
        auto* mutableBefore = const_cast<QuadSurface*>(beforeCrop);
        mutableBefore->save(beforePath.string(), "before", true);
    } catch (const std::exception& ex) {
        qCWarning(lcSegGrowth) << "Failed to save before tifxyz:" << ex.what();
    }

    // Save after tifxyz
    std::filesystem::path afterPath = correctionsDir / "after";
    try {
        auto* mutableAfter = const_cast<QuadSurface*>(afterCrop);
        mutableAfter->save(afterPath.string(), "after", true);
    } catch (const std::exception& ex) {
        qCWarning(lcSegGrowth) << "Failed to save after tifxyz:" << ex.what();
    }

    // Build corrections.json
    nlohmann::json j;
    j["timestamp"] = timestamp;
    j["segment_id"] = segmentId;
    j["volumes"] = volumeIds;
    j["volume_used"] = growthVolumeId;
    j["bbox"] = {
        {"min", {bounds.worldMin[0], bounds.worldMin[1], bounds.worldMin[2]}},
        {"max", {bounds.worldMax[0], bounds.worldMax[1], bounds.worldMax[2]}}
    };

    // Build collections array with points sorted by creation_time
    nlohmann::json collectionsJson = nlohmann::json::array();
    for (const auto& collection : corrections.collections) {
        nlohmann::json collJson;
        collJson["id"] = collection.id;
        collJson["name"] = collection.name;
        collJson["color"] = {collection.color[0], collection.color[1], collection.color[2]};

        // Sort points by creation_time to preserve placement order
        std::vector<ColPoint> sortedPoints = collection.points;
        std::sort(sortedPoints.begin(), sortedPoints.end(),
                  [](const ColPoint& a, const ColPoint& b) {
                      return a.creation_time < b.creation_time;
                  });

        nlohmann::json pointsJson = nlohmann::json::array();
        for (const auto& pt : sortedPoints) {
            nlohmann::json ptJson;
            ptJson["id"] = pt.id;
            ptJson["position"] = {pt.p[0], pt.p[1], pt.p[2]};
            ptJson["creation_time"] = pt.creation_time;
            pointsJson.push_back(ptJson);
        }
        collJson["points"] = pointsJson;

        collectionsJson.push_back(collJson);
    }
    j["collections"] = collectionsJson;

    // Write corrections.json
    std::filesystem::path jsonPath = correctionsDir / "corrections.json";
    try {
        std::ofstream ofs(jsonPath);
        if (ofs.is_open()) {
            ofs << j.dump(2);
            ofs.close();
            qCInfo(lcSegGrowth) << "Saved corrections annotation to" << jsonPath.c_str();
        } else {
            qCWarning(lcSegGrowth) << "Failed to open corrections.json for writing";
        }
    } catch (const std::exception& ex) {
        qCWarning(lcSegGrowth) << "Failed to write corrections.json:" << ex.what();
    }
}

void queueIndexUpdateForBounds(SurfacePatchIndex* index,
                               const std::shared_ptr<QuadSurface>& surface,
                               const cv::Rect& vertexRect)
{
    if (!index || !surface || vertexRect.width <= 0 || vertexRect.height <= 0) {
        return;
    }

    const int rowStart = vertexRect.y;
    const int rowEnd = vertexRect.y + vertexRect.height;
    const int colStart = vertexRect.x;
    const int colEnd = vertexRect.x + vertexRect.width;

    index->queueCellRangeUpdate(surface, rowStart, rowEnd, colStart, colEnd);
}

void synchronizeSurfaceMeta(const std::shared_ptr<VolumePkg>& pkg,
                            QuadSurface* surface,
                            SurfacePanelController* panel)
{
    if (!pkg || !surface) {
        return;
    }

    // getSurface now returns the QuadSurface directly, so we just need to refresh the panel
    const auto loadedIds = pkg->getLoadedSurfaceIDs();
    for (const auto& id : loadedIds) {
        auto loadedSurface = pkg->getSurface(id);
        if (!loadedSurface) {
            continue;
        }

        if (loadedSurface->path == surface->path) {
            // Sync metadata
            loadedSurface->meta = surface->meta;

            if (panel) {
                panel->refreshSurfaceMetrics(id);
            }
        }
    }
}

void refreshSegmentationViewers(ViewerManager* manager)
{
    if (!manager) {
        return;
    }

    manager->forEachViewer([](CVolumeViewer* viewer) {
        if (!viewer) {
            return;
        }

        if (viewer->surfName() == "segmentation") {
            viewer->invalidateVis();
            viewer->renderVisible(true);
        }
    });
}
