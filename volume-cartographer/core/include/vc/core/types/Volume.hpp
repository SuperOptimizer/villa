#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "vc/core/types/Sampling.hpp"

// Forward declarations
namespace vc { class VcDataset; }

namespace vc::cache {
    class TieredChunkCache;
    class DiskStore;
}

struct CompositeParams;

class Volume
{
public:
    // Static flag to skip zarr shape validation against meta.json
    static inline bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, std::string uuid, std::string name);

    ~Volume();


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    static std::shared_ptr<Volume> New(std::filesystem::path path, std::string uuid, std::string name);

    // Create a Volume backed by a remote zarr store over HTTP.
    // Downloads metadata (.zarray files) to a local staging dir, then
    // fetches chunk data on demand via HttpChunkSource.
    static std::shared_ptr<Volume> NewFromUrl(
        const std::string& url,
        const std::filesystem::path& cacheRoot = {});

    [[nodiscard]] bool isRemote() const { return isRemote_; }
    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void saveMetadata();

    [[nodiscard]] int sliceWidth() const;
    [[nodiscard]] int sliceHeight() const;
    [[nodiscard]] int numSlices() const;
    [[nodiscard]] std::array<int, 3> shape() const;
    [[nodiscard]] double voxelSize() const;

    [[nodiscard]] vc::VcDataset *zarrDataset(int level = 0) const;
    [[nodiscard]] size_t numScales() const;

    // Create a TieredChunkCache backed by this volume's zarr data.
    // diskStore: optional shared disk cache (nullptr to disable cold tier).
    std::unique_ptr<vc::cache::TieredChunkCache> createTieredCache(
        std::shared_ptr<vc::cache::DiskStore> diskStore = nullptr) const;

    // --- Cache management ---

    // Lazily create and return the tiered chunk cache for this volume.
    // Thread-safe: creates on first call, returns same cache thereafter.
    vc::cache::TieredChunkCache* tieredCache();

    // Set cache budget (must be called before first tieredCache() access).
    void setCacheBudget(size_t hotBytes, size_t warmBytes = 0);

    // Inject a shared DiskStore for the cold cache tier.
    // Must be called before first tieredCache() access.
    void setDiskStore(std::shared_ptr<vc::cache::DiskStore> store);

    // --- Blocking sampling (CLI / batch) ---

    void sample(cv::Mat_<uint8_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                int level = 0,
                vc::Sampling method = vc::Sampling::Trilinear);

    void sampleComposite(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& baseCoords,
                         const cv::Mat_<cv::Vec3f>& normals,
                         int zStart, int zEnd,
                         const CompositeParams& params,
                         int level = 0);

    // --- Non-blocking sampling (interactive) ---

    // Returns actual pyramid level used. If the requested level's chunks
    // aren't all cached, falls back to coarser levels.
    // Prefetches missing chunks at the requested level in the background.
    int sampleBestEffort(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         int level = 0,
                         vc::Sampling method = vc::Sampling::Trilinear);

    // --- Dimensioned read API ---

    // 0D: single point
    uint8_t read0d(cv::Vec3f point, int level = 0,
                   vc::Sampling method = vc::Sampling::Trilinear);

    // 1D: line sample — origin + step direction, count samples
    void read1d(cv::Mat_<uint8_t>& out,
                cv::Vec3f origin, cv::Vec3f step, int count,
                int level = 0, vc::Sampling method = vc::Sampling::Trilinear);

    // 2D geometric: origin + two axis vectors define the sampling grid
    void read2d(cv::Mat_<uint8_t>& out,
                cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                int w, int h, int level = 0,
                vc::Sampling method = vc::Sampling::Trilinear);
    int read2dBestEffort(cv::Mat_<uint8_t>& out,
                         cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                         int w, int h, int level = 0,
                         vc::Sampling method = vc::Sampling::Trilinear);

    // 2D pre-generated coords (QuadSurface / parametric)
    void read2d(cv::Mat_<uint8_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                int level = 0, vc::Sampling method = vc::Sampling::Trilinear);
    int read2dBestEffort(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         int level = 0, vc::Sampling method = vc::Sampling::Trilinear);

    // 3D pre-generated coords + normals (composite along normals)
    void read3d(cv::Mat_<uint8_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const cv::Mat_<cv::Vec3f>& normals,
                int zStart, int zEnd, const CompositeParams& params,
                int level = 0);
    int read3dBestEffort(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         const cv::Mat_<cv::Vec3f>& normals,
                         int zStart, int zEnd, const CompositeParams& params,
                         int level = 0);

    // 3D geometric: plane + normal direction for composite
    void read3d(cv::Mat_<uint8_t>& out,
                cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                cv::Vec3f normal, int w, int h,
                int zStart, int zEnd, const CompositeParams& params,
                int level = 0);
    int read3dBestEffort(cv::Mat_<uint8_t>& out,
                         cv::Vec3f origin, cv::Vec3f axisU, cv::Vec3f axisV,
                         cv::Vec3f normal, int w, int h,
                         int zStart, int zEnd, const CompositeParams& params,
                         int level = 0);

    // Pin the coarsest pyramid level in the hot tier (never evicted).
    // Guarantees sampleBestEffort() always returns data immediately.
    void pinCoarsestLevel(bool blocking = false);

    // --- Chunk prefetch ---
    void prefetchChunks(const cv::Mat_<cv::Vec3f>& coords, int level);

    // Prefetch all chunks overlapping a world-space axis-aligned bounding box.
    // lo/hi are in world (level 0) coordinates, (x, y, z).
    void prefetchWorldBBox(const cv::Vec3f& lo, const cv::Vec3f& hi, int level);

    // Cancel all pending (not in-flight) async prefetch tasks.
    void cancelPendingPrefetch();

    // --- Query ---
    bool allChunksCached(const cv::Mat_<cv::Vec3f>& coords, int level) const;

    static bool checkDir(std::filesystem::path path);

protected:
    std::filesystem::path path_;
    nlohmann::json metadata_;

    int _width{0};
    int _height{0};
    int _slices{0};

    std::vector<std::unique_ptr<vc::VcDataset>> zarrDs_;
    nlohmann::json zarrGroup_;
    void zarrOpen();

    // Cache ownership
    mutable std::unique_ptr<vc::cache::TieredChunkCache> tieredCache_;
    size_t cacheBudgetHot_ = 8ULL << 30;   // 8 GB default
    size_t cacheBudgetWarm_ = 2ULL << 30;   // 2 GB default
    std::shared_ptr<vc::cache::DiskStore> pendingDiskStore_;

    void ensureTieredCache() const;

    // Bounding box of coords in chunk index space (helper for allChunksCached/prefetch)
    struct ChunkBBox {
        int minIx, maxIx, minIy, maxIy, minIz, maxIz;
    };
    ChunkBBox coordsToChunkBBox(const cv::Mat_<cv::Vec3f>& coords, int level) const;

    // Composite-aware chunk helpers: expand bbox by normal offsets
    bool allCompositeChunksCached(const cv::Mat_<cv::Vec3f>& coords,
                                  const cv::Mat_<cv::Vec3f>& normals,
                                  int zStart, int zEnd, int level) const;
    void prefetchCompositeChunks(const cv::Mat_<cv::Vec3f>& coords,
                                 const cv::Mat_<cv::Vec3f>& normals,
                                 int zStart, int zEnd, int level);

    void loadMetadata();

    // Remote volume state
    bool isRemote_ = false;
    std::string remoteUrl_;
    std::string remoteDelimiter_ = ".";
};
