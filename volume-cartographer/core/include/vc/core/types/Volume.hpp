#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "vc/core/types/Sampling.hpp"
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"  // HttpAuth
#include <utils/zarr.hpp>

namespace vc::cache {
    class TieredChunkCache;
    class DiskStore;
}

struct CompositeParams;

class Volume
{
public:
    // Bounding box of non-zero data in level-0 voxel coordinates (inclusive).
    struct DataBounds {
        int minX = 0, maxX = 0;
        int minY = 0, maxY = 0;
        int minZ = 0, maxZ = 0;
        bool valid = false;
    };

    // Static flag to skip zarr shape validation against meta.json
    static inline bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);

    Volume(std::filesystem::path path, std::string uuid, std::string name);

    ~Volume() noexcept;


    static std::shared_ptr<Volume> New(std::filesystem::path path);

    static std::shared_ptr<Volume> New(std::filesystem::path path, std::string uuid, std::string name);

    // Create a Volume backed by a remote zarr store over HTTP.
    static std::shared_ptr<Volume> NewFromUrl(
        const std::string& url,
        const std::filesystem::path& cacheRoot = {},
        const vc::cache::HttpAuth& auth = {});

    [[nodiscard]] bool isRemote() const noexcept { return isRemote_; }
    [[nodiscard]] std::string id() const noexcept;
    [[nodiscard]] std::string name() const noexcept;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }
    void saveMetadata();

    [[nodiscard]] int sliceWidth() const noexcept;
    [[nodiscard]] int sliceHeight() const noexcept;
    [[nodiscard]] int numSlices() const noexcept;
    [[nodiscard]] std::array<int, 3> shape() const noexcept;
    [[nodiscard]] double voxelSize() const noexcept;

    [[nodiscard]] size_t numScales() const noexcept;

    // Zarr access
    [[nodiscard]] vc::Zarr* zarr(int level = 0) const;

    // Create a TieredChunkCache backed by this volume's zarr data.
    [[nodiscard]] std::unique_ptr<vc::cache::TieredChunkCache> createTieredCache(
        std::shared_ptr<vc::cache::DiskStore> diskStore = nullptr) const;

    // Lazily create and return the tiered chunk cache for this volume.
    [[nodiscard]] vc::cache::TieredChunkCache* tieredCache();

    void setCacheBudget(size_t hotBytes, size_t warmBytes = 0);
    void setDiskStore(std::shared_ptr<vc::cache::DiskStore> store);

    // --- Sampling API (uses TieredChunkCache, z5-independent) ---

    void sample(cv::Mat_<uint8_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const vc::SampleParams& params);

    void sample(cv::Mat_<uint16_t>& out,
                const cv::Mat_<cv::Vec3f>& coords,
                const vc::SampleParams& params);

    int sampleBestEffort(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         const vc::SampleParams& params);

    void sampleComposite(cv::Mat_<uint8_t>& out,
                         const cv::Mat_<cv::Vec3f>& coords,
                         const cv::Mat_<cv::Vec3f>& normals,
                         const vc::SampleParams& params);

    int sampleCompositeBestEffort(cv::Mat_<uint8_t>& out,
                                  const cv::Mat_<cv::Vec3f>& coords,
                                  const cv::Mat_<cv::Vec3f>& normals,
                                  const vc::SampleParams& params);

    void sampleMultiSlice(std::vector<cv::Mat_<uint8_t>>& out,
                          const cv::Mat_<cv::Vec3f>& basePoints,
                          const cv::Mat_<cv::Vec3f>& stepDirs,
                          const std::vector<float>& offsets,
                          const vc::SampleParams& params);

    void sampleMultiSlice(std::vector<cv::Mat_<uint16_t>>& out,
                          const cv::Mat_<cv::Vec3f>& basePoints,
                          const cv::Mat_<cv::Vec3f>& stepDirs,
                          const std::vector<float>& offsets,
                          const vc::SampleParams& params);

    void sampleMultiSliceST(std::vector<cv::Mat_<uint8_t>>& out,
                            const cv::Mat_<cv::Vec3f>& basePoints,
                            const cv::Mat_<cv::Vec3f>& stepDirs,
                            const std::vector<float>& offsets,
                            const vc::SampleParams& params);

    void sampleMultiSliceST(std::vector<cv::Mat_<uint16_t>>& out,
                            const cv::Mat_<cv::Vec3f>& basePoints,
                            const cv::Mat_<cv::Vec3f>& stepDirs,
                            const std::vector<float>& offsets,
                            const vc::SampleParams& params);

    [[nodiscard]] cv::Mat_<cv::Vec3f> computeGradients(const cv::Mat_<cv::Vec3f>& rawPoints,
                                         float dsScale, int level = 0);

    void pinCoarsestLevel(bool blocking = false);

    void prefetchChunks(const cv::Mat_<cv::Vec3f>& coords, int level);
    void prefetchWorldBBox(const cv::Vec3f& lo, const cv::Vec3f& hi, int level);
    void cancelPendingPrefetch();

    [[nodiscard]] const DataBounds& dataBounds() const;
    void computeDataBounds();

    [[nodiscard]] bool allChunksCached(const cv::Mat_<cv::Vec3f>& coords, int level) const;

    [[nodiscard]] static bool checkDir(std::filesystem::path path);

protected:
    std::filesystem::path path_;
    nlohmann::json metadata_;
    bool metadataAutoGenerated_{false};

    int _width{0};
    int _height{0};
    int _slices{0};

    // OMEZarr multi-scale zarr pyramid
    vc::OMEZarr omeZarr_;
    void openZarr();

    // Cache ownership
    mutable std::unique_ptr<vc::cache::TieredChunkCache> tieredCache_;
    mutable std::once_flag cacheOnce_;
    size_t cacheBudgetHot_ = 8ULL << 30;
    size_t cacheBudgetWarm_ = 2ULL << 30;
    std::shared_ptr<vc::cache::DiskStore> pendingDiskStore_;

    void ensureTieredCache() const;

    // Data bounds (lazy-computed)
    mutable DataBounds dataBounds_;
    mutable std::atomic<bool> boundsComputed_{false};
    mutable std::mutex boundsMutex_;

    struct ChunkBBox {
        int minIx, maxIx, minIy, maxIy, minIz, maxIz;
    };
    ChunkBBox coordsToChunkBBox(const cv::Mat_<cv::Vec3f>& coords, int level) const;

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
    vc::cache::HttpAuth remoteAuth_;
};
