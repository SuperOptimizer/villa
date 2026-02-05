#pragma once

#include <cstddef>
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>

#include <vc/core/types/InterpolationMethod.hpp>

// Forward declarations — no z5 or xtensor types in the public API
namespace z5 {
    class Dataset;
}
class ChunkStore;
struct CompositeParams;
template <typename T> class ChunkCache;

class Volume final
{
public:
    // Static flag to skip zarr shape validation against meta.json
    static inline bool skipShapeCheck = false;

    Volume() = delete;

    explicit Volume(std::filesystem::path path);
    Volume(std::filesystem::path path, std::shared_ptr<ChunkStore> store);
    Volume(std::filesystem::path path, const std::string& uuid, const std::string& name);

    ~Volume();

    static std::shared_ptr<Volume> New(std::filesystem::path path);
    static std::shared_ptr<Volume> New(std::filesystem::path path, std::shared_ptr<ChunkStore> store);
    static std::shared_ptr<Volume> New(std::filesystem::path path, const std::string& uuid, const std::string& name);

    // ── Metadata ──────────────────────────────────────────────────────────
    [[nodiscard]] std::string id() const;
    [[nodiscard]] std::string name() const;
    void setName(const std::string& n);
    [[nodiscard]] std::filesystem::path path() const noexcept;
    void saveMetadata();

    [[nodiscard]] int sliceWidth() const noexcept;
    [[nodiscard]] int sliceHeight() const noexcept;
    [[nodiscard]] int numSlices() const noexcept;
    [[nodiscard]] std::array<int, 3> shape() const noexcept;
    [[nodiscard]] double voxelSize() const;

    // ── Scale levels ──────────────────────────────────────────────────────
    [[nodiscard]] size_t numScales() const noexcept;
    [[nodiscard]] std::array<size_t, 3> shapeZYX(int level = 0) const;
    [[nodiscard]] std::array<int, 3> chunkShape(int level = 0) const;

    // ── Cache management ──────────────────────────────────────────────────
    void setChunkStore(std::shared_ptr<ChunkStore> store);
    [[nodiscard]] std::shared_ptr<ChunkStore> chunkStore() const;

    // ── Interpolated reads ────────────────────────────────────────────────
    void readInterpolated(cv::Mat_<uint8_t>& out, const cv::Mat_<cv::Vec3f>& coords,
                          InterpolationMethod method = InterpolationMethod::Trilinear, int level = 0);
    void readInterpolated(cv::Mat_<uint16_t>& out, const cv::Mat_<cv::Vec3f>& coords,
                          InterpolationMethod method = InterpolationMethod::Trilinear, int level = 0);

    // ── Composite rendering ───────────────────────────────────────────────
    void readComposite(cv::Mat_<uint8_t>& out, const cv::Mat_<cv::Vec3f>& baseCoords,
                       const cv::Mat_<cv::Vec3f>& normals, float zStep, int zStart, int zEnd,
                       const CompositeParams& params, int level = 0);
    void readCompositeConstantNormal(cv::Mat_<uint8_t>& out, const cv::Mat_<cv::Vec3f>& baseCoords,
                                     const cv::Vec3f& normal, float zStep, int zStart, int zEnd,
                                     const CompositeParams& params, int level = 0);

    // ── Multi-slice reads ─────────────────────────────────────────────────
    void readMultiSlice(std::vector<cv::Mat_<uint8_t>>& out, const cv::Mat_<cv::Vec3f>& basePoints,
                        const cv::Mat_<cv::Vec3f>& stepDirs, const std::vector<float>& offsets, int level = 0);
    void readMultiSlice(std::vector<cv::Mat_<uint16_t>>& out, const cv::Mat_<cv::Vec3f>& basePoints,
                        const cv::Mat_<cv::Vec3f>& stepDirs, const std::vector<float>& offsets, int level = 0);

    // ── 3D block reads (cv::Mat, no xtensor in API) ─────────────────────
    // Read a 3D sub-volume as a vector of 2D slices (one cv::Mat per Z).
    // offset/size are in ZYX order.
    void readBlock(std::vector<cv::Mat_<uint8_t>>& out,
                   const std::array<int,3>& offsetZYX, const std::array<int,3>& sizeZYX, int level = 0);
    void readBlock(std::vector<cv::Mat_<uint16_t>>& out,
                   const std::array<int,3>& offsetZYX, const std::array<int,3>& sizeZYX, int level = 0);

    // Write a 3D sub-volume from a vector of 2D slices.
    void writeBlock(const std::vector<cv::Mat_<uint8_t>>& data,
                    const std::array<int,3>& offsetZYX, int level = 0);
    void writeBlock(const std::vector<cv::Mat_<uint16_t>>& data,
                    const std::array<int,3>& offsetZYX, int level = 0);

    // ── Gradient computation ──────────────────────────────────────────────
    cv::Mat_<cv::Vec3f> computeGradients(const cv::Mat_<cv::Vec3f>& rawPoints, float dsScale, int level = 0);

    // ── Deprecated bridge — will be removed in Phase 3 ───────────────────
    // Provides raw z5::Dataset* for files not yet migrated.
    [[deprecated("Use Volume methods instead of raw z5::Dataset*")]]
    [[nodiscard]] z5::Dataset* zarrDataset(int level = 0) const;

    // Provides raw ChunkCache pointers for files not yet migrated.
    [[nodiscard]] ChunkCache<uint8_t>* rawCache8() const;
    [[nodiscard]] ChunkCache<uint16_t>* rawCache16() const;

    static bool checkDir(const std::filesystem::path& path);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};
