#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

#include <opencv2/core.hpp>
#include <z5/dataset.hxx>

#include "Surface.hpp"
#include "Slicing.hpp"

template<typename T> class ChunkCache;

// QuadSurface loading/channel flags
constexpr int SURF_LOAD_IGNORE_MASK = (1 << 0);
constexpr int SURF_CHANNEL_NORESIZE = (1 << 1);

// Options for writing TIFF from QuadSurface
struct TiffWriteOptions {
    enum class Compression { NONE, LZW, DEFLATE };
    enum class Predictor  { NONE, HORIZONTAL, FLOATINGPOINT };

    bool forceBigTiff = false;
    int  tileSize = 1024;                 // square tiles
    Compression compression = Compression::LZW;
    Predictor  predictor   = Predictor::FLOATINGPOINT; // for float32
};

struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);

//quads based surface class with a pointer implementing a nominal scale of 1 voxel
class QuadSurface : public Surface
{
public:
    QuadSurface() = default;
    // points will be cloned in constructor
    QuadSurface(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &scale);
    // points will not be cloned in constructor, but pointer stored
    QuadSurface(cv::Mat_<cv::Vec3f> *points, const cv::Vec2f &scale);
    ~QuadSurface() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset) override;
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f loc_raw(const cv::Vec3f &ptr);
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &tgt, float th, int max_iters = 1000) override;
    cv::Size size();
    [[nodiscard]] cv::Vec2f scale() const;
    [[nodiscard]] cv::Vec3f center() const;

    void save(const std::string &path, const std::string &uuid, bool force_overwrite = false);
    void save(const std::filesystem::path &path, bool force_overwrite = false);
    void save_meta();
    // Configure how TIFFs are written
    void setTiffWriteOptions(const TiffWriteOptions& opt) { _tiff_opts = opt; }
    Rect3D bbox();

    bool containsPoint(const cv::Vec3f& point, float tolerance) const;

    virtual cv::Mat_<cv::Vec3f> rawPoints() { return *_points; }
    virtual cv::Mat_<cv::Vec3f> *rawPointsPtr() { return _points; }
    virtual const cv::Mat_<cv::Vec3f> *rawPointsPtr() const { return _points; }

    friend QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search, int step_out);
    friend QuadSurface *smooth_vc_segmentation(QuadSurface *src);
    friend class ControlPointSurface;
    cv::Vec2f _scale;

    void setChannel(const std::string& name, const cv::Mat& channel);
    cv::Mat channel(const std::string& name, int flags = 0);
    void invalidateCache();
    void saveOverwrite();
    void saveSnapshot(int maxBackups = 10);
    std::vector<std::string> channelNames() const;

protected:
    std::unordered_map<std::string, cv::Mat> _channels;
    cv::Mat_<cv::Vec3f>* _points = nullptr;
    cv::Rect _bounds;
    TiffWriteOptions _tiff_opts;
    cv::Vec3f _center;
    Rect3D _bbox = {{-1,-1,-1},{-1,-1,-1}};
};

QuadSurface *load_quad_from_tifxyz(const std::string &path, int flags = 0);
QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search = 100, int step_out = 5);
QuadSurface *smooth_vc_segmentation(QuadSurface *src);

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

QuadSurface* surface_diff(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
QuadSurface* surface_union(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);
QuadSurface* surface_intersection(QuadSurface* a, QuadSurface* b, float tolerance = 2.0);

// Control CUDA usage in GrowPatch (space_tracing_quad_phys). Default is true.
void set_space_tracing_use_cuda(bool enable);

void generate_mask(QuadSurface* surf,
                            cv::Mat_<uint8_t>& mask,
                            cv::Mat_<uint8_t>& img,
                            z5::Dataset* ds_high = nullptr,
                            z5::Dataset* ds_low = nullptr,
                            ChunkCache<uint8_t>* cache = nullptr);
