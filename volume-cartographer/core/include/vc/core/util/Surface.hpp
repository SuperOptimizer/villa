#pragma once
#include <filesystem>
#include <set>
#include <optional>

#include <opencv2/core.hpp> 
#include <nlohmann/json_fwd.hpp>
#include <z5/dataset.hxx>

#include "Slicing.hpp"


#define Z_DBG_GEN_PREFIX "auto_grown_"

#define SURF_LOAD_IGNORE_MASK 1
#define SURF_CHANNEL_NORESIZE 1

struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

struct DSReader
{
    z5::Dataset *ds;
    float scale;
    ChunkCache *cache;
};

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);



//base surface class
class Surface
{
public:
    virtual ~Surface();

    // get a central location point
    virtual cv::Vec3f pointer() = 0;

    //move pointer within internal coordinate system
    virtual void move(cv::Vec3f &ptr, const cv::Vec3f &offset) = 0;
    //does the pointer location contain valid surface data
    virtual bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //nominal pointer coordinates (in "output" coordinates)
    virtual cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    //read coord at pointer location, potentially with (3) offset
    virtual cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) = 0;
    virtual float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) = 0;
    //coordgenerator relative to ptr&offset
    //needs to be deleted after use
    virtual void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) = 0;
    nlohmann::json *meta = nullptr;
    std::filesystem::path path;
    std::string id;
};

class PlaneSurface : public Surface
{
public:
    //Surface API FIXME
    cv::Vec3f pointer() override;
    void move(cv::Vec3f &ptr, const cv::Vec3f &offset);
    bool valid(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override { return true; };
    cv::Vec3f loc(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f coord(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    cv::Vec3f normal(const cv::Vec3f &ptr, const cv::Vec3f &offset = {0,0,0}) override;
    float pointTo(cv::Vec3f &ptr, const cv::Vec3f &coord, float th, int max_iters = 1000) override { abort(); };

    PlaneSurface() {};
    PlaneSurface(cv::Vec3f origin_, cv::Vec3f normal_);

    void gen(cv::Mat_<cv::Vec3f> *coords, cv::Mat_<cv::Vec3f> *normals, cv::Size size, const cv::Vec3f &ptr, float scale, const cv::Vec3f &offset) override;

    float pointDist(cv::Vec3f wp);
    cv::Vec3f project(cv::Vec3f wp, float render_scale = 1.0, float coord_scale = 1.0);
    void setNormal(cv::Vec3f normal);
    void setOrigin(cv::Vec3f origin);
    cv::Vec3f origin();
    float scalarp(cv::Vec3f point) const;
    void setInPlaneRotation(float radians);
    float inPlaneRotation() const { return _inPlaneRotation; }
    cv::Vec3f basisX() const { return _vx; }
    cv::Vec3f basisY() const { return _vy; }
    void setAxisAlignedRotationKey(int key);
    int axisAlignedRotationKey() const { return _axisAlignedRotationKey; }
protected:
    void update();
    cv::Vec3f _normal = {0,0,1};
    cv::Vec3f _origin = {0,0,0};
    cv::Vec3f _vx = {1,0,0};
    cv::Vec3f _vy = {0,1,0};
    float _inPlaneRotation = 0.0f;
    cv::Matx33d _M;
    cv::Vec3d _T;
    int _axisAlignedRotationKey = -1;
};

// Options for writing TIFF from QuadSurface
struct TiffWriteOptions {
    enum class Compression { NONE, LZW, DEFLATE };
    enum class Predictor  { NONE, HORIZONTAL, FLOATINGPOINT };

    bool forceBigTiff = false;
    int  tileSize = 1024;                 // square tiles
    Compression compression = Compression::LZW;
    Predictor  predictor   = Predictor::FLOATINGPOINT; // for float32
};

//quads based surface class with a pointer implementing a nominal scale of 1 voxel
class QuadSurface : public Surface
{
public:
    cv::Vec3f pointer() override;
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


class SurfaceMeta
{
public:
    SurfaceMeta() {};
    SurfaceMeta(const std::filesystem::path &path_, const nlohmann::json &json);
    SurfaceMeta(const std::filesystem::path &path_);
    ~SurfaceMeta();
    void readOverlapping();
    QuadSurface *surface();
    void setSurface(QuadSurface *surf, bool takeOwnership = true);
    std::string name();
    std::filesystem::path path;
    QuadSurface *_surf = nullptr;
    bool _ownsSurface = false;
    Rect3D bbox;
    nlohmann::json *meta = nullptr;
    std::set<std::string> overlapping_str;
    std::set<SurfaceMeta*> overlapping;
    std::optional<std::filesystem::file_time_type> maskTimestamp() const { return maskTimestamp_; }
    static std::optional<std::filesystem::file_time_type> readMaskTimestamp(const std::filesystem::path& dir);

private:
    void cacheMaskTimestamp();
    std::optional<std::filesystem::file_time_type> maskTimestamp_;
};

QuadSurface *load_quad_from_tifxyz(const std::string &path, int flags = 0);
QuadSurface *regularized_local_quad(QuadSurface *src, const cv::Vec3f &ptr, int w, int h, int step_search = 100, int step_out = 5);

bool overlap(SurfaceMeta &a, SurfaceMeta &b, int max_iters = 1000);
bool contains(SurfaceMeta &a, const cv::Vec3f &loc, int max_iters = 1000);
bool contains(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs);
bool contains_any(SurfaceMeta &a, const std::vector<cv::Vec3f> &locs);


float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names);
std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);

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
                            ChunkCache* cache = nullptr);





void normalizeMaskChannel(cv::Mat& mask);
void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names);
std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path);
cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale);
cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale);
cv::Vec3f vx_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n);
cv::Vec3f vy_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n);
void vxy_from_normal(cv::Vec3f orig, cv::Vec3f normal, cv::Vec3f &vx, cv::Vec3f &vy);
cv::Vec3f rotateAroundAxis(const cv::Vec3f& vector, const cv::Vec3f& axis, float angle);
inline float sdist(const cv::Vec3f &a, const cv::Vec3f &b);
inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b);

float search_min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x);
float search_min_loc(const cv::Mat_<cv::Vec3d> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x);
float pointTo_(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo_(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);
float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale);

Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);
bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D rect_from_json(const nlohmann::json &json);

class MultiSurfaceIndex {
    struct Cell {
        std::vector<int> patch_indices;
    };

    std::unordered_map<uint64_t, Cell> grid;
    float cell_size;
    std::vector<Rect3D> patch_bboxes;

    uint64_t hash(int x, int y, int z) const;
public:
    MultiSurfaceIndex(float cell_sz = 100.0f);
    void addPatch(int idx, QuadSurface* patch);
    std::vector<int> getCandidatePatches(const cv::Vec3f& point, float tolerance = 0.0f) const;
    size_t getCellCount() const;
    size_t getPatchCount() const;
};