#include "CChunkedVolumeViewer.hpp"

#include "Constants.hpp"

#include "CState.hpp"
#include "elements/ViewerStatsBar.hpp"
#include "VCSettings.hpp"
#include "ViewerManager.hpp"
#include "overlays/SegmentationOverlayController.hpp"
#include "vc/core/render/Colormaps.hpp"
#include "vc/core/render/McVolumeArray.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/core/util/Surface.hpp"

#include <QApplication>
#include <QCursor>
#include <QElapsedTimer>
#include <QEvent>
#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QGraphicsPathItem>
#include <QGraphicsScene>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QSettings>
#include <QTransform>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <source_location>
#include <sstream>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

namespace {

using vc3d::kResizeSettleMs;
using vc3d::kIntersectionSettleMs;
using vc3d::kStatusRefreshMs;
using vc3d::msToTicks;
constexpr float kMinScale = 0.002f;
constexpr float kMaxScale = 128.0f;
constexpr float kResolutionLodZoomBias = 0.5f;
constexpr float kSegmentationResolutionLodZoomBias = 1.0f;
constexpr int kSurfaceResolutionLevelBias = 1;
constexpr int kInitialSegmentationSurfaceLevel = 5;
constexpr float kPanSmoothingAlpha = 0.65f;
constexpr int kSurfaceCellTileSize = 64;
constexpr std::array<QRgb, 12> kIntersectionPalette = {
    qRgb(255, 120, 120), qRgb(120, 200, 255), qRgb(120, 255, 140),
    qRgb(255, 220, 100), qRgb(220, 140, 255), qRgb(255, 160, 200),
    qRgb(140, 255, 220), qRgb(200, 255, 140), qRgb(255, 180, 120),
    qRgb(180, 200, 255), qRgb(255, 140, 180), qRgb(160, 255, 180),
};
constexpr int kIntersectionZ = 100;

constexpr int kHighlightedIntersectionZ = 110;
constexpr int kActiveIntersectionZ = 120;
constexpr float kActiveIntersectionOpacityScale = 1.2f;
constexpr float kActiveIntersectionWidthScale = 1.3f;
constexpr float kActiveIntersectionMinWidthDelta = 0.75f;
constexpr float kApprovalPlaneIntersectionScale = 1.5f;
constexpr float kFocusProjectionThreshold = 4.0f;

struct IntersectionStyle {
    QRgb color = 0;
    int z = kIntersectionZ;
    int widthQ = 0;

    bool operator==(const IntersectionStyle& other) const
    {
        return color == other.color && z == other.z && widthQ == other.widthQ;
    }
};

struct IntersectionStyleHash {
    size_t operator()(const IntersectionStyle& style) const
    {
        size_t h = std::hash<QRgb>{}(style.color);
        h ^= std::hash<int>{}(style.z) + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(style.widthQ) + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

int dominantAxis(const cv::Vec3f& v, float axisEps = 1e-4f)
{
    int axis = 0;
    float best = std::abs(v[0]);
    for (int i = 1; i < 3; ++i) {
        const float a = std::abs(v[i]);
        if (a > best) {
            best = a;
            axis = i;
        }
    }
    if (best < 1.0f - axisEps)
        return -1;
    for (int i = 0; i < 3; ++i) {
        if (i != axis && std::abs(v[i]) > axisEps)
            return -1;
    }
    return axis;
}

std::string stableHexHash(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : value) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

std::string normalizedVolumeCacheIdentity(const std::shared_ptr<Volume>& volume)
{
    if (!volume)
        return {};
    if (volume->isRemote()) {
        return "remote|" + volume->remoteUrl() +
               "|base=" + std::to_string(volume->baseScaleLevel()) +
               "|id=" + volume->id() +
               "|cache_schema=remote_sharded_ranges_v1";
    }

    std::error_code ec;
    auto path = std::filesystem::weakly_canonical(volume->path(), ec);
    if (ec)
        path = std::filesystem::absolute(volume->path(), ec);
    if (ec)
        path = volume->path();
    return "local|" + path.string() + "|id=" + volume->id();
}

uint32_t alphaBlendArgb(uint32_t base, uint32_t overlay, float alpha)
{
    const float a = std::clamp(alpha, 0.0f, 1.0f);
    const auto mix = [a](uint32_t b, uint32_t o) -> uint32_t {
        return static_cast<uint32_t>(
            std::clamp(std::lround((1.0f - a) * float(b) + a * float(o)), 0L, 255L));
    };
    const uint32_t br = (base >> 16) & 0xFFu;
    const uint32_t bg = (base >> 8) & 0xFFu;
    const uint32_t bb = base & 0xFFu;
    const uint32_t or_ = (overlay >> 16) & 0xFFu;
    const uint32_t og = (overlay >> 8) & 0xFFu;
    const uint32_t ob = overlay & 0xFFu;
    return 0xFF000000u | (mix(br, or_) << 16) | (mix(bg, og) << 8) | mix(bb, ob);
}

QColor activeSegmentationColorForView(const std::string& surfName)
{
    if (surfName == "seg yz" || surfName == "yz plane")
        return QColor(Qt::yellow);
    if (surfName == "seg xz" || surfName == "xz plane")
        return QColor(Qt::red);
    return QColor(255, 140, 0);
}

float activeSegmentationIntersectionWidth(float baseWidth)
{
    return std::max(baseWidth * kActiveIntersectionWidthScale,
                    baseWidth + kActiveIntersectionMinWidthDelta);
}

bool validSurfacePoint(const cv::Vec3f& p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
           p[0] != -1.0f && p[1] != -1.0f && p[2] != -1.0f;
}

bool finiteVec2(const cv::Vec2f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

bool finiteVec3(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

float distance3(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const cv::Vec3f d = a - b;
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

struct SafeFocusProjection {
    bool valid = false;
    const char* reason = "unchecked";
    float dist = -1.0f;
    cv::Vec3f ptr{NAN, NAN, NAN};
    cv::Vec3f loc{NAN, NAN, NAN};
    cv::Vec3f coord{NAN, NAN, NAN};
    cv::Vec2f grid{NAN, NAN};
    int row = -1;
    int col = -1;
    int nearestRow = -1;
    int nearestCol = -1;
};

SafeFocusProjection validateSafeFocusProjection(
    QuadSurface& surface,
    const cv::Vec3f& ptr,
    const cv::Vec3f* focusWorld,
    std::optional<float> pointToDistance = std::nullopt)
{
    SafeFocusProjection result;
    result.ptr = ptr;

    if (!finiteVec3(result.ptr)) {
        result.reason = "non_finite_ptr";
        return result;
    }

    result.loc = surface.loc(result.ptr);
    if (!finiteVec3(result.loc)) {
        result.reason = "non_finite_loc";
        return result;
    }

    if (pointToDistance) {
        result.dist = *pointToDistance;
        if (!std::isfinite(result.dist) ||
            result.dist < 0.0f ||
            result.dist >= kFocusProjectionThreshold) {
            result.reason = "distance_out_of_range";
            return result;
        }
    }

    result.coord = surface.coord(result.ptr);
    if (!validSurfacePoint(result.coord)) {
        result.reason = "invalid_coord";
        return result;
    }

    if (!pointToDistance) {
        result.dist = focusWorld ? distance3(result.coord, *focusWorld) : 0.0f;
        if (!std::isfinite(result.dist) ||
            result.dist < 0.0f ||
            result.dist >= kFocusProjectionThreshold) {
            result.reason = "distance_out_of_range";
            return result;
        }
    }

    result.grid = surface.ptrToGrid(result.ptr);
    if (!finiteVec2(result.grid)) {
        result.reason = "non_finite_grid";
        return result;
    }

    const cv::Mat_<cv::Vec3f>* points = surface.rawPointsPtr();
    if (!points || points->cols < 2 || points->rows < 2) {
        result.reason = "missing_surface_points";
        return result;
    }
    if (result.grid[0] <= 0.0f || result.grid[1] <= 0.0f ||
        result.grid[0] >= static_cast<float>(points->cols - 1) ||
        result.grid[1] >= static_cast<float>(points->rows - 1)) {
        result.reason = "grid_on_or_outside_border";
        return result;
    }

    result.col = static_cast<int>(std::floor(result.grid[0]));
    result.row = static_cast<int>(std::floor(result.grid[1]));
    if (!surface.isQuadValid(result.row, result.col)) {
        result.reason = "invalid_interpolation_quad";
        return result;
    }

    result.nearestCol = static_cast<int>(std::lround(result.grid[0]));
    result.nearestRow = static_cast<int>(std::lround(result.grid[1]));
    if (!surface.isPointValid(result.nearestRow, result.nearestCol)) {
        result.reason = "invalid_nearest_point";
        return result;
    }
    const cv::Vec3f normal = surface.gridNormal(result.nearestRow, result.nearestCol);
    if (!finiteVec3(normal)) {
        result.reason = "non_finite_grid_normal";
        return result;
    }

    result.valid = true;
    result.reason = "valid";
    return result;
}

std::string profileCaller(const std::source_location& caller)
{
    return std::format("{}:{} {}", caller.file_name(), caller.line(), caller.function_name());
}

std::string profileVec3(const cv::Vec3f& v)
{
    return std::format("[{:.3f},{:.3f},{:.3f}]", v[0], v[1], v[2]);
}

std::string profileRect(const cv::Rect& r)
{
    return std::format("[x={},y={},w={},h={}]", r.x, r.y, r.width, r.height);
}

std::string profileRectF(const QRectF& r)
{
    return std::format("[x={:.2f},y={:.2f},w={:.2f},h={:.2f}]",
                       r.x(), r.y(), r.width(), r.height());
}

// End-to-end latency trace (VC3D_LATENCY=1): event -> frame-flip wall-clock ms.
// Separate from --profile (which times internal phases); this answers "how long
// from the nav input to pixels on screen".
bool LatencyLoggingEnabled()
{
    static const bool on = [] {
        const char* e = std::getenv("VC3D_LATENCY");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}

qint64 monotonicNowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

class ProfileScope {
public:
    ProfileScope(const char* event,
                 const char* reason,
                 const std::source_location& caller)
        : event_(event ? event : "")
        , enabled_(ProfileLoggingEnabled())
    {
        if (!enabled_)
            return;
        reason_ = reason ? reason : "";
        caller_ = profileCaller(caller);
        timer_.start();
        Logger()->info("[vc3d-profile] {} begin reason='{}' caller='{}'",
                       event_, reason_, caller_);
    }

    ~ProfileScope()
    {
        finish();
    }

    bool enabled() const
    {
        return enabled_;
    }

    void setDetails(const char* details)
    {
        if (!enabled_)
            return;
        details_ = details ? details : "";
    }

    void setDetails(std::string details)
    {
        if (!enabled_)
            return;
        details_ = std::move(details);
    }

    void finish()
    {
        if (!enabled_ || finished_)
            return;
        finished_ = true;
        Logger()->info("[vc3d-profile] {} end elapsed_ms={} reason='{}' caller='{}'{}{}",
                       event_, timer_.elapsed(), reason_, caller_,
                       details_.empty() ? "" : " ", details_);
    }

private:
    const char* event_ = "";
    std::string reason_;
    std::string caller_;
    bool enabled_ = false;
    bool finished_ = false;
    QElapsedTimer timer_;
    std::string details_;
};

void applyPerPixelNormalOffset(cv::Mat_<cv::Vec3f>& coords,
                               const cv::Mat_<cv::Vec3f>& normals,
                               float zOff)
{
    if (zOff == 0.0f || coords.empty() || normals.empty())
        return;

    const int h = std::min(coords.rows, normals.rows);
    const int w = std::min(coords.cols, normals.cols);
    for (int y = 0; y < h; ++y) {
        auto* coordRow = coords.ptr<cv::Vec3f>(y);
        const auto* normalRow = normals.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            cv::Vec3f& p = coordRow[x];
            const cv::Vec3f& n = normalRow[x];
            if (!validSurfacePoint(p) ||
                !std::isfinite(n[0]) || !std::isfinite(n[1]) || !std::isfinite(n[2])) {
                continue;
            }
            p += n * zOff;
        }
    }
}

std::uint64_t surfaceTileKey(int tx, int ty)
{
    return (std::uint64_t(std::uint32_t(ty)) << 32) | std::uint32_t(tx);
}

std::uint64_t surfaceCellTileKey(int col, int row)
{
    const int tx = col >= 0 ? col / kSurfaceCellTileSize : (col - kSurfaceCellTileSize + 1) / kSurfaceCellTileSize;
    const int ty = row >= 0 ? row / kSurfaceCellTileSize : (row - kSurfaceCellTileSize + 1) / kSurfaceCellTileSize;
    return surfaceTileKey(tx, ty);
}

bool rectContains(const cv::Rect& outer, const cv::Rect& inner)
{
    return inner.x >= outer.x &&
           inner.y >= outer.y &&
           inner.x + inner.width <= outer.x + outer.width &&
           inner.y + inner.height <= outer.y + outer.height;
}

QString formatVec3(const cv::Vec3f& v)
{
    return QString("(%1, %2, %3)")
        .arg(v[0], 0, 'f', 1)
        .arg(v[1], 0, 'f', 1)
        .arg(v[2], 0, 'f', 1);
}

QString formatWholeVolumePosition(const cv::Vec3f& v)
{
    // v is in coordinate/UI order [x, y, z] = [width, height, slices]. Label each
    // component so the scroll Z (slice/depth) axis is unambiguous in the status bar.
    return QString("[X=%1, Y=%2, Z=%3]")
        .arg(v[0], 0, 'f', 0)
        .arg(v[1], 0, 'f', 0)
        .arg(v[2], 0, 'f', 0);
}

QString formatByteSize(std::size_t bytes)
{
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = kKiB * 1024.0;
    constexpr double kGiB = kMiB * 1024.0;
    const double value = static_cast<double>(bytes);
    if (value >= kGiB)
        return QString("%1 GB").arg(value / kGiB, 0, 'f', 2);
    if (value >= kMiB)
        return QString("%1 MB").arg(value / kMiB, 0, 'f', 1);
    if (value >= kKiB)
        return QString("%1 KB").arg(value / kKiB, 0, 'f', 1);
    return QString("%1 B").arg(bytes);
}

QString formatGigabytes(std::size_t bytes)
{
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    return QString("%1").arg(static_cast<double>(bytes) / kGiB, 0, 'f', 1);
}

QString formatMegabytesPerSecond(double bytesPerSecond)
{
    constexpr double kMiB = 1024.0 * 1024.0;
    return QString("%1 MB/s").arg(std::max(0.0, bytesPerSecond) / kMiB, 0, 'f', 1);
}

std::size_t streamingCacheCapacityBytes(const CState* state)
{
    constexpr std::size_t kFallbackCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    if (!state || state->cacheSizeBytes() == 0)
        return kFallbackCapacity;
    return state->cacheSizeBytes();
}

float scaleForSurfaceRenderStartLevel(int renderLevel, int numLevels)
{
    const int maxLevel = std::max(0, numLevels - 1);
    const int clampedRenderLevel = std::clamp(renderLevel, 0, maxLevel);
    int dsLevel = clampedRenderLevel + kSurfaceResolutionLevelBias;
    if (dsLevel > maxLevel)
        dsLevel = maxLevel;

    const float dsScale = static_cast<float>(std::uint64_t{1} << dsLevel);
    return std::clamp(0.75f / (dsScale * kResolutionLodZoomBias), kMinScale, kMaxScale);
}

float scaleForCoarsestPlaneRenderLevel(int numLevels)
{
    const int coarsestLevel = std::max(0, numLevels - 1);
    const float dsScale = static_cast<float>(std::uint64_t{1} << coarsestLevel);
    return std::clamp(0.75f / (dsScale * kResolutionLodZoomBias), kMinScale, kMaxScale);
}

float scaleForCoarsestSegmentationRenderLevel(int numLevels)
{
    const int coarsestLevel = std::max(0, numLevels - 1);
    const float dsScale = static_cast<float>(std::uint64_t{1} << coarsestLevel);
    return std::clamp(0.75f / (dsScale * kSegmentationResolutionLodZoomBias), kMinScale, kMaxScale);
}

std::filesystem::path remoteCacheRootForState(const CState* state)
{
    // Suggestion order: per-volpkg setting first (so projects with an
    // explicit cache stay co-located when no host mount is present), then
    // the user's persisted setting. remoteCachePath() ignores both when
    // /volpkgs or /ephemeral is mounted.
    QString suggestion;
    if (state && state->vpkg()) {
        suggestion = QString::fromStdString(state->vpkg()->remoteCacheRootOrEmpty()).trimmed();
    }
    if (suggestion.isEmpty()) {
        QSettings settings(vc3d::settingsFilePath(), QSettings::IniFormat);
        suggestion =
            settings.value(vc3d::settings::viewer::REMOTE_CACHE_DIR).toString();
    }
    return vc3d::remoteCachePath(suggestion).toStdString();
}

std::shared_ptr<vc::render::IChunkedArray> makeChunkCacheForVolume(const std::shared_ptr<Volume>& volume,
                                                                std::size_t decodedByteCapacity,
                                                                const CState* state)
{
    if (!volume)
        return nullptr;

    vc::render::DecodedCacheOptions options;
    options.decodedByteCapacity = decodedByteCapacity > 0
        ? decodedByteCapacity
        : streamingCacheCapacityBytes(nullptr);
    // The on-disk cache (a single volume.mca) is owned by Volume::createChunkCache, which
    // puts it under the volume's remoteCacheRoot. The GUI resolves that root
    // (/volpkgs|/ephemeral|settings); push it into the volume before the cache builds.
    if (volume->isRemote())
        volume->setRemoteCacheRoot(remoteCacheRootForState(state));

    return volume->createChunkCache(std::move(options));
}

std::shared_ptr<vc::render::IChunkedArray> sharedChunkCacheForVolume(const std::shared_ptr<Volume>& volume,
                                                                  std::size_t decodedByteCapacity,
                                                                  const CState* state)
{
    if (!volume)
        return nullptr;

    const std::size_t capacity = decodedByteCapacity > 0
        ? decodedByteCapacity
        : streamingCacheCapacityBytes(nullptr);
    const std::string key = normalizedVolumeCacheIdentity(volume) +
                            "|decoded=" + std::to_string(capacity) +
                            "|cache=" + (volume->isRemote() ? remoteCacheRootForState(state).string() : std::string{});

    static std::mutex cacheMutex;
    static std::unordered_map<std::string, std::weak_ptr<vc::render::IChunkedArray>> caches;

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = caches.find(key);
        if (it != caches.end()) {
            if (auto cache = it->second.lock())
                return cache;
            caches.erase(it);
        }
    }

    auto cache = makeChunkCacheForVolume(volume, capacity, state);
    if (!cache)
        return nullptr;

    std::lock_guard<std::mutex> lock(cacheMutex);
    auto& slot = caches[key];
    if (auto existing = slot.lock())
        return existing;
    slot = cache;
    return cache;
}

} // namespace

struct CChunkedVolumeViewer::GeneratedSurfaceCache {
    std::mutex mutex;
    bool valid = false;
    Surface* surface = nullptr;
    int fbW = 0;
    int fbH = 0;
    float scale = 0.0f;
    cv::Vec3f offset{0, 0, 0};
    float zOff = 0.0f;
    cv::Vec3f zOffWorldDir{0, 0, 0};
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
};

// Persistent pre-colormap sample buffer + tile bookkeeping for partial
// inter-frame rendering (renderFrame). Only the single-flight render worker
// mutates it; the mutex covers viewer teardown races.
struct CChunkedVolumeViewer::PartialFrameCache {
    std::mutex mutex;
    bool valid = false;
    // camera/settings the values buffer corresponds to (colormap/window are
    // post-sampling and deliberately excluded -- they re-blit, never re-sample)
    const void* array = nullptr;   // the chunk array sampled (volume identity)
    Surface* surface = nullptr;
    int fbW = 0, fbH = 0;
    float scale = 0.f, ptrX = 0.f, ptrY = 0.f, zOff = 0.f;
    cv::Vec3f zOffWorldDir{0, 0, 0};
    std::uint64_t sampleFp = 0;
    double tileMsEma = 0.0;        // measured per-tile render cost -> pass budget
    cv::Mat_<uint8_t> values;      // the sampled frame (pre-colormap)
    // per-tile: gen snapshot at last render, dirty flag, and the region bounding
    // box (picked LOD) the tile samples -- the data-staleness signature.
    struct TileBox { bool any = false; int z0 = 0, z1 = -1, y0 = 0, y1 = -1, x0 = 0, x1 = -1; };
    std::vector<std::uint64_t> tileGen;
    std::vector<uint8_t> tileDirty;
    std::vector<TileBox> tileBox;
};

// NaN test that survives -ffast-math (x!=x / std::isnan are deleted there).
static inline bool ccvNanBits(float f)
{
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

CChunkedVolumeViewer::CChunkedVolumeViewer(CState* state, ViewerManager* manager, QWidget* parent)
    : QWidget(parent)
    , _state(state)
    , _viewerManager(manager)
    , _genSurfaceCache(std::make_shared<GeneratedSurfaceCache>())
    , _partialFrame(std::make_shared<PartialFrameCache>())
{
    _view = new CVolumeViewerView(this);
    _view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _view->setTransformationAnchor(QGraphicsView::NoAnchor);
    _view->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    _view->setRenderHint(QPainter::Antialiasing, false);
    _view->setScrollPanDisabled(true);
    _view->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    connect(_view, &CVolumeViewerView::sendScrolled, this, &CChunkedVolumeViewer::onScrolled);
    connect(_view, &CVolumeViewerView::sendVolumeClicked, this, &CChunkedVolumeViewer::onVolumeClicked);
    connect(_view, &CVolumeViewerView::sendZoom, this, &CChunkedVolumeViewer::onZoom);
    connect(_view, &CVolumeViewerView::sendResized, this, &CChunkedVolumeViewer::onResized);
    connect(_view, &CVolumeViewerView::sendCursorMove, this, &CChunkedVolumeViewer::onCursorMove);
    connect(_view, &CVolumeViewerView::sendPanRelease, this, &CChunkedVolumeViewer::onPanRelease);
    connect(_view, &CVolumeViewerView::sendPanStart, this, &CChunkedVolumeViewer::onPanStart);
    connect(_view, &CVolumeViewerView::sendMousePress, this, &CChunkedVolumeViewer::onMousePress);
    connect(_view, &CVolumeViewerView::sendMouseMove, this, &CChunkedVolumeViewer::onMouseMove);
    connect(_view, &CVolumeViewerView::sendMouseRelease, this, &CChunkedVolumeViewer::onMouseRelease);
    connect(_view, &CVolumeViewerView::sendMouseLeftView, this, [this]() {
        clearLineAnnotationPlacementMarker();
    });
    connect(_view, &CVolumeViewerView::sendMouseDoubleClick, this,
            [this](QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers) {
                const auto cursorPos = cursorVolumePosition(scenePos);
                cv::Vec3f volumePos;
                if (cursorPos) {
                    volumePos = *cursorPos;
                } else {
                    volumePos = sceneToVolume(scenePos);
                }
                emit sendMouseDoubleClickVolume(volumePos, button, modifiers);
            });
    connect(_view, &CVolumeViewerView::sendKeyPress, this, &CChunkedVolumeViewer::onKeyPress);
    connect(_view, &CVolumeViewerView::sendKeyRelease, this, &CChunkedVolumeViewer::onKeyRelease);

    _scene = new QGraphicsScene(this);
    _scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    _view->setScene(_scene);
    _view->setDirectFramebuffer(&_framebuffer);

    // Render, status refresh, resize-settle and intersection rebuild are all
    // driven from the global render clock (ViewerManager::onGlobalTick) via
    // tickRender()/tickIdle() instead of four per-viewer QTimers.

    reloadPerfSettings();

    auto* layout = new QVBoxLayout;
    layout->addWidget(_view);
    setLayout(layout);

    _statsBar = new ViewerStatsBar(this);
    _statsBar->move(10, 5);
}

CChunkedVolumeViewer::~CChunkedVolumeViewer()
{
    quiesceForClose();
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
    clearIntersectionItems();
}

bool CChunkedVolumeViewer::eventFilter(QObject* watched, QEvent* event)
{
    if (event && event->type() == QEvent::Close && watched == parentWidget()) {
        quiesceForClose();
    }
    return QWidget::eventFilter(watched, event);
}

void CChunkedVolumeViewer::quiesceForClose()
{
    if (_closing) {
        return;
    }
    _closing = true;

    if (_viewerManager) {
        _viewerManager->unregisterViewer(this);
    }

    _intersectionTickCountdown = -1;
    _resizeSettleCountdown = -1;
    _renderPending = false;
    ++_renderSerial;

    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
}

void CChunkedVolumeViewer::reloadPerfSettings()
{
    QSettings s(vc3d::settingsFilePath(), QSettings::IniFormat);
    using namespace vc3d::settings;
    _panSensitivity = std::max(0.01f, s.value(viewer::PAN_SENSITIVITY, viewer::PAN_SENSITIVITY_DEFAULT).toFloat());
    _zoomSensitivity = std::max(0.01f, s.value(viewer::ZOOM_SENSITIVITY, viewer::ZOOM_SENSITIVITY_DEFAULT).toFloat());
    _zScrollSensitivity = std::max(0.01f, s.value(viewer::ZSCROLL_SENSITIVITY, viewer::ZSCROLL_SENSITIVITY_DEFAULT).toFloat());
    _voxelSizeOverrideUm = std::max(0.0, s.value(viewer::VOXEL_SIZE_UM, viewer::VOXEL_SIZE_UM_DEFAULT).toDouble());
    updateScalebarScale();   // override may have changed -> refresh the scalebar
    const int interpIdx = s.value(perf::INTERPOLATION_METHOD, 1).toInt();
    _samplingMethod = static_cast<vc::Sampling>(std::clamp(interpIdx, 0, 1));
    _maxDisplayedResolution = std::clamp(
        s.value(viewer::MAX_DISPLAYED_RESOLUTION, viewer::MAX_DISPLAYED_RESOLUTION_DEFAULT).toInt(),
        0,
        5);
}

void CChunkedVolumeViewer::setSurface(const std::string& name)
{
    if (_closing) {
        return;
    }
    _surfName = name;
    if (_state)
        onSurfaceChanged(name, _state->surface(name));
}

Surface* CChunkedVolumeViewer::currentSurface() const
{
    if (!_state) {
        auto shared = _surfWeak.lock();
        return shared ? shared.get() : nullptr;
    }
    return _state->surfaceRaw(_surfName);
}

CChunkedVolumeViewer::CameraState CChunkedVolumeViewer::cameraState() const
{
    CameraState state;
    state.surfacePtrX = _surfacePtrX;
    state.surfacePtrY = _surfacePtrY;
    state.scale = _scale;
    state.zOffset = _zOff;
    state.zOffsetWorldDir = _zOffWorldDir;
    return state;
}

void CChunkedVolumeViewer::applyCameraState(const CameraState& state, bool forceRender)
{
    if (_closing) {
        return;
    }
    _surfacePtrX = state.surfacePtrX;
    _surfacePtrY = state.surfacePtrY;
    _scale = state.scale;
    _zOff = state.zOffset;
    _zOffWorldDir = state.zOffsetWorldDir;
    recalcPyramidLevel();
    _genCacheDirty = true;
    if (forceRender) {
        renderVisible(true, "annotation camera state applied");
    } else {
        scheduleRender("annotation camera state applied");
    }
    emit overlaysUpdated();
}

bool CChunkedVolumeViewer::isRenderQuiescent() const
{
    return !_renderWorkerBusy.load(std::memory_order_acquire)
        && !_renderPending
        && _resizeSettleCountdown < 0;
}

std::size_t CChunkedVolumeViewer::chunkFetchesInFlight() const
{
    return _chunkArray ? _chunkArray->stats().remoteFetchesInFlight : 0;
}

void CChunkedVolumeViewer::rebuildChunkArray()
{
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    _chunkArray.reset();
    if (!_volume)
        return;

    try {
        _chunkArray = sharedChunkCacheForVolume(_volume, streamingCacheCapacityBytes(_state), _state);
    } catch (const std::exception& e) {
        if (_statsBar)
            _statsBar->setItems({QString("Streaming unavailable: %1").arg(e.what())});
        return;
    }

    if (!_chunkArray)
        return;

    // Chunk arrival does NOT schedule a render. In the tick model the global clock
    // renders every tick while downloads are in flight, so data that landed last
    // tick is reflected by the next tick's render without any per-chunk callback.
}

void CChunkedVolumeViewer::OnVolumeChanged(std::shared_ptr<Volume> vol)
{
    if (_closing) {
        return;
    }
    const bool hadVolume = static_cast<bool>(_volume);
    invalidateIntersect();
    if (_surfWeak.lock() == _defaultSurface) {
        _surfWeak.reset();
        _defaultSurface.reset();
    }
    _genCacheDirty = true;
    if (_genSurfaceCache) {
        std::lock_guard lock(_genSurfaceCache->mutex);
        _genSurfaceCache->valid = false;
        _genSurfaceCache->coords.release();
        _genSurfaceCache->normals.release();
    }
    _zOffWorldDir = {0, 0, 0};
    if (_cursorCrosshair)
        _cursorCrosshair->hide();
    clearLineAnnotationPlacementMarker();
    if (_focusMarker)
        _focusMarker->hide();

    _volume = std::move(vol);
    rebuildChunkArray();
    ensureDefaultSurface();
    if (_volume && isAxisAlignedView() && !hadVolume) {
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : static_cast<int>(_volume->numScales());
        _scale = scaleForCoarsestPlaneRenderLevel(n);
    }
    recalcPyramidLevel();   // also pushes the scalebar µm/px (updateScalebarScale)
    updateContentBounds();
    resizeFramebuffer();
    scheduleRender("volume changed");
    renderIntersections("volume changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::invalidateVis()
{
    if (_closing) {
        return;
    }
    _genCacheDirty = true;
}

void CChunkedVolumeViewer::invalidateVisRegion(const std::string&, const cv::Rect&)
{
    invalidateVis();
}

void CChunkedVolumeViewer::onSurfaceChanged(const std::string& name,
                                            const std::shared_ptr<Surface>& surf,
                                            bool isEditUpdate)
{
    if (_closing) {
        return;
    }
    const bool isCurrentSurface = (_surfName == name);
    const auto previousSurface = _surfWeak.lock();
    const bool isSameCurrentSurface = isCurrentSurface && previousSurface && previousSurface == surf;
    const bool isIntersectionTarget =
        _intersectTgts.count(name) != 0 ||
        (_intersectTgts.count("visible_segmentation") != 0 &&
         (name == "segmentation" || _highlightedSurfaceIds.count(name) != 0));

    if (!isCurrentSurface) {
        if (isIntersectionTarget) {
            if (!isEditUpdate) {
                invalidateIntersect(name);
                renderIntersections("intersection target surface changed");
            } else {
                _lastIntersectFp = {};
                if (_deferSegmentationIntersections && name == "segmentation" &&
                    dynamic_cast<PlaneSurface*>(currentSurface())) {
                    _intersectionGeometryCache = {};
                    _deferredSegmentationIntersectionsDirty = true;
                    return;
                }
                scheduleIntersectionRender("intersection target surface changed");
            }
        }
        return;
    }

    std::optional<cv::Vec3f> preservedViewCenter;
    if (!isEditUpdate && !_resetViewOnSurfaceChange && _surfName == "segmentation" &&
        surf && previousSurface && _view && !_framebuffer.isNull()) {
        const QPointF sceneCenter(static_cast<qreal>(_framebuffer.width()) * 0.5,
                                  static_cast<qreal>(_framebuffer.height()) * 0.5);
        preservedViewCenter = cursorVolumePosition(sceneCenter);
    }

    _surfWeak = surf;
    if (isSameCurrentSurface && isEditUpdate) {
        _genCacheDirty = true;
        _zOffWorldDir = {0, 0, 0};
        updateContentBounds();
        updateFocusMarker();
        if (_suppressNextSurfaceEditRender) {
            _suppressNextSurfaceEditRender = false;
        } else {
            scheduleRender("current surface edit update");
        }
        scheduleIntersectionRender("current surface edit update");
        return;
    }

    _genCacheDirty = true;
    _zOffWorldDir = {0, 0, 0};
    invalidateIntersect(name);
    if (!surf) {
        clearIntersectionItems();
        _scene->clear();
        _overlayGroups.clear();
        _cursorCrosshair = nullptr;
        _lineAnnotationPlacementMarker = nullptr;
        _focusMarker = nullptr;
        return;
    }
    updateContentBounds();
    const bool isSegmentationQuadSurface =
        _surfName == "segmentation" && dynamic_cast<QuadSurface*>(surf.get());

    auto setSegmentationPointerFromFocus = [&]() {
        auto* quad = dynamic_cast<QuadSurface*>(surf.get());
        auto* poi = _state ? _state->poi("focus") : nullptr;
        if (!quad || !poi) {
            return false;
        }
        if (!poi->surfaceId.empty() && !quad->id.empty() &&
            poi->surfaceId != quad->id && poi->surfaceId != _surfName) {
            return false;
        }

        if (poi->surfacePtr) {
            const SafeFocusProjection projected =
                validateSafeFocusProjection(*quad, *poi->surfacePtr, &poi->p);
            if (projected.valid) {
                _surfacePtrX = projected.loc[0];
                _surfacePtrY = projected.loc[1];
                return true;
            }
        }

        cv::Vec3f ptr = quad->pointer();
        const float dist = quad->pointTo(ptr, poi->p, kFocusProjectionThreshold, 100, nullptr);
        const SafeFocusProjection projected =
            validateSafeFocusProjection(*quad, ptr, &poi->p, dist);
        if (!projected.valid) {
            return false;
        }

        _surfacePtrX = projected.loc[0];
        _surfacePtrY = projected.loc[1];
        return true;
    };

    if (!isEditUpdate && isSegmentationQuadSurface && !_initializedFirstSegmentationSurface) {
        if (_resetViewOnSurfaceChange) {
            (void)setSegmentationPointerFromFocus();
        }
        _zOff = 0.0f;
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : (_volume ? static_cast<int>(_volume->numScales()) : 1);
        if (_resetViewOnSurfaceChange) {
            _scale = scaleForCoarsestSegmentationRenderLevel(n);
            recalcPyramidLevel();
        }
        _initializedFirstSegmentationSurface = true;
    } else if (!isEditUpdate && _resetViewOnSurfaceChange && isSegmentationQuadSurface) {
        (void)setSegmentationPointerFromFocus();
        _zOff = 0.0f;
        const int n = _chunkArray ? _chunkArray->numLevels()
                                  : (_volume ? static_cast<int>(_volume->numScales()) : 1);
        _scale = scaleForSurfaceRenderStartLevel(kInitialSegmentationSurfaceLevel, n);
        recalcPyramidLevel();
    } else if (preservedViewCenter) {
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f projected = plane->project(*preservedViewCenter, 1.0, 1.0);
            if (std::isfinite(projected[0]) && std::isfinite(projected[1])) {
                _surfacePtrX = projected[0];
                _surfacePtrY = projected[1];
            }
        } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
            cv::Vec3f ptr = quad->pointer();
            auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
            if (quad->pointTo(ptr, *preservedViewCenter, 4.0f, 100, patchIndex) >= 0.0f) {
                const cv::Vec3f loc = quad->loc(ptr);
                if (std::isfinite(loc[0]) && std::isfinite(loc[1])) {
                    _surfacePtrX = loc[0];
                    _surfacePtrY = loc[1];
                }
            }
        }
    }
    updateFocusMarker();
    scheduleRender("current surface changed");
    renderIntersections("current surface changed");
}

void CChunkedVolumeViewer::onSurfaceWillBeDeleted(const std::string&, const std::shared_ptr<Surface>& surf)
{
    if (_closing) {
        return;
    }
    auto current = _surfWeak.lock();
    if (current && current == surf)
        _surfWeak.reset();
}

void CChunkedVolumeViewer::onVolumeClosing()
{
    if (_closing) {
        return;
    }
    if (_chunkCbId != 0 && _chunkArray) {
        _chunkArray->removeChunkReadyListener(_chunkCbId);
        _chunkCbId = 0;
    }
    _chunkArray.reset();
    _volume.reset();
    invalidateIntersect();
    onSurfaceChanged(_surfName, nullptr);
}

void CChunkedVolumeViewer::onPOIChanged(const std::string& name, POI* poi)
{
    if (_closing) {
        return;
    }
    if (name != "focus" || !poi)
        return;
    if (property("vc_viewer_role").toString() == QStringLiteral("annotation")) {
        if (_focusMarker) {
            _focusMarker->hide();
        }
        return;
    }

    auto surf = _surfWeak.lock();
    const bool isPlaneSurface = dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        plane->setOrigin(poi->p);
        if (cv::norm(poi->n) > 0.5f)
            plane->setNormal(poi->n);
        updateContentBounds();
        _genCacheDirty = true;
    }

    updateFocusMarker(poi);
    updateStatusLabel();
    emit overlaysUpdated();
    scheduleRender("focus POI changed");
    if (!isPlaneSurface || !poi->suppressTransientPlaneIntersections)
        renderIntersections("focus POI changed");
}

void CChunkedVolumeViewer::ensureDefaultSurface()
{
    if (_surfWeak.lock() || !_volume || !isAxisAlignedView())
        return;
    const auto shape = _volume->shapeXyz();
    cv::Vec3f center(static_cast<float>(shape[0]) * 0.5f,
                     static_cast<float>(shape[1]) * 0.5f,
                     static_cast<float>(shape[2]) * 0.5f);
    cv::Vec3f normal;
    if (_surfName == "xy plane") normal = {0, 0, 1};
    else if (_surfName == "xz plane" || _surfName == "seg xz") normal = {0, 1, 0};
    else normal = {1, 0, 0};
    _defaultSurface = std::make_shared<PlaneSurface>(center, normal);
    _surfWeak = _defaultSurface;
}

bool CChunkedVolumeViewer::isAxisAlignedView() const
{
    return _surfName == "xy plane" || _surfName == "xz plane" ||
           _surfName == "yz plane" || _surfName == "seg xz" ||
           _surfName == "seg yz";
}

void CChunkedVolumeViewer::updateContentBounds()
{
    auto surf = _surfWeak.lock();
    if (!_volume || !surf)
        return;
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!plane)
        return;

    const auto [w, h, d] = _volume->shapeXyz();
    const float corners[][3] = {
        {0, 0, 0}, {float(w), 0, 0}, {0, float(h), 0}, {float(w), float(h), 0},
        {0, 0, float(d)}, {float(w), 0, float(d)}, {0, float(h), float(d)}, {float(w), float(h), float(d)}
    };
    _contentMinU = _contentMinV = std::numeric_limits<float>::max();
    _contentMaxU = _contentMaxV = std::numeric_limits<float>::lowest();
    for (const auto& c : corners) {
        const cv::Vec3f proj = plane->project({c[0], c[1], c[2]}, 1.0, 1.0);
        _contentMinU = std::min(_contentMinU, proj[0]);
        _contentMinV = std::min(_contentMinV, proj[1]);
        _contentMaxU = std::max(_contentMaxU, proj[0]);
        _contentMaxV = std::max(_contentMaxV, proj[1]);
    }
}

void CChunkedVolumeViewer::recalcPyramidLevel()
{
    const int n = _chunkArray ? _chunkArray->numLevels() : (_volume ? static_cast<int>(_volume->numScales()) : 1);
    const float lodZoomBias = _surfName == "segmentation"
        ? kSegmentationResolutionLodZoomBias
        : kResolutionLodZoomBias;
    const float lodScale = std::max(_scale * lodZoomBias, 1e-6f);
    _dsScaleIdx = std::clamp(
        static_cast<int>(std::floor(std::max(0.0f, std::log2(1.0f / lodScale)))),
        0, std::max(0, n - 1));
    _dsScale = static_cast<float>(std::uint64_t{1} << _dsScaleIdx);
    updateScalebarScale();
}

void CChunkedVolumeViewer::updateScalebarScale()
{
    // The scalebar overlay (CVolumeViewerView::drawForeground) needs µm per scene
    // pixel, where one scene pixel == one rendered framebuffer pixel (the framebuffer
    // is blitted 1:1, so the view transform m11 ~= 1 and does NOT carry the zoom).
    // The zoom + LOD live in the FRAMEBUFFER render, not the view transform:
    //   - 1 framebuffer px = 1/_scale render-LOD voxels (gen steps gridScale/_scale).
    //   - 1 render-LOD voxel = _dsScale level-0 voxels = _dsScale * voxelSize µm.
    // => µm/px = voxelSize * _dsScale / _scale. Must be recomputed on every zoom AND
    // LOD change (both happen here) -- the old code set it once at volume-load with
    // the wrong formula (voxelSize/_dsScale, zoom ignored), so the bar was wrong at
    // every zoom level.
    if (!_view || !_volume)
        return;
    // Voxel size (µm per level-0 voxel): prefer the volume's own metadata, but many
    // .vca archives don't carry one (voxelSize()==0). Fall back to the user-set
    // override (viewer/voxel_size_um in settings) so the scalebar still shows real
    // units. If neither is available the bar can't be physical -> leave the view's
    // default and the overlay shows nothing meaningful.
    double voxel = _volume->voxelSize();
    if (!(voxel > 0.0))
        voxel = _voxelSizeOverrideUm;                 // 0 if unset
    if (!(voxel > 0.0) || !(_scale > 0.0f))
        return;
    const double umPerScenePx =
        voxel * static_cast<double>(_dsScale) / static_cast<double>(_scale);
    _view->setVoxelSize(umPerScenePx, umPerScenePx);
}

void CChunkedVolumeViewer::resizeFramebuffer()
{
    const QSize vpSize = _view->viewport()->size();
    const int w = std::max(1, vpSize.width());
    const int h = std::max(1, vpSize.height());
    if (_framebuffer.isNull() || _framebuffer.width() != w || _framebuffer.height() != h) {
        _framebuffer = QImage(w, h, QImage::Format_RGB32);
        _framebuffer.fill(Qt::black);
    }
    _scene->setSceneRect(0, 0, w, h);
}

void CChunkedVolumeViewer::scheduleRender(const char* reason, std::source_location caller)
{
    if (_closing) {
        return;
    }
    if (ProfileLoggingEnabled()) {
        _pendingRenderReason = reason ? reason : "";
        _pendingRenderCaller = profileCaller(caller);
    }
    ProfileScope profile("scheduleRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' force=false pending={} scale={:.4f} zOff={:.3f}",
            _surfName, _renderPending, _scale, _zOff));
    }
    syncCameraTransform();
    // Latency trace: stamp the FIRST event that made this frame dirty (later
    // coalesced events keep the original origin so we measure worst-case
    // event->flip). Cleared when the frame flips.
    if (LatencyLoggingEnabled() && _latencyOriginNs < 0) {
        _latencyOriginNs = monotonicNowNs();
        _latencyOriginReason = reason;
    }
    _renderPending = true;
    _lastPredictKey = {};   // geometry/settings may have changed: re-predict
    // Route to the one global render clock: it coalesces all viewers' pending
    // flags into a single thaw->batch-apply->freeze->render pass per tick.
    if (_viewerManager) {
        _viewerManager->requestGlobalRender();
        profile.setDetails("action=request_global_render");
    } else {
        profile.setDetails("action=no_manager");
    }
}

// Called from ViewerManager::onGlobalTick (between the cache thaw and freeze):
// if this viewer has a pending change, dispatch its render worker.
void CChunkedVolumeViewer::tickRender(bool force)
{
    // force = the tick decided to render (e.g. streaming) even if this viewer had
    // no view change. Otherwise render only when this viewer is dirty.
    if (_closing || (!_renderPending && !force))
        return;
    // If last tick's render worker hasn't finished, SKIP this tick entirely and
    // leave the dirty flag so the next tick retries. The tick is the only
    // scheduler; a render never outlives its tick by queuing another.
    if (_renderWorkerBusy.load(std::memory_order_acquire)) {
        _renderPending = true;
        return;
    }
    // Hidden/minimized: keep the dirty flag; showEvent re-schedules on reveal.
    if (!isVisible())
        return;
    // Streaming-forced tick with an unchanged camera: skip when the data
    // generation OVER THIS VIEWER'S predicted working set hasn't moved since the
    // frame we last submitted -- nothing this viewport samples changed, so the
    // frame is provably identical. (The volume-global gen re-rendered every
    // viewer whenever a batch landed anywhere in the volume.)
    const std::uint64_t dataGen =
        _chunkArray ? _chunkArray->dataGenerationFor(_predictedRegions) : 0;
    if (!_renderPending && force && dataGen != 0 && dataGen == _lastRenderDataGen)
        return;
    _renderPending = false;
    _lastRenderDataGen = dataGen;
    submitRender("global tick");
    updateStatusLabel();
}

// Called every global tick. Advances the per-viewer debounce/idle deadlines and
// fires the associated work when each reaches zero. Replaces the old
// _statusTimer / _resizeRenderTimer / _intersectionRenderTimer.
void CChunkedVolumeViewer::tickIdle()
{
    if (_closing)
        return;

    // Status refresh: periodic, runs even when idle (background prefetch updates
    // the disk/RAM figures). Re-arm each cycle.
    if (--_statusTickCountdown <= 0) {
        _statusTickCountdown = msToTicks(kStatusRefreshMs);
        updateStatusLabel();
    }

    // Resize-settle: one-shot. When it lands, kick a real render.
    if (_resizeSettleCountdown >= 0 && --_resizeSettleCountdown < 0) {
        scheduleRender("resize settled");
        emit overlaysUpdated();
    }

    // Intersection rebuild: one-shot debounce.
    if (_intersectionTickCountdown >= 0 && --_intersectionTickCountdown < 0) {
        renderIntersections("intersection deadline");
        emit overlaysUpdated();
    }
}

void CChunkedVolumeViewer::showEvent(QShowEvent* event)
{
    QWidget::showEvent(event);
    // tickRender skips hidden viewers; render whatever went stale while hidden.
    scheduleRender("shown");
}

void CChunkedVolumeViewer::requestRender(const char* reason, std::source_location caller)
{
    if (reason && std::string_view(reason) == "push/pull active viewer refresh") {
        _suppressNextSurfaceEditRender = true;
    }
    scheduleRender(reason, caller);
}

void CChunkedVolumeViewer::scheduleIntersectionRender(const char* reason, std::source_location caller)
{
    if (_closing) {
        return;
    }
    if (ProfileLoggingEnabled()) {
        _pendingIntersectionReason = reason ? reason : "";
        _pendingIntersectionCaller = profileCaller(caller);
    }
    ProfileScope profile("scheduleIntersectionRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format("surf='{}' targets={}", _surfName, _intersectTgts.size()));
    }
    if (_deferSegmentationIntersections && dynamic_cast<PlaneSurface*>(currentSurface())) {
        _deferredSegmentationIntersectionsDirty = true;
        profile.setDetails("action=deferred_segmentation_edit");
        return;
    }
    if (_intersectionTickCountdown < 0) {
        _intersectionTickCountdown = msToTicks(kIntersectionSettleMs);
        if (_viewerManager)
            _viewerManager->requestGlobalRender();
        profile.setDetails("action=intersection_deadline_set");
    } else {
        profile.setDetails("action=already_scheduled");
    }
}

void CChunkedVolumeViewer::setSegmentationIntersectionDeferral(bool active)
{
    if (_deferSegmentationIntersections == active) {
        return;
    }

    _deferSegmentationIntersections = active;
    if (!active && _deferredSegmentationIntersectionsDirty) {
        _deferredSegmentationIntersectionsDirty = false;
        scheduleIntersectionRender("deferred segmentation intersection refresh");
    }
}

void CChunkedVolumeViewer::syncCameraTransform()
{
    _camSurfX = _surfacePtrX;
    _camSurfY = _surfacePtrY;
    _camScale = _scale;
    updateFocusMarker();
}

int CChunkedVolumeViewer::renderStartLevel(bool preferSurfaceResolution) const
{
    if (!_chunkArray)
        return 0;

    // `_dsScaleIdx` intentionally waits for about 2x more zoom before moving
    // to a finer level. Surface-resolution views keep their target level to
    // avoid panning blur.
    int level = _dsScaleIdx;
    if (preferSurfaceResolution && _chunkArray && level < _chunkArray->numLevels() - 1)
        level -= kSurfaceResolutionLevelBias;
    level = std::max(level, _maxDisplayedResolution);
    return std::clamp(level, 0, _chunkArray->numLevels() - 1);
}

struct CChunkedVolumeViewer::RenderContext {
    std::uint64_t serial = 0;
    int fbW = 0;
    int fbH = 0;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    float zOff = 0.0f;
    cv::Vec3f zOffWorldDir{0, 0, 0};
    int startLevel = 0;
    vc::Sampling samplingMethod = vc::Sampling::Trilinear;
    CompositeRenderSettings compositeSettings;
    float windowLow = 0.0f;
    float windowHigh = 255.0f;
    std::string baseColormapId;
    std::shared_ptr<Surface> surf;
    std::shared_ptr<vc::render::IChunkedArray> chunkArray;
    std::shared_ptr<Volume> overlayVolume;
    std::shared_ptr<vc::render::IChunkedArray> overlayChunkArray;
    float overlayOpacity = 0.0f;
    std::string overlayColormapId;
    float overlayWindowLow = 0.0f;
    float overlayWindowHigh = 255.0f;
    std::shared_ptr<GeneratedSurfaceCache> genCache;
    bool genCacheDirty = false;
    std::shared_ptr<PartialFrameCache> partialCache;
    std::string profileReason;
    std::string profileCaller;
};

struct CChunkedVolumeViewer::RenderResult {
    std::uint64_t serial = 0;
    QImage framebuffer;
    float surfacePtrX = 0.0f;
    float surfacePtrY = 0.0f;
    float scale = 1.0f;
    qint64 renderFrameElapsedMs = 0;
    int tilesRemaining = 0;        // partial refine: dirty tiles for next tick
};

CChunkedVolumeViewer::RenderResult CChunkedVolumeViewer::renderFrame(RenderContext ctx)
{
    // Always time the whole frame so the latency trace (VC3D_LATENCY) can report
    // worker cost even without --profile. Cheap: one QElapsedTimer.
    QElapsedTimer renderTimer;
    renderTimer.start();
    // Sub-phase timing (only meaningful under --profile). gen = surface coords,
    // sample = chunk sample/decode, blit = colormap LUT + framebuffer write.
    const bool profilePhases = ProfileLoggingEnabled();
    qint64 phaseGenMs = -1, phaseSampleMs = 0, phaseBlitMs = 0;
    bool phaseGenCached = false;
    QElapsedTimer phaseTimer;
    if (ProfileLoggingEnabled()) {
        Logger()->info("[vc3d-profile] renderFrame begin reason='{}' caller='{}' serial={} surf='{}' size={}x{} level={} interactive={} overlay={} composite={} planeComposite={}",
                       ctx.profileReason, ctx.profileCaller, ctx.serial,
                       ctx.surf ? ctx.surf->id : std::string(""),
                       ctx.fbW, ctx.fbH, ctx.startLevel, false,
                       bool(ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f),
                       ctx.compositeSettings.enabled, ctx.compositeSettings.planeEnabled);
    }
    RenderResult result;
    result.serial = ctx.serial;
    result.surfacePtrX = ctx.surfacePtrX;
    result.surfacePtrY = ctx.surfacePtrY;
    result.scale = ctx.scale;
    result.framebuffer = QImage(std::max(1, ctx.fbW), std::max(1, ctx.fbH), QImage::Format_RGB32);
    result.framebuffer.fill(Qt::black);

    auto finishRenderFrameProfile = [&]() {
        result.renderFrameElapsedMs = renderTimer.elapsed();
        if (!ProfileLoggingEnabled())
            return;
        Logger()->info("[vc3d-profile] renderFrame end elapsed_ms={} gen_ms={} genCached={} sample_ms={} blit_ms={} reason='{}' caller='{}' serial={} framebuffer={}x{}",
                       result.renderFrameElapsedMs, phaseGenMs, phaseGenCached,
                       phaseSampleMs, phaseBlitMs, ctx.profileReason, ctx.profileCaller,
                       result.serial, result.framebuffer.width(), result.framebuffer.height());
    };

    if (!ctx.surf || !ctx.chunkArray || ctx.fbW <= 0 || ctx.fbH <= 0) {
        finishRenderFrameProfile();
        return result;
    }

    cv::Mat_<uint8_t> values(ctx.fbH, ctx.fbW, uint8_t(0));
    cv::Mat_<uint8_t> overlayValues;
    const bool planeView = dynamic_cast<PlaneSurface*>(ctx.surf.get()) != nullptr;

    // mc_render composite params from VC's settings. comp=NONE -> a slice; an
    // enabled composite maps method -> reduction and the layer counts to a
    // [t0,t1] slab in LOD-0 voxels. plane and quad use their own layer/enabled
    // fields. mc owns every supported method now.
    const bool composite = planeView ? ctx.compositeSettings.planeEnabled
                                     : ctx.compositeSettings.enabled;
    const bool wantComposite = composite;
    int comp = 0; // MC_COMP_NONE
    if (wantComposite) {
        const std::string& m = ctx.compositeSettings.params.method;
        if (m == "min")            comp = 1;
        else if (m == "mean")      comp = 2;
        else if (m == "max")       comp = 3;
        else if (m == "alpha")     comp = 4;
        else if (m == "stddev")    comp = 5;
        else if (m == "shaded")    comp = 6;
        else if (m == "percentile")comp = 7;
        else if (m == "depth")     comp = 8;
        else if (m == "ink")       comp = 9;
    }
    const int front  = planeView ? ctx.compositeSettings.planeLayersFront
                                 : ctx.compositeSettings.layersFront;
    const int behind = planeView ? ctx.compositeSettings.planeLayersBehind
                                 : ctx.compositeSettings.layersBehind;
    const float zStep = ctx.compositeSettings.reverseDirection ? -1.0f : 1.0f;
    const float t0 = wantComposite ? float(-std::max(0, behind)) * zStep : 0.0f;
    const float t1 = wantComposite ? float( std::max(0, front )) * zStep : 0.0f;
    const float voxPerPixel = ctx.scale > 0.0f ? 1.0f / ctx.scale : 1.0f;

    // surf->gen() builds the W*H world-coord grid (and normals when compositing
    // or offset along z); mc_render samples/composites it directly. genCache
    // memoizes the grid across repaints with identical geometry.
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    const cv::Vec3f offset(ctx.surfacePtrX * ctx.scale - float(ctx.fbW) * 0.5f,
                           ctx.surfacePtrY * ctx.scale - float(ctx.fbH) * 0.5f,
                           0.0f);
    const bool needSurfaceNormals = ctx.zOff != 0.0f || comp != 0;

    bool genCacheHit = false;
    if (ctx.genCache) {
        std::lock_guard lock(ctx.genCache->mutex);
        if (ctx.genCacheDirty) {
            ctx.genCache->valid = false;
            ctx.genCache->coords.release();
            ctx.genCache->normals.release();
        }
        genCacheHit =
            ctx.genCache->valid &&
            ctx.genCache->surface == ctx.surf.get() &&
            ctx.genCache->fbW == ctx.fbW &&
            ctx.genCache->fbH == ctx.fbH &&
            ctx.genCache->scale == ctx.scale &&
            ctx.genCache->offset == offset &&
            ctx.genCache->zOff == ctx.zOff &&
            ctx.genCache->zOffWorldDir == ctx.zOffWorldDir &&
            !ctx.genCache->coords.empty() &&
            (!needSurfaceNormals || !ctx.genCache->normals.empty());
        if (genCacheHit) {
            coords = ctx.genCache->coords;
            if (needSurfaceNormals)
                normals = ctx.genCache->normals;
        }
    }

    phaseGenCached = genCacheHit;
    if (!genCacheHit) {
        if (profilePhases) phaseTimer.restart();
        ctx.surf->gen(&coords, needSurfaceNormals ? &normals : nullptr,
                      cv::Size(ctx.fbW, ctx.fbH), {0, 0, 0}, ctx.scale, offset);
        applyPerPixelNormalOffset(coords, normals, ctx.zOff);
        if (profilePhases) phaseGenMs = phaseTimer.elapsed();

        if (ctx.genCache && !coords.empty()) {
            std::lock_guard lock(ctx.genCache->mutex);
            ctx.genCache->valid = true;
            ctx.genCache->surface = ctx.surf.get();
            ctx.genCache->fbW = ctx.fbW;
            ctx.genCache->fbH = ctx.fbH;
            ctx.genCache->scale = ctx.scale;
            ctx.genCache->offset = offset;
            ctx.genCache->zOff = ctx.zOff;
            ctx.genCache->zOffWorldDir = ctx.zOffWorldDir;
            ctx.genCache->coords = coords;
            ctx.genCache->normals = normals;
        }
    }

    // QuadSurface::gen() returns coords/normals as a cropped ROI view into a
    // padded (w+8) buffer, so the Mat is NOT continuous; mc_render reads a flat
    // w*h*3 array and would shear per row. Clone to continuous when needed.
    if (!coords.empty() && !coords.isContinuous())
        coords = coords.clone();
    if (!normals.empty() && !normals.isContinuous())
        normals = normals.clone();

    // SHADED(6)/PERCENTILE(7) knobs -> mc; null for other modes (mc uses defaults).
    const auto& cp = ctx.compositeSettings.params;
    vc::render::McVolumeArray::ShadeParams shade{
        cp.lightZ, cp.lightY, cp.lightX,
        cp.ambient, cp.diffuse, cp.specular, cp.shininess,
        cp.absorption, cp.shadow, cp.sss, cp.curvature,
        cp.percentile, cp.transmission, cp.inkLockVox};

    auto renderArray = [&](vc::render::IChunkedArray& array,
                           cv::Mat_<uint8_t>& dst, int rcomp) {
        auto* mc = dynamic_cast<vc::render::McVolumeArray*>(&array);
        if (!mc || coords.empty())
            return;
        const bool wantShade = (rcomp == 6 || rcomp == 7 || rcomp == 9);
        mc->render(reinterpret_cast<const float*>(coords.ptr<cv::Vec3f>(0)),
                   (rcomp != 0 && !normals.empty())
                       ? reinterpret_cast<const float*>(normals.ptr<cv::Vec3f>(0))
                       : nullptr,
                   ctx.fbW, ctx.fbH, rcomp, t0, t1, 1.0f,
                   ctx.compositeSettings.params.alphaMin,
                   ctx.compositeSettings.params.alphaOpacity,
                   voxPerPixel, dst.ptr<uint8_t>(0),
                   wantShade ? &shade : nullptr);
    };

    if (profilePhases) phaseTimer.restart();

    // ---- partial inter-frame rendering -------------------------------------
    // Persistent pre-colormap sample buffer + 64px tile grid. Re-render only
    // what changed: same camera -> region-gen-stale tiles; integer-pixel pan ->
    // blit the overlap and render the exposed strips; zoom / z-scroll /
    // geometry scroll -> seed with the previous frame as an instant preview and
    // refine tiles center-out under a per-pass budget (the remainder continues
    // next tick). Window/colormap changes never reach here (LUT-only, below).
    auto* mcArr = dynamic_cast<vc::render::McVolumeArray*>(ctx.chunkArray.get());
    PartialFrameCache* pf = ctx.partialCache.get();
    const bool overlayActive =
        ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f;
    constexpr int TS = 64;                     // tile edge (px)
    const bool partialOk = pf && mcArr && !overlayActive && !coords.empty();

    // Sampling fingerprint: everything that changes sampled VALUES other than
    // the camera (colormap/window are post-sampling and excluded).
    std::uint64_t fp = 0x9E3779B97F4A7C15ull ^ std::uint64_t(comp + 1);
    auto fpAdd = [&fp](float v) {
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        fp = (fp ^ u) * 0x100000001B3ull;
    };
    fpAdd(t0); fpAdd(t1);
    fpAdd(ctx.compositeSettings.params.alphaMin);
    fpAdd(ctx.compositeSettings.params.alphaOpacity);
    fpAdd(shade.lightZ); fpAdd(shade.lightY); fpAdd(shade.lightX);
    fpAdd(shade.ambient); fpAdd(shade.diffuse); fpAdd(shade.specular);
    fpAdd(shade.shininess); fpAdd(shade.absorption); fpAdd(shade.shadow);
    fpAdd(shade.sss); fpAdd(shade.curvature); fpAdd(shade.percentile);
    fpAdd(shade.transmission); fpAdd(shade.inkLock);

    if (!partialOk) {
        renderArray(*ctx.chunkArray, values, comp);
        if (pf) {                              // overlay on / non-mc array: cache off
            std::lock_guard lk(pf->mutex);
            pf->valid = false;
        }
    } else {
        std::lock_guard lk(pf->mutex);
        const int tilesX = (ctx.fbW + TS - 1) / TS;
        const int tilesY = (ctx.fbH + TS - 1) / TS;
        const int nTiles = tilesX * tilesY;
        const int level = mcArr->pickLevel(voxPerPixel);

        enum class Mode { Full, Data, Pan, Seed };
        Mode mode = Mode::Full;
        int panDx = 0, panDy = 0;
        bool seedWarp = false;
        const bool baseMatch = pf->valid && pf->surface == ctx.surf.get() &&
                               pf->array == ctx.chunkArray.get() &&
                               pf->fbW == ctx.fbW && pf->fbH == ctx.fbH &&
                               pf->sampleFp == fp && !pf->values.empty();
        if (baseMatch) {
            const bool geomStable = !ctx.genCacheDirty;
            const bool zSame = pf->zOff == ctx.zOff && pf->zOffWorldDir == ctx.zOffWorldDir;
            const bool camSame = pf->scale == ctx.scale &&
                                 pf->ptrX == ctx.surfacePtrX && pf->ptrY == ctx.surfacePtrY;
            if (geomStable && zSame && camSame) {
                mode = Mode::Data;
            } else if (geomStable && zSame && pf->scale == ctx.scale) {
                const float fdx = (ctx.surfacePtrX - pf->ptrX) * ctx.scale;
                const float fdy = (ctx.surfacePtrY - pf->ptrY) * ctx.scale;
                panDx = int(std::lround(fdx));
                panDy = int(std::lround(fdy));
                const bool integral = std::abs(fdx - float(panDx)) < 1e-3f &&
                                      std::abs(fdy - float(panDy)) < 1e-3f;
                const bool overlaps =
                    std::abs(panDx) < ctx.fbW && std::abs(panDy) < ctx.fbH;
                mode = (integral && overlaps) ? Mode::Pan : Mode::Seed;
            } else {
                mode = Mode::Seed;             // zoom / z-scroll / geometry scroll
                seedWarp = pf->scale != ctx.scale;
            }
        }

        if (pf->values.rows != ctx.fbH || pf->values.cols != ctx.fbW) {
            pf->values.create(ctx.fbH, ctx.fbW);
            pf->values.setTo(uint8_t(0));
            mode = Mode::Full;
        }
        if (int(pf->tileGen.size()) != nTiles) {
            pf->tileGen.assign(size_t(nTiles), 0);
            pf->tileDirty.assign(size_t(nTiles), uint8_t(1));
            pf->tileBox.assign(size_t(nTiles), PartialFrameCache::TileBox{});
        }

        // gen snapshot BEFORE sampling: anything landing during this pass stamps
        // a larger gen and re-dirties its tiles next pass.
        const std::uint64_t genSnap = mcArr->dataGeneration();

        // Per-tile region bounding boxes at the picked level: stride-4 walk of
        // the cached coords (a picked-level region spans >=128px on screen, so a
        // 4px stride cannot skip one), dilated +/-1 region so composite slabs /
        // trilinear edges can't slip past the signature. Rebuilt every pass
        // (~0.1ms) since coords may be new.
        {
            const int sh = 8 + level;          // 256<<level L0 voxels per region
            for (int ty = 0; ty < tilesY; ++ty)
                for (int tx = 0; tx < tilesX; ++tx) {
                    auto& b = pf->tileBox[size_t(ty) * tilesX + tx];
                    b = PartialFrameCache::TileBox{};
                    const int py0 = ty * TS, py1 = std::min(ctx.fbH, py0 + TS);
                    const int px0 = tx * TS, px1 = std::min(ctx.fbW, px0 + TS);
                    for (int y = py0; y < py1; y += 4) {
                        const cv::Vec3f* row = coords[y];
                        for (int x = px0; x < px1; x += 4) {
                            const cv::Vec3f& p = row[x];
                            if (ccvNanBits(p[0]) || ccvNanBits(p[1]) || ccvNanBits(p[2]) ||
                                p[0] < 0.f || p[1] < 0.f || p[2] < 0.f)
                                continue;
                            const int rz = int(p[2]) >> sh;
                            const int ry = int(p[1]) >> sh;
                            const int rx = int(p[0]) >> sh;
                            if (!b.any) {
                                b.z0 = b.z1 = rz; b.y0 = b.y1 = ry; b.x0 = b.x1 = rx;
                                b.any = true;
                            } else {
                                b.z0 = std::min(b.z0, rz); b.z1 = std::max(b.z1, rz);
                                b.y0 = std::min(b.y0, ry); b.y1 = std::max(b.y1, ry);
                                b.x0 = std::min(b.x0, rx); b.x1 = std::max(b.x1, rx);
                            }
                        }
                    }
                    if (b.any) {               // slab / trilinear margin
                        b.z0 = std::max(0, b.z0 - 1); b.z1 += 1;
                        b.y0 = std::max(0, b.y0 - 1); b.y1 += 1;
                        b.x0 = std::max(0, b.x0 - 1); b.x1 += 1;
                    }
                }
        }

        auto markDataStale = [&]() {
            for (int t = 0; t < nTiles; ++t) {
                if (pf->tileDirty[size_t(t)])
                    continue;
                const auto& b = pf->tileBox[size_t(t)];
                if (b.any && mcArr->dataGenerationForBox(level, b.z0, b.z1,
                                                         b.y0, b.y1, b.x0, b.x1) >
                                 pf->tileGen[size_t(t)])
                    pf->tileDirty[size_t(t)] = 1;
            }
        };

        switch (mode) {
        case Mode::Full:
            renderArray(*ctx.chunkArray, pf->values, comp);
            std::fill(pf->tileGen.begin(), pf->tileGen.end(), genSnap);
            std::fill(pf->tileDirty.begin(), pf->tileDirty.end(), uint8_t(0));
            break;
        case Mode::Data:
            markDataStale();
            break;
        case Mode::Pan: {
            // content shifts opposite the camera: new(x,y) = old(x+dx, y+dy)
            cv::Mat_<uint8_t> shifted(ctx.fbH, ctx.fbW, uint8_t(0));
            const int sx0 = std::max(0, panDx), sy0 = std::max(0, panDy);
            const int dx0 = std::max(0, -panDx), dy0 = std::max(0, -panDy);
            const int cw = ctx.fbW - std::abs(panDx), ch = ctx.fbH - std::abs(panDy);
            pf->values(cv::Rect(sx0, sy0, cw, ch))
                .copyTo(shifted(cv::Rect(dx0, dy0, cw, ch)));
            shifted.copyTo(pf->values);
            // a NEW tile fully sourced from clean old tiles inherits min(gen);
            // anything touching the exposed strips is dirty.
            std::vector<std::uint64_t> oldGen = pf->tileGen;
            std::vector<uint8_t> oldDirty = pf->tileDirty;
            for (int ty = 0; ty < tilesY; ++ty)
                for (int tx = 0; tx < tilesX; ++tx) {
                    const int t = ty * tilesX + tx;
                    const int nx0 = tx * TS, nx1 = std::min(ctx.fbW, nx0 + TS) - 1;
                    const int ny0 = ty * TS, ny1 = std::min(ctx.fbH, ny0 + TS) - 1;
                    const int ox0 = nx0 + panDx, ox1 = nx1 + panDx;
                    const int oy0 = ny0 + panDy, oy1 = ny1 + panDy;
                    if (ox0 < 0 || oy0 < 0 || ox1 >= ctx.fbW || oy1 >= ctx.fbH) {
                        pf->tileDirty[size_t(t)] = 1;
                        pf->tileGen[size_t(t)] = 0;
                        continue;
                    }
                    std::uint64_t g = ~0ull;
                    bool dirty = false;
                    for (int sy = oy0 / TS; sy <= oy1 / TS; ++sy)
                        for (int sx = ox0 / TS; sx <= ox1 / TS; ++sx) {
                            const int st = sy * tilesX + sx;
                            dirty = dirty || oldDirty[size_t(st)] != 0;
                            g = std::min(g, oldGen[size_t(st)]);
                        }
                    pf->tileDirty[size_t(t)] = dirty ? 1 : 0;
                    pf->tileGen[size_t(t)] = dirty ? 0 : g;
                }
            markDataStale();
            break;
        }
        case Mode::Seed: {
            if (seedWarp) {
                // zoom preview: map the old frame onto the new camera (nearest).
                // dst -> src: src = (dst + offNew) * scaleOld/scaleNew - offOld
                const float sw = pf->scale / ctx.scale;
                const float offXn = ctx.surfacePtrX * ctx.scale - float(ctx.fbW) * 0.5f;
                const float offYn = ctx.surfacePtrY * ctx.scale - float(ctx.fbH) * 0.5f;
                const float offXo = pf->ptrX * pf->scale - float(ctx.fbW) * 0.5f;
                const float offYo = pf->ptrY * pf->scale - float(ctx.fbH) * 0.5f;
                cv::Matx23f m(sw, 0.f, offXn * sw - offXo,
                              0.f, sw, offYn * sw - offYo);
                cv::Mat seeded;
                cv::warpAffine(pf->values, seeded, m, pf->values.size(),
                               cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                               cv::BORDER_CONSTANT, cv::Scalar(0));
                seeded.copyTo(pf->values);
            }
            // z-scroll / geometry scroll: the old frame stands as the preview.
            std::fill(pf->tileGen.begin(), pf->tileGen.end(), 0);
            std::fill(pf->tileDirty.begin(), pf->tileDirty.end(), uint8_t(1));
            break;
        }
        }

        // Refine: render dirty tiles center-out under a per-pass budget, in ONE
        // stacked mc render call (full SIMD/thread parallelism, no per-tile
        // spawn). The remainder continues next tick (finishRender re-arms).
        if (mode != Mode::Full) {
            std::vector<int> dirtyTiles;
            dirtyTiles.reserve(size_t(nTiles));
            for (int t = 0; t < nTiles; ++t)
                if (pf->tileDirty[size_t(t)])
                    dirtyTiles.push_back(t);
            if (!dirtyTiles.empty()) {
                const float cxp = float(ctx.fbW) * 0.5f, cyp = float(ctx.fbH) * 0.5f;
                auto d2 = [&](int t) {
                    const float ddx = (float(t % tilesX) + 0.5f) * TS - cxp;
                    const float ddy = (float(t / tilesX) + 0.5f) * TS - cyp;
                    return ddx * ddx + ddy * ddy;
                };
                std::sort(dirtyTiles.begin(), dirtyTiles.end(),
                          [&](int a, int b2) { return d2(a) < d2(b2); });
                constexpr double kBudgetMs = 50.0;
                int budget = nTiles;
                if (pf->tileMsEma > 0.0)
                    budget = std::clamp(int(kBudgetMs / pf->tileMsEma), 48, nTiles);
                const int K = std::min<int>(budget, int(dirtyTiles.size()));

                const bool wantN = comp != 0 && !normals.empty();
                cv::Mat_<cv::Vec3f> tp(K * TS, TS, cv::Vec3f(-1.f, -1.f, -1.f));
                cv::Mat_<cv::Vec3f> tn;
                if (wantN) {
                    tn.create(K * TS, TS);
                    tn.setTo(cv::Scalar(0.f, 0.f, 0.f));
                }
                for (int k = 0; k < K; ++k) {
                    const int t = dirtyTiles[size_t(k)];
                    const int px0 = (t % tilesX) * TS, py0 = (t / tilesX) * TS;
                    const int w = std::min(TS, ctx.fbW - px0);
                    const int h = std::min(TS, ctx.fbH - py0);
                    coords(cv::Rect(px0, py0, w, h))
                        .copyTo(tp(cv::Rect(0, k * TS, w, h)));
                    if (wantN)
                        normals(cv::Rect(px0, py0, w, h))
                            .copyTo(tn(cv::Rect(0, k * TS, w, h)));
                }
                cv::Mat_<uint8_t> tout(K * TS, TS, uint8_t(0));
                QElapsedTimer batchTimer;
                batchTimer.start();
                const bool wantShade = (comp == 6 || comp == 7 || comp == 9);
                mcArr->render(reinterpret_cast<const float*>(tp.ptr<cv::Vec3f>(0)),
                              wantN ? reinterpret_cast<const float*>(tn.ptr<cv::Vec3f>(0))
                                    : nullptr,
                              TS, K * TS, comp, t0, t1, 1.0f,
                              ctx.compositeSettings.params.alphaMin,
                              ctx.compositeSettings.params.alphaOpacity,
                              voxPerPixel, tout.ptr<uint8_t>(0),
                              wantShade ? &shade : nullptr);
                const double perTile = double(batchTimer.elapsed()) / std::max(1, K);
                pf->tileMsEma = pf->tileMsEma > 0.0
                                    ? pf->tileMsEma * 0.8 + perTile * 0.2
                                    : perTile;
                for (int k = 0; k < K; ++k) {
                    const int t = dirtyTiles[size_t(k)];
                    const int px0 = (t % tilesX) * TS, py0 = (t / tilesX) * TS;
                    const int w = std::min(TS, ctx.fbW - px0);
                    const int h = std::min(TS, ctx.fbH - py0);
                    tout(cv::Rect(0, k * TS, w, h))
                        .copyTo(pf->values(cv::Rect(px0, py0, w, h)));
                    pf->tileGen[size_t(t)] = genSnap;
                    pf->tileDirty[size_t(t)] = 0;
                }
            }
        }
        for (int t = 0; t < nTiles; ++t)
            result.tilesRemaining += pf->tileDirty[size_t(t)] ? 1 : 0;

        pf->valid = true;
        pf->array = ctx.chunkArray.get();
        pf->surface = ctx.surf.get();
        pf->fbW = ctx.fbW;
        pf->fbH = ctx.fbH;
        pf->scale = ctx.scale;
        pf->ptrX = ctx.surfacePtrX;
        pf->ptrY = ctx.surfacePtrY;
        pf->zOff = ctx.zOff;
        pf->zOffWorldDir = ctx.zOffWorldDir;
        pf->sampleFp = fp;
        values = pf->values;                   // the LUT blit below reads this
    }
    if (profilePhases) phaseSampleMs += phaseTimer.elapsed();

    if (ctx.overlayChunkArray && ctx.overlayVolume && ctx.overlayOpacity > 0.0f) {
        overlayValues.create(ctx.fbH, ctx.fbW);
        overlayValues.setTo(0);
        renderArray(*ctx.overlayChunkArray, overlayValues, 0);
    }

    // INK: stroke-scale local-contrast band-pass at blit time, over the FULL
    // frame (partial tile updates land in `values`; running the band-pass here
    // keeps it seamless across tile boundaries). Applied to a copy -- the
    // persistent sample buffer must stay raw.
    if (comp == 9 && ctx.compositeSettings.params.inkGain > 0.0f &&
        !values.empty() && values.isContinuous()) {
        values = values.clone();
        mc_image_dog(values.ptr<uint8_t>(0), ctx.fbW, ctx.fbH,
                     ctx.compositeSettings.params.inkScaleVox * ctx.scale,
                     ctx.compositeSettings.params.inkGain);
    }

    if (profilePhases) phaseTimer.restart();
    std::array<uint32_t, 256> lut{};
    mc_colormap_lut(lut.data(), ctx.windowLow, ctx.windowHigh,
                    mc_colormap_id(ctx.baseColormapId.c_str()));
    std::array<uint32_t, 256> overlayLut{};
    const bool hasOverlay = !overlayValues.empty() && ctx.overlayOpacity > 0.0f;
    if (hasOverlay) {
        mc_colormap_lut(overlayLut.data(), ctx.overlayWindowLow, ctx.overlayWindowHigh,
                        mc_colormap_id(ctx.overlayColormapId.c_str()));
    }
    auto* fbBits = reinterpret_cast<uint32_t*>(result.framebuffer.bits());
    const int fbStride = result.framebuffer.bytesPerLine() / 4;
    // mc_render emits 0 for any no-material point: real air, confirmed-zero, or
    // not-yet-resident data. All of it is black -- lut[0] is forced black, so the
    // LUT maps 0 -> black directly (no gray "uncovered" placeholder).
    for (int y = 0; y < ctx.fbH; ++y) {
        auto* row = fbBits + size_t(y) * size_t(fbStride);
        const auto* src = values.ptr<uint8_t>(y);
        const auto* overlaySrc = hasOverlay ? overlayValues.ptr<uint8_t>(y) : nullptr;
        for (int x = 0; x < ctx.fbW; ++x) {
            uint32_t pixel = lut[src[x]];
            if (hasOverlay && overlaySrc[x] &&
                overlaySrc[x] >= ctx.overlayWindowLow && overlaySrc[x] <= ctx.overlayWindowHigh) {
                pixel = alphaBlendArgb(pixel, overlayLut[overlaySrc[x]], ctx.overlayOpacity);
            }
            row[x] = pixel;
        }
    }
    if (profilePhases) phaseBlitMs = phaseTimer.elapsed();
    finishRenderFrameProfile();
    return result;
}

// Predict ONLY the 256^3 regions the render will actually sample THIS frame.
// Done by running the surface's own gen() at a DOWNSAMPLED resolution over the
// exact same viewport the render uses -- so we get the in-viewport coords (gen
// clips to the screen rect; a warped sheet's off-screen patch is excluded), map
// each to its 16^3 block, and collapse the distinct blocks to their 256^3 regions.
// Coarse grid (~fb/8) = a few thousand points, cheap. The set is small because the
// rendered slice is thin -- not the whole padded control grid (which sprayed ~46k
// regions across the volume and flooded the download stack).
static constexpr int kPredictDownsample = 8;   // gen at 1/8 res for prediction
static constexpr std::size_t kMaxPredictedRegions = 4096;   // sanity cap

void CChunkedVolumeViewer::predictWorkingSet(std::vector<vc::render::ChunkKey>& out)
{
    // Stash THIS viewer's contribution: tickRender's viewport-local data-gen
    // skip checks against exactly the regions this viewer will sample. The set
    // is a pure function of geometry+viewport -- reuse it while neither changed
    // (the downsampled gen() is the cost; the request blast below stays, since
    // mc dedups present/in-flight/queued regions in O(1)).
    auto surf = _surfWeak.lock();
    const PredictKey key{surf.get(), _framebuffer.width(), _framebuffer.height(),
                         _scale, _surfacePtrX, _surfacePtrY};
    if (!(key == _lastPredictKey) || key.surf == nullptr) {
        _predictedRegions.clear();
        predictWorkingSetInto(_predictedRegions);
        _lastPredictKey = key;
    }
    out.insert(out.end(), _predictedRegions.begin(), _predictedRegions.end());
}

void CChunkedVolumeViewer::predictWorkingSetInto(std::vector<vc::render::ChunkKey>& out) const
{
    if (_closing || !_chunkArray)
        return;
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    const int level = renderStartLevel(plane == nullptr);
    const auto cs = _chunkArray->chunkShape(level);   // (z,y,x) voxels per chunk (16^3)
    const auto ls = _chunkArray->shape(level);        // (z,y,x) level dims
    if (cs[0] <= 0 || cs[1] <= 0 || cs[2] <= 0)
        return;
    const auto s0 = _chunkArray->levelTransform(level).scaleFromLevel0;  // L0 voxel -> this LOD

    const int fbW = _framebuffer.width()  > 0 ? _framebuffer.width()  : 1024;
    const int fbH = _framebuffer.height() > 0 ? _framebuffer.height() : 1024;
    const int D = kPredictDownsample;
    const int gw = std::max(1, fbW / D), gh = std::max(1, fbH / D);
    const float genScale = _scale / float(D);
    // Same offset formula the render uses, at the downsampled fb size.
    const cv::Vec3f offset(_surfacePtrX * genScale - float(gw) * 0.5f,
                           _surfacePtrY * genScale - float(gh) * 0.5f, 0.0f);

    cv::Mat_<cv::Vec3f> coords;
    surf->gen(&coords, nullptr, cv::Size(gw, gh), {0, 0, 0}, genScale, offset);
    if (coords.empty())
        return;

    // 256^3 region = 16 chunks of 16^3 per axis. ChunkKey carries 16^3-BLOCK coords;
    // McVolumeArray::prefetchChunks divides by 16 to get the region index, so we
    // emit the region's corner BLOCK = region*16. Dedup is at REGION granularity.
    // (Cross-viewer + inflight/present dedup is NOT done here: the tick dedups the
    // shared set across viewers, and mc_volume_request_region itself no-ops on
    // present/in-flight/already-queued regions -- mc owns the pipeline state.)
    const int regBlocks = 256 / 16;   // 16^3 blocks per region axis
    std::unordered_set<std::uint64_t> seenRegion;
    for (int y = 0; y < coords.rows; ++y) {
        const cv::Vec3f* row = coords[y];
        for (int x = 0; x < coords.cols; ++x) {
            const cv::Vec3f& p = row[x];
            if (ccvNanBits(p[0]) || ccvNanBits(p[1]) || ccvNanBits(p[2]) ||
                p[0] < 0.f || p[1] < 0.f || p[2] < 0.f)
                continue;   // invalid / hole (NaN check survives -ffast-math)
            // L0 voxel -> this-LOD voxel -> 16^3 block -> 256^3 region.
            const int lz = int(p[2] * float(s0[0])), ly = int(p[1] * float(s0[1])), lx = int(p[0] * float(s0[2]));
            if (unsigned(lz) >= unsigned(ls[0]) || unsigned(ly) >= unsigned(ls[1]) || unsigned(lx) >= unsigned(ls[2]))
                continue;
            const int rz = (lz / cs[0]) / regBlocks, ry = (ly / cs[1]) / regBlocks, rx = (lx / cs[2]) / regBlocks;
            const std::uint64_t k = (std::uint64_t(rz) << 42) ^ (std::uint64_t(ry) << 21) ^ std::uint64_t(rx);
            if (seenRegion.insert(k).second) {
                out.push_back({level, rz * regBlocks, ry * regBlocks, rx * regBlocks});  // region -> corner block
                if (seenRegion.size() >= kMaxPredictedRegions)
                    return;
            }
        }
    }
}

void CChunkedVolumeViewer::submitRender(const char* reason, std::source_location caller)
{
    ProfileScope profile("submitRender", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' volume={} chunkArray={} busy={}",
            _surfName, bool(_volume), bool(_chunkArray),
            _renderWorkerBusy.load(std::memory_order_acquire)));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }

    auto surf = _surfWeak.lock();
    if (!surf || !_volume || !_chunkArray) {
        profile.setDetails("action=skip missing_input");
        return;
    }

    resizeFramebuffer();
    const int fbW = _framebuffer.width();
    const int fbH = _framebuffer.height();
    if (fbW <= 0 || fbH <= 0) {
        if (profile.enabled()) {
            profile.setDetails(std::format("action=skip invalid_framebuffer size={}x{}", fbW, fbH));
        }
        return;
    }

    // One render worker at a time. If one is in flight, skip THIS tick and re-arm
    // the dirty flag: the next tick retries. No queue, no completion self-schedule
    // -- the tick is the only scheduler. (A worker takes <~1 tick normally; if it
    // runs long, ticks coalesce into one retry rather than piling up renders.)
    if (_renderWorkerBusy.exchange(true, std::memory_order_acq_rel)) {
        _renderPending = true;
        profile.setDetails("action=skip_worker_busy_retry_next_tick");
        return;
    }

    _chunkArray->beginViewRequest();
    if (_overlayChunkArray)
        _overlayChunkArray->beginViewRequest();


    RenderContext ctx;
    ctx.serial = ++_renderSerial;
    ctx.fbW = fbW;
    ctx.fbH = fbH;
    ctx.surfacePtrX = _surfacePtrX;
    ctx.surfacePtrY = _surfacePtrY;
    ctx.scale = _scale;
    ctx.zOff = _zOff;
    ctx.zOffWorldDir = _zOffWorldDir;
    ctx.startLevel = renderStartLevel(dynamic_cast<PlaneSurface*>(surf.get()) == nullptr);
    ctx.samplingMethod = _samplingMethod;
    ctx.compositeSettings = _compositeSettings;
    ctx.windowLow = _windowLow;
    ctx.windowHigh = _windowHigh;
    ctx.baseColormapId = _baseColormapId;
    ctx.surf = std::move(surf);
    ctx.chunkArray = _chunkArray;
    ctx.overlayVolume = _overlayVolume;
    ctx.overlayChunkArray = _overlayChunkArray;
    ctx.overlayOpacity = _overlayOpacity;
    ctx.overlayColormapId = _overlayColormapId;
    ctx.overlayWindowLow = _overlayWindowLow;
    ctx.overlayWindowHigh = _overlayWindowHigh;
    ctx.genCache = _genSurfaceCache;
    ctx.genCacheDirty = _genCacheDirty;
    ctx.partialCache = _partialFrame;
    if (ProfileLoggingEnabled()) {
        ctx.profileReason = reason ? reason : "";
        ctx.profileCaller = profileCaller(caller);
    }
    _genCacheDirty = false;
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=worker_start serial={} size={}x{} level={}",
            ctx.serial, ctx.fbW, ctx.fbH, ctx.startLevel));
    }

    QPointer<CChunkedVolumeViewer> guard(this);
    (void)QtConcurrent::run([guard, ctx = std::move(ctx)]() mutable {
        auto result = std::make_shared<RenderResult>(renderFrame(std::move(ctx)));
        QMetaObject::invokeMethod(qApp, [guard, result = std::move(result)]() mutable {
            if (guard)
                guard->finishRenderOnMainThread(std::move(result));
        }, Qt::QueuedConnection);
    });
}

void CChunkedVolumeViewer::finishRenderOnMainThread(std::shared_ptr<RenderResult> result)
{
    ProfileScope profile("finishRenderOnMainThread", "render worker finished",
                         std::source_location::current());
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "result={} serial={} currentSerial={}",
            bool(result), result ? result->serial : 0, _renderSerial));
    }
    _renderWorkerBusy.store(false, std::memory_order_release);
    if (_closing) {
        profile.setDetails("action=drop_closing");
        return;
    }
    if (!result || result->serial != _renderSerial) {
        if (LatencyLoggingEnabled()) {
            // Superseded frame: its user-event origin is unmeasurable. Clear it so a
            // LATER flip doesn't report this stale stamp (bogus multi-second values).
            _latencyOriginNs = -1;
            _latencyOriginReason = nullptr;
        }
        profile.setDetails("action=drop_stale_result");
        return;
    }

    _framebuffer = std::move(result->framebuffer);
    syncCameraTransform();
    if (LatencyLoggingEnabled() && _latencyOriginNs >= 0) {
        const double ms = (monotonicNowNs() - _latencyOriginNs) / 1e6;
        Logger()->info("[vc3d-latency] surf='{}' event->flip {:.1f}ms origin='{}' worker={:.1f}ms",
                       _surfName, ms,
                       _latencyOriginReason ? _latencyOriginReason : "",
                       double(result->renderFrameElapsedMs));
        _latencyOriginNs = -1;
        _latencyOriginReason = nullptr;
    }
    scheduleIntersectionRender("stable render finished");
    emit overlaysUpdated();
    _view->viewport()->update();
    updateStatusLabel();
    if (result->tilesRemaining > 0 && _viewerManager) {
        // partial refine: more dirty tiles -- continue next tick. Set the dirty
        // flag directly (NOT scheduleRender: the camera is unchanged, so the
        // predict memo and latency-origin stamps must survive).
        _renderPending = true;
        _viewerManager->requestGlobalRender();
    }

    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=display serial={} worker_elapsed_ms={} framebuffer={}x{}",
            result->serial, result->renderFrameElapsedMs,
            _framebuffer.width(), _framebuffer.height()));
    }
}

void CChunkedVolumeViewer::renderVisible(bool force, const char* reason, std::source_location caller)
{
    ProfileScope profile("renderVisible", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' force={} pending={} scale={:.4f} zOff={:.3f}",
            _surfName, force, _renderPending, _scale, _zOff));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    if (!force) {
        scheduleRender("renderVisible non-force", caller);
        profile.setDetails("action=schedule");
        return;
    }
    _renderPending = false;
    submitRender("renderVisible force", caller);
    updateStatusLabel();
    profile.setDetails("action=submit");
}

void CChunkedVolumeViewer::setVolumeWindow(float low, float high)
{
    if (_closing) {
        return;
    }
    const float clampedLow = std::clamp(low, 0.0f, 255.0f);
    float clampedHigh = std::clamp(high, 0.0f, 255.0f);
    if (clampedHigh <= clampedLow)
        clampedHigh = std::min(255.0f, clampedLow + 1.0f);
    if (std::abs(_windowLow - clampedLow) < 1e-6f &&
        std::abs(_windowHigh - clampedHigh) < 1e-6f)
        return;
    _windowLow = clampedLow;
    _windowHigh = clampedHigh;
    scheduleRender("volume window changed");
}

void CChunkedVolumeViewer::setOverlayVolume(std::shared_ptr<Volume> volume)
{
    if (_closing) {
        return;
    }
    if (_overlayChunkCbId != 0 && _overlayChunkArray) {
        _overlayChunkArray->removeChunkReadyListener(_overlayChunkCbId);
        _overlayChunkCbId = 0;
    }
    _overlayVolume = std::move(volume);
    _overlayChunkArray.reset();
    if (_overlayVolume) {
        try {
            _overlayChunkArray = sharedChunkCacheForVolume(_overlayVolume, streamingCacheCapacityBytes(_state), _state);
        } catch (const std::exception&) {
            _overlayChunkArray.reset();
        }
        // Overlay chunk arrival, like base, does NOT schedule a render: the tick
        // renders every tick while downloads are in flight.
    }
    scheduleRender("overlay volume changed");
}

void CChunkedVolumeViewer::setOverlayOpacity(float opacity)
{
    if (_closing) {
        return;
    }
    _overlayOpacity = std::clamp(opacity, 0.0f, 1.0f);
    scheduleRender("overlay opacity changed");
}

void CChunkedVolumeViewer::setOverlayColormap(const std::string& colormapId)
{
    if (_closing) {
        return;
    }
    _overlayColormapId = colormapId;
    scheduleRender("overlay colormap changed");
}

void CChunkedVolumeViewer::setOverlayThreshold(float threshold)
{
    if (_closing) {
        return;
    }
    setOverlayWindow(std::max(threshold, 0.0f), _overlayWindowHigh);
}

void CChunkedVolumeViewer::setOverlayWindow(float low, float high)
{
    if (_closing) {
        return;
    }
    _overlayWindowLow = std::clamp(low, 0.0f, 255.0f);
    _overlayWindowHigh = std::clamp(high, _overlayWindowLow + 1.0f, 255.0f);
    scheduleRender("overlay window changed");
}

void CChunkedVolumeViewer::panByF(float dx, float dy)
{
    const float invScale = _panSensitivity / _scale;
    _surfacePtrX -= dx * invScale;
    _surfacePtrY -= dy * invScale;
    if (_contentMaxU > _contentMinU) {
        _surfacePtrX = std::clamp(_surfacePtrX, _contentMinU, _contentMaxU);
        _surfacePtrY = std::clamp(_surfacePtrY, _contentMinV, _contentMaxV);
    }
    scheduleRender("pan");
    refreshSameWrapAnnotationOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::zoomStepsAt(int steps, const QPointF& scenePos)
{
    if (steps == 0)
        return;
    const float factor = std::pow(1.05f, static_cast<float>(steps) * _zoomSensitivity);
    const float newScale = std::clamp(_scale * factor, kMinScale, kMaxScale);
    if (std::abs(newScale - _scale) < _scale * 1e-6f)
        return;
    const float vpW = static_cast<float>(_view->viewport()->width());
    const float vpH = static_cast<float>(_view->viewport()->height());
    const float mx = static_cast<float>(scenePos.x());
    const float my = static_cast<float>(scenePos.y());
    if (mx >= 0 && mx < vpW && my >= 0 && my < vpH) {
        const float dx = mx - vpW * 0.5f;
        const float dy = my - vpH * 0.5f;
        _surfacePtrX += dx * (1.0f / _scale - 1.0f / newScale);
        _surfacePtrY += dy * (1.0f / _scale - 1.0f / newScale);
    }
    _scale = newScale;
    recalcPyramidLevel();
    _genCacheDirty = true;
    resizeFramebuffer();
    scheduleRender("zoom");
    refreshSameWrapAnnotationOverlay();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::adjustZoomByFactor(float factor)
{
    const int steps = (factor > 1.0f) ? 1 : (factor < 1.0f ? -1 : 0);
    zoomStepsAt(steps, QPointF(_view->viewport()->width() * 0.5, _view->viewport()->height() * 0.5));
}

void CChunkedVolumeViewer::notifyInteractiveViewChange(double motionPx)
{
    if (!_volume || !_chunkArray)
        return;

    _genCacheDirty = true;
    scheduleRender("interactive view change final render");
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::adjustSurfaceOffset(float delta)
{
    float maxZ = 10000.0f;
    if (_volume) {
        const auto [w, h, d] = _volume->shapeXyz();
        maxZ = static_cast<float>(std::max({w, h, d}));
    }
    _zOff = std::clamp(_zOff + delta, -maxZ, maxZ);
    _genCacheDirty = true;
    scheduleRender("surface offset changed");
    updateStatusLabel();
}

void CChunkedVolumeViewer::resetSurfaceOffsets()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _zOff = 0.0f;
    _zOffWorldDir = {0, 0, 0};
    _genCacheDirty = true;
    scheduleRender("surface offsets reset");
}

void CChunkedVolumeViewer::fitSurfaceInView()
{
    _surfacePtrX = 0.0f;
    _surfacePtrY = 0.0f;
    _scale = 0.5f;
    recalcPyramidLevel();
    _genCacheDirty = true;
    scheduleRender("fit surface in view");
}

void CChunkedVolumeViewer::centerOnVolumePoint(const cv::Vec3f& point, bool forceRender)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    cv::Vec2f surfacePoint(0.0f, 0.0f);
    bool haveSurfacePoint = false;
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f projected = plane->project(point, 1.0, 1.0);
        surfacePoint = {projected[0], projected[1]};
        haveSurfacePoint = true;
    } else if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (quad->pointTo(ptr, point, 4.0f, 100, patchIndex) >= 0.0f) {
            const cv::Vec3f loc = quad->loc(ptr);
            surfacePoint = {loc[0], loc[1]};
            haveSurfacePoint = true;
        }
    }

    if (!haveSurfacePoint ||
        !std::isfinite(surfacePoint[0]) ||
        !std::isfinite(surfacePoint[1])) {
        return;
    }

    centerOnSurfacePoint(surfacePoint, forceRender);
}

void CChunkedVolumeViewer::centerOnSurfacePoint(const cv::Vec2f& point, bool forceRender)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]))
        return;

    _surfacePtrX = point[0];
    _surfacePtrY = point[1];
    _genCacheDirty = true;
    if (forceRender) {
        renderVisible(true, "center on surface point");
    } else {
        scheduleRender("center on surface point");
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onZoom(int steps, QPointF scenePoint, Qt::KeyboardModifiers modifiers)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return;
    if (modifiers & Qt::ShiftModifier) {
        if (_shiftScrollOverride && _shiftScrollOverride(steps, scenePoint, modifiers)) {
            return;
        }
        if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
            const cv::Vec3f normal = plane->normal({0, 0, 0});
            if (std::isfinite(normal[0]) && std::isfinite(normal[1]) &&
                std::isfinite(normal[2]) && cv::norm(normal) > 0.0f) {
                const float delta = static_cast<float>(steps) * _zScrollSensitivity;
                auto shiftedPlane = std::make_shared<PlaneSurface>(*plane);
                shiftedPlane->setOrigin(plane->origin() + normal * (delta + _zOff));
                _zOff = 0.0f;
                _zOffWorldDir = {0, 0, 0};
                if (_state) {
                    _state->setSurface(_surfName, shiftedPlane, false, true);
                } else {
                    _defaultSurface = shiftedPlane;
                    _surfWeak = _defaultSurface;
                    updateContentBounds();
                    _genCacheDirty = true;
                    scheduleRender("plane slice mouse wheel");
                }
            }
        } else {
            _zOff += static_cast<float>(steps) * _zScrollSensitivity;
            _genCacheDirty = true;
            scheduleRender("z offset mouse wheel");
        }
    } else if (modifiers & Qt::ControlModifier) {
        emit sendSegmentationRadiusWheel(steps, scenePoint, sceneToVolume(scenePoint));
    } else {
        zoomStepsAt(steps > 0 ? 1 : (steps < 0 ? -1 : 0), scenePoint);
    }
}

void CChunkedVolumeViewer::onResized()
{
    if (_closing) {
        return;
    }
    resizeFramebuffer();
    _genCacheDirty = true;
    // Cancel any pending immediate render; coalesce into a resize-settle deadline
    // so a drag-resize doesn't re-render every frame.
    _renderPending = false;
    _resizeSettleCountdown = msToTicks(kResizeSettleMs);
    if (_viewerManager)
        _viewerManager->requestGlobalRender();
    _view->viewport()->update();
}

void CChunkedVolumeViewer::onCursorMove(QPointF scenePos)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_viewerManager) {
        _viewerManager->broadcastLinkedCursor(this, _lastCursorVolumePos);
    }
    if (!_isPanning)
        return;
    const float dx = static_cast<float>(scenePos.x() - _lastPanSceneF.x());
    const float dy = static_cast<float>(scenePos.y() - _lastPanSceneF.y());
    _lastPanSceneF = scenePos;
    if (std::abs(dx) > 0.001f || std::abs(dy) > 0.001f) {
        if (!_panSmoothingInitialized) {
            _smoothedPanDx = dx;
            _smoothedPanDy = dy;
            _panSmoothingInitialized = true;
        } else {
            _smoothedPanDx = kPanSmoothingAlpha * dx + (1.0f - kPanSmoothingAlpha) * _smoothedPanDx;
            _smoothedPanDy = kPanSmoothingAlpha * dy + (1.0f - kPanSmoothingAlpha) * _smoothedPanDy;
        }
        panByF(_smoothedPanDx, _smoothedPanDy);
    }
}

void CChunkedVolumeViewer::onPanStart(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = true;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    _lastPanSceneF = _view->mapToScene(_view->mapFromGlobal(QCursor::pos()));
}

void CChunkedVolumeViewer::onPanRelease(Qt::MouseButton, Qt::KeyboardModifiers)
{
    _isPanning = false;
    _panSmoothingInitialized = false;
    _smoothedPanDx = 0.0f;
    _smoothedPanDy = 0.0f;
    scheduleRender("pan released");
}

void CChunkedVolumeViewer::onVolumeClicked(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    if (_sameWrapAnnotation.enabled() && button == Qt::LeftButton && modifiers.testFlag(Qt::ShiftModifier)) {
        const bool appendToPreview = _sameWrapAnnotation.hasPreview() &&
                                     !_sameWrapAnnotation.shiftReleasedSincePreview();
        if (_sameWrapAnnotation.generatePreview(
                _framebuffer,
                scenePos,
                appendToPreview,
                _scale,
                _pointCollection,
                [this](const QPointF& point) { return sceneToVolume(point); },
                [this](const cv::Vec3f& point) { return volumeToScene(point); },
                [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
                    setOverlayGroup(key, items);
                },
                [this](const std::string& key) { clearOverlayGroup(key); })) {
            return;
        }
    }

    const auto sample = sampleSceneVolume(scenePos);
    if (!sample) {
        return;
    }
    emit sendVolumeClicked(sample->position, sample->normal, sample->surface, button, modifiers);
}

void CChunkedVolumeViewer::setSameWrapAnnotationMode(bool enabled)
{
    _sameWrapAnnotation.setEnabled(enabled);
    if (!enabled) {
        clearSameWrapAnnotationPreview();
    }
}

void CChunkedVolumeViewer::setSameWrapAnnotationSpacing(double spacingVx)
{
    _sameWrapAnnotation.setSpacing(spacingVx);
}

void CChunkedVolumeViewer::setSameWrapAnnotationMergeExisting(bool enabled)
{
    _sameWrapAnnotation.setMergeExistingAnnotations(enabled);
}

void CChunkedVolumeViewer::setSameWrapAnnotationPathType(int pathType)
{
    _sameWrapAnnotation.setPathType(
        pathType == static_cast<int>(SameWrapAnnotationTool::PathType::ShortestPath)
            ? SameWrapAnnotationTool::PathType::ShortestPath
            : SameWrapAnnotationTool::PathType::ConnectedComponents);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::setSameWrapAnnotationFilterType(int filterType)
{
    SameWrapAnnotationTool::ImageFilterType toolFilterType = SameWrapAnnotationTool::ImageFilterType::None;
    if (filterType == static_cast<int>(SameWrapAnnotationTool::ImageFilterType::Median)) {
        toolFilterType = SameWrapAnnotationTool::ImageFilterType::Median;
    } else if (filterType == static_cast<int>(SameWrapAnnotationTool::ImageFilterType::Gaussian)) {
        toolFilterType = SameWrapAnnotationTool::ImageFilterType::Gaussian;
    }
    _sameWrapAnnotation.setImageFilterType(toolFilterType);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::setSameWrapAnnotationFilterKernelSize(int kernelSize)
{
    _sameWrapAnnotation.setImageFilterKernelSize(kernelSize);
    clearSameWrapAnnotationPreview();
}

void CChunkedVolumeViewer::clearSameWrapAnnotationPreview()
{
    _sameWrapAnnotation.clear([this](const std::string& key) { clearOverlayGroup(key); });
}

bool CChunkedVolumeViewer::commitSameWrapAnnotationPreview()
{
    return _sameWrapAnnotation.commit(
        _pointCollection,
        [this](const std::string& key) { clearOverlayGroup(key); });
}

void CChunkedVolumeViewer::refreshSameWrapAnnotationOverlay()
{
    if (!_sameWrapAnnotation.hasPreview()) {
        return;
    }

    _sameWrapAnnotation.refreshOverlay(
        [this](const cv::Vec3f& point) { return volumeToScene(point); },
        [this](const std::string& key, const std::vector<QGraphicsItem*>& items) {
            setOverlayGroup(key, items);
        },
        [this](const std::string& key) { clearOverlayGroup(key); });
}

void CChunkedVolumeViewer::onMousePress(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_sameWrapAnnotation.enabled() && button == Qt::LeftButton &&
        modifiers.testFlag(Qt::ShiftModifier)) {
        return;
    }
    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _bboxStart = QPointF(sp[0], sp[1]);
        _activeBBoxSurfRect = QRectF(_bboxStart, QSizeF(0.0, 0.0));
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMousePressVolume(volumePos, {0, 0, 1}, button, modifiers, scenePos);
    if (_lineAnnotationPlacementPreviewEnabled && button == Qt::LeftButton &&
        modifiers == Qt::NoModifier &&
        std::isfinite(volumePos[0]) && std::isfinite(volumePos[1]) &&
        std::isfinite(volumePos[2])) {
        emit sendLineAnnotationSeedRequested(volumePos, scenePos);
    }
}

void CChunkedVolumeViewer::onMouseMove(QPointF scenePos, Qt::MouseButtons buttons, Qt::KeyboardModifiers modifiers)
{
    Q_UNUSED(modifiers);
    const bool reusedCursorSample = (_lastScenePos == scenePos && _lastCursorVolumePos.has_value());
    if (!reusedCursorSample) {
        _lastCursorVolumePos = cursorVolumePosition(scenePos);
    }
    _lastScenePos = scenePos;
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_bboxMode && _activeBBoxSurfRect && (buttons & Qt::LeftButton)) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        _activeBBoxSurfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    if (_lineAnnotationPlacementPreviewEnabled &&
        std::isfinite(volumePos[0]) && std::isfinite(volumePos[1]) &&
        std::isfinite(volumePos[2])) {
        updateLineAnnotationPlacementMarker(scenePos);
    } else {
        clearLineAnnotationPlacementMarker();
    }
    emit sendMouseMoveVolume(volumePos, buttons, modifiers, scenePos);
}

void CChunkedVolumeViewer::onMouseRelease(QPointF scenePos, Qt::MouseButton button, Qt::KeyboardModifiers modifiers)
{
    _lastScenePos = scenePos;
    _lastCursorVolumePos = cursorVolumePosition(scenePos);
    updateCursorCrosshair(scenePos);
    updateStatusLabel();
    if (_bboxMode && _surfName == "segmentation" && button == Qt::LeftButton &&
        _activeBBoxSurfRect) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const QRectF surfRect = QRectF(_bboxStart, QPointF(sp[0], sp[1])).normalized();
        const int idx = static_cast<int>(_selections.size());
        const QColor color = QColor::fromHsv((idx * 53) % 360, 200, 255);
        _selections.push_back({surfRect, color});
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
        return;
    }
    cv::Vec3f volumePos;
    if (_lastCursorVolumePos) {
        volumePos = *_lastCursorVolumePos;
    } else {
        volumePos = sceneToVolume(scenePos);
    }
    emit sendMouseReleaseVolume(volumePos, button, modifiers, scenePos);
}

void CChunkedVolumeViewer::onPathsChanged(const QList<ViewerOverlayControllerBase::PathPrimitive>& paths)
{
    _drawingPaths.clear();
    _drawingPaths.reserve(static_cast<std::size_t>(paths.size()));
    for (const auto& path : paths) {
        _drawingPaths.push_back(path);
    }
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::onKeyPress(int key, Qt::KeyboardModifiers)
{
    constexpr float kPanPx = 64.0f;
    switch (key) {
        case Qt::Key_Left: panByF(kPanPx, 0); break;
        case Qt::Key_Right: panByF(-kPanPx, 0); break;
        case Qt::Key_Up: panByF(0, kPanPx); break;
        case Qt::Key_Down: panByF(0, -kPanPx); break;
        default: break;
    }
}

void CChunkedVolumeViewer::onKeyRelease(int key, Qt::KeyboardModifiers)
{
    if (key == Qt::Key_Shift) {
        _sameWrapAnnotation.noteShiftReleased();
    }
}

QPointF CChunkedVolumeViewer::surfaceToScene(float surfX, float surfY) const
{
    const float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    const float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    const qreal vx = (surfX - _surfacePtrX) * _scale + vpCx;
    const qreal vy = (surfY - _surfacePtrY) * _scale + vpCy;
    return QPointF(vx, vy);
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurface(const QPointF& scenePos) const
{
    if (_framebuffer.isNull() || _scale <= 0.0f)
        return {0, 0};
    const float vpCx = static_cast<float>(_framebuffer.width()) * 0.5f;
    const float vpCy = static_cast<float>(_framebuffer.height()) * 0.5f;
    return {(static_cast<float>(scenePos.x()) - vpCx) / _scale + _surfacePtrX,
            (static_cast<float>(scenePos.y()) - vpCy) / _scale + _surfacePtrY};
}

QRectF CChunkedVolumeViewer::surfaceRectToSceneRect(const QRectF& surfRect) const
{
    const QPointF a = surfaceToScene(static_cast<float>(surfRect.left()),
                                     static_cast<float>(surfRect.top()));
    const QPointF b = surfaceToScene(static_cast<float>(surfRect.right()),
                                     static_cast<float>(surfRect.bottom()));
    return QRectF(a, b).normalized();
}

cv::Vec2f CChunkedVolumeViewer::sceneToSurfaceCoords(const QPointF& scenePos) const
{
    return sceneToSurface(scenePos);
}

QPointF CChunkedVolumeViewer::volumeToScene(const cv::Vec3f& volPoint)
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return {};
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        const cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    }
    if (auto* quad = dynamic_cast<QuadSurface*>(surf.get())) {
        cv::Vec3f ptr = quad->pointer();
        auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
        if (quad->pointTo(ptr, volPoint, 4.0f, 100, patchIndex) < 0.0f)
            return {};
        const cv::Vec3f loc = quad->loc(ptr);
        return surfaceToScene(loc[0], loc[1]);
    }
    return {};
}

void CChunkedVolumeViewer::updateCursorCrosshair(const QPointF& scenePos)
{
    if (!_scene || !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y()))
        return;

    if (!_cursorCrosshair || !_cursorCrosshair->scene()) {
        QPainterPath path;
        constexpr qreal radius = 6.0;
        constexpr qreal arm = 14.0;
        constexpr qreal gap = 3.0;
        path.addEllipse(QPointF(0.0, 0.0), radius, radius);
        path.moveTo(-arm, 0.0);
        path.lineTo(-gap, 0.0);
        path.moveTo(gap, 0.0);
        path.lineTo(arm, 0.0);
        path.moveTo(0.0, -arm);
        path.lineTo(0.0, -gap);
        path.moveTo(0.0, gap);
        path.lineTo(0.0, arm);

        auto* marker = new QGraphicsPathItem(path);
        QPen pen(QColor(50, 255, 215), 2.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(120.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _cursorCrosshair = marker;
    }

    _cursorCrosshair->setPos(scenePos);
    _cursorCrosshair->show();
}

void CChunkedVolumeViewer::setLineAnnotationPlacementPreviewEnabled(bool enabled)
{
    _lineAnnotationPlacementPreviewEnabled = enabled;
    if (!enabled) {
        clearLineAnnotationPlacementMarker();
    }
}

bool CChunkedVolumeViewer::lineAnnotationPlacementMarkerVisible() const
{
    return _lineAnnotationPlacementMarker && _lineAnnotationPlacementMarker->isVisible();
}

void CChunkedVolumeViewer::updateLineAnnotationPlacementMarker(const QPointF& scenePos)
{
    if (!_lineAnnotationPlacementPreviewEnabled || !_scene ||
        !std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        clearLineAnnotationPlacementMarker();
        return;
    }

    if (!_lineAnnotationPlacementMarker || !_lineAnnotationPlacementMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-6.0, -6.0, 12.0, 12.0);
        QPen pen(QColor(Qt::yellow), 2.0);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(QBrush(QColor(255, 255, 0, 70)));
        marker->setZValue(135.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _lineAnnotationPlacementMarker = marker;
    }

    _lineAnnotationPlacementMarker->setPos(scenePos);
    _lineAnnotationPlacementMarker->show();
}

void CChunkedVolumeViewer::clearLineAnnotationPlacementMarker()
{
    if (_lineAnnotationPlacementMarker) {
        _lineAnnotationPlacementMarker->hide();
    }
}

std::optional<cv::Vec3f> CChunkedVolumeViewer::cursorVolumePosition(const QPointF& scenePos) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return std::nullopt;

    cv::Vec3f p = sceneToVolume(scenePos);
    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        p += plane->normal({0, 0, 0}) * _zOff;
    } else if (_zOff != 0.0f) {
        const cv::Vec2f sp = sceneToSurface(scenePos);
        const cv::Vec3f n = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(n[0]) && std::isfinite(n[1]) && std::isfinite(n[2]))
            p += n * _zOff;
    }
    return p;
}

void CChunkedVolumeViewer::setLinkedCursorVolumePoint(const std::optional<cv::Vec3f>&)
{
    // The chunked viewer shows the cursor marker at its local mouse position.
    // Cross-view projection was unreliable for mixed surface/plane views and is
    // intentionally ignored here.
}

void CChunkedVolumeViewer::updateFocusMarker(POI* poi)
{
    if (!_scene)
        return;
    if (!poi && _state)
        poi = _state->poi("focus");
    if (!poi || !_surfWeak.lock()) {
        if (_focusMarker)
            _focusMarker->hide();
        return;
    }

    if (!_focusMarker || !_focusMarker->scene()) {
        auto* marker = new QGraphicsEllipseItem(-10.0, -10.0, 20.0, 20.0);
        QPen pen(QColor(50, 255, 215), 3.0, Qt::DashDotLine, Qt::RoundCap, Qt::RoundJoin);
        pen.setCosmetic(true);
        marker->setPen(pen);
        marker->setBrush(Qt::NoBrush);
        marker->setZValue(110.0);
        marker->setAcceptedMouseButtons(Qt::NoButton);
        _scene->addItem(marker);
        _focusMarker = marker;
    }

    const QPointF scenePos = volumeToScene(poi->p);
    if (!std::isfinite(scenePos.x()) || !std::isfinite(scenePos.y())) {
        _focusMarker->hide();
        return;
    }

    _focusMarker->setPos(scenePos);
    _focusMarker->show();
}

cv::Vec3f CChunkedVolumeViewer::sceneToVolume(const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf)
        return {0, 0, 0};
    const cv::Vec2f sp = sceneToSurface(scenePoint);
    return surf->coord({0, 0, 0}, {sp[0], sp[1], 0});
}

std::optional<CChunkedVolumeViewer::SceneVolumeSample> CChunkedVolumeViewer::sampleSceneVolume(
    const QPointF& scenePoint) const
{
    auto surf = _surfWeak.lock();
    if (!surf) {
        return std::nullopt;
    }

    const auto cursorPos = cursorVolumePosition(scenePoint);
    SceneVolumeSample sample;
    sample.position = cursorPos ? *cursorPos : sceneToVolume(scenePoint);
    sample.surface = surf.get();
    if (!std::isfinite(sample.position[0]) ||
        !std::isfinite(sample.position[1]) ||
        !std::isfinite(sample.position[2])) {
        return std::nullopt;
    }

    if (auto* plane = dynamic_cast<PlaneSurface*>(surf.get())) {
        sample.normal = plane->normal({0, 0, 0});
    } else {
        const cv::Vec2f sp = sceneToSurface(scenePoint);
        const cv::Vec3f surfaceNormal = surf->normal({0, 0, 0}, {sp[0], sp[1], 0.0f});
        if (std::isfinite(surfaceNormal[0]) &&
            std::isfinite(surfaceNormal[1]) &&
            std::isfinite(surfaceNormal[2]) &&
            cv::norm(surfaceNormal) > 0.0f) {
            sample.normal = surfaceNormal;
        }
    }
    if (!std::isfinite(sample.normal[0]) ||
        !std::isfinite(sample.normal[1]) ||
        !std::isfinite(sample.normal[2]) ||
        cv::norm(sample.normal) <= 0.0f) {
        sample.normal = {0, 0, 1};
    }
    return sample;
}

void CChunkedVolumeViewer::setOverlayGroup(const std::string& key, const std::vector<QGraphicsItem*>& items)
{
    clearOverlayGroup(key);
    _overlayGroups[key] = items;
    for (auto* item : items) {
        if (item && !item->scene())
            _scene->addItem(item);
    }
}

void CChunkedVolumeViewer::clearOverlayGroup(const std::string& key)
{
    auto it = _overlayGroups.find(key);
    if (it == _overlayGroups.end())
        return;
    for (auto* item : it->second)
        delete item;
    _overlayGroups.erase(it);
}

void CChunkedVolumeViewer::clearAllOverlayGroups()
{
    for (auto& [_, items] : _overlayGroups) {
        for (auto* item : items)
            delete item;
    }
    _overlayGroups.clear();
}

std::vector<std::pair<QRectF, QColor>> CChunkedVolumeViewer::selections() const
{
    std::vector<std::pair<QRectF, QColor>> out;
    out.reserve(_selections.size());
    for (const auto& selection : _selections) {
        out.emplace_back(surfaceRectToSceneRect(selection.surfRect), selection.color);
    }
    return out;
}

std::optional<QRectF> CChunkedVolumeViewer::activeBBoxSceneRect() const
{
    if (!_activeBBoxSurfRect)
        return std::nullopt;
    return surfaceRectToSceneRect(*_activeBBoxSurfRect);
}

void CChunkedVolumeViewer::setBBoxMode(bool enabled)
{
    _bboxMode = enabled;
    if (!enabled && _activeBBoxSurfRect) {
        _activeBBoxSurfRect.reset();
        emit overlaysUpdated();
    }
}

QuadSurface* CChunkedVolumeViewer::makeBBoxFilteredSurfaceFromSceneRect(const QRectF& sceneRect)
{
    if (_surfName != "segmentation")
        return nullptr;

    auto surf = _surfWeak.lock();
    auto* quad = dynamic_cast<QuadSurface*>(surf.get());
    if (!quad)
        return nullptr;

    const cv::Mat_<cv::Vec3f> src = quad->rawPoints();
    const int h = src.rows;
    const int w = src.cols;
    if (h <= 0 || w <= 0)
        return nullptr;

    const cv::Vec2f sp0 = sceneToSurface(sceneRect.topLeft());
    const cv::Vec2f sp1 = sceneToSurface(sceneRect.bottomRight());
    QRectF surfRect(QPointF(sp0[0], sp0[1]), QPointF(sp1[0], sp1[1]));
    surfRect = surfRect.normalized();

    const double cx = w * 0.5;
    const double cy = h * 0.5;
    const cv::Vec2f scale = quad->scale();
    if (scale[0] == 0.0f || scale[1] == 0.0f)
        return nullptr;

    const int i0 = std::max(0, static_cast<int>(std::floor(cx + surfRect.left() * scale[0])));
    const int i1 = std::min(w - 1, static_cast<int>(std::ceil(cx + surfRect.right() * scale[0])));
    const int j0 = std::max(0, static_cast<int>(std::floor(cy + surfRect.top() * scale[1])));
    const int j1 = std::min(h - 1, static_cast<int>(std::ceil(cy + surfRect.bottom() * scale[1])));
    if (i0 > i1 || j0 > j1)
        return nullptr;

    cv::Mat_<cv::Vec3f> cropped(j1 - j0 + 1, i1 - i0 + 1, cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            const cv::Vec3f& p = src(j, i);
            if (p[0] == -1.0f && p[1] == -1.0f && p[2] == -1.0f)
                continue;
            const double u = (i - cx) / scale[0];
            const double v = (j - cy) / scale[1];
            if (u >= surfRect.left() && u <= surfRect.right() &&
                v >= surfRect.top() && v <= surfRect.bottom()) {
                cropped(j - j0, i - i0) = p;
            }
        }
    }

    cv::Mat_<cv::Vec3f> cleaned = clean_surface_outliers(cropped);

    auto countValidInCol = [&](int c) {
        int count = 0;
        for (int r = 0; r < cleaned.rows; ++r) {
            if (cleaned(r, c)[0] != -1.0f)
                ++count;
        }
        return count;
    };
    auto countValidInRow = [&](int r) {
        int count = 0;
        for (int c = 0; c < cleaned.cols; ++c) {
            if (cleaned(r, c)[0] != -1.0f)
                ++count;
        }
        return count;
    };

    const int minValidCol = std::max(1, std::min(3, cleaned.rows));
    const int minValidRow = std::max(1, std::min(3, cleaned.cols));
    int left = 0;
    int right = cleaned.cols - 1;
    int top = 0;
    int bottom = cleaned.rows - 1;
    while (left <= right && countValidInCol(left) < minValidCol)
        ++left;
    while (right >= left && countValidInCol(right) < minValidCol)
        --right;
    while (top <= bottom && countValidInRow(top) < minValidRow)
        ++top;
    while (bottom >= top && countValidInRow(bottom) < minValidRow)
        --bottom;

    if (left > right || top > bottom) {
        left = cleaned.cols;
        right = -1;
        top = cleaned.rows;
        bottom = -1;
        for (int j = 0; j < cleaned.rows; ++j) {
            for (int i = 0; i < cleaned.cols; ++i) {
                if (cleaned(j, i)[0] != -1.0f) {
                    left = std::min(left, i);
                    right = std::max(right, i);
                    top = std::min(top, j);
                    bottom = std::max(bottom, j);
                }
            }
        }
        if (right < 0 || bottom < 0)
            return nullptr;
    }

    cv::Mat_<cv::Vec3f> finalPts(bottom - top + 1, right - left + 1,
                                  cv::Vec3f(-1.0f, -1.0f, -1.0f));
    for (int j = top; j <= bottom; ++j) {
        for (int i = left; i <= right; ++i) {
            finalPts(j - top, i - left) = cleaned(j, i);
        }
    }

    return new QuadSurface(finalPts, quad->scale());
}

void CChunkedVolumeViewer::clearSelections()
{
    _selections.clear();
    emit overlaysUpdated();
}

void CChunkedVolumeViewer::invalidateIntersect(const std::string& name)
{
    if (_closing) {
        return;
    }
    if (_segmentationEditActive && name == "segmentation" &&
        dynamic_cast<PlaneSurface*>(currentSurface())) {
        _lastIntersectFp = {};
        _intersectionGeometryCache = {};
        if (_deferSegmentationIntersections) {
            _deferredSegmentationIntersectionsDirty = true;
            return;
        }
        scheduleIntersectionRender("segmentation intersection invalidated during edit");
        return;
    }

    clearIntersectionItems();
    _lastIntersectFp = {};
    _intersectionGeometryCache = {};
    _flattenedIntersectionCache = {};
    _flattenedIntersectionDirtyCells.reset();
}

void CChunkedVolumeViewer::invalidateIntersectRegion(const std::string& name,
                                                     const cv::Rect& changedCells)
{
    if (changedCells.empty()) {
        return;
    }
    if (name.empty() || name != "segmentation") {
        invalidateIntersect(name);
        return;
    }
    if (_surfName != "segmentation") {
        invalidateIntersect(name);
        return;
    }

    _flattenedIntersectionDirtyCells =
        _flattenedIntersectionDirtyCells
            ? rectContains(*_flattenedIntersectionDirtyCells, changedCells)
                  ? *_flattenedIntersectionDirtyCells
                  : (*_flattenedIntersectionDirtyCells | changedCells)
            : changedCells;
    _lastIntersectFp = {};
    scheduleIntersectionRender("segmentation intersection region invalidated");
}

void CChunkedVolumeViewer::setIntersects(const std::set<std::string>& names)
{
    if (_closing || _intersectTgts == names) {
        return;
    }
    _intersectTgts = names;
    invalidateIntersect();
    renderIntersections("setIntersects");
}

void CChunkedVolumeViewer::setPlaneIntersectionLinesVisible(bool visible)
{
    if (_closing || _planeIntersectionLinesVisible == visible) {
        return;
    }
    _planeIntersectionLinesVisible = visible;
    _lastIntersectFp = {};
    auto surf = _surfWeak.lock();
    const bool isPlaneViewer = dynamic_cast<PlaneSurface*>(surf.get()) != nullptr;
    if (!visible && !isPlaneViewer) {
        clearIntersectionItems();
        if (_view) {
            _view->viewport()->update();
        }
        return;
    }
    if (visible && !isPlaneViewer) {
        renderIntersections("setPlaneIntersectionLinesVisible");
    }
}

void CChunkedVolumeViewer::setIntersectionOpacity(float v)
{
    if (_closing) {
        return;
    }
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    if (std::abs(_intersectionOpacity - clamped) < 1e-6f) {
        return;
    }
    _intersectionOpacity = clamped;
    renderIntersections("setIntersectionOpacity");
}

void CChunkedVolumeViewer::setIntersectionThickness(float v)
{
    if (_closing) {
        return;
    }
    const float clamped = std::max(0.0f, v);
    if (std::abs(_intersectionThickness - clamped) < 1e-6f) {
        return;
    }
    _intersectionThickness = clamped;
    renderIntersections("setIntersectionThickness");
}

void CChunkedVolumeViewer::setSurfacePatchSamplingStride(int s)
{
    if (_closing) {
        return;
    }
    const int stride = std::max(1, s);
    if (_surfacePatchSamplingStride == stride) {
        return;
    }
    _surfacePatchSamplingStride = stride;
    invalidateIntersect();
    renderIntersections("setSurfacePatchSamplingStride");
}

void CChunkedVolumeViewer::clearIntersectionItems()
{
    for (auto* item : _intersectionItems) {
        if (item && item->scene())
            _scene->removeItem(item);
        delete item;
    }
    _intersectionItems.clear();
    _flattenedIntersectionCache.tileItems.clear();
    _intersectionItemsHaveCamera = false;
}

void CChunkedVolumeViewer::updateIntersectionPreviewTransform()
{
    if (_intersectionItems.empty() || !_intersectionItemsHaveCamera ||
        _intersectionItemsCamScale <= 0.0f || _camScale <= 0.0f ||
        _framebuffer.isNull()) {
        return;
    }

    const qreal vpCx = qreal(_framebuffer.width()) * 0.5;
    const qreal vpCy = qreal(_framebuffer.height()) * 0.5;
    const qreal scale = qreal(_camScale / _intersectionItemsCamScale);
    const qreal tx = (qreal(_intersectionItemsCamSurfX) - qreal(_camSurfX)) * qreal(_camScale)
                   + vpCx - vpCx * scale;
    const qreal ty = (qreal(_intersectionItemsCamSurfY) - qreal(_camSurfY)) * qreal(_camScale)
                   + vpCy - vpCy * scale;
    const QTransform transform(scale, 0.0, 0.0,
                               0.0, scale, 0.0,
                               tx, ty, 1.0);
    for (auto* item : _intersectionItems) {
        if (item)
            item->setTransform(transform);
    }
}

void CChunkedVolumeViewer::renderFlattenedIntersections(const std::shared_ptr<Surface>& surf,
                                                        const char* reason,
                                                        std::source_location caller)
{
    ProfileScope profile("renderFlattenedIntersections", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format("surf='{}' targets={}", _surfName, _intersectTgts.size()));
    }
    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(surf);
    if (!activeSeg || !_state || _state->surface("segmentation") != activeSeg) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip inactive_segmentation");
        return;
    }

    auto* patchIndex = _viewerManager ? _viewerManager->surfacePatchIndex() : nullptr;
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip empty_patch_index");
        return;
    }

    struct PlaneEntry {
        std::shared_ptr<PlaneSurface> plane;
        QColor color;
    };
    const std::array<std::pair<const char*, QColor>, 3> kPlaneSpecs = {{
        {"seg xy", QColor(255, 140, 0)},
        {"seg xz", QColor(Qt::red)},
        {"seg yz", QColor(Qt::yellow)},
    }};
    std::vector<PlaneEntry> planes;
    planes.reserve(kPlaneSpecs.size());
    for (const auto& [name, color] : kPlaneSpecs) {
        if (!_intersectTgts.count(name))
            continue;
        if (auto p = std::dynamic_pointer_cast<PlaneSurface>(_state->surface(name))) {
            planes.push_back({std::move(p), color});
        }
    }
    if (planes.empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip no_planes");
        return;
    }

    auto mix = [](std::size_t s, std::size_t v) {
        return s ^ (v + 0x9e3779b9u + (s << 6) + (s >> 2));
    };
    auto hashVec = [&](std::size_t s, const cv::Vec3f& v) {
        for (int i = 0; i < 3; ++i)
            s = mix(s, std::hash<int>{}(int(std::lround(v[i] * 1000.0f))));
        return s;
    };

    std::size_t planesHash = 0;
    for (const auto& e : planes) {
        planesHash = hashVec(planesHash, e.plane->origin());
        planesHash = hashVec(planesHash, e.plane->normal({}, {}));
        planesHash = hashVec(planesHash, e.plane->basisX());
        planesHash = hashVec(planesHash, e.plane->basisY());
        planesHash = mix(planesHash, std::hash<uint32_t>{}(uint32_t(e.color.rgba())));
    }

    IntersectFingerprint fp;
    fp.flattenedPlanesHash = planesHash;
    fp.opacityQ = int(std::lround(_intersectionOpacity * 1000.0f));
    fp.thicknessQ = int(std::lround(_intersectionThickness * 1000.0f));
    fp.indexSamplingStride = 1;
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    fp.activeSegHash = std::hash<const void*>{}(activeSeg.get());
    const uint64_t activeGeneration = patchIndex->generation(activeSeg);
    fp.targetGenerationHash = 0;

    // Flattened intersections are built in surface-view scene coordinates.
    // Pan/zoom can reuse the same paths by transforming the existing items.
    fp.cameraHash = 0;
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty() &&
        !_flattenedIntersectionDirtyCells) {
        updateIntersectionPreviewTransform();
        if (profile.enabled()) {
            profile.setDetails(std::format("action=cache_hit planes={} items={}",
                                           planes.size(), _intersectionItems.size()));
        }
        return;
    }

    Rect3D allBounds{cv::Vec3f(0, 0, 0), cv::Vec3f(1, 1, 1)};
    if (_volume) {
        auto [w, h, d] = _volume->shapeXyz();
        allBounds.high = {static_cast<float>(w),
                          static_cast<float>(h),
                          static_cast<float>(d)};
    }

    const float clipTol = 1e-4f;
    const float penWidth = std::max(_intersectionThickness,
                                    kActiveIntersectionMinWidthDelta);
    const float opacity = std::clamp(
        _intersectionOpacity * kActiveIntersectionOpacityScale, 0.0f, 1.0f);

    auto isFiniteScalar = [](double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    };
    auto isFinitePoint = [&](const QPointF& p) {
        return isFiniteScalar(p.x()) && isFiniteScalar(p.y());
    };

    const cv::Mat_<cv::Vec3f>* points = activeSeg->rawPointsPtr();
    if (!points || points->cols < 2 || points->rows < 2) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip no_points");
        return;
    }

    auto cellBounds = [&]() {
        return cv::Rect(0, 0, points->cols - 1, points->rows - 1);
    };
    const int stride = 1;
    const bool cacheCompatible =
        _flattenedIntersectionCache.valid &&
        _flattenedIntersectionCache.surface == activeSeg.get() &&
        _flattenedIntersectionCache.planesHash == planesHash &&
        _flattenedIntersectionCache.indexSamplingStride == stride;
    const bool displayOnlyRefresh =
        cacheCompatible &&
        !_flattenedIntersectionDirtyCells;
    const bool needsFullRebuild = !cacheCompatible;
    std::unordered_map<std::uint64_t, std::unordered_set<int>> dirtyTilePlanes;

    auto removeFlattenedTilePlaneItem = [&](std::uint64_t tileKey, int planeIndex) {
        auto it = _flattenedIntersectionCache.tileItems.find(tileKey);
        if (it == _flattenedIntersectionCache.tileItems.end() ||
            planeIndex < 0 || static_cast<std::size_t>(planeIndex) >= it->second.size()) {
            return;
        }
        auto*& item = it->second[static_cast<std::size_t>(planeIndex)];
        if (!item) {
            return;
        }
        if (item->scene()) {
            _scene->removeItem(item);
        }
        auto vecIt = std::find(_intersectionItems.begin(), _intersectionItems.end(), item);
        if (vecIt != _intersectionItems.end()) {
            _intersectionItems.erase(vecIt);
        }
        delete item;
        item = nullptr;
    };

    auto lineMatchesPlane = [](const FlattenedIntersectionLine& line, int planeIndex) {
        return line.planeIndex == planeIndex;
    };
    auto markChangedCellPlanes = [&](std::uint64_t cellKey,
                                     const std::vector<FlattenedIntersectionLine>& newLines) {
        const int col = int(std::uint32_t(cellKey));
        const int row = int(std::uint32_t(cellKey >> 32));
        const std::uint64_t tileKey = surfaceCellTileKey(col, row);
        const auto oldIt = _flattenedIntersectionCache.cellLines.find(cellKey);
        const auto* oldLines = oldIt == _flattenedIntersectionCache.cellLines.end()
            ? nullptr
            : &oldIt->second;

        for (std::size_t planeIdx = 0; planeIdx < planes.size(); ++planeIdx) {
            const int idx = static_cast<int>(planeIdx);
            std::vector<const FlattenedIntersectionLine*> oldPlaneLines;
            std::vector<const FlattenedIntersectionLine*> newPlaneLines;
            if (oldLines) {
                for (const auto& line : *oldLines) {
                    if (lineMatchesPlane(line, idx)) {
                        oldPlaneLines.push_back(&line);
                    }
                }
            }
            for (const auto& line : newLines) {
                if (lineMatchesPlane(line, idx)) {
                    newPlaneLines.push_back(&line);
                }
            }

            bool changed = oldPlaneLines.size() != newPlaneLines.size();
            if (!changed) {
                for (std::size_t i = 0; i < oldPlaneLines.size(); ++i) {
                    const auto& oldLine = *oldPlaneLines[i];
                    const auto& newLine = *newPlaneLines[i];
                    if (oldLine.a != newLine.a || oldLine.b != newLine.b) {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed) {
                dirtyTilePlanes[tileKey].insert(idx);
            }
        }
    };

    auto rebuildFlattenedCells = [&](const cv::Rect& requestedCells) {
        const cv::Rect cells = requestedCells & cellBounds();
        if (cells.empty()) {
            return;
        }

        const cv::Vec3f center = activeSeg->center();
        const cv::Vec2f gridScale = activeSeg->scale();
        const float cx = center[0] * gridScale[0];
        const float cy = center[1] * gridScale[1];

        const int rowStart = (cells.y / stride) * stride;
        const int colStart = (cells.x / stride) * stride;
        const int rowEnd = cells.y + cells.height;
        const int colEnd = cells.x + cells.width;
        for (int row = rowStart; row < rowEnd && row < points->rows - 1; row += stride) {
            for (int col = colStart; col < colEnd && col < points->cols - 1; col += stride) {
                const std::uint64_t key = surfaceTileKey(col, row);
                std::vector<FlattenedIntersectionLine> lines;

                const int strideX = std::min(stride, points->cols - 1 - col);
                const int strideY = std::min(stride, points->rows - 1 - row);
                if (strideX <= 0 || strideY <= 0) {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines.erase(key);
                    continue;
                }

                const std::array<cv::Vec3f, 4> corners = {
                    (*points)(row, col),
                    (*points)(row, col + strideX),
                    (*points)(row + strideY, col + strideX),
                    (*points)(row + strideY, col),
                };
                if (!validSurfacePoint(corners[0]) || !validSurfacePoint(corners[1]) ||
                    !validSurfacePoint(corners[2]) || !validSurfacePoint(corners[3])) {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines.erase(key);
                    continue;
                }

                const float baseX = static_cast<float>(col);
                const float baseY = static_cast<float>(row);
                const std::array<cv::Vec3f, 4> params = {
                    cv::Vec3f(baseX - cx, baseY - cy, 0.0f),
                    cv::Vec3f(baseX + float(strideX) - cx, baseY - cy, 0.0f),
                    cv::Vec3f(baseX + float(strideX) - cx, baseY + float(strideY) - cy, 0.0f),
                    cv::Vec3f(baseX - cx, baseY + float(strideY) - cy, 0.0f),
                };

                for (int triIdx = 0; triIdx < 2; ++triIdx) {
                    SurfacePatchIndex::TriangleCandidate tri;
                    tri.surface = activeSeg;
                    tri.i = col;
                    tri.j = row;
                    tri.triangleIndex = triIdx;
                    if (triIdx == 0) {
                        tri.world = {corners[0], corners[1], corners[3]};
                        tri.surfaceParams = {params[0], params[1], params[3]};
                    } else {
                        tri.world = {corners[1], corners[2], corners[3]};
                        tri.surfaceParams = {params[1], params[2], params[3]};
                    }

                    if ((tri.world[0][0] < allBounds.low[0] && tri.world[1][0] < allBounds.low[0] && tri.world[2][0] < allBounds.low[0]) ||
                        (tri.world[0][0] > allBounds.high[0] && tri.world[1][0] > allBounds.high[0] && tri.world[2][0] > allBounds.high[0]) ||
                        (tri.world[0][1] < allBounds.low[1] && tri.world[1][1] < allBounds.low[1] && tri.world[2][1] < allBounds.low[1]) ||
                        (tri.world[0][1] > allBounds.high[1] && tri.world[1][1] > allBounds.high[1] && tri.world[2][1] > allBounds.high[1]) ||
                        (tri.world[0][2] < allBounds.low[2] && tri.world[1][2] < allBounds.low[2] && tri.world[2][2] < allBounds.low[2]) ||
                        (tri.world[0][2] > allBounds.high[2] && tri.world[1][2] > allBounds.high[2] && tri.world[2][2] > allBounds.high[2])) {
                        continue;
                    }

                    for (size_t idx = 0; idx < planes.size(); ++idx) {
                        auto seg = SurfacePatchIndex::clipTriangleToPlane(
                            tri, *planes[idx].plane, clipTol);
                        if (!seg) {
                            continue;
                        }
                        const cv::Vec3f a = activeSeg->loc(seg->surfaceParams[0]);
                        const cv::Vec3f b = activeSeg->loc(seg->surfaceParams[1]);
                        const QPointF pa = surfaceToScene(a[0], a[1]);
                        const QPointF pb = surfaceToScene(b[0], b[1]);
                        if (!isFinitePoint(pa) || !isFinitePoint(pb)) {
                            continue;
                        }
                        lines.push_back(FlattenedIntersectionLine{static_cast<int>(idx), pa, pb});
                    }
                }

                if (lines.empty()) {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines.erase(key);
                } else {
                    markChangedCellPlanes(key, lines);
                    _flattenedIntersectionCache.cellLines[key] = std::move(lines);
                }
            }
        }
    };

    if (needsFullRebuild) {
        clearIntersectionItems();
        _flattenedIntersectionCache = {};
        _flattenedIntersectionCache.surface = activeSeg.get();
        _flattenedIntersectionCache.planesHash = planesHash;
        _flattenedIntersectionCache.indexSamplingStride = stride;
        _flattenedIntersectionCache.cellLines.reserve(
            std::size_t(points->rows / stride + 1) * std::size_t(points->cols / stride + 1) / 8 + 1024);
        rebuildFlattenedCells(cellBounds());
        _flattenedIntersectionCache.valid = true;
    } else if (_flattenedIntersectionDirtyCells) {
        rebuildFlattenedCells(*_flattenedIntersectionDirtyCells);
    } else if (!cacheCompatible) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip incompatible_cache");
        return;
    }
    _flattenedIntersectionCache.generation = activeGeneration;
    _flattenedIntersectionDirtyCells.reset();

    _lastIntersectFp = fp;
    _intersectionGeometryCache = {};

    const bool rebuildAllTileItems =
        needsFullRebuild ||
        displayOnlyRefresh ||
        _flattenedIntersectionCache.tileItems.empty();
    if (rebuildAllTileItems) {
        dirtyTilePlanes.clear();
        for (const auto& [cellKey, _] : _flattenedIntersectionCache.cellLines) {
            const int col = int(std::uint32_t(cellKey));
            const int row = int(std::uint32_t(cellKey >> 32));
            const std::uint64_t tileKey = surfaceCellTileKey(col, row);
            for (const auto& line : _) {
                dirtyTilePlanes[tileKey].insert(line.planeIndex);
            }
        }
    }

    if (dirtyTilePlanes.empty()) {
        _intersectionItemsCamSurfX = _camSurfX;
        _intersectionItemsCamSurfY = _camSurfY;
        _intersectionItemsCamScale = _camScale;
        _intersectionItemsHaveCamera = !_intersectionItems.empty();
        _view->viewport()->update();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=no_dirty_tiles planes={} cells={} items={}",
                planes.size(), _flattenedIntersectionCache.cellLines.size(),
                _intersectionItems.size()));
        }
        return;
    }

    std::unordered_map<std::uint64_t, std::vector<QPainterPath>> tilePaths;
    tilePaths.reserve(dirtyTilePlanes.size());
    for (const auto& [tileKey, _] : dirtyTilePlanes) {
        tilePaths.emplace(tileKey, std::vector<QPainterPath>(planes.size()));
    }

    for (const auto& [cellKey, lines] : _flattenedIntersectionCache.cellLines) {
        const int col = int(std::uint32_t(cellKey));
        const int row = int(std::uint32_t(cellKey >> 32));
        const std::uint64_t tileKey = surfaceCellTileKey(col, row);
        auto pathsIt = tilePaths.find(tileKey);
        if (pathsIt == tilePaths.end()) {
            continue;
        }
        auto& paths = pathsIt->second;
        for (const auto& line : lines) {
            if (line.planeIndex < 0 || static_cast<size_t>(line.planeIndex) >= paths.size()) {
                continue;
            }
            const auto dirtyIt = dirtyTilePlanes.find(tileKey);
            if (dirtyIt == dirtyTilePlanes.end() ||
                dirtyIt->second.count(line.planeIndex) == 0) {
                continue;
            }
            paths[line.planeIndex].moveTo(line.a);
            paths[line.planeIndex].lineTo(line.b);
        }
    }

    for (auto& [tileKey, paths] : tilePaths) {
        auto& tileItems = _flattenedIntersectionCache.tileItems[tileKey];
        if (tileItems.size() < paths.size()) {
            tileItems.resize(paths.size(), nullptr);
        }
        const auto dirtyIt = dirtyTilePlanes.find(tileKey);
        for (std::size_t idx = 0; idx < paths.size(); ++idx) {
            if (dirtyIt == dirtyTilePlanes.end() ||
                dirtyIt->second.count(static_cast<int>(idx)) == 0) {
                continue;
            }
            QColor color = planes[idx].color;
            color.setAlphaF(opacity);
            QPen pen(color);
            pen.setWidthF(static_cast<qreal>(penWidth));
            pen.setCapStyle(Qt::RoundCap);
            pen.setJoinStyle(Qt::RoundJoin);
            pen.setCosmetic(true);
            if (paths[idx].isEmpty()) {
                removeFlattenedTilePlaneItem(tileKey, static_cast<int>(idx));
                continue;
            }
            auto*& item = tileItems[idx];
            if (!item) {
                item = new QGraphicsPathItem();
                item->setBrush(Qt::NoBrush);
                item->setZValue(kActiveIntersectionZ);
                item->setAcceptedMouseButtons(Qt::NoButton);
                _scene->addItem(item);
                _intersectionItems.push_back(item);
            }
            item->setTransform(QTransform());
            item->setPath(paths[idx]);
            item->setPen(pen);
        }
    }

    for (auto* item : _intersectionItems) {
        if (item) {
            item->setTransform(QTransform());
        }
    }

    if (_intersectionItems.empty()) {
        _intersectionItemsHaveCamera = false;
        _view->viewport()->update();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=empty_result planes={} cells={} dirtyTiles={}",
                planes.size(), _flattenedIntersectionCache.cellLines.size(),
                dirtyTilePlanes.size()));
        }
        return;
    }

    _intersectionItemsCamSurfX = _camSurfX;
    _intersectionItemsCamSurfY = _camSurfY;
    _intersectionItemsCamScale = _camScale;
    _intersectionItemsHaveCamera = !_intersectionItems.empty();
    _view->viewport()->update();
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=rendered planes={} cells={} dirtyTiles={} items={} fullRebuild={} dirtyCells={}",
            planes.size(), _flattenedIntersectionCache.cellLines.size(),
            dirtyTilePlanes.size(), _intersectionItems.size(), needsFullRebuild,
            bool(_flattenedIntersectionDirtyCells)));
    }
}

void CChunkedVolumeViewer::renderIntersections(const char* reason, std::source_location caller)
{
    ProfileScope profile("renderIntersections", reason, caller);
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "surf='{}' targets={} viewport={} scale={:.4f} zOff={:.3f}",
            _surfName, _intersectTgts.size(),
            _view ? profileRectF(_view->mapToScene(_view->viewport()->rect()).boundingRect())
                  : std::string("[]"),
            _scale, _zOff));
    }
    if (_closing) {
        profile.setDetails("action=skip closing");
        return;
    }
    if (property("vc_viewer_role").toString() == QStringLiteral("annotation")) {
        clearIntersectionItems();
        _lastIntersectFp = {};
        profile.setDetails("action=skip annotation_viewer");
        return;
    }

    auto surf = _surfWeak.lock();
    auto* plane = dynamic_cast<PlaneSurface*>(surf.get());
    if (!surf || !_state || !_viewerManager || !_scene || !_view) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip missing_input");
        return;
    }
    if (!plane) {
        if (!_planeIntersectionLinesVisible) {
            clearIntersectionItems();
            _lastIntersectFp = {};
            profile.setDetails("action=skip flattened_disabled");
            return;
        }
    }

    auto* patchIndex = _viewerManager->surfacePatchIndex();
    if (!patchIndex || patchIndex->empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip empty_patch_index");
        return;
    }

    auto activeSeg = std::dynamic_pointer_cast<QuadSurface>(_state->surface("segmentation"));
    if (!plane) {
        renderFlattenedIntersections(surf, reason, caller);
        profile.setDetails("action=delegated_flattened");
        return;
    }

    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets;
    auto addTarget = [&](const std::string& name) {
        if (auto quad = std::dynamic_pointer_cast<QuadSurface>(_state->surface(name))) {
            if (activeSeg && quad != activeSeg && !activeSeg->id.empty() &&
                quad->id == activeSeg->id) {
                return;
            }
            targets.insert(std::move(quad));
        }
    };
    for (const auto& name : _intersectTgts) {
        if (name == "visible_segmentation") {
            if (_highlightedSurfaceIds.empty()) {
                addTarget("segmentation");
            } else {
                for (const auto& id : _highlightedSurfaceIds)
                    addTarget(id);
            }
        } else {
            addTarget(name);
        }
    }
    if (targets.empty()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip no_targets");
        return;
    }

    QRectF sceneRect = _view->mapToScene(_view->viewport()->rect()).boundingRect();
    if (!sceneRect.isValid()) {
        invalidateIntersect();
        _lastIntersectFp = {};
        profile.setDetails("action=skip invalid_scene_rect");
        return;
    }

    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    const std::array<QPointF, 4> corners = {
        sceneRect.topLeft(), sceneRect.topRight(),
        sceneRect.bottomLeft(), sceneRect.bottomRight(),
    };
    for (const auto& c : corners) {
        cv::Vec2f sp = sceneToSurfaceCoords(c);
        minX = std::min(minX, sp[0]);
        minY = std::min(minY, sp[1]);
        maxX = std::max(maxX, sp[0]);
        maxY = std::max(maxY, sp[1]);
    }
    cv::Rect planeRoi{int(std::floor(minX)), int(std::floor(minY)),
                      std::max(1, int(std::ceil(maxX - minX))),
                      std::max(1, int(std::ceil(maxY - minY)))};

    IntersectFingerprint fp;
    fp.roiX = planeRoi.x;
    fp.roiY = planeRoi.y;
    fp.roiW = planeRoi.width;
    fp.roiH = planeRoi.height;
    auto quantizeVec = [](const cv::Vec3f& v) {
        return std::array<int, 3>{
            int(std::lround(v[0] * 1000.0f)),
            int(std::lround(v[1] * 1000.0f)),
            int(std::lround(v[2] * 1000.0f)),
        };
    };
    fp.planeOriginQ = quantizeVec(plane->origin());
    fp.planeNormalQ = quantizeVec(plane->normal({}, {}));
    fp.planeBasisXQ = quantizeVec(plane->basisX());
    fp.planeBasisYQ = quantizeVec(plane->basisY());
    fp.opacityQ = int(std::lround(_intersectionOpacity * 1000.0f));
    fp.thicknessQ = int(std::lround(_intersectionThickness * 1000.0f));
    fp.indexSamplingStride = patchIndex->samplingStride();
    fp.patchCount = patchIndex->patchCount();
    fp.surfaceCount = patchIndex->surfaceCount();
    size_t th = 0;
    size_t gh = 0;
    for (const auto& t : targets) {
        th ^= std::hash<const void*>{}(t.get()) + 0x9e3779b9u + (th << 6) + (th >> 2);
        gh ^= std::hash<const void*>{}(t.get()) ^
              (std::hash<uint64_t>{}(patchIndex->generation(t)) + 0x9e3779b9u);
    }
    fp.targetHash = th;
    fp.targetGenerationHash = gh;
    fp.activeSegHash = activeSeg ? std::hash<const void*>{}(activeSeg.get()) : 0;
    size_t hh = 0;
    for (const auto& id : _highlightedSurfaceIds)
        hh ^= std::hash<std::string>{}(id) + 0x9e3779b9u + (hh << 6) + (hh >> 2);
    fp.highlightedSurfaceHash = hh;
    fp.cameraHash = (std::hash<int>{}(_framebuffer.width()) + 0x9e3779b9u) ^
                    (std::hash<int>{}(_framebuffer.height()) << 1);
    fp.valid = true;
    if (_lastIntersectFp == fp && !_intersectionItems.empty()) {
        updateIntersectionPreviewTransform();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=cache_hit targetSurfaces={} items={} roi={}",
                targets.size(), _intersectionItems.size(), profileRect(planeRoi)));
        }
        return;
    }

    auto rectContains = [](const cv::Rect& outer, const cv::Rect& inner) {
        return inner.x >= outer.x &&
               inner.y >= outer.y &&
               inner.x + inner.width <= outer.x + outer.width &&
               inner.y + inner.height <= outer.y + outer.height;
    };
    const bool geometryCacheValid =
        _intersectionGeometryCache.valid &&
        rectContains(_intersectionGeometryCache.roi, planeRoi) &&
        _intersectionGeometryCache.planeOriginQ == fp.planeOriginQ &&
        _intersectionGeometryCache.planeNormalQ == fp.planeNormalQ &&
        _intersectionGeometryCache.planeBasisXQ == fp.planeBasisXQ &&
        _intersectionGeometryCache.planeBasisYQ == fp.planeBasisYQ &&
        _intersectionGeometryCache.indexSamplingStride == fp.indexSamplingStride &&
        _intersectionGeometryCache.patchCount == fp.patchCount &&
        _intersectionGeometryCache.surfaceCount == fp.surfaceCount &&
        _intersectionGeometryCache.targetHash == fp.targetHash &&
        _intersectionGeometryCache.targetGenerationHash == fp.targetGenerationHash;

    auto samePlaneRenderInputsExceptRoi = [](const IntersectFingerprint& a,
                                             const IntersectFingerprint& b) {
        return a.valid && b.valid &&
               a.planeOriginQ == b.planeOriginQ &&
               a.planeNormalQ == b.planeNormalQ &&
               a.planeBasisXQ == b.planeBasisXQ &&
               a.planeBasisYQ == b.planeBasisYQ &&
               a.opacityQ == b.opacityQ &&
               a.thicknessQ == b.thicknessQ &&
               a.indexSamplingStride == b.indexSamplingStride &&
               a.patchCount == b.patchCount &&
               a.surfaceCount == b.surfaceCount &&
               a.targetHash == b.targetHash &&
               a.targetGenerationHash == b.targetGenerationHash &&
               a.activeSegHash == b.activeSegHash &&
               a.highlightedSurfaceHash == b.highlightedSurfaceHash &&
               a.cameraHash == b.cameraHash;
    };
    if (geometryCacheValid && !_intersectionItems.empty() &&
        samePlaneRenderInputsExceptRoi(_lastIntersectFp, fp)) {
        _lastIntersectFp = fp;
        updateIntersectionPreviewTransform();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=camera_cache_hit targetSurfaces={} items={} roi={} cacheRoi={}",
                targets.size(), _intersectionItems.size(), profileRect(planeRoi),
                profileRect(_intersectionGeometryCache.roi)));
        }
        return;
    }

    _lastIntersectFp = fp;

    if (!geometryCacheValid) {
        constexpr int kMinPanCachePadding = 256;
        cv::Rect cacheRoi = planeRoi;
        const int padX = std::max(kMinPanCachePadding, planeRoi.width / 2);
        const int padY = std::max(kMinPanCachePadding, planeRoi.height / 2);
        cacheRoi.x -= padX;
        cacheRoi.y -= padY;
        cacheRoi.width += padX * 2;
        cacheRoi.height += padY * 2;

        // Only one compute in flight at a time: while it runs the current scene items
        // stand as the preview. A later render with changed inputs just waits for the
        // in-flight result; if that result is stale (fp differs) it's dropped and the
        // next render re-kicks. This bounds the read-gate to one outstanding read and
        // avoids spawning a worker per pan frame.
        if (_intersectComputeInFlight)
            return;

        // The r-tree query + triangle clip is heavy (~12% of the GUI tick); run it on
        // a worker and build the QGraphicsItems when it lands (Qt scene mutation is
        // main-thread-only). Until then keep the current items as a stale preview.
        // _intersectGen rises on every input change; the finish handler drops a result
        // whose gen no longer matches. The ViewerManager read-gate blocks index
        // mutation while the worker reads.
        const std::uint64_t gen = ++_intersectGen;
        auto planeCopy = std::make_shared<PlaneSurface>(*plane);
        auto targetsCopy = std::make_shared<std::unordered_set<SurfacePatchIndex::SurfacePtr>>(targets);
        _intersectComputeInFlight = true;
        _viewerManager->beginIndexRead();
        QPointer<CChunkedVolumeViewer> guard(this);
        ViewerManager* vm = _viewerManager;            // outlives viewers; always end the read
        SurfacePatchIndex* idx = patchIndex;
        (void)QtConcurrent::run([guard, vm, gen, cacheRoi, fp, planeCopy, targetsCopy, idx]() mutable {
            auto out = std::make_shared<std::unordered_map<SurfacePatchIndex::SurfacePtr,
                std::vector<SurfacePatchIndex::TriangleSegment>>>(
                    idx->computePlaneIntersections(*planeCopy, cacheRoi, *targetsCopy));
            QMetaObject::invokeMethod(qApp, [guard, vm, gen, cacheRoi, fp, out = std::move(out)]() mutable {
                if (guard)
                    guard->finishPlaneIntersectionCompute(gen, cacheRoi, fp, std::move(out));
                else if (vm)
                    vm->endIndexRead();                // viewer gone: still release the gate
            }, Qt::QueuedConnection);
        });
        if (profile.enabled())
            profile.setDetails(std::format("action=async_compute_kick gen={} roi={}",
                                           gen, profileRect(cacheRoi)));
        return;   // scene items rebuilt by finishPlaneIntersectionCompute -> re-entry
    }

    const auto& intersections = _intersectionGeometryCache.intersections;
    if (intersections.empty()) {
        clearIntersectionItems();
        if (profile.enabled()) {
            profile.setDetails(std::format(
                "action=empty_result targetSurfaces={} roi={} geometryCacheHit={}",
                targets.size(), profileRect(planeRoi), geometryCacheValid));
        }
        return;
    }

    std::size_t segmentCount = 0;
    for (const auto& [target, segments] : intersections) {
        (void)target;
        segmentCount += segments.size();
    }

    std::unordered_map<IntersectionStyle, QPainterPath, IntersectionStyleHash> groupedPaths;
    std::unordered_map<IntersectionStyle, QColor, IntersectionStyleHash> groupedColors;
    auto isFiniteScalar = [](double v) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
    };
    auto isFinitePoint = [&](const QPointF& p) {
        return isFiniteScalar(p.x()) && isFiniteScalar(p.y());
    };
    auto planeToScene = [&](const cv::Vec3f& volPoint) {
        cv::Vec3f proj = plane->project(volPoint, 1.0, 1.0);
        return surfaceToScene(proj[0], proj[1]);
    };
    auto* approvalOverlay = _viewerManager ? _viewerManager->segmentationOverlay() : nullptr;
    const bool showApprovalMaskIntersections =
        approvalOverlay && activeSeg && approvalOverlay->hasApprovalMaskData();
    auto paramToApprovalGrid = [&](const cv::Vec3f& param) {
        if (!activeSeg) {
            return QPointF();
        }
        const cv::Vec2f grid = activeSeg->ptrToGrid(param);
        return QPointF(grid[0], grid[1]);
    };
    auto addApprovalMaskIntersection = [&](const SurfacePatchIndex::TriangleSegment& seg,
                                           float renderedOpacity,
                                           float renderedPenWidth) {
        if (!showApprovalMaskIntersections) {
            return;
        }

        const QPointF gridA = paramToApprovalGrid(seg.surfaceParams[0]);
        const QPointF gridB = paramToApprovalGrid(seg.surfaceParams[1]);
        if (!isFinitePoint(gridA) || !isFinitePoint(gridB)) {
            return;
        }

        const int steps = std::max(1, int(std::ceil(std::max(std::abs(gridB.x() - gridA.x()),
                                                             std::abs(gridB.y() - gridA.y())))));
        const float opacity = std::clamp(renderedOpacity * kApprovalPlaneIntersectionScale,
                                         0.0f, 1.0f);
        const float penWidth = std::max(0.0f, renderedPenWidth) *
                               kApprovalPlaneIntersectionScale;

        for (int step = 0; step < steps; ++step) {
            const float t0 = static_cast<float>(step) / static_cast<float>(steps);
            const float t1 = static_cast<float>(step + 1) / static_cast<float>(steps);
            const float tm = (t0 + t1) * 0.5f;

            const float row = static_cast<float>(gridA.y() + (gridB.y() - gridA.y()) * tm);
            const float col = static_cast<float>(gridA.x() + (gridB.x() - gridA.x()) * tm);
            int approvalStatus = 0;
            if (approvalOverlay->queryApprovalBilinear(row, col, &approvalStatus) <= 0.0f ||
                approvalStatus == 0) {
                continue;
            }

            QColor approvalColor = approvalOverlay->queryApprovalColor(
                static_cast<int>(std::round(row)),
                static_cast<int>(std::round(col)));
            if (!approvalColor.isValid()) {
                approvalColor = QColor(0, 255, 0);
            }
            approvalColor.setAlphaF(opacity);
            if (approvalColor.alpha() <= 0) {
                continue;
            }

            const cv::Vec3f world0 = seg.world[0] + (seg.world[1] - seg.world[0]) * t0;
            const cv::Vec3f world1 = seg.world[0] + (seg.world[1] - seg.world[0]) * t1;
            const QPointF a = planeToScene(world0);
            const QPointF b = planeToScene(world1);
            if (!isFinitePoint(a) || !isFinitePoint(b)) {
                continue;
            }

            const IntersectionStyle approvalStyle{
                approvalColor.rgba(),
                kActiveIntersectionZ + 1,
                int(std::lround(penWidth * 1000.0f)),
            };
            QPainterPath& approvalPath = groupedPaths[approvalStyle];
            groupedColors[approvalStyle] = approvalColor;
            approvalPath.moveTo(a);
            approvalPath.lineTo(b);
        }
    };

    for (const auto& [target, segments] : intersections) {
        if (!target || segments.empty())
            continue;

        QColor baseColor;
        int zValue = kIntersectionZ;
        float opacity = _intersectionOpacity;
        float penWidth = _intersectionThickness;
        if (target == activeSeg) {
            baseColor = activeSegmentationColorForView(_surfName);
            zValue = kActiveIntersectionZ;
            opacity *= kActiveIntersectionOpacityScale;
            penWidth = activeSegmentationIntersectionWidth(penWidth);
        } else if (_highlightedSurfaceIds.count(target->id)) {
            baseColor = QColor(0, 220, 255);
            zValue = kHighlightedIntersectionZ;
        } else {
            const auto& id = target->id;
            auto it = _surfaceColorAssignments.find(id);
            size_t idx;
            if (it != _surfaceColorAssignments.end()) {
                idx = it->second;
            } else if (_surfaceColorAssignments.size() < 500) {
                idx = _nextColorIndex++;
                _surfaceColorAssignments[id] = idx;
            } else {
                idx = std::hash<std::string>{}(id);
            }
            baseColor = QColor::fromRgba(kIntersectionPalette[idx % kIntersectionPalette.size()]);
        }
        baseColor.setAlphaF(std::clamp(opacity, 0.0f, 1.0f));
        if (baseColor.alpha() <= 0)
            continue;

        const IntersectionStyle style{
            baseColor.rgba(),
            zValue,
            int(std::lround(std::max(0.0f, penWidth) * 1000.0f)),
        };
        groupedColors[style] = baseColor;
        for (const auto& seg : segments) {
            QPointF a = planeToScene(seg.world[0]);
            QPointF b = planeToScene(seg.world[1]);
            if (!isFinitePoint(a) || !isFinitePoint(b))
                continue;
            groupedPaths[style].moveTo(a);
            groupedPaths[style].lineTo(b);
            if (target == activeSeg) {
                addApprovalMaskIntersection(seg, opacity, penWidth);
            }
        }
    }

    std::size_t itemIndex = 0;
    _intersectionItems.reserve(std::max(_intersectionItems.size(), groupedPaths.size()));
    for (const auto& [style, path] : groupedPaths) {
        if (path.isEmpty())
            continue;
        QGraphicsPathItem* item = nullptr;
        if (itemIndex < _intersectionItems.size()) {
            item = dynamic_cast<QGraphicsPathItem*>(_intersectionItems[itemIndex]);
        }
        if (!item) {
            item = new QGraphicsPathItem();
            item->setBrush(Qt::NoBrush);
            item->setAcceptedMouseButtons(Qt::NoButton);
            _scene->addItem(item);
            if (itemIndex < _intersectionItems.size()) {
                if (_intersectionItems[itemIndex] && _intersectionItems[itemIndex]->scene()) {
                    _scene->removeItem(_intersectionItems[itemIndex]);
                }
                delete _intersectionItems[itemIndex];
                _intersectionItems[itemIndex] = item;
            } else {
                _intersectionItems.push_back(item);
            }
        }
        QPen pen(groupedColors[style]);
        pen.setWidthF(static_cast<qreal>(style.widthQ) / 1000.0);
        pen.setCapStyle(Qt::RoundCap);
        pen.setJoinStyle(Qt::RoundJoin);
        pen.setCosmetic(true);
        item->setTransform(QTransform());
        item->setPath(path);
        item->setPen(pen);
        item->setZValue(style.z);
        ++itemIndex;
    }
    while (_intersectionItems.size() > itemIndex) {
        auto* item = _intersectionItems.back();
        _intersectionItems.pop_back();
        if (item && item->scene()) {
            _scene->removeItem(item);
        }
        delete item;
    }
    _intersectionItemsCamSurfX = _camSurfX;
    _intersectionItemsCamSurfY = _camSurfY;
    _intersectionItemsCamScale = _camScale;
    _intersectionItemsHaveCamera = !_intersectionItems.empty();

    _view->viewport()->update();
    if (profile.enabled()) {
        profile.setDetails(std::format(
            "action=rendered targetSurfaces={} intersectingSurfaces={} segments={} groupedPaths={} items={} roi={} cacheRoi={} geometryCacheHit={}",
            targets.size(), intersections.size(), segmentCount,
            groupedPaths.size(), _intersectionItems.size(), profileRect(planeRoi),
            profileRect(_intersectionGeometryCache.roi), geometryCacheValid));
    }
}

// Worker finished computePlaneIntersections (main thread, via invokeMethod). Release
// the index read-gate, then -- unless superseded -- publish the geometry into the
// cache and re-run renderIntersections, which now hits the geometry cache and builds
// the QGraphicsItems synchronously (cheap). A stale result (gen advanced because the
// camera/surface/targets changed) is dropped; the next render re-kicks.
void CChunkedVolumeViewer::finishPlaneIntersectionCompute(
    std::uint64_t gen, cv::Rect cacheRoi, IntersectFingerprint fp,
    std::shared_ptr<std::unordered_map<SurfacePatchIndex::SurfacePtr,
        std::vector<SurfacePatchIndex::TriangleSegment>>> result)
{
    _intersectComputeInFlight = false;
    if (_viewerManager)
        _viewerManager->endIndexRead();              // release the gate (may apply a deferred swap)
    if (_closing || gen != _intersectGen || !result)
        return;                                      // superseded or shutting down

    _intersectionGeometryCache = {};
    _intersectionGeometryCache.roi = cacheRoi;
    _intersectionGeometryCache.planeOriginQ = fp.planeOriginQ;
    _intersectionGeometryCache.planeNormalQ = fp.planeNormalQ;
    _intersectionGeometryCache.planeBasisXQ = fp.planeBasisXQ;
    _intersectionGeometryCache.planeBasisYQ = fp.planeBasisYQ;
    _intersectionGeometryCache.indexSamplingStride = fp.indexSamplingStride;
    _intersectionGeometryCache.patchCount = fp.patchCount;
    _intersectionGeometryCache.surfaceCount = fp.surfaceCount;
    _intersectionGeometryCache.targetHash = fp.targetHash;
    _intersectionGeometryCache.targetGenerationHash = fp.targetGenerationHash;
    _intersectionGeometryCache.intersections = std::move(*result);
    _intersectionGeometryCache.valid = true;
    // Force a rebuild of the scene items from the now-valid geometry cache: clear the
    // fp memo so the cache-hit early-out doesn't skip the build.
    _lastIntersectFp = {};
    renderIntersections("async intersection ready");
}

void CChunkedVolumeViewer::setHighlightedSurfaceIds(const std::vector<std::string>& ids)
{
    if (_closing) {
        return;
    }
    std::unordered_set<std::string> next;
    next.reserve(ids.size());
    for (const auto& id : ids) {
        next.insert(id);
    }
    if (_highlightedSurfaceIds == next) {
        return;
    }
    _highlightedSurfaceIds.clear();
    _highlightedSurfaceIds = std::move(next);
    renderIntersections("highlighted surface ids changed");
}

const VolumeViewerBase::ActiveSegmentationHandle& CChunkedVolumeViewer::activeSegmentationHandle() const
{
    static ActiveSegmentationHandle handle;
    return handle;
}

const std::vector<ViewerOverlayControllerBase::PathPrimitive>& CChunkedVolumeViewer::drawingPaths() const
{
    return _drawingPaths;
}

const std::map<std::string, cv::Vec3b>& CChunkedVolumeViewer::surfaceOverlays() const
{
    return _surfaceOverlays;
}

void CChunkedVolumeViewer::updateStatusLabel()
{
    if (!_statsBar)
        return;

    // Row 1: view state. Row 2: the data pipeline (cache + per-stage queues).
    QStringList view;
    view << QString("L%1").arg(_dsScaleIdx);
    view << QString("scale %1").arg(_scale, 0, 'f', 2);
    view << QString("%1x%2").arg(_framebuffer.width()).arg(_framebuffer.height());

    if (_compositeSettings.enabled || _compositeSettings.planeEnabled) {
        view << QString("composite %1").arg(QString::fromStdString(_compositeSettings.params.method));
    }

    auto surf = _surfWeak.lock();
    if (_lastCursorVolumePos)
        view << formatWholeVolumePosition(*_lastCursorVolumePos);
    if (dynamic_cast<QuadSurface*>(surf.get())) {
        view << QString("normal offset %1").arg(_zOff, 0, 'f', 1);
        if (_state) {
            if (auto* poi = _state->poi("focus"))
                view << QString("POI %1").arg(formatVec3(poi->p));
        }
    } else if (property("vc_show_custom_normal_offset").toBool()) {
        view << QString("normal offset %1")
                     .arg(property("vc_custom_normal_offset_vx").toDouble(), 0, 'f', 1);
    }

    QStringList pipeline;
    if (_chunkArray) {
        const auto stats = _chunkArray->stats();
        pipeline << QString("RAM %1/%2 GB")
            .arg(formatGigabytes(stats.decodedBytes))
            .arg(formatGigabytes(stats.decodedByteCapacity));
        pipeline << QString("disk %1").arg(formatByteSize(stats.persistentCacheBytes));
        // Gate on the rate (a held sliding-window average, steady through the
        // bursty batch arrivals) rather than the in-flight count, which flickers
        // to 0 between request bursts even while data is still streaming. Show
        // each pipeline stage only while it's >0. (Archive append is synchronous
        // -- there is no archive queue to report.)
        if (stats.remoteDownloadBytesPerSecond > 0.0) {
            const QString rate = formatMegabytesPerSecond(stats.remoteDownloadBytesPerSecond);
            pipeline << (stats.downloading > 0
                ? QString("downloading %1 @ %2").arg(stats.downloading).arg(rate)
                : QString("downloading @ %1").arg(rate));
        }
        if (stats.downloadQueued > 0)
            pipeline << QString("download queued %1").arg(stats.downloadQueued);
        if (stats.decodeQueued > 0)
            pipeline << QString("decode queued %1 (%2)")
                .arg(stats.decodeQueued)
                .arg(formatByteSize(stats.decodeStagingBytes));
        if (stats.encoding > 0)
            pipeline << QString("encoding %1").arg(stats.encoding);
    }

    _statsBar->setItems(view, pipeline);
}
