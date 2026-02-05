#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/SlicingLite.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/LifeTime.hpp"
#include "vc/core/types/ChunkedTensor.hpp"
#include <nlohmann/json.hpp>

#include "vc/core/util/NormalGridVolume.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/tracer/CostFunctions.hpp"
#include "vc/core/util/HashFunctions.hpp"

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/dataset.hxx"

#include <xtensor/views/xview.hpp>

#include <iostream>
#include <cctype>
#include <random>
#include <optional>
#include <cstdlib>
#include <limits>
#include <memory>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h>  // ensure omp_get_max_threads() is declared

#include "vc/tracer/GrowPatch_Internal.hpp"
#include "vc/tracer/Tracer.hpp"
#include "vc/ui/VCCollection.hpp"
#include "vc/tracer/NeuralTracerConnection.h"

namespace { // Anonymous namespace for local helpers

std::optional<uint32_t> environment_seed()
{
    static const std::optional<uint32_t> cached = []() -> std::optional<uint32_t> {
        const char* env = std::getenv("VC_GROWPATCH_RNG_SEED");
        if (!env || *env == '\0') {
            return std::nullopt;
        }

        char* end = nullptr;
        const unsigned long long value = std::strtoull(env, &end, 10);
        if (!end || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }
        return static_cast<uint32_t>(value);
    }();

    return cached;
}

std::mt19937& thread_rng()
{
    static thread_local std::mt19937 rng = [] {
        if (const auto seed = environment_seed()) {
            return std::mt19937(*seed);
        }
        return std::mt19937(std::random_device{}());
    }();
    return rng;
}

[[maybe_unused]] void set_random_perturbation_seed(uint32_t seed)
{
    thread_rng().seed(seed);
}

cv::Vec3d random_perturbation(double max_abs_offset = 0.05) {
    std::uniform_real_distribution<double> dist(-max_abs_offset, max_abs_offset);
    auto& rng = thread_rng();
    return {dist(rng), dist(rng), dist(rng)};
}

struct Vec2iLess {
    [[gnu::always_inline]] bool operator()(const cv::Vec2i& a, const cv::Vec2i& b) const noexcept {
        if (a[0] != b[0]) {
            return a[0] < b[0];
        }
        return a[1] < b[1];
    }
};

static std::vector<cv::Vec2i> parse_growth_directions(const std::vector<std::string>& directions_input)
{
    static const std::vector<cv::Vec2i> kDefaultDirections = {
        {1, 0},    // down / +row
        {0, 1},    // right / +col
        {-1, 0},   // up / -row
        {0, -1},   // left / -col
        {1, 1},    // down-right
        {1, -1},   // down-left
        {-1, 1},   // up-right
        {-1, -1}   // up-left
    };

    if (directions_input.empty()) {
        return kDefaultDirections;
    }

    std::vector<cv::Vec2i> custom;
    custom.reserve(kDefaultDirections.size());
    bool any_valid = false;

    auto append_unique = [&custom](const cv::Vec2i& dir) {
        for (const auto& existing : custom) {
            if (existing[0] == dir[0] && existing[1] == dir[1]) {
                return;
            }
        }
        custom.push_back(dir);
    };

    for (const auto& value : directions_input) {
        std::string normalized;
        normalized.reserve(value.size());
        for (char ch : value) {
            const unsigned char uch = static_cast<unsigned char>(ch);
            char lower = static_cast<char>(std::tolower(uch));
            if (lower == '-' || lower == '_' || lower == ' ') {
                continue;
            }
            normalized.push_back(lower);
        }

        if (normalized.empty()) {
            std::cerr << "Empty growth direction entry ignored" << "\n";
            continue;
        }

        if (normalized == "all" || normalized == "default") {
            return kDefaultDirections;
        }

        auto mark_valid = [&](const cv::Vec2i& dir) {
            append_unique(dir);
            any_valid = true;
        };

        if (normalized == "down") {
            mark_valid({1, 0});
            continue;
        }
        if (normalized == "right") {
            mark_valid({0, 1});
            continue;
        }
        if (normalized == "up") {
            mark_valid({-1, 0});
            continue;
        }
        if (normalized == "left") {
            mark_valid({0, -1});
            continue;
        }
        if (normalized == "downright") {
            mark_valid({1, 1});
            continue;
        }
        if (normalized == "downleft") {
            mark_valid({1, -1});
            continue;
        }
        if (normalized == "upright") {
            mark_valid({-1, 1});
            continue;
        }
        if (normalized == "upleft") {
            mark_valid({-1, -1});
            continue;
        }

        std::cerr << "Unknown growth direction '" << value << "' ignored" << "\n";
    }

    if (!any_valid) {
        return kDefaultDirections;
    }

    return custom;
}

} // namespace

// Out-of-line PointCorrection definitions (class is in GrowPatch_Internal.hpp)
PointCorrection::PointCorrection(const VCCollection& corrections) {
    const auto& collections = corrections.getAllCollections();
    if (collections.empty()) return;

    for (const auto& pair : collections) {
        const auto& collection = pair.second;
        // Allow collections with anchor2d set even if they have no points (drag-and-drop case)
        if (collection.points.empty() && !collection.anchor2d.has_value()) continue;

        CorrectionCollection new_collection;
        new_collection.anchor2d_ = collection.anchor2d;

        std::vector<ColPoint> sorted_points;
        sorted_points.reserve(collection.points.size());
        for (const auto& point_pair : collection.points) {
            sorted_points.push_back(point_pair.second);
        }
        std::sort(sorted_points.begin(), sorted_points.end(), [](const auto& a, const auto& b) {
            return a.id < b.id;
        });

        new_collection.tgts_.reserve(sorted_points.size());
        for (const auto& col_point : sorted_points) {
            new_collection.tgts_.push_back(col_point.p);
        }
        collections_.push_back(new_collection);
    }

    is_valid_ = !collections_.empty();
}

void PointCorrection::init(const cv::Mat_<cv::Vec3f> &points) {
    if (!is_valid_ || points.empty()) {
        is_valid_ = false;
        return;
    }

    QuadSurface tmp(points, {1.0f, 1.0f});

    for (auto& collection : collections_) {
        if (collection.anchor2d_.has_value()) {
            // Use the provided 2D anchor directly - this is the drag-and-drop case
            cv::Vec2f anchor = collection.anchor2d_.value();

            // Bounds check: skip collections where anchor2d is outside surface bounds
            if (anchor[0] < 0 || anchor[0] >= points.cols ||
                anchor[1] < 0 || anchor[1] >= points.rows) {
                std::cout << "Warning: skipping correction with out-of-bounds anchor2d: "
                          << anchor << " (surface size: " << points.cols << "x" << points.rows << ")" << "\n";
                continue;
            }

            std::cout << "using provided anchor2d: " << anchor << "\n";

            // Convert 2D grid location to pointer coordinates
            // pointer coords are relative to center: ptr = grid_loc - center
            // With scale {1,1}, center is {cols/2, rows/2, 0}
            cv::Vec3f ptr = {
                anchor[0] - points.cols / 2.0f,
                anchor[1] - points.rows / 2.0f,
                0.0f
            };

            // Search for all correction points from the anchor position
            for (size_t i = 0; i < collection.tgts_.size(); ++i) {
                float d = tmp.pointTo(ptr, collection.tgts_[i], 100.0f, 0);
                cv::Vec3f loc_3d = tmp.loc_raw(ptr);
                std::cout << "point diff: " << d << loc_3d << "\n";
                cv::Vec2f loc = {loc_3d[0], loc_3d[1]};
                collection.grid_locs_.push_back(loc);
            }
        } else {
            // Original behavior: use first point as anchor
            if (collection.tgts_.empty()) continue;

            cv::Vec3f ptr = tmp.pointer();

            // Initialize anchor point (lowest ID)
            float d = tmp.pointTo(ptr, collection.tgts_[0], 1.0f);
            cv::Vec3f loc_3d = tmp.loc_raw(ptr);
            std::cout << "base diff: " << d << loc_3d << "\n";
            cv::Vec2f loc(loc_3d[0], loc_3d[1]);
            collection.grid_locs_.push_back({loc[0], loc[1]});

            // Initialize other points
            for (size_t i = 1; i < collection.tgts_.size(); ++i) {
                d = tmp.pointTo(ptr, collection.tgts_[i], 100.0f, 0);
                loc_3d = tmp.loc_raw(ptr);
                std::cout << "point diff: " << d << loc_3d << "\n";
                loc = {loc_3d[0], loc_3d[1]};
                collection.grid_locs_.push_back({loc[0], loc[1]});
            }
        }
    }
}

struct NeuralTracerPointInfo {
    cv::Vec2i best_dir;
    std::optional<cv::Vec2i> best_ortho_pos;
    std::optional<cv::Vec2i> best_diag_pos;
    int max_score;
};

static NeuralTracerPointInfo compute_neural_tracer_point_info(
    const cv::Vec2i& p,
    TraceParameters& trace_params)
{
    auto is_valid = [&](const cv::Vec2i& pt) {
        return point_in_bounds(trace_params.state, pt) && (trace_params.state(pt) & STATE_LOC_VALID);
    };

    NeuralTracerPointInfo result;
    result.max_score = -1;

    const cv::Vec2i dirs[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (const auto& dir : dirs) {
        if (!is_valid(p - dir) || !is_valid(p - 2 * dir)) {  // check if center and prev_u (where u is dir-direction) are valid
            continue;
        }

        const cv::Vec2i center_pos = p - dir;

        const cv::Vec2i ortho_dirs[] = {{-dir[1], dir[0]}, {dir[1], -dir[0]}};

        for (const auto& ortho_dir : ortho_dirs) {

            auto current_ortho_pos = is_valid(center_pos - ortho_dir) ? std::optional{center_pos - ortho_dir} : std::nullopt;
            cv::Vec2i diag_dir;  // vector from prev_diag to center
            if (current_ortho_pos) {
                // if prev_u and prev_v both given, then prev_diag should be adjacent to the gap p (not to center)
                diag_dir = dir - ortho_dir;
            } else {
                // if only prev_u not prev_v given, then prev_diag should be opposite to the gap p along ortho_dir
                diag_dir = dir + ortho_dir;
            }
            auto current_diag_pos = is_valid(center_pos - diag_dir) ? std::optional{center_pos - diag_dir} : std::nullopt;

            int current_score =
                current_ortho_pos && current_diag_pos ? 3 :
                current_ortho_pos ? 2 :
                current_diag_pos ? 1 :
                0;  // only prev_u & center valid

            if (current_score > result.max_score) {
                result.max_score = current_score;
                result.best_dir = dir;
                result.best_ortho_pos = current_ortho_pos;
                result.best_diag_pos = current_diag_pos;
            }
        }
    }

    return result;
}

static std::vector<cv::Vec2i> call_neural_tracer_for_points(
    const std::vector<cv::Vec2i>& points,
    TraceParameters& trace_params,
    NeuralTracerConnection* neural_tracer)
{
    if (!neural_tracer) [[unlikely]]
        throw std::logic_error("Neural tracer connection is null");

    std::vector<cv::Vec3f> center_xyzs;
    std::vector<std::optional<cv::Vec3f>> prev_u_xyzs, prev_v_xyzs, prev_diag_xyzs;
    std::vector<cv::Vec2i> points_with_valid_dirs;

    for (auto const &p : points) {

        auto const point_info = compute_neural_tracer_point_info(p, trace_params);

        if (point_info.max_score < 1) {
            // we disallow score = -1 since this implies no neighbors found, and score = 0 since this is likely to create long, poorly-supported tendrils
            std::cout << "warning: max_score = " << point_info.max_score << "\n";
            continue;
        }

        auto get_point = [&](const cv::Vec2i& pos) {
            assert(point_in_bounds(trace_params.state, pos) && (trace_params.state(pos) & STATE_LOC_VALID));
            const auto& p_double_neighbor = trace_params.dpoints(pos);
            return cv::Vec3f(p_double_neighbor[0], p_double_neighbor[1], p_double_neighbor[2]);
        };

        center_xyzs.push_back(get_point(p - point_info.best_dir));
        prev_u_xyzs.push_back(get_point(p - 2 * point_info.best_dir));
        prev_v_xyzs.push_back(point_info.best_ortho_pos.transform(get_point));
        prev_diag_xyzs.push_back(point_info.best_diag_pos.transform(get_point));
        points_with_valid_dirs.push_back(p);
    }

    if (points_with_valid_dirs.empty()) {
        return {};
    }

    auto next_uvs = neural_tracer->get_next_points(center_xyzs, prev_u_xyzs, prev_v_xyzs, prev_diag_xyzs);

    std::vector<cv::Vec2i> successful_points;
    for (int point_idx = 0; point_idx < points_with_valid_dirs.size(); point_idx++) {
        const auto& p = points_with_valid_dirs[point_idx];
        const auto& candidates = next_uvs[point_idx].next_u_xyzs;
        if (!candidates.empty() && cv::norm(candidates[0]) > 1e-6) {
            trace_params.dpoints(p) = {candidates[0][0], candidates[0][1], candidates[0][2]};
            trace_params.state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
            successful_points.push_back(p);
        } else {
            std::cout << "warning: no valid next point found at " << p << "\n";
        }
    }

    return successful_points;
}

// global CUDA to allow use to set to false globally
// in the case they have cuda avail, but do not want to use it
static bool g_use_cuda = true;

// Expose a simple toggle for CUDA usage so tools can honor JSON settings
void set_space_tracing_use_cuda(bool enable) {
    g_use_cuda = enable;
}

template <typename E>
[[gnu::always_inline]] static inline E max_dist_ignore(const E &a, const E &b) noexcept
{
    if (a == E(-1))
        return b;
    if (b == E(-1))
        return a;
    return std::max(a,b);
}

template <typename T, typename E>
static void dist_iteration(T &from, T &to, int s)
{
    E magic = -1;
#pragma omp parallel for collapse(2)
    for(int k=0;k<s;k++)
        for(int j=0;j<s;j++)
            for(int i=0;i<s;i++) {
                E dist = from(k,j,i);
                if (dist == magic) {
                    if (k) dist = max_dist_ignore(dist, from(k-1,j,i));
                    if (k < s-1) dist = max_dist_ignore(dist, from(k+1,j,i));
                    if (j) dist = max_dist_ignore(dist, from(k,j-1,i));
                    if (j < s-1) dist = max_dist_ignore(dist, from(k,j+1,i));
                    if (i) dist = max_dist_ignore(dist, from(k,j,i-1));
                    if (i < s-1) dist = max_dist_ignore(dist, from(k,j,i+1));
                    if (dist != magic)
                        to(k,j,i) = dist+1;
                    else
                        to(k,j,i) = dist;
                }
                else
                    to(k,j,i) = dist;

            }
}

template <typename T, typename E>
static T distance_transform(const T &chunk, int steps, int size)
{
    T c1 = xt::empty<E>(chunk.shape());
    T c2 = xt::empty<E>(chunk.shape());

    c1 = chunk;

    E magic = -1;

    for(int n=0;n<steps/2;n++) {
        dist_iteration<T,E>(c1,c2,size);
        dist_iteration<T,E>(c2,c1,size);
    }

#pragma omp parallel for collapse(2)
    for(int z=0;z<size;z++)
        for(int y=0;y<size;y++)
            for(int x=0;x<size;x++)
                if (c1(z,y,x) == magic)
                    c1(z,y,x) = steps;

    return c1;
}

struct thresholdedDistance
{
    static constexpr int BORDER = 16;
    static constexpr int CHUNK_SIZE = 64;
    static constexpr int FILL_V = 0;
    static constexpr int TH = 170;
    const std::string UNIQUE_ID_STRING = "dqk247q6vz_"+std::to_string(BORDER)+"_"+std::to_string(CHUNK_SIZE)+"_"+std::to_string(FILL_V)+"_"+std::to_string(TH);
    template <typename T, typename E> void compute(const T &large, T &small, const cv::Vec3i &offset_large)
    {
        T outer = xt::empty<E>(large.shape());

        constexpr int s = CHUNK_SIZE+2*BORDER;
        E magic = -1;

#pragma omp parallel for collapse(2)
        for(int z=0;z<s;z++)
            for(int y=0;y<s;y++)
                for(int x=0;x<s;x++)
                    outer(z,y,x) = (large(z,y,x) < TH) ? magic : E(0);

        outer = distance_transform<T,E>(outer, 15, s);

        constexpr int low = BORDER;
        constexpr int high = BORDER + CHUNK_SIZE;

        auto crop_outer = view(outer, xt::range(low,high),xt::range(low,high),xt::range(low,high));

        small = crop_outer;
    }

};


QuadSurface *tracer(z5::Dataset *ds, float scale, ChunkCache<uint8_t> *cache, cv::Vec3f origin, const TracerParams &params, const std::string &cache_root, float voxelsize, std::vector<DirectionField> const &direction_fields, QuadSurface* resume_surf, const std::filesystem::path& tgt_path, const std::string& meta_params_json, const VCCollection* corrections)
{
    std::unique_ptr<NeuralTracerConnection> neural_tracer;
    int pre_neural_gens = 0, neural_batch_size = 1;
    if (!params.neural_socket.empty()) {
        try {
            neural_tracer = std::make_unique<NeuralTracerConnection>(params.neural_socket);
            std::cout << "Neural tracer connection enabled on " << params.neural_socket << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to connect neural tracer: " << e.what() << "\n";
            throw;
        }
        pre_neural_gens = params.pre_neural_generations;
        neural_batch_size = params.neural_batch_size;
        if (!neural_tracer) {
            std::cout << "Neural tracer not active" << "\n";
        }
    }
    TraceData trace_data(direction_fields);
    LossSettings loss_settings;
    loss_settings.applyWeights(params);

    // Optional fitted-3D normals field (direction-field zarr root with x/<scale>,y/<scale>,z/<scale>).
    // IMPORTANT: We auto-derive the correct scale factor from dataset shapes and ignore any JSON scale parameter.
    if (!params.normal3d_zarr_path.empty()) {
        try {
            const std::filesystem::path zarr_root = params.normal3d_zarr_path;

            // Expect fixed layout: <root>/{x,y,z}/0
            const int scale_level = 0;

            // Read delimiter from x/0/.zarray.
            // Also assert direction-field fill_value uses the neutral (128,128,128) convention.
            // We treat that triplet as "no normal".
            const auto assert_fill_value_128 = [&](const char* axis) {
                const std::filesystem::path zarray_axis = zarr_root / axis / "0" / ".zarray";
                if (!std::filesystem::exists(zarray_axis)) {
                    throw std::runtime_error(std::string("Missing ") + axis + "/0/.zarray under normal3d_zarr_path: " + zarr_root.string());
                }
                nlohmann::json j = nlohmann::json::parse(std::ifstream(zarray_axis));
                if (!j.contains("fill_value")) {
                    throw std::runtime_error(std::string("Missing fill_value in ") + axis + "/0/.zarray under normal3d_zarr_path: " + zarr_root.string());
                }
                const int fv = j["fill_value"].get<int>();
                if (fv != 128) {
                    std::stringstream msg;
                    msg << "normal3d_zarr_path fill_value=" << fv << " for " << axis << "/0; expected 128";
                    throw std::runtime_error(msg.str());
                }
            };

            assert_fill_value_128("x");
            assert_fill_value_128("y");
            assert_fill_value_128("z");

            const std::filesystem::path zarray_x = zarr_root / "x" / "0" / ".zarray";
            nlohmann::json j = nlohmann::json::parse(std::ifstream(zarray_x));
            std::string delim = j.value<std::string>("dimension_separator", ".");

            // Assert the direction-field was aligned by vc_ngrids --align-normals.
            try {
                z5::filesystem::handle::File rootFile(zarr_root);
                z5::filesystem::handle::Group root(rootFile, "");
                nlohmann::json attrs;
                z5::filesystem::readAttributes(root, attrs);
                const bool aligned = attrs.value("align_normals", false);
                if (!aligned) {
                    throw std::runtime_error("normal3d_zarr_path is not marked aligned (missing attrs.align_normals=true); run vc_ngrids --align-normals");
                }
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Failed normal3d alignment check: ") + e.what());
            }

            // Derive scale purely from shapes: main volume is full-res, normal zarr is downsampled.
            const auto vol_shape_zyx = ds->shape();
            const int vol_z = static_cast<int>(vol_shape_zyx.at(0));
            const int vol_y = static_cast<int>(vol_shape_zyx.at(1));
            const int vol_x = static_cast<int>(vol_shape_zyx.at(2));

            z5::filesystem::handle::Group dirs_group(zarr_root.string(), z5::FileMode::FileMode::r);
            z5::filesystem::handle::Group x_group(dirs_group, "x");
            z5::filesystem::handle::Dataset x_ds_handle(x_group, "0", delim);
            auto x_ds = z5::filesystem::openDataset(x_ds_handle);
            const auto nshape = x_ds->shape();
            if (nshape.size() != 3) {
                throw std::runtime_error("normal3d x/0 dataset is not 3D");
            }
            const int nz = static_cast<int>(nshape.at(0));
            const int ny = static_cast<int>(nshape.at(1));
            const int nx = static_cast<int>(nshape.at(2));
            if (nz <= 0 || ny <= 0 || nx <= 0) {
                throw std::runtime_error("normal3d x/0 dataset has invalid shape");
            }
            // The normal3d dataset is typically generated on a lattice with
            // n = ceil(vol / step), so vol may NOT be divisible by n.
            // Derive the step (=downsample ratio) by rounding vol/n.
            const double rz_f = static_cast<double>(vol_z) / static_cast<double>(nz);
            const double ry_f = static_cast<double>(vol_y) / static_cast<double>(ny);
            const double rx_f = static_cast<double>(vol_x) / static_cast<double>(nx);

            const int rz = std::max(1, static_cast<int>(std::llround(rz_f)));
            const int ry = std::max(1, static_cast<int>(std::llround(ry_f)));
            const int rx = std::max(1, static_cast<int>(std::llround(rx_f)));

            // Be tolerant of off-by-one shape effects from ceil() at the boundary.
            // Require all axes to agree after rounding.
            if (rz != ry || ry != rx) {
                std::stringstream msg;
                msg << "normal3d downsample ratio differs across axes after rounding: "
                    << "rx=" << rx << " ry=" << ry << " rz=" << rz
                    << " (vol=" << vol_x << "x" << vol_y << "x" << vol_z
                    << ", n=" << nx << "x" << ny << "x" << nz << ")";
                throw std::runtime_error(msg.str());
            }
            const int ratio = rx;

            const float scale_factor = 1.0f / static_cast<float>(ratio);

            std::vector<std::unique_ptr<z5::Dataset>> dss;
            for (auto dim : std::string("xyz")) {
                z5::filesystem::handle::Group dim_group(dirs_group, std::string(&dim, 1));
                z5::filesystem::handle::Dataset ds_handle(dim_group, std::to_string(scale_level), delim);
                dss.push_back(z5::filesystem::openDataset(ds_handle));
            }

            const std::string unique_id = std::to_string(std::hash<std::string>{}(dirs_group.path().string()));
            trace_data.normal3d_field = std::make_unique<Chunked3dVec3fFromUint8>(std::move(dss), scale_factor, cache, cache_root, unique_id + "_n3d");

            // Optional normal-fit diagnostics (written by vc_ngrids) to modulate loss weights.
            // Expected layout: <root>/fit_rms/0 and <root>/fit_frac_short_paths/0 (uint8, ZYX).
            try {
                z5::filesystem::handle::Group g_rms(dirs_group, "fit_rms");
                z5::filesystem::handle::Dataset ds_rms_handle(g_rms, std::to_string(scale_level), delim);
                auto ds_rms = z5::filesystem::openDataset(ds_rms_handle);

                z5::filesystem::handle::Group g_frac(dirs_group, "fit_frac_short_paths");
                z5::filesystem::handle::Dataset ds_frac_handle(g_frac, std::to_string(scale_level), delim);
                auto ds_frac = z5::filesystem::openDataset(ds_frac_handle);

                trace_data.normal3d_fit_quality = std::make_unique<NormalFitQualityWeightField>(
                    std::move(ds_rms), std::move(ds_frac), scale_factor, cache, cache_root, unique_id + "_n3d_fitq");
            } catch (const std::exception& e) {
                std::cerr << "Normal3d fit-quality fields not loaded (optional): " << e.what() << "\n";
                trace_data.normal3d_fit_quality.reset();
            }

            std::cout << "Loaded normal3d zarr field from " << zarr_root
                      << " (ratio=" << ratio
                      << ", scale_factor=" << scale_factor
                      << ", delim='" << delim << "')" << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed to load normal3d zarr field: " << e.what() << "\n";
            trace_data.normal3d_field.reset();
            trace_data.normal3d_fit_quality.reset();
        }
    }

    std::unique_ptr<QuadSurface> reference_surface;
    if (params.reference_surface) {
        const auto& ref_cfg = *params.reference_surface;

        if (!ref_cfg.path.empty()) {
            try {
                reference_surface = load_quad_from_tifxyz(ref_cfg.path);
                loss_settings.reference_raycast.surface = reference_surface.get();
                std::cout << "Loaded reference surface from " << ref_cfg.path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Failed to load reference surface '" << ref_cfg.path << "': " << e.what() << "\n";
            }
        } else {
            std::cerr << "reference_surface parameter provided without a valid path" << "\n";
        }

        if (loss_settings.reference_raycast.surface) {
            loss_settings.reference_raycast.voxel_threshold = std::clamp(ref_cfg.voxel_threshold, 0.0, 255.0);
            loss_settings.w[LossType::REFERENCE_RAY] = std::max(0.0f, static_cast<float>(ref_cfg.penalty_weight));
            loss_settings.reference_raycast.sample_step = std::max(ref_cfg.sample_step, 1e-6);
            loss_settings.reference_raycast.max_distance = ref_cfg.max_distance;
            loss_settings.reference_raycast.min_clearance = std::max(ref_cfg.min_clearance, 0.0);
            loss_settings.reference_raycast.clearance_weight = std::max(ref_cfg.clearance_weight, 0.0);
        }

        if (loss_settings.reference_raycast_enabled()) {
            std::cout << "Reference raycast penalty enabled (threshold="
                      << loss_settings.reference_raycast.voxel_threshold
                      << ", weight=" << loss_settings.w[LossType::REFERENCE_RAY]
                      << ", step=" << loss_settings.reference_raycast.sample_step
                      << ", min_clearance=" << loss_settings.reference_raycast.min_clearance
                      << " (clear_w=" << loss_settings.reference_raycast.clearance_weight << ")";
            if (loss_settings.reference_raycast.max_distance > 0.0) {
                std::cout << ", max_distance=" << loss_settings.reference_raycast.max_distance;
            }
            std::cout << ")" << "\n";
        }
    }
    TraceParameters trace_params;
    int snapshot_interval = params.snapshot_interval;
    int stop_gen = params.generations;

    // Load normal grid first if provided, so we can use its spiral-step
    std::unique_ptr<vc::core::util::NormalGridVolume> ngv;
    if (!params.normal_grid_path.empty()) {
        ngv = std::make_unique<vc::core::util::NormalGridVolume>(params.normal_grid_path);
    }

    // Determine step size with priority: explicit param > normal_grid > resume_surf > default
    float step;
    if (params.step_size > 0.0f) {
        step = params.step_size;
    } else if (ngv) {
        // Use normal grid's spiral-step as authoritative (handles legacy surfaces with wrong scale)
        step = ngv->spiral_step();
    } else if (resume_surf) {
        step = 1.0f / resume_surf->scale()[0];
    } else {
        step = 20.0f;
    }
    trace_params.unit = step*scale;

    // Validate step matches normal grid if explicit step_size was provided
    if (ngv && params.step_size > 0.0f) {
        float ngv_step = ngv->spiral_step();
        if (std::abs(ngv_step - step) > 1e-6) {
            throw std::runtime_error("step_size parameter mismatch between normal grid volume and tracer.");
        }
    }
    trace_data.ngv = ngv.get();

    std::cout << "GrowPatch loss weights:\n"
              << "  DIST: " << loss_settings.w[LossType::DIST]
              << " STRAIGHT: " << loss_settings.w[LossType::STRAIGHT]
              << " DIRECTION: " << loss_settings.w[LossType::DIRECTION]
              << " SNAP: " << loss_settings.w[LossType::SNAP]
              << " NORMAL: " << loss_settings.w[LossType::NORMAL]
              << " NORMAL3DLINE: " << loss_settings.w[LossType::NORMAL3DLINE]
              << " REFERENCE_RAY: " << loss_settings.w[LossType::REFERENCE_RAY]
              << " SDIR: " << loss_settings.w[LossType::SDIR]
              << "\n";
    int rewind_gen = params.rewind_gen;
    loss_settings.z_min = params.z_min;
    loss_settings.z_max = params.z_max;
    loss_settings.y_min = params.y_min;
    loss_settings.y_max = params.y_max;
    loss_settings.x_min = params.x_min;
    loss_settings.x_max = params.x_max;
    loss_settings.flipback_threshold = params.flipback_threshold;
    loss_settings.flipback_weight = params.flipback_weight;
    std::cout << "Anti-flipback: threshold=" << loss_settings.flipback_threshold
              << " weight=" << loss_settings.flipback_weight
              << (loss_settings.flipback_weight == 0 ? " (DISABLED)" : "") << "\n";
    ALifeTime f_timer("empty space tracing\n");



    int w, h;
    if (resume_surf) {
        cv::Mat_<uint16_t> resume_generations = resume_surf->channel("generations");
        if (resume_generations.empty()) {
            cv::Mat_<cv::Vec3f> resume_points = resume_surf->rawPoints();
            resume_generations = cv::Mat_<uint16_t>(resume_points.size(), (uint16_t)0);

            for (int j = 0; j < resume_points.rows; ++j) {
                for (int i = 0; i < resume_points.cols; ++i) {
                    if (resume_points(j,i)[0] != -1) {
                        resume_generations(j,i) = 1;
                    }
                }
            }
            resume_surf->setChannel("generations", resume_generations);
        }
        double min_val, max_val;
        cv::minMaxLoc(resume_generations, &min_val, &max_val);
        int start_gen = (rewind_gen == -1) ? static_cast<int>(max_val) : rewind_gen;
        int gen_diff = std::max(0, stop_gen - start_gen);
        w = resume_generations.cols + 2 * gen_diff + 50;
        h = resume_generations.rows + 2 * gen_diff + 50;
    } else {
        // Calculate the maximum possible size the patch might grow to
        //FIXME show and handle area edge!
        w = 2*stop_gen+50;
        h = w;
    }
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w,h);

    int x0 = w/2;
    int y0 = h/2;

    // Together these represent the cached distance-transform of the thresholded surface volume
    thresholdedDistance compute;
    Chunked3d<uint8_t,thresholdedDistance> proc_tensor(compute, ds, cache, cache_root);

    // Debug: test the chunk cache by reading one voxel
    passTroughComputor pass;
    Chunked3d<uint8_t,passTroughComputor> dbg_tensor(pass, ds, cache);
    trace_data.raw_volume = &dbg_tensor;
    std::cout << "seed val " << origin << " " <<
    static_cast<int>(dbg_tensor(origin[2],origin[1],origin[0])) << "\n";

    auto timer = new ALifeTime("search & optimization ...");

    // This provides a cached interpolated version of the original surface volume
    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp_global(proc_tensor);

    // fringe contains all 2D points around the edge of the patch where we might expand it
    // cands will contain new points adjacent to the fringe that are candidates to expand into
    std::vector<cv::Vec2i> fringe;
    std::vector<cv::Vec2i> cands;
    
    float T = step;
    // float Ts = step*scale;

    
    // The following track the state of the patch; they are each as big as the largest possible patch but initially empty
    // - locs defines the patch! It says for each 2D position, which 3D position it corresponds to
    // - state tracks whether each 2D position is part of the patch yet, and whether its 3D position has been found
    trace_params.dpoints = cv::Mat_<cv::Vec3d>(size,cv::Vec3f(-1,-1,-1));
    trace_params.state = cv::Mat_<uint8_t>(size,0);
    cv::Mat_<uint16_t> generations(size, (uint16_t)0);
    cv::Mat_<cv::Vec3d> surface_normals(size, cv::Vec3d(0,0,0));  // Consistently oriented surface normals
    cv::Mat_<uint8_t> phys_fail(size,0);
    // cv::Mat_<float> init_dist(size,0);
    cv::Mat_<uint16_t> loss_status(cv::Size(w,h),0);
    cv::Rect used_area;
    int generation = 1;

    int succ = 0;  // number of quads successfully added to the patch (each of size approx. step**2)

    int resume_pad_y = 0;
    int resume_pad_x = 0;

    auto create_surface_from_state = [&, &f_timer = *timer]() {
        cv::Rect used_area_safe = used_area;
        used_area_safe.x -= 2;
        used_area_safe.y -= 2;
        used_area_safe.width += 4;
        used_area_safe.height += 4;
        cv::Mat_<cv::Vec3d> points_crop = trace_params.dpoints(used_area_safe);
        cv::Mat_<uint16_t> generations_crop = generations(used_area_safe);

        auto surf = new QuadSurface(points_crop, {1/T, 1/T});
        surf->setChannel("generations", generations_crop);

        if (params.vis_losses) {
            cv::Mat_<float> loss_dist(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_straight(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_normal(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_snap(generations_crop.size(), 0.0f);
            cv::Mat_<float> loss_sdir(generations_crop.size(), 0.0f);

#pragma omp parallel for schedule(dynamic)
            for (int y = 0; y < generations_crop.rows; ++y) {
                for (int x = 0; x < generations_crop.cols; ++x) {
                    cv::Vec2i p = {used_area_safe.y + y, used_area_safe.x + x};
                    if (generations(p) > 0) {
                        ceres::Problem problem_dist, problem_straight, problem_normal, problem_snap, problem_sdir;
                        
                        LossSettings settings_dist, settings_straight, settings_normal, settings_snap, settings_sdir;
                        
                        settings_dist[DIST] = 1.0;
                        add_losses(problem_dist, p, trace_params, trace_data, settings_dist, LOSS_DIST);
                        
                        settings_straight[STRAIGHT] = 1.0;
                        add_losses(problem_straight, p, trace_params, trace_data, settings_straight, LOSS_STRAIGHT);
                        
                        settings_normal[NORMAL] = 1.0;
                        add_losses(problem_normal, p, trace_params, trace_data, settings_normal, LOSS_NORMALSNAP);

                        settings_snap[SNAP] = 1.0;
                        add_losses(problem_snap, p, trace_params, trace_data, settings_snap, LOSS_NORMALSNAP);

                        settings_sdir[SDIR] = 1.0;
                        add_losses(problem_sdir, p, trace_params, trace_data, settings_sdir, LOSS_SDIR);

                        double cost_dist = 0, cost_straight = 0, cost_normal = 0, cost_snap = 0, cost_sdir = 0;
                        problem_dist.Evaluate(ceres::Problem::EvaluateOptions(), &cost_dist, nullptr, nullptr, nullptr);
                        problem_straight.Evaluate(ceres::Problem::EvaluateOptions(), &cost_straight, nullptr, nullptr, nullptr);
                        problem_normal.Evaluate(ceres::Problem::EvaluateOptions(), &cost_normal, nullptr, nullptr, nullptr);
                        problem_snap.Evaluate(ceres::Problem::EvaluateOptions(), &cost_snap, nullptr, nullptr, nullptr);
                        problem_sdir.Evaluate(ceres::Problem::EvaluateOptions(), &cost_sdir, nullptr, nullptr, nullptr);

                        loss_dist(y, x) = sqrt(cost_dist);
                        loss_straight(y, x) = sqrt(cost_straight);
                        loss_normal(y, x) = sqrt(cost_normal);
                        loss_snap(y, x) = sqrt(cost_snap);
                        loss_sdir(y, x) = sqrt(cost_sdir);
                    }
                }
            }
            surf->setChannel("loss_dist", loss_dist);
            surf->setChannel("loss_straight", loss_straight);
            surf->setChannel("loss_normal", loss_normal);
            surf->setChannel("loss_snap", loss_snap);
            surf->setChannel("loss_sdir", loss_sdir);
        }

        const double area_est_vx2 = vc::surface::computeSurfaceAreaVox2(*surf);
        const double voxel_size_d = static_cast<double>(voxelsize);
        const double area_est_cm2 = area_est_vx2 * voxel_size_d * voxel_size_d / 1e8;

        // Populate typed meta fields from meta_params JSON string
        if (!meta_params_json.empty()) {
            auto meta_params = nlohmann::json::parse(meta_params_json);
            for (auto it = meta_params.begin(); it != meta_params.end(); ++it) {
                if (it.key() == "source") {
                    surf->meta.source = it.value().get<std::string>();
                } else {
                    surf->meta.extras[it.key()] = it.value().dump();
                }
            }
        }
        surf->meta.area_vx2 = area_est_vx2;
        surf->meta.area_cm2 = area_est_cm2;
        surf->meta.max_gen = generation;
        surf->meta.seed = cv::Vec3f{origin[0], origin[1], origin[2]};
        surf->meta.elapsed_time_s = f_timer.seconds();
        if (resume_surf && !resume_surf->id.empty()) {
            surf->meta.seed_surface_id = resume_surf->id;
            // Store grid offset for correction point remapping
            // new_coord = old_coord + offset
            const int offset_row = resume_pad_y - used_area_safe.y;
            const int offset_col = resume_pad_x - used_area_safe.x;
            surf->meta.grid_offset = std::array<int, 2>{offset_col, offset_row};
        }

        // Preserve approval and mask channels from resume surface with correct offset
        // Note: These channels are stored at raw points resolution, not scaled size
        if (resume_surf) {
            const int offset_row = resume_pad_y - used_area_safe.y;
            const int offset_col = resume_pad_x - used_area_safe.x;

            // Get raw points size (channels are stored at this resolution)
            const cv::Mat_<cv::Vec3f>* new_points = surf->rawPointsPtr();
            if (!new_points || new_points->empty()) [[unlikely]] {
                return surf;
            }
            const cv::Size raw_size = new_points->size();

            // Preserve approval channel (3-channel BGR image)
            cv::Mat old_approval = resume_surf->channel("approval", SURF_CHANNEL_NORESIZE);
            if (!old_approval.empty()) {
                // Create new approval mask matching old format at raw points resolution
                cv::Mat new_approval;
                if (old_approval.channels() == 3) {
                    new_approval = cv::Mat_<cv::Vec3b>(raw_size, cv::Vec3b(0, 0, 0));
                    for (int r = 0; r < old_approval.rows; ++r) {
                        for (int c = 0; c < old_approval.cols; ++c) {
                            int new_r = r + offset_row;
                            int new_c = c + offset_col;
                            if (new_r >= 0 && new_r < new_approval.rows &&
                                new_c >= 0 && new_c < new_approval.cols) {
                                new_approval.at<cv::Vec3b>(new_r, new_c) = old_approval.at<cv::Vec3b>(r, c);
                            }
                        }
                    }
                } else {
                    // Single channel fallback
                    new_approval = cv::Mat_<uint8_t>(raw_size, static_cast<uint8_t>(0));
                    for (int r = 0; r < old_approval.rows; ++r) {
                        for (int c = 0; c < old_approval.cols; ++c) {
                            int new_r = r + offset_row;
                            int new_c = c + offset_col;
                            if (new_r >= 0 && new_r < new_approval.rows &&
                                new_c >= 0 && new_c < new_approval.cols) {
                                new_approval.at<uint8_t>(new_r, new_c) = old_approval.at<uint8_t>(r, c);
                            }
                        }
                    }
                }
                surf->setChannel("approval", new_approval);
                std::cout << "Preserved approval mask (" << old_approval.channels() << " channels, "
                          << old_approval.cols << "x" << old_approval.rows << " -> "
                          << new_approval.cols << "x" << new_approval.rows
                          << ") with offset (" << offset_row << ", " << offset_col << ")" << "\n";
            }

            // Preserve mask channel (single channel uint8)
            // Layer 0 of mask.tif is the validity mask - it can mask out points that have
            // valid coordinates but shouldn't be used (human corrections).
            // Strategy:
            // 1. Generate fresh validity mask from new surface points (255 if valid, 0 if -1,-1,-1)
            // 2. Overlay old mask values at correct offset (preserving human edits)
            cv::Mat old_mask = resume_surf->channel("mask", SURF_CHANNEL_NORESIZE);
            if (!old_mask.empty()) {
                // Start with validity mask based on actual point data
                cv::Mat_<uint8_t> new_mask(raw_size, static_cast<uint8_t>(0));
                for (int r = 0; r < new_points->rows; ++r) {
                    for (int c = 0; c < new_points->cols; ++c) {
                        const cv::Vec3f& p = (*new_points)(r, c);
                        if (p[0] != -1.0f) {
                            new_mask(r, c) = 255;  // Valid point
                        }
                    }
                }

                // Now overlay the old mask values (preserving human edits that masked out valid points)
                for (int r = 0; r < old_mask.rows; ++r) {
                    for (int c = 0; c < old_mask.cols; ++c) {
                        int new_r = r + offset_row;
                        int new_c = c + offset_col;
                        if (new_r >= 0 && new_r < new_mask.rows &&
                            new_c >= 0 && new_c < new_mask.cols) {
                            // Preserve the old mask value (including human edits that set 0 on valid points)
                            new_mask(new_r, new_c) = old_mask.at<uint8_t>(r, c);
                        }
                    }
                }
                surf->setChannel("mask", new_mask);
                std::cout << "Preserved mask (" << old_mask.cols << "x" << old_mask.rows << " -> "
                          << new_mask.cols << "x" << new_mask.rows
                          << ") with offset (" << offset_row << ", " << offset_col << ")" << "\n";
            }
        }

        return surf;
    };

    cv::Vec3f vx = {1,0,0};
    cv::Vec3f vy = {0,1,0};

    // ceres::Problem big_problem;
    int loss_count = 0;
    double last_elapsed_seconds = 0.0;
    int last_succ = 0;
    int start_gen = 0;

    std::cout << "lets go! " << "\n";

    if (resume_surf) {
        std::cout << "resuime! " << "\n";
        float resume_step = 1.0 / resume_surf->scale()[0];
        // Only validate step match if not using normal_grid (which is authoritative for legacy surfaces)
        if (!ngv && std::abs(resume_step - step) > 1e-6) {
            throw std::runtime_error("Step size parameter mismatch between new trace and resume surface.");
        }

        cv::Mat_<cv::Vec3f> resume_points = resume_surf->rawPoints();
        cv::Mat_<uint16_t> resume_generations = resume_surf->channel("generations");

        resume_pad_x = (w - resume_points.cols) / 2;
        resume_pad_y = (h - resume_points.rows) / 2;

        used_area = cv::Rect(resume_pad_x, resume_pad_y, resume_points.cols, resume_points.rows);
 
        double min_val, max_val;
        cv::minMaxLoc(resume_generations, &min_val, &max_val);
        start_gen = (rewind_gen == -1) ? static_cast<int>(max_val) : rewind_gen;
        generation = start_gen;

        int min_gen = std::numeric_limits<int>::max();
        x0 = -1;
        y0 = -1;
        for (int j = 0; j < resume_points.rows; ++j) {
            for (int i = 0; i < resume_points.cols; ++i) {
                int target_y = resume_pad_y + j;
                int target_x = resume_pad_x + i;
                uint16_t gen = resume_generations.at<uint16_t>(j, i);
                if (gen > 0 && gen <= start_gen && resume_points(j,i)[0] != -1) {
                    trace_params.dpoints(target_y, target_x) = resume_points(j, i);
                    generations(target_y, target_x) = gen;
                    succ++;
                    trace_params.state(target_y, target_x) = STATE_LOC_VALID | STATE_COORD_VALID;
                    if (gen < min_gen) {
                        min_gen = gen;
                        x0 = target_x;
                        y0 = target_y;
                    }
                }
            }
        }

        if (corrections) {
            trace_data.point_correction = PointCorrection(*corrections);
        }

        if (trace_data.point_correction.isValid()) {
            trace_data.point_correction.init(trace_params.dpoints);

            std::cout << "Resuming with " << trace_data.point_correction.all_grid_locs().size() << " correction points." << "\n";
            cv::Mat mask = resume_surf->channel("mask");
            if (!mask.empty()) {
                std::vector<std::vector<cv::Point2f>> all_hulls;
                // For single-point collections (e.g., drag-and-drop), store center and radius
                std::vector<std::pair<cv::Point2f, float>> single_point_regions;

                for (const auto& collection : trace_data.point_correction.collections()) {
                    if (collection.grid_locs_.empty()) continue;

                    if (collection.grid_locs_.size() == 1) {
                        // Single point - use a radius-based region instead of convex hull
                        // This handles the drag-and-drop case where only one correction point is set
                        cv::Point2f center(collection.grid_locs_[0][0], collection.grid_locs_[0][1]);
                        float radius = 8.0f;  // Default radius for single-point corrections
                        single_point_regions.emplace_back(center, radius);
                        std::cout << "single-point correction region at " << center << " with radius " << radius << "\n";
                    } else {
                        std::vector<cv::Point2f> points_for_hull;
                        points_for_hull.reserve(collection.grid_locs_.size());
                        for (const auto& loc : collection.grid_locs_) {
                            points_for_hull.emplace_back(loc[0], loc[1]);
                        }

                        std::vector<cv::Point2f> hull_points;
                        cv::convexHull(points_for_hull, hull_points);
                        if (!hull_points.empty()) {
                            all_hulls.push_back(hull_points);
                        }
                    }
                }

                for (int j = 0; j < mask.rows; ++j) {
                    for (int i = 0; i < mask.cols; ++i) {
                        if (mask.at<uint8_t>(j, i) == 0 && trace_params.state(resume_pad_y + j, resume_pad_x + i)) {
                            int target_y = resume_pad_y + j;
                            int target_x = resume_pad_x + i;
                            cv::Point2f p(target_x, target_y);
                            bool keep = false;

                            // Check convex hull regions
                            for (const auto& hull : all_hulls) {
                                if (cv::pointPolygonTest(hull, p, false) >= 0) {
                                    keep = true;
                                    break;
                                }
                            }

                            // Check single-point circular regions
                            if (!keep) {
                                for (const auto& [center, radius] : single_point_regions) {
                                    float dx = p.x - center.x;
                                    float dy = p.y - center.y;
                                    if (dx * dx + dy * dy <= radius * radius) {
                                        keep = true;
                                        break;
                                    }
                                }
                            }

                            if (!keep) {
                                trace_params.state(target_y, target_x) = 0;
                                trace_params.dpoints(target_y, target_x) = cv::Vec3d(-1,-1,-1);
                                generations(target_y, target_x) = 0;
                                succ--;
                            }
                        }
                    }
                }
            }

            struct OptCenter {
                cv::Vec2i center;
                int radius;
            };
            std::vector<OptCenter> opt_centers;

            for (const auto& collection : trace_data.point_correction.collections()) {
                if (collection.grid_locs_.empty()) continue;

                cv::Vec2f avg_loc(0,0);
                for (const auto& loc : collection.grid_locs_) {
                    avg_loc += loc;
                }
                avg_loc *= (1.0f / collection.grid_locs_.size());

                float max_dist = 0.0f;
                for (const auto& loc : collection.grid_locs_) {
                    max_dist = std::max(max_dist, static_cast<float>(cv::norm(loc - avg_loc)));
                }

                int radius = 8 + static_cast<int>(std::ceil(max_dist));
                cv::Vec2i corr_center_i = { static_cast<int>(std::round(avg_loc[1])), static_cast<int>(std::round(avg_loc[0])) };
                opt_centers.push_back({corr_center_i, radius});

                std::cout << "correction opt centered at " << avg_loc << " with radius " << radius << "\n";
                LossSettings loss_inpaint = loss_settings;
                loss_inpaint[SNAP] *= 0.0;
                loss_inpaint[DIST] *= 0.3;
                loss_inpaint[STRAIGHT] *= 0.1;
                local_optimization(radius, corr_center_i, trace_params, trace_data, loss_inpaint, false, true);
            }

            // if (!tgt_path.empty() && snapshot_interval > 0) {
            //     QuadSurface* surf = create_surface_from_state();
            //     surf->save(tgt_path.string()+"_corr_stage1", true);
            //     delete surf;
            // }

            for (const auto& opt_params : opt_centers) {
                LossSettings loss_inpaint = loss_settings;
                loss_inpaint[DIST] *= 0.3;
                loss_inpaint[STRAIGHT] *= 0.1;
                local_optimization(opt_params.radius, opt_params.center, trace_params, trace_data, loss_inpaint, false, true);
            }

            // if (!tgt_path.empty() && snapshot_interval > 0) {
            //     QuadSurface* surf = create_surface_from_state();
            //     surf->save(tgt_path.string()+"_corr_stage4", true);
            //     delete surf;
            // }

            for (const auto& opt_params : opt_centers) {
                local_optimization(opt_params.radius, opt_params.center, trace_params, trace_data, loss_settings, false, true);
            }

            // if (!tgt_path.empty() && snapshot_interval > 0) {
            //     QuadSurface* surf = create_surface_from_state();
            //     surf->save(tgt_path.string()+"_corr_stage5", true);
            //     delete surf;
            // }

            if (!tgt_path.empty() && snapshot_interval > 0) {
                QuadSurface* surf = create_surface_from_state();
                surf->save(tgt_path, true);
                delete surf;
                std::cout << "saved snapshot in " << tgt_path << "\n";
            }
        }

        // Rebuild fringe from valid points
        for (int j = used_area.y; j < used_area.br().y; ++j) {
            for (int i = used_area.x; i < used_area.br().x; ++i) {
                if (trace_params.state(j, i) & STATE_LOC_VALID) {
                    fringe.push_back({j, i});
                }
            }
        }

        last_succ = succ;
        last_elapsed_seconds = f_timer.seconds();
        std::cout << "Resuming from generation " << generation << " with " << fringe.size() << " points. Initial loss count: " << loss_count << "\n";

    } else {
        // Initialize seed normals with consistent orientation (vx cross vy = +Z direction)
        cv::Vec3d seed_normal = cv::Vec3d(vx).cross(cv::Vec3d(vy));
        seed_normal /= cv::norm(seed_normal);

        if (neural_tracer && pre_neural_gens == 0) {
            std::cout << "Initializing with neural tracer..." << "\n";

            // Bootstrap the first quad with the neural tracer -- we already have the
            // top-left point; we construct top-right, bottom-left and bottom-right

            trace_params.dpoints(y0, x0) = origin;

            // Get hopefully-4 adjacent points; take the one with min or max z-displacement depending on required direction
            auto coordinates = neural_tracer->get_next_points({origin}, {{}}, {{}}, {{}})[0].next_u_xyzs;
            if (coordinates.empty() || cv::norm(coordinates[0]) < 1e-6) {
                std::cout << "no blobs found while bootstrapping (vertex #1, top-right)" << "\n";
                throw std::runtime_error("Neural tracer bootstrap failed at vertex #1");
            }
            // use minimum delta-z; this choice orients the patch
            auto min_delta_z_it = std::min_element(coordinates.begin(), coordinates.end(), [&origin](const cv::Vec3f& a, const cv::Vec3f& b) {
                return std::abs(a[2] - origin[2]) < std::abs(b[2] - origin[2]);
            });
            trace_params.dpoints(y0, x0 + 1) = *min_delta_z_it;

            // Conditioned on center and right, predict below & above (choosing one arbitrarily)
            cv::Vec3f prev_v = trace_params.dpoints(y0, x0 + 1);
            coordinates = neural_tracer->get_next_points({origin}, {{}}, {prev_v}, {{}})[0].next_u_xyzs;
            if (coordinates.empty() || cv::norm(coordinates[0]) < 1e-6) {
                std::cout << "no blobs found while bootstrapping (vertex #2, bottom-left)" << "\n";
                throw std::runtime_error("Neural tracer bootstrap failed at vertex #2");
            }
            trace_params.dpoints(y0 + 1, x0) = coordinates[0];

            // Conditioned on center (top-right of the quad!) and left and below-left, predict below
            cv::Vec3f center_xyz = trace_params.dpoints(y0, x0 + 1);
            prev_v = trace_params.dpoints(y0, x0);
            cv::Vec3f prev_diag = trace_params.dpoints(y0 + 1, x0);
            coordinates = neural_tracer->get_next_points({center_xyz}, {{}}, {prev_v}, {prev_diag})[0].next_u_xyzs;
            if (coordinates.empty() || cv::norm(coordinates[0]) < 1e-6) {
                std::cout << "no blobs found while bootstrapping (vertex #3, bottom-right)" << "\n";
                throw std::runtime_error("Neural tracer bootstrap failed at vertex #3");
            }
            trace_params.dpoints(y0 + 1, x0 + 1) = coordinates[0];

        } else {
            // Initialise the trace at the center of the available area, as a tiny single-quad patch at the seed point
            //these are locations in the local volume!
            trace_params.dpoints(y0,x0) = origin;
            trace_params.dpoints(y0,x0+1) = origin+vx*0.1;
            trace_params.dpoints(y0+1,x0) = origin+vy*0.1;
            trace_params.dpoints(y0+1,x0+1) = origin+vx*0.1 + vy*0.1;
        }

        used_area = cv::Rect(x0,y0,2,2);

        trace_params.state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0+1,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;
        trace_params.state(y0+1,x0+1) = STATE_LOC_VALID | STATE_COORD_VALID;

        generations(y0,x0) = 1;
        generations(y0,x0+1) = 1;
        generations(y0+1,x0) = 1;
        generations(y0+1,x0+1) = 1;

        surface_normals(y0,x0) = seed_normal;
        surface_normals(y0,x0+1) = seed_normal;
        surface_normals(y0+1,x0) = seed_normal;
        surface_normals(y0+1,x0+1) = seed_normal;

        fringe.push_back({y0,x0});
        fringe.push_back({y0+1,x0});
        fringe.push_back({y0,x0+1});
        fringe.push_back({y0+1,x0+1});
    }

    int succ_start = succ;

    // Solve the initial optimisation problem, just placing the first four vertices around the seed
    ceres::Solver::Summary big_summary;
    //just continue on resume no additional global opt
    if (!resume_surf) {
        if (!neural_tracer) {
            local_optimization(8, {y0,x0}, trace_params, trace_data, loss_settings, true);
        }
    }
    else
    {
        if (params.resume_opt == "global") {
            std::cout << "global opt" << "\n";
            local_optimization(100, {y0,x0}, trace_params, trace_data, loss_settings, false, true);
        }
        else if (params.resume_opt == "local") {
            int opt_step = params.resume_local_opt_step;
            if (opt_step <= 0) {
                std::cerr << "WARNING: resume_local_opt_step must be > 0; defaulting to 16" << "\n";
                opt_step = 16;
            }

            int default_radius = opt_step * 2;
            int opt_radius = params.resume_local_opt_radius >= 0 ? params.resume_local_opt_radius : default_radius;
            if (opt_radius <= 0) {
                std::cerr << "WARNING: resume_local_opt_radius must be > 0; defaulting to " << default_radius << "\n";
                opt_radius = default_radius;
            }

            LocalOptimizationConfig resume_local_config;
            resume_local_config.max_iterations = params.resume_local_max_iters;
            if (resume_local_config.max_iterations <= 0) {
                std::cerr << "WARNING: resume_local_max_iters must be > 0; defaulting to 1000" << "\n";
                resume_local_config.max_iterations = 1000;
            }
            resume_local_config.use_dense_qr = params.resume_local_dense_qr;

            std::cout << "local opt (step=" << opt_step
                      << ", radius=" << opt_radius
                      << ", max_iters=" << resume_local_config.max_iterations
                      << ", dense_qr=" << std::boolalpha << resume_local_config.use_dense_qr
                      << std::noboolalpha << ")" << "\n";
            std::vector<cv::Vec2i> opt_local;
            for (int j = used_area.y; j < used_area.br().y; ++j) {
                for (int i = used_area.x; i < used_area.br().x; ++i) {
                    if ((trace_params.state(j, i) & STATE_LOC_VALID) && (i % opt_step == 0 && j % opt_step == 0)) {
                        opt_local.push_back({j, i});
                    }
                }
            }

            std::atomic<int> done = 0;
            if (!opt_local.empty()) {
                OmpThreadPointCol opt_local_threadcol(opt_step*2+1, opt_local);
                int total = opt_local.size();
                auto start_time = std::chrono::high_resolution_clock::now();

                #pragma omp parallel
                while (true)
                {
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(opt_radius, p, trace_params, trace_data, loss_settings, true, false, &resume_local_config);
                    done++;
#pragma omp critical
                    {
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed_seconds = std::chrono::duration<double>(now - start_time).count();
                        double eta_seconds = (elapsed_seconds / done.load()) * (total - done.load());

                        printf("  optimizing... %d/%d (%.2f%%) | elapsed: %.1fs | eta: %.1fs\r",
                               done.load(), total, (100.0 * done.load() / total), elapsed_seconds, eta_seconds);
                        fflush(stdout);
                    }
                }
                printf("\n");
            }
        }
        else if (params.inpaint) {
            cv::Mat mask = resume_surf->channel("mask");
            cv::Mat_<uchar> hole_mask(trace_params.state.size(), (uchar)0);

            cv::Mat active_area_mask(trace_params.state.size(), (uchar)0);
            for (int y = 0; y < trace_params.state.rows; ++y) {
                for (int x = 0; x < trace_params.state.cols; ++x) {
                    if (trace_params.state(y, x) & STATE_LOC_VALID) {
                        active_area_mask.at<uchar>(y, x) = 255;
                    }
                }
            }

            if (!mask.empty()) {
                cv::Mat padded_mask = cv::Mat::zeros(trace_params.state.size(), CV_8U);
                mask.copyTo(padded_mask(used_area));
                cv::bitwise_and(active_area_mask, padded_mask, hole_mask);
            } else {
                active_area_mask.copyTo(hole_mask);
            }

            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(hole_mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

            std::cout << "performing inpaint on " << contours.size() << " potential holes" << "\n";

            int inpaint_count = 0;
            int inpaint_skip = 0;

            // cv::Mat_<cv::Vec3b> vis(hole_mask.size());

    #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < contours.size(); i++) {
                if (hierarchy[i][3] != -1) { // It's a hole
                    cv::Rect roi = cv::boundingRect(contours[i]);

                    int margin = 4;
                    roi.x = std::max(0, roi.x - margin);
                    roi.y = std::max(0, roi.y - margin);
                    roi.width = std::min(hole_mask.cols - roi.x, roi.width + 2 * margin);
                    roi.height = std::min(hole_mask.rows - roi.y, roi.height + 2 * margin);

                    bool insufficient_border =
                        roi.width <= 4 || roi.height <= 4 ||
                        roi.x <= 1 || roi.y <= 1 ||
                        (roi.x + roi.width) > hole_mask.cols - 2 ||
                        (roi.y + roi.height) > hole_mask.rows - 2;
                    if (insufficient_border) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: insufficient margin around roi " << roi << "\n";
                        }
                        continue;
                    }

                    // std::cout << hole_mask.size() << trace_params.state.size() << resume_pad_x << "x" << resume_pad_y << "\n";

                    // cv::Point testp(2492+resume_pad_x, 508+resume_pad_y);
                    // cv::Point testp(2500+resume_pad_x, 566+resume_pad_y);
                    // cv::Point testp(2340+resume_pad_x, 577+resume_pad_y);

                    // cv::rectangle(vis, roi, cv::Scalar(255,255,255));

                    // if (!roi.contains(testp)) {
                    //     // std::cout << "skip " << roi << "\n";
                    //     continue;
                    // }

                    cv::Mat_<uchar> inpaint_mask(roi.size(), (uchar)1);

                    std::vector<cv::Point> hole_contour_roi;
                    for(const auto& p : contours[i]) {
                        hole_contour_roi.push_back({p.x - roi.x, p.y - roi.y});
                    }
                    std::vector<std::vector<cv::Point>> contours_to_fill = {hole_contour_roi};
                    cv::fillPoly(inpaint_mask, contours_to_fill, cv::Scalar(0));

                    // std::cout << "Inpainting hole at " << roi << " - " << inpaint_count << "+" << inpaint_skip << "/" << contours.size() << "\n";
                    bool did_inpaint = false;
                    try {
                        did_inpaint = inpaint(roi, inpaint_mask, trace_params, trace_data);
                    } catch (const cv::Exception& ex) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: OpenCV exception for roi " << roi << " => " << ex.what() << "\n";
                        }
                        continue;
                    } catch (const std::exception& ex) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: exception for roi " << roi << " => " << ex.what() << "\n";
                        }
                        continue;
                    } catch (...) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: unknown exception for roi " << roi << "\n";
                        }
                        continue;
                    }

                    if (!did_inpaint) {
    #pragma omp atomic
                        inpaint_skip++;
    #pragma omp critical
                        {
                            std::cout << "skip inpaint: mask border check failed for roi " << roi << "\n";
                        }
                        continue;
                    }

    #pragma omp critical
                    {
                        if (snapshot_interval > 0 && !tgt_path.empty() && inpaint_count % snapshot_interval == 0) {
                            QuadSurface* surf = create_surface_from_state();
                            surf->save(tgt_path, true);
                            delete surf;
                            std::cout << "saved snapshot in " << tgt_path << " (" << inpaint_count << "+" << inpaint_skip << "/" << contours.size() << ")" << "\n";
                        }
                    }

    #pragma omp atomic
                    inpaint_count++;
                }
                else
    #pragma omp atomic
                    inpaint_skip++;
            }

            // cv::imwrite("vis_inp_rect.tif", vis);
        }
    }

    // Prepare a new set of Ceres options used later during local solves
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-3;

    auto neighs = parse_growth_directions(params.growth_directions);

    int local_opt_r = 3;

    std::cout << "lets start fringe: " << fringe.size() << "\n";

    while (!fringe.empty()) {
        bool global_opt = generation <= 10 && !resume_surf;

        ALifeTime timer_gen;
        timer_gen.del_msg = "time per generation ";

        int phys_fail_count_gen = 0;
        generation++;
        if (stop_gen && generation >= stop_gen)
            break;

        // For every point in the fringe (where we might expand the patch outwards), add to cands all
        // new 2D points we might add to the patch (and later find the corresponding 3D point for)
        for(const auto& p : fringe)
        {
            for(const auto& n : neighs)
                if (bounds.contains(cv::Point(p+n))
                    && (trace_params.state(p+n) & STATE_PROCESSING) == 0
                    && (trace_params.state(p+n) & STATE_LOC_VALID) == 0) {
                    trace_params.state(p+n) |= STATE_PROCESSING;
                    cands.push_back(p+n);
                }
        }
        std::cout << "gen " << generation << " processing " << cands.size() << " fringe cands (total done " << succ << " fringe: " << fringe.size() << ")" << "\n";
        fringe.resize(0);

        std::cout << "cands " << cands.size() << "\n";

        int succ_gen = 0;
        std::vector<cv::Vec2i> succ_gen_ps;

        if (neural_tracer && generation > pre_neural_gens) {
            std::unordered_set<cv::Vec2i> cands_processed;  // subset of cands we've already passed to the neural tracer in this gen
            while (true) {

                float const min_pair_distance = 4.f;  // points closer than this in uv-space aren't processed together

                std::vector<cv::Vec2i> batch_cands;
                batch_cands.reserve(neural_batch_size);
                for (auto const& p : cands) {
                    if (trace_params.state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                        continue;
                    if (cands_processed.contains(p))
                        continue;
                    if (min_dist(p, batch_cands) < min_pair_distance)
                        continue;
                    // TODO: also skip if its neighbors are outside the z-range (c.f. ceres case checking avg)
                    batch_cands.push_back(p);
                    cands_processed.insert(p);
                    if (batch_cands.size() == neural_batch_size)
                        break;
                }

                auto const points_placed = call_neural_tracer_for_points(batch_cands, trace_params, neural_tracer.get());

                for (cv::Vec2i const &p : points_placed) {
                    generations(p) = generation;
                    if (!used_area.contains(cv::Point(p[1],p[0]))) {
                        used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    }
                    fringe.push_back(p);
                    succ_gen_ps.push_back(p);
                }
                succ += points_placed.size();
                succ_gen += points_placed.size();

                if (cands_processed.size() == cands.size())
                    break;

            }  // end loop over batches

        } else {

            // Snapshot positions before per-point optimization for flip detection
            cv::Mat_<cv::Vec3d> positions_before_perpoint = trace_params.dpoints.clone();

            // Configure anti-flipback constraint for per-point optimization
            // Note: new points won't have surface normals yet, but the loss function handles this
            AntiFlipbackConfig perpoint_flipback_config;
            perpoint_flipback_config.anchors = &positions_before_perpoint;
            perpoint_flipback_config.surface_normals = &surface_normals;
            perpoint_flipback_config.threshold = loss_settings.flipback_threshold;
            perpoint_flipback_config.weight = loss_settings.flipback_weight;

            // Build a structure that allows parallel iteration over cands, while avoiding any two threads simultaneously
            // considering two points that are too close to each other...
            OmpThreadPointCol cands_threadcol(local_opt_r*2+1, cands);

            // ...then start iterating over candidates in parallel using the above to yield points
            #pragma omp parallel
            {
                CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                //             int idx = rand() % cands.size();
                while (true) {
                    int r = 1;
                    double phys_fail_th = 0.1;
                    cv::Vec2i p = cands_threadcol.next();
                    if (p[0] == -1)
                        break;

                    if (trace_params.state(p) & (STATE_LOC_VALID | STATE_COORD_VALID))
                        continue;

                    // p is now a 2D point we consider adding to the patch; find the best 3D point to map it to

                    // Iterate all adjacent points that are in the patch, and find their 3D locations
                    int ref_count = 0;
                    cv::Vec3d avg = {0,0,0};
                    std::vector<cv::Vec2i> srcs;
                    for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,trace_params.dpoints.rows-1);oy++)
                        for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,trace_params.dpoints.cols-1);ox++)
                            if (trace_params.state(oy,ox) & STATE_LOC_VALID) {
                                ref_count++;
                                avg += trace_params.dpoints(oy,ox);
                                srcs.push_back({oy,ox});
                            }

                    // Of those adjacent points, find the one that itself has most adjacent in-patch points
                    cv::Vec2i best_l = srcs[0];
                    int best_ref_l = -1;
                    int rec_ref_sum = 0;
                    for(const cv::Vec2i& l : srcs) {
                        int ref_l = 0;
                        for(int oy=std::max(l[0]-r,0);oy<=std::min(l[0]+r,trace_params.dpoints.rows-1);oy++)
                            for(int ox=std::max(l[1]-r,0);ox<=std::min(l[1]+r,trace_params.dpoints.cols-1);ox++)
                                if (trace_params.state(oy,ox) & STATE_LOC_VALID)
                                    ref_l++;

                        rec_ref_sum += ref_l;

                        if (ref_l > best_ref_l) {
                            best_l = l;
                            best_ref_l = ref_l;
                        }
                    }

                    avg /= ref_count;

                    //"fast" skip based on avg xyz value out of limits
                    if (avg[2] < loss_settings.z_min || avg[2] > loss_settings.z_max ||
                        avg[1] < loss_settings.y_min || avg[1] > loss_settings.y_max ||
                        avg[0] < loss_settings.x_min || avg[0] > loss_settings.x_max)
                        continue;



                    cv::Vec3d init = trace_params.dpoints(best_l) + random_perturbation();
                    trace_params.dpoints(p) = init;

                    ceres::Problem problem;
                    trace_params.state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                    int flags = LOSS_DIST | LOSS_STRAIGHT;
                    if (trace_data.normal3d_field)
                        flags |= LOSS_3DNORMALLINE;

                    add_losses(problem, p, trace_params, trace_data, loss_settings, flags);

                    std::vector<double*> parameter_blocks;
                    problem.GetParameterBlocks(&parameter_blocks);
                    for (auto& block : parameter_blocks) {
                        problem.SetParameterBlockConstant(block);
                    }
                    problem.SetParameterBlockVariable(&trace_params.dpoints(p)[0]);

                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);

                    local_optimization(1, p, trace_params, trace_data, loss_settings, true, false, nullptr, &perpoint_flipback_config);
                    if (local_opt_r > 1)
                        local_optimization(local_opt_r, p, trace_params, trace_data, loss_settings, true, false, nullptr, &perpoint_flipback_config);

                    generations(p) = generation;

                    #pragma omp critical
                    {
                        succ++;
                        succ_gen++;
                        if (!used_area.contains(cv::Point(p[1],p[0]))) {
                            used_area = used_area | cv::Rect(p[1],p[0],1,1);
                        }

                        fringe.push_back(p);
                        succ_gen_ps.push_back(p);
                    }
                }  // end parallel iteration over cands
            }
        }

        // Update surface normals for all newly added points and their neighbors
        // This must be done after parallel section completes
        for (const auto& p : succ_gen_ps) {
            update_surface_normal(p, trace_params.dpoints, trace_params.state, surface_normals);
            // Also update neighbors since their geometry may have changed
            static const cv::Vec2i neighbor_offsets[] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
            for (const auto& off : neighbor_offsets) {
                cv::Vec2i neighbor = p + off;
                if (neighbor[0] >= 0 && neighbor[0] < trace_params.dpoints.rows &&
                    neighbor[1] >= 0 && neighbor[1] < trace_params.dpoints.cols &&
                    (trace_params.state(neighbor) & STATE_LOC_VALID)) {
                    update_surface_normal(neighbor, trace_params.dpoints, trace_params.state, surface_normals);
                }
            }
        }

        if (neural_tracer && generation > pre_neural_gens) {
            // Skip optimizations
        } else if (!global_opt) {
            // For late generations, instead of re-solving the global problem, solve many local-ish problems, around each
            // of the newly added points
            std::vector<cv::Vec2i> opt_local;
            for(const auto& p : succ_gen_ps)
                if (p[0] % 4 == 0 && p[1] % 4 == 0)
                    opt_local.push_back(p);

            int done = 0;

            if (!opt_local.empty()) {
                // Snapshot positions before optimization for flip detection
                cv::Mat_<cv::Vec3d> positions_before_opt = trace_params.dpoints.clone();

                // Configure anti-flipback constraint
                AntiFlipbackConfig flipback_config;
                flipback_config.anchors = &positions_before_opt;
                flipback_config.surface_normals = &surface_normals;
                flipback_config.threshold = loss_settings.flipback_threshold;
                flipback_config.weight = loss_settings.flipback_weight;

                OmpThreadPointCol opt_local_threadcol(17, opt_local);

#pragma omp parallel
                while (true)
                {
                    CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor);
                    cv::Vec2i p = opt_local_threadcol.next();
                    if (p[0] == -1)
                        break;

                    local_optimization(8, p, trace_params, trace_data, loss_settings, true, false, nullptr, &flipback_config);

#pragma omp atomic
                    done++;
                }
            }
        }
        else {
            //we do the global opt only every 8 gens, as every add does a small local solve anyweays
            if (generation % 8 == 0) {
                // Snapshot positions before global optimization
                cv::Mat_<cv::Vec3d> positions_before_opt = trace_params.dpoints.clone();

                // Configure anti-flipback constraint
                AntiFlipbackConfig flipback_config;
                flipback_config.anchors = &positions_before_opt;
                flipback_config.surface_normals = &surface_normals;
                flipback_config.threshold = loss_settings.flipback_threshold;
                flipback_config.weight = loss_settings.flipback_weight;

                local_optimization(stop_gen+10, {y0,x0}, trace_params, trace_data, loss_settings, false, true, nullptr, &flipback_config);
            }
        }

        cands.resize(0);

        // --- Speed Reporting ---
        double elapsed_seconds = f_timer.seconds();
        double seconds_this_gen = elapsed_seconds - last_elapsed_seconds;
        int succ_this_gen = succ - last_succ;

        double const vx_per_quad = static_cast<double>(step) * step;
        double const voxelsize_mm = static_cast<double>(voxelsize) / 1000.0;
        double const voxelsize_m = static_cast<double>(voxelsize) / 1000000.0;
        double const mm2_per_quad = vx_per_quad * voxelsize_mm * voxelsize_mm;
        double const m2_per_quad = vx_per_quad * voxelsize_m * voxelsize_m;

        double const total_area_mm2 = succ * mm2_per_quad;
        double const total_area_m2 = succ * m2_per_quad;

        double const total_area_mm2_run = (succ-succ_start) * mm2_per_quad;
        double const total_area_m2_run = (succ-succ_start) * m2_per_quad;

        double avg_speed_mm2_s = (elapsed_seconds > 0) ? (total_area_mm2_run / elapsed_seconds) : 0.0;
        double current_speed_mm2_s = (seconds_this_gen > 0) ? (succ_this_gen * mm2_per_quad / seconds_this_gen) : 0.0;
        double avg_speed_m2_day = (elapsed_seconds > 0) ? (total_area_m2_run / (elapsed_seconds / (24.0 * 3600.0))) : 0.0;

        printf("-> done %d | fringe %ld | area %.2f mm^2 (%.6f m^2) | avg speed %.2f mm^2/s (%.6f m^2/day) | current speed %.2f mm^2/s\n",
               succ, (long)fringe.size(), total_area_mm2, total_area_m2, avg_speed_mm2_s, avg_speed_m2_day, current_speed_mm2_s);

        last_elapsed_seconds = elapsed_seconds;
        last_succ = succ;

        timer_gen.unit = succ_gen * vx_per_quad;
        timer_gen.unit_string = "vx^2";
        // print_accessor_stats();

        if (!tgt_path.empty() && snapshot_interval > 0 && generation % snapshot_interval == 0) {
            QuadSurface* surf = create_surface_from_state();
            surf->save(tgt_path, true);
            delete surf;
            std::cout << "saved snapshot in " << tgt_path << "\n";
        }

    }  // end while fringe is non-empty
    delete timer;

    QuadSurface* surf = create_surface_from_state();

    const double area_est_vx2 = vc::surface::computeSurfaceAreaVox2(*surf);
    const double voxel_size_d = static_cast<double>(voxelsize);
    const double area_est_cm2 = area_est_vx2 * voxel_size_d * voxel_size_d / 1e8;
    printf("generated surface %f vx^2 (%f cm^2)\n", area_est_vx2, area_est_cm2);

    return surf;
}
