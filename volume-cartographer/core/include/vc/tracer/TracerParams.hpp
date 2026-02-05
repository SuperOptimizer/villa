#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

struct TracerParams {
    // === GrowPatch fields (used by tracer()) ===
    std::string neural_socket;           // empty = disabled
    int pre_neural_generations = 0;
    int neural_batch_size = 1;
    std::string normal3d_zarr_path;      // empty = disabled
    std::string normal_grid_path;        // empty = disabled

    // Reference surface (optional block)
    struct ReferenceSurface {
        std::string path;
        double voxel_threshold = 1.0;
        double penalty_weight = 0.5;
        double sample_step = 1.0;
        double max_distance = 250.0;
        double min_clearance = 4.0;
        double clearance_weight = 1.0;
    };
    std::optional<ReferenceSurface> reference_surface;

    int snapshot_interval = 0;
    int generations = 100;
    float step_size = 0.0f;             // 0 = auto
    int rewind_gen = -1;

    // Bounds
    int z_min = -1;
    int z_max = std::numeric_limits<int>::max();
    int y_min = -1;
    int y_max = std::numeric_limits<int>::max();
    int x_min = -1;
    int x_max = std::numeric_limits<int>::max();

    float flipback_threshold = 5.0f;
    float flipback_weight = 1.0f;
    bool vis_losses = false;
    bool inpaint = false;

    // Resume optimization
    std::string resume_opt = "skip";
    int resume_local_opt_step = 16;
    int resume_local_opt_radius = -1;
    int resume_local_max_iters = 1000;
    bool resume_local_dense_qr = false;

    // GUI growth control
    std::string grow_mode = "all";
    int grow_steps = 0;
    int grow_extra_rows = 0;
    int grow_extra_cols = 0;
    std::vector<std::string> growth_directions;

    // Loss weights (for applyWeights)
    float snap_weight = -1.0f;           // -1 = use default
    float normal_weight = -1.0f;
    float normal3dline_weight = -1.0f;
    float straight_weight_loss = -1.0f;  // named differently to avoid clash with GrowSurface field
    float dist_weight = -1.0f;
    float direction_weight = -1.0f;
    float sdir_weight = -1.0f;
    float correction_weight = -1.0f;
    float reference_ray_weight = -1.0f;

    // === GrowSurface fields (used by grow_surf_from_surfs()) ===
    bool flip_x = false;
    int global_steps_per_window = 0;
    float src_step = 20.0f;
    float step = 10.0f;
    int max_width = 80000;
    float local_cost_inl_th = 0.2f;
    float same_surface_th = 2.0f;
    float duplicate_surface_th = -1.0f;
    float remap_attach_surface_th = -1.0f;
    int point_to_max_iters = 10;
    int point_to_seed_max_iters = 1000;
    bool use_surface_patch_index = false;
    int surface_patch_stride = 1;
    float surface_patch_bbox_pad = 0.0f;

    float straight_weight = 0.7f;
    float straight_weight_3D = 4.0f;
    float sliding_w_scale = 1.0f;
    float z_loc_loss_w = 0.1f;
    float dist_loss_2d_w = 1.0f;
    float dist_loss_3d_w = 2.0f;
    float straight_min_count = 1.0f;
    int inlier_base_threshold = 20;

    uint64_t deterministic_seed = 5489;
    double deterministic_jitter_px = 0.15;

    bool pin_approved_points = true;
    bool keep_approved_on_consistency = true;
    bool prefer_approved_in_hr = true;
    float approved_weight_hr = 4.0f;
    int approved_priority_radius = 2;
    bool consider_all_approved_as_candidates = true;
    int approved_min_straight_in_grow = 0;
    int approved_min_count_in_grow = 0;

    int consensus_default_th = 10;
    int consensus_limit_th = 2;
    bool hr_gen_parallel = false;
    bool remap_parallel = false;
    int hr_attach_lr_radius = 1;
    float hr_attach_relax_factor = 2.0f;

    std::optional<std::array<double, 2>> z_range;
    std::filesystem::path tgt_dir;

    bool use_cuda = true;
};
