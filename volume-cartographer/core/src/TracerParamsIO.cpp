#include "vc/tracer/TracerParamsIO.hpp"
#include "vc/tracer/TracerParams.hpp"

#include <nlohmann/json.hpp>

namespace vc::tracer {

TracerParams parseFromJson(const nlohmann::json& j)
{
    TracerParams p;

    // GrowPatch fields
    if (j.contains("neural_socket"))
        p.neural_socket = j["neural_socket"].get<std::string>();
    p.pre_neural_generations = j.value("pre_neural_generations", p.pre_neural_generations);
    p.neural_batch_size = j.value("neural_batch_size", p.neural_batch_size);
    if (j.contains("normal3d_zarr_path") && j["normal3d_zarr_path"].is_string())
        p.normal3d_zarr_path = j["normal3d_zarr_path"].get<std::string>();
    if (j.contains("normal_grid_path") && j["normal_grid_path"].is_string())
        p.normal_grid_path = j["normal_grid_path"].get<std::string>();

    // Reference surface
    if (j.contains("reference_surface")) {
        const auto& ref = j["reference_surface"];
        TracerParams::ReferenceSurface rs;
        if (ref.is_string()) {
            rs.path = ref.get<std::string>();
        } else if (ref.is_object()) {
            if (auto it = ref.find("path"); it != ref.end() && it->is_string())
                rs.path = it->get<std::string>();
            rs.voxel_threshold = ref.value("voxel_threshold", rs.voxel_threshold);
            rs.penalty_weight = ref.value("penalty_weight", rs.penalty_weight);
            rs.sample_step = ref.value("sample_step", rs.sample_step);
            rs.max_distance = ref.value("max_distance", rs.max_distance);
            rs.min_clearance = ref.value("min_clearance", rs.min_clearance);
            rs.clearance_weight = ref.value("clearance_weight", rs.clearance_weight);
        }
        p.reference_surface = rs;
    }

    p.snapshot_interval = j.value("snapshot-interval", p.snapshot_interval);
    p.generations = j.value("generations", p.generations);
    if (j.contains("step_size"))
        p.step_size = j.value("step_size", 20.0f);
    p.rewind_gen = j.value("rewind_gen", p.rewind_gen);

    // Bounds
    p.z_min = j.value("z_min", p.z_min);
    p.z_max = j.value("z_max", p.z_max);
    p.y_min = j.value("y_min", p.y_min);
    p.y_max = j.value("y_max", p.y_max);
    p.x_min = j.value("x_min", p.x_min);
    p.x_max = j.value("x_max", p.x_max);

    p.flipback_threshold = j.value("flipback_threshold", p.flipback_threshold);
    p.flipback_weight = j.value("flipback_weight", p.flipback_weight);
    p.vis_losses = j.value("vis_losses", p.vis_losses);
    p.inpaint = j.value("inpaint", p.inpaint);

    // Resume optimization
    p.resume_opt = j.value("resume_opt", p.resume_opt);
    p.resume_local_opt_step = j.value("resume_local_opt_step", p.resume_local_opt_step);
    p.resume_local_opt_radius = j.value("resume_local_opt_radius", p.resume_local_opt_radius);
    p.resume_local_max_iters = j.value("resume_local_max_iters", p.resume_local_max_iters);
    p.resume_local_dense_qr = j.value("resume_local_dense_qr", p.resume_local_dense_qr);

    // GUI growth control
    p.grow_mode = j.value("grow_mode", p.grow_mode);
    p.grow_steps = j.value("grow_steps", p.grow_steps);
    p.grow_extra_rows = j.value("grow_extra_rows", p.grow_extra_rows);
    p.grow_extra_cols = j.value("grow_extra_cols", p.grow_extra_cols);
    if (j.contains("growth_directions") && j["growth_directions"].is_array()) {
        for (const auto& entry : j["growth_directions"]) {
            if (entry.is_string())
                p.growth_directions.push_back(entry.get<std::string>());
        }
    }

    // Loss weights
    if (j.contains("snap_weight")) p.snap_weight = j["snap_weight"].get<float>();
    if (j.contains("normal_weight")) p.normal_weight = j["normal_weight"].get<float>();
    if (j.contains("normal3dline_weight")) p.normal3dline_weight = j["normal3dline_weight"].get<float>();
    if (j.contains("straight_weight")) {
        // Note: "straight_weight" JSON key maps to both the loss weight (for GrowPatch)
        // and the GrowSurface parameter. We store it in both places.
        float w = j["straight_weight"].get<float>();
        p.straight_weight_loss = w;
        p.straight_weight = w;
    }
    if (j.contains("dist_weight")) p.dist_weight = j["dist_weight"].get<float>();
    if (j.contains("direction_weight")) p.direction_weight = j["direction_weight"].get<float>();
    if (j.contains("sdir_weight")) p.sdir_weight = j["sdir_weight"].get<float>();
    if (j.contains("correction_weight")) p.correction_weight = j["correction_weight"].get<float>();
    if (j.contains("reference_ray_weight")) p.reference_ray_weight = j["reference_ray_weight"].get<float>();

    // GrowSurface fields
    p.flip_x = j.value("flip_x", static_cast<int>(p.flip_x)) != 0;
    p.global_steps_per_window = j.value("global_steps_per_window", p.global_steps_per_window);
    p.src_step = j.value("src_step", p.src_step);
    p.step = j.value("step", p.step);
    p.max_width = j.value("max_width", p.max_width);
    p.local_cost_inl_th = j.value("local_cost_inl_th", p.local_cost_inl_th);
    p.same_surface_th = j.value("same_surface_th", p.same_surface_th);
    p.duplicate_surface_th = j.value("duplicate_surface_th", p.duplicate_surface_th);
    p.remap_attach_surface_th = j.value("remap_attach_surface_th", p.remap_attach_surface_th);
    p.point_to_max_iters = j.value("point_to_max_iters", p.point_to_max_iters);
    p.point_to_seed_max_iters = j.value("point_to_seed_max_iters", p.point_to_seed_max_iters);
    p.use_surface_patch_index = j.value("use_surface_patch_index", p.use_surface_patch_index);
    p.surface_patch_stride = j.value("surface_patch_stride", p.surface_patch_stride);
    p.surface_patch_bbox_pad = j.value("surface_patch_bbox_pad", p.surface_patch_bbox_pad);

    // Note: straight_weight already handled above in loss weights section
    p.straight_weight_3D = j.value("straight_weight_3D", p.straight_weight_3D);
    p.sliding_w_scale = j.value("sliding_w_scale", p.sliding_w_scale);
    p.z_loc_loss_w = j.value("z_loc_loss_w", p.z_loc_loss_w);
    p.dist_loss_2d_w = j.value("dist_loss_2d_w", p.dist_loss_2d_w);
    p.dist_loss_3d_w = j.value("dist_loss_3d_w", p.dist_loss_3d_w);
    p.straight_min_count = j.value("straight_min_count", p.straight_min_count);
    p.inlier_base_threshold = j.value("inlier_base_threshold", p.inlier_base_threshold);

    p.deterministic_seed = static_cast<uint64_t>(j.value("deterministic_seed", static_cast<int>(p.deterministic_seed)));
    p.deterministic_jitter_px = j.value("deterministic_jitter_px", p.deterministic_jitter_px);

    p.pin_approved_points = j.value("pin_approved_points", p.pin_approved_points);
    p.keep_approved_on_consistency = j.value("keep_approved_on_consistency", p.keep_approved_on_consistency);
    p.prefer_approved_in_hr = j.value("prefer_approved_in_hr", p.prefer_approved_in_hr);
    p.approved_weight_hr = j.value("approved_weight_hr", p.approved_weight_hr);
    p.approved_priority_radius = j.value("approved_priority_radius", p.approved_priority_radius);
    p.consider_all_approved_as_candidates = j.value("consider_all_approved_as_candidates", p.consider_all_approved_as_candidates);
    p.approved_min_straight_in_grow = j.value("approved_min_straight_in_grow", p.approved_min_straight_in_grow);
    p.approved_min_count_in_grow = j.value("approved_min_count_in_grow", p.approved_min_count_in_grow);

    p.consensus_default_th = j.value("consensus_default_th", p.consensus_default_th);
    p.consensus_limit_th = j.value("consensus_limit_th", p.consensus_limit_th);
    p.hr_gen_parallel = j.value("hr_gen_parallel", p.hr_gen_parallel);
    p.remap_parallel = j.value("remap_parallel", p.remap_parallel);
    p.hr_attach_lr_radius = j.value("hr_attach_lr_radius", p.hr_attach_lr_radius);
    p.hr_attach_relax_factor = j.value("hr_attach_relax_factor", p.hr_attach_relax_factor);

    // z_range (array of 2 doubles)
    if (j.contains("z_range") && j["z_range"].is_array() && j["z_range"].size() == 2) {
        p.z_range = std::array<double, 2>{
            j["z_range"][0].get<double>(),
            j["z_range"][1].get<double>()
        };
    }

    if (j.contains("tgt_dir"))
        p.tgt_dir = j["tgt_dir"].get<std::string>();

    p.use_cuda = j.value("use_cuda", p.use_cuda);

    return p;
}

nlohmann::json toJson(const TracerParams& p)
{
    nlohmann::json j;

    if (!p.neural_socket.empty()) j["neural_socket"] = p.neural_socket;
    j["pre_neural_generations"] = p.pre_neural_generations;
    j["neural_batch_size"] = p.neural_batch_size;
    if (!p.normal3d_zarr_path.empty()) j["normal3d_zarr_path"] = p.normal3d_zarr_path;
    if (!p.normal_grid_path.empty()) j["normal_grid_path"] = p.normal_grid_path;

    if (p.reference_surface) {
        nlohmann::json ref;
        ref["path"] = p.reference_surface->path;
        ref["voxel_threshold"] = p.reference_surface->voxel_threshold;
        ref["penalty_weight"] = p.reference_surface->penalty_weight;
        ref["sample_step"] = p.reference_surface->sample_step;
        ref["max_distance"] = p.reference_surface->max_distance;
        ref["min_clearance"] = p.reference_surface->min_clearance;
        ref["clearance_weight"] = p.reference_surface->clearance_weight;
        j["reference_surface"] = ref;
    }

    j["snapshot-interval"] = p.snapshot_interval;
    j["generations"] = p.generations;
    if (p.step_size > 0.0f) j["step_size"] = p.step_size;
    j["rewind_gen"] = p.rewind_gen;

    j["z_min"] = p.z_min;
    j["z_max"] = p.z_max;
    j["y_min"] = p.y_min;
    j["y_max"] = p.y_max;
    j["x_min"] = p.x_min;
    j["x_max"] = p.x_max;

    j["flipback_threshold"] = p.flipback_threshold;
    j["flipback_weight"] = p.flipback_weight;
    j["vis_losses"] = p.vis_losses;
    j["inpaint"] = p.inpaint;

    j["resume_opt"] = p.resume_opt;
    j["resume_local_opt_step"] = p.resume_local_opt_step;
    j["resume_local_opt_radius"] = p.resume_local_opt_radius;
    j["resume_local_max_iters"] = p.resume_local_max_iters;
    j["resume_local_dense_qr"] = p.resume_local_dense_qr;

    j["grow_mode"] = p.grow_mode;
    j["grow_steps"] = p.grow_steps;
    j["grow_extra_rows"] = p.grow_extra_rows;
    j["grow_extra_cols"] = p.grow_extra_cols;
    if (!p.growth_directions.empty()) j["growth_directions"] = p.growth_directions;

    if (p.snap_weight >= 0) j["snap_weight"] = p.snap_weight;
    if (p.normal_weight >= 0) j["normal_weight"] = p.normal_weight;
    if (p.normal3dline_weight >= 0) j["normal3dline_weight"] = p.normal3dline_weight;
    if (p.straight_weight_loss >= 0) j["straight_weight"] = p.straight_weight_loss;
    if (p.dist_weight >= 0) j["dist_weight"] = p.dist_weight;
    if (p.direction_weight >= 0) j["direction_weight"] = p.direction_weight;
    if (p.sdir_weight >= 0) j["sdir_weight"] = p.sdir_weight;
    if (p.correction_weight >= 0) j["correction_weight"] = p.correction_weight;
    if (p.reference_ray_weight >= 0) j["reference_ray_weight"] = p.reference_ray_weight;

    j["flip_x"] = p.flip_x;
    j["global_steps_per_window"] = p.global_steps_per_window;
    j["src_step"] = p.src_step;
    j["step"] = p.step;
    j["max_width"] = p.max_width;
    j["local_cost_inl_th"] = p.local_cost_inl_th;
    j["same_surface_th"] = p.same_surface_th;
    j["duplicate_surface_th"] = p.duplicate_surface_th;
    j["remap_attach_surface_th"] = p.remap_attach_surface_th;
    j["point_to_max_iters"] = p.point_to_max_iters;
    j["point_to_seed_max_iters"] = p.point_to_seed_max_iters;
    j["use_surface_patch_index"] = p.use_surface_patch_index;
    j["surface_patch_stride"] = p.surface_patch_stride;
    j["surface_patch_bbox_pad"] = p.surface_patch_bbox_pad;

    // Note: straight_weight already covered in loss weights section above
    j["straight_weight_3D"] = p.straight_weight_3D;
    j["sliding_w_scale"] = p.sliding_w_scale;
    j["z_loc_loss_w"] = p.z_loc_loss_w;
    j["dist_loss_2d_w"] = p.dist_loss_2d_w;
    j["dist_loss_3d_w"] = p.dist_loss_3d_w;
    j["straight_min_count"] = p.straight_min_count;
    j["inlier_base_threshold"] = p.inlier_base_threshold;

    j["deterministic_seed"] = p.deterministic_seed;
    j["deterministic_jitter_px"] = p.deterministic_jitter_px;

    j["pin_approved_points"] = p.pin_approved_points;
    j["keep_approved_on_consistency"] = p.keep_approved_on_consistency;
    j["prefer_approved_in_hr"] = p.prefer_approved_in_hr;
    j["approved_weight_hr"] = p.approved_weight_hr;
    j["approved_priority_radius"] = p.approved_priority_radius;
    j["consider_all_approved_as_candidates"] = p.consider_all_approved_as_candidates;
    j["approved_min_straight_in_grow"] = p.approved_min_straight_in_grow;
    j["approved_min_count_in_grow"] = p.approved_min_count_in_grow;

    j["consensus_default_th"] = p.consensus_default_th;
    j["consensus_limit_th"] = p.consensus_limit_th;
    j["hr_gen_parallel"] = p.hr_gen_parallel;
    j["remap_parallel"] = p.remap_parallel;
    j["hr_attach_lr_radius"] = p.hr_attach_lr_radius;
    j["hr_attach_relax_factor"] = p.hr_attach_relax_factor;

    if (p.z_range) j["z_range"] = {(*p.z_range)[0], (*p.z_range)[1]};
    if (!p.tgt_dir.empty()) j["tgt_dir"] = p.tgt_dir.string();

    j["use_cuda"] = p.use_cuda;

    return j;
}

void applyJsonOverlay(TracerParams& base, const nlohmann::json& overlay)
{
    // Parse the overlay as a fresh TracerParams, then selectively overwrite
    // only the keys that actually appear in the overlay JSON.
    for (auto it = overlay.begin(); it != overlay.end(); ++it) {
        const auto& key = it.key();
        const auto& val = it.value();

        // Mechanical: for each known key, overwrite base field
        if (key == "neural_socket") base.neural_socket = val.get<std::string>();
        else if (key == "pre_neural_generations") base.pre_neural_generations = val.get<int>();
        else if (key == "neural_batch_size") base.neural_batch_size = val.get<int>();
        else if (key == "normal3d_zarr_path") base.normal3d_zarr_path = val.get<std::string>();
        else if (key == "normal_grid_path") base.normal_grid_path = val.get<std::string>();
        else if (key == "reference_surface") {
            TracerParams::ReferenceSurface rs;
            if (val.is_string()) {
                rs.path = val.get<std::string>();
            } else if (val.is_object()) {
                if (auto pit = val.find("path"); pit != val.end() && pit->is_string())
                    rs.path = pit->get<std::string>();
                rs.voxel_threshold = val.value("voxel_threshold", rs.voxel_threshold);
                rs.penalty_weight = val.value("penalty_weight", rs.penalty_weight);
                rs.sample_step = val.value("sample_step", rs.sample_step);
                rs.max_distance = val.value("max_distance", rs.max_distance);
                rs.min_clearance = val.value("min_clearance", rs.min_clearance);
                rs.clearance_weight = val.value("clearance_weight", rs.clearance_weight);
            }
            base.reference_surface = rs;
        }
        else if (key == "snapshot-interval") base.snapshot_interval = val.get<int>();
        else if (key == "generations") base.generations = val.get<int>();
        else if (key == "step_size") base.step_size = val.get<float>();
        else if (key == "rewind_gen") base.rewind_gen = val.get<int>();
        else if (key == "z_min") base.z_min = val.get<int>();
        else if (key == "z_max") base.z_max = val.get<int>();
        else if (key == "y_min") base.y_min = val.get<int>();
        else if (key == "y_max") base.y_max = val.get<int>();
        else if (key == "x_min") base.x_min = val.get<int>();
        else if (key == "x_max") base.x_max = val.get<int>();
        else if (key == "flipback_threshold") base.flipback_threshold = val.get<float>();
        else if (key == "flipback_weight") base.flipback_weight = val.get<float>();
        else if (key == "vis_losses") base.vis_losses = val.get<bool>();
        else if (key == "inpaint") base.inpaint = val.get<bool>();
        else if (key == "resume_opt") base.resume_opt = val.get<std::string>();
        else if (key == "resume_local_opt_step") base.resume_local_opt_step = val.get<int>();
        else if (key == "resume_local_opt_radius") base.resume_local_opt_radius = val.get<int>();
        else if (key == "resume_local_max_iters") base.resume_local_max_iters = val.get<int>();
        else if (key == "resume_local_dense_qr") base.resume_local_dense_qr = val.get<bool>();
        else if (key == "grow_mode") base.grow_mode = val.get<std::string>();
        else if (key == "grow_steps") base.grow_steps = val.get<int>();
        else if (key == "grow_extra_rows") base.grow_extra_rows = val.get<int>();
        else if (key == "grow_extra_cols") base.grow_extra_cols = val.get<int>();
        else if (key == "growth_directions") {
            base.growth_directions.clear();
            if (val.is_array()) {
                for (const auto& e : val)
                    if (e.is_string()) base.growth_directions.push_back(e.get<std::string>());
            }
        }
        else if (key == "snap_weight") base.snap_weight = val.get<float>();
        else if (key == "normal_weight") base.normal_weight = val.get<float>();
        else if (key == "normal3dline_weight") base.normal3dline_weight = val.get<float>();
        else if (key == "straight_weight") { base.straight_weight_loss = val.get<float>(); base.straight_weight = val.get<float>(); }
        else if (key == "dist_weight") base.dist_weight = val.get<float>();
        else if (key == "direction_weight") base.direction_weight = val.get<float>();
        else if (key == "sdir_weight") base.sdir_weight = val.get<float>();
        else if (key == "correction_weight") base.correction_weight = val.get<float>();
        else if (key == "reference_ray_weight") base.reference_ray_weight = val.get<float>();
        else if (key == "flip_x") base.flip_x = val.get<int>() != 0;
        else if (key == "global_steps_per_window") base.global_steps_per_window = val.get<int>();
        else if (key == "src_step") base.src_step = val.get<float>();
        else if (key == "step") base.step = val.get<float>();
        else if (key == "max_width") base.max_width = val.get<int>();
        else if (key == "local_cost_inl_th") base.local_cost_inl_th = val.get<float>();
        else if (key == "same_surface_th") base.same_surface_th = val.get<float>();
        else if (key == "duplicate_surface_th") base.duplicate_surface_th = val.get<float>();
        else if (key == "remap_attach_surface_th") base.remap_attach_surface_th = val.get<float>();
        else if (key == "point_to_max_iters") base.point_to_max_iters = val.get<int>();
        else if (key == "point_to_seed_max_iters") base.point_to_seed_max_iters = val.get<int>();
        else if (key == "use_surface_patch_index") base.use_surface_patch_index = val.get<bool>();
        else if (key == "surface_patch_stride") base.surface_patch_stride = val.get<int>();
        else if (key == "surface_patch_bbox_pad") base.surface_patch_bbox_pad = val.get<float>();
        else if (key == "straight_weight_3D") base.straight_weight_3D = val.get<float>();
        else if (key == "sliding_w_scale") base.sliding_w_scale = val.get<float>();
        else if (key == "z_loc_loss_w") base.z_loc_loss_w = val.get<float>();
        else if (key == "dist_loss_2d_w") base.dist_loss_2d_w = val.get<float>();
        else if (key == "dist_loss_3d_w") base.dist_loss_3d_w = val.get<float>();
        else if (key == "straight_min_count") base.straight_min_count = val.get<float>();
        else if (key == "inlier_base_threshold") base.inlier_base_threshold = val.get<int>();
        else if (key == "deterministic_seed") base.deterministic_seed = static_cast<uint64_t>(val.get<int>());
        else if (key == "deterministic_jitter_px") base.deterministic_jitter_px = val.get<double>();
        else if (key == "pin_approved_points") base.pin_approved_points = val.get<bool>();
        else if (key == "keep_approved_on_consistency") base.keep_approved_on_consistency = val.get<bool>();
        else if (key == "prefer_approved_in_hr") base.prefer_approved_in_hr = val.get<bool>();
        else if (key == "approved_weight_hr") base.approved_weight_hr = val.get<float>();
        else if (key == "approved_priority_radius") base.approved_priority_radius = val.get<int>();
        else if (key == "consider_all_approved_as_candidates") base.consider_all_approved_as_candidates = val.get<bool>();
        else if (key == "approved_min_straight_in_grow") base.approved_min_straight_in_grow = val.get<int>();
        else if (key == "approved_min_count_in_grow") base.approved_min_count_in_grow = val.get<int>();
        else if (key == "consensus_default_th") base.consensus_default_th = val.get<int>();
        else if (key == "consensus_limit_th") base.consensus_limit_th = val.get<int>();
        else if (key == "hr_gen_parallel") base.hr_gen_parallel = val.get<bool>();
        else if (key == "remap_parallel") base.remap_parallel = val.get<bool>();
        else if (key == "hr_attach_lr_radius") base.hr_attach_lr_radius = val.get<int>();
        else if (key == "hr_attach_relax_factor") base.hr_attach_relax_factor = val.get<float>();
        else if (key == "z_range") {
            if (val.is_array() && val.size() == 2) {
                base.z_range = std::array<double, 2>{val[0].get<double>(), val[1].get<double>()};
            }
        }
        else if (key == "tgt_dir") base.tgt_dir = val.get<std::string>();
        else if (key == "use_cuda") base.use_cuda = val.get<bool>();
        // Unknown keys are silently ignored
    }
}

}  // namespace vc::tracer
