# Import os early
import os
import json
import socket
import numpy as np
import zarr
from numcodecs import Blosc
import torch
import torch.nn
import threading
from typing import Dict, Tuple, List, Optional, Any, Union
import torch.distributed as dist
from torch.utils.data import DataLoader
import queue
import time

def find_free_port():
    """Find a free port on localhost that can be used for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]
        
def custom_collate(batch):
    """
    Custom collate function for DataLoader that handles variable-sized inputs.
    Combines a list of samples from dataset into a mini-batch tensor.
    
    Args:
        batch: List of samples from dataset, where each sample is a dict
        
    Returns:
        Dict with batched tensors
    """
    if len(batch) == 0:
        return {}
    
    # Get all keys from the first batch item
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key == 'data':
            # Batch the input data tensors (data is already padded to same shape in dataset)
            result[key] = torch.stack([item[key] for item in batch])
        elif key == 'pos':
            # Keep positions as a list of tuples or tensors
            result[key] = [item[key] for item in batch]
        else:
            # For other keys, default to list collection
            result[key] = [item[key] for item in batch]
            
    return result

from data.vc_dataset import VCDataset
from utils.models.blending import (
    create_gaussian_weights_torch,
    blend_patch_weighted,
    intersects_chunk
)
from data.io.zarrio.zarr_temp_storage import ZarrTempStorage
from data.io.zarrio.zarr_writer_worker import zarr_writer_worker
from utils.models.load_nnunet_model import load_model_for_inference
from utils.models.tta import get_tta_augmented_inputs
from utils.models.helpers import merge_tensors


class ZarrInferer:
    def __init__(self,
                 input_path: str, # Path or ID used by VCDataset
                 output_path: str,
                 model_info: Dict[str, Any],
                 dataset: VCDataset, # Expects an initialized VCDataset
                 dataloader: DataLoader,
                 # patch_size is now primarily derived from model_info inside worker
                 batch_size: int = 4,
                 step_size: float = 0.5,
                 num_write_workers: int = 4,
                 threshold: Optional[float] = None,
                 use_mirroring: bool = False, # Controlled by TTA flags below
                 max_tta_combinations: Optional[int] = 3,
                 use_rotation_tta: bool = True,
                 rotation_weights: List[float] = None,
                 verbose: bool = False,
                 save_probability_maps: bool = True,
                 output_targets: Optional[List[Dict[str, Any]]] = None, # Derived from model_info if None
                 rank: int = 0,
                 edge_weight_boost: float = 0): # Kept for blend_config, but not used in current blend logic
        """
        Performs nnUNet inference using VCDataset and Volume class.

        Writes patch predictions to temporary Zarr storage and blends them sequentially.

        Args:
            input_path: Identifier used by VCDataset (path, scroll ID, etc.).
            output_path: Path to save the final output zarr store (must end with .zarr).
            model_info: Dictionary from load_model_for_inference.
            dataset: Initialized VCDataset instance.
            dataloader: Initialized DataLoader instance wrapping the dataset.
            batch_size: Batch size for model inference.
            step_size: Step size for sliding window (fraction of patch size).
            num_write_workers: Number of threads for writing patches to temp storage.
            threshold: Optional threshold (0-100) for binarizing probability maps.
            use_mirroring: Enable test time augmentation (mirroring and/or rotation).
            max_tta_combinations: Max TTA combinations (0 disables TTA).
            use_rotation_tta: Prioritize rotation TTA if use_mirroring is True.
            rotation_weights: Optional weights for rotation TTA axes.
            verbose: Enable detailed logging.
            save_probability_maps: Save full probabilities (True) or class labels (False).
            output_targets: Optional override for output target configuration.
            rank: Process rank for distributed processing.
            edge_weight_boost: Factor for edge weight boosting (legacy, kept for config).
        """
        self.input_path = input_path # Store for reference, but dataset handles access

        if not output_path.endswith('.zarr'):
            raise ValueError(f"Output path must end with '.zarr', got: {output_path}")
        self.output_path = output_path

        self.model_info = model_info
        self.dataset = dataset
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.tile_step_size = step_size
        self.verbose = verbose
        self.rank = rank

        # Get patch size from model_info (MUST be available)
        if 'patch_size' not in self.model_info:
             raise ValueError("model_info dictionary must contain 'patch_size'")
        self.patch_size = tuple(self.model_info['patch_size'])
        
        # For debugging, print key model info that affects dimensions
        if self.verbose and self.rank == 0:
            print(f"Model info - patch_size: {self.patch_size}")
            print(f"Model info - num_input_channels: {self.model_info.get('num_input_channels', 'N/A')}")
            print(f"Model info - num_seg_heads: {self.model_info.get('num_seg_heads', 'N/A')}")
        # Infer device from model parameters
        self.device_str = 'cpu' # Default
        if 'network' in self.model_info and hasattr(self.model_info['network'], 'parameters'):
             try:
                  self.device_str = str(next(self.model_info['network'].parameters()).device)
             except StopIteration:
                  print("Warning: Could not infer device from model parameters, defaulting to 'cpu'.")
        self.threshold = threshold
        # TTA configuration
        self.use_mirroring = use_mirroring and (max_tta_combinations != 0) # Explicitly disable if max_tta=0
        self.max_tta_combinations = max_tta_combinations
        self.use_rotation_tta = use_rotation_tta and self.use_mirroring # Rotation depends on mirroring being active
        self.rotation_weights = rotation_weights

        self.save_probability_maps = save_probability_maps
        self.edge_weight_boost = edge_weight_boost
        self.total_time = 0.0

        if max_tta_combinations == 0 and self.verbose:
             print("TTA is disabled (max_tta_combinations=0).")
        elif self.verbose:
             print(f"TTA Config: use_mirroring={self.use_mirroring}, use_rotation_tta={self.use_rotation_tta}, max_combs={self.max_tta_combinations}")

        # Derive output targets from model_info if not provided
        num_classes = self.model_info.get('num_seg_heads', 2)
        if output_targets is None:
            self.targets = [{
                "name": "segmentation",
                "channels": num_classes,
                "activation": "sigmoid" if num_classes <= 2 else "softmax",
                "nnunet_output_channels": num_classes
            }]
        else:
             self.targets = output_targets
             # Validate provided targets slightly
             if not isinstance(self.targets, list) or not all(isinstance(t, dict) for t in self.targets):
                 raise ValueError("output_targets must be a list of dictionaries.")

        # Output array name (root level in zarr)
        self.output_array_name = "" # Consistent with previous logic

        # Foreground channel (often index 1 for binary)
        self.nnunet_foreground_channel = 1

        if self.verbose:
            print(f"ZarrInferer Initialized: patch_size={self.patch_size}, step_size={self.tile_step_size}")
            for target in self.targets:
                print(f"  Output target '{target.get('name')}': channels={target.get('channels')}, activation={target.get('activation')}")

        # Writer queue setup
        max_queue = 300 # Consider adjusting based on memory/speed trade-off
        self.writer_queue = queue.Queue(maxsize=max_queue)
        self.num_write_workers = num_write_workers

        # Temp storage - initialized in infer()
        self.temp_storage = None
        self.temp_dir = None # Also initialized in infer()
        self.patch_arrays_by_rank = {} # To store Zarr array references for deferred loading

    # _precompute_chunk_weight_map remains the same

    def _blend_patches(self, patch_arrays, output_arrays, count_arrays):
        """
        Blends patches using Gaussian weighting and deferred loading from temp storage.

        Args:
            patch_arrays: Dict mapping target_name to info (not used directly here).
            output_arrays: Dict mapping target_name to output Zarr arrays (sum arrays).
            count_arrays: Dict mapping target_name to count Zarr arrays.
        """
        if self.verbose:
            print("Starting patch blending phase...")

        patch_size_tuple = tuple(self.patch_size)
        device = self.device_str

        # Gaussian weights setup (standard nnU-Net parameters)
        sigma_scale = 1 / 8
        value_scaling_factor = 10
        blend_weights = create_gaussian_weights_torch(
            patch_size_tuple,
            sigma_scale=sigma_scale,
            value_scaling_factor=value_scaling_factor,
            device=device,
            edge_weight_boost=0 # Use 0 for standard blending logic here
        )
        if self.verbose:
             print(f"Created Gaussian blend weights on device {device} with sigma_scale={sigma_scale}")

        # Process each target
        for target in self.targets:
            tgt_name = target.get("name")
            if tgt_name not in output_arrays or tgt_name not in count_arrays: # Check sum/count arrays
                print(f"Warning: Missing sum or count array for target '{tgt_name}', skipping blending.")
                continue

            # --- Collect Patch Metadata ---
            if self.verbose: print(f"Collecting patch metadata for target '{tgt_name}'...")
            # Use ZarrTempStorage to get metadata for all patches across all ranks
            all_patch_info = self.temp_storage.collect_all_patches(tgt_name)
            if not all_patch_info:
                 print(f"Warning: No patch metadata found for target '{tgt_name}' in temporary storage. Skipping.")
                 continue
            if self.verbose: print(f"Collected metadata for {len(all_patch_info)} patches.")

            # --- Prepare for Deferred Loading ---
            patches_by_rank = {}
            for rank_idx, idx, pos in all_patch_info:
                if rank_idx not in patches_by_rank:
                    patches_by_rank[rank_idx] = []
                patches_by_rank[rank_idx].append((idx, pos))

            self.patch_arrays_by_rank[tgt_name] = {} # Store refs per target
            successful_ranks = 0
            for rank_idx in patches_by_rank:
                try:
                    # Get reference to the Zarr array for this rank/target
                    patches_array_ref, _, _ = self.temp_storage.get_all_patches(rank_idx, tgt_name)
                    if patches_array_ref is not None:
                        self.patch_arrays_by_rank[tgt_name][rank_idx] = patches_array_ref
                        successful_ranks += 1
                    else:
                         print(f"Warning: Could not get patch array reference for rank {rank_idx}, target {tgt_name}")
                except Exception as e:
                     print(f"Error getting patch array reference for rank {rank_idx}, target {tgt_name}: {e}")

            if successful_ranks == 0:
                 print(f"Error: Could not get any patch array references for target '{tgt_name}'. Cannot blend.")
                 continue
            if self.verbose: print(f"Prepared deferred loading references for {successful_ranks} ranks.")


            # --- Blending Loop (Chunk-based) ---
            c, max_z, max_y, max_x = output_arrays[tgt_name].shape
            chunk_size = 256 # Process Z-dimension in chunks

            # Sort patches spatially for potentially better memory access patterns (optional)
            all_patch_info.sort(key=lambda x: (x[2][0], x[2][1], x[2][2]))

            if self.verbose: print(f"Blending target '{tgt_name}' ({c} channels, shape {max_z, max_y, max_x}) using z-chunk size {chunk_size}")

            total_chunks = (max_z + chunk_size - 1) // chunk_size
            z_range_iterator = range(0, max_z, chunk_size)
            if self.rank == 0: # Only show progress bar on rank 0
                 from tqdm import tqdm
                 z_range_iterator = tqdm(z_range_iterator, total=total_chunks, desc=f"Blending {tgt_name}")

            for z_start in z_range_iterator:
                z_end = min(z_start + chunk_size, max_z)
                # if self.verbose: print(f"  Processing z-chunk [{z_start}:{z_end}]...")

                # --- Precompute Weight Map for Chunk ---
                chunk_shape = (z_end - z_start, max_y, max_x)
                weight_map = torch.zeros(chunk_shape, device=device, dtype=torch.float32)

                intersecting_patches_for_chunk = []
                for patch_info in all_patch_info:
                    pz, py, px = patch_info[2]
                    if intersects_chunk(pz, py, px, patch_size_tuple, z_start, z_end):
                        intersecting_patches_for_chunk.append(patch_info)

                # if self.verbose: print(f"    Calculating weight map for {len(intersecting_patches_for_chunk)} intersecting patches...")
                for rank_idx, idx, (z, y, x) in intersecting_patches_for_chunk:
                    # Calculate intersection geometry (global, chunk-relative, patch-relative)
                    patch_z_end = min(z + patch_size_tuple[0], max_z)
                    patch_y_end = min(y + patch_size_tuple[1], max_y)
                    patch_x_end = min(x + patch_size_tuple[2], max_x)
                    global_target_z_start = max(z_start, z)
                    global_target_z_end = min(z_end, patch_z_end)
                    if global_target_z_end <= global_target_z_start: continue # No overlap

                    target_z_start = global_target_z_start - z_start
                    target_z_end = global_target_z_end - z_start
                    target_y_start, target_y_end = y, patch_y_end
                    target_x_start, target_x_end = x, patch_x_end

                    patch_z_start_rel = global_target_z_start - z
                    patch_z_end_rel = global_target_z_end - z
                    patch_y_start_rel, patch_y_end_rel = 0, target_y_end - y
                    patch_x_start_rel, patch_x_end_rel = 0, target_x_end - x

                    # Extract and add weights
                    try:
                        weight_slice = blend_weights[
                            patch_z_start_rel:patch_z_end_rel,
                            patch_y_start_rel:patch_y_end_rel,
                            patch_x_start_rel:patch_x_end_rel
                        ]
                        weight_map[
                            target_z_start:target_z_end,
                            target_y_start:target_y_end,
                            target_x_start:target_x_end
                        ] += weight_slice
                    except IndexError as ie:
                         print(f"Warning: Index error accessing blend_weights for patch {rank_idx, idx, (z,y,x)} intersection. Skipping weight contribution.")
                         print(f"  Weight slice indices: Z=[{patch_z_start_rel}:{patch_z_end_rel}], Y=[{patch_y_start_rel}:{patch_y_end_rel}], X=[{patch_x_start_rel}:{patch_x_end_rel}]")
                         print(f"  Weight map slice indices: Z=[{target_z_start}:{target_z_end}], Y=[{target_y_start}:{target_y_end}], X=[{target_x_start}:{target_x_end}]")

                weight_map = torch.clamp(weight_map, min=1e-8) # Avoid division by zero

                # --- Accumulate Weighted Patches ---
                # Initialize output tensor for the chunk (float32 for accumulation)
                # Check the channel dimension from actual patches to verify
                # We need to ensure this matches what the model produces
                # Do not adjust the channel count based on patches
                # The expected channel count from the target configuration is correct
                # The patches have 3 additional position channels that we will strip

                # Create tensor with correct number of channels based on actual model output
                output_tensor = torch.zeros((c,) + chunk_shape, device=device, dtype=torch.float32)

                # if self.verbose: print(f"    Accumulating {len(intersecting_patches_for_chunk)} weighted patches...")
                processed_in_chunk = 0
                for rank_idx, idx, (z, y, x) in intersecting_patches_for_chunk:
                    # Calculate intersection geometry again (could optimize)
                    patch_z_end = min(z + patch_size_tuple[0], max_z)
                    patch_y_end = min(y + patch_size_tuple[1], max_y)
                    patch_x_end = min(x + patch_size_tuple[2], max_x)
                    global_target_z_start = max(z_start, z)
                    global_target_z_end = min(z_end, patch_z_end)
                    if global_target_z_end <= global_target_z_start: continue

                    target_z_start = global_target_z_start - z_start
                    target_z_end = global_target_z_end - z_start
                    target_y_start, target_y_end = y, patch_y_end
                    target_x_start, target_x_end = x, patch_x_end

                    patch_z_start_rel = global_target_z_start - z
                    patch_z_end_rel = global_target_z_end - z
                    patch_y_start_rel, patch_y_end_rel = 0, target_y_end - y
                    patch_x_start_rel, patch_x_end_rel = 0, target_x_end - x

                    # Load patch data on demand
                    try:
                        if rank_idx not in self.patch_arrays_by_rank[tgt_name]:
                            print(f"Warning: Missing patch array reference for rank {rank_idx}, target {tgt_name}. Skipping patch {idx}.")
                            continue

                        patch_data = self.patch_arrays_by_rank[tgt_name][rank_idx][idx] # Load from Zarr
                        # Patch data has shape (C+3, pZ, pY, pX) where first 3 channels are position info
                        patch_tensor = torch.as_tensor(patch_data, device=device).contiguous() # Move to device
                        
                        # The first 3 channels contain position information added by ZarrTempStorage
                        # The actual model output starts is channels 3+ (channels 0,1,2 are position)
                        
                        # Make sure expected channel count is correct
                        expected_model_channels = output_tensor.shape[0]
                        patch_model_channels = patch_tensor.shape[0] - 3  # Subtract the 3 position channels
                        
                        if patch_model_channels != expected_model_channels:
                            print(f"Channel mismatch: Patch has {patch_model_channels} model channels (plus 3 position), output expects {expected_model_channels}")
                        
                        # Extract only the model output channels (skip first 3 position channels)
                        patch_tensor = patch_tensor[3:] # Only keep the actual model output channels
                        
                        # Blend using the specific function
                        blend_patch_weighted(
                            output_tensor, # Accumulator for the chunk
                            patch_tensor,  # Current patch tensor
                            blend_weights, # Precomputed full patch weights
                            target_z_start, target_z_end, # Chunk-relative Z slice
                            target_y_start, target_y_end, # Chunk-relative Y slice (global coords)
                            target_x_start, target_x_end, # Chunk-relative X slice (global coords)
                            patch_z_start_rel, patch_z_end_rel, # Patch-relative Z slice
                            patch_y_start_rel, patch_y_end_rel, # Patch-relative Y slice
                            patch_x_start_rel, patch_x_end_rel  # Patch-relative X slice
                        )
                        processed_in_chunk += 1
                    except Exception as e:
                         print(f"Error loading/blending patch (target={tgt_name}, rank={rank_idx}, idx={idx}): {e}")
                         # Decide whether to continue or raise

                # --- Normalize and Save Chunk ---
                # if self.verbose: print(f"    Normalizing accumulated chunk by weight map...")
                for c_idx in range(output_tensor.shape[0]):
                    output_tensor[c_idx] /= weight_map

                # Save the normalized chunk back to the SUM array (which now acts as the blended array)
                # The _finalize_arrays step will read from this.
                # if self.verbose: print(f"    Saving normalized chunk to output_arrays['{tgt_name}']...")
                try:
                    # Convert to float16 before saving if needed, but float32 is safer for intermediate
                    output_arrays[tgt_name][:, z_start:z_end] = output_tensor.cpu().numpy() # Write back to Zarr
                except Exception as e:
                     print(f"Error writing normalized chunk back to Zarr for target {tgt_name}, chunk [{z_start}:{z_end}]: {e}")

                # Clean up GPU memory for the chunk
                del output_tensor, weight_map
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            if self.verbose: print(f"Finished blending loop for target '{tgt_name}'.")

        if self.verbose: print("Patch blending phase completed.")


    def _finalize_arrays(self, output_arrays, count_arrays, final_arrays):
        """
        Finalizes the blended arrays stored in `output_arrays`.

        Reads the blended (normalized) data chunk by chunk, applies activation
        (softmax/sigmoid), thresholds if necessary, converts to the final dtype,
        and saves to the `final_arrays`.

        Args:
            output_arrays: Dict mapping target_name to the blended Zarr arrays (previously sum arrays).
            count_arrays: Dict mapping target_name to count Zarr arrays (not used after weight map blending).
            final_arrays: Dict mapping target_name to the final output Zarr arrays.
        """
        if self.verbose:
            print("Finalizing arrays (applying activation, threshold, type conversion)...")

        device = self.device_str

        for target in self.targets:
            tgt_name = target.get("name")
            if tgt_name not in output_arrays or tgt_name not in final_arrays or final_arrays[tgt_name] is None:
                print(f"Warning: Missing blended or final array for target '{tgt_name}'. Skipping finalization.")
                continue

            if self.verbose: print(f"Finalizing target: '{tgt_name}'")

            # Define threshold value (0.0 to 1.0)
            threshold_val = self.threshold / 100.0 if self.threshold is not None else None

            # Get shapes and channel info
            blended_array = output_arrays[tgt_name] # This now holds the blended, normalized data
            final_array = final_arrays[tgt_name]
            num_channels, z_max, y_max, x_max = blended_array.shape
            final_num_channels = final_array.shape[0] # Channels in the final output

            is_multiclass = num_channels > 2
            is_binary = num_channels == 2

            # Determine processing modes
            compute_argmax = not self.save_probability_maps and is_multiclass
            handle_binary = is_binary # Special handling for binary case

            chunk_size = 256 # Process in chunks

            z_range_iterator = range(0, z_max, chunk_size)
            if self.rank == 0:
                from tqdm import tqdm
                z_range_iterator = tqdm(z_range_iterator, desc=f"Finalizing {tgt_name}")

            for z_start in z_range_iterator:
                z_end = min(z_start + chunk_size, z_max)

                # Load blended chunk (logits/pre-activation values)
                # Ensure it's float32 for activation functions
                blended_tensor = torch.as_tensor(blended_array[:, z_start:z_end],
                                                 device=device, dtype=torch.float32).contiguous()

                # --- Apply Activation ---
                if handle_binary:
                    # Apply softmax for binary case (nnUNet standard) -> probabilities
                    activated_tensor = torch.softmax(blended_tensor, dim=0)
                    # For final output, decide based on save_probability_maps
                    if self.save_probability_maps:
                        # Save foreground probability (channel 1) scaled to uint8
                        foreground_prob = activated_tensor[1]
                        final_chunk_data = (foreground_prob * 255).to(torch.uint8)
                        # Check if final array expects 2 channels (prob + mask)
                        if final_num_channels == 2:
                             # Also compute binary mask from argmax
                             binary_mask = torch.argmax(activated_tensor, dim=0).to(torch.uint8) * 255
                             # Combine: Ch0=Prob, Ch1=Mask
                             combined = torch.stack([final_chunk_data, binary_mask], dim=0)
                             final_chunk_data = combined
                        elif final_num_channels == 1:
                             # Only save probability if final array is single channel
                             final_chunk_data = final_chunk_data.unsqueeze(0) # Add channel dim
                        else:
                             print(f"Warning: Unexpected final channel count ({final_num_channels}) for binary prob map saving.")
                             final_chunk_data = final_chunk_data.unsqueeze(0) # Default to saving only prob

                    else:
                        # Save binary mask (argmax) scaled to uint8
                        binary_mask = torch.argmax(activated_tensor, dim=0).to(torch.uint8) * 255
                        final_chunk_data = binary_mask.unsqueeze(0) # Add channel dim

                elif is_multiclass:
                     # Apply softmax for multiclass -> probabilities
                     activated_tensor = torch.softmax(blended_tensor, dim=0)
                     if compute_argmax:
                          # Compute argmax for class labels
                          argmax_result = torch.argmax(activated_tensor, dim=0).to(torch.uint8)
                          final_chunk_data = argmax_result.unsqueeze(0) # Add channel dim
                     else:
                          # Save probability maps scaled to uint8
                          final_chunk_data = (activated_tensor * 255).to(torch.uint8)

                else: # Single channel output (regression or already activated)
                     # Assume data is already suitable or apply sigmoid if needed (e.g., if output was logits)
                     # For simplicity, assume it's ready. If activation is needed, add it here.
                     # Example: activated_tensor = torch.sigmoid(blended_tensor)
                     activated_tensor = blended_tensor # Pass through for now
                     # Apply threshold if saving binary mask from single channel probability
                     if threshold_val is not None and not self.save_probability_maps:
                          final_chunk_data = (activated_tensor >= threshold_val).to(torch.uint8) * 255
                     elif self.save_probability_maps:
                          # Scale single channel probability to uint8
                          final_chunk_data = (activated_tensor.clamp(0, 1) * 255).to(torch.uint8)
                     else: # Shouldn't happen if threshold logic is correct
                          final_chunk_data = activated_tensor.to(torch.uint8) # Cast directly if no threshold/probs


                # --- Thresholding (if applicable and not done via argmax) ---
                # Thresholding is now integrated into the activation logic above where appropriate.

                # --- Type Conversion & Saving ---
                # Ensure final_chunk_data has the correct shape and type for the final array
                if final_chunk_data.shape[0] != final_num_channels:
                    print(f"Warning: Shape mismatch writing final chunk for {tgt_name}. "
                          f"Expected channels {final_num_channels}, got {final_chunk_data.shape[0]}. Adjusting.")
                    # Basic handling: take first channel if final expects 1, or broadcast if needed (less likely)
                    if final_num_channels == 1:
                         final_chunk_data = final_chunk_data[0:1, ...] # Take first channel
                    # Add more sophisticated reshaping/padding if needed

                # Convert to final dtype before writing
                final_dtype_np = final_array.dtype
                if final_chunk_data.dtype != torch.uint8 and final_dtype_np == np.uint8:
                     # If final is uint8 but current is not (e.g., float after sigmoid)
                     # Ensure scaling happened correctly above, then cast
                     pass # Casting should happen implicitly below if needed
                elif final_dtype_np == np.float32 and final_chunk_data.dtype != torch.float32:
                     final_chunk_data = final_chunk_data.float()
                # Add other dtype conversions if necessary

                # Write to final Zarr array
                try:
                     final_array[:, z_start:z_end] = final_chunk_data.cpu().numpy()
                except Exception as write_e:
                     print(f"Error writing final chunk for target {tgt_name}, chunk [{z_start}:{z_end}]: {write_e}")
                     print(f"  Final array shape: {final_array.shape}, dtype: {final_array.dtype}")
                     print(f"  Data chunk shape: {final_chunk_data.shape}, dtype: {final_chunk_data.dtype}")


                # Clean up GPU memory
                del blended_tensor, activated_tensor, final_chunk_data
                if torch.cuda.is_available() and (z_start // chunk_size) % 4 == 0:
                    torch.cuda.empty_cache()

            if self.verbose: print(f"Finished finalizing target: '{tgt_name}'")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.verbose: print("Array finalization complete.")


    # _process_model_outputs remains largely the same, ensure it queues data for ZarrTempStorage

    def _process_model_outputs(self, outputs, positions):
        """
        Process model outputs (raw logits) and queue them for writing to temp storage.
        """
        processed_outputs = {}
        if torch.is_tensor(outputs):
            # Handle single tensor output from model
            # Assume it corresponds to the first target if multiple exist
            first_target_name = self.targets[0].get("name", "segmentation")
            processed_outputs[first_target_name] = outputs
        elif isinstance(outputs, dict):
            processed_outputs = outputs
        else:
            print(f"Warning: Unexpected model output type: {type(outputs)}. Attempting to process first target.")
            first_target_name = self.targets[0].get("name", "segmentation")
            # Try to convert if possible, otherwise skip
            try:
                 processed_outputs[first_target_name] = torch.as_tensor(outputs)
            except Exception:
                 print("Error: Could not convert model output to tensor.")
                 return # Skip processing this batch

        batch_size = next(iter(processed_outputs.values())).shape[0]
        patch_size_tuple = tuple(self.patch_size)

        # Use index from 0 to batch_size-1 for positions list/tensor access
        for i in range(batch_size):
            # Ensure position access is safe
            if i >= len(positions):
                print(f"Warning: Output index {i} exceeds number of positions {len(positions)}. Skipping.")
                continue
            pos = positions[i] # Get the (z,y,x) tuple or tensor element

            for target in self.targets:
                tgt_name = target.get("name")
                if tgt_name not in processed_outputs:
                    print(f"Warning: Target '{tgt_name}' not found in model outputs. Skipping.")
                    continue

                try:
                    # Get the raw logits for this patch and target, move to CPU
                    pred_tensor = processed_outputs[tgt_name][i].detach().cpu()
                    
                    # Debug output for tensor shapes
                    if self.verbose and i < 2:
                        print(f"DEBUG: Model output tensor shape for {tgt_name}[{i}]: {pred_tensor.shape}")
                        print(f"DEBUG: Expected num_seg_heads: {self.model_info.get('num_seg_heads', 'N/A')}")
                        print(f"DEBUG: Target channels: {target.get('channels', 'N/A')}")

                    # Convert to float16 numpy array for storage efficiency in temp zarr
                    # Blending will happen in float32 later
                    pred_array = pred_tensor.to(torch.float16).numpy()

                    # Convert position tensor to tuple if necessary
                    if isinstance(pos, torch.Tensor):
                        pos_tuple = tuple(pos.tolist())
                    else:
                         pos_tuple = tuple(pos) # Assume it's already a tuple/list

                    # Queue the data, position tuple, and target name
                    self.writer_queue.put((pred_array, pos_tuple, tgt_name))

                    # if self.verbose and i < 1 and self.rank == 0: # Log only first patch per batch on rank 0
                    #     print(f"  Queued patch for {tgt_name}: pos={pos_tuple}, shape={pred_array.shape}, dtype={pred_array.dtype}")

                except Exception as e:
                     print(f"Error processing/queuing output for target {tgt_name}, index {i}, position {pos}: {e}")
                     # Log shapes for debugging
                     print(f"  Output tensor shape for {tgt_name}: {processed_outputs[tgt_name].shape}")
                     print(f"  Attempted slice index: {i}")


    def infer(self, skip_blending=False):
        """
        Main inference method.
        Sets up temporary storage, runs model inference, optionally blends patches.
        """
        # --- Setup ---
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        start_global_time = time.time()

        # Create temp directory path
        output_dir = os.path.dirname(self.output_path)
        temp_dir_base = "temp"
        if hasattr(self.dataset, 'num_parts') and self.dataset.num_parts > 1:
            temp_dir_base = f"temp_part{self.dataset.part_id}"
        self.temp_dir = os.path.join(output_dir, temp_dir_base)

        # Create temp dir (rank 0 might do it first, others wait or ignore error)
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except OSError:
             pass # Ignore if exists or race condition

        if self.verbose:
            print(f"Rank {self.rank}: Using temporary storage directory: {self.temp_dir}")

        # Initialize ZarrTempStorage
        self.temp_storage = ZarrTempStorage(
            output_path=self.temp_dir,
            rank=self.rank,
            world_size=world_size,
            verbose=self.verbose,
            num_io_workers=self.num_write_workers # Pass num writers
        )
        # Determine expected patches based on dataset length after distributed splitting
        total_patches_this_rank = len(self.dataset)
        expected_patches = int(total_patches_this_rank * 1.05) + 5 # 5% margin + 5
        self.temp_storage.initialize(expected_patch_count=expected_patches)
        
        # Set the expected patch count for each target
        for target in self.targets:
            target_name = target.get('name')
            self.temp_storage.set_expected_patch_count(target_name, expected_patches)
            
        if self.verbose: print(f"Rank {self.rank}: Initialized temp storage, expecting ~{expected_patches} patches for each target.")

        # --- Save Blend Config (Rank 0 only) ---
        if self.rank == 0:
            blend_config = {
                "patch_size": list(self.patch_size),
                "step_size": self.tile_step_size,
                "threshold": self.threshold,
                "save_probability_maps": self.save_probability_maps,
                "edge_weight_boost": self.edge_weight_boost,
                "output_path": self.output_path,
                 # Add target info needed for blending
                 "targets": self.targets,
                 # Add full volume shape for context
                 "volume_shape": list(self.dataset.input_shape)
            }
            config_path = os.path.join(self.temp_dir, "blend_config.json")
            try:
                with open(config_path, 'w') as f:
                    json.dump(blend_config, f, indent=2)
                if self.verbose: print(f"Saved blending configuration to {config_path}")
            except Exception as e:
                print(f"Warning: Failed to save blending configuration: {e}")

        # --- Prepare Output Arrays (Rank 0 only) ---
        output_arrays = {} # Stores Zarr array references for the blended data (previously sum)
        count_arrays = {} # Stores Zarr array references for counts (not used with weight maps, but keep structure)
        final_arrays = {} # Stores Zarr array references for the final output

        if self.rank == 0:
            if os.path.exists(self.output_path):
                 print(f"Warning: Output path '{self.output_path}' already exists. Overwriting.")
                 # Potentially add overwrite protection here if needed
                 # shutil.rmtree(self.output_path) # Example: remove existing

            input_shape = self.dataset.input_shape # Full shape (C,Z,Y,X) or (Z,Y,X)
            if len(input_shape) == 3: # Add channel dim if missing
                 input_shape = (1,) + input_shape
                 print(f"Warning: Input shape from dataset is 3D {self.dataset.input_shape}. Assuming 1 channel internally: {input_shape}")

            spatial_shape = input_shape[1:] # Z, Y, X
            z_max, y_max, x_max = spatial_shape
            if self.verbose: print(f"Creating output arrays based on spatial shape: {spatial_shape}")

            compressor = Blosc(cname='zstd', clevel=3)
            patch_size_tuple = tuple(self.patch_size)

            output_store = None # To hold the final zarr group reference

            for target in self.targets:
                tgt_name = target.get("name")
                num_classes = target.get("channels", 1)
                is_binary = num_classes == 2

                # Shape for intermediate blended array (stores logits/pre-activation)
                # Always float32 for accumulation/blending precision
                blended_shape = (num_classes, z_max, y_max, x_max)
                blended_dtype = 'float32'

                # Shape and dtype for the final output array
                final_dtype = 'uint8' # Default for segmentation
                if not self.save_probability_maps and num_classes > 2: # Multiclass argmax
                    final_shape = (1, z_max, y_max, x_max)
                elif not self.save_probability_maps and is_binary: # Binary argmax
                    final_shape = (1, z_max, y_max, x_max)
                elif self.save_probability_maps and is_binary: # Binary prob + mask
                     final_shape = (2, z_max, y_max, x_max)
                else: # Multiclass probabilities or single channel output
                    final_shape = (num_classes, z_max, y_max, x_max)

                # Chunking based on patch size, ensuring channel dim is included
                # For final array, use channel dim of final_shape
                blended_chunks = (1,) + patch_size_tuple # Chunk channels individually for blended
                final_chunks = (1,) + patch_size_tuple # Keep consistent chunking

                if self.verbose:
                    print(f"Target '{tgt_name}':")
                    print(f"  Blended (Sum/Weighted) Array: shape={blended_shape}, dtype={blended_dtype}, chunks={blended_chunks}")
                    print(f"  Final Output Array: shape={final_shape}, dtype={final_dtype}, chunks={final_chunks}")

                # Create intermediate blended array in temp storage
                output_arrays[tgt_name] = self.temp_storage.temp_zarr.create_dataset(
                    f"blended_{tgt_name}", # Renamed from "sum_"
                    shape=blended_shape,
                    chunks=blended_chunks,
                    dtype=blended_dtype,
                    compressor=compressor,
                    fill_value=0,
                    write_empty_chunks=False
                )
                # Create dummy count array (not used but keeps structure)
                count_arrays[tgt_name] = self.temp_storage.temp_zarr.create_dataset(
                    f"count_{tgt_name}",
                    shape=spatial_shape,
                    chunks=patch_size_tuple,
                    dtype='uint8',
                    compressor=compressor,
                    fill_value=0,
                    write_empty_chunks=False
                )

                # Create final output array (only if not skipping blending)
                if not skip_blending:
                    if output_store is None:
                         # Open/create the main output Zarr store
                         output_store = zarr.open(self.output_path, mode='w') # Use 'w' to ensure it's fresh

                    # Create dataset within the output store
                    # Handle root array for 'segmentation' target
                    if tgt_name == 'segmentation':
                         # Check if root array already exists (shouldn't with mode='w')
                         if '/' in output_store and isinstance(output_store['/'], zarr.Array):
                              print(f"Warning: Root array already exists in {self.output_path}. Re-using.")
                              final_arrays[tgt_name] = output_store['/']
                              # Verify shape/dtype/chunks? Or assume compatible? For now, assume ok.
                         else:
                              # Create the root array directly
                              final_arrays[tgt_name] = zarr.creation.create(
                                   shape=final_shape, chunks=final_chunks, dtype=final_dtype,
                                   compressor=compressor, store=output_store.store, path=None, # Use path=None for root
                                   overwrite=True, write_empty_chunks=False
                              )
                    else:
                         # Create as a named dataset within the group
                         final_arrays[tgt_name] = output_store.create_dataset(
                              tgt_name,
                              shape=final_shape,
                              chunks=final_chunks,
                              dtype=final_dtype,
                              compressor=compressor,
                              write_empty_chunks=False
                         )
                else:
                    final_arrays[tgt_name] = None # Placeholder if skipping blend

            # Add dataset attributes (like resolution, offset) to the final output store if available
            if not skip_blending and output_store is not None:
                 try:
                      # Example: Copy metadata if available in Volume object
                      if hasattr(self.dataset.volume, 'metadata') and self.dataset.volume.metadata:
                           # Add relevant metadata attributes
                           if 'zattrs' in self.dataset.volume.metadata:
                                # Could copy 'multiscales', 'omero', etc. if appropriate
                                # Be careful not to overwrite essential Zarr attributes
                                pass # Add specific attribute copying here if needed
                      # Add voxel spacing if known (example)
                      # if hasattr(self.dataset.volume, 'resolution'):
                      #    output_store.attrs['resolution'] = self.dataset.volume.resolution
                 except Exception as meta_e:
                      print(f"Warning: Could not write metadata attributes to output Zarr: {meta_e}")


        # --- Sync point ---
        if dist.is_initialized():
            dist.barrier()

        # --- Setup Writer Threads ---
        writer_threads = []
        for worker_id in range(self.num_write_workers):
            thread = threading.Thread(
                target=zarr_writer_worker,
                args=(self.temp_storage, self.writer_queue, worker_id, self.verbose and (self.rank == 0)) # Only rank 0 logs writer details
            )
            thread.daemon = True
            thread.start()
            writer_threads.append(thread)

        # --- Inference Loop ---
        if self.verbose: print(f"Rank {self.rank}: Starting inference loop...")
        network = self.model_info['network']
        network.eval()
        network.to(self.device_str)

        inference_start_time = time.time()
        processed_patches_rank = 0

        progress_bar = None
        if self.rank == 0:
            from tqdm import tqdm
            total_iterations = (total_patches_this_rank + self.batch_size - 1) // self.batch_size
            progress_bar = tqdm(total=total_iterations, desc=f"Infer Rank 0/{world_size}", position=0, leave=True)

        # Use autocast for mixed precision
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.device_str != 'cpu')):
            for batch_idx, batch in enumerate(self.dataloader):
                inputs = batch['data'].to(self.device_str, non_blocking=True).contiguous()
                positions = batch['pos'] # List of tuples

                if inputs.dtype != torch.float32:
                     inputs = inputs.float() # Ensure float32 for model input

                # TTA or Standard Inference
                if self.use_mirroring: # This flag now controls all TTA
                    augmented_inputs, transform_info = get_tta_augmented_inputs(
                        input_tensor=inputs,
                        model_info=self.model_info,
                        max_tta_combinations=self.max_tta_combinations,
                        use_rotation_tta=self.use_rotation_tta, # Use rotation if enabled
                        use_mirroring=not self.use_rotation_tta, # Use mirroring only if rotation is off
                        rotation_weights=self.rotation_weights,
                        verbose=self.verbose and (self.rank==0) and (batch_idx==0), # Limit TTA verbosity
                        rank=self.rank
                    )
                    tta_outputs = []
                    for aug_input, transform in zip(augmented_inputs, transform_info):
                         output = network(aug_input)
                         tta_outputs.append((output, transform))
                    outputs = merge_tensors(tta_outputs) # Merge TTA results
                    del augmented_inputs, tta_outputs # Free memory
                else:
                    outputs = network(inputs) # Standard inference

                # Process and queue outputs
                self._process_model_outputs(outputs, positions)

                processed_patches_rank += inputs.shape[0]
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"patches": f"{processed_patches_rank}/{total_patches_this_rank}"})

                # Clean cache periodically
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                     torch.cuda.empty_cache()

        inference_time = time.time() - inference_start_time
        if progress_bar: progress_bar.close()
        if self.verbose: print(f"Rank {self.rank}: Inference loop finished in {inference_time:.2f}s.")

        # --- Finalize Writers ---
        if self.verbose: print(f"Rank {self.rank}: Waiting for writers...")
        # Wait for queue to empty (with timeout)
        queue_wait_start = time.time()
        while not self.writer_queue.empty():
            if time.time() - queue_wait_start > 60:
                print(f"Rank {self.rank}: Warning - Writer queue not empty after 60s timeout.")
                break
            time.sleep(0.1)
        # Send stop signals
        for _ in range(self.num_write_workers): self.writer_queue.put(None)
        # Join threads (with timeout)
        for thread in writer_threads: thread.join(timeout=10)
        if self.verbose: print(f"Rank {self.rank}: Writers finished.")

        # Finalize counts in temp storage
        for target in self.targets: self.temp_storage.finalize_target(target.get("name"))

        # --- Sync Point ---
        if dist.is_initialized():
            if self.verbose: print(f"Rank {self.rank}: Barrier before blending...")
            dist.barrier()
            if self.verbose: print(f"Rank {self.rank}: Passed barrier.")

        # --- Blending and Finalization (Rank 0 only) ---
        blend_start_time = time.time()
        if not skip_blending:
            if self.rank == 0:
                if self.verbose: print("Rank 0: Starting blending and finalization...")
                try:
                    # Ensure final arrays are accessible (re-open if necessary)
                    reopened_final_arrays = {}
                    final_store = zarr.open(self.output_path, mode='a')
                    for tgt_name in final_arrays:
                        if tgt_name == 'segmentation':
                             reopened_final_arrays[tgt_name] = final_store # Root array
                        else:
                             reopened_final_arrays[tgt_name] = final_store[tgt_name]

                    # Blend using weight map approach (reads temp, writes to intermediate/blended array)
                    self._blend_patches({}, output_arrays, count_arrays) # Pass empty dict for patch_arrays arg

                    # Finalize (reads intermediate/blended, applies activation/threshold, writes to final)
                    self._finalize_arrays(output_arrays, count_arrays, reopened_final_arrays)

                    if self.verbose: print("Rank 0: Blending and finalization complete.")
                except Exception as e:
                     print(f"Rank 0: ERROR during blending/finalization: {e}")
                     import traceback
                     traceback.print_exc()
                     # Mark as preserved to avoid cleanup issues
                     self.temp_dir_preserved = True
                     raise # Re-raise after marking
            else:
                 # Other ranks don't blend but need the placeholder dict for cleanup logic
                 final_arrays = {} # Ensure it exists but is empty
        else:
             if self.rank == 0: print("Skipping blending phase.")
             self.temp_dir_preserved = True # Mark for preservation

        blend_time = time.time() - blend_start_time

        # --- Sync Point and Cleanup ---
        if dist.is_initialized():
            dist.barrier()

        # Cleanup (Rank 0 handles shared temp dir removal)
        if self.rank == 0:
            if hasattr(self, 'temp_dir_preserved') and self.temp_dir_preserved:
                 print(f"Rank 0: Preserving temporary storage: {self.temp_storage.output_path if self.temp_storage else self.temp_dir}")
                 if self.temp_storage: 
                     # Close storage without deleting
                     try:
                         self.temp_storage.close()
                     except AttributeError:
                         print("Warning: temp_storage does not have close method, releasing references")
                         self.temp_storage = None
            else:
                 if self.verbose: print(f"Rank 0: Cleaning up temporary storage...")
                 if self.temp_storage: self.temp_storage.cleanup()
                 # Attempt to remove the main temp directory if it's empty
                 try:
                      if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                           os.rmdir(self.temp_dir)
                           if self.verbose: print(f"Rank 0: Removed empty temp directory {self.temp_dir}")
                 except OSError as e:
                      if self.verbose: print(f"Rank 0: Note - Could not remove temp directory (may not be empty or permission issue): {e}")

        # Clear local references (all ranks)
        output_arrays.clear()
        count_arrays.clear()
        final_arrays.clear()
        self.patch_arrays_by_rank.clear()

        # --- Finish ---
        if dist.is_initialized(): dist.barrier()
        end_global_time = time.time()
        self.total_time = end_global_time - start_global_time
        if self.verbose:
            print(f"Rank {self.rank}: Inference finished. Total time: {self.total_time:.2f}s. Blending time (rank 0): {blend_time:.2f}s.")

# run_worker, single_process_inference, main need updates to pass new args

def run_worker(rank, world_size,
              input_path,
              output_path,
              model_folder,
              hf_model_path,
              hf_token,
              fold,
              checkpoint,
              dist_port,
              batch_size,
              step_size,
              input_format,
              num_workers,
              num_write_workers,
              verbose,
              use_mirroring, # Combined TTA flag
              max_tta_combinations,
              use_rotation_tta,
              rotation_weights,
              save_probability_maps,
              skip_blending,
              threshold,
              dist_backend,
              num_parts,
              part_id,
              # New Volume params
              scroll_id,
              energy,
              resolution,
              segment_id,
              cache, # Renamed from volume_cache
              cache_pool,
              normalization_scheme,
              global_mean,
              global_std,
              return_as_type, # New
              domain, # Renamed from volume_domain
              no_compile=False): # Added no_compile parameter with default value
    """Worker function that runs on each GPU with explicit arguments."""
    import torch.distributed as dist
    import torch
    import os
    import time

    # --- Distributed Setup ---
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(dist_port) # Use the passed port

    backend = dist_backend
    if not torch.cuda.is_available() and backend == 'nccl':
        backend = 'gloo'

    try:
        if verbose and rank==0: print(f"Rank {rank}: Initializing process group ({backend})...")
        dist.init_process_group(backend=backend)
        if verbose and rank==0: print(f"Rank {rank}: Process group initialized.")
    except Exception as e:
        print(f"Rank {rank}: FATAL - Failed to initialize process group: {e}")
        raise

    # --- Device Setup ---
    if torch.cuda.is_available():
        device_idx = rank % torch.cuda.device_count()
        device = f"cuda:{device_idx}"
        torch.cuda.set_device(device_idx)
    else:
        device = "cpu"
    if verbose: print(f"Rank {rank}: Assigned to device {device}")

    # --- Load Model ---
    if verbose and rank==0: print(f"Rank {rank}: Loading model...")
    try:
        fold_parsed = int(fold) if fold.isdigit() else fold
        # Set environment variable for compile flag if no_compile is True
        if no_compile:
            os.environ['nnUNet_compile'] = 'false'
            if verbose and rank == 0:
                print("torch.compile optimization disabled")
            
        model_info = load_model_for_inference(
            model_folder=model_folder,
            hf_model_path=hf_model_path,
            hf_token=hf_token,
            fold=fold_parsed,
            checkpoint_name=checkpoint,
            device_str=device, # Load directly to target device
            use_mirroring=use_mirroring, # Pass TTA flag
            verbose=verbose and (rank == 0),
            rank=rank
        )
        if verbose and rank==0: print(f"Rank {rank}: Model loaded.")
    except Exception as e:
         print(f"Rank {rank}: FATAL - Failed to load model: {e}")
         raise

    # --- Prepare Dataset and Dataloader ---
    patch_size = model_info['patch_size']
    num_input_channels = model_info['num_input_channels']
    num_seg_heads = model_info.get('num_seg_heads', 2)
    output_targets = [{
        "name": "segmentation", "channels": num_seg_heads,
        "activation": "sigmoid" if num_seg_heads <= 2 else "softmax",
        "nnunet_output_channels": num_seg_heads
    }]

    if verbose and rank == 0: print(f"Rank {rank}: Creating VCDataset...")
    try:
        dataset = VCDataset(
            input_path=str(input_path) if input_path is not None else None, # Ensure string or None
            targets=output_targets,
            patch_size=patch_size,
            num_input_channels=num_input_channels,
            input_format=input_format,
            step_size=step_size,
            load_all=False,
            verbose=verbose and (rank == 0),
            num_parts=num_parts,
            part_id=part_id,
            # --- Pass Volume params ---
            scroll_id=scroll_id,
            energy=energy,
            resolution=resolution,
            segment_id=segment_id,
            cache=cache,
            cache_pool=cache_pool,
            normalization_scheme=normalization_scheme,
            global_mean=global_mean,
            global_std=global_std,
            return_as_type=return_as_type, # Pass through
            # return_as_tensor=True, # VCDataset forces True
            domain=domain
        )
        if verbose and rank == 0: print(f"Rank {rank}: VCDataset created.")
    except Exception as e:
         print(f"Rank {rank}: FATAL - Failed to create VCDataset: {e}")
         import traceback
         traceback.print_exc()
         # Attempt cleanup before raising
         if dist.is_initialized(): dist.destroy_process_group()
         raise

    # Distribute dataset patches
    dataset.set_distributed(rank, world_size)
    if verbose: print(f"Rank {rank}: Dataset configured for distributed processing. Length = {len(dataset)}")

    dataloader_workers = max(1, num_workers // world_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=dataloader_workers, pin_memory=True,
        collate_fn=custom_collate, prefetch_factor=4 # Reduced prefetch
    )

    # --- Create ZarrInferer ---
    if verbose and rank == 0: print(f"Rank {rank}: Creating ZarrInferer...")
    write_workers_per_rank = max(1, num_write_workers // world_size)
    try:
        inference = ZarrInferer(
            input_path=str(input_path) if input_path is not None else None, # Pass identifier
            output_path=output_path,
            model_info=model_info,
            dataset=dataset,
            dataloader=dataloader,
            # patch_size already inferred from model_info
            batch_size=batch_size,
            step_size=step_size,
            num_write_workers=write_workers_per_rank,
            threshold=threshold,
            use_mirroring=use_mirroring, # Pass TTA flag
            max_tta_combinations=max_tta_combinations,
            use_rotation_tta=use_rotation_tta,
            rotation_weights=rotation_weights,
            save_probability_maps=save_probability_maps,
            verbose=verbose and (rank == 0),
            rank=rank,
            edge_weight_boost=0, # Keep legacy arg
            output_targets=output_targets
        )
        if verbose and rank == 0: print(f"Rank {rank}: ZarrInferer created.")
    except Exception as e:
         print(f"Rank {rank}: FATAL - Failed to create ZarrInferer: {e}")
         # Attempt cleanup before raising
         if dist.is_initialized(): dist.destroy_process_group()
         raise


    # --- Run Inference ---
    inference_start_time = time.time()
    try:
        if verbose and rank == 0: print(f"Rank {rank}: Starting inference run...")
        inference.infer(skip_blending=skip_blending)
        if verbose and rank == 0: print(f"Rank {rank}: Inference run completed.")

    except Exception as e:
        inference.total_time = time.time() - inference_start_time
        print(f"Rank {rank}: ERROR during inference after {inference.total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        if dist.is_initialized():
            try:
                if verbose: print(f"Rank {rank}: Destroying process group...")
                dist.destroy_process_group()
                if verbose: print(f"Rank {rank}: Process group destroyed.")
            except Exception as e:
                print(f"Rank {rank}: Error destroying process group: {e}")

def single_process_inference(args):
    """Run inference in single process mode"""
    import torch
    import time
    print("Running in single process mode.")

    # Ensure input path is not empty or None and is a string
    input_path = str(args.input) if args.input is not None else None
    if not input_path:
         # Need to check if scroll_id is provided as an alternative identifier
         if args.scroll_id is None and args.segment_id is None:
              raise ValueError("Input path/identifier is required for single process inference.")
         # Use scroll_id or segment_id to construct a default input identifier if path is missing
         elif args.segment_id is not None:
              input_path = str(args.segment_id)
              print(f"Using segment_id {input_path} as input identifier.")
         elif args.scroll_id is not None:
              input_path = f"scroll{args.scroll_id}"
              print(f"Using scroll_id {input_path} as input identifier.")

    # Load Model
    if args.verbose: print("Loading model...")
    try:
        fold_parsed = int(args.fold) if args.fold.isdigit() else args.fold
        
        # Set environment variable for compile flag if args.no_compile is True
        if args.no_compile:
            os.environ['nnUNet_compile'] = 'false'
            if args.verbose:
                print("torch.compile optimization disabled")
                
        model_info = load_model_for_inference(
            model_folder=args.model_folder,
            hf_model_path=args.hf_model_path,
            hf_token=args.hf_token,
            fold=fold_parsed,
            checkpoint_name=args.checkpoint,
            device_str=args.device,
            use_mirroring=not (args.no_tta or args.max_tta_combinations == 0), # Determine TTA flag
            verbose=args.verbose,
            rank=0
        )
        if args.verbose: print("Model loaded.")
    except Exception as e:
         print(f"FATAL: Failed to load model: {e}")
         raise

    # Prepare Dataset and Dataloader
    patch_size = model_info['patch_size']
    num_input_channels = model_info['num_input_channels']
    num_seg_heads = model_info.get('num_seg_heads', 2)
    output_targets = [{
        "name": "segmentation", "channels": num_seg_heads,
        "activation": "sigmoid" if num_seg_heads <= 2 else "softmax",
        "nnunet_output_channels": num_seg_heads
    }]

    if args.verbose: print("Creating VCDataset...")
    try:
        dataset = VCDataset(
            input_path=input_path, # Use validated/constructed input_path
            targets=output_targets,
            patch_size=patch_size,
            num_input_channels=num_input_channels,
            input_format=args.input_format,
            step_size=args.step_size,
            load_all=False,
            verbose=args.verbose,
            num_parts=args.num_parts,
            part_id=args.part_id,
            # Pass Volume params from args
            scroll_id=args.scroll_id,
            energy=args.energy,
            resolution=args.resolution,
            segment_id=args.segment_id,
            cache=False if args.no_volume_cache else args.cache,
            cache_pool=args.cache_pool,
            normalization_scheme=args.normalization_scheme,
            global_mean=args.global_mean,
            global_std=args.global_std,
            return_as_type=args.return_as_type,
            domain=args.volume_domain if hasattr(args, 'volume_domain') and args.volume_domain is not None else args.domain
        )
        if args.verbose: print(f"VCDataset created with length {len(dataset)}.")
    except Exception as e:
         print(f"FATAL: Failed to create VCDataset: {e}")
         import traceback
         traceback.print_exc()
         raise

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=custom_collate, prefetch_factor=4 # Reduced prefetch
    )

    # Create ZarrInferer
    if args.verbose: print("Creating ZarrInferer...")
    try:
        # Determine TTA flags
        use_tta = not (args.no_tta or args.max_tta_combinations == 0)
        use_rotation_tta = not args.no_rotation_tta and use_tta

        inference = ZarrInferer(
            input_path=input_path, # Pass identifier
            output_path=args.output,
            model_info=model_info,
            dataset=dataset,
            dataloader=dataloader,
            # patch_size comes from model_info
            batch_size=args.batch_size,
            step_size=args.step_size,
            num_write_workers=args.num_write_workers,
            threshold=args.threshold,
            use_mirroring=use_tta, # Pass overall TTA flag
            max_tta_combinations=args.max_tta_combinations,
            use_rotation_tta=use_rotation_tta, # Pass specific rotation flag
            rotation_weights=args.rotation_weights,
            save_probability_maps=not args.no_probabilities,
            verbose=args.verbose,
            rank=0,
            edge_weight_boost=0,
            output_targets=output_targets
        )
        if args.verbose: print("ZarrInferer created.")
    except Exception as e:
         print(f"FATAL: Failed to create ZarrInferer: {e}")
         raise

    # Run Inference
    inference_start_time = time.time()
    try:
        if args.verbose: print("Starting inference run...")
        inference.infer(skip_blending=args.skip_blending)
        inference.total_time = time.time() - inference_start_time
        print(f"Inference completed in {inference.total_time:.2f} seconds.")

    except Exception as e:
        inference.total_time = time.time() - inference_start_time
        print(f"ERROR during inference after {inference.total_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup might be needed here even in single process mode
        if hasattr(inference, 'temp_storage') and inference.temp_storage is not None:
             print("Attempting cleanup of temporary storage after error...")
             inference.temp_storage.cleanup()
        raise


def main():
    """
    Main entry point: parses arguments, sets up multiprocessing/distributed environment.
    """
    import argparse
    import torch.multiprocessing as mp
    import torch
    import os
    import signal
    import sys

    # Set start method *before* parsing or anything else
    try:
         mp.set_start_method('spawn', force=True)
         print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
         print("Warning: Multiprocessing start method already set.")

    # Configure global thread settings
    try:
        num_cores = os.cpu_count()
        if num_cores:
            os.environ["NUMEXPR_MAX_THREADS"] = str(num_cores)
            os.environ["OMP_NUM_THREADS"] = str(num_cores)
            os.environ["MKL_NUM_THREADS"] = str(num_cores)
            print(f"Set NUMEXPR/OMP/MKL max threads to {num_cores}")
    except Exception as e:
        print(f"Warning: Could not set global thread limits: {e}")


    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run nnUNet inference using Volume class.')
    # Input/Output
    parser.add_argument('--input', type=str, required=False, help='Path or ID for VCDataset (path, scroll ID, segment ID). Required if not using --scroll_id/--segment_id.')
    parser.add_argument('--output', type=str, required=True, help='Path to output zarr file (must end with .zarr)')
    parser.add_argument('--input_format', type=str, default='zarr', choices=['zarr', 'volume'], help='Hint for input type if ambiguous (e.g., numeric path vs segment ID)')

    # Volume Identification (Alternative to --input for scroll/segment)
    parser.add_argument('--scroll_id', type=str, help='Scroll ID for Volume')
    parser.add_argument('--energy', type=int, help='Energy value for Volume')
    parser.add_argument('--resolution', type=float, help='Resolution value for Volume')
    parser.add_argument('--segment_id', type=int, help='Segment ID for Volume')

    # Volume Configuration
    parser.add_argument('--cache', action='store_true', default=True, help='Enable Volume/TensorStore caching (default: True)')
    parser.add_argument('--no_cache', action='store_false', dest='cache', help='Disable Volume/TensorStore caching')
    parser.add_argument('--no_volume_cache', action='store_true', help='Disable Volume caching (legacy option)')
    parser.add_argument('--cache_pool', type=int, default=1e10, help='TensorStore cache pool size in bytes (default: 1e10)')
    parser.add_argument('--normalization_scheme', type=str, default='instance_zscore', choices=['none', 'instance_zscore', 'global_zscore', 'instance_minmax'], help='Normalization scheme for Volume (default: instance_zscore)')
    parser.add_argument('--global_mean', type=float, help='Global mean for global_zscore')
    parser.add_argument('--global_std', type=float, help='Global std dev for global_zscore')
    parser.add_argument('--return_as_type', type=str, default='np.float32', help='Intermediate NumPy dtype in Volume before tensor conversion (default: np.float32)')
    parser.add_argument('--domain', type=str, choices=['dl.ash2txt', 'local'], help='Domain for Volume source (default: auto-detect)')
    parser.add_argument('--volume_domain', type=str, choices=['dl.ash2txt', 'local'], help='Domain for Volume source (legacy option, use --domain instead)')

    # Model Loading
    parser.add_argument('--model_folder', type=str, help='Path to local nnUNet model folder')
    parser.add_argument('--hf_model_path', type=str, help='Hugging Face model repository path')
    parser.add_argument('--hf_token', type=str, help='Optional Hugging Face token')
    parser.add_argument('--fold', type=str, default='0', help='Model fold (default: 0)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', help='Checkpoint name (default: checkpoint_final.pth)')

    # Inference Parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU (default: 4)')
    parser.add_argument('--step_size', type=float, default=0.5, help='Sliding window step size (fraction of patch size, default: 0.5)')
    parser.add_argument('--threshold', type=float, help='Threshold (0-100) for binarizing output')
    parser.add_argument('--no_probabilities', action='store_true', help='Save class labels instead of probability maps')
    parser.add_argument('--skip_blending', action='store_true', help='Only run inference, save patches to temp storage, skip blending')

    # TTA Parameters
    parser.add_argument('--no_tta', action='store_true', help='Completely disable Test Time Augmentation')
    parser.add_argument('--max_tta_combinations', type=int, default=3, help='Max TTA combinations (0 also disables TTA, default: 3)')
    parser.add_argument('--no_rotation_tta', action='store_true', help='Disable rotation TTA (mirroring TTA might still run if TTA is enabled)')
    parser.add_argument('--rotation_weights', type=float, nargs=3, help='Weights for rotation TTA axes [z, x, y]')

    # Execution/Resource Parameters
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs for distributed inference (default: 1)')
    parser.add_argument('--device', type=str, default='cuda', help='Primary device (used if num_gpus=1, default: cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers per process (default: 4)')
    parser.add_argument('--num_write_workers', type=int, default=4, help='Disk write workers per process (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo', 'mpi'], help='Distributed backend (default: nccl)')
    parser.add_argument('--dist_port', type=int, default=None, help='Master port for distributed init (default: auto-find)')
    parser.add_argument('--no_compile', action='store_true', help='Disable torch.compile optimization')

    # Partitioning Parameters
    parser.add_argument('--num_parts', type=int, default=1, help='Divide volume into N parts along Z-axis (default: 1)')
    parser.add_argument('--part_id', type=int, default=0, help='Process only part ID (0 to N-1, default: 0)')

    args = parser.parse_args()

    # --- Validate Arguments ---
    if not args.output.endswith('.zarr'):
        parser.error("Output path must end with '.zarr'")
    if args.input is None and args.scroll_id is None and args.segment_id is None:
         parser.error("Either --input, --scroll_id, or --segment_id must be provided.")
    if args.model_folder is None and args.hf_model_path is None:
        parser.error("Either --model_folder or --hf_model_path must be provided.")
    if args.normalization_scheme == 'global_zscore' and (args.global_mean is None or args.global_std is None):
        parser.error("--global_mean and --global_std are required when using normalization_scheme='global_zscore'")
    if not (0 <= args.part_id < args.num_parts):
         parser.error(f"--part_id ({args.part_id}) must be less than --num_parts ({args.num_parts}) and non-negative.")

    # Determine TTA configuration
    if args.no_tta or args.max_tta_combinations == 0:
        use_mirroring = False
        use_rotation_tta = False
        max_tta_combinations = 0
        if args.verbose: print("Test Time Augmentation explicitly DISABLED.")
    else:
        use_mirroring = True # Overall TTA flag
        use_rotation_tta = not args.no_rotation_tta
        max_tta_combinations = args.max_tta_combinations
        if args.verbose: print(f"TTA Enabled: Mirroring/Base={use_mirroring}, Rotation={use_rotation_tta}, Max Combs={max_tta_combinations}")


    # --- GPU/CPU Setup ---
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > 0 and available_gpus == 0:
        print("Warning: Requested GPU usage but no CUDA devices found. Running on CPU.")
        args.num_gpus = 0
        args.device = 'cpu'
        args.dist_backend = 'gloo' # NCCL requires CUDA

    if args.num_gpus <= 1:
        # --- Single Process Execution (CPU or 1 GPU) ---
        print(f"Starting inference in single-process mode on device '{args.device}'.")
        # Set CUDA_VISIBLE_DEVICES if using a specific GPU
        if args.device.startswith('cuda:'):
             gpu_id = args.device.split(':')[-1]
             os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
             print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
        elif args.device == 'cuda' and available_gpus > 0:
             os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Default to GPU 0
             print("Set CUDA_VISIBLE_DEVICES=0")

        # Run directly
        single_process_inference(args)

    else:
        # --- Multi-GPU Distributed Execution ---
        num_gpus_to_use = min(args.num_gpus, available_gpus)
        if num_gpus_to_use < args.num_gpus:
            print(f"Warning: Requested {args.num_gpus} GPUs, but only {available_gpus} available. Using {num_gpus_to_use}.")

        print(f"Starting {num_gpus_to_use} worker processes for distributed inference ({args.dist_backend})...")

        # Find free port if not specified
        port = args.dist_port
        if port is None:
             try:
                  port = find_free_port()
                  print(f"Auto-selected port {port} for distributed communication.")
             except RuntimeError as e:
                  print(f"Error finding free port: {e}. Please specify with --dist_port.")
                  sys.exit(1)

        # Set environment for master address/port
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port) # Set for spawned processes

        # Handle no_volume_cache for backward compatibility
        cache_setting = False if args.no_volume_cache else args.cache
        
        # Handle volume_domain for backward compatibility
        domain_setting = args.volume_domain if hasattr(args, 'volume_domain') and args.volume_domain is not None else args.domain
        
        # Prepare arguments for worker function
        worker_args = (
            num_gpus_to_use, # world_size passed to worker
            args.input, args.output,
            args.model_folder, args.hf_model_path, args.hf_token,
            args.fold, args.checkpoint,
            port, # Pass the selected port
            args.batch_size, args.step_size, args.input_format,
            args.num_workers, args.num_write_workers, args.verbose,
            use_mirroring, # Pass derived TTA flag
            max_tta_combinations, # Pass derived TTA combinations
            use_rotation_tta, # Pass derived rotation flag
            args.rotation_weights,
            not args.no_probabilities, # save_probability_maps
            args.skip_blending, args.threshold, args.dist_backend,
            args.num_parts, args.part_id,
            # Pass Volume args explicitly
            args.scroll_id, args.energy, args.resolution, args.segment_id,
            cache_setting, args.cache_pool, args.normalization_scheme,
            args.global_mean, args.global_std, args.return_as_type, domain_setting,
            args.no_compile # Pass the no_compile flag
        )

        # Add signal handling for graceful termination
        def signal_handler(sig, frame):
            print(f"Main process received signal {sig}, attempting graceful shutdown...")
            # How to terminate spawned processes gracefully is tricky.
            # Usually, the OS handles sending signals to the process group.
            # For now, just exit the main process. Child processes should detect parent exit or handle signals.
            sys.exit(1)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            mp.spawn(run_worker, args=worker_args, nprocs=num_gpus_to_use, join=True)
            print("All worker processes finished.")
        except Exception as e:
            print(f"Error during distributed execution: {e}")
            # Potentially add cleanup here if needed
            sys.exit(1)
        finally:
             # Clean up environment variables if needed (optional)
             # del os.environ['MASTER_ADDR']
             # del os.environ['MASTER_PORT']
             pass

if __name__ == "__main__":
    main()