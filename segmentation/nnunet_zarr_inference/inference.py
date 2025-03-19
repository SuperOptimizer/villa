import os
import argparse
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
from typing import Dict, Tuple, List, Optional, Any, Union
import torch.distributed as dist
from collections import defaultdict

# Import nnUNet model loader (with fallback for different import scenarios)
try:
    # First try relative imports (when running as a module)
    from nnunet_zarr_inference.load_nnunet_model import load_model, run_inference
    from nnunet_zarr_inference.inference_dataset import InferenceDataset
except ImportError:
    # Fallback for direct script execution
    from load_nnunet_model import load_model, run_inference
    from inference_dataset import InferenceDataset

class ZarrNNUNetInferenceHandler:
    def __init__(self, 
                 input_path: str,
                 output_path: str,
                 model_folder: str,
                 fold: Union[int, str] = 0,
                 checkpoint_name: str = 'checkpoint_final.pth',
                 patch_size: Optional[Tuple[int, int, int]] = None,
                 batch_size: int = 4,
                 step_size: float = 0.5,
                 num_dataloader_workers: int = 4,
                 num_write_workers: int = 4,
                 input_format: str = 'zarr',
                 load_all: bool = False,
                 write_layers: bool = False,
                 postprocess_only: bool = False,
                 device: str = 'cuda',
                 threshold: Optional[float] = None,
                 use_mirroring: bool = True,
                 max_tta_combinations: Optional[int] = None,
                 full_tta: bool = False,  # New flag for using all TTA combinations
                 verbose: bool = False,
                 keep_intermediates: bool = False,
                 save_probability_maps: bool = True,
                 output_targets: Optional[Dict[str, Dict]] = None):
        """
        Initialize the inference handler for nnUNet models on zarr arrays.
        
        Args:
            input_path: Path to the input zarr store
            output_path: Path to save the output zarr store
            model_folder: Path to the nnUNet model folder
            fold: Which fold to load (default: 0, can also be 'all')
            checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
            patch_size: Optional override for the patch size
            batch_size: Batch size for inference
            step_size: Step size for sliding window prediction as a fraction of patch_size (default: 0.5, nnUNet default)
            num_dataloader_workers: Number of workers for the DataLoader
            num_write_workers: Number of worker threads for asynchronous disk writes
            input_format: Format of the input data ('zarr' supported currently)
            load_all: Whether to load the entire array into memory
            write_layers: Whether to write the sliced z layers to disk
            postprocess_only: Skip the inference pass and only do final averaging + casting
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Optional threshold value (0-100) for binarizing the probability map
            use_mirroring: Enable test time augmentation via mirroring (default: True, matches nnUNet default)
            max_tta_combinations: Maximum number of TTA combinations to use (default: None = auto-detect based on GPU memory)
            full_tta: Whether to use all possible TTA combinations regardless of GPU memory (default: False)
            verbose: Enable detailed output messages during inference (default: False)
            keep_intermediates: Keep intermediate sum and count arrays after processing (default: False)
            save_probability_maps: Save full probability maps for multiclass segmentation (default: True, set to False to save space)
            output_targets: Optional custom output targets configuration
        """
        self.input_path = input_path
        self.output_path = output_path
        self.model_folder = model_folder
        self.fold = fold
        self.checkpoint_name = checkpoint_name
        self.batch_size = batch_size
        self.tile_step_size = step_size  # Using nnUNet's naming convention
        self.num_dataloader_workers = num_dataloader_workers
        self.input_format = input_format
        self.load_all = load_all
        self.postprocess_only = postprocess_only
        self.write_layers = write_layers
        self.device_str = device
        self.threshold = threshold
        self.use_mirroring = use_mirroring
        self.max_tta_combinations = max_tta_combinations
        self.full_tta = full_tta
        self.verbose = verbose
        self.keep_intermediates = keep_intermediates
        self.save_probability_maps = save_probability_maps
        
        # Default output target configuration if not provided
        # For nnUNet binary segmentation, the model outputs 2 channels (background, foreground)
        # But we'll typically only save the foreground channel (index 1) for binary segmentation
        self.output_targets = output_targets or {
            "segmentation": {
                "channels": 1,  # We'll save 1 channel for binary segmentation (foreground only)
                "activation": "softmax",
                "nnunet_output_channels": 2  # nnUNet outputs 2 channels for binary (bg, fg)
            }
        }
        
        # nnUNet binary segmentation configuration
        self.nnunet_foreground_channel = 1  # Second channel (index 1) is foreground in binary segmentation
        
        if self.verbose:
            print(f"Initialized with step_size={self.tile_step_size}")
            for tgt_name, tgt_conf in self.output_targets.items():
                print(f"Output target '{tgt_name}': {tgt_conf}")
        
        # Buffer to accumulate patches before writing
        self.patch_buffer = defaultdict(list)
        self.buffer_positions = []
        self.buffer_size = 32  # Number of patches to accumulate before writing

        # Initialize blend weights as None - will be created when needed
        self.blend_weights = None
        self.blend_weights_4d = None

        # Determine rank for DDP
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # Create a ThreadPoolExecutor for asynchronous writing (if not in postprocess-only mode)
        self.executor = None
        self.write_futures = []
        if not self.postprocess_only:
            self.executor = ThreadPoolExecutor(max_workers=num_write_workers)

        # Limit the number of pending writes to avoid unbounded memory usage
        self.max_pending_writes = num_write_workers * 4
        
        # A lock to protect the read–modify–write update used for blending
        self.write_lock = threading.Lock()
        
        # Load the nnUNet model
        self.model_info = None
        self.patch_size = patch_size  # This will be overridden if None and model is loaded
    
    def _load_nnunet_model(self):
        """
        Load the nnUNet model and return model information.
        """
        try:
            print(f"Loading nnUNet model from {self.model_folder}, fold {self.fold}")
            if self.verbose:
                print(f"Test time augmentation (mirroring): {'enabled' if self.use_mirroring else 'disabled'}")
            model_info = load_model(
                model_folder=self.model_folder,
                fold=self.fold,
                checkpoint_name=self.checkpoint_name,
                device=self.device_str,
                use_mirroring=self.use_mirroring,
                verbose=self.verbose
            )
            
            # Use the model's patch size if none was specified
            if self.patch_size is None:
                self.patch_size = model_info['patch_size']
                if self.verbose:
                    print(f"Using model's patch size: {self.patch_size}")
            
            return model_info
        except Exception as e:
            print(f"Error loading nnUNet model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_blend_weights(self):
        """
        Create a 3D Gaussian window to be used as blending weights, exactly following nnUNet's approach.

        If tile_step_size is 1.0 (no overlap), then simply return an array of ones.
        Otherwise, create a 3D Gaussian centered at the center of the patch.
        
        This implementation directly follows nnUNet's compute_gaussian function in sliding_window_prediction.py.
        """
        if self.tile_step_size == 1.0:
            return np.ones(self.patch_size, dtype=np.float32)

        # Create a temporary zero array and set the center to 1
        tmp = np.zeros(self.patch_size, dtype=np.float32)
        center_coords = [i // 2 for i in self.patch_size]
        tmp[tuple(center_coords)] = 1
        
        # Print debugging info about the gaussian creation
        if self.verbose:
            print(f"Creating Gaussian blend weights:")
            print(f"  - Patch size: {self.patch_size}")
            print(f"  - Center coords: {center_coords}")
            print(f"  - Temp array shape: {tmp.shape}")
        
        # nnUNet uses sigma_scale = 1/8 of patch size
        # We'll go back to using the exact nnUNet approach for consistency
        sigma_scale = 1.0 / 8.0  # Match nnUNet's original approach
        sigmas = [i * sigma_scale for i in self.patch_size]
        
        if self.verbose:
            print(f"  - Using sigma_scale: {sigma_scale}")
            print(f"  - Sigmas: {sigmas}")
        
        # Apply Gaussian filter
        from scipy.ndimage import gaussian_filter
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        
        if self.verbose:
            print(f"  - Gaussian map shape: {gaussian_importance_map.shape}")
            print(f"  - Gaussian min/max: {np.min(gaussian_importance_map):.6f}, {np.max(gaussian_importance_map):.6f}")
        
        # Value scaling factor (higher values will create a more pronounced weighting)
        # nnUNet uses 10.0 - we must use exactly the same
        value_scaling_factor = 0.5  # Set to 1.0 so sum of weights equals ~1.0 at overlaps
        
        # Normalize the Gaussian map - simpler, more reliable approach
        # Scale by value_scaling_factor but preserve the Gaussian shape
        max_value = np.max(gaussian_importance_map)
        gaussian_importance_map = gaussian_importance_map / max_value  # First normalize to [0, 1]
        gaussian_importance_map = gaussian_importance_map * value_scaling_factor  # Then scale
        
        # Ensure no zero values to avoid division by zero later
        mask = gaussian_importance_map == 0
        if np.any(mask):
            gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
            
        if self.verbose:
            print(f"Created Gaussian blend weights with shape {gaussian_importance_map.shape}")
            print(f"  - min: {np.min(gaussian_importance_map):.4f}, max: {np.max(gaussian_importance_map):.4f}")
            print(f"  - value_scaling_factor: {value_scaling_factor}")
            
            # Print central slice to show the Gaussian weight distribution
            if len(self.patch_size) == 3:
                central_z = self.patch_size[0] // 2
                central_slice = gaussian_importance_map[central_z]
                
                # Print a simplified representation of weights for central slice
                central_y = self.patch_size[1] // 2
                row = central_slice[central_y, :]
                print(f"  - Center row weights: {row.min():.3f} → {row.max():.3f}")
                print(f"  - Center point weight: {gaussian_importance_map[central_z, central_y, self.patch_size[2]//2]:.3f}")
                
                # Calculate weight values at various distances from center
                center_val = gaussian_importance_map[central_z, central_y, self.patch_size[2]//2]
                quarter_x = self.patch_size[2] // 4
                half_x = self.patch_size[2] // 2 + (self.patch_size[2] // 4)  # 3/4 of the way to the edge
                quarter_val = gaussian_importance_map[central_z, central_y, quarter_x] 
                half_val = gaussian_importance_map[central_z, central_y, half_x] if half_x < self.patch_size[2] else 0
                edge_val = gaussian_importance_map[central_z, central_y, 0]
                
                print(f"  - Weight falloff from center to edge:")
                print(f"    * center (100%): {center_val:.3f}")
                print(f"    * 1/4 to edge ({quarter_val/center_val*100:.1f}%): {quarter_val:.3f}")
                print(f"    * halfway ({half_val/center_val*100:.1f}%): {half_val:.3f}")
                print(f"    * edge ({edge_val/center_val*100:.1f}%): {edge_val:.3f}")
                
                # Check weights at patch overlap points (for 50% overlap)
                # With 50% overlap, patches overlap starting at patch_size/2
                patch_edge = self.patch_size[2] - 1  # Last index
                overlap_start = self.patch_size[2] // 2  # Where 50% overlap begins
                edge_weight = gaussian_importance_map[central_z, central_y, 0]  # Weight at patch edge
                overlap_weight = gaussian_importance_map[central_z, central_y, overlap_start]  # Weight at overlap point
                
                print(f"    * weight at patch edge: {edge_weight:.3f} ({edge_weight/center_val*100:.1f}% of center)")
                print(f"    * weight at overlap start: {overlap_weight:.3f} ({overlap_weight/center_val*100:.1f}% of center)")
                
                # Check the ratio of weights at the overlap point - ideally should sum to ~1.0
                # For position p = patch_size/2, we want w(p) + w(patch_size-p) ≈ 1.0
                opposite_weight = gaussian_importance_map[central_z, central_y, patch_edge - overlap_start]
                weight_sum = overlap_weight + opposite_weight
                print(f"    * sum of weights at overlap: {overlap_weight:.3f} + {opposite_weight:.3f} = {weight_sum:.3f}")
                print(f"      (ideally should be close to 1.0 for smooth blending)")
            
        return gaussian_importance_map

    def _compute_steps_for_sliding_window(self, image_size, patch_size, step_size):
        """
        Compute the positions for sliding window patches with specified step size.
        This is based on nnUNet's compute_steps_for_sliding_window function.
        
        Args:
            image_size: size of the whole image (Z, Y, X)
            patch_size: size of the patches (Z, Y, X)
            step_size: step size as a fraction of patch_size (0 < step_size <= 1)
            
        Returns:
            List of steps for each dimension
        """
        assert all(i >= j for i, j in zip(image_size, patch_size)), "image size must be larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
        
        # Calculate step sizes in voxels
        target_step_sizes_in_voxels = [int(i * step_size) for i in patch_size]
        
        # Calculate number of steps for each dimension
        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]
        
        # Calculate actual steps for each dimension
        steps = []
        for dim in range(len(patch_size)):
            # The highest step value for this dimension
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # Only one step at position 0
                
            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
            steps.append(steps_here)
            
        return steps
        
    def _initialize_blend_weights(self):
        """
        Initialize blend weights for the current patch size.
        
        This creates two versions of the weights:
        1. blend_weights: standard 3D Gaussian weights for 3D data
        2. blend_weights_4d: same Gaussian but with a singleton channel dimension for 4D data
        
        The two versions enable proper broadcasting during weighted blending
        regardless of whether the data has a channel dimension.
        """
        if len(self.patch_size) == 3:  # 3D patches
            if self.verbose:
                print(f"Initializing Gaussian blend weights for patch size {self.patch_size}")
                
            # Create the 3D Gaussian blend weights
            self.blend_weights = self._create_blend_weights()
            
            # Create 4D version by adding a singleton channel dimension
            # This allows broadcasting to work correctly with multi-channel data
            self.blend_weights_4d = np.expand_dims(self.blend_weights, axis=0)
            
            if self.verbose:
                print(f"Blend weights shape: 3D={self.blend_weights.shape}, 4D={self.blend_weights_4d.shape}")
                print(f"Min/max values: {np.min(self.blend_weights)}, {np.max(self.blend_weights)}")

    def _process_buffer(self, output_arrays: Dict, count_arrays: Dict,
                        patch_buffer: Dict = None, positions: List[Tuple] = None):
        """
        Write accumulated patches to the Zarr arrays, grouping spatially close patches
        to minimize the number of small I/O operations and improve write throughput.
        """
        # Use the provided buffers if any; otherwise, use the instance buffers.
        if patch_buffer is None or positions is None:
            patch_buffer = self.patch_buffer
            positions = self.buffer_positions

        if not patch_buffer or not positions:
            return

        # Group patches by target
        for tgt_name in self.output_targets:
            if tgt_name not in patch_buffer:
                continue
                
            patches_for_target = [patch_buffer[tgt_name][i] for i in range(len(positions))]
            
            # Process all patches for this target at once
            # This reduces the number of zarr array accesses and lock contentions
            self._write_region_patches(patches_for_target, positions,
                                      output_arrays[tgt_name],
                                      count_arrays[tgt_name])

        # If these are the "live" buffers on the instance, clear them.
        if patch_buffer is self.patch_buffer:
            self.patch_buffer.clear()
            self.buffer_positions.clear()

    def _write_region_patches(self, patches: List[np.ndarray], positions: List[Tuple],
                              output_array: zarr.Array, count_array: zarr.Array):
        """Write multiple patches with zero–padding handled.

        For patches that have been zero–padded to reach the full patch size,
        we compute the valid region (i.e. the part that actually falls within the image)
        and only blend that region back into the output arrays.
        
        This optimized version groups spatially close patches to minimize zarr read/write operations.
        """
        # Skip if no patches
        if not patches or not positions:
            return
            
        # Ensure blending weights are initialized outside the lock
        if self.blend_weights is None:
            self._initialize_blend_weights()

        # Get the full image shape from the count array.
        image_z, image_y, image_x = count_array.shape
        full_z, full_y, full_x = self.patch_size

        # Group patches by spatial proximity to minimize region reads/writes
        # First, compute all valid sizes outside the lock
        valid_sizes = []
        for pos in positions:
            z0, y0, x0 = pos
            valid_z = full_z if (z0 + full_z) <= image_z else image_z - z0
            valid_y = full_y if (y0 + full_y) <= image_y else image_y - y0
            valid_x = full_x if (x0 + full_x) <= image_x else image_x - x0
            valid_sizes.append((valid_z, valid_y, valid_x))

        # Compute the bounding box for all patches
        min_z = min(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        min_x = min(pos[2] for pos in positions)
        max_z = max(pos[0] + valid[0] for pos, valid in zip(positions, valid_sizes))
        max_y = max(pos[1] + valid[1] for pos, valid in zip(positions, valid_sizes))
        max_x = max(pos[2] + valid[2] for pos, valid in zip(positions, valid_sizes))

        # Check if bounding box is too large, if so, break it into chunks
        z_range = max_z - min_z
        y_range = max_y - min_y
        x_range = max_x - min_x
        
        # Define max region size (larger regions = fewer I/O operations but more memory)
        # These are chosen to balance memory usage and I/O operations
        max_z_size = full_z * 4  # 4x patch size in Z
        max_y_size = full_y * 4  # 4x patch size in Y
        max_x_size = full_x * 4  # 4x patch size in X
        
        # If region is manageable, process it as a single unit; otherwise chunk it
        # This helps with very large buffers where we have patches from disparate regions
        if z_range <= max_z_size and y_range <= max_y_size and x_range <= max_x_size:
            self._process_region(
                patches, positions, valid_sizes, 
                min_z, min_y, min_x, max_z, max_y, max_x,
                output_array, count_array
            )
        else:
            # Region too large, group patches into smaller chunks
            if self.verbose:
                print(f"Large region detected ({z_range}x{y_range}x{x_range}), chunking for efficient processing")
            
            # Group patches by Z chunks first (most optimal for most volume data)
            z_chunks = range(min_z, max_z, max_z_size)
            for z_start in z_chunks:
                z_end = min(z_start + max_z_size, max_z)
                
                # Find patches that fall within this Z chunk
                z_chunk_indices = [
                    i for i, pos in enumerate(positions) 
                    if (pos[0] < z_end and (pos[0] + valid_sizes[i][0]) > z_start)
                ]
                
                if not z_chunk_indices:
                    continue
                    
                # Process this chunk
                chunk_patches = [patches[i] for i in z_chunk_indices]
                chunk_positions = [positions[i] for i in z_chunk_indices]
                chunk_valid_sizes = [valid_sizes[i] for i in z_chunk_indices]
                
                # Find bounds within this chunk
                chunk_min_z = max(min(pos[0] for pos in chunk_positions), z_start)
                chunk_min_y = min(pos[1] for pos in chunk_positions)
                chunk_min_x = min(pos[2] for pos in chunk_positions)
                chunk_max_z = min(max(pos[0] + vs[0] for pos, vs in zip(chunk_positions, chunk_valid_sizes)), z_end)
                chunk_max_y = max(pos[1] + vs[1] for pos, vs in zip(chunk_positions, chunk_valid_sizes))
                chunk_max_x = max(pos[2] + vs[2] for pos, vs in zip(chunk_positions, chunk_valid_sizes))
                
                self._process_region(
                    chunk_patches, chunk_positions, chunk_valid_sizes,
                    chunk_min_z, chunk_min_y, chunk_min_x, 
                    chunk_max_z, chunk_max_y, chunk_max_x,
                    output_array, count_array
                )
    
    def _process_region(self, patches, positions, valid_sizes,
                       min_z, min_y, min_x, max_z, max_y, max_x,
                       output_array, count_array):
        """
        Process a single region of patches with one read-modify-write cycle.
        This implementation follows nnUNet's approach in predict_from_raw_data.py.
        """
        if self.verbose:
            # Print detailed shape information to debug dimension issues
            batch_size = len(patches)
            print(f"\n--- PROCESS REGION ---")
            print(f"Region: z={min_z}-{max_z}, y={min_y}-{max_y}, x={min_x}-{max_x}")
            print(f"Processing {batch_size} patches")
            if batch_size > 0:
                print(f"First patch shape: {patches[0].shape}")
                print(f"Blend weights shape: {self.blend_weights.shape}")
                print(f"Blend weights 4D shape: {self.blend_weights_4d.shape}")
                print(f"Output array shape: {output_array.shape}")
                print(f"Count array shape: {count_array.shape}")
            
        with self.write_lock:
            # Check output_array dimensionality and adapt access pattern accordingly
            output_ndim = len(output_array.shape)
            
            if self.verbose:
                print(f"Output array dimensionality: {output_ndim}")
            
            # Determine if we have 3D or 4D data
            has_channel_dim = len(patches[0].shape) == 4
            
            if self.verbose:
                print(f"Processing data with has_channel_dim={has_channel_dim}, output_ndim={output_ndim}")
                print(f"First patch shape: {patches[0].shape}")
            
            # CRITICAL FIX: Always use the correct weights based on data dimensionality
            # Use 4D weights (with channel dim) when patch has channel, 3D weights when it doesn't
            base_weights = self.blend_weights_4d if has_channel_dim else self.blend_weights
            
            if self.verbose:
                print(f"Using base weights with shape: {base_weights.shape}, ndim={base_weights.ndim}")
            
            # Fetch the current region (single I/O operation)
            if output_ndim == 4 and has_channel_dim:
                # Both have 4D: C,Z,Y,X
                region_sum = output_array[:, min_z:max_z, min_y:max_y, min_x:max_x]
                if self.verbose:
                    print(f"4D output + 4D patch case")
            elif output_ndim == 3 and has_channel_dim:
                # This shouldn't happen anymore after our fixes
                print(f"ERROR: Dimension mismatch detected:")
                print(f"- Output array: shape={output_array.shape}, ndim={output_ndim}")
                print(f"- Patch data: shape={patches[0].shape}, ndim={len(patches[0].shape)}")
                print(f"- Channel flag: has_channel_dim={has_channel_dim}")
                
                # Attempt to recover by reducing patch dimension
                tmp_patches = [p[0:1] for p in patches]  # Keep only first channel
                patches = tmp_patches  # Replace patches with reduced versions
                region_sum = output_array[min_z:max_z, min_y:max_y, min_x:max_x]
                base_weights = self.blend_weights  # Switch to 3D weights
                has_channel_dim = False  # Reset flag since we've modified patches
            elif output_ndim == 3 and not has_channel_dim:
                # Both are 3D: Z,Y,X
                region_sum = output_array[min_z:max_z, min_y:max_y, min_x:max_x]
                if self.verbose:
                    print(f"3D output + 3D patch case")
            else:
                # Unusual case - shouldn't happen
                raise ValueError(f"Incompatible dimensions: output array has {output_ndim} dimensions " 
                               f"but patch data has {len(patches[0].shape)} dimensions")

            region_count = count_array[min_z:max_z, min_y:max_y, min_x:max_x]

            # Accumulate all patches into the region
            for patch, pos, valid in zip(patches, positions, valid_sizes):
                valid_z, valid_y, valid_x = valid
                z0, y0, x0 = pos
                z_rel = z0 - min_z
                y_rel = y0 - min_y
                x_rel = x0 - min_x

                # Skip if patch is completely outside the region
                if (z_rel >= max_z - min_z) or (y_rel >= max_y - min_y) or (x_rel >= max_x - min_x) or \
                   (z_rel + valid_z <= 0) or (y_rel + valid_y <= 0) or (x_rel + valid_x <= 0):
                    continue

                # Adjust relative coordinates if they are negative (patch partially outside region)
                if z_rel < 0:
                    valid_z += z_rel
                    patch = patch[:, -z_rel:, :, :] if has_channel_dim else patch[-z_rel:, :, :]
                    z_rel = 0
                if y_rel < 0:
                    valid_y += y_rel
                    patch = patch[:, :, -y_rel:, :] if has_channel_dim else patch[:, -y_rel:, :]
                    y_rel = 0
                if x_rel < 0:
                    valid_x += x_rel
                    patch = patch[:, :, :, -x_rel:] if has_channel_dim else patch[:, :, -x_rel:]
                    x_rel = 0

                # Adjust valid region if it extends beyond region bounds
                if z_rel + valid_z > max_z - min_z:
                    valid_z = max_z - min_z - z_rel
                if y_rel + valid_y > max_y - min_y:
                    valid_y = max_y - min_y - y_rel
                if x_rel + valid_x > max_x - min_x:
                    valid_x = max_x - min_x - x_rel

                # Crop the patch and blending weights to the valid region.
                if self.verbose:
                    print(f"\n--- PATCH PROCESSING ---")
                    print(f"Patch shape before cropping: {patch.shape}")
                    print(f"Valid region: z={valid_z}, y={valid_y}, x={valid_x}")
                    print(f"has_channel_dim: {has_channel_dim}")
                
                if has_channel_dim:
                    patch_valid = patch[:, :valid_z, :valid_y, :valid_x]
                    
                    if self.verbose:
                        print(f"Patch shape after cropping: {patch_valid.shape}")
                    
                    # Get the appropriate weights slice - need to handle both 3D and 4D weights consistently
                    if self.verbose:
                        print(f"Slicing base_weights with shape {base_weights.shape}")
                        
                    # Handle different dimensionality of base_weights
                    if base_weights.ndim == 4:  # 4D weights (with channel dim)
                        weights_slice = base_weights[:, :valid_z, :valid_y, :valid_x]
                        if self.verbose:
                            print(f"Sliced 4D weights -> shape {weights_slice.shape}")
                    else:  # 3D weights (no channel dim)
                        weights_slice = base_weights[:valid_z, :valid_y, :valid_x]
                        if self.verbose:
                            print(f"Sliced 3D weights -> shape {weights_slice.shape}")
                    
                    if self.verbose:
                        print(f"Weights slice shape before expansion: {weights_slice.shape}")
                    
                    # nnUNet applies weights per-channel by broadcasting
                    # For 4D data, we need to ensure the weights have the right shape for broadcasting
                    if patch_valid.ndim == 4:
                        # Get the exact expected shape from the patch
                        expected_shape = (1,) + patch_valid.shape[1:]
                        
                        if self.verbose:
                            print(f"Patch shape is {patch_valid.shape}, expected weights shape is {expected_shape}")
                        
                        # We need to handle any mismatch in dimensions directly
                        # This is simplest and most reliable approach
                        if weights_slice.shape != expected_shape:
                            if self.verbose:
                                print(f"Weights shape {weights_slice.shape} doesn't match expected {expected_shape}")
                                print(f"Creating new weights with exact patch shape...")
                            
                            # Check if we even have the right dimensionality
                            if weights_slice.ndim != len(expected_shape):
                                # Add or remove dimensions as needed
                                if weights_slice.ndim < len(expected_shape):
                                    # Add dimensions (expand_dims) to match
                                    while weights_slice.ndim < len(expected_shape):
                                        weights_slice = np.expand_dims(weights_slice, axis=0)
                                        if self.verbose:
                                            print(f"Added dimension, now: {weights_slice.shape}")
                                else:
                                    # Remove dimensions (squeeze) to match
                                    while weights_slice.ndim > len(expected_shape):
                                        weights_slice = np.squeeze(weights_slice, axis=0)
                                        if self.verbose:
                                            print(f"Removed dimension, now: {weights_slice.shape}")
                            
                            # Check if the dimensions still don't match exactly
                            if weights_slice.shape != expected_shape:
                                if self.verbose:
                                    print(f"Need to resize weights from {weights_slice.shape} to {expected_shape}")
                                
                                # The most reliable approach - create a new array with the right shape
                                # and sample the correct values from the weights_slice
                                new_weights = np.zeros(expected_shape, dtype=weights_slice.dtype)
                                
                                # Determine valid region to copy (min of each dimension)
                                copy_shape = tuple(min(s1, s2) for s1, s2 in zip(weights_slice.shape, expected_shape))
                                
                                if self.verbose:
                                    print(f"Copy shape for valid region: {copy_shape}")
                                
                                # Create slices for copying
                                src_slices = tuple(slice(0, s) for s in copy_shape)
                                dst_slices = tuple(slice(0, s) for s in copy_shape)
                                
                                # Copy valid region 
                                new_weights[dst_slices] = weights_slice[src_slices]
                                
                                # Replace the weights with our correctly sized version
                                weights_slice = new_weights
                                
                                if self.verbose:
                                    print(f"New weights created with shape: {weights_slice.shape}")
                        else:
                            if self.verbose:
                                print(f"Weights shape already matches expected: {weights_slice.shape}")
                    
                    # Apply weights (this is exactly how nnUNet does it)
                    weighted_patch = patch_valid * weights_slice
                    
                    if self.verbose:
                        print(f"Weighted patch shape: {weighted_patch.shape}")
                    
                else:
                    patch_valid = patch[:valid_z, :valid_y, :valid_x]
                    weights_slice = base_weights[:valid_z, :valid_y, :valid_x]
                    weighted_patch = patch_valid * weights_slice
                    
                    if self.verbose:
                        print(f"3D case - Weighted patch shape: {weighted_patch.shape}")

                # Update region with weighted patch - this follows nnUNet's approach:
                # 1. Apply Gaussian weights to the prediction (prediction *= gaussian)
                # 2. Add to accumulated predictions (predicted_logits[sl] += prediction)
                # 3. Add Gaussian weights to count array (n_predictions[sl[1:]] += gaussian)
                
                if output_ndim == 4 and has_channel_dim:
                    # Handle potential channel count mismatches and dimension mismatches
                    if self.verbose:
                        print(f"\n--- REGION UPDATE ---")
                        print(f"Updating region with weighted patch")
                        print(f"Weighted patch shape: {weighted_patch.shape}")
                        target_region_shape = region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x].shape
                        print(f"Target region shape: {target_region_shape}")
                        print(f"Relative coords: z_rel={z_rel}, y_rel={y_rel}, x_rel={x_rel}")
                        print(f"Valid sizes: z={valid_z}, y={valid_y}, x={valid_x}")
                    
                    try:
                        # First check for channel count mismatches (multiclass segmentation case)
                        if weighted_patch.shape[0] != region_sum.shape[0]:
                            # Special case - channel counts don't match
                            if self.verbose:
                                print(f"WARNING: Channel count mismatch during blending:")
                                print(f"- Patch channels: {weighted_patch.shape[0]}")
                                print(f"- Region channels: {region_sum.shape[0]}")
                                
                            # Match dimensions by padding or truncating the patch
                            if weighted_patch.shape[0] < region_sum.shape[0]:
                                # Need to pad the patch with zeros for missing channels
                                if self.verbose:
                                    print(f"Padding patch to match channel count")
                                temp = np.zeros((region_sum.shape[0], valid_z, valid_y, valid_x), dtype=weighted_patch.dtype)
                                temp[:weighted_patch.shape[0]] = weighted_patch
                                if self.verbose:
                                    print(f"Padded temp shape: {temp.shape}")
                                region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += temp
                            else:
                                # Need to truncate the patch to first N channels
                                if self.verbose:
                                    print(f"Truncating patch to first {region_sum.shape[0]} channels")
                                region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch[:region_sum.shape[0]]
                        else:
                            # Normal case - channels match exactly
                            # Handle direct addition with proper broadcasting
                            target_region = region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x]
                            
                            if self.verbose:
                                print(f"Direct channel match case:")
                                print(f"- Weighted patch shape: {weighted_patch.shape}, ndim={weighted_patch.ndim}")
                                print(f"- Target region shape: {target_region.shape}, ndim={target_region.ndim}")
                            
                            # If shapes don't match exactly, reshape for proper broadcasting
                            if weighted_patch.ndim != target_region.ndim:
                                if self.verbose:
                                    print(f"Reshaping weighted_patch from {weighted_patch.shape} to match region dimensions {target_region.ndim}")
                                
                                # Let's replace the complex reshaping with a more direct approach
                                # The issue is likely with how the extra dimension gets added
                                
                                if weighted_patch.ndim == 4 and target_region.ndim == 5:
                                    # This is the problematic case - we need to match a 5D target with a 4D patch
                                    if self.verbose:
                                        print(f"Special case: 4D patch to 5D target")
                                        
                                    # Get the exact shape of the target region
                                    target_shape = target_region.shape
                                    if self.verbose:
                                        print(f"Target shape dimensions: {target_shape}")
                                        
                                    # Create a temporary array with the exact target shape
                                    # This avoids broadcasting issues
                                    temp_weighted = np.zeros(target_shape, dtype=weighted_patch.dtype)
                                    
                                    # Copy the weighted patch into the temp array
                                    # This handles the shape mismatch explicitly
                                    temp_weighted[0] = weighted_patch
                                    
                                    if self.verbose:
                                        print(f"Created temp array with shape: {temp_weighted.shape}")
                                    
                                    # Now add the temp array with the exact matching shape
                                    region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += temp_weighted
                                    
                                else:
                                    # Old approach for other cases
                                    if weighted_patch.ndim > target_region.ndim:
                                        # Squeeze out extra dimensions
                                        if self.verbose:
                                            print(f"Squeezing down from {weighted_patch.ndim} to {target_region.ndim} dimensions")
                                        while weighted_patch.ndim > target_region.ndim:
                                            weighted_patch = np.squeeze(weighted_patch, axis=0)
                                            if self.verbose:
                                                print(f"After squeeze: {weighted_patch.shape}")
                                    else:
                                        # Add dimensions as needed
                                        if self.verbose:
                                            print(f"Expanding from {weighted_patch.ndim} to {target_region.ndim} dimensions")
                                        while weighted_patch.ndim < target_region.ndim:
                                            weighted_patch = np.expand_dims(weighted_patch, axis=0)
                                            if self.verbose:
                                                print(f"After expand: {weighted_patch.shape}")
                                    
                                    # Now do the addition with the reshaped patch
                                    region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                            else:
                                # Shapes match in dimensions, do direct addition
                                if self.verbose:
                                    print(f"Dimensions match exactly - direct addition")
                                region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                            
                    except Exception as e:
                        # Print detailed error with shapes to help diagnose the issue
                        print(f"ERROR during patch addition: {str(e)}")
                        print(f"- Weighted patch: shape={weighted_patch.shape}, dtype={weighted_patch.dtype}, ndim={weighted_patch.ndim}")
                        print(f"- Region sum total shape: {region_sum.shape}")
                        print(f"- Target slice: z_rel={z_rel}, y_rel={y_rel}, x_rel={x_rel}, valid_z={valid_z}, valid_y={valid_y}, valid_x={valid_x}")
                        print(f"- Target region shape: {region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x].shape}")
                        # Rethrow the exception
                        raise
                elif output_ndim == 3 and has_channel_dim:
                    # This path should not be reached due to earlier fix
                    region_sum[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch[0]
                else:
                    region_sum[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                
                # Update count array with the weights - exactly as nnUNet does
                # In nnUNet: n_predictions[sl[1:]] += gaussian
                # The key is that the count array doesn't have channel dimension in nnUNet,
                # so we need to ensure our weights don't have channel dimension when updating count
                if weights_slice.ndim == 4:
                    # For 4D weights, use the first channel for count (removes channel dimension)
                    region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weights_slice[0]
                else:
                    # For 3D weights, use directly
                    region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weights_slice

            # Write back the updated regions - single I/O operation
            if output_ndim == 4 and has_channel_dim:
                output_array[:, min_z:max_z, min_y:max_y, min_x:max_x] = region_sum
            else:
                output_array[min_z:max_z, min_y:max_y, min_x:max_x] = region_sum
            count_array[min_z:max_z, min_y:max_y, min_x:max_x] = region_count

    def _process_model_outputs(self, outputs, positions: List[Tuple],
                               output_arrays: Dict, count_arrays: Dict):
        """
        Process model outputs with buffering and offload writing asynchronously when full.
        This function handles both single tensor outputs (standard nnUNet) and dictionary outputs.
        """
        # For nnUNet, the output is a tensor, not a dict - convert it to match target format
        if torch.is_tensor(outputs):
            # Convert single tensor output to dict based on output_targets
            processed_outputs = {}
            for tgt_name in self.output_targets:
                processed_outputs[tgt_name] = outputs
        else:
            # Already in dict format
            processed_outputs = outputs
            
        # Process each patch and add to buffer
        for i, pos in enumerate(positions):
            for tgt_name in self.output_targets:
                # Get the prediction tensor for this target
                pred = processed_outputs[tgt_name][i].cpu().numpy()
                
                # Apply activation based on the target config
                activation_str = self.output_targets[tgt_name].get("activation", "none").lower()
                # Note: We've already applied activations in the inference loop
                
                # We're already handling the channel selection upstream
                # (in the previous step where we extracted just the foreground channel)
                # Here we just verify the shape is as expected
                num_channels = pred.shape[0]
                
                # Just a sanity check to ensure we have the expected number of channels
                expected_channels = self.output_targets[tgt_name]["channels"]
                
                if self.verbose and num_channels != expected_channels:
                    # This can happen after the first batch when we've detected multiclass
                    # and updated the output_targets, but have not yet updated the local variable
                    print(f"Note: Channel count for {tgt_name}: expected={expected_channels}, actual={num_channels}")
                    print(f"This is normal when processing multiclass segmentation models")
                
                self.patch_buffer[tgt_name].append(pred)
            self.buffer_positions.append(pos)

        # When buffer is full, process it asynchronously
        if len(self.buffer_positions) >= self.buffer_size:
            # Check if any futures are done without blocking
            # This helps keep the main thread running without waiting
            done, not_done = wait(self.write_futures, timeout=0, return_when=FIRST_COMPLETED)
            for future in done:
                # Clean up completed futures
                if future.exception() is not None:
                    print(f"Warning: A write future encountered an exception: {future.exception()}")
                self.write_futures.remove(future)
            
            # Make local copies of the current buffers and clear the instance buffers.
            local_patch_buffer = {k: v[:] for k, v in self.patch_buffer.items()}
            local_positions = self.buffer_positions[:]
            self.patch_buffer.clear()
            self.buffer_positions.clear()
            
            # Offload asynchronous writing.
            future = self.executor.submit(self._process_buffer, output_arrays, count_arrays,
                                          local_patch_buffer, local_positions)
            self.write_futures.append(future)

            # Only block if we absolutely have to (when memory pressure is high)
            if len(self.write_futures) >= self.max_pending_writes:
                # Wait for at least one job to complete
                if self.verbose:
                    print(f"Waiting for write futures to complete ({len(self.write_futures)} pending)")
                done, not_done = wait(self.write_futures, return_when=FIRST_COMPLETED)
                # Convert the set back to a list
                self.write_futures = list(not_done)

    def infer(self):
        """Run inference with the nnUNet model on the zarr array."""
        # Verify input path exists and is a valid zarr array
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")
        
        try:
            # Try to open the input zarr array to check if it's valid
            if not self.postprocess_only:  # Skip this check in postprocess-only mode
                _ = zarr.open(self.input_path, mode='r')
        except Exception as e:
            raise ValueError(f"Error opening input zarr array at {self.input_path}: {str(e)}")
            
        # Create a synchronizer for concurrent writes
        os.makedirs(self.output_path, exist_ok=True)
        sync_path = os.path.join(self.output_path, ".zarr_sync")
        synchronizer = zarr.ProcessSynchronizer(sync_path)
        store_path = os.path.join(self.output_path, "predictions.zarr")
        
        # Load the nnUNet model if not in postprocess-only mode
        if not self.postprocess_only:
            self.model_info = self._load_nnunet_model()
            network = self.model_info['network']

        if not self.postprocess_only:
            # Only rank 0 creates the Zarr store and datasets.
            if self.rank == 0:
                if os.path.isdir(store_path) and not self.postprocess_only:
                    raise FileExistsError(f"Zarr store '{store_path}' already exists.")
                zarr_store = zarr.open(store_path, mode='w', synchronizer=synchronizer)
                output_arrays = {}
                count_arrays = {}

                # Create a temporary dataset to determine the full output shape.
                dataset_temp = InferenceDataset(
                    input_path=self.input_path,
                    targets=self.output_targets,
                    model_info=self.model_info,
                    patch_size=self.patch_size,
                    input_format=self.input_format,
                    step_size=self.tile_step_size,
                    load_all=self.load_all,
                    verbose=self.verbose
                )
                
                # Get the shape of the input array for output shape determination
                input_shape = dataset_temp.input_shape
                if len(input_shape) == 3:  # 3D array (Z,Y,X)
                    z_max, y_max, x_max = input_shape
                elif len(input_shape) == 4:  # 4D array (C,Z,Y,X)
                    _, z_max, y_max, x_max = input_shape
                
                chunk_z, chunk_y, chunk_x = self.patch_size
                compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

                for tgt_name, tgt_conf in self.output_targets.items():
                    c = tgt_conf["channels"]
                    
                    # Check if the nnUNet model outputs multilabel segmentation (num_classes > 2)
                    # or if the target has multiple channels
                    num_model_outputs = 1
                    if self.model_info is not None:
                        # For nnUNet, get number of segmentation heads
                        num_model_outputs = self.model_info.get('num_seg_heads', 1)
                        # For multiclass segmentation where the network has multiple segmentation heads
                        # Update the channel count to match
                        if num_model_outputs > 2 and tgt_name == "segmentation":
                            c = num_model_outputs
                            tgt_conf["channels"] = c
                            tgt_conf["nnunet_output_channels"] = c
                            print(f"Detected multiclass model with {c} classes from model_info")
                    
                    # Always use 4D output with channel dimension
                    # This ensures we handle nnUNet's binary and multiclass segmentation output correctly
                    out_shape = (c, z_max, y_max, x_max)
                    chunks = (c, chunk_z, chunk_y, chunk_x)
                    if self.verbose:
                        print(f"Using 4D output shape for target '{tgt_name}': {out_shape}, chunks: {chunks}")

                    sum_ds = zarr_store.create_dataset(
                        name=f"{tgt_name}_sum",
                        shape=out_shape,
                        chunks=chunks,
                        dtype='float32',
                        compressor=compressor,
                        fill_value=0,
                        synchronizer=synchronizer
                    )
                    cnt_ds = zarr_store.create_dataset(
                        name=f"{tgt_name}_count",
                        shape=(z_max, y_max, x_max),
                        chunks=(chunk_z, chunk_y, chunk_x),
                        dtype='float32',
                        compressor=compressor,
                        fill_value=0,
                        synchronizer=synchronizer
                    )

                    output_arrays[tgt_name] = sum_ds
                    count_arrays[tgt_name] = cnt_ds

            # Wait for rank 0 to create the store.
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])

            if self.rank != 0:
                zarr_store = zarr.open(store_path, mode='r+', synchronizer=synchronizer)
                output_arrays = {
                    tgt_name: zarr_store[f"{tgt_name}_sum"]
                    for tgt_name in self.output_targets
                }
                count_arrays = {
                    tgt_name: zarr_store[f"{tgt_name}_count"]
                    for tgt_name in self.output_targets
                }

            # Get device for inference
            if self.device_str.startswith('cuda'):
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f'cuda:{local_rank}')
            else:
                device = torch.device(self.device_str)

            # Add verbose flag to model_info for debugging
            if self.model_info is not None and self.verbose:
                self.model_info['verbose'] = True
                
            # Create dataset and dataloader
            dataset = InferenceDataset(
                input_path=self.input_path,
                targets=self.output_targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,  # This is a factor (0.5 = 50% overlap)
                load_all=self.load_all,
                verbose=self.verbose
            )
            
            if self.verbose:
                print(f"Created dataset with tile_step_size={self.tile_step_size}")
                print(f"  - Will generate {len(dataset)} patches")

            # Set up distributed sampler if needed
            sampler = None
            if dist.is_initialized():
                sampler = DistributedSampler(dataset, shuffle=False)
                sampler.set_epoch(0)

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=self.num_dataloader_workers,
                prefetch_factor=2,
                pin_memory=True,
                persistent_workers=False
            )

            # Run inference
            network.eval()
            
            # Determine optimal TTA batch multiplier using a single test patch
            # This gives us accurate GPU memory measurements
            if torch.cuda.is_available() and self.use_mirroring and self.rank == 0:
                print("Determining optimal TTA batch multiplier...")
                # Get the first batch to test memory usage
                test_data = next(iter(loader))
                test_patches = test_data["image"].to(device)
                
                # Run a single forward pass with TTA to accurately measure memory usage
                with torch.no_grad():
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    
                    # First run a normal forward pass 
                    _ = network(test_patches)
                    torch.cuda.synchronize(device)
                    
                    # Now run with a single TTA flip to measure memory requirements
                    # This gives us an accurate picture of TTA memory usage
                    # We'll use the first allowed mirroring axis
                    if 'allowed_mirroring_axes' in self.model_info and self.model_info['allowed_mirroring_axes']:
                        # Get first mirroring axis and adjust for batch/channel dimensions
                        axis = self.model_info['allowed_mirroring_axes'][0]
                        mirror_axis = axis + 2  # +2 for batch and channel dimensions
                        
                        # Create flipped input
                        flipped_input = torch.flip(test_patches, [mirror_axis])
                        
                        # Forward pass with flipped input
                        _ = network(flipped_input)
                        torch.cuda.synchronize(device)
                
                # Get peak memory usage for this forward pass
                peak_memory = torch.cuda.max_memory_allocated(device)
                current_memory = torch.cuda.memory_allocated(device)
                total_memory = torch.cuda.get_device_properties(device).total_memory
                
                # Calculate safe batch multiplier (4GB safety margin)
                safety_margin = 4 * (1024**3)  # 4GB in bytes
                available_memory = total_memory - current_memory - safety_margin
                
                # How many batches of this size can we fit with the remaining memory?
                if peak_memory > 0:
                    max_multiplier = int(available_memory / peak_memory)
                    # Use at least 1, at most 8 multiplier
                    max_multiplier = max(1, min(8, max_multiplier))
                    
                    print(f"GPU Memory Analysis:")
                    print(f"  Total GPU memory: {total_memory/(1024**3):.1f}GB")
                    print(f"  Current usage: {current_memory/(1024**3):.1f}GB")
                    print(f"  Peak usage for one batch: {peak_memory/(1024**3):.1f}GB")
                    print(f"  Available memory (with safety margin): {available_memory/(1024**3):.1f}GB")
                    print(f"  TTA batch multiplier: {max_multiplier}")
                    
                    # Update the max_tta_combinations parameter
                    # Handle different TTA modes
                    if self.full_tta:
                        # Full TTA mode - use all combinations (parallel or sequential based on memory)
                        self.max_tta_combinations = None  # None means use all combinations
                        print(f"Using FULL TTA mode with all available combinations")
                        print(f"Parallel processing multiplier: {max_multiplier}")
                    else:
                        # Primary axes mode - always prioritize the 3 main axes
                        min_desired_axes = 3
                        
                        # If user specified a value, respect it (up to memory limits)
                        if self.max_tta_combinations is not None:
                            # User-specified value
                            if max_multiplier < self.max_tta_combinations:
                                print(f"WARNING: Requested TTA combinations {self.max_tta_combinations} exceeds safe limit.")
                                if max_multiplier >= min_desired_axes:
                                    print(f"Reducing to {max_multiplier} (still covers all primary axes)")
                                else:
                                    print(f"Reducing to {max_multiplier} (insufficient memory for all primary axes)")
                                self.max_tta_combinations = max_multiplier
                        else:
                            # Always use at least the 3 primary axes - regardless of memory constraints
                            # We'll process them sequentially if needed
                            self.max_tta_combinations = min_desired_axes  # Always use 3 primary axes
                            
                            # But adapt the parallel processing based on memory
                            if max_multiplier >= min_desired_axes:
                                print(f"Using TTA with all {min_desired_axes} primary axes (parallel processing)")
                            else:
                                print(f"Using TTA with all {min_desired_axes} primary axes (sequential processing due to memory constraints)")
                    
                    # Also pass the parallel processing multiplier to the run_inference function
                    self.parallel_tta_multiplier = max_multiplier
                
                # Clear GPU memory
                del test_patches
                torch.cuda.empty_cache()
            
            if self.rank == 0:
                iterator = tqdm(enumerate(loader), total=len(loader),
                                desc="Running nnUNet inference on patches...")
            else:
                iterator = enumerate(loader)

            with torch.no_grad(), torch.amp.autocast(device_type=device.type):
                for batch_idx, data in iterator:
                    patches = data["image"].to(device)
                    indices = data["index"]  # these are the indices from the dataset
                    positions = [dataset.all_positions[i] for i in indices]

                    # Run inference with the nnUNet model
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    
                    # Use run_inference for nnUNet models with our optimized TTA settings
                    # Pass both parameters:
                    # - max_tta_combinations: Controls which combinations to use (None = all)
                    # - parallel_tta_multiplier: Controls how many to process in parallel (for performance)
                    outputs = run_inference(
                        self.model_info, 
                        patches, 
                        max_tta_combinations=self.max_tta_combinations,
                        parallel_tta_multiplier=getattr(self, 'parallel_tta_multiplier', None)
                    )
                    
                    # Process outputs based on activation functions
                    processed_outputs = {}
                    
                    # For the first batch, verify nnUNet output format
                    if batch_idx == 0:
                        # For nnUNet, outputs is typically a tensor
                        if torch.is_tensor(outputs):
                            # Get the output shape to determine number of channels
                            num_channels = outputs.shape[1]
                            print(f"nnUNet model output shape: {outputs.shape}, channels: {num_channels}")
                            
                            # Check if this is a binary or multiclass segmentation
                            if num_channels == 2:
                                print(f"Detected standard nnUNet binary segmentation output (2 channels).")
                                print(f"Will extract foreground channel (index {self.nnunet_foreground_channel}) for final output.")
                            else:
                                print(f"Detected multiclass segmentation output with {num_channels} channels.")
                                print(f"Processing all {num_channels} classes.")
                                
                                # Update the output target configuration to match the model output
                                for t_name in self.output_targets:
                                    self.output_targets[t_name]["channels"] = num_channels
                                    self.output_targets[t_name]["nnunet_output_channels"] = num_channels
                                
                                # Only print this message on rank 0
                                if self.rank == 0:
                                    print(f"Automatically updated output configuration to handle {num_channels} classes.")
                    
                    # Apply activations for each target
                    for t_name in self.output_targets:
                        # For nnUNet, the output might be a single tensor, not a dict with keys
                        if torch.is_tensor(outputs):
                            raw_output = outputs
                        else:
                            raw_output = outputs[t_name]
                            
                        t_conf = self.output_targets[t_name]
                        activation_str = t_conf.get("activation", "none").lower()
                        
                        # Apply activation function
                        if activation_str == "sigmoid":
                            activated = torch.sigmoid(raw_output)
                        elif activation_str == "softmax":
                            activated = torch.softmax(raw_output, dim=1)
                        else:
                            activated = raw_output
                        
                        # Handle different segmentation types
                        if activated.shape[1] == 2:
                            # Binary segmentation: extract only the foreground channel (index 1)
                            processed_outputs[t_name] = activated[:, self.nnunet_foreground_channel:self.nnunet_foreground_channel+1]
                        elif t_name == "segmentation" and activated.shape[1] > 2:
                            # Multiclass segmentation: keep all channels
                            processed_outputs[t_name] = activated
                        else:
                            # Default behavior: keep all channels
                            processed_outputs[t_name] = activated
                            
                    self._process_model_outputs(processed_outputs, positions, output_arrays, count_arrays)

            # Process any remaining patches in the buffer.
            if self.buffer_positions:
                # Create a copy of the buffer to avoid race conditions
                local_patch_buffer = {k: v[:] for k, v in self.patch_buffer.items()}
                local_positions = self.buffer_positions[:]
                self.patch_buffer.clear()
                self.buffer_positions.clear()
                
                # Process the final buffer
                self._process_buffer(output_arrays, count_arrays, local_patch_buffer, local_positions)
                
            # Wait for and process all pending futures
            if self.write_futures:
                if self.verbose:
                    print(f"Waiting for {len(self.write_futures)} remaining write operations to complete...")
                for i, future in enumerate(self.write_futures):
                    try:
                        # Process futures one by one to catch any exceptions
                        future.result()
                        if self.verbose and (i + 1) % 10 == 0:  # Print progress every 10 futures
                            print(f"Processed {i+1}/{len(self.write_futures)} pending writes")
                    except Exception as e:
                        print(f"Error in write future: {e}")
                self.write_futures.clear()

            # Shut down the executor.
            if self.executor is not None:
                self.executor.shutdown(wait=True)

        else:
            # Postprocess-only: open the existing store.
            zarr_store = zarr.open(store_path, mode='r+', synchronizer=synchronizer)

        # Barrier: ensure all processes finish inference before postprocessing.
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

        # Only rank 0 performs postprocessing.
        if self.rank == 0:
            self._optimized_postprocessing(zarr_store)

        # Final barrier to ensure all processes complete postprocessing.
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])
            
        # Output location info
        if self.rank == 0:
            print(f"\nFinal output saved to {store_path}")
            
        # Clean up the synchronizer
        if os.path.exists(sync_path):
            if self.verbose and self.rank == 0:
                print(f"Cleaning up zarr synchronizer at {sync_path}")
            try:
                import shutil
                shutil.rmtree(sync_path)
                if self.verbose and self.rank == 0:
                    print(f"Successfully removed synchronizer directory")
            except Exception as e:
                print(f"Warning: Could not remove synchronizer directory: {e}")

    def _optimized_postprocessing(self, zarr_store):
        """Optimized post-processing with improved vector handling and optional thresholding"""
        for tgt_name in self.output_targets:
            sum_ds = zarr_store[f"{tgt_name}_sum"]
            cnt_ds = zarr_store[f"{tgt_name}_count"]
            is_normals = (tgt_name.lower() == "normals")
            chunk_size = sum_ds.chunks[-3]

            final_dtype = "uint16" if is_normals else "uint8"
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

            # Check if this is a multiclass segmentation
            is_multiclass = sum_ds.shape[0] > 2 if len(sum_ds.shape) >= 4 else False
            
            # Determine if we should save probability maps
            # For multiclass, we might skip them to save space
            # For binary segmentation or non-segmentation targets, always save them
            should_save_probabilities = (
                self.save_probability_maps or  # User explicitly wants probability maps
                not is_multiclass or          # Binary segmentation - always save probabilities
                tgt_name != "segmentation"    # Not a segmentation target - always save values
            )
            
            # Create dataset for probability output if needed
            final_ds = None
            if should_save_probabilities:
                # Use a more descriptive name for segmentation output
                if tgt_name == "segmentation":
                    final_name = f"{tgt_name}_probabilities"
                else:
                    final_name = f"{tgt_name}_final"
                    
                final_ds = zarr_store.create_dataset(
                    name=final_name,
                    shape=sum_ds.shape,
                    chunks=sum_ds.chunks,
                    dtype=final_dtype,
                    compressor=compressor,
                    fill_value=0
                )
                
                if self.verbose and is_multiclass and tgt_name == "segmentation":
                    print(f"Saving full probability maps for {sum_ds.shape[0]} classes (use --skip_probability_maps to save space)")
            elif self.verbose and is_multiclass and tgt_name == "segmentation":
                print(f"Skipping probability maps for {sum_ds.shape[0]} classes to save disk space")
            
            # For multiclass segmentation, also create an argmax output
            argmax_ds = None
            if is_multiclass and tgt_name == "segmentation":
                if self.verbose:
                    print(f"Creating argmax output for multiclass segmentation with {sum_ds.shape[0]} classes")
                argmax_ds = zarr_store.create_dataset(
                    name=f"{tgt_name}_argmax",
                    shape=sum_ds.shape[1:],  # Remove channel dimension
                    chunks=sum_ds.chunks[1:],  # Remove channel dimension from chunks
                    dtype="uint8",
                    compressor=compressor,
                    fill_value=0
                )
            
            # Create additional dataset for thresholded output if threshold is specified
            thresholded_ds = None
            if self.threshold is not None and not is_normals:
                if self.verbose:
                    print(f"Threshold value set to {self.threshold}% - will create binary threshold output")
                    
                # For multiclass, create same shape as probabilities
                thresholded_ds = zarr_store.create_dataset(
                    name=f"{tgt_name}_threshold",
                    shape=sum_ds.shape,
                    chunks=sum_ds.chunks,
                    dtype="uint8",
                    compressor=compressor,
                    fill_value=0
                )

            # Determine if we have 3D or 4D arrays
            sum_ds_ndim = len(sum_ds.shape)
            
            # Print shape info if verbose
            if self.verbose:
                print(f"Postprocessing {tgt_name}: sum_ds shape {sum_ds.shape}, count_ds shape {cnt_ds.shape}")
            
            for z0 in tqdm(range(0, sum_ds.shape[-3], chunk_size),
                           desc=f"Processing {tgt_name}"):
                z1 = min(z0 + chunk_size, sum_ds.shape[-3])
                
                # Process differently based on array dimensions
                if sum_ds_ndim == 4:  # 4D array (C,Z,Y,X)
                    sum_chunk = sum_ds[:, z0:z1].copy()
                    count_chunk = cnt_ds[z0:z1].copy()
                    count_chunk = np.expand_dims(count_chunk, axis=0)
                    count_chunk = np.broadcast_to(count_chunk, sum_chunk.shape)
                else:  # 3D array (Z,Y,X)
                    sum_chunk = sum_ds[z0:z1].copy()
                    count_chunk = cnt_ds[z0:z1].copy()

                # Create mask of non-zero counts
                mask = count_chunk > 0

                # Save the normalized probability data for thresholding later
                normalized_chunk = sum_chunk.copy()
                
                # Normalize based on count
                if is_normals and sum_ds_ndim == 4:
                    # Normalize vector magnitude for normals
                    eps = 1e-8
                    mag = np.sqrt(np.sum(sum_chunk ** 2, axis=0)) + eps
                    sum_chunk = sum_chunk / np.expand_dims(mag, axis=0)
                else:
                    # Standard normalization (divide by count where count > 0)
                    if self.verbose and z0 == 0:  # Only print this once
                        print(f"Normalizing with count array:")
                        print(f"  - count_chunk shape: {count_chunk.shape}")
                        print(f"  - count range: {np.min(count_chunk[mask]):.4f} to {np.max(count_chunk[mask]):.4f}")
                        print(f"  - before normalization: sum_chunk range: {np.min(sum_chunk[mask]):.4f} to {np.max(sum_chunk[mask]):.4f}")
                        
                        # Show histogram of count values to diagnose blending issues
                        unique_counts = np.unique(count_chunk[mask])
                        print(f"  - unique count values: {len(unique_counts)} values")
                        print(f"  - count distribution: {unique_counts[:10]}{'...' if len(unique_counts) > 10 else ''}")
                        
                        # Check for regions with very high or very low counts
                        high_count_pct = np.sum(count_chunk > np.max(count_chunk) * 0.8) / np.sum(mask) * 100
                        print(f"  - high count regions (>80% max): {high_count_pct:.2f}% of non-zero voxels")
                    
                    # Apply normalization where count > 0
                    sum_chunk[mask] /= count_chunk[mask]
                    normalized_chunk[mask] /= count_chunk[mask]
                    
                    if self.verbose and z0 == 0:  # Only print this once
                        print(f"  - after normalization: sum_chunk range: {np.min(sum_chunk[mask]):.4f} to {np.max(sum_chunk[mask]):.4f}")
                        
                        # Check for potential issues in result after normalization
                        if np.any(sum_chunk[mask] > 1.1):  # Values over 1.1 might indicate normalization issues
                            print(f"  - WARNING: Found values > 1.1 after normalization, max={np.max(sum_chunk[mask]):.4f}")
                            print(f"    This could indicate issues with the blending weights")

                # Apply final transformation based on target type
                if is_normals:
                    # For normals: map from [-1,1] to [0,65535]
                    sum_chunk = ((sum_chunk + 1.0) / 2.0 * 65535.0).clip(0, 65535).astype(np.uint16)
                else:
                    # For segmentation: handle based on target type
                    if tgt_name == "segmentation" and self.output_targets[tgt_name].get("activation") == "softmax":
                        # For softmax-activated binary segmentation from nnUNet:
                        # Scale to [0,255] for grayscale output (where higher values indicate higher foreground confidence)
                        sum_chunk = (sum_chunk * 255.0).clip(0, 255).astype(np.uint8)
                        if self.verbose:
                            print(f"Processed foreground probability map: min={np.min(sum_chunk)}, max={np.max(sum_chunk)}")
                    else:
                        # Standard scaling for other output types
                        sum_chunk = (sum_chunk * 255.0).clip(0, 255).astype(np.uint8)

                # Create thresholded output if requested
                if thresholded_ds is not None:
                    # Calculate threshold value from the percentage
                    # For probability maps that range from 0-1, we convert the percentage to a value between 0-1
                    threshold_value = self.threshold / 100.0
                    
                    # Apply threshold to the normalized chunk
                    if sum_ds_ndim == 4:
                        # For 4D data, threshold each channel separately
                        thresholded_chunk = (normalized_chunk >= threshold_value).astype(np.uint8) * 255
                    else:
                        # For 3D data
                        thresholded_chunk = (normalized_chunk >= threshold_value).astype(np.uint8) * 255
                    
                    # Write thresholded chunk to dataset
                    if sum_ds_ndim == 4 and len(thresholded_ds.shape) == 4:
                        # Both 4D
                        thresholded_ds[:, z0:z1] = thresholded_chunk
                    elif sum_ds_ndim == 3 and len(thresholded_ds.shape) == 3:
                        # Both 3D
                        thresholded_ds[z0:z1] = thresholded_chunk
                    elif sum_ds_ndim == 4 and len(thresholded_ds.shape) == 3:
                        # Handle 4D to 3D case
                        if thresholded_chunk.shape[0] == 1:
                            # Single channel
                            thresholded_ds[z0:z1] = thresholded_chunk[0]
                        else:
                            # Multiple channels - convert to binary using logical OR across channels
                            combined = np.any(thresholded_chunk > 0, axis=0).astype(np.uint8) * 255
                            thresholded_ds[z0:z1] = combined

                # Process and write back data
                # First, always process argmax for multiclass segmentation
                if is_multiclass and argmax_ds is not None and sum_chunk.shape[0] > 2:
                    # Calculate argmax along channel dimension (excluding background)
                    # By convention in medical imaging, channel 0 is often background
                    # For argmax output, we'll actually use argmax from channel 1 onwards
                    # and add 1 to the result to get class indices starting from 1
                    if self.verbose and z0 == 0:  # Print only once
                        print("Computing argmax for multiclass segmentation (excluding background)")
                    
                    # Skip background (channel 0) for argmax calculation
                    foreground_chunk = sum_chunk[1:, :, :, :]
                    
                    # Get argmax of foreground channels and add 1 to get class indices starting from 1
                    # Channels with zero probability will become class 0 (background)
                    # Apply a mask to only assign classes where there's sufficient probability
                    foreground_max = np.max(foreground_chunk, axis=0)
                    background_prob = sum_chunk[0, :, :, :]
                    
                    # Where foreground probability is higher than background, use argmax+1
                    # Otherwise use 0 (background)
                    mask = foreground_max > background_prob
                    
                    # Initialize with zeros (background)
                    result = np.zeros_like(mask, dtype=np.uint8)
                    
                    # Where foreground wins, use argmax+1
                    if np.any(mask):
                        # Get argmax of foreground channels only
                        argmax_foreground = np.argmax(foreground_chunk, axis=0)
                        # Add 1 to convert to class indices (1-based)
                        result[mask] = argmax_foreground[mask] + 1
                        
                    # Write to argmax dataset
                    argmax_ds[z0:z1] = result
                
                # Then write probability maps if needed
                if final_ds is not None:
                    if sum_ds_ndim == 4 and len(final_ds.shape) == 4:
                        # Both 4D
                        final_ds[:, z0:z1] = sum_chunk
                    elif sum_ds_ndim == 3 and len(final_ds.shape) == 3:  
                        # Both 3D
                        final_ds[z0:z1] = sum_chunk
                    elif sum_ds_ndim == 4 and len(final_ds.shape) == 3:
                        # Sum is 4D but final is 3D (e.g., taking argmax)
                        if sum_chunk.shape[0] > 1:
                            # Multi-class: convert to single channel via argmax
                            final_ds[z0:z1] = np.argmax(sum_chunk, axis=0).astype(np.uint8)
                        else:
                            # Single class: just take first channel
                            final_ds[z0:z1] = sum_chunk[0]
                    else:
                        # Handle unusual case
                        raise ValueError(f"Incompatible dimensions in postprocessing: sum_ds has {sum_ds_ndim} dimensions " 
                                      f"but final_ds has {len(final_ds.shape)} dimensions")
        
        # Clean up intermediate arrays if not keeping them
        if not self.keep_intermediates:
            for tgt_name in self.output_targets:
                if self.verbose:
                    print(f"Cleaning up intermediate arrays for {tgt_name}")
                # Delete sum and count arrays
                if f"{tgt_name}_sum" in zarr_store:
                    del zarr_store[f"{tgt_name}_sum"]
                if f"{tgt_name}_count" in zarr_store:
                    del zarr_store[f"{tgt_name}_count"]


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for nnUNet models on zarr arrays with DDP support."
    )
    parser.add_argument("--input_path", type=str, required=True,
                      help="Path to the input zarr array")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Path to save the output predictions")
    parser.add_argument("--model_folder", type=str, required=True,
                      help="Path to the nnUNet model folder")
    parser.add_argument("--fold", type=str, default="0",
                      help="Fold to use for inference (default: 0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth",
                      help="Checkpoint file name to use (default: checkpoint_final.pth)")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Batch size for inference (default: 4)")
    parser.add_argument("--step_size", type=float, default=0.5,
                      help="Step size for sliding window as a fraction of patch size (default: 0.5, nnUNet default)")
    parser.add_argument("--num_dataloader_workers", type=int, default=4,
                      help="Number of workers for the DataLoader (default: 4)")
    parser.add_argument("--num_write_workers", type=int, default=4,
                      help="Number of worker threads for asynchronous disk writes (default: 4)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run inference on ('cuda' or 'cpu') (default: cuda)")
    parser.add_argument("--threshold", type=float, 
                      help="Apply threshold to probability map (value 0-100, represents percentage)")
    parser.add_argument("--disable_tta", action="store_true",
                      help="Disable test time augmentation (mirroring) for faster but potentially less accurate inference")
    parser.add_argument("--max_tta_combinations", type=int, default=None,
                      help="Maximum number of TTA combinations to use (default: None = auto-detect based on GPU memory)")
    parser.add_argument("--full_tta", action="store_true",
                      help="Use all possible TTA combinations (slower but potentially more accurate)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable detailed output messages during inference")
    parser.add_argument("--write_layers", action="store_true",
                      help="Write the sliced z layers to disk")
    parser.add_argument("--postprocess_only", action="store_true",
                      help="Skip the inference pass and only do final averaging + casting")
    parser.add_argument("--keep_intermediates", action="store_true",
                      help="Keep intermediate sum and count arrays after processing")
    parser.add_argument("--skip_probability_maps", action="store_true",
                      help="Skip storing full probability maps for multiclass segmentation to save disk space")
    parser.add_argument("--load_all", action="store_true",
                      help="Load the entire input array into memory (use with caution!)")
    
    args = parser.parse_args()

    # Set the CUDA device before initializing the process group.
    # Initialize distributed process group if running with DDP
    world_size = 1
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)  # Set the device first!
        dist.init_process_group(backend='nccl', init_method='env://')
        device = f'cuda:{local_rank}'
        
        # Only rank 0 should print this message
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"Running in distributed mode with {world_size} processes")
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Automatically adjust worker counts when running with DDP
    # This ensures we use the same total resources regardless of process count
    dataloader_workers = max(1, args.num_dataloader_workers // world_size)
    write_workers = max(1, args.num_write_workers // world_size)
    
    # Only rank 0 prints the adjusted worker counts
    if ('RANK' not in os.environ) or (int(os.environ.get("RANK", 0)) == 0):
        if world_size > 1:
            print(f"Adjusting worker counts for {world_size} processes:")
            print(f"  - Dataloader workers per process: {dataloader_workers} (total: {dataloader_workers * world_size})")
            print(f"  - Write workers per process: {write_workers} (total: {write_workers * world_size})")

    inference_handler = ZarrNNUNetInferenceHandler(
        input_path=args.input_path,
        output_path=args.output_path,
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        step_size=args.step_size,
        num_dataloader_workers=dataloader_workers,  # Use adjusted worker count
        num_write_workers=write_workers,            # Use adjusted worker count
        write_layers=args.write_layers,
        postprocess_only=args.postprocess_only,
        device=device,
        threshold=args.threshold,
        use_mirroring=not args.disable_tta,  # Invert flag to match nnUNet's behavior
        max_tta_combinations=args.max_tta_combinations,
        full_tta=args.full_tta,  # Whether to use all TTA combinations
        verbose=args.verbose,
        keep_intermediates=args.keep_intermediates,
        save_probability_maps=not args.skip_probability_maps,  # Invert flag for intuitive CLI
        load_all=args.load_all
    )
    
    inference_handler.infer()  # Run inference and postprocessing

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()