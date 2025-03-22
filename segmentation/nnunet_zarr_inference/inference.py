import os
import sys
import numpy as np
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import threading
from typing import Dict, Tuple, List, Optional, Any, Union
import torch.distributed as dist
from scipy.ndimage import gaussian_filter
import uuid
import queue

# Add parent directory to sys.path for direct imports when running the script directly
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Now we can import as a module
    # print(f"Added {parent_dir} to sys.path")

# Import blending and activation
if __name__ == "__main__" or not __package__:
    # Direct script execution
    from nnunet_zarr_inference.blending_torch import (create_gaussian_weights_torch,
                                                      blend_patch_torch)
else:
    # Module import
    from .blending_torch import (create_gaussian_weights_torch,
                                 blend_patch_torch)

# Import our zarr cache
if __name__ == "__main__" or not __package__:
    # Direct script execution
    from nnunet_zarr_inference.zarr_cache import ZarrArrayLRUCache
else:
    # Module import
    from .zarr_cache import ZarrArrayLRUCache

# Try to configure NumPy and related libraries to use multiple threads
try:
    # Determine optimal number of threads for various operations
    num_physical_cores = os.cpu_count() // 2 if os.cpu_count() else 4
    num_threads = min(8, num_physical_cores)

    # Configure NumPy threading
    np.set_num_threads(num_threads)

    # Try to configure OpenMP for libraries like MKL
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
except (AttributeError, ImportError):
    pass

# Determine if this is being run as a module or directly
import sys
import os

if __name__ == "__main__" or not __package__:
    # Add parent directory to sys.path for direct script execution
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Import directly
    from nnunet_zarr_inference.load_nnunet_model import load_model, run_inference
    from nnunet_zarr_inference.inference_dataset import InferenceDataset
    from nnunet_zarr_inference.helpers import compute_steps_for_sliding_window_tuple
else:
    # Relative imports for package/module usage
    from .load_nnunet_model import load_model, run_inference
    from .inference_dataset import InferenceDataset
    from .helpers import compute_steps_for_sliding_window_tuple


class SequentialZarrNNUNetInference:
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
                 device: str = 'cuda',
                 threshold: Optional[float] = None,
                 use_mirroring: bool = True,
                 max_tta_combinations: Optional[int] = None,
                 verbose: bool = False,
                 save_probability_maps: bool = True,
                 output_targets: Optional[List[Dict[str, Any]]] = None,
                 rank: int = 0,
                 edge_weight_boost: float = 0.5,
                 cache_size: int = 256,
                 max_cache_bytes: float = 4.0):
        """
        A sequential approach to nnUNet inference on zarr arrays.

        This implementation differs from the original ZarrNNUNetInferenceHandler by:
        1. Using a sequential writing approach to avoid chunk conflicts
        2. Writing each patch to a separate chunk in a temporary storage
        3. Only performing the Gaussian blending at the end during a final reduction pass

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
            device: Device to run inference on ('cuda' or 'cpu')
            threshold: Optional threshold value (0-100) for binarizing the probability map
            use_mirroring: Enable test time augmentation via mirroring (default: True, matches nnUNet default)
            max_tta_combinations: Maximum number of TTA combinations to use (default: None = auto-detect based on GPU memory)
            verbose: Enable detailed output messages during inference (default: False)
            save_probability_maps: Save full probability maps for multiclass segmentation (default: True, set to False to save space)
            output_targets: Optional list of output target configurations, each a dictionary with 'name' and other parameters
            rank: Process rank for distributed processing (default: 0, only process 0 will print verbose outputs)
            edge_weight_boost: Factor to boost Gaussian weights at patch edges (default: 0.5). Higher values reduce artifacts
                               at volume boundaries but may affect blending quality elsewhere. Set to 0 for original behavior.
            cache_size: Number of zarr chunks to cache in memory (default: 256)
            max_cache_bytes: Maximum memory in GB to use for zarr cache (default: 4.0)
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
        self.device_str = device
        self.threshold = threshold
        self.use_mirroring = use_mirroring
        self.max_tta_combinations = max_tta_combinations
        self.verbose = verbose
        self.save_probability_maps = save_probability_maps
        self.rank = rank  # Store rank for distributed training awareness
        self.edge_weight_boost = edge_weight_boost  # Store edge weight boost factor
        self.cache_size = cache_size  # Number of zarr chunks to cache
        self.max_cache_bytes = max_cache_bytes  # Maximum cache size in GB

        # Convert patch_size to tuple if it's a list or another sequence
        if patch_size is not None:
            self.patch_size = tuple(patch_size)

        # Use the provided output_targets or default to a standard configuration
        # Always use the list format expected by InferenceDataset
        self.targets = [
            {
                "name": "segmentation",
                "channels": 2,  # nnUNet always outputs at least 2 channels (background, foreground)
                "activation": "sigmoid",
                "nnunet_output_channels": 2  # This is the minimum number of channels
            }
        ]

        # Define foreground channel index for reference (not used for special handling)
        self.nnunet_foreground_channel = 1  # Second channel (index 1) is foreground in binary segmentation

        if self.verbose:
            print(f"Initialized with step_size={self.tile_step_size}")
            for target in self.targets:
                print(f"Output target '{target.get('name')}': {target}")

        # Determine rank for DDP
        self.rank = 0
        if dist.is_initialized():
            self.rank = dist.get_rank()

        # Writer queue for patches
        self.writer_queue = queue.Queue()

        # Load the nnUNet model
        self.model_info = None
        self.patch_size = patch_size  # This will be overridden if None and model is loaded

        # Number of writer threads
        self.num_write_workers = num_write_workers

        # For TTA, we'll use only the 3 primary axis for mirroring
        self.tta_directions = [(0,), (1,), (2,)]  # Only axis-aligned flips

    def _load_nnunet_model(self):
        """
        Load the nnUNet model and return model information.
        """
        try:
            # Only print from rank 0
            if self.rank == 0:
                print(f"Loading nnUNet model from {self.model_folder}, fold {self.fold}")

            if self.verbose and self.rank == 0:
                print(f"Test time augmentation (mirroring): {'enabled' if self.use_mirroring else 'disabled'}")

            # Set verbose to False for non-rank-0 processes to avoid duplicate messages
            local_verbose = self.verbose and self.rank == 0

            model_info = load_model(
                model_folder=self.model_folder,
                fold=self.fold,
                checkpoint_name=self.checkpoint_name,
                device=self.device_str,
                use_mirroring=self.use_mirroring,
                verbose=local_verbose,
                rank=self.rank  # Pass rank to load_model
            )

            # Use the model's patch size if none was specified
            if self.patch_size is None:
                # Always convert to tuple for consistency (model_info often has it as a list)
                self.patch_size = tuple(model_info['patch_size'])
                if self.verbose and self.rank == 0:
                    print(f"Using model's patch size: {self.patch_size}")

            # Report multiclass vs binary if rank 0
            if self.rank == 0:
                num_classes = model_info.get('num_seg_heads', 0)
                if num_classes > 2:
                    print(f"Detected multiclass model with {num_classes} classes from model_info")
                elif num_classes == 2:
                    print(f"Detected binary segmentation model from model_info")

            return model_info
        except Exception as e:
            print(f"Error loading nnUNet model (rank {self.rank}): {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_blend_weights(self, device=None, edge_weight_boost=0.5):
        """
        Create a 3D Gaussian window to be used as blending weights using PyTorch.
        This is a reimplementation of the original nnUNet approach with edge enhancement
        to reduce artifacts at volume boundaries.

        Args:
            device: Optional PyTorch device to create the weights on. If None, uses self.device_str.
            edge_weight_boost: Factor to boost edge weights (0.0 = no boost, 1.0 = full plateau)
                            Higher values reduce edge artifacts but may introduce other blending effects.
        """
        # Ensure patch_size is a tuple
        patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

        # For step size of 1.0, no blending is needed
        if self.tile_step_size == 1.0:
            return torch.ones(patch_size_tuple, dtype=torch.float32)

        # For segmentation, use more conservative settings to avoid edge artifacts
        if any(target.get('name') == 'segmentation' for target in self.targets):
            sigma_scale = 1.0 / 8.0  # Larger sigma for smoother transitions
            value_scaling_factor = 10  # Smaller scaling for more balanced weights
            edge_weight_boost = 0.0  # Disable edge boosting for segmentation
        else:
            sigma_scale = 1.0 / 8.0  # Match nnUNet's original approach
            value_scaling_factor = 10  # Set to create smooth transitions between patches

        # Print debugging info about the gaussian creation
        if self.verbose:
            print(f"Creating Gaussian blend weights:")
            print(f"  - Patch size: {self.patch_size}")
            print(f"  - Using sigma_scale: {sigma_scale}")
            print(f"  - Value scaling factor: {value_scaling_factor}")
            print(f"  - Edge weight boost: {edge_weight_boost}")

        # Determine device for PyTorch operations
        if device is None:
            device = torch.device(
                self.device_str if torch.cuda.is_available() and self.device_str.startswith('cuda') else 'cpu')

        gaussian_importance_map = create_gaussian_weights_torch(
            patch_size_tuple,
            sigma_scale=sigma_scale,
            value_scaling_factor=value_scaling_factor,
            device=device,
            edge_weight_boost=edge_weight_boost
        )

        if self.verbose:
            print(f"Created Gaussian blend weights with shape {gaussian_importance_map.shape}")
            print(f"  - min: {torch.min(gaussian_importance_map):.4f}, max: {torch.max(gaussian_importance_map):.4f}")
            print(f"  - device: {gaussian_importance_map.device}")

        return gaussian_importance_map

    def _create_memmap_arrays(self, target_name, num_expected_patches, patch_shape, temp_dir):
        """
        Create storage for patches and positions.

        Uses a memory-efficient approach:
        - Patches stored in memory-mapped uint8 arrays for efficiency with large data
        - Positions stored in regular Python list (in memory) for faster access
        - Index counter stored as a thread-safe integer value in memory

        Args:
            target_name: Name of the target
            num_expected_patches: Expected number of patches
            patch_shape: Shape of a single patch (C, Z, Y, X)
            temp_dir: Directory to store memory-mapped files

        Returns:
            Tuple of (patches_array, positions_list, counter, file_path)
        """
        # Create file path for memory mapped patches array
        patches_file = os.path.join(temp_dir, f"{target_name}_patches_{uuid.uuid4().hex}.npy")

        # Create memory-mapped array for patches
        # For segmentation, we need to store raw logits as float32
        # For other targets, we use uint8 to save memory
        if target_name == "segmentation":
            dtype = 'float32'
        else:
            dtype = 'uint8'

        patches_array = np.memmap(
            patches_file,
            dtype=dtype,
            mode='w+',
            shape=(num_expected_patches,) + patch_shape
        )

        # Use a regular Python list for positions (memory efficient enough)
        # Pre-allocate with None values to maintain array-like indexing
        positions_list = [None] * num_expected_patches

        # Use a thread-safe counter object
        counter = {'value': 0, 'lock': threading.Lock()}

        if self.verbose:
            print(f"Created storage for {target_name}:")
            print(f"  - Patches: memory-mapped uint8 array with shape {patches_array.shape}")
            print(f"  - Positions: in-memory list with {len(positions_list)} slots")

        # Return storage objects (patches, positions, counter, file_path)
        return patches_array, positions_list, counter, patches_file

    def _get_next_index(self, counter):
        """
        Get the next available index and increment the counter atomically.

        Uses an in-memory counter object with a lock for thread safety.

        Args:
            counter: A dictionary with 'value' and 'lock' keys

        Returns:
            Current index before increment
        """
        with counter['lock']:
            current_index = counter['value']
            counter['value'] += 1

        return current_index

    def _writer_worker(self, target_arrays, work_queue, worker_id):
        """
        Worker thread that writes patches to memory-mapped arrays and positions to in-memory lists.

        Args:
            target_arrays: Dictionary mapping target names to tuples of (patches_array, positions_list, counter, file_path).
            work_queue: Queue for writer tasks
            worker_id: Unique ID for this worker
        """
        try:
            while True:
                # Get an item from the queue
                item = work_queue.get()

                # Check for sentinel value to terminate
                if item is None:
                    if self.verbose:
                        print(f"Writer {worker_id} received termination signal")
                    work_queue.task_done()
                    break

                # Unpack the item
                patch, position, target_name = item

                # Get position info
                z, y, x = position

                # Get arrays for this target
                if target_name not in target_arrays:
                    # Skip if arrays don't exist for this target
                    print(f"Warning: No arrays found for target {target_name}")
                    work_queue.task_done()
                    continue

                # Get the first array tuple from the list (we only write to the first one)
                patches_array, positions_list, counter, file_path = target_arrays[target_name][0]

                # Get the next available index from the thread-safe counter
                idx = self._get_next_index(counter)

                # Bounds check to prevent index errors
                if idx >= patches_array.shape[0]:
                    print(f"Warning: Index {idx} exceeds array size {patches_array.shape[0]}. Skipping patch.")
                    work_queue.task_done()
                    continue

                # Write patch to memory-mapped array
                patches_array[idx] = patch

                # Write position to in-memory list
                positions_list[idx] = (z, y, x)

                # Flush the patches array to ensure changes are written to disk
                patches_array.flush()

                # Log patch details
                if self.verbose and idx < 3:
                    print(f"Writer {worker_id}: Wrote patch with shape {patch.shape} at index {idx} for {target_name}")
                elif self.verbose and idx % 1000 == 0:
                    print(f"Writer {worker_id}: Progress - wrote patch {idx} for {target_name}")

                # Mark task as done
                work_queue.task_done()

        except Exception as e:
            print(f"Error in writer thread {worker_id}: {str(e)}")
            if 'patch' in locals():
                print(f"Patch shape: {patch.shape}")
                print(f"Patch dtype: {patch.dtype}")
            import traceback
            traceback.print_exc()
            work_queue.task_done()

    def _blend_patches(self, patch_arrays, output_arrays, count_arrays):
        """
        Blend all patches into the final output array using Gaussian weights.
        Optimized version with PyTorch acceleration for better performance.
        Keeps everything in PyTorch tensors throughout the process to minimize CPU-GPU transfers.

        Args:
            patch_arrays: Dictionary mapping target names to patch data. It can be either:
                1. A single tuple (patches_array, positions_list, counter, file_path)
                2. A list of such tuples for handling multiple ranks in DDP mode
            output_arrays: Dictionary mapping target names to output arrays (NumPy/memmap)
            count_arrays: Dictionary mapping target names to count arrays (NumPy/memmap)

        Returns:
            Dictionary mapping target names to PyTorch tensors for output and count arrays,
            to be used directly in _finalize_arrays to avoid unnecessary transfers
        """
        if self.verbose:
            print("Starting patch blending phase...")

        # Ensure patch_size is a tuple
        patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

        # Create Gaussian blend weights - use device from model
        device = self.device_str
        blend_weights = self._create_blend_weights(
            device=device,
            edge_weight_boost=self.edge_weight_boost
        )

        # We don't need the torch versions of the intersection functions anymore
        # since we're handling the intersections directly in NumPy

        # Dictionary to store output/count tensors
        tensor_dict = {}

        # Process each target
        for target in self.targets:
            tgt_name = target.get("name")
            if tgt_name not in patch_arrays:
                print(f"Warning: No patches found for target {tgt_name}")
                continue

            if self.verbose:
                print(f"Blending patches for target: {tgt_name}")

            # Handle both single array and list of arrays formats
            patches_data_list = patch_arrays[tgt_name]
            if not isinstance(patches_data_list, list):
                patches_data_list = [patches_data_list]

            total_patches = 0
            for patches_array, positions_list, counter, _ in patches_data_list:
                total_patches += counter['value']

            if self.verbose:
                print(f"  - Total patches to blend: {total_patches}")

            # Get array dimensions
            if len(output_arrays[tgt_name].shape) == 4:  # 4D output (C,Z,Y,X)
                c, max_z, max_y, max_x = output_arrays[tgt_name].shape
            else:  # 3D output (Z,Y,X)
                max_z, max_y, max_x = output_arrays[tgt_name].shape
                c = 1

            # Convert output and count arrays to PyTorch tensors once
            # Need to copy to make them writable and avoid warnings
            # Keep them on GPU throughout the process
            output_tensor = torch.from_numpy(output_arrays[tgt_name].copy()).to(device)
            count_tensor = torch.from_numpy(count_arrays[tgt_name].copy()).to(device)

            # Store these tensors for later use in finalize
            tensor_dict[tgt_name] = (output_tensor, count_tensor)

            # We'll process all patches at once to prevent double-processing at chunk boundaries
            if self.verbose:
                print(f"Processing patches in a single pass to avoid boundary issues")

            # Create a combined list of all candidate patches from all ranks
            all_candidates = []

            # First, collect all patch candidates across all ranks
            for rank_idx, (patches_array, positions_list, counter, _) in enumerate(patches_data_list):
                n_patches = counter['value']

                for i in range(n_patches):
                    # Skip None values (should not happen but check to be safe)
                    if positions_list[i] is None:
                        continue

                    # Get position tuple (z, y, x)
                    pos = positions_list[i]

                    # Store rank, index, and position
                    all_candidates.append((rank_idx, i, pos))

            if len(all_candidates) == 0:
                continue

            if self.verbose:
                print(f"  - Processing all {len(all_candidates)} patches in a single pass")

            # Sort all candidates by z-coordinate then y-coordinate for better cache locality
            all_candidates.sort(key=lambda x: (x[2][0], x[2][1]))  # Sort by z, then y coordinates

            # Process all patches from all ranks
            for rank_idx, idx, pos in all_candidates:
                # Get the correct patch data based on rank index
                patches_array, positions_list, _, _ = patches_data_list[rank_idx]

                # Extract position
                z, y, x = pos

                # Calculate intersection coordinates directly
                # Calculate bounds for the output region
                patch_z_end = min(z + patch_size_tuple[0], max_z)
                target_z_start = max(0, z)
                target_z_end = min(max_z, patch_z_end)

                target_y_start = y
                target_y_end = min(max_y, y + patch_size_tuple[1])

                target_x_start = x
                target_x_end = min(max_x, x + patch_size_tuple[2])

                # Calculate offsets in the patch
                patch_z_start_rel = max(0, 0 - z)  # Relative to start of volume (0), not chunk
                patch_z_end_rel = min(patch_size_tuple[0], max_z - z)
                patch_y_start_rel = 0
                patch_y_end_rel = min(patch_size_tuple[1], max_y - y)
                patch_x_start_rel = 0
                patch_x_end_rel = min(patch_size_tuple[2], max_x - x)

                # Skip if patch has invalid dimensions
                if (patch_y_end_rel <= patch_y_start_rel or
                        patch_x_end_rel <= patch_x_start_rel or
                        patch_z_end_rel <= patch_z_start_rel):
                    continue

                # Organize coordinates for blending
                target_coords = (target_z_start, target_z_end,
                                 target_y_start, target_y_end,
                                 target_x_start, target_x_end)

                patch_coords = (patch_z_start_rel, patch_z_end_rel,
                                patch_y_start_rel, patch_y_end_rel,
                                patch_x_start_rel, patch_x_end_rel)

                # Convert patch to PyTorch tensor - need to copy to make it writable
                # This avoids the "non-writable tensor" warning when converting from memmap
                # For segmentation, patches are now stored as float32 raw logits
                # For other targets, patches are still uint8 for memory efficiency
                patch_tensor = torch.from_numpy(patches_array[idx].copy()).to(device)

                # Unpack coordinates for blending
                target_z_start, target_z_end, target_y_start, target_y_end, target_x_start, target_x_end = target_coords
                patch_z_start_rel, patch_z_end_rel, patch_y_start_rel, patch_y_end_rel, patch_x_start_rel, patch_x_end_rel = patch_coords

                # Blend the patch using PyTorch
                blend_patch_torch(
                    output_tensor, count_tensor,
                    patch_tensor, blend_weights,
                    target_z_start, target_z_end,
                    target_y_start, target_y_end,
                    target_x_start, target_x_end,
                    patch_z_start_rel, patch_z_end_rel,
                    patch_y_start_rel, patch_y_end_rel,
                    patch_x_start_rel, patch_x_end_rel
                )

                # Periodically clean up to free GPU memory
                if len(all_candidates) > 100 and (all_candidates.index((rank_idx, idx, pos)) + 1) % 50 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                # We don't copy back to CPU after each chunk - keep everything on GPU
                # Suggest garbage collection after each major chunk for GPU memory
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # Once we're done with a target, suggest explicit garbage collection
            if self.verbose:
                print(f"Completed blending for {tgt_name}")
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return tensor_dict

    def _finalize_arrays(self, output_arrays, count_arrays, final_arrays, tensor_dict=None):
        """
        Divide the sum arrays by the count arrays to get the final result.
        Also handles thresholding and conversion to appropriate data type.

        If save_probability_maps is False, computes argmax over channels for multiclass segmentation.
        For binary segmentation with exactly 2 channels, extracts only channel 1 and applies sigmoid.

        This is done chunk by chunk to avoid loading entire large arrays into memory.
        Optimized with PyTorch for better performance on GPU.

        Args:
            output_arrays: Dictionary mapping target names to output arrays (NumPy/memmap)
            count_arrays: Dictionary mapping target names to count arrays (NumPy/memmap)
            final_arrays: Dictionary mapping target names to final output arrays (Zarr)
            tensor_dict: Optional dictionary with target names mapping to PyTorch tensors
                         containing output and count arrays already on GPU
        """
        if self.verbose:
            print("Finalizing arrays..")
            print("Information about arrays:")
            for tgt_name in output_arrays:
                print(
                    f"  - output_arrays[{tgt_name}] shape: {output_arrays[tgt_name].shape}, dtype: {output_arrays[tgt_name].dtype}")
                print(
                    f"  - count_arrays[{tgt_name}] shape: {count_arrays[tgt_name].shape}, dtype: {count_arrays[tgt_name].dtype}")
                print(
                    f"  - final_arrays[{tgt_name}] shape: {final_arrays[tgt_name].shape}, dtype: {final_arrays[tgt_name].dtype}")
                print(f"  - final_arrays[{tgt_name}] type: {type(final_arrays[tgt_name])}")
                print(f"  - save_probability_maps: {self.save_probability_maps}")

        # Get device from model
        device = self.device_str

        # Import the normalize_chunk_torch function
        if __name__ == "__main__" or not __package__:
            from nnunet_zarr_inference.blending_torch import normalize_chunk_torch
        else:
            from .blending_torch import normalize_chunk_torch

        for target in self.targets:
            tgt_name = target.get("name")
            if self.verbose:
                print(f"Processing {tgt_name}...")

            # Define threshold_val at the beginning for each target
            # This ensures it's available for all code paths
            threshold_val = None
            if self.threshold is not None:
                threshold_val = self.threshold / 100.0

            if self.verbose:
                print(f"Using threshold_val = {threshold_val} (from self.threshold = {self.threshold})")

            num_channels, z_max, y_max, x_max = output_arrays[tgt_name].shape

            # Identify segmentation types based on channel count
            is_multiclass = num_channels > 2
            is_binary = num_channels == 2

            # Find the target config from our list
            target_config = next((t for t in self.targets if t.get('name') == tgt_name), None)

            # Determine processing modes based on segmentation type and flags
            # Multiclass segmentation with argmax (when not saving probability maps)
            compute_argmax = not self.save_probability_maps and is_multiclass

            # Binary segmentation handling - always extract channel 1 and apply sigmoid
            extract_binary = is_binary

            if compute_argmax and self.verbose:
                print(f"Will compute argmax over {num_channels} channels for {tgt_name}")

            if extract_binary and self.verbose:
                if self.save_probability_maps:
                    print(
                        f"Binary segmentation detected: will use both channels with softmax and save probabilities")
                else:
                    print(f"Binary segmentation detected: will use both channels with softmax and threshold")

            # Get the PyTorch tensors if provided, otherwise create them now
            if tensor_dict and tgt_name in tensor_dict:
                output_tensor_full, count_tensor_full = tensor_dict[tgt_name]
                if self.verbose:
                    print(f"Using GPU tensors from tensor_dict for {tgt_name}")
            else:
                # Convert to PyTorch tensors if not already provided
                # Need to copy to make them writable and avoid warnings
                output_tensor_full = torch.from_numpy(output_arrays[tgt_name].copy()).to(device)
                # Convert uint8 count array to PyTorch tensor - conversion to float happens in normalize_chunk_torch
                count_tensor_full = torch.from_numpy(count_arrays[tgt_name].copy()).to(device)
                if self.verbose:
                    print(f"Created new GPU tensors for {tgt_name}")

            chunk_size = 256

            # Only use tqdm progress bar on rank 0
            z_range = range(0, z_max, chunk_size)
            if self.rank == 0:
                z_range = tqdm(z_range, desc=f"Finalizing {tgt_name}")

            for z_start in z_range:
                z_end = min(z_start + chunk_size, z_max)
                z_range_size = z_end - z_start

                # Get the slices for this region
                count_tensor = count_tensor_full[z_start:z_end]
                output_tensor = output_tensor_full[:, z_start:z_end]

                # Determine processing approach based on configuration
                if compute_argmax:
                    # Multiclass case with argmax - compute argmax over channels
                    if self.verbose:
                        print(f"Computing argmax over {output_tensor.shape[0]} channels using torch.argmax")

                    # Create safe count tensor (avoid division by zero)
                    safe_count_tensor = torch.clamp(count_tensor, min=1.0)

                    # Normalize directly in PyTorch using broadcasting
                    normalized_tensor = output_tensor / safe_count_tensor.unsqueeze(0)

                    # Compute argmax along channel dimension (dim=0)
                    argmax_tensor = torch.argmax(normalized_tensor, dim=0)

                    # Reshape to add channel dimension and convert to uint8
                    # Create a new tensor with the correct shape and type
                    dest_tensor = torch.zeros((1,) + argmax_tensor.shape,
                                              dtype=torch.uint8,
                                              device=device)
                    dest_tensor[0] = argmax_tensor

                    # Move to CPU for transfer to zarr
                    dest_cpu = dest_tensor.cpu().numpy()

                    # Add detailed diagnostics before writing to zarr
                    if self.verbose:
                        print(f"About to write to zarr array - diagnostics:")
                        print(f"  - dest_cpu shape: {dest_cpu.shape}")
                        print(f"  - final_arrays[{tgt_name}] shape: {final_arrays[tgt_name].shape}")
                        print(f"  - dest_cpu has non-zero values: {np.any(dest_cpu > 0)}, max={np.max(dest_cpu)}")

                    # Copy to the zarr array
                    final_arrays[tgt_name][:, z_start:z_end] = dest_cpu

                    if self.verbose:
                        print(f"Copied results to zarr: source max={torch.max(dest_tensor).item()}")

                elif extract_binary:
                    # Binary segmentation case - use both channels and apply softmax
                    if self.verbose:
                        print(f"Processing binary segmentation: using both channels and applying softmax")

                    # Debug the original output tensor
                    if self.verbose and self.rank == 0:
                        print(f"Binary output channels: {output_tensor.shape[0]}")
                        print(
                            f"Channel 0 (background) range: {torch.min(output_tensor[0]).item():.4f} to {torch.max(output_tensor[0]).item():.4f}")
                        print(
                            f"Channel 1 (foreground) range: {torch.min(output_tensor[1]).item():.4f} to {torch.max(output_tensor[1]).item():.4f}")

                    # Create safe count (avoid division by zero) using torch
                    safe_count_tensor = torch.clamp(count_tensor, min=1.0)

                    # Normalize logits by count array (both channels)
                    normalized_tensor = output_tensor / safe_count_tensor.unsqueeze(0)

                    # Debug normalized tensor
                    if self.verbose and self.rank == 0:
                        print(
                            f"Normalized tensor range: {torch.min(normalized_tensor).item():.4f} to {torch.max(normalized_tensor).item():.4f}")

                    # Apply softmax across channel dimension (0) to get proper probabilities for both classes
                    softmax_tensor = torch.nn.functional.softmax(normalized_tensor, dim=0)

                    # Extract foreground probability (class 1)
                    foreground_prob = softmax_tensor[1]

                    # Debug softmax output
                    if self.verbose and self.rank == 0:
                        # Check raw normalized tensor values before softmax
                        print(f"Normalized tensor stats before softmax:")
                        print(
                            f"  - Background channel: min={torch.min(normalized_tensor[0]).item():.4f}, max={torch.max(normalized_tensor[0]).item():.4f}, mean={torch.mean(normalized_tensor[0]).item():.4f}")
                        print(
                            f"  - Foreground channel: min={torch.min(normalized_tensor[1]).item():.4f}, max={torch.max(normalized_tensor[1]).item():.4f}, mean={torch.mean(normalized_tensor[1]).item():.4f}")

                        # Show the channel difference - positive means foreground is predicted
                        diff = normalized_tensor[1] - normalized_tensor[0]
                        print(
                            f"  - Channel difference (fg-bg): min={torch.min(diff).item():.4f}, max={torch.max(diff).item():.4f}, mean={torch.mean(diff).item():.4f}")

                        # See how many pixels are near the decision boundary
                        boundary_pixels = ((diff > -0.1) & (diff < 0.1)).sum().item()
                        total_pixels = diff.numel()
                        print(
                            f"  - Pixels near decision boundary (±0.1): {boundary_pixels} ({boundary_pixels / total_pixels * 100:.2f}%)")

                        # Show the softmax results
                        print(f"Softmax results:")
                        print(
                            f"  - Background range: {torch.min(softmax_tensor[0]).item():.4f} to {torch.max(softmax_tensor[0]).item():.4f}")
                        print(
                            f"  - Foreground range: {torch.min(softmax_tensor[1]).item():.4f} to {torch.max(softmax_tensor[1]).item():.4f}")

                        # Show histogram of foreground probabilities
                        hist = torch.histc(foreground_prob, bins=10, min=0.0, max=1.0)
                        bin_edges = torch.linspace(0, 1, 11)
                        print(f"Foreground probability histogram:")
                        for i in range(10):
                            print(f"  - {bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}: {hist[i].item():.0f} pixels")

                    if self.save_probability_maps:
                        # When save_probability_maps is True, save foreground probability and binary mask
                        # Use argmax for binary mask for consistent boundary handling

                        # Save just the foreground probability channel (index 1)
                        prob_tensor = (softmax_tensor[1] * 255).to(torch.uint8)

                        # Generate binary mask with argmax for maximum consistency
                        # This is how nnUNet creates binary segmentations - argmax across channels
                        binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                        binary_tensor = binary_mask * 255  # Scale to 0/255 for visualization

                        # Create 2-channel output: foreground prob, binary mask
                        dest_tensor = torch.zeros((2,) + binary_tensor.shape,
                                                  dtype=torch.uint8,
                                                  device=device)
                        dest_tensor[0] = prob_tensor  # Channel 0: foreground probability
                        dest_tensor[1] = binary_tensor  # Channel 1: binary mask (0/255)

                        if self.verbose and self.rank == 0:
                            print(f"Saving 2-channel binary segmentation output:")
                            print(
                                f"  - Channel 0: Foreground probabilities (0-255), range: {torch.min(prob_tensor).item()}-{torch.max(prob_tensor).item()}")
                            print(
                                f"  - Channel 1: Binary mask (0/255), using argmax, unique values: {torch.unique(binary_tensor).cpu().numpy()}")
                            print(f"  - Final tensor shape: {dest_tensor.shape}")
                    else:
                        # When save_probability_maps is False: save only binary mask
                        # Use argmax for consistency with nnUNet approach
                        binary_mask = torch.argmax(softmax_tensor, dim=0).to(torch.uint8)
                        binary_tensor = binary_mask * 255  # Scale to 0/255
                        dest_tensor = binary_tensor.unsqueeze(0)  # Add channel dimension for single-channel output

                        if self.verbose:
                            print(f"Saving only binary mask from argmax (single channel)")

                    # Move to CPU for transfer to zarr
                    dest_cpu = dest_tensor.cpu().numpy()

                    # Debug the data being written to zarr
                    if self.verbose and self.rank == 0:
                        print(f"Writing to zarr array: shape={dest_cpu.shape}, dtype={dest_cpu.dtype}")
                        if dest_cpu.shape[0] == 2:  # Binary segmentation with probabilities
                            print(f"  - Channel 0 (probabilities) range: {np.min(dest_cpu[0])}-{np.max(dest_cpu[0])}")
                            print(f"  - Channel 1 (binary mask) range: {np.min(dest_cpu[1])}-{np.max(dest_cpu[1])}")
                            print(f"  - Unique values in binary mask: {np.unique(dest_cpu[1])}")

                    # Write the result to the final array
                    final_arrays[tgt_name][:, z_start:z_end] = dest_cpu

                else:
                    # Standard case for multiclass segmentation (saving full probability maps)
                    # Use channel count from the output tensor
                    c = output_tensor.shape[0]

                    # Debugging info
                    if self.verbose:
                        output_max = torch.max(output_tensor).item()
                        count_max = torch.max(count_tensor).item()
                        count_min = torch.min(count_tensor).item()
                        print(
                            f"Before normalization - output max: {output_max:.4f}, count range: {count_min:.4f}-{count_max:.4f}")
                        print(f"Processing segmentation with {c} channels")

                    # Create safe count tensor (avoid division by zero)
                    safe_count_tensor = torch.clamp(count_tensor, min=1.0)

                    # Normalize directly in PyTorch using broadcasting
                    normalized_tensor = output_tensor / safe_count_tensor.unsqueeze(0)

                    # Apply softmax to get proper probabilities
                    softmax_tensor = torch.nn.functional.softmax(normalized_tensor, dim=0)

                    # Convert to uint8 (0-255 range)
                    dest_tensor = (softmax_tensor * 255).to(torch.uint8)

                    # Apply threshold if provided
                    if threshold_val is not None:
                        # Convert threshold to uint8 scale
                        threshold_uint8 = int(threshold_val * 255)
                        dest_tensor = (dest_tensor >= threshold_uint8).to(torch.uint8) * 255

                    # Move to CPU for transfer to zarr
                    dest_cpu = dest_tensor.cpu().numpy()

                    # Copy the results back to the zarr array
                    final_arrays[tgt_name][:, z_start:z_end] = dest_cpu

                # Clean up temporary tensors
                del normalized_tensor
                if (z_start // chunk_size) % 4 == 0:
                    # Cleanup GPU memory
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            # Cleanup full tensors
            del output_tensor_full
            del count_tensor_full

            # Final cleanup
            torch.cuda.empty_cache()

    def _process_model_outputs(self, outputs, positions):
        """
        Process model outputs and submit them to the writer queue.
        """
        # For nnUNet, the output is a tensor, not a dict - convert it
        if torch.is_tensor(outputs):
            # Convert single tensor output to dict based on targets
            processed_outputs = {}
            for target in self.targets:
                tgt_name = target.get("name")
                processed_outputs[tgt_name] = outputs

            if self.verbose:
                print(f"Converted tensor output to dict with keys: {list(processed_outputs.keys())}")
                print(f"Tensor shape: {outputs.shape}")
        else:
            # Already in dict format
            processed_outputs = outputs
            if self.verbose:
                print(f"Output already in dict format with keys: {list(processed_outputs.keys())}")

        # Get the expected patch shape
        patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

        # Process each patch and add to the queue
        for i, pos in enumerate(positions):
            for target in self.targets:
                tgt_name = target.get("name")
                # Get the prediction tensor for this target
                try:
                    # Get model output as tensor first (keeping on CPU)
                    pred_tensor = processed_outputs[tgt_name][i].cpu()

                    # Store raw logits directly, without applying activation
                    # We'll apply softmax only once, after blending all patches
                    if tgt_name == "segmentation":
                        # Get number of channels
                        num_channels = pred_tensor.shape[0]

                        # For logits, keep as float for proper blending
                        # Note: we're storing the raw network outputs directly
                        pred_uint8 = pred_tensor.numpy()

                        if self.verbose and i < 3:
                            print(f"Storing raw logits for {tgt_name} with {num_channels} channels")
                            print(f"  - Value range: {pred_uint8.min():.4f}-{pred_uint8.max():.4f}")
                    else:
                        # For non-segmentation targets, scale to uint8 range (0-255)
                        pred_uint8 = (pred_tensor * 255).to(torch.uint8).numpy()

                    # Print shape information for the first few patches
                    if self.verbose and i < 3:
                        print(
                            f"Patch for {tgt_name} - output shape: {pred_uint8.shape}, expected shape: (C, {patch_size_tuple[0]}, {patch_size_tuple[1]}, {patch_size_tuple[2]})")
                        print(f"  - Value range: {pred_uint8.min()}-{pred_uint8.max()}, dtype: {pred_uint8.dtype}")

                    # Add to the writer queue as uint8
                    self.writer_queue.put((pred_uint8, pos, tgt_name))
                except Exception as e:
                    print(f"Error processing output for target {tgt_name}, position {pos}: {str(e)}")
                    print(f"Output keys: {list(processed_outputs.keys())}")
                    if tgt_name in processed_outputs:
                        print(f"Output shape for {tgt_name}: {processed_outputs[tgt_name].shape}")
                    else:
                        print(f"Target {tgt_name} not found in outputs")
                    raise

    def infer(self):

        # Verify input path exists and is a valid zarr array
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

        try:
            # Try to open the input zarr array to check if it's valid
            zarr.open(self.input_path, mode='r')
        except Exception as e:
            raise ValueError(f"Error opening input zarr array at {self.input_path}: {str(e)}")

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        store_path = os.path.join(self.output_path, "predictions.zarr")

        # Create a temporary directory for memory-mapped arrays
        temp_dir = os.path.join(self.output_path, f"temp_memmap_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)

        # Load the nnUNet model
        self.model_info = self._load_nnunet_model()
        network = self.model_info['network']

        # Only rank 0 creates the Zarr store and datasets
        if self.rank == 0:
            if os.path.isdir(store_path):
                raise FileExistsError(f"Zarr store '{store_path}' already exists.")

            # Create a temporary dataset to determine the full output shape
            dataset_temp = InferenceDataset(
                input_path=self.input_path,
                targets=self.targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all,
                verbose=self.verbose,
                cache_size=self.cache_size,
                max_cache_bytes=self.max_cache_bytes
            )

            # Get the shape of the input array for output shape determination
            input_shape = dataset_temp.input_shape
            if len(input_shape) == 3:  # 3D array (Z,Y,X)
                z_max, y_max, x_max = input_shape
            elif len(input_shape) == 4:  # 4D array (C,Z,Y,X)
                _, z_max, y_max, x_max = input_shape

            # Calculate total number of patches
            steps = compute_steps_for_sliding_window_tuple(
                (z_max, y_max, x_max), self.patch_size, self.tile_step_size
            )
            z_steps, y_steps, x_steps = steps
            total_patches = len(z_steps) * len(y_steps) * len(x_steps)

            if self.verbose:
                print(f"Total patches to process: {total_patches}")
                print(f"Z steps: {len(z_steps)}, Y steps: {len(y_steps)}, X steps: {len(x_steps)}")

            # Define compression for zarr arrays
            compressor = Blosc(cname='zstd', clevel=3)

            # Setup data structures for each target
            for target in self.targets:
                tgt_name = target.get("name")
                c = target.get("channels")

                # For nnUNet, get number of segmentation heads
                num_model_outputs = self.model_info.get('num_seg_heads', 1)
                # For multiclass segmentation, update the channel count
                if num_model_outputs > 2 and tgt_name == "segmentation":
                    c = num_model_outputs
                    target["channels"] = c
                    target["nnunet_output_channels"] = c
                    print(f"Detected multiclass model with {c} classes from model_info")

                # Convert patch_size to tuple if it's a list
                patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

                if self.verbose:
                    print(
                        f"Will use memory-mapped arrays for intermediate storage with shape ({c}, {patch_size_tuple[0]}, {patch_size_tuple[1]}, {patch_size_tuple[2]})")

            # Create final output zarr arrays
            final_zarr = zarr.open(store_path, mode='w')
            output_arrays = {}
            count_arrays = {}
            final_arrays = {}

            # Create sum, count, and final arrays for each target
            for target in self.targets:
                tgt_name = target.get("name")
                c = target.get("channels")

                # Always use 4D output with channel dimension for consistency
                out_shape = (c, z_max, y_max, x_max)
                chunks = (c, self.patch_size[0], self.patch_size[1], self.patch_size[2])

                if self.verbose:
                    print(f"Creating arrays for target '{tgt_name}': shape={out_shape}, chunks={chunks}")

                # Create memory-mapped arrays for blending
                # Convert patch_size to tuple if it's a list
                patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size

                # Create temporary directory for memory-mapped files
                temp_dir = os.path.join(self.output_path, f"temp_memmap_{uuid.uuid4().hex}")
                os.makedirs(temp_dir, exist_ok=True)

                # Create file paths for memory-mapped arrays
                sum_file = os.path.join(temp_dir, f"{tgt_name}_sum.npy")
                count_file = os.path.join(temp_dir, f"{tgt_name}_count.npy")

                if self.verbose:
                    print(f"Creating memory-mapped sum array with shape {out_shape}")
                    print(f"Creating memory-mapped count array with shape {(z_max, y_max, x_max)}")

                # Create memory-mapped array for output sum
                output_arrays[tgt_name] = np.memmap(
                    sum_file,
                    dtype='float32',
                    mode='w+',
                    shape=out_shape
                )

                # Create memory-mapped array for count
                count_arrays[tgt_name] = np.memmap(
                    count_file,
                    dtype='float32',
                    mode='w+',
                    shape=(z_max, y_max, x_max)
                )

                # Initialize arrays to zero
                output_arrays[tgt_name][:] = 0
                count_arrays[tgt_name][:] = 0

                # Flush to ensure initialization is written to disk
                output_arrays[tgt_name].flush()
                count_arrays[tgt_name].flush()

                # Create final output array with appropriate format
                # Get number of model outputs from model info
                num_model_outputs = 1
                if self.model_info is not None:
                    # For nnUNet, get number of segmentation heads
                    num_model_outputs = self.model_info.get('num_seg_heads', 1)

                # Determine the output format based on segmentation type and configuration
                is_multiclass = num_model_outputs > 2
                is_binary = num_model_outputs == 2

                # Case 1: Multiclass segmentation with argmax (single channel, uint8)
                if not self.save_probability_maps and is_multiclass and tgt_name == "segmentation":
                    # For debugging
                    if self.verbose:
                        print(f"Using optimized single-channel output for multiclass segmentation")
                        print(f"  - save_probability_maps: {self.save_probability_maps}")
                        print(f"  - num_model_outputs: {num_model_outputs}")
                        print(f"  - activation: {target.get('activation', 'softmax')}")

                    # Multiclass case - use uint8 for class indices after argmax
                    output_shape = (1, z_max, y_max, x_max)
                    output_chunks = (1,) + patch_size_tuple
                    output_dtype = 'uint8'

                    if self.verbose:
                        print(f"Creating argmax output array with shape {output_shape}, dtype {output_dtype}")

                    # Create the zarr array
                    zarr_array = final_zarr.create_dataset(
                        name=tgt_name,
                        shape=output_shape,
                        chunks=output_chunks,
                        dtype=output_dtype,
                        compressor=compressor,
                        fill_value=0
                    )

                    # Always use direct zarr array to ensure data is properly persisted
                    final_arrays[tgt_name] = zarr_array
                    if self.verbose:
                        print(f"Using direct zarr array for {tgt_name} with NO caching")

                # Case 2: Binary segmentation - output format depends on save_probability_maps
                elif is_binary and tgt_name == "segmentation":
                    if self.save_probability_maps:
                        # When save_probability_maps is True: 2-channel output
                        if self.verbose:
                            print(f"Using 2-channel output for binary segmentation")
                            print(f"  - num_model_outputs: {num_model_outputs}")
                            print(f"  - activation: {target.get('activation', 'sigmoid')}")
                            print(f"  - channel 0: probability values (0-255)")
                            print(f"  - channel 1: thresholded binary mask (0/255)")

                        # 2-channel output format
                        output_shape = (2, z_max, y_max, x_max)
                        output_chunks = (2,) + patch_size_tuple
                    else:
                        # When save_probability_maps is False: single-channel thresholded output
                        if self.verbose:
                            print(f"Using single-channel output for binary segmentation")
                            print(f"  - num_model_outputs: {num_model_outputs}")
                            print(f"  - activation: {target.get('activation', 'sigmoid')}")
                            print(f"  - output: thresholded binary mask (0/255)")

                        # Single-channel output format
                        output_shape = (1, z_max, y_max, x_max)
                        output_chunks = (1,) + patch_size_tuple

                    # Both formats use uint8
                    output_dtype = 'uint8'

                    if self.verbose:
                        print(
                            f"Creating binary segmentation output array with shape {output_shape}, dtype {output_dtype}")

                    # Create the zarr array
                    zarr_array = final_zarr.create_dataset(
                        name=tgt_name,
                        shape=output_shape,
                        chunks=output_chunks,
                        dtype=output_dtype,
                        compressor=compressor,
                        fill_value=0
                    )

                    # Always use direct zarr array to ensure data is properly persisted
                    final_arrays[tgt_name] = zarr_array
                    if self.verbose:
                        print(f"Using direct zarr array for {tgt_name} with NO caching")

                # Case 3: Standard probability maps with all channels (default case)
                else:
                    # Standard probability maps with uint8 dtype to save space
                    if self.verbose:
                        print(f"Creating probability map array with shape {out_shape}, dtype uint8")

                    # Create the zarr array
                    zarr_array = final_zarr.create_dataset(
                        name=tgt_name,
                        shape=out_shape,
                        chunks=chunks,  # Use chunks that match patch size
                        dtype='uint8',  # Using uint8 for all outputs
                        compressor=compressor,
                        fill_value=0
                    )

                    # Always use direct zarr array to ensure data is properly persisted
                    final_arrays[tgt_name] = zarr_array
                    if self.verbose:
                        print(f"Using direct zarr array for {tgt_name} with NO caching")

            # Create memory-mapped arrays for each target
            # First, estimate the number of patches we'll need to process

            dataset_temp = InferenceDataset(
                input_path=self.input_path,
                targets=self.targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all,
                verbose=self.verbose,
                cache_size=self.cache_size,
                max_cache_bytes=self.max_cache_bytes
            )

            # Calculate total number of patches
            # Add 10% extra padding to be safe
            input_shape = dataset_temp.input_shape
            if len(input_shape) == 3:  # 3D array (Z,Y,X)
                z_max, y_max, x_max = input_shape
            elif len(input_shape) == 4:  # 4D array (C,Z,Y,X)
                _, z_max, y_max, x_max = input_shape

            steps = compute_steps_for_sliding_window_tuple(
                (z_max, y_max, x_max), self.patch_size, self.tile_step_size
            )
            z_steps, y_steps, x_steps = steps
            num_expected_patches = int(len(z_steps) * len(y_steps) * len(x_steps) * 1.1)  # 10% extra

            # Create memory-mapped arrays for each target
            target_arrays = {}
            for target in self.targets:
                tgt_name = target.get("name")
                # Get the channel dimension
                channels = target.get("channels")

                # Check if the nnUNet model outputs multilabel segmentation (num_classes > 2)
                if self.model_info is not None:
                    # For nnUNet, get number of segmentation heads
                    num_model_outputs = self.model_info.get('num_seg_heads', 1)
                    # For multiclass segmentation, update the channel count
                    if num_model_outputs > 2 and tgt_name == "segmentation":
                        channels = num_model_outputs

                # Create memory-mapped arrays for this target
                # If we're storing patches for a binary segmentation with 1 channel,
                # all ranks need to use the same format
                if (tgt_name == "segmentation" and
                        target.get("activation", "softmax") == "sigmoid"):
                    # For binary segmentation, always use the same number of channels for writing patches
                    # This ensures all ranks use the same format
                    patch_shape = (target.get("channels"),) + tuple(self.patch_size)
                    if self.verbose:
                        print(f"Binary segmentation: using patch shape {patch_shape} for {tgt_name}")
                else:
                    # Standard patch shape for other cases
                    patch_shape = (channels,) + tuple(self.patch_size)

                array_tuple = self._create_memmap_arrays(
                    target_name=tgt_name,
                    num_expected_patches=num_expected_patches,
                    patch_shape=patch_shape,
                    temp_dir=temp_dir
                )

                # Store as a list to allow for merging patches from multiple ranks later
                target_arrays[tgt_name] = [array_tuple]

            # Start writer worker threads
            writer_threads = []
            for worker_id in range(self.num_write_workers):
                thread = threading.Thread(
                    target=self._writer_worker,
                    args=(target_arrays, self.writer_queue, worker_id)
                )
                thread.daemon = True
                thread.start()
                writer_threads.append(thread)

            # Create the dataset for inference
            dataset = InferenceDataset(
                input_path=self.input_path,
                targets=self.targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all,
                verbose=self.verbose,
                cache_size=self.cache_size,
                max_cache_bytes=self.max_cache_bytes
            )

            # Create a dataloader with DistributedSampler if using DDP
            if dist.is_initialized():
                sampler = DistributedSampler(dataset)
                if self.verbose:
                    print(f"Rank {self.rank}: Using DistributedSampler with {len(dataset)} total samples")
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    num_workers=self.num_dataloader_workers,
                    pin_memory=True,
                    prefetch_factor=8
                )

                # Set the epoch for the sampler to ensure proper shuffling
                sampler.set_epoch(0)
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_dataloader_workers,
                    pin_memory=True,
                    prefetch_factor=8
                )

            # Run inference
            print(f"Rank {self.rank}: Running inference with {len(dataloader)} batches...")

            # Use tqdm only on rank 0 for verbosity
            batch_iter = dataloader
            if self.rank == 0:
                batch_iter = tqdm(dataloader, desc=f"Inference (Rank {self.rank})")

            for batch in batch_iter:
                # Get the batch data
                images = batch["image"].to(self.device_str)
                indices = batch["index"]

                # Get positions for this batch
                positions = [dataset.all_positions[i.item()] for i in indices]

                # Run inference with mirroring TTA if enabled
                if self.use_mirroring:
                    # Process all output targets using the model_info for TTA
                    outputs = run_inference(
                        model_info=self.model_info,
                        input_tensor=images,
                        max_tta_combinations=3,  # Use only the 3 primary axes for TTA
                        parallel_tta_multiplier=None,
                        rank=self.rank  # Pass rank to control logging
                    )
                    # Print shape information for debugging if verbose (only from rank 0)
                    if self.verbose and self.rank == 0 and isinstance(outputs, torch.Tensor):
                        print(f"Output shape: {outputs.shape}, type: {type(outputs)}")
                        print(f"Output Dtype: {outputs.dtype}")
                else:
                    # Without TTA, run standard inference
                    if self.rank == 0:
                        print("Running inference with no TTA")
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        outputs = network(images)

                self._process_model_outputs(outputs, positions)

            # Wait for all writer tasks to complete
            self.writer_queue.join()

            # Signal writer threads to terminate
            for _ in range(self.num_write_workers):
                self.writer_queue.put(None)

            # Wait for all writer threads to finish
            for thread in writer_threads:
                thread.join()

            print(f"Rank {self.rank}: All writer threads completed")

            print(f"Rank {self.rank}: All inference and writing complete.")

            # If using DDP, wait for all ranks to complete inference
            if dist.is_initialized():
                # Check target arrays before barrier
                for tgt_name in target_arrays:
                    for array_tuple in target_arrays[tgt_name]:
                        patches_array, positions_list, counter, file_path = array_tuple
                        num_processed = counter['value']
                        print(f"Rank {self.rank}: Processed {num_processed} patches for {tgt_name}")

                print(f"Rank {self.rank}: Waiting for all ranks to complete inference...")

                # Simple barrier - process group is already initialized with device ID
                try:
                    dist.barrier()
                    print(f"Rank {self.rank}: All ranks have completed inference, continuing...")
                except Exception as e:
                    print(f"Rank {self.rank}: Error in barrier: {e}")
                    import traceback
                    traceback.print_exc()

                # For DDP, gather patches from all ranks
                world_size = dist.get_world_size()
                if world_size > 1:
                    print(f"Rank {self.rank}: Gathering patches and positions from {world_size} ranks...")

                    # For each non-zero rank, receive its counter values and positions
                    for rank in range(1, world_size):
                        try:
                            # Process each target's memory-mapped arrays
                            for target in self.targets:
                                tgt_name = target.get("name")

                                # Set up a tensor to receive the counter value
                                counter_tensor = torch.zeros(1, dtype=torch.int64, device=self.device_str)

                                # Receive the counter value from this rank
                                dist.recv(counter_tensor, src=rank)
                                n_patches = counter_tensor.item()

                                print(
                                    f"Rank {self.rank}: Received counter value {n_patches} from rank {rank} for target {tgt_name}")

                                if n_patches == 0:
                                    print(f"Rank {self.rank}: No patches found for rank {rank}, target {tgt_name}")
                                    continue

                                # Receive position data for each patch
                                # Use int32 to match the sender (positions are relatively small)
                                positions_tensor = torch.zeros((n_patches, 3), dtype=torch.int32,
                                                               device=self.device_str)
                                dist.recv(positions_tensor, src=rank)

                                # Find the corresponding patches directory for this rank
                                rank_temp_dir = None
                                for entry in os.listdir(self.output_path):
                                    if entry.startswith(f"temp_memmap_rank{rank}_"):
                                        rank_temp_dir = os.path.join(self.output_path, entry)
                                        break

                                if rank_temp_dir and os.path.exists(rank_temp_dir):
                                    print(f"Rank {self.rank}: Processing patches from {rank_temp_dir}")

                                    # Find the corresponding patches files
                                    patches_files = [f for f in os.listdir(rank_temp_dir)
                                                     if f.startswith(f"{tgt_name}_patches_") and f.endswith(".npy")]

                                    if not patches_files:
                                        print(
                                            f"Rank {self.rank}: Missing patch files for rank {rank}, target {tgt_name}")
                                        continue

                                    # Open the memory-mapped arrays for patches
                                    patches_path = os.path.join(rank_temp_dir, patches_files[0])

                                    # Get the channel dimension from the output target
                                    target_config = next((t for t in self.targets if t.get("name") == tgt_name), None)
                                    channels = target_config.get("channels") if target_config else 2

                                    patch_shape = (channels,) + tuple(self.patch_size)

                                    if self.verbose:
                                        print(f"Using patch shape {patch_shape} for rank {rank}, target {tgt_name}")

                                    # Load the memmap arrays using the counter value received
                                    patches_array = np.memmap(
                                        patches_path,
                                        dtype='uint8',
                                        mode='r',
                                        shape=(n_patches,) + patch_shape
                                    )

                                    # Convert positions tensor to list of tuples
                                    positions_list = [None] * n_patches
                                    for i in range(n_patches):
                                        z = positions_tensor[i, 0].item()
                                        y = positions_tensor[i, 1].item()
                                        x = positions_tensor[i, 2].item()
                                        positions_list[i] = (z, y, x)

                                    # Create a counter dict to match the expected format
                                    counter = {'value': n_patches, 'lock': threading.Lock()}

                                    # Add to the existing target arrays for this target
                                    if tgt_name not in target_arrays:
                                        target_arrays[tgt_name] = []

                                    # Store with the new format
                                    target_arrays[tgt_name].append((
                                        patches_array,
                                        positions_list,
                                        counter,
                                        patches_path
                                    ))

                                    print(
                                        f"Rank {self.rank}: Added {n_patches} patches from rank {rank} for target {tgt_name}")
                                else:
                                    print(f"Rank {self.rank}: Could not find temp directory for rank {rank}")

                            print(f"Rank {self.rank}: Successfully processed patches from rank {rank}")
                        except Exception as e:
                            print(f"Rank {self.rank}: Error processing patches from rank {rank}: {str(e)}")
                            import traceback
                            traceback.print_exc()

            # Use the memory-mapped patch arrays for blending
            # This returns PyTorch tensors we can use directly in finalize
            tensor_dict = self._blend_patches(target_arrays, output_arrays, count_arrays)

            # Pass the PyTorch tensors to _finalize_arrays to avoid unnecessary transfers
            self._finalize_arrays(output_arrays, count_arrays, final_arrays, tensor_dict=tensor_dict)

            # Clean up temporary files
            print(f"Rank {self.rank}: Cleaning up temporary files...")

            # Get a list of target names to avoid modifying the dictionary during iteration
            target_names = list(output_arrays.keys())

            # Close the memory-mapped arrays
            for tgt_name in target_names:
                # Delete to close the memory-mapped files
                del output_arrays[tgt_name]
                del count_arrays[tgt_name]

                # Clean up all patch arrays
                if tgt_name in target_arrays:
                    for patch_data_tuple in target_arrays[tgt_name]:
                        patches_array, positions_list, _, _ = patch_data_tuple
                        del patches_array
                        # No need to explicitly delete positions_list as it's a regular Python list

            # Remove the temporary directory with memory-mapped files
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

            # Signal other ranks that rank 0 is done with their data
            if dist.is_initialized():
                print(f"Rank {self.rank}: Signaling completion to other ranks...")
                # Simple barrier - process group is already initialized with device ID
                try:
                    dist.barrier()
                    print(f"Rank {self.rank}: Final barrier passed successfully")
                except Exception as e:
                    print(f"Rank {self.rank}: Error in final barrier: {e}")
                    import traceback
                    traceback.print_exc()

            # Report cache stats if verbose
            if self.verbose:
                print(f"Rank {self.rank}: Cache statistics for final arrays:")
                for tgt_name, array in final_arrays.items():
                    if isinstance(array, ZarrArrayLRUCache):
                        stats = array.get_stats()
                        hit_rate = stats['hit_rate']
                        print(
                            f"  - {tgt_name}: hits={stats['hits']}, misses={stats['misses']}, hit rate={hit_rate:.1f}%, cache size={stats['cache_size']}/{stats['max_size']}")

            print(f"Rank {self.rank}: Inference complete. Results saved to {store_path}")

        else:
            # For non-zero ranks in distributed mode
            if self.verbose:
                print(f"Rank {self.rank}: Starting distributed inference")

            # Load the model (all ranks need to load the model)
            self.model_info = self._load_nnunet_model()
            network = self.model_info['network']

            # Create the dataset for inference
            dataset = InferenceDataset(
                input_path=self.input_path,
                targets=self.targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all,
                verbose=self.verbose
            )

            # Create a dataloader with DistributedSampler
            sampler = DistributedSampler(dataset)
            if self.verbose:
                print(f"Rank {self.rank}: Using DistributedSampler with {len(dataset)} total samples")

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_dataloader_workers,
                pin_memory=True,
                prefetch_factor=8
            )

            # Set the epoch for the sampler to ensure proper shuffling
            sampler.set_epoch(0)

            # Create a temporary directory for this rank to store memory-mapped arrays
            temp_dir = os.path.join(self.output_path, f"temp_memmap_rank{self.rank}_{uuid.uuid4().hex}")
            os.makedirs(temp_dir, exist_ok=True)

            # First, estimate the number of patches we'll need to process
            dataset_temp = InferenceDataset(
                input_path=self.input_path,
                targets=self.targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all,
                verbose=self.verbose
            )

            # Calculate total number of patches
            input_shape = dataset_temp.input_shape
            if len(input_shape) == 3:  # 3D array (Z,Y,X)
                z_max, y_max, x_max = input_shape
            elif len(input_shape) == 4:  # 4D array (C,Z,Y,X)
                _, z_max, y_max, x_max = input_shape

            # Calculate number of patches with DistributedSampler
            # This will be different from rank 0 as each rank gets a subset
            steps = compute_steps_for_sliding_window_tuple(
                (z_max, y_max, x_max), self.patch_size, self.tile_step_size
            )
            z_steps, y_steps, x_steps = steps
            total_patches = len(z_steps) * len(y_steps) * len(x_steps)
            world_size = dist.get_world_size()
            patches_per_rank = (total_patches + world_size - 1) // world_size  # Ceiling division
            num_expected_patches = int(patches_per_rank * 1.1)  # Add 10% extra

            # Create memory-mapped arrays for each target
            target_arrays = {}
            for target in self.targets:
                tgt_name = target.get("name")
                # Get the channel dimension
                channels = target.get("channels")

                # Check if the nnUNet model outputs multilabel segmentation (num_classes > 2)
                if self.model_info is not None:
                    # For nnUNet, get number of segmentation heads
                    num_model_outputs = self.model_info.get('num_seg_heads', 1)
                    # For multiclass segmentation, update the channel count
                    if num_model_outputs > 2 and tgt_name == "segmentation":
                        channels = num_model_outputs
                        target["channels"] = channels
                        target["nnunet_output_channels"] = channels

                # Create memory-mapped arrays for this target
                patch_size_tuple = tuple(self.patch_size) if isinstance(self.patch_size, list) else self.patch_size
                patch_shape = (channels,) + patch_size_tuple

                array_tuple = self._create_memmap_arrays(
                    target_name=tgt_name,
                    num_expected_patches=num_expected_patches,
                    patch_shape=patch_shape,
                    temp_dir=temp_dir
                )

                # Store as a list to maintain consistency with rank 0's format
                target_arrays[tgt_name] = [array_tuple]

            # Start writer worker threads
            writer_threads = []
            writer_queue = queue.Queue()

            for worker_id in range(self.num_write_workers):
                thread = threading.Thread(
                    target=self._writer_worker,
                    args=(target_arrays, writer_queue, worker_id)
                )
                thread.daemon = True
                thread.start()
                writer_threads.append(thread)

            # Run inference
            if self.verbose:
                print(f"Rank {self.rank}: Running inference with {len(dataloader)} batches...")

            # Use tqdm only on rank 0 for verbosity
            batch_iter = dataloader
            if self.rank == 0:
                batch_iter = tqdm(dataloader, desc=f"Inference (Rank {self.rank})")

            for batch in batch_iter:
                # Get the batch data
                images = batch["image"].to(self.device_str)
                indices = batch["index"]

                # Get positions for this batch
                positions = [dataset.all_positions[i.item()] for i in indices]

                # Run inference with mirroring TTA if enabled
                if self.use_mirroring:
                    # Process all output targets using the model_info for TTA
                    outputs = run_inference(
                        model_info=self.model_info,
                        input_tensor=images,
                        max_tta_combinations=3,  # Use only the 3 primary axes for TTA
                        parallel_tta_multiplier=None
                    )
                else:
                    # Without TTA, run standard inference
                    with torch.no_grad(), torch.amp.autocast('cuda'):
                        outputs = network(images)

                # Process outputs and submit to the writer queue
                for i, pos in enumerate(positions):
                    for target in self.targets:
                        tgt_name = target.get("name")
                        # Get the prediction tensor for this target
                        if torch.is_tensor(outputs):
                            # Simple case - always keep all channels for consistency
                            pred = outputs[i].cpu().numpy()
                        else:
                            pred = outputs[tgt_name][i].cpu().numpy()

                        # Add to the writer queue
                        writer_queue.put((pred, pos, tgt_name))

            # Wait for all writer tasks to complete
            writer_queue.join()

            # Signal writer threads to terminate
            for _ in range(self.num_write_workers):
                writer_queue.put(None)

            # Wait for all writer threads to finish
            for thread in writer_threads:
                thread.join()

            if self.verbose:
                print(f"Rank {self.rank}: All inference and writing complete.")

            # Check the count of processed patches before barrier
            if dist.is_initialized():
                for tgt_name in target_arrays:
                    for array_tuple in target_arrays[tgt_name]:
                        patches_array, positions_list, counter, file_path = array_tuple
                        num_processed = counter['value']
                        print(f"Rank {self.rank}: Processed {num_processed} patches for {tgt_name}")

                print(f"Rank {self.rank}: All data processing complete, waiting at barrier...")

                # Make all processes wait here until all are done with inference
                try:
                    # Simple barrier - process group is already initialized with device ID
                    dist.barrier()
                    print(f"Rank {self.rank}: Passed barrier successfully")
                except Exception as e:
                    print(f"Rank {self.rank}: Error in barrier: {e}")
                    import traceback
                    traceback.print_exc()

            # Only rank 0 will blend the patches from all ranks
            # We need to share the counter values and positions with rank 0
            # This is done using PyTorch distributed communication (dist.send/dist.recv)
            # The memory-mapped patches are accessed from disk, and the positions are shared via tensors
            if dist.is_initialized() and self.rank != 0:
                # Share counter values and positions with rank 0 for each target
                for tgt_name in target_arrays:
                    for array_tuple in target_arrays[tgt_name]:
                        patches_array, positions_list, counter, file_path = array_tuple

                        # Get the counter value
                        counter_value = counter['value']

                        # Convert counter to tensor for sharing
                        counter_tensor = torch.tensor([counter_value], dtype=torch.int64, device=self.device_str)

                        # Share counter value with rank 0
                        dist.send(counter_tensor, dst=0)

                        # Share actual positions for each patch
                        if counter_value > 0:
                            # Convert positions_list to tensor for sharing
                            # Each position is (z, y, x) - pack into a single tensor
                            # Use int32 since positions are relatively small (max ~30,000)
                            positions_tensor = torch.zeros((counter_value, 3), dtype=torch.int32,
                                                           device=self.device_str)

                            # Fill the tensor with positions
                            for i in range(counter_value):
                                if positions_list[i] is not None:
                                    z, y, x = positions_list[i]
                                    positions_tensor[i, 0] = z
                                    positions_tensor[i, 1] = y
                                    positions_tensor[i, 2] = x

                            # Share positions with rank 0
                            dist.send(positions_tensor, dst=0)

                        print(
                            f"Rank {self.rank}: Shared counter value {counter_value} and positions for target {tgt_name}")

            # Wait for rank 0 to signal it's done with all data
            if dist.is_initialized():
                print(f"Rank {self.rank}: Waiting for final signal from rank 0...")
                # Simple barrier - process group is already initialized with device ID
                try:
                    dist.barrier()
                    print(f"Rank {self.rank}: Final barrier passed successfully, exiting")
                except Exception as e:
                    print(f"Rank {self.rank}: Error in final barrier: {e}")
                    import traceback
                    traceback.print_exc()

            # Now clean up
            if self.verbose:
                print(f"Rank {self.rank}: Cleaning up temporary files...")

            # Close and delete memory-mapped arrays
            # Get a list of target names to avoid modifying the dictionary during iteration
            target_names = list(target_arrays.keys())

            for tgt_name in target_names:
                # Close all arrays in the list
                for patches_array, positions_list, _, _ in target_arrays[tgt_name]:
                    # Close the memory-mapped patches array
                    # (positions_list is a regular Python list, no need to explicitly delete)
                    del patches_array

            # Remove the temporary directory with all memory-mapped files
            import shutil
            import glob

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            for f in glob.glob(os.path.join(self.output_path, "temp_memmap*")):
                shutil.rmtree(f)

            if self.verbose:
                print(f"Rank {self.rank}: Done.")

            if self.verbose:
                print(f"Rank {self.rank}: Done.")


# Main function for standalone execution
def main():
    import argparse
    import torch.distributed as dist

    # Initialize process group for DDP if running with torch.distributed
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        if local_rank == 0:
            print(f"Initializing distributed process group (local_rank={local_rank})")

        # Initialize process group without explicit device_id
        # torchrun already sets the necessary environment variables
        print(f"Rank {local_rank}: Initializing process group with NCCL backend")
        dist.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser(description='Run sequential nnUNet inference on a zarr array')
    parser.add_argument('--input', type=str, required=True, help='Path to input zarr array')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to nnUNet model folder')
    parser.add_argument('--fold', type=str, default='0', help='Model fold to use (default: 0)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth',
                        help='Checkpoint name (default: checkpoint_final.pth)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference (default: 4)')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='Step size as fraction of patch size (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (default: cuda)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_write_workers', type=int, default=4, help='Number of writer threads (default: 4)')
    parser.add_argument('--threshold', type=float, help='Optional threshold (0-100) for binarizing the output')
    parser.add_argument('--no_mirroring', action='store_true', help='Disable test time augmentation via mirroring')
    parser.add_argument('--no_probabilities', action='store_true',
                        help='Do not save probability maps, save argmax for multiclass segmentation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--edge_weight_boost', type=float, default=0,
                        help='Factor to boost weights at patch edges (0.0-1.0) to reduce boundary artifacts (default: 0)')
    parser.add_argument('--cache_size', type=int, default=256,
                        help='Number of zarr chunks to cache in memory (default: 256)')
    parser.add_argument('--max_cache_bytes', type=float, default=4.0,
                        help='Maximum memory in GB to use for zarr cache (default: 4.0)')

    args = parser.parse_args()

    # Convert fold to int if numeric
    try:
        fold = int(args.fold)
    except ValueError:
        fold = args.fold  # Keep as string if not numeric (e.g., "all")

    # Set the device explicitly if using DDP
    if local_rank != -1:
        # When using torchrun, each process should use the GPU assigned by LOCAL_RANK
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # Use modulo in case there are more ranks than GPUs
            device = f"cuda:{local_rank % device_count}"
            print(f"Rank {local_rank}: Assigned to device {device} (out of {device_count} GPUs)")
        else:
            # Fallback to CPU if no CUDA devices
            device = "cpu"
            print(f"Rank {local_rank}: No CUDA devices available, using CPU")
    else:
        device = args.device

    # Adjust workers based on world size for DDP
    if dist.is_initialized():
        world_size = dist.get_world_size()
        dataloader_workers = max(1, args.num_workers // world_size)
        write_workers = max(1, args.num_write_workers // world_size)
        if local_rank == 0:
            print(f"Adjusting workers for {world_size} processes:")
            print(f"  - Dataloader workers: {args.num_workers} -> {dataloader_workers} per process")
            print(f"  - Write workers: {args.num_write_workers} -> {write_workers} per process")
    else:
        dataloader_workers = args.num_workers
        write_workers = args.num_write_workers

    # Create inference handler
    inference = SequentialZarrNNUNetInference(
        input_path=args.input,
        output_path=args.output,
        model_folder=args.model_folder,
        fold=fold,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        step_size=args.step_size,
        num_dataloader_workers=dataloader_workers,
        num_write_workers=write_workers,
        device=device,
        threshold=args.threshold,
        use_mirroring=not args.no_mirroring,
        save_probability_maps=not args.no_probabilities,
        verbose=args.verbose and (local_rank == -1 or local_rank == 0),  # Only be verbose on rank 0
        edge_weight_boost=args.edge_weight_boost,  # Pass the edge weight boost parameter
        cache_size=args.cache_size,  # Pass the cache size parameter
        max_cache_bytes=args.max_cache_bytes  # Pass the max cache bytes parameter
    )

    # Run inference
    inference.infer()

    # Cleanup distributed process group
    if local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()