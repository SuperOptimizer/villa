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

# Import nnUNet model loader
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
        
        # Default output target configuration if not provided
        # For nnUNet, we expect 2 channels (background and foreground)
        # But we'll only save the foreground channel (channel 1)
        self.output_targets = output_targets or {
            "segmentation": {
                "channels": 1,  # We'll only save 1 channel after extracting the foreground
                "activation": "softmax",
                "nnunet_output_channels": 2  # nnUNet outputs 2 channels (bg, fg)
            }
        }
        
        # nnUNet always produces 2 channels (bg, fg) for binary segmentation
        self.nnunet_foreground_channel = 1  # Second channel (index 1) is foreground
        
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
            model_info = load_model(
                model_folder=self.model_folder,
                fold=self.fold,
                checkpoint_name=self.checkpoint_name,
                device=self.device_str
            )
            
            # Use the model's patch size if none was specified
            if self.patch_size is None:
                self.patch_size = model_info['patch_size']
                print(f"Using model's patch size: {self.patch_size}")
            
            return model_info
        except Exception as e:
            print(f"Error loading nnUNet model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_blend_weights(self):
        """
        Create a 3D Gaussian window to be used as blending weights, following nnUNet's approach.

        If tile_step_size is 1.0 (no overlap), then simply return an array of ones.
        Otherwise, create a 3D Gaussian centered at the center of the patch.
        
        This implementation follows nnUNet's compute_gaussian function.
        """
        if self.tile_step_size == 1.0:
            return np.ones(self.patch_size, dtype=np.float32)

        # Create a temporary zero array and set the center to 1
        tmp = np.zeros(self.patch_size, dtype=np.float32)
        center_coords = [i // 2 for i in self.patch_size]
        tmp[tuple(center_coords)] = 1
        
        # Define sigma based on patch_size - using same scale as nnUNet (1/8)
        sigma_scale = 1.0 / 8.0
        sigmas = [i * sigma_scale for i in self.patch_size]
        
        # Apply Gaussian filter
        from scipy.ndimage import gaussian_filter
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        
        # Value scaling factor (higher values will create a more pronounced weighting)
        # nnUNet uses 10, we'll use the same
        value_scaling_factor = 10.0
        
        # Normalize the Gaussian map
        gaussian_importance_map = gaussian_importance_map / (np.max(gaussian_importance_map) / value_scaling_factor)
        
        # Ensure no zero values to avoid division by zero later
        mask = gaussian_importance_map == 0
        if np.any(mask):
            gaussian_importance_map[mask] = np.min(gaussian_importance_map[~mask])
            
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
        """Initialize blend weights for the current patch size"""
        if len(self.patch_size) == 3:  # 3D patches
            self.blend_weights = self._create_blend_weights()
            # Create 4D version for multi-channel data
            self.blend_weights_4d = np.expand_dims(self.blend_weights, axis=0)

    def _process_buffer(self, output_arrays: Dict, count_arrays: Dict,
                        patch_buffer: Dict = None, positions: List[Tuple] = None):
        """
        Write accumulated patches to the Zarr arrays, one patch at a time,
        instead of grouping patches into large bounding boxes.
        """
        # Use the provided buffers if any; otherwise, use the instance buffers.
        if patch_buffer is None or positions is None:
            patch_buffer = self.patch_buffer
            positions = self.buffer_positions

        if not patch_buffer or not positions:
            return

        # Loop over each patch and write it directly.
        for i, pos in enumerate(positions):
            for tgt_name in self.output_targets:
                patch_data = patch_buffer[tgt_name][i]
                # We call _write_region_patches() with just a single patch in the list.
                self._write_region_patches([patch_data], [pos],
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
        """
        with self.write_lock:
            # Ensure blending weights are initialized.
            if self.blend_weights is None:
                self._initialize_blend_weights()

            # Get the full image shape from the count array.
            image_z, image_y, image_x = count_array.shape
            full_z, full_y, full_x = self.patch_size

            # Compute valid sizes for each patch (i.e. how much of the patch actually falls inside the image).
            valid_sizes = []
            for pos in positions:
                z0, y0, x0 = pos
                valid_z = full_z if (z0 + full_z) <= image_z else image_z - z0
                valid_y = full_y if (y0 + full_y) <= image_y else image_y - y0
                valid_x = full_x if (x0 + full_x) <= image_x else image_x - x0
                valid_sizes.append((valid_z, valid_y, valid_x))

            # Compute the union region bounds for all patches.
            min_z = min(pos[0] for pos in positions)
            min_y = min(pos[1] for pos in positions)
            min_x = min(pos[2] for pos in positions)
            max_z = max(pos[0] + valid[0] for pos, valid in zip(positions, valid_sizes))
            max_y = max(pos[1] + valid[1] for pos, valid in zip(positions, valid_sizes))
            max_x = max(pos[2] + valid[2] for pos, valid in zip(positions, valid_sizes))

            # Check output_array dimensionality and adapt access pattern accordingly
            output_ndim = len(output_array.shape)
            
            # Determine if we have 3D or 4D data
            has_channel_dim = len(patches[0].shape) == 4
            
            # Handle differently based on output dimensions
            if output_ndim == 4 and has_channel_dim:
                # Both have 4D: C,Z,Y,X
                region_sum = output_array[:, min_z:max_z, min_y:max_y, min_x:max_x]
                base_weights = self.blend_weights_4d
            elif output_ndim == 3 and has_channel_dim:
                # Let's inspect the shapes to provide better debugging info
                print(f"ERROR: Dimension mismatch detected:")
                print(f"- Output array: shape={output_array.shape}, ndim={output_ndim}")
                print(f"- Patch data: shape={patches[0].shape}, ndim={len(patches[0].shape)}")
                print(f"- Channel flag: has_channel_dim={has_channel_dim}")
                print(f"This indicates a serious dimension mismatch that shouldn't occur - the arrays were not properly recreated.")
                
                # Attempt to recover by reducing patch dimension
                tmp_patches = [p[0:1] for p in patches]  # Keep only first channel
                patches = tmp_patches  # Replace patches with reduced versions
                region_sum = output_array[min_z:max_z, min_y:max_y, min_x:max_x]
                base_weights = self.blend_weights 
                has_channel_dim = False  # Reset flag since we've modified patches
            elif output_ndim == 3 and not has_channel_dim:
                # Both are 3D: Z,Y,X
                region_sum = output_array[min_z:max_z, min_y:max_y, min_x:max_x]
                base_weights = self.blend_weights
            else:
                # Handle unusual case of 3D patch and 4D output - shouldn't happen but just in case
                raise ValueError(f"Incompatible dimensions: output array has {output_ndim} dimensions " 
                               f"but patch data has {len(patches[0].shape)} dimensions")

            region_count = count_array[min_z:max_z, min_y:max_y, min_x:max_x]

            # Loop over patches and add only the valid region.
            for patch, pos, valid in zip(patches, positions, valid_sizes):
                valid_z, valid_y, valid_x = valid
                z0, y0, x0 = pos
                z_rel = z0 - min_z
                y_rel = y0 - min_y
                x_rel = x0 - min_x

                # Crop the patch and blending weights to the valid region.
                if has_channel_dim:
                    patch_valid = patch[:, :valid_z, :valid_y, :valid_x]
                    # Make sure we have 4D weights for 4D patch
                    if base_weights.ndim == 3:
                        # Need to expand base_weights to match channel dimension
                        local_weights = np.expand_dims(base_weights[:valid_z, :valid_y, :valid_x], axis=0)
                    else:
                        local_weights = base_weights[:, :valid_z, :valid_y, :valid_x]
                else:
                    patch_valid = patch[:valid_z, :valid_y, :valid_x]
                    local_weights = base_weights[:valid_z, :valid_y, :valid_x]

                weighted_patch = patch_valid * local_weights

                # Update region sum based on whether we have channel dimension
                if output_ndim == 4 and has_channel_dim:
                    region_sum[:, z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                elif output_ndim == 3 and has_channel_dim:
                    # This path should not be reached due to earlier fix, but just in case
                    print(f"Warning: attempting to add 4D data to 3D array - taking first channel only")
                    region_sum[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch[0]
                else:
                    region_sum[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += weighted_patch
                
                # For count array, adjust as needed based on weights dimensionality
                if local_weights.ndim == 4:
                    # For multi-channel weights, just use one channel for count
                    region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += local_weights[0]
                else:
                    # Single channel weights - use as is
                    region_count[z_rel:z_rel + valid_z, y_rel:y_rel + valid_y, x_rel:x_rel + valid_x] += local_weights

            # Write back the updated regions.
            if output_ndim == 4 and has_channel_dim:
                output_array[:, min_z:max_z, min_y:max_y, min_x:max_x] = region_sum
            else:
                output_array[min_z:max_z, min_y:max_y, min_x:max_x] = region_sum
            count_array[min_z:max_z, min_y:max_y, min_x:max_x] = region_count

    def _process_model_outputs(self, outputs, positions: List[Tuple],
                               output_arrays: Dict, count_arrays: Dict):
        """Process model outputs with buffering and offload writing asynchronously when full"""
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
                if num_channels != self.output_targets[tgt_name]["channels"]:
                    print(f"Warning: Expected {self.output_targets[tgt_name]['channels']} channel(s) " 
                          f"but got {num_channels} channels in the processed output.")
                    # This shouldn't happen since we're controlling the channel count upstream
                
                self.patch_buffer[tgt_name].append(pred)
            self.buffer_positions.append(pos)

        # When buffer is full, process it asynchronously
        if len(self.buffer_positions) >= self.buffer_size:
            # Make local copies of the current buffers and clear the instance buffers.
            local_patch_buffer = {k: v[:] for k, v in self.patch_buffer.items()}
            local_positions = self.buffer_positions[:]
            self.patch_buffer.clear()
            self.buffer_positions.clear()
            
            # Offload asynchronous writing.
            future = self.executor.submit(self._process_buffer, output_arrays, count_arrays,
                                          local_patch_buffer, local_positions)
            self.write_futures.append(future)

            # Wait (block) if too many pending write futures to avoid unbounded memory usage
            if len(self.write_futures) >= self.max_pending_writes:
                done, _ = wait(self.write_futures, return_when=FIRST_COMPLETED)
                for future in done:
                    self.write_futures.remove(future)

    def infer(self):
        """Run inference with the nnUNet model on the zarr array."""
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
                    load_all=self.load_all
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
                    
                    # Always use 4D output with channel dimension
                    # This ensures we handle nnUNet's binary segmentation output (2 channels) correctly
                    out_shape = (c, z_max, y_max, x_max)
                    chunks = (c, chunk_z, chunk_y, chunk_x)
                    print(f"Using 4D output shape for target '{tgt_name}': {out_shape}")

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

            # Create dataset and dataloader
            dataset = InferenceDataset(
                input_path=self.input_path,
                targets=self.output_targets,
                model_info=self.model_info,
                patch_size=self.patch_size,
                input_format=self.input_format,
                step_size=self.tile_step_size,
                load_all=self.load_all
            )

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
                    
                    # Use run_inference for nnUNet models
                    outputs = run_inference(self.model_info, patches)
                    
                    # Process outputs based on activation functions
                    processed_outputs = {}
                    
                    # For the first batch, verify nnUNet output format
                    if batch_idx == 0:
                        # For nnUNet, outputs is typically a tensor
                        if torch.is_tensor(outputs):
                            # Get the output shape to determine number of channels
                            num_channels = outputs.shape[1]
                            print(f"nnUNet model output shape: {outputs.shape}, channels: {num_channels}")
                            
                            # Check if this is a typical nnUNet output (should have 2 channels for binary segmentation)
                            if num_channels != 2:
                                print(f"Warning: Expected 2 channels from nnUNet but got {num_channels} channels.")
                                print(f"This might not be a standard binary nnUNet model.")
                            else:
                                print(f"Detected standard nnUNet binary segmentation output (2 channels).")
                                print(f"Will extract foreground channel (index {self.nnunet_foreground_channel}) for final output.")
                    
                    # Apply activations for each target and extract only the foreground channel
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
                        
                        # For nnUNet with 2 channels, extract only the foreground channel (index 1)
                        if activated.shape[1] == 2:
                            processed_outputs[t_name] = activated[:, self.nnunet_foreground_channel:self.nnunet_foreground_channel+1]
                        else:
                            processed_outputs[t_name] = activated
                            
                    self._process_model_outputs(processed_outputs, positions, output_arrays, count_arrays)

            # Process any remaining patches in the buffer.
            if self.buffer_positions:
                self._process_buffer(output_arrays, count_arrays)
            for future in self.write_futures:
                future.result()

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
            print(f"Final output saved to {store_path}")

    def _optimized_postprocessing(self, zarr_store):
        """Optimized post-processing with improved vector handling and optional thresholding"""
        for tgt_name in self.output_targets:
            sum_ds = zarr_store[f"{tgt_name}_sum"]
            cnt_ds = zarr_store[f"{tgt_name}_count"]
            is_normals = (tgt_name.lower() == "normals")
            chunk_size = sum_ds.chunks[-3]

            final_dtype = "uint16" if is_normals else "uint8"
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)

            # Create dataset for probability output
            final_ds = zarr_store.create_dataset(
                name=f"{tgt_name}_final",
                shape=sum_ds.shape,
                chunks=sum_ds.chunks,
                dtype=final_dtype,
                compressor=compressor,
                fill_value=0
            )
            
            # Create additional dataset for thresholded output if threshold is specified
            thresholded_ds = None
            if self.threshold is not None and not is_normals:
                print(f"Threshold value set to {self.threshold}% - will create binary threshold output")
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
            
            # Print shape info
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
                    sum_chunk[mask] /= count_chunk[mask]
                    normalized_chunk[mask] /= count_chunk[mask]

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

                # Write probability chunks back to final dataset
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
    parser.add_argument("--write_layers", action="store_true",
                      help="Write the sliced z layers to disk")
    parser.add_argument("--postprocess_only", action="store_true",
                      help="Skip the inference pass and only do final averaging + casting")
    parser.add_argument("--load_all", action="store_true",
                      help="Load the entire input array into memory (use with caution!)")
    
    args = parser.parse_args()

    # Set the CUDA device before initializing the process group.
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)  # Set the device first!
        dist.init_process_group(backend='nccl', init_method='env://')
        device = f'cuda:{local_rank}'
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    inference_handler = ZarrNNUNetInferenceHandler(
        input_path=args.input_path,
        output_path=args.output_path,
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        step_size=args.step_size,
        num_dataloader_workers=args.num_dataloader_workers,
        num_write_workers=args.num_write_workers,
        write_layers=args.write_layers,
        postprocess_only=args.postprocess_only,
        device=device,
        threshold=args.threshold,
        load_all=args.load_all
    )
    
    inference_handler.infer()  # Run inference and postprocessing

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()