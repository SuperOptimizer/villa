from pathlib import Path
import os
import json
import numpy as np
import torch
import fsspec
import zarr
from torch.utils.data import Dataset
import albumentations as A
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from utils.utils import find_mask_patches, find_mask_patches_2d, pad_or_crop_3d, pad_or_crop_2d
from utils.io.patch_cache_utils import (
    get_data_checksums,
    load_cached_patches,
    save_computed_patches,
    save_intensity_properties,
    load_intensity_properties
)
from ..training.normalization import get_normalization
from .intensity_sampling import compute_intensity_properties_parallel

class BaseDataset(Dataset):
    """
    A PyTorch Dataset base class for handling both 2D and 3D data from various sources.
    
    Subclasses must implement the _initialize_volumes() method to specify how
    data is loaded from their specific data source.
    """
    def __init__(self,
                 mgr,
                 image_transforms=None,
                 volume_transforms=None):
        """
        Initialize the dataset with configuration from the manager.
        
        Parameters
        ----------
        mgr : ConfigManager
            Manager containing configuration parameters
        image_transforms : list, optional
            2D image transformations via albumentations
        volume_transforms : list, optional
            3D volume transformations
        """
        super().__init__()
        self.mgr = mgr

        self.model_name = mgr.model_name
        self.targets = mgr.targets               # e.g. {"ink": {...}, "normals": {...}}
        self.patch_size = mgr.train_patch_size   # Expected to be [z, y, x]
        self.min_labeled_ratio = mgr.min_labeled_ratio
        self.min_bbox_percent = mgr.min_bbox_percent
        self.dilate_label = mgr.dilate_label

        # New binarization control parameters
        self.binarize_labels = mgr.binarize_labels
        self.target_value = mgr.target_value
        
        # Skip patch validation (defaults to False)
        self.skip_patch_validation = getattr(mgr, 'skip_patch_validation', False)
        
        # Initialize normalization (will be set after computing intensity properties)
        self.normalization_scheme = getattr(mgr, 'normalization_scheme', 'zscore')
        self.intensity_properties = getattr(mgr, 'intensity_properties', {})
        self.normalizer = None  # Will be initialized after volumes are loaded

        self.image_transforms = image_transforms
        self.volume_transforms = volume_transforms
        self.target_volumes = {}
        self.valid_patches = []
        self.is_2d_dataset = None  
        
        # Store data_path as an attribute for cache handling
        self.data_path = Path(mgr.data_path) if hasattr(mgr, 'data_path') else None
        
        # Cache-related attributes
        self.cache_enabled = getattr(mgr, 'cache_valid_patches', True)
        self.cache_dir = None
        if self.data_path is not None:
            self.cache_dir = self.data_path / '.patches_cache'
            print(f"Cache directory: {self.cache_dir}")
            print(f"Cache enabled: {self.cache_enabled}")
        
        self._initialize_volumes()
        ref_target = list(self.target_volumes.keys())[0]
        ref_volume = self.target_volumes[ref_target][0]['data']['label']
        self.is_2d_dataset = len(ref_volume.shape) == 2
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
        else:
            print("Detected 3D dataset")
        
        # Try to load intensity properties from JSON file first
        loaded_from_cache = False
        if self.cache_enabled and self.cache_dir is not None and self.normalization_scheme in ['zscore', 'ct'] and not self.intensity_properties:
            # Try to load from separate JSON file
            print("\nChecking for cached intensity properties...")
            intensity_result = load_intensity_properties(self.cache_dir)
            if intensity_result is not None:
                cached_intensity_properties, cached_normalization_scheme = intensity_result
                if cached_normalization_scheme == self.normalization_scheme:
                    self.intensity_properties = cached_intensity_properties
                    self.mgr.intensity_properties = cached_intensity_properties
                    if hasattr(self.mgr, 'dataset_config'):
                        self.mgr.dataset_config['intensity_properties'] = cached_intensity_properties
                    print("\nLoaded intensity properties from JSON cache - skipping computation")
                    print("Cached intensity properties:")
                    for key, value in cached_intensity_properties.items():
                        print(f"  {key}: {value:.4f}")
                    loaded_from_cache = True
                else:
                    print(f"Cached normalization scheme '{cached_normalization_scheme}' doesn't match current '{self.normalization_scheme}'")
            
        # Also check patches cache for backward compatibility
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None and not loaded_from_cache:
            config_params = self._get_config_params()
            cache_result = load_cached_patches(
                self.cache_dir,
                config_params,
                self.data_path
            )
            if cache_result is not None:
                cached_patches, cached_intensity_properties, cached_normalization_scheme = cache_result
                # If we have cached intensity properties and don't already have them, use the cached ones
                if cached_intensity_properties and not self.intensity_properties:
                    self.intensity_properties = cached_intensity_properties
                    self.mgr.intensity_properties = cached_intensity_properties
                    if hasattr(self.mgr, 'dataset_config'):
                        self.mgr.dataset_config['intensity_properties'] = cached_intensity_properties
                    print("\nLoaded intensity properties from patches cache - skipping computation")
                    print("Cached intensity properties:")
                    for key, value in cached_intensity_properties.items():
                        print(f"  {key}: {value:.4f}")
                    loaded_from_cache = True
                    # Also save to JSON for visibility
                    save_intensity_properties(self.cache_dir, cached_intensity_properties, cached_normalization_scheme or self.normalization_scheme)
        
        # Compute intensity properties if not provided and not loaded from cache
        if self.normalization_scheme in ['zscore', 'ct'] and not self.intensity_properties and not loaded_from_cache:
            print(f"\nComputing intensity properties for {self.normalization_scheme} normalization...")
            self.intensity_properties = compute_intensity_properties_parallel(self.target_volumes, sample_ratio=0.001, max_samples=1000000)
            # Update the config manager with computed properties
            if hasattr(self.mgr, 'intensity_properties'):
                self.mgr.intensity_properties = self.intensity_properties
                self.mgr.dataset_config['intensity_properties'] = self.intensity_properties
            
            # Save to separate JSON file for visibility
            if self.cache_enabled and self.cache_dir is not None:
                save_intensity_properties(self.cache_dir, self.intensity_properties, self.normalization_scheme)
        
        # Now initialize the normalizer with the computed or provided properties
        self.normalizer = get_normalization(self.normalization_scheme, self.intensity_properties)
        
        self._get_valid_patches()

    def _initialize_volumes(self):
        """
        Initialize volumes from the data source.
        
        This method must be implemented by subclasses to specify how
        data is loaded from their specific data source (napari, TIFs, Zarr, etc.).
        
        The implementation should populate self.target_volumes in the format:
        {
            'target_name': [
                {
                    'data': {
                        'data': numpy_array,      # Image data
                        'label': numpy_array,     # Label data  
                        'mask': numpy_array       # Mask data (optional)
                    }
                },
                ...  # Additional volumes for this target
            ],
            ...  # Additional targets
        }
        """
        raise NotImplementedError("Subclasses must implement _initialize_volumes() method")

    def _get_all_sliding_window_positions(self, volume_shape, patch_size, stride=None):
        """
        Generate all possible sliding window positions for a volume.
        
        Parameters
        ----------
        volume_shape : tuple
            Shape of the volume (2D or 3D)
        patch_size : tuple
            Size of patches to extract
        stride : tuple, optional
            Stride for sliding window, defaults to 50% overlap
            
        Returns
        -------
        list
            List of positions as dictionaries with 'start_pos' key
        """
        if len(volume_shape) == 2:
            # 2D case
            H, W = volume_shape
            h, w = patch_size
            
            if stride is None:
                stride = (h // 2, w // 2)  # 50% overlap by default
            
            positions = []
            # Calculate total iterations for progress bar
            y_positions = list(range(0, H - h + 1, stride[0]))
            total_positions = len(y_positions) * len(range(0, W - w + 1, stride[1]))
            
            with tqdm(total=total_positions, desc="Generating 2D sliding window positions", leave=False) as pbar:
                for y in y_positions:
                    for x in range(0, W - w + 1, stride[1]):
                        positions.append({
                            'start_pos': [0, y, x]  # [dummy_z, y, x] for 2D
                        })
                        pbar.update(1)
            
            # Ensure we cover the edges
            # Add patches at the bottom edge if needed
            if H - h > positions[-1]['start_pos'][1]:
                for x in range(0, W - w + 1, stride[1]):
                    positions.append({
                        'start_pos': [0, H - h, x]
                    })
            
            # Add patches at the right edge if needed
            if W - w > positions[-1]['start_pos'][2]:
                for y in range(0, H - h + 1, stride[0]):
                    positions.append({
                        'start_pos': [0, y, W - w]
                    })
            
            # Add the bottom-right corner if needed
            if H - h > positions[-1]['start_pos'][1] or W - w > positions[-1]['start_pos'][2]:
                positions.append({
                    'start_pos': [0, H - h, W - w]
                })
                
        else:
            # 3D case
            D, H, W = volume_shape
            d, h, w = patch_size
            
            if stride is None:
                stride = (d // 2, h // 2, w // 2)  # 50% overlap by default
            
            positions = []
            # Calculate total iterations for progress bar
            z_positions = list(range(0, D - d + 1, stride[0]))
            y_positions = list(range(0, H - h + 1, stride[1]))
            x_positions = list(range(0, W - w + 1, stride[2]))
            total_positions = len(z_positions) * len(y_positions) * len(x_positions)
            
            with tqdm(total=total_positions, desc="Generating 3D sliding window positions", leave=False) as pbar:
                for z in z_positions:
                    for y in y_positions:
                        for x in x_positions:
                            positions.append({
                                'start_pos': [z, y, x]
                            })
                            pbar.update(1)
            
            # Ensure we cover the edges in 3D
            # This is more complex but follows the same principle
            # Add patches to cover the edges if the stride doesn't naturally cover them
            if D - d > 0 and (D - d) % stride[0] != 0:
                for y in range(0, H - h + 1, stride[1]):
                    for x in range(0, W - w + 1, stride[2]):
                        positions.append({
                            'start_pos': [D - d, y, x]
                        })
            
            if H - h > 0 and (H - h) % stride[1] != 0:
                for z in range(0, D - d + 1, stride[0]):
                    for x in range(0, W - w + 1, stride[2]):
                        positions.append({
                            'start_pos': [z, H - h, x]
                        })
            
            if W - w > 0 and (W - w) % stride[2] != 0:
                for z in range(0, D - d + 1, stride[0]):
                    for y in range(0, H - h + 1, stride[1]):
                        positions.append({
                            'start_pos': [z, y, W - w]
                        })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for pos in positions:
            pos_tuple = tuple(pos['start_pos'])
            if pos_tuple not in seen:
                seen.add(pos_tuple)
                unique_positions.append(pos)
        
        return unique_positions

    def _get_config_params(self):
        """
        Get configuration parameters for caching.
        
        Returns
        -------
        dict
            Configuration parameters
        """
        return {
            'patch_size': self.patch_size,
            'min_labeled_ratio': self.min_labeled_ratio,
            'min_bbox_percent': self.min_bbox_percent,
            'skip_patch_validation': self.skip_patch_validation,
            'targets': sorted(self.targets.keys()),
            'is_2d_dataset': self.is_2d_dataset
        }
    
    def _get_valid_patches(self):
        """Find valid patches based on mask coverage and labeled ratio requirements."""
        # Try to load from cache first
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None:
            print("\nAttempting to load patches from cache...")
            config_params = self._get_config_params()
            print(f"Cache configuration: {config_params}")
            cache_result = load_cached_patches(
                self.cache_dir,
                config_params,
                self.data_path
            )
            if cache_result is not None:
                cached_patches, cached_intensity_properties, cached_normalization_scheme = cache_result
                self.valid_patches = cached_patches
                print(f"Successfully loaded {len(self.valid_patches)} patches from cache\n")
                
                # Load cached intensity properties if available and not already set
                if cached_intensity_properties and not self.intensity_properties:
                    self.intensity_properties = cached_intensity_properties
                    self.mgr.intensity_properties = cached_intensity_properties
                    if hasattr(self.mgr, 'dataset_config'):
                        self.mgr.dataset_config['intensity_properties'] = cached_intensity_properties
                    print("Loaded intensity properties from cache - skipping computation")
                    print("Cached intensity properties:")
                    for key, value in cached_intensity_properties.items():
                        print(f"  {key}: {value:.4f}")
                    
                return
            else:
                print("No valid cache found, will compute patches...")
            
        # If no valid cache, compute patches
        print("Computing valid patches...")
        ref_target = list(self.target_volumes.keys())[0]
        total_volumes = len(self.target_volumes[ref_target])

        for vol_idx, volume_info in enumerate(tqdm(self.target_volumes[ref_target], 
                                                    desc="Processing volumes", 
                                                    total=total_volumes)):
            vdata = volume_info['data']
            is_2d = len(vdata['label'].shape) == 2
            
            if self.skip_patch_validation:
                # Skip validation - generate all sliding window positions
                print(f"Skipping patch validation for volume {vol_idx} - using all sliding window positions")
                
                volume_shape = vdata['label'].shape
                if is_2d:
                    patch_size = self.patch_size[:2]  # [h, w]
                else:
                    patch_size = self.patch_size  # [d, h, w]
                
                patches = self._get_all_sliding_window_positions(volume_shape, patch_size)
                print(f"Generated {len(patches)} patches from {'2D' if is_2d else '3D'} data using sliding window")
            else:
                # Original validation logic
                label_data = vdata['label']  # Get the label data explicitly
                
                # Check if mask is available
                has_mask = 'mask' in vdata
                
                if has_mask:
                    mask_data = vdata['mask']
                    print(f"Using mask for patch extraction in volume {vol_idx}")
                else:
                    # When no mask is available, we'll skip mask checks entirely
                    # The mask_data shape is still needed for the patch finding functions
                    mask_data = label_data  # Just for shape, won't be used with skip_mask_check=True
                    print(f"No mask found for volume {vol_idx}, skipping mask coverage checks")
                
                if is_2d:
                    h, w = self.patch_size[0], self.patch_size[1]  # y, x
                    patches = find_mask_patches_2d(
                        mask_data,
                        label_data,
                        patch_size=[h, w], 
                        min_mask_coverage=1.0,
                        min_labeled_ratio=self.min_labeled_ratio,
                        skip_mask_check=not has_mask  # Skip mask check if no mask provided
                    )
                    print(f"Found {len(patches)} patches from 2D data with min labeled ratio {self.min_labeled_ratio}")
                else:
                    patches = find_mask_patches(
                        mask_data,
                        label_data,
                        patch_size=self.patch_size, 
                        min_mask_coverage=1.0,
                        min_labeled_ratio=self.min_labeled_ratio,
                        skip_mask_check=not has_mask  # Skip mask check if no mask provided
                    )
                    print(f"Found {len(patches)} patches from 3D data with min labeled ratio {self.min_labeled_ratio}")

            for p in patches:
                self.valid_patches.append({
                    "volume_index": vol_idx,
                    "position": p["start_pos"]  # (z,y,x)
                })
        
        # Save to cache after computing all patches
        if self.cache_enabled and self.cache_dir is not None and self.data_path is not None:
            print(f"\nAttempting to save {len(self.valid_patches)} patches to cache...")
            config_params = self._get_config_params()
            success = save_computed_patches(
                self.valid_patches,
                self.cache_dir,
                config_params,
                self.data_path,
                intensity_properties=self.intensity_properties,
                normalization_scheme=self.normalization_scheme
            )
            if success:
                print(f"Successfully saved patches to cache directory: {self.cache_dir}\n")
            else:
                print("Failed to save patches to cache\n")
        else:
            if not self.cache_enabled:
                print("Patch caching is disabled")
            elif self.cache_dir is None:
                print("Cache directory is not set")
            elif self.data_path is None:
                print("Data path is not set")

    def __len__(self):
        return len(self.valid_patches)

    def _validate_dimensionality(self, data_item, ref_item=None):
        """
        Validate and ensure consistent dimensionality between different data samples.
        
        Parameters
        ----------
        data_item : numpy.ndarray
            The data item to validate
        ref_item : numpy.ndarray, optional
            A reference item to compare against
            
        Returns
        -------
        bool
            True if the data is 2D, False if 3D
        """
        is_2d = len(data_item.shape) == 2
        
        if ref_item is not None:
            ref_is_2d = len(ref_item.shape) == 2
            if is_2d != ref_is_2d:
                raise ValueError(
                    f"Dimensionality mismatch: Data item is {'2D' if is_2d else '3D'} "
                    f"but reference item is {'2D' if ref_is_2d else '3D'}"
                )
        
        return is_2d
            
    def _extract_patch_coords(self, patch_info):
        """
        Extract patch coordinates and sizes based on dataset dimensionality.
        
        Parameters
        ----------
        patch_info : dict
            Dictionary containing patch position information
        
        Returns
        -------
        tuple
            (z, y, x, dz, dy, dx, is_2d) coordinates and dimensions
        """
        if self.is_2d_dataset:
            # For 2D, position is [dummy_z, y, x] and patch_size should be [h, w]
            _, y, x = patch_info["position"]  # Unpack properly ignoring dummy z value
            
            # Handle patch_size dimensionality - take last 2 dimensions for 2D
            if len(self.patch_size) >= 2:
                dy, dx = self.patch_size[-2:]  # Take last 2 elements (height, width)
            else:
                raise ValueError(f"patch_size {self.patch_size} insufficient for 2D data")
                
            z, dz = 0, 0  # Not used for 2D
            is_2d = True
        else:
            # For 3D, position is (z, y, x) and patch_size is (d, h, w)
            z, y, x = patch_info["position"]
            
            # Handle patch_size dimensionality
            if len(self.patch_size) >= 3:
                dz, dy, dx = self.patch_size[:3]  # Take first 3 elements
            elif len(self.patch_size) == 2:
                # 2D patch_size for 3D data - assume depth of 1
                dy, dx = self.patch_size
                dz = 1
            else:
                raise ValueError(f"patch_size {self.patch_size} insufficient for 3D data")
                
            is_2d = False
            
        return z, y, x, dz, dy, dx, is_2d
    
    def _extract_image_patch(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract and normalize an image patch from the volume.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
        z, y, x : int
            Starting coordinates
        dz, dy, dx : int
            Patch dimensions
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        numpy.ndarray
            Normalized image patch with channel dimension [C, H, W] or [C, D, H, W]
        """
        # Get the image from the first target (all targets share the same image)
        first_target_name = list(self.target_volumes.keys())[0]
        img_arr = self.target_volumes[first_target_name][vol_idx]['data']['data']
        
        # Extract image patch with appropriate dimensionality
        if is_2d:
            img_patch = img_arr[y:y+dy, x:x+dx]
            img_patch = pad_or_crop_2d(img_patch, (dy, dx))
        else:
            img_patch = img_arr[z:z+dz, y:y+dy, x:x+dx]
            img_patch = pad_or_crop_3d(img_patch, (dz, dy, dx))
        
        # Apply normalization
        if self.normalizer is not None:
            img_patch = self.normalizer.run(img_patch)
        else:
            # If no normalizer, just convert to float32
            img_patch = img_patch.astype(np.float32)
        
        # Add channel dimension
        img_patch = img_patch[np.newaxis, ...]  # Shape: [1, H, W] or [1, D, H, W]
        
        return np.ascontiguousarray(img_patch).copy()
    
    def _extract_label_patches(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract all label patches for all targets.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
        z, y, x : int
            Starting coordinates
        dz, dy, dx : int
            Patch dimensions
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        dict
            Dictionary of label patches for each target
        """
        label_patches = {}
        
        for t_name, volumes_list in self.target_volumes.items():
            volume_info = volumes_list[vol_idx]
            label_arr = volume_info['data']['label']
            
            if is_2d:
                label_patch = label_arr[y:y+dy, x:x+dx]
                
                # Apply binarization only if configured to do so
                if self.binarize_labels:
                    target_value = self._get_target_value(t_name)
                    
                    if isinstance(target_value, dict):
                        # Check if this is a multi-class config with regions
                        if 'mapping' in target_value:
                            # New format with mapping and optional regions
                            mapping = target_value['mapping']
                            regions = target_value.get('regions', {})
                            
                            # First apply standard mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in mapping.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            
                            # Then apply regions (which override)
                            for region_id, source_classes in regions.items():
                                # Create mask for any pixel that belongs to source classes
                                region_mask = np.zeros_like(new_label_patch, dtype=bool)
                                for source_class in source_classes:
                                    region_mask |= (new_label_patch == source_class)
                                # Override those pixels with the region ID
                                new_label_patch[region_mask] = region_id
                            
                            label_patch = new_label_patch
                        else:
                            # Old format: direct mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in target_value.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            label_patch = new_label_patch
                    else:
                        # Single class binarization: keep zeros as zero, set non-zeros to target_value
                        binary_mask = (label_patch > 0)
                        new_label_patch = np.zeros_like(label_patch)
                        new_label_patch[binary_mask] = target_value
                        label_patch = new_label_patch
                        
                label_patch = pad_or_crop_2d(label_patch, (dy, dx))
            else:
                label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]
                
                # Apply binarization only if configured to do so
                if self.binarize_labels:
                    target_value = self._get_target_value(t_name)
                    
                    if isinstance(target_value, dict):
                        # Check if this is a multi-class config with regions
                        if 'mapping' in target_value:
                            # New format with mapping and optional regions
                            mapping = target_value['mapping']
                            regions = target_value.get('regions', {})
                            
                            # First apply standard mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in mapping.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            
                            # Then apply regions (which override)
                            for region_id, source_classes in regions.items():
                                # Create mask for any pixel that belongs to source classes
                                region_mask = np.zeros_like(new_label_patch, dtype=bool)
                                for source_class in source_classes:
                                    region_mask |= (new_label_patch == source_class)
                                # Override those pixels with the region ID
                                new_label_patch[region_mask] = region_id
                            
                            label_patch = new_label_patch
                        else:
                            # Old format: direct mapping
                            new_label_patch = np.zeros_like(label_patch)
                            for original_val, new_val in target_value.items():
                                mask = (label_patch == original_val)
                                new_label_patch[mask] = new_val
                            label_patch = new_label_patch
                    else:
                        # Single class binarization: keep zeros as zero, set non-zeros to target_value
                        binary_mask = (label_patch > 0)
                        new_label_patch = np.zeros_like(label_patch)
                        new_label_patch[binary_mask] = target_value
                        label_patch = new_label_patch
                        
                label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))
            
            # Add channel dimension
            label_patch = label_patch[np.newaxis, ...]  # Shape: [1, H, W] or [1, D, H, W]
            
            # Ensure consistent data type
            label_patch = np.ascontiguousarray(label_patch).astype(np.float32).copy()
            label_patches[t_name] = label_patch
            
        return label_patches
    
    def _get_target_value(self, t_name):
        """
        Extract target value from configuration.
        
        Parameters
        ----------
        t_name : str
            Target name
            
        Returns
        -------
        int or dict
            Target value(s) - can be int for single class or dict for multi-class
        """
        # Check if target_value is configured
        if isinstance(self.target_value, dict) and t_name in self.target_value:
            return self.target_value[t_name]
        elif isinstance(self.target_value, int):
            return self.target_value
        else:
            # Default to 1 if not configured
            print(f"Warning: No target value configured for '{t_name}', defaulting to 1")
            return 1
    
    def _extract_ignore_mask(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract ignore mask if available.
        
        Parameters
        ----------
        vol_idx : int
            Volume index
        z, y, x : int
            Starting coordinates
        dz, dy, dx : int
            Patch dimensions
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        dict or None
            Dictionary of ignore masks per target if available, None otherwise
        """
        # Check only the first target for the mask
        first_target_name = list(self.target_volumes.keys())[0]
        volume_info = self.target_volumes[first_target_name][vol_idx]
        vdata = volume_info['data']
        
        if 'mask' in vdata:
            # Use explicit mask if available - same mask for all targets
            mask_arr = vdata['mask']
            
            if is_2d:
                mask_patch = mask_arr[y:y+dy, x:x+dx]
                mask_patch = pad_or_crop_2d(mask_patch, (dy, dx))
            else:
                mask_patch = mask_arr[z:z+dz, y:y+dy, x:x+dx]
                mask_patch = pad_or_crop_3d(mask_patch, (dz, dy, dx))
            
            # Convert to binary mask and invert (1 = ignore, 0 = compute)
            single_mask = (mask_patch == 0).astype(np.float32)
            
            # Add channel dimension
            single_mask = single_mask[np.newaxis, ...]  # [1, H, W] or [1, D, H, W]
            
            # Return same mask for all targets
            return {t_name: single_mask for t_name in self.target_volumes.keys()}
            
        elif hasattr(self.mgr, 'compute_loss_on_label') and self.mgr.compute_loss_on_label:
            # Create separate masks from each target's labels
            ignore_masks = {}
            
            for t_name, volumes_list in self.target_volumes.items():
                label_arr = volumes_list[vol_idx]['data']['label']
                
                # Extract label patch
                if is_2d:
                    label_patch = label_arr[y:y+dy, x:x+dx]
                    label_patch = pad_or_crop_2d(label_patch, (dy, dx))
                else:
                    label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]
                    label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))
                
                # Create binary mask from this target's label (1 = ignore unlabeled, 0 = compute on labeled)
                mask = (label_patch == 0).astype(np.float32)
                # Add channel dimension
                mask = mask[np.newaxis, ...]  # Shape: [1, H, W] or [1, D, H, W]
                ignore_masks[t_name] = mask
            
            return ignore_masks
        
        return None
    
    def _apply_transforms(self, img_patch, label_patches, is_2d):
        """
        Apply transforms to image and label patches.
        
        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch
        label_patches : dict
            Dictionary of label patches for each target
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        tuple
            (transformed_img, transformed_labels)
        """
        if is_2d and self.image_transforms:
            # For 2D, use albumentations transformations
            transform_input = {"image": img_patch}
            
            # Add all label patches to the transformation input
            for t_name, label_patch in label_patches.items():
                mask_key = f"mask_{t_name}"
                transform_input[mask_key] = label_patch
            
            # Apply the transformations to image and all masks
            transformed = self.image_transforms(**transform_input)
            
            # Get the transformed image
            img_patch = transformed["image"]
            
            # Get all transformed labels
            for t_name in label_patches.keys():
                mask_key = f"mask_{t_name}"
                label_patches[t_name] = transformed[mask_key]
        elif not is_2d and self.volume_transforms:
            # 3D transformations affect only the image for now
            # TODO: Implement proper 3D synchronized transformations
            vol_augmented = self.volume_transforms(volume=img_patch)
            img_patch = vol_augmented["volume"]
        
        return img_patch, label_patches
    
    def _prepare_tensors(self, img_patch, label_patches, ignore_mask, vol_idx, is_2d):
        """
        Convert numpy arrays to PyTorch tensors.
        
        All input arrays already have channel dimensions from extraction methods.
        
        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch with shape [C, H, W] or [C, D, H, W]
        label_patches : dict
            Dictionary of label patches for each target with shape [C, H, W] or [C, D, H, W]
        ignore_mask : dict or None
            Dictionary of ignore masks per target if available
        vol_idx : int
            Volume index
        is_2d : bool
            Whether the data is 2D
            
        Returns
        -------
        dict
            Dictionary of tensors for training
        """
        data_dict = {}
        
        # Simply convert image to tensor (already has channel dimension)
        data_dict["image"] = torch.from_numpy(img_patch)
        
        # Add ignore masks if available
        if ignore_mask is not None:
            data_dict["ignore_masks"] = {
                t_name: torch.from_numpy(mask) 
                for t_name, mask in ignore_mask.items()
            }
        
        # Process all labels based on target configuration
        for t_name, label_patch in label_patches.items():
            # Check if this is a multi-class target
            target_value = self._get_target_value(t_name)
            is_multiclass = (
                isinstance(target_value, dict) and 
                'mapping' in target_value and 
                ('regions' in target_value or len(target_value['mapping']) > 2)
            )
            
            # Get the number of output channels for this target
            out_channels = self.targets[t_name].get("out_channels", 2)
            
            if is_multiclass or out_channels > 2:
                # Multi-class segmentation - update metadata
                unique_vals = np.unique(label_patch)
                num_classes = int(unique_vals.max()) + 1
                self.targets[t_name]["out_channels"] = max(num_classes, out_channels)
            
            # Check if we need to convert to one-hot for binary segmentation
            if out_channels == 2 and label_patch.shape[0] == 1:
                # Binary segmentation with 2 output channels - convert to one-hot
                # label_patch has shape [1, H, W] or [1, D, H, W]
                label_tensor = torch.from_numpy(label_patch).long()
                
                # Create one-hot encoding
                if is_2d:
                    # Shape: [1, H, W] -> [2, H, W]
                    one_hot = torch.zeros(2, label_tensor.shape[1], label_tensor.shape[2])
                    one_hot[0] = (label_tensor[0] == 0).float()  # Background channel
                    one_hot[1] = (label_tensor[0] == 1).float()  # Foreground channel
                else:
                    # Shape: [1, D, H, W] -> [2, D, H, W]
                    one_hot = torch.zeros(2, label_tensor.shape[1], label_tensor.shape[2], label_tensor.shape[3])
                    one_hot[0] = (label_tensor[0] == 0).float()  # Background channel
                    one_hot[1] = (label_tensor[0] == 1).float()  # Foreground channel
                
                data_dict[t_name] = one_hot
            else:
                # Simply convert to tensor (already has proper shape and channel dimension)
                data_dict[t_name] = torch.from_numpy(label_patch)
        
        return data_dict
    
    def __getitem__(self, index):
        """
        Get a patch from the dataset.
        
        Parameters
        ----------
        index : int
            Index of the patch
            
        Returns
        -------
        dict
            Dictionary of tensors for training
        """
        # 1. Get patch info and coordinates
        patch_info = self.valid_patches[index]
        vol_idx = patch_info["volume_index"]
        z, y, x, dz, dy, dx, is_2d = self._extract_patch_coords(patch_info)
        
        # 2. Extract and normalize image patch
        img_patch = self._extract_image_patch(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # 3. Extract label patches for all targets
        label_patches = self._extract_label_patches(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # 4. Extract ignore mask if available
        ignore_mask = self._extract_ignore_mask(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # 5. Apply transforms to image and labels
        img_patch, label_patches = self._apply_transforms(img_patch, label_patches, is_2d)
        
        # 6. Convert to tensors and format for the model
        data_dict = self._prepare_tensors(img_patch, label_patches, ignore_mask, vol_idx, is_2d)
        
        return data_dict
