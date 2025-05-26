from pathlib import Path
import os
import json
import numpy as np
import torch
import fsspec
import zarr
from torch.utils.data import Dataset
import albumentations as A

from utils.utils import find_mask_patches, find_mask_patches_2d, pad_or_crop_3d, pad_or_crop_2d

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

        self.image_transforms = image_transforms
        self.volume_transforms = volume_transforms
        self.target_volumes = {}
        self.valid_patches = []
        self.is_2d_dataset = None  
        
        self._initialize_volumes()
        ref_target = list(self.target_volumes.keys())[0]
        ref_volume = self.target_volumes[ref_target][0]['data']['label']
        self.is_2d_dataset = len(ref_volume.shape) == 2
        
        if self.is_2d_dataset:
            print("Detected 2D dataset")
        else:
            print("Detected 3D dataset")
        
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

    def _get_valid_patches(self):
        """Find valid patches based on mask coverage and labeled ratio requirements."""
        print("Computing valid patches...")
        ref_target = list(self.target_volumes.keys())[0]

        for vol_idx, volume_info in enumerate(self.target_volumes[ref_target]):
            vdata = volume_info['data']
            is_2d = len(vdata['label'].shape) == 2
            
            label_data = vdata['label']  # Get the label data explicitly
            
            # Use mask if available, otherwise use label data for patch finding
            if 'mask' in vdata:
                mask_data = vdata['mask']
                print(f"Using mask for patch extraction in volume {vol_idx}")
            else:
                if hasattr(label_data, 'shape') and hasattr(label_data, '__array__'):
                    label_np = np.asarray(label_data)
                    mask_data = (label_np > 0).astype(np.float32)
                else:
                    mask_data = (label_data > 0).astype(np.float32)
                print(f"No mask found for volume {vol_idx}, using label data for patch extraction")
            
            if is_2d:
                h, w = self.patch_size[0], self.patch_size[1]  # y, x
                patches = find_mask_patches_2d(
                    mask_data,
                    label_data,
                    patch_size=[h, w], 
                    min_mask_coverage=1.0,
                    min_labeled_ratio=self.min_labeled_ratio
                )
                print(f"Found {len(patches)} patches from 2D data with min labeled ratio {self.min_labeled_ratio}")
            else:
                patches = find_mask_patches(
                    mask_data,
                    label_data,
                    patch_size=self.patch_size, 
                    min_mask_coverage=1.0,
                    min_labeled_ratio=self.min_labeled_ratio
                )
                print(f"Found {len(patches)} patches from 3D data with min labeled ratio {self.min_labeled_ratio}")

            for p in patches:
                self.valid_patches.append({
                    "volume_index": vol_idx,
                    "position": p["start_pos"]  # (z,y,x)
                })

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
            # For 2D, position is [dummy_z, y, x] and patch_size is [h, w]
            _, y, x = patch_info["position"]  # Unpack properly ignoring dummy z value
            dy, dx = self.patch_size
            z, dz = 0, 0  # Not used for 2D
            is_2d = True
        else:
            # For 3D, position is (z, y, x) and patch_size is (d, h, w)
            z, y, x = patch_info["position"]
            dz, dy, dx = self.patch_size
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
            Normalized image patch
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
        
        # Normalize to [0, 1] range
        img_patch = img_patch.astype(np.float32)
        min_val = np.min(img_patch)
        max_val = np.max(img_patch)
        if max_val > min_val:
            img_patch = (img_patch - min_val) / (max_val - min_val)
        
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
    
    def _extract_loss_mask(self, vol_idx, z, y, x, dz, dy, dx, is_2d):
        """
        Extract loss mask if available.
        
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
        numpy.ndarray or None
            Loss mask if available, None otherwise
        """
        # Check only the first target for the mask
        first_target_name = list(self.target_volumes.keys())[0]
        volume_info = self.target_volumes[first_target_name][vol_idx]
        vdata = volume_info['data']
        
        if 'mask' in vdata:
            mask_arr = vdata['mask']
            
            if is_2d:
                mask_patch = mask_arr[y:y+dy, x:x+dx]
                mask_patch = pad_or_crop_2d(mask_patch, (dy, dx))
            else:
                mask_patch = mask_arr[z:z+dz, y:y+dy, x:x+dx]
                mask_patch = pad_or_crop_3d(mask_patch, (dz, dy, dx))
            
            # Convert to binary mask
            return (mask_patch > 0).astype(np.float32)
        
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
    
    def _prepare_tensors(self, img_patch, label_patches, loss_mask, vol_idx, is_2d):
        """
        Convert numpy arrays to PyTorch tensors with proper formatting.
        
        Parameters
        ----------
        img_patch : numpy.ndarray
            Image patch
        label_patches : dict
            Dictionary of label patches for each target
        loss_mask : numpy.ndarray or None
            Loss mask if available
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
        
        # Add channel dimension to image
        if is_2d and img_patch.ndim == 2:
            img_patch = img_patch[None, ...]  # Shape: (1, H, W)
        elif not is_2d and img_patch.ndim == 3:
            img_patch = img_patch[None, ...]  # Shape: (1, D, H, W)
        
        # Add image to data dict
        data_dict["image"] = torch.from_numpy(img_patch)
        
        # Add loss mask if available
        if loss_mask is not None:
            data_dict["loss_mask"] = torch.from_numpy(loss_mask)
        
        # Process all labels based on target configuration
        for t_name, label_patch in label_patches.items():
            # Check if this is a multi-class target with regions
            target_value = self._get_target_value(t_name)
            is_multiclass_with_regions = (
                isinstance(target_value, dict) and 
                'mapping' in target_value and 
                ('regions' in target_value or len(target_value['mapping']) > 2)
            )
            
            if is_multiclass_with_regions:
                # Multi-class segmentation: use integer labels
                # Determine number of classes (max label value + 1)
                unique_vals = np.unique(label_patch)
                num_classes = int(unique_vals.max()) + 1
                
                # Update target configuration for multi-class
                self.targets[t_name]["out_channels"] = num_classes
                self.targets[t_name]["loss_fn"] = "CrossEntropyLoss"
                
                # Convert to long tensor for CrossEntropyLoss
                label_tensor = torch.from_numpy(label_patch.astype(np.int64))
            else:
                # Binary segmentation: use binary mask
                loss_fn = self.targets[t_name].get("loss_fn", "SoftDiceLoss")
                self.targets[t_name]["out_channels"] = 1
                
                # For binary tasks: use binary mask with a single channel
                binary_mask = (label_patch > 0).astype(np.float32)
                
                # Add channel dimension
                if is_2d:
                    # 2D case: [1, H, W]
                    binary_mask = binary_mask[np.newaxis, ...]
                else:
                    # 3D case: [1, D, H, W]
                    binary_mask = binary_mask[np.newaxis, ...]
                
                label_tensor = torch.from_numpy(binary_mask)
            
            data_dict[t_name] = label_tensor
        
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
        
        # 4. Extract loss mask if available
        loss_mask = self._extract_loss_mask(vol_idx, z, y, x, dz, dy, dx, is_2d)
        
        # 5. Apply transforms to image and labels
        img_patch, label_patches = self._apply_transforms(img_patch, label_patches, is_2d)
        
        # 6. Convert to tensors and format for the model
        data_dict = self._prepare_tensors(img_patch, label_patches, loss_mask, vol_idx, is_2d)
        
        return data_dict
