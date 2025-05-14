from pathlib import Path
import os
import json
import numpy as np
import torch
import fsspec
import zarr
from torch.utils.data import Dataset
from skimage.morphology import dilation, ball, skeletonize
import albumentations as A


from utils import find_valid_patches, find_valid_patches_2d, pad_or_crop_3d, pad_or_crop_2d

class NapariDataset(Dataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from napari.
    
    This dataset automatically detects if the provided data is 2D or 3D and 
    handles it appropriately throughout the data loading process. It maintains
    the original dimensionality of the data, ensuring 2D data stays 2D and 3D
    data stays 3D in tensors.
    
    When working with 2D data:
    - Only y and x dimensions of patch_size are used
    - Patches are extracted using specialized 2D patch finder
    - 2D data is kept as 2D (H,W) with only a channel dimension added (C,H,W)
    - 2D transformations are applied directly to 2D data
    
    When working with 3D data:
    - Full z, y, x dimensions are used
    - Standard 3D patch extraction is performed
    - 3D transformations are applied, resulting in (C,D,H,W) tensors
    
    This dual handling ensures that models designed for either 2D or 3D data
    can work with the same dataset infrastructure.
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

        self.image_transforms = image_transforms
        self.volume_transforms = volume_transforms

        # Initialize volumes and detect data dimensionality
        # This dict will contain target data in the format:
        # {
        #   "ink": [
        #       {
        #         'data': {'label': <zarr array>, 'data': <zarr array>},
        #         'out_channels': 1,
        #         ...
        #       },
        #       { ... },
        #   ],
        #   "normals": [ ... ],
        # }
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
        Initialize volumes from the ConfigManager's get_images method.
        This populates the target_volumes dictionary with data and metadata
        for each target type.
        """
        self.target_volumes = self.mgr.get_images()

    def _get_valid_patches(self):
        print("Computing valid patches...")
        ref_target = list(self.target_volumes.keys())[0]

        for vol_idx, volume_info in enumerate(self.target_volumes[ref_target]):
            label_data = volume_info['data']['label']  

            is_2d = len(label_data.shape) == 2
            
            if is_2d:
                # For 2D, patch_size is [h, w]
                h, w = self.patch_size[0], self.patch_size[1]  # y, x
                
                patches = find_valid_patches_2d(
                    label_data,
                    patch_size=[h, w],
                    bbox_threshold=self.min_bbox_percent,
                    label_threshold=self.min_labeled_ratio,
                )
                
                print(f"Found {len(patches)} valid 2D patches")
            else:
                patches = find_valid_patches(
                    label_data,
                    patch_size=self.patch_size,
                    bbox_threshold=self.min_bbox_percent,
                    label_threshold=self.min_labeled_ratio,
                )

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
            
    def __getitem__(self, index):
        patch_info = self.valid_patches[index]
        vol_idx = patch_info["volume_index"]
        
        # Handle 2D vs 3D case differently
        if self.is_2d_dataset:
            # For 2D, position is [dummy_z, y, x] and patch_size is [h, w]
            # The find_valid_patches_2d function adds a dummy z=0 value
            _, y, x = patch_info["position"]  # Unpack properly ignoring dummy z value
            dy, dx = self.patch_size
            z, dz = 0, 0  # Not used for 2D
            is_2d = True
        else:
            # For 3D, position is (z, y, x) and patch_size is (d, h, w)
            z, y, x = patch_info["position"]
            dz, dy, dx = self.patch_size
            is_2d = False

        data_dict = {}
        
        # First, extract the image patch
        # Get the image from the first target (all targets share the same image)
        first_target_name = list(self.target_volumes.keys())[0]
        img_arr = self.target_volumes[first_target_name][vol_idx]['data']['data']
        
        # Extract image patch
        if is_2d:
            img_patch = img_arr[y:y+dy, x:x+dx]
        else:
            img_patch = img_arr[z:z+dz, y:y+dy, x:x+dx]
        
        # Apply min-max normalization (scale to 0-1)
        img_patch = img_patch.astype(np.float32)  # Ensure float32 type before normalization
        min_val = np.min(img_patch)
        max_val = np.max(img_patch)
        if max_val > min_val:
            img_patch = (img_patch - min_val) / (max_val - min_val)
        
        img_patch = np.ascontiguousarray(img_patch).copy()
        
        # Now extract all label patches
        label_patches = {}
        for t_name, volumes_list in self.target_volumes.items():
            volume_info = volumes_list[vol_idx]
            vdata = volume_info['data']
            out_c = volume_info['out_channels']
            label_arr = vdata['label']
            
            if is_2d:
                # Extract and process 2D label patch
                label_patch = label_arr[y:y+dy, x:x+dx]
                
                if hasattr(self.mgr, 'dilate_label') and self.mgr.dilate_label > 0:
                    from scipy import ndimage
                    for _ in range(self.mgr.dilate_label):
                        label_patch = ndimage.binary_dilation(label_patch)
                
                # Extract target value from t_name if it has format like "32_3"
                target_value = 1
                if "_" in t_name:
                    suffix = t_name.split("_")[-1]
                    if suffix.isdigit():
                        target_value = int(suffix)
                
                # Process label: keep zeros as zero, set non-zeros to target_value
                binary_mask = (label_patch > 0)
                label_patch = np.zeros_like(label_patch)
                label_patch[binary_mask] = target_value
                label_patch = pad_or_crop_2d(label_patch, (dy, dx))
            else:
                # Extract and process 3D label patch
                label_patch = label_arr[z:z+dz, y:y+dy, x:x+dx]
                
                if hasattr(self.mgr, 'dilate_label') and self.mgr.dilate_label > 0:
                    for _ in range(self.mgr.dilate_label):
                        label_patch = dilation(label_patch, ball(1))
                
                # Extract target value from t_name if it has format like "32_3"
                target_value = 1
                if "_" in t_name:
                    suffix = t_name.split("_")[-1]
                    if suffix.isdigit():
                        target_value = int(suffix)
                
                # Process label: keep zeros as zero, set non-zeros to target_value
                binary_mask = (label_patch > 0)
                label_patch = np.zeros_like(label_patch)
                label_patch[binary_mask] = target_value
                label_patch = pad_or_crop_3d(label_patch, (dz, dy, dx))
            
            # Ensure consistent data type with image for albumentations transformations
            label_patch = np.ascontiguousarray(label_patch).astype(np.uint8).copy()
            # Store the label patch with its target name
            label_patches[t_name] = label_patch
        
        # Apply transformations to both image and all labels at once
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
            # For 3D, use volume transformations if provided
            vol_augmented = self.volume_transforms(volume=img_patch)
            img_patch = vol_augmented["volume"]
            # Note: 3D label transformations would need to be handled separately
            # if needed in the future, as it's not part of the current implementation
        
        # Add channel dimension to image and convert to tensor
        if is_2d:
            # For 2D data: add channel dimension to get (C, H, W)
            if img_patch.ndim == 2:
                img_patch = img_patch[None, ...]  # Shape: (1, H, W)
        else:
            # For 3D data: add channel dimension to get (C, D, H, W)
            if img_patch.ndim == 3:
                img_patch = img_patch[None, ...]  # Shape: (1, D, H, W)
        
        # Store the image in the data dictionary
        data_dict["image"] = torch.from_numpy(img_patch)
        
        # Process all labels and add to data dictionary
        for t_name, label_patch in label_patches.items():
            out_c = self.target_volumes[t_name][vol_idx]['out_channels']
            
            # Add channel dimension if needed for the labels
            if is_2d and label_patch.ndim == 2 and out_c == 1:
                label_patch = label_patch[None, ...]  # Shape: (1, H, W)
            elif not is_2d and label_patch.ndim == 3 and out_c == 1:
                label_patch = label_patch[None, ...]  # Shape: (1, D, H, W)
            
            # Add to data dictionary
            data_dict[t_name] = torch.from_numpy(label_patch)
        
        return data_dict
