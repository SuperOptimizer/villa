import numpy as np
import zarr
import tifffile
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from .base_dataset import BaseDataset


def convert_tif_to_zarr_worker(args):
    """
    Worker function to convert a single TIF file to a Zarr array.
    This function is defined at module level to be picklable for multiprocessing.
    
    Parameters
    ----------
    args : tuple
        (tif_path, zarr_group_path, array_name, patch_size)
        
    Returns
    -------
    tuple
        (array_name, shape, success, error_msg)
    """
    tif_path, zarr_group_path, array_name, patch_size = args
    
    try:
        # Read the TIFF file
        img = tifffile.imread(str(tif_path))
        
        # Convert to float32
        img = img.astype(np.uint8)
        
        # Use patch size directly as chunks
        if len(img.shape) == 2:  # 2D
            chunks = tuple(patch_size[:2])  # [h, w]
        else:  # 3D
            chunks = tuple(patch_size)  # [d, h, w]
        
        # Open the Zarr group and create the array
        group = zarr.open_group(str(zarr_group_path), mode='a')
        group.create_dataset(
            array_name,
            data=img,
            shape=img.shape,
            dtype=np.uint8,
            chunks=chunks,
            compressor=None,
            overwrite=True,
            write_empty_chunks=False
        )
        
        return array_name, img.shape, True, None
        
    except Exception as e:
        return array_name, None, False, str(e)

class TifDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from TIFF files.
    
    This dataset automatically converts TIFF files to Zarr format on first use
    for much faster random access during training. The Zarr files are stored
    as groups in the data path:
    - images.zarr/  (contains image1, image2, etc. as arrays)
    - labels.zarr/  (contains image1_ink, image2_ink, etc. as arrays)  
    - masks.zarr/   (contains image1_ink, image2_ink, etc. as arrays)
    """
    
    def __init__(self, mgr, image_transforms=None, volume_transforms=None):
        """
        Initialize the TIF dataset with configuration from the manager.
        
        By default, TIF datasets skip patch validation for performance,
        considering all sliding window positions as valid.
        
        Parameters
        ----------
        mgr : ConfigManager
            Manager containing configuration parameters
        image_transforms : list, optional
            2D image transformations via albumentations
        volume_transforms : list, optional
            3D volume transformations
        """
        # Always set skip_patch_validation to True for TIF datasets
        mgr.skip_patch_validation = True
        
        super().__init__(mgr, image_transforms, volume_transforms)
    
    def _get_or_create_zarr_groups(self):
        """
        Get or create the Zarr groups for images, labels, and masks.
        
        Returns
        -------
        tuple
            (images_group, labels_group, masks_group)
        """
        # Create paths for the Zarr groups
        images_zarr_path = self.data_path / "images.zarr"
        labels_zarr_path = self.data_path / "labels.zarr"
        masks_zarr_path = self.data_path / "masks.zarr"
        
        # Open or create the groups
        images_group = zarr.open_group(str(images_zarr_path), mode='a')
        labels_group = zarr.open_group(str(labels_zarr_path), mode='a')
        masks_group = zarr.open_group(str(masks_zarr_path), mode='a')
        
        return images_group, labels_group, masks_group
    
    def _tif_to_zarr_array(self, tif_path, zarr_group, array_name):
        """
        Convert a TIFF file to a Zarr array within a group.
        
        Parameters
        ----------
        tif_path : Path
            Path to the TIFF file
        zarr_group : zarr.Group
            The Zarr group to store the array in
        array_name : str
            Name for the array in the Zarr group
            
        Returns
        -------
        zarr.Array
            The created Zarr array
        """
        # Read the TIFF file
        img = tifffile.imread(str(tif_path))
        
        # Convert to float32
        img = img.astype(np.float32)
        
        # Use patch size directly as chunks
        if len(img.shape) == 2:  # 2D
            chunks = tuple(self.patch_size[:2])  # [h, w]
        else:  # 3D
            chunks = tuple(self.patch_size)  # [d, h, w]
        
        z_array = zarr_group.create_dataset(
            array_name,
            data=img,
            shape=img.shape,
            dtype=np.uint8,
            chunks=chunks,
            compressor=None,
            overwrite=True,
            write_empty_chunks=False
        )
        
        return z_array
    
    def _needs_update(self, tif_file, zarr_group, array_name):
        """
        Check if a TIF file is newer than its corresponding Zarr array.
        
        Parameters
        ----------
        tif_file : Path
            Path to the TIFF file
        zarr_group : zarr.Group
            The Zarr group containing the array
        array_name : str
            Name of the array in the group
            
        Returns
        -------
        bool
            True if the TIF file is newer and needs updating
        """
        if array_name not in zarr_group:
            return True
        
        # Check modification times
        tif_mtime = os.path.getmtime(tif_file)
        
        # For groups, check the array metadata file modification time
        group_store_path = Path(zarr_group.store.path)
        if group_store_path.exists():
            array_meta_path = group_store_path / array_name / ".zarray"
            if array_meta_path.exists():
                zarr_mtime = os.path.getmtime(array_meta_path)
                return tif_mtime > zarr_mtime
        
        return True
    
    def _ensure_zarr_array(self, tif_file, zarr_group, array_name):
        """
        Ensure a Zarr array exists in the group, creating or updating as needed.
        
        Parameters
        ----------
        tif_file : Path
            Path to the TIFF file
        zarr_group : zarr.Group
            The Zarr group containing the array
        array_name : str
            Name of the array in the group
            
        Returns
        -------
        zarr.Array
            The Zarr array (either existing or newly created)
        """
        # Check if array exists in the group
        if array_name in zarr_group:
            # Check if we need to update (TIF is newer)
            tif_mtime = os.path.getmtime(tif_file)
            
            # For groups, we check the group's store path modification time
            group_store_path = Path(zarr_group.store.path)
            if group_store_path.exists():
                # Get the array metadata file modification time
                array_meta_path = group_store_path / array_name / ".zarray"
                if array_meta_path.exists():
                    zarr_mtime = os.path.getmtime(array_meta_path)
                    
                    if zarr_mtime >= tif_mtime:
                        # Cache is up to date, return existing array
                        return zarr_group[array_name]
        
        # Need to create or update the array
        print(f"Converting {tif_file.name} to Zarr format...")
        return self._tif_to_zarr_array(tif_file, zarr_group, array_name)
    
    def _initialize_volumes(self):
        """
        Initialize volumes from TIFF files, converting to Zarr format for fast access.
        
        Expected directory structure:
        
        For multi-task scenarios:
        data_path/
        ├── images/
        │   ├── image1.tif      # Single image file
        │   ├── image2.tif      # Single image file
        │   └── ...
        ├── labels/
        │   ├── image1_ink.tif
        │   ├── image1_damage.tif
        │   ├── image2_ink.tif
        │   ├── image2_damage.tif
        │   └── ...
        └── masks/
            ├── image1_ink.tif
            ├── image1_damage.tif
            ├── image2_ink.tif
            ├── image2_damage.tif
            └── ...
            
        For single-task scenarios:
        data_path/
        ├── images/
        │   ├── image1_ink.tif
        │   ├── image2_ink.tif
        │   └── ...
        ├── labels/
        │   ├── image1_ink.tif
        │   ├── image2_ink.tif
        │   └── ...
        └── masks/
            ├── image1_ink.tif
            ├── image2_ink.tif
            └── ...
        """
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for TIF dataset")
        
        self.data_path = Path(self.mgr.data_path)
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        images_dir = self.data_path / "images"
        labels_dir = self.data_path / "labels"
        masks_dir = self.data_path / "masks"
        
        # Check required directories exist
        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory does not exist: {labels_dir}")
        
        # Get or create Zarr groups
        images_group, labels_group, masks_group = self._get_or_create_zarr_groups()
        
        # Get the configured targets
        configured_targets = set(self.mgr.targets.keys())
        print(f"Looking for configured targets: {configured_targets}")
        
        # Find all label files to determine which images and targets we need
        label_files = list(labels_dir.glob("*.tif*"))
        if not label_files:
            raise ValueError(f"No TIFF files found in {labels_dir}")
        
        # Group files by target and image identifier
        targets_data = defaultdict(lambda: defaultdict(dict))
        
        # Track files to convert for progress bar
        files_to_process = []
        
        # First pass: identify all files that need processing
        for label_file in label_files:
            stem = label_file.stem  # Remove .tif extension
            
            # Parse label filename: image1_ink.tif -> image_id="image1", target="ink"
            if '_' not in stem:
                print(f"Skipping label file without underscore: {label_file.name}")
                continue
            
            # Split on the last underscore to handle cases like "image1_test_ink"
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                print(f"Invalid label filename format: {label_file.name}")
                continue
            
            image_id, target = parts
            
            # Only process targets that are in the configuration
            if target not in configured_targets:
                print(f"Skipping {image_id}_{target} - not in configured targets")
                continue
            
            # Look for corresponding image file
            # First try without task suffix (multi-task scenario)
            image_file = images_dir / f"{image_id}.tif"
            if not image_file.exists():
                image_file = images_dir / f"{image_id}.tiff"
            
            # If not found, try with task suffix (single-task/backward compatibility)
            if not image_file.exists():
                image_file = images_dir / f"{image_id}_{target}.tif"
                if not image_file.exists():
                    image_file = images_dir / f"{image_id}_{target}.tiff"
                    if not image_file.exists():
                        raise ValueError(f"Image file not found for {image_id} (tried {image_id}.tif, {image_id}.tiff, {image_id}_{target}.tif, and {image_id}_{target}.tiff)")
            
            # Look for mask file (always with task suffix)
            mask_file = masks_dir / f"{image_id}_{target}.tif"
            if not mask_file.exists():
                mask_file = masks_dir / f"{image_id}_{target}.tiff"
            
            files_to_process.append((target, image_id, image_file, label_file, mask_file if mask_file.exists() else None))
        
        # Collect all conversion tasks that need to be done
        conversion_tasks = []
        array_info = {}  # Track which arrays go where
        
        for target, image_id, image_file, label_file, mask_file in files_to_process:
            # Determine array names
            if image_file.name.endswith(f"_{target}.tif") or image_file.name.endswith(f"_{target}.tiff"):
                image_array_name = f"{image_id}_{target}"
            else:
                image_array_name = image_id
            
            # Check if conversions are needed
            images_zarr_path = self.data_path / "images.zarr"
            labels_zarr_path = self.data_path / "labels.zarr" 
            masks_zarr_path = self.data_path / "masks.zarr"
            
            # Image conversion
            if image_array_name not in images_group or self._needs_update(image_file, images_group, image_array_name):
                conversion_tasks.append((image_file, images_zarr_path, image_array_name, self.patch_size))
            
            # Label conversion
            label_array_name = f"{image_id}_{target}"
            if label_array_name not in labels_group or self._needs_update(label_file, labels_group, label_array_name):
                conversion_tasks.append((label_file, labels_zarr_path, label_array_name, self.patch_size))
            
            # Mask conversion if available
            mask_array_name = None
            if mask_file:
                mask_array_name = f"{image_id}_{target}"
                if mask_array_name not in masks_group or self._needs_update(mask_file, masks_group, mask_array_name):
                    conversion_tasks.append((mask_file, masks_zarr_path, mask_array_name, self.patch_size))
            
            # Store info for later
            array_info[(target, image_id)] = {
                'image_array_name': image_array_name,
                'label_array_name': label_array_name,
                'mask_array_name': mask_array_name if mask_file else None
            }
        
        # Perform parallel conversions if needed
        if conversion_tasks:
            print(f"\nConverting {len(conversion_tasks)} TIF files to Zarr format using {cpu_count()} workers...")
            
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                # Submit all tasks
                futures = {executor.submit(convert_tif_to_zarr_worker, task): task for task in conversion_tasks}
                
                # Process completed tasks with progress bar
                with tqdm(total=len(futures), desc="Converting TIFs to Zarr") as pbar:
                    for future in as_completed(futures):
                        array_name, shape, success, error_msg = future.result()
                        
                        if success:
                            pbar.set_description(f"Converted {array_name}")
                        else:
                            print(f"ERROR converting {array_name}: {error_msg}")
                        
                        pbar.update(1)
            
            print("✓ Conversion complete!")
        else:
            print("✓ All Zarr arrays are up to date!")
        
        # Now load all arrays from the Zarr groups
        print("\nLoading Zarr arrays...")
        
        for target, image_id, image_file, label_file, mask_file in files_to_process:
            info = array_info[(target, image_id)]
            
            # Load arrays from groups
            data_array = images_group[info['image_array_name']]
            label_array = labels_group[info['label_array_name']]
            
            # Store in the nested dictionary
            data_dict = {
                'data': data_array,
                'label': label_array
            }
            
            # Load mask if available
            if info['mask_array_name']:
                mask_array = masks_group[info['mask_array_name']]
                data_dict['mask'] = mask_array
            
            targets_data[target][image_id] = data_dict
            print(f"Loaded {image_id}_{target} with shape {data_array.shape}")
        
        # Check that all configured targets were found
        found_targets = set(targets_data.keys())
        missing_targets = configured_targets - found_targets
        if missing_targets:
            raise ValueError(f"Configured targets not found in data: {missing_targets}")
        
        # Convert to the expected format for BaseDataset
        self.target_volumes = {}
        
        for target, images_dict in targets_data.items():
            self.target_volumes[target] = []
            
            for image_id, data_dict in images_dict.items():
                volume_info = {
                    'data': data_dict
                }
                self.target_volumes[target].append(volume_info)
            
            print(f"Target '{target}' has {len(self.target_volumes[target])} volumes")
        
        print(f"\nTotal targets loaded: {list(self.target_volumes.keys())}")
        print("✓ Zarr cache ready - data access will be much faster!")
