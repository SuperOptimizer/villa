import numpy as np
import dask_image.imread
from pathlib import Path
from collections import defaultdict
from .base_dataset import BaseDataset

class TifDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from TIFF files.
    
    This dataset loads TIFF files lazily using dask_image to avoid loading
    all data into memory at once, while still supporting numpy array slicing.
    """
    
    def _initialize_volumes(self):
        """
        Initialize volumes from TIFF files using dask_image for lazy loading.
        
        Expected directory structure:
        data_path/
        ├── images/
        │   ├── image1_ink.tif
        │   ├── image1_normals.tif
        │   ├── image2_ink.tif
        │   └── ...
        ├── labels/
        │   ├── image1_ink.tif
        │   ├── image1_normals.tif
        │   ├── image2_ink.tif
        │   └── ...
        └── masks/
            ├── image1_ink.tif
            ├── image1_normals.tif
            ├── image2_ink.tif
            └── ...
        """
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for TIF dataset")
        
        data_path = Path(self.mgr.data_path)
        if not data_path.exists():
            raise ValueError(f"Data path does not exist: {data_path}")
        
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"
        masks_dir = data_path / "masks"
        
        # Check required directories exist
        if not images_dir.exists():
            raise ValueError(f"Images directory does not exist: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"Labels directory does not exist: {labels_dir}")
        
        # Get the configured targets
        configured_targets = set(self.mgr.targets.keys())
        print(f"Looking for configured targets: {configured_targets}")
        
        # Find all image files and parse their names
        image_files = list(images_dir.glob("*.tif*"))
        if not image_files:
            raise ValueError(f"No TIFF files found in {images_dir}")
        
        # Group files by target and image identifier
        targets_data = defaultdict(lambda: defaultdict(dict))
        
        for image_file in image_files:
            # Parse filename: image1_ink.tif -> image_id="image1", target="ink"
            stem = image_file.stem  # Remove .tif extension
            if '_' not in stem:
                raise ValueError(f"Invalid filename format: {image_file.name}. Expected format: imageN_target.tif")
            
            # Split on the last underscore to handle cases like "image1_test_ink"
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid filename format: {image_file.name}. Expected format: imageN_target.tif")
            
            image_id, target = parts
            
            # Only process targets that are in the configuration
            if target not in configured_targets:
                print(f"Skipping {image_id}_{target} - not in configured targets")
                continue
            
            # Find corresponding label and mask files
            label_file = labels_dir / image_file.name
            mask_file = masks_dir / image_file.name
            
            if not label_file.exists():
                raise ValueError(f"Corresponding label file not found: {label_file}")
            
            # Create dask arrays for lazy loading using dask_image.imread
            data_array = dask_image.imread.imread(str(image_file))
            label_array = dask_image.imread.imread(str(label_file))
            
            # Ensure float32 dtype for arrays
            data_array = data_array.astype(np.float32)
            label_array = label_array.astype(np.float32)
            
            # Store in the nested dictionary - only include mask if it exists
            data_dict = {
                'data': data_array,
                'label': label_array
            }
            
            # Load mask if available
            if mask_file.exists():
                mask_array = dask_image.imread.imread(str(mask_file))
                mask_array = mask_array.astype(np.float32)
                data_dict['mask'] = mask_array
                print(f"Found mask for {image_id}_{target}")
            else:
                print(f"No mask file found for {image_id}_{target}, will use no mask")
            
            targets_data[target][image_id] = data_dict
            
            print(f"Registered {image_id}_{target} with shape {data_array.shape} (lazy loading)")
        
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
        
        print(f"Total targets loaded: {list(self.target_volumes.keys())}")
        print("Note: All data is loaded lazily with dask_image - actual file reading happens on demand")
