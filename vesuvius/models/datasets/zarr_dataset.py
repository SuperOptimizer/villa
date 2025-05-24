import numpy as np
import zarr
from pathlib import Path
from collections import defaultdict
from .base_dataset import BaseDataset

class ZarrDataset(BaseDataset):
    """
    A PyTorch Dataset for handling both 2D and 3D data from Zarr files.
    
    This dataset loads Zarr files which are already lazily loaded by design,
    supporting numpy array slicing without loading all data into memory.
    """
    
    def _initialize_volumes(self):
        """
        Initialize volumes from Zarr files.
        
        Expected directory structure:
        data_path/
        ├── images/
        │   ├── image1_ink.zarr/
        │   ├── image1_normals.zarr/
        │   ├── image2_ink.zarr/
        │   └── ...
        ├── labels/
        │   ├── image1_ink.zarr/
        │   ├── image1_normals.zarr/
        │   ├── image2_ink.zarr/
        │   └── ...
        └── masks/
            ├── image1_ink.zarr/
            ├── image1_normals.zarr/
            ├── image2_ink.zarr/
            └── ...
        """
        if not hasattr(self.mgr, 'data_path'):
            raise ValueError("ConfigManager must have 'data_path' attribute for Zarr dataset")
        
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
        
        # Find all zarr directories and parse their names
        zarr_dirs = [d for d in images_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']
        if not zarr_dirs:
            raise ValueError(f"No .zarr directories found in {images_dir}")
        
        # Group files by target and image identifier
        targets_data = defaultdict(lambda: defaultdict(dict))
        
        for zarr_dir in zarr_dirs:
            # Parse directory name: image1_ink.zarr -> image_id="image1", target="ink"
            stem = zarr_dir.stem  # Remove .zarr extension
            if '_' not in stem:
                raise ValueError(f"Invalid directory name format: {zarr_dir.name}. Expected format: imageN_target.zarr")
            
            # Split on the last underscore to handle cases like "image1_test_ink"
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid directory name format: {zarr_dir.name}. Expected format: imageN_target.zarr")
            
            image_id, target = parts
            
            # Only process targets that are in the configuration
            if target not in configured_targets:
                print(f"Skipping {image_id}_{target} - not in configured targets")
                continue
            
            # Find corresponding label and mask directories
            label_dir = labels_dir / zarr_dir.name
            mask_dir = masks_dir / zarr_dir.name
            
            if not label_dir.exists():
                raise ValueError(f"Corresponding label directory not found: {label_dir}")
            
            # Open zarr arrays - these are already lazily loaded
            try:
                data_array = zarr.open(str(zarr_dir), mode='r')
                label_array = zarr.open(str(label_dir), mode='r')
                
                # Load mask if available
                if mask_dir.exists():
                    mask_array = zarr.open(str(mask_dir), mode='r')
                    print(f"Found mask for {image_id}_{target}")
                else:
                    # Create a lazy mask wrapper that generates ones when accessed
                    class DefaultMaskArray:
                        def __init__(self, reference_array):
                            self.reference_array = reference_array
                            
                        @property
                        def shape(self):
                            return self.reference_array.shape
                            
                        @property
                        def dtype(self):
                            return np.float32
                            
                        def __getitem__(self, key):
                            # Return ones with the same slice shape as the reference
                            ref_slice = self.reference_array[key]
                            return np.ones_like(ref_slice, dtype=np.float32)
                    
                    mask_array = DefaultMaskArray(label_array)
                    print(f"No mask directory found for {image_id}_{target}, will create default mask")
                
                # Store in the nested dictionary
                targets_data[target][image_id] = {
                    'data': data_array,
                    'label': label_array,
                    'mask': mask_array
                }
                
                print(f"Registered {image_id}_{target} with shape {data_array.shape} (lazy zarr)")
                
            except Exception as e:
                raise ValueError(f"Error opening zarr directory {zarr_dir}: {e}")
        
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
        print("Note: All data is loaded lazily with zarr - actual chunk reading happens on demand")
