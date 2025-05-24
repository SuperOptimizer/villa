"""
Create synthetic test datasets for zarr and tif dataset testing.

This module creates realistic test data that matches the expected structure
and format of the vesuvius datasets for comprehensive testing.
"""

import numpy as np
import zarr
import tifffile
from pathlib import Path
import shutil
import os
from typing import Dict, List, Tuple, Optional


class TestDatasetCreator:
    """Creates synthetic test datasets for zarr and tif formats."""
    
    def __init__(self, base_path: str = "vesuvius/models/tests/temp_data"):
        """
        Initialize the test dataset creator.
        
        Parameters
        ----------
        base_path : str
            Base directory where test data will be created
        """
        self.base_path = Path(base_path)
        self.zarr_path = self.base_path / "zarr_test"
        self.tif_path = self.base_path / "tif_test"
        
    def cleanup(self):
        """Remove all test data."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            print(f"Cleaned up test data at {self.base_path}")
    
    def create_synthetic_data_3d(self, shape: Tuple[int, int, int], 
                                 noise_level: float = 0.1,
                                 seed: Optional[int] = None) -> np.ndarray:
        """
        Create synthetic 3D volume data.
        
        Parameters
        ----------
        shape : tuple
            Shape (depth, height, width)
        noise_level : float
            Amount of noise to add
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Synthetic 3D data
        """
        if seed is not None:
            np.random.seed(seed)
            
        d, h, w = shape
        
        # Create base patterns
        # Gradient patterns
        z_grad = np.linspace(0, 1, d)[:, None, None]
        y_grad = np.linspace(0, 1, h)[None, :, None] 
        x_grad = np.linspace(0, 1, w)[None, None, :]
        
        # Sinusoidal patterns for more realistic textures
        z_coords = np.arange(d)[:, None, None] / d * 2 * np.pi
        y_coords = np.arange(h)[None, :, None] / h * 2 * np.pi
        x_coords = np.arange(w)[None, None, :] / w * 2 * np.pi
        
        # Combine patterns
        data = (0.3 * z_grad + 
                0.2 * y_grad + 
                0.2 * x_grad +
                0.1 * np.sin(z_coords * 3) + 
                0.1 * np.sin(y_coords * 2) + 
                0.1 * np.sin(x_coords * 4))
        
        # Add noise
        noise = np.random.normal(0, noise_level, shape)
        data = data + noise
        
        # Normalize to [0, 1] and convert to float32
        data = (data - data.min()) / (data.max() - data.min())
        return data.astype(np.float32)
    
    def create_synthetic_data_2d(self, shape: Tuple[int, int], 
                                 noise_level: float = 0.1,
                                 seed: Optional[int] = None) -> np.ndarray:
        """
        Create synthetic 2D image data.
        
        Parameters
        ----------
        shape : tuple
            Shape (height, width)
        noise_level : float
            Amount of noise to add
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Synthetic 2D data
        """
        if seed is not None:
            np.random.seed(seed)
            
        h, w = shape
        
        # Create base patterns
        y_grad = np.linspace(0, 1, h)[:, None]
        x_grad = np.linspace(0, 1, w)[None, :]
        
        # Sinusoidal patterns
        y_coords = np.arange(h)[:, None] / h * 2 * np.pi
        x_coords = np.arange(w)[None, :] / w * 2 * np.pi
        
        # Combine patterns
        data = (0.4 * y_grad + 
                0.4 * x_grad +
                0.1 * np.sin(y_coords * 2) + 
                0.1 * np.sin(x_coords * 3))
        
        # Add noise
        noise = np.random.normal(0, noise_level, shape)
        data = data + noise
        
        # Normalize to [0, 1] and convert to float32
        data = (data - data.min()) / (data.max() - data.min())
        return data.astype(np.float32)
    
    def create_synthetic_labels(self, data: np.ndarray, 
                                coverage: float = 0.3,
                                seed: Optional[int] = None) -> np.ndarray:
        """
        Create synthetic binary labels based on data intensity.
        
        Parameters
        ----------
        data : np.ndarray
            Source data to base labels on
        coverage : float
            Approximate fraction of positive labels
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Binary labels (0 or 1)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Create labels based on intensity threshold with some randomness
        threshold = np.percentile(data, (1 - coverage) * 100)
        base_labels = (data > threshold).astype(np.float32)
        
        # Add some noise to make it more realistic
        noise = np.random.random(data.shape)
        # Flip some labels randomly (5% noise)
        flip_mask = noise < 0.05
        base_labels[flip_mask] = 1 - base_labels[flip_mask]
        
        return base_labels.astype(np.float32)
    
    def create_synthetic_masks(self, shape: Tuple[int, ...], 
                               coverage: float = 0.8,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        Create synthetic masks indicating valid regions.
        
        Parameters
        ----------
        shape : tuple
            Shape of the mask
        coverage : float
            Fraction of the mask that should be valid (1.0)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Binary mask (0 or 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create a mask with some realistic patterns
        mask = np.ones(shape, dtype=np.float32)
        
        # Remove some random regions to simulate invalid areas
        if len(shape) == 3:
            d, h, w = shape
            # Remove some random 3D regions
            for _ in range(int((1 - coverage) * 20)):
                z = np.random.randint(0, d)
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                size = np.random.randint(5, 15)
                
                z_end = min(z + size, d)
                y_end = min(y + size, h)
                x_end = min(x + size, w)
                
                mask[z:z_end, y:y_end, x:x_end] = 0
        else:
            h, w = shape
            # Remove some random 2D regions
            for _ in range(int((1 - coverage) * 10)):
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                size = np.random.randint(5, 20)
                
                y_end = min(y + size, h)
                x_end = min(x + size, w)
                
                mask[y:y_end, x:x_end] = 0
        
        return mask
    
    def create_zarr_dataset(self, 
                           targets: List[str] = ["ink", "normals"],
                           image_ids: List[str] = ["image1", "image2"],
                           shape_3d: Tuple[int, int, int] = (32, 128, 128),
                           include_masks: bool = True):
        """
        Create a complete zarr test dataset (3D only).
        
        Parameters
        ----------
        targets : list
            List of target names
        image_ids : list  
            List of image identifiers
        shape_3d : tuple
            Shape for 3D volumes
        include_masks : bool
            Whether to create mask files
        """
        print(f"Creating zarr test dataset at {self.zarr_path}")
        
        # Create directory structure
        images_dir = self.zarr_path / "images"
        labels_dir = self.zarr_path / "labels"
        masks_dir = self.zarr_path / "masks"
        
        for dir_path in [images_dir, labels_dir, masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        file_count = 0
        for i, image_id in enumerate(image_ids):
            for j, target in enumerate(targets):
                seed = 42 + file_count  # Consistent seeds for reproducibility
                
                # Create synthetic 3D data
                data = self.create_synthetic_data_3d(shape_3d, seed=seed)
                labels = self.create_synthetic_labels(data, coverage=0.25, seed=seed+1)
                
                # File names
                filename = f"{image_id}_{target}.zarr"
                
                # Save as zarr arrays (without compressor for zarr v3 compatibility)
                zarr.save_array(str(images_dir / filename), data)
                zarr.save_array(str(labels_dir / filename), labels)
                
                if include_masks:
                    mask = self.create_synthetic_masks(shape_3d, coverage=0.85, seed=seed+2)
                    zarr.save_array(str(masks_dir / filename), mask)
                
                file_count += 1
                print(f"Created zarr files for {filename} - 3D shape: {shape_3d}")
        
        print(f"Zarr dataset creation complete. Created {file_count} file sets.")
    
    def create_tif_dataset(self, 
                          targets: List[str] = ["ink", "normals"],
                          image_ids: List[str] = ["image1", "image2"],
                          shape_3d: Tuple[int, int, int] = (32, 128, 128),
                          shape_2d: Tuple[int, int] = (128, 128),
                          include_masks: bool = True,
                          mixed_dimensions: bool = True):
        """
        Create a complete TIF test dataset.
        
        Parameters
        ----------
        targets : list
            List of target names
        image_ids : list
            List of image identifiers  
        shape_3d : tuple
            Shape for 3D volumes
        shape_2d : tuple
            Shape for 2D images
        include_masks : bool
            Whether to create mask files
        mixed_dimensions : bool
            Whether to mix 2D and 3D data
        """
        print(f"Creating TIF test dataset at {self.tif_path}")
        
        # Create directory structure
        images_dir = self.tif_path / "images"
        labels_dir = self.tif_path / "labels"
        masks_dir = self.tif_path / "masks"
        
        for dir_path in [images_dir, labels_dir, masks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        file_count = 0
        for i, image_id in enumerate(image_ids):
            for j, target in enumerate(targets):
                # Decide on dimensionality
                if mixed_dimensions:
                    is_3d = (i + j) % 2 == 0  # Alternate between 2D and 3D
                else:
                    is_3d = True  # Default to 3D
                
                shape = shape_3d if is_3d else shape_2d
                seed = 42 + file_count  # Consistent seeds for reproducibility
                
                # Create synthetic data
                if is_3d:
                    data = self.create_synthetic_data_3d(shape, seed=seed)
                else:
                    data = self.create_synthetic_data_2d(shape, seed=seed)
                
                labels = self.create_synthetic_labels(data, coverage=0.25, seed=seed+1)
                
                # File names
                filename = f"{image_id}_{target}.tif"
                
                # Save as TIFF files
                tifffile.imwrite(str(images_dir / filename), data)
                tifffile.imwrite(str(labels_dir / filename), labels)
                
                if include_masks:
                    mask = self.create_synthetic_masks(shape, coverage=0.85, seed=seed+2)
                    tifffile.imwrite(str(masks_dir / filename), mask)
                
                file_count += 1
                print(f"Created TIF files for {filename} - {'3D' if is_3d else '2D'} shape: {shape}")
        
        print(f"TIF dataset creation complete. Created {file_count} file sets.")
    
    def create_all_test_datasets(self):
        """Create both zarr and TIF test datasets with various scenarios."""
        print("Creating comprehensive test datasets...")
        
        # Standard datasets - zarr is 3D only
        self.create_zarr_dataset(
            targets=["ink", "normals"],
            image_ids=["image1", "image2"]
        )
        
        # TIF can still have mixed dimensions
        self.create_tif_dataset(
            targets=["ink", "normals"], 
            image_ids=["image1", "image2"],
            mixed_dimensions=True
        )
        
        # Create some edge case datasets
        # Single target, single image (minimal case) - zarr 3D only
        self.create_zarr_dataset(
            targets=["ink"],
            image_ids=["single"]
        )
        
        # Multiple targets (multi-task scenario) - TIF  
        self.create_tif_dataset(
            targets=["ink", "normals", "sheet", "affinities"],
            image_ids=["multi"],
            mixed_dimensions=False,
            include_masks=False  # Test default mask creation
        )
        
        print("All test datasets created successfully!")


def create_test_data():
    """Convenience function to create test data."""
    creator = TestDatasetCreator()
    creator.create_all_test_datasets()
    return creator


def cleanup_test_data():
    """Convenience function to cleanup test data."""
    creator = TestDatasetCreator()
    creator.cleanup()


if __name__ == "__main__":
    # Create test data when run directly
    create_test_data()
    print("Test data creation complete!")
