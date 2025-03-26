import torch
import numpy as np
from typing import Union, Optional, List, Tuple, Dict, Any
import random
from ..base import BaseAugmentation, ArrayLike
import torch.nn.functional as F

class RandomRotation(BaseAugmentation):
    """
    Randomly rotate the volume and/or label.
    
    This augmentation can perform:
    1. Random 90-degree rotations around one of the three main axes (default)
    2. Arbitrary angle rotations in 3D space (when full=True)
    
    For arbitrary rotations, different interpolation methods are used:
    - Volume: Bilinear/trilinear interpolation for smooth transformation
    - Label: Nearest-neighbor interpolation to preserve label classes
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        full: bool = False,
        axes: Optional[List[int]] = None,
        angles: Optional[List[float]] = None,
        angle_range: Optional[Tuple[float, float]] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the random rotation augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        full : bool, default=False
            If True, perform arbitrary angle rotations. 
            If False, perform only 90-degree rotations (k*90 where k ∈ {0,1,2,3})
        axes : Optional[List[int]], default=None
            Axes to rotate around. For 3D data, this can be [0,1,2] for [z,y,x].
            If None, all axes will be considered for 90-degree rotations.
            For arbitrary rotations, these are the axes around which to rotate.
        angles : Optional[List[float]], default=None
            Specific angles in degrees to randomly select from.
            Only used when full=True. Overrides angle_range if provided.
        angle_range : Optional[Tuple[float, float]], default=None
            Range of angles in degrees for random selection.
            Only used when full=True and angles=None.
            Default is (-30, 30) if not provided.
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.full = full
        
        # For 90-degree rotations
        if axes is None:
            # Default to all three axes for 3D data
            self.axes = [0, 1, 2]  # z, y, x axes
        else:
            self.axes = axes
            
        # For arbitrary angle rotations
        self.angles = angles
        if angle_range is None:
            self.angle_range = (-30, 30)  # Default range in degrees
        else:
            self.angle_range = angle_range
            
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the rotation transformation to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data
        """
        # Determine if we're dealing with a volume or label
        # This helps us choose the right interpolation method
        is_label = False
        
        # Check if this is the label data
        if self.label is not None and data is self.label:
            is_label = True
        
        # Perform full arbitrary rotation if requested
        if self.full:
            return self._apply_arbitrary_rotation(data, is_label)
        else:
            # Otherwise do 90-degree rotations
            return self._apply_90_degree_rotation(data)
    
    def _apply_90_degree_rotation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply 90-degree rotation to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
            
        Returns
        -------
        torch.Tensor
            Transformed data with 90-degree rotation
        """
        # Select a random axis from available axes
        axis = random.choice(self.axes)
        
        # Select a random number of 90-degree rotations (0, 1, 2, or 3)
        k = random.randint(0, 3)
        
        # Skip if no rotation (k=0)
        if k == 0:
            return data
        
        # Get data dimensionality
        ndim = len(data.shape)
        
        # For 3D volumes (assuming order is [channels, z, y, x] or [z, y, x])
        if ndim == 4:
            # For [channels, z, y, x] format
            dims = list(range(1, 4))  # Skip channel dimension
            rot_dims = [(dims[i], dims[j]) for i, j in [(1, 2), (0, 2), (0, 1)]][axis]
        elif ndim == 3:
            # For [z, y, x] format
            rot_dims = [(1, 2), (0, 2), (0, 1)][axis]
        else:
            raise ValueError(f"Unsupported data dimensionality: {ndim}. Expected 3 or 4.")
        
        # Apply the rotation
        result = data
        for _ in range(k):
            result = torch.rot90(result, dims=rot_dims)
            
        return result
    
    def _apply_arbitrary_rotation(self, data: torch.Tensor, is_label: bool) -> torch.Tensor:
        """Apply arbitrary angle rotation to the data with appropriate interpolation.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to transform
        is_label : bool
            Whether the data is a label map (True) or volume (False)
            
        Returns
        -------
        torch.Tensor
            Transformed data with arbitrary rotation
        """
        # Determine the angles for rotation
        if self.angles is not None:
            # Choose from the provided angles
            angle_x = random.choice(self.angles) if 0 in self.axes else 0
            angle_y = random.choice(self.angles) if 1 in self.axes else 0
            angle_z = random.choice(self.angles) if 2 in self.axes else 0
        else:
            # Generate random angles within the specified range
            angle_x = random.uniform(*self.angle_range) if 0 in self.axes else 0
            angle_y = random.uniform(*self.angle_range) if 1 in self.axes else 0
            angle_z = random.uniform(*self.angle_range) if 2 in self.axes else 0
        
        # Convert to radians
        angle_x = torch.tensor(angle_x * np.pi / 180.0).to(self.device)
        angle_y = torch.tensor(angle_y * np.pi / 180.0).to(self.device)
        angle_z = torch.tensor(angle_z * np.pi / 180.0).to(self.device)
        
        # Create rotation matrices
        # Rotation around X axis
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle_x), -torch.sin(angle_x)],
            [0, torch.sin(angle_x), torch.cos(angle_x)]
        ], device=self.device)
        
        # Rotation around Y axis
        R_y = torch.tensor([
            [torch.cos(angle_y), 0, torch.sin(angle_y)],
            [0, 1, 0],
            [-torch.sin(angle_y), 0, torch.cos(angle_y)]
        ], device=self.device)
        
        # Rotation around Z axis
        R_z = torch.tensor([
            [torch.cos(angle_z), -torch.sin(angle_z), 0],
            [torch.sin(angle_z), torch.cos(angle_z), 0],
            [0, 0, 1]
        ], device=self.device)
        
        # Combined rotation matrix
        R = torch.mm(R_z, torch.mm(R_y, R_x))
        
        # Get data dimensionality and shape
        ndim = len(data.shape)
        
        # Determine if we have a channel dimension
        has_channel_dim = (ndim == 4)
        
        # Handle single-channel 3D data ([C, D, H, W]) or just ([D, H, W])
        if has_channel_dim:
            num_channels = data.shape[0]
            depth, height, width = data.shape[1:]
        else:
            num_channels = 1
            depth, height, width = data.shape
            # Add channel dimension for processing
            data = data.unsqueeze(0)
        
        # Create a grid of coordinates
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )
        
        # Stack coordinates and reshape for matrix multiplication
        coords = torch.stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Apply rotation to coordinates
        rotated_coords = torch.mm(R, coords)
        
        # Reshape back
        x_rotated = rotated_coords[0].reshape(depth, height, width)
        y_rotated = rotated_coords[1].reshape(depth, height, width)
        z_rotated = rotated_coords[2].reshape(depth, height, width)
        
        # Stack rotated coordinates for grid_sample
        grid = torch.stack([x_rotated, y_rotated, z_rotated], dim=-1)
        
        # Expand grid for multiple channels
        grid = grid.unsqueeze(0).expand(num_channels, -1, -1, -1, -1)
        
        # Choose the appropriate interpolation mode based on data type
        if is_label:
            # For label maps, use nearest-neighbor interpolation to preserve classes
            mode = 'nearest'
        else:
            # For volume data, use trilinear interpolation for smooth results
            mode = 'bilinear'  # Note: 'bilinear' becomes trilinear in 3D
        
        # Apply the transformation using grid_sample
        # Add batch dimension for grid_sample
        data = data.unsqueeze(0)  # [1, C, D, H, W] or [1, 1, D, H, W]
        
        # Apply the transformation
        rotated_data = F.grid_sample(
            data, 
            grid.unsqueeze(0),  # Add batch dim: [1, C, D, H, W, 3]
            mode=mode,
            align_corners=True,
            padding_mode='zeros'
        )
        
        # Remove the batch dimension
        rotated_data = rotated_data.squeeze(0)
        
        # Remove the channel dimension if the original data didn't have it
        if not has_channel_dim:
            rotated_data = rotated_data.squeeze(0)
            
        return rotated_data