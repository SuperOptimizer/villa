import torch
import numpy as np
from typing import Union, Optional, Tuple, List, Dict
import random
from ..base import BaseAugmentation, ArrayLike
import torch.nn.functional as F

class ElasticDeformation(BaseAugmentation):
    """
    Applies elastic deformation to 3D volumes.
    
    This augmentation creates a random displacement field and applies it to 
    deform the volume. It is useful for simulating elastic deformations in 
    medical imaging to improve model robustness and is based on the method
    introduced by Simard et al. for handwritten digit recognition.
    
    The deformation is controlled by:
    - strength: How large the displacement vectors can be (alpha parameter)
    - grid_size: The control point spacing (inversely related to sigma parameter)
    
    Different interpolation modes are used for:
    - Volume data: Bilinear/trilinear interpolation for smooth transformation
    - Label data: Nearest-neighbor interpolation to preserve label classes
    """
    
    def __init__(
        self,
        volume: Optional[ArrayLike] = None,
        label: Optional[ArrayLike] = None,
        strength: float = 10.0,
        grid_size: int = 32,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = 'cuda',
        **kwargs
    ):
        """Initialize the elastic deformation augmentation.
        
        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        strength : float, default=10.0
            The maximum displacement in each direction
            Higher values create stronger deformations
        grid_size : int, default=32
            The spacing between control points in the deformation grid
            Smaller values create more fine-grained deformations
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)
        
        self.strength = strength
        self.grid_size = grid_size
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _create_deformation_field(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Create a random deformation field.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the input data (D, H, W) or (C, D, H, W)
            
        Returns
        -------
        torch.Tensor
            Deformation field with shape (D, H, W, 3)
        """
        # Get spatial dimensions (D, H, W)
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape
        
        # Create a coarse grid of control points
        grid_depth = max(2, depth // self.grid_size)
        grid_height = max(2, height // self.grid_size)
        grid_width = max(2, width // self.grid_size)
        
        # Generate random displacements at control points
        # Each control point gets a (dx, dy, dz) displacement
        control_points = torch.randn(
            3, grid_depth, grid_height, grid_width, 
            device=self.device
        ) * self.strength
        
        # Interpolate to full resolution using trilinear interpolation
        # We need to convert from (3, D, H, W) to (D, H, W, 3) format
        displacement_field = F.interpolate(
            control_points.unsqueeze(0),  # Add batch dim
            size=(depth, height, width),
            mode='trilinear',
            align_corners=True
        ).squeeze(0).permute(1, 2, 3, 0)
        
        return displacement_field
    
    def _apply_deformation(self, data: torch.Tensor, displacement: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        """Apply the deformation field to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data to deform
        displacement : torch.Tensor
            Displacement field
        mode : str, default='bilinear'
            Interpolation mode for grid_sample
            
        Returns
        -------
        torch.Tensor
            Deformed data
        """
        # Get data dimensions
        if data.dim() == 4:  # (C, D, H, W)
            channels, depth, height, width = data.shape
            spatial_dims = (depth, height, width)
        else:  # (D, H, W)
            depth, height, width = data.shape
            spatial_dims = (depth, height, width)
            # Add channel dimension for processing
            data = data.unsqueeze(0)
            channels = 1
        
        # Create base grid (normalized -1 to 1 coordinates)
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )
        
        # Create sampling grid for grid_sample
        base_grid = torch.stack([x, y, z], dim=-1)
        
        # Scale displacement to normalized coordinates
        scale = torch.tensor([
            2.0 / (width - 1), 
            2.0 / (height - 1), 
            2.0 / (depth - 1)
        ], device=self.device)
        
        scaled_displacement = displacement * scale
        
        # Add displacement to base grid
        sampling_grid = base_grid + scaled_displacement
        
        # Reshape and expand grid for multiple channels
        sampling_grid = sampling_grid.unsqueeze(0).expand(channels, -1, -1, -1, -1)
        
        # Add batch dimension
        data = data.unsqueeze(0)  # (1, C, D, H, W)
        
        # Apply the deformation using grid_sample
        deformed_data = F.grid_sample(
            data, 
            sampling_grid,
            mode=mode,
            padding_mode='border',
            align_corners=True
        )
        
        # Remove batch dimension
        deformed_data = deformed_data.squeeze(0)
        
        # Remove channel dimension if original data didn't have it
        if channels == 1 and data.dim() - 1 == 3:  # Original was 3D
            deformed_data = deformed_data.squeeze(0)
        
        return deformed_data
    
    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the deformation transformation to the data.
        
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
        is_label = False
        
        # Check if this is the label data
        if self.label is not None and data is self.label:
            is_label = True
        
        # Create deformation field based on data shape
        displacement = self._create_deformation_field(data.shape)
        
        # Choose the appropriate interpolation mode
        if is_label:
            # For label maps, use nearest-neighbor interpolation to preserve classes
            mode = 'nearest'
        else:
            # For volume data, use bilinear/trilinear interpolation for smooth results
            mode = 'bilinear'  # Note: 'bilinear' becomes trilinear in 3D
        
        # Apply the deformation
        deformed_data = self._apply_deformation(data, displacement, mode=mode)
        
        return deformed_data


class SplineDeformation(BaseAugmentation):
    """
    Applies thin-plate spline deformation to 3D volumes along the z-axis.

    This augmentation creates smooth curvilinear splines that span the entire
    depth of the volume, with control points that are displaced according to
    specified vectors. This is particularly useful for simulating warping and
    bending in scroll-like structures.

    The spline deformation allows for controlled, smooth warping that preserves
    continuity through the volume depth while introducing realistic distortions.

    Different interpolation modes are used for:
    - Volume data: Bilinear/trilinear interpolation for smooth transformation
    - Label data: Nearest-neighbor interpolation to preserve label classes
    """

    def __init__(
            self,
            volume: Optional[ArrayLike] = None,
            label: Optional[ArrayLike] = None,
            num_control_points: int = 8,
            max_displacement: float = 10.0,
            smoothness: float = 2.0,
            depth_coherence: float = 0.8,
            seed: Optional[int] = None,
            device: Optional[Union[str, torch.device]] = 'cuda',
            **kwargs
    ):
        """Initialize the spline deformation augmentation.

        Parameters
        ----------
        volume : Optional[ArrayLike], default=None
            Input volume data as numpy array, torch tensor, or cupy array
        label : Optional[ArrayLike], default=None
            Input label data as numpy array, torch tensor, or cupy array
        num_control_points : int, default=8
            Number of control points per spline
        max_displacement : float, default=10.0
            Maximum displacement magnitude for control points
        smoothness : float, default=2.0
            Controls the smoothness of the deformation (higher = smoother)
        depth_coherence : float, default=0.8
            Controls how much the deformation vectors are correlated across depths
            (0 = independent at each depth, 1 = identical across all depths)
        seed : Optional[int], default=None
            Random seed for reproducibility
        device : Optional[Union[str, torch.device]], default='cuda'
            Device to place tensor data on
        **kwargs
            Additional keyword arguments
        """
        super().__init__(volume, label, device, **kwargs)

        self.num_control_points = num_control_points
        self.max_displacement = max_displacement
        self.smoothness = smoothness
        self.depth_coherence = depth_coherence

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _generate_control_points(self, shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate control points and their displacements.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the input data (D, H, W) or (C, D, H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            source_points: Control point coordinates (N, 3)
            dest_points: Displaced control point coordinates (N, 3)
        """
        # Get spatial dimensions (D, H, W)
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape

        # Calculate number of splines based on image size
        # One spline per 32 pixels (or user parameter)
        spline_spacing = 32
        num_splines_y = max(3, height // spline_spacing)
        num_splines_x = max(3, width // spline_spacing)

        # Place control points in a grid pattern across the y,x plane
        # For each y,x position, place control points along the z-axis
        y_positions = torch.linspace(0, height - 1, num_splines_y, device=self.device)
        x_positions = torch.linspace(0, width - 1, num_splines_x, device=self.device)
        z_positions = torch.linspace(0, depth - 1, self.num_control_points, device=self.device)

        # Create mesh grid for the source control points
        grid_y, grid_x, grid_z = torch.meshgrid(y_positions, x_positions, z_positions, indexing='ij')

        # Reshape to get list of 3D coordinates (y, x, z)
        source_points = torch.stack([grid_z.flatten(), grid_y.flatten(), grid_x.flatten()], dim=1)

        # Generate displacement vectors with some correlation across the Z dimension
        # This ensures splines maintain coherence through depth

        # Start with random displacements at each depth
        displacement_vectors = torch.randn(
            num_splines_y, num_splines_x, self.num_control_points, 2,  # 2 for y and x directions
            device=self.device
        ) * self.max_displacement

        # Apply smoothing along z-axis to ensure depth coherence
        if self.depth_coherence > 0:
            # Create a smoothing kernel for depth
            kernel_size = min(5, self.num_control_points)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size

            depth_kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size

            # Apply smoothing along z dimension (dim=2)
            # Add batch and channel dimensions for conv1d
            smoothed = F.conv1d(
                displacement_vectors.reshape(-1, 1, self.num_control_points),
                depth_kernel,
                padding=kernel_size // 2
            ).squeeze(1).reshape(num_splines_y, num_splines_x, self.num_control_points, 2)

            # Combine original and smoothed based on depth_coherence
            displacement_vectors = (
                    (1 - self.depth_coherence) * displacement_vectors +
                    self.depth_coherence * smoothed
            )

        # Add displacement to y and x coordinates (not to z, to maintain depth ordering)
        displacement_flat = displacement_vectors.view(-1, 2)

        # Create destination points by adding displacements to source points
        dest_points = source_points.clone()
        dest_points[:, 1:3] += displacement_flat  # Apply to y and x coordinates

        # Ensure points stay within bounds
        dest_points[:, 0] = torch.clamp(dest_points[:, 0], 0, depth - 1)
        dest_points[:, 1] = torch.clamp(dest_points[:, 1], 0, height - 1)
        dest_points[:, 2] = torch.clamp(dest_points[:, 2], 0, width - 1)

        return source_points, dest_points

    def _compute_thin_plate_spline(
            self,
            source_points: torch.Tensor,
            dest_points: torch.Tensor,
            shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Compute thin plate spline deformation field.

        Parameters
        ----------
        source_points : torch.Tensor
            Control point coordinates (N, 3)
        dest_points : torch.Tensor
            Displaced control point coordinates (N, 3)
        shape : Tuple[int, ...]
            Shape of the output deformation field

        Returns
        -------
        torch.Tensor
            Deformation field with shape (D, H, W, 3)
        """
        # Get spatial dimensions
        if len(shape) == 4:  # (C, D, H, W)
            depth, height, width = shape[1:]
        else:  # (D, H, W)
            depth, height, width = shape

        # Normalize control point coordinates to [-1, 1] for numerical stability
        source_points_norm = torch.zeros_like(source_points)
        source_points_norm[:, 0] = 2 * (source_points[:, 0] / (depth - 1)) - 1  # z
        source_points_norm[:, 1] = 2 * (source_points[:, 1] / (height - 1)) - 1  # y
        source_points_norm[:, 2] = 2 * (source_points[:, 2] / (width - 1)) - 1  # x

        dest_points_norm = torch.zeros_like(dest_points)
        dest_points_norm[:, 0] = 2 * (dest_points[:, 0] / (depth - 1)) - 1  # z
        dest_points_norm[:, 1] = 2 * (dest_points[:, 1] / (height - 1)) - 1  # y
        dest_points_norm[:, 2] = 2 * (dest_points[:, 2] / (width - 1)) - 1  # x

        # Create target grid points
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )
        grid_points = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

        # Compute pairwise distances for radial basis function
        n_control_points = source_points_norm.shape[0]
        n_grid_points = grid_points.shape[0]

        # Compute distances between grid points and source control points
        # This is the most computationally expensive part
        expanded_grid_points = grid_points.unsqueeze(1).expand(-1, n_control_points, -1)
        expanded_control_points = source_points_norm.unsqueeze(0).expand(n_grid_points, -1, -1)

        # Compute squared Euclidean distance
        squared_dist = torch.sum((expanded_grid_points - expanded_control_points) ** 2, dim=2)

        # Apply radial basis function with smoothness parameter
        # U(r) = r^2 * log(r) for thin plate spline
        rbf = squared_dist * torch.log(torch.clamp(squared_dist, min=1e-8))
        rbf = rbf / self.smoothness

        # Compute displacements based on the RBF values
        displacements = torch.zeros((n_grid_points, 3), device=self.device)

        # For each dimension (z, y, x), compute weights based on control point displacements
        for dim in range(3):
            # Compute weights for this dimension
            control_displacements = dest_points_norm[:, dim] - source_points_norm[:, dim]
            displacements[:, dim] = torch.matmul(rbf, control_displacements)

        # Reshape displacements to match input shape
        displacement_field = displacements.reshape(depth, height, width, 3)

        return displacement_field

    def _apply_deformation(self, data: torch.Tensor, displacement_field: torch.Tensor,
                           mode: str = 'bilinear') -> torch.Tensor:
        """Apply the spline deformation to the data.

        Parameters
        ----------
        data : torch.Tensor
            Input data to deform
        displacement_field : torch.Tensor
            Displacement field with shape (D, H, W, 3)
        mode : str, default='bilinear'
            Interpolation mode for grid_sample

        Returns
        -------
        torch.Tensor
            Deformed data
        """
        # Get data dimensions
        if data.dim() == 4:  # (C, D, H, W)
            channels, depth, height, width = data.shape
            spatial_dims = (depth, height, width)
        else:  # (D, H, W)
            depth, height, width = data.shape
            spatial_dims = (depth, height, width)
            # Add channel dimension for processing
            data = data.unsqueeze(0)
            channels = 1

        # Create base grid (normalized -1 to 1 coordinates)
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=self.device),
            torch.linspace(-1, 1, height, device=self.device),
            torch.linspace(-1, 1, width, device=self.device),
            indexing='ij'
        )
        base_grid = torch.stack([x, y, z], dim=-1)  # Grid shape: [D, H, W, 3]

        # Add displacement to base grid
        sampling_grid = base_grid + displacement_field

        # Reshape and expand grid for multiple channels
        sampling_grid = sampling_grid.unsqueeze(0).expand(channels, -1, -1, -1, -1)

        # Add batch dimension
        data = data.unsqueeze(0)  # (1, C, D, H, W)

        # Apply the deformation using grid_sample
        deformed_data = F.grid_sample(
            data,
            sampling_grid,
            mode=mode,
            padding_mode='border',
            align_corners=True
        )

        # Remove batch dimension
        deformed_data = deformed_data.squeeze(0)

        # Remove channel dimension if original data didn't have it
        if channels == 1 and len(spatial_dims) == 3:  # Original was 3D
            deformed_data = deformed_data.squeeze(0)

        return deformed_data

    def _apply_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the spline deformation transformation to the data.

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
        is_label = False

        # Check if this is the label data
        if self.label is not None and data is self.label:
            is_label = True

        # Generate control points for the thin plate spline
        source_points, dest_points = self._generate_control_points(data.shape)

        # Compute deformation field using thin plate spline
        displacement_field = self._compute_thin_plate_spline(
            source_points, dest_points, data.shape
        )

        # Choose appropriate interpolation mode
        if is_label:
            # For label maps, use nearest-neighbor interpolation to preserve classes
            mode = 'nearest'
        else:
            # For volume data, use bilinear/trilinear interpolation for smooth results
            mode = 'bilinear'  # Note: 'bilinear' becomes trilinear in 3D context

        # Apply the deformation
        deformed_data = self._apply_deformation(data, displacement_field, mode=mode)

        return deformed_data