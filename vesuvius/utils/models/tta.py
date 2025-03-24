import torch
from typing import Dict, List, Any, Optional, Tuple
from vesuvius.utils.models.helpers import merge_tensors


def get_tta_transformations(
    input_tensor: torch.Tensor,
    use_mirroring: bool = True,
    allowed_mirroring_axes: Optional[List[int]] = None,
    max_tta_combinations: Optional[int] = None,
    use_rotation_tta: bool = False,
    verbose: bool = False,
    rank: int = 0
) -> List[Dict[str, Any]]:
    """
    Generate test time augmentation (TTA) transformations for an input tensor.
    
    Args:
        input_tensor: Input tensor of shape [B, C, ...] with dimensions matching model's expected input
        use_mirroring: Whether to use mirroring-based TTA
        allowed_mirroring_axes: List of axes to mirror (without batch and channel dimensions)
        max_tta_combinations: If set, limits which TTA combinations to use
        use_rotation_tta: If True, use rotation-based TTA instead of mirror-based TTA
        verbose: Whether to print verbose output
        rank: Process rank for distributed processing (for logging)
        
    Returns:
        List of dictionaries, each containing:
            - 'transform_type': str - Type of transform ('identity', 'flip', or 'rotation')
            - 'axes': Tuple or List - Axes used for transformation
            - 'weight': float - Weight for this transformation in blending
            - 'func': Callable - Function to apply the transformation to input tensor
            - 'inverse_func': Callable - Function to revert the transformation on output tensor
    """
    transforms = []
    
    # Always add the identity transformation (no augmentation)
    transforms.append({
        'transform_type': 'identity',
        'axes': None,
        'weight': 1.0,
        'func': lambda x: x,
        'inverse_func': lambda x: x
    })
    
    # If TTA is disabled, just return the identity transform
    if not use_mirroring:
        if verbose and rank == 0:
            print("Test time augmentation is disabled")
        return transforms
    
    # Use rotation-based TTA if requested and has 3D input (5D tensor with batch and channel dims)
    if use_rotation_tta and len(input_tensor.shape) == 5:  # [B, C, Z, Y, X]
        if verbose and rank == 0:
            print("Using rotation-based TTA (each axis as top)")
        
        # Define how many rotations we'll perform (default to 3 for 3D volumes)
        num_rotations = 3 if max_tta_combinations is None else min(3, max_tta_combinations)
        
        # Set up default weights
        weights = [1.0, 1.0, 1.0]
        
        # Helper functions for rotations
        def rotate_tensor(x, axes_order):
            """
            Reorder axes of the input tensor
            axes_order is a list specifying the new order of spatial dimensions
            e.g. [0, 2, 1] would swap Y and X axes
            """
            # Keep batch and channel dimensions unchanged
            return x.permute(0, 1, *[i+2 for i in axes_order])
            
        def rotate_back(x, axes_order):
            """Calculate the inverse permutation"""
            inverse_order = [0, 1]  # Batch and channel stay the same
            for i in range(len(axes_order)):
                inverse_order.append(axes_order.index(i) + 2)
            return x.permute(inverse_order)
        
        # First transform is already in the list (identity/original orientation)
        
        # Apply rotation 2: Make X the top [B, C, X, Y, Z]
        if num_rotations >= 2:
            axes_order = [2, 1, 0]  # New order of spatial dimensions
            transforms.append({
                'transform_type': 'rotation',
                'axes': axes_order,
                'weight': weights[1],
                'func': lambda x, axes=axes_order: rotate_tensor(x, axes),
                'inverse_func': lambda x, axes=axes_order: rotate_back(x, axes)
            })
            
        # Apply rotation 3: Make Y the top [B, C, Y, X, Z]
        if num_rotations >= 3:
            axes_order = [1, 2, 0]  # New order of spatial dimensions
            transforms.append({
                'transform_type': 'rotation',
                'axes': axes_order,
                'weight': weights[2],
                'func': lambda x, axes=axes_order: rotate_tensor(x, axes),
                'inverse_func': lambda x, axes=axes_order: rotate_back(x, axes)
            })
            
    # Otherwise, use standard mirror-based TTA
    elif allowed_mirroring_axes is not None and not use_rotation_tta:
        # Adjust mirror axes to account for batch and channel dimensions
        mirror_axes = [i + 2 for i in allowed_mirroring_axes]  # +2 for batch and channel dimensions
        if verbose and rank == 0:
            print(f"Using mirror-based TTA with mirroring axes: {allowed_mirroring_axes}")
        
        # Import itertools for combinations
        import itertools
        
        # Generate all possible combinations of mirroring axes
        axes_combinations = [
            c for i in range(len(mirror_axes)) 
            for c in itertools.combinations(mirror_axes, i + 1)
        ]
        
        # Store original combinations count
        original_combinations_count = len(axes_combinations)
        
        # Limit the number of combinations if specified
        if max_tta_combinations is not None and original_combinations_count > max_tta_combinations:
            if verbose and rank == 0:
                print(f"Limiting TTA combinations from {original_combinations_count} to {max_tta_combinations}")
            
            # For exactly 3 combinations, use the 3 single-axis flips
            if max_tta_combinations == 3:
                # Get the single-axis flips
                single_axis_flips = [c for c in axes_combinations if len(c) == 1]
                
                if len(single_axis_flips) == 3:
                    # Found all 3 primary axis flips - use these
                    axes_combinations = single_axis_flips
            
            else:
                # For other counts, prioritize single-axis flips first, then others
                single_axis_flips = [c for c in axes_combinations if len(c) == 1]
                other_flips = [c for c in axes_combinations if len(c) > 1]
                
                # Sort the remaining flips by length
                other_flips.sort(key=len)
                
                # Determine how many additional flips we can include
                num_primary = len(single_axis_flips)
                
                # Make sure we include at least the single-axis flips if possible
                if max_tta_combinations >= num_primary:
                    # Include all single-axis flips + as many others as will fit
                    remaining_slots = max_tta_combinations - num_primary
                    axes_combinations = single_axis_flips + other_flips[:remaining_slots]
                    
                    if verbose and rank == 0:
                        print(f"  - Included all {num_primary} single-axis flips + {remaining_slots} additional flips")
                else:
                    # Not enough slots even for all single-axis flips
                    axes_combinations.sort(key=len)
                    axes_combinations = axes_combinations[:max_tta_combinations]
                    
                    if verbose and rank == 0:
                        print(f"  - WARNING: Not enough capacity for all single-axis flips")
        
        # Add all mirror transformations
        for axes in axes_combinations:
            # Add to transforms list
            transforms.append({
                'transform_type': 'flip',
                'axes': axes,
                'weight': 1.0,
                'func': lambda x, ax=axes: torch.flip(x, ax),
                'inverse_func': lambda x, ax=axes: torch.flip(x, ax)  # Mirror is its own inverse
            })
    
    else:
        # No TTA being applied (rotation TTA failed or mirror TTA disabled)
        if verbose and rank == 0:
            if use_rotation_tta:
                print("Rotation TTA requested but input is not 5D, using original prediction only")
            else:
                print("No mirroring axes specified or TTA disabled, using original prediction only")
    
    # Normalize weights if we have more than just the identity transform
    if len(transforms) > 1:
        weight_sum = sum(t['weight'] for t in transforms)
        for t in transforms:
            t['weight'] = t['weight'] / weight_sum
    
    return transforms


def apply_tta_transforms(input_tensor: torch.Tensor, transforms: List[Dict[str, Any]]) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Apply a list of TTA transformations to an input tensor.
    
    Args:
        input_tensor: The input tensor to transform
        transforms: List of transformation dictionaries from get_tta_transformations()
        
    Returns:
        List of tuples, each containing:
            - The transformed input tensor
            - The transform dictionary for reference (used for inverse transform)
    """
    transformed_inputs = []
    
    for transform in transforms:
        # Apply the transformation function
        transformed = transform['func'](input_tensor)
        transformed_inputs.append((transformed, transform))
    
    return transformed_inputs


def combine_tta_outputs(outputs: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> torch.Tensor:
    """
    Combine the outputs from multiple TTA transformations.
    
    Args:
        outputs: List of tuples, each containing:
            - Output tensor from model
            - Transform dictionary used to generate the input
            
    Returns:
        Combined output tensor
    """
    # Use the general-purpose tensor merging function
    return merge_tensors(outputs)


def get_tta_augmented_inputs(
    input_tensor: torch.Tensor,
    model_info: Dict[str, Any],
    max_tta_combinations: Optional[int] = None,
    use_rotation_tta: bool = False,
    rotation_weights: Optional[List[float]] = None,
    verbose: bool = False,
    rank: int = 0
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """
    Generate a batch of TTA-augmented inputs for inference.
    
    Args:
        input_tensor: Input tensor of shape [B, C, ...] with dimensions matching model's expected input
        model_info: Dictionary with model information
        max_tta_combinations: If set, limits which TTA combinations to use
        use_rotation_tta: If True, use rotation-based TTA instead of mirror-based TTA
        rotation_weights: Optional list of weights for rotation-based TTA
        verbose: Whether to print verbose output
        rank: Process rank for distributed processing
        
    Returns:
        Tuple containing:
            - List of transformed input tensors
            - List of transform dictionaries for inverse transforms
    """
    # Configure TTA settings from model_info
    use_mirroring = model_info.get('use_mirroring', True)
    allowed_mirroring_axes = model_info.get('allowed_mirroring_axes', None)
    
    # Get transformations
    transforms = get_tta_transformations(
        input_tensor=input_tensor,
        use_mirroring=use_mirroring,
        allowed_mirroring_axes=allowed_mirroring_axes,
        max_tta_combinations=max_tta_combinations,
        use_rotation_tta=use_rotation_tta,
        verbose=verbose,
        rank=rank
    )
    
    # If using custom rotation weights, override the weights in the transforms
    if rotation_weights is not None and use_rotation_tta:
        weights = rotation_weights
        if len(weights) < 3:
            # Pad with 1.0 if needed
            weights = weights + [1.0] * (3 - len(weights))
        elif len(weights) > 3:
            # Truncate if too many
            weights = weights[:3]
            
        # Apply the weights to the rotation transforms
        weight_idx = 0
        for i, transform in enumerate(transforms):
            if transform['transform_type'] in ['identity', 'rotation']:
                if weight_idx < len(weights):
                    transform['weight'] = weights[weight_idx]
                    weight_idx += 1
        
        # Normalize weights
        weight_sum = sum(t['weight'] for t in transforms)
        for t in transforms:
            t['weight'] = t['weight'] / weight_sum
    
    # Apply transformations to get augmented inputs
    augmented_inputs = []
    transform_info = []
    
    for transform in transforms:
        # Apply the transformation
        transformed = transform['func'](input_tensor)
        augmented_inputs.append(transformed)
        transform_info.append(transform)
    
    return augmented_inputs, transform_info