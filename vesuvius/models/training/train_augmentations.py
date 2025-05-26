import albumentations as A


def compose_augmentations(dimension, additional_targets=None):
    """
    Create and return training augmentations composition.
    
    Parameters
    ----------
    dimension : str
        Either '2d' or '3d' to specify which augmentations to apply
    additional_targets : dict, optional
        Dictionary mapping target names to target types for albumentations.
        For example: {'mask_sheet': 'mask', 'mask_normals': 'mask'}
        
    Returns
    -------
    albumentations.Compose or None
        Composed augmentation pipeline for training (2D) or None (3D placeholder)
    """
    if additional_targets is None:
        additional_targets = {}

    if dimension == '2d':
        # --- Augmentations (2D only) ---
        image_transforms = A.Compose([

            A.OneOf([
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip()
            ], p=0.3),

        ], additional_targets=additional_targets, p=1.0)  # Always apply the composition

        return image_transforms
    
    elif dimension == '3d':
        # TODO: Implement 3D augmentations
        # For now, return None as placeholder
        return None
    
    else:
        raise ValueError(f"Invalid dimension '{dimension}'. Must be '2d' or '3d'.")
