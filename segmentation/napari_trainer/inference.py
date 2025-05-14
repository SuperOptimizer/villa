import os
import numpy as np
import torch
import napari
from torch.nn import functional as F
from pathlib import Path
import json
from model.build_network_from_config import NetworkFromConfig
import torch.nn as nn
from tqdm.auto import tqdm

class ModelLoader:
    """Class to load a model from a checkpoint with its configuration"""
    
    def __init__(self, checkpoint_path, config_manager=None):
        """
        Initialize the model loader
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint file (.pth)
        config_manager : object, optional
            An existing config manager object. If provided, will use this instead of creating a new one.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Store the provided config manager if any
        self.config_manager = config_manager
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        """
        Load the model and configuration directly from the checkpoint file
        
        Returns
        -------
        model : torch.nn.Module
            The loaded model
        model_config : dict
            The model configuration
            
        Raises
        ------
        ValueError
            If no model configuration is found in the checkpoint file
        """
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Always require model_config in the checkpoint
        if 'model_config' not in checkpoint:
            raise ValueError(f"No model configuration found in checkpoint file: {self.checkpoint_path}")
        
        # Get model configuration from the checkpoint
        model_config = checkpoint['model_config']
        print("Using model configuration from checkpoint")
        
        # If a config manager was provided, update it with the config from the checkpoint
        if self.config_manager is not None:
            print("Updating provided config manager with checkpoint configuration")
            config_mgr = self.config_manager
            config_mgr.model_config = model_config
            config_mgr.inference_config = model_config
        else:
            # Create appropriate config manager based on what's available
            try:
                from main_window import ConfigManager
                config_mgr = ConfigManager(verbose=True)
                
                # Set up the config manager with the model config from checkpoint
                config_mgr.model_config = model_config
                config_mgr.inference_config = model_config
                config_mgr.train_patch_size = model_config.get("train_patch_size", model_config.get("patch_size", [64, 64, 64]))
                config_mgr.train_batch_size = model_config.get("batch_size", 2)
                config_mgr.in_channels = model_config.get("in_channels", 1)
                # Make sure spacing dimension matches the patch size dimension
                config_mgr.spacing = [1] * len(config_mgr.train_patch_size)
                config_mgr.targets = model_config.get("targets", {"default": {"out_channels": 1, "loss_fn": "BCEDiceLoss"}})
                config_mgr.autoconfigure = model_config.get("autoconfigure", True)
                config_mgr.model_name = model_config.get("model_name", self.checkpoint_path.stem.split('_')[0])
                config_mgr.tr_configs = {}
                config_mgr.tr_info = {"model_name": model_config.get("model_name", "model")}
                config_mgr.dataset_config = {}
            except ImportError:
                # If we can't import ConfigManager, create a simple namespace
                from types import SimpleNamespace
                config_mgr = SimpleNamespace(
                    model_config=model_config,
                    inference_config=model_config,
                    train_patch_size=model_config.get("train_patch_size", model_config.get("patch_size", [64, 64, 64])),
                    train_batch_size=model_config.get("batch_size", 2),
                    in_channels=model_config.get("in_channels", 1),
                    spacing=[1] * len(model_config.get("train_patch_size", model_config.get("patch_size", [64, 64, 64]))),
                    targets=model_config.get("targets", {"default": {"out_channels": 1, "loss_fn": "BCEDiceLoss"}}),
                    autoconfigure=model_config.get("autoconfigure", True),
                    model_name=model_config.get("model_name", self.checkpoint_path.stem.split('_')[0]),
                    tr_configs={},
                    tr_info={"model_name": model_config.get("model_name", "model")},
                    dataset_config={}
                )
        
        # Create the model
        model = NetworkFromConfig(config_mgr)
        model.load_state_dict(checkpoint['model'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully with configuration: {model_config.get('model_name', 'unknown')}")
        return model, model_config


def create_gaussian_window(patch_size, sigma_scale=1/4, device=None):
    """
    Create a gaussian window for blending patches, works for both 2D and 3D
    
    Parameters
    ----------
    patch_size : tuple
        Size of the patch (depth, height, width) for 3D or (height, width) for 2D
    sigma_scale : float
        Scale factor for standard deviation relative to patch size
    device : torch.device
        Device to place the window on
        
    Returns
    -------
    window : torch.Tensor
        Gaussian window of shape (1, 1, D, H, W) for 3D or (1, 1, H, W) for 2D
    """
    # Determine if we're creating a 2D or 3D window
    is_3d = len(patch_size) == 3
    
    if is_3d:
        # 3D case - create coordinate grids for 3D
        d_coords = torch.linspace(-1, 1, patch_size[0])
        y_coords = torch.linspace(-1, 1, patch_size[1])
        x_coords = torch.linspace(-1, 1, patch_size[2])
        grid_d, grid_y, grid_x = torch.meshgrid(d_coords, y_coords, x_coords, indexing='ij')
        
        # Compute 3D gaussian
        sigma = sigma_scale * min(patch_size)
        gaussian = torch.exp(-(grid_d**2 + grid_y**2 + grid_x**2) / (2 * sigma**2))
        
        # Reshape to (1, 1, D, H, W) for broadcasting
        gaussian = gaussian.view(1, 1, *patch_size)
    else:
        # 2D case - create coordinate grids for 2D
        y_coords = torch.linspace(-1, 1, patch_size[0])
        x_coords = torch.linspace(-1, 1, patch_size[1])
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute 2D gaussian
        sigma = sigma_scale * min(patch_size)
        gaussian = torch.exp(-(grid_y**2 + grid_x**2) / (2 * sigma**2))
        
        # Reshape to (1, 1, H, W) for broadcasting
        gaussian = gaussian.view(1, 1, *patch_size)
    
    # Move to device if provided
    if device is not None:
        gaussian = gaussian.to(device)
    
    return gaussian


def sliding_window_inference(model, data, patch_size, overlap=0.5, batch_size=1, verbose=True, **kwargs):
    """
    Perform sliding window inference on a 2D image or 3D volume
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference
    data : numpy.ndarray
        Input image/volume of shape (H, W) or (C, H, W) for 2D
        or (D, H, W) or (C, D, H, W) for 3D
    patch_size : tuple
        Size of patches to extract (height, width) for 2D
        or (depth, height, width) for 3D
    overlap : float
        Overlap between patches (0-1)
    batch_size : int
        Batch size for inference
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    output : dict
        Dictionary of output tensors, one for each task
    """
    device = next(model.parameters()).device
    
    # Determine if we're working with 2D or 3D data based on patch size
    is_3d = len(patch_size) == 3
    
    # Handle different input shapes
    input_ndim = len(data.shape)
    
    if is_3d:
        if input_ndim == 3:  # (D, H, W)
            data = data[np.newaxis, ...]  # Add channel dimension (1, D, H, W)
    else:
        if input_ndim == 2:  # (H, W)
            data = data[np.newaxis, ...]  # Add channel dimension (1, H, W)
    
    # Convert to PyTorch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data).float()
    
    # Add batch dimension if not present
    if is_3d and data.dim() == 4:  # (C, D, H, W)
        data = data.unsqueeze(0)  # Add batch dimension (1, C, D, H, W)
    elif not is_3d and data.dim() == 3:  # (C, H, W)
        data = data.unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    data = data.to(device)
    
    # Get spatial dimensions based on whether we're in 2D or 3D mode
    if is_3d:
        _, _, depth, height, width = data.shape
        spatial_dims = (depth, height, width)
    else:
        _, _, height, width = data.shape
        spatial_dims = (height, width)
    
    # Calculate step sizes for patches with overlap
    steps = [int(p * (1 - overlap)) for p in patch_size]
    
    # Calculate the number of patches in each dimension
    num_patches = []
    for i, (dim, patch, step) in enumerate(zip(spatial_dims, patch_size, steps)):
        num_patches.append(max(1, (dim - patch) // step + 1) if dim > patch else 1)
    
    # Initialize output tensors and weight mask for blending
    outputs = {}
    importance_maps = {}
    
    # Create gaussian window for blending
    gaussian_window = create_gaussian_window(patch_size, device=device)
    
    # Get the task names from the model (keys of the output dictionary)
    # Use a dummy forward pass to get the keys
    with torch.no_grad():
        dummy_input = torch.zeros((1, data.shape[1], *patch_size), device=device)
        dummy_output = model(dummy_input)
        task_names = list(dummy_output.keys())
    
    # Initialize aggregation variables for each task
    for task_name in task_names:
        # Get the number of output channels for this task
        num_classes = dummy_output[task_name].shape[1]
        if is_3d:
            outputs[task_name] = torch.zeros((1, num_classes, *spatial_dims), device=device)
            importance_maps[task_name] = torch.zeros((1, 1, *spatial_dims), device=device)
        else:
            outputs[task_name] = torch.zeros((1, num_classes, *spatial_dims), device=device)
            importance_maps[task_name] = torch.zeros((1, 1, *spatial_dims), device=device)
    
    # Generate all patch indices
    if is_3d:
        total_patches = num_patches[0] * num_patches[1] * num_patches[2]
        if verbose:
            print(f"Processing {total_patches} 3D patches with size {patch_size} (overlap: {overlap})")
            patch_iterator = tqdm([(d, h, w) for d in range(num_patches[0]) 
                                for h in range(num_patches[1]) 
                                for w in range(num_patches[2])], 
                                desc="3D Sliding Window", unit="patch")
        else:
            patch_iterator = [(d, h, w) for d in range(num_patches[0]) 
                            for h in range(num_patches[1]) 
                            for w in range(num_patches[2])]
    else:
        total_patches = num_patches[0] * num_patches[1]
        if verbose:
            print(f"Processing {total_patches} 2D patches with size {patch_size} (overlap: {overlap})")
            patch_iterator = tqdm([(h, w) for h in range(num_patches[0]) for w in range(num_patches[1])], 
                                desc="2D Sliding Window", unit="patch")
        else:
            patch_iterator = [(h, w) for h in range(num_patches[0]) for w in range(num_patches[1])]
    
    # Process all patches
    patches = []
    positions = []
    patch_count = 0
    
    for indices in patch_iterator:
        # Calculate start positions based on dimensionality
        if is_3d:
            d_idx, h_idx, w_idx = indices
            start_positions = (
                min(d_idx * steps[0], spatial_dims[0] - patch_size[0]),
                min(h_idx * steps[1], spatial_dims[1] - patch_size[1]),
                min(w_idx * steps[2], spatial_dims[2] - patch_size[2])
            )
            # Extract patch
            patch = data[:, :, 
                        start_positions[0]:start_positions[0] + patch_size[0], 
                        start_positions[1]:start_positions[1] + patch_size[1], 
                        start_positions[2]:start_positions[2] + patch_size[2]]
        else:
            h_idx, w_idx = indices
            start_positions = (
                min(h_idx * steps[0], spatial_dims[0] - patch_size[0]),
                min(w_idx * steps[1], spatial_dims[1] - patch_size[1])
            )
            # Extract patch
            patch = data[:, :, 
                        start_positions[0]:start_positions[0] + patch_size[0], 
                        start_positions[1]:start_positions[1] + patch_size[1]]
        
        # Store patch and position
        patches.append(patch)
        positions.append(start_positions)
        
        # Update counter for batch processing
        patch_count += 1
        
        # Process in batches
        last_batch = patch_count == total_patches
        if len(patches) == batch_size or last_batch:
            # Stack patches into batch
            batch = torch.cat(patches, dim=0)
            
            # Forward pass
            with torch.no_grad():
                batch_output = model(batch)
            
            # Put predictions back into output tensor with gaussian blending
            for b_idx in range(len(patches)):
                start_positions = positions[b_idx]
                
                # Apply predictions for each task
                for task_name in task_names:
                    task_output = batch_output[task_name][b_idx:b_idx+1]
                    
                    # Apply gaussian weighting
                    weighted_output = task_output * gaussian_window
                    
                    # Add to output and importance map - using different indexing for 2D vs 3D
                    if is_3d:
                        start_d, start_h, start_w = start_positions
                        outputs[task_name][:, :, 
                                        start_d:start_d + patch_size[0], 
                                        start_h:start_h + patch_size[1], 
                                        start_w:start_w + patch_size[2]] += weighted_output
                        importance_maps[task_name][:, :, 
                                                start_d:start_d + patch_size[0], 
                                                start_h:start_h + patch_size[1], 
                                                start_w:start_w + patch_size[2]] += gaussian_window
                    else:
                        start_h, start_w = start_positions
                        outputs[task_name][:, :, 
                                        start_h:start_h + patch_size[0], 
                                        start_w:start_w + patch_size[1]] += weighted_output
                        importance_maps[task_name][:, :, 
                                                start_h:start_h + patch_size[0], 
                                                start_w:start_w + patch_size[1]] += gaussian_window
            
            # Clear for next batch
            patches = []
            positions = []
    
    # Normalize by importance map (prevent division by zero)
    results = {}
    for task_name in task_names:
        # Add small epsilon to avoid division by zero
        normalized = outputs[task_name] / (importance_maps[task_name] + 1e-8)
        results[task_name] = normalized.squeeze().cpu().numpy()
    
    return results


def run_inference(viewer, layer, checkpoint_path, patch_size=None, overlap=0.25, batch_size=2, num_dataloader_workers=4):
    """
    Run inference on a napari layer and add the results as new layers.
    Automatically detects 2D or 3D data and uses the appropriate inference method.
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance
    layer : napari.layers.Layer
        Layer to run inference on
    checkpoint_path : str or Path
        Path to the model checkpoint
    patch_size : tuple, optional
        Size of patches to use for inference. If None, will be extracted from the model config.
        For 2D: (height, width)
        For 3D: (depth, height, width)
    overlap : float
        Overlap between patches (0-1)
    batch_size : int
        Batch size for inference
    num_dataloader_workers : int
        Number of worker threads for data loading
        
    Returns
    -------
    new_layers : list of napari.layers.Layer
        List of new layers added to the viewer
    """
    # Try to access the global config manager from main_window
    config_manager = None
    try:
        from main_window import _config_manager
        if _config_manager is not None:
            config_manager = _config_manager
            print("Using global config manager from main_window")
    except (ImportError, AttributeError):
        # Fall back to using the wrapper in inference.py
        print("No global config manager available, using default config")
    
    # Load the model with the config manager if available
    loader = ModelLoader(checkpoint_path, config_manager=config_manager)
    model, model_config = loader.load()
    
    # Get model name from checkpoint path
    checkpoint_path = Path(checkpoint_path)
    model_name = checkpoint_path.stem
    
    # Get the input data
    image_data = layer.data
    
    # Determine if input is 2D or 3D
    is_3d = image_data.ndim == 3 and image_data.shape[0] > 1
    
    # If patch_size is not explicitly provided, extract from model config
    if patch_size is None:
        # Try to get patch size from config
        if 'patch_size' in model_config:
            config_patch_size = model_config['patch_size']
            print(f"Using patch_size from model_config: {config_patch_size}")
            patch_size = tuple(config_patch_size)
        elif 'train_patch_size' in model_config:
            config_patch_size = model_config['train_patch_size']
            print(f"Using train_patch_size from model_config: {config_patch_size}")
            patch_size = tuple(config_patch_size)
        else:
            raise ValueError("No patch_size or train_patch_size found in model config. Cannot proceed with inference.")
    
    print(f"Final patch size for inference: {patch_size}")
    print(f"Data shape: {image_data.shape}, Using {'3D' if is_3d else '2D'} inference")
    
    # Verify that patch size dimensionality matches the data dimensionality
    if is_3d and len(patch_size) != 3:
        raise ValueError(f"3D data requires 3D patch size, but got {patch_size}")
    elif not is_3d and len(patch_size) != 2:
        raise ValueError(f"2D data requires 2D patch size, but got {patch_size}")
    
    # Verify model dimensions match data dimensions
    model_dims = len(patch_size)
    data_dims = 3 if is_3d else 2
    if model_dims != data_dims:
        raise ValueError(f"Model dimensionality ({model_dims}D) must match data dimensionality ({data_dims}D)")
    
    # Run unified sliding window inference
    new_layers = []
    
    try:
        # Use the unified sliding window inference function
        results = sliding_window_inference(
            model, 
            image_data, 
            patch_size, 
            overlap=overlap,
            batch_size=batch_size
        )
        
        # Add results as new layers
        for task_name, result in results.items():
            # Apply activation if needed for binary segmentation tasks
            if result.shape[0] == 2:  # Binary segmentation
                # Apply softmax and take foreground probability
                result_tensor = torch.from_numpy(result)
                result = F.softmax(result_tensor, dim=0).numpy()[1]
            
            # Create layer name
            layer_name = f"{layer.name}_{model_name}_{task_name}"
            
            # Add as image layer
            new_layer = viewer.add_image(
                result,
                name=layer_name,
                colormap='magma',
                blending='additive'
            )
            new_layers.append(new_layer)
    except Exception as e:
        raise e
    
    print(f"Inference completed. Added {len(new_layers)} new layers to the viewer.")
    return new_layers


# Keep the old function name for backward compatibility
def run_inference_on_layer(*args, **kwargs):
    """Legacy function name for backward compatibility, calls run_inference"""
    return run_inference(*args, **kwargs)


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference on napari layers")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--patch_size", type=int, nargs='+', help="Patch size for inference (for 2D: height width, for 3D: depth height width)")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap between patches (0-1)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    args = parser.parse_args()
    
    # Start napari
    viewer = napari.Viewer()
    
    # Add some example data if none is loaded
    if len(viewer.layers) == 0:
        print("No layers found. Please load an image or label layer and try again.")
        print("Example: Add a sample image to get started.")
        # Add sample image
        sample = np.random.rand(512, 512)
        viewer.add_image(sample, name="Sample")
    
    # Define callback for inference button
    def run_inference_callback():
        # Get the currently selected layer
        current_layer = viewer.layers.selection.active
        if current_layer is None:
            print("No layer selected. Please select a layer.")
            return
        
        # Parse patch size
        patch_size = None
        if args.patch_size:
            if len(args.patch_size) == 2:
                patch_size = tuple(args.patch_size)  # 2D
            elif len(args.patch_size) == 3:
                patch_size = tuple(args.patch_size)  # 3D
            else:
                print(f"Invalid patch size: {args.patch_size}. Expected 2 or 3 values.")
                return
            
        # Run inference
        run_inference(
            viewer, 
            current_layer, 
            args.checkpoint,
            patch_size=patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size
        )
    
    # Add button widget to run inference
    import warnings
    from qtpy.QtWidgets import QPushButton
    
    # Create button widget
    btn = QPushButton("Run Inference")
    
    # Connect button directly to inference function
    def on_click():
        run_inference_callback()
        
    btn.clicked.connect(on_click)
    
    # Add button to viewer
    viewer.window.add_dock_widget(btn, name="Inference")
    
    # Run napari
    napari.run()
