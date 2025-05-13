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
        
        # Try to find corresponding config file in same directory
        self.config_path = self.checkpoint_path.with_name(
            self.checkpoint_path.stem + "_config.json"
        )
        
        # If not found, look in parent or model-specific directory
        if not self.config_path.exists():
            # Check if we're in a model-specific directory and try parent
            parent_dir = self.checkpoint_path.parent.parent
            model_name = self.checkpoint_path.parent.name
            
            # Try parent directory
            parent_config = parent_dir / f"{model_name}_config.json"
            if parent_config.exists():
                print(f"Found config in parent directory: {parent_config}")
                self.config_path = parent_config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        """
        Load the model and configuration
        
        Returns
        -------
        model : torch.nn.Module
            The loaded model
        """
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # If a real config manager was provided, use it directly
        if self.config_manager is not None:
            print("Using provided config manager")
            config_mgr = self.config_manager
            model_config = getattr(config_mgr, 'model_config', {})
        else:
            # No config manager was provided - import and create a real one
            try:
                from main_window import ConfigManager
                
                # Get model configuration, with fallbacks
                model_config = None
                
                # First try to get config from checkpoint
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    print("Using model configuration from checkpoint")
                
                # Next try separate config file if found
                elif self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config_data = json.load(f)
                        model_config = config_data.get('model', {})
                    print(f"Using model configuration from {self.config_path}")
                
                # Finally, fall back to default configuration
                else:
                    print("No model configuration found. Using default configuration.")
                    model_config = {
                        "model_name": self.checkpoint_path.stem.split('_')[0],
                        "in_channels": 1,
                        "out_channels": 1,
                        "train_patch_size": [64, 64, 64],
                        "autoconfigure": True,
                        "conv_op": "nn.Conv3d"
                    }
                
                # Create a real config manager
                config_mgr = ConfigManager(verbose=True)
                
                # First check if there's a complete config in the JSON file
                if self.config_path.exists():
                    print(f"Reading complete config from {self.config_path}")
                    try:
                        with open(self.config_path, 'r') as f:
                            config_data = json.load(f)
                        
                        # Initialize config sections
                        config_mgr.model_config = config_data
                        config_mgr.inference_config = config_data
                        config_mgr.tr_configs = {}
                        config_mgr.tr_info = {"model_name": config_data.get("model_name", "model")}
                        config_mgr.dataset_config = {}
                        
                        # Set essential attributes from the config
                        # Check for and print patch_size value explicitly for debugging
                        if "patch_size" in config_data:
                            print(f"Found patch_size in config: {config_data['patch_size']}")
                        config_mgr.train_patch_size = config_data.get("patch_size", [64, 64, 64])
                        config_mgr.train_batch_size = config_data.get("batch_size", 2)
                        config_mgr.in_channels = config_data.get("in_channels", 1)
                        config_mgr.autoconfigure = config_data.get("autoconfigure", False)
                        config_mgr.model_name = config_data.get("model_name", self.checkpoint_path.stem.split('_')[0])
                        config_mgr.spacing = [1, 1, 1]
                        config_mgr.targets = config_data.get("targets", {"default": {"out_channels": 1, "loss_fn": "BCEDiceLoss"}})
                        
                        print(f"Successfully loaded full configuration from {self.config_path}")
                        
                    except Exception as e:
                        print(f"Error loading config from file: {e}")
                        # Continue with fallback approach below
                
                # If no config file or error occurred, set up with model_config 
                if not hasattr(config_mgr, 'model_config') or not config_mgr.model_config:
                    # Set up the config manager with our data
                    config_mgr.model_config = model_config
                    config_mgr.inference_config = model_config  # Add inference_config
                    config_mgr.train_patch_size = model_config.get("train_patch_size", model_config.get("patch_size", [64, 64, 64]))
                    config_mgr.train_batch_size = model_config.get("batch_size", 2)
                    config_mgr.in_channels = model_config.get("in_channels", 1)
                    config_mgr.spacing = [1, 1, 1]
                    config_mgr.targets = model_config.get("targets", {"default": {"out_channels": 1, "loss_fn": "BCEDiceLoss"}})
                    config_mgr.autoconfigure = model_config.get("autoconfigure", True)
                    config_mgr.model_name = model_config.get("model_name", self.checkpoint_path.stem.split('_')[0])
                
            except ImportError:
                # If we can't import ConfigManager, we're in a standalone script
                # Create an object with the minimum required attributes
                print("Cannot import ConfigManager. Creating a minimal configuration object.")
                from types import SimpleNamespace
                
                # Get model configuration, with fallbacks
                model_config = None
                
                # First try to get config from checkpoint
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    print("Using model configuration from checkpoint")
                
                # Next try separate config file if found
                elif self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config_data = json.load(f)
                        model_config = config_data.get('model', {})
                    print(f"Using model configuration from {self.config_path}")
                
                # Finally, fall back to default configuration
                else:
                    print("No model configuration found. Using default configuration.")
                    model_config = {
                        "model_name": self.checkpoint_path.stem.split('_')[0],
                        "in_channels": 1,
                        "out_channels": 1,
                        "train_patch_size": [64, 64, 64],
                        "autoconfigure": True,
                        "conv_op": "nn.Conv3d"
                    }
                
                # Create a simple object with the necessary attributes
                config_mgr = SimpleNamespace(
                    model_config=model_config,
                    inference_config=model_config,  # Add inference_config
                    train_patch_size=model_config.get("train_patch_size", model_config.get("patch_size", [64, 64, 64])),
                    train_batch_size=model_config.get("batch_size", 2),
                    in_channels=model_config.get("in_channels", 1),
                    spacing=[1, 1, 1],
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


def sliding_window_inference_2d(model, image, patch_size, overlap=0.5, batch_size=4, verbose=True, **kwargs):
    """
    Perform sliding window inference on a 2D image
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference
    image : numpy.ndarray
        Input image of shape (H, W) or (C, H, W)
    patch_size : tuple
        Size of patches to extract (height, width)
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
    
    # Handle different input shapes
    if len(image.shape) == 2:  # (H, W)
        image = image[np.newaxis, ...]  # Add channel dimension (1, H, W)
    
    # Convert to PyTorch tensor
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    
    if image.dim() == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)
    
    image = image.to(device)
    
    _, _, height, width = image.shape
    
    # Calculate step sizes for patches with overlap
    step_h = int(patch_size[0] * (1 - overlap))
    step_w = int(patch_size[1] * (1 - overlap))
    
    # Calculate the number of patches in each dimension
    num_patches_h = max(1, (height - patch_size[0]) // step_h + 1) if height > patch_size[0] else 1
    num_patches_w = max(1, (width - patch_size[1]) // step_w + 1) if width > patch_size[1] else 1
    
    # Initialize output tensors and weight mask for blending
    outputs = {}
    importance_maps = {}
    
    # Create gaussian window for blending
    gaussian_window = create_gaussian_window(patch_size, device=device)
    
    # Get the task names from the model (keys of the output dictionary)
    # Use a dummy forward pass to get the keys
    with torch.no_grad():
        dummy_input = torch.zeros((1, image.shape[1], patch_size[0], patch_size[1]), device=device)
        dummy_output = model(dummy_input)
        task_names = list(dummy_output.keys())
    
    # Initialize aggregation variables for each task
    for task_name in task_names:
        # Get the number of output channels for this task
        num_classes = dummy_output[task_name].shape[1]
        outputs[task_name] = torch.zeros((1, num_classes, height, width), device=device)
        importance_maps[task_name] = torch.zeros((1, 1, height, width), device=device)
    
    # Process all patches
    patches = []
    positions = []
    
    # Generate all patches
    total_patches = num_patches_h * num_patches_w
    
    if verbose:
        print(f"Processing {total_patches} patches with size {patch_size} (overlap: {overlap})")
        # Create a flattened list of all patch indices for tqdm
        patch_iterator = tqdm([(h, w) for h in range(num_patches_h) for w in range(num_patches_w)], 
                              desc="2D Sliding Window", unit="patch")
    else:
        patch_iterator = [(h, w) for h in range(num_patches_h) for w in range(num_patches_w)]
    
    for h_idx, w_idx in patch_iterator:
        # Calculate start positions
        start_h = min(h_idx * step_h, height - patch_size[0])
        start_w = min(w_idx * step_w, width - patch_size[1])
        
        # Extract patch
        patch = image[:, :, start_h:start_h + patch_size[0], start_w:start_w + patch_size[1]]
        
        # Store patch and position
        patches.append(patch)
        positions.append((start_h, start_w))
        
        # Process in batches
        if len(patches) == batch_size or (h_idx == num_patches_h - 1 and w_idx == num_patches_w - 1):
            # Stack patches into batch
            batch = torch.cat(patches, dim=0)
            
            # Forward pass
            with torch.no_grad():
                batch_output = model(batch)
            
            # Put predictions back into output tensor with gaussian blending
            for b_idx in range(len(patches)):
                start_h, start_w = positions[b_idx]
                
                # Apply predictions for each task
                for task_name in task_names:
                    task_output = batch_output[task_name][b_idx:b_idx+1]
                    
                    # Apply gaussian weighting
                    weighted_output = task_output * gaussian_window
                    
                    # Add to output and importance map
                    outputs[task_name][:, :, start_h:start_h + patch_size[0], start_w:start_w + patch_size[1]] += weighted_output
                    importance_maps[task_name][:, :, start_h:start_h + patch_size[0], start_w:start_w + patch_size[1]] += gaussian_window
            
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


def sliding_window_inference_3d(model, volume, patch_size, overlap=0.5, batch_size=1, verbose=True, **kwargs):
    """
    Perform sliding window inference on a 3D volume
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference
    volume : numpy.ndarray
        Input volume of shape (D, H, W) or (C, D, H, W)
    patch_size : tuple
        Size of patches to extract (depth, height, width)
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
    
    # Handle different input shapes
    if len(volume.shape) == 3:  # (D, H, W)
        volume = volume[np.newaxis, ...]  # Add channel dimension (1, D, H, W)
    
    # Convert to PyTorch tensor
    if not isinstance(volume, torch.Tensor):
        volume = torch.from_numpy(volume).float()
    
    if volume.dim() == 4:  # (C, D, H, W)
        volume = volume.unsqueeze(0)  # Add batch dimension (1, C, D, H, W)
    
    volume = volume.to(device)
    
    _, _, depth, height, width = volume.shape
    
    # Calculate step sizes for patches with overlap
    step_d = int(patch_size[0] * (1 - overlap))
    step_h = int(patch_size[1] * (1 - overlap))
    step_w = int(patch_size[2] * (1 - overlap))
    
    # Calculate the number of patches in each dimension
    num_patches_d = max(1, (depth - patch_size[0]) // step_d + 1) if depth > patch_size[0] else 1
    num_patches_h = max(1, (height - patch_size[1]) // step_h + 1) if height > patch_size[1] else 1
    num_patches_w = max(1, (width - patch_size[2]) // step_w + 1) if width > patch_size[2] else 1
    
    # Initialize output tensors and weight mask for blending
    outputs = {}
    importance_maps = {}
    
    # Create gaussian window for blending
    gaussian_window = create_gaussian_window(patch_size, device=device)
    
    # Get the task names from the model (keys of the output dictionary)
    # Use a dummy forward pass to get the keys
    with torch.no_grad():
        dummy_input = torch.zeros((1, volume.shape[1], *patch_size), device=device)
        dummy_output = model(dummy_input)
        task_names = list(dummy_output.keys())
    
    # Initialize aggregation variables for each task
    for task_name in task_names:
        # Get the number of output channels for this task
        num_classes = dummy_output[task_name].shape[1]
        outputs[task_name] = torch.zeros((1, num_classes, depth, height, width), device=device)
        importance_maps[task_name] = torch.zeros((1, 1, depth, height, width), device=device)
    
    # Process all patches
    patches = []
    positions = []
    
    # Generate all patches
    total_patches = num_patches_d * num_patches_h * num_patches_w
    
    if verbose:
        print(f"Processing {total_patches} 3D patches with size {patch_size} (overlap: {overlap})")
        # Create a flattened list of all patch indices for tqdm
        patch_iterator = tqdm([(d, h, w) for d in range(num_patches_d) 
                               for h in range(num_patches_h) 
                               for w in range(num_patches_w)], 
                              desc="3D Sliding Window", unit="patch")
    else:
        patch_iterator = [(d, h, w) for d in range(num_patches_d) 
                        for h in range(num_patches_h) 
                        for w in range(num_patches_w)]
    
    patch_count = 0  # Keep track for batch processing
    for d_idx, h_idx, w_idx in patch_iterator:
        # Calculate start positions
        start_d = min(d_idx * step_d, depth - patch_size[0])
        start_h = min(h_idx * step_h, height - patch_size[1])
        start_w = min(w_idx * step_w, width - patch_size[2])
        
        # Extract patch
        patch = volume[:, :, start_d:start_d + patch_size[0], start_h:start_h + patch_size[1], start_w:start_w + patch_size[2]]
        
        # Store patch and position
        patches.append(patch)
        positions.append((start_d, start_h, start_w))
        
        # Update counter for batch processing
        patch_count += 1
        
        # Process in batches
        if len(patches) == batch_size or patch_count == total_patches:
            # Stack patches into batch
            batch = torch.cat(patches, dim=0)
            
            # Forward pass
            with torch.no_grad():
                batch_output = model(batch)
            
            # Put predictions back into output tensor with gaussian blending
            for b_idx in range(len(patches)):
                start_d, start_h, start_w = positions[b_idx]
                
                # Apply predictions for each task
                for task_name in task_names:
                    task_output = batch_output[task_name][b_idx:b_idx+1]
                    
                    # Apply gaussian weighting
                    weighted_output = task_output * gaussian_window
                    
                    # Add to output and importance map
                    outputs[task_name][:, :, start_d:start_d + patch_size[0], start_h:start_h + patch_size[1], start_w:start_w + patch_size[2]] += weighted_output
                    importance_maps[task_name][:, :, start_d:start_d + patch_size[0], start_h:start_h + patch_size[1], start_w:start_w + patch_size[2]] += gaussian_window
            
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
    
    # First, check if we're in a model-specific directory structure
    if checkpoint_path.parent.name == checkpoint_path.parent.parent.name:
        # We're in a subdirectory named after the model
        model_name = checkpoint_path.parent.name
    else:
        # Traditional path - extract from filename
        model_name = checkpoint_path.stem
        if "_" in model_name:
            # Remove epoch number if present (e.g. model_name_10 -> model_name)
            parts = model_name.split("_")
            if parts[-1].isdigit():
                model_name = "_".join(parts[:-1])
    
    # Get the input data
    image_data = layer.data
    
    # Determine if input is 2D or 3D
    is_3d = image_data.ndim == 3 and image_data.shape[0] > 1
    
    # If patch_size is not provided, extract from model config
    if patch_size is None:
        if 'train_patch_size' in model_config:
            config_patch_size = model_config['train_patch_size']
            
            # Handle different dimensions for 2D/3D
            if is_3d and len(config_patch_size) == 3:
                # Use as is for 3D
                patch_size = tuple(config_patch_size)
            elif is_3d and len(config_patch_size) == 2:
                # Convert 2D patch to 3D by adding a z dimension
                patch_size = (min(16, image_data.shape[0]), *config_patch_size)
            elif not is_3d and len(config_patch_size) == 3:
                # Use y,x dimensions for 2D
                patch_size = tuple(config_patch_size[1:])
            else:
                # Use as is for 2D
                patch_size = tuple(config_patch_size)
        else:
            # Set default patch sizes
            if is_3d:
                patch_size = (16, 128, 128)  # default 3D patch size
            else:
                patch_size = (256, 256)  # default 2D patch size
    
    print(f"Using patch size: {patch_size}")
    print(f"Data shape: {image_data.shape}, Using {'3D' if is_3d else '2D'} inference")
    
    # For 3D data, use 3D inference directly
    new_layers = []
    
    if is_3d:
        # Run 3D inference
        try:
            results = sliding_window_inference_3d(
                model, 
                image_data, 
                patch_size, 
                overlap=overlap,
                batch_size=batch_size
            )
            
            # Add 3D results as new layers
            for task_name, result_3d in results.items():
                # Apply activation if needed
                if result_3d.shape[0] == 2:  # Binary segmentation
                    # Apply softmax and take foreground probability
                    result_3d_tensor = torch.from_numpy(result_3d)
                    result_3d = F.softmax(result_3d_tensor, dim=0).numpy()[1]
                
                # Create layer name
                layer_name = f"{layer.name}_{model_name}_{task_name}"
                
                # Add as image layer
                new_layer = viewer.add_image(
                    result_3d,
                    name=layer_name,
                    colormap='magma',
                    blending='additive'
                )
                new_layers.append(new_layer)
        except Exception as e:
            print(f"Error during 3D inference: {e}")
            print("Falling back to slice-by-slice 2D inference")
            
            # Fall back to slice-by-slice 2D inference
            results_3d = {}
            num_slices = image_data.shape[0]
            
            print(f"Processing 3D volume with {num_slices} slices")
            
            for z in range(num_slices):
                print(f"Processing slice {z+1}/{num_slices} ({(z+1)/num_slices*100:.1f}%)")
                
                # Get the 2D slice
                slice_data = image_data[z]
                
                # Run inference on this slice
                slice_results = sliding_window_inference_2d(
                    model, 
                    slice_data, 
                    patch_size[-2:],  # Use only height, width for 2D
                    overlap=overlap,
                    batch_size=batch_size
                )
                
                # Initialize 3D results arrays if not already done
                if not results_3d:
                    print(f"Initializing output arrays for {len(slice_results)} tasks")
                    for task_name, result in slice_results.items():
                        results_3d[task_name] = np.zeros((image_data.shape[0], *result.shape), dtype=np.float32)
                
                # Store the slice results
                for task_name, result in slice_results.items():
                    results_3d[task_name][z] = result
                
                print(f"Completed slice {z+1}/{num_slices}")
            
            # Add 3D results as new layers
            for task_name, result_3d in results_3d.items():
                # Apply activation if needed
                if result_3d.shape[1] == 2:  # Binary segmentation
                    # Apply softmax and take foreground probability for each slice
                    result_3d_tensor = torch.from_numpy(result_3d)
                    result_3d = F.softmax(result_3d_tensor, dim=1).numpy()[:, 1]
                
                # Create layer name
                layer_name = f"{layer.name}_{model_name}_{task_name}"
                
                # Add as image layer
                new_layer = viewer.add_image(
                    result_3d,
                    name=layer_name,
                    colormap='magma',
                    blending='additive'
                )
                new_layers.append(new_layer)
    else:
        # Run 2D inference on single image
        results = sliding_window_inference_2d(
            model, 
            image_data, 
            patch_size,
            overlap=overlap,
            batch_size=batch_size
        )
        
        # Add results as new layers
        for task_name, result in results.items():
            # Apply activation based on task
            # For segmentation-like tasks (output_channels=2), take the second channel (foreground)
            if result.shape[0] == 2:
                # Apply softmax and take foreground probability
                result = F.softmax(torch.from_numpy(result), dim=0).numpy()[1]
            
            # Create a meaningful layer name
            layer_name = f"{layer.name}_{model_name}_{task_name}"
            
            # Add as image layer
            new_layer = viewer.add_image(
                result, 
                name=layer_name,
                colormap='magma',
                blending='additive'
            )
            new_layers.append(new_layer)
    
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
    from napari.qt.threading import thread_worker
    import warnings
    from qtpy.QtWidgets import QPushButton
    
    # Create button widget
    btn = QPushButton("Run Inference")
    
    @thread_worker
    def inference_worker():
        run_inference_callback()
        return True
        
    # Connect button to worker
    def on_click():
        worker = inference_worker()
        worker.start()
        
    btn.clicked.connect(on_click)
    
    # Add button to viewer
    viewer.window.add_dock_widget(btn, name="Inference")
    
    # Run napari
    napari.run()
