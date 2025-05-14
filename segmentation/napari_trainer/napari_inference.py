import napari
from magicgui import magicgui
from pathlib import Path
import os
import torch
from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np

# Import the inference functionality
from inference import run_inference, ModelLoader

# Function to get available layers for inference
def get_data_layer_choices(viewer):
    """Get available image layers as choices for inference"""
    return [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]

@magicgui(
    call_button="Run Inference",
    layout="vertical",
    model_path={"label": "Model Checkpoint", "widget_type": "FileEdit", "filter": "PyTorch Checkpoint (*.pth)"},
    layer={"label": "Layer for inference", "choices": get_data_layer_choices},
    batch_size={"label": "Batch Size", "widget_type": "SpinBox", "min": 1, "max": 64, "step": 1, "value": 2},
    overlap={"label": "Overlap", "widget_type": "FloatSpinBox", "min": 0.0, "max": 0.75, "step": 0.05, "value": 0.25},
)
def inference_widget(
    viewer: napari.Viewer,
    model_path: str,
    layer: napari.layers.Layer,
    batch_size: int = 2,
    overlap: float = 0.25,
) -> Optional[napari.layers.Image]:
    """
    Run inference on a selected layer using a trained model.
    
    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance
    model_path : str
        Path to the model checkpoint (.pth file)
    layer : napari.layers.Layer
        The layer to run inference on
    batch_size : int
        Number of patches to process simultaneously
    overlap : float
        Overlap between patches (0-1)
        
    Returns
    -------
    Optional[napari.layers.Image]
        The new image layer to be added to the viewer, or None if inference fails
    """
    if not model_path or not os.path.exists(model_path):
        print("Please select a valid model checkpoint file")
        return None
    
    if layer is None:
        print("Please select a layer for inference")
        return None
    
    # Get data dimensionality
    image_data = layer.data
    is_3d = image_data.ndim == 3 and image_data.shape[0] > 1
    data_dims = 3 if is_3d else 2
    print(f"Input data is {data_dims}D with shape {image_data.shape}")
    
    # Try to access the global config manager
    config_manager = None
    try:
        from main_window import _config_manager
        if _config_manager is not None:
            config_manager = _config_manager
    except (ImportError, AttributeError):
        pass
    
    # Load the model
    model_loader = ModelLoader(model_path, config_manager=config_manager)
    model, model_config = model_loader.load()
    
    # Get patch size from model config
    patch_size = None
    if 'patch_size' in model_config:
        patch_size = tuple(model_config['patch_size'])
        print(f"Using patch_size from model config: {patch_size}")
    elif 'train_patch_size' in model_config:
        patch_size = tuple(model_config['train_patch_size'])
        print(f"Using train_patch_size from model config: {patch_size}")
    else:
        raise ValueError("No patch_size or train_patch_size found in model config. Cannot proceed with inference.")
    
    # SIMPLE DIMENSION CHECK: Model dims must match data dims
    model_dims = len(patch_size)
    if model_dims != data_dims:
        raise ValueError(f"Model dimensionality ({model_dims}D) must match data dimensionality ({data_dims}D)")
    
    # Run inference directly (no threading)
    try:
        # Use sliding_window_inference directly to get results without adding to viewer
        from inference import sliding_window_inference
        
        # Load the model directly here
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # Get results from inference
        results = sliding_window_inference(
            model, 
            layer.data, 
            patch_size, 
            overlap=overlap,
            batch_size=batch_size
        )
        
        # Get model name from checkpoint path
        model_name = Path(model_path).stem
        
        # Create and return the first result as a napari.layers.Image
        # (We assume the model has at least one output)
        for task_name, result in results.items():
            # Apply activation if needed for binary segmentation tasks
            if result.shape[0] == 2:  # Binary segmentation
                # Apply softmax and take foreground probability
                result_tensor = torch.from_numpy(result)
                result = torch.nn.functional.softmax(result_tensor, dim=0).numpy()[1]
            
            # Create layer name
            layer_name = f"{layer.name}_{model_name}_{task_name}"
            
            # Return the result as a new image layer
            return napari.layers.Image(
                result,
                name=layer_name,
                colormap='magma',
                blending='additive'
            )
        
        # If no results were processed, return None
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Inference error: {str(e)}")
        return None

# This function will be automatically detected by napari
def napari_experimental_provide_dock_widget():
    # Return the magicgui widget
    return inference_widget

if __name__ == "__main__":
    # Create a new viewer
    viewer = napari.Viewer()
    
    # Create sample data if needed
    if len(viewer.layers) == 0:
        sample = np.random.rand(512, 512)
        viewer.add_image(sample, name="Sample")
    
    # Create and add our widget
    viewer.window.add_dock_widget(
        inference_widget(), 
        name="Model Inference"
    )
    
    # Run napari
    napari.run()
