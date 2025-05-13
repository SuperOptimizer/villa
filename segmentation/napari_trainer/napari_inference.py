import napari
from napari.types import LayerDataTuple
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
    num_dataloader_workers={"label": "Number of Workers", "widget_type": "SpinBox", "min": 0, "max": 32, "step": 1, "value": 4},
)
def inference_widget(
    viewer: napari.Viewer,
    model_path: str,
    layer: napari.layers.Layer,
    batch_size: int = 2,
    overlap: float = 0.25,
    num_dataloader_workers: int = 4,
) -> None:
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
    num_dataloader_workers : int
        Number of workers for parallel data loading
    """
    if not model_path or not os.path.exists(model_path):
        print("Please select a valid model checkpoint file")
        return
    
    if layer is None:
        print("Please select a layer for inference")
        return
    
    try:
        # Load the model and get config
        model_loader = ModelLoader(model_path)
        
        # Check for config files in various locations
        checkpoint_path = Path(model_path)
        
        # 1. Look in the same directory as the checkpoint
        config_path = checkpoint_path.with_name(checkpoint_path.stem + "_config.json")
        
        # 2. Look in parent directory if we're in a model-specific directory
        if not config_path.exists():
            model_name = checkpoint_path.parent.name
            parent_config = checkpoint_path.parent.parent / f"{model_name}_config.json"
            if parent_config.exists():
                config_path = parent_config
                print(f"Found config in parent directory: {config_path}")
        
        # 3. Check if config is embedded in the checkpoint itself
        checkpoint = None
        patch_size = None
        if not config_path.exists():
            try:
                # Load the checkpoint to check for embedded config
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_config' in checkpoint:
                    print("Found configuration in checkpoint")
                    # Extract patch size directly from the checkpoint
                    if 'train_patch_size' in checkpoint['model_config']:
                        patch_size = checkpoint['model_config']['train_patch_size']
                        print(f"Using patch size from checkpoint: {patch_size}")
                        # For 3D patch sizes, use the last 2 dimensions for 2D inference
                        if len(patch_size) == 3:
                            patch_size = tuple(patch_size[1:])  # Use y, x dimensions
            except Exception as e:
                print(f"Error loading checkpoint to check for config: {e}")
                # We'll continue with config file approach
        
        # Process config file if it exists
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Extract patch size from config (handles both model.train_patch_size and inference.patch_size)
                if 'model' in config_data and 'train_patch_size' in config_data['model']:
                    patch_size = config_data['model']['train_patch_size']
                elif 'train_patch_size' in config_data:
                    patch_size = config_data['train_patch_size']
                elif 'inference' in config_data and 'patch_size' in config_data['inference']:
                    patch_size = config_data['inference']['patch_size']
                else:
                    print("Patch size not found in config, using default (256, 256)")
                    patch_size = (256, 256)
            except Exception as e:
                print(f"Error reading config file: {e}")
                print("Using default patch size (256, 256)")
                patch_size = (256, 256)
        else:
            print(f"No configuration file found")
            print("Using default patch size (256, 256)")
            patch_size = (256, 256)
            
            # For 3D patch sizes, use the last 2 dimensions for 2D inference
            if len(patch_size) == 3:
                patch_size = tuple(patch_size[1:])  # Use y, x dimensions
            
            print(f"Using patch size: {patch_size}")
        
        # Run inference using thread_worker to avoid UI blocking
        from napari.qt.threading import thread_worker
        
        @thread_worker
        def inference_thread():
            try:
                run_inference(
                    viewer,
                    layer,
                    model_path,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    overlap=overlap,
                    num_dataloader_workers=num_dataloader_workers
                )
                return True
            except Exception as e:
                import traceback
                traceback.print_exc()
                return str(e)
        
        # Connect to thread events
        worker = inference_thread()
        
        def on_done(result):
            if isinstance(result, str):  # Error message
                print(f"Inference failed: {result}")
        
        def on_error(error):
            print(f"Inference error: {str(error)}")
        
        worker.finished.connect(on_done)
        worker.errored.connect(on_error)
        worker.start()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during inference: {str(e)}")

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
