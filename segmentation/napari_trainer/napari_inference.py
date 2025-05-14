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
            # No defaults - require patch size in config
            raise ValueError("No patch_size or train_patch_size found in model config. Cannot proceed with inference.")
        
        # SIMPLE DIMENSION CHECK: Model dims must match data dims
        model_dims = len(patch_size)
        if model_dims != data_dims:
            raise ValueError(f"Model dimensionality ({model_dims}D) must match data dimensionality ({data_dims}D)")
        
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
