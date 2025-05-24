import napari
from magicgui import magicgui, widgets
import napari.viewer
import scipy.ndimage

from .inference_widget import inference_widget
from models.config_manager import ConfigManager
from PIL import Image
import numpy as np
from pathlib import Path
from collections.abc import Sequence
from copy import deepcopy
import napari.layers
import json
import yaml
from pathlib import Path
import torch.nn as nn


Image.MAX_IMAGE_PIXELS = None

_config_manager = None

def get_images_from_napari(verbose=False):
    """
    Extract image/label pairs from napari viewer for the dataset.
    Labels are expected to be named {image_name}_{suffix} where suffix is a unique identifier.
    For each image layer, multiple label layers can be associated.
    
    Returns
    -------
    tuple
        (targets_dict, data_dict) where:
        - targets_dict: Dictionary with target configurations 
        - data_dict: Dictionary with target names as keys and list of volume data as values
    """
    viewer = napari.current_viewer()
    if viewer is None:
        raise ValueError("No active viewer found")
    
    all_layers = list(viewer.layers)
    layer_names = [layer.name for layer in all_layers]
    image_layers = [layer for layer in all_layers if isinstance(layer, napari.layers.Image)]
    
    if not image_layers:
        raise ValueError("No image layers found in the viewer")
    
    targets_dict = {}
    data_dict = {}
    
    for image_layer in image_layers:
        image_name = image_layer.name
        
        if verbose:
            print(f"Processing image layer: {image_name}")
            print(f"Available layers: {layer_names}")
        
        # Find all label layers that correspond to this image layer
        # Pattern: image_name_suffix
        matching_label_layers = []
        
        for layer in all_layers:
            if (isinstance(layer, napari.layers.Labels) and 
                layer.name.startswith(f"{image_name}_")):
                suffix = layer.name[len(image_name) + 1:]  # Extract suffix after image_name_
                if verbose:
                    print(f"Found matching label layer: {layer.name} with suffix: {suffix}")
                matching_label_layers.append((suffix, layer))
        
        if not matching_label_layers:
            if verbose:
                print(f"No matching label layers found for image: {image_name}")
            continue
        
        mask_layer = None
        mask_layer_name = f"{image_name}_mask"
        for layer in all_layers:
            if isinstance(layer, napari.layers.Labels) and layer.name == mask_layer_name:
                mask_layer = layer
                print(f"Found mask layer for image {image_name}: {mask_layer_name}")
                break
        
        # For each matching label layer, create a target
        for target_suffix, label_layer in matching_label_layers:
            if target_suffix == "mask":
                continue

            target_name = target_suffix
            if target_name not in targets_dict:
                targets_dict[target_name] = {
                    "out_channels": 1, 
                    "activation": "sigmoid" 
                }
                print(f"DEBUG: Creating new target '{target_name}'")
                data_dict[target_name] = []
            
            # Prepare the data dictionary
            data_entry = {
                'data': deepcopy(image_layer.data),
                'label': deepcopy(label_layer.data)
            }
            
            if mask_layer is not None:
                data_entry['mask'] = deepcopy(mask_layer.data)
                print(f"Including mask for target {target_name}")
            
            data_dict[target_name].append({
                'data': data_entry,
                'out_channels': targets_dict[target_name]["out_channels"],
                'name': f"{image_name}_{target_name}"
            })
            
            if verbose:
                print(f"Added target {target_name} with data from {image_name} and {label_layer.name}")
    
    if not targets_dict:
        raise ValueError("No valid image-label pairs found. Label layers should be named as image_name_suffix.")
    
    if verbose:
        print(f"Final targets dictionary: {targets_dict}")
        print(f"Final data dictionary keys: {list(data_dict.keys())}")
            
    return targets_dict, data_dict

@magicgui(filenames={"label": "select config file", "filter": "*.yaml"},
          auto_call=True)
def filespicker(filenames: Sequence[Path] = str(Path(__file__).parent.parent.parent / 'models' / 'configs' / 'default_config.yaml')) -> Sequence[Path]:
    print("selected config : ", filenames)
    if filenames and _config_manager is not None:
        # Load the first selected file into the config manager
        _config_manager.load_config(filenames[0])
        print(f"Config loaded from {filenames[0]}")
    return filenames

@magicgui(
    call_button='run training',
    patch_size_z={'widget_type': 'SpinBox', 'label': 'Patch Size Z', 'min': 0, 'max': 4096, 'value': 0},
    patch_size_x={'widget_type': 'SpinBox', 'label': 'Patch Size X', 'min': 0, 'max': 4096, 'value': 128},
    patch_size_y={'widget_type': 'SpinBox', 'label': 'Patch Size Y', 'min': 0, 'max': 4096, 'value': 128},
    min_labeled_percentage={'widget_type': 'SpinBox', 'label': 'Min Labeled Percentage', 'min': 0.0, 'max': 100.0, 'step': 1.0, 'value': 10.0},
    max_epochs={'widget_type': 'SpinBox', 'label': 'Max Epochs', 'min': 1, 'max': 1000, 'value': 5},
    loss_function={'widget_type': 'ComboBox', 'choices': ["BCELoss", "BCEWithLogitsLoss", "MSELoss", 
                                                         "L1Loss", "SoftDiceLoss"], 'value': "SoftDiceLoss"}
)
def run_training(patch_size_z: int = 128, patch_size_x: int = 128, patch_size_y: int = 128,
                min_labeled_percentage: float = 10.0,
                max_epochs: int = 5,
                loss_function: str = "SoftDiceLoss"):
    if _config_manager is None:
        print("Error: No configuration loaded. Please load a config file first.")
        return
    
    print("Starting training process...")
    print("Using images and labels from current viewer")
    
    # Update configuration with new parameters using the generic interface
    new_patch_size = [patch_size_z, patch_size_x, patch_size_y]
    min_labeled_ratio = min_labeled_percentage / 100.0
    
    _config_manager.update_config(
        patch_size=new_patch_size,
        min_labeled_ratio=min_labeled_ratio,
        max_epochs=max_epochs,
        loss_function=loss_function
    )
    
    try:
        # Extract napari data using our napari-specific function
        targets_dict, data_dict = get_images_from_napari(verbose=_config_manager.verbose)
        
        # Set targets and data using the generic ConfigManager interface
        _config_manager.set_targets_and_data(targets_dict, data_dict)
        
    except Exception as e:
        print(f"Error detecting images and labels: {e}")
        raise
    
    if not hasattr(_config_manager, 'targets') or not _config_manager.targets:
        raise ValueError("No targets defined. Please make sure you have label layers named {image_name}_{target_name} in the viewer.")
    
    from models.run.train import BaseTrainer
    _config_manager.data_format = 'napari'
    
    trainer = BaseTrainer(mgr=_config_manager, verbose=True)
    print("Starting training...")
    trainer.train()

def main():
    viewer = napari.Viewer()
    global _config_manager
    _config_manager = ConfigManager(verbose=True)
    # Use an absolute path based on the location of this script
    default_config_path = Path(__file__).parent.parent.parent / 'models' / 'configs' / 'default_config.yaml'
    _config_manager.load_config(default_config_path)
    print(f"Default config loaded from {default_config_path}")

    file_picker_widget = filespicker
    viewer.window.add_dock_widget(file_picker_widget, area='right', name="config file")
    viewer.window.add_dock_widget(run_training, area='right', name="training")
    viewer.window.add_dock_widget(inference_widget, area='right', name="inference")

    napari.run()

if __name__ == "__main__":
    main()
