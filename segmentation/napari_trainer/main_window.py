import napari
from magicgui import magicgui, widgets
from magicclass import magicclass, field, vfield, FieldGroup
import napari.viewer
import scipy.ndimage

from napari_inference import inference_widget
import cv2
from PIL import Image
import os
import threading
import numpy as np
from pathlib import Path
from collections.abc import Sequence
from typing import Optional, List, Tuple, Dict, Any, Union, Set
from copy import deepcopy
# Import BaseTrainer inside functions instead of at module level to avoid circular imports
import napari.layers

Image.MAX_IMAGE_PIXELS = None
import json
import yaml
from pathlib import Path
import torch.nn as nn

class ConfigManager:
    def __init__(self, verbose):
        # Initialize empty
        self._config_path = None
        self.data = None
        self.verbose = verbose

    def load_config(self, config_path):
        config_path = Path(config_path)
        self._config_path = config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Store the main config sections
        self.tr_info = config.get("tr_setup", {})
        self.tr_configs = config.get("tr_config", {})
        self.model_config = config.get("model_config", {}) 
        self.dataset_config = config.get("dataset_config", {})
        self.inference_config = config.get("inference_config", {})
        
        # Initialize all the derived attributes
        self._init_attributes()
        
        return config
    
    def _init_attributes(self):
        
        # Training setup
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        self.in_channels = 1

        self.model_name = self.tr_info.get("model_name", "Model")
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.95))
        self.dilate_label = bool(self.tr_info.get("dilate_label", False))

        ckpt_out_base = self.tr_info.get("ckpt_out_base", "./checkpoints/")
        self.ckpt_out_base = Path(ckpt_out_base)
        if not self.ckpt_out_base.exists():
            self.ckpt_out_base.mkdir(parents=True)
        ckpt_path = self.tr_info.get("checkpoint_path", None)
        self.checkpoint_path = Path(ckpt_path) if ckpt_path else None
        self.load_weights_only = bool(self.tr_info.get("load_weights_only", False))

        # Training config
        self.optimizer = self.tr_configs.get("optimizer", "AdamW")
        self.initial_lr = float(self.tr_configs.get("initial_lr", 1e-3))
        self.weight_decay = float(self.tr_configs.get("weight_decay", 0))
        self.train_batch_size = int(self.tr_configs.get("batch_size", 2))
        self.gradient_accumulation = int(self.tr_configs.get("gradient_accumulation", 1))
        self.max_steps_per_epoch = int(self.tr_configs.get("max_steps_per_epoch", 500))
        self.max_val_steps_per_epoch = int(self.tr_configs.get("max_val_steps_per_epoch", 25))
        self.train_num_dataloader_workers = int(self.tr_configs.get("num_dataloader_workers", 4))
        self.max_epoch = int(self.tr_configs.get("max_epoch", 1000))

        # Dataset config
        self.min_labeled_ratio = float(self.dataset_config.get("min_labeled_ratio", 0.10))
        self.min_bbox_percent = float(self.dataset_config.get("min_bbox_percent", 0.95))
        
        # Only initialize targets from config if not already created dynamically
        if not hasattr(self, 'targets') or not self.targets:
            self.targets = self.dataset_config.get("targets", {})
            if self.verbose and self.targets:
                print(f"Loaded targets from config: {self.targets}")

        # model config
        self.use_timm = self.model_config.get("use_timm", False)
        self.timm_encoder_class = self.model_config.get("timm_encoder_class", None)
        
        # Auto-detect dimensionality from patch_size and set appropriate convolution
        if len(self.train_patch_size) == 2:
            # Set 2D convolutions for 2D data
            self.model_config["conv_op"] = "nn.Conv2d"
            # Make sure spacing dimension matches patch size dimension
            self.spacing = [1] * len(self.train_patch_size)
            if self.verbose:
                print(f"Detected 2D patch size {self.train_patch_size}, setting conv_op to nn.Conv2d")
        else:
            # Default to 3D convolutions for 3D data
            self.model_config["conv_op"] = "nn.Conv3d"
            # Make sure spacing dimension matches patch size dimension
            self.spacing = [1] * len(self.train_patch_size)
            if self.verbose:
                print(f"Detected 3D patch size {self.train_patch_size}, setting conv_op to nn.Conv3d")

        # channel configuration
        self.in_channels = 1
        self.out_channels = ()
        for target_name, task_info in self.targets.items():
            # Look for either 'out_channels' or 'channels' in the task info
            if 'out_channels' in task_info:
                channels = task_info['out_channels']
            elif 'channels' in task_info:
                channels = task_info['channels']
            else:
                raise ValueError(f"Target {target_name} is missing channels specification (either 'channels' or 'out_channels')")
            self.out_channels += (channels,)

        # Inference config
        self.infer_checkpoint_path = self.inference_config.get("checkpoint_path", None)
        self.infer_patch_size = tuple(self.inference_config.get("patch_size", self.train_patch_size))
        self.infer_batch_size = int(self.inference_config.get("batch_size", self.train_batch_size))
        self.infer_output_targets = self.inference_config.get("output_targets", ['all'])
        self.infer_overlap = float(self.inference_config.get("overlap", 0.25))
        self.load_strict = bool(getattr(self.inference_config, "load_strict", True))
        self.infer_num_dataloader_workers = int(
            getattr(self.inference_config, "num_dataloader_workers", self.train_num_dataloader_workers))

    def get_images(self):
        """
        Returns image/label pairs as np arrays formatted for the dataset.
        Labels are expected to be named {image_name}_{suffix} where suffix is a unique identifier.
        For each image layer, multiple label layers can be associated.
        
        Returns
        -------
        dict
            Dictionary with targets containing volume data and metadata
        """
        # Get the current viewer and its layers
        viewer = napari.current_viewer()
        if viewer is None:
            raise ValueError("No active viewer found")
        
        # Get all layers for faster lookup
        all_layers = list(viewer.layers)
        layer_names = [layer.name for layer in all_layers]
        
        # Find image layers
        image_layers = [layer for layer in all_layers if isinstance(layer, napari.layers.Image)]
        
        if not image_layers:
            raise ValueError("No image layers found in the viewer")
        
        # Build targets dictionary based on naming patterns
        self.targets = {}
        result = {}
        
        for image_layer in image_layers:
            image_name = image_layer.name
            
            if self.verbose:
                print(f"Processing image layer: {image_name}")
                print(f"Available layers: {layer_names}")
            
            # Find all label layers that correspond to this image layer
            # Pattern: image_name_suffix
            matching_label_layers = []
            
            for layer in all_layers:
                if (isinstance(layer, napari.layers.Labels) and 
                    layer.name.startswith(f"{image_name}_")):
                    suffix = layer.name[len(image_name) + 1:]  # Extract suffix after image_name_
                    if self.verbose:
                        print(f"Found matching label layer: {layer.name} with suffix: {suffix}")
                    matching_label_layers.append((suffix, layer))
            
            if not matching_label_layers:
                if self.verbose:
                    print(f"No matching label layers found for image: {image_name}")
                continue
            
            # For each matching label layer, create a target
            for target_suffix, label_layer in matching_label_layers:
                # Use the suffix as the target name
                target_name = target_suffix
                
                # Create the target if it doesn't exist
                if target_name not in self.targets:
                    self.targets[target_name] = {"out_channels": 1, "loss_fn": "BCEWithLogitsLoss"}
                    result[target_name] = []
                
                # Append the data for this target
                result[target_name].append({
                    'data': {
                        'data': deepcopy(image_layer.data),
                        'label': deepcopy(label_layer.data)
                    },
                    'out_channels': self.targets[target_name]["out_channels"],
                    'name': f"{image_name}_{target_name}"
                })
                
                if self.verbose:
                    print(f"Added target {target_name} with data from {image_name} and {label_layer.name}")
        
        if not self.targets:
            raise ValueError("No valid image-label pairs found. Label layers should be named as image_name_suffix.")
        
        # Update out_channels in ConfigManager
        self.out_channels = tuple(task_info["out_channels"] for task_info in self.targets.values())
        
        # Check the dimensionality of the data to ensure it matches with the current configuration
        if result:
            # Get the first target's first item to check dimensionality
            first_target = next(iter(result.values()))[0]
            img_data = first_target['data']['data']
            data_is_2d = len(img_data.shape) == 2
            config_is_2d = len(self.train_patch_size) == 2
            
            # If there's a mismatch between data and configuration, update the configuration
            if data_is_2d != config_is_2d:
                if data_is_2d:
                    # Data is 2D but config is 3D
                    if self.verbose:
                        print(f"Data is 2D but config is for 3D. Updating conv_op to nn.Conv2d")
                    # Set 2D convolutions for 2D data
                    self.model_config["conv_op"] = "nn.Conv2d"
                    # Keep existing patch_size dimensions but adapt to 2D
                    if len(self.train_patch_size) > 2:
                        self.train_patch_size = self.train_patch_size[-2:]
                        self.tr_configs["patch_size"] = list(self.train_patch_size)
                        # Update spacing to match patch size dimensions
                        self.spacing = [1] * len(self.train_patch_size)
                        if self.verbose:
                            print(f"Adjusted patch_size to {self.train_patch_size} for 2D data")
                            print(f"Updated spacing to {self.spacing}")
                else:
                    # Data is 3D but config is 2D
                    if self.verbose:
                        print(f"Data is 3D but config is for 2D. Updating conv_op to nn.Conv3d")
                    # Set 3D convolutions for 3D data
                    self.model_config["conv_op"] = "nn.Conv3d"
                    # Update spacing to match 3D dimensions
                    self.spacing = [1, 1, 1]
                    if self.verbose:
                        print(f"Updated spacing to {self.spacing}")
        
        if self.verbose:
            print(f"Final targets dictionary: {self.targets}")
            print(f"Final output channels: {self.out_channels}")
                
        return result

    def save_config(self):
        # Make a deep copy of existing configs to avoid modifying the originals
        tr_setup = deepcopy(self.tr_info)
        tr_config = deepcopy(self.tr_configs)
        model_config = deepcopy(self.model_config)
        dataset_config = deepcopy(self.dataset_config)
        inference_config = deepcopy(self.inference_config)
        
        # Ensure the targets from our dynamic layer detection are saved
        if hasattr(self, 'targets') and self.targets:
            dataset_config["targets"] = deepcopy(self.targets)
            
            # Also update model_config with the targets for inference
            model_config["targets"] = deepcopy(self.targets)
            
            if self.verbose:
                print(f"Saving targets to config: {self.targets}")
        
        combined_config = {
            "tr_setup": tr_setup,
            "tr_config": tr_config,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "inference_config": inference_config,
        }

        original_stem = self._config_path.stem  # e.g. "my_config"
        original_ext = self._config_path.suffix  # e.g. ".yaml"
        original_parent = self._config_path.parent

        # Create the new filename with "_final" inserted
        final_filename = f"{original_stem}_final{original_ext}"

        # Full path to the new file
        final_path = original_parent / final_filename

        # Write out the YAML
        with final_path.open("w") as f:
            yaml.safe_dump(combined_config, f, sort_keys=False)

        print(f"Configuration saved to: {final_path}")

    def _print_summary(self):
        print("____________________________________________")
        print("Training Setup (tr_info):")
        for k, v in self.tr_info.items():
            print(f"  {k}: {v}")

        print("\nTraining Config (tr_configs):")
        for k, v in self.tr_configs.items():
            print(f"  {k}: {v}")

        print("\nDataset Config (dataset_config):")
        for k, v in self.dataset_config.items():
            print(f"  {k}: {v}")

        print("\nInference Config (inference_config):")
        for k, v in self.inference_config.items():
            print(f"  {k}: {v}")
        print("____________________________________________")

# Global variables to hold important references
_config_manager = None
_loss_fn_widget = None

# Function to pick config file, with callback to load it into ConfigManager
@magicgui(filenames={"label": "select config file", "filter": "*.yaml"},
          auto_call=True)
def filespicker(filenames: Sequence[Path]) -> Sequence[Path]:
    print("selected config : ", filenames)
    if filenames and _config_manager is not None:
        # Load the first selected file into the config manager
        _config_manager.load_config(filenames[0])
        print(f"Config loaded from {filenames[0]}")
    return filenames

@magicgui(call_button='run training')
def run_training():
    """
    Start training with the currently loaded configuration and visible layers.
    
    Label Naming Convention:
    For each image layer, you can have multiple label layers that will be treated as separate targets.
    Labels must be named as: {image_name}_{target_name}
    
    Examples:
    - If you have an image layer "32" and label layers "32_1" and "32_2", 
      the system will create targets named "1" and "2".
    - If you have multiple image layers like "image1" and "image2" with corresponding
      label layers "image1_ink" and "image2_ink", the system will create a target named "ink".
    
    All labels with the same suffix (target name) will be grouped together for training.
    """
    if _config_manager is None:
        print("Error: No configuration loaded. Please load a config file first.")
        return
    
    print("Starting training process...")
    print("Using images and labels from current viewer")
    
    # First, get images and labels from the current viewer
    try:
        # This will populate the targets
        _config_manager.get_images()
    except Exception as e:
        print(f"Error detecting images and labels: {e}")
        raise
    
    # Verify targets exist in the config
    if not hasattr(_config_manager, 'targets') or not _config_manager.targets:
        raise ValueError("No targets defined. Please make sure you have label layers named {image_name}_{target_name} in the viewer.")
    
    # Import BaseTrainer here to avoid circular imports
    from train import BaseTrainer
    
    # Initialize trainer with our config manager
    trainer = BaseTrainer(mgr=_config_manager, verbose=True)
    
    # Start training
    print("Starting training...")
    trainer.train()
    



if __name__ == "__main__":
    viewer = napari.Viewer()

    # Create config manager and store in global variable
    _config_manager = ConfigManager(verbose=True)

    # Create the file picker widget
    file_picker_widget = filespicker

    # Add widgets to the viewer
    viewer.window.add_dock_widget(file_picker_widget, area='right', name="Config File")


    # Add training and inference widgets
    viewer.window.add_dock_widget(run_training, area='right', name="Training")
    viewer.window.add_dock_widget(inference_widget, area='right', name="Inference")

    napari.run()
