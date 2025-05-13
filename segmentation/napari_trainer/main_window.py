import napari
from magicgui import magicgui
from magicclass import magicclass, field, vfield, FieldGroup
import napari.viewer
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
        self.targets = self.dataset_config.get("targets", {})

        # model config
        self.use_timm = self.model_config.get("use_timm", False)
        self.timm_encoder_class = self.model_config.get("timm_encoder_class", None)

        # channel configuration
        self.in_channels = 1
        self.spacing = [1, 1, 1]
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
        Labels must be named either {image}_label or {image}_labels
        
        Returns
        -------
        dict
            Dictionary with targets containing volume data and metadata
        """
        # Get the current viewer and its layers
        viewer = napari.current_viewer()
        if viewer is None:
            raise ValueError("No active viewer found")
        
        # Initialize lists to store paired images and labels
        images = []
        labels = []
        
        # Get all layer names for faster lookup
        layer_names = [layer.name for layer in viewer.layers]
        
        # Find image layers and their corresponding label layers
        image_layers = [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]
        
        # Check that each image has a corresponding label layer
        for image_layer in image_layers:
            image_name = image_layer.name
            label_name = f"{image_name}_label"
            
            if self.verbose:
                print(f"Looking for label layer matching {image_name}")
                print(f"Available layers: {layer_names}")
            
            # Find matching label layer
            label_layer = None
            for layer in viewer.layers:
                if layer.name == label_name or layer.name == f"{image_name}_labels":
                    label_layer = layer
                    if self.verbose:
                        print(f"Found matching label layer: {layer.name}")
                    break
                    
            if label_layer is None:
                raise ValueError(f"Cannot locate matching label for {image_name}, available layers: {layer_names}")
            
            # Add the pair to our lists
            images.append(deepcopy(image_layer.data))
            labels.append(deepcopy(label_layer.data))
    
        if self.verbose: 
            for im_idx, img in enumerate(images):
                print(f"image: {im_idx}")
                print(f"dtype: {img.dtype}")
                print(f"shape: {img.shape}")
                
            for l_idx, lbl in enumerate(labels):
                print(f"label: {l_idx}")
                print(f"dtype: {lbl.dtype}")
                print(f"shape: {lbl.shape}")
        
        # Create the formatted dictionary compatible with dataset.py
        result = {}
        for target_name, task_info in self.targets.items():
            # Look for either 'out_channels' or 'channels' in the task info
            if 'out_channels' in task_info:
                out_channels = task_info['out_channels']
            elif 'channels' in task_info:
                out_channels = task_info['channels']
            else:
                raise ValueError(f"Target {target_name} is missing channels specification")
            
            volumes = []
            for i in range(len(images)):
                volumes.append({
                    'data': {
                        'data': images[i],
                        'label': labels[i]
                    },
                    'out_channels': out_channels,
                    'name': f"volume_{i}"
                })
            
            result[target_name] = volumes
                
        return result

    def save_config(self):
        
        combined_config = {
            "tr_setup": self.tr_info,
            "tr_config": self.tr_configs,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "inference_config": self.inference_config,
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

# Global variable to hold the config manager instance
_config_manager = None

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
    This will use images and their corresponding label layers to train the model.
    
    Example: If you have an image layer "16" and a label layer "16_labels", 
    this function will properly create the dataset, find valid patches,
    and then run training.
    """
    if _config_manager is None:
        print("Error: No configuration loaded. Please load a config file first.")
        return
    
    print("Starting training process...")
    print("Using images and labels from current viewer")
    
    # Verify targets exist in the config
    if not hasattr(_config_manager, 'targets') or not _config_manager.targets:
        print("Warning: No targets defined in configuration. Adding default target.")
        # Add a default target if none exists
        _config_manager.targets = {"ink": {"out_channels": 1, "loss_fn": "BCEDiceLoss"}}
    
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
    
    viewer.window.add_dock_widget(file_picker_widget, area='right')
    viewer.window.add_dock_widget(run_training, area='right')
    viewer.window.add_dock_widget(inference_widget, area='right')
    
    napari.run()
