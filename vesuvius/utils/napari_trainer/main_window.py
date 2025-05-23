import napari
from magicgui import magicgui, widgets
import napari.viewer
import scipy.ndimage

from .inference_widget import inference_widget
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
from utils.utils import determine_dimensionality


Image.MAX_IMAGE_PIXELS = None

class ConfigManager:
    def __init__(self, verbose):
        self._config_path = None
        self.data = None
        self.verbose = verbose
        self.selected_loss_function = "BCEWithLogitsLoss"

    def load_config(self, config_path):
        config_path = Path(config_path)
        self._config_path = config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.tr_info = config.get("tr_setup", {})
        self.tr_configs = config.get("tr_config", {})
        self.model_config = config.get("model_config", {}) 
        self.dataset_config = config.get("dataset_config", {})
        self.inference_config = config.get("inference_config", {})
        
        self._init_attributes()
        
        return config
    
    def _init_attributes(self):
        self.train_patch_size = tuple(self.tr_configs.get("patch_size", [192, 192, 192]))
        self.in_channels = 2

        self.model_name = self.tr_info.get("model_name", "Model")
        self.autoconfigure = bool(self.tr_info.get("autoconfigure", True))
        self.tr_val_split = float(self.tr_info.get("tr_val_split", 0.95))
        self.dilate_label = int(self.tr_info.get("dilate_label", 0))
        self.compute_loss_on_label = bool(self.tr_info.get("compute_loss_on_label", True))

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
        
        # Use the centralized dimensionality function to set appropriate operations
        dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
        self.model_config["conv_op"] = dim_props["conv_op"]
        self.model_config["pool_op"] = dim_props["pool_op"]
        self.model_config["norm_op"] = dim_props["norm_op"]
        self.model_config["dropout_op"] = dim_props["dropout_op"]
        self.spacing = dim_props["spacing"]
        self.op_dims = dim_props["op_dims"]

        # channel configuration
        self.in_channels = 1  # Changed from 2 to 1 to match actual input data
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
        viewer = napari.current_viewer()
        if viewer is None:
            raise ValueError("No active viewer found")
        
        all_layers = list(viewer.layers)
        layer_names = [layer.name for layer in all_layers]
        image_layers = [layer for layer in all_layers if isinstance(layer, napari.layers.Image)]
        
        if not image_layers:
            raise ValueError("No image layers found in the viewer")
        
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
                if target_name not in self.targets:

                    self.targets[target_name] = {
                        "out_channels": 1, 
                        "loss_fn": self.selected_loss_function,
                        "activation": "sigmoid" 
                    }
                    print(f"DEBUG: Creating new target '{target_name}' with loss function '{self.selected_loss_function}'")
                    result[target_name] = []
                
                # Prepare the data dictionary
                data_dict = {
                    'data': deepcopy(image_layer.data),
                    'label': deepcopy(label_layer.data)
                }
                
                if mask_layer is not None:
                    data_dict['mask'] = deepcopy(mask_layer.data)
                    print(f"Including mask for target {target_name}")
                
                result[target_name].append({
                    'data': data_dict,
                    'out_channels': self.targets[target_name]["out_channels"],
                    'name': f"{image_name}_{target_name}"
                })
                
                if self.verbose:
                    print(f"Added target {target_name} with data from {image_name} and {label_layer.name}")
        
        if not self.targets:
            raise ValueError("No valid image-label pairs found. Label layers should be named as image_name_suffix.")
        
        self.out_channels = tuple(task_info["out_channels"] for task_info in self.targets.values())
        
        if result:
            first_target = next(iter(result.values()))[0]
            img_data = first_target['data']['data']
            data_is_2d = len(img_data.shape) == 2
            config_is_2d = len(self.train_patch_size) == 2
            
            if data_is_2d != config_is_2d:
                # Reconfigure dimensionality based on actual data
                if data_is_2d:
                    # Data is 2D but config is 3D
                    if self.verbose:
                        print(f"Data is 2D but config is for 3D. Reconfiguring for 2D operations.")
                    
                    # Keep existing patch_size dimensions but adapt to 2D
                    if len(self.train_patch_size) > 2:
                        self.train_patch_size = self.train_patch_size[-2:]
                        self.tr_configs["patch_size"] = list(self.train_patch_size)
                    
                    # Update all dimension-dependent configurations
                    dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
                    self.model_config["conv_op"] = dim_props["conv_op"]
                    self.model_config["pool_op"] = dim_props["pool_op"]
                    self.model_config["norm_op"] = dim_props["norm_op"]
                    self.model_config["dropout_op"] = dim_props["dropout_op"]
                    self.spacing = dim_props["spacing"]
                    self.op_dims = dim_props["op_dims"]
                else:
                    # Data is 3D but config is 2D
                    if self.verbose:
                        print(f"Data is 3D but config is for 2D. Reconfiguring for 3D operations.")
                    
                    # Update all dimension-dependent configurations
                    dim_props = determine_dimensionality([1, 1, 1], self.verbose)  # Use dummy 3D patch
                    self.model_config["conv_op"] = dim_props["conv_op"]
                    self.model_config["pool_op"] = dim_props["pool_op"]
                    self.model_config["norm_op"] = dim_props["norm_op"]
                    self.model_config["dropout_op"] = dim_props["dropout_op"]
                    self.spacing = dim_props["spacing"]
                    self.op_dims = dim_props["op_dims"]
        
        if self.verbose:
            print(f"Final targets dictionary: {self.targets}")
            print(f"Final output channels: {self.out_channels}")
                
        return result

    def save_config(self):
        tr_setup = deepcopy(self.tr_info)
        tr_config = deepcopy(self.tr_configs)
        model_config = deepcopy(self.model_config)
        dataset_config = deepcopy(self.dataset_config)
        inference_config = deepcopy(self.inference_config)
        
        if hasattr(self, 'targets') and self.targets:
            dataset_config["targets"] = deepcopy(self.targets)
            
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

        # Create a specific directory for this model's checkpoints
        model_ckpt_dir = Path(self.ckpt_out_base) / self.model_name
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create a config filename matching the model name
        config_filename = f"{self.model_name}_config.yaml"

        # Full path to the new file in the checkpoint directory
        config_path = model_ckpt_dir / config_filename

        # Write out the YAML
        with config_path.open("w") as f:
            yaml.safe_dump(combined_config, f, sort_keys=False)

        print(f"Configuration saved to: {config_path}")

    def update_config_from_widget(self, widget=None):
        if widget is not None:
            patch_size_z = widget.patch_size_z.value
            patch_size_x = widget.patch_size_x.value
            patch_size_y = widget.patch_size_y.value
            min_labeled_percentage = widget.min_labeled_percentage.value
            max_epochs = widget.max_epochs.value
            loss_function = widget.loss_function.value
            new_patch_size = [patch_size_z, patch_size_x, patch_size_y]
            self.train_patch_size = tuple(new_patch_size)
            self.tr_configs["patch_size"] = new_patch_size
            self.min_labeled_ratio = min_labeled_percentage / 100.0
            self.dataset_config["min_labeled_ratio"] = self.min_labeled_ratio
            
            # Use centralized dimensionality function to set appropriate operations
            dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
            self.model_config["conv_op"] = dim_props["conv_op"]
            self.model_config["pool_op"] = dim_props["pool_op"]
            self.model_config["norm_op"] = dim_props["norm_op"]
            self.model_config["dropout_op"] = dim_props["dropout_op"]
            self.spacing = dim_props["spacing"]
            self.op_dims = dim_props["op_dims"]
            
            # Set max_steps_per_epoch and max_val_steps_per_epoch to None so that train.py will use the dataset length
            self.max_steps_per_epoch = None
            self.max_val_steps_per_epoch = None
            self.max_epoch = max_epochs
            self.tr_configs["max_epoch"] = max_epochs
            self.selected_loss_function = loss_function
            
            if self.verbose:
                print(f"Updated training parameters:")
                print(f"  - Patch size: [{patch_size_z}, {patch_size_x}, {patch_size_y}]")
                print(f"  - Min labeled ratio: {self.min_labeled_ratio:.2f} ({min_labeled_percentage}%)")
                print(f"  - Max epochs: {max_epochs}")
                print(f"  - Loss function: {loss_function}")
        
        if hasattr(self, 'selected_loss_function') and hasattr(self, 'targets') and self.targets:
            print(f"DEBUG: Applying loss function '{self.selected_loss_function}' to all targets")
            for target_name in self.targets:
                self.targets[target_name]["loss_fn"] = self.selected_loss_function
                print(f"DEBUG: Set target '{target_name}' loss_fn to '{self.selected_loss_function}'")
            
            if self.verbose:
                print(f"Applied loss function '{self.selected_loss_function}' to all targets during config update")
    
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

_config_manager = None
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
    
    _config_manager.update_config_from_widget(run_training)
    
    try:
        _config_manager.get_images()
    except Exception as e:
        print(f"Error detecting images and labels: {e}")
        raise
    
    if not hasattr(_config_manager, 'targets') or not _config_manager.targets:
        raise ValueError("No targets defined. Please make sure you have label layers named {image_name}_{target_name} in the viewer.")
    
    from models.train import BaseTrainer
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
