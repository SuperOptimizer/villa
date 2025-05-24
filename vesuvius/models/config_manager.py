from PIL import Image
import numpy as np
from pathlib import Path
from copy import deepcopy
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

    def set_targets_and_data(self, targets_dict, data_dict):
        """
        Generic method to set targets and data from any source (napari, TIF, zarr, etc.)
        
        Parameters
        ----------
        targets_dict : dict
            Dictionary with target names as keys and target configuration as values
            Example: {"ink": {"out_channels": 1, "loss_fn": "BCEWithLogitsLoss", "activation": "sigmoid"}}
        data_dict : dict
            Dictionary with target names as keys and list of volume data as values
            Example: {"ink": [{"data": {...}, "out_channels": 1, "name": "image1_ink"}]}
        """
        self.targets = deepcopy(targets_dict)
        
        # Apply current loss function to all targets if not already set
        for target_name in self.targets:
            if "loss_fn" not in self.targets[target_name]:
                self.targets[target_name]["loss_fn"] = self.selected_loss_function
        
        # Update out_channels tuple
        self.out_channels = tuple(task_info["out_channels"] for task_info in self.targets.values())
        
        # Auto-configure dimensionality if data is provided
        if data_dict:
            first_target = next(iter(data_dict.values()))[0]
            if 'data' in first_target and 'data' in first_target['data']:
                img_data = first_target['data']['data']
                data_is_2d = len(img_data.shape) == 2
                config_is_2d = len(self.train_patch_size) == 2
                
                if data_is_2d != config_is_2d:
                    self._reconfigure_dimensionality(data_is_2d)
        
        if self.verbose:
            print(f"Set targets: {list(self.targets.keys())}")
            print(f"Output channels: {self.out_channels}")
        
        return data_dict
    
    def _reconfigure_dimensionality(self, data_is_2d):
        """
        Reconfigure model operations based on data dimensionality
        
        Parameters
        ----------
        data_is_2d : bool
            True if data is 2D, False if 3D
        """
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

    def update_config(self, patch_size=None, min_labeled_ratio=None, max_epochs=None, loss_function=None):
        """
        Generic method to update configuration parameters
        
        Parameters
        ----------
        patch_size : tuple or list, optional
            New patch size, e.g., (128, 128, 128) or (128, 128)
        min_labeled_ratio : float, optional
            Minimum labeled ratio (0.0 to 1.0)
        max_epochs : int, optional
            Maximum number of epochs
        loss_function : str, optional
            Loss function name
        """
        if patch_size is not None:
            if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2:
                self.train_patch_size = tuple(patch_size)
                self.tr_configs["patch_size"] = list(patch_size)
                
                # Use centralized dimensionality function to set appropriate operations
                dim_props = determine_dimensionality(self.train_patch_size, self.verbose)
                self.model_config["conv_op"] = dim_props["conv_op"]
                self.model_config["pool_op"] = dim_props["pool_op"]
                self.model_config["norm_op"] = dim_props["norm_op"]
                self.model_config["dropout_op"] = dim_props["dropout_op"]
                self.spacing = dim_props["spacing"]
                self.op_dims = dim_props["op_dims"]
                
                if self.verbose:
                    print(f"Updated patch size: {self.train_patch_size}")
        
        if min_labeled_ratio is not None:
            self.min_labeled_ratio = float(min_labeled_ratio)
            self.dataset_config["min_labeled_ratio"] = self.min_labeled_ratio
            if self.verbose:
                print(f"Updated min labeled ratio: {self.min_labeled_ratio:.2f}")
        
        if max_epochs is not None:
            # Set max_steps_per_epoch and max_val_steps_per_epoch to None so that train.py will use the dataset length
            self.max_steps_per_epoch = None
            self.max_val_steps_per_epoch = None
            self.max_epoch = int(max_epochs)
            self.tr_configs["max_epoch"] = self.max_epoch
            if self.verbose:
                print(f"Updated max epochs: {self.max_epoch}")
        
        if loss_function is not None:
            self.selected_loss_function = loss_function
            # Apply to existing targets if they exist
            if hasattr(self, 'targets') and self.targets:
                for target_name in self.targets:
                    self.targets[target_name]["loss_fn"] = self.selected_loss_function
                if self.verbose:
                    print(f"Applied loss function '{self.selected_loss_function}' to all targets")
            elif self.verbose:
                print(f"Set loss function: {self.selected_loss_function}")
    
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
