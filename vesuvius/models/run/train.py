from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.training.lr_schedulers import get_scheduler, PolyLRScheduler
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.utils import init_weights_he
import albumentations as A
from models.datasets import NapariDataset, TifDataset, ZarrDataset
from utils.plotting import save_debug
from models.build.build_network_from_config import NetworkFromConfig

from models.training.loss.losses import _create_loss


from models.training.optimizers import create_optimizer
# Augmentations will be handled directly in this file
from models.augmentation.transforms.utils.compose import ComposeTransforms
from models.augmentation.transforms.utils.random import RandomTransform
from models.augmentation.transforms.spatial.mirroring import MirrorTransform
from models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform


class BaseTrainer:
    def __init__(self,
                 mgr=None,
                 verbose: bool = True):
        """
        Initialize the trainer with a config manager instance

        Parameters
        ----------
        mgr : ConfigManager, optional
            If provided, use this config manager instance instead of creating a new one
        verbose : bool
            Whether to print verbose output
        """
        if mgr is not None:
            self.mgr = mgr
        else:
            # Import ConfigManager here to avoid circular imports
            from vesuvius.models.configuration.config_manager import ConfigManager
            self.mgr = ConfigManager(verbose)

    # --- build model --- #
    def _build_model(self):
        # Ensure model_config is initialized
        # If running directly, we need to make sure config is not None
        if not hasattr(self.mgr, 'model_config') or self.mgr.model_config is None:
            print("Initializing model_config with defaults")
            self.mgr.model_config = {
                "train_patch_size": self.mgr.train_patch_size,
                "in_channels": self.mgr.in_channels,
                "model_name": self.mgr.model_name,
                "autoconfigure": self.mgr.autoconfigure,
                "conv_op": "nn.Conv2d" if len(self.mgr.train_patch_size) == 2 else "nn.Conv3d"
            }

        model = NetworkFromConfig(self.mgr)
        return model

    def _create_training_transforms(self):
        """
        Create training transforms using custom batchgeneratorsv2.
        Returns None for validation (no augmentations).
        """
        dimension = len(self.mgr.train_patch_size)

        if dimension == 2:
            # 2D transforms (no transpose transform)
            transforms = [
                RandomTransform(
                    MirrorTransform(allowed_axes=(0, 1)), 
                    apply_probability=0.5
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.5),
                        synchronize_channels=True,
                        synchronize_axes=False,
                        p_per_channel=1.0
                    ),
                    apply_probability=0.2
                )
            ]
        else:
            # 3D transforms
            transforms = [
                RandomTransform(
                    MirrorTransform(allowed_axes=(0, 1, 2)), 
                    apply_probability=0.5
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.0),
                        synchronize_channels=True,
                        synchronize_axes=False,
                        p_per_channel=1.0
                    ),
                    apply_probability=0.2
                )
            ]

            # Only add transpose transform if all three dimensions (z, y, x) are equal
            patch_d, patch_h, patch_w = self.mgr.train_patch_size
            if patch_d == patch_h == patch_w:
                transforms.insert(1, RandomTransform(
                    TransposeAxesTransform(allowed_axes={0, 1, 2}), 
                    apply_probability=0.5
                ))
                print(f"Added transpose transform for 3D (equal dimensions: {patch_d}x{patch_h}x{patch_w})")
            else:
                print(f"Skipped transpose transform for 3D (unequal dimensions: {patch_d}x{patch_h}x{patch_w})")

        return ComposeTransforms(transforms)

    # --- configure dataset --- #
    def _configure_dataset(self):
        # Get data format from config manager, default to zarr
        data_format = getattr(self.mgr, 'data_format', 'zarr')

        # Note: Augmentations are now handled in the training loop, not in the dataset
        if data_format == 'napari':
            dataset = NapariDataset(mgr=self.mgr,
                                   image_transforms=None,
                                   volume_transforms=None)
        elif data_format == 'tif':
            dataset = TifDataset(mgr=self.mgr,
                                image_transforms=None,
                                volume_transforms=None)
        elif data_format == 'zarr':
            dataset = ZarrDataset(mgr=self.mgr,
                                 image_transforms=None,
                                 volume_transforms=None)
        else:
            raise ValueError(f"Unsupported data format: {data_format}. "
                           f"Supported formats are: 'napari', 'tif', 'zarr'")

        print(f"Using {data_format} dataset format")
        return dataset

    # --- losses ---- #
    def _build_loss(self):
        # Use the centralized _create_loss function to instantiate loss functions
        loss_fns = {}
        for task_name, task_info in self.mgr.targets.items():
            loss_fn_name = task_info.get("loss_fn", "BCEDiceLoss")
            print(f"DEBUG: Target {task_name} using loss function: {loss_fn_name}")
            
            # Get loss kwargs from task info, or use empty dict for defaults
            loss_config = task_info.get("loss_kwargs", {})
            
            # Get other parameters that might be needed
            weight = loss_config.get("weight", None)
            ignore_index = loss_config.get("ignore_index", None)
            pos_weight = loss_config.get("pos_weight", None)
            
            # If compute_loss_on_label is set and no ignore_index is specified, use -100
            if hasattr(self.mgr, 'compute_loss_on_label') and self.mgr.compute_loss_on_label and ignore_index is None:
                ignore_index = -100
                print(f"Setting ignore_index=-100 for target '{task_name}' due to compute_loss_on_label=True")
            
            # Create the loss function using the factory
            try:
                loss_fns[task_name] = _create_loss(
                    name=loss_fn_name,
                    loss_config=loss_config,
                    weight=weight,
                    ignore_index=ignore_index,
                    pos_weight=pos_weight
                )
            except RuntimeError as e:
                raise ValueError(f"Failed to create loss function '{loss_fn_name}' for target '{task_name}': {str(e)}")

        return loss_fns

    # --- optimizer ---- #
    def _get_optimizer(self, model):

        optimizer_config = {
            'name': self.mgr.optimizer,
            'learning_rate': self.mgr.initial_lr,
            'weight_decay': self.mgr.weight_decay
        }

        return create_optimizer(optimizer_config, model)

    # --- scheduler --- #
    def _get_scheduler(self, optimizer):
        # Get scheduler type from config or use 'poly' as default
        scheduler_type = getattr(self.mgr, 'scheduler', 'poly')

        # Get scheduler-specific kwargs from config if available
        scheduler_kwargs = getattr(self.mgr, 'scheduler_kwargs', {})

        # Use the factory function to create the scheduler
        scheduler = get_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            initial_lr=self.mgr.initial_lr,
            max_steps=self.mgr.max_epoch,
            **scheduler_kwargs
        )

        print(f"Using {scheduler_type} learning rate scheduler")
        return scheduler

    # --- scaler --- #
    def _get_scaler(self, device_type='cuda'):
        # for cuda, we can use a grad scaler for mixed precision training
        # for mps or cpu, we create a dummy scaler that does nothing
        if device_type == 'cuda':
            return torch.amp.GradScaler()
        else:
            class DummyScaler:
                def scale(self, loss):
                    return loss

                def unscale_(self, optimizer):
                    pass

                def step(self, optimizer):
                    optimizer.step()

                def update(self):
                    pass

            return DummyScaler()

    # --- dataloaders --- #
    def _configure_dataloaders(self, dataset):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        train_val_split = self.mgr.tr_val_split
        split = int(np.floor(train_val_split * dataset_size))
        train_indices, val_indices = indices[:split], indices[split:]
        batch_size = self.mgr.train_batch_size

        device_type = 'mps' if hasattr(torch, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

        # For MPS device, set num_workers=0 to avoid pickling error
        if device_type == 'mps':
            num_workers = 0
        else:
            # For CUDA or CPU, use the configured number of workers
            # If train_num_dataloader_workers is not set, default to 0
            if hasattr(self.mgr, 'train_num_dataloader_workers'):
                num_workers = self.mgr.train_num_dataloader_workers
            else:
                num_workers = 4

        train_dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=(device_type == 'cuda'),  # pin_memory=True for CUDA
                                num_workers=num_workers)

        val_dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=(device_type == 'cuda'), 
                                    num_workers=num_workers)

        return train_dataloader, val_dataloader, train_indices, val_indices


    def train(self):

        dataset = self._configure_dataset()
        
        # Auto-detect channels from dataset if needed
        self.mgr.auto_detect_channels(dataset)
        
        model = self._build_model()
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        scheduler = self._get_scheduler(optimizer)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA device")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using CPU device")

        # Apply weight initialization with recommended negative_slope=0.2 for LeakyReLU
        # Do this BEFORE device transfer and compilation

        model.apply(lambda module: init_weights_he(module, neg_slope=0.2))

        model = model.to(device)

        # Only compile the model if it's on CUDA (not supported on MPS/CPU)
        if device.type == 'cuda':
            model = torch.compile(model, mode="default", fullgraph=False)

        # Create a no_op context manager as it might be needed for MPS
        if not hasattr(torch, 'no_op'):
            # Define a simple no-op context manager if not available
            class NullContextManager:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass

            torch.no_op = lambda: NullContextManager()

        # Create appropriate scaler for the device
        scaler = self._get_scaler(device.type)

        train_dataloader, val_dataloader, train_indices, val_indices = self._configure_dataloaders(dataset)

        # Save initial configuration to checkpoint directory if requested
        if model.save_config:
            self.mgr.save_config()

        start_epoch = 0

        # Create base checkpoint directory if it doesn't exist
        os.makedirs(self.mgr.ckpt_out_base, exist_ok=True)

        # Create a specific directory for this model's checkpoints and configs
        model_ckpt_dir = os.path.join(self.mgr.ckpt_out_base, self.mgr.model_name)
        os.makedirs(model_ckpt_dir, exist_ok=True)

        # Check for a valid, non-empty checkpoint path
        valid_checkpoint = (self.mgr.checkpoint_path is not None and 
                           self.mgr.checkpoint_path != "" and 
                           Path(self.mgr.checkpoint_path).exists())

        if valid_checkpoint:
            print(f"Loading checkpoint from {self.mgr.checkpoint_path}")
            checkpoint = torch.load(self.mgr.checkpoint_path, map_location=device)

            # Check if this checkpoint has model configuration
            if 'model_config' in checkpoint:
                print("Found model configuration in checkpoint, using it to initialize the model")

                # Update the manager with the saved configuration if needed
                if hasattr(self.mgr, 'targets') and 'targets' in checkpoint['model_config']:
                    self.mgr.targets = checkpoint['model_config']['targets']
                    print(f"Updated targets from checkpoint: {self.mgr.targets}")
            
            # Check for normalization information in checkpoint
            if 'normalization_scheme' in checkpoint:
                print(f"Found normalization scheme in checkpoint: {checkpoint['normalization_scheme']}")
                self.mgr.normalization_scheme = checkpoint['normalization_scheme']
                if hasattr(self.mgr, 'dataset_config'):
                    self.mgr.dataset_config['normalization_scheme'] = checkpoint['normalization_scheme']
            
            if 'intensity_properties' in checkpoint:
                print("Found intensity properties in checkpoint")
                self.mgr.intensity_properties = checkpoint['intensity_properties']
                if hasattr(self.mgr, 'dataset_config'):
                    self.mgr.dataset_config['intensity_properties'] = checkpoint['intensity_properties']
                # Print the loaded intensity properties
                print("Loaded intensity properties:")
                for key, value in checkpoint['intensity_properties'].items():
                    print(f"  {key}: {value:.4f}")

                # We may need to rebuild the model with the saved configuration
                if model.autoconfigure != checkpoint['model_config'].get('autoconfigure', True):
                    print("Model autoconfiguration differs, rebuilding model from checkpoint config")

                    # Create a version of the manager with the checkpoint's configuration
                    class ConfigWrapper:
                        def __init__(self, config_dict, base_mgr):
                            self.__dict__.update(config_dict)
                            # Add any missing attributes from the base manager
                            for attr_name in dir(base_mgr):
                                if not attr_name.startswith('__') and not hasattr(self, attr_name):
                                    setattr(self, attr_name, getattr(base_mgr, attr_name))

                    config_wrapper = ConfigWrapper(checkpoint['model_config'], self.mgr)
                    model = NetworkFromConfig(config_wrapper)

                    model = model.to(device)
                    # Only compile for CUDA with explicit parameters
                    if device.type == 'cuda':
                        model = torch.compile(model, mode="default", fullgraph=False)

                    # Also recreate optimizer since the model parameters changed
                    optimizer = self._get_optimizer(model)

            # Load model weights
            model.load_state_dict(checkpoint['model'])

            if not self.mgr.load_weights_only:
                # Only load optimizer, scheduler, epoch if we are NOT in "weights_only" mode
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            else:
                # Start a 'new run' from epoch 0 or 1
                start_epoch = 0
                scheduler = self._get_scheduler(optimizer)
                print("Loaded model weights only; starting new training run from epoch 1.")

        global_step = 0
        grad_accumulate_n = self.mgr.gradient_accumulation

        # Initialize training transforms
        train_transforms = self._create_training_transforms()

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()

            train_running_losses = {t_name: 0.0 for t_name in self.mgr.targets}
            # Reset loss components for new epoch
            if hasattr(self, '_loss_components'):
                self._loss_components = {t_name: {'dice': [], 'ce': [], 'total': []} for t_name in self.mgr.targets}
            # Use the length of train indices as the number of iterations per epoch
            train_dataloader_iter = iter(train_dataloader)

            # Determine number of iterations based on max_steps_per_epoch if set
            if hasattr(self.mgr, 'max_steps_per_epoch') and self.mgr.max_steps_per_epoch and self.mgr.max_steps_per_epoch > 0:
                num_iters = min(len(train_indices), self.mgr.max_steps_per_epoch)
            else:
                num_iters = len(train_indices)  # Use all data (current behavior)

            pbar = tqdm(range(num_iters), total=num_iters)
            steps = 0

            for i in pbar:
                try:
                    data_dict = next(train_dataloader_iter)
                except StopIteration:
                    # Reset iterator if we run out of data
                    train_dataloader_iter = iter(train_dataloader)
                    data_dict = next(train_dataloader_iter)

                if epoch == 0 and i == 0:
                    print("Items from the first batch -- Double check that your shapes and values are expected:")
                    for item in data_dict:
                        if isinstance(data_dict[item], dict):
                            # Handle dictionary items like ignore_masks
                            print(f"{item}: (dictionary with keys: {list(data_dict[item].keys())})")
                            for sub_key, sub_val in data_dict[item].items():
                                print(f"  {sub_key}: {sub_val.dtype}")
                                print(f"  {sub_key}: {sub_val.shape}")
                                print(f"  {sub_key}: min : {sub_val.min()} max : {sub_val.max()}")
                        else:
                            # Handle tensor items
                            print(f"{item}: {data_dict[item].dtype}")
                            print(f"{item}: {data_dict[item].shape}")
                            print(f"{item}: min : {data_dict[item].min()} max : {data_dict[item].max()}")

                global_step += 1

                # Apply augmentations to individual samples in the batch (only during training)
                if train_transforms is not None:
                    # Apply transforms to each sample in the batch individually
                    batch_size = data_dict["image"].shape[0]
                    # Initialize transformed_batch with proper structure
                    transformed_batch = {}
                    for key in data_dict.keys():
                        if key == "ignore_masks":
                            # Initialize ignore_masks as a dictionary of lists
                            transformed_batch[key] = {mask_key: [] for mask_key in data_dict[key].keys()}
                        else:
                            transformed_batch[key] = []

                    for batch_idx in range(batch_size):
                        # Extract single sample from batch
                        sample_dict = {}
                        for key, value in data_dict.items():
                            if key == "ignore_masks":
                                # Handle ignore_masks dictionary
                                sample_dict[key] = {
                                    mask_key: mask_value[batch_idx] 
                                    for mask_key, mask_value in value.items()
                                }
                            else:
                                sample_dict[key] = value[batch_idx]

                        # Apply transforms to single sample
                        transformed_sample = train_transforms(**sample_dict)

                        # Collect transformed samples
                        for key, value in transformed_sample.items():
                            if key == "ignore_masks":
                                # Handle ignore_masks dictionary
                                if key not in transformed_batch:
                                    transformed_batch[key] = {mask_key: [] for mask_key in value.keys()}
                                for mask_key, mask_value in value.items():
                                    transformed_batch[key][mask_key].append(mask_value)
                            else:
                                transformed_batch[key].append(value)

                    # Stack samples back into batch
                    for key, value_list in transformed_batch.items():
                        if key == "ignore_masks":
                            # Handle ignore_masks dictionary
                            data_dict[key] = {
                                mask_key: torch.stack(mask_value_list) 
                                for mask_key, mask_value_list in value_list.items()
                            }
                        else:
                            data_dict[key] = torch.stack(value_list)

                    # --- NEW: make sure each target has proper channel dimension ---
                    for t_name in self.mgr.targets:
                        if t_name in data_dict:  # Check if this target exists in the batch
                            t_tensor = data_dict[t_name]
                            # expected dims: B + C + spatial (len(patch_size))
                            expected_dims = 2 + len(self.mgr.train_patch_size)

                            if t_tensor.dim() == expected_dims - 1:
                                # Missing channel dimension - check target config for expected channels
                                target_info = self.mgr.targets[t_name]
                                expected_channels = target_info.get("out_channels", 1)

                                if expected_channels == 1:
                                    # Single channel case - add channel dimension at dim=1
                                    data_dict[t_name] = t_tensor.unsqueeze(1)
                                else:
                                    # Multi-channel case - this shouldn't happen as multi-channel targets
                                    # should maintain their channel dimension through augmentations
                                    # But if it does, we need to handle it appropriately
                                    print(f"Warning: Target {t_name} expected {expected_channels} channels but channel dim was squeezed")
                                    data_dict[t_name] = t_tensor.unsqueeze(1)  # Add single channel for now

                inputs = data_dict["image"].to(device, dtype=torch.float32)
                # Get the ignore masks dictionary if it exists
                ignore_masks = None
                if "ignore_masks" in data_dict:
                    ignore_masks = {
                        t_name: mask.to(device, dtype=torch.float32)
                        for t_name, mask in data_dict["ignore_masks"].items()
                    }

                # Create targets_dict excluding both image and ignore_masks
                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k != "image" and k != "ignore_masks"
                }

                # forward
                # Use device-specific autocast or context manager
                context = (
                    torch.amp.autocast(device.type) if device.type == 'cuda' 
                    else torch.amp.autocast('cpu') if device.type == 'cpu' 
                    else torch.no_op() if device.type == 'mps' 
                    else torch.no_op()
                )

                with context:
                    outputs = model(inputs)
                    total_loss = 0.0
                    per_task_losses = {}

                    for t_name, t_gt in targets_dict.items():
                        t_pred = outputs[t_name]
                        t_loss_fn = loss_fns[t_name]
                        task_weight = self.mgr.targets[t_name].get("weight", 1.0)



                        # Apply the per-target ignore mask if available
                        if ignore_masks is not None and t_name in ignore_masks:
                            ignore_mask = ignore_masks[t_name].to(device, dtype=torch.float32)

                            # Print a message showing we're using the custom mask
                            if steps == 0 and i == 0:
                                print(f"Using custom ignore mask for target {t_name}")

                            # Check if loss function has MaskingLossWrapper
                            if hasattr(t_loss_fn, 'ignore_index'):
                                ignore_label = t_loss_fn.ignore_index
                            else:
                                # Default ignore index if not wrapped
                                ignore_label = -100

                            # Ensure ignore mask has the same number of dimensions as target
                            if ignore_mask.dim() == t_gt.dim() - 1:
                                # Add channel dimension to match target
                                ignore_mask = ignore_mask.unsqueeze(1)

                            # Apply mask to target: set regions where mask is 1 to ignore_label
                            t_gt = torch.where(ignore_mask == 1, torch.tensor(ignore_label, dtype=t_gt.dtype, device=t_gt.device), t_gt)

                        # Calculate the loss (no need to pass mask, it's now embedded in the target)
                        t_loss = t_loss_fn(t_pred, t_gt)
                        
                        # Extract individual loss components if available
                        if hasattr(t_loss_fn, 'last_dc_loss') and hasattr(t_loss_fn, 'last_ce_loss'):
                            # Store individual components for display
                            if not hasattr(self, '_loss_components'):
                                self._loss_components = {t_name: {'dice': [], 'ce': [], 'total': []} for t_name in self.mgr.targets}
                            
                            dice_score = 1.0 - t_loss_fn.last_dc_loss  # Convert from loss to score
                            ce_loss = t_loss_fn.last_ce_loss
                            
                            self._loss_components[t_name]['dice'].append(dice_score)
                            self._loss_components[t_name]['ce'].append(ce_loss)
                            self._loss_components[t_name]['total'].append(t_loss.item())

                        # Apply task weight to the loss
                        total_loss += task_weight * t_loss
                        train_running_losses[t_name] += t_loss.item()
                        per_task_losses[t_name] = t_loss.item()

                # backward
                # loss \ accumulation steps to maintain same effective batch size
                total_loss = total_loss / grad_accumulate_n
                # backward
                scaler.scale(total_loss).backward()

                if (i + 1) % grad_accumulate_n == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1

                desc_parts = []
                for t_name in self.mgr.targets:
                    # Check if we have individual loss components
                    if hasattr(self, '_loss_components') and t_name in self._loss_components and len(self._loss_components[t_name]['dice']) > 0:
                        # Use recent values for smoother display
                        recent_dice = np.mean(self._loss_components[t_name]['dice'][-50:])
                        recent_ce = np.mean(self._loss_components[t_name]['ce'][-50:])
                        recent_total = np.mean(self._loss_components[t_name]['total'][-50:])
                        desc_parts.append(f"{t_name}: Dice={recent_dice:.3f}, CE={recent_ce:.3f}, Total={recent_total:.4f}")
                    else:
                        # Fallback to simple loss display
                        avg_t_loss = train_running_losses[t_name] / steps if steps > 0 else 0
                        desc_parts.append(f"{t_name}: {avg_t_loss:.4f}")

                desc_str = f"Epoch {epoch + 1} => " + " | ".join(desc_parts)
                pbar.set_description(desc_str)

            pbar.close()

            # Apply any remaining gradients at the end of the epoch
            if steps % grad_accumulate_n != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Step the scheduler once after each epoch
            scheduler.step()

            # Print epoch summary with individual components
            print(f"\n[Train] Epoch {epoch + 1} completed.")
            for t_name in self.mgr.targets:
                # Avoid division by zero
                epoch_avg = train_running_losses[t_name] / steps if steps > 0 else 0
                
                # Print individual components if available
                if hasattr(self, '_loss_components') and t_name in self._loss_components and len(self._loss_components[t_name]['dice']) > 0:
                    avg_dice = np.mean(self._loss_components[t_name]['dice'])
                    avg_ce = np.mean(self._loss_components[t_name]['ce'])
                    print(f"  {t_name}: Avg Dice Score = {avg_dice:.3f}, Avg CE Loss = {avg_ce:.3f}, Avg Total Loss = {epoch_avg:.4f}")
                else:
                    print(f"  {t_name}: Avg Loss = {epoch_avg:.4f}")

            # Get model and checkpoint path within the model-specific directory
            ckpt_path = f"{model_ckpt_dir}/{self.mgr.model_name}_{epoch + 1}.pth"

            # Save checkpoint with model weights and training state
            # Include the model configuration directly in the checkpoint
            # Also include normalization information
            checkpoint_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'model_config': model.final_config  # Save the model configuration
            }
            
            # Add normalization information from the dataset
            if hasattr(dataset, 'normalization_scheme'):
                checkpoint_data['normalization_scheme'] = dataset.normalization_scheme
            if hasattr(dataset, 'intensity_properties'):
                checkpoint_data['intensity_properties'] = dataset.intensity_properties
                
            torch.save(checkpoint_data, ckpt_path)

            print(f"Checkpoint saved to: {ckpt_path}")

            # Configuration is already saved in the checkpoint, no need to save it again here

            # clean up old checkpoints and configs -- currently just keeps 10 newest
            # Path to the model-specific checkpoint directory
            ckpt_dir = Path(model_ckpt_dir)

            # Get all checkpoint files (.pth) and sort by modification time
            all_checkpoints = sorted(
                ckpt_dir.glob(f"{self.mgr.model_name}_*.pth"),
                key=lambda x: x.stat().st_mtime
            )

            # Get all config files (.yaml) and sort by modification time
            all_configs = sorted(
                ckpt_dir.glob(f"{self.mgr.model_name}_*.yaml"),
                key=lambda x: x.stat().st_mtime
            )

            # Keep only the 10 newest checkpoint files
            while len(all_checkpoints) > 10:
                oldest = all_checkpoints.pop(0)
                oldest.unlink()
                print(f"Removed old checkpoint: {oldest}")

            # Keep only the 1 newest config file (others are redundant)
            while len(all_configs) > 1:
                oldest = all_configs.pop(0)
                oldest.unlink()
                print(f"Removed old config: {oldest}")

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_running_losses = {t_name: 0.0 for t_name in self.mgr.targets}
                    val_steps = 0

                    # Use the length of val indices as the number of iterations for validation
                    val_dataloader_iter = iter(val_dataloader)

                    # Determine number of validation iterations based on max_val_steps_per_epoch if set
                    if hasattr(self.mgr, 'max_val_steps_per_epoch') and self.mgr.max_val_steps_per_epoch and self.mgr.max_val_steps_per_epoch > 0:
                        num_val_iters = min(len(val_indices), self.mgr.max_val_steps_per_epoch)
                    else:
                        num_val_iters = len(val_indices)  # Use all data (current behavior)

                    pbar = tqdm(range(num_val_iters), total=num_val_iters)
                    for i in pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            # Reset iterator if we run out of data
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        # Get the ignore masks dictionary if it exists
                        ignore_masks = None
                        if "ignore_masks" in data_dict:
                            ignore_masks = {
                                t_name: mask.to(device, dtype=torch.float32)
                                for t_name, mask in data_dict["ignore_masks"].items()
                            }

                        # Create targets_dict excluding both image and ignore_masks
                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image" and k != "ignore_masks"
                        }

                        # Use the same context as in training
                        context = (
                            torch.amp.autocast(device.type) if device.type == 'cuda' 
                            else torch.amp.autocast('cpu') if device.type == 'cpu' 
                            else torch.no_op() if device.type == 'mps' 
                            else torch.no_op()
                        )

                        with context:
                            outputs = model(inputs)
                            total_val_loss = 0.0
                            for t_name, t_gt in targets_dict.items():
                                t_pred = outputs[t_name]
                                t_loss_fn = loss_fns[t_name]

                                # Apply the per-target ignore mask if available
                                if ignore_masks is not None and t_name in ignore_masks:
                                    ignore_mask = ignore_masks[t_name].to(device, dtype=torch.float32)

                                    # Check if loss function has MaskingLossWrapper
                                    if hasattr(t_loss_fn, 'ignore_index'):
                                        ignore_label = t_loss_fn.ignore_index
                                    else:
                                        # Default ignore index if not wrapped
                                        ignore_label = -100

                                    # Ensure ignore mask has the same number of dimensions as target
                                    if ignore_mask.dim() == t_gt.dim() - 1:
                                        # Add channel dimension to match target
                                        ignore_mask = ignore_mask.unsqueeze(1)

                                    # Apply mask to target: set regions where mask is 1 to ignore_label
                                    t_gt = torch.where(ignore_mask == 1, torch.tensor(ignore_label, dtype=t_gt.dtype, device=t_gt.device), t_gt)

                                # Calculate the loss (no need to pass mask, it's now embedded in the target)
                                t_loss = t_loss_fn(t_pred, t_gt)

                                total_val_loss += t_loss
                                val_running_losses[t_name] += t_loss.item()

                            val_steps += 1

                            if i == 0:
                                b_idx = 0  # pick which sample in the batch to visualize
                                # Slicing shape: [1, c, z, y, x ]
                                inputs_first = inputs[b_idx: b_idx + 1]

                                targets_dict_first = {}
                                for t_name, t_tensor in targets_dict.items():
                                    targets_dict_first[t_name] = t_tensor[b_idx: b_idx + 1]

                                outputs_dict_first = {}
                                for t_name, p_tensor in outputs.items():
                                    outputs_dict_first[t_name] = p_tensor[b_idx: b_idx + 1]

                                # create debug visualization (gif for 3D, png for 2D) in the model-specific directory
                                debug_img_path = f"{model_ckpt_dir}/{self.mgr.model_name}_debug.gif"
                                save_debug(
                                    input_volume=inputs_first,
                                    targets_dict=targets_dict_first,
                                    outputs_dict=outputs_dict_first,
                                    tasks_dict=self.mgr.targets, # your dictionary, e.g. {"sheet": {"activation":"sigmoid"}, "normals": {"activation":"none"}}
                                    epoch=epoch,
                                    save_path=debug_img_path
                                )

                    desc_parts = []
                    for t_name in self.mgr.targets:
                        # Avoid division by zero
                        avg_loss_for_t = val_running_losses[t_name] / val_steps if val_steps > 0 else 0
                        desc_parts.append(f"{t_name} {avg_loss_for_t:.4f}")
                    desc_str = "Val: " + " | ".join(desc_parts)
                    pbar.set_description(desc_str)

                pbar.close()

                # Final avg for each task
                print(f"\n[Validation] Epoch {epoch + 1} summary:")
                for t_name in self.mgr.targets:
                    # Avoid division by zero
                    val_avg = val_running_losses[t_name] / val_steps if val_steps > 0 else 0
                    print(f"  Task '{t_name}': Avg validation loss = {val_avg:.4f}")

            # Scheduler step happens once per epoch after training

        print('Training Finished!')

        # Save final model with configuration in the model-specific directory
        final_model_path = f"{model_ckpt_dir}/{self.mgr.model_name}_final.pth"

        # Save the complete checkpoint with configuration embedded
        final_checkpoint_data = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': self.mgr.max_epoch - 1,
            'model_config': model.final_config
        }
        
        # Add normalization information from the dataset
        if hasattr(dataset, 'normalization_scheme'):
            final_checkpoint_data['normalization_scheme'] = dataset.normalization_scheme
        if hasattr(dataset, 'intensity_properties'):
            final_checkpoint_data['intensity_properties'] = dataset.intensity_properties
            
        torch.save(final_checkpoint_data, final_model_path)

        print(f"Final model saved to {final_model_path}")
        print(f"Model configuration is embedded in the checkpoint")



def detect_targets_from_data(mgr):
    """
    Detect available targets from the data directory structure.

    Parameters
    ----------
    mgr : ConfigManager
        The configuration manager instance
    """
    data_path = Path(mgr.data_path)
    images_dir = data_path / "images"

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Find target names from image files
    targets = set()

    if mgr.data_format == "zarr":
        # Look for .zarr directories with format imageN_target.zarr
        zarr_dirs = [d for d in images_dir.iterdir() if d.is_dir() and d.suffix == '.zarr']
        for zarr_dir in zarr_dirs:
            stem = zarr_dir.stem  # Remove .zarr extension
            if '_' in stem:
                target = stem.rsplit('_', 1)[1]
                targets.add(target)

    elif mgr.data_format == "tif":
        # Look for .tif files with format imageN_target.tif
        tif_files = images_dir.glob("*.tif")
        for tif_file in tif_files:
            stem = tif_file.stem
            if '_' in stem:
                target = stem.rsplit('_', 1)[1]
                targets.add(target)

    elif mgr.data_format == "napari":
        # For napari, targets would need to be configured externally
        # as napari datasets are typically loaded programmatically
        print("Warning: Target detection not implemented for napari format. Please configure targets in config file.")
        return

    # Only create default target configuration if no targets are already configured
    if targets and (not hasattr(mgr, 'targets') or not mgr.targets):
        mgr.targets = {}
        for target in sorted(targets):
            mgr.targets[target] = {
                "out_channels": 2,  # Always use at least 2 channels
                "activation": "softmax",  # Use softmax for multi-channel output
                "loss_fn": "CrossEntropyLoss"  # Use CrossEntropyLoss for 2-channel binary
            }

        print(f"Detected targets from data: {list(targets)}")
    elif targets and mgr.targets:
        # Check if all detected targets are configured
        configured_targets = set(mgr.targets.keys())
        missing_targets = targets - configured_targets
        if missing_targets:
            print(f"Warning: Detected targets {missing_targets} are not configured in the config file")
        print(f"Using configured targets: {list(mgr.targets.keys())}")
    else:
        print("No targets detected from data. Please configure targets in config file.")


def apply_loss_functions_to_targets(mgr, loss_list):
    """
    Apply loss functions to targets in order.

    Parameters
    ----------
    mgr : ConfigManager
        The configuration manager instance
    loss_list : list
        List of loss function names
    """
    # If no targets are configured yet, detect them from the data
    if not hasattr(mgr, 'targets') or not mgr.targets:
        detect_targets_from_data(mgr)

    if not mgr.targets:
        raise ValueError("No targets found. Please ensure your data directory has the correct structure or configure targets in a config file.")

    target_names = list(mgr.targets.keys())

    # Apply loss functions in order
    for i, target_name in enumerate(target_names):
        if i < len(loss_list):
            loss_fn = loss_list[i]
        else:
            # If more targets than loss functions, use the last loss function
            loss_fn = loss_list[-1] 

        mgr.targets[target_name]["loss_fn"] = loss_fn
        print(f"Applied {loss_fn} to target '{target_name}'")


def update_config_from_args(mgr, args):
    """
    Update ConfigManager with command line arguments.

    Parameters
    ----------
    mgr : ConfigManager
        The configuration manager instance
    args : argparse.Namespace
        Parsed command line arguments
    """
    # Set data path (maps to -i argument)
    mgr.data_path = Path(args.input)

    # Set data format
    mgr.data_format = args.format

    # Set checkpoint output directory
    mgr.ckpt_out_base = Path(args.output)
    mgr.tr_info["ckpt_out_base"] = str(mgr.ckpt_out_base)

    # Update optional parameters if provided
    if args.batch_size is not None:
        mgr.train_batch_size = args.batch_size
        mgr.tr_configs["batch_size"] = args.batch_size

    if args.patch_size is not None:
        # Parse patch size from string like "192,192,192" or "256,256"
        try:
            patch_size = [int(x.strip()) for x in args.patch_size.split(',')]
            mgr.update_config(patch_size=patch_size)
        except ValueError as e:
            raise ValueError(f"Invalid patch size format: {args.patch_size}. Expected comma-separated integers like '192,192,192'")

    if args.train_split is not None:
        if not 0.0 <= args.train_split <= 1.0:
            raise ValueError(f"Train split must be between 0.0 and 1.0, got {args.train_split}")
        mgr.tr_val_split = args.train_split
        mgr.tr_info["tr_val_split"] = args.train_split

    if args.loss_on_label_only:
        mgr.compute_loss_on_label = True
        mgr.tr_info["compute_loss_on_label"] = True

    # Handle max steps per epoch if provided
    if args.max_steps_per_epoch is not None:
        mgr.max_steps_per_epoch = args.max_steps_per_epoch
        mgr.tr_configs["max_steps_per_epoch"] = args.max_steps_per_epoch

    if args.max_val_steps_per_epoch is not None:
        mgr.max_val_steps_per_epoch = args.max_val_steps_per_epoch
        mgr.tr_configs["max_val_steps_per_epoch"] = args.max_val_steps_per_epoch

    # Handle model name
    if args.model_name is not None:
        mgr.model_name = args.model_name
        mgr.tr_info["model_name"] = args.model_name
        if mgr.verbose:
            print(f"Set model name: {mgr.model_name}")

    # Handle nonlinearity/activation function
    if args.nonlin is not None:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["nonlin"] = args.nonlin
        if mgr.verbose:
            print(f"Set activation function: {args.nonlin}")

    # Handle squeeze and excitation
    if args.se:
        if not hasattr(mgr, 'model_config') or mgr.model_config is None:
            mgr.model_config = {}
        mgr.model_config["squeeze_excitation"] = True
        mgr.model_config["squeeze_excitation_reduction_ratio"] = args.se_reduction_ratio
        if mgr.verbose:
            print(f"Enabled squeeze and excitation with reduction ratio: {args.se_reduction_ratio}")

    # Handle loss functions
    if args.loss is not None:
        # Parse loss function list like "[SoftDiceLoss, BCEWithLogitsLoss]"
        try:
            import ast
            loss_list = ast.literal_eval(args.loss)
            if not isinstance(loss_list, list):
                loss_list = [loss_list]  # Convert single item to list
        except (ValueError, SyntaxError):
            # Try parsing as comma-separated string
            loss_list = [s.strip() for s in args.loss.split(',')]

        # Apply loss functions to targets in order
        apply_loss_functions_to_targets(mgr, loss_list)


def main():
    """Main entry point for the training script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Vesuvius neural networks for ink detection and segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-i", "--input", required=True, 
                       help="Input directory containing images/, labels/, and optionally masks/ subdirectories")
    parser.add_argument("-o", "--output", required=True,
                       help="Output directory for saving checkpoints and configurations")
    parser.add_argument("--format", required=True, choices=["tif", "zarr", "napari"],
                       help="Data format (tif: TIFF files, zarr: Zarr arrays, napari: Napari layers)")

    # Optional arguments
    parser.add_argument("--batch-size", type=int, 
                       help="Training batch size (default: from config or 2)")
    parser.add_argument("--patch-size", type=str,
                       help="Patch size as comma-separated values, e.g., '192,192,192' for 3D or '256,256' for 2D")
    parser.add_argument("--loss", type=str, 
                       help="Loss functions as a list, e.g., '[SoftDiceLoss, BCEWithLogitsLoss]' or comma-separated")
    parser.add_argument("--train-split", type=float,
                       help="Training/validation split ratio (0.0-1.0, default: 0.95)")
    parser.add_argument("--loss-on-label-only", action="store_true",
                       help="Compute loss only on labeled regions (use masks for loss calculation)")
    parser.add_argument("--config-path", type=str,
                       help="Path to configuration YAML file (if not provided, uses defaults)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output for debugging")
    parser.add_argument("--max-steps-per-epoch", type=int, default=200,
                       help="Maximum training steps per epoch (if not set, uses all data)")
    parser.add_argument("--max-val-steps-per-epoch", type=int, default=30,
                       help="Maximum validation steps per epoch (if not set, uses all data)")
    parser.add_argument("--model-name", type=str,
                       help="Model name for checkpoints and logging (default: from config or 'Model')")
    parser.add_argument("--nonlin", type=str, choices=["LeakyReLU", "ReLU", "SwiGLU", "swiglu", "GLU", "glu"],
                       help="Activation function to use in the model (default: from config or 'LeakyReLU')")
    parser.add_argument("--se", action="store_true", help="Enable squeeze and excitation modules in the encoder")
    parser.add_argument("--se-reduction-ratio", type=float, default=0.0625,
                       help="Squeeze excitation reduction ratio (default: 0.0625 = 1/16)")

    args = parser.parse_args()

    # Validate required arguments
    if not Path(args.input).exists():
        raise ValueError(f"Input directory does not exist: {args.input}")

    # Create output directory if it doesn't exist
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Initialize ConfigManager
    from models.configuration.config_manager import ConfigManager
    mgr = ConfigManager(verbose=args.verbose)

    # Load config file if provided, otherwise initialize with defaults
    if args.config_path:
        if not Path(args.config_path).exists():
            raise ValueError(f"Config file does not exist: {args.config_path}")
        mgr.load_config(args.config_path)
        print(f"Loaded configuration from: {args.config_path}")
    else:
        # Initialize with defaults - first set up the basic config dictionaries
        mgr.tr_info = {}
        mgr.tr_configs = {}
        mgr.model_config = {}
        mgr.dataset_config = {}
        mgr._init_attributes()
        print("Initialized with default configuration")

    # Update configuration with command line arguments
    update_config_from_args(mgr, args)

    # Create and run trainer
    trainer = BaseTrainer(mgr=mgr, verbose=args.verbose)

    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == '__main__':
    main()

# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]
