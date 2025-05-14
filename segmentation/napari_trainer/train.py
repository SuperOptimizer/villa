from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils import init_weights_he
import albumentations as A
from dataset import NapariDataset
from plotting import save_debug
from model.build_network_from_config import NetworkFromConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from losses import (
    MaskedCosineLoss, BCEDiceLoss, DC_and_CE_loss, 
    DC_and_BCE_loss, DC_and_topk_loss, SoftDiceLoss, 
    GeneralizedDiceLoss, WeightedCrossEntropyLoss,
    DC_SkelREC_and_CE_loss, TopKLoss, RobustCrossEntropyLoss
)


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
            from main_window import ConfigManager
            self.mgr = ConfigManager(verbose)

    # --- build model --- #
    def _build_model(self):
        # Ensure model_config and inference_config are initialized
        if hasattr(self.mgr, 'update_config'):
            # If running from the GUI, ensure configs are updated
            self.mgr.update_config()
        else:
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
            
            if not hasattr(self.mgr, 'inference_config') or self.mgr.inference_config is None:
                print("Initializing inference_config with model_config defaults")
                self.mgr.inference_config = self.mgr.model_config.copy()

        model = NetworkFromConfig(self.mgr)
        model.apply(lambda module: init_weights_he(module, neg_slope=1e-2))

        return model

    def _compose_augmentations(self):

        # --- Augmentations (2D only) ---
        image_transforms = A.Compose([
            # Intensity transformations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.Illumination(p=1.0),
            ], p=0.5),

            # Noise transformations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MultiplicativeNoise(),
            ], p=0.5),

            # Blur and quality transformations
            A.OneOf([
                A.MotionBlur(blur_limit=7),
                A.Defocus(radius=(3, 7)),
                A.Downscale(scale_min=0.8, scale_max=0.99),
                A.GaussianBlur(blur_limit=(3, 7)),
            ], p=0.5),
            
            # Spatial transformations
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1),
            ], p=0.5),
        ], p=1.0)  # Always apply the composition

        return image_transforms

    # --- configure dataset --- #
    def _configure_dataset(self):

        image_transforms = self._compose_augmentations()

        dataset = NapariDataset(mgr=self.mgr,
                                     image_transforms=image_transforms,
                                     volume_transforms=None)

        return dataset

    # --- losses ---- #
    def _build_loss(self):
        # if you override this you need to allow for a loss fn to apply to every single
        # possible target in the dictionary of targets . the easiest is probably
        # to add it in losses.loss, import it here, and then add it to the map
        LOSS_FN_MAP = {
            # Basic losses
            "BCELoss": BCELoss,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "MSELoss": MSELoss,
            "RobustCrossEntropyLoss": RobustCrossEntropyLoss,
            
            # Segmentation-specific losses
            "BCEDiceLoss": BCEDiceLoss,
            "SoftDiceLoss": SoftDiceLoss,
            "GeneralizedDiceLoss": GeneralizedDiceLoss,
            "WeightedCrossEntropyLoss": WeightedCrossEntropyLoss,
            "TopKLoss": TopKLoss,
            
            # Combined losses
            "DC_and_CE_loss": DC_and_CE_loss,
            "DC_and_BCE_loss": DC_and_BCE_loss,
            "DC_and_topk_loss": DC_and_topk_loss,
            
            # Special losses
            "MaskedCosineLoss": MaskedCosineLoss,
            "DC_SkelREC_and_CE_loss": DC_SkelREC_and_CE_loss,
        }

        loss_fns = {}
        for task_name, task_info in self.mgr.targets.items():
            loss_fn = task_info.get("loss_fn", "DC_and_CE_loss")
            if loss_fn not in LOSS_FN_MAP:
                raise ValueError(f"Loss function {loss_fn} not found in LOSS_FN_MAP. Add it to the mapping and try again.")
            loss_kwargs = task_info.get("loss_kwargs", {})
            loss_fns[task_name] = LOSS_FN_MAP[loss_fn](**loss_kwargs)

        return loss_fns

    # --- optimizer ---- #
    def _get_optimizer(self, model):
        from model.optimizers import create_optimizer
        
        # Map ConfigManager params to what create_optimizer expects
        optimizer_config = {
            'name': self.mgr.optimizer,
            'learning_rate': self.mgr.initial_lr,
            'weight_decay': self.mgr.weight_decay
        }
        
        return create_optimizer(optimizer_config, model)

    # --- scheduler --- #
    def _get_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=self.mgr.max_epoch,
                                      eta_min=0)
        return scheduler

    # --- scaler --- #
    def _get_scaler(self, device_type='cuda'):
        if device_type == 'cuda':
            # Only create a GradScaler for CUDA
            return torch.amp.GradScaler()
        else:
            # For MPS/CPU, return a dummy scaler that does nothing
            class DummyScaler:
                def scale(self, loss):
                    return loss
                
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
        
        # Check if we're using MPS device for Apple Silicon
        device_type = 'mps' if hasattr(torch, 'mps') and torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # For MPS device, set num_workers=0 to avoid pickling error
        num_workers = 0 if device_type == 'mps' else self.mgr.train_num_dataloader_workers
        
        train_dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=SubsetRandomSampler(train_indices),
                                pin_memory=(device_type == 'cuda'),  # Only use pin_memory for CUDA
                                num_workers=num_workers)
        val_dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    sampler=SubsetRandomSampler(val_indices),
                                    pin_memory=(device_type == 'cuda'),  # Only use pin_memory for CUDA
                                    num_workers=num_workers)

        return train_dataloader, val_dataloader


    def train(self):

        model = self._build_model()
        optimizer = self._get_optimizer(model)
        loss_fns = self._build_loss()
        dataset = self._configure_dataset()
        scheduler = self._get_scheduler(optimizer)

        # Determine the best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA device")
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
        
        model = model.to(device)
        
        # Only compile the model if it's on CUDA (not supported on MPS/CPU)
        if device.type == 'cuda':
            model = torch.compile(model)

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

        train_dataloader, val_dataloader = self._configure_dataloaders(dataset)

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
                
                # We may need to rebuild the model with the saved configuration
                if model.autoconfigure != checkpoint['model_config'].get('autoconfigure', True):
                    print("Model autoconfiguration differs, rebuilding model from checkpoint config")
                    from model.build_network_from_config import NetworkFromConfig
                    
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
                    model.apply(lambda module: init_weights_he(module, neg_slope=1e-2))
                    model = model.to(device)
                    model = torch.compile(model)
                    
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

        # ---- training! ----- #
        for epoch in range(start_epoch, self.mgr.max_epoch):
            model.train()

            train_running_losses = {t_name: 0.0 for t_name in self.mgr.targets}
            # Always run exactly max_steps_per_epoch iterations, regardless of dataset size
            train_dataloader_iter = iter(train_dataloader)
            pbar = tqdm(range(self.mgr.max_steps_per_epoch), total=self.mgr.max_steps_per_epoch)
            steps = 0

            for i in pbar:
                try:
                    data_dict = next(train_dataloader_iter)
                except StopIteration:
                    # Reset iterator if we run out of data
                    train_dataloader_iter = iter(train_dataloader)
                    data_dict = next(train_dataloader_iter)

                if epoch == 0 and i == 0:
                    for item in data_dict:
                        print(f"Items from the first batch -- Double check that your shapes and values are expected:")
                        print(f"{item}: {data_dict[item].dtype}")
                        print(f"{item}: {data_dict[item].shape}")
                        print(f"{item}: min : {data_dict[item].min()} max : {data_dict[item].max()}")

                global_step += 1

                inputs = data_dict["image"].to(device, dtype=torch.float32)
                targets_dict = {
                    k: v.to(device, dtype=torch.float32)
                    for k, v in data_dict.items()
                    if k != "image"
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

                        # check if skeleton is available in data_dict (e.g. "fibers_skel")
                        skel_key = f"{t_name}_skel"
                        if skel_key in data_dict:
                            t_skel = data_dict[skel_key].to(device, dtype=torch.float32)
                            t_loss = t_loss_fn(t_pred, t_gt, t_skel) * task_weight
                        else:
                            t_loss = t_loss_fn(t_pred, t_gt) * task_weight

                        total_loss += t_loss
                        train_running_losses[t_name] += t_loss.item()
                        per_task_losses[t_name] = t_loss.item()

                # backward
                # loss \ accumulation steps to maintain same effective batch size
                total_loss = total_loss / grad_accumulate_n
                # backward
                scaler.scale(total_loss).backward()

                if (i + 1) % grad_accumulate_n == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                steps += 1

                desc_parts = []
                for t_name in self.mgr.targets:
                    # Avoid division by zero
                    avg_t_loss = train_running_losses[t_name] / steps if steps > 0 else 0
                    desc_parts.append(f"{t_name}: {avg_t_loss:.4f}")

                desc_str = f"Epoch {epoch + 1} => " + " | ".join(desc_parts)
                pbar.set_description(desc_str)

            pbar.close()

            # Apply any remaining gradients at the end of the epoch
            if steps % grad_accumulate_n != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for t_name in self.mgr.targets:
                # Avoid division by zero
                epoch_avg = train_running_losses[t_name] / steps if steps > 0 else 0


            print(f"[Train] Epoch {epoch + 1} completed.")

            # Get model and checkpoint path within the model-specific directory
            ckpt_path = f"{model_ckpt_dir}/{self.mgr.model_name}_{epoch + 1}.pth"
            
            # Save checkpoint with model weights and training state
            # Include the model configuration directly in the checkpoint
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'model_config': model.final_config  # Save the model configuration
            }, ckpt_path)
            
            print(f"Checkpoint saved to: {ckpt_path}")
                    
            # Save the full configuration via the ConfigManager
            self.mgr.save_config()

            # clean up old checkpoints -- currently just keeps 10 newest
            # Path to the model-specific checkpoint directory
            ckpt_dir = Path(model_ckpt_dir)
            all_checkpoints = sorted(
                ckpt_dir.glob(f"{self.mgr.model_name}_*.pth"),
                key=lambda x: x.stat().st_mtime
            )

            # if more than 10, remove the oldest
            while len(all_checkpoints) > 10:
                oldest = all_checkpoints.pop(0)
                oldest.unlink()  #

            # ---- validation ----- #
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_running_losses = {t_name: 0.0 for t_name in self.mgr.targets}
                    val_steps = 0

                    # Always run exactly max_val_steps_per_epoch iterations
                    val_dataloader_iter = iter(val_dataloader)
                    pbar = tqdm(range(self.mgr.max_val_steps_per_epoch), total=self.mgr.max_val_steps_per_epoch)
                    for i in pbar:
                        try:
                            data_dict = next(val_dataloader_iter)
                        except StopIteration:
                            # Reset iterator if we run out of data
                            val_dataloader_iter = iter(val_dataloader)
                            data_dict = next(val_dataloader_iter)

                        inputs = data_dict["image"].to(device, dtype=torch.float32)
                        targets_dict = {
                            k: v.to(device, dtype=torch.float32)
                            for k, v in data_dict.items()
                            if k != "image"
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
                for t_name in self.mgr.targets:
                    # Avoid division by zero
                    val_avg = val_running_losses[t_name] / val_steps if val_steps > 0 else 0
                    print(f"Task '{t_name}', epoch {epoch + 1} avg val loss: {val_avg:.4f}")

            scheduler.step()

        print('Training Finished!')
        
        # Save final model with configuration in the model-specific directory
        final_model_path = f"{model_ckpt_dir}/{self.mgr.model_name}_final.pth"
        
        # Save the complete checkpoint with configuration embedded
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': self.mgr.max_epoch - 1,
            'model_config': model.final_config
        }, final_model_path)
        
        print(f"Final model saved to {final_model_path}")
        print(f"Model configuration is embedded in the checkpoint")
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train script for MultiTaskUnet.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    trainer = BaseTrainer(verbose=args.verbose)
    trainer.train()

# During training, you'll get a dict with all outputs
# outputs = model(input_tensor)
# sheet_pred = outputs['sheet']          # Shape: [B, 1, D, H, W]
# normals_pred = outputs['normals']      # Shape: [B, 3, D, H, W]
# affinities_pred = outputs['affinities']  # Shape: [B, N_affinities, D, H, W]
