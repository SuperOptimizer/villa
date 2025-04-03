"""
Custom wrapper for the inference script using our own NetworkFromConfig model.
This script serves as an adapter between our model loading system and the inference system.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Union

from models.run.inference import ZarrInferer
from models.run.alts import load_vesuvius_model_for_inference
from data.vc_dataset import VCDataset


def run_inference(
    config_file: str,
    input_path: str,
    output_path: str,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    patch_size: Optional[tuple] = None,
    use_mirroring: bool = True,
    threshold: Optional[float] = None,
    batch_size: int = 1,
    step_size: float = 0.5,
    verbose: bool = False,
    num_workers: int = 4,
    rank: int = 0
):
    """
    Run inference using our custom NetworkFromConfig model.
    
    Args:
        config_file: Path to configuration file
        input_path: Path to input data
        output_path: Path to save output
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on ('cuda' or 'cpu')
        patch_size: Optional override for patch size
        use_mirroring: Enable test time augmentation via mirroring
        threshold: Optional threshold for binarization
        batch_size: Batch size for inference
        step_size: Step size for sliding window as fraction of patch size
        verbose: Enable detailed output messages
        num_workers: Number of parallel workers
        rank: Distributed rank
    """
    # Load model using our custom loader
    model_info = load_vesuvius_model_for_inference(
        config_file=config_file,
        checkpoint_path=checkpoint_path,
        device_str=device,
        use_mirroring=use_mirroring,
        patch_size=patch_size,
        verbose=verbose,
        rank=rank
    )
    
    # Get output channels from model
    num_output_channels = model_info.get('num_seg_heads', 2)
    
    # Configure targets for the dataset and inferer
    # The target name should match what's in the model
    if hasattr(model_info['network'], 'targets') and model_info['network'].targets:
        # Use the targets from the model
        targets = []
        for target_name, target_info in model_info['network'].targets.items():
            targets.append({
                "name": target_name,
                "channels": target_info.get("out_channels", 2),
                "activation": target_info.get("activation", "softmax" if target_info.get("out_channels", 2) > 1 else "sigmoid"),
                "nnunet_output_channels": target_info.get("out_channels", 2)
            })
    else:
        # Default target configuration (binary segmentation)
        targets = [{
            "name": "segmentation",
            "channels": num_output_channels,
            "activation": "softmax" if num_output_channels > 1 else "sigmoid",
            "nnunet_output_channels": num_output_channels
        }]
    
    # Configure the dataset
    dataset = VCDataset(
        input_path=input_path,
        targets=targets,
        patch_size=model_info['patch_size'],
        num_input_channels=model_info['num_input_channels'],
        step_size=step_size,
        verbose=verbose,
        mode='infer'
    )
    
    # Configure the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Prepare output path
    if not output_path.endswith('.zarr'):
        output_path = f"{output_path}.zarr"
    
    # Create the inferer
    inferer = ZarrInferer(
        input_path=input_path,
        output_path=output_path,
        model_info=model_info,
        dataset=dataset,
        dataloader=dataloader,
        patch_size=model_info['patch_size'],
        batch_size=batch_size,
        step_size=step_size,
        num_write_workers=num_workers,
        threshold=threshold,
        use_mirroring=use_mirroring,
        verbose=verbose,
        save_probability_maps=True,
        output_targets=targets,
        rank=rank
    )
    
    # Run inference
    inferer.infer()


if __name__ == "__main__":
    import torch
    
    parser = argparse.ArgumentParser(description="Run inference using a custom NetworkFromConfig model")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--output", required=True, help="Path to save output")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to run inference on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--step_size", type=float, default=0.5, help="Step size for sliding window")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for binarization")
    parser.add_argument("--no_mirroring", action="store_true", help="Disable test time augmentation")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed output messages")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Configure optional patch size
    patch_size = None  # Use default from model configuration
    
    # Run inference
    run_inference(
        config_file=args.config,
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        device=args.device,
        patch_size=patch_size,
        use_mirroring=not args.no_mirroring,
        threshold=args.threshold,
        batch_size=args.batch_size,
        step_size=args.step_size,
        verbose=args.verbose,
        num_workers=args.num_workers
    )