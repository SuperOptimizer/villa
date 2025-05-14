import torch
import os
import sys
from pathlib import Path
from model.build_network_from_config import NetworkFromConfig
from types import SimpleNamespace
import numpy as np

def test_model_loading(checkpoint_path):
    print(f"Testing model loading from checkpoint: {checkpoint_path}")
    
    # Load the checkpoint
    device = torch.device("cpu")  # Use CPU to avoid GPU issues
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if model_config exists in the checkpoint
    if 'model_config' not in checkpoint:
        print("ERROR: No model_config found in checkpoint")
        return False
    
    # Get model configuration from the checkpoint
    model_config = checkpoint['model_config']
    print(f"Found model_config in checkpoint with keys: {list(model_config.keys())}")
    
    # Create a simple config manager using the model_config
    config_mgr = SimpleNamespace(
        model_config=model_config,
        inference_config=model_config,
        train_patch_size=model_config.get("train_patch_size", model_config.get("patch_size", [64, 64, 64])),
        train_batch_size=model_config.get("batch_size", 2),
        in_channels=model_config.get("in_channels", 1),
        spacing=[1, 1, 1],
        targets=model_config.get("targets", {"default": {"out_channels": 1, "loss_fn": "BCEDiceLoss"}}),
        autoconfigure=model_config.get("autoconfigure", True),
        model_name=model_config.get("model_name", Path(checkpoint_path).stem.split('_')[0]),
        tr_configs={},
        tr_info={"model_name": model_config.get("model_name", "model")},
        dataset_config={}
    )
    
    # Try to instantiate the model
    try:
        print("Creating model from config...")
        model = NetworkFromConfig(config_mgr)
        
        # Load the model state dict to verify compatibility
        print("Loading model weights...")
        model.load_state_dict(checkpoint['model'])
        
        # Put model in eval mode
        model.eval()
        
        # Create a random input tensor to test forward pass
        input_shape = [1, model_config.get("in_channels", 1)]
        input_shape.extend(model_config.get("patch_size", [64, 64]))
        
        print(f"Creating test input with shape: {input_shape}")
        test_input = torch.randn(*input_shape)
        
        # Try forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            outputs = model(test_input)
        
        # Check outputs
        print(f"Forward pass successful! Output keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        
        print("Model loading and instantiation successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to create or run model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    checkpoint_path = "checkpoints/test_1638/test_1638_1.pth"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    test_model_loading(checkpoint_path)
