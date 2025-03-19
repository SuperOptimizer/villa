import os
import torch
from typing import Union, List, Tuple, Dict, Any
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import nnunetv2
from torch._dynamo import OptimizedModule

# Define the module's public interface
__all__ = ['load_model', 'run_inference']


def load_model(model_folder: str, fold: Union[int, str] = 0, checkpoint_name: str = 'checkpoint_final.pth', 
            device='cuda', custom_plans_json=None, custom_dataset_json=None, use_mirroring: bool = True, 
            verbose: bool = False):
    """
    Load a trained nnUNet model from a model folder.
    
    Args:
        model_folder: Path to the model folder containing plans.json, dataset.json and fold_X folders
        fold: Which fold to load (default: 0, can also be 'all')
        checkpoint_name: Name of the checkpoint file (default: checkpoint_final.pth)
        device: Device to load the model on ('cuda' or 'cpu')
        custom_plans_json: Optional custom plans.json to use instead of the one in model_folder
        custom_dataset_json: Optional custom dataset.json to use instead of the one in model_folder
        use_mirroring: Enable test time augmentation via mirroring (default: True)
        verbose: Enable detailed output messages during loading (default: False)
        
    Returns:
        network: The loaded model
        parameters: The model parameters
    """
    # Load dataset and plans - check if we're in a fold directory
    model_path = model_folder
    if os.path.basename(model_folder).startswith('fold_'):
        # We're inside a fold directory, move up one level
        model_path = os.path.dirname(model_folder)
    
    dataset_json = custom_dataset_json if custom_dataset_json is not None else load_json(join(model_path, 'dataset.json'))
    plans = custom_plans_json if custom_plans_json is not None else load_json(join(model_path, 'plans.json'))
    plans_manager = PlansManager(plans)
    
    # Load checkpoint - handle if we're already in a fold directory or not
    if os.path.basename(model_folder).startswith('fold_'):
        checkpoint_file = join(model_folder, checkpoint_name)
    else:
        checkpoint_file = join(model_folder, f'fold_{fold}', checkpoint_name)
    print(f"Loading checkpoint: {checkpoint_file}")
    try:
        print(f"Attempting to load checkpoint from: {checkpoint_file}")
        try:
            # Try with weights_only=False first (required for PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
            print("Loaded checkpoint with weights_only=False")
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            print("Falling back to loading without weights_only parameter")
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    
    # Get configuration
    configuration_manager = plans_manager.get_configuration(configuration_name)
    
    # Determine input channels and number of output classes
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    
    # Find the trainer class
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                               trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if trainer_class is None:
        raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer.')
    
    # Build the network architecture (without deep supervision for inference)
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False
    )
    
    # Load model parameters
    network_state_dict = checkpoint['network_weights']
    
    # Move to the specified device
    device = torch.device(device)
    network = network.to(device)
    
    # Load the state dict
    if not isinstance(network, OptimizedModule):
        network.load_state_dict(network_state_dict)
    else:
        network._orig_mod.load_state_dict(network_state_dict)
    
    # Set to evaluation mode
    network.eval()
    
    # Compile by default unless explicitly disabled
    should_compile = True
    if 'nnUNet_compile' in os.environ.keys():
        should_compile = os.environ['nnUNet_compile'].lower() in ('true', '1', 't')
    
    if should_compile and not isinstance(network, OptimizedModule):
        print('Using torch.compile for potential performance improvement')
        try:
            network = torch.compile(network)
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
            print("Continuing with uncompiled model")
    
    # Get allowed mirroring axes from checkpoint if available
    inference_allowed_mirroring_axes = None
    if 'inference_allowed_mirroring_axes' in checkpoint.keys():
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes']
    
    # Return useful information for inference
    model_info = {
        'network': network,
        'checkpoint': checkpoint,
        'plans_manager': plans_manager,
        'configuration_manager': configuration_manager,
        'dataset_json': dataset_json,
        'label_manager': label_manager,
        'trainer_name': trainer_name,
        'num_input_channels': num_input_channels,
        'num_seg_heads': label_manager.num_segmentation_heads,
        'patch_size': configuration_manager.patch_size,
        'use_mirroring': use_mirroring,
        'allowed_mirroring_axes': inference_allowed_mirroring_axes,
        'verbose': verbose
    }
    
    return model_info


def run_inference(model_info: Dict[str, Any], input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Run inference with a loaded nnUNet model
    
    Args:
        model_info: Dictionary returned by the load_model function
        input_tensor: Input tensor of shape [B, C, ...] with dimensions matching the model's expected input size
                    (B: batch size, C: channels matching num_input_channels)
                    
    Returns:
        output: Model prediction tensor
    """
    network = model_info['network']
    network.eval()  # Ensure network is in evaluation mode
    
    # Check if input has correct number of channels
    expected_channels = model_info['num_input_channels']
    if input_tensor.shape[1] != expected_channels:
        raise ValueError(f"Expected input tensor with {expected_channels} channels, got {input_tensor.shape[1]}")
    
    # Check if input has correct dimensions (allowing for batch dimension)
    patch_size = model_info['patch_size']
    expected_ndim = len(patch_size) + 2  # +2 for batch and channel dimensions
    if len(input_tensor.shape) != expected_ndim:
        raise ValueError(f"Expected {expected_ndim}-dimensional input, got {len(input_tensor.shape)}")
    
    # Make sure input is on the right device
    device = next(network.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Check if TTA (test time augmentation) should be used
    use_mirroring = model_info.get('use_mirroring', True)  # Default to True (nnUNet default)
    allowed_mirroring_axes = model_info.get('allowed_mirroring_axes', None)
    verbose = model_info.get('verbose', False)  # Default to False
    
    # If TTA is disabled or no mirroring axes are specified, just run standard inference
    if not use_mirroring or allowed_mirroring_axes is None:
        if verbose:
            if not use_mirroring:
                print("Test time augmentation (mirroring) is disabled")
            elif allowed_mirroring_axes is None:
                print("No mirroring axes specified in model checkpoint, test time augmentation not possible")
        with torch.no_grad():
            output = network(input_tensor)
        return output
    
    # Implementation of TTA through mirroring (matches nnUNet's approach)
    with torch.no_grad():
        # Get standard prediction without mirroring
        prediction = network(input_tensor)
        
        # Adjust mirror axes to account for batch and channel dimensions
        mirror_axes = [i + 2 for i in allowed_mirroring_axes]  # +2 for batch and channel dimensions
        if verbose:
            print(f"Using test time augmentation with mirroring axes: {allowed_mirroring_axes}")
        
        # Import itertools for combinations
        import itertools
        
        # Generate all possible combinations of mirroring axes
        axes_combinations = [
            c for i in range(len(mirror_axes)) 
            for c in itertools.combinations(mirror_axes, i + 1)
        ]
        
        # For each combination, mirror input, predict, and mirror prediction back
        for axes in axes_combinations:
            # Mirror input
            mirrored_input = torch.flip(input_tensor, axes)
            # Predict
            mirrored_output = network(mirrored_input)
            # Mirror prediction back to original orientation
            prediction += torch.flip(mirrored_output, axes)
        
        # Average all predictions (original + mirrored variants)
        prediction /= (len(axes_combinations) + 1)
        
    return prediction


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load a trained nnUNet model')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the model folder')
    parser.add_argument('--fold', type=str, default='0', help='Fold to load (default: 0, can also be "all")')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_final.pth', 
                      help='Checkpoint file name (default: checkpoint_final.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load model on (default: cuda)')
    parser.add_argument('--custom_plans', type=str, default=None, 
                      help='Path to custom plans.json file (optional)')
    parser.add_argument('--custom_dataset', type=str, default=None, 
                      help='Path to custom dataset.json file (optional)')
    parser.add_argument('--test_inference', action='store_true', 
                      help='Run a test inference with random input')
    parser.add_argument('--disable_tta', action='store_true',
                      help='Disable test time augmentation (mirroring) for faster inference')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable detailed output messages during loading and inference')
    
    args = parser.parse_args()
    
    # Load custom JSON files if specified
    custom_plans_json = load_json(args.custom_plans) if args.custom_plans else None
    custom_dataset_json = load_json(args.custom_dataset) if args.custom_dataset else None
    
    # Load the model
    model_info = load_model(
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        device=args.device,
        custom_plans_json=custom_plans_json,
        custom_dataset_json=custom_dataset_json,
        use_mirroring=not args.disable_tta,
        verbose=args.verbose
    )
    
    # Print model information
    network = model_info['network']
    checkpoint = model_info['checkpoint']
    print("\nModel loaded successfully!")
    print(f"Trainer: {model_info['trainer_name']}")
    print(f"Configuration: {checkpoint['init_args']['configuration']}")
    print(f"Model type: {type(network).__name__}")
    print(f"Model is on device: {next(network.parameters()).device}")
    print(f"Input channels: {model_info['num_input_channels']}")
    print(f"Output segmentation heads: {model_info['num_seg_heads']}")
    print(f"Expected patch size: {model_info['patch_size']}")
    
    # Show TTA status if verbose
    if args.verbose:
        use_mirroring = model_info.get('use_mirroring', True)
        mirroring_axes = model_info.get('allowed_mirroring_axes', None)
        if use_mirroring and mirroring_axes is not None:
            print(f"Test time augmentation: Enabled with mirroring axes {mirroring_axes}")
        else:
            print(f"Test time augmentation: {'Disabled by user' if not use_mirroring else 'Not available (no mirroring axes in checkpoint)'}")
    
    # Run a test inference if requested
    if args.test_inference:
        print("\nRunning test inference with random input...")
        patch_size = model_info['patch_size']
        input_channels = model_info['num_input_channels']
        
        # Create a random input batch with the right dimensions
        if len(patch_size) == 2:  # 2D model
            dummy_input = torch.randn(1, input_channels, *patch_size, device=args.device)
        else:  # 3D model
            dummy_input = torch.randn(1, input_channels, *patch_size, device=args.device)
        
        # Run inference
        with torch.no_grad():
            try:
                output = network(dummy_input)
                print(f"Test inference successful!")
                print(f"Input shape: {dummy_input.shape}")
                print(f"Output shape: {output.shape}")
                del dummy_input, output
                torch.cuda.empty_cache()  # Clean up GPU memory
            except Exception as e:
                print(f"Test inference failed with error: {e}")