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
            verbose: bool = False, rank: int = 0):
    """
    MODIFIED version to add tracing for slow loading
    """
    # Only print from rank 0 by default
    if rank == 0:
        print(f"Starting load_model for {model_folder}, fold={fold}, device={device}")
    import time
    start_time = time.time()
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
        rank: Distributed rank of the process (default: 0, used to suppress output from non-rank-0 processes)
        
    Returns:
        network: The loaded model
        parameters: The model parameters
    """
    # Load dataset and plans - check if we're in a fold directory
    model_path = model_folder
    if os.path.basename(model_folder).startswith('fold_'):
        # We're inside a fold directory, move up one level
        model_path = os.path.dirname(model_folder)
    
    # Check for dataset.json and plans.json
    dataset_json_path = join(model_path, 'dataset.json')
    plans_json_path = join(model_path, 'plans.json')
    
    if custom_dataset_json is None and not os.path.exists(dataset_json_path):
        error_msg = f"ERROR: dataset.json not found at: {dataset_json_path}\n"
        error_msg += f"\nThis file is required for nnUNet model loading.\n"
        if os.path.isdir(model_path):
            error_msg += f"Contents of model directory ({model_path}):\n"
            error_msg += f"  {', '.join(os.listdir(model_path))}\n"
        raise FileNotFoundError(error_msg)
        
    if custom_plans_json is None and not os.path.exists(plans_json_path):
        error_msg = f"ERROR: plans.json not found at: {plans_json_path}\n"
        error_msg += f"\nThis file is required for nnUNet model loading.\n"
        if os.path.isdir(model_path):
            error_msg += f"Contents of model directory ({model_path}):\n"
            error_msg += f"  {', '.join(os.listdir(model_path))}\n"
        raise FileNotFoundError(error_msg)
    
    # Load the JSON files
    try:
        dataset_json = custom_dataset_json if custom_dataset_json is not None else load_json(dataset_json_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset.json: {str(e)}")
        
    try:
        plans = custom_plans_json if custom_plans_json is not None else load_json(plans_json_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load plans.json: {str(e)}")
        
    try:
        plans_manager = PlansManager(plans)
    except Exception as e:
        raise RuntimeError(f"Failed to create PlansManager: {str(e)}")
    
    # Load checkpoint - handle if we're already in a fold directory or not
    if os.path.basename(model_folder).startswith('fold_'):
        checkpoint_file = join(model_folder, checkpoint_name)
    else:
        checkpoint_file = join(model_folder, f'fold_{fold}', checkpoint_name)
    
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_file):
        # List available folds and checkpoints to help the user
        available_folds = []
        if os.path.isdir(model_folder):
            for item in os.listdir(model_folder):
                if item.startswith('fold_') and os.path.isdir(join(model_folder, item)):
                    available_folds.append(item)
                    
        error_msg = f"ERROR: Checkpoint file not found: {checkpoint_file}\n"
        if available_folds:
            error_msg += "\nAvailable folds in this model folder:\n"
            for fold_dir in available_folds:
                fold_path = join(model_folder, fold_dir)
                checkpoints = [f for f in os.listdir(fold_path) if f.endswith('.pth')]
                if checkpoints:
                    error_msg += f"  - {fold_dir}: {', '.join(checkpoints)}\n"
                else:
                    error_msg += f"  - {fold_dir}: No checkpoint files found\n"
        else:
            error_msg += f"\nThe model folder does not contain any 'fold_X' subdirectories.\n"
            if os.path.isdir(model_folder):
                error_msg += f"Contents of {model_folder}:\n"
                error_msg += f"  {', '.join(os.listdir(model_folder))}\n"
            else:
                error_msg += f"The model folder does not exist or is not accessible: {model_folder}\n"
                
        error_msg += "\nPlease check:\n"
        error_msg += "1. The model_folder path is correct\n"
        error_msg += "2. The fold number is correct\n"
        error_msg += "3. The checkpoint_name is correct\n"
        
        raise FileNotFoundError(error_msg)
        
    if rank == 0:  # Only print from rank 0
        print(f"Loading checkpoint: {checkpoint_file}")
    try:
        if rank == 0:
            print(f"Attempting to load checkpoint from: {checkpoint_file}")
        try:
            # Try with weights_only=False first (required for PyTorch 2.6+)
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
            if rank == 0:
                print("Loaded checkpoint with weights_only=False")
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            if rank == 0:
                print("Falling back to loading without weights_only parameter")
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading checkpoint (rank {rank}): {str(e)}")
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
        if rank == 0:
            print('Using torch.compile for potential performance improvement')
        try:
            network = torch.compile(network)
        except Exception as e:
            if rank == 0:
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


def run_inference(model_info: Dict[str, Any], input_tensor: torch.Tensor, 
                max_tta_combinations: int = None,
                parallel_tta_multiplier: int = None,
                rank: int = 0) -> torch.Tensor:
    """
    Run inference with a loaded nnUNet model
    
    Args:
        model_info: Dictionary returned by the load_model function
        input_tensor: Input tensor of shape [B, C, ...] with dimensions matching the model's expected input size
                    (B: batch size, C: channels matching num_input_channels)
        max_tta_combinations: If set, limits which TTA combinations to use
                    (None = use all combinations, primarily affects accuracy)
        parallel_tta_multiplier: If set, controls how many combinations to process in parallel
                    (primarily affects performance/memory usage)
                    
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
        if verbose and rank == 0:
            if not use_mirroring:
                print("Test time augmentation (mirroring) is disabled")
            elif allowed_mirroring_axes is None:
                print("No mirroring axes specified in model checkpoint, test time augmentation not possible")
        with torch.no_grad():
            output = network(input_tensor)
        return output
    
    # Print TTA performance optimization information if verbose
    if verbose and rank == 0:
        # Check if we're using the optimized batch TTA
        batch_size = input_tensor.shape[0]
        using_batch_taa = batch_size <= 2 and torch.cuda.is_available()
        
        if max_tta_combinations is not None:
            if max_tta_combinations == 3:
                print(f"Using TTA with only the 3 primary axis flips (memory-efficient)")
            else:
                print(f"Using TTA with {max_tta_combinations} combinations")
        else:
            print(f"Using TTA with all possible combinations (may be memory-intensive)")
            
        if using_batch_taa:
            print(f"Batch size {batch_size} allows for parallel TTA processing")
        else:
            print(f"Using sequential TTA processing due to batch size {batch_size}")
            
        # Memory info if available
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            mem_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            print(f"GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    
    # Implementation of TTA through mirroring (matches nnUNet's approach but optimized)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # Adjust mirror axes to account for batch and channel dimensions
        mirror_axes = [i + 2 for i in allowed_mirroring_axes]  # +2 for batch and channel dimensions
        if verbose and rank == 0:
            print(f"Using test time augmentation with mirroring axes: {allowed_mirroring_axes}")
        
        # Import itertools for combinations
        import itertools
        
        # Generate all possible combinations of mirroring axes
        axes_combinations = [
            c for i in range(len(mirror_axes)) 
            for c in itertools.combinations(mirror_axes, i + 1)
        ]
        
        # Store original combinations count for more informative messages
        original_combinations_count = len(axes_combinations)
        
        # Limit the number of combinations if specified
        if max_tta_combinations is not None and original_combinations_count > max_tta_combinations:
            if verbose and rank == 0:
                print(f"Limiting TTA combinations from {original_combinations_count} to {max_tta_combinations}")
            
            # Special case for exactly 3 combinations - ALWAYS use the 3 single-axis flips
            if max_tta_combinations == 3:
                # Get the single-axis flips - these are the primary axes we always want
                single_axis_flips = [c for c in axes_combinations if len(c) == 1]
                
                if len(single_axis_flips) == 3:
                    # Found all 3 primary axis flips - use these
                    axes_combinations = single_axis_flips
                    
                    if verbose and rank == 0:
                        print(f"  - Using the 3 primary axis flips (most important for view coverage)")
                else:
                    # Something unexpected happened - the single axis flips aren't exactly 3
                    # Fall back to standard prioritization
                    axes_combinations.sort(key=len)  # Prioritize by number of flipped axes (fewer = better)
                    axes_combinations = axes_combinations[:max_tta_combinations]
                    
                    if verbose and rank == 0:
                        print(f"  - WARNING: Could not identify exactly 3 single-axis flips, using first {max_tta_combinations}")
            else:
                # For other counts, prioritize single-axis flips first, then others
                single_axis_flips = [c for c in axes_combinations if len(c) == 1]
                other_flips = [c for c in axes_combinations if len(c) > 1]
                
                # Sort the remaining flips by length (prioritize 2-axis over 3-axis flips)
                other_flips.sort(key=len)
                
                # Determine how many additional flips we can include beyond the primary ones
                num_primary = len(single_axis_flips)
                
                # Make sure we include at least the single-axis flips if possible
                if max_tta_combinations >= num_primary:
                    # Include all single-axis flips + as many others as will fit
                    remaining_slots = max_tta_combinations - num_primary
                    axes_combinations = single_axis_flips + other_flips[:remaining_slots]
                    
                    if verbose and rank == 0:
                        print(f"  - Included all {num_primary} single-axis flips + {remaining_slots} additional flips")
                else:
                    # Not enough slots even for all single-axis flips
                    # For medical data, still prioritize by axes length to get primary views
                    axes_combinations.sort(key=len)  # Prioritize by number of flipped axes
                    axes_combinations = axes_combinations[:max_tta_combinations]
                    
                    if verbose and rank == 0:
                        print(f"  - WARNING: Not enough capacity for all single-axis flips, using first {max_tta_combinations}")
        
        # Calculate total number of predictions for averaging
        num_predictions = len(axes_combinations) + 1
        
        # Get standard prediction without mirroring
        prediction = network(input_tensor)
        
        # Determine the batch processing strategy based on parallel_tta_multiplier
        # This controls HOW we process combinations (in parallel or sequentially)
        use_batched_tta = False
        if parallel_tta_multiplier is not None and parallel_tta_multiplier > 1 and torch.cuda.is_available():
            use_batched_tta = True
            max_batch_multiplier = parallel_tta_multiplier
            
            if verbose and rank == 0:
                print(f"Using parallel TTA processing with multiplier: {max_batch_multiplier}")
        elif verbose and rank == 0:
            print(f"Using sequential TTA processing")
            
        if use_batched_tta:
            # Process TTA combinations in batches to maximize GPU utilization
            combo_batches = [axes_combinations[i:i+max_batch_multiplier] 
                            for i in range(0, len(axes_combinations), max_batch_multiplier)]
            
            for combo_batch in combo_batches:
                # Process multiple mirror combinations in parallel
                batch_inputs = []
                for axes in combo_batch:
                    batch_inputs.append(torch.flip(input_tensor, axes))
                
                if batch_inputs:  # Make sure we have some inputs
                    # Concatenate along batch dimension
                    batched_input = torch.cat(batch_inputs, dim=0)
                    
                    # Single forward pass for multiple mirror combinations
                    batched_output = network(batched_input)
                    
                    # Split and process the results
                    split_outputs = torch.split(batched_output, input_tensor.shape[0])
                    
                    for i, axes in enumerate(combo_batch):
                        # Flip each output back to original orientation and add to prediction
                        prediction += torch.flip(split_outputs[i], axes)
        else:
            # Standard sequential processing of TTA combinations
            if verbose and torch.cuda.is_available() and rank == 0:
                # Use the actual number of combinations we're using, not the original count
                print(f"Using sequential TTA processing for {len(axes_combinations)} combinations")
                
            for axes in axes_combinations:
                # Mirror input
                mirrored_input = torch.flip(input_tensor, axes)
                # Predict
                mirrored_output = network(mirrored_input)
                # Mirror prediction back to original orientation
                prediction += torch.flip(mirrored_output, axes)
        
        # Average all predictions (original + mirrored variants)
        prediction /= num_predictions
        
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