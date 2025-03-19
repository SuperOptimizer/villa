"""
Example script demonstrating how to use the nnUNet zarr inference module.
"""
import os
import argparse
from inference import ZarrNNUNetInferenceHandler

def run_example(args):
    """
    Run nnUNet inference on a zarr array using the ZarrNNUNetInferenceHandler.
    """
    # Define custom output targets if needed
    # This is optional and if not provided, a default single-channel segmentation target will be used
    output_targets = None
    if args.multi_class:
        # Example multi-class target for semantic segmentation
        output_targets = {
            "segmentation": {
                "channels": args.num_classes,  # Number of output classes (including background)
                "activation": "softmax"  # Apply softmax activation
            }
        }
    elif args.multi_target:
        # Example multi-target output for a model that predicts both segmentation and normals
        output_targets = {
            "segmentation": {
                "channels": 1,
                "activation": "sigmoid"  # Apply sigmoid activation for binary segmentation
            },
            "normals": {
                "channels": 3,  # 3 channels for XYZ normal vectors
                "activation": "none"  # No activation for normals
            }
        }

    # Initialize the inference handler
    inference_handler = ZarrNNUNetInferenceHandler(
        input_path=args.input_path,
        output_path=args.output_path,
        model_folder=args.model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        overlap=args.overlap,
        num_dataloader_workers=args.num_workers,
        num_write_workers=args.num_write_workers,
        device=args.device,
        output_targets=output_targets,
        load_all=args.load_all
    )

    # Run inference
    inference_handler.infer()
    
    print(f"Inference completed. Results saved to {args.output_path}/predictions.zarr")
    print(f"Available datasets: ")
    for tgt_name in output_targets or {"segmentation": {"channels": 1}}:
        print(f"  - {tgt_name}_sum: Raw sum of predictions")
        print(f"  - {tgt_name}_count: Count of predictions per voxel")
        print(f"  - {tgt_name}_final: Final averaged and normalized predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for nnUNet zarr inference")
    
    # Required arguments
    parser.add_argument("--input_path", type=str, required=True, 
                        help="Path to the input zarr array")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the output predictions")
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Path to the nnUNet model folder")
    
    # Optional arguments
    parser.add_argument("--fold", type=str, default="0",
                        help="Fold to use for inference (default: 0)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth",
                        help="Checkpoint file name to use (default: checkpoint_final.pth)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference (default: 4)")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Overlap between patches as a fraction (default: 0.25)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for the DataLoader (default: 4)")
    parser.add_argument("--num_write_workers", type=int, default=4,
                        help="Number of worker threads for asynchronous disk writes (default: 4)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on ('cuda' or 'cpu') (default: cuda)")
    parser.add_argument("--load_all", action="store_true",
                        help="Load the entire input array into memory (use with caution!)")
                        
    # Output target configuration options
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--multi_class", action="store_true",
                             help="Use multi-class segmentation output target")
    target_group.add_argument("--multi_target", action="store_true",
                             help="Use multiple output targets (segmentation + normals)")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes for multi-class segmentation (default: 2)")
    
    args = parser.parse_args()
    run_example(args)