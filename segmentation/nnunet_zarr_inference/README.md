# nnUNet Zarr Inference

This package provides utilities for running inference with nnUNet models on zarr array data.

## Features

- Load trained nnUNet models (any valid nnUNet checkpoint)
- Efficient sliding window inference on large 3D zarr arrays
- Multi-GPU support with DDP (Distributed Data Parallel)
- Automatic model compilation for improved performance
- Asynchronous disk writing to minimize I/O bottlenecks
- Automatic overlapping patch blending with Gaussian windows
- Test time augmentation support with mirroring (like nnUNet)
- Thresholding to generate binary segmentation masks
- Automatic cleanup of intermediate arrays

## Requirements

- PyTorch (1.10+)
- nnUNetV2
- zarr
- numcodecs
- tqdm
- numpy

## Usage

### Basic Usage

```bash
python -m nnunet_zarr_inference.inference \
  --input_path /path/to/input.zarr \
  --output_path /path/to/output \
  --model_folder /path/to/nnunet/model \
  --fold 0 \
  --batch_size 4 \
  --step_size 0.5
```

### With Thresholding

```bash
python -m nnunet_zarr_inference.inference \
  --input_path /path/to/input.zarr \
  --output_path /path/to/output \
  --model_folder /path/to/nnunet/model \
  --fold 0 \
  --threshold 50  # 50% confidence threshold
```

### Multi-GPU Inference with DDP

```bash
# Using torchrun (recommended for PyTorch 1.10+)
torchrun --nproc_per_node=2 -m nnunet_zarr_inference.inference \
  --input_path /path/to/input.zarr \
  --output_path /path/to/output \
  --model_folder /path/to/nnunet/model \
  --fold 0 \
  --batch_size 4 \
  --num_dataloader_workers 8  # Will be automatically divided by number of processes

# For better performance, set OMP_NUM_THREADS
export OMP_NUM_THREADS=4  # Total CPU cores / Number of processes
torchrun --nproc_per_node=2 -m nnunet_zarr_inference.inference \
  --input_path /path/to/input.zarr \
  --output_path /path/to/output \
  --model_folder /path/to/nnunet/model \
  --fold 0
```

When using DDP, worker counts are automatically adjusted:
- You specify the **total** number of workers you want across all processes
- The script automatically divides this by the number of processes
- This ensures efficient CPU utilization without oversubscription

### Arguments

- `--input_path`: Path to the input zarr array
- `--output_path`: Path to save the output predictions
- `--model_folder`: Path to the nnUNet model folder
- `--fold`: Fold to use for inference (default: 0)
- `--checkpoint`: Checkpoint file name to use (default: checkpoint_final.pth)
- `--batch_size`: Batch size for inference (default: 4)
- `--step_size`: Step size for sliding window as a fraction of patch size (default: 0.5, nnUNet default)
- `--num_dataloader_workers`: Number of workers for the DataLoader (default: 4). When using DDP, this is the total worker count across all processes.
- `--num_write_workers`: Number of worker threads for asynchronous disk writes (default: 4). When using DDP, this is the total worker count across all processes.
- `--device`: Device to run inference on ('cuda' or 'cpu') (default: cuda)
- `--threshold`: Apply threshold to probability map (value 0-100, represents percentage)
- `--disable_tta`: Disable test time augmentation (mirroring) for faster but potentially less accurate inference
- `--verbose`: Enable detailed output messages during inference
- `--keep_intermediates`: Keep intermediate sum and count arrays after processing
- `--write_layers`: Write the sliced z layers to disk
- `--postprocess_only`: Skip the inference pass and only do final averaging + casting
- `--load_all`: Load the entire input array into memory (use with caution!)

## API Usage

You can also use the Python API directly:

```python
from nnunet_zarr_inference.inference import ZarrNNUNetInferenceHandler

# Initialize the inference handler
inference_handler = ZarrNNUNetInferenceHandler(
    input_path="/path/to/input.zarr",
    output_path="/path/to/output",
    model_folder="/path/to/nnunet/model",
    fold=0,
    batch_size=4,
    step_size=0.5,  # Controls sliding window overlap (nnUNet default)
    threshold=50,   # Optional: Apply 50% threshold for binary segmentation
    use_mirroring=True,  # Use test time augmentation (default)
    verbose=False,  # Minimal console output
    keep_intermediates=False,  # Clean up intermediates (default)
    num_dataloader_workers=4,
    num_write_workers=4
)

# Run inference
inference_handler.infer()
```

## Output

By default, the output will be saved as a zarr array with the following datasets:

- `segmentation_probabilities`: Probability maps scaled to 0-255 (uint8)
- `segmentation_threshold`: Binary segmentation if a threshold was specified (uint8)

If `--keep_intermediates` is specified, these additional arrays will be kept:
- `segmentation_sum`: Sum of all predictions (float32)
- `segmentation_count`: Count of predictions per voxel (float32)

The default target name is "segmentation", but this can be customized using the output_targets parameter.

## Environment Variables

- `OMP_NUM_THREADS`: Controls the number of OpenMP threads per process. Recommended to set this to (Total CPU cores / Number of processes).
- `nnUNet_compile`: Controls model compilation (default is enabled). Set to 'false' to disable compilation if you encounter issues.