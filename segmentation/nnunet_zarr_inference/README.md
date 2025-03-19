# nnUNet Zarr Inference

This package provides utilities for running inference with nnUNet models on zarr array data.

## Features

- Load trained nnUNet models (any valid nnUNet checkpoint)
- Efficient sliding window inference on large 3D zarr arrays
- Multi-GPU support with DDP (Distributed Data Parallel)
- Asynchronous disk writing to minimize I/O bottlenecks
- Automatic overlapping patch blending with Gaussian windows

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
python inference.py \
  --input_path /path/to/input.zarr \
  --output_path /path/to/output \
  --model_folder /path/to/nnunet/model \
  --fold 0 \
  --batch_size 4 \
  --overlap 0.25
```

### Multi-GPU Inference

```bash
torchrun --nproc_per_node=2 inference.py \
  --input_path /path/to/input.zarr \
  --output_path /path/to/output \
  --model_folder /path/to/nnunet/model \
  --fold 0 \
  --batch_size 4
```

### Arguments

- `--input_path`: Path to the input zarr array
- `--output_path`: Path to save the output predictions
- `--model_folder`: Path to the nnUNet model folder
- `--fold`: Fold to use for inference (default: 0)
- `--checkpoint`: Checkpoint file name to use (default: checkpoint_final.pth)
- `--batch_size`: Batch size for inference (default: 4)
- `--overlap`: Overlap between patches as a fraction (default: 0.25)
- `--num_dataloader_workers`: Number of workers for the DataLoader (default: 4)
- `--num_write_workers`: Number of worker threads for asynchronous disk writes (default: 4)
- `--device`: Device to run inference on ('cuda' or 'cpu') (default: cuda)
- `--write_layers`: Write the sliced z layers to disk
- `--postprocess_only`: Skip the inference pass and only do final averaging + casting
- `--load_all`: Load the entire input array into memory (use with caution!)

## API Usage

You can also use the Python API directly:

```python
from inference import ZarrNNUNetInferenceHandler

# Initialize the inference handler
inference_handler = ZarrNNUNetInferenceHandler(
    input_path="/path/to/input.zarr",
    output_path="/path/to/output",
    model_folder="/path/to/nnunet/model",
    fold=0,
    batch_size=4,
    overlap=0.25,
    num_dataloader_workers=4,
    num_write_workers=4
)

# Run inference
inference_handler.infer()
```

## Output

The output will be saved as a zarr array with the following datasets:

- `{target_name}_sum`: Sum of all predictions
- `{target_name}_count`: Count of predictions per voxel
- `{target_name}_final`: Final averaged predictions (uint8 or uint16)

The default target name is "segmentation", but this can be customized using the output_targets parameter.