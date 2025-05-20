import torch
import numpy as np
import zarr
import os
import json
import multiprocessing
import threading
import fsspec
from concurrent.futures import ThreadPoolExecutor
# fork causes issues on windows and w/ tensorstore , force to spawn
multiprocessing.set_start_method('spawn', force=True)
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.models.load_nnunet_model import load_model_for_inference
from data.vc_dataset import VCDataset

class Inferer():
    def __init__(self,
                 model_path: str = None,
                 input_dir: str = None,
                 output_dir: str = None,
                 input_format: str = 'zarr',
                 tta_type: str = 'mirroring', # 'mirroring' or 'rotation'
                 # tta_combinations: int = 3,
                 # tta_rotation_weights: [list, tuple] = (1, 1, 1),
                 do_tta: bool = True,
                 num_parts: int = 1,
                 part_id: int = 0,
                 overlap: float = 0.5,
                 batch_size: int = 1,
                 patch_size: [list, tuple] = None,
                 save_softmax: bool = False,
                 normalization_scheme: str = 'instance_zscore',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_dataloader_workers: int = 4,
                 verbose: bool = False,
                 skip_empty_patches: bool = True,  # Skip empty/homogeneous patches
                 # parmas to get passed to Volume 
                 scroll_id: [str, int] = None,
                 segment_id: [str, int] = None,
                 energy: int = None,
                 resolution: float = None,
                 compressor_name: str = 'zstd',
                 compression_level: int = 1,
                 hf_token: str = None
                 ):

        self.model_path = model_path
        self.input = input_dir
        self.do_tta = do_tta
        self.tta_type = tta_type
        # self.tta_combinations = tta_combinations
        # self.tta_rotation_weights = tta_rotation_weights
        self.num_parts = num_parts
        self.part_id = part_id
        self.overlap = overlap
        self.batch_size = batch_size
        self.patch_size = tuple(patch_size) if patch_size is not None else None  # Can be None, will derive from model
        self.save_softmax = save_softmax
        self.verbose = verbose
        self.normalization_scheme = normalization_scheme
        self.input_format = input_format
        self.device = torch.device(device)
        self.num_dataloader_workers = num_dataloader_workers
        self.skip_empty_patches = skip_empty_patches
        self.scroll_id = scroll_id
        self.segment_id = segment_id
        self.energy = energy
        self.resolution = resolution
        self.compressor_name = compressor_name
        self.compression_level = compression_level
        self.hf_token = hf_token
        self.model_patch_size = None
        self.num_classes = None

        # --- Validation ---
        if not self.input or self.model_path is None:
            raise ValueError("Input directory and model path must be provided.")
        if self.num_parts > 1:
            if self.part_id < 0 or self.part_id >= self.num_parts:
                raise ValueError(f"Invalid part_id {self.part_id} for num_parts {self.num_parts}.")
        if self.overlap < 0 or self.overlap > 1:
            raise ValueError(f"Invalid overlap value {self.overlap}. Must be between 0 and 1.")
        if self.tta_type not in ['mirroring', 'rotation']:
             raise ValueError(f"Invalid tta_type '{self.tta_type}'. Must be 'mirroring' or 'rotation'.")
        # Defer patch size validation until after model loading if not explicitly provided
        if self.patch_size is not None and self.tta_type == 'rotation':
            if len(self.patch_size) != 3:
                raise ValueError(f"Rotation TTA requires 3D patch size, got {self.patch_size}.")

        # --- Output Setup ---
        self._temp_dir_obj = None
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            raise ValueError("Output directory must be provided.")

        # --- Placeholders ---
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.output_store = None
        self.num_classes = None
        self.num_total_patches = None
        self.current_patch_write_index = 0


    def _load_model(self):
        # check if model_path is a Hugging Face model path (starts with "hf://")
        if isinstance(self.model_path, str) and self.model_path.startswith("hf://"):
            hf_model_path = self.model_path.replace("hf://", "")
            if self.verbose:
                print(f"Loading model from Hugging Face repo: {hf_model_path}")
            model_info = load_model_for_inference(
                model_folder=None,
                hf_model_path=hf_model_path,
                hf_token=self.hf_token if hasattr(self, 'hf_token') else None,
                device_str=str(self.device),
                verbose=self.verbose
            )
        else:
            # Load from local path
            if self.verbose:
                print(f"Loading model from local path: {self.model_path}")
            model_info = load_model_for_inference(
                model_folder=self.model_path,
                device_str=str(self.device),
                verbose=self.verbose
            )
        
        # model loader returns a dict, network is the actual model
        model = model_info['network']
        model.eval()
        
        # patch size and number of classes from model_info
        self.model_patch_size = tuple(model_info.get('patch_size', (192, 192, 192)))
        self.num_classes = model_info.get('num_seg_heads', None)
        
        # use models patch size if one wasn't specified
        if self.patch_size is None:
            self.patch_size = self.model_patch_size
            if self.verbose:
                print(f"Using model's patch size: {self.patch_size}")
        else:
            if self.verbose and self.patch_size != self.model_patch_size:
                print(f"Warning: Using user-provided patch size {self.patch_size} instead of model's default: {self.model_patch_size}")
        
        # Validate patch size for rotation TTA if needed
        if self.patch_size is not None and self.tta_type == 'rotation':
            if len(self.patch_size) != 3:
                raise ValueError(f"Rotation TTA requires 3D patch size, got {self.patch_size}.")
        
        # Confirm num_classes if it couldn't be determined from model_info
        if self.num_classes is None:
            if self.verbose:
                print("Number of classes not found in model_info, performing dummy inference...")
            
            # Determine input channels from model_info if possible
            input_channels = model_info.get('num_input_channels', 1)
            dummy_input_shape = (1, input_channels, *self.patch_size)
            dummy_input = torch.randn(dummy_input_shape, device=self.device)
            
            try:
                with torch.no_grad():
                    dummy_output = model(dummy_input)
                self.num_classes = dummy_output.shape[1]  # N, C, D, H, W
                if self.verbose:
                    print(f"Inferred number of output classes via dummy inference: {self.num_classes}")
            except Exception as e:
                print(f"Warning: Could not automatically determine number of classes via dummy inference: {e}")
                print("Ensure your model is loaded correctly and check the expected input shape.")
                # Default to binary segmentation as fallback
                self.num_classes = 2
                print(f"Using default num_classes: {self.num_classes}")

        return model

    def _create_dataset_and_loader(self):
        # Use step_size instead of overlap (step_size is [0-1] representing stride as fraction of patch size)
        # step_size of 0.5 means 50% overlap
        self.dataset = VCDataset(
            input_path=self.input,
            patch_size=self.patch_size,
            step_size=self.overlap,
            num_parts=self.num_parts,
            part_id=self.part_id,
            normalization_scheme=self.normalization_scheme,
            input_format=self.input_format,
            verbose=self.verbose,
            mode='infer',
            # Pass skip_empty_patches flag
            skip_empty_patches=self.skip_empty_patches,
            # Pass Volume-specific parameters
            scroll_id=self.scroll_id,
            segment_id=self.segment_id,
            energy=self.energy,
            resolution=self.resolution
        )

        expected_attr_name = 'all_positions'
        if not hasattr(self.dataset, expected_attr_name) or getattr(self.dataset, expected_attr_name) is None:
            raise AttributeError(f"The VCDataset instance must calculate and provide an "
                                 f"'{expected_attr_name}' attribute (list of coordinate tuples).")

        self.patch_start_coords_list = getattr(self.dataset, expected_attr_name)
        self.num_total_patches = len(self.patch_start_coords_list)

        # ensure dataset __len__ matches coordinate list length
        if len(self.dataset) != self.num_total_patches:
            print(f"Warning: Dataset __len__ ({len(self.dataset)}) mismatch with "
                  f"{expected_attr_name} length ({self.num_total_patches}). Using {expected_attr_name} list length.")

        if self.num_total_patches == 0:
            raise RuntimeError(
                f"Dataset for part {self.part_id}/{self.num_parts} is empty (based on calculated coordinates in '{expected_attr_name}'). Check input data and partitioning.")

        if self.verbose:
            print(f"Total patches to process for part {self.part_id}: {self.num_total_patches}")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            pin_memory=True if self.device != torch.device('cpu') else False,
            collate_fn=VCDataset.collate_fn  # we use custom collate fn here to tag patches that contain only zeros 
                                             # so we don't run them through the model 
        )
        return self.dataset, self.dataloader
        
    def _get_zarr_compressor(self):
        if self.compressor_name.lower() == 'zstd':
            return zarr.Blosc(cname='zstd', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.compressor_name.lower() == 'lz4':
            return zarr.Blosc(cname='lz4', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.compressor_name.lower() == 'zlib':
            return zarr.Blosc(cname='zlib', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.compressor_name.lower() == 'none':
            return None
        else:
            return zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE)

    def _create_output_stores(self):
        if self.num_classes is None or self.patch_size is None or self.num_total_patches is None:
            raise RuntimeError("Cannot create output stores: model/patch info missing.")
        if not self.patch_start_coords_list:
            raise RuntimeError("Cannot create output stores: patch coordinates not available.")

        compressor = self._get_zarr_compressor()
        output_shape = (self.num_total_patches, self.num_classes, *self.patch_size)
        output_chunks = (1, self.num_classes, *self.patch_size)  # Chunk by individual patch
        main_store_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
        
        # Use fsspec to create mapper for the zarr store
        # Use proper storage options for S3 paths
        storage_options = {}
        if main_store_path.startswith('s3://'):
            print(f"Detected S3 output path for logits, using anon=False to use AWS credentials from environment")
            storage_options = {'anon': False}
            
        logits_mapper = fsspec.get_mapper(main_store_path, storage_options=storage_options)
        self.output_store = zarr.open(
            logits_mapper, 
            mode='w',  
            shape=output_shape,
            chunks=output_chunks,
            dtype=np.float16,  
            compressor=compressor,
            write_empty_chunks=False  # we skip empty chunks here so we don't write all zero patches to the array but keep
                                      # the proper indices for later re-zarring 
        )
        
        self.coords_store_path = os.path.join(self.output_dir, f"coordinates_part_{self.part_id}.zarr")
        coord_shape = (self.num_total_patches, len(self.patch_size))
        coord_chunks = (min(self.num_total_patches, 4096), len(self.patch_size))
        
        # Coordinates may also be on S3
        coords_storage_options = {}
        if self.coords_store_path.startswith('s3://'):
            print(f"Detected S3 output path for coordinates, using anon=False to use AWS credentials from environment")
            coords_storage_options = {'anon': False}
            
        coords_mapper = fsspec.get_mapper(self.coords_store_path, storage_options=coords_storage_options)
        coords_store = zarr.open(
            coords_mapper,
            mode='w',
            shape=coord_shape,
            chunks=coord_chunks,
            dtype=np.int32,
            compressor=compressor,
            write_empty_chunks=False  
        )
        
        try:
            original_volume_shape = None
            if hasattr(self.dataset, 'input_shape'):
                if len(self.dataset.input_shape) == 4:  # has channel dimension
                    original_volume_shape = list(self.dataset.input_shape[1:])
                else:  # no channel dimension
                    original_volume_shape = list(self.dataset.input_shape)
                if self.verbose:
                    print(f"Derived original volume shape from dataset.input_shape: {original_volume_shape}")
            
            # store some metadata we might later want 
            self.output_store.attrs['patch_size'] = list(self.patch_size)
            self.output_store.attrs['overlap'] = self.overlap
            self.output_store.attrs['part_id'] = self.part_id
            self.output_store.attrs['num_parts'] = self.num_parts
            
            if original_volume_shape:
                self.output_store.attrs['original_volume_shape'] = original_volume_shape
            
            coords_store.attrs['part_id'] = self.part_id
            coords_store.attrs['num_parts'] = self.num_parts
            
        except Exception as e:
            print(f"Warning: Failed to write custom attributes: {e}")

        coords_np = np.array(self.patch_start_coords_list, dtype=np.int32)
        coords_store[:] = coords_np
        
        if self.verbose: 
            print(f"Created output stores: {main_store_path} and {self.coords_store_path}")
        
        return self.output_store

    def _process_batches(self):
        thread_local = threading.local()
        self.current_patch_write_index = 0
        max_workers = min(16, os.cpu_count() or 4)
        
        def get_zarr_array():
            if not hasattr(thread_local, 'zarr_array'):
                zarr_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
                
                # Use proper storage options for S3 paths
                storage_options = {}
                if zarr_path.startswith('s3://'):
                    if self.verbose:
                        print(f"Thread using S3 path with anon=False: {zarr_path}")
                    storage_options = {'anon': False}
                    
                mapper = fsspec.get_mapper(zarr_path, storage_options=storage_options)
                thread_local.zarr_array = zarr.open(mapper, mode='r+')
            return thread_local.zarr_array
        
        def write_patch(write_index, patch_data):
            zarr_array = get_zarr_array()
            zarr_array[write_index] = patch_data
            return write_index
            
        with tqdm(total=self.num_total_patches, desc=f"Inferring Part {self.part_id}") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for batch_data in self.dataloader:
                    if isinstance(batch_data, (list, tuple)):
                        input_batch = batch_data[0].to(self.device)
                        is_empty_flags = [False] * input_batch.shape[0]
                    elif isinstance(batch_data, dict):
                        input_batch = batch_data['data'].to(self.device)
                        is_empty_flags = batch_data.get('is_empty', [False] * input_batch.shape[0])
                    else:
                        input_batch = batch_data.to(self.device)
                        is_empty_flags = [False] * input_batch.shape[0]
                    
                    # Skip invalid batches
                    if input_batch is None or input_batch.shape[0] == 0:
                        if self.verbose:
                            print("Skipping batch with no valid data")
                        continue
                    
                    batch_size = input_batch.shape[0]
                    output_shape = (batch_size, self.num_classes, *self.patch_size)
                    output_batch = torch.zeros(output_shape, device=self.device, dtype=input_batch.dtype)
                    
                    # Find non-empty patches that need model inference
                    non_empty_indices = [i for i, is_empty in enumerate(is_empty_flags) if not is_empty]
                    
                    # Only perform inference if there are non-empty patches
                    if non_empty_indices:
                        non_empty_input = input_batch[non_empty_indices]
                        
                        # Perform inference with or without TTA
                        with torch.no_grad(), torch.amp.autocast('cuda'):
                            if self.do_tta:
                                # --- TTA ---
                                outputs_batch_tta = []  # Store list of outputs for each TTA for the batch

                                if self.tta_type == 'mirroring':
                                    # Apply model to original and mirrored versions (but only for non-empty patches)
                                    m0 = self.model(non_empty_input)
                                    m1 = self.model(torch.flip(non_empty_input, dims=[-1]))
                                    m2 = self.model(torch.flip(non_empty_input, dims=[-2]))
                                    m3 = self.model(torch.flip(non_empty_input, dims=[-3]))
                                    m4 = self.model(torch.flip(non_empty_input, dims=[-1, -2]))
                                    m5 = self.model(torch.flip(non_empty_input, dims=[-1, -3]))
                                    m6 = self.model(torch.flip(non_empty_input, dims=[-2, -3]))
                                    m7 = self.model(torch.flip(non_empty_input, dims=[-1, -2, -3]))

                                    # Reverse the flips on the outputs before averaging
                                    outputs_batch_tta = [
                                        m0,
                                        torch.flip(m1, dims=[-1]),
                                        torch.flip(m2, dims=[-2]),
                                        torch.flip(m3, dims=[-3]),
                                        torch.flip(m4, dims=[-1, -2]),
                                        torch.flip(m5, dims=[-1, -3]),
                                        torch.flip(m6, dims=[-2, -3]),
                                        torch.flip(m7, dims=[-1, -2, -3])
                                    ]

                                elif self.tta_type == 'rotation':
                                    r0 = self.model(non_empty_input)
                                    r1 = self.model(torch.rot90(non_empty_input, k=1, dims=(-2, -1)))  # 90 deg
                                    r2 = self.model(torch.rot90(non_empty_input, k=2, dims=(-2, -1)))  # 180 deg
                                    r3 = self.model(torch.rot90(non_empty_input, k=3, dims=(-2, -1)))  # 270 deg

                                    # Rotate outputs back before averaging
                                    outputs_batch_tta = [
                                        r0,
                                        torch.rot90(r1, k=-1, dims=(-2, -1)),  # -90 deg
                                        torch.rot90(r2, k=-2, dims=(-2, -1)),  # -180 deg
                                        torch.rot90(r3, k=-3, dims=(-2, -1))   # -270 deg
                                    ]

                                # --- Merge TTA results for the batch ---
                                stacked_outputs = torch.stack(outputs_batch_tta, dim=0)
                                non_empty_output = torch.mean(stacked_outputs, dim=0)

                            else:
                                # --- No TTA ---
                                non_empty_output = self.model(non_empty_input) 
                        
                        # Place non-empty patch outputs in the correct positions in output_batch
                        for idx, original_idx in enumerate(non_empty_indices):
                            output_batch[original_idx] = non_empty_output[idx]
                    
                    else:
                        if self.verbose:
                            print("Batch contains only empty patches, skipping model inference")
                    
                    output_np = output_batch.cpu().numpy().astype(np.float16)
                    current_batch_size = output_np.shape[0]
                    
                    patch_indices = batch_data.get('index', list(range(current_batch_size)))
                    
                    # Submit each patch for writing, now including both empty and non-empty patches
                    for i in range(current_batch_size):
                        patch_data = output_np[i]  # Shape: (C, Z, Y, X)
                        write_index = patch_indices[i]
                        future = executor.submit(write_patch, write_index, patch_data)
                        futures.append(future)
                        
                    completed = [f for f in futures if f.done()]
                    for future in completed:
                        try:
                            _ = future.result() 
                            pbar.update(1)
                            self.current_patch_write_index += 1
                        except Exception as e:
                            print(f"Error writing patch: {e}")
                    
                    futures = [f for f in futures if not f.done()]
                
                for future in futures:
                    try:
                        _ = future.result()
                        pbar.update(1)
                        self.current_patch_write_index += 1
                    except Exception as e:
                        print(f"Error writing patch: {e}")
        
        if self.verbose:
            print(f"Finished writing {self.current_patch_write_index} non-empty patches.")
        
        if not self.skip_empty_patches and self.current_patch_write_index != self.num_total_patches:
            print(f"Warning: Expected {self.num_total_patches} patches, but wrote {self.current_patch_write_index}.")

    def _run_inference(self):
        if self.verbose: print("Loading model...")
        self.model = self._load_model()

        if self.verbose: print("Creating dataset and dataloader...")
        self._create_dataset_and_loader()

        if self.num_total_patches > 0:
            if self.verbose: print("Creating output stores...")
            self._create_output_stores()

            if self.verbose: print("Starting inference and writing logits...")
            self._process_batches()
        else:
            print(f"Skipping processing for part {self.part_id} as no patches were found.")

        if self.verbose: print("Inference complete.")

    def infer(self):
        try:
            self._run_inference()
            main_output_path = os.path.join(self.output_dir, f"logits_part_{self.part_id}.zarr")
            return main_output_path, self.coords_store_path
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc() 


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Run nnUNet inference on Zarr data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the nnUNet model folder')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input Zarr volume')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to store output predictions')
    parser.add_argument('--input_format', type=str, default='zarr', help='Input format (zarr, volume)')
    parser.add_argument('--tta_type', type=str, default='mirroring', choices=['mirroring', 'rotation'], 
                      help='TTA type (mirroring or rotation)')
    parser.add_argument('--disable_tta', action='store_true', help='Disable test time augmentation')
    parser.add_argument('--num_parts', type=int, default=1, help='Number of parts to split processing into')
    parser.add_argument('--part_id', type=int, default=0, help='Part ID to process (0-indexed)')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between patches (0-1)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--patch_size', type=str, default=None, 
                      help='Optional: Override patch size, comma-separated (e.g., "192,192,192"). If not provided, uses the model\'s default patch size.')
    parser.add_argument('--save_softmax', action='store_true', help='Save softmax outputs')
    parser.add_argument('--normalization', type=str, default='instance_zscore', 
                      help='Normalization scheme (instance_zscore, global_zscore, instance_minmax, none)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda, cpu)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--skip-empty-patches', dest='skip_empty_patches', action='store_true', 
                      help='Skip patches that are empty (all values the same). Default: True')
    parser.add_argument('--no-skip-empty-patches', dest='skip_empty_patches', action='store_false',
                      help='Process all patches, even if they appear empty')
    parser.set_defaults(skip_empty_patches=True)
    
    # Add arguments for Zarr compression
    parser.add_argument('--zarr-compressor', type=str, default='zstd',
                      choices=['zstd', 'lz4', 'zlib', 'none'],
                      help='Zarr compression algorithm')
    parser.add_argument('--zarr-compression-level', type=int, default=3,
                      help='Compression level (1-9, higher = better compression but slower)')
    
    # Add arguments for the updated Volume class
    parser.add_argument('--scroll_id', type=str, default=None, help='Scroll ID to use (if input_format is volume)')
    parser.add_argument('--segment_id', type=str, default=None, help='Segment ID to use (if input_format is volume)')
    parser.add_argument('--energy', type=int, default=None, help='Energy level to use (if input_format is volume)')
    parser.add_argument('--resolution', type=float, default=None, help='Resolution to use (if input_format is volume)')
    
    # Add arguments for Hugging Face model loading
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token for accessing private repositories')
    
    args = parser.parse_args()
    
    # Parse optional patch size if provided
    patch_size = None
    if args.patch_size:
        try:
            patch_size = tuple(map(int, args.patch_size.split(',')))
            print(f"Using user-specified patch size: {patch_size}")
        except Exception as e:
            print(f"Error parsing patch_size: {e}")
            print("Expected format: comma-separated integers, e.g. '192,192,192'")
            print("Using model's default patch size instead.")
    
    # Convert scroll_id and segment_id if needed
    scroll_id = args.scroll_id
    segment_id = args.segment_id
    
    if scroll_id is not None and scroll_id.isdigit():
        scroll_id = int(scroll_id)
    
    if segment_id is not None and segment_id.isdigit():
        segment_id = int(segment_id)
    
    print("\n--- Initializing Inferer ---")
    inferer = Inferer(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_format=args.input_format,
        tta_type=args.tta_type,
        do_tta=not args.disable_tta,
        num_parts=args.num_parts,
        part_id=args.part_id,
        overlap=args.overlap,
        batch_size=args.batch_size,
        patch_size=patch_size,  # Will use model's patch size if None
        save_softmax=args.save_softmax,
        normalization_scheme=args.normalization,
        device=args.device,
        verbose=args.verbose,
        skip_empty_patches=args.skip_empty_patches,  # Skip empty patches flag
        # Pass Volume-specific parameters to VCDataset
        scroll_id=scroll_id,
        segment_id=segment_id,
        energy=args.energy,
        resolution=args.resolution,
        # Pass Zarr compression settings
        compressor_name=args.zarr_compressor,
        compression_level=args.zarr_compression_level,
        # Pass Hugging Face parameters
        hf_token=args.hf_token
    )

    try:
        print("\n--- Starting Inference ---")
        logits_path, coords_path = inferer.infer()

        if logits_path and coords_path and os.path.exists(logits_path) and os.path.exists(coords_path):
            print(f"\n--- Inference Finished ---")
            print(f"Output logits saved to: {logits_path}")

            print("\n--- Inspecting Output Store ---")
            try:
                 # Open the zarr store using fsspec
                 logits_mapper = fsspec.get_mapper(logits_path)
                 output_store = zarr.open(logits_mapper, mode='r')
                 print(f"Output shape: {output_store.shape}")
                 print(f"Output dtype: {output_store.dtype}")
                 print(f"Output chunks: {output_store.chunks}")
            except Exception as inspect_e:
                print(f"Could not inspect output Zarr: {inspect_e}")
                
            # Print empty patches report if skip_empty_patches was enabled
            if inferer.skip_empty_patches and hasattr(inferer.dataset, 'get_empty_patches_report'):
                report = inferer.dataset.get_empty_patches_report()
                print("\n--- Empty Patches Report ---")
                print(f"  Empty Patches Skipped: {report['total_skipped']}")
                print(f"  Total Available Positions: {report['total_positions']}")
                if report['total_skipped'] > 0:
                    print(f"  Skip Ratio: {report['skip_ratio']:.2%}")
                    print(f"  Effective Speedup: {1/(1-report['skip_ratio']):.2f}x")

            print("\n--- Inspecting Coordinate Store ---")
            try:
                coords_mapper = fsspec.get_mapper(coords_path)
                coords_store = zarr.open(coords_mapper, mode='r')
                print(f"Coords shape: {coords_store.shape}")
                print(f"Coords dtype: {coords_store.dtype}")
                first_few_coords = coords_store[0:5]
                print(f"First few coordinates:\n{first_few_coords}")
            except Exception as inspect_e:
                print(f"Could not inspect coordinate Zarr: {inspect_e}")
            return 0
        else:
             print("\n--- Inference finished, but output path seems invalid or wasn't created. ---")
             return 1

    except Exception as main_e:
        print(f"\n--- Inference Failed ---")
        print(f"Error: {main_e}")
        import traceback
        traceback.print_exc()
        return 1

# --- Command line usage ---
if __name__ == '__main__':
    import sys
    sys.exit(main())
