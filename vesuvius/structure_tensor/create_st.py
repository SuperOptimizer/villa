import argparse
import shutil
import torch
from torch import nn
import torch.nn.functional as F
import fsspec
import numpy as np
import os
import subprocess
import time
import traceback
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import zarr
import numcodecs

from models.run.inference import Inferer
from data.utils import open_zarr


class StructureTensorInferer(Inferer, nn.Module):
    """
    Inherits all of Inferer's I/O, patching, zarr & scheduling machinery,
    but replaces the nnU-Net inference with a 6‐channel 3D structure tensor.
    """
    def __init__(self,
                 *args,
                 sigma: float = 1.0,
                 smooth_components: bool = False,
                 volume: int = None,  # Add volume attribute
                 **kwargs):
        # --- Initialize Module first so register_buffer exists ---
        nn.Module.__init__(self)

        # --- Remove any incoming normalization_scheme so it can't collide ---
        kwargs.pop('normalization_scheme', None)

        # --- Now initialize Inferer, forcing normalization_scheme='none' ---
        Inferer.__init__(self, *args, normalization_scheme='none', **kwargs)

        self.num_classes = 6
        self.do_tta = False
        self.sigma = sigma
        self.smooth_components = smooth_components
        self.volume = volume  # Initialize volume attribute

        # --- Auto-infer patch_size from the input Zarr's chunking if none given ---
        if self.patch_size is None:
            store = open_zarr(
                path=self.input,
                mode='r',
                storage_options={'anon': False}
                    if str(self.input).startswith('s3://') else None
            )
            chunks = store.chunks  # e.g. (1, pZ, pY, pX) or (pZ, pY, pX)
            if len(chunks) == 4:
                # drop the channel‐chunk
                self.patch_size = tuple(chunks[1:])
            elif len(chunks) == 3:
                self.patch_size = tuple(chunks)
            else:
                raise ValueError(
                    f"Cannot infer patch_size from input chunks={chunks}; "
                    "please supply --patch-size Z,Y,X"
                )
            if self.verbose:
                print(f"Inferred patch_size {self.patch_size} from input Zarr chunking")

        # — precompute 3D Gaussian kernel  —
        if self.sigma > 0:
            radius = int(3 * self.sigma)
            coords = torch.arange(-radius, radius + 1,
                                  device=self.device, dtype=torch.float32)
            g1 = torch.exp(-coords**2 / (2 * self.sigma * self.sigma))
            g1 = g1 / g1.sum()
            g3 = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
            # store both kernel and pad:
            self.register_buffer("_gauss3d",   g3[None,None])     # [1,1,D,H,W]
            self.register_buffer("_gauss3d_tensor",
                                 self._gauss3d.expand(6, -1, -1, -1, -1))
            self._pad     = radius

        # Build 3D Pavel Holoborodko kernels and store as plain tensors
        # http://www.holoborodko.com/pavel/image-processing/edge-detection/
        # derivative kernel, smoothing kernel
        
        dev   = self.device
        dtype = torch.float32
        d = torch.tensor([2.,1.,-16.,-27.,0.,27.,16.,-1.,-2.], device=dev, dtype=dtype) # derivative kernel
        s = torch.tensor([1., 4., 6., 4., 1.], device=dev, dtype=dtype) # smoothing kernel

        # depth‐derivative with y/x smoothing
        kz = (d.view(9,1,1) * s.view(1,5,1) * s.view(1,1,5)) / (96*16*16)
        # height‐derivative with z/x smoothing
        ky = (s.view(5,1,1) * d.view(1,9,1) * s.view(1,1,5)) / (96*16*16)
        # width‐derivative with z/y smoothing
        kx = (s.view(5,1,1) * s.view(1,5,1) * d.view(1,1,9)) / (96*16*16)

        self.register_buffer("pavel_kz", kz[None,None])
        self.register_buffer("pavel_ky", ky[None,None])
        self.register_buffer("pavel_kx", kx[None,None])

    def _load_model(self):
        """
        No model to load—just ensure num_classes is set.
        """
        self.num_classes = 6
        # patch_size logic from base class still applies (it will fall back
        # to user-specified patch_size or the model default, but here model
        # default is never used).
        return None

    def _create_output_stores(self):
        """
        Override to create a single output zarr with the full volume shape
        and write patches directly to their positions.
        """
        if self.num_classes is None or self.patch_size is None:
            raise RuntimeError("Cannot create output stores: model/patch info missing.")
        if not self.patch_start_coords_list:
            raise RuntimeError("Cannot create output stores: patch coordinates not available.")

        # Get the original volume shape
        if hasattr(self.dataset, 'input_shape'):
            if len(self.dataset.input_shape) == 4:  # has channel dimension
                original_volume_shape = list(self.dataset.input_shape[1:])
            else:  # no channel dimension
                original_volume_shape = list(self.dataset.input_shape)
        else:
            raise RuntimeError("Cannot determine original volume shape from dataset")

        # Check if we're in multi-GPU mode by seeing if output_dir ends with .zarr
        # and num_parts > 1, which indicates we should open existing shared store
        if self.num_parts > 1 and self.output_dir.endswith('.zarr'):
            # Multi-GPU mode: open existing shared store
            main_store_path = self.output_dir
            print(f"Opening existing shared store at: {main_store_path}")
            
            self.output_store = open_zarr(
                path=main_store_path, 
                mode='r+',  # Open in read-write mode
                storage_options={'anon': False} if main_store_path.startswith('s3://') else None,
                verbose=self.verbose
            )
        else:
            # Single-GPU mode: create new store
            # Create output path - for structure tensor, we create the full volume directly
            main_store_path = os.path.join(self.output_dir, f"structure_tensor_part_{self.part_id}.zarr")
            
            # Shape is (6 channels, Z, Y, X) for the full volume
            output_shape = (self.num_classes, *original_volume_shape)
            
            # Use the same chunking as patch size for efficient writing
            output_chunks = (self.num_classes, *self.patch_size)
            
            compressor = self._get_zarr_compressor()
            
            print(f"Creating output store at: {main_store_path}")
            print(f"Full volume shape: {output_shape}")
            print(f"Chunk shape: {output_chunks}")
            
            # Create the zarr array using our helper function
            self.output_store = open_zarr(
                path=main_store_path, 
                mode='w',  
                storage_options={'anon': False} if main_store_path.startswith('s3://') else None,
                verbose=self.verbose,
                shape=output_shape,
                chunks=output_chunks,
                dtype=np.float32,  
                compressor=compressor,
                write_empty_chunks=False
            )
            
            # Store metadata
            try:
                self.output_store.attrs['patch_size'] = list(self.patch_size)
                self.output_store.attrs['overlap'] = self.overlap
                self.output_store.attrs['part_id'] = self.part_id
                self.output_store.attrs['num_parts'] = self.num_parts
                self.output_store.attrs['original_volume_shape'] = original_volume_shape
                self.output_store.attrs['sigma'] = self.sigma
                self.output_store.attrs['smooth_components'] = self.smooth_components
            except Exception as e:
                print(f"Warning: Failed to write custom attributes: {e}")
        
        # Set coords_store_path to None since we're not creating it
        self.coords_store_path = None
        
        if self.verbose: 
            print(f"Created output store: {main_store_path}")
        
        return self.output_store
    
    def compute_structure_tensor(self, x: torch.Tensor, sigma=None):
        # x: [N,1,Z,Y,X]
        if sigma is None: sigma = self.sigma
        if sigma > 0:
            x = F.conv3d(x, self._gauss3d, padding=(self._pad,)*3)

        # 2) apply Pavel
        gz = F.conv3d(x, self.pavel_kz, padding=(4,2,2))
        gy = F.conv3d(x, self.pavel_ky, padding=(2,4,2))
        gx = F.conv3d(x, self.pavel_kx, padding=(2,2,4))

        # 3) build tensor components
        Jxx = gx * gx
        Jyx = gx * gy
        Jzx = gx * gz
        Jyy = gy * gy
        Jzy = gy * gz
        Jzz = gz * gz

        # stack into [N,6, Z,Y,X]
        J = torch.stack([Jzz, Jzy, Jzx, Jyy, Jyx, Jxx], dim=1)

        # drop that singleton channel axis → [N,6,D,H,W]
        if J.dim() == 6 and J.shape[2] == 1:
            J = J.squeeze(2)

        # now group‐conv each of the 6 channels with your Gaussian:
        if sigma > 0 and self.smooth_components:
            # build one filter per channel
            J = F.conv3d(J, weight=self._gauss3d_tensor, padding=(self._pad,)*3, groups=6)

        return J

    def _run_inference(self):
        """
        Skip model loading entirely, just build dataset, stores, then process.
        """
        if self.verbose: print("Preparing dataset & output stores for structure‐tensor...")
        # load_model is a no‐op now
        self.model = self._load_model()
        # dataset + dataloader
        self._create_dataset_and_loader()
        # zarr stores for logits & coords
        self._create_output_stores()
        # compute & write structure tensor
        self._process_batches()

    def infer(self):
        """
        Override to return just the output path (not a tuple with coords path).
        """
        try:
            self._run_inference()
            # Return the correct output path based on whether we're in multi-GPU mode
            if self.num_parts > 1 and self.output_dir.endswith('.zarr'):
                # Multi-GPU mode: return the shared store path
                main_output_path = self.output_dir
            else:
                # Single-GPU mode: return the individual part path
                main_output_path = os.path.join(self.output_dir, f"structure_tensor_part_{self.part_id}.zarr")
            return main_output_path
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_batches(self):
        """
        Iterate over patches using DataLoader, compute J, and write directly to positions in the full volume zarr.
        """
        numcodecs.blosc.use_threads = False

        total = self.num_total_patches
        store = self.output_store
        processed_count = 0

        with tqdm(total=total, desc="Struct-Tensor") as pbar:
            for batch_data in self.dataloader:
                # Handle different batch data formats
                if isinstance(batch_data, dict):
                    data_batch = batch_data['data']
                    pos_batch = batch_data.get('pos', [])
                    indices_batch = batch_data.get('index', [])
                else:
                    # Fallback for simple tensor batches
                    data_batch = batch_data
                    pos_batch = []
                    indices_batch = []
                
                # Move batch to device
                data_batch = data_batch.to(self.device).float()
                batch_size = data_batch.shape[0]
                
                # Process each item in the batch
                for i in range(batch_size):
                    data = data_batch[i]  # Shape: (C, Z, Y, X) or (Z, Y, X)
                    
                    # Get position
                    if pos_batch and i < len(pos_batch):
                        pos = pos_batch[i]
                    elif indices_batch and i < len(indices_batch):
                        idx = indices_batch[i]
                        pos = self.patch_start_coords_list[idx]
                    else:
                        # Fallback to sequential indexing
                        pos = self.patch_start_coords_list[processed_count + i]
                    
                    # Extract coordinates
                    z_start, y_start, x_start = pos
                    z_end = z_start + self.patch_size[0]
                    y_end = y_start + self.patch_size[1]
                    x_end = x_start + self.patch_size[2]
                    
                    # fiber‐volume mask
                    if self.volume is not None:
                        data = (data == self.volume).float()
                    
                    # --- Ensure shape is [1, C, Z, Y, X] ---
                    if data.ndim == 3:
                        x = data.unsqueeze(0).unsqueeze(0)  # (1,1,Z,Y,X)
                    elif data.ndim == 4:
                        x = data.unsqueeze(0)               # (1,C,Z,Y,X)
                    else:
                        raise RuntimeError(f"Unexpected patch data ndim={data.ndim}")
                    
                    # --- Compute structure tensor ---
                    # J has shape [1, 6, Z, Y, X]
                    with torch.no_grad():
                        J = self.compute_structure_tensor(x, sigma=self.sigma)
                    
                    # --- Bring to numpy and cast ---
                    out_np = J.cpu().numpy().astype(np.float32)
                    
                    # --- Defensive squeeze of any extra singleton dim ---
                    # Target shape is (6, Z, Y, X)
                    if out_np.ndim == 5:
                        # Could be (1,6,Z,Y,X) or (6,1,Z,Y,X)
                        # First try drop batch axis if present
                        if out_np.shape[0] == 1:
                            out_np = out_np[0]
                        # Then drop any singleton channel axis
                        if out_np.ndim == 4 and out_np.shape[0] == 1 and self.num_classes != 1:
                            # unlikely, but just in case
                            out_np = out_np[0]
                        # Or if shape is (6,1,Z,Y,X), drop the middle
                        if out_np.ndim == 5 and out_np.shape[1] == 1:
                            out_np = out_np[:,0]
                    
                    # Final check
                    if out_np.shape != (self.num_classes, *self.patch_size):
                        raise RuntimeError(
                            f"After squeeze, expected out_np shape {(self.num_classes, *self.patch_size)}, "
                            f"but got {out_np.shape}"
                        )
                    
                    # --- Write directly to position in full volume Zarr ---
                    store[:, z_start:z_end, y_start:y_end, x_start:x_end] = out_np
                    
                    # Update progress
                    pbar.update(1)
                
                processed_count += batch_size
                self.current_patch_write_index = processed_count

        if self.verbose:
            print(f"Written {self.current_patch_write_index}/{total} patches.")


class ChunkDataset(Dataset):
    """Dataset of spatial chunk bounds for structure‐tensor eigen decomposition."""
    def __init__(self, input_path, chunks, device):
        self.input_path = input_path
        self.chunks = chunks
        self.device = device

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # read block
        z0, z1, y0, y1, x0, x1 = self.chunks[idx]
        src = open_zarr(
            path=self.input_path, mode='r',
            storage_options={'anon': False} if self.input_path.startswith('s3://') else None
        )
        block_np = src[:, z0:z1, y0:y1, x0:x1].astype('float32')
        block = torch.from_numpy(block_np).to(self.device)  # [6, dz, dy, dx]
        return idx, (z0, z1, y0, y1, x0, x1), block


# solve the eigenvalue problem and sanitize the output
def _eigh_and_sanitize(M: torch.Tensor):
    w, v = torch.linalg.eigh(M) 
    # sanitize once
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return w, v


# compute the eigenvectors (and the eigenvalues)
def _compute_eigenvectors(
    block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, dz, dy, dx = block.shape
    N = dz * dy * dx

    # build + sanitize
    x = block.view(6, N)
    M = torch.empty((N,3,3), dtype=block.dtype, device=block.device)
    M[:, 0, 0] = x[0]; M[:, 0, 1] = x[1]; M[:, 0, 2] = x[2]
    M[:, 1, 0] = x[1]; M[:, 1, 1] = x[3]; M[:, 1, 2] = x[4]
    M[:, 2, 0] = x[2]; M[:, 2, 1] = x[4]; M[:, 2, 2] = x[5]
    M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    zero_mask = M.abs().sum(dim=(1,2)) == 0

    batch_size = 1048576
    # eigen-decomp (either whole or in chunks)
    if batch_size is None or N <= batch_size:
        w, v = _eigh_and_sanitize(M)
    else:
        ws = []; vs = []
        for chunk in M.split(batch_size, dim=0):
            wi, vi = _eigh_and_sanitize(chunk)
            ws.append(wi); vs.append(vi)
        w = torch.cat(ws, 0)
        v = torch.cat(vs, 0)

    # zero out truly‐empty voxels without branching
    # zero_mask: [N], w: [N,3], v: [N,3,3]
    mask_w = zero_mask.unsqueeze(-1)             # [N,1]
    w = w.masked_fill(mask_w, 0.0)               # [N,3]
    mask_v = mask_w.unsqueeze(-1)                # [N,1,1]
    v = v.masked_fill(mask_v, 0.0)               # [N,3,3]

    # reshape back
    eigvals = w.transpose(0,1).view(3, dz, dy, dx)
    eigvecs = (
        v
        .permute(0,2,1)
        .reshape(N,9)
        .transpose(0,1)
        .view(9, dz, dy, dx)
    )
    return eigvals, eigvecs


# compiling the function
_compute_eigenvectors = torch.compile(
    _compute_eigenvectors,
    mode="max-autotune-no-cudagraphs",
    fullgraph=True,
)


def _finalize_structure_tensor_torch(
    input_path, output_path, chunk_size, num_workers, compressor, verbose, swap_eigenvectors=False
):
    # open input
    src = open_zarr(
        path=input_path, mode='r',
        storage_options={'anon': False} if input_path.startswith('s3://') else None,
        verbose=verbose
    )
    C, Z, Y, X = src.shape
    assert C == 6, f"Expect 6 channels, got {C}"

    # chunk dims
    if chunk_size is None:
        # src.chunks == (1, cz, cy, cx)
        cz, cy, cx = src.chunks[1:]
    else:
        cz, cy, cx = chunk_size
    if verbose:
        print(f"[Eigen] using chunks (dz,dy,dx)=({cz},{cy},{cx})")

    # prepare eigenvectors output zarr
    out_chunks = (1, cz, cy, cx)
    open_zarr(
        path=output_path, mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        verbose=verbose,
        shape=(9, Z, Y, X),
        chunks=out_chunks,
        compressor=compressor,
        dtype=np.float32,
        write_empty_chunks=False,
        overwrite=True
    )

    # prepare eigenvalue output zarr (3 channels)
    eigval_path = output_path.replace('.zarr', '_eigenvalues.zarr')
    open_zarr(
        path=eigval_path, mode='w',
        storage_options={'anon': False} if output_path.startswith('s3://') else None,
        shape=(3, Z, Y, X),
        chunks=out_chunks,
        compressor=compressor,
        dtype=np.float32,
        write_empty_chunks=False,
        overwrite=True
    )
    # build chunk list
    def gen_bounds():
        for z0 in range(0, Z, cz):
            for y0 in range(0, Y, cy):
                for x0 in range(0, X, cx):
                    yield (z0, min(z0+cz,Z),
                           y0, min(y0+cy,Y),
                           x0, min(x0+cx,X))
    chunks = list(gen_bounds())
    if verbose:
        print(f"[Eigen] {len(chunks)} chunks to solve the eigenvalue problem on")

    # 5) Dataset & DataLoader (with a no‐op collate so bounds come through untouched)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ChunkDataset(input_path, chunks, device)
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: batch[0]  # return the single sample directly
    )

    # 6) Process each chunk
    for idx, bounds, block in tqdm(loader, desc="[Eigen] Chunks", disable=not verbose):
        # idx: int, bounds: (z0,z1,y0,y1,x0,x1), block: tensor [1,6,dz,dy,dx]
        z0, z1, y0, y1, x0, x1 = bounds
        # squeeze off the dummy batch dim
        with torch.no_grad():
            eigvals_block_gpu, eigvecs_block_gpu = _compute_eigenvectors(block) #TODO: fix batch size intelligently
        eigvals_block = eigvals_block_gpu.cpu().numpy()
        eigvecs_block = eigvecs_block_gpu.cpu().numpy()

        del block, eigvals_block_gpu, eigvecs_block_gpu
        torch.cuda.empty_cache()

        if swap_eigenvectors:
            # 1) reshape eigenvectors into [3 eigenvectors, 3 components, dz,dy,dx]
            v = eigvecs_block.reshape(3, 3, *eigvecs_block.shape[1:])  # → [3,3,dz,dy,dx]
            # eigenvalues are already [3, dz,dy,dx]
            w = eigvals_block                                        # → [3,dz,dy,dx]

            # 2) swap eigenvector #0 <-> #1 and their eigenvalues
            #    this exchanges the *first* and *second* principal directions
            v[[0, 1], :, ...] = v[[1, 0], :, ...]
            w[[0, 1],    ...] = w[[1, 0],    ...]

            # 3) flatten back to your Zarr layout
            eigvecs_block = v.reshape(9, *eigvecs_block.shape[1:])
            eigvals_block  = w

        # write
        dst_vec = open_zarr(
            path=output_path, mode='r+',
            storage_options={'anon': False} if output_path.startswith('s3://') else None
        )
        dst_val = open_zarr(
            path=eigval_path, mode='r+',
            storage_options={'anon': False} if output_path.startswith('s3://') else None
        )
        dst_vec[:, z0:z1, y0:y1, x0:x1] = eigvecs_block
        dst_val[:, z0:z1, y0:y1, x0:x1] = eigvals_block

    if verbose:
        print(f"[Eigen] eigenvectors → {output_path}")
        print(f"[Eigen] eigenvalues  → {eigval_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute 3D structure tensor or eigenvalues/eigenvectors')
    
    # Basic I/O arguments
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Path to the input Zarr volume')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Path to store output results')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='structure-tensor',
                        choices=['structure-tensor', 'eigenanalysis'],
                        help='Mode of operation: compute structure tensor or perform eigenanalysis')
    
    # Structure tensor computation arguments
    parser.add_argument('--structure-tensor', action='store_true', dest='structure_tensor',
                        help='Compute 6-channel 3D structure tensor (sets mode to structure-tensor)')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Gaussian σ for structure-tensor smoothing')
    parser.add_argument('--smooth-components', action='store_true',
                        help='After computing Jxx…Jzz, apply a second Gaussian smoothing to each channel')
    parser.add_argument('--volume', type=int, default=None,
                        help='Volume ID for fiber-volume masking')
    
    # Patch processing arguments
    parser.add_argument('--patch_size', type=str, default=None, 
                        help='Override patch size, comma-separated (e.g., "192,192,192")')
    parser.add_argument('--overlap', type=float, default=0.0, 
                        help='Overlap between patches (0-1), default 0.0 for structure tensor')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for inference')
    parser.add_argument('--num_parts', type=int, default=1, 
                        help='Number of parts to split processing into')
    parser.add_argument('--part_id', type=int, default=0, 
                        help='Part ID to process (0-indexed)')
    
    # Eigenanalysis arguments
    parser.add_argument('--eigen-input', type=str, default=None,
                        help='Input path for eigenanalysis (6-channel structure tensor zarr)')
    parser.add_argument('--eigen-output', type=str, default=None,
                        help='Output path for eigenvectors')
    parser.add_argument('--chunk-size', type=str, default=None,
                        help='Chunk size for eigenanalysis, comma-separated (e.g., "64,64,64")')
    parser.add_argument('--swap-eigenvectors', action='store_true',
                        help='Swap eigenvectors 0 and 1')
    parser.add_argument('--delete-intermediate', action='store_true',
                        help='Delete intermediate structure tensor after eigenanalysis')
    
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to use (cuda, cpu)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose output')
    parser.add_argument('--zarr-compressor', type=str, default='zstd',
                        choices=['zstd', 'lz4', 'zlib', 'none'],
                        help='Zarr compression algorithm')
    parser.add_argument('--zarr-compression-level', type=int, default=3,
                        help='Compression level (1-9)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Handle mode logic
    if args.structure_tensor:
        args.mode = 'structure-tensor'
    
    # Parse patch size if provided
    patch_size = None
    if args.patch_size:
        try:
            patch_size = tuple(map(int, args.patch_size.split(',')))
            print(f"Using user-specified patch size: {patch_size}")
        except Exception as e:
            print(f"Error parsing patch_size: {e}")
            print("Using default patch size.")
    
    # Parse chunk size for eigenanalysis
    chunk_size = None
    if args.chunk_size:
        try:
            chunk_size = tuple(map(int, args.chunk_size.split(',')))
        except Exception as e:
            print(f"Error parsing chunk_size: {e}")
    
    # Get compressor
    if args.zarr_compressor.lower() == 'zstd':
        compressor = zarr.Blosc(cname='zstd', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'lz4':
        compressor = zarr.Blosc(cname='lz4', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'zlib':
        compressor = zarr.Blosc(cname='zlib', clevel=args.zarr_compression_level, shuffle=zarr.Blosc.SHUFFLE)
    elif args.zarr_compressor.lower() == 'none':
        compressor = None
    else:
        compressor = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE)
    
    # Compile the compute_structure_tensor method
    StructureTensorInferer.compute_structure_tensor = torch.compile(
        StructureTensorInferer.compute_structure_tensor,
        mode="reduce-overhead",
        fullgraph=True
    )
    
    if args.mode == 'structure-tensor':
        # Run structure tensor computation
        print("\n--- Initializing Structure Tensor Inferer ---")
        inferer = StructureTensorInferer(
            model_path='dummy',  # Not used for structure tensor
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sigma=args.sigma,
            smooth_components=args.smooth_components,
            volume=args.volume,
            num_parts=args.num_parts,
            part_id=args.part_id,
            overlap=args.overlap,
            batch_size=args.batch_size,
            patch_size=patch_size,
            device=args.device,
            verbose=args.verbose,
            compressor_name=args.zarr_compressor,
            compression_level=args.zarr_compression_level,
            num_dataloader_workers=args.num_workers
        )
        
        try:
            print("\n--- Starting Structure Tensor Computation ---")
            result = inferer.infer()
            
            # Handle either single return value or tuple
            if isinstance(result, tuple):
                logits_path = result[0]
            else:
                logits_path = result
            
            if logits_path:
                print(f"\n--- Structure Tensor Computation Finished ---")
                print(f"Output tensor saved to: {logits_path}")
                
                # Optionally run eigenanalysis immediately
                if args.eigen_output:
                    print("\n--- Running Eigenanalysis ---")
                    _finalize_structure_tensor_torch(
                        input_path=logits_path,
                        output_path=args.eigen_output,
                        chunk_size=chunk_size,
                        num_workers=args.num_workers,
                        compressor=compressor,
                        verbose=args.verbose,
                        swap_eigenvectors=args.swap_eigenvectors
                    )
                    
                    # Delete intermediate if requested
                    if args.delete_intermediate:
                        if logits_path.startswith('s3://'):
                            fs = fsspec.filesystem(logits_path.split('://')[0], anon=False)
                            fs.rm(logits_path, recursive=True)
                        else:
                            shutil.rmtree(logits_path, ignore_errors=True)
                        print(f"Deleted intermediate tensor: {logits_path}")
                
        except Exception as e:
            print(f"\n--- Structure Tensor Computation Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    elif args.mode == 'eigenanalysis':
        # Run eigenanalysis only
        if not args.eigen_input:
            print("Error: --eigen-input must be provided for eigenanalysis mode")
            return 1
        if not args.eigen_output:
            print("Error: --eigen-output must be provided for eigenanalysis mode")
            return 1
            
        print("\n--- Running Eigenanalysis ---")
        try:
            _finalize_structure_tensor_torch(
                input_path=args.eigen_input,
                output_path=args.eigen_output,
                chunk_size=chunk_size,
                num_workers=args.num_workers,
                compressor=compressor,
                verbose=args.verbose,
                swap_eigenvectors=args.swap_eigenvectors
            )
            
            # Delete intermediate if requested
            if args.delete_intermediate:
                if args.eigen_input.startswith('s3://'):
                    fs = fsspec.filesystem(args.eigen_input.split('://')[0], anon=False)
                    fs.rm(args.eigen_input, recursive=True)
                else:
                    shutil.rmtree(args.eigen_input, ignore_errors=True)
                print(f"Deleted intermediate tensor: {args.eigen_input}")
                
            print("\n--- Eigenanalysis Completed Successfully ---")
            
        except Exception as e:
            print(f"\n--- Eigenanalysis Failed ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
