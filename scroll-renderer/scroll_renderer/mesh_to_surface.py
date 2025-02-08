"""
Vesuvius Challenge Team 2024
originally integrated from ThaumatoAnakalyptor
"""

import multiprocessing
import psutil
# Force 'spawn' to avoid segfaults or concurrency issues with fork-unsafe libraries
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # If the context is already set elsewhere, we can ignore
    pass

import glob
import shutil
import logging
import vesuvius
from vesuvius import Volume
import dask.array as da
from dask_image.imread import imread
import open3d as o3d
import argparse
import os
import tempfile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore, Lock

# Set max_split_size_mb to a smaller value
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # or another value you deem appropriate

import torch.distributed
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import tifffile
import cv2
import zarr
from multiprocessing import cpu_count
import time
import warnings

from .rendering_utils.interpolate_image_3d import extract_from_image_4d

#################################################################
# Decide number of workers and prefetch
##################################################################
def calculate_batch_and_prefetch(workers, x, ram_fraction=0.8, max_batches=1024):
    """
    Calculate the optimal batch size and prefetch size based on available RAM and workers.
    
    workers: Number of predefined workers (W)
    x: Dimension of each item (X x X x X)
    ram_fraction: Fraction of available RAM to use (default: 80%)
    max_batches: Maximum number of batches to iterate through for batch size (default: 1024)
    
    Returns: optimal_batch_size, optimal_prefetch
    """
    # Get available RAM in bytes
    available_ram = psutil.virtual_memory().available * ram_fraction

    # Memory per element (uint16 = 2 bytes)
    element_size = 2
    
    # Initialize results
    optimal_batch_size = 1
    optimal_prefetch = 1

    for batch_size in range(1, max_batches + 1):
        # Calculate memory usage for one batch
        batch_memory = (batch_size+1) * x * x * x * element_size
        
        # Calculate total memory for workers
        workers_memory = workers * batch_memory
        
        # Check if workers alone exceed memory
        if workers_memory >= available_ram:
            break  # No need to check further if workers' memory exceeds available RAM
        
        # Calculate remaining memory for prefetch
        remaining_memory = available_ram - workers_memory
        
        # Calculate maximum prefetch that fits into remaining memory
        max_prefetch = remaining_memory // batch_memory
        
        # Update optimal values
        optimal_batch_size = batch_size
        optimal_prefetch = max_prefetch

    return min(optimal_batch_size, 32), max(min(4, int(optimal_prefetch)), 2)

##################################################################
# SegmentWriter
##################################################################

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentWriter(BasePredictionWriter):
    def __init__(self, save_path, image_size, r, max_queue_size=10, max_workers=1, display=True, dtype='uint16'):
        super().__init__(write_interval="batch")
        self.save_path = save_path
        self.image_size = image_size
        self.r = r
        self.display = display
        self.dtype = dtype
        self._max_queue_size = max(max_queue_size, max_workers)
        self._max_workers = max_workers

        # Use a memmap file for the volume instead of shared memory.
        self.volume_memmap = None
        self.memmap_filename = os.path.join(self.save_path, "surface_volume.dat")
        
        self.executor = None
        self.semaphore = None
        self.futures = []
        self.num_workers = cpu_count()
        self.trainer_rank = None
        self.image = None
        self.display_lock = Lock()

    def setup(self, trainer, pl_module, stage=None):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self._max_workers)
        if self.semaphore is None:
            self.semaphore = Semaphore(self._max_queue_size)
        self.trainer_rank = trainer.global_rank if trainer.world_size > 1 else 0

        # Ensure output directory exists
        os.makedirs(self.save_path, exist_ok=True)
        # Create (or overwrite) the memmap file.
        shape = (2 * self.r + 1, self.image_size[0], self.image_size[1])
        dtype_np = np.uint16 if self.dtype == 'uint16' else np.uint8
        self.volume_memmap = np.memmap(self.memmap_filename, dtype=dtype_np, mode='w+', shape=shape)
        self.volume_memmap[:] = 0  # Initialize with zeros

    def process_and_write_data(self, prediction):
        try:
            if prediction is None or len(prediction) == 0:
                return
            values, indexes_3d = prediction
            if indexes_3d.shape[0] == 0:
                return

            # Write directly into the memmap.
            self.volume_memmap[indexes_3d[:, 0], indexes_3d[:, 1], indexes_3d[:, 2]] = values

            if self.trainer_rank == 0:
                self.process_display_progress()
        except Exception as e:
            logger.exception("Error in process_and_write_data: %s", e)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        if self.trainer_rank is None:
            self.trainer_rank = trainer.global_rank if trainer.world_size > 1 else 0

        self.semaphore.acquire()
        future = self.executor.submit(self.process_and_write_data, prediction)
        future.add_done_callback(lambda fut: self.semaphore.release())
        self.futures.append(future)

        if self.trainer_rank == 0:
            self.display_progress()

    def teardown(self, trainer, pl_module, stage=None):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        self.semaphore = None
        if self.volume_memmap is not None:
            self.volume_memmap.flush()
            del self.volume_memmap
            self.volume_memmap = None

    def process_display_progress(self):
        if not self.display:
            return
        try:
            if self.dtype == 'uint16':
                image = self.volume_memmap[self.r].astype(np.float32) / 65535.0
            else:
                image = self.volume_memmap[self.r].astype(np.float32) / 255.0
            screen_y, screen_x = 2560, 1440
            if (screen_y * image.shape[1]) // image.shape[0] > screen_x:
                screen_y = (screen_x * image.shape[0]) // image.shape[1]
            else:
                screen_x = (screen_y * image.shape[1]) // image.shape[0]
            image = cv2.resize(image, (screen_x, screen_y))
            image = image.T[::-1, :]
            with self.display_lock:
                self.image = image
        except Exception as e:
            logger.exception("Error in process_display_progress: %s", e)

    def display_progress(self):
        if not self.display or self.trainer_rank != 0:
            return
        with self.display_lock:
            if self.image is None:
                return
            try:
                cv2.imshow("Surface Volume", self.image)
                cv2.waitKey(1)
            except Exception as e:
                logger.exception("Error in display_progress: %s", e)

    def wait_for_all_writes_to_complete(self):
        for future in tqdm(self.futures, desc="Finalizing writes"):
            future.result()
        torch.distributed.barrier()

    def write_to_disk(self, flag='jpg'):
        logger.info("Waiting for all writes to complete")
        self.wait_for_all_writes_to_complete()

        # Only rank 0 writes to disk.
        if self.trainer_rank != 0:
            return

        os.makedirs(self.save_path, exist_ok=True)

        # If the volume memmap was not created in this process (e.g. in the main process),
        # try to open the existing file.
        if self.volume_memmap is None:
            shape = (2 * self.r + 1, self.image_size[0], self.image_size[1])
            dtype_np = np.uint16 if self.dtype == 'uint16' else np.uint8
            try:
                self.volume_memmap = np.memmap(self.memmap_filename, dtype=dtype_np, mode='r', shape=shape)
            except Exception as e:
                logger.exception("Could not attach to existing memmap: %s", e)
                return

        if flag == 'tif':
            self.write_tif()
        elif flag == 'jpg':
            self.write_jpg()
        elif flag == 'memmap':
            self.write_memmap()
        elif flag == 'npz':
            self.write_npz()
        elif flag == 'zarr':
            self.write_zarr()
        else:
            logger.error("Invalid flag for write_to_disk")
            return

        # Flush and clean up the memmap.
        if self.volume_memmap is not None:
            self.volume_memmap.flush()
            del self.volume_memmap
            self.volume_memmap = None

        logger.info("Segment written to disk")

    def write_zarr(self):
        # Write out in chunks to avoid holding two copies in memory
        chunk_size = (16, 16, 16)
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
        zarr_path = os.path.join(self.save_path, "surface_volume.zarr")
        z = zarr.open_array(zarr_path, mode='w', shape=self.volume_memmap.shape,
                             dtype=self.dtype, chunks=chunk_size, compressor=compressor)
        for i in range(0, self.volume_memmap.shape[0], chunk_size[0]):
            z[i:i+chunk_size[0]] = self.volume_memmap[i:i+chunk_size[0]]

    def write_tif(self):
        def save_tif(i, filename):
            image = self.volume_memmap[i]
            image = image.T[::-1, :]
            tifffile.imsave(filename, image)

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for i in range(self.volume_memmap.shape[0]):
                i_str = str(i).zfill(len(str(self.volume_memmap.shape[0])))
                filename = os.path.join(self.save_path, f"{i_str}.tif")
                futures.append(executor.submit(save_tif, i, filename))
            for future in tqdm(as_completed(futures), desc="Writing TIF"):
                future.result()

        # Create a composite image
        composite_image = np.zeros(
            (self.volume_memmap.shape[1], self.volume_memmap.shape[2]),
            dtype=np.float32
        )
        for i in range(self.volume_memmap.shape[0]):
            composite_image = np.maximum(composite_image, self.volume_memmap[i])
        if self.dtype == 'uint16':
            composite_image = composite_image.astype(np.uint16)
        elif self.dtype == 'uint8':
            composite_image = composite_image.astype(np.uint8)
        composite_image = composite_image.T[::-1, :]
        tifffile.imsave(os.path.join(os.path.dirname(self.save_path), "composite.tif"), composite_image)

    def write_jpg(self, quality=60, chunk_size=10):
        total_slices = self.volume_memmap.shape[0]
        slice_indices = list(range(total_slices))

        def save_chunk(chunk):
            for i in chunk:
                image = self.volume_memmap[i]
                if self.dtype == 'uint16':
                    image = (image / 257).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
                image = image.T[::-1, :]
                i_str = str(i).zfill(len(str(total_slices)))
                filename = os.path.join(self.save_path, f"{i_str}.jpg")
                cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for i in range(0, total_slices, chunk_size):
                chunk = slice_indices[i: i + chunk_size]
                futures.append(executor.submit(save_chunk, chunk))
            for future in as_completed(futures):
                future.result()

    def write_memmap(self):
        memmap_path = os.path.join(self.save_path, "surface_volume.memmap")
        memmap_out = np.memmap(memmap_path, dtype=self.dtype, mode='w+', shape=self.volume_memmap.shape)
        memmap_out[:] = self.volume_memmap[:]
        del memmap_out

    def write_npz(self):
        npz_path = os.path.join(self.save_path, "surface_volume.npz")
        np.savez_compressed(npz_path, surface_volume=self.volume_memmap)

##################################################################
# MeshDataset
##################################################################

class MeshDataset(Dataset):
    """
    Dataset class for rendering a mesh.
    NOTE: We do NOT store unpicklable objects here.
    """
    def __init__(
        self,
        path,
        scroll,
        scroll_format="grid cells",
        output_path=None,
        grid_size=500,
        r=32,
        max_side_triangle=10,
        max_workers=1,
        display=False
    ):
        self.grid_size = grid_size
        self.path = path
        self.scroll_format = scroll_format
        self.scroll_name = scroll
        if self.scroll_format == "grid cells":
            self.grid_template = scroll
        elif self.scroll_format == "zarr":
            self.zarr = self.load_zarr(scroll)
            self.zarr_shape = self.zarr.shape
        elif self.scroll_format == "tifstack":
            self.zarr = self.load_tifstack(scroll)
            self.zarr_shape = self.zarr.shape
        elif self.scroll_format == "remote":
            self.scroll = Volume(self.scroll_name, cache=True)
            self.scroll_shape = self.scroll.shape(0)

        # Keep original offset; add 1 to r as in the original code.
        self.r = r + 1
        self.max_side_triangle = max_side_triangle

        # Output path is read by ppm_and_texture(); if not given, use the directory of the obj file.
        output_path = os.path.dirname(path) if output_path is None else output_path
        self.output_path = output_path

        self.load_mesh(path)
        self.grids_to_process = self.init_grids_to_process()
        self.adjust_triangle_sizes()

    def load_zarr(self, dirname):
        stack_array = zarr.open(dirname, mode="r")
        print("Contents of the group:", list(stack_array.groups()))
        print("Arrays in the group:", list(stack_array.arrays()))
        stack_array = stack_array[0]
        print(f"zarr shape: {stack_array.shape}")
        return stack_array

    def load_tifstack(self, path):
        pattern = os.path.join(path, "*.tif")
        stack_array = imread(pattern)
        self.zarr_shape = stack_array.shape
        print(f"Dask shape: {self.zarr_shape}")
        return stack_array

    def parse_mtl_for_texture_filenames(self, mtl_filepath):
        texture_filenames = []
        with open(mtl_filepath, 'r') as file:
            for line in file:
                if line.strip().startswith('map_Kd'):
                    parts = line.split()
                    if len(parts) > 1:
                        texture_filenames.append(parts[1])
        return texture_filenames
    
    def generate_mask_png(self):
        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)
        for triangle in self.uv:
            triangle = triangle.astype(np.int32)
            try:
                cv2.fillPoly(mask, [triangle], 255)
            except Exception:
                pass
        mask = mask[::-1, :]
        cv2.imwrite(
            os.path.join(
                os.path.dirname(self.path),
                os.path.basename(self.path).split(".")[0] + "_mask.png"
            ),
            mask
        )

    def load_mesh(self, path):
        working_path = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]

        tif_path = os.path.join(working_path, f"{base_name}.tif")
        png_path = os.path.join(working_path, f"{base_name}.png")
        mtl_path = os.path.join(working_path, f"{base_name}.mtl")

        if os.path.exists(tif_path):
            image_path = tif_path
            print(f"Found TIF image at: {image_path}")
        elif os.path.exists(png_path):
            image_path = png_path
            print(f"Found PNG image at: {image_path}")
        elif os.path.exists(mtl_path):
            texture_filenames = self.parse_mtl_for_texture_filenames(mtl_path)
            if texture_filenames:
                image_path = os.path.join(working_path, texture_filenames[0])
                print(f"Found material texture image at: {image_path}")
            else:
                image_path = None
                print("No corresponding TIF, PNG, or MTL image found.")
        else:
            image_path = None
            print("No corresponding TIF or PNG image found.")

        print("Texture Image Name:", image_path)
        if image_path:
            with Image.open(image_path) as img:
                y_size, x_size = img.size
        print(f"Y-size: {y_size}, X-size: {x_size}")

        print(f"Loading mesh from {path}")
        with tempfile.NamedTemporaryFile(suffix=".obj") as temp_file:
            temp_path = temp_file.name
            shutil.copy(path, temp_path)
            mesh = o3d.io.read_triangle_mesh(temp_path)
        print(f"Loaded mesh from {path}")

        # Convert to numpy arrays; do NOT store the mesh object
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.triangles = np.asarray(mesh.triangles)
        uv = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
        del mesh

        # Scale UV coordinates to image size
        self.uv = uv * np.array([y_size, x_size])
        self.image_size = (y_size, x_size)

        # Generate the mask image
        self.generate_mask_png()

        self.triangles_vertices = self.vertices[self.triangles]
        self.triangles_normals = self.normals[self.triangles]

    def adjust_triangle_sizes(self):
        """
        Splits large triangles into smaller ones based on max_side_triangle.
        """
        triangles_vertices = self.triangles_vertices
        triangles_normals = self.triangles_normals
        uv = self.uv

        print(f"Original triangles: {triangles_vertices.shape[0]}, {triangles_normals.shape[0]}, {uv.shape[0]}")

        uv_good = []
        triangles_vertices_good = []
        triangles_normals_good = []

        def current_progress(maxS):
            return np.log(2) - np.log(self.max_side_triangle / maxS)

        with tqdm(total=100, desc="Adjusting triangle sizes") as pbar:
            start = None
            while True:
                triangle_min_uv = np.min(uv, axis=1)
                triangle_max_uv = np.max(uv, axis=1)
                side_lengths = np.ceil(triangle_max_uv) - np.floor(triangle_min_uv)
                max_side_lengths = np.max(side_lengths, axis=0)

                if start is None:
                    start = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = 0
                else:
                    now = current_progress(max_side_lengths[0]) + current_progress(max_side_lengths[1])
                    progress = max(1 - now / start, 0)
                pbar.n = int(progress * 100)
                pbar.refresh()

                mask_large_side = np.any(side_lengths > self.max_side_triangle, axis=1)
                if mask_large_side.shape[0] == 0:
                    break

                uv_good_ = uv[~mask_large_side]
                triangles_vertices_good_ = triangles_vertices[~mask_large_side]
                triangles_normals_good_ = triangles_normals[~mask_large_side]

                uv_good.append(uv_good_)
                triangles_vertices_good.append(triangles_vertices_good_)
                triangles_normals_good.append(triangles_normals_good_)

                if not np.any(mask_large_side):
                    break

                uv_large = uv[mask_large_side]
                side_lengths = side_lengths[mask_large_side]
                triangle_min_uv_ = np.expand_dims(triangle_min_uv[mask_large_side], axis=1)
                triangle_max_uv_ = np.expand_dims(triangle_max_uv[mask_large_side], axis=1)
                triangles_vertices_large = triangles_vertices[mask_large_side]
                triangles_normals_large = triangles_normals[mask_large_side]

                mask_larger_side_x = side_lengths[:, 0] >= side_lengths[:, 1]
                mask_larger_side_y = ~mask_larger_side_x

                mask_uv_x_min = uv_large[:, :, 0] == triangle_min_uv_[:, :, 0]
                mask_uv_x_max = uv_large[:, :, 0] == triangle_max_uv_[:, :, 0]
                mask_uv_y_min = uv_large[:, :, 1] == triangle_min_uv_[:, :, 1]
                mask_uv_y_max = uv_large[:, :, 1] == triangle_max_uv_[:, :, 1]

                mask_x_min = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_min)
                mask_x_max = np.logical_and(mask_larger_side_x[:, None], mask_uv_x_max)
                mask_y_min = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_min)
                mask_y_max = np.logical_and(mask_larger_side_y[:, None], mask_uv_y_max)

                mask_x_min_ = np.zeros_like(mask_x_min)
                mask_x_max_ = np.zeros_like(mask_x_max)
                mask_y_min_ = np.zeros_like(mask_y_min)
                mask_y_max_ = np.zeros_like(mask_y_max)

                idx_x_min = np.argmax(mask_x_min, axis=1)
                idx_x_max = np.argmax(mask_x_max, axis=1)
                idx_y_min = np.argmax(mask_y_min, axis=1)
                idx_y_max = np.argmax(mask_y_max, axis=1)
                ix = np.arange(mask_x_min.shape[0])

                mask_x_min_[ix, idx_x_min] = True
                mask_x_max_[ix, idx_x_max] = True
                mask_y_min_[ix, idx_y_min] = True
                mask_y_max_[ix, idx_y_max] = True

                mask_x_min__ = np.logical_and(mask_x_min, mask_x_min_)
                mask_x_max__ = np.logical_and(mask_x_max, mask_x_max_)
                mask_y_min__ = np.logical_and(mask_y_min, mask_y_min_)
                mask_y_max__ = np.logical_and(mask_y_max, mask_y_max_)

                mask_min = np.logical_or(mask_x_min__, mask_y_min__)
                mask_max = np.logical_or(mask_x_max__, mask_y_max__)

                new_vertices = (triangles_vertices_large[mask_min] + triangles_vertices_large[mask_max]) / 2
                new_normals = (triangles_normals_large[mask_min] + triangles_normals_large[mask_max]) / 2
                new_uv = (uv_large[mask_min] + uv_large[mask_max]) / 2

                new_triangles_vertices_0 = np.copy(triangles_vertices_large)
                new_triangles_vertices_0[mask_min] = new_vertices
                new_triangles_normals_0 = np.copy(triangles_normals_large)
                new_triangles_normals_0[mask_min] = new_normals

                new_triangles_vertices_1 = np.copy(triangles_vertices_large)
                new_triangles_vertices_1[mask_max] = new_vertices
                new_triangles_normals_1 = np.copy(triangles_normals_large)
                new_triangles_normals_1[mask_max] = new_normals

                new_uv_0 = np.copy(uv_large)
                new_uv_0[mask_min] = new_uv
                new_uv_1 = np.copy(uv_large)
                new_uv_1[mask_max] = new_uv

                triangles_vertices = np.concatenate((new_triangles_vertices_0, new_triangles_vertices_1), axis=0)
                triangles_normals = np.concatenate((new_triangles_normals_0, new_triangles_normals_1), axis=0)
                uv = np.concatenate((new_uv_0, new_uv_1), axis=0)

            pbar.n = 100
            pbar.refresh()

        self.triangles_vertices = np.concatenate(triangles_vertices_good, axis=0)
        self.triangles_normals = np.concatenate(triangles_normals_good, axis=0)
        self.uv = np.concatenate(uv_good, axis=0)

        print(f"Adjusted triangles: {self.triangles_vertices.shape[0]}, {self.triangles_normals.shape[0]}, {self.uv.shape[0]}")

    def init_grids_to_process(self):
        triangles_vertices = self.triangles_vertices.reshape(-1, 3)
        grids_to_process = set(
            map(tuple, np.maximum((triangles_vertices / self.grid_size).astype(int), 0))
        )

        mask_changing_vertices = np.floor(
            ((triangles_vertices - self.r) / self.grid_size).astype(int)
        ) != np.floor(
            ((triangles_vertices + self.r) / self.grid_size).astype(int)
        )
        mask_changing_vertices = np.any(mask_changing_vertices, axis=1)
        changing_vertices = triangles_vertices[mask_changing_vertices]

        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    candidate_grids = np.floor(
                        (changing_vertices + np.array([x*self.r, y*self.r, z*self.r])) / self.grid_size
                    ).astype(int)
                    valid_grids = candidate_grids[(candidate_grids >= 0).all(axis=1)]
                    grids_to_process.update(set(map(tuple, valid_grids)))

        grids_to_process = sorted(list(grids_to_process))
        print(f"Number of grids to process: {len(grids_to_process)}")
        return grids_to_process

    def extract_triangles_mask(self, grid_index):
        selected_triangles_mask = np.any(
            np.all(
                np.logical_and(
                    self.triangles_vertices >= np.array(grid_index) * self.grid_size - 2*self.r - 1e-7,
                    self.triangles_vertices <= (np.array(grid_index)+1)*self.grid_size + 2*self.r + 1e-7
                ),
                axis=2
            ),
            axis=1
        )
        return selected_triangles_mask

    def load_grid(self, grid_index, uint8=False):
        if self.scroll_format == "grid cells":
            return self.load_grid_cell(grid_index, uint8)
        elif self.scroll_format == "zarr":
            return self.load_grid_zarr(grid_index, uint8)
        elif self.scroll_format == "tifstack":
            return self.load_grid_dask(grid_index, uint8)
        elif self.scroll_format == "remote":
            return self.load_grid_remote(grid_index)
        else:
            raise NotImplementedError(f"Scroll format {self.scroll_format} not implemented.")

    def load_grid_cell(self, grid_index, uint8=False):
        grid_index_ = np.asarray(grid_index)[[1, 0, 2]]
        path = self.grid_template.format(
            grid_index_[0]+1, grid_index_[1]+1, grid_index_[2]+1
        )
        if not os.path.exists(path):
            return None
        with tifffile.TiffFile(path) as tif:
            grid_cell = tif.asarray()
        if uint8:
            grid_cell = (grid_cell // 256).astype(np.uint8)
        return grid_cell

    def load_grid_zarr(self, grid_index, uint8=False):
        grid_index_ = np.asarray(grid_index)[[2, 1, 0]]
        grid_start = grid_index_ * self.grid_size
        grid_end = grid_start + self.grid_size
        grid_end = [min(grid_end[i], self.zarr_shape[i]) for i in range(3)]
        grid_cell = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint16)
        zarr_grid = self.zarr[grid_start[0]:grid_end[0],
                              grid_start[1]:grid_end[1],
                              grid_start[2]:grid_end[2]]
        grid_cell[:zarr_grid.shape[0], :zarr_grid.shape[1], :zarr_grid.shape[2]] = zarr_grid
        if uint8:
            grid_cell = (grid_cell // 256).astype(np.uint8)
        return grid_cell
    
    def load_grid_remote(self, grid_index):
        grid_index_ = np.asarray(grid_index)[[2, 1, 0]]
        grid_start = grid_index_ * self.grid_size
        grid_end = grid_start + self.grid_size
        grid_end = [min(grid_end[i], self.scroll_shape[i]) for i in range(3)]
        grid_cell = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint8)
        scroll_grid = self.scroll[grid_start[0]:grid_end[0],
                                  grid_start[1]:grid_end[1],
                                  grid_start[2]:grid_end[2], 0]
        grid_cell[:scroll_grid.shape[0], :scroll_grid.shape[1], :scroll_grid.shape[2]] = scroll_grid
        return grid_cell
    
    def load_grid_dask(self, grid_index, uint8=False):
        grid_index_ = np.asarray(grid_index)[[2, 1, 0]]
        grid_start = grid_index_ * self.grid_size
        grid_end = grid_start + self.grid_size
        grid_end = [min(grid_end[i], self.zarr_shape[i]) for i in range(3)]
        cell_shape = (
            grid_end[0] - grid_start[0],
            grid_end[1] - grid_start[1],
            grid_end[2] - grid_start[2]
        )
        zarr_grid = self.zarr[
            grid_start[0]:grid_end[0],
            grid_start[1]:grid_end[1],
            grid_start[2]:grid_end[2]
        ].compute()
        grid_cell = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.uint16)
        grid_cell[:cell_shape[0], :cell_shape[1], :cell_shape[2]] = zarr_grid
        if uint8:
            grid_cell = (grid_cell // 256).astype(np.uint8)
        return grid_cell

    def __len__(self):
        return len(self.grids_to_process)

    def __getitem__(self, idx):
        grid_index = self.grids_to_process[idx]
        triangles_mask = self.extract_triangles_mask(grid_index)
        vertices = self.triangles_vertices[triangles_mask]
        normals = self.triangles_normals[triangles_mask]
        uv = self.uv[triangles_mask]
        grid_cell = self.load_grid(grid_index)
        if grid_cell is None:
            return None, None, None, None, None
        grid_cell = grid_cell.astype(np.float32)
        vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        uv_tensor = torch.tensor(uv, dtype=torch.float32)
        grid_cell_tensor = torch.tensor(grid_cell, dtype=torch.float32)
        grid_coord = torch.tensor(np.array(grid_index) * self.grid_size, dtype=torch.int32)
        return grid_coord, grid_cell_tensor, vertices_tensor, normals_tensor, uv_tensor

##################################################################
# PPMAndTextureModel
##################################################################
class PPMAndTextureModel(pl.LightningModule):
    def __init__(self, r: int = 32, max_side_triangle: int = 10, max_triangles_per_loop: int = 5000, dtype='uint16'):
        print("instantiating model")
        super().__init__()
        self.r = r
        self.value_dtype = dtype
        self.max_side_triangle = max_side_triangle
        self.max_triangles_per_loop = max_triangles_per_loop
        self.new_order = [2,1,0]
        self.epsilon = 1e-7

    def ppm(self, pts, tri):
        """
        Barycentric coords for point-in-triangle test
        """
        v0 = tri[:, 2, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
        v1 = tri[:, 1, :].unsqueeze(1) - tri[:, 0, :].unsqueeze(1)
        v2 = pts - tri[:, 0, :].unsqueeze(1)

        dot00 = v0.pow(2).sum(dim=2)
        dot01 = (v0 * v1).sum(dim=2)
        dot11 = v1.pow(2).sum(dim=2)
        dot02 = (v2 * v0).sum(dim=2)
        dot12 = (v2 * v1).sum(dim=2)

        invDenom = 1 / (dot00 * dot11 - dot01.pow(2))
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        
        is_inside = (u >= -self.epsilon) & (v >= -self.epsilon) & ((u + v) <= 1+self.epsilon)
        w = 1 - u - v

        bary_coords = torch.stack([u, v, w], dim=2)
        bary_coords = normalize(bary_coords, p=1, dim=2)
        return bary_coords, is_inside

    def create_grid_points_tensor(self, starting_points, w, h):
        device = starting_points.device
        n = starting_points.shape[0]

        dx = torch.arange(w, device=device)
        dy = torch.arange(h, device=device)
        mesh_dx, mesh_dy = torch.meshgrid(dx, dy, indexing='xy')
        offset_grid = torch.stack((mesh_dx, mesh_dy), dim=2).view(-1, 2)

        starting_points_expanded = starting_points.view(n, 1, 2)
        grid_points = starting_points_expanded + offset_grid
        return grid_points

    def forward(self, x):
        # x is (grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index)
        grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index = x
        if grid_cells is None:
            return None

        min_uv, _ = torch.min(uv_coords_triangles, dim=1)
        min_uv = torch.floor(min_uv)

        nr_triangles = vertices.shape[0]
        values_list = []
        grid_points_list = []

        for i in range(0, nr_triangles, self.max_triangles_per_loop):
            min_uv_ = min_uv[i:i+self.max_triangles_per_loop]
            grid_coords_ = grid_coords[i:i+self.max_triangles_per_loop]
            vertices_ = vertices[i:i+self.max_triangles_per_loop]
            normals_ = normals[i:i+self.max_triangles_per_loop]
            uv_coords_triangles_ = uv_coords_triangles[i:i+self.max_triangles_per_loop]
            grid_index_ = grid_index[i:i+self.max_triangles_per_loop]

            grid_points = self.create_grid_points_tensor(min_uv_, self.max_side_triangle, self.max_side_triangle)
            baryicentric_coords, is_inside = self.ppm(grid_points, uv_coords_triangles_)
            grid_points = grid_points[is_inside]

            vertices_ = vertices_[:, self.new_order, :]
            normals_ = normals_[:, self.new_order, :]

            coords = torch.einsum('ijk,isj->isk', vertices_, baryicentric_coords)
            norms = torch.einsum('ijk,isj->isk', normals_, baryicentric_coords)
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            if norms.dim() == 2:
                norms = norms.unsqueeze(0)
            norms = normalize(norms, dim=2)

            grid_index_ = grid_index_.unsqueeze(-1).expand(-1, baryicentric_coords.shape[1])
            grid_index_ = grid_index_[is_inside]
            grid_coords_ = grid_coords_.unsqueeze(-2).expand(-1, baryicentric_coords.shape[1], -1)

            coords = coords - grid_coords_
            coords = coords[is_inside]
            norms = norms[is_inside]

            coords = coords[:, self.new_order]
            norms = norms[:, self.new_order]

            r_arange = torch.arange(-self.r, self.r+1, device=coords.device).reshape(1, -1, 1)
            coords = coords.unsqueeze(-2).expand(-1, 2*self.r+1, -1) + r_arange * norms.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
            grid_index_ = grid_index_.unsqueeze(-1).unsqueeze(-1).expand(-1, 2*self.r+1, -1)

            r_arange = r_arange.expand(grid_points.shape[0], -1, -1) + self.r
            grid_points = grid_points.unsqueeze(-2).expand(-1, 2*self.r+1, -1)
            grid_points = torch.cat((grid_points, r_arange), dim=-1)

            mask_coords = (
                (coords[:, :, 0] >= 0) & (coords[:, :, 0] < grid_cells.shape[1]) &
                (coords[:, :, 1] >= 0) & (coords[:, :, 1] < grid_cells.shape[2]) &
                (coords[:, :, 2] >= 0) & (coords[:, :, 2] < grid_cells.shape[3])
            )

            coords = coords[mask_coords]
            grid_points = grid_points[mask_coords]
            grid_index_ = grid_index_[mask_coords]

            # Extract values
            values = extract_from_image_4d(grid_cells, grid_index_, coords)
            values = values.reshape(-1)
            grid_points = grid_points.reshape(-1, 3)

            torch.cuda.empty_cache()

            # reorder grid_points to (z, x, y)
            grid_points = grid_points[:, [2, 0, 1]]

            values_list.append(values)
            grid_points_list.append(grid_points)

        del grid_cells, grid_index, min_uv, vertices, normals, uv_coords_triangles
        torch.cuda.empty_cache()

        if len(values_list) == 0:
            return None, None

        values = torch.cat(values_list, dim=0)

        grid_points = torch.cat(grid_points_list, dim=0)

        if self.value_dtype == 'uint16':
            values = values.cpu().numpy().astype(np.uint16)
        else:
            values = values.cpu().numpy().astype(np.uint8)

        grid_points = grid_points.cpu().numpy().astype(np.int32)
        return values, grid_points


##################################################################
# Custom Collate
##################################################################

def custom_collate_fn(batch):
    try:
        grid_cells = []
        vertices = []
        normals = []
        uv_coords_triangles = []
        grid_index = []
        grid_coords = []

        for i, items in enumerate(batch):
            if items is None:
                continue
            grid_coord, grid_cell, vertice, normal, uv_coords_triangle = items
            if grid_cell is None:
                continue
            if len(grid_cell) == 0:
                continue
            if grid_cell.size()[0] == 0:
                continue
            grid_cells.append(grid_cell)
            vertices.append(vertice)
            normals.append(normal)
            uv_coords_triangles.append(uv_coords_triangle)
            grid_index.extend([i]*vertice.shape[0])
            grid_coord = grid_coord.unsqueeze(0).expand(vertice.shape[0], -1)
            grid_coords.extend(grid_coord)
            
        if len(grid_cells) == 0:
            return None, None, None, None, None, None

        grid_cells = torch.stack(grid_cells, dim=0)
        vertices = torch.cat(vertices, dim=0)
        normals = torch.cat(normals, dim=0)
        uv_coords_triangles = torch.cat(uv_coords_triangles, dim=0)
        grid_index = torch.tensor(grid_index, dtype=torch.int32)
        grid_coords = torch.stack(grid_coords, dim=0)

        return grid_coords, grid_cells, vertices, normals, uv_coords_triangles, grid_index
    except:
        print("Error collating")
        return None, None, None, None, None, None

##################################################################
# Main function
##################################################################

def ppm_and_texture(
    obj_path,
    scroll,
    output_path=None,
    grid_size=500,
    gpus=1,
    r=32,
    format='jpg',
    max_side_triangle=10,
    max_triangles_per_loop=5000,
    display=False,
    nr_workers=None,
    remote=False,
    dtype='uint16'
):
    # Automatically determine the number of GPUs
    gpus = min(int(gpus), torch.cuda.device_count()) if torch.cuda.is_available() else 0

    # Decide how many workers to use
    if gpus > 0:
        num_threads = multiprocessing.cpu_count() // int(gpus)
    else:
        num_threads = multiprocessing.cpu_count()

    num_workers = min(num_threads, multiprocessing.cpu_count() - 1)
    max_workers = max(1, multiprocessing.cpu_count() - 1)

    if nr_workers is not None:
        num_workers = nr_workers
        max_workers = nr_workers

    # Detect scroll format
    is_zarr = scroll.endswith(".zarr")
    if remote:
        scroll_format = "remote"
        grid_size = 128
        dtype = 'uint8'
    elif is_zarr:
        scroll_format = "zarr"
        grid_size = 128
    elif os.path.isdir(scroll):
        pattern = os.path.join(scroll, "cell_yxz_[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9].tif")
        matching_files = glob.glob(pattern)
        if matching_files:
            scroll_format = "grid cells"
            scroll = os.path.join(scroll, "cell_yxz_{:03d}_{:03d}_{:03d}.tif")
        else:
            scroll_format = "tifstack"
            grid_size = 128
    else:
        scroll_format = "tifstack"  # fallback
        grid_size = 128

    print(f"Calculating batch size and prefetch factor for a total of {num_workers} workers.")

    batch_size, prefetch_factor = calculate_batch_and_prefetch(num_workers, grid_size, ram_fraction=0.8)
    print(f"Chosen batch size = {batch_size}, prefetch factor = {prefetch_factor}")

    # Create the dataset
    dataset = MeshDataset(
        path=obj_path,
        scroll=scroll,
        scroll_format=scroll_format,
        output_path=output_path,
        grid_size=grid_size,
        r=r,
        max_side_triangle=max_side_triangle,
        max_workers=max_workers,
        display=display
    )

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    # Create the model
    model = PPMAndTextureModel(
        r=r,
        max_side_triangle=max_side_triangle,
        max_triangles_per_loop=max_triangles_per_loop,
        dtype=dtype
    )

    model = torch.compile(model)

    # Create the SegmentWriter callback outside the dataset (to avoid pickling issues)
    write_path = os.path.join(dataset.output_path, "layers")
    writer = SegmentWriter(
        save_path=write_path,
        image_size=dataset.image_size,
        r=r,
        max_queue_size=10,
        max_workers=max_workers,
        display=display,
        dtype=dtype
    )

    trainer = pl.Trainer(
        callbacks=[writer],
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else multiprocessing.cpu_count() - num_workers,
        strategy="ddp"
    )

    print("Start Rendering")
    trainer.predict(model, dataloaders=dataloader, return_predictions=False)
    print("Rendering done")
    writer.write_to_disk(format)

    # Final barrier if needed
    torch.distributed.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj', type=str)
    parser.add_argument('scroll', type=str, help="Path to the grid cells, tifstack, zarr file or vesuvius canonical scroll name.")
    parser.add_argument('--output_path', type=str, default=None, help="Output folder path that shall contain the layers folder.")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--r', type=int, default=32)
    parser.add_argument('--format', type=str, default='jpg')
    parser.add_argument('--max_side_triangle', type=int, default=10)
    parser.add_argument('--triangle_batch', type=int, default=5000)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--nr_workers', type=int, default=None)
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()

    print(f"Rendering args: {args}")
    if args.display:
        print("[INFO]: Displaying the rendering image slows down the rendering process by about 20%.")

    ppm_and_texture(
        obj_path=args.obj,
        scroll=args.scroll,
        output_path=args.output_path,
        gpus=args.gpus,
        r=args.r,
        format=args.format,
        max_side_triangle=args.max_side_triangle,
        max_triangles_per_loop=args.triangle_batch,
        display=args.display,
        nr_workers=args.nr_workers,
        remote=args.remote
    )

if __name__ == '__main__':
    main()
