"""
Improved writer worker implementation for ZarrTempStorage.

This module contains the worker thread implementation for writing patches to ZarrTempStorage
with a more efficient parallel I/O approach that directly writes to the underlying zarr store.
"""
import multiprocessing
import numpy as np
import os
import time
import zarr
import threading
import logging
import traceback
from queue import Queue, Empty
from typing import Dict, Any, Tuple, Optional, List
from numcodecs import Blosc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('zarr_writer')

class ParallelZarrWriter:
    """
    A parallel zarr writer that distributes I/O across multiple processes.
    """
    def __init__(self, zarr_path: str, num_workers: int = 4, verbose: bool = False):
        """
        Initialize a parallel zarr writer.
        
        Args:
            zarr_path: Path to the zarr store
            num_workers: Number of worker processes to use
            verbose: Enable verbose output
        """
        self.zarr_path = zarr_path
        self.num_workers = num_workers
        self.verbose = verbose
        self.logger = logger
        
        # Shared array metadata
        self.arrays: Dict[str, Dict[str, Any]] = {}
        
        # Create per-worker queues
        self.queues = [multiprocessing.Queue() for _ in range(num_workers)]
        self.processes = []
        self.stopping = False
        
        # Start worker processes
        self._start_workers()
        
        if verbose:
            self.logger.info(f"Initialized ParallelZarrWriter with {num_workers} workers for {zarr_path}")
    
    def _start_workers(self):
        """Start the worker processes"""
        for worker_id in range(self.num_workers):
            p = multiprocessing.Process(
                target=self._worker_process,
                args=(worker_id, self.queues[worker_id], self.zarr_path, self.verbose)
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
            
            if self.verbose:
                self.logger.info(f"Started worker process {worker_id} (PID: {p.pid})")
    
    @staticmethod
    def _worker_process(worker_id: int, work_queue: multiprocessing.Queue, 
                       zarr_path: str, verbose: bool):
        """
        Worker process function that performs actual writes to the zarr store.
        
        Args:
            worker_id: ID of this worker
            work_queue: Queue for work items
            zarr_path: Path to zarr store
            verbose: Enable verbose logging
        """
        # Each process opens its own connection to the zarr store
        start_time = time.time()
        if verbose:
            print(f"Worker {worker_id}: Initializing and opening zarr store at {zarr_path}")
        
        # Wait for the zarr store to exist before trying to open it
        while not os.path.exists(zarr_path) and time.time() - start_time < 60:
            time.sleep(0.1)
        
        if not os.path.exists(zarr_path):
            print(f"Worker {worker_id}: Error - Zarr store not found at {zarr_path}")
            return
            
        try:
            zarr_store = zarr.open(zarr_path, mode='a')
            
            # Track open arrays to avoid reopening
            open_arrays = {}
            
            if verbose:
                print(f"Worker {worker_id}: Successfully opened zarr store")
                
            patch_count = 0
            
            while True:
                try:
                    # Get work from queue with timeout
                    item = work_queue.get(timeout=5)
                    
                    # Check for sentinel
                    if item is None:
                        if verbose:
                            print(f"Worker {worker_id}: Received shutdown signal")
                        break
                        
                    # Unpack work item
                    array_path, index, data = item
                    
                    try:
                        # Get or open the array
                        if array_path not in open_arrays:
                            # Parse path to navigate zarr hierarchy
                            parts = array_path.split('/')
                            current = zarr_store
                            
                            # Navigate to the correct group/array
                            for i, part in enumerate(parts[:-1]):
                                if part:  # Skip empty parts from leading/trailing/double slashes
                                    if part not in current:
                                        if verbose:
                                            print(f"Worker {worker_id}: Group '{part}' not found, available: {list(current.keys())}")
                                        # Try next level
                                        current = current.create_group(part)
                                    else:
                                        current = current[part]
                            
                            # Get the actual array
                            array_name = parts[-1]
                            if array_name in current:
                                open_arrays[array_path] = current[array_name]
                            else:
                                if verbose:
                                    print(f"Worker {worker_id}: Array '{array_name}' not found, available: {list(current.keys())}")
                                # Skip this item
                                continue
                                
                        # Get the array
                        array = open_arrays[array_path]
                        
                        # Write the data
                        array[index] = data
                        patch_count += 1
                        
                        if verbose and (patch_count < 5 or patch_count % 100 == 0):
                            print(f"Worker {worker_id}: Wrote patch {patch_count} to {array_path}[{index}], shape={data.shape}")
                            
                    except Exception as e:
                        print(f"Worker {worker_id}: Error writing to array {array_path}[{index}]: {e}")
                        if verbose:
                            traceback.print_exc()
                        
                except Empty:
                    # Just a timeout, check if we should exit
                    continue
                except Exception as e:
                    print(f"Worker {worker_id}: Error in main loop: {e}")
                    if verbose:
                        traceback.print_exc()
            
            if verbose:
                print(f"Worker {worker_id}: Exiting after writing {patch_count} patches")
                
        except Exception as e:
            print(f"Worker {worker_id}: Critical error: {e}")
            traceback.print_exc()
    
    def _get_next_queue(self) -> int:
        """Simple round-robin queue selection"""
        # Just use modulo to distribute across queues
        # This could be enhanced with smarter load balancing
        next_id = getattr(self, '_next_queue_id', 0)
        self._next_queue_id = (next_id + 1) % self.num_workers
        return next_id
    
    def write_patch(self, array_path: str, index: int, data: np.ndarray):
        """
        Write a patch to a zarr array using a worker process.
        
        Args:
            array_path: Path to the array within the zarr store (e.g., "rank_0/patches/segmentation")
            index: Index to write at
            data: Data to write
        """
        # Choose a worker using round-robin
        worker_id = self._get_next_queue()
        
        # Send to that worker's queue
        self.queues[worker_id].put((array_path, index, data))
    
    def register_array(self, array_path: str, shape: Tuple, dtype: np.dtype, 
                       chunks: Optional[Tuple] = None):
        """
        Register an array's metadata (not required, for future enhancements).
        
        Args:
            array_path: Path to the array in the zarr hierarchy
            shape: Array shape
            dtype: Array data type
            chunks: Chunk size (optional)
        """
        self.arrays[array_path] = {
            'shape': shape,
            'dtype': dtype,
            'chunks': chunks
        }
    
    def shutdown(self):
        """Shutdown all worker processes gracefully"""
        if self.stopping:
            return
            
        self.stopping = True
        
        if self.verbose:
            self.logger.info("Shutting down parallel zarr writer...")
            
        # Send sentinel to each worker
        for q in self.queues:
            q.put(None)
            
        # Wait for processes to terminate
        for i, p in enumerate(self.processes):
            try:
                p.join(timeout=5)
                if p.is_alive():
                    self.logger.warning(f"Worker {i} did not terminate gracefully, forcing termination")
                    p.terminate()
            except Exception as e:
                self.logger.error(f"Error shutting down worker {i}: {e}")
                
        # Clear process list
        self.processes = []
        
        if self.verbose:
            self.logger.info("Parallel zarr writer shutdown complete")


def zarr_writer_worker(temp_storage, work_queue, worker_id, verbose=False):
    """
    Worker thread that writes patches to ZarrTempStorage.
    
    This version acts as an intermediary, taking items from the queue
    and forwarding them to the ParallelZarrWriter.
    
    Args:
        temp_storage: ZarrTempStorage instance
        work_queue: Queue for writer tasks
        worker_id: Unique ID for this worker
        verbose: Enable verbose output
    """
    try:
        while True:
            # Get an item from the queue
            item = work_queue.get()
            
            # Check for sentinel value to terminate
            if item is None:
                if verbose:
                    print(f"Writer {worker_id} received termination signal")
                work_queue.task_done()
                break
                
            # Unpack the item
            patch, position, target_name = item
            
            # Get position info
            z, y, x = position
            
            # Store patch in zarr
            try:
                idx = temp_storage.store_patch(patch, (z, y, x), target_name)
                
                # Log patch details
                if verbose and idx < 3:
                    print(f"Writer {worker_id}: Wrote patch with shape {patch.shape} at index {idx} for {target_name}")
                elif verbose and idx % 1000 == 0:
                    print(f"Writer {worker_id}: Progress - wrote patch {idx} for {target_name}")
                    
            except Exception as e:
                print(f"Error storing patch in zarr: {str(e)}")
                
            # Mark task as done
            work_queue.task_done()
            
    except Exception as e:
        print(f"Error in zarr writer thread {worker_id}: {str(e)}")
        if 'patch' in locals():
            print(f"Patch shape: {patch.shape}")
            print(f"Patch dtype: {patch.dtype}")
        import traceback
        traceback.print_exc()
        work_queue.task_done()