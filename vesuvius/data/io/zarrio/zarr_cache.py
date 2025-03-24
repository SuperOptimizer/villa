import threading
from collections import OrderedDict
from utils.models.counter import AtomicCounter


class ZarrArrayLRUCache:
    """
    Optimized LRU Cache for zarr array access that caches chunks.
    This improves performance by reducing disk I/O operations for frequently accessed chunks.
    Uses fine-grained locking only for cache structure modifications.
    
    Features:
    - Can limit cache by number of entries or total memory usage
    - Automatically detects and uses zarr chunk boundaries for optimal caching
    - Thread-safe for concurrent access
    """
    def __init__(self, zarr_array, max_size=256, max_bytes_gb=4.0):
        """
        Initialize a new LRU cache for zarr array access with memory limits.
        
        Args:
            zarr_array: The zarr array to cache
            max_size: Maximum number of entries to cache (default: 256)
            max_bytes_gb: Maximum memory usage in GB (default: 4.0)
        """
        self.zarr_array = zarr_array
        self.max_size = max_size
        self._shape = zarr_array.shape
        self._dtype = zarr_array.dtype
        self._chunks = getattr(zarr_array, 'chunks', None)
        
        # Set maximum memory limit in bytes
        self.max_bytes = int(max_bytes_gb * 1024**3)  # Convert GB to bytes
        self.current_bytes = 0
        
        # Determine element size for memory calculations
        if hasattr(self._dtype, 'itemsize'):
            self.item_size_bytes = self._dtype.itemsize
        else:
            # Default to 4 bytes (like float32) if itemsize not available
            self.item_size_bytes = 4
        
        # Create an OrderedDict for the cache
        self.cache = OrderedDict()
        
        # Add a lock for cache structure modifications only
        # Using RLock to allow reentrant acquisition
        self.cache_lock = threading.RLock()
        
        # Use atomic counters for stats to avoid locking
        self.hits = AtomicCounter(0)
        self.misses = AtomicCounter(0)
        self.memory_limits_triggered = AtomicCounter(0)
        
        # Print some debug info about zarr array
        print(f"ZarrArrayLRUCache: Array shape {self._shape}, dtype {self._dtype}, " + 
              f"chunks {self._chunks if self._chunks else 'None'}")
        print(f"ZarrArrayLRUCache: Max entries {max_size}, max memory {max_bytes_gb:.1f} GB")
        
    @property
    def shape(self):
        return self._shape
        
    @property
    def dtype(self):
        return self._dtype
        
    @property
    def chunks(self):
        return self._chunks
    
    def _make_cache_key(self, key):
        """Convert a slice key to a hashable cache key."""
        if isinstance(key, slice):
            # Handle a single slice object
            return (key.start, key.stop, key.step)
        elif isinstance(key, tuple):
            # Handle tuple of indices/slices
            result = []
            for k in key:
                if isinstance(k, slice):
                    result.append((k.start, k.stop, k.step))
                else:
                    result.append(k)
            return tuple(result)
        # For any other type (like a single integer index)
        return key
    
    def _calculate_data_size(self, data):
        """Calculate memory size of data in bytes"""
        if hasattr(data, 'nbytes'):
            return data.nbytes
        elif hasattr(data, 'size'):
            # For arrays that have size but not nbytes
            return data.size * self.item_size_bytes
        else:
            # Fallback calculation
            try:
                import numpy as np
                if isinstance(data, (list, tuple)):
                    # Convert to numpy array for calculation
                    data_np = np.array(data)
                    return data_np.nbytes
                return 0  # Can't determine size
            except Exception:
                return 0  # Can't determine size
    
    def _enforce_memory_limit(self):
        """Remove items from cache until memory usage is below limit"""
        while self.current_bytes > self.max_bytes and len(self.cache) > 0:
            # Remove least recently used item
            key, data = self.cache.popitem(last=False)
            # Subtract its size from the total
            data_size = self._calculate_data_size(data)
            self.current_bytes -= data_size
            self.memory_limits_triggered.increment_and_get()
            
    def __getitem__(self, key):
        """Get an item from the cache if it exists, otherwise get it from the zarr array.
        
        Uses chunk-aware slicing to reduce memory usage when the zarr array has a chunking pattern.
        """
        cache_key = self._make_cache_key(key)
        
        # First check without lock for better concurrency
        if cache_key in self.cache:
            # We need a lock to modify the OrderedDict structure
            with self.cache_lock:
                if cache_key in self.cache:  # Re-check after acquiring lock
                    # Cache hit - move to end (most recently used)
                    data = self.cache[cache_key]
                    self.cache.move_to_end(cache_key)
                    self.hits.increment_and_get()
                    return data
        
        # Cache miss - get from zarr array
        self.misses.increment_and_get()
        data = self.zarr_array[key]
        
        # Update cache but only for reasonably sized arrays
        data_size = self._calculate_data_size(data)
        # Skip caching if data is too large (more than 25% of max allowed)
        if data_size > self.max_bytes * 0.25:
            return data
            
        with self.cache_lock:
            # Check if the key was added while we were getting the data
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                return self.cache[cache_key]
            
            # Check if adding this would exceed our memory limit
            if self.current_bytes + data_size > self.max_bytes:
                # Remove items until we have enough space
                self._enforce_memory_limit()
                
            # If we still don't have enough space, don't cache this item
            if self.current_bytes + data_size > self.max_bytes:
                return data
                
            # If cache is full by count, remove least recently used item
            if len(self.cache) >= self.max_size:
                removed_key, removed_data = self.cache.popitem(last=False)
                removed_size = self._calculate_data_size(removed_data)
                self.current_bytes -= removed_size
                
            # Add to cache
            self.cache[cache_key] = data
            self.current_bytes += data_size
        
        return data
        
    def __setitem__(self, key, value):
        """Set an item in the cache and in the zarr array."""
        # Update the zarr array
        self.zarr_array[key] = value
        
        # Update the cache
        cache_key = self._make_cache_key(key)
        data_size = self._calculate_data_size(value)
        
        # Skip caching if data is too large (more than 25% of max allowed)
        if data_size > self.max_bytes * 0.25:
            return
            
        with self.cache_lock:
            # If this key is already in cache, remove its size first
            if cache_key in self.cache:
                old_data = self.cache[cache_key]
                old_size = self._calculate_data_size(old_data)
                self.current_bytes -= old_size
            
            # Check if adding this would exceed our memory limit
            if self.current_bytes + data_size > self.max_bytes:
                # Remove items until we have enough space
                self._enforce_memory_limit()
                
            # If we still don't have enough space, don't cache this item
            if self.current_bytes + data_size > self.max_bytes:
                return
                
            # If cache is full by count, remove least recently used item
            if len(self.cache) >= self.max_size:
                removed_key, removed_data = self.cache.popitem(last=False)
                removed_size = self._calculate_data_size(removed_data)
                self.current_bytes -= removed_size
                
            # Add to cache (or update existing)
            self.cache[cache_key] = value
            self.current_bytes += data_size
            
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            
    def get_stats(self):
        """Get cache statistics."""
        # Get atomic counter values
        hits = self.hits.get()
        misses = self.misses.get()
        memory_limits_triggered = self.memory_limits_triggered.get()
        
        with self.cache_lock:  # Just for cache structure access
            current_size = len(self.cache)
            memory_mb = self.current_bytes / (1024**2)
            memory_limit_mb = self.max_bytes / (1024**2)
            
            # Calculate stats
            total = hits + misses
            hit_rate = hits / total * 100 if total > 0 else 0
            
            return {
                'hits': hits,
                'misses': misses,
                'total': total,
                'hit_rate': hit_rate,
                'cache_entries': current_size,
                'max_entries': self.max_size,
                'memory_used_mb': round(memory_mb, 2),
                'memory_limit_mb': round(memory_limit_mb, 2),
                'memory_used_percent': round((self.current_bytes / self.max_bytes) * 100, 1) if self.max_bytes > 0 else 0,
                'memory_limits_triggered': memory_limits_triggered
            }
        
    def clear(self):
        """Clear the cache."""
        with self.cache_lock:
            self.cache.clear()
            # Create new atomic counters instead of resetting
            # as there's no direct way to reset them
            self.hits = AtomicCounter(0)
            self.misses = AtomicCounter(0)
            self.memory_limits_triggered = AtomicCounter(0)
            
    def flush_all(self):
        """
        Force all cached items to be flushed to disk.
        This is particularly important before cleanup to ensure all data is written.
        """
        # For zarr arrays, we just need to ensure the store is synchronized
        # First check if the zarr array has a store attribute
        if hasattr(self.zarr_array, 'store') and hasattr(self.zarr_array.store, 'flush'):
            self.zarr_array.store.flush()
        
        # Then check if there's a synchronizer that needs flushing
        if hasattr(self.zarr_array, 'store') and hasattr(self.zarr_array.store, 'synchronizer') and \
           self.zarr_array.store.synchronizer is not None:
            self.zarr_array.store.synchronizer.flush()
        
        # Finally, if the array itself has a flush method, call it
        if hasattr(self.zarr_array, 'flush'):
            self.zarr_array.flush()
            
        # Clear our cache to help with memory usage
        with self.cache_lock:
            self.cache.clear()