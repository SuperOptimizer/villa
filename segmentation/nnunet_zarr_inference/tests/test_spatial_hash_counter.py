import unittest
import sys
import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spatial_hash_counter import SpatialHashCounter

class TestSpatialHashCounter(unittest.TestCase):
    def test_basic_mapping(self):
        """Test basic position-to-index mapping"""
        # Create a counter with simple dimensions
        counter = SpatialHashCounter(10, 10, 10)
        
        # Check if positions map to expected indices
        positions = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (5, 6, 7)
        ]
        
        # Get indices for each position
        indices = [counter.position_to_index(pos) for pos in positions]
        
        # Verify we get increasing indices starting from 0
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[1], 1)
        self.assertEqual(indices[2], 2)
        self.assertEqual(indices[3], 3)
        self.assertEqual(indices[4], 4)
        
        # Verify reusing the same position returns the same index
        self.assertEqual(counter.position_to_index((0, 0, 0)), 0)
        self.assertEqual(counter.position_to_index((5, 6, 7)), 4)
        
    def test_index_uniqueness(self):
        """Test that each position maps to a unique index"""
        counter = SpatialHashCounter(10, 10, 10)
        
        # Create a set of positions
        positions = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (5, 6, 7),
            (9, 9, 9)
        ]
        
        # Get indices for all positions
        indices = [counter.position_to_index(pos) for pos in positions]
        
        # Check that all indices are unique
        self.assertEqual(len(indices), len(set(indices)), 
                         f"Indices should be unique, got: {indices}")
        
    def test_unique_positions(self):
        """Test the get_unique_positions method"""
        counter = SpatialHashCounter(10, 10, 10)
        
        # Create positions in random order
        positions = [
            (5, 6, 7),
            (0, 0, 0),
            (9, 9, 9),
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, 1)
        ]
        
        # Get sorted position-index pairs
        position_indices = counter.get_unique_positions(positions)
        
        # Check if positions are sorted by spatial order
        sorted_positions = [pos for pos, _ in position_indices]
        
        # Expected order is by z, then y, then x
        expected_order = [(0, 0, 0), (0, 0, 1), (0, 1, 0), 
                          (1, 0, 0), (5, 6, 7), (9, 9, 9)]
        
        self.assertEqual(sorted_positions, expected_order)
        
    def test_compute_total_positions(self):
        """Test the compute_total_positions method"""
        counter = SpatialHashCounter(10, 10, 10)
        
        # Initially should be 0 registered positions
        self.assertEqual(counter.compute_total_positions(), 0)
        
        # Register some positions
        positions = [(0, 0, 0), (1, 2, 3), (4, 5, 6)]
        for pos in positions:
            counter.position_to_index(pos)
            
        # Should now have 3 registered positions
        self.assertEqual(counter.compute_total_positions(), 3)
        
    def test_thread_safety(self):
        """Test thread safety of position registration"""
        counter = SpatialHashCounter(1000, 1000, 1000)
        
        # Number of threads and positions per thread
        num_threads = 10
        positions_per_thread = 1000
        
        # Track all registered indices across threads
        all_indices = []
        indices_lock = threading.Lock()
        
        def register_positions(thread_id):
            """Worker function to register positions from a thread"""
            thread_indices = []
            
            for i in range(positions_per_thread):
                # Generate position based on thread_id and i
                # Use a formula that ensures unique positions across threads
                pos = (thread_id, i // 100, i % 100)
                
                # Register the position
                idx = counter.position_to_index(pos)
                thread_indices.append(idx)
                
                # Small delay to increase thread interleaving
                if i % 100 == 0:
                    time.sleep(0.001)
            
            # Add indices to the global list
            with indices_lock:
                all_indices.extend(thread_indices)
            
            return thread_indices
        
        # Run threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_positions, thread_id) 
                      for thread_id in range(num_threads)]
            
            # Wait for all threads to complete
            results = [future.result() for future in futures]
        
        # Check total number of registered positions
        expected_total = num_threads * positions_per_thread
        self.assertEqual(counter.compute_total_positions(), expected_total, 
                        f"Expected {expected_total} registered positions")
        
        # Check for duplicate indices
        all_indices.sort()
        unique_indices = set(all_indices)
        
        self.assertEqual(len(all_indices), len(unique_indices), 
                        "There should be no duplicate indices")

if __name__ == "__main__":
    unittest.main()