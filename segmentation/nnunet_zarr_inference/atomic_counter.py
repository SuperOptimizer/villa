import atomics

class AtomicCounter:
    """
    Thread-safe counter implementation using the 'atomics' package.
    
    This provides fast, lock-free increment operations while maintaining
    accurate counts across threads.
    """
    def __init__(self, initial_value=0):
        """
        Initialize a new AtomicCounter with the given initial value.
        
        Args:
            initial_value: Starting value for the counter (default: 0)
        """
        # Create an atomic integer (4 bytes width for standard 32-bit int)
        self.counter = atomics.atomic(width=4, atype=atomics.INT)
        # Store the initial value directly - this will be the result of the first get()
        # and the next increment_and_get() will return initial_value + 1
        self.counter.store(initial_value)
        
    def increment_and_get(self):
        """
        Atomically increments by one and returns the new value.
        
        This operation is completely thread-safe and lock-free.
        
        Returns:
            The incremented value
        """
        # fetch_add returns the old value, add 1 to get the new value
        old_value = self.counter.fetch_add(1)
        return old_value + 1
        
    def get(self):
        """
        Get the current counter value.
        
        Returns:
            The current value
        """
        return self.counter.load()


# For testing
if __name__ == "__main__":
    import time
    from concurrent.futures import ThreadPoolExecutor
    
    # Create counter
    counter = AtomicCounter()
    
    # Function to increment counter in a thread
    def increment_task(thread_id, iterations):
        results = []
        for i in range(iterations):
            value = counter.increment_and_get()
            results.append(value)
            time.sleep(0.001)  # Small delay to increase interleaving
        return thread_id, results
    
    # Run with multiple threads
    num_threads = 5
    iterations_per_thread = 100
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(increment_task, i, iterations_per_thread) 
                  for i in range(num_threads)]
        
        # Collect results
        all_results = {}
        all_values = []
        for future in futures:
            thread_id, values = future.result()
            all_results[thread_id] = values
            all_values.extend(values)
            print(f"Thread {thread_id}: first value={values[0]}, last value={values[-1]}")
    
    # Check total count
    total_increments = num_threads * iterations_per_thread
    reported_count = counter.get()
    
    print(f"\nTotal expected increments: {total_increments}")
    print(f"Counter.get() reports: {reported_count}")
    
    # Verify uniqueness
    all_values.sort()
    duplicates = [all_values[i] for i in range(1, len(all_values)) 
                 if all_values[i] == all_values[i-1]]
    
    if duplicates:
        print(f"ERROR: Found {len(duplicates)} duplicate values!")
    else:
        print("No duplicate values found - each increment got a unique value")
        
    # Verify sequence completeness
    expected = set(range(1, total_increments + 1))
    actual = set(all_values)
    missing = expected - actual
    
    if missing:
        print(f"ERROR: Missing {len(missing)} values!")
    else:
        print(f"All values from 1 to {total_increments} present")