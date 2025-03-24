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
