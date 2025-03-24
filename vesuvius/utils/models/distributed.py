def dataset_set_distributed(self, rank: int, world_size: int):
    """
    Configure this dataset for distributed data parallel processing.

    This method divides the dataset's patch positions among processes,
    so each process works on a different subset of patches.

    Args:
        rank: The rank of this process (0 to world_size-1)
        world_size: Total number of processes
    """
    if world_size <= 1 or rank < 0 or rank >= world_size:
        # No need to distribute or invalid configuration
        return

    # Get total number of positions
    total_positions = len(self.all_positions)

    if self.verbose:
        print(f"Rank {rank}: Distributing {total_positions} positions among {world_size} processes")

    # Calculate positions for this rank (simple chunking)
    # Each rank gets approximately total_positions / world_size positions
    positions_per_rank = total_positions // world_size
    remainder = total_positions % world_size

    # Calculate start and end indices
    # Ranks with ID < remainder get one extra position
    start_idx = rank * positions_per_rank + min(rank, remainder)
    end_idx = start_idx + positions_per_rank + (1 if rank < remainder else 0)

    # Take only the positions assigned to this rank
    self.all_positions = self.all_positions[start_idx:end_idx]

    if self.verbose:
        print(f"Rank {rank}: Processing {len(self.all_positions)} positions ({start_idx} to {end_idx - 1})")