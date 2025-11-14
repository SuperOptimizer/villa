#include "vc/core/types/ChunkedTensor.hpp"
#include "vc/core/types/ChunkedTensor_impl.hpp"

#include <iostream>

// ============================================================================
//  print_accessor_stats implementation
// ============================================================================

void print_accessor_stats()
{
    std::cout << "acc miss/total " << miss << " " << total << " " << static_cast<double>(miss)/total << std::endl;
    std::cout << "chunk compute overhead/total " << chunk_compute_collisions << " " << chunk_compute_total << " " << static_cast<double>(chunk_compute_collisions)/chunk_compute_total << std::endl;
}

// ============================================================================
//  Explicit template instantiations for passTroughComputor
// ============================================================================

// Chunked3d
template class Chunked3d<uint8_t, passTroughComputor>;

// Chunked3dAccessor
template class Chunked3dAccessor<uint8_t, passTroughComputor>;

// CachedChunked3dInterpolator
template class CachedChunked3dInterpolator<uint8_t, passTroughComputor>;
