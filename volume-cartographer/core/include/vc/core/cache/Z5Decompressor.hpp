#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "ChunkData.hpp"
#include "ChunkKey.hpp"

// Forward declarations — avoid pulling in z5 headers
namespace z5 {
class Dataset;
}

namespace vc::cache {

// Create a DecompressFn that uses z5::Dataset's decompressor.
//
// The returned function decompresses raw bytes (read from disk or network)
// into a ChunkData containing the voxel values. Handles uint8 and uint16
// datasets, with optional uint16→uint8 downsampling (divide by 257).
//
// datasets: one z5::Dataset* per pyramid level (index = level).
// The dataset is used to determine chunk shape, dtype, and decompressor.
// Pointers must remain valid for the lifetime of the returned function.
DecompressFn makeZ5Decompressor(const std::vector<z5::Dataset*>& datasets);

// Convenience overload for a single dataset (level 0 only).
DecompressFn makeZ5Decompressor(z5::Dataset* ds);

}  // namespace vc::cache
