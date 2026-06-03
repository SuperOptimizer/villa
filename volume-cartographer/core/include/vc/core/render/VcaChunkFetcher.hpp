#pragma once

// Backend that serves 32^3 chunks from a local volume-compressor (.vca) archive.
// One mmap'd archive (VcaArchive) is shared by a per-LOD VcaChunkFetcher; fetch()
// decodes a single 32^3 atom via libvc. See libs/vc/vc.h and docs/SPEC.md upstream.

#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"

#include <filesystem>

namespace vc::render {

// Open a local .vca archive as an 8-LOD chunked pyramid of 32^3 chunks.
// Throws std::runtime_error if the file can't be mmap'd or vc_open rejects it.
OpenedChunkedZarr openVcaArchive(const std::filesystem::path& vcaPath);

} // namespace vc::render
