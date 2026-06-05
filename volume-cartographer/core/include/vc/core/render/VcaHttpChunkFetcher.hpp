#pragma once

// Backend that STREAMS a volume-compressor (.vca) archive from a remote HTTPS/S3
// URL without downloading it whole. libvc's streaming reader (vc_open_streaming)
// pulls byte ranges through a callback backed by VcaHttpBlockCache: a persistent,
// fixed-block range cache that range-GETs missing blocks via utils::HttpClient
// (AWS SigV4) and serves repeats from a local sparse file. See VcaChunkFetcher.hpp
// for the local-file equivalent; this is the remote, lazily-fetched variant.

#include "vc/core/render/ZarrChunkFetcher.hpp"   // OpenedChunkedZarr
#include "vc/core/util/RemoteAuth.hpp"            // vc::HttpAuth

#include <filesystem>
#include <string>

namespace vc::render {

// Open a remote .vca as an 8-LOD chunked pyramid of 32^3 chunks, streamed on
// demand. `httpsUrl` is the resolved https:// URL (s3:// already converted),
// `auth` the AWS creds, `cacheDir` where the persistent block cache lives.
// Throws std::runtime_error if the header can't be fetched or vc_open rejects it.
OpenedChunkedZarr openHttpVcaArchive(const std::string& httpsUrl,
                                     const vc::HttpAuth& auth,
                                     const std::filesystem::path& cacheDir);

} // namespace vc::render
