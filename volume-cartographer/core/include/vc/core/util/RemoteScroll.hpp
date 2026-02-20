#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "vc/core/cache/HttpMetadataFetcher.hpp"

namespace vc {

struct RemoteScrollInfo {
    std::string baseUrl;                    // HTTPS base URL of the scroll root
    std::vector<std::string> volumeNames;   // e.g. ["20230205180739.zarr"]
    std::vector<std::string> segmentIds;    // e.g. ["20230205180739"]
    cache::HttpAuth auth;
};

// Probe a remote scroll root URL for volumes/ and segments/ subdirectories.
// Returns discovered volume names and segment IDs.
// If the URL doesn't look like a scroll (no volumes/ or segments/), returns
// empty lists.
RemoteScrollInfo discoverRemoteScroll(const std::string& httpsUrl, const cache::HttpAuth& auth);

// Download a single remote segment's tifxyz files to a local cache directory.
// Downloads: segments/<segId>/mesh/tifxyz/{meta.json, x.tif, y.tif, z.tif}
// to: cacheDir/segments/<segId>/
// Skips download if all 4 files already exist locally.
// Returns the local directory containing the downloaded segment files.
std::filesystem::path downloadRemoteSegment(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth);

}  // namespace vc
