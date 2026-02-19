#pragma once

#include <filesystem>
#include <string>

namespace vc::cache {

// Optional auth config for HTTP requests (e.g. AWS SigV4 signing).
struct HttpAuth {
    bool awsSigv4 = false;
    std::string region;  // e.g. "us-east-1"
};

struct RemoteZarrInfo {
    std::string url;
    std::filesystem::path stagingDir;  // local dir with .zarray files
    std::string delimiter = ".";
    int numLevels = 0;
};

// Fetch zarr metadata from a remote URL, write to local staging dir.
// Probes 0/.zarray, 1/.zarray, ... until 404.
// Creates: <stagingRoot>/<volumeId>/.zgroup, <volumeId>/0/.zarray, etc.
// Also synthesizes meta.json with dimensions from level 0's shape.
RemoteZarrInfo fetchRemoteZarrMetadata(
    const std::string& url,
    const std::filesystem::path& stagingRoot,
    const HttpAuth& auth = {});

// Fetch URL body as string. Empty on failure.
std::string httpGetString(const std::string& url, const HttpAuth& auth = {});

}  // namespace vc::cache
