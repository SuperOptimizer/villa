#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vc::cache {

// Optional auth config for HTTP requests (e.g. AWS SigV4 signing).
struct HttpAuth {
    bool awsSigv4 = false;
    std::string region;        // e.g. "us-east-1"
    std::string accessKey;     // AWS_ACCESS_KEY_ID
    std::string secretKey;     // AWS_SECRET_ACCESS_KEY
    std::string sessionToken;  // AWS_SESSION_TOKEN (optional, for STS)
};

struct RemoteZarrInfo {
    std::string url;
    std::filesystem::path stagingDir;  // local dir with .zarray files
    std::string delimiter = ".";
    int numLevels = 0;
};

// Result of S3 ListObjectsV2 with delimiter.
struct S3ListResult {
    std::vector<std::string> prefixes;  // subdirectory prefixes (CommonPrefixes)
    std::vector<std::string> objects;   // object keys (Contents)
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

// List objects under an S3 prefix using ListObjectsV2.
// httpsBaseUrl should be the bucket URL (https://bucket.s3.region.amazonaws.com/prefix/).
// Returns prefixes (subdirectories) and object keys.
S3ListResult s3ListObjects(const std::string& httpsBaseUrl, const HttpAuth& auth = {});

// Download a URL to a local file. Returns true on success.
// Uses atomic write (temp file + rename).
bool httpDownloadFile(const std::string& url, const std::filesystem::path& dest, const HttpAuth& auth = {});

}  // namespace vc::cache
