#include "vc/core/util/RemoteScroll.hpp"

#include <cstdio>

namespace vc {

RemoteScrollInfo discoverRemoteScroll(const std::string& httpsUrl, const cache::HttpAuth& auth)
{
    // Normalize: ensure trailing slash
    std::string baseUrl = httpsUrl;
    while (!baseUrl.empty() && baseUrl.back() == '/') baseUrl.pop_back();

    RemoteScrollInfo info;
    info.baseUrl = baseUrl;
    info.auth = auth;

    // Probe volumes/
    std::fprintf(stderr, "[RemoteScroll] Probing %s/volumes/\n", baseUrl.c_str());
    auto volList = cache::s3ListObjects(baseUrl + "/volumes/", auth);
    for (const auto& name : volList.prefixes) {
        std::fprintf(stderr, "[RemoteScroll]   volume: %s\n", name.c_str());
        info.volumeNames.push_back(name);
    }

    // Probe segments/
    std::fprintf(stderr, "[RemoteScroll] Probing %s/segments/\n", baseUrl.c_str());
    auto segList = cache::s3ListObjects(baseUrl + "/segments/", auth);
    for (const auto& name : segList.prefixes) {
        std::fprintf(stderr, "[RemoteScroll]   segment: %s\n", name.c_str());
        info.segmentIds.push_back(name);
    }

    std::fprintf(stderr, "[RemoteScroll] Found %zu volumes, %zu segments\n",
                 info.volumeNames.size(), info.segmentIds.size());

    return info;
}

std::filesystem::path downloadRemoteSegment(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth)
{
    namespace fs = std::filesystem;

    // Local destination: cacheDir/segments/<segId>/
    auto localDir = cacheDir / "segments" / segmentId;
    fs::create_directories(localDir);

    // Files to download from the remote tifxyz directory
    const std::vector<std::string> files = {"meta.json", "x.tif", "y.tif", "z.tif"};

    // Check if all files already exist
    bool allExist = true;
    for (const auto& f : files) {
        if (!fs::exists(localDir / f)) {
            allExist = false;
            break;
        }
    }

    if (allExist) {
        std::fprintf(stderr, "[RemoteScroll] Segment %s already cached at %s\n",
                     segmentId.c_str(), localDir.c_str());
        return localDir;
    }

    // Download each file
    // Remote path: baseUrl/segments/<segId>/mesh/tifxyz/<file>
    std::string remoteBase = baseUrl + "/segments/" + segmentId + "/mesh/tifxyz/";

    for (const auto& f : files) {
        auto localPath = localDir / f;
        if (fs::exists(localPath)) {
            std::fprintf(stderr, "[RemoteScroll]   %s already exists, skipping\n", f.c_str());
            continue;
        }

        std::string url = remoteBase + f;
        std::fprintf(stderr, "[RemoteScroll]   Downloading %s -> %s\n",
                     url.c_str(), localPath.c_str());

        if (!cache::httpDownloadFile(url, localPath, auth)) {
            std::fprintf(stderr, "[RemoteScroll]   FAILED to download %s\n", f.c_str());
            // Continue with other files — partial downloads are handled by the
            // caller checking if meta.json exists
        }
    }

    return localDir;
}

}  // namespace vc
