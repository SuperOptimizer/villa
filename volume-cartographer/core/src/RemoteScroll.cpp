#include "vc/core/util/RemoteScroll.hpp"

#include <cstdio>
#include <fstream>

#include "vc/core/util/LoadJson.hpp"

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

    // If the very first request has an auth error, bail out early
    if (volList.authError) {
        info.authError = true;
        info.authErrorMessage = volList.errorMessage;
        std::fprintf(stderr, "[RemoteScroll] Auth error: %s\n", volList.errorMessage.c_str());
        return info;
    }

    for (const auto& name : volList.prefixes) {
        std::fprintf(stderr, "[RemoteScroll]   volume: %s\n", name.c_str());
        info.volumeNames.push_back(name);
    }

    // Probe paths/ first (full volpkg format)
    std::fprintf(stderr, "[RemoteScroll] Probing %s/paths/\n", baseUrl.c_str());
    auto pathsList = cache::s3ListObjects(baseUrl + "/paths/", auth);
    if (!pathsList.prefixes.empty()) {
        info.segmentSource = RemoteSegmentSource::Paths;
        for (const auto& name : pathsList.prefixes) {
            std::fprintf(stderr, "[RemoteScroll]   path segment: %s\n", name.c_str());
            info.segmentIds.push_back(name);
        }
    } else {
        // Fall back to segments/ (lite format)
        std::fprintf(stderr, "[RemoteScroll] Probing %s/segments/\n", baseUrl.c_str());
        auto segList = cache::s3ListObjects(baseUrl + "/segments/", auth);
        info.segmentSource = RemoteSegmentSource::Segments;
        for (const auto& name : segList.prefixes) {
            std::fprintf(stderr, "[RemoteScroll]   segment: %s\n", name.c_str());
            info.segmentIds.push_back(name);
        }
    }

    std::fprintf(stderr, "[RemoteScroll] Found %zu volumes, %zu segments (source: %s)\n",
                 info.volumeNames.size(), info.segmentIds.size(),
                 info.segmentSource == RemoteSegmentSource::Paths ? "paths" : "segments");

    return info;
}

std::filesystem::path downloadRemoteSegment(
    const std::string& baseUrl,
    const std::string& segmentId,
    const std::filesystem::path& cacheDir,
    const cache::HttpAuth& auth,
    RemoteSegmentSource source)
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

    // Build remote base URL depending on source format
    std::string remoteBase;
    if (source == RemoteSegmentSource::Direct) {
        // External URL: baseUrl is already the parent of segment dirs
        remoteBase = baseUrl + "/" + segmentId + "/";
    } else if (source == RemoteSegmentSource::Paths) {
        // Full volpkg: paths/<segId>/<file>
        remoteBase = baseUrl + "/paths/" + segmentId + "/";
    } else {
        // Lite format: segments/<segId>/mesh/tifxyz/<file>
        remoteBase = baseUrl + "/segments/" + segmentId + "/mesh/tifxyz/";
    }

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

    // Patch meta.json if required fields are missing (safety net for lite format)
    auto metaPath = localDir / "meta.json";
    if (fs::exists(metaPath)) {
        try {
            std::ifstream ifs(metaPath);
            auto meta = nlohmann::json::parse(ifs);
            ifs.close();

            bool patched = false;
            if (!meta.contains("type")) {
                meta["type"] = "seg";
                patched = true;
            }
            if (!meta.contains("uuid")) {
                meta["uuid"] = segmentId;
                patched = true;
            }
            if (!meta.contains("format")) {
                meta["format"] = "tifxyz";
                patched = true;
            }

            if (patched) {
                std::fprintf(stderr, "[RemoteScroll]   Patched meta.json for segment %s\n",
                             segmentId.c_str());
                std::ofstream ofs(metaPath);
                ofs << meta.dump(2);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[RemoteScroll]   Warning: failed to patch meta.json: %s\n",
                         e.what());
        }
    }

    return localDir;
}

}  // namespace vc
