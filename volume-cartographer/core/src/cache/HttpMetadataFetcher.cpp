#include "vc/core/cache/HttpMetadataFetcher.hpp"

#include <cstdio>
#include <fstream>
#include <optional>
#include <nlohmann/json.hpp>

#ifdef VC_USE_CURL
#include <curl/curl.h>
#endif

namespace vc::cache {

// ---- curl helpers -----------------------------------------------------------

#ifdef VC_USE_CURL
static size_t stringWriteCallback(
    char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* str = static_cast<std::string*>(userdata);
    str->append(ptr, size * nmemb);
    return size * nmemb;
}
#endif

std::string httpGetString(const std::string& url, const HttpAuth& auth)
{
#ifdef VC_USE_CURL
    std::string response;

    CURL* curl = curl_easy_init();
    if (!curl) return {};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stringWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    struct curl_slist* headers = nullptr;
    std::string userpwd;
    if (auth.awsSigv4) {
        std::string sigv4 = "aws:amz:" + auth.region + ":s3";
        curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, sigv4.c_str());
        userpwd = auth.accessKey + ":" + auth.secretKey;
        curl_easy_setopt(curl, CURLOPT_USERPWD, userpwd.c_str());
        if (!auth.sessionToken.empty()) {
            std::string hdr = "x-amz-security-token: " + auth.sessionToken;
            headers = curl_slist_append(headers, hdr.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        }
    }

    CURLcode res = curl_easy_perform(curl);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) return {};
    return response;
#else
    (void)url;
    (void)auth;
    return {};
#endif
}

// ---- metadata fetcher -------------------------------------------------------

static std::string deriveVolumeId(const std::string& url)
{
    // Extract last non-empty path component from URL
    std::string u = url;
    while (!u.empty() && u.back() == '/') u.pop_back();
    auto pos = u.rfind('/');
    if (pos != std::string::npos) {
        return u.substr(pos + 1);
    }
    return u;
}

static void writeFile(const std::filesystem::path& path, const std::string& content)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
}

static std::string readFile(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// Check if staging dir already has valid cached metadata (meta.json + at least 0/.zarray).
// If so, return the info without hitting the network.
static std::optional<RemoteZarrInfo> tryLoadCachedMetadata(
    const std::string& baseUrl,
    const std::filesystem::path& stagingDir)
{
    namespace fs = std::filesystem;

    auto metaPath = stagingDir / "meta.json";
    auto level0Zarray = stagingDir / "0" / ".zarray";
    if (!fs::exists(metaPath) || !fs::exists(level0Zarray)) {
        return std::nullopt;
    }

    // Count cached levels
    int numLevels = 0;
    for (int lvl = 0; lvl < 20; lvl++) {
        if (fs::exists(stagingDir / std::to_string(lvl) / ".zarray")) {
            numLevels++;
        } else {
            break;
        }
    }

    if (numLevels == 0) return std::nullopt;

    // Parse delimiter from level 0
    std::string delimiter = ".";
    auto zarray0 = readFile(level0Zarray);
    if (!zarray0.empty()) {
        try {
            auto j = nlohmann::json::parse(zarray0);
            if (j.contains("dimension_separator")) {
                delimiter = j["dimension_separator"].get<std::string>();
            }
        } catch (...) {}
    }

    std::fprintf(stderr, "[REMOTE] Using cached metadata from %s (%d levels)\n",
                 stagingDir.c_str(), numLevels);

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels
    };
}

RemoteZarrInfo fetchRemoteZarrMetadata(
    const std::string& url,
    const std::filesystem::path& stagingRoot,
    const HttpAuth& auth)
{
    // Normalize URL: strip trailing slashes
    std::string baseUrl = url;
    while (!baseUrl.empty() && baseUrl.back() == '/') baseUrl.pop_back();

    std::string volumeId = deriveVolumeId(baseUrl);
    auto stagingDir = stagingRoot / volumeId;

    // Clean up stale sibling .chunks dir from older code path
    {
        auto staleChunks = stagingRoot / (volumeId + ".chunks");
        std::error_code ec;
        if (std::filesystem::exists(staleChunks, ec)) {
            std::filesystem::remove_all(staleChunks, ec);
        }
    }

    // Try cached metadata first — avoids network round-trips on subsequent opens
    if (auto cached = tryLoadCachedMetadata(baseUrl, stagingDir)) {
        return *cached;
    }

    std::filesystem::create_directories(stagingDir);

    std::fprintf(stderr, "[REMOTE] Fetching metadata for %s -> %s\n",
                 baseUrl.c_str(), stagingDir.c_str());

    // Fetch .zgroup
    auto zgroup = httpGetString(baseUrl + "/.zgroup", auth);
    if (!zgroup.empty()) {
        writeFile(stagingDir / ".zgroup", zgroup);
    } else {
        // Synthesize minimal .zgroup
        writeFile(stagingDir / ".zgroup", R"({"zarr_format":2})");
    }

    // Fetch .zattrs (optional, may 404)
    auto zattrs = httpGetString(baseUrl + "/.zattrs", auth);
    if (!zattrs.empty()) {
        writeFile(stagingDir / ".zattrs", zattrs);
    }

    // Probe levels: 0, 1, 2, ... until 404
    std::string delimiter = ".";
    int numLevels = 0;
    nlohmann::json level0Meta;

    for (int lvl = 0; lvl < 20; lvl++) {
        std::string levelStr = std::to_string(lvl);
        auto zarray = httpGetString(baseUrl + "/" + levelStr + "/.zarray", auth);
        if (zarray.empty()) {
            std::fprintf(stderr, "[REMOTE] Level %d: no .zarray, stopping\n", lvl);
            break;
        }

        auto levelDir = stagingDir / levelStr;
        writeFile(levelDir / ".zarray", zarray);
        numLevels++;

        std::fprintf(stderr, "[REMOTE] Level %d: fetched .zarray (%zu bytes)\n",
                     lvl, zarray.size());

        // Parse level 0 for shape and delimiter
        if (lvl == 0) {
            try {
                level0Meta = nlohmann::json::parse(zarray);
                if (level0Meta.contains("dimension_separator")) {
                    delimiter = level0Meta["dimension_separator"].get<std::string>();
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[REMOTE] Warning: failed to parse level 0 .zarray: %s\n", e.what());
            }
        }
    }

    if (numLevels == 0) {
        throw std::runtime_error("No zarr levels found at " + baseUrl);
    }

    // Synthesize meta.json from level 0 shape
    int width = 0, height = 0, slices = 0;
    if (level0Meta.contains("shape") && level0Meta["shape"].is_array() &&
        level0Meta["shape"].size() >= 3) {
        // zarr shape is [z, y, x]
        slices = level0Meta["shape"][0].get<int>();
        height = level0Meta["shape"][1].get<int>();
        width  = level0Meta["shape"][2].get<int>();
    }

    nlohmann::json meta;
    meta["uuid"] = volumeId;
    meta["name"] = volumeId;
    meta["type"] = "vol";
    meta["width"] = width;
    meta["height"] = height;
    meta["slices"] = slices;
    meta["format"] = "zarr";
    meta["voxelsize"] = 0;
    meta["min"] = 0;
    meta["max"] = 255;

    writeFile(stagingDir / "meta.json", meta.dump(2));

    std::fprintf(stderr, "[REMOTE] Metadata complete: %d levels, shape=[%d, %d, %d] delimiter='%s'\n",
                 numLevels, slices, height, width, delimiter.c_str());

    return RemoteZarrInfo{
        .url = baseUrl,
        .stagingDir = stagingDir,
        .delimiter = delimiter,
        .numLevels = numLevels
    };
}

}  // namespace vc::cache
