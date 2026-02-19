#include "vc/core/cache/ChunkSource.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::cache {

// =============================================================================
// FileSystemChunkSource
// =============================================================================

FileSystemChunkSource::FileSystemChunkSource(
    const std::filesystem::path& zarrRoot,
    const std::string& delimiter)
    : root_(zarrRoot), delimiter_(delimiter)
{
    discoverLevels();
}

FileSystemChunkSource::FileSystemChunkSource(
    const std::filesystem::path& zarrRoot,
    const std::string& delimiter,
    std::vector<LevelMeta> levels)
    : root_(zarrRoot), delimiter_(delimiter), levels_(std::move(levels))
{
}

void FileSystemChunkSource::discoverLevels()
{
    // Discover pyramid levels by scanning for numbered subdirectories
    // with .zarray metadata files.
    std::vector<int> levelNums;
    for (auto& entry : std::filesystem::directory_iterator(root_)) {
        if (!entry.is_directory()) continue;
        auto name = entry.path().filename().string();
        // Check if directory name is a number
        bool isNum = !name.empty() &&
                     std::all_of(name.begin(), name.end(), ::isdigit);
        if (isNum) {
            levelNums.push_back(std::stoi(name));
        }
    }
    std::sort(levelNums.begin(), levelNums.end());

    levels_.clear();
    for (int lvl : levelNums) {
        auto zarrayPath = root_ / std::to_string(lvl) / ".zarray";
        if (!std::filesystem::exists(zarrayPath)) continue;

        std::ifstream f(zarrayPath);
        if (!f.is_open()) continue;

        nlohmann::json meta;
        try {
            f >> meta;
        } catch (...) {
            continue;
        }

        LevelMeta lm{};
        if (meta.contains("shape") && meta["shape"].is_array()) {
            auto& s = meta["shape"];
            if (s.size() >= 3) {
                lm.shape = {s[0].get<int>(), s[1].get<int>(), s[2].get<int>()};
            }
        }
        if (meta.contains("chunks") && meta["chunks"].is_array()) {
            auto& c = meta["chunks"];
            if (c.size() >= 3) {
                lm.chunkShape = {
                    c[0].get<int>(), c[1].get<int>(), c[2].get<int>()};
            }
        }
        levels_.push_back(lm);
    }
}

std::filesystem::path FileSystemChunkSource::chunkPath(
    const ChunkKey& key) const
{
    // zarr v2: <root>/<level>/<iz>.<iy>.<ix>
    std::string name = std::to_string(key.iz) + delimiter_ +
                       std::to_string(key.iy) + delimiter_ +
                       std::to_string(key.ix);
    return root_ / std::to_string(key.level) / name;
}

std::vector<uint8_t> FileSystemChunkSource::fetch(const ChunkKey& key)
{
    auto path = chunkPath(key);

    // Log first few fetches and all failures for diagnostics
    static std::atomic<int> fetchCount{0};
    int n = fetchCount.fetch_add(1, std::memory_order_relaxed);
    if (n < 5) {
        std::fprintf(stderr, "[TILED] ChunkSource::fetch #%d: lvl=%d (%d,%d,%d) path=%s\n",
                     n, key.level, key.iz, key.iy, key.ix, path.c_str());
    }

    // Use POSIX I/O for performance (matches existing ChunkCache pattern)
    int fd = ::open(path.c_str(), O_RDONLY | O_NOATIME);
    if (fd < 0) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (n < 5) {
                std::fprintf(stderr, "[TILED] ChunkSource::fetch: FAILED to open %s (errno=%d)\n",
                             path.c_str(), errno);
            }
            return {};
        }
    }

    struct stat sb;
    if (::fstat(fd, &sb) != 0) {
        ::close(fd);
        return {};
    }

    auto fileSize = static_cast<size_t>(sb.st_size);
    if (fileSize == 0) {
        ::close(fd);
        return {};
    }

    std::vector<uint8_t> buf(fileSize);
    size_t total = 0;
    while (total < fileSize) {
        ssize_t n = ::read(fd, buf.data() + total, fileSize - total);
        if (n <= 0) {
            ::close(fd);
            return {};
        }
        total += static_cast<size_t>(n);
    }
    ::close(fd);
    return buf;
}

bool FileSystemChunkSource::exists(const ChunkKey& key) const
{
    return std::filesystem::exists(chunkPath(key));
}

int FileSystemChunkSource::numLevels() const
{
    return static_cast<int>(levels_.size());
}

std::array<int, 3> FileSystemChunkSource::chunkShape(int level) const
{
    if (level < 0 || level >= static_cast<int>(levels_.size()))
        return {0, 0, 0};
    return levels_[level].chunkShape;
}

std::array<int, 3> FileSystemChunkSource::levelShape(int level) const
{
    if (level < 0 || level >= static_cast<int>(levels_.size()))
        return {0, 0, 0};
    return levels_[level].shape;
}

// =============================================================================
// HttpChunkSource
// =============================================================================

#ifdef VC_USE_CURL
#include <curl/curl.h>

static size_t curlWriteCallback(
    char* ptr, size_t size, size_t nmemb, void* userdata)
{
    auto* vec = static_cast<std::vector<uint8_t>*>(userdata);
    size_t bytes = size * nmemb;
    vec->insert(vec->end(), ptr, ptr + bytes);
    return bytes;
}
#endif

HttpChunkSource::HttpChunkSource(
    const std::string& baseUrl,
    const std::string& delimiter,
    std::vector<LevelMeta> levels,
    HttpAuth auth)
    : baseUrl_(baseUrl), delimiter_(delimiter), levels_(std::move(levels)), auth_(std::move(auth))
{
    // Remove trailing slash from base URL
    while (!baseUrl_.empty() && baseUrl_.back() == '/') {
        baseUrl_.pop_back();
    }

#ifdef VC_USE_CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
}

HttpChunkSource::~HttpChunkSource()
{
#ifdef VC_USE_CURL
    curl_global_cleanup();
#endif
}

std::string HttpChunkSource::chunkUrl(const ChunkKey& key) const
{
    return baseUrl_ + "/" + std::to_string(key.level) + "/" +
           std::to_string(key.iz) + delimiter_ + std::to_string(key.iy) +
           delimiter_ + std::to_string(key.ix);
}

std::vector<uint8_t> HttpChunkSource::fetch(const ChunkKey& key)
{
#ifdef VC_USE_CURL
    std::string url = chunkUrl(key);
    std::vector<uint8_t> response;

    CURL* curl = curl_easy_init();
    if (!curl) return {};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);  // thread-safe
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    if (auth_.awsSigv4) {
        std::string sigv4 = "aws:amz:" + auth_.region + ":s3";
        curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, sigv4.c_str());
    }

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) return {};
    return response;
#else
    (void)key;
    return {};
#endif
}

bool HttpChunkSource::exists(const ChunkKey& key) const
{
#ifdef VC_USE_CURL
    std::string url = chunkUrl(key);

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    if (auth_.awsSigv4) {
        std::string sigv4 = "aws:amz:" + auth_.region + ":s3";
        curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, sigv4.c_str());
    }

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    return res == CURLE_OK;
#else
    (void)key;
    return false;
#endif
}

int HttpChunkSource::numLevels() const
{
    return static_cast<int>(levels_.size());
}

std::array<int, 3> HttpChunkSource::chunkShape(int level) const
{
    if (level < 0 || level >= static_cast<int>(levels_.size()))
        return {0, 0, 0};
    return levels_[level].chunkShape;
}

std::array<int, 3> HttpChunkSource::levelShape(int level) const
{
    if (level < 0 || level >= static_cast<int>(levels_.size()))
        return {0, 0, 0};
    return levels_[level].shape;
}

}  // namespace vc::cache
