// vc_cache_prefetch — populate a remote volume's local mca cache without the GUI.
//
// Opens the remote volume (c3d/raw zarr, or a remote .mca archive) exactly like
// VC3D does, then walks every 256^3 region of the requested LOD levels in order,
// pulling each through the same fetch path the viewer uses: fetch native chunks
// (or compressed .mca blobs), encode/append into the per-volume volume.mca, done.
// A later VC3D session on the same volume serves everything from disk.
//
//   vc_cache_prefetch s3://bucket/vol.zarr/ --levels 5,4,3,2,1
//
// The cache lands where VC3D will look for it: /volpkgs/remote_cache or
// /ephemeral/remote_cache when those mounts exist, else --cache-root, else
// ~/.VC3D/remote_cache.

#include "vc/core/types/Volume.hpp"
#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/util/RemoteAuth.hpp"
#include "vc/core/util/Logging.hpp"

#include <boost/program_options.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>
#include <atomic>
#include <string>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace {

// VC3D's cache-root priority (see apps/VC3D/VCSettings.hpp): forced mounts win,
// then the explicit override, then the per-user default.
fs::path resolveCacheRoot(const std::string& override_)
{
    for (const char* mount : {"/volpkgs", "/ephemeral"}) {
        std::error_code ec;
        if (fs::is_directory(mount, ec)) {
            const fs::path p = fs::path(mount) / "remote_cache";
            fs::create_directories(p, ec);
            if (ec || !fs::is_directory(p)) {
                std::cerr << "error: " << mount << " exists but " << p
                          << " is not writable\n";
                std::exit(1);
            }
            return p;
        }
    }
    if (!override_.empty())
        return override_;
    const char* home = std::getenv("HOME");
    return fs::path(home ? home : ".") / ".VC3D" / "remote_cache";
}

std::vector<int> parseLevels(const std::string& spec, int numLevels)
{
    std::vector<int> levels;
    if (spec.empty()) {   // default: every level, coarsest first
        for (int l = numLevels - 1; l >= 0; --l)
            levels.push_back(l);
        return levels;
    }
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const int l = std::stoi(item);
        if (l < 0 || l >= numLevels)
            throw std::runtime_error("level " + item + " out of range (volume has " +
                                     std::to_string(numLevels) + " levels)");
        levels.push_back(l);
    }
    return levels;
}

std::string fmtBytes(double b)
{
    char buf[32];
    if (b >= 1e9) std::snprintf(buf, sizeof buf, "%.2f GB", b / 1e9);
    else if (b >= 1e6) std::snprintf(buf, sizeof buf, "%.1f MB", b / 1e6);
    else std::snprintf(buf, sizeof buf, "%.0f KB", b / 1e3);
    return buf;
}

} // namespace

int main(int argc, char** argv)
{
    po::options_description opts("vc_cache_prefetch options");
    // clang-format off
    opts.add_options()
        ("help,h", "show help")
        ("url", po::value<std::string>(), "remote volume URL (s3:// or https://; zarr root or .mca)")
        ("levels", po::value<std::string>()->default_value(""),
            "comma-separated LOD levels in prefetch order, e.g. 5,4,3,2,1 "
            "(default: all levels, coarsest first)")
        ("cache-root", po::value<std::string>()->default_value(""),
            "cache root override (ignored when /volpkgs or /ephemeral exists)")
        ("threads", po::value<int>()->default_value(0), "fetch worker threads (0 = default)")
        ("budget-gb", po::value<double>()->default_value(1.0), "resident decode-cache budget");
    // clang-format on
    po::positional_options_description pos;
    pos.add("url", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(opts).positional(pos).run(), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n" << opts << "\n";
        return 1;
    }
    if (vm.count("help") || !vm.count("url")) {
        std::cout << "usage: vc_cache_prefetch <url> [--levels 5,4,3,2,1]\n\n" << opts << "\n";
        return vm.count("help") ? 0 : 1;
    }

    const auto url = vm["url"].as<std::string>();
    const auto cacheRoot = resolveCacheRoot(vm["cache-root"].as<std::string>());
    std::error_code ec;
    fs::create_directories(cacheRoot, ec);

    // same credential chain as VC3D (libs3: profile/SSO/IMDS/env/INI).
    const auto auth = vc::loadAwsCredentials();

    std::shared_ptr<Volume> volume;
    try {
        volume = Volume::NewFromUrl(url, cacheRoot, auth);
    } catch (const std::exception& e) {
        std::cerr << "error: cannot open " << url << ": " << e.what() << "\n";
        return 1;
    }
    if (vm["threads"].as<int>() > 0)
        volume->setIOThreads(vm["threads"].as<int>());
    volume->setCacheBudget(static_cast<size_t>(vm["budget-gb"].as<double>() * (1ull << 30)));

    auto* cache = volume->chunkedCache();
    const int numLevels = cache->numLevels();
    std::cout << "volume " << volume->id() << "  levels=" << numLevels
              << "  cache=" << (cacheRoot / volume->id()).string() << "\n";

    // the persistent mca cache re-expresses levels as 16^3 blocks; anything else
    // means it did not engage (non-uint8 volume) and there is nothing to persist.
    if (cache->chunkShape(0)[0] != 16) {
        std::cerr << "error: mca cache did not engage for this volume "
                     "(non-uint8 dtype?) — nothing to prefetch\n";
        return 1;
    }

    std::vector<int> levels;
    try {
        levels = parseLevels(vm["levels"].as<std::string>(), numLevels);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    constexpr int kRegion = 256;
    constexpr int kBlocksPerRegion = 16;
    const auto t0 = std::chrono::steady_clock::now();

    const int nThreads = vm["threads"].as<int>() > 0 ? vm["threads"].as<int>() : 16;
    auto* cc = static_cast<vc::render::ChunkCache*>(cache);

    // Build ONE shard queue across all levels, in level order, with no barrier
    // between levels: workers flow from one level's shards straight into the next.
    // Each shard is downloaded once (parallel GET) and its inner chunks encoded
    // in parallel inside prefetchShardBlocking.
    struct Shard { int level, rz0, ry0, rx0; };
    std::vector<Shard> shards;
    std::size_t totalRegions = 0;
    for (const int level : levels) {
        const auto shape = cache->shape(level);
        const int rz = (shape[0] + kRegion - 1) / kRegion;
        const int ry = (shape[1] + kRegion - 1) / kRegion;
        const int rx = (shape[2] + kRegion - 1) / kRegion;
        totalRegions += std::size_t(rz) * ry * rx;
        const auto fb = cc->shardBatch(level, 0, 0, 0);
        const int b = std::max(1, fb.edgeChunks / kBlocksPerRegion);
        for (int z = 0; z < rz; z += b)
            for (int y = 0; y < ry; y += b)
                for (int x = 0; x < rx; x += b)
                    shards.push_back({level, z, y, x});
        std::cout << "level " << level << ": " << shape[0] << "x" << shape[1] << "x"
                  << shape[2] << "  (" << (std::size_t(rz) * ry * rx) << " regions, batch=" << b
                  << ")\n";
    }
    std::cout << shards.size() << " shards across " << levels.size() << " level(s), "
              << totalRegions << " regions, " << nThreads << " threads\n";

    {
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> team;
        for (int t = 0; t < nThreads; ++t)
            team.emplace_back([&] {
                for (;;) {
                    const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= shards.size())
                        return;
                    const Shard& s = shards[i];
                    cc->prefetchShardBlocking(s.level, s.rz0 * kBlocksPerRegion,
                                              s.ry0 * kBlocksPerRegion, s.rx0 * kBlocksPerRegion);
                }
            });
        while (next.load() < shards.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const auto stats = cc->stats();
            std::printf("\r  %zu/%zu shards  net %s/s  disk %s   ",
                        std::min(next.load(), shards.size()), shards.size(),
                        fmtBytes(stats.remoteDownloadBytesPerSecond).c_str(),
                        fmtBytes(double(stats.persistentCacheBytes)).c_str());
            std::fflush(stdout);
        }
        for (auto& th : team)
            th.join();
        std::printf("\n");
    }

    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto stats = static_cast<vc::render::ChunkCache*>(cache)->stats();
    std::cout << "done in " << static_cast<int>(secs) << "s; cache size "
              << fmtBytes(double(stats.persistentCacheBytes)) << "\n";
    return 0;
}
