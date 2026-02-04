#include "vc/core/types/ChunkedTensor.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include <unistd.h>

void print_accessor_stats()
{
    std::cout << "acc miss/total " << miss << " " << total << " " << double(miss)/total << '\n';
    std::cout << "chunk compute overhead/total " << chunk_compute_collisions << " " << chunk_compute_total << " " << double(chunk_compute_collisions)/chunk_compute_total << '\n';
}

static std::string tmp_name_proc_thread()
{
    std::stringstream ss;
    ss << "tmp_" << getpid() << "_" << std::this_thread::get_id();
    return ss.str();
}

std::filesystem::path resolve_chunk_cache_dir(
    const std::filesystem::path& cache_root,
    const std::filesystem::path& root,
    bool persistent,
    const std::filesystem::path& ds_path)
{
    std::filesystem::path cache_dir;

    for(int r=0;r<1000 && cache_dir.empty();r++) {
        std::set<std::string> paths;
        if (persistent) {
            for (auto const& entry : std::filesystem::directory_iterator(root))
                if (std::filesystem::is_directory(entry) && std::filesystem::exists(entry.path()/"meta.json") && std::filesystem::is_regular_file(entry.path()/"meta.json")) {
                    paths.insert(entry.path());
                    try {
                        std::ifstream meta_f(entry.path()/"meta.json");
                        nlohmann::json meta = nlohmann::json::parse(meta_f);
                        // Skip entries with invalid or non-existent dataset paths
                        if (!meta.contains("dataset_source_path") || !meta["dataset_source_path"].is_string())
                            continue;
                        std::filesystem::path src_candidate(meta["dataset_source_path"].get<std::string>());
                        if (!std::filesystem::exists(src_candidate))
                            continue;
                        if (!std::filesystem::exists(ds_path))
                            continue;
                        std::filesystem::path src = std::filesystem::canonical(src_candidate);
                        std::filesystem::path cur = std::filesystem::canonical(ds_path);
                        if (src == cur) {
                            cache_dir = entry.path();
                            break;
                        }
                    } catch (const std::exception&) {
                        // Ignore malformed cache entries or paths we cannot canonicalize
                        continue;
                    }
                }

            if (!cache_dir.empty())
                continue;
        }

        //try generating our own cache dir atomically
        std::filesystem::path tmp_dir = cache_root/tmp_name_proc_thread();
        std::filesystem::create_directories(tmp_dir);

        if (persistent) {
            nlohmann::json meta;
            meta["dataset_source_path"] = std::filesystem::canonical(ds_path).string();
            std::ofstream o(tmp_dir/"meta.json");
            o << std::setw(4) << meta << std::endl;

            std::filesystem::path tgt_path;
            for(int i=0;i<1000;i++) {
                tgt_path = root/std::to_string(i);
                if (paths.count(tgt_path.string()))
                    continue;
                try {
                    std::filesystem::rename(tmp_dir, tgt_path);
                }
                catch (std::filesystem::filesystem_error&){
                    continue;
                }
                cache_dir = tgt_path;
                break;
            }
        }
        else {
            cache_dir = tmp_dir;
        }
    }

    return cache_dir;
}
