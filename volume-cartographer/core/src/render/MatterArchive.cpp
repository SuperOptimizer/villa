#include "vc/core/render/MatterArchive.hpp"

#include <filesystem>
#include <stdexcept>

#include "vc/core/util/Logging.hpp"

extern "C" {
#include "mc_archive_api.h"
#include "mc_cache.h"
}

namespace vc::render {

struct MatterArchive::Impl {
    mc_archive* a = nullptr;
    mc_cache* cache = nullptr;
};

MatterArchive::MatterArchive(std::string path, std::array<int, 3> shape0, float quality,
                             std::size_t cacheBytes)
    : impl_(std::make_unique<Impl>())
    , path_(std::move(path))
    , shape0_(shape0)
    , quality_(quality)
{
    // shape0 is (z,y,x); mc takes (nx,ny,nz) and pads each axis to 256 internally.
    auto open = [&] {
        return mc_archive_open_dims(path_.c_str(), shape0_[2], shape0_[1], shape0_[0],
                                    quality);
    };
    impl_->a = open();
    if (!impl_->a && std::filesystem::exists(path_)) {
        // stale archive (older format version / different dims): it's a rebuildable
        // cache, so delete and recreate.
        Logger()->warn("MatterArchive: {} is stale/incompatible; recreating", path_);
        std::error_code ec;
        std::filesystem::remove(path_, ec);
        impl_->a = open();
    }
    if (!impl_->a)
        throw std::runtime_error("MatterArchive: mc_archive_open failed for " + path_);
    impl_->cache = mc_cache_new_archive(cacheBytes, impl_->a);
    if (!impl_->cache) {
        mc_archive_close(impl_->a);
        throw std::runtime_error("MatterArchive: mc_cache_new_archive failed for " + path_);
    }
}

MatterArchive::~MatterArchive()
{
    if (impl_ && impl_->cache)
        mc_cache_free(impl_->cache);
    if (impl_ && impl_->a)
        mc_archive_close(impl_->a);
}

bool MatterArchive::appendChunkRaw(int lod, int cz, int cy, int cx, const std::uint8_t* vox256)
{
    if (!vox256) return false;
    return mc_archive_append_chunk_raw(impl_->a, lod, cz, cy, cx, vox256) == 0;
}

bool MatterArchive::hasChunk(int lod, int cz, int cy, int cx) const
{
    return mc_archive_chunk_coverage(impl_->a, lod, cz, cy, cx) == MC_PRESENT;
}

void MatterArchive::decodeBlock(int lod, int cz, int cy, int cx, int bz, int by, int bx,
                                std::uint8_t* dst4096) const
{
    // mc_cache keys on GLOBAL block coords.
    mc_cache_get_copy(impl_->cache, lod,
                      cz * kBlocksPerAxis + bz,
                      cy * kBlocksPerAxis + by,
                      cx * kBlocksPerAxis + bx,
                      dst4096);
}

MatterArchive::CacheStats MatterArchive::cacheStats() const
{
    mc_cache_stats s{};
    mc_cache_get_stats(impl_->cache, &s);
    CacheStats out;
    out.hits = s.hits;
    out.misses = s.misses;
    out.evictions = s.evictions;
    out.usedBytes = s.used * std::size_t{4096};
    out.capacityBytes = s.slots * std::size_t{4096};
    return out;
}

}  // namespace vc::render
