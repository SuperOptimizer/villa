#include "vc/core/render/MatterArchive.hpp"

#include <cstring>
#include <stdexcept>

extern "C" {
#include "mc_archive_api.h"
}

namespace vc::render {

struct MatterArchive::Impl {
    mc_archive* a = nullptr;
};

static int alignUp(int v, int a) { return ((v + a - 1) / a) * a; }

MatterArchive::MatterArchive(std::string path, int dim0, float quality)
    : impl_(std::make_unique<Impl>())
    , path_(std::move(path))
    , dim0_(dim0)
    , quality_(quality)
{
    const int dimAligned = alignUp(dim0, kChunk);
    impl_->a = mc_archive_open(path_.c_str(), dimAligned, quality);
    if (!impl_->a)
        throw std::runtime_error("MatterArchive: mc_archive_open failed for " + path_);
}

MatterArchive::~MatterArchive()
{
    if (impl_ && impl_->a) {
        mc_archive_close(impl_->a);
        impl_->a = nullptr;
    }
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
    const std::uint64_t off = mc_archive_chunk_offset(impl_->a, lod, cz, cy, cx);
    mc_archive_decode_block(impl_->a, off, bz, by, bx, dst4096);
}

void MatterArchive::decodeChunk(int lod, int cz, int cy, int cx, std::uint8_t* dst) const
{
    const std::uint64_t off = mc_archive_chunk_offset(impl_->a, lod, cz, cy, cx);
    std::uint8_t blk[kBlock * kBlock * kBlock];
    for (int bz = 0; bz < kBlocksPerAxis; ++bz)
        for (int by = 0; by < kBlocksPerAxis; ++by)
            for (int bx = 0; bx < kBlocksPerAxis; ++bx) {
                mc_archive_decode_block(impl_->a, off, bz, by, bx, blk);
                // scatter the 16^3 block into the 256^3 chunk buffer.
                for (int z = 0; z < kBlock; ++z)
                    for (int y = 0; y < kBlock; ++y) {
                        const int gz = bz * kBlock + z;
                        const int gy = by * kBlock + y;
                        const int gx = bx * kBlock;
                        std::memcpy(
                            dst + ((static_cast<std::size_t>(gz) * kChunk + gy) * kChunk + gx),
                            blk + ((static_cast<std::size_t>(z) * kBlock + y) * kBlock),
                            kBlock);
                    }
            }
}

}  // namespace vc::render
