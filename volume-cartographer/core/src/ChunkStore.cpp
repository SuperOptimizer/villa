#include "vc/core/types/ChunkStore.hpp"

#include "vc/core/util/ChunkCache.hpp"

struct ChunkStore::Impl {
    ChunkCache<uint8_t> cache8;
    ChunkCache<uint16_t> cache16;

    explicit Impl(size_t maxBytes)
        : cache8(maxBytes), cache16(maxBytes) {}
};

ChunkStore::ChunkStore(size_t maxBytes)
    : pImpl_(std::make_unique<Impl>(maxBytes)) {}

ChunkStore::~ChunkStore() = default;

void ChunkStore::setMaxBytes(size_t maxBytes)
{
    pImpl_->cache8.setMaxBytes(maxBytes);
    pImpl_->cache16.setMaxBytes(maxBytes);
}

ChunkStore::Stats ChunkStore::stats() const
{
    auto s8 = pImpl_->cache8.stats();
    auto s16 = pImpl_->cache16.stats();
    return {
        s8.hits + s16.hits,
        s8.misses + s16.misses,
        s8.evictions + s16.evictions,
        s8.bytesRead + s16.bytesRead
    };
}

void ChunkStore::resetStats()
{
    pImpl_->cache8.resetStats();
    pImpl_->cache16.resetStats();
}

void ChunkStore::clear()
{
    pImpl_->cache8.clear();
    pImpl_->cache16.clear();
}

ChunkCache<uint8_t>& ChunkStore::cache8()
{
    return pImpl_->cache8;
}

ChunkCache<uint16_t>& ChunkStore::cache16()
{
    return pImpl_->cache16;
}
