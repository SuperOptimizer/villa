#include "vc4d/cache/tiered_cache.hpp"

#include <algorithm>
#include <ranges>

namespace vc4d {

TieredCache::TieredCache(Config config)
    : config_(std::move(config)) {}

TieredCache::~TieredCache() = default;

std::span<const uint8_t> TieredCache::get(ChunkKey key) {
    std::unique_lock lock(mutex_);

    // Check hot tier first
    if (auto it = hot_.find(key); it != hot_.end()) {
        it->second.last_access = ++access_counter_;
        return it->second.data;
    }

    // Check warm tier — promote to hot
    if (auto it = warm_.find(key); it != warm_.end()) {
        auto compressed = std::move(it->second);
        warm_bytes_ -= compressed.size();
        warm_.erase(it);

        // TODO: Decompress compressed -> decompressed
        // For now, treat warm data as already decompressed
        auto& entry = hot_[key];
        entry.data = std::move(compressed);
        entry.last_access = ++access_counter_;
        hot_bytes_ += entry.data.size();

        evict_hot_to_budget();
        return entry.data;
    }

    return {};
}

void TieredCache::put(ChunkKey key, std::vector<uint8_t> data) {
    std::unique_lock lock(mutex_);

    auto size = data.size();
    auto& entry = hot_[key];
    if (!entry.data.empty())
        hot_bytes_ -= entry.data.size();

    entry.data = std::move(data);
    entry.last_access = ++access_counter_;
    hot_bytes_ += size;

    evict_hot_to_budget();
}

void TieredCache::put_compressed(ChunkKey key, std::vector<uint8_t> compressed) {
    std::unique_lock lock(mutex_);

    auto size = compressed.size();
    if (auto it = warm_.find(key); it != warm_.end())
        warm_bytes_ -= it->second.size();

    warm_[key] = std::move(compressed);
    warm_bytes_ += size;

    evict_warm_to_budget();
}

bool TieredCache::contains(ChunkKey key) const {
    std::shared_lock lock(mutex_);
    return hot_.contains(key) || warm_.contains(key);
}

void TieredCache::pin(ChunkKey key) {
    std::unique_lock lock(mutex_);
    if (auto it = hot_.find(key); it != hot_.end())
        it->second.pinned = true;
}

void TieredCache::set_hot_budget(size_t bytes) {
    std::unique_lock lock(mutex_);
    config_.hot_budget_bytes = bytes;
    evict_hot_to_budget();
}

size_t TieredCache::hot_usage() const {
    std::shared_lock lock(mutex_);
    return hot_bytes_;
}

size_t TieredCache::warm_usage() const {
    std::shared_lock lock(mutex_);
    return warm_bytes_;
}

void TieredCache::clear() {
    std::unique_lock lock(mutex_);
    hot_.clear();
    warm_.clear();
    hot_bytes_ = 0;
    warm_bytes_ = 0;
}

void TieredCache::evict_hot_to_budget() {
    while (hot_bytes_ > config_.hot_budget_bytes && !hot_.empty()) {
        // Find LRU non-pinned entry
        auto oldest = hot_.end();
        uint64_t oldest_access = UINT64_MAX;

        for (auto it = hot_.begin(); it != hot_.end(); ++it) {
            if (!it->second.pinned && it->second.last_access < oldest_access) {
                oldest_access = it->second.last_access;
                oldest = it;
            }
        }

        if (oldest == hot_.end()) break;  // Only pinned entries remain

        // Demote to warm tier (compress in real impl)
        auto& entry = oldest->second;
        hot_bytes_ -= entry.data.size();
        warm_bytes_ += entry.data.size();
        warm_[oldest->first] = std::move(entry.data);
        hot_.erase(oldest);

        evict_warm_to_budget();
    }
}

void TieredCache::evict_warm_to_budget() {
    while (warm_bytes_ > config_.warm_budget_bytes && !warm_.empty()) {
        // Evict oldest warm entry (FIFO — could be improved to LRU)
        auto it = warm_.begin();
        warm_bytes_ -= it->second.size();
        // TODO: Write to disk store before discarding
        warm_.erase(it);
    }
}

} // namespace vc4d
