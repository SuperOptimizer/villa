#include "ChunkCache.hpp"

#include <utils/thread_pool.hpp>

#include "vc/core/util/Logging.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace vc::render {

namespace {

constexpr auto kDownloadStatsWindow = std::chrono::seconds{3};
constexpr auto kPersistentCacheSizeScanInterval = std::chrono::seconds{2};
constexpr int kViewEpochPriorityStride = 1024;

std::size_t normalizedWorkerCount(std::size_t requested)
{
    return std::max<std::size_t>(1, requested);
}

utils::PriorityThreadPool& chunkWorkerPool(std::size_t workerCount)
{
    // Keep chunk I/O executors process-wide instead of viewer/cache-owned.
    // Destroying a viewer invalidates its cache state, but does not join
    // blocking file/HTTP reads from the UI thread.
    static std::mutex mutex;
    static std::unordered_map<std::size_t, std::unique_ptr<utils::PriorityThreadPool>> pools;

    workerCount = normalizedWorkerCount(workerCount);
    std::lock_guard lock(mutex);
    auto& pool = pools[workerCount];
    if (!pool)
        pool = std::make_unique<utils::PriorityThreadPool>(workerCount);
    return *pool;
}

std::string fetchErrorMessage(const ChunkFetchResult& fetch)
{
    if (!fetch.message.empty())
        return fetch.message;
    switch (fetch.status) {
    case ChunkFetchStatus::HttpError:
        return fetch.httpStatus > 0 ? "HTTP error " + std::to_string(fetch.httpStatus) : "HTTP error";
    case ChunkFetchStatus::IoError:
        return "I/O error";
    case ChunkFetchStatus::DecodeError:
        return "decode error";
    case ChunkFetchStatus::Found:
    case ChunkFetchStatus::Missing:
        return {};
    }
    return "chunk fetch error";
}

} // namespace

ChunkCache::ChunkCache(std::vector<LevelInfo> levels,
                       std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
                       double fillValue,
                       ChunkDtype dtype)
    : ChunkCache(std::move(levels), std::move(fetchers), fillValue, dtype, Options{})
{
}

ChunkCache::ChunkCache(std::vector<LevelInfo> levels,
                       std::vector<std::shared_ptr<IChunkFetcher>> fetchers,
                       double fillValue,
                       ChunkDtype dtype,
                       Options options)
    : state_(std::make_shared<State>(std::move(levels), std::move(fetchers), fillValue, dtype, std::move(options)))
{
    if (state_->levels_.empty())
        throw std::invalid_argument("ChunkCache requires at least one level");
    if (state_->levels_.size() != state_->fetchers_.size())
        throw std::invalid_argument("ChunkCache level/fetcher count mismatch");
    for (std::size_t i = 0; i < state_->levels_.size(); ++i) {
        const bool missingLevel =
            state_->levels_[i].shape[0] == 0 &&
            state_->levels_[i].shape[1] == 0 &&
            state_->levels_[i].shape[2] == 0;
        if (!state_->fetchers_[i] && !missingLevel)
            throw std::invalid_argument("ChunkCache fetcher must not be null for present level");
        for (int dim : state_->levels_[i].shape) {
            if (dim < 0)
                throw std::invalid_argument("ChunkCache level shape must be non-negative");
        }
        for (int dim : state_->levels_[i].chunkShape) {
            if (dim <= 0)
                throw std::invalid_argument("ChunkCache chunk shape must be positive");
        }
    }
}

ChunkCache::~ChunkCache()
{
    invalidate();
}

int ChunkCache::numLevels() const
{
    return static_cast<int>(state_->levels_.size());
}

std::array<int, 3> ChunkCache::shape(int level) const
{
    return state_->levels_.at(static_cast<std::size_t>(level)).shape;
}

std::array<int, 3> ChunkCache::chunkShape(int level) const
{
    return state_->levels_.at(static_cast<std::size_t>(level)).chunkShape;
}

ChunkDtype ChunkCache::dtype() const
{
    return state_->dtype_;
}

double ChunkCache::fillValue() const
{
    return state_->fillValue_;
}

IChunkedArray::LevelTransform ChunkCache::levelTransform(int level) const
{
    return state_->levels_.at(static_cast<std::size_t>(level)).transform;
}

ChunkResult ChunkCache::tryGetChunk(int level, int iz, int iy, int ix)
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix};
    if (level >= 0 && level < static_cast<int>(state->fetchers_.size()) &&
        !state->fetchers_[static_cast<std::size_t>(level)]) {
        return ChunkResult{
            ChunkStatus::Missing,
            state->dtype_,
            state->levels_[static_cast<std::size_t>(level)].chunkShape,
            {},
            {}};
    }
    if (!isValidKey(*state, key))
        return ChunkResult{ChunkStatus::AllFill, state->dtype_, {}, {}, {}};

    std::unique_lock lock(state->mutex_);
    auto it = state->entries_.find(key);
    if (it != state->entries_.end()) {
        if (it->second.status == EntryStatus::InFlight) {
            return ChunkResult{ChunkStatus::MissQueued, state->dtype_, state->levels_[level].chunkShape, {}, {}};
        }
        return resultFromEntryLocked(*state, key, it->second);
    }

    state->entries_.emplace(key, Entry{});
    queueFetchLocked(state, key, state->generation_, 0);
    return ChunkResult{ChunkStatus::MissQueued, state->dtype_, state->levels_[level].chunkShape, {}, {}};
}

ChunkResult ChunkCache::getChunkBlocking(int level, int iz, int iy, int ix)
{
    auto state = state_;
    const ChunkKey key{level, iz, iy, ix};
    if (level >= 0 && level < static_cast<int>(state->fetchers_.size()) &&
        !state->fetchers_[static_cast<std::size_t>(level)]) {
        return ChunkResult{
            ChunkStatus::Missing,
            state->dtype_,
            state->levels_[static_cast<std::size_t>(level)].chunkShape,
            {},
            {}};
    }
    if (!isValidKey(*state, key))
        return ChunkResult{ChunkStatus::AllFill, state->dtype_, {}, {}, {}};

    std::unique_lock lock(state->mutex_);
    auto [it, inserted] = state->entries_.emplace(key, Entry{});
    if (inserted)
        queueFetchLocked(state, key, state->generation_, 0);
    waitForResolvedLocked(*state, lock, key);
    it = state->entries_.find(key);
    if (it == state->entries_.end())
        return ChunkResult{ChunkStatus::Error, state->dtype_, state->levels_[level].chunkShape, {}, "chunk invalidated"};
    return resultFromEntryLocked(*state, key, it->second);
}

void ChunkCache::prefetchChunks(const std::vector<ChunkKey>& keys, bool wait, int priorityOffset)
{
    auto state = state_;
    std::unique_lock lock(state->mutex_);
    for (const auto& key : keys) {
        if (!isValidKey(*state, key))
            continue;
        auto [it, inserted] = state->entries_.emplace(key, Entry{});
        if (inserted) {
            queueFetchLocked(state, key, state->generation_, priorityOffset);
        } else if (it->second.status == EntryStatus::InFlight &&
                   key.level + priorityOffset < it->second.basePriority) {
            queueFetchLocked(state, key, state->generation_, priorityOffset);
        }
    }
    if (!wait)
        return;

    state->cv_.wait(lock, [&] {
        for (const auto& key : keys) {
            if (!isValidKey(*state, key))
                continue;
            auto it = state->entries_.find(key);
            if (it != state->entries_.end() && it->second.status == EntryStatus::InFlight)
                return false;
        }
        return true;
    });
}

IChunkedArray::ChunkReadyCallbackId ChunkCache::addChunkReadyListener(ChunkReadyCallback cb)
{
    auto state = state_;
    std::lock_guard lock(state->mutex_);
    const auto id = state->nextCallbackId_++;
    state->callbacks_.emplace(id, std::move(cb));
    return id;
}

void ChunkCache::removeChunkReadyListener(ChunkReadyCallbackId id)
{
    auto state = state_;
    std::lock_guard lock(state->mutex_);
    state->callbacks_.erase(id);
}

ChunkCache::Stats ChunkCache::stats() const
{
    auto state = state_;
    std::optional<std::filesystem::path> mcaPath;
    bool scanMcaBytes = false;
    Stats result;
    {
        std::lock_guard lock(state->mutex_);
        const auto now = std::chrono::steady_clock::now();
        pruneDownloadHistoryLocked(*state, now);

        std::size_t recentBytes = 0;
        for (const auto& [time, bytes] : state->remoteDownloadHistory_) {
            (void)time;
            recentBytes += bytes;
        }

        result.decodedBytes = state->decodedBytes_;
        result.decodedByteCapacity = state->options_.decodedByteCapacity;
        result.remoteFetchesInFlight = state->remoteFetchesInFlight_;
        result.remoteDownloadBytesPerSecond =
            static_cast<double>(recentBytes) /
            std::chrono::duration<double>(kDownloadStatsWindow).count();
        // "disk" cache size = the single on-disk volume.mca file (the only on-disk
        // cache). Throttled stat() of the file size (it grows as chunks are appended).
        result.persistentCacheBytes = state->cachedPersistentCacheBytes_;
        mcaPath = state->options_.mcaPath;
        scanMcaBytes = mcaPath.has_value() &&
            (state->lastPersistentCacheSizeScan_ == std::chrono::steady_clock::time_point{} ||
             now - state->lastPersistentCacheSizeScan_ >= kPersistentCacheSizeScanInterval);
        if (scanMcaBytes)
            state->lastPersistentCacheSizeScan_ = now;
    }
    if (scanMcaBytes) {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(*mcaPath, ec);
        result.persistentCacheBytes = ec ? 0 : static_cast<std::size_t>(sz);
        std::lock_guard lock(state->mutex_);
        state->cachedPersistentCacheBytes_ = result.persistentCacheBytes;
    }
    return result;
}

void ChunkCache::invalidate()
{
    auto state = state_;
    {
        std::lock_guard lock(state->mutex_);
        ++state->generation_;
        state->entries_.clear();
        state->lru_.clear();
        state->decodedBytes_ = 0;
        state->remoteFetchesInFlight_ = 0;
        state->remoteDownloadHistory_.clear();
    }
    state->cv_.notify_all();
}

void ChunkCache::beginViewRequest()
{
    auto state = state_;
    std::lock_guard lock(state->mutex_);
    if (state->viewEpoch_ == std::numeric_limits<utils::PriorityThreadPool::Priority>::max())
        state->viewEpoch_ = 1;
    else
        ++state->viewEpoch_;
}

ChunkResult ChunkCache::resultFromEntryLocked(State& state, const ChunkKey& key, Entry& entry)
{
    ChunkResult result;
    result.dtype = state.dtype_;
    result.shape = state.levels_[static_cast<std::size_t>(key.level)].chunkShape;

    switch (entry.status) {
    case EntryStatus::InFlight:
        result.status = ChunkStatus::MissQueued;
        break;
    case EntryStatus::Missing:
        result.status = ChunkStatus::Missing;
        touchLocked(state, key, entry);
        break;
    case EntryStatus::AllFill:
        result.status = ChunkStatus::AllFill;
        touchLocked(state, key, entry);
        break;
    case EntryStatus::Data:
        result.status = ChunkStatus::Data;
        result.bytes = entry.bytes;
        touchLocked(state, key, entry);
        break;
    case EntryStatus::Error:
        result.status = ChunkStatus::Error;
        result.error = entry.error;
        touchLocked(state, key, entry);
        break;
    }
    return result;
}

void ChunkCache::queueFetchLocked(const std::shared_ptr<State>& state,
                                  const ChunkKey& key,
                                  std::uint64_t generation,
                                  int priorityOffset)
{
    auto it = state->entries_.find(key);
    if (it == state->entries_.end())
        return;
    Entry& entry = it->second;
    entry.status = EntryStatus::InFlight;
    entry.basePriority = key.level + priorityOffset;
    const auto epochBias = state->viewEpoch_;
    entry.priority = entry.basePriority - epochBias * kViewEpochPriorityStride;
    const std::uint64_t fetchSerial = state->nextFetchSerial_++;
    entry.fetchSerial = fetchSerial;

    const auto priority = entry.priority;
    std::weak_ptr<State> weakState = state;
    chunkWorkerPool(state->options_.maxConcurrentReads).submit(priority, [weakState, key, generation, fetchSerial] {
        if (auto state = weakState.lock())
            fetchAndStore(state, key, generation, fetchSerial);
    });
}

void ChunkCache::fetchAndStore(const std::shared_ptr<State>& state,
                               ChunkKey key,
                               std::uint64_t generation,
                               std::uint64_t fetchSerial)
{
    {
        std::lock_guard lock(state->mutex_);
        if (generation != state->generation_)
            return;
        auto it = state->entries_.find(key);
        if (it == state->entries_.end() || it->second.fetchSerial != fetchSerial)
            return;
    }

    ChunkFetchResult fetch;
    bool trackedRemoteFetch = false;
    try {
        // On-disk caching is handled by the mca cache inside the fetcher (the
        // MatterCacheFetcher decorator owns the single volume.mca). The ChunkCache no
        // longer has a per-chunk persistent cache; it just calls fetch().
        trackedRemoteFetch = true;
        {
            std::lock_guard lock(state->mutex_);
            ++state->remoteFetchesInFlight_;
        }
        fetch = state->fetchers_.at(static_cast<std::size_t>(key.level))->fetch(key);
    } catch (const std::exception& e) {
        fetch.status = ChunkFetchStatus::IoError;
        fetch.message = e.what();
        Logger()->error(
            "ChunkCache caught chunk fetch exception for {}/{}/{}/{}: {}",
            key.level,
            key.iz,
            key.iy,
            key.ix,
            fetch.message);
    } catch (...) {
        fetch.status = ChunkFetchStatus::IoError;
        fetch.message = "unknown chunk fetch exception";
        Logger()->error(
            "ChunkCache caught unknown chunk fetch exception for {}/{}/{}/{}",
            key.level,
            key.iz,
            key.iy,
            key.ix);
    }

    {
        std::lock_guard lock(state->mutex_);
        if (trackedRemoteFetch && state->remoteFetchesInFlight_ > 0)
            --state->remoteFetchesInFlight_;
        if (trackedRemoteFetch && fetch.status == ChunkFetchStatus::Found && !fetch.bytes.empty()) {
            const auto now = std::chrono::steady_clock::now();
            state->remoteDownloadHistory_.emplace_back(now, fetch.bytes.size());
            pruneDownloadHistoryLocked(*state, now);
        }
        if (generation != state->generation_)
            return;
        auto it = state->entries_.find(key);
        if (it == state->entries_.end() || it->second.fetchSerial != fetchSerial)
            return;
        storeFetchResultLocked(state, key, std::move(fetch));
    }
    state->cv_.notify_all();
    notifyListeners(state);
}

void ChunkCache::storeFetchResultLocked(const std::shared_ptr<State>& state,
                                        const ChunkKey& key,
                                        ChunkFetchResult fetch)
{
    auto it = state->entries_.find(key);
    if (it == state->entries_.end())
        return;

    Entry& entry = it->second;
    if (entry.inLru) {
        state->lru_.erase(entry.lruIt);
        entry.inLru = false;
    }
    if (entry.status == EntryStatus::Data)
        state->decodedBytes_ -= entry.decodedBytes;

    entry.bytes.reset();
    entry.error.clear();
    entry.decodedBytes = 0;

    switch (fetch.status) {
    case ChunkFetchStatus::Found: {
        if (fetch.bytes.size() != expectedChunkBytes(*state, key)) {
            entry.status = EntryStatus::Error;
            entry.error = "decoded chunk byte size does not match full chunk shape";
            break;
        }
        if (state->options_.detectAllFillChunks && isAllFill(*state, fetch.bytes)) {
            entry.status = EntryStatus::AllFill;
            break;
        }
        entry.status = EntryStatus::Data;
        entry.decodedBytes = fetch.bytes.size();
        entry.bytes = std::make_shared<const std::vector<std::byte>>(std::move(fetch.bytes));
        state->decodedBytes_ += entry.decodedBytes;
        break;
    }
    case ChunkFetchStatus::Missing:
        entry.status = EntryStatus::Missing;
        break;
    case ChunkFetchStatus::HttpError:
    case ChunkFetchStatus::IoError:
    case ChunkFetchStatus::DecodeError:
        entry.status = EntryStatus::Error;
        entry.error = fetchErrorMessage(fetch);
        break;
    }

    touchLocked(*state, key, entry);
    enforceCapacityLocked(state);
}


void ChunkCache::pruneDownloadHistoryLocked(State& state, std::chrono::steady_clock::time_point now)
{
    const auto cutoff = now - kDownloadStatsWindow;
    while (!state.remoteDownloadHistory_.empty() &&
           state.remoteDownloadHistory_.front().first < cutoff) {
        state.remoteDownloadHistory_.pop_front();
    }
}

void ChunkCache::touchLocked(State& state, const ChunkKey& key, Entry& entry)
{
    if (entry.status == EntryStatus::InFlight)
        return;
    if (entry.inLru)
        state.lru_.erase(entry.lruIt);
    state.lru_.push_front(key);
    entry.lruIt = state.lru_.begin();
    entry.inLru = true;
}

void ChunkCache::enforceCapacityLocked(const std::shared_ptr<State>& state)
{
    while ((state->decodedBytes_ > state->options_.decodedByteCapacity ||
            state->entries_.size() > state->options_.metadataEntryCapacity) &&
           !state->lru_.empty()) {
        const ChunkKey victim = state->lru_.back();
        state->lru_.pop_back();
        auto it = state->entries_.find(victim);
        if (it == state->entries_.end())
            continue;
        Entry& entry = it->second;
        entry.inLru = false;
        if (entry.status == EntryStatus::Data)
            state->decodedBytes_ -= entry.decodedBytes;
        state->entries_.erase(it);
    }
}

bool ChunkCache::isValidKey(const State& state, const ChunkKey& key)
{
    if (key.level < 0 || key.level >= static_cast<int>(state.levels_.size()))
        return false;
    if (!state.fetchers_[static_cast<std::size_t>(key.level)])
        return false;
    const auto& level = state.levels_[static_cast<std::size_t>(key.level)];
    const std::array<int, 3> coords{key.iz, key.iy, key.ix};
    for (int axis = 0; axis < 3; ++axis) {
        if (coords[axis] < 0)
            return false;
        const int chunks = (level.shape[axis] + level.chunkShape[axis] - 1) / level.chunkShape[axis];
        if (coords[axis] >= chunks)
            return false;
    }
    return true;
}

bool ChunkCache::isAllFill(const State& state, const std::vector<std::byte>& bytes)
{
    if (state.dtype_ == ChunkDtype::UInt8) {
        const auto fill = static_cast<unsigned char>(std::clamp(
            state.fillValue_, 0.0, static_cast<double>(std::numeric_limits<unsigned char>::max())));
        return std::all_of(bytes.begin(), bytes.end(), [fill](std::byte value) {
            return static_cast<unsigned char>(value) == fill;
        });
    }

    const auto fill = static_cast<std::uint16_t>(std::clamp(
        state.fillValue_, 0.0, static_cast<double>(std::numeric_limits<std::uint16_t>::max())));
    if (bytes.size() % sizeof(std::uint16_t) != 0)
        return false;
    const auto* ptr = reinterpret_cast<const std::uint16_t*>(bytes.data());
    const std::size_t count = bytes.size() / sizeof(std::uint16_t);
    return std::all_of(ptr, ptr + count, [fill](std::uint16_t value) {
        return value == fill;
    });
}

std::size_t ChunkCache::dtypeSize(ChunkDtype dtype)
{
    switch (dtype) {
    case ChunkDtype::UInt8:
        return 1;
    case ChunkDtype::UInt16:
        return 2;
    }
    return 1;
}

std::size_t ChunkCache::expectedChunkBytes(const State& state, const ChunkKey& key)
{
    const auto& chunk = state.levels_[static_cast<std::size_t>(key.level)].chunkShape;
    return static_cast<std::size_t>(chunk[0]) *
           static_cast<std::size_t>(chunk[1]) *
           static_cast<std::size_t>(chunk[2]) *
           dtypeSize(state.dtype_);
}

void ChunkCache::notifyListeners(const std::shared_ptr<State>& state)
{
    std::vector<ChunkReadyCallback> callbacks;
    {
        std::lock_guard lock(state->mutex_);
        callbacks.reserve(state->callbacks_.size());
        for (const auto& [id, cb] : state->callbacks_) {
            (void)id;
            callbacks.push_back(cb);
        }
    }
    for (auto& cb : callbacks) {
        if (cb)
            cb();
    }
}

void ChunkCache::waitForResolvedLocked(State& state, std::unique_lock<std::mutex>& lock, const ChunkKey& key)
{
    state.cv_.wait(lock, [&] {
        auto it = state.entries_.find(key);
        return it == state.entries_.end() || it->second.status != EntryStatus::InFlight;
    });
}

} // namespace vc::render
