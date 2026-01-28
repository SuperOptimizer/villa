#include "vc/core/util/ChunkCache.hpp"

#include "z5/dataset.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include <algorithm>

// Helper to read a chunk from disk
template<typename T>
static xt::xarray<T>* readChunkFromDisk(const z5::Dataset& ds, size_t ix, size_t iy, size_t iz)
{
    z5::types::ShapeType chunkId = {ix, iy, iz};

    if (!ds.chunkExists(chunkId)) {
        return nullptr;
    }

    if (!ds.isZarr())
        throw std::runtime_error("only zarr datasets supported currently!");
    if (ds.getDtype() != z5::types::Datatype::uint8 && ds.getDtype() != z5::types::Datatype::uint16)
        throw std::runtime_error("only uint8_t/uint16 zarrs supported currently!");

    const auto& maxChunkShape = ds.defaultChunkShape();

    xt::xarray<T>* out = new xt::xarray<T>();
    *out = xt::empty<T>(maxChunkShape);

    // Handle based on both dataset dtype and target type T
    if (ds.getDtype() == z5::types::Datatype::uint8) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            ds.readChunk(chunkId, out->data());
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (ds.getDtype() == z5::types::Datatype::uint16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            ds.readChunk(chunkId, out->data());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            // Dataset is uint16, target is uint8 - need conversion
            xt::xarray<uint16_t> tmp = xt::empty<uint16_t>(maxChunkShape);
            ds.readChunk(chunkId, tmp.data());

            const size_t maxChunkSize = ds.defaultChunkSize();
            uint8_t* p8 = out->data();
            uint16_t* p16 = tmp.data();
            for (size_t i = 0; i < maxChunkSize; i++)
                p8[i] = p16[i] / 257;
        }
    }

    return out;
}

template<typename T>
ChunkCache<T>::ChunkCache(size_t maxBytes) : _maxBytes(maxBytes)
{
}

template<typename T>
void ChunkCache<T>::setMaxBytes(size_t maxBytes)
{
    _maxBytes = maxBytes;
    // Recompute maxChunks if already initialized
    if (_ds && _cw > 0 && _ch > 0 && _cd > 0) {
        size_t chunkBytes = static_cast<size_t>(_cw) * _ch * _cd * sizeof(T);
        _maxChunks = (_maxBytes > 0) ? (_maxBytes / chunkBytes) : 0;
    }
}

template<typename T>
ChunkCache<T>::~ChunkCache()
{
    clear();
}

template<typename T>
void ChunkCache<T>::init(z5::Dataset* ds)
{
    if (_ds == ds) return;  // Already initialized for this dataset

    clear();
    _ds = ds;

    if (!ds) return;

    // Get chunk dimensions
    const auto& blockShape = ds->chunking().blockShape();
    _cw = static_cast<int>(blockShape[0]);
    _ch = static_cast<int>(blockShape[1]);
    _cd = static_cast<int>(blockShape[2]);

    // Compute bit shifts for power-of-2 chunk sizes
    _cwShift = log2_pow2(_cw);
    _chShift = log2_pow2(_ch);
    _cdShift = log2_pow2(_cd);
    _cwMask = _cw - 1;
    _chMask = _ch - 1;
    _cdMask = _cd - 1;

    // Get dataset dimensions
    const auto& dsShape = ds->shape();
    _sx = static_cast<int>(dsShape[0]);
    _sy = static_cast<int>(dsShape[1]);
    _sz = static_cast<int>(dsShape[2]);

    // Compute number of chunks
    _chunksX = (_sx + _cw - 1) / _cw;
    _chunksY = (_sy + _ch - 1) / _ch;
    _chunksZ = (_sz + _cd - 1) / _cd;

    // Compute maxChunks from maxBytes and actual chunk size
    size_t chunkBytes = static_cast<size_t>(_cw) * _ch * _cd * sizeof(T);
    _maxChunks = (_maxBytes > 0) ? (_maxBytes / chunkBytes) : 0;

    // Allocate flat array of entries
    size_t totalChunks = static_cast<size_t>(_chunksX) * _chunksY * _chunksZ;
    _entries.resize(totalChunks);
    for (size_t i = 0; i < totalChunks; i++) {
        _entries[i].chunk.store(nullptr, std::memory_order_relaxed);
        _entries[i].lastAccess.store(0, std::memory_order_relaxed);
    }

    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);
}

template<typename T>
xt::xarray<T>* ChunkCache<T>::get(int ix, int iy, int iz)
{
    // Bounds check
    if (ix < 0 || ix >= _chunksX || iy < 0 || iy >= _chunksY || iz < 0 || iz >= _chunksZ)
        return nullptr;

    size_t i = idx(ix, iy, iz);
    Entry& entry = _entries[i];

    // Fast path: atomic load (lock-free)
    xt::xarray<T>* ptr = entry.chunk.load(std::memory_order_acquire);
    if (ptr) {
        // Update access time for NRU (relaxed ordering is fine)
        entry.lastAccess.store(_generation.fetch_add(1, std::memory_order_relaxed),
                               std::memory_order_relaxed);
        return ptr;
    }

    // Slow path: serialize chunk loading (blosc2 decompression is not thread-safe)
    {
        std::lock_guard<std::mutex> lock(_loadMutex);

        // Re-check after acquiring lock — another thread may have loaded it
        ptr = entry.chunk.load(std::memory_order_acquire);
        if (ptr) {
            entry.lastAccess.store(_generation.fetch_add(1, std::memory_order_relaxed),
                                   std::memory_order_relaxed);
            return ptr;
        }

        xt::xarray<T>* newChunk = loadChunk(ix, iy, iz);
        if (!newChunk) return nullptr;

        // Check if we need to evict before inserting
        if (_maxChunks > 0 && _cachedCount.load(std::memory_order_relaxed) >= _maxChunks) {
            std::lock_guard<std::mutex> evictLock(_evictionMutex);
            evictIfNeeded();
        }

        entry.chunk.store(newChunk, std::memory_order_release);
        entry.lastAccess.store(_generation.fetch_add(1, std::memory_order_relaxed),
                               std::memory_order_relaxed);
        _cachedCount.fetch_add(1, std::memory_order_relaxed);
        return newChunk;
    }
}

template<typename T>
xt::xarray<T>* ChunkCache<T>::getIfCached(int ix, int iy, int iz) const
{
    if (ix < 0 || ix >= _chunksX || iy < 0 || iy >= _chunksY || iz < 0 || iz >= _chunksZ)
        return nullptr;

    return _entries[idx(ix, iy, iz)].chunk.load(std::memory_order_acquire);
}

template<typename T>
void ChunkCache<T>::prefetch(int minIx, int minIy, int minIz, int maxIx, int maxIy, int maxIz)
{
    // Clamp to valid range
    minIx = std::max(0, minIx);
    minIy = std::max(0, minIy);
    minIz = std::max(0, minIz);
    maxIx = std::min(_chunksX - 1, maxIx);
    maxIy = std::min(_chunksY - 1, maxIy);
    maxIz = std::min(_chunksZ - 1, maxIz);

    // Load all chunks in the region (can be parallelized)
    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int iz = minIz; iz <= maxIz; iz++) {
        for (int iy = minIy; iy <= maxIy; iy++) {
            for (int ix = minIx; ix <= maxIx; ix++) {
                get(ix, iy, iz);  // get() handles the atomic caching
            }
        }
    }
}

template<typename T>
void ChunkCache<T>::clear()
{
    for (size_t i = 0; i < _entries.size(); i++) {
        xt::xarray<T>* ptr = _entries[i].chunk.exchange(nullptr, std::memory_order_relaxed);
        delete ptr;
    }
    _entries.clear();

    _ds = nullptr;
    _cw = _ch = _cd = 0;
    _sx = _sy = _sz = 0;
    _chunksX = _chunksY = _chunksZ = 0;
    _cwShift = _chShift = _cdShift = 0;
    _cwMask = _chMask = _cdMask = 0;
    _maxChunks = 0;  // Will be recomputed in init() from _maxBytes
    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);
}

template<typename T>
void ChunkCache<T>::evictIfNeeded()
{
    // Called with _evictionMutex held
    if (_maxChunks == 0) return;

    size_t currentCount = _cachedCount.load(std::memory_order_relaxed);
    if (currentCount < _maxChunks) return;

    // Collect entries with their access times
    struct EvictCandidate {
        size_t index;
        uint64_t lastAccess;
    };
    std::vector<EvictCandidate> candidates;
    candidates.reserve(currentCount);

    for (size_t i = 0; i < _entries.size(); i++) {
        if (_entries[i].chunk.load(std::memory_order_relaxed) != nullptr) {
            candidates.push_back({i, _entries[i].lastAccess.load(std::memory_order_relaxed)});
        }
    }

    if (candidates.empty()) return;

    // Sort by access time (oldest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const EvictCandidate& a, const EvictCandidate& b) {
                  return a.lastAccess < b.lastAccess;
              });

    // Evict oldest 50% to avoid repeated eviction cycles
    size_t toEvict = std::max<size_t>(1, candidates.size() / 2);
    size_t evicted = 0;

    for (size_t j = 0; j < toEvict && j < candidates.size(); j++) {
        size_t i = candidates[j].index;
        xt::xarray<T>* ptr = _entries[i].chunk.exchange(nullptr, std::memory_order_relaxed);
        if (ptr) {
            delete ptr;
            evicted++;
        }
    }

    _cachedCount.fetch_sub(evicted, std::memory_order_relaxed);
}

template<typename T>
xt::xarray<T>* ChunkCache<T>::loadChunk(int ix, int iy, int iz)
{
    if (!_ds) return nullptr;
    try {
        return readChunkFromDisk<T>(*_ds, static_cast<size_t>(ix), static_cast<size_t>(iy), static_cast<size_t>(iz));
    } catch (const std::exception&) {
        return nullptr;
    }
}

// Explicit template instantiations
template class ChunkCache<uint8_t>;
template class ChunkCache<uint16_t>;
