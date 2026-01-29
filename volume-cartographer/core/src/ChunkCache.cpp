#include "vc/core/util/ChunkCache.hpp"

#include <algorithm>

using namespace volcart::zarr;

// Helper to read a chunk from disk
template<typename T>
static std::shared_ptr<Tensor3D<T>> readChunkFromDisk(ZarrDataset& ds, size_t ix, size_t iy, size_t iz)
{
    std::vector<std::size_t> chunkId = {ix, iy, iz};

    Dtype dtype = ds.getDtype();
    if (dtype != Dtype::UInt8 && dtype != Dtype::UInt16)
        throw std::runtime_error("only uint8_t/uint16 zarrs supported currently!");

    const auto& maxChunkShape = ds.chunkShape();

    auto out = std::make_shared<Tensor3D<T>>(maxChunkShape[0], maxChunkShape[1], maxChunkShape[2]);

    bool ok = false;
    if (dtype == Dtype::UInt8) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            ok = ds.readChunk(chunkId, out->data());
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (dtype == Dtype::UInt16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            ok = ds.readChunk(chunkId, out->data());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            Tensor3D<uint16_t> tmp(maxChunkShape[0], maxChunkShape[1], maxChunkShape[2]);
            ok = ds.readChunk(chunkId, tmp.data());

            if (ok) {
                const size_t maxChunkSize = ds.defaultChunkSize();
                uint8_t* p8 = out->data();
                uint16_t* p16 = tmp.data();
                for (size_t i = 0; i < maxChunkSize; i++)
                    p8[i] = p16[i] / 257;
            }
        }
    }

    if (!ok) return nullptr;
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
void ChunkCache<T>::init(ZarrDataset* ds)
{
    if (_ds == ds) return;

    clear();

    if (!ds) {
        _ds = nullptr;
        return;
    }

    const auto& blockShape = ds->chunkShape();
    _cw = static_cast<int>(blockShape[0]);
    _ch = static_cast<int>(blockShape[1]);
    _cd = static_cast<int>(blockShape[2]);

    _cwShift = log2_pow2(_cw);
    _chShift = log2_pow2(_ch);
    _cdShift = log2_pow2(_cd);
    _cwMask = _cw - 1;
    _chMask = _ch - 1;
    _cdMask = _cd - 1;

    const auto& dsShape = ds->shape();
    _sx = static_cast<int>(dsShape[0]);
    _sy = static_cast<int>(dsShape[1]);
    _sz = static_cast<int>(dsShape[2]);

    _chunksX = (_sx + _cw - 1) / _cw;
    _chunksY = (_sy + _ch - 1) / _ch;
    _chunksZ = (_sz + _cd - 1) / _cd;

    size_t chunkBytes = static_cast<size_t>(_cw) * _ch * _cd * sizeof(T);
    _maxChunks = (_maxBytes > 0) ? (_maxBytes / chunkBytes) : 0;

    _totalSlots = static_cast<size_t>(_chunksX) * _chunksY * _chunksZ;
    _chunks = std::make_unique<std::atomic<ChunkPtr>[]>(_totalSlots);
    _rawChunks = std::make_unique<std::atomic<const Tensor3D<T>*>[]>(_totalSlots);
    for (size_t i = 0; i < _totalSlots; i++)
        _rawChunks[i].store(nullptr, std::memory_order_relaxed);
    _lastAccess = std::make_unique<std::atomic<uint64_t>[]>(_totalSlots);

    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);

    _ds = ds;
}

template<typename T>
auto ChunkCache<T>::get(int ix, int iy, int iz) -> ChunkPtr
{
    if (ix < 0 || ix >= _chunksX || iy < 0 || iy >= _chunksY || iz < 0 || iz >= _chunksZ)
        return nullptr;

    size_t i = idx(ix, iy, iz);

    // Fast path: lock-free atomic read
    ChunkPtr cached = _chunks[i].load(std::memory_order_acquire);
    if (cached) {
        _lastAccess[i].store(_generation.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        return cached;
    }

    // Slow path: load from disk (per-chunk lock to avoid duplicate reads)
    std::lock_guard<std::mutex> diskLock(_lockPool[i % kLockPoolSize]);

    // Re-check after acquiring disk lock
    cached = _chunks[i].load(std::memory_order_acquire);
    if (cached) {
        _lastAccess[i].store(_generation.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
        return cached;
    }

    ChunkPtr newChunk = loadChunk(ix, iy, iz);
    if (!newChunk) return nullptr;

    {
        std::lock_guard<std::mutex> evictLock(_evictionMutex);
        if (_maxChunks > 0 && _cachedCount.load(std::memory_order_relaxed) >= _maxChunks) {
            evictIfNeeded();
        }
    }

    _rawChunks[i].store(newChunk.get(), std::memory_order_relaxed);
    _chunks[i].store(newChunk, std::memory_order_release);
    _lastAccess[i].store(_generation.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);
    _cachedCount.fetch_add(1, std::memory_order_relaxed);
    return newChunk;
}

template<typename T>
Tensor3D<T>* ChunkCache<T>::getRaw(int ix, int iy, int iz)
{
    auto ptr = get(ix, iy, iz);
    return ptr.get();
}

template<typename T>
const Tensor3D<T>* ChunkCache<T>::getRawFast(int ix, int iy, int iz) const
{
    if (ix < 0 || ix >= _chunksX || iy < 0 || iy >= _chunksY || iz < 0 || iz >= _chunksZ)
        return nullptr;

    size_t i = idx(ix, iy, iz);
    return _rawChunks[i].load(std::memory_order_relaxed);
}

template<typename T>
auto ChunkCache<T>::getIfCached(int ix, int iy, int iz) const -> ChunkPtr
{
    if (ix < 0 || ix >= _chunksX || iy < 0 || iy >= _chunksY || iz < 0 || iz >= _chunksZ)
        return nullptr;

    size_t i = idx(ix, iy, iz);
    return _chunks[i].load(std::memory_order_acquire);
}

template<typename T>
void ChunkCache<T>::prefetch(int minIx, int minIy, int minIz, int maxIx, int maxIy, int maxIz)
{
    minIx = std::max(0, minIx);
    minIy = std::max(0, minIy);
    minIz = std::max(0, minIz);
    maxIx = std::min(_chunksX - 1, maxIx);
    maxIy = std::min(_chunksY - 1, maxIy);
    maxIz = std::min(_chunksZ - 1, maxIz);

    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int iz = minIz; iz <= maxIz; iz++) {
        for (int iy = minIy; iy <= maxIy; iy++) {
            for (int ix = minIx; ix <= maxIx; ix++) {
                get(ix, iy, iz);
            }
        }
    }
}

template<typename T>
void ChunkCache<T>::clear()
{
    for (size_t i = 0; i < _totalSlots; i++) {
        _rawChunks[i].store(nullptr, std::memory_order_relaxed);
        _chunks[i].store(nullptr, std::memory_order_relaxed);
    }
    _chunks.reset();
    _rawChunks.reset();
    _lastAccess.reset();
    _totalSlots = 0;

    _ds = nullptr;
    _cw = _ch = _cd = 0;
    _sx = _sy = _sz = 0;
    _chunksX = _chunksY = _chunksZ = 0;
    _cwShift = _chShift = _cdShift = 0;
    _cwMask = _chMask = _cdMask = 0;
    _maxChunks = 0;
    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);
}

template<typename T>
void ChunkCache<T>::flush()
{
    for (size_t i = 0; i < _totalSlots; i++) {
        _rawChunks[i].store(nullptr, std::memory_order_relaxed);
        _chunks[i].store(nullptr, std::memory_order_relaxed);
    }
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

    struct EvictCandidate {
        size_t index;
        uint64_t lastAccess;
    };
    std::vector<EvictCandidate> candidates;
    candidates.reserve(currentCount);

    for (size_t i = 0; i < _totalSlots; i++) {
        if (_chunks[i].load(std::memory_order_relaxed)) {
            candidates.push_back({i, _lastAccess[i].load(std::memory_order_relaxed)});
        }
    }

    if (candidates.empty()) return;

    std::sort(candidates.begin(), candidates.end(),
              [](const EvictCandidate& a, const EvictCandidate& b) {
                  return a.lastAccess < b.lastAccess;
              });

    size_t target = _maxChunks * 3 / 4;
    size_t toEvict = (currentCount > target) ? (currentCount - target) : 1;
    size_t evicted = 0;

    for (size_t j = 0; j < toEvict && j < candidates.size(); j++) {
        _rawChunks[candidates[j].index].store(nullptr, std::memory_order_relaxed);
        _chunks[candidates[j].index].store(nullptr, std::memory_order_release);
        evicted++;
    }

    _cachedCount.fetch_sub(evicted, std::memory_order_relaxed);
}

template<typename T>
auto ChunkCache<T>::loadChunk(int ix, int iy, int iz) -> ChunkPtr
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
