#include "vc/core/util/ChunkCache.hpp"

#include <algorithm>

using namespace volcart::zarr;

// Helper to read a chunk from disk via IChunkSource
template<typename T>
static std::shared_ptr<Tensor3D<T>> readChunkFromSource(IChunkSource& src, size_t iz, size_t iy, size_t ix)
{
    Dtype dtype = src.volDtype();
    if (dtype != Dtype::UInt8 && dtype != Dtype::UInt16)
        throw std::runtime_error("only uint8_t/uint16 sources supported currently!");

    const auto chunkShape = src.volChunkShape();

    auto out = std::make_shared<Tensor3D<T>>(chunkShape[0], chunkShape[1], chunkShape[2]);

    bool ok = false;
    if (dtype == Dtype::UInt8) {
        if constexpr (std::is_same_v<T, uint8_t>) {
            ok = src.volReadChunk(iz, iy, ix, out->data());
        } else {
            throw std::runtime_error("Cannot read uint8 dataset into uint16 array");
        }
    }
    else if (dtype == Dtype::UInt16) {
        if constexpr (std::is_same_v<T, uint16_t>) {
            ok = src.volReadChunk(iz, iy, ix, out->data());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            size_t chunkElems = src.volChunkElements();
            Tensor3D<uint16_t> tmp(chunkShape[0], chunkShape[1], chunkShape[2]);
            ok = src.volReadChunk(iz, iy, ix, tmp.data());

            if (ok) {
                uint8_t* p8 = out->data();
                uint16_t* p16 = tmp.data();
                for (size_t i = 0; i < chunkElems; i++)
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
    if (_src && _cz > 0 && _cy > 0 && _cx > 0) {
        size_t chunkBytes = static_cast<size_t>(_cz) * _cy * _cx * sizeof(T);
        _maxChunks = (_maxBytes > 0) ? (_maxBytes / chunkBytes) : 0;
    }
}

template<typename T>
ChunkCache<T>::~ChunkCache()
{
    clear();
}

template<typename T>
void ChunkCache<T>::init(IChunkSource* src)
{
    if (_src == src) return;

    clear();

    if (!src) {
        _src = nullptr;
        return;
    }

    const auto chunkShape = src->volChunkShape();
    _cz = static_cast<int>(chunkShape[0]);
    _cy = static_cast<int>(chunkShape[1]);
    _cx = static_cast<int>(chunkShape[2]);

    _czShift = log2_pow2(_cz);
    _cyShift = log2_pow2(_cy);
    _cxShift = log2_pow2(_cx);
    _czMask = _cz - 1;
    _cyMask = _cy - 1;
    _cxMask = _cx - 1;

    const auto dsShape = src->volShape();
    _sz = static_cast<int>(dsShape[0]);
    _sy = static_cast<int>(dsShape[1]);
    _sx = static_cast<int>(dsShape[2]);

    _chunksZ = (_sz + _cz - 1) / _cz;
    _chunksY = (_sy + _cy - 1) / _cy;
    _chunksX = (_sx + _cx - 1) / _cx;

    size_t chunkBytes = static_cast<size_t>(_cz) * _cy * _cx * sizeof(T);
    _maxChunks = (_maxBytes > 0) ? (_maxBytes / chunkBytes) : 0;

    _totalSlots = static_cast<size_t>(_chunksZ) * _chunksY * _chunksX;
    _chunks = std::make_unique<std::atomic<ChunkPtr>[]>(_totalSlots);
    _rawChunks = std::make_unique<std::atomic<const Tensor3D<T>*>[]>(_totalSlots);
    for (size_t i = 0; i < _totalSlots; i++)
        _rawChunks[i].store(nullptr, std::memory_order_relaxed);
    _lastAccess = std::make_unique<std::atomic<uint64_t>[]>(_totalSlots);

    _cachedCount.store(0, std::memory_order_relaxed);
    _generation.store(0, std::memory_order_relaxed);

    _src = src;
}

template<typename T>
auto ChunkCache<T>::get(int iz, int iy, int ix) -> ChunkPtr
{
    if (iz < 0 || iz >= _chunksZ || iy < 0 || iy >= _chunksY || ix < 0 || ix >= _chunksX)
        return nullptr;

    size_t i = idx(iz, iy, ix);

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

    ChunkPtr newChunk = loadChunk(iz, iy, ix);
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
Tensor3D<T>* ChunkCache<T>::getRaw(int iz, int iy, int ix)
{
    auto ptr = get(iz, iy, ix);
    return ptr.get();
}

template<typename T>
const Tensor3D<T>* ChunkCache<T>::getRawFast(int iz, int iy, int ix) const
{
    if (iz < 0 || iz >= _chunksZ || iy < 0 || iy >= _chunksY || ix < 0 || ix >= _chunksX)
        return nullptr;

    size_t i = idx(iz, iy, ix);
    return _rawChunks[i].load(std::memory_order_relaxed);
}

template<typename T>
auto ChunkCache<T>::getIfCached(int iz, int iy, int ix) const -> ChunkPtr
{
    if (iz < 0 || iz >= _chunksZ || iy < 0 || iy >= _chunksY || ix < 0 || ix >= _chunksX)
        return nullptr;

    size_t i = idx(iz, iy, ix);
    return _chunks[i].load(std::memory_order_acquire);
}

template<typename T>
void ChunkCache<T>::prefetch(int minIz, int minIy, int minIx, int maxIz, int maxIy, int maxIx)
{
    minIz = std::max(0, minIz);
    minIy = std::max(0, minIy);
    minIx = std::max(0, minIx);
    maxIz = std::min(_chunksZ - 1, maxIz);
    maxIy = std::min(_chunksY - 1, maxIy);
    maxIx = std::min(_chunksX - 1, maxIx);

    #pragma omp parallel for collapse(3) schedule(dynamic, 1)
    for (int ix = minIx; ix <= maxIx; ix++) {
        for (int iy = minIy; iy <= maxIy; iy++) {
            for (int iz = minIz; iz <= maxIz; iz++) {
                get(iz, iy, ix);
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

    _src = nullptr;
    _cz = _cy = _cx = 0;
    _sz = _sy = _sx = 0;
    _chunksZ = _chunksY = _chunksX = 0;
    _czShift = _cyShift = _cxShift = 0;
    _czMask = _cyMask = _cxMask = 0;
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
auto ChunkCache<T>::loadChunk(int iz, int iy, int ix) -> ChunkPtr
{
    if (!_src) return nullptr;
    try {
        return readChunkFromSource<T>(*_src, static_cast<size_t>(iz), static_cast<size_t>(iy), static_cast<size_t>(ix));
    } catch (const std::exception&) {
        return nullptr;
    }
}

// Explicit template instantiations
template class ChunkCache<uint8_t>;
template class ChunkCache<uint16_t>;
