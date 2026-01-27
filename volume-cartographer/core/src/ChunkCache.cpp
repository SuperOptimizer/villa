#include "vc/core/util/ChunkCache.hpp"

#include <algorithm>
#include <vector>
#include <cassert>

template<typename T>
ChunkCache<T>::ChunkCache(size_t size) : _size(size)
{
}

template<typename T>
ChunkCache<T>::~ChunkCache()
{
    for (auto& it : _store) {
        it.second.reset();
    }
}

template<typename T>
int ChunkCache<T>::groupIdx(const std::string& name)
{
    if (!_group_store.count(name)) {
        _group_store[name] = _group_store.size() + 1;
    }
    return _group_store[name];
}

template<typename T>
void ChunkCache<T>::put(const cv::Vec4i& idx, volcart::zarr::Tensor3D<T>* ar)
{
    // Evict old entries if cache is full
    if (_stored >= _size) {
        using KP = std::pair<cv::Vec4i, uint64_t>;
        std::vector<KP> gen_list(_gen_store.begin(), _gen_store.end());
        std::sort(gen_list.begin(), gen_list.end(),
                  [](const KP& a, const KP& b) { return a.second < b.second; });

        for (const auto& it : gen_list) {
            std::shared_ptr<volcart::zarr::Tensor3D<T>> cached = _store[it.first];
            if (cached.get()) {
                size_t bytes = cached->nbytes();
                _stored -= bytes;
            }
            // Always erase entries (including null entries for non-existent chunks)
            _store.erase(it.first);
            _gen_store.erase(it.first);

            // We delete 10% of cache content to amortize sorting costs
            if (_stored < 0.9 * _size) {
                break;
            }
        }
    }

    // Update storage accounting
    if (ar) {
        if (_store.count(idx)) {
            assert(_store[idx].get());
            _stored -= _store[idx]->nbytes();
        }
        _stored += ar->nbytes();
    }

    _store[idx].reset(ar);
    _generation++;
    _gen_store[idx] = _generation;
}

template<typename T>
std::shared_ptr<volcart::zarr::Tensor3D<T>> ChunkCache<T>::get(const cv::Vec4i& idx)
{
    auto res = _store.find(idx);
    if (res == _store.end()) {
        return nullptr;
    }

    _generation++;
    _gen_store[idx] = _generation;

    return res->second;
}

template<typename T>
bool ChunkCache<T>::has(const cv::Vec4i& idx)
{
    return _store.count(idx) > 0;
}

template<typename T>
void ChunkCache<T>::reset()
{
    _gen_store.clear();
    _group_store.clear();
    _store.clear();

    _generation = 0;
    _stored = 0;
}

template<typename T>
void ChunkCache<T>::getStats(size_t& storedBytes, size_t& maxBytes, size_t& numEntries, size_t& numNullEntries) const
{
    storedBytes = _stored;
    maxBytes = _size;
    numEntries = _store.size();
    numNullEntries = 0;
    for (const auto& it : _store) {
        if (!it.second.get()) {
            numNullEntries++;
        }
    }
}

// Explicit template instantiations
template class ChunkCache<uint8_t>;
template class ChunkCache<uint16_t>;
