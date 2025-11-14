#include "vc/core/util/ChunkCache.hpp"

template<typename T>
ChunkCache<T>::~ChunkCache()
{
    for(auto &it : _store)
        it.second.reset();
}

template<typename T>
int ChunkCache<T>::groupIdx(const std::string &name)
{
    if (!_group_store.count(name))
        _group_store[name] = _group_store.size()+1;

     return _group_store[name];
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
void ChunkCache<T>::put(const cv::Vec4i &idx, xt::xarray<T> *ar)
{
    if (_stored >= _size) {
        using KP = std::pair<cv::Vec4i, uint64_t>;
        std::vector<KP> gen_list(_gen_store.begin(), _gen_store.end());
        std::sort(gen_list.begin(), gen_list.end(), [](const KP &a, const KP &b){ return a.second < b.second; });
        for(const auto& it : gen_list) {
            std::shared_ptr<xt::xarray<T>> cached_ar = _store[it.first];
            if (cached_ar.get()) {
                size_t size = cached_ar.get()->storage().size() * sizeof(T);
                cached_ar.reset();
                _stored -= size;

                _store.erase(it.first);
                _gen_store.erase(it.first);
            }

            //we delete 10% of cache content to amortize sorting costs
            if (_stored < 0.9*_size) {
                break;
            }
        }
    }

    if (ar) {
        if (_store.count(idx)) {
            assert(_store[idx].get());
            _stored -= _store[idx]->size() * sizeof(T);
        }
        _stored += ar->size() * sizeof(T);
    }
    _store[idx].reset(ar);
    _generation++;
    _gen_store[idx] = _generation;
}

template<typename T>
std::shared_ptr<xt::xarray<T>> ChunkCache<T>::get(const cv::Vec4i &idx)
{
    auto res = _store.find(idx);
    if (res == _store.end())
        return nullptr;

    _generation++;
    _gen_store[idx] = _generation;

    return res->second;
}

template<typename T>
bool ChunkCache<T>::has(const cv::Vec4i &idx)
{
    return _store.count(idx);
}

// Explicit template instantiations for the types we support
template class ChunkCache<uint8_t>;
template class ChunkCache<uint16_t>;
