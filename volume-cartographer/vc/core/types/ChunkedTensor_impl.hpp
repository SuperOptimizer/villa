#pragma once

// This file contains template method implementations for ChunkedTensor.hpp
// Include this file in .cpp files that need to explicitly instantiate the templates

#include "vc/core/types/ChunkedTensor.hpp"

// ============================================================================
//  Chunked3d method implementations
// ============================================================================

template <typename T, typename C>
Chunked3d<T,C>::Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache<T> *cache)
    : _compute_f(compute_f), _ds(ds), _cache(cache)
{
    _border = compute_f.BORDER;
}

template <typename T, typename C>
Chunked3d<T,C>::~Chunked3d()
{
    if (!_persistent)
        remove_all(_cache_dir);
}

template <typename T, typename C>
Chunked3d<T,C>::Chunked3d(C &compute_f, z5::Dataset *ds, ChunkCache<T> *cache, const std::filesystem::path &cache_root)
    : _compute_f(compute_f), _ds(ds), _cache(cache)
{
    _border = compute_f.BORDER;

    if (_ds)
        _shape = {_ds->shape()[0],_ds->shape()[1],_ds->shape()[2]};

    if (cache_root.empty())
        return;

    if (!_compute_f.UNIQUE_ID_STRING.size())
        throw std::runtime_error("requested fs cache for compute function without identifier");

    std::filesystem::path root = cache_root/_compute_f.UNIQUE_ID_STRING;

    std::filesystem::create_directories(root);

    if (!_ds)
        _persistent = false;

    //create cache dir while others are competing to do the same
    for(int r=0;r<1000 && _cache_dir.empty();r++) {
        std::set<std::string> paths;
        if (_persistent) {
            for (auto const& entry : std::filesystem::directory_iterator(root))
                if (std::filesystem::is_directory(entry) && std::filesystem::exists(entry.path()/"meta.json") && std::filesystem::is_regular_file(entry.path()/"meta.json")) {
                    paths.insert(entry.path());
                    try {
                        std::ifstream meta_f(entry.path()/"meta.json");
                        nlohmann::json meta = nlohmann::json::parse(meta_f);
                        // Skip entries with invalid or non-existent dataset paths
                        if (!meta.contains("dataset_source_path") || !meta["dataset_source_path"].is_string())
                            continue;
                        std::filesystem::path src_candidate(meta["dataset_source_path"].get<std::string>());
                        if (!std::filesystem::exists(src_candidate))
                            continue;
                        if (!std::filesystem::exists(ds->path()))
                            continue;
                        std::filesystem::path src = std::filesystem::canonical(src_candidate);
                        std::filesystem::path cur = std::filesystem::canonical(ds->path());
                        if (src == cur) {
                            _cache_dir = entry.path();
                            break;
                        }
                    } catch (const std::exception&) {
                        // Ignore malformed cache entries or paths we cannot canonicalize
                        continue;
                    }
                }

            if (!_cache_dir.empty())
                continue;
        }

        //try generating our own cache dir atomically
        std::filesystem::path tmp_dir = cache_root/tmp_name_proc_thread();
        std::filesystem::create_directories(tmp_dir);

        if (_persistent) {
            nlohmann::json meta;
            meta["dataset_source_path"] = std::filesystem::canonical(ds->path()).string();
            std::ofstream o(tmp_dir/"meta.json");
            o << std::setw(4) << meta << std::endl;

            std::filesystem::path tgt_path;
            for(int i=0;i<1000;i++) {
                tgt_path = root/std::to_string(i);
                if (paths.count(tgt_path.string()))
                    continue;
                try {
                    std::filesystem::rename(tmp_dir, tgt_path);
                }
                catch (std::filesystem::filesystem_error&){
                    continue;
                }
                _cache_dir = tgt_path;
                break;
            }
        }
        else {
            _cache_dir = tmp_dir;
        }
    }

    if (_cache_dir.empty())
        throw std::runtime_error("could not create cache dir - maybe too many caches in cache root (max 1000!)");
}

template <typename T, typename C>
size_t Chunked3d<T,C>::calc_off(const cv::Vec3i &p)
{
    auto s = C::CHUNK_SIZE;
    return p[0] + p[1]*s + p[2]*s*s;
}

template <typename T, typename C>
T& Chunked3d<T,C>::operator()(const cv::Vec3i &p)
{
    auto s = C::CHUNK_SIZE;
    cv::Vec3i id = {p[0]/s,p[1]/s,p[2]/s};

    if (!_chunks.count(id))
        cache_chunk(id);

    return _chunks[id][calc_off({p[0]-id[0]*s,p[1]-id[1]*s,p[2]-id[2]*s})];
}

template <typename T, typename C>
T& Chunked3d<T,C>::operator()(int z, int y, int x)
{
    return operator()({z,y,x});
}

template <typename T, typename C>
T& Chunked3d<T,C>::safe_at(const cv::Vec3i &p)
{
    const auto s = C::CHUNK_SIZE;
    const cv::Vec3i id{ p[0]/s, p[1]/s, p[2]/s };

    {
        std::shared_lock<std::shared_mutex> rlock(_mutex);
        if (auto it = _chunks.find(id); it != _chunks.end()) {
            T* chunk = it->second;
            return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
        }
    }
    // compute/load outside shared lock
    T* chunk = cache_chunk_safe(id);
    return chunk[calc_off({p[0]-id[0]*s, p[1]-id[1]*s, p[2]-id[2]*s})];
}

template <typename T, typename C>
T& Chunked3d<T,C>::safe_at(int z, int y, int x)
{
    return safe_at({z,y,x});
}

template <typename T, typename C>
std::filesystem::path Chunked3d<T,C>::id_path(const std::filesystem::path &dir, const cv::Vec3i &id)
{
    return dir / (std::to_string(id[0]) + "." + std::to_string(id[1]) + "." + std::to_string(id[2]));
}

template <typename T, typename C>
T* Chunked3d<T,C>::cache_chunk_safe_mmap(const cv::Vec3i &id)
{
    auto s = C::CHUNK_SIZE;

    std::filesystem::path tgt_path = id_path(_cache_dir, id);
    size_t len = s*s*s;
    size_t len_bytes = len*sizeof(T);

    if (std::filesystem::exists(tgt_path)) {
        int fd = open(tgt_path.string().c_str(), O_RDWR);
        T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);

        _mutex.lock();
        if (!_chunks.count(id)) {
            _chunks[id] = chunk;
        }
        else {
#pragma omp atomic
            chunk_compute_collisions++;
            munmap(chunk, len_bytes);
            chunk = _chunks[id];
        }
#pragma omp atomic
        chunk_compute_total++;
        _mutex.unlock();

        return chunk;
    }

    std::filesystem::path tmp_path;
    _mutex.lock();
    std::stringstream ss;
    ss << this << "_" << std::this_thread::get_id() << "_" << _tmp_counter++;
    tmp_path = std::filesystem::path(_cache_dir) / ss.str();
    _mutex.unlock();
    int fd = open(tmp_path.string().c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
        const int err = errno;
        std::stringstream msg;
        msg << "Chunked3d: open failed for " << tmp_path << ": " << std::strerror(err);
        throw std::runtime_error(msg.str());
    }
    int ret = ftruncate(fd, len_bytes);
    if (ret != 0) {
        const int err = errno;
        close(fd);
        std::stringstream msg;
        msg << "Chunked3d: ftruncate failed for " << tmp_path
            << " (" << len_bytes << " bytes): " << std::strerror(err);
        throw std::runtime_error(msg.str());
    }
    T *chunk = (T*)mmap(NULL, len_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    cv::Vec3i offset =
    {id[0]*s-_border,
        id[1]*s-_border,
        id[2]*s-_border};

    CHUNKT small = xt::empty<T>({s,s,s});
    CHUNKT large;
    if (_ds) {
        large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);

        readArea3D(large, offset, _ds, _cache);
    }

    _compute_f.template compute<CHUNKT,T>(large, small, offset);

    for(int i=0;i<len;i++)
        chunk[i] = (&small(0,0,0))[i];

    _mutex.lock();
    if (!_chunks.count(id)) {
        _chunks[id] = chunk;
        int ret = rename(tmp_path.string().c_str(), tgt_path.string().c_str());

        if (ret)
            throw std::runtime_error("oops rename failed!");
    }
    else {
#pragma omp atomic
        chunk_compute_collisions++;
        munmap(chunk, len_bytes);
        unlink(tmp_path.string().c_str());
        chunk = _chunks[id];
    }
#pragma omp atomic
    chunk_compute_total++;
    _mutex.unlock();

    return chunk;
}

template <typename T, typename C>
T* Chunked3d<T,C>::cache_chunk_safe_alloc(const cv::Vec3i &id)
{
    auto s = C::CHUNK_SIZE;
    CHUNKT small = xt::empty<T>({s,s,s});

    cv::Vec3i offset =
    {id[0]*s-_border,
        id[1]*s-_border,
        id[2]*s-_border};

    CHUNKT large;
    if (_ds) {
        large = xt::empty<T>({s+2*_border,s+2*_border,s+2*_border});
        large = xt::full_like(large, C::FILL_V);

        readArea3D(large, offset, _ds, _cache);
    }

    _compute_f.template compute<CHUNKT,T>(large, small, offset);

    T *chunk = nullptr;

    _mutex.lock();
    if (!_chunks.count(id)) {
        chunk = (T*)malloc(s*s*s*sizeof(T));
        memcpy(chunk, &small(0,0,0), s*s*s*sizeof(T));
        _chunks[id] = chunk;
    }
    else {
#pragma omp atomic
        chunk_compute_collisions++;
        chunk = _chunks[id];
    }
#pragma omp atomic
    chunk_compute_total++;
    _mutex.unlock();

    return chunk;
}

template <typename T, typename C>
T* Chunked3d<T,C>::cache_chunk_safe(const cv::Vec3i &id)
{
    if (_cache_dir.empty())
        return cache_chunk_safe_alloc(id);
    else
        return cache_chunk_safe_mmap(id);
}

template <typename T, typename C>
T* Chunked3d<T,C>::cache_chunk(const cv::Vec3i &id)
{
    return cache_chunk_safe(id);
}

template <typename T, typename C>
T* Chunked3d<T,C>::chunk(const cv::Vec3i &id)
{
    if (!_chunks.count(id))
        return cache_chunk(id);
    return _chunks[id];
}

template <typename T, typename C>
T* Chunked3d<T,C>::chunk_safe(const cv::Vec3i &id)
{
    T *chunk = nullptr;
    _mutex.lock_shared();
    if (_chunks.count(id)) {
        chunk = _chunks[id];
        _mutex.unlock();
    }
    else {
        _mutex.unlock();
        chunk = cache_chunk_safe(id);
    }

    return chunk;
}

template <typename T, typename C>
std::vector<int> Chunked3d<T,C>::shape()
{
    return _shape;
}

// ============================================================================
//  Chunked3dAccessor method implementations
// ============================================================================

template <typename T, typename C>
Chunked3dAccessor<T,C>::Chunked3dAccessor(Chunked3d<T,C> &ar) : _ar(ar) {}

template <typename T, typename C>
Chunked3dAccessor<T,C> Chunked3dAccessor<T,C>::create(Chunked3d<T,C> &ar)
{
    return Chunked3dAccessor(ar);
}

template <typename T, typename C>
T& Chunked3dAccessor<T,C>::operator()(const cv::Vec3i &p)
{
    auto s = C::CHUNK_SIZE;

    if (_corner[0] == -1)
        get_chunk(p);
    else {
        bool miss = false;
        for(int i=0;i<3;i++)
            if (p[i] < _corner[i])
                miss = true;
        for(int i=0;i<3;i++)
            if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                miss = true;
        if (miss)
            get_chunk(p);
    }

    total++;

    return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
}

template <typename T, typename C>
T& Chunked3dAccessor<T,C>::operator()(int z, int y, int x)
{
    return operator()({z,y,x});
}

template <typename T, typename C>
T& Chunked3dAccessor<T,C>::safe_at(const cv::Vec3i &p)
{
    auto s = C::CHUNK_SIZE;

    if (_corner[0] == -1)
        get_chunk_safe(p);
    else {
        bool miss = false;
        for(int i=0;i<3;i++)
            if (p[i] < _corner[i])
                miss = true;
        for(int i=0;i<3;i++)
            if (p[i] >= _corner[i]+C::CHUNK_SIZE)
                miss = true;
        if (miss)
            get_chunk_safe(p);
    }

    #pragma omp atomic
    total++;

    return _chunk[_ar.calc_off({p[0]-_corner[0],p[1]-_corner[1],p[2]-_corner[2]})];
}

template <typename T, typename C>
T& Chunked3dAccessor<T,C>::safe_at(int z, int y, int x)
{
    return safe_at({z,y,x});
}

template <typename T, typename C>
void Chunked3dAccessor<T,C>::get_chunk(const cv::Vec3i &p)
{
    miss++;
    cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
    _chunk = _ar.chunk(id);
    _corner = id*C::CHUNK_SIZE;
}

template <typename T, typename C>
void Chunked3dAccessor<T,C>::get_chunk_safe(const cv::Vec3i &p)
{
    #pragma omp atomic
    miss++;
    cv::Vec3i id = {p[0]/C::CHUNK_SIZE,p[1]/C::CHUNK_SIZE,p[2]/C::CHUNK_SIZE};
    _chunk = _ar.chunk_safe(id);
    _corner = id*C::CHUNK_SIZE;
}

// ============================================================================
//  CachedChunked3dInterpolator method implementations
// ============================================================================

template <typename T, typename C>
CachedChunked3dInterpolator<T,C>::CachedChunked3dInterpolator(Chunked3d<T,C>& ar)
    : _ar(ar), _shape(ar.shape())
{}

template <typename T, typename C>
double CachedChunked3dInterpolator<T,C>::val(const double& v) const
{
    return v;
}

template <typename T, typename C>
Chunked3dAccessor<T,C>& CachedChunked3dInterpolator<T,C>::local_accessor() const
{
    // Per-thread, bounded cache keyed by the underlying array address.
    // (Multiple interpolators over the same array share one accessor.)
    struct TLS {
        std::unordered_map<const void*, Chunked3dAccessor<T,C>> map;
        std::deque<const void*> order;          // FIFO ~ LRU-ish
    };
    thread_local TLS tls;

    constexpr std::size_t kMax = CCI_TLS_MAX;

    const void* key = static_cast<const void*>(&_ar);

    if (auto it = tls.map.find(key); it != tls.map.end()) {
        return it->second;
    }

    // Evict oldest if at capacity
    if (tls.map.size() >= kMax && !tls.order.empty()) {
        const void* old = tls.order.front();
        tls.order.pop_front();
        tls.map.erase(old);
    }

    auto [it2, inserted] = tls.map.emplace(key, Chunked3dAccessor<T,C>{_ar});
    if (inserted) tls.order.push_back(key);
    return it2->second;
}
