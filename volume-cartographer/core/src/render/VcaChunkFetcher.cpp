#include "vc/core/render/VcaChunkFetcher.hpp"

#include <vc.h> // libvc (vendored at libs/vc); extern "C" guarded internally

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vc::render {

namespace {

// Owns the mmap'd .vca file and the libvc reader handle. vc_open BORROWS the
// buffer, so the mapping must outlive the handle — both live here together and
// are torn down in reverse on destruction. Shared by all per-LOD fetchers.
class VcaArchive {
public:
    explicit VcaArchive(const std::filesystem::path& path)
    {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("vca: cannot open " + path.string());
        }
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
            ::close(fd_);
            throw std::runtime_error("vca: cannot stat " + path.string());
        }
        len_ = static_cast<std::size_t>(st.st_size);
        void* p = ::mmap(nullptr, len_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("vca: mmap failed for " + path.string());
        }
        buf_ = static_cast<const std::uint8_t*>(p);
        archive_ = vc_open(buf_, len_);
        if (!archive_) {
            ::munmap(const_cast<std::uint8_t*>(buf_), len_);
            ::close(fd_);
            throw std::runtime_error("vca: vc_open rejected " + path.string());
        }
    }

    ~VcaArchive()
    {
        if (archive_) vc_close(archive_);
        if (buf_) ::munmap(const_cast<std::uint8_t*>(buf_), len_);
        if (fd_ >= 0) ::close(fd_);
    }

    VcaArchive(const VcaArchive&) = delete;
    VcaArchive& operator=(const VcaArchive&) = delete;

    vc_archive* handle() const { return archive_; }

private:
    int fd_ = -1;
    const std::uint8_t* buf_ = nullptr;
    std::size_t len_ = 0;
    vc_archive* archive_ = nullptr;
};

// One fetcher per LOD. Decodes a single 32^3 atom on fetch().
class VcaChunkFetcher final : public IChunkFetcher {
public:
    VcaChunkFetcher(std::shared_ptr<VcaArchive> archive, int level)
        : archive_(std::move(archive)), level_(level)
    {
    }

    ChunkFetchResult fetch(const ChunkKey& key) override
    {
        ChunkFetchResult result;
        vc_archive* a = archive_->handle();

        // ChunkKey grid coords (iz,iy,ix) == libvc atom coords (az,ay,ax).
        const std::uint32_t az = static_cast<std::uint32_t>(key.iz);
        const std::uint32_t ay = static_cast<std::uint32_t>(key.iy);
        const std::uint32_t ax = static_cast<std::uint32_t>(key.ix);

        const vc_cover cover = vc_atom_coverage(a, level_, az, ay, ax);
        if (cover != VC_PRESENT) {
            // ABSENT or KNOWN_ZERO: no content. ChunkCache renders this as
            // all-fill (zeros). For a finished local archive there is nothing to
            // download, so Missing is the right terminal state.
            result.status = ChunkFetchStatus::Missing;
            return result;
        }

        // libvc decode output is raster (z,y,x) — exactly the byte order
        // ChunkResult::bytes is indexed in (ChunkedPlaneSampler), so no transpose.
        auto bytes = std::vector<std::byte>(VC_ATOM3);
        const vc_status s = vc_decode_atom(
            a, level_, static_cast<int>(ax), static_cast<int>(ay),
            static_cast<int>(az), reinterpret_cast<std::uint8_t*>(bytes.data()));
        if (s != VC_OK) {
            result.status = ChunkFetchStatus::DecodeError;
            result.message = "vc_decode_atom status " + std::to_string(s);
            return result;
        }
        result.status = ChunkFetchStatus::Found;
        result.bytes = std::move(bytes);
        return result;
    }

private:
    std::shared_ptr<VcaArchive> archive_;
    int level_;
};

} // namespace

OpenedChunkedZarr openVcaArchive(const std::filesystem::path& vcaPath)
{
    auto archive = std::make_shared<VcaArchive>(vcaPath);
    vc_archive* a = archive->handle();

    OpenedChunkedZarr opened;
    opened.dtype = ChunkDtype::UInt8;
    opened.fillValue = 0.0;

    for (int lod = 0; lod < VC_NLOD; ++lod) {
        vc_dims d{};
        if (vc_lod_dims(a, lod, &d) != VC_OK) {
            break; // pyramid ends at the first LOD the archive doesn't carry
        }

        opened.levelNumbers.push_back(lod);
        // VC3D shape/chunkShape are [z,y,x]; libvc dims are (nx,ny,nz).
        opened.shapes.push_back({static_cast<int>(d.nz),
                                 static_cast<int>(d.ny),
                                 static_cast<int>(d.nx)});
        const std::array<int, 3> chunk32{static_cast<int>(VC_ATOM),
                                         static_cast<int>(VC_ATOM),
                                         static_cast<int>(VC_ATOM)};
        opened.chunkShapes.push_back(chunk32);
        opened.storageChunkShapes.push_back(chunk32);

        IChunkedArray::LevelTransform t;
        // Map a LOD0 voxel coord DOWN to this level: divide by 2^lod (the sampler
        // computes levelCoord = p0 * scaleFromLevel0). Same convention as zarr.
        const double invScale = 1.0 / static_cast<double>(std::uint64_t{1} << lod);
        t.scaleFromLevel0 = {invScale, invScale, invScale};
        t.offsetFromLevel0 = {0.0, 0.0, 0.0};
        opened.transforms.push_back(t);

        opened.fetchers.push_back(
            std::make_shared<VcaChunkFetcher>(archive, lod));
    }

    if (opened.fetchers.empty()) {
        throw std::runtime_error("vca: no LODs in " + vcaPath.string());
    }
    return opened;
}

} // namespace vc::render
