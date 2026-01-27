#include "vc/core/zarr/ZarrDataset.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace volcart::zarr
{

// --- Dtype utilities ---

std::string dtypeToString(Dtype dtype)
{
    switch (dtype) {
        case Dtype::UInt8:
            return "<u1";
        case Dtype::UInt16:
            return "<u2";
        case Dtype::Float32:
            return "<f4";
        default:
            return "";
    }
}

Dtype dtypeFromString(const std::string& s)
{
    // Handle both little-endian (<) and native (|) byte order markers
    if (s == "<u1" || s == "|u1" || s == "uint8") {
        return Dtype::UInt8;
    }
    if (s == "<u2" || s == ">u2" || s == "uint16") {
        return Dtype::UInt16;
    }
    if (s == "<f4" || s == ">f4" || s == "float32") {
        return Dtype::Float32;
    }
    return Dtype::Unknown;
}

std::size_t dtypeSize(Dtype dtype)
{
    switch (dtype) {
        case Dtype::UInt8:
            return 1;
        case Dtype::UInt16:
            return 2;
        case Dtype::Float32:
            return 4;
        default:
            return 0;
    }
}

// --- ZarrDataset implementation ---

ZarrDataset::ZarrDataset(const std::filesystem::path& path) : path_(path)
{
    if (!std::filesystem::exists(path_)) {
        throw std::runtime_error(
            "ZarrDataset: path does not exist: " + path_.string());
    }
    loadMetadata();
}

ZarrDataset::ZarrDataset(
    const std::filesystem::path& path,
    const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& chunks,
    Dtype dtype,
    const std::string& compressor,
    const nlohmann::json& compressorOpts)
    : path_(path), shape_(shape), chunks_(chunks), dtype_(dtype)
{
    // Create directory
    std::filesystem::create_directories(path_);

    // Setup compressor
    if (compressor == "blosc") {
        codec_ = std::make_unique<BloscCodec>(compressorOpts);
    }

    // Default fill value based on dtype
    switch (dtype_) {
        case Dtype::UInt8:
        case Dtype::UInt16:
            fillValue_ = 0;
            break;
        case Dtype::Float32:
            fillValue_ = 0.0;
            break;
        default:
            fillValue_ = nullptr;
    }

    writeMetadata();
}

void ZarrDataset::loadMetadata()
{
    auto zarrayPath = path_ / ".zarray";
    if (!std::filesystem::exists(zarrayPath)) {
        throw std::runtime_error(
            "ZarrDataset: .zarray not found at " + zarrayPath.string());
    }

    std::ifstream f(zarrayPath);
    nlohmann::json meta;
    f >> meta;

    // Parse shape
    shape_ = meta["shape"].get<std::vector<std::size_t>>();

    // Parse chunks
    chunks_ = meta["chunks"].get<std::vector<std::size_t>>();

    // Parse dtype
    dtype_ = dtypeFromString(meta["dtype"].get<std::string>());
    if (dtype_ == Dtype::Unknown) {
        throw std::runtime_error(
            "ZarrDataset: unsupported dtype: " +
            meta["dtype"].get<std::string>());
    }

    // Parse dimension separator (defaults to '/')
    if (meta.contains("dimension_separator")) {
        auto sep = meta["dimension_separator"].get<std::string>();
        if (!sep.empty()) {
            dimSeparator_ = sep[0];
        }
    }

    // Parse compressor
    if (meta.contains("compressor") && !meta["compressor"].is_null()) {
        auto compMeta = meta["compressor"];
        if (compMeta.contains("id") && compMeta["id"] == "blosc") {
            codec_ = std::make_unique<BloscCodec>(compMeta);
        }
    }

    // Parse fill_value
    if (meta.contains("fill_value")) {
        fillValue_ = meta["fill_value"];
    }
}

void ZarrDataset::writeMetadata() const
{
    nlohmann::json meta;
    meta["zarr_format"] = 2;
    meta["shape"] = shape_;
    meta["chunks"] = chunks_;
    meta["dtype"] = dtypeToString(dtype_);
    meta["order"] = "C";  // C-order (row-major in numpy terms, but we use column-major internally)
    meta["fill_value"] = fillValue_;
    meta["dimension_separator"] = std::string(1, dimSeparator_);

    if (codec_) {
        meta["compressor"] = codec_->toJson();
    } else {
        meta["compressor"] = nullptr;
    }

    meta["filters"] = nullptr;

    auto zarrayPath = path_ / ".zarray";
    std::ofstream f(zarrayPath);
    f << meta.dump(2);
}

std::size_t ZarrDataset::defaultChunkSize() const noexcept
{
    std::size_t size = 1;
    for (auto c : chunks_) {
        size *= c;
    }
    return size;
}

bool ZarrDataset::chunkExists(const std::vector<std::size_t>& chunkId) const
{
    return std::filesystem::exists(chunkPath(chunkId));
}

void ZarrDataset::getChunkShape(
    const std::vector<std::size_t>& chunkId,
    std::vector<std::size_t>& shapeOut) const
{
    shapeOut.resize(shape_.size());
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        std::size_t chunkStart = chunkId[i] * chunks_[i];
        std::size_t chunkEnd = std::min(chunkStart + chunks_[i], shape_[i]);
        shapeOut[i] = chunkEnd - chunkStart;
    }
}

std::filesystem::path ZarrDataset::chunkPath(
    const std::vector<std::size_t>& chunkId) const
{
    std::string name;
    for (std::size_t i = 0; i < chunkId.size(); ++i) {
        if (i > 0) {
            name += dimSeparator_;
        }
        name += std::to_string(chunkId[i]);
    }
    return path_ / name;
}

void ZarrDataset::ensureChunkDir(const std::vector<std::size_t>& chunkId) const
{
    auto cp = chunkPath(chunkId);
    auto parentDir = cp.parent_path();
    if (!parentDir.empty() && !std::filesystem::exists(parentDir)) {
        std::filesystem::create_directories(parentDir);
    }
}

bool ZarrDataset::readChunk(
    const std::vector<std::size_t>& chunkId, void* buffer) const
{
    auto cp = chunkPath(chunkId);
    if (!std::filesystem::exists(cp)) {
        return false;
    }

    // Read compressed data
    std::ifstream f(cp, std::ios::binary);
    f.seekg(0, std::ios::end);
    auto fileSize = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> compressed(static_cast<std::size_t>(fileSize));
    f.read(reinterpret_cast<char*>(compressed.data()), fileSize);

    if (codec_) {
        // Decompress
        std::size_t chunkBytes = defaultChunkSize() * dtypeSize(dtype_);
        codec_->decompress(
            compressed.data(), compressed.size(), buffer, chunkBytes);
    } else {
        // Raw data
        std::memcpy(buffer, compressed.data(), compressed.size());
    }

    return true;
}

void ZarrDataset::writeChunk(
    const std::vector<std::size_t>& chunkId,
    const void* buffer,
    std::size_t size)
{
    ensureChunkDir(chunkId);

    std::vector<std::uint8_t> output;
    if (codec_) {
        output = codec_->compress(buffer, size, dtypeSize(dtype_));
    } else {
        output.resize(size);
        std::memcpy(output.data(), buffer, size);
    }

    auto cp = chunkPath(chunkId);
    std::ofstream f(cp, std::ios::binary);
    f.write(reinterpret_cast<const char*>(output.data()),
            static_cast<std::streamsize>(output.size()));
}

// --- Template implementations for readSubarray/writeSubarray ---

template <typename T>
void ZarrDataset::readSubarray(
    Tensor3D<T>& out,
    const std::vector<std::size_t>& offset,
    const std::vector<std::size_t>& reqShape) const
{
    // Validate dimensions
    if (offset.size() != 3 || reqShape.size() != 3) {
        throw std::runtime_error(
            "ZarrDataset::readSubarray: requires 3D offset and shape");
    }

    // Resize output if needed
    if (out.shape()[0] != reqShape[0] || out.shape()[1] != reqShape[1] ||
        out.shape()[2] != reqShape[2]) {
        out.resize(reqShape[0], reqShape[1], reqShape[2]);
    }

    // Initialize with fill value
    T fillVal{0};
    if (fillValue_.is_number()) {
        fillVal = fillValue_.get<T>();
    }
    out.fill(fillVal);

    // Calculate chunk range
    std::vector<std::size_t> startChunk(3), endChunk(3);
    for (int i = 0; i < 3; ++i) {
        startChunk[i] = offset[i] / chunks_[i];
        endChunk[i] = (offset[i] + reqShape[i] + chunks_[i] - 1) / chunks_[i];
    }

    // Temporary buffer for reading chunks
    std::vector<T> chunkBuffer(defaultChunkSize());

    // Iterate over chunks
    for (std::size_t cz = startChunk[0]; cz < endChunk[0]; ++cz) {
        for (std::size_t cy = startChunk[1]; cy < endChunk[1]; ++cy) {
            for (std::size_t cx = startChunk[2]; cx < endChunk[2]; ++cx) {
                std::vector<std::size_t> chunkId = {cz, cy, cx};

                // Try to read chunk
                if (!readChunk(chunkId, chunkBuffer.data())) {
                    continue;  // Chunk doesn't exist, use fill value
                }

                // Calculate overlap between chunk and requested region
                std::size_t chunkStartZ = cz * chunks_[0];
                std::size_t chunkStartY = cy * chunks_[1];
                std::size_t chunkStartX = cx * chunks_[2];

                // Source range within chunk
                std::size_t srcZ0 =
                    (offset[0] > chunkStartZ) ? offset[0] - chunkStartZ : 0;
                std::size_t srcY0 =
                    (offset[1] > chunkStartY) ? offset[1] - chunkStartY : 0;
                std::size_t srcX0 =
                    (offset[2] > chunkStartX) ? offset[2] - chunkStartX : 0;

                std::size_t srcZ1 = std::min(
                    chunks_[0], offset[0] + reqShape[0] - chunkStartZ);
                std::size_t srcY1 = std::min(
                    chunks_[1], offset[1] + reqShape[1] - chunkStartY);
                std::size_t srcX1 = std::min(
                    chunks_[2], offset[2] + reqShape[2] - chunkStartX);

                // Destination range within output
                std::size_t dstZ0 = chunkStartZ + srcZ0 - offset[0];
                std::size_t dstY0 = chunkStartY + srcY0 - offset[1];
                std::size_t dstX0 = chunkStartX + srcX0 - offset[2];

                // Copy data (zarr C-order: z,y,x with x varying fastest)
                // Our tensor is column-major with dim0 varying fastest
                // We need to handle the mapping properly
                for (std::size_t sz = srcZ0; sz < srcZ1; ++sz) {
                    for (std::size_t sy = srcY0; sy < srcY1; ++sy) {
                        for (std::size_t sx = srcX0; sx < srcX1; ++sx) {
                            // Zarr C-order index: z*chunks[1]*chunks[2] + y*chunks[2] + x
                            std::size_t srcIdx =
                                sz * chunks_[1] * chunks_[2] +
                                sy * chunks_[2] + sx;

                            std::size_t dz = dstZ0 + (sz - srcZ0);
                            std::size_t dy = dstY0 + (sy - srcY0);
                            std::size_t dx = dstX0 + (sx - srcX0);

                            out(dz, dy, dx) = chunkBuffer[srcIdx];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void ZarrDataset::writeSubarray(
    const Tensor3D<T>& data,
    const std::vector<std::size_t>& offset)
{
    const auto& dataShape = data.shape();

    // Validate dimensions
    if (offset.size() != 3) {
        throw std::runtime_error(
            "ZarrDataset::writeSubarray: requires 3D offset");
    }

    // Calculate chunk range
    std::vector<std::size_t> startChunk(3), endChunk(3);
    for (int i = 0; i < 3; ++i) {
        startChunk[i] = offset[i] / chunks_[i];
        endChunk[i] = (offset[i] + dataShape[i] + chunks_[i] - 1) / chunks_[i];
    }

    // Temporary buffer for writing chunks
    std::vector<T> chunkBuffer(defaultChunkSize());

    // Iterate over chunks
    for (std::size_t cz = startChunk[0]; cz < endChunk[0]; ++cz) {
        for (std::size_t cy = startChunk[1]; cy < endChunk[1]; ++cy) {
            for (std::size_t cx = startChunk[2]; cx < endChunk[2]; ++cx) {
                std::vector<std::size_t> chunkId = {cz, cy, cx};

                // Try to read existing chunk data (for partial overwrites)
                bool existingChunk = readChunk(chunkId, chunkBuffer.data());
                if (!existingChunk) {
                    // Initialize with fill value
                    T fillVal{0};
                    if (fillValue_.is_number()) {
                        fillVal = fillValue_.get<T>();
                    }
                    std::fill(chunkBuffer.begin(), chunkBuffer.end(), fillVal);
                }

                // Calculate overlap
                std::size_t chunkStartZ = cz * chunks_[0];
                std::size_t chunkStartY = cy * chunks_[1];
                std::size_t chunkStartX = cx * chunks_[2];

                // Destination range within chunk
                std::size_t dstZ0 =
                    (offset[0] > chunkStartZ) ? offset[0] - chunkStartZ : 0;
                std::size_t dstY0 =
                    (offset[1] > chunkStartY) ? offset[1] - chunkStartY : 0;
                std::size_t dstX0 =
                    (offset[2] > chunkStartX) ? offset[2] - chunkStartX : 0;

                std::size_t dstZ1 = std::min(
                    chunks_[0], offset[0] + dataShape[0] - chunkStartZ);
                std::size_t dstY1 = std::min(
                    chunks_[1], offset[1] + dataShape[1] - chunkStartY);
                std::size_t dstX1 = std::min(
                    chunks_[2], offset[2] + dataShape[2] - chunkStartX);

                // Source range within input data
                std::size_t srcZ0 = chunkStartZ + dstZ0 - offset[0];
                std::size_t srcY0 = chunkStartY + dstY0 - offset[1];
                std::size_t srcX0 = chunkStartX + dstX0 - offset[2];

                // Copy data
                for (std::size_t dz = dstZ0; dz < dstZ1; ++dz) {
                    for (std::size_t dy = dstY0; dy < dstY1; ++dy) {
                        for (std::size_t dx = dstX0; dx < dstX1; ++dx) {
                            std::size_t sz = srcZ0 + (dz - dstZ0);
                            std::size_t sy = srcY0 + (dy - dstY0);
                            std::size_t sx = srcX0 + (dx - dstX0);

                            // Zarr C-order index
                            std::size_t dstIdx =
                                dz * chunks_[1] * chunks_[2] +
                                dy * chunks_[2] + dx;

                            chunkBuffer[dstIdx] = data(sz, sy, sx);
                        }
                    }
                }

                // Write chunk
                writeChunk(
                    chunkId,
                    chunkBuffer.data(),
                    defaultChunkSize() * sizeof(T));
            }
        }
    }
}

// Explicit instantiations
template void ZarrDataset::readSubarray(
    Tensor3D<std::uint8_t>&,
    const std::vector<std::size_t>&,
    const std::vector<std::size_t>&) const;
template void ZarrDataset::readSubarray(
    Tensor3D<std::uint16_t>&,
    const std::vector<std::size_t>&,
    const std::vector<std::size_t>&) const;
template void ZarrDataset::readSubarray(
    Tensor3D<float>&,
    const std::vector<std::size_t>&,
    const std::vector<std::size_t>&) const;

template void ZarrDataset::writeSubarray(
    const Tensor3D<std::uint8_t>&,
    const std::vector<std::size_t>&);
template void ZarrDataset::writeSubarray(
    const Tensor3D<std::uint16_t>&,
    const std::vector<std::size_t>&);
template void ZarrDataset::writeSubarray(
    const Tensor3D<float>&,
    const std::vector<std::size_t>&);

}  // namespace volcart::zarr
