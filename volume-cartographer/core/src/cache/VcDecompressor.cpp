#include "vc/core/cache/VcDecompressor.hpp"
#include <utils/zarr.hpp>

#include <cstring>
#include <stdexcept>

namespace vc::cache {

DecompressFn makeVcDecompressor(const std::vector<vc::Zarr*>& datasets)
{
    // Capture the dataset pointers (caller guarantees lifetime)
    return [datasets](const std::vector<uint8_t>& compressed,
                      const ChunkKey& key) -> ChunkDataPtr {
        if (key.level < 0 ||
            key.level >= static_cast<int>(datasets.size()) ||
            !datasets[key.level]) {
            return nullptr;
        }

        vc::Zarr& ds = *datasets[key.level];
        const auto& chunkShape = ds.chunks();
        const size_t chunkSize = ds.chunkSize();
        const auto dtype = ds.dtype();

        auto result = std::make_shared<ChunkData>();
        result->shape = {
            static_cast<int>(chunkShape[0]),
            static_cast<int>(chunkShape[1]),
            static_cast<int>(chunkShape[2])};

        // The compressed buffer needs to be char-based for VcDataset API
        auto compressedCopy =
            std::vector<char>(compressed.begin(), compressed.end());

        if (dtype == vc::DType::UInt8) {
            result->elementSize = 1;
            result->bytes.resize(chunkSize);
            ds.decompress(compressedCopy, result->bytes.data(), chunkSize);
        } else if (dtype == vc::DType::UInt16) {
            // Decompress as uint16 then convert to uint8 (divide by 257)
            result->elementSize = 1;
            std::vector<uint16_t> tmp(chunkSize);
            ds.decompress(compressedCopy, tmp.data(), chunkSize);

            result->bytes.resize(chunkSize);
            for (size_t i = 0; i < chunkSize; i++) {
                result->bytes[i] = static_cast<uint8_t>(tmp[i] / 257);
            }
        } else {
            return nullptr;  // unsupported dtype
        }

        return result;
    };
}

DecompressFn makeVcDecompressor(vc::Zarr* ds)
{
    return makeVcDecompressor(std::vector<vc::Zarr*>{ds});
}

}  // namespace vc::cache
