#include "vc/core/cache/Z5Decompressor.hpp"

#include "z5/dataset.hxx"
#include "z5/types/types.hxx"

#include <cstring>
#include <stdexcept>

namespace vc::cache {

DecompressFn makeZ5Decompressor(const std::vector<z5::Dataset*>& datasets)
{
    // Capture the dataset pointers (caller guarantees lifetime)
    return [datasets](const std::vector<uint8_t>& compressed,
                      const ChunkKey& key) -> ChunkDataPtr {
        if (key.level < 0 ||
            key.level >= static_cast<int>(datasets.size()) ||
            !datasets[key.level]) {
            return nullptr;
        }

        z5::Dataset& ds = *datasets[key.level];
        const auto& chunkShape = ds.defaultChunkShape();
        const size_t chunkSize = ds.defaultChunkSize();
        const auto dtype = ds.getDtype();

        auto result = std::make_shared<ChunkData>();
        result->shape = {
            static_cast<int>(chunkShape[0]),
            static_cast<int>(chunkShape[1]),
            static_cast<int>(chunkShape[2])};

        // Decompress into the appropriate dtype
        // The compressed buffer needs to be non-const for z5's API
        auto compressedCopy =
            std::vector<char>(compressed.begin(), compressed.end());

        if (dtype == z5::types::Datatype::uint8) {
            result->elementSize = 1;
            result->bytes.resize(chunkSize);
            ds.decompress(compressedCopy, result->bytes.data(), chunkSize);
        } else if (dtype == z5::types::Datatype::uint16) {
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

DecompressFn makeZ5Decompressor(z5::Dataset* ds)
{
    return makeZ5Decompressor(std::vector<z5::Dataset*>{ds});
}

}  // namespace vc::cache
