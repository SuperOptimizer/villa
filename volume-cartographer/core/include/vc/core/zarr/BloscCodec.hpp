#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace volcart::zarr
{

/**
 * @brief Blosc compression/decompression codec for zarr chunks.
 */
class BloscCodec
{
public:
    BloscCodec();
    explicit BloscCodec(const nlohmann::json& config);

    std::vector<std::uint8_t>
    compress(const void* src, std::size_t size, std::size_t typesize) const;

    std::size_t decompress(
        const void* src,
        std::size_t srcSize,
        void* dst,
        std::size_t dstCapacity) const;

    static bool isBlosc(const void* data, std::size_t size);
    static std::size_t decompressedSize(const void* data);

    const std::string& compressorName() const noexcept { return cname_; }
    int compressionLevel() const noexcept { return clevel_; }
    int shuffleMode() const noexcept { return shuffle_; }

    nlohmann::json toJson() const;
    nlohmann::json toJsonV3(std::size_t typesize) const;

private:
    std::string cname_ = "zstd";
    int clevel_ = 1;
    int shuffle_ = 1;
    std::size_t blocksize_ = 0;

    static void initBlosc();
};

}  // namespace volcart::zarr
