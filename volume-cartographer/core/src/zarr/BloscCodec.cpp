#include "vc/core/zarr/BloscCodec.hpp"

#include <blosc2.h>

#include <mutex>
#include <stdexcept>

namespace volcart::zarr
{

namespace
{
std::once_flag bloscInitFlag;
}

void BloscCodec::initBlosc()
{
    std::call_once(bloscInitFlag, []() { blosc2_init(); });
}

BloscCodec::BloscCodec()
{
    initBlosc();
}

BloscCodec::BloscCodec(const nlohmann::json& config) : BloscCodec()
{
    if (config.contains("cname") && config["cname"].is_string()) {
        cname_ = config["cname"].get<std::string>();
    }
    if (config.contains("clevel") && config["clevel"].is_number_integer()) {
        clevel_ = config["clevel"].get<int>();
    }
    if (config.contains("shuffle")) {
        if (config["shuffle"].is_number_integer()) {
            shuffle_ = config["shuffle"].get<int>();
        } else if (config["shuffle"].is_string()) {
            auto s = config["shuffle"].get<std::string>();
            if (s == "noshuffle")
                shuffle_ = 0;
            else if (s == "shuffle")
                shuffle_ = 1;
            else if (s == "bitshuffle")
                shuffle_ = 2;
        }
    }
    if (config.contains("blocksize") && config["blocksize"].is_number_integer()) {
        blocksize_ = config["blocksize"].get<std::size_t>();
    }
}

std::vector<std::uint8_t> BloscCodec::compress(
    const void* src, std::size_t size, std::size_t typesize) const
{
    std::size_t maxDstSize = size + BLOSC2_MAX_OVERHEAD;
    std::vector<std::uint8_t> dst(maxDstSize);

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.typesize = static_cast<int32_t>(typesize);
    cparams.compcode = blosc2_compname_to_compcode(cname_.c_str());
    cparams.clevel = clevel_;
    cparams.filters[BLOSC2_MAX_FILTERS - 1] =
        (shuffle_ == 2) ? BLOSC_BITSHUFFLE : (shuffle_ == 1) ? BLOSC_SHUFFLE
                                                             : BLOSC_NOSHUFFLE;
    cparams.blocksize = static_cast<int32_t>(blocksize_);
    cparams.nthreads = 1;

    blosc2_context* cctx = blosc2_create_cctx(cparams);
    if (!cctx) {
        throw std::runtime_error("BloscCodec::compress: failed to create context");
    }

    int compressedSize = blosc2_compress_ctx(
        cctx,
        src,
        static_cast<int32_t>(size),
        dst.data(),
        static_cast<int32_t>(maxDstSize));

    blosc2_free_ctx(cctx);

    if (compressedSize < 0) {
        throw std::runtime_error(
            "BloscCodec::compress failed with error code " +
            std::to_string(compressedSize));
    }

    dst.resize(static_cast<std::size_t>(compressedSize));
    return dst;
}

std::size_t BloscCodec::decompress(
    const void* src,
    std::size_t srcSize,
    void* dst,
    std::size_t dstCapacity) const
{
    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    dparams.nthreads = 1;

    blosc2_context* dctx = blosc2_create_dctx(dparams);
    if (!dctx) {
        throw std::runtime_error(
            "BloscCodec::decompress: failed to create context");
    }

    int result = blosc2_decompress_ctx(
        dctx,
        src,
        static_cast<int32_t>(srcSize),
        dst,
        static_cast<int32_t>(dstCapacity));

    blosc2_free_ctx(dctx);

    if (result < 0) {
        throw std::runtime_error(
            "BloscCodec::decompress failed with error code " +
            std::to_string(result));
    }

    return static_cast<std::size_t>(result);
}

bool BloscCodec::isBlosc(const void* data, std::size_t size)
{
    if (size < BLOSC_MIN_HEADER_LENGTH) {
        return false;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    return bytes[0] >= 0x01 && bytes[0] <= 0x02;
}

std::size_t BloscCodec::decompressedSize(const void* data)
{
    int32_t nbytes, cbytes, blocksize;
    blosc2_cbuffer_sizes(data, &nbytes, &cbytes, &blocksize);
    return static_cast<std::size_t>(nbytes);
}

nlohmann::json BloscCodec::toJson() const
{
    return {
        {"id", "blosc"},
        {"cname", cname_},
        {"clevel", clevel_},
        {"shuffle", shuffle_},
        {"blocksize", blocksize_}};
}

nlohmann::json BloscCodec::toJsonV3(std::size_t typesize) const
{
    std::string shuffleStr = (shuffle_ == 2)   ? "bitshuffle"
                             : (shuffle_ == 1) ? "shuffle"
                                               : "noshuffle";
    return {
        {"name", "blosc"},
        {"configuration",
         {{"cname", cname_},
          {"clevel", clevel_},
          {"shuffle", shuffleStr},
          {"typesize", typesize},
          {"blocksize", blocksize_}}}};
}

}  // namespace volcart::zarr
