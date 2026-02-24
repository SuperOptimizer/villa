#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace utils {

class Tensor;

enum class Compression : std::uint16_t {
    None     = 1,
    LZW      = 5,
    Deflate  = 8,
    ZStd     = 50000,
    LZ4      = 34892,
    PackBits = 32773,
};

enum class Predictor : std::uint16_t {
    None       = 1,
    Horizontal = 2,
};

enum class Orientation : std::uint16_t {
    TopLeft     = 1,
    TopRight    = 2,
    BottomRight = 3,
    BottomLeft  = 4,
    LeftTop     = 5,
    RightTop    = 6,
    RightBottom = 7,
    LeftBottom  = 8,
};

struct TiffMetadata {
    std::string image_description;  // tag 270
    std::string software;           // tag 305
    std::string date_time;          // tag 306
};

enum class SampleFormat : std::uint8_t {
    UInt  = 1,
    Int   = 2,
    Float = 3,
};

enum class PhotoInterp : std::uint8_t {
    MinIsBlack = 0,
    MinIsWhite = 1,
    RGB        = 2,
};

struct TiffWriteOptions {
    Compression compression = Compression::None;
    Predictor predictor = Predictor::None;
    std::uint32_t rows_per_strip = 0; // 0 = single strip (backward compat)
    std::uint32_t tile_width  = 0;    // both must be set for tile mode
    std::uint32_t tile_height = 0;    // must be multiples of 16
    Orientation orientation = Orientation::TopLeft;
    TiffMetadata metadata;
};

class TiffImage {
public:
    ~TiffImage();
    TiffImage(TiffImage&& other) noexcept;
    auto operator=(TiffImage&& other) noexcept -> TiffImage&;
    TiffImage(const TiffImage& other);
    auto operator=(const TiffImage& other) -> TiffImage&;

    // Read first page from file or memory
    [[nodiscard]] static auto read(const std::string& path)
        -> std::expected<TiffImage, std::string>;
    [[nodiscard]] static auto decode(std::span<const std::byte> data)
        -> std::expected<TiffImage, std::string>;

    // Write to file or encode to memory
    [[nodiscard]] auto write(const std::string& path,
                             Compression comp = Compression::None) const
        -> std::expected<void, std::string>;
    [[nodiscard]] auto write(const std::string& path,
                             const TiffWriteOptions& opts) const
        -> std::expected<void, std::string>;
    [[nodiscard]] auto encode(Compression comp = Compression::None) const
        -> std::expected<std::vector<std::byte>, std::string>;
    [[nodiscard]] auto encode(const TiffWriteOptions& opts) const
        -> std::expected<std::vector<std::byte>, std::string>;

    // Create from raw pixel data (row-major, interleaved channels)
    [[nodiscard]] static auto from_data(const void* pixels,
                                        std::uint32_t width, std::uint32_t height,
                                        std::uint16_t channels,
                                        SampleFormat fmt = SampleFormat::UInt,
                                        std::uint16_t bits_per_sample = 8)
        -> std::expected<TiffImage, std::string>;

    // Create from tensor (shape [H,W] or [H,W,C])
    [[nodiscard]] static auto from_tensor(const Tensor& tensor)
        -> std::expected<TiffImage, std::string>;

    // Properties
    [[nodiscard]] auto width() const noexcept -> std::uint32_t;
    [[nodiscard]] auto height() const noexcept -> std::uint32_t;
    [[nodiscard]] auto channels() const noexcept -> std::uint16_t;
    [[nodiscard]] auto bits_per_sample() const noexcept -> std::uint16_t;
    [[nodiscard]] auto sample_format() const noexcept -> SampleFormat;
    [[nodiscard]] auto photo_interp() const noexcept -> PhotoInterp;
    [[nodiscard]] auto pixels() const noexcept -> const void*;
    [[nodiscard]] auto nbytes() const noexcept -> std::size_t;
    [[nodiscard]] auto orientation() const noexcept -> Orientation;
    [[nodiscard]] auto metadata() const noexcept -> const TiffMetadata&;
    [[nodiscard]] auto extra_samples() const noexcept -> const std::vector<std::uint16_t>&;

    // Convert to tensor (shape [H,W] for 1-channel, [H,W,C] for multi-channel)
    [[nodiscard]] auto to_tensor() const -> Tensor;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit TiffImage(std::unique_ptr<Impl> impl);

    friend auto tiff_encode_pages(std::span<const TiffImage> pages,
                                  Compression comp)
        -> std::expected<std::vector<std::byte>, std::string>;
    friend auto tiff_encode_pages(std::span<const TiffImage> pages,
                                  const TiffWriteOptions& opts)
        -> std::expected<std::vector<std::byte>, std::string>;
    friend auto tiff_decode_pages(std::span<const std::byte> data)
        -> std::expected<std::vector<TiffImage>, std::string>;
    friend auto tiff_mmap(const std::string& path)
        -> std::expected<Tensor, std::string>;
};

// Multi-page TIFF I/O
[[nodiscard]] auto tiff_read_pages(const std::string& path)
    -> std::expected<std::vector<TiffImage>, std::string>;
[[nodiscard]] auto tiff_decode_pages(std::span<const std::byte> data)
    -> std::expected<std::vector<TiffImage>, std::string>;
[[nodiscard]] auto tiff_write_pages(const std::string& path,
                                    std::span<const TiffImage> pages,
                                    Compression comp = Compression::None)
    -> std::expected<void, std::string>;
[[nodiscard]] auto tiff_write_pages(const std::string& path,
                                    std::span<const TiffImage> pages,
                                    const TiffWriteOptions& opts)
    -> std::expected<void, std::string>;
[[nodiscard]] auto tiff_encode_pages(std::span<const TiffImage> pages,
                                     Compression comp = Compression::None)
    -> std::expected<std::vector<std::byte>, std::string>;
[[nodiscard]] auto tiff_encode_pages(std::span<const TiffImage> pages,
                                     const TiffWriteOptions& opts)
    -> std::expected<std::vector<std::byte>, std::string>;

// Memory-mapped TIFF (zero-copy, uncompressed single-page only)
[[nodiscard]] auto tiff_mmap(const std::string& path)
    -> std::expected<Tensor, std::string>;

} // namespace utils
