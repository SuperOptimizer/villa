#include "utils/tiff.hpp"
#include "utils/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include <lz4.h>
#include <zlib.h>
#include <zstd.h>

namespace utils {

// ---------------------------------------------------------------------------
// Byte-order helpers
// ---------------------------------------------------------------------------

enum class ByteOrder : std::uint8_t { Little, Big };

static auto read_u16(const std::byte* p, ByteOrder bo) -> std::uint16_t {
    if (bo == ByteOrder::Little) {
        return static_cast<std::uint16_t>(static_cast<unsigned>(p[0]) |
               (static_cast<unsigned>(p[1]) << 8));
    }
    return static_cast<std::uint16_t>((static_cast<unsigned>(p[0]) << 8) |
           static_cast<unsigned>(p[1]));
}

static auto read_u32(const std::byte* p, ByteOrder bo) -> std::uint32_t {
    if (bo == ByteOrder::Little) {
        return static_cast<std::uint32_t>(p[0]) |
               (static_cast<std::uint32_t>(p[1]) << 8) |
               (static_cast<std::uint32_t>(p[2]) << 16) |
               (static_cast<std::uint32_t>(p[3]) << 24);
    }
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

static void write_u32_le(std::byte* p, std::uint32_t v) {
    p[0] = static_cast<std::byte>(v & 0xFF);
    p[1] = static_cast<std::byte>((v >> 8) & 0xFF);
    p[2] = static_cast<std::byte>((v >> 16) & 0xFF);
    p[3] = static_cast<std::byte>((v >> 24) & 0xFF);
}

static void push_u16_le(std::vector<std::byte>& buf, std::uint16_t v) {
    buf.push_back(static_cast<std::byte>(v & 0xFF));
    buf.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
}

static void push_u32_le(std::vector<std::byte>& buf, std::uint32_t v) {
    buf.push_back(static_cast<std::byte>(v & 0xFF));
    buf.push_back(static_cast<std::byte>((v >> 8) & 0xFF));
    buf.push_back(static_cast<std::byte>((v >> 16) & 0xFF));
    buf.push_back(static_cast<std::byte>((v >> 24) & 0xFF));
}

// Byte-swap a 16-bit sample in place
static void swap_bytes_16(void* data, std::size_t count) {
    auto* p = static_cast<std::uint8_t*>(data);
    for (std::size_t i = 0; i < count; ++i) {
        std::swap(p[i * 2], p[i * 2 + 1]);
    }
}

// Byte-swap a 32-bit sample in place
static void swap_bytes_32(void* data, std::size_t count) {
    auto* p = static_cast<std::uint8_t*>(data);
    for (std::size_t i = 0; i < count; ++i) {
        std::swap(p[i * 4], p[i * 4 + 3]);
        std::swap(p[i * 4 + 1], p[i * 4 + 2]);
    }
}

// ---------------------------------------------------------------------------
// TIFF constants
// ---------------------------------------------------------------------------

static constexpr std::uint16_t kTagImageWidth         = 256;
static constexpr std::uint16_t kTagImageLength         = 257;
static constexpr std::uint16_t kTagBitsPerSample       = 258;
static constexpr std::uint16_t kTagCompression         = 259;
static constexpr std::uint16_t kTagPhotometric         = 262;
static constexpr std::uint16_t kTagStripOffsets         = 273;
static constexpr std::uint16_t kTagSamplesPerPixel     = 277;
static constexpr std::uint16_t kTagRowsPerStrip        = 278;
static constexpr std::uint16_t kTagStripByteCounts     = 279;
static constexpr std::uint16_t kTagXResolution         = 282;
static constexpr std::uint16_t kTagYResolution         = 283;
static constexpr std::uint16_t kTagResolutionUnit      = 296;
static constexpr std::uint16_t kTagTileWidth            = 322;
static constexpr std::uint16_t kTagTileLength           = 323;
static constexpr std::uint16_t kTagImageDescription     = 270;
static constexpr std::uint16_t kTagOrientation          = 274;
static constexpr std::uint16_t kTagPlanarConfig         = 284;
static constexpr std::uint16_t kTagSoftware             = 305;
static constexpr std::uint16_t kTagDateTime             = 306;
static constexpr std::uint16_t kTagPredictor            = 317;
static constexpr std::uint16_t kTagTileOffsets          = 324;
static constexpr std::uint16_t kTagTileByteCounts       = 325;
static constexpr std::uint16_t kTagExtraSamples         = 338;
static constexpr std::uint16_t kTagSampleFormat         = 339;

// TIFF field types
static constexpr std::uint16_t kTypeByte     = 1;
static constexpr std::uint16_t kTypeAscii    = 2;
static constexpr std::uint16_t kTypeShort    = 3;
static constexpr std::uint16_t kTypeLong     = 4;
static constexpr std::uint16_t kTypeRational = 5;

// ---------------------------------------------------------------------------
// PackBits compression
// ---------------------------------------------------------------------------

static auto packbits_decompress(const std::byte* src, std::size_t src_len,
                                std::size_t expected_len) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(expected_len);
    std::size_t pos = 0;

    while (pos < src_len && out.size() < expected_len) {
        auto n = static_cast<std::int8_t>(src[pos++]);
        if (n >= 0) {
            // Literal run: copy next n+1 bytes
            auto count = static_cast<std::size_t>(n) + 1;
            if (pos + count > src_len) break;
            out.insert(out.end(), src + pos, src + pos + count);
            pos += count;
        } else if (n != -128) {
            // Repeat run: repeat next byte 1-n times
            auto count = static_cast<std::size_t>(1 - n);
            if (pos >= src_len) break;
            auto val = src[pos++];
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(val);
            }
        }
        // n == -128: no-op
    }
    return out;
}

static auto packbits_compress(const std::byte* src, std::size_t len) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(len + len / 128 + 1);
    std::size_t pos = 0;

    while (pos < len) {
        // Check for a run of identical bytes
        std::size_t run = 1;
        while (pos + run < len && run < 128 && src[pos + run] == src[pos]) {
            ++run;
        }
        if (run >= 3) {
            out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(1 - static_cast<int>(run))));
            out.push_back(src[pos]);
            pos += run;
        } else {
            // Literal run
            std::size_t lit_start = pos;
            std::size_t lit_len = 0;
            while (pos < len && lit_len < 128) {
                // Check if a run of 3+ starts here
                std::size_t r = 1;
                while (pos + r < len && r < 128 && src[pos + r] == src[pos]) {
                    ++r;
                }
                if (r >= 3 && lit_len > 0) break;
                if (r >= 3) {
                    // Include one byte, then break for the run
                    ++pos;
                    ++lit_len;
                    break;
                }
                ++pos;
                ++lit_len;
            }
            out.push_back(static_cast<std::byte>(static_cast<std::uint8_t>(lit_len - 1)));
            out.insert(out.end(), src + lit_start, src + lit_start + lit_len);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// LZW compression (TIFF variant: MSB-first bit packing)
// ---------------------------------------------------------------------------

static constexpr int kLzwClearCode = 256;
static constexpr int kLzwEoiCode   = 257;
static constexpr int kLzwFirstCode = 258;
static constexpr int kLzwMaxBits   = 12;
static constexpr int kLzwMaxTable  = 1 << kLzwMaxBits; // 4096

// Bit reader: reads codes MSB-first from a byte stream
struct LzwBitReader {
    const std::byte* data;
    std::size_t len;
    std::size_t byte_pos = 0;
    int bit_pos = 0; // bits consumed in current byte (0..7, MSB first)

    auto read_bits(int nbits) -> int {
        int result = 0;
        for (int i = 0; i < nbits; ++i) {
            if (byte_pos >= len) return -1;
            int bit = (static_cast<int>(data[byte_pos]) >> (7 - bit_pos)) & 1;
            result = (result << 1) | bit;
            ++bit_pos;
            if (bit_pos == 8) {
                bit_pos = 0;
                ++byte_pos;
            }
        }
        return result;
    }
};

static auto lzw_decompress(const std::byte* src, std::size_t src_len,
                            std::size_t expected_len) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(expected_len);

    // Table: each entry is (prefix_code, suffix_byte, length)
    // For single-byte entries (0-255): prefix=-1, suffix=byte, length=1
    struct Entry {
        int prefix;
        std::uint8_t suffix;
        std::uint16_t length;
    };

    std::vector<Entry> table;
    table.reserve(kLzwMaxTable);

    auto reset_table = [&]() {
        table.clear();
        for (int i = 0; i < 256; ++i) {
            table.push_back({-1, static_cast<std::uint8_t>(i), 1});
        }
        table.push_back({-1, 0, 0}); // 256 = clear
        table.push_back({-1, 0, 0}); // 257 = eoi
    };

    // Decode a table entry to bytes
    auto decode_entry = [&](int code) {
        auto& e = table[static_cast<std::size_t>(code)];
        auto start = out.size();
        out.resize(out.size() + e.length);
        auto idx = static_cast<int>(code);
        for (auto pos = static_cast<std::ptrdiff_t>(start + e.length - 1);
             pos >= static_cast<std::ptrdiff_t>(start); --pos) {
            out[static_cast<std::size_t>(pos)] =
                static_cast<std::byte>(table[static_cast<std::size_t>(idx)].suffix);
            idx = table[static_cast<std::size_t>(idx)].prefix;
        }
    };

    LzwBitReader reader{src, src_len};
    int code_bits = 9;

    reset_table();

    int code = reader.read_bits(code_bits);
    if (code != kLzwClearCode) return out; // must start with clear

    code_bits = 9;
    reset_table();

    int prev_code = -1;
    while (out.size() < expected_len) {
        code = reader.read_bits(code_bits);
        if (code < 0 || code == kLzwEoiCode) break;

        if (code == kLzwClearCode) {
            reset_table();
            code_bits = 9;
            prev_code = -1;
            continue;
        }

        auto table_size = static_cast<int>(table.size());
        if (code < table_size) {
            decode_entry(code);
            if (prev_code >= 0 && table_size < kLzwMaxTable) {
                // first byte of current entry
                auto first_byte = static_cast<std::uint8_t>(
                    out[out.size() - table[static_cast<std::size_t>(code)].length]);
                auto prev_len = table[static_cast<std::size_t>(prev_code)].length;
                table.push_back({prev_code, first_byte,
                                 static_cast<std::uint16_t>(prev_len + 1)});
            }
        } else if (code == table_size && prev_code >= 0) {
            // Special case: code not in table yet
            auto first_byte_pos = out.size() -
                                  table[static_cast<std::size_t>(prev_code)].length;
            auto first_byte = static_cast<std::uint8_t>(out[first_byte_pos]);
            auto prev_len = table[static_cast<std::size_t>(prev_code)].length;
            table.push_back({prev_code, first_byte,
                             static_cast<std::uint16_t>(prev_len + 1)});
            decode_entry(code);
        } else {
            break; // invalid code
        }

        // Increase code size if needed (early change)
        table_size = static_cast<int>(table.size());
        if (table_size > (1 << code_bits) - 1 && code_bits < kLzwMaxBits) {
            ++code_bits;
        }

        prev_code = code;
    }
    return out;
}

// Bit writer: writes codes MSB-first
struct LzwBitWriter {
    std::vector<std::byte>& buf;
    int pending = 0;
    int pending_bits = 0;

    void write_bits(int code, int nbits) {
        pending = (pending << nbits) | code;
        pending_bits += nbits;
        while (pending_bits >= 8) {
            pending_bits -= 8;
            buf.push_back(static_cast<std::byte>((pending >> pending_bits) & 0xFF));
        }
    }

    void flush() {
        if (pending_bits > 0) {
            buf.push_back(static_cast<std::byte>((pending << (8 - pending_bits)) & 0xFF));
            pending = 0;
            pending_bits = 0;
        }
    }
};

static auto lzw_compress(const std::byte* src, std::size_t len) -> std::vector<std::byte> {
    std::vector<std::byte> out;
    out.reserve(len + len / 4);
    LzwBitWriter writer{out};

    // Hash table for string matching: maps (prefix, byte) -> code
    // Use a simple open-addressing hash table
    static constexpr int kHashSize = 8192;
    struct HashEntry {
        int prefix;
        std::uint8_t suffix;
        int code;
    };

    std::vector<HashEntry> hash_table(kHashSize, {-1, 0, -1});

    int next_code = kLzwFirstCode;
    int code_bits = 9;

    auto reset = [&]() {
        std::fill(hash_table.begin(), hash_table.end(), HashEntry{-1, 0, -1});
        next_code = kLzwFirstCode;
        code_bits = 9;
    };

    auto hash_func = [](int prefix, std::uint8_t suffix) -> std::size_t {
        auto key = static_cast<std::size_t>(
            (static_cast<unsigned>(prefix) << 8) ^ suffix);
        return (key * 2654435761u) % kHashSize;
    };

    auto find_or_insert = [&](int prefix, std::uint8_t suffix) -> int {
        auto h = hash_func(prefix, suffix);
        for (std::size_t i = 0; i < kHashSize; ++i) {
            auto idx = (h + i) % kHashSize;
            if (hash_table[idx].code == -1) {
                // Not found - insert
                if (next_code < kLzwMaxTable) {
                    hash_table[idx] = {prefix, suffix, next_code};
                    ++next_code;
                }
                return -1;
            }
            if (hash_table[idx].prefix == prefix &&
                hash_table[idx].suffix == suffix) {
                return hash_table[idx].code;
            }
        }
        return -1;
    };

    writer.write_bits(kLzwClearCode, code_bits);

    if (len == 0) {
        writer.write_bits(kLzwEoiCode, code_bits);
        writer.flush();
        return out;
    }

    int current = static_cast<std::uint8_t>(src[0]);
    for (std::size_t i = 1; i < len; ++i) {
        auto byte = static_cast<std::uint8_t>(src[i]);
        int code = find_or_insert(current, byte);
        if (code >= 0) {
            current = code;
        } else {
            writer.write_bits(current, code_bits);
            // Check if we need to increase code bits
            if (next_code > (1 << code_bits) && code_bits < kLzwMaxBits) {
                ++code_bits;
            }
            if (next_code >= kLzwMaxTable) {
                writer.write_bits(kLzwClearCode, code_bits);
                reset();
            }
            current = byte;
        }
    }

    writer.write_bits(current, code_bits);
    writer.write_bits(kLzwEoiCode, code_bits);
    writer.flush();
    return out;
}

// ---------------------------------------------------------------------------
// IFD parsing helper
// ---------------------------------------------------------------------------

struct IfdEntry {
    std::uint16_t tag;
    std::uint16_t type;
    std::uint32_t count;
    std::uint32_t value_offset; // inline value or file offset
};

struct IfdInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t bits_per_sample = 8;
    std::uint16_t compression = 1;
    std::uint16_t photometric = 0;
    std::uint16_t samples_per_pixel = 1;
    std::uint32_t rows_per_strip = 0xFFFFFFFF;
    std::uint16_t sample_format = 1;
    std::uint16_t predictor = 1;
    std::uint16_t orientation = 1;
    std::uint16_t planar_config = 1;
    std::vector<std::uint16_t> extra_samples;
    std::vector<std::uint32_t> strip_offsets;
    std::vector<std::uint32_t> strip_byte_counts;
    // Tile fields
    std::uint32_t tile_width = 0;
    std::uint32_t tile_height = 0;
    std::vector<std::uint32_t> tile_offsets;
    std::vector<std::uint32_t> tile_byte_counts;
    // Metadata
    std::string image_description;
    std::string software;
    std::string date_time;

    [[nodiscard]] auto is_tiled() const -> bool {
        return tile_width > 0 && tile_height > 0;
    }
};

static auto type_size(std::uint16_t type) -> std::size_t {
    switch (type) {
        case 1: return 1; // BYTE
        case 2: return 1; // ASCII
        case 3: return 2; // SHORT
        case 4: return 4; // LONG
        case 5: return 8; // RATIONAL
        default: return 0;
    }
}

static auto read_tag_value_u32(const IfdEntry& entry, const std::byte* /*data*/,
                                std::size_t /*data_len*/, ByteOrder bo) -> std::uint32_t {
    if (entry.type == kTypeShort && entry.count == 1) {
        // Inline SHORT: left-justified in the 4-byte value/offset field.
        // After read_u32, LE puts it in low 16 bits, BE in high 16 bits.
        if (bo == ByteOrder::Little) return entry.value_offset & 0xFFFF;
        return (entry.value_offset >> 16) & 0xFFFF;
    }
    if (entry.type == kTypeLong && entry.count == 1) {
        return entry.value_offset;
    }
    if (entry.type == kTypeByte && entry.count == 1) {
        return entry.value_offset & 0xFF;
    }
    return entry.value_offset;
}

static auto read_tag_array_u32(const IfdEntry& entry, const std::byte* data,
                                std::size_t data_len, ByteOrder bo) -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> result;
    result.reserve(entry.count);

    auto elem_size = type_size(entry.type);
    bool is_inline = (entry.count * elem_size <= 4);
    const std::byte* src = nullptr;

    std::byte inline_buf[4];
    if (is_inline) {
        // Value is stored inline in the value_offset field
        // We need to reconstruct the raw bytes in file byte order
        if (bo == ByteOrder::Little) {
            write_u32_le(inline_buf, entry.value_offset);
        } else {
            // big-endian: store as big-endian
            inline_buf[0] = static_cast<std::byte>((entry.value_offset >> 24) & 0xFF);
            inline_buf[1] = static_cast<std::byte>((entry.value_offset >> 16) & 0xFF);
            inline_buf[2] = static_cast<std::byte>((entry.value_offset >> 8) & 0xFF);
            inline_buf[3] = static_cast<std::byte>(entry.value_offset & 0xFF);
        }
        src = inline_buf;
    } else {
        if (entry.value_offset + entry.count * elem_size > data_len) return result;
        src = data + entry.value_offset;
    }

    for (std::uint32_t i = 0; i < entry.count; ++i) {
        if (entry.type == kTypeShort) {
            result.push_back(read_u16(src + i * 2, bo));
        } else if (entry.type == kTypeLong) {
            result.push_back(read_u32(src + i * 4, bo));
        } else if (entry.type == kTypeByte) {
            result.push_back(static_cast<std::uint32_t>(src[i]));
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// String tag reader
// ---------------------------------------------------------------------------

static auto read_tag_string(const IfdEntry& entry, const std::byte* data,
                             std::size_t data_len, ByteOrder bo) -> std::string {
    if (entry.type != kTypeAscii || entry.count == 0) return {};

    const std::byte* src = nullptr;
    std::byte inline_buf[4];

    if (entry.count <= 4) {
        // Inline value
        if (bo == ByteOrder::Little) {
            write_u32_le(inline_buf, entry.value_offset);
        } else {
            inline_buf[0] = static_cast<std::byte>((entry.value_offset >> 24) & 0xFF);
            inline_buf[1] = static_cast<std::byte>((entry.value_offset >> 16) & 0xFF);
            inline_buf[2] = static_cast<std::byte>((entry.value_offset >> 8) & 0xFF);
            inline_buf[3] = static_cast<std::byte>(entry.value_offset & 0xFF);
        }
        src = inline_buf;
    } else {
        if (entry.value_offset + entry.count > data_len) return {};
        src = data + entry.value_offset;
    }

    // Count excludes or includes null terminator — strip trailing nulls
    auto len = static_cast<std::size_t>(entry.count);
    while (len > 0 && src[len - 1] == std::byte{0}) --len;
    return std::string(reinterpret_cast<const char*>(src), len);
}

static auto parse_ifd(const std::byte* data, std::size_t data_len,
                       std::uint32_t ifd_offset, ByteOrder bo)
    -> std::expected<std::pair<IfdInfo, std::uint32_t>, std::string> {

    if (ifd_offset + 2 > data_len) {
        return std::unexpected("IFD offset out of bounds");
    }

    auto num_entries = read_u16(data + ifd_offset, bo);
    auto entries_end = ifd_offset + 2 + static_cast<std::uint32_t>(num_entries) * 12 + 4;
    if (entries_end > data_len) {
        return std::unexpected("IFD entries extend past end of data");
    }

    IfdInfo info;
    for (std::uint16_t i = 0; i < num_entries; ++i) {
        auto pos = ifd_offset + 2 + static_cast<std::uint32_t>(i) * 12;
        IfdEntry entry{};
        entry.tag = read_u16(data + pos, bo);
        entry.type = read_u16(data + pos + 2, bo);
        entry.count = read_u32(data + pos + 4, bo);
        entry.value_offset = read_u32(data + pos + 8, bo);

        switch (entry.tag) {
            case kTagImageWidth:
                info.width = read_tag_value_u32(entry, data, data_len, bo);
                break;
            case kTagImageLength:
                info.height = read_tag_value_u32(entry, data, data_len, bo);
                break;
            case kTagBitsPerSample: {
                // For multi-channel images, BitsPerSample is an array.
                // Read the first element (we require all channels same bps).
                auto bps_arr = read_tag_array_u32(entry, data, data_len, bo);
                if (!bps_arr.empty()) {
                    info.bits_per_sample = static_cast<std::uint16_t>(bps_arr[0]);
                }
                break;
            }
            case kTagCompression:
                info.compression = static_cast<std::uint16_t>(
                    read_tag_value_u32(entry, data, data_len, bo));
                break;
            case kTagPhotometric:
                info.photometric = static_cast<std::uint16_t>(
                    read_tag_value_u32(entry, data, data_len, bo));
                break;
            case kTagSamplesPerPixel:
                info.samples_per_pixel = static_cast<std::uint16_t>(
                    read_tag_value_u32(entry, data, data_len, bo));
                break;
            case kTagRowsPerStrip:
                info.rows_per_strip = read_tag_value_u32(entry, data, data_len, bo);
                break;
            case kTagStripOffsets:
                info.strip_offsets = read_tag_array_u32(entry, data, data_len, bo);
                break;
            case kTagStripByteCounts:
                info.strip_byte_counts = read_tag_array_u32(entry, data, data_len, bo);
                break;
            case kTagTileWidth:
                info.tile_width = read_tag_value_u32(entry, data, data_len, bo);
                break;
            case kTagTileLength:
                info.tile_height = read_tag_value_u32(entry, data, data_len, bo);
                break;
            case kTagTileOffsets:
                info.tile_offsets = read_tag_array_u32(entry, data, data_len, bo);
                break;
            case kTagTileByteCounts:
                info.tile_byte_counts = read_tag_array_u32(entry, data, data_len, bo);
                break;
            case kTagImageDescription:
                info.image_description = read_tag_string(entry, data, data_len, bo);
                break;
            case kTagOrientation:
                info.orientation = static_cast<std::uint16_t>(
                    read_tag_value_u32(entry, data, data_len, bo));
                break;
            case kTagPlanarConfig:
                info.planar_config = static_cast<std::uint16_t>(
                    read_tag_value_u32(entry, data, data_len, bo));
                break;
            case kTagSoftware:
                info.software = read_tag_string(entry, data, data_len, bo);
                break;
            case kTagDateTime:
                info.date_time = read_tag_string(entry, data, data_len, bo);
                break;
            case kTagPredictor:
                info.predictor = static_cast<std::uint16_t>(
                    read_tag_value_u32(entry, data, data_len, bo));
                break;
            case kTagExtraSamples: {
                auto es_arr = read_tag_array_u32(entry, data, data_len, bo);
                info.extra_samples.reserve(es_arr.size());
                for (auto v : es_arr) {
                    info.extra_samples.push_back(static_cast<std::uint16_t>(v));
                }
                break;
            }
            case kTagSampleFormat: {
                auto sf_arr = read_tag_array_u32(entry, data, data_len, bo);
                if (!sf_arr.empty()) {
                    info.sample_format = static_cast<std::uint16_t>(sf_arr[0]);
                }
                break;
            }
            default:
                break;
        }
    }

    auto next_ifd_offset_pos = ifd_offset + 2 + static_cast<std::uint32_t>(num_entries) * 12;
    auto next_ifd = read_u32(data + next_ifd_offset_pos, bo);
    return std::pair{info, next_ifd};
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct TiffImage::Impl {
    std::vector<std::byte> pixel_data;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint16_t channels = 1;
    std::uint16_t bits_per_sample = 8;
    SampleFormat sample_format = SampleFormat::UInt;
    PhotoInterp photo_interp = PhotoInterp::MinIsBlack;
    Orientation orientation = Orientation::TopLeft;
    TiffMetadata metadata;
    std::vector<std::uint16_t> extra_samples;

    [[nodiscard]] auto bytes_per_sample() const -> std::size_t {
        return bits_per_sample / 8;
    }

    [[nodiscard]] auto row_bytes() const -> std::size_t {
        return static_cast<std::size_t>(width) * channels * bytes_per_sample();
    }

    [[nodiscard]] auto total_bytes() const -> std::size_t {
        return static_cast<std::size_t>(height) * row_bytes();
    }
};

// Intermediate decoded image data (avoids private access in static helper)
struct DecodedImageData {
    std::vector<std::byte> pixel_data;
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint16_t channels{1};
    std::uint16_t bits_per_sample{8};
    SampleFormat sample_format{SampleFormat::UInt};
    PhotoInterp photo_interp{PhotoInterp::MinIsBlack};
    Orientation orientation{Orientation::TopLeft};
    TiffMetadata metadata;
    std::vector<std::uint16_t> extra_samples;
};

// ---------------------------------------------------------------------------
// Decode a single IFD into raw image data
// ---------------------------------------------------------------------------

// Decompress a chunk of data (strip or tile)
static auto decompress_chunk(const std::byte* src, std::size_t src_len,
                              std::size_t expected_len, std::uint16_t compression)
    -> std::expected<std::vector<std::byte>, std::string> {
    std::vector<std::byte> decompressed;
    switch (compression) {
        case 1: // None
            if (src_len < expected_len) {
                return std::unexpected("uncompressed chunk too small");
            }
            decompressed.assign(src, src + expected_len);
            break;
        case 5: // LZW
            decompressed = lzw_decompress(src, src_len, expected_len);
            if (decompressed.size() < expected_len) {
                return std::unexpected("LZW decompression produced insufficient data");
            }
            break;
        case 8: { // Deflate
            decompressed.resize(expected_len);
            z_stream strm{};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
            if (inflateInit2(&strm, 15 + 32) != Z_OK) {
#pragma GCC diagnostic pop
                return std::unexpected("Deflate: inflateInit2 failed");
            }
            strm.next_in = reinterpret_cast<Bytef*>(
                const_cast<std::byte*>(src));
            strm.avail_in = static_cast<uInt>(src_len);
            strm.next_out = reinterpret_cast<Bytef*>(decompressed.data());
            strm.avail_out = static_cast<uInt>(expected_len);
            auto ret = inflate(&strm, Z_FINISH);
            inflateEnd(&strm);
            if (ret != Z_STREAM_END) {
                return std::unexpected("Deflate decompression failed");
            }
            break;
        }
        case 50000: { // ZStd
            decompressed.resize(expected_len);
            auto result = ZSTD_decompress(decompressed.data(), expected_len,
                                           src, src_len);
            if (ZSTD_isError(result)) {
                return std::unexpected(std::string("ZStd decompression failed: ") +
                                       ZSTD_getErrorName(result));
            }
            break;
        }
        case 34892: { // LZ4
            decompressed.resize(expected_len);
            auto result = LZ4_decompress_safe(
                reinterpret_cast<const char*>(src),
                reinterpret_cast<char*>(decompressed.data()),
                static_cast<int>(src_len),
                static_cast<int>(expected_len));
            if (result < 0) {
                return std::unexpected("LZ4 decompression failed");
            }
            break;
        }
        case 32773: // PackBits
            decompressed = packbits_decompress(src, src_len, expected_len);
            if (decompressed.size() < expected_len) {
                return std::unexpected("PackBits decompression produced insufficient data");
            }
            break;
        default:
            return std::unexpected("unsupported compression: " +
                                   std::to_string(compression));
    }
    return decompressed;
}

// Compress a chunk of data (strip or tile)
static auto compress_chunk(const std::byte* src, std::size_t len,
                            Compression comp)
    -> std::expected<std::vector<std::byte>, std::string> {
    switch (static_cast<std::uint16_t>(comp)) {
        case 1: // None
            return std::vector<std::byte>(src, src + len);
        case 5: // LZW
            return lzw_compress(src, len);
        case 8: { // Deflate
            auto bound = deflateBound(nullptr, static_cast<uLong>(len));
            std::vector<std::byte> out(bound);
            z_stream strm{};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
            if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED,
                             15, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
#pragma GCC diagnostic pop
                return std::unexpected("Deflate: deflateInit2 failed");
            }
            strm.next_in = reinterpret_cast<Bytef*>(const_cast<std::byte*>(src));
            strm.avail_in = static_cast<uInt>(len);
            strm.next_out = reinterpret_cast<Bytef*>(out.data());
            strm.avail_out = static_cast<uInt>(out.size());
            auto ret = deflate(&strm, Z_FINISH);
            deflateEnd(&strm);
            if (ret != Z_STREAM_END) {
                return std::unexpected("Deflate compression failed");
            }
            out.resize(strm.total_out);
            return out;
        }
        case 50000: { // ZStd
            auto bound = ZSTD_compressBound(len);
            std::vector<std::byte> out(bound);
            auto result = ZSTD_compress(out.data(), bound, src, len, 3);
            if (ZSTD_isError(result)) {
                return std::unexpected(std::string("ZStd compression failed: ") +
                                       ZSTD_getErrorName(result));
            }
            out.resize(result);
            return out;
        }
        case 34892: { // LZ4
            auto bound = LZ4_compressBound(static_cast<int>(len));
            std::vector<std::byte> out(static_cast<std::size_t>(bound));
            auto result = LZ4_compress_default(
                reinterpret_cast<const char*>(src),
                reinterpret_cast<char*>(out.data()),
                static_cast<int>(len), bound);
            if (result <= 0) {
                return std::unexpected("LZ4 compression failed");
            }
            out.resize(static_cast<std::size_t>(result));
            return out;
        }
        case 32773: // PackBits
            return packbits_compress(src, len);
        default:
            return std::unexpected("unsupported compression for writing");
    }
}

// ---------------------------------------------------------------------------
// Horizontal differencing predictor
// ---------------------------------------------------------------------------

// Apply predictor (encode path): pixel[i] = pixel[i] - pixel[i-1], per row
static void apply_horizontal_predictor(std::byte* data, std::uint32_t width,
                                        std::uint32_t height, std::uint16_t channels,
                                        std::uint16_t bytes_per_sample) {
    auto pixel_bytes = static_cast<std::size_t>(channels) * bytes_per_sample;
    auto row_bytes = static_cast<std::size_t>(width) * pixel_bytes;
    for (std::uint32_t y = 0; y < height; ++y) {
        auto* row = data + y * row_bytes;
        // Process right-to-left so we don't clobber values we still need
        for (std::size_t x = row_bytes - 1; x >= pixel_bytes; --x) {
            row[x] = static_cast<std::byte>(
                static_cast<std::uint8_t>(row[x]) -
                static_cast<std::uint8_t>(row[x - pixel_bytes]));
        }
    }
}

// Unapply predictor (decode path): pixel[i] = pixel[i] + pixel[i-1], per row
static void unapply_horizontal_predictor(std::byte* data, std::uint32_t width,
                                          std::uint32_t height, std::uint16_t channels,
                                          std::uint16_t bytes_per_sample) {
    auto pixel_bytes = static_cast<std::size_t>(channels) * bytes_per_sample;
    auto row_bytes = static_cast<std::size_t>(width) * pixel_bytes;
    for (std::uint32_t y = 0; y < height; ++y) {
        auto* row = data + y * row_bytes;
        for (std::size_t x = pixel_bytes; x < row_bytes; ++x) {
            row[x] = static_cast<std::byte>(
                static_cast<std::uint8_t>(row[x]) +
                static_cast<std::uint8_t>(row[x - pixel_bytes]));
        }
    }
}

static auto decode_ifd(const std::byte* data, std::size_t data_len,
                        const IfdInfo& info, ByteOrder bo)
    -> std::expected<DecodedImageData, std::string> {

    if (info.width == 0 || info.height == 0) {
        return std::unexpected("invalid image dimensions");
    }

    auto bps = info.bits_per_sample;
    if (bps != 8 && bps != 16 && bps != 32) {
        return std::unexpected("unsupported bits per sample: " + std::to_string(bps));
    }

    auto spp = info.samples_per_pixel;
    if (spp == 0 || spp > 4) {
        return std::unexpected("unsupported samples per pixel: " + std::to_string(spp));
    }

    auto bytes_per_sample = static_cast<std::size_t>(bps) / 8;
    auto pixel_bytes = bytes_per_sample * spp;
    auto row_bytes = static_cast<std::size_t>(info.width) * pixel_bytes;
    auto total_bytes = static_cast<std::size_t>(info.height) * row_bytes;

    std::vector<std::byte> pixels(total_bytes);

    if (info.is_tiled()) {
        // ------ Tiled decode ------
        auto tw = info.tile_width;
        auto th = info.tile_height;
        auto tiles_across = (info.width + tw - 1) / tw;
        auto tiles_down = (info.height + th - 1) / th;
        auto expected_tiles = tiles_across * tiles_down;

        if (info.tile_offsets.size() != expected_tiles ||
            info.tile_byte_counts.size() != expected_tiles) {
            return std::unexpected("tile offset/count mismatch: expected " +
                                   std::to_string(expected_tiles) + " tiles");
        }

        auto tile_row_bytes = static_cast<std::size_t>(tw) * pixel_bytes;
        auto full_tile_bytes = static_cast<std::size_t>(tw) * th * pixel_bytes;

        for (std::uint32_t ty = 0; ty < tiles_down; ++ty) {
            for (std::uint32_t tx = 0; tx < tiles_across; ++tx) {
                auto tile_idx = ty * tiles_across + tx;
                auto tile_offset = info.tile_offsets[tile_idx];
                auto tile_byte_count = info.tile_byte_counts[tile_idx];

                if (tile_offset + tile_byte_count > data_len) {
                    return std::unexpected("tile data extends past end of file");
                }

                auto chunk = decompress_chunk(data + tile_offset, tile_byte_count,
                                               full_tile_bytes, info.compression);
                if (!chunk) return std::unexpected(chunk.error());

                // Unapply predictor on tile data
                if (info.predictor == 2) {
                    unapply_horizontal_predictor(chunk->data(), tw, th,
                                                  spp, static_cast<std::uint16_t>(bytes_per_sample));
                }

                // Copy tile data into output, handling edge tiles
                auto actual_w = std::min(tw, info.width - tx * tw);
                auto actual_h = std::min(th, info.height - ty * th);
                auto copy_row_bytes = static_cast<std::size_t>(actual_w) * pixel_bytes;

                for (std::uint32_t row = 0; row < actual_h; ++row) {
                    auto dst_y = ty * th + row;
                    auto dst_offset = dst_y * row_bytes +
                                      static_cast<std::size_t>(tx * tw) * pixel_bytes;
                    auto src_offset = static_cast<std::size_t>(row) * tile_row_bytes;
                    std::memcpy(pixels.data() + dst_offset,
                                chunk->data() + src_offset, copy_row_bytes);
                }
            }
        }
    } else {
        // ------ Strip decode ------
        auto rows_per_strip = info.rows_per_strip;
        if (rows_per_strip > info.height) rows_per_strip = info.height;

        auto num_strips = (info.height + rows_per_strip - 1) / rows_per_strip;
        if (info.strip_offsets.size() != num_strips ||
            info.strip_byte_counts.size() != num_strips) {
            return std::unexpected("strip offset/count mismatch: expected " +
                                   std::to_string(num_strips) + " strips, got " +
                                   std::to_string(info.strip_offsets.size()) + " offsets and " +
                                   std::to_string(info.strip_byte_counts.size()) + " counts");
        }

        std::size_t out_pos = 0;
        for (std::uint32_t s = 0; s < num_strips; ++s) {
            auto strip_offset = info.strip_offsets[s];
            auto strip_bytes = info.strip_byte_counts[s];

            if (strip_offset + strip_bytes > data_len) {
                return std::unexpected("strip data extends past end of file");
            }

            auto strip_rows = std::min(rows_per_strip, info.height - s * rows_per_strip);
            auto expected_bytes = static_cast<std::size_t>(strip_rows) * row_bytes;

            auto chunk = decompress_chunk(data + strip_offset, strip_bytes,
                                           expected_bytes, info.compression);
            if (!chunk) return std::unexpected(chunk.error());

            // Unapply predictor on strip data
            if (info.predictor == 2) {
                unapply_horizontal_predictor(chunk->data(), info.width, strip_rows,
                                              spp, static_cast<std::uint16_t>(bytes_per_sample));
            }

            auto copy_len = std::min(expected_bytes, total_bytes - out_pos);
            std::memcpy(pixels.data() + out_pos, chunk->data(), copy_len);
            out_pos += copy_len;
        }
    }

    // Byte-swap multi-byte samples if file is big-endian and we're on little-endian
    if (bo == ByteOrder::Big && bps > 8) {
        auto num_samples = total_bytes / bytes_per_sample;
        if (bps == 16) {
            swap_bytes_16(pixels.data(), num_samples);
        } else if (bps == 32) {
            swap_bytes_32(pixels.data(), num_samples);
        }
    }

    // Convert planar (RRRGGGBBB) to chunky (RGBRGBRGB) if needed
    if (info.planar_config == 2 && spp > 1) {
        auto num_pixels = static_cast<std::size_t>(info.width) * info.height;
        std::vector<std::byte> chunky(total_bytes);
        auto plane_bytes = num_pixels * bytes_per_sample;
        for (std::size_t px = 0; px < num_pixels; ++px) {
            for (std::uint16_t ch = 0; ch < spp; ++ch) {
                auto src_off = static_cast<std::size_t>(ch) * plane_bytes +
                               px * bytes_per_sample;
                auto dst_off = px * pixel_bytes + static_cast<std::size_t>(ch) * bytes_per_sample;
                std::memcpy(chunky.data() + dst_off, pixels.data() + src_off,
                            bytes_per_sample);
            }
        }
        pixels = std::move(chunky);
    }

    DecodedImageData result;
    result.pixel_data = std::move(pixels);
    result.width = info.width;
    result.height = info.height;
    result.channels = spp;
    result.bits_per_sample = bps;

    if (info.sample_format == 2) {
        result.sample_format = SampleFormat::Int;
    } else if (info.sample_format == 3) {
        result.sample_format = SampleFormat::Float;
    } else {
        result.sample_format = SampleFormat::UInt;
    }

    if (info.photometric == 0) {
        result.photo_interp = PhotoInterp::MinIsBlack;
    } else if (info.photometric == 1) {
        result.photo_interp = PhotoInterp::MinIsWhite;
    } else {
        result.photo_interp = PhotoInterp::RGB;
    }

    auto ori = info.orientation;
    if (ori >= 1 && ori <= 8) {
        result.orientation = static_cast<Orientation>(ori);
    }
    result.extra_samples = info.extra_samples;
    result.metadata.image_description = info.image_description;
    result.metadata.software = info.software;
    result.metadata.date_time = info.date_time;

    return result;
}

// ---------------------------------------------------------------------------
// TiffImage special members
// ---------------------------------------------------------------------------

TiffImage::TiffImage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TiffImage::~TiffImage() = default;
TiffImage::TiffImage(TiffImage&& other) noexcept = default;
auto TiffImage::operator=(TiffImage&& other) noexcept -> TiffImage& = default;

TiffImage::TiffImage(const TiffImage& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}

auto TiffImage::operator=(const TiffImage& other) -> TiffImage& {
    if (this != &other) {
        impl_ = std::make_unique<Impl>(*other.impl_);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

auto TiffImage::width() const noexcept -> std::uint32_t { return impl_->width; }
auto TiffImage::height() const noexcept -> std::uint32_t { return impl_->height; }
auto TiffImage::channels() const noexcept -> std::uint16_t { return impl_->channels; }
auto TiffImage::bits_per_sample() const noexcept -> std::uint16_t { return impl_->bits_per_sample; }
auto TiffImage::sample_format() const noexcept -> SampleFormat { return impl_->sample_format; }
auto TiffImage::photo_interp() const noexcept -> PhotoInterp { return impl_->photo_interp; }
auto TiffImage::pixels() const noexcept -> const void* { return impl_->pixel_data.data(); }
auto TiffImage::nbytes() const noexcept -> std::size_t { return impl_->pixel_data.size(); }
auto TiffImage::orientation() const noexcept -> Orientation { return impl_->orientation; }
auto TiffImage::metadata() const noexcept -> const TiffMetadata& { return impl_->metadata; }
auto TiffImage::extra_samples() const noexcept -> const std::vector<std::uint16_t>& { return impl_->extra_samples; }

// ---------------------------------------------------------------------------
// Read / Decode
// ---------------------------------------------------------------------------

auto TiffImage::decode(std::span<const std::byte> buf)
    -> std::expected<TiffImage, std::string> {

    if (buf.size() < 8) {
        return std::unexpected("data too small for TIFF header");
    }

    ByteOrder bo;
    if (buf[0] == std::byte{0x49} && buf[1] == std::byte{0x49}) {
        bo = ByteOrder::Little;
    } else if (buf[0] == std::byte{0x4D} && buf[1] == std::byte{0x4D}) {
        bo = ByteOrder::Big;
    } else {
        return std::unexpected("invalid TIFF byte order marker");
    }

    auto magic = read_u16(buf.data() + 2, bo);
    if (magic != 42) {
        return std::unexpected("invalid TIFF magic number");
    }

    auto first_ifd = read_u32(buf.data() + 4, bo);
    if (first_ifd == 0) {
        return std::unexpected("no IFD found");
    }

    auto result = parse_ifd(buf.data(), buf.size(), first_ifd, bo);
    if (!result) return std::unexpected(result.error());

    auto decoded = decode_ifd(buf.data(), buf.size(), result->first, bo);
    if (!decoded) return std::unexpected(decoded.error());

    auto impl = std::make_unique<Impl>();
    impl->pixel_data      = std::move(decoded->pixel_data);
    impl->width            = decoded->width;
    impl->height           = decoded->height;
    impl->channels         = decoded->channels;
    impl->bits_per_sample  = decoded->bits_per_sample;
    impl->sample_format    = decoded->sample_format;
    impl->photo_interp     = decoded->photo_interp;
    impl->orientation      = decoded->orientation;
    impl->metadata         = std::move(decoded->metadata);
    impl->extra_samples    = std::move(decoded->extra_samples);
    return TiffImage(std::move(impl));
}

auto TiffImage::read(const std::string& path) -> std::expected<TiffImage, std::string> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::unexpected("cannot open file: " + path);
    }
    auto size = file.tellg();
    file.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(buf.data()), size);
    if (!file) {
        return std::unexpected("failed to read file: " + path);
    }
    return decode(buf);
}

// ---------------------------------------------------------------------------
// Encode / Write
// ---------------------------------------------------------------------------

static void write_ifd_entry(std::vector<std::byte>& buf, std::uint16_t tag,
                             std::uint16_t type, std::uint32_t count,
                             std::uint32_t value) {
    push_u16_le(buf, tag);
    push_u16_le(buf, type);
    push_u32_le(buf, count);
    push_u32_le(buf, value);
}

// ---------------------------------------------------------------------------
// Shared encode_page helper — used by both TiffImage::encode and
// tiff_encode_pages.  Appends one page's data + IFD to `buf`.
// Returns {ifd_offset, next_ifd_field_position}.
// ---------------------------------------------------------------------------

struct EncodePageResult {
    std::uint32_t ifd_offset;
    std::size_t next_ifd_field_pos;
};

static auto encode_page(std::vector<std::byte>& buf,
                         const DecodedImageData& img,
                         const TiffWriteOptions& opts)
    -> std::expected<EncodePageResult, std::string> {

    bool tile_mode = opts.tile_width > 0 && opts.tile_height > 0;
    if (tile_mode && opts.rows_per_strip != 0) {
        return std::unexpected("cannot set both tile dimensions and rows_per_strip");
    }
    if (tile_mode) {
        if (opts.tile_width % 16 != 0 || opts.tile_height % 16 != 0) {
            return std::unexpected("tile dimensions must be multiples of 16");
        }
    }

    bool use_predictor = opts.predictor == Predictor::Horizontal;

    auto bps_bytes = static_cast<std::size_t>(img.bits_per_sample) / 8;
    auto pixel_bytes = static_cast<std::size_t>(img.channels) * bps_bytes;
    auto row_bytes = static_cast<std::size_t>(img.width) * pixel_bytes;

    // ----- Compress data into chunks (strips or tiles) -----
    std::vector<std::uint32_t> chunk_offsets;
    std::vector<std::uint32_t> chunk_byte_counts;

    if (tile_mode) {
        auto tw = opts.tile_width;
        auto th = opts.tile_height;
        auto tiles_across = (img.width + tw - 1) / tw;
        auto tiles_down = (img.height + th - 1) / th;
        auto tile_row_bytes = static_cast<std::size_t>(tw) * pixel_bytes;
        auto full_tile_bytes = static_cast<std::size_t>(tw) * th * pixel_bytes;

        for (std::uint32_t ty = 0; ty < tiles_down; ++ty) {
            for (std::uint32_t tx = 0; tx < tiles_across; ++tx) {
                std::vector<std::byte> tile_buf(full_tile_bytes, std::byte{0});

                auto actual_w = std::min(tw, img.width - tx * tw);
                auto actual_h = std::min(th, img.height - ty * th);

                for (std::uint32_t row = 0; row < actual_h; ++row) {
                    auto src_y = ty * th + row;
                    auto src_offset = src_y * row_bytes +
                                      static_cast<std::size_t>(tx * tw) * pixel_bytes;
                    auto dst_offset = static_cast<std::size_t>(row) * tile_row_bytes;
                    auto copy_bytes = static_cast<std::size_t>(actual_w) * pixel_bytes;
                    std::memcpy(tile_buf.data() + dst_offset,
                                img.pixel_data.data() + src_offset, copy_bytes);
                }

                if (use_predictor) {
                    apply_horizontal_predictor(tile_buf.data(), tw, th,
                                               img.channels,
                                               static_cast<std::uint16_t>(bps_bytes));
                }

                auto compressed = compress_chunk(tile_buf.data(), full_tile_bytes,
                                                  opts.compression);
                if (!compressed) return std::unexpected(compressed.error());

                chunk_offsets.push_back(static_cast<std::uint32_t>(buf.size()));
                buf.insert(buf.end(), compressed->begin(), compressed->end());
                chunk_byte_counts.push_back(
                    static_cast<std::uint32_t>(compressed->size()));
            }
        }
    } else {
        auto rps = opts.rows_per_strip;
        if (rps == 0) rps = img.height;
        auto num_strips = (img.height + rps - 1) / rps;

        for (std::uint32_t s = 0; s < num_strips; ++s) {
            auto strip_rows = std::min(rps, img.height - s * rps);
            auto strip_bytes = static_cast<std::size_t>(strip_rows) * row_bytes;
            auto strip_start = static_cast<std::size_t>(s) *
                               static_cast<std::size_t>(rps) * row_bytes;

            const std::byte* chunk_src = img.pixel_data.data() + strip_start;
            std::vector<std::byte> pred_buf;

            if (use_predictor) {
                pred_buf.assign(chunk_src, chunk_src + strip_bytes);
                apply_horizontal_predictor(pred_buf.data(), img.width, strip_rows,
                                            img.channels,
                                            static_cast<std::uint16_t>(bps_bytes));
                chunk_src = pred_buf.data();
            }

            auto compressed = compress_chunk(chunk_src, strip_bytes, opts.compression);
            if (!compressed) return std::unexpected(compressed.error());

            chunk_offsets.push_back(static_cast<std::uint32_t>(buf.size()));
            buf.insert(buf.end(), compressed->begin(), compressed->end());
            chunk_byte_counts.push_back(
                static_cast<std::uint32_t>(compressed->size()));
        }
    }

    auto num_chunks = static_cast<std::uint32_t>(chunk_offsets.size());

    while (buf.size() % 2 != 0) buf.push_back(std::byte{0});

    // ----- Extra data area (multi-value tags, strings, etc.) -----
    // Helper to write a string and return its offset
    auto write_string_data = [&](const std::string& s) -> std::uint32_t {
        auto off = static_cast<std::uint32_t>(buf.size());
        for (char c : s) buf.push_back(static_cast<std::byte>(c));
        buf.push_back(std::byte{0}); // null terminator
        return off;
    };

    // BitsPerSample: inline if count * 2 <= 4 (i.e. channels <= 2), else external
    std::uint32_t bps_value = 0;
    bool bps_inline = (img.channels * 2u <= 4);
    if (img.channels == 1) {
        bps_value = img.bits_per_sample;
    } else if (bps_inline) {
        // Pack channels shorts into a 4-byte LE value
        for (std::uint16_t c = 0; c < img.channels; ++c) {
            bps_value |= static_cast<std::uint32_t>(img.bits_per_sample) << (c * 16);
        }
    } else {
        bps_value = static_cast<std::uint32_t>(buf.size());
        for (std::uint16_t c = 0; c < img.channels; ++c) {
            push_u16_le(buf, img.bits_per_sample);
        }
    }

    // SampleFormat: same inline/external logic
    std::uint32_t sf_value = 0;
    bool sf_inline = (img.channels * 2u <= 4);
    if (img.channels == 1) {
        sf_value = static_cast<std::uint32_t>(img.sample_format);
    } else if (sf_inline) {
        auto sfv = static_cast<std::uint16_t>(img.sample_format);
        for (std::uint16_t c = 0; c < img.channels; ++c) {
            sf_value |= static_cast<std::uint32_t>(sfv) << (c * 16);
        }
    } else {
        while (buf.size() % 2 != 0) buf.push_back(std::byte{0});
        sf_value = static_cast<std::uint32_t>(buf.size());
        for (std::uint16_t c = 0; c < img.channels; ++c) {
            push_u16_le(buf, static_cast<std::uint16_t>(img.sample_format));
        }
    }

    while (buf.size() % 2 != 0) buf.push_back(std::byte{0});
    auto resolution_offset = static_cast<std::uint32_t>(buf.size());
    push_u32_le(buf, 72);
    push_u32_le(buf, 1);

    std::uint32_t chunk_offsets_offset = 0;
    std::uint32_t chunk_counts_offset = 0;
    if (num_chunks > 1) {
        while (buf.size() % 2 != 0) buf.push_back(std::byte{0});
        chunk_offsets_offset = static_cast<std::uint32_t>(buf.size());
        for (auto off : chunk_offsets) push_u32_le(buf, off);

        chunk_counts_offset = static_cast<std::uint32_t>(buf.size());
        for (auto cnt : chunk_byte_counts) push_u32_le(buf, cnt);
    }

    // ExtraSamples data offset (for >2 extra samples needing external storage)
    std::uint32_t extra_samples_offset = 0;
    std::vector<std::uint16_t> extra_samples_vals;
    // Auto-detect extra samples
    if (img.photo_interp == PhotoInterp::RGB && img.channels == 4) {
        extra_samples_vals = {2}; // unassociated alpha
    } else if (img.photo_interp == PhotoInterp::MinIsBlack && img.channels >= 2) {
        extra_samples_vals.resize(img.channels - 1, 0); // unspecified
    }
    if (extra_samples_vals.size() > 2) {
        while (buf.size() % 2 != 0) buf.push_back(std::byte{0});
        extra_samples_offset = static_cast<std::uint32_t>(buf.size());
        for (auto v : extra_samples_vals) push_u16_le(buf, v);
    }

    // Metadata string offsets
    std::uint32_t desc_offset = 0;
    std::uint32_t soft_offset = 0;
    std::uint32_t dt_offset = 0;
    const auto& meta = opts.metadata;
    if (!meta.image_description.empty() && meta.image_description.size() + 1 > 4) {
        desc_offset = write_string_data(meta.image_description);
    }
    if (!meta.software.empty() && meta.software.size() + 1 > 4) {
        soft_offset = write_string_data(meta.software);
    }
    if (!meta.date_time.empty() && meta.date_time.size() + 1 > 4) {
        dt_offset = write_string_data(meta.date_time);
    }

    // ----- Build IFD tag vector, then sort by tag number -----
    struct IfdTagEntry {
        std::uint16_t tag;
        std::uint16_t type;
        std::uint32_t count;
        std::uint32_t value;
    };

    std::vector<IfdTagEntry> tags;
    tags.reserve(20);

    tags.push_back({kTagImageWidth, kTypeShort, 1, img.width});
    tags.push_back({kTagImageLength, kTypeShort, 1, img.height});

    tags.push_back({kTagBitsPerSample, kTypeShort, img.channels, bps_value});

    tags.push_back({kTagCompression, kTypeShort, 1,
                    static_cast<std::uint32_t>(opts.compression)});
    tags.push_back({kTagPhotometric, kTypeShort, 1,
                    static_cast<std::uint32_t>(img.photo_interp)});

    // Metadata strings (tags 270, 305, 306)
    if (!meta.image_description.empty()) {
        auto count = static_cast<std::uint32_t>(meta.image_description.size() + 1);
        std::uint32_t val = desc_offset;
        if (count <= 4) {
            // Inline: pack bytes into 4-byte value (LE)
            val = 0;
            for (std::size_t i = 0; i < meta.image_description.size(); ++i) {
                val |= static_cast<std::uint32_t>(
                    static_cast<std::uint8_t>(meta.image_description[i])) << (i * 8);
            }
        }
        tags.push_back({kTagImageDescription, kTypeAscii, count, val});
    }

    if (!tile_mode) {
        if (num_chunks == 1) {
            tags.push_back({kTagStripOffsets, kTypeLong, 1, chunk_offsets[0]});
        } else {
            tags.push_back({kTagStripOffsets, kTypeLong, num_chunks,
                            chunk_offsets_offset});
        }
    }

    tags.push_back({kTagOrientation, kTypeShort, 1,
                    static_cast<std::uint32_t>(opts.orientation)});

    tags.push_back({kTagSamplesPerPixel, kTypeShort, 1, img.channels});

    if (!tile_mode) {
        auto rps = opts.rows_per_strip;
        if (rps == 0) rps = img.height;
        tags.push_back({kTagRowsPerStrip, kTypeShort, 1, rps});

        if (num_chunks == 1) {
            tags.push_back({kTagStripByteCounts, kTypeLong, 1,
                            chunk_byte_counts[0]});
        } else {
            tags.push_back({kTagStripByteCounts, kTypeLong, num_chunks,
                            chunk_counts_offset});
        }
    }

    tags.push_back({kTagXResolution, kTypeRational, 1, resolution_offset});
    tags.push_back({kTagYResolution, kTypeRational, 1, resolution_offset});
    tags.push_back({kTagResolutionUnit, kTypeShort, 1, 2});

    if (!meta.software.empty()) {
        auto count = static_cast<std::uint32_t>(meta.software.size() + 1);
        std::uint32_t val = soft_offset;
        if (count <= 4) {
            val = 0;
            for (std::size_t i = 0; i < meta.software.size(); ++i) {
                val |= static_cast<std::uint32_t>(
                    static_cast<std::uint8_t>(meta.software[i])) << (i * 8);
            }
        }
        tags.push_back({kTagSoftware, kTypeAscii, count, val});
    }

    if (!meta.date_time.empty()) {
        auto count = static_cast<std::uint32_t>(meta.date_time.size() + 1);
        std::uint32_t val = dt_offset;
        if (count <= 4) {
            val = 0;
            for (std::size_t i = 0; i < meta.date_time.size(); ++i) {
                val |= static_cast<std::uint32_t>(
                    static_cast<std::uint8_t>(meta.date_time[i])) << (i * 8);
            }
        }
        tags.push_back({kTagDateTime, kTypeAscii, count, val});
    }

    if (use_predictor) {
        tags.push_back({kTagPredictor, kTypeShort, 1, 2});
    }

    if (tile_mode) {
        tags.push_back({kTagTileWidth, kTypeShort, 1, opts.tile_width});
        tags.push_back({kTagTileLength, kTypeShort, 1, opts.tile_height});

        if (num_chunks == 1) {
            tags.push_back({kTagTileOffsets, kTypeLong, 1, chunk_offsets[0]});
        } else {
            tags.push_back({kTagTileOffsets, kTypeLong, num_chunks,
                            chunk_offsets_offset});
        }
        if (num_chunks == 1) {
            tags.push_back({kTagTileByteCounts, kTypeLong, 1,
                            chunk_byte_counts[0]});
        } else {
            tags.push_back({kTagTileByteCounts, kTypeLong, num_chunks,
                            chunk_counts_offset});
        }
    }

    // ExtraSamples tag
    if (!extra_samples_vals.empty()) {
        auto es_count = static_cast<std::uint32_t>(extra_samples_vals.size());
        std::uint32_t es_val = 0;
        if (es_count <= 2) {
            // Inline: pack shorts into 4-byte value (LE)
            for (std::size_t i = 0; i < extra_samples_vals.size(); ++i) {
                es_val |= static_cast<std::uint32_t>(extra_samples_vals[i]) << (i * 16);
            }
        } else {
            es_val = extra_samples_offset;
        }
        tags.push_back({kTagExtraSamples, kTypeShort, es_count, es_val});
    }

    tags.push_back({kTagSampleFormat, kTypeShort, img.channels, sf_value});

    // Sort tags by tag number (TIFF spec requires ascending order)
    std::sort(tags.begin(), tags.end(),
              [](const IfdTagEntry& a, const IfdTagEntry& b) {
                  return a.tag < b.tag;
              });

    // ----- Write IFD -----
    while (buf.size() % 2 != 0) buf.push_back(std::byte{0});
    auto ifd_pos = static_cast<std::uint32_t>(buf.size());

    push_u16_le(buf, static_cast<std::uint16_t>(tags.size()));
    for (const auto& t : tags) {
        write_ifd_entry(buf, t.tag, t.type, t.count, t.value);
    }

    auto next_ifd_pos = buf.size();
    push_u32_le(buf, 0);

    return EncodePageResult{ifd_pos, next_ifd_pos};
}

// ---------------------------------------------------------------------------
// TiffImage::encode / write (options-based, with backward-compat overloads)
// ---------------------------------------------------------------------------

auto TiffImage::encode(const TiffWriteOptions& opts) const
    -> std::expected<std::vector<std::byte>, std::string> {
    std::vector<std::byte> buf;
    buf.reserve(impl_->total_bytes() + 256);

    // TIFF header
    buf.push_back(std::byte{0x49});
    buf.push_back(std::byte{0x49});
    push_u16_le(buf, 42);
    push_u32_le(buf, 0); // placeholder for first IFD offset

    DecodedImageData data;
    data.pixel_data = impl_->pixel_data;
    data.width = impl_->width;
    data.height = impl_->height;
    data.channels = impl_->channels;
    data.bits_per_sample = impl_->bits_per_sample;
    data.sample_format = impl_->sample_format;
    data.photo_interp = impl_->photo_interp;

    auto result = encode_page(buf, data, opts);
    if (!result) return std::unexpected(result.error());

    write_u32_le(buf.data() + 4, result->ifd_offset);
    return buf;
}

auto TiffImage::encode(Compression comp) const
    -> std::expected<std::vector<std::byte>, std::string> {
    return encode(TiffWriteOptions{comp, Predictor::None, 0, 0, 0, Orientation::TopLeft, {}});
}

auto TiffImage::write(const std::string& path, const TiffWriteOptions& opts) const
    -> std::expected<void, std::string> {
    auto encoded = encode(opts);
    if (!encoded) return std::unexpected(encoded.error());

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected("cannot create file: " + path);
    }
    file.write(reinterpret_cast<const char*>(encoded->data()),
               static_cast<std::streamsize>(encoded->size()));
    if (!file) {
        return std::unexpected("failed to write file: " + path);
    }
    return {};
}

auto TiffImage::write(const std::string& path, Compression comp) const
    -> std::expected<void, std::string> {
    return write(path, TiffWriteOptions{comp, Predictor::None, 0, 0, 0, Orientation::TopLeft, {}});
}

// ---------------------------------------------------------------------------
// from_data
// ---------------------------------------------------------------------------

auto TiffImage::from_data(const void* pixels, std::uint32_t width, std::uint32_t height,
                           std::uint16_t channels, SampleFormat fmt,
                           std::uint16_t bits_per_sample)
    -> std::expected<TiffImage, std::string> {
    if (!pixels) return std::unexpected("pixels pointer is null");
    if (width == 0 || height == 0) return std::unexpected("dimensions must be non-zero");
    if (channels == 0 || channels > 4) {
        return std::unexpected("channels must be 1-4");
    }
    if (bits_per_sample != 8 && bits_per_sample != 16 && bits_per_sample != 32) {
        return std::unexpected("bits_per_sample must be 8, 16, or 32");
    }
    if (fmt == SampleFormat::Float && bits_per_sample != 32) {
        return std::unexpected("float format requires 32 bits per sample");
    }

    auto impl = std::make_unique<TiffImage::Impl>();
    impl->width = width;
    impl->height = height;
    impl->channels = channels;
    impl->bits_per_sample = bits_per_sample;
    impl->sample_format = fmt;

    if (channels == 1) {
        impl->photo_interp = PhotoInterp::MinIsBlack;
    } else {
        impl->photo_interp = PhotoInterp::RGB;
    }

    auto nbytes = impl->total_bytes();
    impl->pixel_data.resize(nbytes);
    std::memcpy(impl->pixel_data.data(), pixels, nbytes);

    return TiffImage(std::move(impl));
}

// ---------------------------------------------------------------------------
// Tensor interop
// ---------------------------------------------------------------------------

auto TiffImage::from_tensor(const Tensor& tensor)
    -> std::expected<TiffImage, std::string> {
    auto ndim = tensor.ndim();
    if (ndim != 2 && ndim != 3) {
        return std::unexpected("tensor must be 2D [H,W] or 3D [H,W,C]");
    }

    auto height = static_cast<std::uint32_t>(tensor.shape()[0]);
    auto width = static_cast<std::uint32_t>(tensor.shape()[1]);
    std::uint16_t channels = 1;
    if (ndim == 3) {
        channels = static_cast<std::uint16_t>(tensor.shape()[2]);
        if (channels == 0 || channels > 4) {
            return std::unexpected("channel count must be 1-4, got " +
                                   std::to_string(channels));
        }
    }

    if (!tensor.is_contiguous()) {
        auto contig = tensor.contiguous();
        return from_tensor(contig);
    }

    SampleFormat fmt{};
    std::uint16_t bps = 0;
    switch (tensor.dtype()) {
        case DType::UInt8:   fmt = SampleFormat::UInt;  bps = 8;  break;
        case DType::UInt16:  fmt = SampleFormat::UInt;  bps = 16; break;
        case DType::Int8:    fmt = SampleFormat::Int;   bps = 8;  break;
        case DType::Int16:   fmt = SampleFormat::Int;   bps = 16; break;
        case DType::Float32: fmt = SampleFormat::Float; bps = 32; break;
        default:
            return std::unexpected("unsupported dtype for TIFF: " +
                                   std::string(dtype_name(tensor.dtype())));
    }

    return from_data(tensor.data_ptr(), width, height, channels, fmt, bps);
}

auto TiffImage::to_tensor() const -> Tensor {
    DType dt{};
    auto bps = impl_->bits_per_sample;
    switch (impl_->sample_format) {
        case SampleFormat::UInt:
            if (bps == 8) dt = DType::UInt8;
            else if (bps == 16) dt = DType::UInt16;
            else dt = DType::UInt32;
            break;
        case SampleFormat::Int:
            if (bps == 8) dt = DType::Int8;
            else if (bps == 16) dt = DType::Int16;
            else dt = DType::Int32;
            break;
        case SampleFormat::Float:
            dt = DType::Float32;
            break;
    }

    std::vector<std::size_t> shape;
    if (impl_->channels == 1) {
        shape = {impl_->height, impl_->width};
    } else {
        shape = {impl_->height, impl_->width, impl_->channels};
    }

    auto tensor = Tensor::from_data(impl_->pixel_data.data(), shape, dt);
    return std::move(tensor.value());
}

// ---------------------------------------------------------------------------
// Multi-page TIFF I/O
// ---------------------------------------------------------------------------

auto tiff_decode_pages(std::span<const std::byte> buf)
    -> std::expected<std::vector<TiffImage>, std::string> {

    if (buf.size() < 8) {
        return std::unexpected("data too small for TIFF header");
    }

    ByteOrder bo;
    if (buf[0] == std::byte{0x49} && buf[1] == std::byte{0x49}) {
        bo = ByteOrder::Little;
    } else if (buf[0] == std::byte{0x4D} && buf[1] == std::byte{0x4D}) {
        bo = ByteOrder::Big;
    } else {
        return std::unexpected("invalid TIFF byte order marker");
    }

    auto magic = read_u16(buf.data() + 2, bo);
    if (magic != 42) {
        return std::unexpected("invalid TIFF magic number");
    }

    std::vector<TiffImage> pages;
    auto ifd_offset = read_u32(buf.data() + 4, bo);

    while (ifd_offset != 0) {
        auto result = parse_ifd(buf.data(), buf.size(), ifd_offset, bo);
        if (!result) return std::unexpected(result.error());

        auto decoded = decode_ifd(buf.data(), buf.size(), result->first, bo);
        if (!decoded) return std::unexpected(decoded.error());

        auto impl = std::make_unique<TiffImage::Impl>();
        impl->pixel_data      = std::move(decoded->pixel_data);
        impl->width            = decoded->width;
        impl->height           = decoded->height;
        impl->channels         = decoded->channels;
        impl->bits_per_sample  = decoded->bits_per_sample;
        impl->sample_format    = decoded->sample_format;
        impl->photo_interp     = decoded->photo_interp;
        impl->orientation      = decoded->orientation;
        impl->metadata         = std::move(decoded->metadata);
        impl->extra_samples    = std::move(decoded->extra_samples);
        pages.push_back(TiffImage(std::move(impl)));
        ifd_offset = result->second;
    }

    if (pages.empty()) {
        return std::unexpected("no pages found in TIFF");
    }
    return pages;
}

auto tiff_read_pages(const std::string& path)
    -> std::expected<std::vector<TiffImage>, std::string> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::unexpected("cannot open file: " + path);
    }
    auto size = file.tellg();
    file.seekg(0);
    std::vector<std::byte> buf(static_cast<std::size_t>(size));
    file.read(reinterpret_cast<char*>(buf.data()), size);
    if (!file) {
        return std::unexpected("failed to read file: " + path);
    }
    return tiff_decode_pages(buf);
}

auto tiff_encode_pages(std::span<const TiffImage> pages, const TiffWriteOptions& opts)
    -> std::expected<std::vector<std::byte>, std::string> {
    if (pages.empty()) {
        return std::unexpected("no pages to encode");
    }

    std::vector<std::byte> buf;
    buf.reserve(4096);

    // Header
    buf.push_back(std::byte{0x49});
    buf.push_back(std::byte{0x49});
    push_u16_le(buf, 42);
    push_u32_le(buf, 0); // placeholder for first IFD offset

    std::size_t prev_next_ifd_pos = 0;

    for (std::size_t page_idx = 0; page_idx < pages.size(); ++page_idx) {
        const auto* impl = pages[page_idx].impl_.get();
        DecodedImageData data;
        data.pixel_data = impl->pixel_data;
        data.width = impl->width;
        data.height = impl->height;
        data.channels = impl->channels;
        data.bits_per_sample = impl->bits_per_sample;
        data.sample_format = impl->sample_format;
        data.photo_interp = impl->photo_interp;

        auto result = encode_page(buf, data, opts);
        if (!result) return std::unexpected(result.error());

        if (page_idx == 0) {
            write_u32_le(buf.data() + 4, result->ifd_offset);
        } else {
            write_u32_le(buf.data() + prev_next_ifd_pos, result->ifd_offset);
        }
        prev_next_ifd_pos = result->next_ifd_field_pos;
    }

    return buf;
}

auto tiff_encode_pages(std::span<const TiffImage> pages, Compression comp)
    -> std::expected<std::vector<std::byte>, std::string> {
    return tiff_encode_pages(pages, TiffWriteOptions{comp, Predictor::None, 0, 0, 0, Orientation::TopLeft, {}});
}

auto tiff_write_pages(const std::string& path, std::span<const TiffImage> pages,
                       const TiffWriteOptions& opts) -> std::expected<void, std::string> {
    auto encoded = tiff_encode_pages(pages, opts);
    if (!encoded) return std::unexpected(encoded.error());

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected("cannot create file: " + path);
    }
    file.write(reinterpret_cast<const char*>(encoded->data()),
               static_cast<std::streamsize>(encoded->size()));
    if (!file) {
        return std::unexpected("failed to write file: " + path);
    }
    return {};
}

auto tiff_write_pages(const std::string& path, std::span<const TiffImage> pages,
                       Compression comp) -> std::expected<void, std::string> {
    return tiff_write_pages(path, pages, TiffWriteOptions{comp, Predictor::None, 0, 0, 0, Orientation::TopLeft, {}});
}

// ---------------------------------------------------------------------------
// mmap zero-copy
// ---------------------------------------------------------------------------

} // namespace utils

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace utils {

auto tiff_mmap(const std::string& path) -> std::expected<Tensor, std::string> {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return std::unexpected("cannot open file: " + path);
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        return std::unexpected("cannot stat file: " + path);
    }
    auto file_size = static_cast<std::size_t>(st.st_size);

    void* mapping = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapping == MAP_FAILED) {
        return std::unexpected("mmap failed for: " + path);
    }

    auto* data = static_cast<const std::byte*>(mapping);

    // Parse TIFF header
    if (file_size < 8) {
        ::munmap(mapping, file_size);
        return std::unexpected("file too small for TIFF header");
    }

    ByteOrder bo;
    if (data[0] == std::byte{0x49} && data[1] == std::byte{0x49}) {
        bo = ByteOrder::Little;
    } else if (data[0] == std::byte{0x4D} && data[1] == std::byte{0x4D}) {
        bo = ByteOrder::Big;
        ::munmap(mapping, file_size);
        return std::unexpected("mmap requires native (little-endian) byte order");
    } else {
        ::munmap(mapping, file_size);
        return std::unexpected("invalid TIFF byte order marker");
    }

    auto magic = read_u16(data + 2, bo);
    if (magic != 42) {
        ::munmap(mapping, file_size);
        return std::unexpected("invalid TIFF magic number");
    }

    auto first_ifd = read_u32(data + 4, bo);
    if (first_ifd == 0) {
        ::munmap(mapping, file_size);
        return std::unexpected("no IFD found");
    }

    auto ifd_result = parse_ifd(data, file_size, first_ifd, bo);
    if (!ifd_result) {
        ::munmap(mapping, file_size);
        return std::unexpected(ifd_result.error());
    }

    const auto& info = ifd_result->first;

    // Validate constraints for mmap
    if (info.compression != 1) {
        ::munmap(mapping, file_size);
        return std::unexpected("mmap requires uncompressed TIFF");
    }

    if (info.is_tiled()) {
        ::munmap(mapping, file_size);
        return std::unexpected("mmap does not support tiled TIFF");
    }

    if (info.width == 0 || info.height == 0) {
        ::munmap(mapping, file_size);
        return std::unexpected("invalid image dimensions");
    }

    auto bps = info.bits_per_sample;
    if (bps != 8 && bps != 16 && bps != 32) {
        ::munmap(mapping, file_size);
        return std::unexpected("unsupported bits per sample: " + std::to_string(bps));
    }

    auto spp = info.samples_per_pixel;
    auto bytes_per_sample = static_cast<std::size_t>(bps) / 8;
    auto row_bytes = static_cast<std::size_t>(info.width) * spp * bytes_per_sample;
    auto expected_total = static_cast<std::size_t>(info.height) * row_bytes;

    // Verify strip contiguity
    auto rows_per_strip = info.rows_per_strip;
    if (rows_per_strip > info.height) rows_per_strip = info.height;
    auto num_strips = (info.height + rows_per_strip - 1) / rows_per_strip;

    if (info.strip_offsets.size() != num_strips ||
        info.strip_byte_counts.size() != num_strips) {
        ::munmap(mapping, file_size);
        return std::unexpected("strip offset/count mismatch");
    }

    // Check strips are contiguous
    auto pixel_start = info.strip_offsets[0];
    std::size_t total_strip_bytes = 0;
    for (std::uint32_t s = 0; s < num_strips; ++s) {
        if (info.strip_offsets[s] != pixel_start + total_strip_bytes) {
            ::munmap(mapping, file_size);
            return std::unexpected("mmap requires contiguous strips");
        }
        total_strip_bytes += info.strip_byte_counts[s];
    }

    if (total_strip_bytes != expected_total) {
        ::munmap(mapping, file_size);
        return std::unexpected("strip data size mismatch");
    }

    if (pixel_start + expected_total > file_size) {
        ::munmap(mapping, file_size);
        return std::unexpected("pixel data extends past end of file");
    }

    // Determine DType
    DType dt{};
    if (info.sample_format == 3) {
        if (bps == 32) dt = DType::Float32;
        else {
            ::munmap(mapping, file_size);
            return std::unexpected("unsupported float bit depth");
        }
    } else if (info.sample_format == 2) {
        if (bps == 8) dt = DType::Int8;
        else if (bps == 16) dt = DType::Int16;
        else dt = DType::Int32;
    } else {
        if (bps == 8) dt = DType::UInt8;
        else if (bps == 16) dt = DType::UInt16;
        else dt = DType::UInt32;
    }

    // Build shape
    std::vector<std::size_t> shape;
    if (spp == 1) {
        shape = {info.height, info.width};
    } else {
        shape = {info.height, info.width, static_cast<std::size_t>(spp)};
    }

    // Create shared_ptr with munmap deleter
    auto storage = std::shared_ptr<void>(mapping, [file_size](void* p) {
        ::munmap(p, file_size);
    });

    auto* pixel_ptr = static_cast<void*>(
        static_cast<std::byte*>(mapping) + pixel_start);

    return Tensor::adapt(std::move(storage), pixel_ptr, shape, dt);
}

} // namespace utils
