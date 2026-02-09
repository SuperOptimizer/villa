#include "vc/core/util/Tiff.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <numeric>
#include <opencv2/imgproc.hpp>

// ============================================================================
// TIFF constants (tag IDs)
// ============================================================================
namespace {

constexpr uint16_t TAG_ImageWidth        = 256;
constexpr uint16_t TAG_ImageLength       = 257;
constexpr uint16_t TAG_BitsPerSample     = 258;
constexpr uint16_t TAG_Compression       = 259;
constexpr uint16_t TAG_Photometric       = 262;
constexpr uint16_t TAG_SamplesPerPixel   = 277;
constexpr uint16_t TAG_XResolution       = 282;
constexpr uint16_t TAG_YResolution       = 283;
constexpr uint16_t TAG_PlanarConfig      = 284;
constexpr uint16_t TAG_ResolutionUnit    = 296;
constexpr uint16_t TAG_Predictor         = 317;
constexpr uint16_t TAG_TileWidth         = 322;
constexpr uint16_t TAG_TileLength        = 323;
constexpr uint16_t TAG_TileOffsets       = 324;
constexpr uint16_t TAG_TileByteCounts    = 325;
constexpr uint16_t TAG_SampleFormat      = 339;

// TIFF data types
constexpr uint16_t TIFF_SHORT    = 3;
constexpr uint16_t TIFF_LONG     = 4;
constexpr uint16_t TIFF_RATIONAL = 5;

constexpr uint16_t PLANARCONFIG_CONTIG = 1;
constexpr uint16_t PREDICTOR_NONE       = 1;
constexpr uint16_t PREDICTOR_HORIZONTAL = 2;

// ============================================================================
// LZW Codec
// ============================================================================
constexpr int LZW_CLEAR_CODE = 256;
constexpr int LZW_EOI_CODE   = 257;
constexpr int LZW_FIRST_CODE = 258;
constexpr int LZW_MAX_CODE   = 4095;
constexpr int LZW_MAX_BITS   = 12;

// LZW decode (MSB-first bitstream, TIFF-style)
std::vector<uint8_t> lzw_decode(const uint8_t* src, size_t srcLen)
{
    std::vector<uint8_t> out;
    out.reserve(srcLen * 3);

    // String table: each entry is (prefix_code, suffix_byte)
    // For codes 0-255, the string is just that byte.
    struct Entry {
        int prefix;    // -1 for root entries
        uint8_t suffix;
        uint16_t length;
    };

    std::vector<Entry> table;
    table.reserve(4096);
    auto resetTable = [&]() {
        table.clear();
        for (int i = 0; i < 256; i++)
            table.push_back({-1, static_cast<uint8_t>(i), 1});
        // clear and eoi entries
        table.push_back({-1, 0, 0}); // 256 = clear
        table.push_back({-1, 0, 0}); // 257 = eoi
    };

    // Decode a table entry into output
    auto decodeString = [&](int code) {
        size_t pos = out.size();
        int c = code;
        while (c >= 0 && c < static_cast<int>(table.size())) {
            out.push_back(table[c].suffix);
            c = table[c].prefix;
            if (c == -1) break;
        }
        // Reverse the appended bytes
        std::reverse(out.begin() + pos, out.end());
    };

    auto firstChar = [&](int code) -> uint8_t {
        while (code >= 0 && table[code].prefix >= 0)
            code = table[code].prefix;
        return table[code].suffix;
    };

    // Bit reader (MSB-first)
    size_t bitPos = 0;
    auto readBits = [&](int nBits) -> int {
        int result = 0;
        for (int i = nBits - 1; i >= 0; --i) {
            size_t byteIdx = bitPos / 8;
            int bitIdx = 7 - static_cast<int>(bitPos % 8);
            if (byteIdx < srcLen) {
                if (src[byteIdx] & (1 << bitIdx))
                    result |= (1 << i);
            }
            bitPos++;
        }
        return result;
    };

    resetTable();
    int codeBits = 9;
    int nextCode = LZW_FIRST_CODE;

    // First code must be CLEAR
    int code = readBits(codeBits);
    if (code != LZW_CLEAR_CODE) return out;

    code = readBits(codeBits);
    if (code == LZW_EOI_CODE) return out;

    decodeString(code);
    int oldCode = code;

    while (bitPos / 8 < srcLen + 1) {
        code = readBits(codeBits);
        if (code == LZW_EOI_CODE) break;
        if (code == LZW_CLEAR_CODE) {
            resetTable();
            codeBits = 9;
            nextCode = LZW_FIRST_CODE;
            code = readBits(codeBits);
            if (code == LZW_EOI_CODE) break;
            decodeString(code);
            oldCode = code;
            continue;
        }

        if (code < static_cast<int>(table.size())) {
            decodeString(code);
            uint8_t fc = firstChar(code);
            table.push_back({oldCode, fc,
                static_cast<uint16_t>(table[oldCode].length + 1)});
            nextCode++;
        } else {
            // code == nextCode (the not-yet-in-table case)
            uint8_t fc = firstChar(oldCode);
            decodeString(oldCode);
            out.push_back(fc);
            table.push_back({oldCode, fc,
                static_cast<uint16_t>(table[oldCode].length + 1)});
            nextCode++;
        }

        if (nextCode > (1 << codeBits) - 1 && codeBits < LZW_MAX_BITS)
            codeBits++;

        oldCode = code;
    }

    return out;
}

// LZW encode (MSB-first bitstream, TIFF-style)
std::vector<uint8_t> lzw_encode(const uint8_t* data, size_t dataLen)
{
    std::vector<uint8_t> out;
    out.reserve(dataLen + dataLen / 8 + 64);

    // Bit writer (MSB-first)
    size_t bitPos = 0;
    auto writeBits = [&](int value, int nBits) {
        for (int i = nBits - 1; i >= 0; --i) {
            size_t byteIdx = bitPos / 8;
            int bitIdx = 7 - static_cast<int>(bitPos % 8);
            if (byteIdx >= out.size())
                out.push_back(0);
            if (value & (1 << i))
                out[byteIdx] |= (1 << bitIdx);
            bitPos++;
        }
    };

    // Hash table for string lookup: key = (prefix << 8) | byte, value = code
    // Use a simple open-addressing hash table
    constexpr int HASH_SIZE = 8192;  // power of 2, > 4096
    struct HashEntry { int key; int code; };
    std::vector<HashEntry> hashTable(HASH_SIZE, {-1, -1});

    auto hashLookup = [&](int key) -> int {
        int h = (key * 2654435761u) >> 19; // Fibonacci hash to 13 bits
        h &= (HASH_SIZE - 1);
        while (true) {
            if (hashTable[h].key == key) return hashTable[h].code;
            if (hashTable[h].key == -1) return -1;
            h = (h + 1) & (HASH_SIZE - 1);
        }
    };

    auto hashInsert = [&](int key, int code) {
        int h = (key * 2654435761u) >> 19;
        h &= (HASH_SIZE - 1);
        while (hashTable[h].key != -1)
            h = (h + 1) & (HASH_SIZE - 1);
        hashTable[h] = {key, code};
    };

    auto resetTable = [&]() {
        std::fill(hashTable.begin(), hashTable.end(), HashEntry{-1, -1});
        for (int i = 0; i < 256; i++)
            hashInsert((-1 << 8) | i, i); // sentinel prefix
    };

    int codeBits = 9;
    int nextCode = LZW_FIRST_CODE;

    resetTable();
    writeBits(LZW_CLEAR_CODE, codeBits);

    if (dataLen == 0) {
        writeBits(LZW_EOI_CODE, codeBits);
        // Pad to byte boundary
        if (bitPos % 8 != 0)
            out.resize((bitPos + 7) / 8);
        return out;
    }

    int w = data[0]; // current string = first byte

    for (size_t i = 1; i < dataLen; i++) {
        uint8_t c = data[i];
        int key = (w << 8) | c;
        int found = hashLookup(key);

        if (found >= 0) {
            w = found;
        } else {
            writeBits(w, codeBits);

            if (nextCode <= LZW_MAX_CODE) {
                hashInsert(key, nextCode);
                nextCode++;
                if (nextCode > (1 << codeBits) && codeBits < LZW_MAX_BITS)
                    codeBits++;
            } else {
                // Table full, emit clear code and reset
                writeBits(LZW_CLEAR_CODE, codeBits);
                resetTable();
                codeBits = 9;
                nextCode = LZW_FIRST_CODE;
            }

            w = c;
        }
    }

    writeBits(w, codeBits);
    writeBits(LZW_EOI_CODE, codeBits);

    // Pad to byte boundary
    out.resize((bitPos + 7) / 8);
    return out;
}

// ============================================================================
// PackBits Codec
// ============================================================================
std::vector<uint8_t> packbits_decode(const uint8_t* src, size_t srcLen, size_t expectedLen)
{
    std::vector<uint8_t> out;
    out.reserve(expectedLen);
    size_t i = 0;
    while (i < srcLen && out.size() < expectedLen) {
        int8_t n = static_cast<int8_t>(src[i++]);
        if (n >= 0) {
            int count = n + 1;
            for (int j = 0; j < count && i < srcLen && out.size() < expectedLen; j++)
                out.push_back(src[i++]);
        } else if (n != -128) {
            int count = -n + 1;
            if (i < srcLen) {
                uint8_t val = src[i++];
                for (int j = 0; j < count && out.size() < expectedLen; j++)
                    out.push_back(val);
            }
        }
        // n == -128: no-op
    }
    return out;
}

std::vector<uint8_t> packbits_encode(const uint8_t* data, size_t dataLen)
{
    std::vector<uint8_t> out;
    out.reserve(dataLen + dataLen / 128 + 2);
    size_t i = 0;
    while (i < dataLen) {
        // Check for run of identical bytes
        size_t runStart = i;
        while (i + 1 < dataLen && data[i] == data[i + 1] && (i - runStart) < 127)
            i++;
        size_t runLen = i - runStart + 1;
        i = runStart;

        if (runLen >= 3) {
            // Emit run
            out.push_back(static_cast<uint8_t>(-(static_cast<int>(runLen) - 1)));
            out.push_back(data[i]);
            i += runLen;
        } else {
            // Emit literal
            size_t litStart = i;
            while (i < dataLen) {
                if (i + 2 < dataLen && data[i] == data[i + 1] && data[i] == data[i + 2])
                    break;
                i++;
                if (i - litStart >= 128) break;
            }
            size_t litLen = i - litStart;
            out.push_back(static_cast<uint8_t>(litLen - 1));
            out.insert(out.end(), data + litStart, data + litStart + litLen);
        }
    }
    return out;
}

// ============================================================================
// Horizontal predictor (differencing)
// ============================================================================
void predictor_decode(uint8_t* data, uint32_t width, uint32_t height,
                      uint16_t spp, uint16_t bytesPerSample)
{
    const size_t rowStride = static_cast<size_t>(width) * spp * bytesPerSample;
    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + y * rowStride;
        // Running sum per sample channel
        for (uint32_t x = 1; x < width; x++) {
            for (uint16_t s = 0; s < spp; s++) {
                size_t off = (static_cast<size_t>(x) * spp + s) * bytesPerSample;
                size_t prev = (static_cast<size_t>(x - 1) * spp + s) * bytesPerSample;
                for (uint16_t b = 0; b < bytesPerSample; b++) {
                    row[off + b] += row[prev + b];
                }
            }
        }
    }
}

void predictor_encode(uint8_t* data, uint32_t width, uint32_t height,
                      uint16_t spp, uint16_t bytesPerSample)
{
    const size_t rowStride = static_cast<size_t>(width) * spp * bytesPerSample;
    for (uint32_t y = 0; y < height; y++) {
        uint8_t* row = data + y * rowStride;
        // Running difference per sample channel (process right to left)
        for (uint32_t x = width - 1; x >= 1; x--) {
            for (uint16_t s = 0; s < spp; s++) {
                size_t off = (static_cast<size_t>(x) * spp + s) * bytesPerSample;
                size_t prev = (static_cast<size_t>(x - 1) * spp + s) * bytesPerSample;
                for (uint16_t b = 0; b < bytesPerSample; b++) {
                    row[off + b] -= row[prev + b];
                }
            }
        }
    }
}

// ============================================================================
// Decompress a tile/strip
// ============================================================================
std::vector<uint8_t> decompress(const uint8_t* src, size_t srcLen,
                                uint16_t compression, size_t expectedLen)
{
    switch (compression) {
        case tiff::None:
            return {src, src + srcLen};
        case tiff::LZW:
            return lzw_decode(src, srcLen);
        case tiff::PackBits:
            return packbits_decode(src, srcLen, expectedLen);
        default:
            throw std::runtime_error("Unsupported TIFF compression: " +
                                     std::to_string(compression));
    }
}

// Compress tile data
std::vector<uint8_t> compress(const uint8_t* data, size_t dataLen,
                              uint16_t compression)
{
    switch (compression) {
        case tiff::None:
            return {data, data + dataLen};
        case tiff::LZW:
            return lzw_encode(data, dataLen);
        case tiff::PackBits:
            return packbits_encode(data, dataLen);
        default:
            throw std::runtime_error("Unsupported TIFF compression for writing: " +
                                     std::to_string(compression));
    }
}

// ============================================================================
// Binary I/O helpers (little-endian)
// ============================================================================
void write_u16(std::ofstream& f, uint16_t v)
{
    uint8_t buf[2] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8)};
    f.write(reinterpret_cast<char*>(buf), 2);
}

void write_u32(std::ofstream& f, uint32_t v)
{
    uint8_t buf[4] = {
        static_cast<uint8_t>(v),
        static_cast<uint8_t>(v >> 8),
        static_cast<uint8_t>(v >> 16),
        static_cast<uint8_t>(v >> 24)
    };
    f.write(reinterpret_cast<char*>(buf), 4);
}

uint16_t read_u16(const uint8_t* p, bool bigEndian)
{
    if (bigEndian) return (static_cast<uint16_t>(p[0]) << 8) | p[1];
    return p[0] | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t read_u32(const uint8_t* p, bool bigEndian)
{
    if (bigEndian)
        return (static_cast<uint32_t>(p[0]) << 24) |
               (static_cast<uint32_t>(p[1]) << 16) |
               (static_cast<uint32_t>(p[2]) << 8) | p[3];
    return p[0] | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// ============================================================================
// TIFF writing parameters
// ============================================================================
struct TiffParams {
    int bits;
    int sampleFormat;
    int elemSize;
    int channels;
};

TiffParams getTiffParams(int cvType) {
    switch (cvType) {
        case CV_8UC1:  return {8,  tiff::UInt,  1, 1};
        case CV_8UC3:  return {8,  tiff::UInt,  1, 3};
        case CV_16UC1: return {16, tiff::UInt,  2, 1};
        case CV_32FC1: return {32, tiff::Float, 4, 1};
        case CV_32FC3: return {32, tiff::Float, 4, 3};
        case CV_64FC1: return {64, tiff::Float, 8, 1};
        default:
            throw std::runtime_error("Unsupported cv::Mat type for TIFF: " +
                                     std::to_string(cvType));
    }
}

void fillTileBuffer(std::vector<uint8_t>& buf, int cvType, float padValue) {
    switch (cvType) {
        case CV_8UC1:
        case CV_8UC3:
            std::fill(buf.begin(), buf.end(), static_cast<uint8_t>(0));
            break;
        case CV_16UC1:
            std::fill(reinterpret_cast<uint16_t*>(buf.data()),
                     reinterpret_cast<uint16_t*>(buf.data() + buf.size()),
                     static_cast<uint16_t>(0));
            break;
        case CV_32FC1:
        case CV_32FC3:
            std::fill(reinterpret_cast<float*>(buf.data()),
                     reinterpret_cast<float*>(buf.data() + buf.size()),
                     padValue);
            break;
        case CV_64FC1:
            std::fill(reinterpret_cast<double*>(buf.data()),
                     reinterpret_cast<double*>(buf.data() + buf.size()),
                     static_cast<double>(padValue));
            break;
    }
}

cv::Mat convertWithScaling(const cv::Mat& img, int targetType) {
    if (img.type() == targetType)
        return img;

    cv::Mat result;
    const int srcType = img.type();

    if (srcType == CV_8UC1 && targetType == CV_16UC1) {
        img.convertTo(result, CV_16UC1, 257.0);
    } else if (srcType == CV_8UC1 && targetType == CV_32FC1) {
        img.convertTo(result, CV_32FC1, 1.0 / 255.0);
    } else if (srcType == CV_16UC1 && targetType == CV_8UC1) {
        img.convertTo(result, CV_8UC1, 1.0 / 257.0);
    } else if (srcType == CV_16UC1 && targetType == CV_32FC1) {
        img.convertTo(result, CV_32FC1, 1.0 / 65535.0);
    } else if (srcType == CV_32FC1 && targetType == CV_8UC1) {
        img.convertTo(result, CV_8UC1, 255.0);
    } else if (srcType == CV_32FC1 && targetType == CV_16UC1) {
        img.convertTo(result, CV_16UC1, 65535.0);
    } else {
        throw std::runtime_error("Unsupported type conversion");
    }

    return result;
}

// ============================================================================
// TIFF IFD tag helper
// ============================================================================
struct IFDTag {
    uint16_t tag;
    uint16_t type;
    uint32_t count;
    uint32_t valueOrOffset;
};

void writeTag(std::ofstream& f, const IFDTag& t) {
    write_u16(f, t.tag);
    write_u16(f, t.type);
    write_u32(f, t.count);
    write_u32(f, t.valueOrOffset);
}

// ============================================================================
// Write complete TIFF file: header + tile data + IFD + offset arrays
// ============================================================================
void writeTiffFile(std::ofstream& f,
                   const std::vector<std::vector<uint8_t>>& compressedTiles,
                   uint32_t imgW, uint32_t imgH,
                   uint32_t tileW, uint32_t tileH,
                   uint16_t bps, uint16_t spp, uint16_t sampleFormat,
                   uint16_t compression, bool usePredictor)
{
    // Header: "II" (little-endian) + 42 + placeholder offset to IFD
    f.write("II", 2);
    write_u16(f, 42);
    uint32_t ifdOffsetPos = static_cast<uint32_t>(f.tellp());
    write_u32(f, 0); // placeholder

    // Write compressed tile data, record offsets and sizes
    uint32_t numTiles = static_cast<uint32_t>(compressedTiles.size());
    std::vector<uint32_t> tileOffsets(numTiles);
    std::vector<uint32_t> tileByteCounts(numTiles);

    for (uint32_t i = 0; i < numTiles; i++) {
        tileOffsets[i] = static_cast<uint32_t>(f.tellp());
        tileByteCounts[i] = static_cast<uint32_t>(compressedTiles[i].size());
        f.write(reinterpret_cast<const char*>(compressedTiles[i].data()),
                compressedTiles[i].size());
    }

    // Prepare extra data arrays that go after the IFD
    // BitsPerSample array (if spp > 1, can't fit in 4 bytes for SHORT)
    std::vector<uint16_t> bpsArray(spp, bps);
    std::vector<uint16_t> sfArray(spp, sampleFormat);

    // Compute IFD position
    uint32_t ifdOffset = static_cast<uint32_t>(f.tellp());

    // Go back and write real IFD offset
    f.seekp(ifdOffsetPos);
    write_u32(f, ifdOffset);
    f.seekp(ifdOffset);

    // Build tag list
    std::vector<IFDTag> tags;
    uint16_t photometric = (spp == 3) ? tiff::RGB : tiff::MinIsBlack;

    // Determine how many tags
    int numTags = 14; // base tags (11 + XRes + YRes + ResUnit)
    if (usePredictor) numTags++;

    write_u16(f, static_cast<uint16_t>(numTags));

    // We need to know where extra data goes (after IFD + next offset)
    // IFD size: 2 (count) + numTags * 12 + 4 (next IFD) = 2 + numTags*12 + 4
    uint32_t afterIFD = ifdOffset + 2 + static_cast<uint32_t>(numTags) * 12 + 4;
    uint32_t extraOff = afterIFD;

    // Plan extra data layout
    uint32_t tileOffsetsOff = extraOff;
    extraOff += numTiles * 4;
    uint32_t tileByteCountsOff = extraOff;
    extraOff += numTiles * 4;
    uint32_t bpsArrayOff = 0;
    if (spp > 2) {  // > 2 shorts don't fit in value field
        bpsArrayOff = extraOff;
        extraOff += spp * 2;
    }
    uint32_t sfArrayOff = 0;
    if (spp > 2) {
        sfArrayOff = extraOff;
        extraOff += spp * 2;
    }
    // Resolution RATIONAL values: 72/1 = {72, 1} — 8 bytes each, share one copy
    uint32_t resolutionOff = extraOff;
    extraOff += 8; // one RATIONAL (numerator + denominator)

    // Write tags in numeric order
    // 256 ImageWidth
    writeTag(f, {TAG_ImageWidth, TIFF_LONG, 1, imgW});
    // 257 ImageLength
    writeTag(f, {TAG_ImageLength, TIFF_LONG, 1, imgH});
    // 258 BitsPerSample
    if (spp <= 2) {
        uint32_t val = bps;
        if (spp == 2) val = bps | (static_cast<uint32_t>(bps) << 16);
        writeTag(f, {TAG_BitsPerSample, TIFF_SHORT, spp, val});
    } else {
        writeTag(f, {TAG_BitsPerSample, TIFF_SHORT, spp, bpsArrayOff});
    }
    // 259 Compression
    writeTag(f, {TAG_Compression, TIFF_SHORT, 1, compression});
    // 262 Photometric
    writeTag(f, {TAG_Photometric, TIFF_SHORT, 1, photometric});
    // 277 SamplesPerPixel
    writeTag(f, {TAG_SamplesPerPixel, TIFF_SHORT, 1, spp});
    // 282 XResolution
    writeTag(f, {TAG_XResolution, TIFF_RATIONAL, 1, resolutionOff});
    // 283 YResolution
    writeTag(f, {TAG_YResolution, TIFF_RATIONAL, 1, resolutionOff});
    // 284 PlanarConfig
    writeTag(f, {TAG_PlanarConfig, TIFF_SHORT, 1, PLANARCONFIG_CONTIG});
    // 296 ResolutionUnit
    writeTag(f, {TAG_ResolutionUnit, TIFF_SHORT, 1, 2}); // 2 = inches
    // 317 Predictor (if applicable)
    if (usePredictor) {
        writeTag(f, {TAG_Predictor, TIFF_SHORT, 1, PREDICTOR_HORIZONTAL});
    }
    // 322 TileWidth
    writeTag(f, {TAG_TileWidth, TIFF_LONG, 1, tileW});
    // 323 TileLength
    writeTag(f, {TAG_TileLength, TIFF_LONG, 1, tileH});
    // 324 TileOffsets
    writeTag(f, {TAG_TileOffsets, TIFF_LONG, numTiles, tileOffsetsOff});
    // 325 TileByteCounts
    writeTag(f, {TAG_TileByteCounts, TIFF_LONG, numTiles, tileByteCountsOff});
    // 339 SampleFormat
    if (spp <= 2) {
        uint32_t val = sampleFormat;
        if (spp == 2) val = sampleFormat | (static_cast<uint32_t>(sampleFormat) << 16);
        writeTag(f, {TAG_SampleFormat, TIFF_SHORT, spp, val});
    } else {
        writeTag(f, {TAG_SampleFormat, TIFF_SHORT, spp, sfArrayOff});
    }

    // Next IFD offset = 0 (single image)
    write_u32(f, 0);

    // Write extra data
    // Tile offsets
    for (uint32_t i = 0; i < numTiles; i++)
        write_u32(f, tileOffsets[i]);
    // Tile byte counts
    for (uint32_t i = 0; i < numTiles; i++)
        write_u32(f, tileByteCounts[i]);
    // BitsPerSample array (if needed)
    if (spp > 2) {
        for (uint16_t i = 0; i < spp; i++)
            write_u16(f, bpsArray[i]);
    }
    // SampleFormat array (if needed)
    if (spp > 2) {
        for (uint16_t i = 0; i < spp; i++)
            write_u16(f, sfArray[i]);
    }
    // Resolution RATIONAL: 72/1
    write_u32(f, 72);
    write_u32(f, 1);
}

} // anonymous namespace

// ============================================================================
// TiffReader::Impl
// ============================================================================
struct TiffReader::Impl {
    std::vector<uint8_t> fileData;
    bool bigEndian = false;

    // Parsed IFD
    std::vector<uint32_t> tileOffsets;
    std::vector<uint32_t> tileByteCounts;
    uint16_t predictor = PREDICTOR_NONE;

    // For scanline-based: strip offsets/counts
    std::vector<uint32_t> stripOffsets;
    std::vector<uint32_t> stripByteCounts;
    uint32_t rowsPerStrip = 0;

    uint16_t readU16(size_t off) const {
        return read_u16(fileData.data() + off, bigEndian);
    }
    uint32_t readU32(size_t off) const {
        return read_u32(fileData.data() + off, bigEndian);
    }

    // Read an array of LONG values from offset
    std::vector<uint32_t> readU32Array(uint32_t offset, uint32_t count) const {
        std::vector<uint32_t> result(count);
        for (uint32_t i = 0; i < count; i++)
            result[i] = readU32(offset + i * 4);
        return result;
    }
    std::vector<uint32_t> readU16ArrayAsU32(uint32_t offset, uint32_t count) const {
        std::vector<uint32_t> result(count);
        for (uint32_t i = 0; i < count; i++)
            result[i] = readU16(offset + i * 2);
        return result;
    }
};

TiffReader::TiffReader(const std::filesystem::path& path)
    : _impl(std::make_unique<Impl>())
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Failed to open TIFF: " + path.string());

    auto sz = f.tellg();
    f.seekg(0);
    _impl->fileData.resize(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(_impl->fileData.data()), sz);
    f.close();

    const auto& data = _impl->fileData;
    if (data.size() < 8)
        throw std::runtime_error("TIFF too small: " + path.string());

    // Check byte order
    if (data[0] == 'I' && data[1] == 'I')
        _impl->bigEndian = false;
    else if (data[0] == 'M' && data[1] == 'M')
        _impl->bigEndian = true;
    else
        throw std::runtime_error("Not a TIFF file: " + path.string());

    uint16_t magic = _impl->readU16(2);
    if (magic != 42)
        throw std::runtime_error("Invalid TIFF magic number: " + path.string());

    uint32_t ifdOffset = _impl->readU32(4);
    if (ifdOffset + 2 > data.size())
        throw std::runtime_error("IFD offset out of range: " + path.string());

    uint16_t numTags = _impl->readU16(ifdOffset);

    // Parse tags into a map
    struct TagInfo { uint16_t type; uint32_t count; uint32_t valueOrOffset; };
    std::map<uint16_t, TagInfo> tagMap;

    for (uint16_t i = 0; i < numTags; i++) {
        size_t off = ifdOffset + 2 + i * 12;
        if (off + 12 > data.size()) break;
        uint16_t tag = _impl->readU16(off);
        uint16_t type = _impl->readU16(off + 2);
        uint32_t count = _impl->readU32(off + 4);
        uint32_t val = _impl->readU32(off + 8);
        tagMap[tag] = {type, count, val};
    }

    // Helper to get tag value (first value, as uint32)
    auto getTag = [&](uint16_t tag, uint32_t def = 0) -> uint32_t {
        auto it = tagMap.find(tag);
        if (it == tagMap.end()) return def;
        auto& ti = it->second;
        if (ti.type == TIFF_SHORT && ti.count == 1)
            return ti.valueOrOffset & 0xFFFF;
        return ti.valueOrOffset;
    };

    _width  = getTag(TAG_ImageWidth);
    _height = getTag(TAG_ImageLength);
    _bps    = static_cast<uint16_t>(getTag(TAG_BitsPerSample, 8));
    _spp    = static_cast<uint16_t>(getTag(TAG_SamplesPerPixel, 1));
    _sf     = static_cast<uint16_t>(getTag(TAG_SampleFormat, tiff::UInt));
    _compression = static_cast<uint16_t>(getTag(TAG_Compression, tiff::None));
    _impl->predictor = static_cast<uint16_t>(getTag(TAG_Predictor, PREDICTOR_NONE));

    // Handle BitsPerSample for multi-sample: read first value from array
    if (_spp > 1) {
        auto it = tagMap.find(TAG_BitsPerSample);
        if (it != tagMap.end() && it->second.count > 1) {
            if (it->second.type == TIFF_SHORT) {
                // If count <= 2, values are packed in valueOrOffset
                if (it->second.count <= 2) {
                    _bps = static_cast<uint16_t>(it->second.valueOrOffset & 0xFFFF);
                } else {
                    _bps = _impl->readU16(it->second.valueOrOffset);
                }
            }
        }
    }

    // Read tile info
    _tileW = getTag(TAG_TileWidth, 0);
    _tileH = getTag(TAG_TileLength, 0);

    if (_tileW > 0 && _tileH > 0) {
        // Tiled
        auto it = tagMap.find(TAG_TileOffsets);
        if (it != tagMap.end()) {
            uint32_t count = it->second.count;
            if (it->second.type == TIFF_LONG)
                _impl->tileOffsets = _impl->readU32Array(it->second.valueOrOffset, count);
        }
        it = tagMap.find(TAG_TileByteCounts);
        if (it != tagMap.end()) {
            uint32_t count = it->second.count;
            if (it->second.type == TIFF_LONG)
                _impl->tileByteCounts = _impl->readU32Array(it->second.valueOrOffset, count);
            else if (it->second.type == TIFF_SHORT)
                _impl->tileByteCounts = _impl->readU16ArrayAsU32(it->second.valueOrOffset, count);
        }
    } else {
        // Scanline-based: read strip offsets/counts
        constexpr uint16_t TAG_StripOffsets = 273;
        constexpr uint16_t TAG_StripByteCounts = 279;
        constexpr uint16_t TAG_RowsPerStrip = 278;

        _impl->rowsPerStrip = getTag(TAG_RowsPerStrip, _height);

        auto it = tagMap.find(TAG_StripOffsets);
        if (it != tagMap.end()) {
            uint32_t count = it->second.count;
            if (count == 1) {
                _impl->stripOffsets = {it->second.valueOrOffset};
            } else if (it->second.type == TIFF_LONG) {
                _impl->stripOffsets = _impl->readU32Array(it->second.valueOrOffset, count);
            } else if (it->second.type == TIFF_SHORT) {
                _impl->stripOffsets = _impl->readU16ArrayAsU32(it->second.valueOrOffset, count);
            }
        }
        it = tagMap.find(TAG_StripByteCounts);
        if (it != tagMap.end()) {
            uint32_t count = it->second.count;
            if (count == 1) {
                _impl->stripByteCounts = {it->second.valueOrOffset};
            } else if (it->second.type == TIFF_LONG) {
                _impl->stripByteCounts = _impl->readU32Array(it->second.valueOrOffset, count);
            } else if (it->second.type == TIFF_SHORT) {
                _impl->stripByteCounts = _impl->readU16ArrayAsU32(it->second.valueOrOffset, count);
            }
        }
    }
}

TiffReader::~TiffReader() = default;

int TiffReader::cvType() const
{
    if (_spp == 3 && _bps == 32 && _sf == tiff::Float) return CV_32FC3;
    if (_spp == 3 && _bps == 8) return CV_8UC3;
    if (_spp == 1) {
        switch (_bps) {
            case 8:  return CV_8UC1;
            case 16: return (_sf == tiff::Float) ? CV_16FC1 : CV_16UC1;
            case 32: return (_sf == tiff::Float) ? CV_32FC1 : CV_32SC1;
            case 64: return CV_64FC1;
        }
    }
    return CV_8UC(_spp);
}

uint32_t TiffReader::tilesAcross() const
{
    if (_tileW == 0) return 0;
    return (_width + _tileW - 1) / _tileW;
}

uint32_t TiffReader::tilesDown() const
{
    if (_tileH == 0) return 0;
    return (_height + _tileH - 1) / _tileH;
}

size_t TiffReader::tileBytes() const
{
    return static_cast<size_t>(_tileW) * _tileH * _spp * ((_bps + 7) / 8);
}

void TiffReader::readTile(uint32_t tileX, uint32_t tileY, void* buf, size_t bufSize)
{
    if (!isTiled())
        throw std::runtime_error("TiffReader::readTile called on non-tiled TIFF");

    uint32_t idx = tileY * tilesAcross() + tileX;
    if (idx >= _impl->tileOffsets.size())
        throw std::runtime_error("Tile index out of range");

    uint32_t offset = _impl->tileOffsets[idx];
    uint32_t compLen = _impl->tileByteCounts[idx];

    size_t expectedLen = tileBytes();
    auto decoded = decompress(_impl->fileData.data() + offset, compLen,
                              _compression, expectedLen);

    // Apply predictor
    if (_impl->predictor == PREDICTOR_HORIZONTAL && decoded.size() == expectedLen) {
        predictor_decode(decoded.data(), _tileW, _tileH, _spp, (_bps + 7) / 8);
    }

    size_t copyLen = std::min(decoded.size(), bufSize);
    std::memcpy(buf, decoded.data(), copyLen);
    // Zero-pad if decoded is shorter
    if (copyLen < bufSize)
        std::memset(static_cast<uint8_t*>(buf) + copyLen, 0, bufSize - copyLen);
}

cv::Mat TiffReader::readAll()
{
    int type = cvType();
    cv::Mat result(static_cast<int>(_height), static_cast<int>(_width), type);

    const int bytesPerSample = (_bps + 7) / 8;
    const int pixelBytes = bytesPerSample * _spp;

    if (isTiled()) {
        size_t tbytes = tileBytes();
        std::vector<uint8_t> tileBuf(tbytes);

        for (uint32_t ty = 0; ty < tilesDown(); ty++) {
            for (uint32_t tx = 0; tx < tilesAcross(); tx++) {
                readTile(tx, ty, tileBuf.data(), tbytes);

                uint32_t x0 = tx * _tileW;
                uint32_t y0 = ty * _tileH;
                uint32_t dx = std::min(_tileW, _width - x0);
                uint32_t dy = std::min(_tileH, _height - y0);

                for (uint32_t row = 0; row < dy; row++) {
                    const uint8_t* src = tileBuf.data() +
                        static_cast<size_t>(row) * _tileW * pixelBytes;
                    uint8_t* dst = result.ptr<uint8_t>(static_cast<int>(y0 + row)) +
                        x0 * pixelBytes;
                    std::memcpy(dst, src, dx * pixelBytes);
                }
            }
        }
    } else {
        // Scanline/strip-based
        uint32_t rowsPerStrip = _impl->rowsPerStrip;
        if (rowsPerStrip == 0) rowsPerStrip = _height;

        for (size_t s = 0; s < _impl->stripOffsets.size(); s++) {
            uint32_t stripStart = static_cast<uint32_t>(s) * rowsPerStrip;
            uint32_t stripRows = std::min(rowsPerStrip, _height - stripStart);
            size_t expectedLen = static_cast<size_t>(stripRows) * _width * pixelBytes;

            auto decoded = decompress(
                _impl->fileData.data() + _impl->stripOffsets[s],
                _impl->stripByteCounts[s],
                _compression, expectedLen);

            if (_impl->predictor == PREDICTOR_HORIZONTAL && decoded.size() == expectedLen) {
                predictor_decode(decoded.data(), _width, stripRows, _spp,
                                 static_cast<uint16_t>(bytesPerSample));
            }

            for (uint32_t row = 0; row < stripRows; row++) {
                uint8_t* dst = result.ptr<uint8_t>(static_cast<int>(stripStart + row));
                size_t rowOff = static_cast<size_t>(row) * _width * pixelBytes;
                size_t copyLen = std::min(
                    static_cast<size_t>(_width) * pixelBytes,
                    decoded.size() > rowOff ? decoded.size() - rowOff : 0);
                if (copyLen > 0)
                    std::memcpy(dst, decoded.data() + rowOff, copyLen);
                if (copyLen < static_cast<size_t>(_width) * pixelBytes)
                    std::memset(dst + copyLen, 0,
                                static_cast<size_t>(_width) * pixelBytes - copyLen);
            }
        }
    }

    // Convert RGB to BGR for 8-bit 3-channel (OpenCV uses BGR internally).
    // TIFF stores RGB; to match cv::imread behavior, swap R and B channels.
    // Skip for 32F 3-channel (e.g. XYZ coordinate data, not color).
    if (_spp == 3 && _bps == 8) {
        cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    }

    return result;
}

// ============================================================================
// TiffWriter::Impl
// ============================================================================
struct TiffWriter::Impl {
    std::ofstream file;
    std::vector<std::vector<uint8_t>> compressedTiles;
    uint32_t tilesAcross = 0;
    uint32_t tilesDown = 0;
    bool closed = false;
};

TiffWriter::TiffWriter(const std::filesystem::path& path,
                       uint32_t width, uint32_t height,
                       int cvType,
                       uint32_t tileW,
                       uint32_t tileH,
                       float padValue,
                       uint16_t compression)
    : _impl(std::make_unique<Impl>()),
      _width(width), _height(height), _tileW(tileW), _tileH(tileH),
      _cvType(cvType), _padValue(padValue), _compression(compression), _path(path)
{
    const auto params = getTiffParams(cvType);
    _elemSize = params.elemSize * params.channels;

    _impl->tilesAcross = (width + tileW - 1) / tileW;
    _impl->tilesDown = (height + tileH - 1) / tileH;
    _impl->compressedTiles.resize(
        static_cast<size_t>(_impl->tilesAcross) * _impl->tilesDown);

    _tileBuf.resize(static_cast<size_t>(tileW) * tileH * _elemSize);
}

TiffWriter::~TiffWriter() {
    close();
}

TiffWriter::TiffWriter(TiffWriter&& other) noexcept
    : _impl(std::move(other._impl)),
      _width(other._width), _height(other._height),
      _tileW(other._tileW), _tileH(other._tileH),
      _cvType(other._cvType), _elemSize(other._elemSize),
      _padValue(other._padValue), _compression(other._compression),
      _tileBuf(std::move(other._tileBuf)), _path(std::move(other._path))
{
}

TiffWriter& TiffWriter::operator=(TiffWriter&& other) noexcept {
    if (this != &other) {
        close();
        _impl = std::move(other._impl);
        _width = other._width;
        _height = other._height;
        _tileW = other._tileW;
        _tileH = other._tileH;
        _cvType = other._cvType;
        _elemSize = other._elemSize;
        _padValue = other._padValue;
        _compression = other._compression;
        _tileBuf = std::move(other._tileBuf);
        _path = std::move(other._path);
    }
    return *this;
}

bool TiffWriter::isOpen() const { return _impl && !_impl->closed; }

void TiffWriter::writeTile(uint32_t x0, uint32_t y0, const cv::Mat& tile) {
    if (!_impl || _impl->closed)
        throw std::runtime_error("TiffWriter: file not open");
    if (tile.type() != _cvType)
        throw std::runtime_error("TiffWriter: tile type mismatch");

    const uint32_t dx = static_cast<uint32_t>(tile.cols);
    const uint32_t dy = static_cast<uint32_t>(tile.rows);

    fillTileBuffer(_tileBuf, _cvType, _padValue);

    for (uint32_t ty = 0; ty < dy; ++ty) {
        const uint8_t* src = tile.ptr<uint8_t>(static_cast<int>(ty));
        std::memcpy(_tileBuf.data() + ty * _tileW * _elemSize,
                   src, dx * _elemSize);
    }

    // Apply predictor before compression
    const auto params = getTiffParams(_cvType);
    bool usePredictor = (_compression == tiff::LZW);

    std::vector<uint8_t> tileData(_tileBuf);
    if (usePredictor) {
        predictor_encode(tileData.data(), _tileW, _tileH,
                        static_cast<uint16_t>(params.channels),
                        static_cast<uint16_t>(params.elemSize));
    }

    auto compressed = compress(tileData.data(), tileData.size(), _compression);

    uint32_t tx = x0 / _tileW;
    uint32_t tyIdx = y0 / _tileH;
    uint32_t idx = tyIdx * _impl->tilesAcross + tx;
    if (idx < _impl->compressedTiles.size())
        _impl->compressedTiles[idx] = std::move(compressed);
}

void TiffWriter::writeRawTile(uint32_t tileX, uint32_t tileY,
                              const void* data, size_t len) {
    if (!_impl || _impl->closed)
        throw std::runtime_error("TiffWriter: file not open");

    // "raw" here means decoded data — we need to compress it
    const auto params = getTiffParams(_cvType);
    bool usePredictor = (_compression == tiff::LZW);

    std::vector<uint8_t> tileData(static_cast<const uint8_t*>(data),
                                  static_cast<const uint8_t*>(data) + len);
    if (usePredictor) {
        predictor_encode(tileData.data(), _tileW, _tileH,
                        static_cast<uint16_t>(params.channels),
                        static_cast<uint16_t>(params.elemSize));
    }

    auto compressed = compress(tileData.data(), tileData.size(), _compression);

    uint32_t idx = tileY * _impl->tilesAcross + tileX;
    if (idx < _impl->compressedTiles.size())
        _impl->compressedTiles[idx] = std::move(compressed);
}

void TiffWriter::close() {
    if (!_impl || _impl->closed) return;
    _impl->closed = true;

    const auto params = getTiffParams(_cvType);
    bool usePredictor = (_compression == tiff::LZW);

    // Fill any tiles that weren't written with empty compressed data
    size_t rawTileSize = static_cast<size_t>(_tileW) * _tileH * _elemSize;
    for (auto& ct : _impl->compressedTiles) {
        if (ct.empty()) {
            std::vector<uint8_t> emptyTile(rawTileSize, 0);
            if (_cvType == CV_32FC1 || _cvType == CV_32FC3) {
                float* fp = reinterpret_cast<float*>(emptyTile.data());
                size_t n = rawTileSize / sizeof(float);
                std::fill(fp, fp + n, _padValue);
            } else if (_cvType == CV_64FC1) {
                double* dp = reinterpret_cast<double*>(emptyTile.data());
                size_t n = rawTileSize / sizeof(double);
                std::fill(dp, dp + n, static_cast<double>(_padValue));
            }
            if (usePredictor) {
                predictor_encode(emptyTile.data(), _tileW, _tileH,
                                static_cast<uint16_t>(params.channels),
                                static_cast<uint16_t>(params.elemSize));
            }
            ct = compress(emptyTile.data(), emptyTile.size(), _compression);
        }
    }

    std::ofstream f(_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Failed to open TIFF for writing: " + _path.string());

    writeTiffFile(f, _impl->compressedTiles,
                  _width, _height, _tileW, _tileH,
                  static_cast<uint16_t>(params.bits),
                  static_cast<uint16_t>(params.channels),
                  static_cast<uint16_t>(params.sampleFormat),
                  _compression, usePredictor);
}

// ============================================================================
// writeTiff (free function)
// ============================================================================
void writeTiff(const std::filesystem::path& outPath,
               const cv::Mat& img,
               int cvType,
               uint32_t tileW,
               uint32_t tileH,
               float padValue,
               uint16_t compression)
{
    if (img.empty())
        throw std::runtime_error("Empty image for " + outPath.string());
    if (img.channels() != 1)
        throw std::runtime_error("Expected single-channel image for " + outPath.string());

    const int outType = (cvType < 0) ? img.type() : cvType;
    const cv::Mat outImg = convertWithScaling(img, outType);

    TiffWriter writer(outPath,
                      static_cast<uint32_t>(outImg.cols),
                      static_cast<uint32_t>(outImg.rows),
                      outType, tileW, tileH, padValue, compression);

    const uint32_t W = static_cast<uint32_t>(outImg.cols);
    const uint32_t H = static_cast<uint32_t>(outImg.rows);

    for (uint32_t y0 = 0; y0 < H; y0 += tileH) {
        const uint32_t dy = std::min(tileH, H - y0);
        for (uint32_t x0 = 0; x0 < W; x0 += tileW) {
            const uint32_t dx = std::min(tileW, W - x0);
            cv::Mat tile = outImg(cv::Range(static_cast<int>(y0),
                                            static_cast<int>(y0 + dy)),
                                  cv::Range(static_cast<int>(x0),
                                            static_cast<int>(x0 + dx)));
            writer.writeTile(x0, y0, tile);
        }
    }

    writer.close();
}

// ============================================================================
// tiff::imread / tiff::imwrite
// ============================================================================
namespace tiff {

cv::Mat imread(const std::filesystem::path& path)
{
    TiffReader reader(path);
    return reader.readAll();
}

void imwrite(const std::filesystem::path& path, const cv::Mat& img,
             uint16_t compression)
{
    if (img.empty())
        throw std::runtime_error("tiff::imwrite: empty image for " + path.string());

    int type = img.type();
    const auto params = getTiffParams(type);

    uint32_t tileW = 256, tileH = 256;
    if (img.cols <= 256 && img.rows <= 256) {
        tileW = static_cast<uint32_t>(img.cols);
        tileH = static_cast<uint32_t>(img.rows);
        // Round up to multiple of 16 (TIFF spec recommends multiples of 16)
        tileW = ((tileW + 15) / 16) * 16;
        tileH = ((tileH + 15) / 16) * 16;
        if (tileW == 0) tileW = 16;
        if (tileH == 0) tileH = 16;
    }

    TiffWriter writer(path,
                      static_cast<uint32_t>(img.cols),
                      static_cast<uint32_t>(img.rows),
                      type, tileW, tileH, 0.0f, compression);

    const uint32_t W = static_cast<uint32_t>(img.cols);
    const uint32_t H = static_cast<uint32_t>(img.rows);

    for (uint32_t y0 = 0; y0 < H; y0 += tileH) {
        const uint32_t dy = std::min(tileH, H - y0);
        for (uint32_t x0 = 0; x0 < W; x0 += tileW) {
            const uint32_t dx = std::min(tileW, W - x0);

            cv::Mat tile;
            if (type == CV_8UC3) {
                // Convert BGR to RGB for TIFF
                cv::Mat roi = img(cv::Range(static_cast<int>(y0),
                                            static_cast<int>(y0 + dy)),
                                  cv::Range(static_cast<int>(x0),
                                            static_cast<int>(x0 + dx)));
                cv::cvtColor(roi, tile, cv::COLOR_BGR2RGB);
            } else {
                tile = img(cv::Range(static_cast<int>(y0),
                                     static_cast<int>(y0 + dy)),
                           cv::Range(static_cast<int>(x0),
                                     static_cast<int>(x0 + dx)));
            }
            writer.writeTile(x0, y0, tile);
        }
    }

    writer.close();
}

std::vector<cv::Mat> imreadmulti(const std::filesystem::path& path)
{
    // Read entire file
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error("Failed to open TIFF: " + path.string());

    auto sz = f.tellg();
    f.seekg(0);
    auto fileData = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(fileData->data()), sz);
    f.close();

    const auto& data = *fileData;
    if (data.size() < 8)
        throw std::runtime_error("TIFF too small: " + path.string());

    bool bigEndian;
    if (data[0] == 'I' && data[1] == 'I')
        bigEndian = false;
    else if (data[0] == 'M' && data[1] == 'M')
        bigEndian = true;
    else
        throw std::runtime_error("Not a TIFF file: " + path.string());

    uint16_t magic = read_u16(data.data() + 2, bigEndian);
    if (magic != 42)
        throw std::runtime_error("Invalid TIFF magic: " + path.string());

    std::vector<cv::Mat> layers;
    uint32_t ifdOffset = read_u32(data.data() + 4, bigEndian);

    while (ifdOffset != 0 && ifdOffset + 2 <= data.size()) {
        // Parse IFD at this offset
        uint16_t numTags = read_u16(data.data() + ifdOffset, bigEndian);
        if (ifdOffset + 2 + static_cast<size_t>(numTags) * 12 + 4 > data.size())
            break;

        struct TagInfo { uint16_t type; uint32_t count; uint32_t valueOrOffset; };
        std::map<uint16_t, TagInfo> tagMap;

        for (uint16_t i = 0; i < numTags; i++) {
            size_t off = ifdOffset + 2 + i * 12;
            uint16_t tag = read_u16(data.data() + off, bigEndian);
            uint16_t type = read_u16(data.data() + off + 2, bigEndian);
            uint32_t count = read_u32(data.data() + off + 4, bigEndian);
            uint32_t val = read_u32(data.data() + off + 8, bigEndian);
            tagMap[tag] = {type, count, val};
        }

        auto getTag = [&](uint16_t tag, uint32_t def = 0) -> uint32_t {
            auto it = tagMap.find(tag);
            if (it == tagMap.end()) return def;
            auto& ti = it->second;
            if (ti.type == TIFF_SHORT && ti.count == 1)
                return ti.valueOrOffset & 0xFFFF;
            return ti.valueOrOffset;
        };

        uint32_t imgW = getTag(TAG_ImageWidth);
        uint32_t imgH = getTag(TAG_ImageLength);
        uint16_t bps = static_cast<uint16_t>(getTag(TAG_BitsPerSample, 8));
        uint16_t spp = static_cast<uint16_t>(getTag(TAG_SamplesPerPixel, 1));
        uint16_t sf = static_cast<uint16_t>(getTag(TAG_SampleFormat, tiff::UInt));
        uint16_t comp = static_cast<uint16_t>(getTag(TAG_Compression, tiff::None));
        uint16_t pred = static_cast<uint16_t>(getTag(TAG_Predictor, PREDICTOR_NONE));
        uint32_t tileW = getTag(TAG_TileWidth, 0);
        uint32_t tileH = getTag(TAG_TileLength, 0);

        // Handle multi-sample BitsPerSample
        if (spp > 1) {
            auto it = tagMap.find(TAG_BitsPerSample);
            if (it != tagMap.end() && it->second.count > 1) {
                if (it->second.type == TIFF_SHORT) {
                    if (it->second.count <= 2) {
                        bps = static_cast<uint16_t>(it->second.valueOrOffset & 0xFFFF);
                    } else {
                        bps = read_u16(data.data() + it->second.valueOrOffset, bigEndian);
                    }
                }
            }
        }

        // Determine cv type
        int cvType;
        if (spp == 3 && bps == 32 && sf == tiff::Float) cvType = CV_32FC3;
        else if (spp == 3 && bps == 8) cvType = CV_8UC3;
        else if (spp == 1) {
            switch (bps) {
                case 8:  cvType = CV_8UC1; break;
                case 16: cvType = (sf == tiff::Float) ? CV_16FC1 : CV_16UC1; break;
                case 32: cvType = (sf == tiff::Float) ? CV_32FC1 : CV_32SC1; break;
                case 64: cvType = CV_64FC1; break;
                default: cvType = CV_8UC1; break;
            }
        } else {
            cvType = CV_8UC(spp);
        }

        int bytesPerSample = (bps + 7) / 8;
        int pixelBytes = bytesPerSample * spp;
        cv::Mat result(static_cast<int>(imgH), static_cast<int>(imgW), cvType);

        if (tileW > 0 && tileH > 0) {
            // Tiled
            auto tileOffsetsIt = tagMap.find(TAG_TileOffsets);
            auto tileByteCountsIt = tagMap.find(TAG_TileByteCounts);
            if (tileOffsetsIt != tagMap.end() && tileByteCountsIt != tagMap.end()) {
                uint32_t numTiles = tileOffsetsIt->second.count;
                uint32_t tilesAcross = (imgW + tileW - 1) / tileW;

                for (uint32_t ti = 0; ti < numTiles; ti++) {
                    uint32_t tOff, tLen;
                    if (tileOffsetsIt->second.type == TIFF_LONG)
                        tOff = read_u32(data.data() + tileOffsetsIt->second.valueOrOffset + ti * 4, bigEndian);
                    else
                        tOff = read_u16(data.data() + tileOffsetsIt->second.valueOrOffset + ti * 2, bigEndian);

                    if (tileByteCountsIt->second.type == TIFF_LONG)
                        tLen = read_u32(data.data() + tileByteCountsIt->second.valueOrOffset + ti * 4, bigEndian);
                    else
                        tLen = read_u16(data.data() + tileByteCountsIt->second.valueOrOffset + ti * 2, bigEndian);

                    size_t expectedLen = static_cast<size_t>(tileW) * tileH * pixelBytes;
                    auto decoded = decompress(data.data() + tOff, tLen, comp, expectedLen);

                    if (pred == PREDICTOR_HORIZONTAL && decoded.size() == expectedLen) {
                        predictor_decode(decoded.data(), tileW, tileH, spp, bytesPerSample);
                    }

                    uint32_t tx = ti % tilesAcross;
                    uint32_t ty = ti / tilesAcross;
                    uint32_t x0 = tx * tileW;
                    uint32_t y0 = ty * tileH;
                    uint32_t dx = std::min(tileW, imgW - x0);
                    uint32_t dy = std::min(tileH, imgH - y0);

                    for (uint32_t row = 0; row < dy; row++) {
                        const uint8_t* src = decoded.data() +
                            static_cast<size_t>(row) * tileW * pixelBytes;
                        uint8_t* dst = result.ptr<uint8_t>(static_cast<int>(y0 + row)) +
                            x0 * pixelBytes;
                        size_t copyLen = std::min(static_cast<size_t>(dx * pixelBytes),
                                                  decoded.size() - static_cast<size_t>(row) * tileW * pixelBytes);
                        if (row * tileW * pixelBytes < decoded.size())
                            std::memcpy(dst, src, copyLen);
                    }
                }
            }
        } else {
            // Strip-based
            constexpr uint16_t TAG_StripOffsets = 273;
            constexpr uint16_t TAG_StripByteCounts = 279;
            constexpr uint16_t TAG_RowsPerStrip = 278;

            uint32_t rowsPerStrip = getTag(TAG_RowsPerStrip, imgH);

            auto stripOffsetsIt = tagMap.find(TAG_StripOffsets);
            auto stripByteCountsIt = tagMap.find(TAG_StripByteCounts);
            if (stripOffsetsIt != tagMap.end() && stripByteCountsIt != tagMap.end()) {
                uint32_t numStrips = stripOffsetsIt->second.count;
                for (uint32_t si = 0; si < numStrips; si++) {
                    uint32_t sOff, sLen;
                    if (numStrips == 1) {
                        sOff = stripOffsetsIt->second.valueOrOffset;
                        sLen = stripByteCountsIt->second.valueOrOffset;
                    } else {
                        if (stripOffsetsIt->second.type == TIFF_LONG)
                            sOff = read_u32(data.data() + stripOffsetsIt->second.valueOrOffset + si * 4, bigEndian);
                        else
                            sOff = read_u16(data.data() + stripOffsetsIt->second.valueOrOffset + si * 2, bigEndian);
                        if (stripByteCountsIt->second.type == TIFF_LONG)
                            sLen = read_u32(data.data() + stripByteCountsIt->second.valueOrOffset + si * 4, bigEndian);
                        else
                            sLen = read_u16(data.data() + stripByteCountsIt->second.valueOrOffset + si * 2, bigEndian);
                    }

                    uint32_t stripStart = si * rowsPerStrip;
                    uint32_t stripRows = std::min(rowsPerStrip, imgH - stripStart);
                    size_t expectedLen = static_cast<size_t>(stripRows) * imgW * pixelBytes;

                    auto decoded = decompress(data.data() + sOff, sLen, comp, expectedLen);

                    if (pred == PREDICTOR_HORIZONTAL && decoded.size() == expectedLen) {
                        predictor_decode(decoded.data(), imgW, stripRows, spp,
                                         static_cast<uint16_t>(bytesPerSample));
                    }

                    for (uint32_t row = 0; row < stripRows; row++) {
                        uint8_t* dst = result.ptr<uint8_t>(static_cast<int>(stripStart + row));
                        size_t rowOff = static_cast<size_t>(row) * imgW * pixelBytes;
                        size_t copyLen = std::min(
                            static_cast<size_t>(imgW) * pixelBytes,
                            decoded.size() > rowOff ? decoded.size() - rowOff : 0);
                        if (copyLen > 0)
                            std::memcpy(dst, decoded.data() + rowOff, copyLen);
                    }
                }
            }
        }

        // Convert RGB to BGR for 8-bit 3-channel
        if (spp == 3 && bps == 8) {
            cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
        }

        layers.push_back(std::move(result));

        // Follow next IFD offset
        size_t nextIfdPos = ifdOffset + 2 + static_cast<size_t>(numTags) * 12;
        ifdOffset = read_u32(data.data() + nextIfdPos, bigEndian);
    }

    return layers;
}

void imwritemulti(const std::filesystem::path& path,
                  const std::vector<cv::Mat>& layers)
{
    if (layers.empty())
        throw std::runtime_error("tiff::imwritemulti: no layers");

    // For each layer, compress all tiles into memory, then write
    // everything with chained IFDs.

    struct LayerData {
        std::vector<std::vector<uint8_t>> compressedTiles;
        uint32_t imgW, imgH, tileW, tileH;
        uint16_t bps, spp, sampleFormat;
        uint16_t compression;
        bool usePredictor;
    };

    std::vector<LayerData> allLayers;
    allLayers.reserve(layers.size());

    for (const auto& img : layers) {
        if (img.empty())
            throw std::runtime_error("tiff::imwritemulti: empty layer");

        int type = img.type();
        const auto params = getTiffParams(type);

        uint32_t tileW = 256, tileH = 256;
        if (img.cols <= 256 && img.rows <= 256) {
            tileW = static_cast<uint32_t>(img.cols);
            tileH = static_cast<uint32_t>(img.rows);
            tileW = ((tileW + 15) / 16) * 16;
            tileH = ((tileH + 15) / 16) * 16;
            if (tileW == 0) tileW = 16;
            if (tileH == 0) tileH = 16;
        }

        uint32_t imgW = static_cast<uint32_t>(img.cols);
        uint32_t imgH = static_cast<uint32_t>(img.rows);
        uint32_t tilesAcross = (imgW + tileW - 1) / tileW;
        uint32_t tilesDown = (imgH + tileH - 1) / tileH;
        uint32_t numTiles = tilesAcross * tilesDown;

        uint16_t bps = static_cast<uint16_t>(params.bits);
        uint16_t spp = static_cast<uint16_t>(params.channels);
        uint16_t sf = static_cast<uint16_t>(params.sampleFormat);
        uint16_t comp = tiff::LZW;
        bool usePredictor = (comp == tiff::LZW);
        int elemSize = params.elemSize * params.channels;

        std::vector<std::vector<uint8_t>> compressedTiles(numTiles);
        std::vector<uint8_t> tileBuf(static_cast<size_t>(tileW) * tileH * elemSize);

        for (uint32_t ty = 0; ty < tilesDown; ty++) {
            for (uint32_t tx = 0; tx < tilesAcross; tx++) {
                uint32_t x0 = tx * tileW;
                uint32_t y0 = ty * tileH;
                uint32_t dx = std::min(tileW, imgW - x0);
                uint32_t dy = std::min(tileH, imgH - y0);

                fillTileBuffer(tileBuf, type, 0.0f);

                cv::Mat srcTile;
                if (type == CV_8UC3) {
                    cv::Mat roi = img(cv::Range(static_cast<int>(y0),
                                                static_cast<int>(y0 + dy)),
                                      cv::Range(static_cast<int>(x0),
                                                static_cast<int>(x0 + dx)));
                    cv::cvtColor(roi, srcTile, cv::COLOR_BGR2RGB);
                } else {
                    srcTile = img(cv::Range(static_cast<int>(y0),
                                            static_cast<int>(y0 + dy)),
                                  cv::Range(static_cast<int>(x0),
                                            static_cast<int>(x0 + dx)));
                }

                for (uint32_t row = 0; row < dy; row++) {
                    const uint8_t* src = srcTile.ptr<uint8_t>(static_cast<int>(row));
                    std::memcpy(tileBuf.data() + row * tileW * elemSize,
                               src, dx * elemSize);
                }

                std::vector<uint8_t> tileData(tileBuf);
                if (usePredictor) {
                    predictor_encode(tileData.data(), tileW, tileH, spp,
                                    static_cast<uint16_t>(params.elemSize));
                }

                uint32_t idx = ty * tilesAcross + tx;
                compressedTiles[idx] = compress(tileData.data(), tileData.size(), comp);
            }
        }

        allLayers.push_back({std::move(compressedTiles), imgW, imgH, tileW, tileH,
                             bps, spp, sf, comp, usePredictor});
    }

    // Now write all layers to file with chained IFDs
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Failed to open TIFF for writing: " + path.string());

    // Header
    f.write("II", 2);
    write_u16(f, 42);
    // We'll patch the first IFD offset after writing all tile data
    uint32_t headerIfdOffsetPos = static_cast<uint32_t>(f.tellp());
    write_u32(f, 0); // placeholder

    // Write all tile data for all layers and record offsets
    struct LayerOffsets {
        std::vector<uint32_t> tileOffsets;
        std::vector<uint32_t> tileByteCounts;
    };
    std::vector<LayerOffsets> layerOffsets(allLayers.size());

    for (size_t li = 0; li < allLayers.size(); li++) {
        auto& ld = allLayers[li];
        uint32_t numTiles = static_cast<uint32_t>(ld.compressedTiles.size());
        layerOffsets[li].tileOffsets.resize(numTiles);
        layerOffsets[li].tileByteCounts.resize(numTiles);
        for (uint32_t ti = 0; ti < numTiles; ti++) {
            layerOffsets[li].tileOffsets[ti] = static_cast<uint32_t>(f.tellp());
            layerOffsets[li].tileByteCounts[ti] = static_cast<uint32_t>(ld.compressedTiles[ti].size());
            f.write(reinterpret_cast<const char*>(ld.compressedTiles[ti].data()),
                    ld.compressedTiles[ti].size());
        }
    }

    // Now write IFDs, chaining each to the next
    std::vector<uint32_t> nextIfdPatchPositions; // positions of next-IFD fields to patch

    for (size_t li = 0; li < allLayers.size(); li++) {
        auto& ld = allLayers[li];
        auto& lo = layerOffsets[li];
        uint32_t numTiles = static_cast<uint32_t>(ld.compressedTiles.size());
        uint16_t spp = ld.spp;

        uint32_t ifdOffset = static_cast<uint32_t>(f.tellp());

        // Patch previous pointer to this IFD
        if (li == 0) {
            // Patch header
            f.seekp(headerIfdOffsetPos);
            write_u32(f, ifdOffset);
            f.seekp(ifdOffset);
        } else {
            // Patch previous IFD's next-IFD field
            uint32_t patchPos = nextIfdPatchPositions.back();
            f.seekp(patchPos);
            write_u32(f, ifdOffset);
            f.seekp(ifdOffset);
        }

        // Build tags
        int numTagsVal = 14; // base tags (11 + XRes + YRes + ResUnit)
        if (ld.usePredictor) numTagsVal++;

        write_u16(f, static_cast<uint16_t>(numTagsVal));

        uint32_t afterIFD = ifdOffset + 2 + static_cast<uint32_t>(numTagsVal) * 12 + 4;
        uint32_t extraOff = afterIFD;

        uint32_t tileOffsetsOff = extraOff;
        extraOff += numTiles * 4;
        uint32_t tileByteCountsOff = extraOff;
        extraOff += numTiles * 4;
        uint32_t bpsArrayOff = 0;
        if (spp > 2) {
            bpsArrayOff = extraOff;
            extraOff += spp * 2;
        }
        uint32_t sfArrayOff = 0;
        if (spp > 2) {
            sfArrayOff = extraOff;
            extraOff += spp * 2;
        }
        uint32_t resolutionOff = extraOff;
        extraOff += 8; // one RATIONAL (72/1)

        uint16_t photometric = (spp == 3) ? tiff::RGB : tiff::MinIsBlack;

        writeTag(f, {TAG_ImageWidth, TIFF_LONG, 1, ld.imgW});
        writeTag(f, {TAG_ImageLength, TIFF_LONG, 1, ld.imgH});
        if (spp <= 2) {
            uint32_t val = ld.bps;
            if (spp == 2) val = ld.bps | (static_cast<uint32_t>(ld.bps) << 16);
            writeTag(f, {TAG_BitsPerSample, TIFF_SHORT, spp, val});
        } else {
            writeTag(f, {TAG_BitsPerSample, TIFF_SHORT, spp, bpsArrayOff});
        }
        writeTag(f, {TAG_Compression, TIFF_SHORT, 1, ld.compression});
        writeTag(f, {TAG_Photometric, TIFF_SHORT, 1, photometric});
        writeTag(f, {TAG_SamplesPerPixel, TIFF_SHORT, 1, spp});
        writeTag(f, {TAG_XResolution, TIFF_RATIONAL, 1, resolutionOff});
        writeTag(f, {TAG_YResolution, TIFF_RATIONAL, 1, resolutionOff});
        writeTag(f, {TAG_PlanarConfig, TIFF_SHORT, 1, PLANARCONFIG_CONTIG});
        writeTag(f, {TAG_ResolutionUnit, TIFF_SHORT, 1, 2}); // inches
        if (ld.usePredictor) {
            writeTag(f, {TAG_Predictor, TIFF_SHORT, 1, PREDICTOR_HORIZONTAL});
        }
        writeTag(f, {TAG_TileWidth, TIFF_LONG, 1, ld.tileW});
        writeTag(f, {TAG_TileLength, TIFF_LONG, 1, ld.tileH});
        writeTag(f, {TAG_TileOffsets, TIFF_LONG, numTiles, tileOffsetsOff});
        writeTag(f, {TAG_TileByteCounts, TIFF_LONG, numTiles, tileByteCountsOff});
        if (spp <= 2) {
            uint32_t val = ld.sampleFormat;
            if (spp == 2) val = ld.sampleFormat | (static_cast<uint32_t>(ld.sampleFormat) << 16);
            writeTag(f, {TAG_SampleFormat, TIFF_SHORT, spp, val});
        } else {
            writeTag(f, {TAG_SampleFormat, TIFF_SHORT, spp, sfArrayOff});
        }

        // Next IFD offset (0 for now, patched later if not last)
        uint32_t nextIfdFieldPos = static_cast<uint32_t>(f.tellp());
        nextIfdPatchPositions.push_back(nextIfdFieldPos);
        write_u32(f, 0);

        // Extra data: tile offsets
        for (uint32_t ti = 0; ti < numTiles; ti++)
            write_u32(f, lo.tileOffsets[ti]);
        // Tile byte counts
        for (uint32_t ti = 0; ti < numTiles; ti++)
            write_u32(f, lo.tileByteCounts[ti]);
        // BPS array
        if (spp > 2) {
            for (uint16_t s = 0; s < spp; s++)
                write_u16(f, ld.bps);
        }
        // SF array
        if (spp > 2) {
            for (uint16_t s = 0; s < spp; s++)
                write_u16(f, ld.sampleFormat);
        }
        // Resolution RATIONAL: 72/1
        write_u32(f, 72);
        write_u32(f, 1);
    }
}

// ============================================================================
// mergeTiffParts implementation
// ============================================================================

bool mergeTiffParts(const std::string& outputPath, int numParts)
{
    std::filesystem::path outDir(outputPath);
    if (!std::filesystem::is_directory(outDir)) outDir = outDir.parent_path();
    if (outDir.empty()) outDir = ".";

    std::map<std::filesystem::path, std::vector<std::filesystem::path>> groups;
    for (auto& entry : std::filesystem::directory_iterator(outDir)) {
        std::string fname = entry.path().filename().string();
        auto pos = fname.find(".part");
        if (pos == std::string::npos) continue;
        auto dotTif = fname.find(".tif", pos);
        if (dotTif == std::string::npos) continue;
        groups[outDir / (fname.substr(0, pos) + ".tif")].push_back(entry.path());
    }
    if (groups.empty()) {
        std::cerr << "No .partN.tif files found in " << outDir << "\n";
        return false;
    }

    std::cout << "Merging " << groups.size() << " TIFF(s) from " << numParts << " parts..." << std::endl;
    for (auto& [finalPath, partFiles] : groups) {
        std::sort(partFiles.begin(), partFiles.end());
        TiffReader firstReader(partFiles[0]);
        uint32_t w = firstReader.width();
        uint32_t h = firstReader.height();
        uint32_t tw = firstReader.tileWidth();
        uint32_t th = firstReader.tileHeight();
        uint16_t comp = firstReader.compression();
        int outCvType = firstReader.cvType();

        TiffWriter outWriter(finalPath, w, h, outCvType, tw, th, 0.0f, comp);

        size_t tileSizeBytes = firstReader.tileBytes();
        std::vector<uint8_t> buf(tileSizeBytes, 0);
        std::vector<uint8_t> zeroBuf(tileSizeBytes, 0);

        uint32_t tilesX = (w + tw - 1) / tw;
        uint32_t tilesY = (h + th - 1) / th;
        size_t merged = 0;
        for (uint32_t ty = 0; ty < tilesY; ty++) {
            for (uint32_t tx = 0; tx < tilesX; tx++) {
                bool found = false;
                for (auto& pf : partFiles) {
                    try {
                        TiffReader partReader(pf);
                        if (tx < partReader.tilesAcross() && ty < partReader.tilesDown()) {
                            partReader.readTile(tx, ty, buf.data(), tileSizeBytes);
                            if (buf != zeroBuf) {
                                found = true;
                                break;
                            }
                        }
                    } catch (...) {
                        continue;
                    }
                }
                if (found) {
                    outWriter.writeRawTile(tx, ty, buf.data(), tileSizeBytes);
                    merged++;
                }
            }
        }
        outWriter.close();

        for (auto& pf : partFiles)
            std::filesystem::remove(pf);
        std::cout << "  " << finalPath.filename().string() << ": " << merged << " tiles from " << partFiles.size() << " parts\n";
    }
    std::cout << "Merge complete.\n";
    return true;
}

} // namespace tiff
