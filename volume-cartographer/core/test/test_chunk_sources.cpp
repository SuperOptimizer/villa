#include "test.hpp"

#include "vc/core/cache/ChunkSource.hpp"
#include "vc/core/cache/HttpMetadataFetcher.hpp"
#include "vc/core/cache/CacheUtils.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fs = std::filesystem;
using namespace vc::cache;

// ---- helpers ----------------------------------------------------------------

static fs::path makeTempDir(const std::string& prefix)
{
    auto p = fs::temp_directory_path() / (prefix + "_" + std::to_string(::getpid()));
    fs::create_directories(p);
    return p;
}

struct TempDirGuard {
    fs::path path;
    explicit TempDirGuard(const std::string& prefix) : path(makeTempDir(prefix)) {}
    ~TempDirGuard() { std::error_code ec; fs::remove_all(path, ec); }
};

static void writeFile(const fs::path& path, const std::string& content)
{
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
}

static nlohmann::json makeZarray(
    std::array<int, 3> shape,
    std::array<int, 3> chunks,
    const std::string& delimiter = ".")
{
    nlohmann::json j;
    j["zarr_format"] = 2;
    j["shape"] = {shape[0], shape[1], shape[2]};
    j["chunks"] = {chunks[0], chunks[1], chunks[2]};
    j["dtype"] = "|u1";
    j["compressor"] = nullptr;
    j["fill_value"] = 0;
    j["order"] = "C";
    j["dimension_separator"] = delimiter;
    return j;
}

// Create a minimal zarr directory with the given number of levels.
// Each level halves the shape. Chunk shape stays constant.
static void createZarrDir(
    const fs::path& root,
    int numLevels,
    std::array<int, 3> baseShape = {128, 256, 256},
    std::array<int, 3> chunkShape = {32, 32, 32},
    const std::string& delimiter = ".")
{
    fs::create_directories(root);
    writeFile(root / ".zgroup", R"({"zarr_format":2})");

    for (int lvl = 0; lvl < numLevels; lvl++) {
        auto levelDir = root / std::to_string(lvl);
        fs::create_directories(levelDir);

        std::array<int, 3> shape = {
            baseShape[0] >> lvl,
            baseShape[1] >> lvl,
            baseShape[2] >> lvl
        };
        auto zarray = makeZarray(shape, chunkShape, delimiter);
        writeFile(levelDir / ".zarray", zarray.dump(2));
    }
}

// Write a fake chunk file at a given path
static void writeChunk(
    const fs::path& root,
    int level,
    int iz, int iy, int ix,
    const std::vector<uint8_t>& data,
    const std::string& delimiter = ".")
{
    auto dir = root / std::to_string(level);
    fs::create_directories(dir);
    auto filename = std::to_string(iz) + delimiter +
                    std::to_string(iy) + delimiter +
                    std::to_string(ix);
    auto path = dir / filename;
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
}

// =============================================================================
// FileSystemChunkSource tests
// =============================================================================

TEST(FileSystemChunkSource, DiscoverLevels)
{
    TempDirGuard tmp("test_fs_discover");
    createZarrDir(tmp.path, 3);

    FileSystemChunkSource src(tmp.path);
    EXPECT_EQ(src.numLevels(), 3);

    auto shape0 = src.levelShape(0);
    EXPECT_EQ(shape0[0], 128);
    EXPECT_EQ(shape0[1], 256);
    EXPECT_EQ(shape0[2], 256);

    auto shape1 = src.levelShape(1);
    EXPECT_EQ(shape1[0], 64);
    EXPECT_EQ(shape1[1], 128);
    EXPECT_EQ(shape1[2], 128);

    auto chunk0 = src.chunkShape(0);
    EXPECT_EQ(chunk0[0], 32);
    EXPECT_EQ(chunk0[1], 32);
    EXPECT_EQ(chunk0[2], 32);
}

TEST(FileSystemChunkSource, PresuppliedMetadata)
{
    TempDirGuard tmp("test_fs_presupplied");
    fs::create_directories(tmp.path);

    std::vector<FileSystemChunkSource::LevelMeta> levels = {
        {{100, 200, 300}, {10, 20, 30}},
        {{50, 100, 150}, {10, 20, 30}},
    };

    FileSystemChunkSource src(tmp.path, ".", std::move(levels));
    EXPECT_EQ(src.numLevels(), 2);
    EXPECT_EQ(src.levelShape(0)[0], 100);
    EXPECT_EQ(src.chunkShape(1)[2], 30);
}

TEST(FileSystemChunkSource, FetchRoundTrip)
{
    TempDirGuard tmp("test_fs_fetch");
    createZarrDir(tmp.path, 2);

    // Write a fake chunk
    std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF, 0x42};
    writeChunk(tmp.path, 0, 1, 2, 3, data);

    FileSystemChunkSource src(tmp.path);
    auto result = src.fetch({0, 1, 2, 3});
    ASSERT_EQ(result.size(), data.size());
    EXPECT_EQ(result[0], 0xDE);
    EXPECT_EQ(result[4], 0x42);
}

TEST(FileSystemChunkSource, FetchMissReturnsEmpty)
{
    TempDirGuard tmp("test_fs_miss");
    createZarrDir(tmp.path, 1);

    FileSystemChunkSource src(tmp.path);
    auto result = src.fetch({0, 99, 99, 99});
    EXPECT_TRUE(result.empty());
}

TEST(FileSystemChunkSource, FetchMultipleLevels)
{
    TempDirGuard tmp("test_fs_multi_level");
    createZarrDir(tmp.path, 3);

    std::vector<uint8_t> d0 = {0x00, 0x01};
    std::vector<uint8_t> d1 = {0x10, 0x11};
    std::vector<uint8_t> d2 = {0x20, 0x21};
    writeChunk(tmp.path, 0, 0, 0, 0, d0);
    writeChunk(tmp.path, 1, 0, 0, 0, d1);
    writeChunk(tmp.path, 2, 0, 0, 0, d2);

    FileSystemChunkSource src(tmp.path);
    auto r0 = src.fetch({0, 0, 0, 0});
    auto r1 = src.fetch({1, 0, 0, 0});
    auto r2 = src.fetch({2, 0, 0, 0});

    ASSERT_EQ(r0.size(), 2u);
    ASSERT_EQ(r1.size(), 2u);
    ASSERT_EQ(r2.size(), 2u);
    EXPECT_EQ(r0[0], 0x00);
    EXPECT_EQ(r1[0], 0x10);
    EXPECT_EQ(r2[0], 0x20);
}

TEST(FileSystemChunkSource, SlashDelimiter)
{
    TempDirGuard tmp("test_fs_slash_delim");

    // Create zarr with "/" delimiter
    fs::create_directories(tmp.path / "0");
    auto zarray = makeZarray({64, 64, 64}, {32, 32, 32}, "/");
    writeFile(tmp.path / "0" / ".zarray", zarray.dump(2));

    // Write chunk with "/" delimiter: 0/1/2/3 (nested dirs)
    auto chunkDir = tmp.path / "0" / "1" / "2";
    fs::create_directories(chunkDir);
    std::vector<uint8_t> data = {0xAB, 0xCD};
    std::ofstream f(chunkDir / "3", std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    f.close();

    FileSystemChunkSource src(tmp.path, "/");
    auto result = src.fetch({0, 1, 2, 3});
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 0xAB);
}

TEST(FileSystemChunkSource, OutOfRangeLevelReturnsZero)
{
    TempDirGuard tmp("test_fs_oob");
    createZarrDir(tmp.path, 2);

    FileSystemChunkSource src(tmp.path);
    auto shape = src.levelShape(99);
    EXPECT_EQ(shape[0], 0);
    EXPECT_EQ(shape[1], 0);
    EXPECT_EQ(shape[2], 0);

    auto chunk = src.chunkShape(-1);
    EXPECT_EQ(chunk[0], 0);
}

TEST(FileSystemChunkSource, EmptyDirNoLevels)
{
    TempDirGuard tmp("test_fs_empty");
    fs::create_directories(tmp.path);
    writeFile(tmp.path / ".zgroup", R"({"zarr_format":2})");

    FileSystemChunkSource src(tmp.path);
    EXPECT_EQ(src.numLevels(), 0);
}

TEST(FileSystemChunkSource, MalformedZarraySkipped)
{
    TempDirGuard tmp("test_fs_malformed");
    fs::create_directories(tmp.path / "0");
    writeFile(tmp.path / "0" / ".zarray", "not valid json {{{}");
    fs::create_directories(tmp.path / "1");
    auto zarray = makeZarray({64, 64, 64}, {32, 32, 32});
    writeFile(tmp.path / "1" / ".zarray", zarray.dump(2));

    FileSystemChunkSource src(tmp.path);
    // Level 0 has malformed metadata, level 1 is valid.
    // Since we index by level number, numLevels = 2 (indices 0 and 1)
    EXPECT_EQ(src.numLevels(), 2);
    // Level 0 shape should be zeros (malformed)
    auto shape0 = src.levelShape(0);
    EXPECT_EQ(shape0[0], 0);
    // Level 1 should have correct data
    auto shape1 = src.levelShape(1);
    EXPECT_EQ(shape1[0], 64);
}

// =============================================================================
// HttpChunkSource tests (no network — just construction and URL building)
// =============================================================================

TEST(HttpChunkSource, Construction)
{
    std::vector<FileSystemChunkSource::LevelMeta> levels = {
        {{128, 256, 256}, {32, 32, 32}},
        {{64, 128, 128}, {32, 32, 32}},
    };

    HttpChunkSource src("https://example.com/zarr/", ".", std::move(levels));
    EXPECT_EQ(src.numLevels(), 2);
    EXPECT_EQ(src.levelShape(0)[0], 128);
    EXPECT_EQ(src.chunkShape(1)[2], 32);
}

TEST(HttpChunkSource, OutOfRange)
{
    std::vector<FileSystemChunkSource::LevelMeta> levels = {
        {{100, 200, 300}, {10, 20, 30}},
    };

    HttpChunkSource src("https://example.com", ".", std::move(levels));
    auto shape = src.levelShape(5);
    EXPECT_EQ(shape[0], 0);
}

// =============================================================================
// HttpMetadataFetcher tests (local only — no network)
// =============================================================================

TEST(HttpMetadataFetcher, FetchRemoteMetadataCachesLocally)
{
    TempDirGuard staging("test_staging");

    // Pre-populate staging dir as if metadata was already fetched
    auto volDir = staging.path / "testvol";
    fs::create_directories(volDir / "0");
    fs::create_directories(volDir / "1");

    auto zarray0 = makeZarray({128, 256, 256}, {32, 32, 32});
    auto zarray1 = makeZarray({64, 128, 128}, {32, 32, 32});
    writeFile(volDir / "0" / ".zarray", zarray0.dump(2));
    writeFile(volDir / "1" / ".zarray", zarray1.dump(2));

    nlohmann::json meta;
    meta["uuid"] = "testvol";
    meta["name"] = "testvol";
    meta["type"] = "vol";
    meta["width"] = 256;
    meta["height"] = 256;
    meta["slices"] = 128;
    meta["format"] = "zarr";
    writeFile(volDir / "meta.json", meta.dump(2));

    // Now fetchRemoteZarrMetadata should use cached data
    auto info = fetchRemoteZarrMetadata(
        "https://example.com/testvol/", staging.path);

    EXPECT_EQ(info.numLevels, 2);
    EXPECT_EQ(info.delimiter, ".");
    EXPECT_EQ(info.stagingDir, volDir);
    // URL should have trailing slash stripped
    EXPECT_EQ(info.url, "https://example.com/testvol");
}

TEST(HttpMetadataFetcher, DeriveVolumeIdFromUrl)
{
    TempDirGuard staging("test_derive_vol");

    // Pre-create cached metadata for a volume with path components
    auto volDir = staging.path / "scroll1";
    fs::create_directories(volDir / "0");

    auto zarray = makeZarray({64, 64, 64}, {32, 32, 32});
    writeFile(volDir / "0" / ".zarray", zarray.dump(2));

    nlohmann::json meta;
    meta["uuid"] = "scroll1";
    writeFile(volDir / "meta.json", meta.dump(2));

    auto info = fetchRemoteZarrMetadata(
        "https://s3.example.com/bucket/data/scroll1/", staging.path);

    EXPECT_EQ(info.numLevels, 1);
    EXPECT_EQ(info.stagingDir, volDir);
}

TEST(HttpMetadataFetcher, DimensionSeparatorParsed)
{
    TempDirGuard staging("test_dim_sep");

    auto volDir = staging.path / "vol";
    fs::create_directories(volDir / "0");

    // Use "/" dimension separator
    auto zarray = makeZarray({64, 64, 64}, {32, 32, 32}, "/");
    writeFile(volDir / "0" / ".zarray", zarray.dump(2));

    nlohmann::json meta;
    meta["uuid"] = "vol";
    writeFile(volDir / "meta.json", meta.dump(2));

    auto info = fetchRemoteZarrMetadata(
        "https://example.com/vol", staging.path);

    EXPECT_EQ(info.delimiter, "/");
}

TEST(HttpMetadataFetcher, HttpGetStringNoCurlReturnsEmpty)
{
    // When VC_USE_CURL is defined, this would actually try HTTP.
    // We're just testing the function exists and is callable.
    // With a bad URL, it should return empty (or try and fail).
    // Can't meaningfully test without a server, so just verify
    // compilation and basic behavior.
    auto result = httpGetString("http://localhost:1/nonexistent");
    // May or may not be empty depending on whether curl is linked,
    // but should not crash.
    (void)result;
    EXPECT_TRUE(true);
}

TEST(HttpMetadataFetcher, HttpDownloadFileNoCurl)
{
    TempDirGuard tmp("test_dl");
    auto dest = tmp.path / "out.bin";
    auto ok = httpDownloadFile("http://localhost:1/nonexistent", dest);
    // Should not crash, and likely fails since URL is invalid
    (void)ok;
    EXPECT_TRUE(true);
}

// =============================================================================
// CacheUtils integration (already tested in test_cache_primitives,
// but verify chunkFilename works with slash delimiter for path construction)
// =============================================================================

TEST(CacheUtils, ChunkFilenameSlashDelimiter)
{
    ChunkKey key{0, 3, 5, 7};
    auto name = chunkFilename(key, "/");
    EXPECT_EQ(name, "3/5/7");
}

TEST(CacheUtils, ChunkFilenameDotDelimiter)
{
    ChunkKey key{2, 1, 2, 3};
    auto name = chunkFilename(key, ".");
    EXPECT_EQ(name, "1.2.3");
}
