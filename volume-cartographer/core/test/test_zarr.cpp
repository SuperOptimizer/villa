#include "test.hpp"
#include "vc/core/types/Zarr.hpp"
#include "vc/core/types/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <vector>

using namespace zarr;

// =============================================================================
// Helper: allclose for vc::Tensor
// =============================================================================

template<typename T>
bool allclose(const vc::Tensor<T>& a, const vc::Tensor<T>& b,
              double rtol = 1e-5, double atol = 1e-8)
{
    if (a.shape() != b.shape()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = std::abs(static_cast<double>(a.data()[i]) -
                               static_cast<double>(b.data()[i]));
        double limit = atol + rtol * std::abs(static_cast<double>(b.data()[i]));
        if (diff > limit) return false;
    }
    return true;
}

// =============================================================================
// 1. MemoryStore tests
// =============================================================================

TEST(MemoryStore, SetGetRoundTrip) {
    MemoryStore store;
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    auto r = store.set("my/key", data);
    ASSERT_TRUE(r.has_value());

    auto got = store.get("my/key");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, data);
}

TEST(MemoryStore, ExistsAndErase) {
    MemoryStore store;
    EXPECT_FALSE(store.exists("foo"));

    std::vector<uint8_t> data = {10, 20};
    ASSERT_TRUE(store.set("foo", data).has_value());
    EXPECT_TRUE(store.exists("foo"));

    auto er = store.erase("foo");
    ASSERT_TRUE(er.has_value());
    EXPECT_FALSE(store.exists("foo"));
}

TEST(MemoryStore, EraseNonexistent) {
    MemoryStore store;
    // Erasing a key that doesn't exist should succeed silently
    auto er = store.erase("nonexistent");
    ASSERT_TRUE(er.has_value());
}

TEST(MemoryStore, GetNonexistentReturnsError) {
    MemoryStore store;
    auto got = store.get("no_such_key");
    ASSERT_FALSE(got.has_value());
    EXPECT_FALSE(got.error().empty());
}

TEST(MemoryStore, ListPrefix) {
    MemoryStore store;
    std::vector<uint8_t> d = {0};
    store.set("data/a", d);
    store.set("data/b", d);
    store.set("meta/c", d);

    auto keys = store.list_prefix("data/");
    EXPECT_EQ(keys.size(), 2u);
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys[0], "data/a");
    EXPECT_EQ(keys[1], "data/b");
}

TEST(MemoryStore, ListPrefixEmpty) {
    MemoryStore store;
    std::vector<uint8_t> d = {1};
    store.set("x", d);
    store.set("y/z", d);
    // Empty prefix returns all keys
    auto keys = store.list_prefix("");
    EXPECT_EQ(keys.size(), 2u);
}

TEST(MemoryStore, ListDir) {
    MemoryStore store;
    std::vector<uint8_t> d = {0};
    store.set("root/file.txt", d);
    store.set("root/sub/nested", d);
    store.set("root/sub2/deep/item", d);

    auto [keys, prefixes] = store.list_dir("root");
    EXPECT_EQ(keys.size(), 1u);
    if (!keys.empty()) EXPECT_EQ(keys[0], "file.txt");

    std::sort(prefixes.begin(), prefixes.end());
    EXPECT_EQ(prefixes.size(), 2u);
    if (prefixes.size() == 2u) {
        EXPECT_EQ(prefixes[0], "sub");
        EXPECT_EQ(prefixes[1], "sub2");
    }
}

TEST(MemoryStore, GetPartial) {
    MemoryStore store;
    std::vector<uint8_t> data = {10, 20, 30, 40, 50, 60};
    store.set("blob", data);

    auto partial = store.get_partial("blob", 2, 3);
    ASSERT_TRUE(partial.has_value());
    EXPECT_EQ(partial->size(), 3u);
    EXPECT_EQ((*partial)[0], 30);
    EXPECT_EQ((*partial)[1], 40);
    EXPECT_EQ((*partial)[2], 50);
}

TEST(MemoryStore, GetPartialBeyondEnd) {
    MemoryStore store;
    std::vector<uint8_t> data = {1, 2, 3};
    store.set("small", data);

    auto partial = store.get_partial("small", 1, 100);
    ASSERT_TRUE(partial.has_value());
    EXPECT_EQ(partial->size(), 2u);
    EXPECT_EQ((*partial)[0], 2);
    EXPECT_EQ((*partial)[1], 3);
}

TEST(MemoryStore, GetPartialNonexistent) {
    MemoryStore store;
    auto partial = store.get_partial("missing", 0, 10);
    ASSERT_FALSE(partial.has_value());
}

// =============================================================================
// 2. FilesystemStore tests
// =============================================================================

TEST(FilesystemStore, OpenCreatesDirectory) {
    auto subdir = std::filesystem::temp_directory_path() /
                  ("zarr_test_fsopen_" + std::to_string(std::random_device{}()));
    EXPECT_FALSE(std::filesystem::exists(subdir));

    auto store = FilesystemStore::open(subdir);
    ASSERT_TRUE(store.has_value());
    EXPECT_TRUE(std::filesystem::exists(subdir));
    EXPECT_TRUE(std::filesystem::is_directory(subdir));

    std::error_code ec;
    std::filesystem::remove_all(subdir, ec);
}

TEST(FilesystemStore, SetGetRoundTrip) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fsrt_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());

    std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF};
    auto r = (*store)->set("test/key", data);
    ASSERT_TRUE(r.has_value());

    auto got = (*store)->get("test/key");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, data);

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FilesystemStore, ExistsAndErase) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fsee_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());

    std::vector<uint8_t> data = {42};
    (*store)->set("item", data);
    EXPECT_TRUE((*store)->exists("item"));

    (*store)->erase("item");
    EXPECT_FALSE((*store)->exists("item"));

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FilesystemStore, GetNonexistentReturnsError) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fsne_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());

    auto got = (*store)->get("no_such_file");
    ASSERT_FALSE(got.has_value());

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FilesystemStore, ListPrefix) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fslp_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());
    std::vector<uint8_t> d = {1};
    (*store)->set("data/a.bin", d);
    (*store)->set("data/b.bin", d);
    (*store)->set("other/c.bin", d);

    auto keys = (*store)->list_prefix("data");
    EXPECT_EQ(keys.size(), 2u);

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FilesystemStore, ListDir) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fsld_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());
    std::vector<uint8_t> d = {1};
    (*store)->set("root/file.txt", d);
    (*store)->set("root/sub/nested.txt", d);

    auto [keys, prefixes] = (*store)->list_dir("root");
    EXPECT_EQ(keys.size(), 1u);
    EXPECT_EQ(prefixes.size(), 1u);

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FilesystemStore, GetPartial) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fsgp_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());

    std::vector<uint8_t> data = {10, 20, 30, 40, 50};
    (*store)->set("blob", data);

    auto partial = (*store)->get_partial("blob", 1, 3);
    ASSERT_TRUE(partial.has_value());
    EXPECT_EQ(partial->size(), 3u);
    EXPECT_EQ((*partial)[0], 20);
    EXPECT_EQ((*partial)[1], 30);
    EXPECT_EQ((*partial)[2], 40);

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FilesystemStore, RootAccessor) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_test_fsra_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto store = FilesystemStore::open(tmp_dir);
    ASSERT_TRUE(store.has_value());
    // root() should return a path that resolves to the same location
    auto root = (*store)->root();
    EXPECT_TRUE(std::filesystem::equivalent(root, tmp_dir));

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

// =============================================================================
// 3. CodecConfig factory tests
// =============================================================================

TEST(CodecConfigFactory, Blosc) {
    auto c = CodecConfig::blosc("zstd", 3, 2, 4, 0);
    EXPECT_EQ(c.id, CodecId::Blosc);
    EXPECT_EQ(c.blosc_cname, "zstd");
    EXPECT_EQ(c.blosc_clevel, 3);
    EXPECT_EQ(c.blosc_shuffle, 2);
    EXPECT_EQ(c.blosc_typesize, 4);
    EXPECT_EQ(c.blosc_blocksize, 0);
}

TEST(CodecConfigFactory, BloscDefaults) {
    auto c = CodecConfig::blosc();
    EXPECT_EQ(c.id, CodecId::Blosc);
    EXPECT_EQ(c.blosc_cname, "lz4");
    EXPECT_EQ(c.blosc_clevel, 5);
    EXPECT_EQ(c.blosc_shuffle, 1);
}

TEST(CodecConfigFactory, Gzip) {
    auto c = CodecConfig::gzip(7);
    EXPECT_EQ(c.id, CodecId::Gzip);
    EXPECT_EQ(c.level, 7);
}

TEST(CodecConfigFactory, Zstd) {
    auto c = CodecConfig::zstd(10);
    EXPECT_EQ(c.id, CodecId::Zstd);
    EXPECT_EQ(c.level, 10);
}

TEST(CodecConfigFactory, Lz4) {
    auto c = CodecConfig::lz4(2);
    EXPECT_EQ(c.id, CodecId::Lz4);
    EXPECT_EQ(c.acceleration, 2);
}

TEST(CodecConfigFactory, Bz2) {
    auto c = CodecConfig::bz2(9);
    EXPECT_EQ(c.id, CodecId::Bz2);
    EXPECT_EQ(c.level, 9);
}

TEST(CodecConfigFactory, Bytes) {
    auto c = CodecConfig::bytes("big");
    EXPECT_EQ(c.id, CodecId::Bytes);
    EXPECT_EQ(c.endian, "big");
}

TEST(CodecConfigFactory, Transpose) {
    auto c = CodecConfig::transpose({2, 0, 1});
    EXPECT_EQ(c.id, CodecId::Transpose);
    ASSERT_EQ(c.transpose_order.size(), 3u);
    EXPECT_EQ(c.transpose_order[0], 2u);
    EXPECT_EQ(c.transpose_order[1], 0u);
    EXPECT_EQ(c.transpose_order[2], 1u);
}

TEST(CodecConfigFactory, Crc32c) {
    auto c = CodecConfig::crc32c();
    EXPECT_EQ(c.id, CodecId::Crc32c);
}

// =============================================================================
// 4. ZarrArray V2 tests
// =============================================================================

TEST(ZarrArrayV2, CreateAndMetadata) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {100, 200};
    opts.chunks = {10, 20};
    opts.dtype = Dtype::float32;
    opts.order = Order::C;
    opts.fill_value = 0.0;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(store, "myarr", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->version(), ZarrVersion::v2);
    EXPECT_EQ(arr->dtype(), Dtype::float32);
    EXPECT_EQ(arr->ndim(), 2u);

    auto shape = arr->shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 100u);
    EXPECT_EQ(shape[1], 200u);

    auto chunks = arr->chunks();
    ASSERT_EQ(chunks.size(), 2u);
    EXPECT_EQ(chunks[0], 10u);
    EXPECT_EQ(chunks[1], 20u);

    auto& meta = arr->metadata();
    EXPECT_EQ(meta.order(), Order::C);
    EXPECT_NEAR(meta.fill_value(), 0.0, 1e-12);
}

TEST(ZarrArrayV2, WriteReadChunk) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(store, "arr", opts);
    ASSERT_TRUE(arr.has_value());

    // Prepare a 4x4 float32 chunk (16 floats = 64 bytes)
    std::vector<float> chunk_data(16);
    std::iota(chunk_data.begin(), chunk_data.end(), 1.0f);
    std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(chunk_data.data()),
                                   chunk_data.size() * sizeof(float));

    std::vector<std::size_t> idx = {0, 1};
    auto wr = arr->write_chunk(idx, bytes);
    ASSERT_TRUE(wr.has_value());

    auto rd = arr->read_chunk(idx);
    ASSERT_TRUE(rd.has_value());
    ASSERT_EQ(rd->size(), 16u * sizeof(float));

    auto* out = reinterpret_cast<const float*>(rd->data());
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i + 1));
    }
}

TEST(ZarrArrayV2, WriteReadRegion) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(store, "reg", opts);
    ASSERT_TRUE(arr.has_value());

    // Write a 3x3 region starting at (1,2)
    std::vector<int32_t> region(9);
    for (int i = 0; i < 9; ++i) region[i] = 100 + i;

    std::vector<std::size_t> offset = {1, 2};
    std::vector<std::size_t> rshape = {3, 3};
    auto wr = arr->write_region(offset, rshape, region.data());
    ASSERT_TRUE(wr.has_value());

    // Read back
    std::vector<int32_t> out(9, 0);
    auto rd = arr->read_region(offset, rshape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(out[i], 100 + i);
    }
}

TEST(ZarrArrayV2, TensorRoundTrip) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4, 6};
    opts.chunks = {4, 6};
    opts.dtype = Dtype::float64;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(store, "tens", opts);
    ASSERT_TRUE(arr.has_value());

    // Create tensor with known data
    auto t = vc::zeros<double>({4, 6});
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            t(r, c) = static_cast<double>(r * 6 + c);

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {4, 6};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto t2 = vc::zeros<double>({4, 6});
    auto rd = arr->read_region(off, shape, t2.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_EQ(t2.shape().size(), 2u);
    EXPECT_EQ(t2.shape()[0], 4u);
    EXPECT_EQ(t2.shape()[1], 6u);

    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            EXPECT_NEAR(t2(r, c), static_cast<double>(r * 6 + c), 1e-12);
}

TEST(ZarrArrayV2, Resize) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 10};
    opts.chunks = {5, 5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(store, "resizable", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->shape()[0], 10u);
    EXPECT_EQ(arr->shape()[1], 10u);

    std::vector<std::size_t> new_shape = {20, 15};
    auto r = arr->resize(new_shape);
    ASSERT_TRUE(r.has_value());

    EXPECT_EQ(arr->shape()[0], 20u);
    EXPECT_EQ(arr->shape()[1], 15u);
}

TEST(ZarrArrayV2, OpenAfterCreate) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {16, 16};
    opts.chunks = {8, 8};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    auto arr = ZarrArray::create(store, "opentest", opts);
    ASSERT_TRUE(arr.has_value());

    // Write some data
    std::vector<int32_t> data(16 * 16);
    std::iota(data.begin(), data.end(), 0);
    std::vector<std::size_t> offset = {0, 0};
    std::vector<std::size_t> shape = {16, 16};
    arr->write_region(offset, shape, data.data());

    // Re-open and read
    auto arr2 = ZarrArray::open(store, "opentest");
    ASSERT_TRUE(arr2.has_value());
    EXPECT_EQ(arr2->version(), ZarrVersion::v2);
    EXPECT_EQ(arr2->dtype(), Dtype::int32);

    std::vector<int32_t> out(16 * 16, -1);
    auto rd = arr2->read_region(offset, shape, out.data());
    ASSERT_TRUE(rd.has_value());
    for (int i = 0; i < 256; ++i) {
        EXPECT_EQ(out[i], i);
    }
}

// =============================================================================
// 5. ZarrArray V3 tests
// =============================================================================

TEST(ZarrArrayV3, CreateAndMetadata) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {64, 128};
    opts.chunks = {16, 32};
    opts.dtype = Dtype::float64;
    opts.order = Order::C;
    opts.fill_value = -1.0;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3arr", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->version(), ZarrVersion::v3);
    EXPECT_EQ(arr->dtype(), Dtype::float64);
    EXPECT_EQ(arr->ndim(), 2u);
    EXPECT_EQ(arr->shape()[0], 64u);
    EXPECT_EQ(arr->shape()[1], 128u);
    EXPECT_EQ(arr->chunks()[0], 16u);
    EXPECT_EQ(arr->chunks()[1], 32u);
    EXPECT_NEAR(arr->metadata().fill_value(), -1.0, 1e-12);
}

TEST(ZarrArrayV3, WriteReadChunk) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3chunk", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<float> chunk(16);
    for (int i = 0; i < 16; ++i) chunk[i] = static_cast<float>(i) * 0.5f;
    std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(chunk.data()),
                                   chunk.size() * sizeof(float));

    std::vector<std::size_t> idx = {1, 0};
    auto wr = arr->write_chunk(idx, bytes);
    ASSERT_TRUE(wr.has_value());

    auto rd = arr->read_chunk(idx);
    ASSERT_TRUE(rd.has_value());
    ASSERT_EQ(rd->size(), 16u * sizeof(float));

    auto* out = reinterpret_cast<const float*>(rd->data());
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) * 0.5f);
    }
}

TEST(ZarrArrayV3, WriteReadRegion) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {12, 12};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3reg", opts);
    ASSERT_TRUE(arr.has_value());

    // Write a 5x7 region at (2, 3) -- spans multiple chunks
    std::vector<float> region(5 * 7);
    for (int i = 0; i < 35; ++i) region[i] = static_cast<float>(i + 10);

    std::vector<std::size_t> offset = {2, 3};
    std::vector<std::size_t> rshape = {5, 7};
    auto wr = arr->write_region(offset, rshape, region.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<float> out(35, -999.0f);
    auto rd = arr->read_region(offset, rshape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 35; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i + 10));
    }
}

TEST(ZarrArrayV3, TensorRoundTrip) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {6, 8};
    opts.chunks = {3, 4};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3tens", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::zeros<int32_t>({6, 8});
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < 8; ++c)
            t(r, c) = static_cast<int32_t>(r * 10 + c);

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {6, 8};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto t2 = vc::zeros<int32_t>({6, 8});
    auto rd = arr->read_region(off, shape, t2.data());
    ASSERT_TRUE(rd.has_value());
    ASSERT_EQ(t2.shape().size(), 2u);
    EXPECT_EQ(t2.shape()[0], 6u);
    EXPECT_EQ(t2.shape()[1], 8u);

    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < 8; ++c)
            EXPECT_EQ(t2(r, c), static_cast<int32_t>(r * 10 + c));
}

TEST(ZarrArrayV3, ReadTensorSubregion) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 10};
    opts.chunks = {5, 5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3sub", opts);
    ASSERT_TRUE(arr.has_value());

    auto full = vc::zeros<float>({10, 10});
    for (std::size_t r = 0; r < 10; ++r)
        for (std::size_t c = 0; c < 10; ++c)
            full(r, c) = static_cast<float>(r * 10 + c);

    std::vector<std::size_t> off0 = {0, 0};
    std::vector<std::size_t> full_shape = {10, 10};
    arr->write_region(off0, full_shape, full.data());

    // Read a 3x4 subregion at (2, 3)
    std::vector<std::size_t> off = {2, 3};
    std::vector<std::size_t> sh = {3, 4};
    auto sub = vc::zeros<float>({3, 4});
    auto rd = arr->read_region(off, sh, sub.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_EQ(sub.shape()[0], 3u);
    EXPECT_EQ(sub.shape()[1], 4u);

    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            EXPECT_FLOAT_EQ(sub(r, c), static_cast<float>((r + 2) * 10 + (c + 3)));
}

TEST(ZarrArrayV3, OpenAfterCreate) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {20, 20};
    opts.chunks = {10, 10};
    opts.dtype = Dtype::float64;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3open", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::Tensor<double>({20, 20}, vc::Layout::RowMajor, 1.0);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {20, 20};
    arr->write_region(off, shape, t.data());

    auto arr2 = ZarrArray::open(store, "v3open");
    ASSERT_TRUE(arr2.has_value());
    EXPECT_EQ(arr2->version(), ZarrVersion::v3);
    EXPECT_EQ(arr2->dtype(), Dtype::float64);
    EXPECT_EQ(arr2->shape()[0], 20u);

    auto t2 = vc::zeros<double>({20, 20});
    auto rd = arr2->read_region(off, shape, t2.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(t, t2));
}

TEST(ZarrArrayV3, Resize) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "v3resize", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<std::size_t> new_shape = {16, 32};
    auto r = arr->resize(new_shape);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(arr->shape()[0], 16u);
    EXPECT_EQ(arr->shape()[1], 32u);
}

// =============================================================================
// 6. ZarrGroup tests
// =============================================================================

TEST(ZarrGroupV3, CreateAndOpen) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    auto grp2 = ZarrGroup::open(store, "");
    ASSERT_TRUE(grp2.has_value());
}

TEST(ZarrGroupV2, CreateAndOpen) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "root", ZarrVersion::v2);
    ASSERT_TRUE(grp.has_value());

    auto grp2 = ZarrGroup::open(store, "root");
    ASSERT_TRUE(grp2.has_value());
}

TEST(ZarrGroupV3, CreateAndOpenChildArrays) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    CreateOptions opts;
    opts.shape = {10, 10};
    opts.chunks = {5, 5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = grp->create_array("data", opts);
    ASSERT_TRUE(arr.has_value());

    auto arr2 = grp->open_array("data");
    ASSERT_TRUE(arr2.has_value());
    EXPECT_EQ(arr2->shape()[0], 10u);
    EXPECT_EQ(arr2->dtype(), Dtype::float32);
}

TEST(ZarrGroupV3, CreateAndOpenChildGroups) {
    MemoryStore store;
    auto root = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(root.has_value());

    auto child = root->create_group("subgroup");
    ASSERT_TRUE(child.has_value());

    auto child2 = root->open_group("subgroup");
    ASSERT_TRUE(child2.has_value());
}

TEST(ZarrGroupV3, ListArraysAndGroups) {
    MemoryStore store;
    auto root = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(root.has_value());

    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    root->create_array("arr1", opts);
    root->create_array("arr2", opts);
    root->create_group("grp1");
    root->create_group("grp2");
    root->create_group("grp3");

    auto arrays = root->list_arrays();
    std::sort(arrays.begin(), arrays.end());
    ASSERT_EQ(arrays.size(), 2u);
    EXPECT_EQ(arrays[0], "arr1");
    EXPECT_EQ(arrays[1], "arr2");

    auto groups = root->list_groups();
    std::sort(groups.begin(), groups.end());
    ASSERT_EQ(groups.size(), 3u);
    EXPECT_EQ(groups[0], "grp1");
    EXPECT_EQ(groups[1], "grp2");
    EXPECT_EQ(groups[2], "grp3");

    auto [arrs, grps] = root->list_children();
    EXPECT_EQ(arrs.size(), 2u);
    EXPECT_EQ(grps.size(), 3u);
}

TEST(ZarrGroupV3, ContainsArrayAndGroup) {
    MemoryStore store;
    auto root = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(root.has_value());

    CreateOptions opts;
    opts.shape = {5};
    opts.chunks = {5};
    opts.version = ZarrVersion::v3;

    root->create_array("myarray", opts);
    root->create_group("mygroup");

    EXPECT_TRUE(root->contains_array("myarray"));
    EXPECT_FALSE(root->contains_array("mygroup"));
    EXPECT_FALSE(root->contains_array("nonexistent"));

    EXPECT_TRUE(root->contains_group("mygroup"));
    EXPECT_FALSE(root->contains_group("myarray"));
    EXPECT_FALSE(root->contains_group("nonexistent"));
}

TEST(ZarrGroupV3, Attributes) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    auto attrs = grp->attributes();
    ASSERT_TRUE(attrs.has_value());
    EXPECT_EQ(*attrs, "{}");

    auto r = grp->set_attributes(R"({"description":"test group","version":42})");
    ASSERT_TRUE(r.has_value());

    auto attrs2 = grp->attributes();
    ASSERT_TRUE(attrs2.has_value());
    // Verify attributes contain expected content
    EXPECT_NE(attrs2->find("description"), std::string::npos);
    EXPECT_NE(attrs2->find("test group"), std::string::npos);
    EXPECT_NE(attrs2->find("version"), std::string::npos);
}

TEST(ZarrGroupV2, ListArraysAndGroups) {
    MemoryStore store;
    auto root = ZarrGroup::create(store, "root", ZarrVersion::v2);
    ASSERT_TRUE(root.has_value());

    CreateOptions opts;
    opts.shape = {8};
    opts.chunks = {4};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;

    root->create_array("data", opts);
    root->create_group("meta");

    auto arrays = root->list_arrays();
    ASSERT_EQ(arrays.size(), 1u);
    EXPECT_EQ(arrays[0], "data");

    auto groups = root->list_groups();
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0], "meta");
}

TEST(ZarrGroupV2, Attributes) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "root", ZarrVersion::v2);
    ASSERT_TRUE(grp.has_value());

    auto attrs = grp->attributes();
    ASSERT_TRUE(attrs.has_value());
    EXPECT_EQ(*attrs, "{}");

    grp->set_attributes(R"({"key":"value"})");
    auto attrs2 = grp->attributes();
    ASSERT_TRUE(attrs2.has_value());
    EXPECT_NE(attrs2->find("key"), std::string::npos);
    EXPECT_NE(attrs2->find("value"), std::string::npos);
}

// =============================================================================
// 7. Codec tests
// =============================================================================

TEST(Codecs, BloscCompression) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {32, 32};
    opts.chunks = {16, 16};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::blosc("lz4", 5, 1, 4, 0)};

    auto arr = ZarrArray::create(store, "blosc_arr", opts);
    ASSERT_TRUE(arr.has_value());

    // Write known data via region
    std::vector<float> data(32 * 32);
    for (int i = 0; i < 32 * 32; ++i) data[i] = static_cast<float>(i) * 1.5f;

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {32, 32};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<float> out(32 * 32, -1.0f);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 32 * 32; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) * 1.5f);
    }
}

TEST(Codecs, GzipCompression) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {20, 20};
    opts.chunks = {10, 10};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::gzip(5)};

    auto arr = ZarrArray::create(store, "gzip_arr", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<int32_t> data(20 * 20);
    std::iota(data.begin(), data.end(), -200);

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {20, 20};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<int32_t> out(20 * 20, 0);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 400; ++i) {
        EXPECT_EQ(out[i], -200 + i);
    }
}

TEST(Codecs, ZstdCompression) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {16, 16};
    opts.chunks = {8, 8};
    opts.dtype = Dtype::float64;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::zstd(3)};

    auto arr = ZarrArray::create(store, "zstd_arr", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<double> data(16 * 16);
    for (int i = 0; i < 256; ++i) data[i] = static_cast<double>(i) * 0.01;

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {16, 16};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<double> out(256, -1.0);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(out[i], static_cast<double>(i) * 0.01, 1e-12);
    }
}

TEST(Codecs, BytesOnlyNoCompression) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes()};

    auto arr = ZarrArray::create(store, "bytes_arr", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<uint8_t> data(64);
    std::iota(data.begin(), data.end(), static_cast<uint8_t>(0));

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {8, 8};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<uint8_t> out(64, 255);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(out[i], static_cast<uint8_t>(i));
    }
}

TEST(Codecs, BloscV2Compression) {
    // Test v2-style array with blosc compression
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {24, 24};
    opts.chunks = {12, 12};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v2;
    opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;
    opts.codecs = {CodecConfig::blosc("lz4", 5, 1, 4)};

    auto arr = ZarrArray::create(store, "v2blosc", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::zeros<float>({24, 24});
    for (std::size_t r = 0; r < 24; ++r)
        for (std::size_t c = 0; c < 24; ++c)
            t(r, c) = static_cast<float>(r + c);

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {24, 24};
    arr->write_region(off, shape, t.data());
    auto t2 = vc::zeros<float>({24, 24});
    auto rd = arr->read_region(off, shape, t2.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(t, t2));
}

// =============================================================================
// 8. Free function tests
// =============================================================================

TEST(FreeFunction, SaveLoadFloat32) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_ff_f32_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto t = vc::zeros<float>({5, 10});
    for (std::size_t r = 0; r < 5; ++r)
        for (std::size_t c = 0; c < 10; ++c)
            t(r, c) = static_cast<float>(r * 10 + c) * 0.1f;

    auto path = tmp_dir / "float32_array";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {5, 10};
        opts.chunks = {5, 10};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
        std::vector<std::size_t> off = {0, 0};
        std::vector<std::size_t> shape = {5, 10};
        auto wr = arr->write_region(off, shape, t.data());
        ASSERT_TRUE(wr.has_value());
    }

    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        auto arr2 = ZarrArray::open(store, "");
        ASSERT_TRUE(arr2.has_value());
        EXPECT_EQ(arr2->dtype(), Dtype::float32);
        EXPECT_EQ(arr2->shape().size(), 2u);
        EXPECT_EQ(arr2->shape()[0], 5u);
        EXPECT_EQ(arr2->shape()[1], 10u);
        auto t2 = vc::zeros<float>({5, 10});
        std::vector<std::size_t> off = {0, 0};
        std::vector<std::size_t> shape = {5, 10};
        auto rd = arr2->read_region(off, shape, t2.data());
        ASSERT_TRUE(rd.has_value());
        EXPECT_TRUE(allclose(t, t2));
    }

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FreeFunction, SaveLoadFloat64) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_ff_f64_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto t = vc::zeros<double>({3, 7});
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 7; ++c)
            t(r, c) = static_cast<double>(r * 7 + c) * 3.14;

    auto path = tmp_dir / "float64_array";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {3, 7};
        opts.chunks = {3, 7};
        opts.dtype = Dtype::float64;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
        std::vector<std::size_t> off = {0, 0};
        std::vector<std::size_t> shape = {3, 7};
        arr->write_region(off, shape, t.data());
    }

    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        auto arr2 = ZarrArray::open(store, "");
        ASSERT_TRUE(arr2.has_value());
        EXPECT_EQ(arr2->dtype(), Dtype::float64);
        auto t2 = vc::zeros<double>({3, 7});
        std::vector<std::size_t> off = {0, 0};
        std::vector<std::size_t> shape = {3, 7};
        arr2->read_region(off, shape, t2.data());
        EXPECT_TRUE(allclose(t, t2));
    }

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FreeFunction, SaveLoadInt32) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_ff_i32_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    auto t = vc::zeros<int32_t>({8, 4});
    for (std::size_t r = 0; r < 8; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            t(r, c) = static_cast<int32_t>(r * 4 + c) - 16;

    auto path = tmp_dir / "int32_array";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {8, 4};
        opts.chunks = {8, 4};
        opts.dtype = Dtype::int32;
        opts.version = ZarrVersion::v2;
        opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
        std::vector<std::size_t> off = {0, 0};
        std::vector<std::size_t> shape = {8, 4};
        arr->write_region(off, shape, t.data());
    }

    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        auto arr2 = ZarrArray::open(store, "");
        ASSERT_TRUE(arr2.has_value());
        EXPECT_EQ(arr2->dtype(), Dtype::int32);
        auto t2 = vc::zeros<int32_t>({8, 4});
        std::vector<std::size_t> off = {0, 0};
        std::vector<std::size_t> shape = {8, 4};
        arr2->read_region(off, shape, t2.data());
        // Exact match for integers
        for (std::size_t i = 0; i < t.size(); i++)
            EXPECT_EQ(t.data()[i], t2.data()[i]);
    }

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FreeFunction, DetectZarrVersion) {
    auto tmp_dir = std::filesystem::temp_directory_path() /
                   ("zarr_ff_ver_" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(tmp_dir);

    // Create a V3 array
    {
        auto path = tmp_dir / "v3_detect";
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {4, 4};
        opts.chunks = {4, 4};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());

        auto ver = detect_zarr_version(store, "");
        ASSERT_TRUE(ver.has_value());
        EXPECT_EQ(*ver, ZarrVersion::v3);
    }

    // Create a V2 array
    {
        auto path = tmp_dir / "v2_detect";
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {4, 4};
        opts.chunks = {4, 4};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v2;
        opts.chunk_key_encoding = ChunkKeyEncoding::DotSeparated;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());

        auto ver = detect_zarr_version(store, "");
        ASSERT_TRUE(ver.has_value());
        EXPECT_EQ(*ver, ZarrVersion::v2);
    }

    std::error_code ec;
    std::filesystem::remove_all(tmp_dir, ec);
}

TEST(FreeFunction, DetectVersionNoMetadata) {
    MemoryStore store;
    auto ver = detect_zarr_version(store, "");
    ASSERT_FALSE(ver.has_value());
}

// =============================================================================
// 9. Edge case tests
// =============================================================================

TEST(EdgeCases, OneDimensionalArray) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {100};
    opts.chunks = {25};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "1d", opts);
    ASSERT_TRUE(arr.has_value());
    EXPECT_EQ(arr->ndim(), 1u);

    std::vector<float> data(100);
    std::iota(data.begin(), data.end(), 0.0f);

    std::vector<std::size_t> off = {0};
    std::vector<std::size_t> shape = {100};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<float> out(100, -1.0f);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i));
    }
}

TEST(EdgeCases, HighDimensional4D) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {2, 3, 4, 5};
    opts.chunks = {2, 3, 4, 5};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "4d", opts);
    ASSERT_TRUE(arr.has_value());
    EXPECT_EQ(arr->ndim(), 4u);

    const std::size_t total = 2 * 3 * 4 * 5;
    std::vector<int32_t> data(total);
    std::iota(data.begin(), data.end(), 0);

    std::vector<std::size_t> off = {0, 0, 0, 0};
    std::vector<std::size_t> shape = {2, 3, 4, 5};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<int32_t> out(total, -1);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_EQ(out[i], static_cast<int32_t>(i));
    }
}

TEST(EdgeCases, FillValueForMissingChunks) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.fill_value = 42.0;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "fill", opts);
    ASSERT_TRUE(arr.has_value());

    // Read a chunk that was never written -- should be fill_value
    std::vector<std::size_t> idx = {0, 0};
    auto chunk = arr->read_chunk(idx);
    ASSERT_TRUE(chunk.has_value());
    ASSERT_EQ(chunk->size(), 4u * 4u * sizeof(float));

    auto* fptr = reinterpret_cast<const float*>(chunk->data());
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(fptr[i], 42.0f);
    }
}

TEST(EdgeCases, FillValueReadRegion) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4, 4};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::int32;
    opts.fill_value = 99.0;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "fillreg", opts);
    ASSERT_TRUE(arr.has_value());

    // Read the whole array without writing -- should be fill value
    std::vector<int32_t> out(16, 0);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {4, 4};
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(out[i], 99);
    }
}

TEST(EdgeCases, RegionSpanningMultipleChunks) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {16, 16};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "multispan", opts);
    ASSERT_TRUE(arr.has_value());

    // Write a 10x10 region at offset (3,3) -- spans chunks in both dimensions
    const std::size_t rsize = 10;
    std::vector<float> data(rsize * rsize);
    for (std::size_t i = 0; i < rsize * rsize; ++i)
        data[i] = static_cast<float>(i) + 0.25f;

    std::vector<std::size_t> off = {3, 3};
    std::vector<std::size_t> shape = {rsize, rsize};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<float> out(rsize * rsize, -1.0f);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    for (std::size_t i = 0; i < rsize * rsize; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i) + 0.25f);
    }
}

TEST(EdgeCases, WriteTensorAtOffset) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {12, 12};
    opts.chunks = {4, 4};
    opts.dtype = Dtype::float32;
    opts.fill_value = 0.0;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "tensoff", opts);
    ASSERT_TRUE(arr.has_value());

    // Write a small 3x3 tensor at offset (2, 5)
    auto sub = vc::full<float>({3, 3}, 7.0f);
    std::vector<std::size_t> offset = {2, 5};
    std::vector<std::size_t> sub_shape = {3, 3};
    auto wr = arr->write_region(offset, sub_shape, sub.data());
    ASSERT_TRUE(wr.has_value());

    // Read back the subregion
    auto t2 = vc::zeros<float>({3, 3});
    auto rd = arr->read_region(offset, sub_shape, t2.data());
    ASSERT_TRUE(rd.has_value());

    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(t2(r, c), 7.0f);

    // Verify that area outside the written region is fill value (0)
    std::vector<std::size_t> off0 = {0, 0};
    std::vector<std::size_t> s1 = {2, 5};
    auto corner = vc::zeros<float>({2, 5});
    auto rd2 = arr->read_region(off0, s1, corner.data());
    ASSERT_TRUE(rd2.has_value());
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 5; ++c)
            EXPECT_FLOAT_EQ(corner(r, c), 0.0f);
}

TEST(EdgeCases, ArrayAttributes) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "attrtest", opts);
    ASSERT_TRUE(arr.has_value());

    auto r = arr->set_attributes(R"({"units":"meters","scale":2.5})");
    ASSERT_TRUE(r.has_value());

    auto attrs = arr->attributes();
    ASSERT_TRUE(attrs.has_value());
    EXPECT_NE(attrs->find("units"), std::string::npos);
    EXPECT_NE(attrs->find("meters"), std::string::npos);
    EXPECT_NE(attrs->find("scale"), std::string::npos);
}

TEST(EdgeCases, AutoChunkWhenChunksEmpty) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {50, 100};
    // chunks intentionally left empty -> auto-chunk
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "auto", opts);
    ASSERT_TRUE(arr.has_value());

    // Auto-chunking should clamp to shape (both <= 1024)
    EXPECT_EQ(arr->chunks()[0], 50u);
    EXPECT_EQ(arr->chunks()[1], 100u);
}

TEST(EdgeCases, CreateEmptyShapeFails) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto arr = ZarrArray::create(store, "empty", opts);
    ASSERT_FALSE(arr.has_value());
}

TEST(EdgeCases, DifferentIntegerDtypes) {
    // Test round-trip with various integer dtypes
    auto test_dtype = [](Dtype dt, std::size_t elem_size) {
        MemoryStore store;
        CreateOptions opts;
        opts.shape = {8};
        opts.chunks = {8};
        opts.dtype = dt;
        opts.version = ZarrVersion::v3;

        auto arr = ZarrArray::create(store, "dt_" + dtypeToV3String(dt), opts);
        ASSERT_TRUE(arr.has_value());
        EXPECT_EQ(arr->dtype(), dt);

        // Write raw chunk of zeros, read back
        std::vector<uint8_t> data(8 * elem_size, 0);
        // Write value 1 at first element
        if (elem_size == 1) data[0] = 1;
        else if (elem_size == 2) { int16_t v = 1; std::memcpy(data.data(), &v, 2); }
        else if (elem_size == 4) { int32_t v = 1; std::memcpy(data.data(), &v, 4); }
        else if (elem_size == 8) { int64_t v = 1; std::memcpy(data.data(), &v, 8); }

        std::vector<std::size_t> idx = {0};
        auto wr = arr->write_chunk(idx, data);
        ASSERT_TRUE(wr.has_value());

        auto rd = arr->read_chunk(idx);
        ASSERT_TRUE(rd.has_value());
        ASSERT_EQ(rd->size(), data.size());
        EXPECT_EQ((*rd)[0], data[0]);
    };

    test_dtype(Dtype::int8, 1);
    test_dtype(Dtype::int16, 2);
    test_dtype(Dtype::int32, 4);
    test_dtype(Dtype::int64, 8);
    test_dtype(Dtype::uint8, 1);
    test_dtype(Dtype::uint16, 2);
    test_dtype(Dtype::uint32, 4);
    test_dtype(Dtype::uint64, 8);
}

// -- Sharding tests -----------------------------------------------------------

TEST(Sharding, Basic2DSharding) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {16, 16};
    opts.chunks = {8, 8};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    opts.codecs = {
        CodecConfig::bytes(),
        CodecConfig::sharding({4, 4},
            {CodecConfig::bytes(), CodecConfig::blosc()},
            {CodecConfig::bytes(), CodecConfig::crc32c()})
    };
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // Write with known values
    auto t = vc::zeros<float>({16, 16});
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 16; ++j)
            t(i, j) = static_cast<float>(i * 16 + j);

    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {16, 16};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto rt = vc::zeros<float>({16, 16});
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(t, rt));
}

TEST(Sharding, NoInnerCompression) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {12, 12};
    opts.chunks = {6, 6};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v3;
    opts.codecs = {
        CodecConfig::bytes(),
        CodecConfig::sharding({3, 3},
            {CodecConfig::bytes()},
            {CodecConfig::bytes()})
    };
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::full<int32_t>({12, 12}, 42);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {12, 12};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto rt = vc::zeros<int32_t>({12, 12});
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    for (std::size_t i = 0; i < rt.size(); i++)
        EXPECT_EQ(t.data()[i], rt.data()[i]);
}

TEST(Sharding, Sharding1D) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {100};
    opts.chunks = {20};
    opts.dtype = Dtype::float64;
    opts.version = ZarrVersion::v3;
    opts.codecs = {
        CodecConfig::bytes(),
        CodecConfig::sharding({5},
            {CodecConfig::bytes(), CodecConfig::zstd(1)},
            {CodecConfig::bytes()})
    };
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::zeros<double>({100});
    for (std::size_t i = 0; i < 100; ++i) t.data()[i] = static_cast<double>(i);

    std::vector<std::size_t> off = {0};
    std::vector<std::size_t> shape = {100};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto rt = vc::zeros<double>({100});
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(t, rt));
}

TEST(Sharding, ShardingRegionIO) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {20, 20};
    opts.chunks = {10, 10};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    opts.codecs = {
        CodecConfig::bytes(),
        CodecConfig::sharding({5, 5},
            {CodecConfig::bytes(), CodecConfig::gzip(1)},
            {CodecConfig::bytes()})
    };
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // Write a subregion
    auto sub = vc::full<float>({5, 5}, 99.0f);
    std::vector<std::size_t> offset = {3, 3};
    std::vector<std::size_t> shape = {5, 5};
    auto wr = arr->write_region(offset, shape, sub.data());
    ASSERT_TRUE(wr.has_value());

    // Read back the subregion
    auto out = vc::zeros<float>({5, 5});
    auto rd = arr->read_region(offset, shape, out.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(sub, out));
}

TEST(Sharding, Sharding3D) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {8, 8, 8};
    opts.chunks = {4, 4, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    opts.codecs = {
        CodecConfig::bytes(),
        CodecConfig::sharding({2, 2, 2},
            {CodecConfig::bytes()},
            {CodecConfig::bytes()})
    };
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::zeros<float>({8, 8, 8});
    float val = 0.0f;
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t j = 0; j < 8; ++j)
            for (std::size_t k = 0; k < 8; ++k)
                t(i, j, k) = val++;

    std::vector<std::size_t> off = {0, 0, 0};
    std::vector<std::size_t> shape = {8, 8, 8};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto rt = vc::zeros<float>({8, 8, 8});
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(t, rt));
}

// -- Computed Properties tests ------------------------------------------------

TEST(ComputedProperties, SizeItemsizeNbytes) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 20};
    opts.chunks = {5, 10};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->size(), 200u);
    EXPECT_EQ(arr->itemsize(), 4u);
    EXPECT_EQ(arr->nbytes(), 800u);
}

TEST(ComputedProperties, Nchunks) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 20};
    opts.chunks = {5, 10};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // ceil(10/5) * ceil(20/10) = 2 * 2 = 4
    EXPECT_EQ(arr->nchunks(), 4u);
}

TEST(ComputedProperties, NchunksInitialized) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 10};
    opts.chunks = {5, 5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // No chunks written yet
    EXPECT_EQ(arr->nchunks_initialized(), 0u);

    // Write a tensor to fill the array
    auto t = vc::Tensor<float>({10, 10}, vc::Layout::RowMajor, 1.0f);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {10, 10};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    // All 4 chunks should be initialized now
    EXPECT_EQ(arr->nchunks_initialized(), 4u);
}

TEST(ComputedProperties, FillValueOrderPath) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10};
    opts.chunks = {5};
    opts.dtype = Dtype::float64;
    opts.fill_value = 42.0;
    opts.order = Order::F;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "myarray", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_NEAR(arr->fill_value(), 42.0, 1e-12);
    EXPECT_EQ(arr->order(), Order::F);
    EXPECT_EQ(arr->path(), "myarray");
    EXPECT_EQ(arr->name(), "myarray");
}

TEST(ComputedProperties, ReadOnly) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10};
    opts.chunks = {5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_FALSE(arr->read_only());
    arr->set_read_only(true);
    EXPECT_TRUE(arr->read_only());
}

TEST(ComputedProperties, Info) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 20};
    opts.chunks = {5, 10};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "data", opts);
    ASSERT_TRUE(arr.has_value());

    auto info = arr->info();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("float32"), std::string::npos);
    EXPECT_NE(info.find("10"), std::string::npos);
}

// -- Granular Attribute tests -------------------------------------------------

TEST(GranularAttrs, ArraySetGetDelete) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10};
    opts.chunks = {5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->num_attributes(), 0u);
    EXPECT_FALSE(arr->contains_attribute("foo"));

    // Set an attribute
    auto set_result = arr->set_attribute("foo", "42");
    ASSERT_TRUE(set_result.has_value());

    EXPECT_TRUE(arr->contains_attribute("foo"));
    EXPECT_EQ(arr->num_attributes(), 1u);

    auto val = arr->get_attribute("foo");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "42");

    // Set a string attribute
    set_result = arr->set_attribute("name", "\"hello\"");
    ASSERT_TRUE(set_result.has_value());
    EXPECT_EQ(arr->num_attributes(), 2u);

    // Keys
    auto keys = arr->attribute_keys();
    EXPECT_EQ(keys.size(), 2u);

    // Delete
    auto del = arr->delete_attribute("foo");
    ASSERT_TRUE(del.has_value());
    EXPECT_EQ(arr->num_attributes(), 1u);
    EXPECT_FALSE(arr->contains_attribute("foo"));
    EXPECT_TRUE(arr->contains_attribute("name"));
}

TEST(GranularAttrs, GroupSetGetDelete) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    EXPECT_EQ(grp->num_attributes(), 0u);

    auto set_result = grp->set_attribute("key", "[1,2,3]");
    ASSERT_TRUE(set_result.has_value());

    EXPECT_TRUE(grp->contains_attribute("key"));
    EXPECT_EQ(grp->num_attributes(), 1u);

    auto val = grp->get_attribute("key");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "[1,2,3]");

    auto del = grp->delete_attribute("key");
    ASSERT_TRUE(del.has_value());
    EXPECT_EQ(grp->num_attributes(), 0u);
}

// -- Append tests -------------------------------------------------------------

TEST(Append, AppendAxis0) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {5, 3};
    opts.chunks = {5, 3};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // Write initial data
    auto t1 = vc::Tensor<float>({5, 3}, vc::Layout::RowMajor, 1.0f);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {5, 3};
    auto wr = arr->write_region(off, shape, t1.data());
    ASSERT_TRUE(wr.has_value());

    // Append more rows
    auto t2 = vc::full<float>({3, 3}, 2.0f);
    std::vector<std::size_t> t2_shape = {3, 3};
    auto app = arr->append(t2.data(), t2_shape, 0);
    ASSERT_TRUE(app.has_value());

    // Shape should be (8, 3)
    auto s = arr->shape();
    EXPECT_EQ(s[0], 8u);
    EXPECT_EQ(s[1], 3u);

    // Read back the full array
    auto rt = vc::zeros<float>({8, 3});
    std::vector<std::size_t> full_off = {0, 0};
    std::vector<std::size_t> full_shape = {8, 3};
    auto rd = arr->read_region(full_off, full_shape, rt.data());
    ASSERT_TRUE(rd.has_value());

    // First 5 rows should be 1.0
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(rt(i, j), 1.0f);

    // Last 3 rows should be 2.0
    for (std::size_t i = 5; i < 8; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(rt(i, j), 2.0f);
}

// -- Open by path tests -------------------------------------------------------

TEST(OpenByPath, ArrayOpenByPath) {
    // Create a zarr array on disk, then reopen via path
    std::string path = "/tmp/zarr_test_open_by_path";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {10};
        opts.chunks = {5};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
        auto t = vc::Tensor<float>({10}, vc::Layout::RowMajor, 1.0f);
        std::vector<std::size_t> off = {0};
        std::vector<std::size_t> shape = {10};
        auto wr = arr->write_region(off, shape, t.data());
        ASSERT_TRUE(wr.has_value());
    }

    // Open by path
    auto arr = ZarrArray::open(std::filesystem::path(path));
    ASSERT_TRUE(arr.has_value());
    EXPECT_EQ(arr->size(), 10u);

    auto rt = vc::zeros<float>({10});
    std::vector<std::size_t> off = {0};
    std::vector<std::size_t> shape = {10};
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(OpenByPath, OpenZarrFreeFunction) {
    std::string path = "/tmp/zarr_test_open_zarr_func";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {6};
        opts.chunks = {3};
        opts.dtype = Dtype::int32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
    }

    auto arr = open_zarr(path, OpenMode::Read);
    ASSERT_TRUE(arr.has_value());
    EXPECT_TRUE(arr->read_only());
    EXPECT_EQ(arr->size(), 6u);
    EXPECT_EQ(arr->dtype(), Dtype::int32);

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(OpenByPath, OpenZarrGroupFreeFunction) {
    std::string path = "/tmp/zarr_test_open_zarr_group_func";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
        ASSERT_TRUE(grp.has_value());
    }

    auto grp = open_zarr_group(path, OpenMode::ReadWrite);
    ASSERT_TRUE(grp.has_value());
    EXPECT_FALSE(grp->read_only());

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// -- ZarrGroup navigation property tests --------------------------------------

TEST(GroupNav, PathNameVersionNchildren) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "root", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    EXPECT_EQ(grp->path(), "root");
    EXPECT_EQ(grp->name(), "root");
    EXPECT_EQ(grp->version(), ZarrVersion::v3);
    EXPECT_EQ(grp->nchildren(), 0u);

    // Create a child array
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto child_arr = grp->create_array("data", opts);
    ASSERT_TRUE(child_arr.has_value());

    // Create a child group
    auto child_grp = grp->create_group("subgroup");
    ASSERT_TRUE(child_grp.has_value());

    EXPECT_EQ(grp->nchildren(), 2u);
}

TEST(GroupNav, InfoAndReadOnly) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    EXPECT_FALSE(grp->read_only());
    grp->set_read_only(true);
    EXPECT_TRUE(grp->read_only());

    auto info = grp->info();
    EXPECT_FALSE(info.empty());
}

// -- Computed Properties -- additional tests -----------------------------------

TEST(ComputedProperties, StoreRef) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // store_ref should point back to same store
    const Store& ref = arr->store_ref();
    // Verify it's functional by checking that the metadata key exists
    EXPECT_TRUE(ref.exists("zarr.json"));
}

TEST(ComputedProperties, NameNestedPath) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto child = grp->create_array("deep/nested/array", opts);
    ASSERT_TRUE(child.has_value());
    // name() should return basename
    EXPECT_EQ(child->name(), "array");
    EXPECT_EQ(child->path(), "deep/nested/array");
}

TEST(ComputedProperties, EmptyPathName) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());
    // Empty path -> name should be "/" or ""
    EXPECT_TRUE(arr->path().empty() || arr->path() == "/");
}

TEST(ComputedProperties, NchunksNonEven) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {7, 13};
    opts.chunks = {3, 5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());
    // ceil(7/3) * ceil(13/5) = 3 * 3 = 9
    EXPECT_EQ(arr->nchunks(), 9u);
}

TEST(ComputedProperties, AllDtypeSizes) {
    MemoryStore store;
    auto check = [&](Dtype dt, std::size_t expected_size) {
        CreateOptions opts;
        opts.shape = {4};
        opts.chunks = {4};
        opts.dtype = dt;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "arr_" + dtypeToV3String(dt), opts);
        ASSERT_TRUE(arr.has_value());
        EXPECT_EQ(arr->itemsize(), expected_size);
        EXPECT_EQ(arr->nbytes(), 4u * expected_size);
    };
    check(Dtype::float32, 4);
    check(Dtype::float64, 8);
    check(Dtype::int8, 1);
    check(Dtype::int16, 2);
    check(Dtype::int32, 4);
    check(Dtype::int64, 8);
    check(Dtype::uint8, 1);
    check(Dtype::uint16, 2);
    check(Dtype::uint32, 4);
    check(Dtype::uint64, 8);
}

TEST(ComputedProperties, NchunksInitializedPartial) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 10};
    opts.chunks = {5, 5};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->nchunks_initialized(), 0u);

    // Write only one chunk's worth of data
    auto t = vc::Tensor<float>({5, 5}, vc::Layout::RowMajor, 1.0f);
    std::vector<std::size_t> offset = {0, 0};
    std::vector<std::size_t> shape = {5, 5};
    auto wr = arr->write_region(offset, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    EXPECT_EQ(arr->nchunks_initialized(), 1u);
}

TEST(ComputedProperties, V2Properties) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {10, 10};
    opts.chunks = {5, 5};
    opts.dtype = Dtype::int32;
    opts.fill_value = -1.0;
    opts.order = Order::F;
    opts.version = ZarrVersion::v2;
    auto arr = ZarrArray::create(store, "v2arr", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_EQ(arr->version(), ZarrVersion::v2);
    EXPECT_EQ(arr->size(), 100u);
    EXPECT_EQ(arr->itemsize(), 4u);
    EXPECT_EQ(arr->nchunks(), 4u);
    EXPECT_NEAR(arr->fill_value(), -1.0, 1e-12);
    EXPECT_EQ(arr->order(), Order::F);
    EXPECT_EQ(arr->path(), "v2arr");
}

// -- Granular Attributes -- additional tests -----------------------------------

TEST(GranularAttrs, OverwriteAttribute) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto r1 = arr->set_attribute("key", "\"first\"");
    ASSERT_TRUE(r1.has_value());
    auto r2 = arr->set_attribute("key", "\"second\"");
    ASSERT_TRUE(r2.has_value());

    EXPECT_EQ(arr->num_attributes(), 1u);
    auto val = arr->get_attribute("key");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "\"second\"");
}

TEST(GranularAttrs, ComplexJsonValues) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // Nested object
    auto r = arr->set_attribute("meta", R"({"x":1,"y":[2,3]})");
    ASSERT_TRUE(r.has_value());

    auto val = arr->get_attribute("meta");
    ASSERT_TRUE(val.has_value());
    // The JSON should round-trip (order may differ but content is preserved)
    EXPECT_NE(val->find("\"x\""), std::string::npos);
    EXPECT_NE(val->find("\"y\""), std::string::npos);

    // Boolean
    r = arr->set_attribute("flag", "true");
    ASSERT_TRUE(r.has_value());
    val = arr->get_attribute("flag");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "true");

    // Null
    r = arr->set_attribute("empty", "null");
    ASSERT_TRUE(r.has_value());
    val = arr->get_attribute("empty");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "null");
}

TEST(GranularAttrs, DeleteNonexistent) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto del = arr->delete_attribute("nonexistent");
    // Deleting a nonexistent key should either succeed silently or return error
    // Either way it shouldn't crash
    EXPECT_EQ(arr->num_attributes(), 0u);
}

TEST(GranularAttrs, GetNonexistent) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto val = arr->get_attribute("nonexistent");
    EXPECT_FALSE(val.has_value());
}

TEST(GranularAttrs, GroupAttributeKeys) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    EXPECT_TRUE(grp->attribute_keys().empty());

    auto r1 = grp->set_attribute("alpha", "1");
    ASSERT_TRUE(r1.has_value());
    auto r2 = grp->set_attribute("beta", "2");
    ASSERT_TRUE(r2.has_value());
    auto r3 = grp->set_attribute("gamma", "3");
    ASSERT_TRUE(r3.has_value());

    auto keys = grp->attribute_keys();
    EXPECT_EQ(keys.size(), 3u);
    // Check all keys present (order may vary)
    std::sort(keys.begin(), keys.end());
    EXPECT_EQ(keys[0], "alpha");
    EXPECT_EQ(keys[1], "beta");
    EXPECT_EQ(keys[2], "gamma");
}

TEST(GranularAttrs, AttrsWithBulkInterop) {
    // Verify granular and bulk attrs interop correctly
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // Set via bulk, read via granular
    auto bulk_set = arr->set_attributes(R"({"initial":"value"})");
    ASSERT_TRUE(bulk_set.has_value());

    EXPECT_TRUE(arr->contains_attribute("initial"));
    auto val = arr->get_attribute("initial");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "\"value\"");

    // Granular set, then bulk read
    auto r = arr->set_attribute("added", "42");
    ASSERT_TRUE(r.has_value());

    auto bulk = arr->attributes();
    ASSERT_TRUE(bulk.has_value());
    EXPECT_NE(bulk->find("\"initial\""), std::string::npos);
    EXPECT_NE(bulk->find("\"added\""), std::string::npos);
}

// -- Append -- additional tests -----------------------------------------------

TEST(Append, AppendAxis1) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {3, 4};
    opts.chunks = {3, 4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    auto t1 = vc::Tensor<float>({3, 4}, vc::Layout::RowMajor, 1.0f);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {3, 4};
    auto wr = arr->write_region(off, shape, t1.data());
    ASSERT_TRUE(wr.has_value());

    auto t2 = vc::full<float>({3, 2}, 5.0f);
    std::vector<std::size_t> t2_shape = {3, 2};
    auto app = arr->append(t2.data(), t2_shape, 1);
    ASSERT_TRUE(app.has_value());

    auto s = arr->shape();
    EXPECT_EQ(s[0], 3u);
    EXPECT_EQ(s[1], 6u);

    auto rt = vc::zeros<float>({3, 6});
    std::vector<std::size_t> full_off = {0, 0};
    std::vector<std::size_t> full_shape = {3, 6};
    auto rd = arr->read_region(full_off, full_shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    // First 4 cols: 1.0
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_FLOAT_EQ(rt(i, j), 1.0f);
    // Last 2 cols: 5.0
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 4; j < 6; ++j)
            EXPECT_FLOAT_EQ(rt(i, j), 5.0f);
}

TEST(Append, AppendMultipleTimes) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {0, 3};
    opts.chunks = {5, 3};
    opts.dtype = Dtype::int32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());
    EXPECT_EQ(arr->size(), 0u);

    // Append three batches
    for (int batch = 0; batch < 3; ++batch) {
        auto t = vc::full<int32_t>({2, 3}, static_cast<int32_t>(batch + 1));
        std::vector<std::size_t> t_shape = {2, 3};
        auto app = arr->append(t.data(), t_shape, 0);
        ASSERT_TRUE(app.has_value());
    }

    auto s = arr->shape();
    EXPECT_EQ(s[0], 6u);
    EXPECT_EQ(s[1], 3u);

    auto rt = vc::zeros<int32_t>({6, 3});
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> full_shape = {6, 3};
    auto rd = arr->read_region(off, full_shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    for (std::size_t i = 0; i < 6; ++i) {
        int expected = static_cast<int>(i / 2) + 1;
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_EQ(rt(i, j), expected);
    }
}

TEST(Append, AppendDimMismatch) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {5, 3};
    opts.chunks = {5, 3};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = ZarrArray::create(store, "", opts);
    ASSERT_TRUE(arr.has_value());

    // Wrong number of columns
    auto t = vc::Tensor<float>({2, 4}, vc::Layout::RowMajor, 1.0f);
    std::vector<std::size_t> t_shape = {2, 4};
    auto app = arr->append(t.data(), t_shape, 0);
    EXPECT_FALSE(app.has_value());
}

TEST(Append, AppendOnFilesystem) {
    std::string path = "/tmp/zarr_test_append_fs";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {4};
        opts.chunks = {4};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());

        auto t1 = vc::Tensor<float>({4}, vc::Layout::RowMajor, 1.0f);
        std::vector<std::size_t> off = {0};
        std::vector<std::size_t> shape = {4};
        auto wr = arr->write_region(off, shape, t1.data());
        ASSERT_TRUE(wr.has_value());

        auto t2 = vc::full<float>({3}, 2.0f);
        std::vector<std::size_t> t2_shape = {3};
        auto app = arr->append(t2.data(), t2_shape, 0);
        ASSERT_TRUE(app.has_value());
    }

    // Reopen and verify
    auto arr = open_zarr(path);
    ASSERT_TRUE(arr.has_value());
    auto s = arr->shape();
    EXPECT_EQ(s[0], 7u);

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// -- Open by path -- additional tests -----------------------------------------

TEST(OpenByPath, OpenZarrAllModes) {
    std::string path = "/tmp/zarr_test_all_modes";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {8};
        opts.chunks = {4};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
    }

    // Read mode
    {
        auto arr = open_zarr(path, OpenMode::Read);
        ASSERT_TRUE(arr.has_value());
        EXPECT_TRUE(arr->read_only());
    }
    // Write mode
    {
        auto arr = open_zarr(path, OpenMode::Write);
        ASSERT_TRUE(arr.has_value());
        EXPECT_FALSE(arr->read_only());
    }
    // ReadWrite mode
    {
        auto arr = open_zarr(path, OpenMode::ReadWrite);
        ASSERT_TRUE(arr.has_value());
        EXPECT_FALSE(arr->read_only());
    }
    // Append mode
    {
        auto arr = open_zarr(path, OpenMode::Append);
        ASSERT_TRUE(arr.has_value());
        EXPECT_FALSE(arr->read_only());
    }

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(OpenByPath, OpenZarrGroupAllModes) {
    std::string path = "/tmp/zarr_test_group_all_modes";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
        ASSERT_TRUE(grp.has_value());
    }

    auto grp_r = open_zarr_group(path, OpenMode::Read);
    ASSERT_TRUE(grp_r.has_value());
    EXPECT_TRUE(grp_r->read_only());

    auto grp_w = open_zarr_group(path, OpenMode::Write);
    ASSERT_TRUE(grp_w.has_value());
    EXPECT_FALSE(grp_w->read_only());

    auto grp_a = open_zarr_group(path, OpenMode::Append);
    ASSERT_TRUE(grp_a.has_value());
    EXPECT_FALSE(grp_a->read_only());

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(OpenByPath, GroupOpenByPath) {
    std::string path = "/tmp/zarr_test_group_open_path";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
        ASSERT_TRUE(grp.has_value());
        CreateOptions opts;
        opts.shape = {4};
        opts.chunks = {4};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = grp->create_array("data", opts);
        ASSERT_TRUE(arr.has_value());
    }

    auto grp = ZarrGroup::open(std::filesystem::path(path));
    ASSERT_TRUE(grp.has_value());
    EXPECT_EQ(grp->nchildren(), 1u);
    EXPECT_TRUE(grp->contains_array("data"));

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

TEST(OpenByPath, OpenNonexistentPath) {
    auto arr = open_zarr("/tmp/zarr_test_nonexistent_12345");
    EXPECT_FALSE(arr.has_value());

    auto grp = open_zarr_group("/tmp/zarr_test_nonexistent_12345");
    EXPECT_FALSE(grp.has_value());
}

TEST(OpenByPath, SetOwnedStorePersistence) {
    // Verify that set_owned_store keeps the store alive
    std::string path = "/tmp/zarr_test_owned_store_persist";
    {
        auto store_result = FilesystemStore::open(path);
        ASSERT_TRUE(store_result.has_value());
        auto& store = **store_result;
        CreateOptions opts;
        opts.shape = {10};
        opts.chunks = {5};
        opts.dtype = Dtype::float32;
        opts.version = ZarrVersion::v3;
        auto arr = ZarrArray::create(store, "", opts);
        ASSERT_TRUE(arr.has_value());
        auto t = vc::Tensor<float>({10}, vc::Layout::RowMajor, 1.0f);
        std::vector<std::size_t> off = {0};
        std::vector<std::size_t> shape = {10};
        auto wr = arr->write_region(off, shape, t.data());
        ASSERT_TRUE(wr.has_value());
    }

    // open_zarr uses set_owned_store internally
    auto arr = open_zarr(path);
    ASSERT_TRUE(arr.has_value());

    // The store should remain valid even though FilesystemStore::open was scoped
    auto rt = vc::zeros<float>({10});
    std::vector<std::size_t> off = {0};
    std::vector<std::size_t> shape = {10};
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_EQ(rt.shape()[0], 10u);

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
}

// -- ZarrGroup navigation -- additional tests ---------------------------------

TEST(GroupNav, StoreRef) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    const Store& ref = grp->store_ref();
    EXPECT_TRUE(ref.exists("zarr.json"));
}

TEST(GroupNav, NameForRootGroup) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());
    // Root group path is empty or "/"
    auto p = grp->path();
    EXPECT_TRUE(p.empty() || p == "/");
}

TEST(GroupNav, NestedGroupNavigation) {
    MemoryStore store;
    auto root = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(root.has_value());

    auto sub1 = root->create_group("level1");
    ASSERT_TRUE(sub1.has_value());

    auto sub2 = sub1->create_group("level2");
    ASSERT_TRUE(sub2.has_value());

    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto leaf = sub2->create_array("leaf", opts);
    ASSERT_TRUE(leaf.has_value());

    // Navigate from root
    EXPECT_TRUE(root->contains_group("level1"));
    EXPECT_EQ(root->nchildren(), 1u);

    auto opened = root->open_group("level1");
    ASSERT_TRUE(opened.has_value());
    EXPECT_TRUE(opened->contains_group("level2"));

    auto opened2 = opened->open_group("level2");
    ASSERT_TRUE(opened2.has_value());
    EXPECT_TRUE(opened2->contains_array("leaf"));
}

TEST(GroupNav, ListChildrenSeparated) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;

    auto a1 = grp->create_array("arr1", opts);
    ASSERT_TRUE(a1.has_value());
    auto a2 = grp->create_array("arr2", opts);
    ASSERT_TRUE(a2.has_value());
    auto g1 = grp->create_group("grp1");
    ASSERT_TRUE(g1.has_value());

    auto arrays = grp->list_arrays();
    auto groups = grp->list_groups();
    EXPECT_EQ(arrays.size(), 2u);
    EXPECT_EQ(groups.size(), 1u);

    auto [all_arrs, all_grps] = grp->list_children();
    EXPECT_EQ(all_arrs.size(), 2u);
    EXPECT_EQ(all_grps.size(), 1u);
}

TEST(GroupNav, ContainsArrayAndGroup) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    EXPECT_FALSE(grp->contains_array("x"));
    EXPECT_FALSE(grp->contains_group("x"));

    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = grp->create_array("x", opts);
    ASSERT_TRUE(arr.has_value());

    EXPECT_TRUE(grp->contains_array("x"));
    EXPECT_FALSE(grp->contains_group("x"));

    auto sub = grp->create_group("y");
    ASSERT_TRUE(sub.has_value());

    EXPECT_FALSE(grp->contains_array("y"));
    EXPECT_TRUE(grp->contains_group("y"));
}

TEST(GroupNav, GroupInfo) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "mygroup", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = grp->create_array("data", opts);
    ASSERT_TRUE(arr.has_value());

    auto info = grp->info();
    EXPECT_FALSE(info.empty());
    EXPECT_NE(info.find("mygroup"), std::string::npos);
}

TEST(GroupNav, V2GroupProperties) {
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v2);
    ASSERT_TRUE(grp.has_value());

    EXPECT_EQ(grp->version(), ZarrVersion::v2);
    EXPECT_EQ(grp->nchildren(), 0u);
    EXPECT_FALSE(grp->read_only());
}

TEST(GroupNav, GroupAttrsWithChildren) {
    // Verify group attrs work alongside child creation
    MemoryStore store;
    auto grp = ZarrGroup::create(store, "", ZarrVersion::v3);
    ASSERT_TRUE(grp.has_value());

    auto r = grp->set_attribute("description", "\"top-level group\"");
    ASSERT_TRUE(r.has_value());

    CreateOptions opts;
    opts.shape = {4};
    opts.chunks = {4};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v3;
    auto arr = grp->create_array("data", opts);
    ASSERT_TRUE(arr.has_value());

    // Attrs should still be intact after child creation
    EXPECT_TRUE(grp->contains_attribute("description"));
    auto val = grp->get_attribute("description");
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, "\"top-level group\"");
}

// -- V2 Filters tests ---------------------------------------------------------

TEST(Filters, V2MultipleCodecsInPipeline) {
    // V2 array with gzip filter + blosc compressor
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {20, 20};
    opts.chunks = {10, 10};
    opts.dtype = Dtype::float32;
    opts.version = ZarrVersion::v2;
    // For v2, first codecs are filters, last is compressor
    opts.codecs = {CodecConfig::gzip(1), CodecConfig::blosc()};
    auto arr = ZarrArray::create(store, "test", opts);
    ASSERT_TRUE(arr.has_value());

    auto t = vc::Tensor<float>({20, 20}, vc::Layout::RowMajor, 1.0f);
    std::vector<std::size_t> off = {0, 0};
    std::vector<std::size_t> shape = {20, 20};
    auto wr = arr->write_region(off, shape, t.data());
    ASSERT_TRUE(wr.has_value());

    auto rt = vc::zeros<float>({20, 20});
    auto rd = arr->read_region(off, shape, rt.data());
    ASSERT_TRUE(rd.has_value());
    EXPECT_TRUE(allclose(t, rt));
}

// =============================================================================
// Video Codec tests
// =============================================================================

TEST(CodecConfigFactory, H264) {
    auto c = CodecConfig::h264(18, "fast", 8);
    EXPECT_EQ(c.id, CodecId::H264);
    EXPECT_EQ(c.video_crf, 18);
    EXPECT_EQ(c.video_preset, "fast");
    EXPECT_EQ(c.video_bit_depth, 8);
}

TEST(CodecConfigFactory, H265) {
    auto c = CodecConfig::h265(20, "slow", 10);
    EXPECT_EQ(c.id, CodecId::H265);
    EXPECT_EQ(c.video_crf, 20);
    EXPECT_EQ(c.video_preset, "slow");
    EXPECT_EQ(c.video_bit_depth, 10);
}

TEST(CodecConfigFactory, AV1) {
    auto c = CodecConfig::av1(30, "6", 8);
    EXPECT_EQ(c.id, CodecId::AV1);
    EXPECT_EQ(c.video_crf, 30);
    EXPECT_EQ(c.video_preset, "6");
    EXPECT_EQ(c.video_bit_depth, 8);
}

#ifdef VC_HAS_VIDEO_CODECS

TEST(VideoCodecs, H264RoundTripUInt8) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {32, 32, 32};
    opts.chunks = {32, 32, 32};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::h264(18, "ultrafast")};

    auto arr = ZarrArray::create(store, "h264_test", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<uint8_t> data(32 * 32 * 32);
    std::iota(data.begin(), data.end(), uint8_t(0));

    std::vector<size_t> off = {0, 0, 0};
    std::vector<size_t> shape = {32, 32, 32};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<uint8_t> out(32 * 32 * 32, 0);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    // Lossy: check PSNR is reasonable
    double mse = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double diff = static_cast<double>(data[i]) - static_cast<double>(out[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(data.size());
    double psnr = 10.0 * std::log10(255.0 * 255.0 / mse);
    EXPECT_GT(psnr, 20.0);
}

TEST(VideoCodecs, H264RoundTripLossy) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {32, 32, 32};
    opts.chunks = {32, 32, 32};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::h264(23, "ultrafast")};

    auto arr = ZarrArray::create(store, "h264_lossy", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<uint8_t> data(32 * 32 * 32);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data) v = static_cast<uint8_t>(dist(rng));

    std::vector<size_t> off = {0, 0, 0};
    std::vector<size_t> shape = {32, 32, 32};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<uint8_t> out(32 * 32 * 32, 0);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    // Lossy: check approximate match (PSNR should be reasonable)
    double mse = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double diff = static_cast<double>(data[i]) - static_cast<double>(out[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(data.size());
    double psnr = 10.0 * std::log10(255.0 * 255.0 / mse);
    EXPECT_GT(psnr, 20.0);
}

TEST(VideoCodecs, H265RoundTripUInt8) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {32, 32, 32};
    opts.chunks = {32, 32, 32};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::h265(20, "ultrafast")};

    auto arr = ZarrArray::create(store, "h265_test", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<uint8_t> data(32 * 32 * 32);
    std::iota(data.begin(), data.end(), uint8_t(0));

    std::vector<size_t> off = {0, 0, 0};
    std::vector<size_t> shape = {32, 32, 32};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<uint8_t> out(32 * 32 * 32, 0);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    // Lossy: check PSNR is reasonable
    double mse = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double diff = static_cast<double>(data[i]) - static_cast<double>(out[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(data.size());
    double psnr = 10.0 * std::log10(255.0 * 255.0 / mse);
    EXPECT_GT(psnr, 20.0);
}

TEST(VideoCodecs, AV1RoundTripUInt8) {
    // SVT-AV1 requires minimum 64x64 dimensions
    // AV1 via SVT-AV1 uses YUV420P -- lossy for grayscale data
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {64, 64, 64};
    opts.chunks = {64, 64, 64};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::av1(23, "6")};

    auto arr = ZarrArray::create(store, "av1_test", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<uint8_t> data(64 * 64 * 64);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data) v = static_cast<uint8_t>(dist(rng));

    std::vector<size_t> off = {0, 0, 0};
    std::vector<size_t> shape = {64, 64, 64};
    auto wr = arr->write_region(off, shape, data.data());
    ASSERT_TRUE(wr.has_value());

    std::vector<uint8_t> out(64 * 64 * 64, 0);
    auto rd = arr->read_region(off, shape, out.data());
    ASSERT_TRUE(rd.has_value());

    // AV1 is lossy -- check PSNR is reasonable
    double mse = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double diff = static_cast<double>(data[i]) - static_cast<double>(out[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(data.size());
    double psnr = 10.0 * std::log10(255.0 * 255.0 / mse);
    EXPECT_GT(psnr, 15.0);
}

TEST(VideoCodecs, MetadataSerializationVideoCodec) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {32, 32, 32};
    opts.chunks = {32, 32, 32};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::h264(18, "fast")};

    auto arr = ZarrArray::create(store, "meta_test", opts);
    ASSERT_TRUE(arr.has_value());

    // Re-open and verify metadata
    auto arr2 = ZarrArray::open(store, "meta_test");
    ASSERT_TRUE(arr2.has_value());
    auto codecs = arr2->metadata().codecs();
    EXPECT_GE(codecs.size(), 2u);
    if (codecs.size() >= 2u) {
        auto& video_cc = codecs[1]; // bytes is first, h264 second
        EXPECT_EQ(video_cc.id, CodecId::H264);
        EXPECT_EQ(video_cc.video_crf, 18);
        EXPECT_EQ(video_cc.video_preset, "fast");
        EXPECT_EQ(video_cc.video_bit_depth, 8);
    }
}

TEST(VideoCodecs, Non3DChunkShapeFails) {
    MemoryStore store;
    CreateOptions opts;
    opts.shape = {32, 32};
    opts.chunks = {32, 32};
    opts.dtype = Dtype::uint8;
    opts.version = ZarrVersion::v3;
    opts.codecs = {CodecConfig::bytes(), CodecConfig::h264()};

    auto arr = ZarrArray::create(store, "bad_shape", opts);
    ASSERT_TRUE(arr.has_value());

    std::vector<uint8_t> data(32 * 32, 128);
    std::vector<size_t> off = {0, 0};
    std::vector<size_t> shape = {32, 32};
    auto wr = arr->write_region(off, shape, data.data());
    EXPECT_FALSE(wr.has_value());
    EXPECT_TRUE(wr.error().find("3D") != std::string::npos);
}

#endif // VC_HAS_VIDEO_CODECS
