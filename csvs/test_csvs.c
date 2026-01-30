/*
 * Build with desired codec support:
 *   gcc -std=gnu23 -O2 -Wall -o test_csvs test_csvs.c -llz4 -lm
 *   gcc -std=gnu23 -O2 -Wall -DCSVS_HAS_ZSTD -o test_csvs test_csvs.c -llz4 -lzstd -lm
 */
#define CSVS_IMPLEMENTATION
#include "csvs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define TEST_DIR "/tmp/test_csvs_vol"

static void cleanup(void)
{
    system("rm -rf " TEST_DIR);
}

static void test_create_and_open(void)
{
    printf("test_create_and_open... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {64, 128, 256};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 256,
                       CSVS_CODEC_LZ4, 1) == 0);
    assert(vol.chunk_size == 32);
    assert(vol.shard_size == 256);
    assert(vol.chunks_per_shard == 8);
    assert(vol.padded_shape[0] == 256);
    assert(vol.padded_shape[1] == 256);
    assert(vol.padded_shape[2] == 256);
    assert(vol.codec == CSVS_CODEC_LZ4);
    assert(vol.shard_cache != NULL);
    csvs_close(&vol);

    /* reopen */
    assert(csvs_open(&vol, TEST_DIR) == 0);
    assert(vol.shape[0] == 64);
    assert(vol.shape[1] == 128);
    assert(vol.shape[2] == 256);
    assert(vol.chunk_size == 32);
    assert(vol.codec == CSVS_CODEC_LZ4);
    assert(vol.shard_cache != NULL);
    csvs_close(&vol);

    printf("OK\n");
}

static void test_lz4_roundtrip(void)
{
    printf("test_lz4_roundtrip... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 64,
                       CSVS_CODEC_LZ4, 1) == 0);

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    uint8_t* rbuf = (uint8_t*)malloc(voxels);

    for (size_t i = 0; i < voxels; i++)
        wbuf[i] = (uint8_t)(i & 0xFF);

    assert(csvs_write_chunk(&vol, 0, 0, 0, wbuf, voxels) == 0);
    assert(csvs_read_chunk(&vol, 0, 0, 0, rbuf) == 0);
    assert(memcmp(wbuf, rbuf, voxels) == 0);

    /* unwritten chunk */
    assert(csvs_read_chunk(&vol, 0, 0, 1, rbuf) == 0);
    for (size_t i = 0; i < voxels; i++) assert(rbuf[i] == 0);

    free(wbuf);
    free(rbuf);
    csvs_close(&vol);
    printf("OK\n");
}

#ifdef CSVS_HAS_ZSTD
static void test_zstd_roundtrip(void)
{
    printf("test_zstd_roundtrip... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 64,
                       CSVS_CODEC_ZSTD, 3) == 0);

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    uint8_t* rbuf = (uint8_t*)malloc(voxels);

    for (size_t i = 0; i < voxels; i++)
        wbuf[i] = (uint8_t)(i & 0xFF);

    assert(csvs_write_chunk(&vol, 0, 0, 0, wbuf, voxels) == 0);
    assert(csvs_read_chunk(&vol, 0, 0, 0, rbuf) == 0);
    assert(memcmp(wbuf, rbuf, voxels) == 0);

    /* verify meta codec */
    csvs_volume vol2;
    assert(csvs_open(&vol2, TEST_DIR) == 0);
    assert(vol2.codec == CSVS_CODEC_ZSTD);
    csvs_close(&vol2);

    free(wbuf);
    free(rbuf);
    csvs_close(&vol);
    printf("OK\n");
}
#endif

static void test_multiple_chunks(void)
{
    printf("test_multiple_chunks... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {128, 128, 128};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 64,
                       CSVS_CODEC_LZ4, 1) == 0);

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    uint8_t* rbuf = (uint8_t*)malloc(voxels);

    size_t coords[][3] = {{0,0,0}, {0,0,1}, {1,1,1}, {0,1,0}};
    size_t nc = sizeof(coords)/sizeof(coords[0]);

    for (size_t c = 0; c < nc; c++) {
        for (size_t i = 0; i < voxels; i++)
            wbuf[i] = (uint8_t)((c * 37 + i) & 0xFF);
        assert(csvs_write_chunk(&vol, coords[c][0], coords[c][1], coords[c][2],
                                wbuf, voxels) == 0);
    }

    for (size_t c = 0; c < nc; c++) {
        assert(csvs_read_chunk(&vol, coords[c][0], coords[c][1], coords[c][2],
                               rbuf) == 0);
        for (size_t i = 0; i < voxels; i++) {
            uint8_t expected = (uint8_t)((c * 37 + i) & 0xFF);
            assert(rbuf[i] == expected);
        }
    }

    free(wbuf);
    free(rbuf);
    csvs_close(&vol);
    printf("OK\n");
}

/* ---- error statistics helper ------------------------------------------ */

static int cmp_uint(const void* a, const void* b)
{
    unsigned ua = *(const unsigned*)a;
    unsigned ub = *(const unsigned*)b;
    return (ua > ub) - (ua < ub);
}

static void print_error_stats(const char* label, const uint8_t* orig,
                               const uint8_t* decoded, size_t n)
{
    unsigned* errs = (unsigned*)malloc(n * sizeof(unsigned));
    double sum = 0, sqsum = 0;
    unsigned max_err = 0;
    size_t exact = 0;

    for (size_t i = 0; i < n; i++) {
        int d = (int)orig[i] - (int)decoded[i];
        unsigned ae = (unsigned)(d < 0 ? -d : d);
        errs[i] = ae;
        sum += ae;
        sqsum += (double)ae * ae;
        if (ae > max_err) max_err = ae;
        if (ae == 0) exact++;
    }

    qsort(errs, n, sizeof(unsigned), cmp_uint);

    double mean = sum / (double)n;
    double mse = sqsum / (double)n;
    double psnr = mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : 999.0;
    unsigned p50  = errs[n / 2];
    unsigned p90  = errs[(size_t)(n * 0.90)];
    unsigned p99  = errs[(size_t)(n * 0.99)];
    unsigned p999 = errs[(size_t)(n * 0.999)];

    printf("  %s: mean=%.2f MSE=%.1f PSNR=%.1fdB max=%u "
           "p50=%u p90=%u p99=%u p99.9=%u exact=%.1f%%\n",
           label, mean, mse, psnr, max_err,
           p50, p90, p99, p999,
           100.0 * (double)exact / (double)n);

    free(errs);
}

/* ---- video codec tests ------------------------------------------------ */

#ifdef CSVS_HAS_H264
static void test_h264_roundtrip(void)
{
    printf("test_h264_roundtrip...\n");
    cleanup();

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    uint8_t* rbuf = (uint8_t*)malloc(voxels);

    for (size_t i = 0; i < voxels; i++)
        wbuf[i] = (uint8_t)((i * 7) & 0xFF);

    int qps[] = {0, 10, 20, 30, 40, 51};
    for (size_t q = 0; q < sizeof(qps)/sizeof(qps[0]); q++) {
        cleanup();
        csvs_volume vol;
        size_t shape[3] = {64, 64, 64};
        assert(csvs_create(&vol, TEST_DIR, shape, cs, 64,
                           CSVS_CODEC_H264, qps[q]) == 0);

        assert(csvs_write_chunk(&vol, 0, 0, 0, wbuf, voxels) == 0);
        assert(csvs_read_chunk(&vol, 0, 0, 0, rbuf) == 0);

        char label[32];
        snprintf(label, sizeof(label), "h264 QP=%2d", qps[q]);
        print_error_stats(label, wbuf, rbuf, voxels);

        csvs_close(&vol);
    }

    /* unwritten chunk returns zeros */
    cleanup();
    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, cs, 64,
                       CSVS_CODEC_H264, 0) == 0);
    assert(csvs_read_chunk(&vol, 0, 0, 1, rbuf) == 0);
    for (size_t i = 0; i < voxels; i++) assert(rbuf[i] == 0);
    csvs_close(&vol);

    free(wbuf);
    free(rbuf);
    printf("  OK\n");
}
#endif

#ifdef CSVS_HAS_AV1
static void test_av1_roundtrip(void)
{
    printf("test_av1_roundtrip...\n");
    cleanup();

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    uint8_t* rbuf = (uint8_t*)malloc(voxels);

    for (size_t i = 0; i < voxels; i++)
        wbuf[i] = (uint8_t)((i * 7) & 0xFF);

    int cqs[] = {0, 10, 20, 30, 40, 63};
    for (size_t q = 0; q < sizeof(cqs)/sizeof(cqs[0]); q++) {
        cleanup();
        csvs_volume vol;
        size_t shape[3] = {64, 64, 64};
        assert(csvs_create(&vol, TEST_DIR, shape, cs, 64,
                           CSVS_CODEC_AV1, cqs[q]) == 0);

        assert(csvs_write_chunk(&vol, 0, 0, 0, wbuf, voxels) == 0);
        assert(csvs_read_chunk(&vol, 0, 0, 0, rbuf) == 0);

        char label[32];
        snprintf(label, sizeof(label), "av1  CQ=%2d", cqs[q]);
        print_error_stats(label, wbuf, rbuf, voxels);

        csvs_close(&vol);
    }

    /* unwritten chunk returns zeros */
    cleanup();
    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, cs, 64,
                       CSVS_CODEC_AV1, 0) == 0);
    assert(csvs_read_chunk(&vol, 0, 0, 1, rbuf) == 0);
    for (size_t i = 0; i < voxels; i++) assert(rbuf[i] == 0);
    csvs_close(&vol);

    free(wbuf);
    free(rbuf);
    printf("  OK\n");
}
#endif

static void test_read_shard(void)
{
    printf("test_read_shard... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 64,
                       CSVS_CODEC_LZ4, 1) == 0);

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    size_t n = csvs__chunks_in_shard(&vol);

    /* Write a shard with even chunks only */
    uint8_t* wdata = (uint8_t*)malloc(n * voxels);
    uint8_t* wmask = (uint8_t*)malloc(n);
    for (size_t i = 0; i < n; i++) {
        wmask[i] = (i % 2 == 0) ? 1 : 0;
        uint8_t* chunk = wdata + i * voxels;
        for (size_t j = 0; j < voxels; j++)
            chunk[j] = (uint8_t)((i * 100 + j) & 0xFF);
    }
    assert(csvs_write_shard(&vol, 0, 0, 0, wdata, wmask, voxels) == 0);

    /* Read it back */
    uint8_t* rdata = (uint8_t*)calloc(n, voxels);
    uint8_t* rmask = (uint8_t*)calloc(n, 1);
    assert(csvs_read_shard(&vol, 0, 0, 0, rdata, rmask) == 0);

    for (size_t i = 0; i < n; i++) {
        assert(rmask[i] == wmask[i]);
        if (wmask[i])
            assert(memcmp(wdata + i * voxels, rdata + i * voxels, voxels) == 0);
    }

    /* Read missing shard */
    memset(rmask, 0xFF, n);
    assert(csvs_read_shard(&vol, 1, 0, 0, rdata, rmask) == 0);
    for (size_t i = 0; i < n; i++)
        assert(rmask[i] == 0);

    free(wdata);
    free(wmask);
    free(rdata);
    free(rmask);
    csvs_close(&vol);
    printf("OK\n");
}

static void test_read_region(void)
{
    printf("test_read_region... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 64,
                       CSVS_CODEC_LZ4, 1) == 0);

    size_t cs = 32;
    size_t voxels = cs * cs * cs;

    /* Write chunk (0,0,0) with known pattern */
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    for (size_t i = 0; i < voxels; i++)
        wbuf[i] = (uint8_t)(i & 0xFF);
    assert(csvs_write_chunk(&vol, 0, 0, 0, wbuf, voxels) == 0);

    /* Read a sub-region fully inside chunk (0,0,0) */
    size_t rz = 4, ry = 4, rx = 4;
    size_t rn = rz * ry * rx;
    uint8_t* rbuf = (uint8_t*)calloc(rn, 1);
    assert(csvs_read_region(&vol, 2, 2, 2, rz, ry, rx, rbuf) == 0);

    for (size_t z = 0; z < rz; z++) {
        for (size_t y = 0; y < ry; y++) {
            for (size_t x = 0; x < rx; x++) {
                size_t ri = z * ry * rx + y * rx + x;
                size_t ci = (z + 2) * cs * cs + (y + 2) * cs + (x + 2);
                assert(rbuf[ri] == (uint8_t)(ci & 0xFF));
            }
        }
    }

    /* Read region spanning into unwritten chunk — zeros there */
    size_t bz = 8, by = 8, bx = 8;
    size_t bn = bz * by * bx;
    uint8_t* bbuf = (uint8_t*)calloc(bn, 1);
    assert(csvs_read_region(&vol, 28, 28, 28, bz, by, bx, bbuf) == 0);

    for (size_t z = 0; z < 4; z++) {
        for (size_t y = 0; y < 4; y++) {
            for (size_t x = 0; x < 4; x++) {
                size_t ri = z * by * bx + y * bx + x;
                size_t ci = (z + 28) * cs * cs + (y + 28) * cs + (x + 28);
                assert(bbuf[ri] == (uint8_t)(ci & 0xFF));
            }
        }
    }
    for (size_t z = 4; z < bz; z++) {
        for (size_t y = 0; y < by; y++) {
            for (size_t x = 0; x < bx; x++) {
                size_t ri = z * by * bx + y * bx + x;
                assert(bbuf[ri] == 0);
            }
        }
    }

    free(wbuf);
    free(rbuf);
    free(bbuf);
    csvs_close(&vol);
    printf("OK\n");
}

static void test_write_chunk_pwrite(void)
{
    printf("test_write_chunk_pwrite... ");
    cleanup();

    csvs_volume vol;
    size_t shape[3] = {64, 64, 64};
    assert(csvs_create(&vol, TEST_DIR, shape, 32, 64,
                       CSVS_CODEC_LZ4, 1) == 0);

    size_t cs = 32;
    size_t voxels = cs * cs * cs;
    uint8_t* wbuf = (uint8_t*)malloc(voxels);
    uint8_t* rbuf = (uint8_t*)malloc(voxels);

    for (size_t i = 0; i < voxels; i++) wbuf[i] = (uint8_t)(i + 100);
    assert(csvs_write_chunk(&vol, 0, 0, 0, wbuf, voxels) == 0);

    for (size_t i = 0; i < voxels; i++) wbuf[i] = (uint8_t)(i + 200);
    assert(csvs_write_chunk(&vol, 0, 0, 1, wbuf, voxels) == 0);

    assert(csvs_read_chunk(&vol, 0, 0, 0, rbuf) == 0);
    for (size_t i = 0; i < voxels; i++)
        assert(rbuf[i] == (uint8_t)(i + 100));

    assert(csvs_read_chunk(&vol, 0, 0, 1, rbuf) == 0);
    for (size_t i = 0; i < voxels; i++)
        assert(rbuf[i] == (uint8_t)(i + 200));

    free(wbuf);
    free(rbuf);
    csvs_close(&vol);
    printf("OK\n");
}

int main(void)
{
    test_create_and_open();
    test_lz4_roundtrip();
#ifdef CSVS_HAS_ZSTD
    test_zstd_roundtrip();
#endif
    test_multiple_chunks();
#ifdef CSVS_HAS_H264
    test_h264_roundtrip();
#endif
#ifdef CSVS_HAS_AV1
    test_av1_roundtrip();
#endif
    test_read_shard();
    test_read_region();
    test_write_chunk_pwrite();
    cleanup();
    printf("All tests passed.\n");
    return 0;
}
