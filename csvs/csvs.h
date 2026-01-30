/*
 * csvs.h — Chunked Sharded Volume Storage (uint8 voxels)
 *
 * Single-file stb-style header-only C library.
 *
 * Usage:
 *   #define CSVS_IMPLEMENTATION
 *   #include "csvs.h"
 *
 * Required: lz4 (-llz4)
 * Optional (define before including):
 *   #define CSVS_HAS_ZSTD    — enables zstd codec (-lzstd)
 *   #define CSVS_HAS_H264    — enables h264 codec via openh264 (-lopenh264)
 *   #define CSVS_HAS_H265    — enables h265/hevc codec via x265+libde265 (-lx265 -lde265)
 *   #define CSVS_HAS_AV1     — enables av1 codec via libaom+dav1d (-laom -ldav1d)
 */
#ifndef CSVS_H
#define CSVS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CSVS_CODEC_LZ4,
    CSVS_CODEC_ZSTD,
    CSVS_CODEC_H264,
    CSVS_CODEC_H265,
    CSVS_CODEC_AV1
} csvs_codec;

#ifndef CSVS_SHARD_CACHE_CAP
#define CSVS_SHARD_CACHE_CAP 64
#endif

typedef struct {
    void*    data;   /* mmap'd region */
    size_t   len;
    int      fd;
    uint64_t key;    /* shard coord hash, 0 = empty */
} csvs__shard_map;

typedef struct {
    char path[1024];
    size_t shape[3];          /* z, y, x actual dimensions */
    size_t padded_shape[3];   /* padded to shard multiple */
    uint32_t chunk_size;
    uint32_t shard_size;
    uint32_t chunks_per_shard; /* per axis */
    csvs_codec codec;
    int codec_level;
    csvs__shard_map* shard_cache;
} csvs_volume;

int csvs_open(csvs_volume* vol, const char* path);
int csvs_create(csvs_volume* vol, const char* path,
                const size_t shape[3], uint32_t chunk_size,
                uint32_t shard_size,
                csvs_codec codec, int codec_level);
int csvs_read_chunk(const csvs_volume* vol,
                    size_t cz, size_t cy, size_t cx, void* buf);
int csvs_write_chunk(const csvs_volume* vol,
                     size_t cz, size_t cy, size_t cx,
                     const void* buf, size_t raw_size);
int csvs_write_shard(const csvs_volume* vol,
                     size_t sz, size_t sy, size_t sx,
                     const void* chunks, const uint8_t* mask,
                     size_t raw_chunk_size);
int csvs_read_shard(const csvs_volume* vol,
                    size_t sz, size_t sy, size_t sx,
                    void* chunks, uint8_t* mask);
int csvs_read_region(const csvs_volume* vol,
                     size_t z0, size_t y0, size_t x0,
                     size_t zn, size_t yn, size_t xn,
                     void* buf);
void csvs_close(csvs_volume* vol);

const char* csvs_codec_str(csvs_codec c);

#ifdef __cplusplus
}
#endif

#endif /* CSVS_H */

/* ======================================================================== */
#ifdef CSVS_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <lz4.h>

#ifdef CSVS_HAS_ZSTD
#include <zstd.h>
#endif

#ifdef CSVS_HAS_H264
#include <wels/codec_api.h>
#include <wels/codec_app_def.h>
#include <wels/codec_ver.h>
#endif

#ifdef CSVS_HAS_H265
#include <x265.h>
#include <libde265/de265.h>
#endif

#ifdef CSVS_HAS_AV1
#include <aom/aom_encoder.h>
#include <aom/aomcx.h>
#include <dav1d/dav1d.h>
#endif

#define CSVS__MISSING_OFF  UINT64_MAX
#define CSVS__MISSING_LEN  UINT64_MAX
#define CSVS__INDEX_ENTRY  16

/* ---- codec helpers ---------------------------------------------------- */

const char* csvs_codec_str(csvs_codec c)
{
    switch (c) {
    case CSVS_CODEC_LZ4:  return "lz4";
    case CSVS_CODEC_ZSTD: return "zstd";
    case CSVS_CODEC_H264: return "h264";
    case CSVS_CODEC_H265: return "h265";
    case CSVS_CODEC_AV1:  return "av1";
    }
    return "unknown";
}

static int csvs__parse_codec(const char* s, csvs_codec* out)
{
    static const struct { const char* name; csvs_codec val; } tbl[] = {
        {"lz4",  CSVS_CODEC_LZ4},  {"zstd", CSVS_CODEC_ZSTD},
        {"h264", CSVS_CODEC_H264}, {"h265", CSVS_CODEC_H265},
        {"av1",  CSVS_CODEC_AV1},
    };
    for (size_t i = 0; i < sizeof(tbl)/sizeof(tbl[0]); i++) {
        if (strcmp(s, tbl[i].name) == 0) { *out = tbl[i].val; return 0; }
    }
    return -1;
}

/* ---- math / path helpers ---------------------------------------------- */

static size_t csvs__round_up(size_t v, size_t m)
{
    return ((v + m - 1) / m) * m;
}

static void csvs__meta_path(char* out, size_t n, const char* base)
{
    snprintf(out, n, "%s/meta.ini", base);
}

static void csvs__shard_dir(char* out, size_t n, const char* base)
{
    snprintf(out, n, "%s/shards", base);
}

static void csvs__shard_path(char* out, size_t n, const char* base,
                             size_t sz, size_t sy, size_t sx)
{
    snprintf(out, n, "%s/shards/%zu_%zu_%zu.shard", base, sz, sy, sx);
}

/* ---- INI parse / write ------------------------------------------------ */

static int csvs__parse_meta(csvs_volume* vol)
{
    char path[1088];
    csvs__meta_path(path, sizeof(path), vol->path);

    FILE* f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    int got_format = 0;
    char codec_str[32] = "lz4";
    char dtype_str[32] = {0};

    while (fgets(line, sizeof(line), f)) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = 0;
        char* eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        char* key = line;
        char* val = eq + 1;

        if (strcmp(key, "format") == 0) {
            if (strcmp(val, "csvs") != 0) { fclose(f); return -1; }
            got_format = 1;
        } else if (strcmp(key, "shape") == 0) {
            sscanf(val, "%zu,%zu,%zu", &vol->shape[0], &vol->shape[1], &vol->shape[2]);
        } else if (strcmp(key, "chunk_size") == 0) {
            vol->chunk_size = (uint32_t)atoi(val);
        } else if (strcmp(key, "shard_size") == 0) {
            vol->shard_size = (uint32_t)atoi(val);
        } else if (strcmp(key, "dtype") == 0) {
            snprintf(dtype_str, sizeof(dtype_str), "%s", val);
        } else if (strcmp(key, "codec") == 0) {
            snprintf(codec_str, sizeof(codec_str), "%s", val);
        } else if (strcmp(key, "codec_level") == 0) {
            vol->codec_level = atoi(val);
        }
    }
    fclose(f);

    if (!got_format) return -1;
    /* Accept "uint8" or empty dtype for compatibility; reject anything else */
    if (dtype_str[0] && strcmp(dtype_str, "uint8") != 0) return -1;
    if (csvs__parse_codec(codec_str, &vol->codec) != 0) return -1;

    vol->chunks_per_shard = vol->shard_size / vol->chunk_size;
    for (int i = 0; i < 3; i++)
        vol->padded_shape[i] = csvs__round_up(vol->shape[i], vol->shard_size);

    return 0;
}

static int csvs__write_meta(const csvs_volume* vol)
{
    char path[1088];
    csvs__meta_path(path, sizeof(path), vol->path);

    FILE* f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "format=csvs\n");
    fprintf(f, "version=1\n");
    fprintf(f, "shape=%zu,%zu,%zu\n", vol->shape[0], vol->shape[1], vol->shape[2]);
    fprintf(f, "chunk_size=%u\n", vol->chunk_size);
    fprintf(f, "shard_size=%u\n", vol->shard_size);
    fprintf(f, "dtype=uint8\n");
    fprintf(f, "codec=%s\n", csvs_codec_str(vol->codec));
    fprintf(f, "codec_level=%d\n", vol->codec_level);
    fclose(f);
    return 0;
}

/* ---- shard index I/O -------------------------------------------------- */

static size_t csvs__chunks_in_shard(const csvs_volume* vol)
{
    size_t c = vol->chunks_per_shard;
    return c * c * c;
}

static int csvs__read_index(const csvs_volume* vol, const char* shard_path,
                            uint64_t* index, size_t* data_end)
{
    size_t n = csvs__chunks_in_shard(vol);
    size_t idx_bytes = n * CSVS__INDEX_ENTRY;

    int fd = open(shard_path, O_RDONLY);
    if (fd < 0) {
        for (size_t i = 0; i < n; i++) {
            index[i * 2]     = CSVS__MISSING_OFF;
            index[i * 2 + 1] = CSVS__MISSING_LEN;
        }
        *data_end = 0;
        return 0;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < idx_bytes) {
        close(fd);
        return -1;
    }

    *data_end = (size_t)st.st_size - idx_bytes;
    ssize_t rd = pread(fd, index, idx_bytes, (off_t)*data_end);
    close(fd);
    if (rd < 0 || (size_t)rd != idx_bytes) return -1;
    return 0;
}

static size_t csvs__chunk_index_in_shard(const csvs_volume* vol,
                                         size_t lz, size_t ly, size_t lx)
{
    size_t c = vol->chunks_per_shard;
    return (lz * c + ly) * c + lx;
}

/* ---- shard cache (mmap-based) ----------------------------------------- */

static uint64_t csvs__shard_key(size_t sz, size_t sy, size_t sx)
{
    uint64_t k = ((uint64_t)sz * 73856093ULL) ^
                 ((uint64_t)sy * 19349669ULL) ^
                 ((uint64_t)sx * 83492791ULL);
    return k ? k : 1;
}

static csvs__shard_map* csvs__cache_lookup(const csvs_volume* vol,
                                            uint64_t key)
{
    if (!vol->shard_cache) return NULL;
    uint32_t slot = (uint32_t)(key % CSVS_SHARD_CACHE_CAP);
    csvs__shard_map* m = &vol->shard_cache[slot];
    if (m->key == key && m->data != NULL) return m;
    return NULL;
}

static void csvs__cache_evict(csvs__shard_map* m)
{
    if (m->data) {
        munmap(m->data, m->len);
        close(m->fd);
        m->data = NULL;
        m->len = 0;
        m->fd = -1;
        m->key = 0;
    }
}

static void csvs__cache_invalidate(const csvs_volume* vol,
                                    size_t sz, size_t sy, size_t sx)
{
    if (!vol->shard_cache) return;
    uint64_t key = csvs__shard_key(sz, sy, sx);
    uint32_t slot = (uint32_t)(key % CSVS_SHARD_CACHE_CAP);
    csvs__shard_map* m = &vol->shard_cache[slot];
    if (m->key == key) csvs__cache_evict(m);
}

static const csvs__shard_map* csvs__mmap_shard(const csvs_volume* vol,
                                                size_t sz, size_t sy, size_t sx)
{
    uint64_t key = csvs__shard_key(sz, sy, sx);

    csvs__shard_map* cached = csvs__cache_lookup(vol, key);
    if (cached) return cached;

    char spath[1200];
    csvs__shard_path(spath, sizeof(spath), vol->path, sz, sy, sx);

    int fd = open(spath, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        close(fd);
        return NULL;
    }

    void* data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    if (vol->shard_cache) {
        uint32_t slot = (uint32_t)(key % CSVS_SHARD_CACHE_CAP);
        csvs__shard_map* m = &vol->shard_cache[slot];
        if (m->key != key) csvs__cache_evict(m);
        m->data = data;
        m->len = (size_t)st.st_size;
        m->fd = fd;
        m->key = key;
        return m;
    }

    static thread_local csvs__shard_map tls_map;
    csvs__cache_evict(&tls_map);
    tls_map.data = data;
    tls_map.len = (size_t)st.st_size;
    tls_map.fd = fd;
    tls_map.key = key;
    return &tls_map;
}

static void csvs__init_cache(csvs_volume* vol)
{
    vol->shard_cache = (csvs__shard_map*)calloc(
        CSVS_SHARD_CACHE_CAP, sizeof(csvs__shard_map));
}

static void csvs__free_cache(csvs_volume* vol)
{
    if (!vol->shard_cache) return;
    for (uint32_t i = 0; i < CSVS_SHARD_CACHE_CAP; i++)
        csvs__cache_evict(&vol->shard_cache[i]);
    free(vol->shard_cache);
    vol->shard_cache = NULL;
}

/* ======================================================================
 * Video codec compressed format (h264, av1):
 *   [uint32_t frame_sizes[chunk_size]]   — size of each encoded frame
 *   [frame_0 data][frame_1 data]...      — concatenated bitstream
 * ====================================================================== */

/* ---- H.264 via openh264 ----------------------------------------------- */
#ifdef CSVS_HAS_H264

static int csvs__h264_compress(const uint8_t* frames, uint32_t w, uint32_t h,
                               uint32_t nframes, int qp,
                               void* dst, int dst_cap, int* out_size)
{
    ISVCEncoder* enc = NULL;
    if (WelsCreateSVCEncoder(&enc) != 0 || !enc) return -1;

    SEncParamExt param;
    (*enc)->GetDefaultParams(enc, &param);
    param.iUsageType = SCREEN_CONTENT_REAL_TIME;
    param.iPicWidth = (int)w;
    param.iPicHeight = (int)h;
    param.fMaxFrameRate = 30.0f;
    param.iRCMode = RC_OFF_MODE;
    param.iNumRefFrame = 1;
    param.iMultipleThreadIdc = 1;
    param.bEnableFrameSkip = 0;
    param.sSpatialLayers[0].iVideoWidth  = (int)w;
    param.sSpatialLayers[0].iVideoHeight = (int)h;
    param.sSpatialLayers[0].fFrameRate = 30.0f;
    param.sSpatialLayers[0].iSpatialBitrate = 0;
    param.sSpatialLayers[0].iDLayerQp = qp;
    param.iSpatialLayerNum = 1;
    param.iTemporalLayerNum = 1;

    if ((*enc)->InitializeExt(enc, &param) != 0) {
        WelsDestroySVCEncoder(enc);
        return -1;
    }

    int video_fmt = videoFormatI420;
    (*enc)->SetOption(enc, ENCODER_OPTION_DATAFORMAT, &video_fmt);

    size_t frame_pixels = (size_t)w * h;
    size_t uv_size = (size_t)(w / 2) * (h / 2);
    uint8_t* uv = (uint8_t*)malloc(uv_size);
    memset(uv, 128, uv_size);

    uint32_t* frame_sizes = (uint32_t*)dst;
    uint8_t* payload = (uint8_t*)dst + nframes * sizeof(uint32_t);
    size_t payload_cap = (size_t)dst_cap - nframes * sizeof(uint32_t);
    size_t written = 0;

    int rc = 0;
    for (uint32_t f = 0; f < nframes; f++) {
        SSourcePicture pic;
        memset(&pic, 0, sizeof(pic));
        pic.iPicWidth  = (int)w;
        pic.iPicHeight = (int)h;
        pic.iColorFormat = videoFormatI420;
        pic.iStride[0] = (int)w;
        pic.iStride[1] = (int)(w / 2);
        pic.iStride[2] = (int)(w / 2);
        pic.pData[0] = (unsigned char*)(frames + f * frame_pixels);
        pic.pData[1] = uv;
        pic.pData[2] = uv;

        SFrameBSInfo info;
        memset(&info, 0, sizeof(info));
        if ((*enc)->EncodeFrame(enc, &pic, &info) != 0) { rc = -1; break; }

        size_t frame_bytes = 0;
        if (info.eFrameType != videoFrameTypeSkip) {
            for (int layer = 0; layer < info.iLayerNum; layer++) {
                SLayerBSInfo* li = &info.sLayerInfo[layer];
                int lsz = 0;
                for (int n = 0; n < li->iNalCount; n++) lsz += li->pNalLengthInByte[n];
                if (written + (size_t)lsz > payload_cap) { rc = -1; break; }
                memcpy(payload + written, li->pBsBuf, (size_t)lsz);
                frame_bytes += (size_t)lsz;
                written += (size_t)lsz;
            }
            if (rc != 0) break;
        }
        frame_sizes[f] = (uint32_t)frame_bytes;
    }

    free(uv);
    (*enc)->Uninitialize(enc);
    WelsDestroySVCEncoder(enc);

    if (rc == 0)
        *out_size = (int)(nframes * sizeof(uint32_t) + written);
    return rc;
}

static int csvs__h264_decompress(const void* src, int src_size,
                                 uint8_t* frames, uint32_t w, uint32_t h,
                                 uint32_t nframes)
{
    ISVCDecoder* dec = NULL;
    if (WelsCreateDecoder(&dec) != 0 || !dec) return -1;

    SDecodingParam dp;
    memset(&dp, 0, sizeof(dp));
    dp.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_AVC;
    if ((*dec)->Initialize(dec, &dp) != 0) {
        WelsDestroyDecoder(dec);
        return -1;
    }

    const uint32_t* frame_sizes = (const uint32_t*)src;
    const uint8_t* payload = (const uint8_t*)src + nframes * sizeof(uint32_t);
    size_t frame_pixels = (size_t)w * h;
    size_t offset = 0;
    int rc = 0;

    for (uint32_t f = 0; f < nframes; f++) {
        uint32_t fsz = frame_sizes[f];
        if (fsz == 0) {
            memset(frames + f * frame_pixels, 0, frame_pixels);
            continue;
        }

        uint8_t* dst_ptrs[3] = {0};
        SBufferInfo buf_info;
        memset(&buf_info, 0, sizeof(buf_info));
        DECODING_STATE ds = (*dec)->DecodeFrameNoDelay(
            dec, (unsigned char*)(payload + offset), (int)fsz, dst_ptrs, &buf_info);

        if (ds != dsErrorFree && ds != dsNoParamSets) { rc = -1; break; }

        if (buf_info.iBufferStatus == 1) {
            int stride = buf_info.UsrData.sSystemBuffer.iStride[0];
            const uint8_t* y = dst_ptrs[0];
            for (uint32_t row = 0; row < h; row++)
                memcpy(frames + f * frame_pixels + row * w,
                       y + row * stride, w);
        } else {
            memset(frames + f * frame_pixels, 0, frame_pixels);
        }
        offset += fsz;
    }

    (*dec)->Uninitialize(dec);
    WelsDestroyDecoder(dec);
    return rc;
}

#endif /* CSVS_HAS_H264 */

/* ---- H.265/HEVC via x265 (encode) + libde265 (decode) ----------------- */
#ifdef CSVS_HAS_H265

static int csvs__h265_compress(const uint8_t* frames, uint32_t w, uint32_t h,
                               uint32_t nframes, int qp,
                               void* dst, int dst_cap, int* out_size)
{
    x265_param* param = x265_param_alloc();
    if (!param) return -1;
    x265_param_default_preset(param, "ultrafast", "fastdecode");

    param->sourceWidth = (int)w;
    param->sourceHeight = (int)h;
    param->fpsNum = 30;
    param->fpsDenom = 1;
    param->internalCsp = X265_CSP_I400;
    param->rc.rateControlMode = X265_RC_CQP;
    param->rc.qp = qp;
    param->bRepeatHeaders = 1;
    param->totalFrames = (int)nframes;
    param->logLevel = X265_LOG_NONE;
    param->frameNumThreads = 1;
    param->bEnableWavefront = 0;
    param->bframes = 0;  /* no B-frames — ensures no delayed output */

    x265_encoder* enc = x265_encoder_open(param);
    if (!enc) {
        x265_param_free(param);
        return -1;
    }

    x265_picture* pic = x265_picture_alloc();
    x265_picture_init(param, pic);

    uint32_t* frame_sizes = (uint32_t*)dst;
    uint8_t* payload = (uint8_t*)dst + nframes * sizeof(uint32_t);
    size_t payload_cap = (size_t)dst_cap - nframes * sizeof(uint32_t);
    size_t written = 0;
    size_t frame_pixels = (size_t)w * h;
    int rc = 0;

    /* Encode frames */
    for (uint32_t f = 0; f < nframes; f++) {
        pic->planes[0] = (void*)(frames + f * frame_pixels);
        pic->stride[0] = (int)w;

        x265_nal* nals = NULL;
        uint32_t nal_count = 0;
        int ret = x265_encoder_encode(enc, &nals, &nal_count, pic, NULL);
        if (ret < 0) { rc = -1; break; }

        size_t frame_bytes = 0;
        for (uint32_t n = 0; n < nal_count; n++) {
            size_t sz = (size_t)nals[n].sizeBytes;
            if (written + sz > payload_cap) { rc = -1; break; }
            memcpy(payload + written, nals[n].payload, sz);
            frame_bytes += sz;
            written += sz;
        }
        if (rc != 0) break;
        frame_sizes[f] = (uint32_t)frame_bytes;
    }

    /* Flush delayed frames */
    if (rc == 0) {
        for (;;) {
            x265_nal* nals = NULL;
            uint32_t nal_count = 0;
            int ret = x265_encoder_encode(enc, &nals, &nal_count, NULL, NULL);
            if (ret <= 0) break;
            for (uint32_t n = 0; n < nal_count; n++) {
                size_t sz = (size_t)nals[n].sizeBytes;
                if (written + sz > payload_cap) { rc = -1; break; }
                memcpy(payload + written, nals[n].payload, sz);
                written += sz;
            }
            if (rc != 0) break;
        }
    }

    x265_picture_free(pic);
    x265_encoder_close(enc);
    x265_param_free(param);

    if (rc == 0)
        *out_size = (int)(nframes * sizeof(uint32_t) + written);
    return rc;
}

static int csvs__h265_decompress(const void* src, int src_size,
                                 uint8_t* frames, uint32_t w, uint32_t h,
                                 uint32_t nframes)
{
    de265_decoder_context* ctx = de265_new_decoder();
    if (!ctx) return -1;

    de265_set_parameter_bool(ctx, DE265_DECODER_PARAM_SUPPRESS_FAULTY_PICTURES, 0);
    de265_start_worker_threads(ctx, 0);

    const uint32_t* frame_sizes = (const uint32_t*)src;
    const uint8_t* payload = (const uint8_t*)src + nframes * sizeof(uint32_t);
    size_t frame_pixels = (size_t)w * h;
    size_t offset = 0;
    int rc = 0;
    uint32_t decoded = 0;

    /* Push all data, pull decoded frames */
    for (uint32_t f = 0; f < nframes; f++) {
        uint32_t fsz = frame_sizes[f];
        if (fsz == 0) {
            memset(frames + decoded * frame_pixels, 0, frame_pixels);
            decoded++;
            continue;
        }

        de265_error err = de265_push_data(ctx, payload + offset, fsz, offset, NULL);
        if (err != DE265_OK) { rc = -1; break; }
        offset += fsz;

        /* Decode available frames */
        int more = 1;
        while (more) {
            err = de265_decode(ctx, &more);
            if (err == DE265_ERROR_WAITING_FOR_INPUT_DATA)
                break;
            if (err != DE265_OK)
                break;

            const struct de265_image* img = de265_get_next_picture(ctx);
            if (img && decoded < nframes) {
                int stride = 0;
                const uint8_t* y = de265_get_image_plane(img, 0, &stride);
                if (y) {
                    for (uint32_t row = 0; row < h; row++)
                        memcpy(frames + decoded * frame_pixels + row * w,
                               y + row * stride, w);
                } else {
                    memset(frames + decoded * frame_pixels, 0, frame_pixels);
                }
                decoded++;
            }
        }
    }

    /* Flush remaining */
    if (rc == 0) {
        de265_flush_data(ctx);
        int more = 1;
        while (more && decoded < nframes) {
            de265_error err = de265_decode(ctx, &more);
            if (err != DE265_OK && err != DE265_ERROR_WAITING_FOR_INPUT_DATA)
                break;
            const struct de265_image* img = de265_get_next_picture(ctx);
            if (img && decoded < nframes) {
                int stride = 0;
                const uint8_t* y = de265_get_image_plane(img, 0, &stride);
                if (y) {
                    for (uint32_t row = 0; row < h; row++)
                        memcpy(frames + decoded * frame_pixels + row * w,
                               y + row * stride, w);
                } else {
                    memset(frames + decoded * frame_pixels, 0, frame_pixels);
                }
                decoded++;
            }
        }
    }

    /* Zero any undelivered frames */
    while (decoded < nframes) {
        memset(frames + decoded * frame_pixels, 0, frame_pixels);
        decoded++;
    }

    de265_free_decoder(ctx);
    return rc;
}

#endif /* CSVS_HAS_H265 */

/* ---- AV1 via libaom (encode) + dav1d (decode) ------------------------- */
#ifdef CSVS_HAS_AV1

static int csvs__av1_compress(const uint8_t* frames, uint32_t w, uint32_t h,
                              uint32_t nframes, int qp,
                              void* dst, int dst_cap, int* out_size)
{
    aom_codec_enc_cfg_t cfg;
    if (aom_codec_enc_config_default(aom_codec_av1_cx(), &cfg, 0) != AOM_CODEC_OK)
        return -1;

    cfg.g_w = w;
    cfg.g_h = h;
    cfg.g_timebase.num = 1;
    cfg.g_timebase.den = 30;
    cfg.rc_end_usage = AOM_Q;
    cfg.g_threads = 1;
    cfg.g_lag_in_frames = 0;
    cfg.g_error_resilient = 0;

    aom_codec_ctx_t ctx;
    if (aom_codec_enc_init(&ctx, aom_codec_av1_cx(), &cfg, 0) != AOM_CODEC_OK)
        return -1;

    aom_codec_control(&ctx, AOME_SET_CPUUSED, 10);
    aom_codec_control(&ctx, AOME_SET_CQ_LEVEL, qp);

    aom_image_t img;
    if (!aom_img_alloc(&img, AOM_IMG_FMT_I420, w, h, 1)) {
        aom_codec_destroy(&ctx);
        return -1;
    }
    memset(img.planes[1], 128, (size_t)(img.stride[1]) * ((h + 1) / 2));
    memset(img.planes[2], 128, (size_t)(img.stride[2]) * ((h + 1) / 2));

    uint32_t* frame_sizes = (uint32_t*)dst;
    uint8_t* payload = (uint8_t*)dst + nframes * sizeof(uint32_t);
    size_t payload_cap = (size_t)dst_cap - nframes * sizeof(uint32_t);
    size_t written = 0;
    size_t frame_pixels = (size_t)w * h;
    int rc = 0;

    for (uint32_t f = 0; f <= nframes; f++) {
        aom_image_t* img_ptr = NULL;
        if (f < nframes) {
            const uint8_t* src_y = frames + f * frame_pixels;
            for (uint32_t row = 0; row < h; row++)
                memcpy(img.planes[0] + row * img.stride[0],
                       src_y + row * w, w);
            img_ptr = &img;
        }

        if (aom_codec_encode(&ctx, img_ptr, (aom_codec_pts_t)f, 1, 0)
            != AOM_CODEC_OK) { rc = -1; break; }

        const aom_codec_cx_pkt_t* pkt;
        aom_codec_iter_t iter = NULL;
        size_t frame_bytes = 0;
        while ((pkt = aom_codec_get_cx_data(&ctx, &iter))) {
            if (pkt->kind != AOM_CODEC_CX_FRAME_PKT) continue;
            size_t sz = pkt->data.frame.sz;
            if (written + sz > payload_cap) { rc = -1; break; }
            memcpy(payload + written, pkt->data.frame.buf, sz);
            frame_bytes += sz;
            written += sz;
        }
        if (rc != 0) break;
        if (f < nframes)
            frame_sizes[f] = (uint32_t)frame_bytes;
    }

    aom_img_free(&img);
    aom_codec_destroy(&ctx);

    if (rc == 0)
        *out_size = (int)(nframes * sizeof(uint32_t) + written);
    return rc;
}

static int csvs__av1_decompress(const void* src, int src_size,
                                uint8_t* frames, uint32_t w, uint32_t h,
                                uint32_t nframes)
{
    Dav1dSettings settings;
    dav1d_default_settings(&settings);
    settings.n_threads = 1;

    Dav1dContext* ctx = NULL;
    if (dav1d_open(&ctx, &settings) != 0) return -1;

    const uint32_t* frame_sizes = (const uint32_t*)src;
    const uint8_t* payload = (const uint8_t*)src + nframes * sizeof(uint32_t);
    size_t frame_pixels = (size_t)w * h;
    size_t offset = 0;
    int rc = 0;

    for (uint32_t f = 0; f < nframes; f++) {
        uint32_t fsz = frame_sizes[f];
        if (fsz == 0) {
            memset(frames + f * frame_pixels, 0, frame_pixels);
            continue;
        }

        Dav1dData data = {0};
        uint8_t* dbuf = dav1d_data_create(&data, fsz);
        if (!dbuf) { rc = -1; break; }
        memcpy(dbuf, payload + offset, fsz);
        offset += fsz;

        int res = dav1d_send_data(ctx, &data);
        while (res == DAV1D_ERR(EAGAIN)) {
            Dav1dPicture pic = {0};
            dav1d_get_picture(ctx, &pic);
            dav1d_picture_unref(&pic);
            res = dav1d_send_data(ctx, &data);
        }

        Dav1dPicture pic = {0};
        res = dav1d_get_picture(ctx, &pic);
        if (res == 0) {
            const uint8_t* y = (const uint8_t*)pic.data[0];
            ptrdiff_t stride = pic.stride[0];
            for (uint32_t row = 0; row < h; row++)
                memcpy(frames + f * frame_pixels + row * w,
                       y + row * stride, w);
            dav1d_picture_unref(&pic);
        } else {
            memset(frames + f * frame_pixels, 0, frame_pixels);
        }
    }

    for (;;) {
        Dav1dPicture pic = {0};
        if (dav1d_get_picture(ctx, &pic) != 0) break;
        dav1d_picture_unref(&pic);
    }

    dav1d_close(&ctx);
    return rc;
}

#endif /* CSVS_HAS_AV1 */

/* ---- codec dispatch --------------------------------------------------- */

static int csvs__compress_bound(const csvs_volume* vol, int src_size)
{
    switch (vol->codec) {
    case CSVS_CODEC_LZ4:
        return LZ4_compressBound(src_size);
#ifdef CSVS_HAS_ZSTD
    case CSVS_CODEC_ZSTD:
        return (int)ZSTD_compressBound((size_t)src_size);
#endif
    case CSVS_CODEC_H264:
    case CSVS_CODEC_H265:
    case CSVS_CODEC_AV1:
        return src_size * 2 + (int)vol->chunk_size * 4;
    default:
        return src_size * 2;
    }
}

static int csvs__compress(const csvs_volume* vol,
                          const void* src, int src_size,
                          void* dst, int dst_cap)
{
    switch (vol->codec) {
    case CSVS_CODEC_LZ4:
        return LZ4_compress_fast((const char*)src, (char*)dst,
                                 src_size, dst_cap, vol->codec_level);
#ifdef CSVS_HAS_ZSTD
    case CSVS_CODEC_ZSTD: {
        size_t r = ZSTD_compress(dst, (size_t)dst_cap, src, (size_t)src_size,
                                 vol->codec_level);
        return ZSTD_isError(r) ? -1 : (int)r;
    }
#endif
#ifdef CSVS_HAS_H264
    case CSVS_CODEC_H264: {
        uint32_t cs = vol->chunk_size;
        int out_size = 0;
        int rc = csvs__h264_compress((const uint8_t*)src, cs, cs, cs,
                                     vol->codec_level, dst, dst_cap, &out_size);
        return rc == 0 ? out_size : -1;
    }
#endif
#ifdef CSVS_HAS_H265
    case CSVS_CODEC_H265: {
        uint32_t cs = vol->chunk_size;
        int out_size = 0;
        int rc = csvs__h265_compress((const uint8_t*)src, cs, cs, cs,
                                     vol->codec_level, dst, dst_cap, &out_size);
        return rc == 0 ? out_size : -1;
    }
#endif
#ifdef CSVS_HAS_AV1
    case CSVS_CODEC_AV1: {
        uint32_t cs = vol->chunk_size;
        int out_size = 0;
        int rc = csvs__av1_compress((const uint8_t*)src, cs, cs, cs,
                                    vol->codec_level, dst, dst_cap, &out_size);
        return rc == 0 ? out_size : -1;
    }
#endif
    default:
        return -1;
    }
}

static int csvs__decompress(const csvs_volume* vol,
                            const void* src, int src_size,
                            void* dst, int dst_size)
{
    switch (vol->codec) {
    case CSVS_CODEC_LZ4:
        return LZ4_decompress_safe((const char*)src, (char*)dst,
                                   src_size, dst_size) > 0 ? 0 : -1;
#ifdef CSVS_HAS_ZSTD
    case CSVS_CODEC_ZSTD: {
        size_t r = ZSTD_decompress(dst, (size_t)dst_size, src, (size_t)src_size);
        return ZSTD_isError(r) ? -1 : 0;
    }
#endif
#ifdef CSVS_HAS_H264
    case CSVS_CODEC_H264: {
        uint32_t cs = vol->chunk_size;
        return csvs__h264_decompress(src, src_size, (uint8_t*)dst, cs, cs, cs);
    }
#endif
#ifdef CSVS_HAS_H265
    case CSVS_CODEC_H265: {
        uint32_t cs = vol->chunk_size;
        return csvs__h265_decompress(src, src_size, (uint8_t*)dst, cs, cs, cs);
    }
#endif
#ifdef CSVS_HAS_AV1
    case CSVS_CODEC_AV1: {
        uint32_t cs = vol->chunk_size;
        return csvs__av1_decompress(src, src_size, (uint8_t*)dst, cs, cs, cs);
    }
#endif
    default:
        return -1;
    }
}

/* ---- zero check ------------------------------------------------------- */

static int csvs__is_zero(const void* buf, size_t len)
{
    const uint8_t* p = (const uint8_t*)buf;
    for (size_t i = 0; i < len; i++)
        if (p[i]) return 0;
    return 1;
}

/* ---- Core API --------------------------------------------------------- */

int csvs_open(csvs_volume* vol, const char* path)
{
    memset(vol, 0, sizeof(*vol));
    snprintf(vol->path, sizeof(vol->path), "%s", path);
    vol->codec_level = 1;
    int rc = csvs__parse_meta(vol);
    if (rc == 0) csvs__init_cache(vol);
    return rc;
}

int csvs_create(csvs_volume* vol, const char* path,
                const size_t shape[3], uint32_t chunk_size,
                uint32_t shard_size,
                csvs_codec codec, int codec_level)
{
    memset(vol, 0, sizeof(*vol));
    snprintf(vol->path, sizeof(vol->path), "%s", path);
    memcpy(vol->shape, shape, 3 * sizeof(size_t));
    vol->chunk_size = chunk_size;
    vol->shard_size = shard_size;
    vol->chunks_per_shard = shard_size / chunk_size;
    vol->codec = codec;
    vol->codec_level = codec_level;

    for (int i = 0; i < 3; i++)
        vol->padded_shape[i] = csvs__round_up(shape[i], shard_size);

    mkdir(path, 0755);
    char sdir[1088];
    csvs__shard_dir(sdir, sizeof(sdir), path);
    mkdir(sdir, 0755);

    int rc = csvs__write_meta(vol);
    if (rc == 0) csvs__init_cache(vol);
    return rc;
}

int csvs_read_chunk(const csvs_volume* vol,
                    size_t cz, size_t cy, size_t cx, void* buf)
{
    uint32_t cps = vol->chunks_per_shard;
    size_t sz = cz / cps, sy = cy / cps, sx = cx / cps;
    size_t lz = cz % cps, ly = cy % cps, lx = cx % cps;

    size_t n = csvs__chunks_in_shard(vol);
    size_t idx_bytes = n * CSVS__INDEX_ENTRY;
    size_t cs = vol->chunk_size;
    size_t raw_size = cs * cs * cs;

    const csvs__shard_map* m = csvs__mmap_shard(vol, sz, sy, sx);
    if (m) {
        if (m->len < idx_bytes) return -1;

        size_t data_end = m->len - idx_bytes;
        const uint64_t* index = (const uint64_t*)((const char*)m->data + data_end);
        size_t ci = csvs__chunk_index_in_shard(vol, lz, ly, lx);
        uint64_t off = index[ci * 2];
        uint64_t nbytes = index[ci * 2 + 1];

        if (off == CSVS__MISSING_OFF && nbytes == CSVS__MISSING_LEN) {
            memset(buf, 0, raw_size);
            return 0;
        }

        if (off + nbytes > data_end) return -1;

        const void* comp = (const char*)m->data + off;
        return csvs__decompress(vol, comp, (int)nbytes, buf, (int)raw_size);
    }

    memset(buf, 0, raw_size);
    return 0;
}

int csvs_write_chunk(const csvs_volume* vol,
                     size_t cz, size_t cy, size_t cx,
                     const void* buf, size_t raw_size)
{
    uint32_t cps = vol->chunks_per_shard;
    size_t sz = cz / cps, sy = cy / cps, sx = cx / cps;
    size_t lz = cz % cps, ly = cy % cps, lx = cx % cps;

    char spath[1200];
    csvs__shard_path(spath, sizeof(spath), vol->path, sz, sy, sx);

    size_t n = csvs__chunks_in_shard(vol);
    size_t idx_bytes = n * CSVS__INDEX_ENTRY;
    uint64_t* index = (uint64_t*)malloc(idx_bytes);
    if (!index) return -1;

    size_t data_end;
    if (csvs__read_index(vol, spath, index, &data_end) != 0) {
        free(index);
        return -1;
    }

    csvs__cache_invalidate(vol, sz, sy, sx);

    size_t ci = csvs__chunk_index_in_shard(vol, lz, ly, lx);

    /* If the chunk is all zero, mark it missing instead of storing it */
    if (csvs__is_zero(buf, raw_size)) {
        index[ci * 2]     = CSVS__MISSING_OFF;
        index[ci * 2 + 1] = CSVS__MISSING_LEN;

        /* If all chunks are now missing, delete the shard file */
        int all_missing = 1;
        for (size_t i = 0; i < n; i++) {
            if (!(index[i * 2] == CSVS__MISSING_OFF &&
                  index[i * 2 + 1] == CSVS__MISSING_LEN)) {
                all_missing = 0;
                break;
            }
        }
        if (all_missing) {
            free(index);
            unlink(spath);
        } else {
            /* Rewrite index in place with this chunk marked missing */
            int fd = open(spath, O_RDWR);
            if (fd < 0) { free(index); return -1; }
            ssize_t w = pwrite(fd, index, idx_bytes, (off_t)data_end);
            close(fd);
            free(index);
            if (w < 0 || (size_t)w != idx_bytes) return -1;
        }
        return 0;
    }

    int bound = csvs__compress_bound(vol, (int)raw_size);
    char* comp = (char*)malloc((size_t)bound);
    if (!comp) { free(index); return -1; }

    int comp_size = csvs__compress(vol, buf, (int)raw_size, comp, bound);
    if (comp_size <= 0) { free(comp); free(index); return -1; }

    index[ci * 2]     = (uint64_t)data_end;
    index[ci * 2 + 1] = (uint64_t)comp_size;

    int fd = open(spath, O_RDWR | O_CREAT, 0644);
    if (fd < 0) { free(comp); free(index); return -1; }

    ssize_t w1 = pwrite(fd, comp, (size_t)comp_size, (off_t)data_end);
    free(comp);
    if (w1 < 0 || (size_t)w1 != (size_t)comp_size) {
        free(index);
        close(fd);
        return -1;
    }

    size_t new_data_end = data_end + (size_t)comp_size;
    ssize_t w2 = pwrite(fd, index, idx_bytes, (off_t)new_data_end);
    free(index);
    if (w2 < 0 || (size_t)w2 != idx_bytes) {
        close(fd);
        return -1;
    }

    if (ftruncate(fd, (off_t)(new_data_end + idx_bytes)) != 0) {
        close(fd);
        return -1;
    }
    close(fd);
    return 0;
}

int csvs_write_shard(const csvs_volume* vol,
                     size_t sz, size_t sy, size_t sx,
                     const void* chunks, const uint8_t* mask,
                     size_t raw_chunk_size)
{
    size_t n = csvs__chunks_in_shard(vol);
    size_t idx_bytes = n * CSVS__INDEX_ENTRY;
    const char* src_bytes = (const char*)chunks;

    /* Check if all present chunks are zero — if so, delete shard */
    int any_nonzero = 0;
    for (size_t i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        if (!csvs__is_zero(src_bytes + i * raw_chunk_size, raw_chunk_size)) {
            any_nonzero = 1;
            break;
        }
    }

    char spath[1200];
    csvs__shard_path(spath, sizeof(spath), vol->path, sz, sy, sx);
    csvs__cache_invalidate(vol, sz, sy, sx);

    if (!any_nonzero) {
        unlink(spath);
        return 0;
    }

    uint64_t* index = (uint64_t*)malloc(idx_bytes);
    if (!index) return -1;

    int bound = csvs__compress_bound(vol, (int)raw_chunk_size);
    char* comp = (char*)malloc((size_t)bound);
    if (!comp) { free(index); return -1; }

    int fd = open(spath, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { free(comp); free(index); return -1; }

    size_t offset = 0;

    for (size_t i = 0; i < n; i++) {
        if ((mask && !mask[i]) ||
            csvs__is_zero(src_bytes + i * raw_chunk_size, raw_chunk_size)) {
            index[i * 2]     = CSVS__MISSING_OFF;
            index[i * 2 + 1] = CSVS__MISSING_LEN;
            continue;
        }
        int csz = csvs__compress(vol, src_bytes + i * raw_chunk_size,
                                 (int)raw_chunk_size, comp, bound);
        if (csz <= 0) { close(fd); free(comp); free(index); return -1; }

        ssize_t w = pwrite(fd, comp, (size_t)csz, (off_t)offset);
        if (w < 0 || (size_t)w != (size_t)csz) {
            close(fd); free(comp); free(index); return -1;
        }
        index[i * 2]     = (uint64_t)offset;
        index[i * 2 + 1] = (uint64_t)csz;
        offset += (size_t)csz;
    }

    ssize_t w = pwrite(fd, index, idx_bytes, (off_t)offset);
    close(fd);
    free(comp);
    free(index);
    return (w < 0 || (size_t)w != idx_bytes) ? -1 : 0;
}

int csvs_read_shard(const csvs_volume* vol,
                    size_t sz, size_t sy, size_t sx,
                    void* chunks, uint8_t* mask)
{
    size_t n = csvs__chunks_in_shard(vol);
    size_t idx_bytes = n * CSVS__INDEX_ENTRY;
    size_t cs = vol->chunk_size;
    size_t raw_chunk_size = cs * cs * cs;

    const csvs__shard_map* m = csvs__mmap_shard(vol, sz, sy, sx);
    if (!m) {
        if (mask) memset(mask, 0, n);
        memset(chunks, 0, n * raw_chunk_size);
        return 0;
    }

    if (m->len < idx_bytes) return -1;

    size_t data_end = m->len - idx_bytes;
    const uint64_t* index = (const uint64_t*)((const char*)m->data + data_end);
    char* out = (char*)chunks;

    for (size_t i = 0; i < n; i++) {
        uint64_t off = index[i * 2];
        uint64_t nbytes = index[i * 2 + 1];

        if (off == CSVS__MISSING_OFF && nbytes == CSVS__MISSING_LEN) {
            if (mask) mask[i] = 0;
            memset(out + i * raw_chunk_size, 0, raw_chunk_size);
            continue;
        }

        if (off + nbytes > data_end) return -1;

        const void* comp = (const char*)m->data + off;
        int rc = csvs__decompress(vol, comp, (int)nbytes,
                                  out + i * raw_chunk_size, (int)raw_chunk_size);
        if (rc != 0) return -1;
        if (mask) mask[i] = 1;
    }

    return 0;
}

int csvs_read_region(const csvs_volume* vol,
                     size_t z0, size_t y0, size_t x0,
                     size_t zn, size_t yn, size_t xn,
                     void* buf)
{
    size_t cs = vol->chunk_size;
    size_t chunk_voxels = cs * cs * cs;

    size_t oz = zn, oy = yn, ox = xn;
    memset(buf, 0, oz * oy * ox);

    size_t cz0 = z0 / cs, cy0 = y0 / cs, cx0 = x0 / cs;
    size_t cz1 = (z0 + zn + cs - 1) / cs;
    size_t cy1 = (y0 + yn + cs - 1) / cs;
    size_t cx1 = (x0 + xn + cs - 1) / cs;

    void* chunk_buf = malloc(chunk_voxels);
    if (!chunk_buf) return -1;

    uint32_t cps = vol->chunks_per_shard;

    for (size_t cz = cz0; cz < cz1; cz++) {
        for (size_t cy = cy0; cy < cy1; cy++) {
            for (size_t cx = cx0; cx < cx1; cx++) {
                size_t vz = cz * cs, vy = cy * cs, vx = cx * cs;

                size_t oz0 = vz > z0 ? vz : z0;
                size_t oy0 = vy > y0 ? vy : y0;
                size_t ox0 = vx > x0 ? vx : x0;
                size_t oz1 = vz + cs < z0 + zn ? vz + cs : z0 + zn;
                size_t oy1 = vy + cs < y0 + yn ? vy + cs : y0 + yn;
                size_t ox1 = vx + cs < x0 + xn ? vx + cs : x0 + xn;

                if (oz0 >= oz1 || oy0 >= oy1 || ox0 >= ox1) continue;

                size_t sz = cz / cps, sy = cy / cps, sx = cx / cps;
                size_t lz = cz % cps, ly = cy % cps, lx = cx % cps;

                size_t n = csvs__chunks_in_shard(vol);
                size_t idx_bytes = n * CSVS__INDEX_ENTRY;
                int have_data = 0;

                const csvs__shard_map* m = csvs__mmap_shard(vol, sz, sy, sx);
                if (m && m->len >= idx_bytes) {
                    size_t data_end = m->len - idx_bytes;
                    const uint64_t* index = (const uint64_t*)((const char*)m->data + data_end);
                    size_t ci = csvs__chunk_index_in_shard(vol, lz, ly, lx);
                    uint64_t off = index[ci * 2];
                    uint64_t nbytes = index[ci * 2 + 1];

                    if (!(off == CSVS__MISSING_OFF && nbytes == CSVS__MISSING_LEN) &&
                        off + nbytes <= data_end) {
                        const void* comp = (const char*)m->data + off;
                        if (csvs__decompress(vol, comp, (int)nbytes,
                                             chunk_buf, (int)chunk_voxels) == 0) {
                            have_data = 1;
                        }
                    }
                }

                if (!have_data) continue;

                size_t copy_x = ox1 - ox0;
                for (size_t z = oz0; z < oz1; z++) {
                    for (size_t y = oy0; y < oy1; y++) {
                        size_t src_off = (z - vz) * cs * cs +
                                         (y - vy) * cs +
                                         (ox0 - vx);
                        size_t dst_off = (z - z0) * oy * ox +
                                         (y - y0) * ox +
                                         (ox0 - x0);
                        memcpy((char*)buf + dst_off,
                               (const char*)chunk_buf + src_off, copy_x);
                    }
                }
            }
        }
    }

    free(chunk_buf);
    return 0;
}

void csvs_close(csvs_volume* vol)
{
    csvs__free_cache(vol);
    memset(vol, 0, sizeof(*vol));
}

#endif /* CSVS_IMPLEMENTATION */
