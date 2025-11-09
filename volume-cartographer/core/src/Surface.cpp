#include "vc/core/util/Surface.hpp"

#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
//TODO remove
#include <opencv2/highgui.hpp>

#include <unordered_map>
#include <nlohmann/json.hpp>
#include <system_error>
#include <cmath>
#include <limits>
#include <cerrno>
#include <algorithm>
#include <vector>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

// Use libtiff for BigTIFF; fall back to OpenCV if not present.
#include <tiffio.h>

void normalizeMaskChannel(cv::Mat& mask)
{
    if (mask.empty()) {
        return;
    }

    cv::Mat singleChannel;
    if (mask.channels() == 1) {
        singleChannel = mask;
    } else if (mask.channels() == 3) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGR2GRAY);
    } else if (mask.channels() == 4) {
        cv::cvtColor(mask, singleChannel, cv::COLOR_BGRA2GRAY);
    } else {
        std::vector<cv::Mat> channels;
        cv::split(mask, channels);
        if (!channels.empty()) {
            singleChannel = channels[0];
        } else {
            singleChannel = mask;
        }
    }

    if (singleChannel.depth() != CV_8U) {
        cv::Mat converted;
        singleChannel.convertTo(converted, CV_8U);
        singleChannel = converted;
    }

    mask = singleChannel;
}


void write_overlapping_json(const std::filesystem::path& seg_path, const std::set<std::string>& overlapping_names) {
    nlohmann::json overlap_json;
    overlap_json["overlapping"] = std::vector<std::string>(overlapping_names.begin(), overlapping_names.end());

    std::ofstream o(seg_path / "overlapping.json");
    o << std::setw(4) << overlap_json << std::endl;
}

std::set<std::string> read_overlapping_json(const std::filesystem::path& seg_path) {
    std::set<std::string> overlapping;
    std::filesystem::path json_path = seg_path / "overlapping.json";

    if (std::filesystem::exists(json_path)) {
        std::ifstream i(json_path);
        nlohmann::json overlap_json;
        i >> overlap_json;

        if (overlap_json.contains("overlapping")) {
            for (const auto& name : overlap_json["overlapping"]) {
                overlapping.insert(name.get<std::string>());
            }
        }
    }

    return overlapping;
}



//NOTE we have 3 coordiante systems. Nominal (voxel volume) coordinates, internal relative (ptr) coords (where _center is at 0/0) and internal absolute (_points) coordinates where the upper left corner is at 0/0.
cv::Vec3f internal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return internal + cv::Vec3f(nominal[0]*scale[0], nominal[1]*scale[1], nominal[2]);
}

cv::Vec3f nominal_loc(const cv::Vec3f &nominal, const cv::Vec3f &internal, const cv::Vec2f &scale)
{
    return nominal + cv::Vec3f(internal[0]/scale[0], internal[1]/scale[1], internal[2]);
}

Surface::~Surface()
{
    if (meta) {
        delete meta;
    }
}

//given origin and normal, return the normalized vector v which describes a point : origin + v which lies in the plane and maximizes v.x at the cost of v.y,v.z
cv::Vec3f vx_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    //impossible
    if (n[1] == 0 && n[2] == 0)
        return {0,0,0};

    //also trivial
    if (n[0] == 0)
        return {1,0,0};

    cv::Vec3f v = {1,0,0};

    if (n[1] == 0) {
        v[1] = 0;
        //either n1 or n2 must be != 0, see first edge case
        v[2] = -n[0]/n[2];
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    if (n[2] == 0) {
        //either n1 or n2 must be != 0, see first edge case
        v[1] = -n[0]/n[1];
        v[2] = 0;
        cv::normalize(v, v, 1,0, cv::NORM_L2);
        return v;
    }

    v[1] = -n[0]/(n[1]+n[2]);
    v[2] = v[1];
    cv::normalize(v, v, 1,0, cv::NORM_L2);

    return v;
}

cv::Vec3f vy_from_orig_norm(const cv::Vec3f &o, const cv::Vec3f &n)
{
    cv::Vec3f v = vx_from_orig_norm({o[1],o[0],o[2]}, {n[1],n[0],n[2]});
    return {v[1],v[0],v[2]};
}

void vxy_from_normal(cv::Vec3f orig, cv::Vec3f normal, cv::Vec3f &vx, cv::Vec3f &vy)
{
    vx = vx_from_orig_norm(orig, normal);
    vy = vy_from_orig_norm(orig, normal);

    //TODO will there be a jump around the midpoint?
    if (abs(vx[0]) >= abs(vy[1]))
        vy = cv::Mat(normal).cross(cv::Mat(vx));
    else
        vx = cv::Mat(normal).cross(cv::Mat(vy));

    //FIXME probably not the right way to normalize the direction?
    if (vx[0] < 0)
        vx *= -1;
    if (vy[1] < 0)
        vy *= -1;
}

cv::Vec3f rotateAroundAxis(const cv::Vec3f& vector, const cv::Vec3f& axis, float angle)
{
    if (std::abs(angle) <= std::numeric_limits<float>::epsilon()) {
        return vector;
    }

    cv::Vec3d axis_d(axis[0], axis[1], axis[2]);
    double axis_norm = cv::norm(axis_d);
    if (axis_norm == 0.0) {
        return vector;
    }
    axis_d /= axis_norm;

    cv::Mat R;
    cv::Mat rot_vec = (cv::Mat_<double>(3, 1)
        << axis_d[0] * static_cast<double>(angle),
           axis_d[1] * static_cast<double>(angle),
           axis_d[2] * static_cast<double>(angle));
    cv::Rodrigues(rot_vec, R);

    cv::Mat v = (cv::Mat_<double>(3, 1) << vector[0], vector[1], vector[2]);
    cv::Mat res = R * v;
    return cv::Vec3f(static_cast<float>(res.at<double>(0)),
                     static_cast<float>(res.at<double>(1)),
                     static_cast<float>(res.at<double>(2)));
}

inline float sdist(const cv::Vec3f &a, const cv::Vec3f &b)
{
    cv::Vec3f d = a-b;
    return d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
}

inline cv::Vec2f mul(const cv::Vec2f &a, const cv::Vec2f &b)
{
    return{a[0]*b[0],a[1]*b[1]};
}

// Overload for cv::Vec3f
float search_min_loc(const cv::Mat_<cv::Vec3f> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        return -1;
    }

    bool changed = true;
    cv::Vec3f val = at_int(points, loc);
    out = val;
    float best = sdist(val, tgt);
    float res;

    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    cv::Vec2f step = init_step;

    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);

            //just skip if out of bounds
            if (!boundary.contains(cv::Point(cand)))
                continue;

            val = at_int(points, cand);
            res = sdist(val, tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step[0] < min_step_x)
            break;
    }

    return sqrt(best);
}

// Overload for cv::Vec3d
float search_min_loc(const cv::Mat_<cv::Vec3d> &points, cv::Vec2f &loc, cv::Vec3f &out, cv::Vec3f tgt, cv::Vec2f init_step, float min_step_x)
{
    cv::Rect boundary(1,1,points.cols-2,points.rows-2);
    if (!boundary.contains(cv::Point(loc))) {
        out = {-1,-1,-1};
        return -1;
    }

    bool changed = true;
    cv::Vec3d val_d = at_int(points, loc);
    cv::Vec3f val = cv::Vec3f(val_d[0], val_d[1], val_d[2]);
    out = val;
    float best = sdist(val, tgt);
    float res;

    std::vector<cv::Vec2f> search = {{0,-1},{0,1},{-1,-1},{-1,0},{-1,1},{1,-1},{1,0},{1,1}};
    cv::Vec2f step = init_step;

    while (changed) {
        changed = false;

        for(auto &off : search) {
            cv::Vec2f cand = loc+mul(off,step);

            //just skip if out of bounds
            if (!boundary.contains(cv::Point(cand)))
                continue;

            val_d = at_int(points, cand);
            val = cv::Vec3f(val_d[0], val_d[1], val_d[2]);
            res = sdist(val, tgt);
            if (res < best) {
                changed = true;
                best = res;
                loc = cand;
                out = val;
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        changed = true;

        if (step[0] < min_step_x)
            break;
    }

    return sqrt(best);
}

// Overload for cv::Vec3f
float pointTo_(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    loc = cv::Vec2f(points.cols/2,points.rows/2);
    cv::Vec3f _out;

    cv::Vec2f step_small = {std::max(1.0f,scale),std::max(1.0f,scale)};
    float min_mul = std::min(0.1*points.cols/scale,0.1*points.rows/scale);
    cv::Vec2f step_large = {min_mul*scale,min_mul*scale};

    assert(points.cols > 3);
    assert(points.rows > 3);

    float dist = search_min_loc(points, loc, _out, tgt, step_small, scale*0.1);

    if (dist < th && dist >= 0) {
        return dist;
    }

    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(points.cols/scale+points.rows/scale);

    //FIXME is this excessive?
    int r_full = 0;
    for(int r=0;r<10*max_iters && r_full < max_iters;r++) {
        //FIXME skipn invalid init locs!
        loc = {1 + (rand() % (points.cols-3)), 1 + (rand() % (points.rows-3))};

        if (points(loc[1],loc[0])[0] == -1)
            continue;

        r_full++;

        float dist = search_min_loc(points, loc, _out, tgt, step_large, scale*0.1);

        if (dist < th && dist >= 0) {
            dist = search_min_loc(points, loc, _out, tgt, step_small, scale*0.1);
            return dist;
        } else if (dist >= 0 && dist < min_dist) {
            min_loc = loc;
            min_dist = dist;
        }
    }

    loc = min_loc;
    return min_dist;
}

// Overload for cv::Vec3d
float pointTo_(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    loc = cv::Vec2f(points.cols/2,points.rows/2);
    cv::Vec3f _out;

    cv::Vec2f step_small = {std::max(1.0f,scale),std::max(1.0f,scale)};
    float min_mul = std::min(0.1*points.cols/scale,0.1*points.rows/scale);
    cv::Vec2f step_large = {min_mul*scale,min_mul*scale};

    assert(points.cols > 3);
    assert(points.rows > 3);

    float dist = search_min_loc(points, loc, _out, tgt, step_small, scale*0.1);

    if (dist < th && dist >= 0) {
        return dist;
    }

    cv::Vec2f min_loc = loc;
    float min_dist = dist;
    if (min_dist < 0)
        min_dist = 10*(points.cols/scale+points.rows/scale);

    //FIXME is this excessive?
    int r_full = 0;
    for(int r=0;r<10*max_iters && r_full < max_iters;r++) {
        //FIXME skipn invalid init locs!
        loc = {1 + (rand() % (points.cols-3)), 1 + (rand() % (points.rows-3))};

        if (points(loc[1],loc[0])[0] == -1)
            continue;

        r_full++;

        float dist = search_min_loc(points, loc, _out, tgt, step_large, scale*0.1);

        if (dist < th && dist >= 0) {
            dist = search_min_loc(points, loc, _out, tgt, step_small, scale*0.1);
            return dist;
        } else if (dist >= 0 && dist < min_dist) {
            min_loc = loc;
            min_dist = dist;
        }
    }

    loc = min_loc;
    return min_dist;
}

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3d> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    return pointTo_(loc, points, tgt, th, max_iters, scale);
}

float pointTo(cv::Vec2f &loc, const cv::Mat_<cv::Vec3f> &points, const cv::Vec3f &tgt, float th, int max_iters, float scale)
{
    return pointTo_(loc, points, tgt, th, max_iters, scale);
}


QuadSurface *load_quad_from_tifxyz(const std::string &path, int flags)
{
    auto read_band_into = [](const std::filesystem::path& fpath,
                             cv::Mat_<cv::Vec3f>& points,
                             int channel,
                             int& outW, int& outH) -> void
    {
        TIFF* tif = TIFFOpen(fpath.string().c_str(), "r");
        if (!tif) {
            throw std::runtime_error("Failed to open TIFF: " + fpath.string());
        }

        // Basic geometry
        uint32_t W=0, H=0;
        if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &W) ||
            !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &H)) {
            TIFFClose(tif);
            throw std::runtime_error("TIFF missing width/height: " + fpath.string());
        }

        uint16_t spp = 1; TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
        if (spp != 1) {
            TIFFClose(tif);
            throw std::runtime_error("Expected 1 sample per pixel in " + fpath.string());
        }
        uint16_t bps = 0;  TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bps);
        uint16_t fmt = SAMPLEFORMAT_UINT; TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &fmt);
        const int bytesPer = (bps + 7) / 8;
        if (!(bps==8 || bps==16 || bps==32 || bps==64)) {
            TIFFClose(tif);
            throw std::runtime_error("Unsupported BitsPerSample in " + fpath.string());
        }

        // Allocate destination if first band
        if (points.empty()) {
            points.create(static_cast<int>(H), static_cast<int>(W));
            // Initialize as invalid
            for (int y=0;y<points.rows;++y)
                for (int x=0;x<points.cols;++x)
                    points(y,x) = cv::Vec3f(-1.f,-1.f,-1.f);
            outW = static_cast<int>(W);
            outH = static_cast<int>(H);
        } else {
            if (outW != static_cast<int>(W) || outH != static_cast<int>(H)) {
                TIFFClose(tif);
                throw std::runtime_error("Band size mismatch in " + fpath.string());
            }
        }

        auto to_float = [fmt,bps,bytesPer](const uint8_t* p)->float {
            float v=0.f;
            switch(fmt) {
                case SAMPLEFORMAT_IEEEFP:
                    if (bps==32) { std::memcpy(&v, p, 4); return v; }
                    if (bps==64) { double d=0.0; std::memcpy(&d, p, 8); return static_cast<float>(d); }
                    break;
                case SAMPLEFORMAT_UINT:
                {
                    if (bps==8)  { uint8_t  t=*p; return static_cast<float>(t); }
                    if (bps==16) { uint16_t t; std::memcpy(&t,p,2); return static_cast<float>(t); }
                    if (bps==32) { uint32_t t; std::memcpy(&t,p,4); return static_cast<float>(t); }
                } break;
                case SAMPLEFORMAT_INT:
                {
                    if (bps==8)  { int8_t  t=*reinterpret_cast<const int8_t*>(p); return static_cast<float>(t); }
                    if (bps==16) { int16_t t; std::memcpy(&t,p,2); return static_cast<float>(t); }
                    if (bps==32) { int32_t t; std::memcpy(&t,p,4); return static_cast<float>(t); }
                } break;
                default: break;
            }
            // Last resort: treat as 32-bit float bytes
            std::memcpy(&v, p, std::min(bytesPer, (int)sizeof(float)));
            return v;
        };

        if (TIFFIsTiled(tif)) {
            uint32_t tileW=0, tileH=0;
            TIFFGetField(tif, TIFFTAG_TILEWIDTH,  &tileW);
            TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileH);
            if (tileW==0 || tileH==0) {
                TIFFClose(tif);
                throw std::runtime_error("Invalid tile geometry in " + fpath.string());
            }
            const tmsize_t tileBytes = TIFFTileSize(tif);
            std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes));

            for (uint32_t y0=0; y0<H; y0+=tileH) {
                const uint32_t dy = std::min(tileH, H - y0);
                for (uint32_t x0=0; x0<W; x0+=tileW) {
                    const uint32_t dx = std::min(tileW, W - x0);
                    const ttile_t tidx = TIFFComputeTile(tif, x0, y0, 0, 0);
                    tmsize_t n = TIFFReadEncodedTile(tif, tidx, tileBuf.data(), tileBytes);
                    if (n < 0) {
                        // fill with zeros on read error
                        std::fill(tileBuf.begin(), tileBuf.end(), 0);
                    }
                    for (uint32_t ty=0; ty<dy; ++ty) {
                        const uint8_t* row = tileBuf.data() + (static_cast<size_t>(ty)*tileW*bytesPer);
                        for (uint32_t tx=0; tx<dx; ++tx) {
                            float fv = to_float(row + static_cast<size_t>(tx)*bytesPer);
                            cv::Vec3f& dst = points(static_cast<int>(y0+ty), static_cast<int>(x0+tx));
                            dst[channel] = fv;
                        }
                    }
                }
            }
        } else {
            const tmsize_t scanBytes = TIFFScanlineSize(tif);
            std::vector<uint8_t> scanBuf(static_cast<size_t>(scanBytes));
            for (uint32_t y=0; y<H; ++y) {
                if (TIFFReadScanline(tif, scanBuf.data(), y, 0) != 1) {
                    std::fill(scanBuf.begin(), scanBuf.end(), 0);
                }
                const uint8_t* row = scanBuf.data();
                for (uint32_t x=0; x<W; ++x) {
                    float fv = to_float(row + static_cast<size_t>(x)*bytesPer);
                    cv::Vec3f& dst = points(static_cast<int>(y), static_cast<int>(x));
                    dst[channel] = fv;
                }
            }
        }
        TIFFClose(tif);
    };

    // Read meta first (scale, uuid, etc.)
    std::ifstream meta_f((std::filesystem::path(path)/"meta.json").string());
    if (!meta_f.is_open() || !meta_f.good()) {
        throw std::runtime_error("Cannot open meta.json at: " + path);
    }
    nlohmann::json metadata = nlohmann::json::parse(meta_f);
    cv::Vec2f scale = {metadata["scale"][0].get<float>(), metadata["scale"][1].get<float>()};

    cv::Mat_<cv::Vec3f>* points = new cv::Mat_<cv::Vec3f>();
    int W=0, H=0;
    read_band_into(std::filesystem::path(path)/"x.tif", *points, 0, W, H);
    read_band_into(std::filesystem::path(path)/"y.tif", *points, 1, W, H);
    read_band_into(std::filesystem::path(path)/"z.tif", *points, 2, W, H);

    // Invalidate by z<=0
    for (int j=0;j<points->rows;++j)
        for (int i=0;i<points->cols;++i)
            if ((*points)(j,i)[2] <= 0.f)
                (*points)(j,i) = cv::Vec3f(-1.f,-1.f,-1.f);

    // Optional mask
    const std::filesystem::path maskPath = std::filesystem::path(path)/"mask.tif";
    if (!(flags & SURF_LOAD_IGNORE_MASK) && std::filesystem::exists(maskPath)) {
        TIFF* mtif = TIFFOpen(maskPath.string().c_str(), "r");
        if (mtif) {
            uint32_t mW=0, mH=0;
            TIFFGetField(mtif, TIFFTAG_IMAGEWIDTH, &mW);
            TIFFGetField(mtif, TIFFTAG_IMAGELENGTH, &mH);
            if (mW!=0 && mH!=0) {
                uint16_t bps=0, fmt=SAMPLEFORMAT_UINT;
                TIFFGetFieldDefaulted(mtif, TIFFTAG_BITSPERSAMPLE, &bps);
                TIFFGetFieldDefaulted(mtif, TIFFTAG_SAMPLEFORMAT, &fmt);
                uint16_t spp=1;
                TIFFGetFieldDefaulted(mtif, TIFFTAG_SAMPLESPERPIXEL, &spp);
                uint16_t planarConfig = PLANARCONFIG_CONTIG;
                TIFFGetFieldDefaulted(mtif, TIFFTAG_PLANARCONFIG, &planarConfig);
                const int bytesPer = (bps+7)/8;
                const uint16_t samplesPerPixel = std::max<uint16_t>(1, spp);
                const bool isPlanarSeparate = (planarConfig == PLANARCONFIG_SEPARATE);
                const size_t pixelStride = static_cast<size_t>(bytesPer) *
                                           static_cast<size_t>(isPlanarSeparate ? 1 : samplesPerPixel);
                auto computeScaleFactor = [](uint32_t maskDim, int targetDim) -> uint32_t {
                    if (maskDim == static_cast<uint32_t>(targetDim)) {
                        return 1;
                    } else if (targetDim > 0 && (maskDim % static_cast<uint32_t>(targetDim)) == 0) {
                        return maskDim / static_cast<uint32_t>(targetDim);
                    } else {
                        return 0;
                    }
                };
                const uint32_t scaleX = computeScaleFactor(mW, W);
                const uint32_t scaleY = computeScaleFactor(mH, H);
                if (scaleX != 0 && scaleY != 0) {
                    // Channel 0 encodes mask validity; later channels remain untouched.
                    const double retainThreshold = [&]{
                        switch(fmt) {
                            case SAMPLEFORMAT_UINT:
                            case SAMPLEFORMAT_INT:
                            case SAMPLEFORMAT_IEEEFP:
                                return 255.0;
                        }
                        return 255.0;
                    }();
                    auto to_valid = [fmt,bps,bytesPer,retainThreshold](const uint8_t* p)->bool{
                        switch(fmt) {
                            case SAMPLEFORMAT_IEEEFP:
                                if (bps==32) { float v; std::memcpy(&v,p,4); return v>=static_cast<float>(retainThreshold); }
                                if (bps==64) { double d; std::memcpy(&d,p,8); return d>=retainThreshold; }
                                break;
                            case SAMPLEFORMAT_UINT:
                                if (bps==8)  return (*p)>=retainThreshold;
                                if (bps==16) { uint16_t t; std::memcpy(&t,p,2); return t>=static_cast<uint16_t>(retainThreshold); }
                                if (bps==32) { uint32_t t; std::memcpy(&t,p,4); return t>=static_cast<uint32_t>(retainThreshold); }
                                break;
                            case SAMPLEFORMAT_INT:
                                if (bps==8)  { int8_t t=*reinterpret_cast<const int8_t*>(p);  return t>=static_cast<int8_t>(retainThreshold); }
                                if (bps==16) { int16_t t; std::memcpy(&t,p,2); return t>=static_cast<int16_t>(retainThreshold); }
                                if (bps==32) { int32_t t; std::memcpy(&t,p,4); return t>=static_cast<int32_t>(retainThreshold); }
                                break;
                        }
                        // default
                        return (*p)>=retainThreshold;
                    };
                    const cv::Vec3f invalidPoint(-1.f,-1.f,-1.f);
                    auto invalidate = [&](uint32_t srcX, uint32_t srcY){
                        const uint32_t dstX = (scaleX == 1) ? srcX : (srcX / scaleX);
                        const uint32_t dstY = (scaleY == 1) ? srcY : (srcY / scaleY);
                        if (dstX < static_cast<uint32_t>(W) && dstY < static_cast<uint32_t>(H)) {
                            (*points)(static_cast<int>(dstY), static_cast<int>(dstX)) = invalidPoint;
                        }
                    };
                    if (TIFFIsTiled(mtif)) {
                        uint32_t tileW=0, tileH=0;
                        TIFFGetField(mtif, TIFFTAG_TILEWIDTH,  &tileW);
                        TIFFGetField(mtif, TIFFTAG_TILELENGTH, &tileH);
                        const tmsize_t tileBytes = TIFFTileSize(mtif);
                        std::vector<uint8_t> tileBuf(static_cast<size_t>(tileBytes));
                        for (uint32_t y0=0; y0<mH; y0+=tileH) {
                            const uint32_t dy = std::min(tileH, mH - y0);
                            for (uint32_t x0=0; x0<mW; x0+=tileW) {
                                const uint32_t dx = std::min(tileW, mW - x0);
                                const ttile_t tidx = TIFFComputeTile(mtif, x0, y0, 0, 0);
                                tmsize_t n = TIFFReadEncodedTile(mtif, tidx, tileBuf.data(), tileBytes);
                                if (n < 0) std::fill(tileBuf.begin(), tileBuf.end(), 0);
                                const size_t tileRowStride = static_cast<size_t>(tileW) * pixelStride;
                                for (uint32_t ty=0; ty<dy; ++ty) {
                                    const uint8_t* row = tileBuf.data() + static_cast<size_t>(ty) * tileRowStride;
                                    for (uint32_t tx=0; tx<dx; ++tx) {
                                        const uint8_t* px = row + static_cast<size_t>(tx) * pixelStride;
                                        if (!to_valid(px)) {
                                            invalidate(x0 + tx, y0 + ty);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        const tmsize_t scanBytes = TIFFScanlineSize(mtif);
                        std::vector<uint8_t> scanBuf(static_cast<size_t>(scanBytes));
                        for (uint32_t y=0; y<mH; ++y) {
                            if (TIFFReadScanline(mtif, scanBuf.data(), y, 0) != 1) {
                                std::fill(scanBuf.begin(), scanBuf.end(), 0);
                            }
                            const uint8_t* row = scanBuf.data();
                            for (uint32_t x=0; x<mW; ++x) {
                                const uint8_t* px = row + static_cast<size_t>(x) * pixelStride;
                                if (!to_valid(px)) {
                                    invalidate(x, y);
                                }
                            }
                        }
                    }
                }
            }
            TIFFClose(mtif);
        }
    }

    QuadSurface *surf = new QuadSurface(points, scale);
    surf->path = path;
    surf->id   = metadata["uuid"];
    surf->meta = new nlohmann::json(metadata);

    // Register extra channels lazily (left as OpenCV-based on-demand load).
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.path().extension() == ".tif") {
            std::string filename = entry.path().stem().string();
            if (filename != "x" && filename != "y" && filename != "z") {
                surf->setChannel(filename, cv::Mat());
            }
        }
    }
    return surf;
}

Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p)
{
    Rect3D res = a;
    for(int d=0;d<3;d++) {
        res.low[d] = std::min(res.low[d], p[d]);
        res.high[d] = std::max(res.high[d], p[d]);
    }

    return res;
}


bool intersect(const Rect3D &a, const Rect3D &b)
{
    for(int d=0;d<3;d++) {
        if (a.high[d] < b.low[d])
            return false;
        if (a.low[d] > b.high[d])
            return false;
    }

    return true;
}

Rect3D rect_from_json(const nlohmann::json &json)
{
    return {{json[0][0],json[0][1],json[0][2]},{json[1][0],json[1][1],json[1][2]}};
}



uint64_t MultiSurfaceIndex::hash(int x, int y, int z) const {
    // Ensure non-negative values for hashing
    uint32_t ux = static_cast<uint32_t>(x + 1000000);
    uint32_t uy = static_cast<uint32_t>(y + 1000000);
    uint32_t uz = static_cast<uint32_t>(z + 1000000);
    return (static_cast<uint64_t>(ux) << 40) |
           (static_cast<uint64_t>(uy) << 20) |
           static_cast<uint64_t>(uz);
}

MultiSurfaceIndex::MultiSurfaceIndex(float cell_sz) : cell_size(cell_sz) {}

void MultiSurfaceIndex::addPatch(int idx, QuadSurface* patch) {
    Rect3D bbox = patch->bbox();
    patch_bboxes.push_back(bbox);

    // Expand bbox slightly to handle edge cases
    int x0 = std::floor((bbox.low[0] - cell_size) / cell_size);
    int y0 = std::floor((bbox.low[1] - cell_size) / cell_size);
    int z0 = std::floor((bbox.low[2] - cell_size) / cell_size);
    int x1 = std::ceil((bbox.high[0] + cell_size) / cell_size);
    int y1 = std::ceil((bbox.high[1] + cell_size) / cell_size);
    int z1 = std::ceil((bbox.high[2] + cell_size) / cell_size);

    for (int z = z0; z <= z1; z++) {
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                grid[hash(x, y, z)].patch_indices.push_back(idx);
            }
        }
    }
}

std::vector<int> MultiSurfaceIndex::getCandidatePatches(const cv::Vec3f& point, float tolerance) const {
    // Get the cell containing this point
    int x = std::floor(point[0] / cell_size);
    int y = std::floor(point[1] / cell_size);
    int z = std::floor(point[2] / cell_size);

    // If tolerance is specified, check neighboring cells too
    std::set<int> unique_patches;

    if (tolerance > 0) {
        int cell_radius = std::ceil(tolerance / cell_size);
        for (int dz = -cell_radius; dz <= cell_radius; dz++) {
            for (int dy = -cell_radius; dy <= cell_radius; dy++) {
                for (int dx = -cell_radius; dx <= cell_radius; dx++) {
                    auto it = grid.find(hash(x + dx, y + dy, z + dz));
                    if (it != grid.end()) {
                        for (int idx : it->second.patch_indices) {
                            unique_patches.insert(idx);
                        }
                    }
                }
            }
        }
    } else {
        auto it = grid.find(hash(x, y, z));
        if (it != grid.end()) {
            for (int idx : it->second.patch_indices) {
                unique_patches.insert(idx);
            }
        }
    }

    // Filter by bounding box for extra safety
    std::vector<int> result;
    for (int idx : unique_patches) {
        const Rect3D& bbox = patch_bboxes[idx];
        if (point[0] >= bbox.low[0] - tolerance &&
            point[0] <= bbox.high[0] + tolerance &&
            point[1] >= bbox.low[1] - tolerance &&
            point[1] <= bbox.high[1] + tolerance &&
            point[2] >= bbox.low[2] - tolerance &&
            point[2] <= bbox.high[2] + tolerance) {
            result.push_back(idx);
        }
    }

    return result;
}

size_t MultiSurfaceIndex::getCellCount() const { return grid.size(); }
size_t MultiSurfaceIndex::getPatchCount() const { return patch_bboxes.size(); }
