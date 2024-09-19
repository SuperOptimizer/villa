#include "vc/core/types/Volume.hpp"

#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <jxl/codestream_header.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/resizable_parallel_runner_cxx.h>
#include <jxl/types.h>

#include "vc/core/io/TIFFIO.hpp"

#include "z5/attributes.hxx"
#include "z5/dataset.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/metadata.hxx"
#include "z5/handle.hxx"
#include "z5/types/types.hxx"
#include "z5/util/util.hxx"
#include "z5/util/blocking.hxx"
#include "z5/util/format_data.hxx"
#include "z5/factory.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "xtensor/xarray.hpp"

namespace fs = volcart::filesystem;
namespace tio = volcart::tiffio;

using namespace volcart;

// Load a Volume from disk
Volume::Volume(fs::path path) : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    width_ = metadata_.get<int>("width");
    height_ = metadata_.get<int>("height");
    slices_ = metadata_.get<int>("slices");
    numSliceCharacters_ = std::to_string(slices_).size();

    std::vector<std::mutex> init_mutexes(slices_);

    slice_mutexes_.swap(init_mutexes);
    
    zarrOpen();
}

// Setup a Volume from a folder of slices
Volume::Volume(fs::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name)),
          slice_mutexes_(slices_)
{
    metadata_.set("type", "vol");
    metadata_.set("width", width_);
    metadata_.set("height", height_);
    metadata_.set("slices", slices_);
    metadata_.set("voxelsize", double{});
    metadata_.set("min", double{});
    metadata_.set("max", double{});    

    zarrOpen();
}

void Volume::zarrOpen()
{
    if (metadata_.hasKey("format") && metadata_.get<std::string>("format") == "zarr") {
        std::cout << "zarropen" << "\n";
        isZarr = true;
        zarrFile_ = new z5::filesystem::handle::File(path_);
        z5::filesystem::handle::Group group(path_, z5::FileMode::FileMode::r);
        z5::readAttributes(group, zarrGroup_);
        
        // z5::filesystem::handle::Dataset ds_handle(fs::path(path_ / "1"), z5::FileMode::FileMode::r, "/");
        z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
        zarrDs_ = z5::filesystem::openDataset(ds_handle);
    }
}

// Load a Volume from disk, return a pointer
auto Volume::New(fs::path path) -> Volume::Pointer
{
    return std::make_shared<Volume>(path);
}

// Set a Volume from a folder of slices, return a pointer
auto Volume::New(fs::path path, std::string uuid, std::string name)
    -> Volume::Pointer
{
    return std::make_shared<Volume>(path, uuid, name);
}

auto Volume::sliceWidth() const -> int { return width_; }
auto Volume::sliceHeight() const -> int { return height_; }
auto Volume::numSlices() const -> int { return slices_; }
auto Volume::voxelSize() const -> double
{
    return metadata_.get<double>("voxelsize");
}
auto Volume::min() const -> double { return metadata_.get<double>("min"); }
auto Volume::max() const -> double { return metadata_.get<double>("max"); }

void Volume::setSliceWidth(int w)
{
    width_ = w;
    metadata_.set("width", w);
}

void Volume::setSliceHeight(int h)
{
    height_ = h;
    metadata_.set("height", h);
}

void Volume::setNumberOfSlices(std::size_t numSlices)
{
    slices_ = numSlices;
    numSliceCharacters_ = std::to_string(numSlices).size();
    metadata_.set("slices", numSlices);
}

void Volume::setVoxelSize(double s) { metadata_.set("voxelsize", s); }
void Volume::setMin(double m) { metadata_.set("min", m); }
void Volume::setMax(double m) { metadata_.set("max", m); }

auto Volume::bounds() const -> Volume::Bounds
{
    return {
        {0, 0, 0},
        {static_cast<double>(width_), static_cast<double>(height_),
         static_cast<double>(slices_)}};
}

auto Volume::isInBounds(double x, double y, double z) const -> bool
{
    return x >= 0 && x < width_ && y >= 0 && y < height_ && z >= 0 &&
           z < slices_;
}

auto Volume::isInBounds(const cv::Vec3d& v) const -> bool
{
    return isInBounds(v(0), v(1), v(2));
}

auto Volume::getSlicePath(int index) const -> fs::path
{
    if (metadata_.hasKey("img_ptn")) {
        char buf[64];
        snprintf(buf,64,metadata_.get<std::string>("img_ptn").c_str(), index);
        
        return path_ / buf;
    }
    else {
        std::stringstream ss;
        ss << std::setw(numSliceCharacters_) << std::setfill('0') << index
        << ".tif";
        return path_ / ss.str();
    }
}

auto Volume::getSliceData(int index) const -> cv::Mat
{
    if (cacheSlices_ && !isZarr) {
        return cache_slice_(index);
    }
    return load_slice_(index);
}

auto Volume::getSliceDataCopy(int index) const -> cv::Mat
{
    return getSliceData(index).clone();
}

auto Volume::getSliceDataRect(int index, cv::Rect rect) const -> cv::Mat
{
    auto whole_img = getSliceData(index);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return whole_img(rect);
}

auto Volume::getSliceDataRectCopy(int index, cv::Rect rect) const -> cv::Mat
{
    auto whole_img = getSliceData(index);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return whole_img(rect).clone();
}

void Volume::setSliceData(int index, const cv::Mat& slice, bool compress)
{
    if (isZarr)
        throw std::runtime_error("setSliceData() not supported for zarr volumes");
    
    auto slicePath = getSlicePath(index);
    tio::WriteTIFF(
        slicePath.string(), slice,
        (compress) ? tiffio::Compression::LZW : tiffio::Compression::NONE);
}

auto Volume::intensityAt(int x, int y, int z) const -> std::uint16_t
{
    // clang-format off
    if (x < 0 || x >= sliceWidth() ||
        y < 0 || y >= sliceHeight() ||
        z < 0 || z >= numSlices()) {
        return 0;
    }
    // clang-format on
    return getSliceData(z).at<std::uint16_t>(y, x);
}

// Trilinear Interpolation
// From: https://en.wikipedia.org/wiki/Trilinear_interpolation
auto Volume::interpolateAt(double x, double y, double z) const -> std::uint16_t
{
    // insert safety net
    if (!isInBounds(x, y, z)) {
        return 0;
    }

    double intPart;
    double dx = std::modf(x, &intPart);
    auto x0 = static_cast<int>(intPart);
    int x1 = x0 + 1;
    double dy = std::modf(y, &intPart);
    auto y0 = static_cast<int>(intPart);
    int y1 = y0 + 1;
    double dz = std::modf(z, &intPart);
    auto z0 = static_cast<int>(intPart);
    int z1 = z0 + 1;

    auto c00 =
        intensityAt(x0, y0, z0) * (1 - dx) + intensityAt(x1, y0, z0) * dx;
    auto c10 =
        intensityAt(x0, y1, z0) * (1 - dx) + intensityAt(x1, y0, z0) * dx;
    auto c01 =
        intensityAt(x0, y0, z1) * (1 - dx) + intensityAt(x1, y0, z1) * dx;
    auto c11 =
        intensityAt(x0, y1, z1) * (1 - dx) + intensityAt(x1, y1, z1) * dx;

    auto c0 = c00 * (1 - dy) + c10 * dy;
    auto c1 = c01 * (1 - dy) + c11 * dy;

    auto c = c0 * (1 - dz) + c1 * dz;
    return static_cast<std::uint16_t>(cvRound(c));
}

auto Volume::reslice(
    const cv::Vec3d& center,
    const cv::Vec3d& xvec,
    const cv::Vec3d& yvec,
    int width,
    int height) const -> Reslice
{
    auto xnorm = cv::normalize(xvec);
    auto ynorm = cv::normalize(yvec);
    auto origin = center - ((width / 2) * xnorm + (height / 2) * ynorm);

    cv::Mat m(height, width, CV_16UC1);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            m.at<std::uint16_t>(h, w) =
                interpolateAt(origin + (h * ynorm) + (w * xnorm));
        }
    }

    return Reslice(m, origin, xnorm, ynorm);
}

void throw_run_path(const fs::path &path, const std::string msg)
{
    throw std::runtime_error(msg + " for " + path.string());
}

cv::Mat read_jxl(const fs::path &path)
{
    //adapted from https://github.com/libjxl/libjxl/blob/main/examples/decode_oneshot.cc
    
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> data(file_size);
    if (!file.read(data.data(), file_size))
        throw_run_path(path, "read error");
    
    
    size_t w, h;
    cv::Mat_<uint8_t> img;
    
    // Multi-threaded parallel runner.
    auto runner = JxlResizableParallelRunnerMake(nullptr);
    auto dec = JxlDecoderMake(nullptr);
    
    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE))
        throw_run_path(path, "JxlDecoderSubscribeEvents Error");
        
    if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(), JxlResizableParallelRunner, runner.get()))
        throw_run_path(path, "JxlDecoderSetParallelRunner failed");
            
    JxlBasicInfo info;
    JxlPixelFormat format = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    //FIXME check format in the file and keep that?
            
    JxlDecoderSetInput(dec.get(), (uint8_t*)&data[0], file_size);
    JxlDecoderCloseInput(dec.get());
            
    for (;;) {
        JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
        
        if (status == JXL_DEC_ERROR)
            throw_run_path(path, "Decoder error\n");
        else if (status == JXL_DEC_NEED_MORE_INPUT)
            throw_run_path(path, "Error, already provided all input");
        else if (status == JXL_DEC_BASIC_INFO) {
            if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info))
                throw_run_path(path, "JxlDecoderGetBasicInfo failed");
            w = info.xsize;
            h = info.ysize;
            JxlResizableParallelRunnerSetThreads(runner.get(), JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
        } else if (status == JXL_DEC_COLOR_ENCODING) {
            std::cout << "Ignoring jxl ICC profile" << "\n";
        } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            size_t buffer_size;
            if (JXL_DEC_SUCCESS != JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size))
                throw_run_path(path, "JxlDecoderImageOutBufferSize failed");
            
            if (buffer_size != w * h)
                throw_run_path(path, "Invalid Buffer size");

            img.create(h, w);
            void* pixels_buffer = static_cast<void*>(img.ptr(0));
            if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels_buffer, buffer_size))
                throw_run_path(path, "JxlDecoderSetImageOutBuffer failed\n");

        } else if (status == JXL_DEC_FULL_IMAGE) {
            // Nothing to do. Do not yet return. If the image is an animation, more
            // full frames may be decoded. This example only keeps the last one.
        } else if (status == JXL_DEC_SUCCESS) {
            // All decoding successfully finished.
            // It's not required to call JxlDecoderReleaseInput(dec.get()) here since
            // the decoder will be destroyed.
            return img;
        } else {
            std::runtime_error("Unknown decoder Error");
        }
    }
}

std::ostream& operator<< (std::ostream& out, const xt::xarray<uint8_t>::shape_type &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b\b]"; // use two ANSI backspace characters '\b' to overwrite final ", "
    }
    return out;
}

auto Volume::load_slice_(int index) const -> cv::Mat
{
    {
        std::unique_lock<std::shared_mutex> lock(print_mutex_);
        std::cout << "Requested to load slice " << index << std::endl;
    }
    
    if (isZarr) {
        throw std::runtime_error("load_slice_ not implemented for Zarr");
    }
    
    auto slicePath = getSlicePath(index);
    cv::Mat mat;
    if (slicePath.extension() == ".tif") {
        try {
            mat = tio::ReadTIFF(slicePath.string());
        } catch (std::runtime_error) {
        }
    }
    else if (slicePath.extension() == ".jxl") {
        mat = read_jxl(slicePath);
        mat.convertTo(mat, CV_16UC1, 257);
    }
    else {
        mat = cv::imread(slicePath, cv::IMREAD_UNCHANGED);
    }
    
    if (!mat.empty() && (mat.size().width != sliceWidth() || mat.size().height != sliceHeight()))
    {
        cv::resize(mat, mat, cv::Size(sliceWidth(),sliceHeight()), 0,0, cv::INTER_CUBIC);
    }
    return mat;
}

auto Volume::cache_slice_(int index) const -> cv::Mat
{
    // Check if the slice is in the cache.
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (cache_->contains(index)) {
            return cache_->get(index);
        }
    }

    {
        // Get the lock for this slice.
        auto& mutex = slice_mutexes_[index];

        // If the slice is not in the cache, get exclusive access to this slice's mutex.
        std::unique_lock<std::mutex> lock(mutex);
        // Check again to ensure the slice has not been added to the cache while waiting for the lock.
        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex_);
            if (cache_->contains(index)) {
                return cache_->get(index);
            }
        }
        // Load the slice and add it to the cache.
        {
            auto slice = load_slice_(index);
            std::unique_lock<std::shared_mutex> lock(cache_mutex_);
            cache_->put(index, slice);
            return slice;
        }
    }

}


void Volume::cachePurge() const 
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    cache_->purge();
}

z5::Dataset *Volume::zarrDataset()
{
    return zarrDs_.get();
}
