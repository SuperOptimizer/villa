#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/filesystem/handle.hxx"
#include "z5/factory.hxx"

#include <xtensor/xview.hpp>

#include <opencv2/highgui.hpp>

struct passTroughComputor
{
    enum {BORDER = 20};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    template <typename T> void compute(const T &large, T &small)
    {
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

int main(int argc, char *argv[])
{
    assert(argc == 2);

    const char *vol_path = argv[1];
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);
    
    passTroughComputor compute;

    ChunkCache chunk_cache(10e9);

    Chunked3d<uint8_t,passTroughComputor> proc_tensor(compute, ds.get(), &chunk_cache);

    int w = 4000;
    int h = 4000;

    cv::Mat_<uint8_t> img(cv::Size(w,h),0);
    for(int j=1000;j<2000;j++) {
        for(int i=1000;i<2000;i++)
            img(j,i) = proc_tensor(4000,j-10,i-10);
    }

    cv::imwrite("slice.tif", img);

    
    return EXIT_SUCCESS;
}
