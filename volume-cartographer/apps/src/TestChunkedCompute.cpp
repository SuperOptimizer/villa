#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/filesystem/handle.hxx"
#include "z5/factory.hxx"

#include <xtensor/xview.hpp>

#include <opencv2/highgui.hpp>

struct passTroughComputor
{
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    template <typename T, typename E> void compute(const T &large, T &small)
    {
        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(large, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

struct thresholdedDistance
{
    enum {BORDER = 0};
    enum {CHUNK_SIZE = 32};
    enum {FILL_V = 0};
    template <typename T, typename E> void compute(const T &large, T &small)
    {
        T c1 = xt::empty<E>(large.shape());
        T c2 = xt::empty<E>(large.shape());

        c1 = large;

        T &p1 = c1;
        T &p2 = c2;

        int s = CHUNK_SIZE+2*BORDER;

#pragma omp parallel for
        for(int z=0;z<s;z++)
            for(int y=0;y<s;y++)
                for(int x=0;x<s;x++)
                    if (p1(z,y,x) >= 50)
                        p1(z,y,x) = 1;
                    else
                        p1(z,y,x) = 0;


        int low = int(BORDER);
        int high = int(BORDER)+int(CHUNK_SIZE);
        small = view(p1, xt::range(low,high),xt::range(low,high),xt::range(low,high));
    }
};

int main(int argc, char *argv[])
{
    assert(argc == 2);

    const char *vol_path = argv[1];
    std::string group_name = "0";
    z5::filesystem::handle::Group group(vol_path, z5::FileMode::FileMode::r);
    z5::filesystem::handle::Dataset ds_handle(group, group_name, nlohmann::json::parse(std::ifstream(std::filesystem::path(vol_path)/group_name/".zarray")).value<>("dimension_separator","."));
    std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

    std::cout << ds.get()->shape() << std::endl;
    
    passTroughComputor compute;

    ChunkCache chunk_cache(10e9);

    Chunked3d<uint8_t,passTroughComputor> proc_tensor(compute, ds.get(), &chunk_cache);

    int w = ds.get()->shape(2);
    int h = ds.get()->shape(1);

    cv::Mat_<uint8_t> img(cv::Size(w,h),0);
    // Chunked3dAccessor acc(proc_tensor);
    // for(int j=0;j<h-32;j+=32) {
    //     std::cout << j << std::endl;
    //     for(int i=0;i<w-32;i+=32)
    //         for(int y=j;y<j+32;y++)
    //             for(int x=i;x<i+32;x++)
    //                 img(y,x) = acc(4000,y,x);
    //     for(int j=0;j<h;j++) {
    //         std::cout << j << std::endl;
    //         for(int i=0;i<w;i++)
    //             img(j,i) = acc(4000,j,i);
    // }
#pragma omp parallel
    {
        Chunked3dAccessor acc(proc_tensor);
#pragma omp for
        for(int j=0;j<h;j++) {
            for(int i=0;i<w;i++)
                img(j,i) = acc.safe_at(2000,j,i);
        }

    }

    cv::imwrite("slice.tif", img);

    
    return EXIT_SUCCESS;
}
