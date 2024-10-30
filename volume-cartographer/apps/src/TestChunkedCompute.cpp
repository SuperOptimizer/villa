#include "vc/core/types/ChunkedTensor.hpp"

#include "z5/filesystem/handle.hxx"
#include "z5/factory.hxx"

struct passTroughComputor
{
    size_t border = 0;
    void compute(const xt::xarray<uint8_t> &large, xt::xarray<uint8_t> &small)
    {
        small = large;
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
    
    return EXIT_SUCCESS;
}
