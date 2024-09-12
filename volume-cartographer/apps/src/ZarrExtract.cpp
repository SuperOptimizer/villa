#include <nlohmann/json.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xio.hpp>

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include <opencv2/highgui.hpp>

using shape = z5::types::ShapeType;
using namespace xt::placeholders;

std::ostream& operator<< (std::ostream& out, const std::vector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

// std::ostream& operator<< (std::ostream& out, const xt::svector<size_t> &v) {
//     if ( !v.empty() ) {
//         out << '[';
//         for(auto &v : v)
//             out << v << ",";
//         out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
//     }
//     return out;
// }


shape chunkId(const std::unique_ptr<z5::Dataset> &ds, shape coord)
{
    shape div = ds->chunking().blockShape();
    shape id = coord;
    for(int i=0;i<id.size();i++)
        id[i] /= div[i];
    return id;
}

shape idCoord(const std::unique_ptr<z5::Dataset> &ds, shape id)
{
    shape mul = ds->chunking().blockShape();
    shape coord = id;
    for(int i=0;i<coord.size();i++)
        coord[i] *= mul[i];
    return coord;
}

void readInterpolated3D(xt::xarray<uint8_t> &out, std::unique_ptr<z5::Dataset> &ds, const xt::xarray<float> &coords)
{
    // auto dims = xt::range(_,coords.shape().size()-2);
    // std::vector<long int> da = dims;
    std::vector<int> dims;
    for(int i=0;i<coords.shape().size()-1;i++)
        dims.push_back(i);
    xt::xarray<float> upper = xt::amax(coords, dims);
    xt::xarray<float> lower = xt::amin(coords, dims);
    
    // std::cout << "maxshape" << .shape() << std::endl;
    
    shape offset(3);
    shape size(3);
    for(int i=0;i<3;i++) {
        offset[i] = lower[i];
        size[i] = ceil(upper[i]) - offset[i]+1;
    }
    std::cout << "offset" << offset << std::endl;
    std::cout << "size" << size << std::endl;
    
    xt::xarray<uint8_t> buf(size);
    
    z5::multiarray::readSubarray<uint8_t>(*ds, buf, offset.begin(), std::thread::hardware_concurrency());
    
    auto out_shape = coords.shape();
    out_shape.back() = 1;
    out = xt::empty<uint8_t>(out_shape);
    
    auto iter_coords = xt::axis_slice_begin(coords, 2);
    auto iter_out = xt::axis_slice_begin(out, 2);
    auto end_coords = xt::axis_slice_end(coords, 2);
    auto end_out = xt::axis_slice_end(out, 2);
    while(iter_coords != end_coords)
    {
        *iter_out = buf((*iter_coords)(0)-offset[0],(*iter_coords)(1)-offset[1],(*iter_coords)(2)-offset[2]);
        iter_coords++;
        iter_out++;
    }
    
    // std::cout << "read to buf" << std::endl;
}

int main(int argc, char *argv[])
{
  assert(argc == 2);
  // z5::filesystem::handle::File f(argv[1]);
  z5::filesystem::handle::Group group(argv[1], z5::FileMode::FileMode::r);
  z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
  std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

  std::cout << "ds shape " << ds->shape() << std::endl;
  std::cout << "ds shape via chunk " << ds->chunking().shape() << std::endl;
  std::cout << "chunk shape shape " << ds->chunking().blockShape() << std::endl;

  //read the chunk around pixel coord
  shape coord = shape({0,2000,2000});

  shape id = chunkId(ds, coord);

  fs::path path;
  ds->chunkPath(id, path);

  std::cout << "id " << id << " path " << path << std::endl;

  int threads = static_cast<int>(std::thread::hardware_concurrency());

  z5::types::ShapeType offset = idCoord(ds, id);
  xt::xarray<uint8_t>::shape_type shape = ds->chunking().blockShape();
  shape[0] = 1;
  xt::xarray<uint8_t> array(shape);

  z5::multiarray::readSubarray<uint8_t>(*ds, array, offset.begin(), threads);

  cv::Mat m = cv::Mat(shape[1], shape[2], CV_8U, array.data());

  cv::imwrite("extract.tif", m);

  xt::xarray<float> coords = xt::zeros<float>({500,500,3});
  xt::xarray<uint8_t> img;
  
  for(int y=0;y<500;y++)
      for(int x=0;x<500;x++) {
          coords(y,x,0) = 0;
          coords(y,x,1) = y+2000;
          coords(y,x,2) = x+2000;
      }

  readInterpolated3D(img,ds,coords);

  
  m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  
  cv::imwrite("img.tif", m);
  
  return 0;
}
