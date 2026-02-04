#include <map>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/core.hpp>
#include <z5/common.hxx>
#include <z5/dataset.hxx>
#include <z5/filesystem/factory.hxx>
#include <z5/util/blocking.hxx>

#include "z5/filesystem/handle.hxx"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/StreamOperators.hpp"
#include "vc/core/util/ChunkCache.hpp"


using shape = z5::types::ShapeType;



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

void timed_plane_slice(Surface &plane, z5::Dataset *ds, int size, ChunkCache<uint8_t> *cache, std::string msg, bool nearest_neighbor)
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    cv::Mat_<uint8_t> img;

    auto start = std::chrono::high_resolution_clock::now();
    plane.gen(&coords, &normals, {size, size}, plane.pointer(), 1.0, {0,0,0});
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s gen_coords() " << msg << "\n";
    start = std::chrono::high_resolution_clock::now();
    readInterpolated3D(img, ds, coords, cache, nearest_neighbor);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing  " << size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << msg << "\n";
}


int main(int argc, char *argv[])
{
  assert(argc == 2 || argc == 3);
  // z5::filesystem::handle::File f(argv[1]);
  z5::filesystem::handle::Group group(argv[1], z5::FileMode::FileMode::r);
  z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
  std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

   bool nearest_neighbor =  (argc == 3 && strncmp(argv[2],"nearest",7) == 0);

  std::cout << "ds shape " << ds->shape() << "\n";
  std::cout << "ds shape via chunk " << ds->chunking().shape() << "\n";
  std::cout << "chunk shape shape " << ds->chunking().blockShape() << "\n";
  if (nearest_neighbor) {
    std::cout << "doing nearest neighbor interpolation" << "\n";
  }

  cv::Mat_<cv::Vec3f> coords;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<uint8_t> img;
  
//   std::vector<cv::Mat> chs;
//   cv::imreadmulti("../grid_slice_coords.tif", chs, cv::IMREAD_UNCHANGED);
//   cv::Mat_<cv::Vec3f> points;
//   cv::merge(chs, points);
//
//   QuadSurface gen_grid(points, {1,1});
  
  PlaneSurface gen_plane({2000,2000,2000},{0.5,0.5,0.5});
  // PlaneSurface gen_plane({2000,2000,2000},{0.0,0.0,1.0});
  // PlaneSurface gen_plane({0,0,0},{0.0,0.0,1.0});
  // PlaneSurface gen_plane({2000,2000,0},{0.0,0.0,1.0});
  
  PlaneSurface plane_x({2000,2000,2000},{1.0,0.0,0.0});
  PlaneSurface plane_y({2000,2000,2000},{0.0,1.0,0.0});
  PlaneSurface plane_z({2000,2000,2000},{0.0,0.0,1.0});
  
  // gen_plane.gen_coords(coords, 1000, 1000);
  // gen_grid.gen(&coords, &normals, {1000, 1000}, gen_grid.pointer(), 1.0, {0,0,0});

    ChunkCache<uint8_t> chunk_cache(10*10e9);

  // auto start = std::chrono::high_resolution_clock::now();
  // readInterpolated3D(img,ds.get(),coords, &chunk_cache);
  // auto end = std::chrono::high_resolution_clock::now();
  // std::cout << std::chrono::duration<double>(end-start).count() << "s cold" << "\n";
  
  // cv::imwrite("plane.tif", img);
  
//   for(int r=0;r<10;r++) {
//     start = std::chrono::high_resolution_clock::now();
//     readInterpolated3D(img,ds.get(),coords, &chunk_cache);
//     end = std::chrono::high_resolution_clock::now();
//     std::cout << std::chrono::duration<double>(end-start).count() << "s cached" << "\n";
//   }
//

  const int size = 1024;

  std::cout << "testing different slice directions / caching" << "\n";
  for(int r=0;r<3;r++) {
      timed_plane_slice(plane_x, ds.get(), size, &chunk_cache, "yz cold", nearest_neighbor);
      timed_plane_slice(plane_x, ds.get(), size, &chunk_cache, "yz", nearest_neighbor);
      timed_plane_slice(plane_y, ds.get(), size, &chunk_cache, "xz cold", nearest_neighbor);
      timed_plane_slice(plane_y, ds.get(), size, &chunk_cache, "xz", nearest_neighbor);
      timed_plane_slice(plane_z, ds.get(), size, &chunk_cache, "xy cold", nearest_neighbor);
      timed_plane_slice(plane_z, ds.get(), size, &chunk_cache, "xy", nearest_neighbor);
      timed_plane_slice(gen_plane, ds.get(), size, &chunk_cache, "diag cold", nearest_neighbor);
      timed_plane_slice(gen_plane, ds.get(), size, &chunk_cache, "diag", nearest_neighbor);
  }


  {
    auto start = std::chrono::high_resolution_clock::now();

    for(float shift = -50;shift<50;shift++) {
        PlaneSurface plane_s({2000,2000,2000+shift},{0.0,0.0,1.0});

        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        cv::Mat_<uint8_t> img;

        plane_s.gen(&coords, &normals, {size, size}, plane_s.pointer(), 1.0, {0,0,0});

        readInterpolated3D(img, ds.get(), coords, &chunk_cache);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing / " << 100*size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << " shift (cold)"  << "\n";
  }

  {
      auto start = std::chrono::high_resolution_clock::now();

      for(float shift = -50;shift<50;shift++) {
          PlaneSurface plane_s({2000,2000,2000+shift},{0.0,0.0,1.0});

          cv::Mat_<cv::Vec3f> coords;
          cv::Mat_<cv::Vec3f> normals;
          cv::Mat_<uint8_t> img;

          plane_s.gen(&coords, &normals, {size, size}, plane_s.pointer(), 1.0, {0,0,0});

          readInterpolated3D(img, ds.get(), coords, &chunk_cache);
      }

      auto end = std::chrono::high_resolution_clock::now();
      std::cout << std::chrono::duration<double>(end-start).count() << "s slicing / " << 100*size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << " shift (warm)"  << "\n";
  }


  
  // readInterpolated3D(img,ds.get(),coords);
  // m = cv::Mat(img.shape(0), img.shape(1), CV_8U, img.data());
  // cv::imwrite("plane.tif", m);
  
  
  // m = cv::Mat(coords.shape(0), coords.shape(1), CV_32FC3, coords.data());
  // std::vector<cv::Mat> chs;
  // cv::split(m, chs);
  // cv::imwrite("coords_x.tif", chs[2]);
  // cv::imwrite("coords_y.tif", chs[1]);
  // cv::imwrite("coords_z.tif", chs[0]);
    
  return 0;
}
