#include <map>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "vc/core/types/Volume.hpp"
#include "vc/core/util/PlaneSurface.hpp"


void timed_plane_slice(Surface &plane, Volume &vol, int size, const std::string& msg, bool nearest_neighbor)
{
    cv::Mat_<cv::Vec3f> coords;
    cv::Mat_<cv::Vec3f> normals;
    cv::Mat_<uint8_t> img;

    auto start = std::chrono::high_resolution_clock::now();
    plane.gen(&coords, &normals, {size, size}, plane.pointer(), 1.0, {0,0,0});
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s gen_coords() " << msg << "\n";
    start = std::chrono::high_resolution_clock::now();
    auto method = nearest_neighbor ? InterpolationMethod::Nearest : InterpolationMethod::Trilinear;
    vol.readInterpolated(img, coords, method, 1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(end-start).count() << "s slicing  " << size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << msg << "\n";
}


int main(int argc, char *argv[])
{
  assert(argc == 2 || argc == 3);

  auto volume = Volume::New(argv[1]);

  bool nearest_neighbor = (argc == 3 && strncmp(argv[2],"nearest",7) == 0);

  auto shape = volume->shapeZYX(1);
  std::cout << "ds shape " << shape[0] << "x" << shape[1] << "x" << shape[2] << "\n";
  if (nearest_neighbor) {
    std::cout << "doing nearest neighbor interpolation" << "\n";
  }

  cv::Mat_<cv::Vec3f> coords;
  cv::Mat_<cv::Vec3f> normals;
  cv::Mat_<uint8_t> img;

  PlaneSurface gen_plane({2000,2000,2000},{0.5,0.5,0.5});

  PlaneSurface plane_x({2000,2000,2000},{1.0,0.0,0.0});
  PlaneSurface plane_y({2000,2000,2000},{0.0,1.0,0.0});
  PlaneSurface plane_z({2000,2000,2000},{0.0,0.0,1.0});

  const int size = 1024;

  std::cout << "testing different slice directions / caching" << "\n";
  for(int r=0;r<3;r++) {
      timed_plane_slice(plane_x, *volume, size, "yz cold", nearest_neighbor);
      timed_plane_slice(plane_x, *volume, size, "yz", nearest_neighbor);
      timed_plane_slice(plane_y, *volume, size, "xz cold", nearest_neighbor);
      timed_plane_slice(plane_y, *volume, size, "xz", nearest_neighbor);
      timed_plane_slice(plane_z, *volume, size, "xy cold", nearest_neighbor);
      timed_plane_slice(plane_z, *volume, size, "xy", nearest_neighbor);
      timed_plane_slice(gen_plane, *volume, size, "diag cold", nearest_neighbor);
      timed_plane_slice(gen_plane, *volume, size, "diag", nearest_neighbor);
  }


  {
    auto start = std::chrono::high_resolution_clock::now();

    for(float shift = -50;shift<50;shift++) {
        PlaneSurface plane_s({2000,2000,2000+shift},{0.0,0.0,1.0});

        cv::Mat_<cv::Vec3f> coords;
        cv::Mat_<cv::Vec3f> normals;
        cv::Mat_<uint8_t> img;

        plane_s.gen(&coords, &normals, {size, size}, plane_s.pointer(), 1.0, {0,0,0});

        volume->readInterpolated(img, coords, InterpolationMethod::Trilinear, 1);
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

          volume->readInterpolated(img, coords, InterpolationMethod::Trilinear, 1);
      }

      auto end = std::chrono::high_resolution_clock::now();
      std::cout << std::chrono::duration<double>(end-start).count() << "s slicing / " << 100*size*size/1024.0/1024.0/std::chrono::duration<double>(end-start).count() << "MiB/s " << " shift (warm)"  << "\n";
  }

  return 0;
}
