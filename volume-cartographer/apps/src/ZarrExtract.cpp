#include <nlohmann/json.hpp>
#include "xtensor/xarray.hpp"

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/filesystem/dataset.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

using shape = z5::types::ShapeType;

std::ostream& operator<< (std::ostream& out, const std::vector<size_t> &v) {
    if ( !v.empty() ) {
        out << '[';
        for(auto &v : v)
            out << v << ",";
        out << "\b]"; // use ANSI backspace character '\b' to overwrite final ", "
    }
    return out;
}

shape chunkId(const std::unique_ptr<z5::Dataset> &ds, shape coord)
{
    shape div = ds->chunking().blockShape();
    shape id = coord;
    for(int i=0;i<coord.size();i++)
        id[i] /= div[i];
    return id;
}

int main(int argc, char *argv[]) {

  assert(argc == 2);
  // z5::filesystem::handle::File f(argv[1]);
  z5::filesystem::handle::Group group(argv[1], z5::FileMode::FileMode::r);
  z5::filesystem::handle::Dataset ds_handle(group, "1", "/");
  std::unique_ptr<z5::Dataset> ds = z5::filesystem::openDataset(ds_handle);

  std::cout << "ds shape " << ds->shape() << "\n";
  std::cout << "ds shape via chunk " << ds->chunking().shape() << "\n";
  std::cout << "chunk shape shape " << ds->chunking().blockShape() << "\n";

  //read the chunk around pixel coord
  shape coord = shape({0,2000,2000});

  shape id = chunkId(ds, coord);

  fs::path path;
  ds->chunkPath(id, path);

  std::cout << "id " << id << " path " << path << "\n";


  return 0;
}
