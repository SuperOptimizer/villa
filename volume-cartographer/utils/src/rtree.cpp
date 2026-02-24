#include "utils/rtree.hpp"

// Explicit instantiation to verify the template compiles for common
// dimensionalities. Being header-only, all real usage is inlined at the call
// site, but these instantiations catch errors at library build time.
template class utils::RTree<2>;
template class utils::RTree<3>;
