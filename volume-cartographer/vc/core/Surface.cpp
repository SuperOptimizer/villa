#include "vc/core/util/Surface.hpp"

#include <nlohmann/json.hpp>

Surface::~Surface()
{
    if (meta) {
        delete meta;
    }
}
