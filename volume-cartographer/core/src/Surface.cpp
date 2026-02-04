#include "vc/core/util/Surface.hpp"

#include <nlohmann/json.hpp>

// Special member functions defined here where nlohmann::json is complete
Surface::Surface() = default;
Surface::~Surface() = default;
Surface::Surface(Surface&&) noexcept = default;
Surface& Surface::operator=(Surface&&) noexcept = default;
