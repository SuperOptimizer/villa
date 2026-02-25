#include "vc/core/types/Tensor.hpp"

// Explicit template instantiations for common types
namespace vc {

template class Tensor<uint8_t>;
template class Tensor<uint16_t>;
template class Tensor<float>;

template class TensorView<uint8_t>;
template class TensorView<uint16_t>;
template class TensorView<float>;

template class TensorAdaptor<uint8_t>;
template class TensorAdaptor<uint16_t>;
template class TensorAdaptor<float>;

} // namespace vc
