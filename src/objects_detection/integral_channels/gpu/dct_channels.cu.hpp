
//#include <cudatemplates/devicememorypitched.hpp>
//#include <cudatemplates/devicememoryreference.hpp>
//#include <boost/cstdint.hpp>

//include this header avoid duplicate codes for gpu_channels definition
#include "integral_channels.cu.hpp"

namespace doppia {
namespace integral_channels {

//typedef Cuda::DeviceMemoryPitched3D<boost::uint8_t> gpu_channels_t;
//__global__ void dct_channels_kernel(gpu_channels_t::KernelData input_channel);
void compute_dct_channels(gpu_channels_t &input_channel);


} // end of namespace integral_channels
} // end of namespace doppia


