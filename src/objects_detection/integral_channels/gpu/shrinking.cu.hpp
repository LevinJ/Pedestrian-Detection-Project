
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/devicememoryreference.hpp>
#include <boost/cstdint.hpp>

namespace doppia {
namespace integral_channels {

//typedef boost::uint16_t channel_element_t;
typedef boost::uint8_t channel_element_t;

//typedef Cuda::DeviceMemoryPitched2D<channel_element_t> gpu_channel_t;
typedef Cuda::DeviceMemoryReference2D<channel_element_t> gpu_channel_ref_t;
typedef Cuda::DeviceMemoryPitched3D<channel_element_t> gpu_channels_t;

/// shrinking_factor 1 means no resizing, 2 half the size, 4 a quarter of the size
void shrink_channel(const gpu_channel_ref_t &input_channel, gpu_channel_ref_t &shrunk_channel, const int shrinking_factor);

void shrink_channels(const gpu_channels_t &input_channels, gpu_channels_t &shrunk_channels, const int shrinking_factor);

} // end of namespace integral_channels
} // end of namespace doppia


