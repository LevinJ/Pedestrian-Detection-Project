#ifndef INTEGRALCHANNELSCOMPUTER_HPP
#define INTEGRALCHANNELSCOMPUTER_HPP

/// This header file defines bootstrapping::integral_channels_computer_t

#if true and defined(USE_GPU)
#define USE_GPU_INTEGRAL_CHANNELS_COMPUTER
#endif

#if defined(USE_GPU_INTEGRAL_CHANNELS_COMPUTER)
#include "objects_detection/integral_channels/GpuIntegralChannelsForPedestrians.hpp"
#else
#include "objects_detection/integral_channels/IntegralChannelsForPedestrians.hpp"
#endif

namespace bootstrapping
{

#if defined(USE_GPU_INTEGRAL_CHANNELS_COMPUTER)
typedef doppia::GpuIntegralChannelsForPedestrians integral_channels_computer_t;
#else
typedef doppia::IntegralChannelsForPedestrians integral_channels_computer_t;
#endif

typedef integral_channels_computer_t::integral_channels_t integral_channels_t;
typedef integral_channels_computer_t::integral_channels_view_t integral_channels_view_t;
typedef integral_channels_computer_t::integral_channels_const_view_t integral_channels_const_view_t;


} // end of namespace bootstrapping


#endif // INTEGRALCHANNELSCOMPUTER_HPP
