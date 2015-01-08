#ifndef INTEGRAL_CHANNELS_HELPERS_HPP
#define INTEGRAL_CHANNELS_HELPERS_HPP

#include "applications/bootstrapping_lib/IntegralChannelsComputer.hpp"
#include "helpers/geometry.hpp"

namespace boosted_learning {

typedef bootstrapping::integral_channels_computer_t integral_channels_computer_t;
typedef integral_channels_computer_t::integral_channels_t integral_channels_t;
typedef integral_channels_computer_t::integral_channels_const_view_t integral_channels_const_view_t;

typedef doppia::geometry::point_xy<int> point_t;

/// the out integral channel will be resized to the required dimensions
void get_integral_channels(const integral_channels_t &in,
                           const point_t &modelWindowSize, const point_t &dataOffset, const int resizing_factor,
                           integral_channels_t &out);

/// the out integral channel will be resized to the required dimensions
void get_integral_channels(const integral_channels_t &in,
                           const int inX, const int inY, const int inW, const int inH,
                           const int resizing_factor,
                           integral_channels_t &out);

integral_channels_const_view_t
get_integral_channels_view(const integral_channels_t &in,
                           const int inX, const int inY, const int inW, const int inH,
                           const int resizing_factor);

} // end of namespace boosted_learning


#endif // INTEGRAL_CHANNELS_HELPERS_HPP
