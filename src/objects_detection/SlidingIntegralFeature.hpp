#ifndef SLIDINGINTEGRALFEATURE_HPP
#define SLIDINGINTEGRALFEATURE_HPP

#include "SoftCascadeOverIntegralChannelsModel.hpp"
#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include <cstdio>

namespace doppia {

/// Small helper class to build fast IntegralChannels detectors
/// Helper class that implements a sliding integral feature
/// tracks all four corners
/// all methods inlined for performance reasons
/// this class will do zero memory checks, the user is responsible of avoiding segmentation faults
class SlidingIntegralFeature
{
public:

    typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;
    typedef IntegralChannelsForPedestrians::const_integral_channel_t const_integral_channel_t;

    SlidingIntegralFeature(const IntegralChannelsFeature &feature,
                           const integral_channels_t &integral_channels,
                           const size_t row_index,  const size_t col_index,
                           const uint8_t xstride);

    /// make the feature move xstride pixels to the right
    void slide();

    /// get the integral value of the feature
    float get_value();

protected:

    const uint8_t xstride;
    typedef const_integral_channel_t::element const* const_value_pointer_t;

    const_value_pointer_t top_left_p, top_right_p, bottom_left_p, bottom_right_p;

};

inline
SlidingIntegralFeature::SlidingIntegralFeature(const IntegralChannelsFeature &feature,
                                               const integral_channels_t &integral_channels,
                                               const size_t row_index,
                                               const size_t col_index,
                                               const uint8_t xstride_)
    : xstride(xstride_)
{
    const const_integral_channel_t channel = integral_channels[feature.channel_index];
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    // if row_index or col_index are too high, some of these pointers may be fall outside the channel memory
    top_left_p     = &(channel[row_index + box.min_corner().y()][col_index + box.min_corner().x()]);
    top_right_p    = &(channel[row_index + box.min_corner().y()][col_index + box.max_corner().x()]);
    bottom_left_p  = &(channel[row_index + box.max_corner().y()][col_index + box.min_corner().x()]);
    bottom_right_p = &(channel[row_index + box.max_corner().y()][col_index + box.max_corner().x()]);
    return;
}


inline
void SlidingIntegralFeature::slide()
{
    // if called to many times, the pointers will fall outside the channel memory
    top_left_p += xstride;
    top_right_p += xstride;
    bottom_left_p += xstride;
    bottom_right_p += xstride;
    return;
}


inline
float SlidingIntegralFeature::get_value()
{
    // based on http://en.wikipedia.org/wiki/Summed_area_table
    /*const const_integral_channel_t::element
            &a = *top_left_p,
            &b = *top_right_p,
            &c = *bottom_right_p,
            &d = *bottom_left_p;*/

    // using const_integral_channel_t::element could cause overflows on a+b
    const float a = *top_left_p, b = *top_right_p, c = *bottom_right_p, d = *bottom_left_p;
    float feature_value = a +c -b -d;

    const bool print_each_feature_value = false;
    if(print_each_feature_value)
    { // useful for debugging
        printf("feature_value == %.3f\n", feature_value);
    }

    return feature_value;
}


} // end of namespace doppia

#endif // SLIDINGINTEGRALFEATURE_HPP
