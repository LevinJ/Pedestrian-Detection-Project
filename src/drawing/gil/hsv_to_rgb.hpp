#ifndef DOPPIA_HSV_TO_RGB_HPP
#define DOPPIA_HSV_TO_RGB_HPP

#include <boost/gil/pixel.hpp>
#include <boost/gil/typedefs.hpp>

namespace doppia {

/// @param hue should be in range [0,1]
/// @param saturation should be in range [0,1]
/// @param value should be in range [0,1]
boost::gil::rgb8c_pixel_t hsv_to_rgb(const float hue, const float saturation, const float value);

void hsv_to_rgb( float H, float S, float V, float& R, float& G, float& B );

} // end of namespace doppia

#endif // DOPPIA_CONVERT_HSV_TO_RGB_HPP
