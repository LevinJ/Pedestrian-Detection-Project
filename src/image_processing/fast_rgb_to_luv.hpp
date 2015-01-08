#ifndef FAST_RGB_TO_LUV_HPP
#define FAST_RGB_TO_LUV_HPP

#include <boost/gil/typedefs.hpp>

namespace doppia {

void fast_rgb_to_luv(const boost::gil::rgb8c_view_t &rgb_view,
                     const boost::gil::dev3n8_view_t &luv_view);

void fast_rgb_to_luv(const boost::gil::rgb8c_view_t &rgb_view,
                     const boost::gil::dev3n8_planar_view_t &luv_view);


void fast_rgb_to_luv(const boost::gil::rgb16c_view_t &rgb_view,
                     const boost::gil::dev3n16_view_t &luv_view);

} // end of namespace doppia

#endif // FAST_RGB_TO_LUV_HPP
