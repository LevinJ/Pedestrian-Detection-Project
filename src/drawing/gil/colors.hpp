#ifndef COLORS_HPP
#define COLORS_HPP

#include <boost/gil/typedefs.hpp>

namespace doppia {

namespace rgb8_colors
{
extern const boost::gil::rgb8c_pixel_t
white, black,
pink, dark_pink,
red, dark_red,
orange, dark_orange,
brown, dark_brown,
yellow, dark_yellow,
gray,
green, dark_green,
cyan, dark_cyan,
blue, dark_blue,
violet, dark_violet,
magenta, dark_magenta;

//extern const boost::gil::rgb8c_pixel_t jet_color_map[640];
extern const  boost::uint8_t jet_color_map[640][3];

} // end of namespace rgb8c_colors


} // end of namespace doppia

#endif // COLORS_HPP
