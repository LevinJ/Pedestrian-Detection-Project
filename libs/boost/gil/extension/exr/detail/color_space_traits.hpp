//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#ifndef EXR_COLORSPACE_TRAITS
#define EXR_COLORSPACE_TRAITS

#include<boost/gil/gray.hpp>
#include<boost/gil/rgb.hpp>
#include<boost/gil/rgba.hpp>

namespace boost
{
namespace gil
{

namespace detail
{
template<class CS> struct exr_color_space_traits {};

template<> struct exr_color_space_traits<gil::gray_t> { static const char *channel_name[1];};
const char *exr_color_space_traits<gil::gray_t>::channel_name[1] = { "Y"};

template<> struct exr_color_space_traits<gil::rgb_t> { static const char *channel_name[3];};
const char *exr_color_space_traits<gil::rgb_t>::channel_name[3] = { "R", "G", "B"};

template<> struct exr_color_space_traits<gil::rgba_t> { static const char *channel_name[4];};
const char *exr_color_space_traits<gil::rgba_t>::channel_name[4] = { "R", "G", "B", "A"};

} // namespace
} // namespace
} // namespace

#endif
