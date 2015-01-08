//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXR_CHANNEL_TRAITS
#define EXR_CHANNEL_TRAITS

#include "../half/channel.hpp"

#include<OpenEXR/ImfPixelType.h>

namespace boost
{
namespace gil
{

namespace detail
{
template<class C> struct exr_channel2pixel_type {};
template<> struct exr_channel2pixel_type<gil::bits16f>	{ BOOST_STATIC_CONSTANT( Imf::PixelType, value = Imf::HALF);};
template<> struct exr_channel2pixel_type<gil::bits32f>	{ BOOST_STATIC_CONSTANT( Imf::PixelType, value = Imf::FLOAT);};
template<> struct exr_channel2pixel_type<gil::bits32>	{ BOOST_STATIC_CONSTANT( Imf::PixelType, value = Imf::UINT);};
}

} // namespace
} // namespace

#endif
