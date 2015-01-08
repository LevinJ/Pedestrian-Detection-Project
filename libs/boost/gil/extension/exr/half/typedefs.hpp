//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef GIL_HALF_TYPEDEFS_HPP
#define GIL_HALF_TYPEDEFS_HPP

#include<boost/gil/typedefs.hpp>

#include"channel.hpp"

namespace boost
{
namespace gil
{

GIL_DEFINE_BASE_TYPEDEFS(16f,gray)
GIL_DEFINE_BASE_TYPEDEFS(16f,bgr)
GIL_DEFINE_BASE_TYPEDEFS(16f,argb)
GIL_DEFINE_BASE_TYPEDEFS(16f,abgr)
GIL_DEFINE_BASE_TYPEDEFS(16f,bgra)

GIL_DEFINE_ALL_TYPEDEFS(16f,rgb)
GIL_DEFINE_ALL_TYPEDEFS(16f,rgba)
GIL_DEFINE_ALL_TYPEDEFS(16f,cmyk)

template <int N> struct devicen_t;
template <int N> struct devicen_layout_t;

GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(16f,dev2n, devicen_t<2>, devicen_layout_t<2>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(16f,dev3n, devicen_t<3>, devicen_layout_t<3>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(16f,dev4n, devicen_t<4>, devicen_layout_t<4>)
GIL_DEFINE_ALL_TYPEDEFS_INTERNAL(16f,dev5n, devicen_t<5>, devicen_layout_t<5>)

} // namespace
} // namespace

#endif
