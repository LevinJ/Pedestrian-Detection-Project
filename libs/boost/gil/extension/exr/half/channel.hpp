//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef GIL_EXR_CHANNEL_HPP
#define GIL_EXR_CHANNEL_HPP

#include<OpenEXR/half.h>

#include<boost/type_traits/is_class.hpp>

#include<boost/gil/channel.hpp>
#include<boost/gil/channel_algorithm.hpp>

namespace boost
{

	template<> struct is_class<half> : public false_type{};

namespace gil
{

struct half_zero { static half apply() { return half( 0.0f);}};
struct half_one  { static half apply() { return half( 1.0f);}};

typedef scoped_channel_value<half,half_zero,half_one> bits16f;

// channel conversion
template <> struct channel_converter_unsigned<bits16f,bits8> : public std::unary_function<bits16f,bits8>
{
    bits8   operator()(bits16f x) const
	{
		return static_cast<bits8>( half(x) * 255.0f + 0.5f);
	}
};

template <> struct channel_converter_unsigned<bits8,bits16f> : public std::unary_function<bits8,bits16f>
{
    bits16f operator()(bits8   x) const { return static_cast<bits16f>( half(x) / 255.0f); }
};

template <> struct channel_converter_unsigned<bits16f,bits16> : public std::unary_function<bits16f,bits16>
{
    bits16 operator()(bits16f x) const { return static_cast<bits16>( half(x) * 65535.0f + 0.5f); }
};

template <> struct channel_converter_unsigned<bits16,bits16f> : public std::unary_function<bits16,bits16f>
{
    bits16f operator()(bits16  x) const { return static_cast<bits16f>( half(x) / 65535.0f); }
};

// floating point
template <> struct channel_converter_unsigned<bits32f,bits16f> : public std::unary_function<bits32f,bits16f>
{
    bits16f operator()(bits32f  x) const { return bits16f( float(x));}
};

template <> struct channel_converter_unsigned<bits16f,bits32f> : public std::unary_function<bits16f,bits32f>
{
    bits32f operator()(bits16f  x) const { return bits32f( half(x));}
};

template<> struct channel_multiplier_unsigned<bits16f> : public std::binary_function<bits16f,bits16f,bits16f>
{
    bits16f operator()(bits16f a, bits16f b) const { return bits16f( half(a) * half(b));}
};

} // namespace
} // namespace

#endif
