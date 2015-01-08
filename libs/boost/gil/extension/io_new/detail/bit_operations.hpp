/*
    Copyright 2007-2008 Christian Henning, Andreas Pokorny, Lubomir Bourdev
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_IO_DETAIL_BIT_OPERATIONS_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_IO_DETAIL_BIT_OPERATIONS_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief
/// \author Christian Henning \n
///
/// \date   2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

namespace boost { namespace gil { namespace detail {

// 1110 1100 -> 0011 0111
template< typename Buffer
        , typename IsBitAligned
        >
struct mirror_bits
{
   mirror_bits() {}

   void operator() ( Buffer& ) {}
};


// The functor will generate a lookup table since the
// mirror operation is quite costly. Some basic benchmarks
// have proven it.
template< typename Buffer >
struct mirror_bits< Buffer
                  , mpl::true_
                  >
{
   mirror_bits()
   {
        byte_t i = 0;
        do
        {
            _lookup[i] = mirror( i );
        }
        while( i++ != 255 );
   }

   void operator() ( Buffer& buf )
   {
        for_each( buf.begin()
                , buf.end()
                , bind( &mirror_bits< Buffer
                                    , mpl::true_
                                    >::lookup
                       , *this
                       , _1
                       )
                );
   }

private:

    void lookup( byte_t& c )
    {
        c = _lookup[ c ];
    }

    static byte_t mirror( byte_t c )
    {
        byte_t result = 0;
        for( int i = 0; i < 8; ++i )
        {
            result = result << 1;
            result |= ( c & 1 );
            c = c >> 1;
        }

        return result;
    }
 
private:
 
    array< byte_t, 256 > _lookup;
};


// 0011 1111 -> 1100 0000
template< typename Buffer
        , typename IsBitAligned
        >
struct negate_bits
{
    void operator() ( Buffer& ) {}
};

template< typename Buffer >
struct negate_bits< Buffer, mpl::true_ >
{
    void operator() ( Buffer& buf )
    {
        for_each( buf.begin()
                , buf.end()
                , negate_bits< Buffer, mpl::true_ >::negate
                );
    }

private:

    static void negate( byte_t& b )
    {
        b = ~b;
    }
};


// 11101100 -> 11001110
template< typename Buffer
        , typename IsBitAligned
        >
struct swap_half_bytes
{
    void operator() ( Buffer& ) {}
};

template< typename Buffer >
struct swap_half_bytes< Buffer
                      , mpl::true_
                      >
{
    void operator() ( Buffer& buf )
    {
        for_each( buf.begin()
                , buf.end()
                , swap_half_bytes< Buffer, mpl::true_ >::swap
                );
    }

private:

    static void swap( byte_t& c )
    {
        c = (( c << 4 ) & 0xF0 ) | (( c >> 4 ) & 0x0F );
    }
};

template< typename Buffer >
struct do_nothing
{
   do_nothing() {}

   void operator() ( Buffer& ) {}
};

/// Count consecutive zeros on the right
template< typename T >
inline
unsigned int trailing_zeros( T x )
throw()
{
    unsigned int n = 0;

    x = ~x & (x - 1);

    while( x )
    {
        n = n + 1;
        x = x >> 1;
    }

    return n;
}

/// Counts ones in a bit-set
template< typename T >
inline
unsigned int count_ones( T x )
throw()
{
    unsigned int n = 0;

    while( x )
    {
	    // clear the least significant bit set
	    x &= x - 1;
	    ++n;
    }

    return n;
}


} // namespace detail
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_IO_DETAIL_BIT_OPERATIONS_HPP_INCLUDED
