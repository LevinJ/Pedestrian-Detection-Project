/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_CONVERT_SCALE_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_CONVERT_SCALE_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <boost/gil/gil_all.hpp>

#include "ipl_image_wrapper.hpp"

namespace boost { namespace gil { namespace opencv {

inline
void convert_scale_( const ipl_image_wrapper& src
                   , ipl_image_wrapper&        dst
                   , const double&             scale = 1.0
                   , const double&             shift = 0.0
                   )
{
   cvConvertScale( src.get()
                 , dst.get()
                 , scale
                 , shift
                 );
}

template< typename View_Src
        , typename View_Dst
        >
inline
void convert_scale( View_Src&     src
                  , View_Dst&     dst
                  , const double& scale = 1.0
                  , const double& shift = 0.0
                  )
{
/*
    // color spaces must be equal
    BOOST_STATIC_ASSERT(( boost::is_same< typename color_space_type< View_Src >::type
                                        , typename color_space_type< View_Dst >::type
                                        >::type::value
                       ));

    // destination image should have 8 bit unsigned channels
    BOOST_STATIC_ASSERT(( boost::is_same< bits8
                                        , typename channel_type< View_Dst >::type
                                        >::type::value
                       ));
*/

    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    convert_scale_( src_ipl
                  , dst_ipl
                  , scale
                  , shift
                  );
}


inline
void convert_scale_abs_( const ipl_image_wrapper& src
                      , ipl_image_wrapper&        dst
                      , const double&             scale = 1.0
                      , const double&             shift = 0.0
                      )
{
   cvConvertScaleAbs( src.get()
                    , dst.get()
                    , scale
                    , shift
                    );
}

template< typename View_Src
        , typename View_Dst
        >
inline
void convert_scale_abs( View_Src&     src
                      , View_Dst&     dst
                      , const double& scale = 1.0
                      , const double& shift = 0.0
                      )
{
    // color spaces must be equal
    BOOST_STATIC_ASSERT(( boost::is_same< typename color_space_type< View_Src >::type
                                        , typename color_space_type< View_Dst >::type
                                        >::type::value
                       ));

    // destination image should have 8 bit unsigned channels
    BOOST_STATIC_ASSERT(( boost::is_same< bits8
                                        , typename channel_type< View_Dst >::type
                                        >::type::value
                       ));

    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    convert_scale_abs_( src_ipl
                      , dst_ipl
                      , scale
                      , shift
                      );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_CONVERT_SCALE_HPP_INCLUDED
