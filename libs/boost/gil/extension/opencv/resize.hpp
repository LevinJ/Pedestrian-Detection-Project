/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_RESIZE_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_RESIZE_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include "ipl_image_wrapper.hpp"

namespace boost { namespace gil { namespace opencv {

struct interpolation_base {};

struct nearest_neigbor : interpolation_base, boost::mpl::int_< CV_INTER_NN >     {};
struct bilinear        : interpolation_base, boost::mpl::int_< CV_INTER_LINEAR > {};
struct area            : interpolation_base, boost::mpl::int_< CV_INTER_AREA >   {};
struct bicubic         : interpolation_base, boost::mpl::int_< CV_INTER_CUBIC >  {};

template< typename Interpolation >
void resize( const ipl_image_wrapper& src
           , ipl_image_wrapper&       dst
           , const Interpolation& 
           , typename boost::enable_if< typename boost::is_base_of< interpolation_base 
                                                                  , Interpolation
                                                                  >::type
                                      >::type* ptr = 0
           )
{
   cvResize( src.get()
           , dst.get()
           , Interpolation::type::value
           );
}

template< typename View
        , typename Interpolation
        >
void resize( View                 src
           , View                 dst
           , const Interpolation& interpolation
           , typename boost::enable_if< typename boost::is_base_of< interpolation_base 
                                                                  , Interpolation
                                                                  >::type
                                      >::type* ptr = 0
           )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    resize( src_ipl
          , dst_ipl
          , interpolation
          );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_RESIZE_HPP_INCLUDED
