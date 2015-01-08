/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_SMOOTH_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_SMOOTH_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include <boost/type_traits/is_base_of.hpp>

#include <boost/utility/enable_if.hpp>

#include "ipl_image_wrapper.hpp"

namespace boost { namespace gil { namespace opencv {

struct smooth_base {};

struct blur_no_scale : smooth_base, boost::mpl::int_< CV_BLUR_NO_SCALE > {};
struct blur          : smooth_base, boost::mpl::int_< CV_BLUR          > {};
struct gaussian      : smooth_base, boost::mpl::int_< CV_GAUSSIAN      > {};
struct median        : smooth_base, boost::mpl::int_< CV_MEDIAN        > {};
struct bilateral     : smooth_base, boost::mpl::int_< CV_BILATERAL     > {};

template< typename Smooth >
void smooth( const ipl_image_wrapper& src
           , ipl_image_wrapper&       dst
           , const Smooth&            smooth
           , size_t                   param1 = 3
           , size_t                   param2 = 0
           , size_t                   param3 = 0
           , size_t                   param4 = 0
           , typename boost::enable_if< typename boost::is_base_of< smooth_base 
                                                                  , Smooth
                                                                  >::type
                                      >::type* ptr = 0
           )
           
{
   cvSmooth( src.get()
           , dst.get()
           , Smooth::type::value
           , param1
           , param2
           , param3
           , param4
           );
}

template< typename View
        , typename Smooth
        >
void smooth( View          src
           , View          dst
           , const Smooth& smooth_type
           , size_t        param1 = 3
           , size_t        param2 = 0
           , size_t        param3 = 0
           , size_t        param4 = 0
           , typename boost::enable_if< typename boost::is_base_of< smooth_base 
                                                                  , Smooth
                                                                  >::type
                                      >::type* ptr = 0
           )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    smooth( src_ipl
          , dst_ipl
          , smooth_type
          );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_SMOOTH_HPP_INCLUDED
