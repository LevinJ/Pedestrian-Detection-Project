/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_UTILITIES_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_UTILITIES_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

//#include <opencv/cv.h>
#include <opencv2/core/core_c.h>

#include <boost/shared_array.hpp>

#include <boost/gil/gil_all.hpp>

#include <vector>

namespace boost { namespace gil { namespace opencv {

typedef gil::point2<ptrdiff_t> point_t;
typedef std::vector< point_t > curve_t;
typedef std::vector< curve_t > curve_vec_t;

typedef boost::shared_array<CvPoint> cvpoint_array_t;
typedef std::vector< cvpoint_array_t > cvpoint_array_vec_t;

inline
CvPoint make_cvPoint( point_t point )
{
   return cvPoint( static_cast< int >( point.x )
                 , static_cast< int >( point.y )
                 );
}

inline
cvpoint_array_t make_cvPoint_array( const curve_t& curve )
{
   std::size_t curve_size = curve.size();

   cvpoint_array_t cvpoint_array( new CvPoint[ curve.size() ] );

   for( std::size_t i = 0; i < curve_size ; ++i )
   {
      cvpoint_array[i] = make_cvPoint( curve[i] );
   }

   return cvpoint_array;
}

inline
CvSize make_cvSize( point_t point )
{
   return cvSize( static_cast< int >( point.x )
                , static_cast< int >( point.y )
                );
}

template< class PIXEL >
inline
CvScalar make_cvScalar( const PIXEL& pixel )
{
   CvScalar s;
   for( int i = 0; i < num_channels<PIXEL>::value; ++i )
   {
      s.val[i] = pixel[i];
   }

   return s;
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_UTILITIES_HPP_INCLUDED
