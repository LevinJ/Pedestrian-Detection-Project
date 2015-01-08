/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_DRAWING_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_DRAWING_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include <boost/scoped_array.hpp>

#include "ipl_image_wrapper.hpp"

namespace boost { namespace gil { namespace opencv {

struct four_connected_line  : boost::mpl::int_< 4 > {};
struct eight_connected_line : boost::mpl::int_< 8 > {};

struct cv_fill : boost::mpl::int_< CV_FILLED > {};

struct cv_aa : boost::mpl::int_< CV_AA > {};


/// When chaining operators we don't want to reconvert to
/// ipl_image all the time.

/// rectangle

// Use cv_fill as thickness to fill the rectangle.

template< typename Color
        , typename Line_Type
        >
inline
void drawRectangle( ipl_image_wrapper&  ipl_image
                  , point_t             start
                  , point_t             end
                  , const Color&        color
                  , std::size_t         thickness
                  , const Line_Type&    
                  )
{
   cvRectangle( ipl_image.get()
              , make_cvPoint ( start )
              , make_cvPoint ( end   )
              , make_cvScalar( color )
              , thickness
              , typename Line_Type::type::value
              );
}

template< typename View
        , typename Line_Type
        >
inline
void drawRectangle( View                      view
                  , point_t                   start
                  , point_t                   end
                  , typename View::value_type color
                  , std::size_t               thickness
                  , const Line_Type&          line_type )
{
   drawRectangle( create_ipl_image( view )
                , start
                , end
                , color
                , thickness
                , line_type
                );
}

/// circle

template< typename Color
        , typename Line_Type
        >
inline
void drawCircle( ipl_image_wrapper& ipl_image
               , const point_t&     center
               , std::size_t        radius
               , const Color&       color
               , std::size_t        thickness
               , const Line_Type&
               )
{
   cvCircle( ipl_image.get()
           , make_cvPoint( center )
           , radius
           , make_cvScalar( color )
           , thickness
           , typename Line_Type::type::value
           );
}

template< typename View
        , typename Line_Type
        >
inline
void drawCircle( View                      view
               , point_t                   center
               , std::size_t               radius
               , typename View::value_type color
               , std::size_t               thickness
               , const Line_Type&          line_type
               )
{
   drawCircle( create_ipl_image( view )
             , center
             , radius
             , color
             , thickness
             , line_type
             );
}

/// ellipse

template< typename Color
        , typename Line_Type
        >
inline
void drawEllipse( ipl_image_wrapper& ipl_image
                , const point_t&     center
                , const point_t&     axes
                , const double&      angle
                , const double&      start_angle
                , const double&      end_angle
                , const Color&       color
                , std::size_t        thickness
                , const Line_Type&
                )
{
   cvEllipse( ipl_image.get()
            , make_cvPoint( center )
            , make_cvSize ( axes   )
            , angle
            , start_angle
            , end_angle
            , make_cvScalar( color )
            , thickness
            , typename Line_Type::type::value
            );
}

template< typename View
        , typename Line_Type
        >
inline
void drawEllipse( View                      view
                , const point_t&            center
                , const point_t&            axes
                , const double&             angle
                , const double&             start_angle
                , const double&             end_angle
                , typename View::value_type color
                , std::size_t               thickness
                , const Line_Type&          line_type
                )
{
   drawEllipse( create_ipl_image( view )
              , center
              , axes
              , angle
              , start_angle
              , end_angle
              , color
              , thickness
              , line_type
              );
}

/// line

template< typename Color
        , typename Line_Type
        >
inline
void drawLine( ipl_image_wrapper& ipl_image
             , point_t            start
             , point_t            end
             , Color              color
             , std::size_t        line_width
             , const Line_Type&
             )
{
   cvLine( ipl_image.get()
         , make_cvPoint( start )
         , make_cvPoint( end )
         , make_cvScalar( color )
         , line_width
         , Line_Type::type::value
         );
}

template< typename View
        , typename Line_Type
        >
inline
void drawLine( View&                     view
             , point_t                   start
             , point_t                   end
             , typename View::value_type color
             , std::size_t               line_width
             , const Line_Type&          line_type
             )
{
   drawLine( create_ipl_image( view )
           , start
           , end
           , color
           , line_width
           , line_type
           );
}

/// polyline

template< typename Color
        , typename Line_Type
        >
inline
void drawPolyLine( ipl_image_wrapper& ipl_image
                 , const curve_vec_t& curves
                 , bool               is_closed
                 , Color              color
                 , std::size_t        thickness
                 , const Line_Type&
                 )
{
    const std::size_t num_curves = curves.size();

    std::vector<int> num_points_per_curve( num_curves );

    for( std::size_t i = 0; i < num_curves; ++i )
    {
      num_points_per_curve[i] = curves[i].size();
    }

    cvpoint_array_vec_t pp( num_curves );
    boost::scoped_array<CvPoint*> curve_array( new CvPoint*[num_curves] );

    for( std::size_t i = 0; i < num_curves; ++i )
    {
        pp[i] = make_cvPoint_array( curves[i] );

        curve_array[i] = pp[i].get();
    }
   
    cvPolyLine( ipl_image.get()
              , curve_array.get()  // needs to be pointer to C array of CvPoints.
              , &num_points_per_curve.front()
              , curves.size()
              , is_closed
              , make_cvScalar( color )
              , thickness
              , Line_Type::type::value
              );
}

template< typename View
        , typename Line_Type
        >
inline
void drawPolyLine( View&                     view
                 , const curve_vec_t&        curves
                 , bool                      is_closed
                 , typename View::value_type color
                 , std::size_t               thickness
                 , const Line_Type&          line_type
                 )
{
   drawPolyLine( create_ipl_image( view )
               , curves
               , is_closed
               , color
               , thickness
               , line_type
               );
}

template< typename Color
        , typename Line_Type
        >
inline
void drawFillPoly( ipl_image_wrapper& ipl_image
                 , const curve_vec_t& curves
                 , const Color&       color
                 , const Line_Type&
                 )
{
   const std::size_t num_curves = curves.size();

    std::vector<int> num_points_per_curve( num_curves );

    for( std::size_t i = 0; i < num_curves; ++i )
    {
      num_points_per_curve[i] = curves[i].size();
    }

    cvpoint_array_vec_t pp( num_curves );

    boost::scoped_array<CvPoint*> curve_array( new CvPoint*[num_curves] );

    for( std::size_t i = 0; i < num_curves; ++i )
    {
        pp[i] = make_cvPoint_array( curves[i] );

        curve_array[i] = pp[i].get();
    }
   
    cvFillPoly( ipl_image.get()
              , curve_array.get()           // needs to be pointer to C array of CvPoints.
              , &num_points_per_curve.front()
              , curves.size()
              , make_cvScalar( color )
              );
}

template< typename View
        , typename Line_Type
        >
inline
void drawFillPoly( View&                     view
                 , const curve_vec_t&        curves
                 , typename View::value_type color
                 , const Line_Type&          line_type
                 )
{
   drawFillPoly( create_ipl_image( view )
               , curves
               , color
               , line_type
               );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_DRAWING_HPP_INCLUDED
