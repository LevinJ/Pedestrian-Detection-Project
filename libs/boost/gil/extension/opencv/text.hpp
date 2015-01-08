/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_TEXT_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_TEXT_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include <boost\shared_ptr.hpp>

#include "ipl_image_wrapper.hpp"
#include "drawing.hpp"

namespace boost { namespace gil { namespace opencv {

struct font_hershey_simplex        : boost::mpl::int_< CV_FONT_HERSHEY_SIMPLEX > {};
struct font_hershey_plain          : boost::mpl::int_< CV_FONT_HERSHEY_PLAIN > {};
struct font_hershey_duplex         : boost::mpl::int_< CV_FONT_HERSHEY_DUPLEX > {};
struct font_hershey_complex        : boost::mpl::int_< CV_FONT_HERSHEY_COMPLEX > {};
struct font_hershey_triplex        : boost::mpl::int_< CV_FONT_HERSHEY_TRIPLEX > {};
struct font_hershey_complex_small  : boost::mpl::int_< CV_FONT_HERSHEY_COMPLEX_SMALL > {};
struct font_hershey_script_simplex : boost::mpl::int_< CV_FONT_HERSHEY_SCRIPT_SIMPLEX > {};
struct font_hershey_script_complex : boost::mpl::int_< CV_FONT_HERSHEY_SCRIPT_COMPLEX > {};


typedef boost::shared_ptr< CvFont > ipl_font_wrapper;


template< typename Font_Face
        , typename Line_Type
        >
inline
ipl_font_wrapper create_ipl_font( const Font_Face& font_face
                                , double           hscale
                                , double           vscale
                                , const Line_Type& 
                                , double           shear     = 0
                                , std::size_t      thickness = 1
                                )
{
    ipl_font_wrapper ipl_font( new CvFont() );

    cvInitFont( ipl_font.get()
              , typename Font_Face::type::value
              , hscale
              , vscale
              , shear
              , thickness
              , typename Line_Type::type::value
              );

    return ipl_font;
}

template< typename Color >
inline
void putText( ipl_image_wrapper&       ipl_image
            , const std::string&       text
            , point_t                  org
            , const ipl_font_wrapper&  ipl_font
            , const Color&             color
            )
{
    cvPutText( ipl_image.get()
             , text.c_str()
             , make_cvPoint ( org )
             , ipl_font.get()
             , make_cvScalar( color )
             );
}

template< typename View
        , typename Color
        >
inline
void putText( View                     v
            , const std::string&       text
            , point_t                  org
            , const ipl_font_wrapper&  ipl_font
            , const Color&             color
            )
{
    ipl_image_wrapper ipl = create_ipl_image( v );

    putText( ipl
           , text
           , org
           , ipl_font
           , color
           );
}

inline
void getTextSize( const std::string&       text
                , const ipl_font_wrapper&  ipl_font
                , point_t&                 size
                , int&                     baseline
                )
{
    CvSize cv_size;

    cvGetTextSize( text.c_str()
                 , ipl_font.get()
                 , &cv_size
                 , &baseline
                 );

    size.x = cv_size.width;
    size.y = cv_size.height;
}

} // namespace opencv
} // namespace gil
} // namespace boost


#endif // BOOST_GIL_EXTENSION_OPENCV_TEXT_HPP_INCLUDED
