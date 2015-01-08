/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_CONVERT_COLOR_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_CONVERT_COLOR_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include <boost/static_assert.hpp>

#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/toolbox/xyz.hpp>
#include <boost/gil/extension/toolbox/hsl.hpp>
#include <boost/gil/extension/toolbox/hsv.hpp>
#include <boost/gil/extension/toolbox/lab.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>

#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/utility/enable_if.hpp>

#include "ipl_image_wrapper.hpp"

namespace boost { namespace gil { namespace opencv {

typedef bit_aligned_image3_type<  5,  5,  5, bgr_layout_t >::type bgr555_image_t;
typedef bgr555_image_t::view_t::value_type bgr555_pixel_t;

typedef bit_aligned_image3_type<  5,  6,  5, bgr_layout_t >::type bgr565_image_t;
typedef bgr565_image_t::view_t::value_type bgr565_pixel_t;

template< typename PixelRefT>
struct is_bit_aligned : mpl::false_{};

template <typename B, typename C, typename L, bool M>  
struct is_bit_aligned<bit_aligned_pixel_reference<B,C,L,M> > : mpl::true_{};

template <typename B, typename C, typename L, bool M>  
struct is_bit_aligned<const bit_aligned_pixel_reference<B,C,L,M> > : mpl::true_{};

template <typename B, typename C, typename L>  
struct is_bit_aligned<packed_pixel<B,C,L> > : mpl::true_{};

template <typename B, typename C, typename L>  
struct is_bit_aligned<const packed_pixel<B,C,L> > : mpl::true_{};

/*
// not implemented
#define  CV_BGR2YCrCb   36
#define  CV_RGB2YCrCb   37
#define  CV_YCrCb2BGR   38
#define  CV_YCrCb2RGB   39

#define  CV_BayerBG2BGR 46
#define  CV_BayerGB2BGR 47
#define  CV_BayerRG2BGR 48
#define  CV_BayerGR2BGR 49

#define  CV_BayerBG2RGB CV_BayerRG2BGR
#define  CV_BayerGB2RGB CV_BayerGR2BGR
#define  CV_BayerRG2RGB CV_BayerBG2BGR
#define  CV_BayerGR2RGB CV_BayerGB2BGR

#define  CV_BGR2Luv     50
#define  CV_RGB2Luv     51
#define  CV_Luv2BGR     58
#define  CV_Luv2RGB     59
*/


template< typename T1, typename T2 > struct is_supported : boost::mpl::false_ {};

// 3 channel to 4 channel with equal layout
template<> struct is_supported< bgr_layout_t, bgra_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2BGRA; };
template<> struct is_supported< rgb_layout_t, rgba_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2RGBA; };

// 4 channel to 3 channel with equal layout
template<> struct is_supported< bgra_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGRA2BGR; };
template<> struct is_supported< rgba_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2RGB; };

// 3 channel to 4 channel with different layout
template<> struct is_supported< bgr_layout_t, rgba_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2RGBA; };
template<> struct is_supported< rgb_layout_t, bgra_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2BGRA; };

// 4 channel to 3 channel with different layout
template<> struct is_supported< rgba_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2BGR; };
template<> struct is_supported< bgra_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2BGRA; };

// 3 channel to 3 channel with different layout
template<> struct is_supported< bgr_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2BGR; };
template<> struct is_supported< rgb_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2BGR; };

// 4 channel to 4 channel with different layout
template<> struct is_supported< bgra_layout_t, rgba_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGRA2RGBA; };
template<> struct is_supported< rgba_layout_t, bgra_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2BGRA; };

// BGR to Gray
template<> struct is_supported< bgr_layout_t, gray_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2GRAY; };

// RGB to Gray
template<> struct is_supported< rgb_layout_t, gray_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2GRAY; };

// Gray to BGR
template<> struct is_supported< gray_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_GRAY2BGR; };

// Gray to RGB
template<> struct is_supported< gray_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_GRAY2RGB; };

// Gray to BGRA
template<> struct is_supported< gray_layout_t, bgra_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_GRAY2BGRA; };

// Gray to RGBA
template<> struct is_supported< gray_layout_t, rgba_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_GRAY2RGBA; };

// BGRA to Gray
template<> struct is_supported< bgra_layout_t, gray_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGRA2GRAY; };

// RGBA to Gray
template<> struct is_supported< rgba_layout_t, gray_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2GRAY; };

// BGR to BGR565
template<> struct is_supported< bgr_layout_t, bgr565_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2BGR565; };

// RGB to BGR565
template<> struct is_supported< rgb_layout_t, bgr565_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2BGR565; };

// BGR565 to BGR
template<> struct is_supported< bgr565_pixel_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5652BGR; };

// BGR565 to RGB
template<> struct is_supported< bgr565_pixel_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5652RGB; };

// BGRA to BGR565
template<> struct is_supported< bgra_layout_t, bgr565_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_BGRA2BGR565; };

// RGBA to BGR565
template<> struct is_supported< rgba_layout_t, bgr565_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2BGR565; };

// BGR565 to BGRA
template<> struct is_supported< bgr565_pixel_t, bgra_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5652BGRA; };

// BGR565 to RGBA
template<> struct is_supported< bgr565_pixel_t, rgba_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5652RGBA; };

// Gray to BGR565
template<> struct is_supported< gray_layout_t, bgr565_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_GRAY2BGR565; };

// BGR565 to Gray
template<> struct is_supported< bgr565_pixel_t, gray_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5652GRAY; };

// BGR to BGR555
template<> struct is_supported< bgr_layout_t, bgr555_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2BGR555; };

// RGB to BGR555
template<> struct is_supported< rgb_layout_t, bgr555_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2BGR555; };

// BGR555 to BGR
template<> struct is_supported< bgr555_pixel_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5552BGR; };

// BGR555 to RGB
template<> struct is_supported< bgr555_pixel_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5552RGB; };

// BGR to BGR555
template<> struct is_supported< bgra_layout_t, bgr555_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_BGRA2BGR555; };

// RGBA to BGR555
template<> struct is_supported< rgba_layout_t, bgr555_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_RGBA2BGR555; };

// BGR555 to BGRA
template<> struct is_supported< bgr555_pixel_t, bgra_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5552BGRA; };

// BGR555 to RGBA
template<> struct is_supported< bgr555_pixel_t, rgba_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5552RGBA; };

// Gray to BGR555
template<> struct is_supported< gray_layout_t, bgr555_pixel_t > : public boost::mpl::true_ 
{ static const int code = CV_GRAY2BGR555; };

// BGR555 to Gray
template<> struct is_supported< bgr555_pixel_t, gray_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR5552GRAY; };


// BGR to XYZ
template<> struct is_supported< bgr_layout_t, xyz_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2XYZ; };

// RGB to XYZ
template<> struct is_supported< rgb_layout_t, xyz_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2XYZ; };

// XYZ to BGR
template<> struct is_supported< xyz_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_XYZ2BGR; };

// XYZ to RGB
template<> struct is_supported< xyz_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_XYZ2RGB; };

// BGR to HSV
template<> struct is_supported< bgr_layout_t, hsv_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2HSV; };

// RGB to HSV
template<> struct is_supported< rgb_layout_t, hsv_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2HSV; };

// BGR to Lab
template<> struct is_supported< bgr_layout_t, lab_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2Lab; };

// RGB to Lab
template<> struct is_supported< rgb_layout_t, lab_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2Lab; };

// BGR to HLS
template<> struct is_supported< bgr_layout_t, hsl_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_BGR2HLS; };

// RGB to HLS
template<> struct is_supported< rgb_layout_t, hsl_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_RGB2HLS; };

// HSV to BGR
template<> struct is_supported< hsv_layout_t, bgr_layout_t > : public boost::mpl::true_
{ static const int code = CV_HSV2BGR; };

// HSV to RGB
template<> struct is_supported< hsv_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_HSV2RGB; };

// Lab to BGR
template<> struct is_supported< lab_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_Lab2BGR; };

// Lab to RGB
template<> struct is_supported< lab_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_Lab2RGB; };

// HLS to BGR
template<> struct is_supported< hsl_layout_t, bgr_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_HLS2BGR; };

// HLS to RGB
template<> struct is_supported< hsl_layout_t, rgb_layout_t > : public boost::mpl::true_ 
{ static const int code = CV_HLS2RGB; };

// Allowed channel types

template< typename Channel > struct allowed_channel_type : boost::mpl::false_ {};
template <> struct allowed_channel_type< bits8   > : boost::mpl::true_ {};
template <> struct allowed_channel_type< bits16  > : boost::mpl::true_ {};
template <> struct allowed_channel_type< bits32f > : boost::mpl::true_ {};

template< typename Is_Supported >
inline
void cvtcolor_impl( const ipl_image_wrapper& src
                  , ipl_image_wrapper&       dst
                  , const Is_Supported&
                  , typename boost::enable_if< typename boost::is_base_of< boost::mpl::false_
                                                                         , Is_Supported
                                                                         >::type
                                             >::type* ptr = 0
                  )
{
    // conversion isn't supported
    BOOST_STATIC_ASSERT(( 0 ));
}

template< typename Is_Supported >
inline
void cvtcolor_impl( const ipl_image_wrapper& src
                  , ipl_image_wrapper&       dst
                  , const Is_Supported&
                  , typename boost::enable_if< typename boost::is_base_of< boost::mpl::true_
                                                                         , Is_Supported
                                                                         >::type
                                             >::type* ptr = 0
                  )
{
    cvCvtColor( src.get()
              , dst.get()
              , Is_Supported::code
              );
}

template< typename View_Src
        , typename View_Dst
        >
inline
void cvtcolor( View_Src src
             , View_Dst dst
             , const boost::mpl::true_& // is bit aligned
             )
{
    typedef typename View_Src::value_type::layout_t SrcLayout;
    typedef typename View_Dst::value_type::layout_t DstLayout;

    typedef typename boost::mpl::if_< is_bit_aligned< View_Src::value_type >
                                    , View_Src::value_type
                                    , View_Src::value_type::layout_t
                                    >::type src_t;

    typedef typename boost::mpl::if_< is_bit_aligned< View_Dst::value_type >
                                    , View_Dst::value_type
                                    , View_Dst::value_type::layout_t
                                    >::type dst_t;

    /// @todo: Not implemented yet. Make sure we can create bit aligned ipl images.
    BOOST_STATIC_ASSERT(( 0 ));

/*
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    cvtcolor_impl( src_ipl
                 , dst_ipl
                 , is_supported< src_t
                               , dst_t
                               >()
                 );
*/
}

template< typename View_Src
        , typename View_Dst
        >
inline
void cvtcolor( View_Src src
             , View_Dst dst
             , const boost::mpl::false_& // is bit aligned
             )
{
    typedef typename channel_type< View_Src >::type src_channel_t;
    typedef typename channel_type< View_Dst >::type dst_channel_t;

    // Only 8u, 16u, and 32f is allowed.
    BOOST_STATIC_ASSERT(( boost::mpl::or_< allowed_channel_type< src_channel_t >
                                         , allowed_channel_type< dst_channel_t >
                                         >::value ));

    // Channel depths need to match.
    BOOST_STATIC_ASSERT(( boost::is_same< src_channel_t
                                        , dst_channel_t >::value ));

    typedef typename View_Src::value_type::layout_t SrcLayout;
    typedef typename View_Dst::value_type::layout_t DstLayout;

    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    cvtcolor_impl( src_ipl
                 , dst_ipl
                 , is_supported< SrcLayout
                               , DstLayout
                               >()
                 );
}

template< typename View_Src
        , typename View_Dst
        >
inline
void cvtcolor( View_Src src
             , View_Dst dst
             )
{
    if(  std::max( src.dimensions().x, dst.dimensions().x ) == 0
      && std::max( src.dimensions().y, dst.dimensions().y ) == 0
      )
    {
        throw std::exception( "Image doesn't have dimensions ( empty image )." );
    }
    


    if( src.dimensions() != dst.dimensions() )
    {
        throw std::exception( "Image's dimensions don't match." );
    }

    // There is some special code for dealing with bit_aligned images.
    // OpenCV seems only to support rgb|bgr555 or rgb|bgr565 images.
    typedef boost::mpl::or_< is_bit_aligned< View_Src::value_type >::type
                           , is_bit_aligned< View_Dst::value_type >::type
                           >::type is_bit_aligned_t;

    cvtcolor( src
            , dst
            , is_bit_aligned_t()
            );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_CONVERT_COLOR_HPP_INCLUDED
