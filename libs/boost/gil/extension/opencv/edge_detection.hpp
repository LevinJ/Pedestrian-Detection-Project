/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_EDGE_DETECTION_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_EDGE_DETECTION_HPP_INCLUDED

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

struct aperture_base {};

struct aperture1 : aperture_base, boost::mpl::int_< 1 > {};
struct aperture3 : aperture_base, boost::mpl::int_< 3 > {};
struct aperture5 : aperture_base, boost::mpl::int_< 5 > {};
struct aperture7 : aperture_base, boost::mpl::int_< 7 > {};
struct aperture_scharr : aperture_base, boost::mpl::int_< CV_SCHARR > {};

// default template parameter for function, only for classes
template< typename Aperture >
inline
void sobel( const ipl_image_wrapper& src
          , ipl_image_wrapper&       dst
          , const Aperture&
          , size_t                   x_order = 1
          , size_t                   y_order = 0
          , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                 , Aperture
                                                                 >::type
                                     >::type* ptr = 0
          )
{
   cvSobel( src.get()
          , dst.get()
          , x_order
          , y_order
          , Aperture::type::value
          );
}

template< typename View
        , typename Aperture
        >
inline
void sobel( View                  src
          , View                  dst
          , const Aperture&       aperture
          , std::size_t x_order = 1
          , std::size_t y_order = 0
          , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                 , Aperture
                                                                 >::type
                                     >::type* ptr = 0
          )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    sobel( src_ipl
         , dst_ipl
         , aperture
         , x_order
         , y_order
         );
}

template< typename Aperture >
inline
void laplace( const ipl_image_wrapper& src
            , ipl_image_wrapper&       dst
            , const Aperture&
            , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                   , Aperture
                                                                   >::type
                                       >::type* ptr = 0
            )
{
   cvLaplace( src.get()
            , dst.get()
            , Aperture::type::value
            );
}

template< typename View_Src
        , typename View_Dst
        , typename Aperture
        >
inline
void laplace( View_Src        src
            , View_Dst        dst
            , const Aperture& aperture
            , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                   , Aperture
                                                                   >::type
                                       >::type* ptr = 0
            )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    laplace( src_ipl
           , dst_ipl
           , aperture
           );
}

template< typename Aperture >
inline
void canny( const ipl_image_wrapper& src
          , ipl_image_wrapper&       dst
          , double                   threshold1
          , double                   threshold2
          , const Aperture&
          , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                 , Aperture
                                                                 >::type
                                       >::type* ptr = 0
            )
{
   cvCanny( src.get()
          , dst.get()
          , threshold1
          , threshold2
          , Aperture::type::value
          );
}

template< typename View
        , typename Aperture
        >
inline
void canny( View            src
          , View            dst
          , double          threshold1
          , double          threshold2
          , const Aperture& aperture
          , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                 , Aperture
                                                                 >::type
                                     >::type* ptr = 0
            )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    canny( src_ipl
         , dst_ipl
         , threshold1
         , threshold2
         , aperture
         );
}

template< typename Aperture >
inline
void precorner_detect( const ipl_image_wrapper& src
                     , ipl_image_wrapper&       dst
                     , const Aperture&
                     , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                            , Aperture
                                                                            >::type
                                                >::type* ptr = 0
                     )
{
   cvPreCornerDetect( src.get()
                    , dst.get()
                    , Aperture::type::value
                    );
}

template< typename View_Src
        , typename View_Dst
        , typename Aperture
        >
inline
void precorner_detect( View_Src        src
                     , View_Dst        dst
                     , const Aperture& aperture
                     , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                            , Aperture
                                                                            >::type
                                                >::type* ptr = 0
                      )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    precorner_detect( src_ipl
                    , dst_ipl
                    , aperture
                    );
}

template< typename Aperture >
inline
void corner_eigen_vals_and_vecs( const ipl_image_wrapper& src
                               , ipl_image_wrapper&       dst
                               , const std::size_t        block_size
                               , const Aperture&
                               , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                                      , Aperture
                                                                                      >::type
                                                          >::type* ptr = 0
                               )
{
    if( src.get()->width * 6 != dst.get()->width )
    {
        throw std::runtime_error( "Destination image must be 6 times wider." );
    }

    cvCornerEigenValsAndVecs( src.get()
                            , dst.get()
                            , block_size
                            , Aperture::type::value
                            );
}

template< typename View_Src
        , typename View_Dst
        , typename Aperture
        >
inline
void corner_eigen_vals_and_vecs( View_Src          src
                               , View_Dst          dst
                               , const std::size_t block_size
                               , const Aperture&   aperture
                               , typename boost::enable_if< typename boost::is_base_of< aperture_base
                                                                                      , Aperture
                                                                                      >::type
                                                          >::type* ptr = 0
                               )
{
    ipl_image_wrapper src_ipl = create_ipl_image( src );
    ipl_image_wrapper dst_ipl = create_ipl_image( dst );

    corner_eigen_vals_and_vecs( src_ipl
                              , dst_ipl
                              , block_size
                              , aperture
                              );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_EDGE_DETECTION_HPP_INCLUDED
