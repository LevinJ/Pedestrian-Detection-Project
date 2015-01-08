/*
    Copyright 2008 Christian Henning
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_OPENCV_IPL_IMAGE_WRAPPER_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_OPENCV_IPL_IMAGE_WRAPPER_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file               
/// \brief
/// \author Christian Henning \n
///         
/// \date 2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

#include <boost/shared_ptr.hpp>
#include <boost/gil/gil_all.hpp>
#include <stdexcept>

#include "utilities.hpp"

namespace boost { namespace gil { namespace opencv {

template < typename Channel > struct ipl_channel_type : boost::mpl::false_ {};
template<> struct ipl_channel_type< bits8 >   : boost::mpl::int_< IPL_DEPTH_8U  > {};
template<> struct ipl_channel_type< bits16 >  : boost::mpl::int_< IPL_DEPTH_16U > {};
template<> struct ipl_channel_type< bits32f > : boost::mpl::int_< IPL_DEPTH_32F > {};
template<> struct ipl_channel_type< double >  : boost::mpl::int_< IPL_DEPTH_64F > {};
template<> struct ipl_channel_type< bits8s >  : boost::mpl::int_< IPL_DEPTH_8S  > {};
template<> struct ipl_channel_type< bits16s > : boost::mpl::int_< IPL_DEPTH_16S > {};
template<> struct ipl_channel_type< bits32s > : boost::mpl::int_< IPL_DEPTH_32S > {};


/**
 *
 * ipl_image_wrapper encloses a IplImage pointer. Value semantics, like
 * copying, are supported by using shared_ptr.
 *
 **/
class ipl_image_wrapper
{
public:

    typedef boost::shared_ptr< IplImage > ipl_image_ptr_t;

public:
    ipl_image_wrapper() {}
    ipl_image_wrapper( IplImage* img ) : _img( img, ipl_deleter ) {}

    IplImage*       get()       { return _img.get(); }
    const IplImage* get() const { return _img.get(); }

private:

    static void ipl_deleter( IplImage* ipl_img )
    {
        if( ipl_img )
        {
            cvReleaseImageHeader( &ipl_img );
        }
    }

   ipl_image_ptr_t _img;
};

/**
  Will violate the constness of the view
  */
template< typename View >
inline
ipl_image_wrapper create_ipl_image( View view )
{
    typedef typename channel_type< View >::type channel_t;

    IplImage* img;

    if(( img = cvCreateImageHeader( make_cvSize( view.dimensions() )
                                  , ipl_channel_type<channel_t>::type::value
                                  , num_channels<View>::value
                                  )) == NULL )
    {
        throw std::runtime_error( "Cannot create IPL image." );
    }

    cvSetData( img
             , const_cast<void *>( static_cast<const void *>(&view.begin()[0]) )
             , num_channels<View>::value * view.width() * sizeof( channel_t ) );

    return ipl_image_wrapper( img );
}

} // namespace opencv
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_OPENCV_IPL_IMAGE_WRAPPER_HPP_INCLUDED
