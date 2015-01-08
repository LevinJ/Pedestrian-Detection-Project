/*
    Copyright 2007-2008 Christian Henning, Lubomir Bourdev
    Use, modification and distribution are subject to the Boost Software License,
    Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
*/

/*************************************************************************************************/

#ifndef BOOST_GIL_EXTENSION_IO_TIFF_IO_READ_HPP_INCLUDED
#define BOOST_GIL_EXTENSION_IO_TIFF_IO_READ_HPP_INCLUDED

////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief
/// \author Christian Henning, Lubomir Bourdev \n
///
/// \date   2007-2008 \n
///
////////////////////////////////////////////////////////////////////////////////////////

extern "C" {
#include "tiff.h"
#include "tiffio.h"
}

#include <algorithm>
#include <string>
#include <vector>
#include <boost/static_assert.hpp>

#include <boost/gil/extension/io_new/detail/base.hpp>
#include <boost/gil/extension/io_new/detail/conversion_policies.hpp>
#include <boost/gil/extension/io_new/detail/bit_operations.hpp>
#include <boost/gil/extension/io_new/detail/row_buffer_helper.hpp>
#include <boost/gil/extension/io_new/detail/io_device.hpp>
#include <boost/gil/extension/io_new/detail/reader_base.hpp>

#include "device.hpp"
#include "is_allowed.hpp"


namespace boost { namespace gil { namespace detail {

template < int K >
struct plane_recursion
{
   template< typename View
           , typename Device
           , typename ConversionPolicy
           >
   static
   void read_plane( const View& dst_view
                  , reader< Device
                          , tiff_tag
                          , ConversionPolicy >* p
                  )
   {
      typedef typename kth_channel_view_type< K, View >::type plane_t;
      plane_t plane = kth_channel_view<K>( dst_view );
      if(!p->_io_dev.is_tiled())
        p->read_data( plane, K );
      else
          p->read_tiled_data( plane, K );

      plane_recursion< K - 1 >::read_plane( dst_view, p );
   }
};

template <>
struct plane_recursion< -1 >
{
   template< typename View
           , typename Device
           , typename ConversionPolicy
           >
   static
   void read_plane( const View&               /* dst_view */
                  , reader< Device
                          , tiff_tag
                          , ConversionPolicy
                          >*                  /* p         */
                  )
    {}
};

template< typename Device
        , typename ConversionPolicy
        >
class reader< Device
            , tiff_tag
            , ConversionPolicy
            >
    : public reader_base< tiff_tag
                        , ConversionPolicy >
{
public:

    reader( Device&                                device
          , const image_read_settings< tiff_tag >& settings
          )
    : reader_base< tiff_tag
                 , ConversionPolicy
                 >( settings )
    , _io_dev( device )
    {}

    reader( Device&                                                device
          , const typename ConversionPolicy::color_converter_type& cc
          , const image_read_settings< tiff_tag >&                 settings
          )
    : reader_base< tiff_tag
                 , ConversionPolicy
                 >( cc
                  , settings
                  )
    , _io_dev( device )
    {}

   image_read_info<tiff_tag> get_info() const
   {
      image_read_info<tiff_tag> info;

      io_error_if( _io_dev.template get_property<tiff_image_width>               ( info._width ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_image_height>              ( info._height ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_compression>               ( info._compression ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_samples_per_pixel>         ( info._samples_per_pixel ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_bits_per_sample>           ( info._bits_per_sample ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_sample_format>             ( info._sample_format ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_planar_configuration>      ( info._planar_configuration ) == false
                 , "cannot read tiff tag." );
      io_error_if( _io_dev.template get_property<tiff_photometric_interpretation>( info._photometric_interpretation  ) == false
                 , "cannot read tiff tag." );

      // Tile tags
      if(_io_dev.is_tiled())
      {
          io_error_if( !_io_dev.template get_property<tiff_tile_width>      ( info._tile_width )
                       , "cannot read tiff_tile_width tag." );
          io_error_if( !_io_dev.template get_property<tiff_tile_length>     ( info._tile_length )
                       , "cannot read tiff_tile_length tag." );
          io_error_if( !_io_dev.template get_property<tiff_tile_offsets>    ( info._tile_offsets )
                       , "cannot read tiff_tile_offsets tag." );
          io_error_if( !_io_dev.template get_property<tiff_tile_byte_counts>( info._tile_byte_counts )
                       , "cannot read tiff_tile_byte_counts tag." );
      }

      return info;
   }

    // only works for homogeneous image types
    template< typename View >
    void apply( View& dst_view )
    {
        if( this->_info._photometric_interpretation == PHOTOMETRIC_PALETTE )
        {
            // Steps:
            // 1. Read indices. It's an array of grayX_pixel_t.
            // 2. Read palette. It's an array of rgb16_pixel_t.
            // 3. ??? Create virtual image or transform the two arrays
            //    into a rgb16_image_t object. The latter might
            //    be a good first solution.

            switch( this->_info._bits_per_sample )
            {
                case 1:  { read_palette_image< gray1_image_t  >( dst_view ); break; }
                case 2:  { read_palette_image< gray2_image_t  >( dst_view ); break; }
                case 4:  { read_palette_image< gray4_image_t  >( dst_view ); break; }
                case 8:  { read_palette_image< gray8_image_t  >( dst_view ); break; }
                case 16: { read_palette_image< gray16_image_t >( dst_view ); break; }

                default: { io_error( "Not supported palette " ); }
            }

            return;

        }
        else
        {
            // In case we only read the image the user's type and
            // the tiff type need to compatible. Which means:
            // color_spaces_are_compatible && channels_are_pairwise_compatible

            typedef typename is_same< ConversionPolicy
                                    , read_and_no_convert
                                    >::type is_read_and_convert_t;

            io_error_if( !is_allowed< View >( this->_info
                                            , is_read_and_convert_t()
                                            )
                       , "Image types aren't compatible."
                       );

            if( this->_info._planar_configuration == PLANARCONFIG_SEPARATE )
            {
                plane_recursion< num_channels< View >::value - 1 >::read_plane( dst_view
                                                                              , this
                                                                              );
            }
            else if( this->_info._planar_configuration == PLANARCONFIG_CONTIG )
            {
                if(_io_dev.is_tiled())
                {
                    read_tiled_data( dst_view, 0 );
                }
                else
                {
                    read_data( dst_view, 0 );
                }
            }
            else
            {
                io_error( "Wrong planar configuration setting." );
            }
        }
    }

private:

   template< typename PaletteImage
           , typename View
           >
   void read_palette_image( const View& dst_view )
   {
      PaletteImage indices( this->_info._width  - this->_settings._top_left.x
                          , this->_info._height - this->_settings._top_left.y );

      read_data( view( indices ), 0 );

      read_palette_image( dst_view
                        , view( indices )
                        , typename is_same< View, rgb16_view_t >::type() );
   }

   template< typename View
           , typename Indices_View
           >
   void read_palette_image( const View&         dst_view
                          , const Indices_View& indices_view
                          , mpl::true_   // is View rgb16_view_t
                          )
   {
      tiff_color_map::red_t   red   = NULL;
      tiff_color_map::green_t green = NULL;
      tiff_color_map::blue_t  blue  = NULL;

      _io_dev.get_field_defaulted( red, green, blue );

      typedef typename channel_traits<
                    typename element_type<
                            typename Indices_View::value_type >::type >::value_type channel_t;

      int num_colors = channel_traits< channel_t >::max_value();

      rgb16_planar_view_t palette = planar_rgb_view( num_colors
                                                   , 1
                                                   , red
                                                   , green
                                                   , blue
                                                   , sizeof( bits16 ) * num_colors );

      typename rgb16_planar_view_t::x_iterator palette_it = palette.row_begin( 0 );

      for( typename rgb16_view_t::y_coord_t y = 0; y < dst_view.height(); ++y )
      {
         typename rgb16_view_t::x_iterator it  = dst_view.row_begin( y );
         typename rgb16_view_t::x_iterator end = dst_view.row_end( y );

         typename Indices_View::x_iterator indices_it = indices_view.row_begin( y );

         for( ; it != end; ++it, ++indices_it )
         {
            bits16 i = gil::at_c<0>( *indices_it );

            *it = palette[i];
         }
      }
   }

   template< typename View
           , typename Indices_View
           >
   inline
   void read_palette_image( const View&         /* dst_view     */
                          , const Indices_View& /* indices_view */
                          , mpl::false_  // is View rgb16_view_t
                          )
   {
      io_error( "User supplied image type must be rgb16_image_t." );
   }

   template< typename Buffer >
   void skip_over_rows( Buffer& buffer
                      , int     plane
                      )
   {
      if( this->_info._compression != COMPRESSION_NONE )
      {
         // Skipping over rows is not possible for compressed images(  no random access ). See man
         // page ( diagnostics section ) for more information.
         for( std::ptrdiff_t row = 0; row < this->_settings._top_left.y; ++row )
         {
            _io_dev.read_scanline( buffer
                                 , row
                                 , static_cast< tsample_t >( plane ));
         }
      }
   }

   template< typename View >
   void read_tiled_data( const View& dst_view
                         , int         plane     )
   {
       typedef typename is_bit_aligned< typename View::value_type >::type is_view_bit_aligned_t;
       typedef row_buffer_helper_view< View > row_buffer_helper_t;
       typedef typename row_buffer_helper_t::buffer_t   buffer_t;
       typedef typename row_buffer_helper_t::iterator_t it_t;

       // TIFFReadTile needs a buffer with enough memory allocated. This size can easily be computed by TIFFTileSize (implemented in device.hpp)
       std::size_t size_to_allocate = _io_dev.get_tile_size();
       row_buffer_helper_t row_buffer_helper( size_to_allocate, true );
       uint32_t plain_tile_size = this->_info._tile_width*this->_info._tile_length;

       //@todo Is _io_dev.are_bytes_swapped() == true when reading bit_aligned images?
       //      If the following fires then we need to pass a boolean to the constructor.
       io_error_if( is_bit_aligned< View >::value && !_io_dev.are_bytes_swapped()
                    , "Cannot be read."
                    );

       mirror_bits< buffer_t
               , typename is_bit_aligned< View >::type
               > mirror_bits;

       for (unsigned int y = 0; y < this->_info._height; y += this->_info._tile_length)
       {
           for (unsigned int x = 0; x < this->_info._width; x += this->_info._tile_width)
           {
               uint32_t current_tile_width  = (x+this->_info._tile_width<this->_info._width  ) ? this->_info._tile_width  : this->_info._width -x;
               uint32_t current_tile_length = (y+this->_info._tile_length<this->_info._height) ? this->_info._tile_length : this->_info._height-y;

               _io_dev.read_tile( row_buffer_helper.buffer()
                                  , x
                                  , y
                                  , 0
                                  , static_cast< tsample_t >( plane )
                                  );

               mirror_bits( row_buffer_helper.buffer() );

               View tile_subimage_view = subimage_view(dst_view, x, y, current_tile_width, current_tile_length);

               if ( current_tile_width*current_tile_length == plain_tile_size )
               {
                   it_t first = row_buffer_helper.begin();
                   it_t last  = first + current_tile_width*current_tile_length;

                   this->_cc_policy.read( first
                                          , last
                                          , tile_subimage_view.begin()
                                          );
               }
               else
               {
                   // When the current tile is smaller than a "normal" tile (image borders), we have to iterate over each row
                   // to get the firsts 'current_tile_width' pixels.
                   for(unsigned int tile_row=0;tile_row<current_tile_length;++tile_row)
                   {
                       it_t first = row_buffer_helper.begin() + tile_row*this->_info._tile_width;
                       it_t last  = first + current_tile_width;

                       this->_cc_policy.read( first
                                              , last
                                              , tile_subimage_view.begin() + tile_row*current_tile_width
                                              );
                   }
               }
           }
       }
   }

   template< typename View >
   void read_data( const View& dst_view
                 , int         plane     )
   {
      typedef typename is_bit_aligned< typename View::value_type >::type is_view_bit_aligned_t;

      typedef row_buffer_helper_view< View > row_buffer_helper_t;

      typedef typename row_buffer_helper_t::buffer_t   buffer_t;
      typedef typename row_buffer_helper_t::iterator_t it_t;

      std::size_t size_to_allocate = buffer_size< typename View::value_type >( dst_view.width()
                                                                             , is_view_bit_aligned_t() );
      row_buffer_helper_t row_buffer_helper( size_to_allocate, true );

      it_t begin = row_buffer_helper.begin();

      it_t first = begin + this->_settings._top_left.x;
      it_t last  = first + this->_settings._dim.x; // one after last element

      skip_over_rows( row_buffer_helper.buffer()
                    , plane
                    );


      //@todo Is _io_dev.are_bytes_swapped() == true when reading bit_aligned images?
      //      If the following fires then we need to pass a boolean to the constructor.
      io_error_if( is_bit_aligned< View >::value && !_io_dev.are_bytes_swapped()
                 , "Cannot be read."
                 );

      mirror_bits< buffer_t
                 , typename is_bit_aligned< View >::type
                 > mirror_bits;

      for( std::ptrdiff_t row = 0
         ; row < this->_settings._dim.y
         ; ++row
         )
      {
         _io_dev.read_scanline( row_buffer_helper.buffer()
                              , row
                              , static_cast< tsample_t >( plane )
                              );

         mirror_bits( row_buffer_helper.buffer() );

         this->_cc_policy.read( first
                              , last
                              , dst_view.row_begin( row ));
      }
   }

   template< typename Pixel >
   std::size_t buffer_size( std::size_t width
                          , mpl::false_ // is_bit_aligned
                          )
   {
      std::size_t scanline_size_in_bytes = _io_dev.get_scanline_size();

      std::size_t element_size = sizeof( Pixel );

      return  std::max( width
                      , (( scanline_size_in_bytes + element_size - 1 ) / element_size ));
   }

   template< typename Pixel >
   std::size_t buffer_size( std::size_t /* width */
                          , mpl::true_  // is_bit_aligned
                          )
   {
      return _io_dev.get_scanline_size();
   }

private:

   Device& _io_dev;

   template < int K > friend struct plane_recursion;
};

struct tiff_type_format_checker
{
    tiff_type_format_checker( const image_read_info< tiff_tag >& info )
    : _info( info )
    {}

    template< typename Image >
    bool apply()
    {
        typedef typename Image::view_t view_t;

        return is_allowed< view_t >( _info
                                   , mpl::true_()
                                   );
    }

private:

    const image_read_info< tiff_tag >& _info;
};

struct tiff_read_is_supported
{
    template< typename View >
    struct apply : public is_read_supported< typename get_pixel_type< View >::type
                                           , tiff_tag
                                           >
    {};
};

template< typename Device
        >
class dynamic_image_reader< Device
                          , tiff_tag
                          >
    : public reader< Device
                   , tiff_tag
                   , detail::read_and_no_convert
                   >
{
    typedef reader< Device
                  , tiff_tag
                  , detail::read_and_no_convert
                  > parent_t;

public:

    dynamic_image_reader( Device&                                device
                        , const image_read_settings< tiff_tag >& settings
                        )
    : parent_t( device
              , settings
              )
    {}

    template< typename Images >
    void apply( any_image< Images >& images )
    {
        tiff_type_format_checker format_checker( this->_info );

        if( !construct_matched( images
                              , format_checker
                              ))
        {
            io_error( "No matching image type between those of the given any_image and that of the file" );
        }
        else
        {
            init_image( images
                      , this->_info
                      );

            dynamic_io_fnobj< tiff_read_is_supported
                            , parent_t
                            > op( this );

            apply_operation( view( images )
                           , op
                           );
        }
    }
};


} // namespace detail
} // namespace gil
} // namespace boost

#endif // BOOST_GIL_EXTENSION_IO_TIFF_IO_READ_HPP_INCLUDED
