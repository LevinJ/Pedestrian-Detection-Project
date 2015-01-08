//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXR_WRITER
#define EXR_WRITER

#include<vector>

#include<OpenEXR/ImfOutputFile.h>
#include<OpenEXR/ImfChannelList.h>

#include<boost/gil/gil_all.hpp>

#include"../half/channel.hpp"

#include"channel_traits.hpp"
#include"color_space_traits.hpp"

namespace boost
{
namespace gil
{

class exr_writer
{
public:

	exr_writer( const char *filename) : filename_(filename) {}

	template<typename View>
	void apply( const View& view, Imf::Header& header)
	{
	typedef typename color_space_type<View>::type color_space_t;
	typedef typename channel_type<View>::type channel_t;
	typedef pixel<channel_t, layout<color_space_t > > pixel_t;

		std::size_t n = num_channels<View>::value;		

		Imath::Box2i dw( header.dataWindow());
		int width  = dw.max.x - dw.min.x + 1;

		std::size_t xstride = sizeof( pixel_t);
		std::size_t ystride = xstride * width;

		std::vector<pixel_t> row( width);
		
		Imf::FrameBuffer frameBuffer;

                for( std::size_t i=0;i<n;++i)
		{
			const char *cname = detail::exr_color_space_traits<color_space_t>::channel_name[i];
			frameBuffer.insert( cname, Imf::Slice( Imf::FLOAT, 0, 0, 0, 1, 1));
			header.channels().insert( cname, Imf::Channel( detail::exr_channel2pixel_type<channel_t>::value));
		}

		Imf::OutputFile file( filename_, header);

		int y0 = dw.min.y;
		int y1 = dw.max.y;

		char *ptr = reinterpret_cast<char *>( &row.front()) - (y0 * ystride) - (dw.min.x * xstride);
		
		for( int y=y0;y<=y1;++y)
		{
			std::copy( view.row_begin( y - y0), view.row_begin( y - y0)+width, &row.front());
		
			char *p = ptr;
                        for( std::size_t i=0;i<n;++i)
			{
				const char *cname = detail::exr_color_space_traits<color_space_t>::channel_name[i];
				frameBuffer[cname] = Imf::Slice( detail::exr_channel2pixel_type<channel_t>::value, p, xstride, ystride);
				p += sizeof( channel_t);
			}

			file.setFrameBuffer( frameBuffer);
			file.writePixels( 1);
			ptr -= ystride;
		}		
	}

private:

	const char *filename_;
};

} // namespace
} // namespace

#endif
