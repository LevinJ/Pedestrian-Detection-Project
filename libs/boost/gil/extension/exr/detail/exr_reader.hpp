//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef EXR_READER
#define EXR_READER

#include<vector>

#include<OpenEXR/ImfInputFile.h>

#include<boost/gil/gil_all.hpp>

#include"../half/channel.hpp"

#include"channel_traits.hpp"
#include"color_space_traits.hpp"

namespace boost
{
namespace gil
{

class exr_reader
{
public:

	exr_reader( Imf::InputFile& file) : file_(file) {}

	template<typename View>
	void apply( const View& view, const Imath::Box2i& crop)
	{
	typedef typename color_space_type<View>::type color_space_t;
	typedef typename channel_type<View>::type channel_t;
	typedef pixel<channel_t, layout<color_space_t > > pixel_t;

		std::size_t n = num_channels<View>::value;
		Imath::Box2i dw( file_.header().dataWindow());
		int width  = dw.max.x - dw.min.x + 1;
		std::vector<pixel_t> row( width);

		assert( view.width()  == (crop.max.x - crop.min.x + 1));
		assert( view.height() == (crop.max.y - crop.min.y + 1));

		Imf::FrameBuffer frameBuffer;

		for( int i=0;i<n;++i)
			frameBuffer.insert( detail::exr_color_space_traits<color_space_t>::channel_name[i], Imf::Slice( Imf::FLOAT, 0, 0, 0, 1, 1, 0.0));
		
		std::size_t xstride = sizeof( pixel_t);
		std::size_t ystride = xstride * width;
	
		int y0 = crop.min.y; int x0 = crop.min.x;
		int y1 = crop.max.y; int x1 = crop.max.x;
		int xbegin = crop.min.x - dw.min.x;
		int xend   = xbegin + x1 - x0 + 1;
		
		char *ptr = reinterpret_cast<char *>( &row.front()) - (y0 * ystride) - (dw.min.x * xstride);
		
		for( int y=y0;y<y1;++y)
		{
			char *p = ptr;
			for( int i=0;i<n;++i)
			{
				const char *cname = detail::exr_color_space_traits<color_space_t>::channel_name[i];
				frameBuffer[cname] = Imf::Slice( detail::exr_channel2pixel_type<channel_t>::value, p, xstride, ystride, 1, 1, 0.0);
				p += sizeof( channel_t);
			}

			file_.setFrameBuffer( frameBuffer);
			file_.readPixels( y, y + 1);

			std::copy( &row.front() + xbegin, &row.front() + xend, view.row_begin( y - y0));
			ptr -= ystride;
		}
	}

private:

	Imf::InputFile& file_;
};

} // namespace
} // namespace

#endif
