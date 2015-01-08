//  Copyright Esteban Tovagliari 2007. Use, modification and
//  distribution is subject to the Boost Software License, Version
//  1.0. (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef GIL_EXR_IO_HPP
#define GIL_EXR_IO_HPP

#include"detail/exr_reader.hpp"
#include"detail/exr_writer.hpp"

namespace boost
{
namespace gil
{

template<class View>
inline void exr_read_view( const char *filename, const View& view)
{
	Imf::InputFile file( filename);
	Imath::Box2i crop( file.header().dataWindow());
	exr_reader m( file);
	m.apply( view, crop);
}

template<class View>
inline void exr_read_view( Imf::InputFile& file, const View& view)
{
	Imath::Box2i crop( file.header().dataWindow());
	exr_reader m( file);
	m.apply( view, crop);
}

template<class View>
inline void exr_read_view( Imf::InputFile& file, const View& view, const Imath::Box2i& crop)
{
	exr_reader m( file);
	m.apply( view, crop);
}

template<class Image>
inline void exr_read_image( const char *filename, Image& img)
{
	Imf::InputFile file( filename);
	exr_read_image( file, img);
}

template<class Image>
inline void exr_read_image( Imf::InputFile& file, Image& img)
{
	Imath::Box2i crop( file.header().dataWindow());
	img.recreate( crop.max.x - crop.min.x + 1, crop.max.y - crop.min.y + 1);
	exr_reader m( file);
	m.apply( view(img), crop);
}

template<class Image>
inline void exr_read_image( Imf::InputFile& file, Image& img, const Imath::Box2i& crop)
{
	img.recreate( crop.max.x - crop.min.x + 1, crop.max.y - crop.min.y + 1);
	exr_reader m( file);
	m.apply( view(img), crop);
}

template<class View>
inline void exr_write_view( const char *filename, const View& view)
{
	Imf::Header header( view.width(), view.height(), 1);
	exr_write_view( filename, header, view);
}

template<class View>
inline void exr_write_view( const char *filename, Imf::Header& header, const View& view)
{
	exr_writer m( filename);
	m.apply( view, header);
}

} // namespace
} // namespace

#endif
