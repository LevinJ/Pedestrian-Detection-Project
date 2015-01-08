#include "AlignedImage.hpp"

#include <boost/gil/image_view_factory.hpp>

namespace doppia {


using namespace boost;
using namespace boost::gil;

AlignedImage::AlignedImage()
{
    // nothing to do here
    return;
}


AlignedImage::~AlignedImage()
{
    // nothing to do here
    return;
}




AlignedImage::AlignedImage(const point_t dimensions)
{
    resize(dimensions);
    return;
}

AlignedImage::AlignedImage(const size_t width, const size_t height)
{
    resize(width, height);
    return;
}


bool AlignedImage::empty() const
{
    return data.empty();
}

void AlignedImage::resize(const point_t dimensions)
{
    resize(dimensions.x, dimensions.y);
    return;
}

void AlignedImage::resize(const size_t width, const size_t height)
{
    // the in memory images need to have row with multiples of 16 bytes --

    const size_t input_row_size_in_bytes = width * sizeof(rgb8_pixel_t);
    const size_t row_size_modulo_16 = (input_row_size_in_bytes % 16);
    size_t data_row_size_in_bytes = input_row_size_in_bytes;
    if (row_size_modulo_16 > 0)
    {
        data_row_size_in_bytes+= (16 - row_size_modulo_16);
    }
    assert(data_row_size_in_bytes >= input_row_size_in_bytes);

    //printf("input_row_size_in_bytes == %zi\n", input_row_size_in_bytes);
    //printf("row_size_modulo_16 == %zi\n", row_size_modulo_16);
    //printf("data_row_size_in_bytes == %zi\n", data_row_size_in_bytes);
    //printf("Creating image_data of size ( %.3f, %zi)\n",
    //       static_cast<float>(data_row_size_in_bytes) / sizeof(rgb8_pixel_t), input_dimensions.y);

    data.resize(boost::extents[height][data_row_size_in_bytes]);

    image_view = interleaved_view(width, height,
                                  reinterpret_cast<rgb8_pixel_t *>(data.data()),
                                  data_row_size_in_bytes);
    image_const_view = image_view;
    return;
}

const AlignedImage::point_t &AlignedImage::dimensions() const
{
    return image_view.dimensions();
}

AlignedImage::data_t &AlignedImage::get_multi_array()
{
    return data;
}

const AlignedImage::data_t &AlignedImage::get_multi_array() const
{
    return data;
}


const AlignedImage::view_t &AlignedImage::get_view()
{
    return image_view;
}

const AlignedImage::const_view_t &AlignedImage::get_view() const
{
    return image_const_view;
}

} // end of namespace doppia
