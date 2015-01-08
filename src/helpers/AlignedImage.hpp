#ifndef ALIGNEDIMAGE_HPP
#define ALIGNEDIMAGE_HPP

#include <boost/multi_array.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view.hpp>

#include <boost/cstdint.hpp>

#include <Eigen/Core>

namespace doppia {

/// helper class that provides access to an image where
/// every row is 16 bytes memory aligned
class AlignedImage
{

public:
    typedef Eigen::aligned_allocator<boost::uint8_t> aligned_uint8_allocator;
    typedef boost::multi_array< boost::uint8_t, 2, aligned_uint8_allocator> data_t;
    typedef boost::gil::rgb8_view_t view_t;
    typedef boost::gil::rgb8c_view_t const_view_t;
    typedef view_t::point_t point_t;

    AlignedImage();
    AlignedImage(const point_t dimensions);
    AlignedImage(const size_t width, const size_t height);

    ~AlignedImage();

    bool empty() const;

    void resize(const point_t dimensions);
    void resize(const size_t width, const size_t height);

    const point_t &dimensions() const;

    data_t &get_multi_array();
    const data_t &get_multi_array() const;

    const view_t &get_view();
    const const_view_t &get_view() const;

protected:

    data_t data;
    view_t image_view;
    const_view_t image_const_view;
};

} // end of namespace doppia

#endif // ALIGNEDIMAGE_HPP
