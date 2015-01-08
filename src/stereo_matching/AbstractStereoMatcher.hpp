#ifndef ABSTRACTSTEREOMATCHER_HPP
#define ABSTRACTSTEREOMATCHER_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include <boost/mpl/vector.hpp>

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/gil/extension/dynamic_image/dynamic_image_all.hpp>
//#include <boost/gil/extension/dynamic_image/any_image.hpp>

#include <cstdio>

namespace doppia
{

class AbstractStereoMatcher
{

public:
    /// input image are either rgb8 or gray8
    typedef boost::mpl::vector<boost::gil::gray8_image_t, boost::gil::rgb8_image_t> input_images_t;

    // FIXME some applications may need float instead of bits8
    typedef boost::gil::gray8_image_t disparity_map_t;

protected:
    boost::scoped_ptr<disparity_map_t> disparity_map_p;
    boost::gil::any_image<input_images_t>::const_view_t left, right;

private:
    /// storage used if color to gray convertion is requested
    boost::gil::gray8_image_t gray_left_image, gray_right_image;

protected:

    bool convert_from_rgb_to_gray;

    disparity_map_t::view_t disparity_map_view;

    /// the disparities in the image are expected to
    /// be contained in [0, max_disparity]
    int max_disparity;

    bool first_disparity_map_computation;

public:
    typedef boost::gil::any_image<input_images_t>::const_view_t input_image_view_t;

    static boost::program_options::options_description get_args_options();

    AbstractStereoMatcher(const boost::program_options::variables_map &options);
    virtual ~AbstractStereoMatcher();

    virtual void set_rectified_images_pair(input_image_view_t &left, input_image_view_t &right);

    virtual void compute_disparity_map();

    disparity_map_t::const_view_t get_disparity_map();

protected:
    virtual void compute_disparity_map(boost::gil::gray8c_view_t &left, boost::gil::gray8c_view_t &right) = 0;

    virtual void compute_disparity_map(boost::gil::rgb8c_view_t  &left, boost::gil::rgb8c_view_t &right) = 0;

};

} // end of namespace doppia

#endif // ABSTRACTSTEREOMATCHER_HPP
