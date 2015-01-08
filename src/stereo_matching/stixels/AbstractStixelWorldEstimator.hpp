#ifndef ABSTRACTSTIXELWORLDESTIMATOR_HPP
#define ABSTRACTSTIXELWORLDESTIMATOR_HPP


#include "stereo_matching/AbstractStereoMatcher.hpp"
#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "Stixel.hpp"

namespace doppia
{


class AbstractStixelWorldEstimator
{
public:

    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8_view_t input_image_view_t;
    typedef boost::gil::rgb8c_view_t input_image_const_view_t;

    virtual void set_rectified_images_pair(const input_image_const_view_t &left,
                                           const input_image_const_view_t &right);

    virtual void compute() = 0;

    virtual const GroundPlane &get_ground_plane() const = 0;
    virtual const stixels_t &get_stixels() const = 0;
    virtual int get_stixel_width() const = 0;

    typedef std::vector<int> ground_plane_corridor_t;

    /// for each row in the image, assuming that is the object's bottom position,
    /// will return the expect top object row
    /// top row value is -1 for rows above the horizon
    virtual const ground_plane_corridor_t &get_ground_plane_corridor() = 0;

    virtual ~AbstractStixelWorldEstimator();

protected:

    input_image_const_view_t input_left_view, input_right_view;

};

} // end of namespace doppia


#endif // ABSTRACTSTIXELWORLDESTIMATOR_HPP
