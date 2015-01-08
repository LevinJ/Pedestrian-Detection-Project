#ifndef DOPPIA_ABSTRACTSTIXELSESTIMATOR_HPP
#define DOPPIA_ABSTRACTSTIXELSESTIMATOR_HPP

#include "Stixel.hpp"

#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"

#include <boost/gil/typedefs.hpp>

namespace doppia {

class AbstractStixelsEstimator
{
public:

    typedef boost::gil::rgb8c_view_t input_image_const_view_t;

    AbstractStixelsEstimator();
    virtual ~AbstractStixelsEstimator();


    /// Set the pair of rectified images corresponding to the computed cost volume
    virtual void set_rectified_images_pair(input_image_const_view_t &left, input_image_const_view_t &right) = 0;

    /// Provide the best estimate available for the ground plane
    virtual void set_ground_plane_estimate(const GroundPlane &ground_plane,
                                           const GroundPlaneEstimator::line_t &v_disparity_ground_line) = 0;

    virtual void compute() = 0;

    const stixels_t &get_stixels() const;

protected:

    stixels_t the_stixels;

};

} // end of namespace doppia

#endif // DOPPIA_ABSTRACTSTIXELSESTIMATOR_HPP
