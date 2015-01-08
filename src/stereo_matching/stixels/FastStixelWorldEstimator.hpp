#ifndef FASTSTIXELWORLDESTIMATOR_HPP
#define FASTSTIXELWORLDESTIMATOR_HPP

#include "AbstractStixelWorldEstimator.hpp"

#include "video_input/AbstractVideoInput.hpp"

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace objects_detection {
class ObjectsDetectionLibGui; // forward declaration
}

namespace stixel_world {
class StixelWorldLibGui; // forward declaration
}

namespace doppia {

class AbstractStixelsEstimator;
class FastGroundPlaneEstimator;
class FastStixelsEstimator;
class MetricStereoCamera;

class ObjectsDetectionApplication; // used for iros2012 hack

class FastStixelWorldEstimator : public AbstractStixelWorldEstimator
{
public:

    static boost::program_options::options_description get_args_options();

    FastStixelWorldEstimator(const boost::program_options::variables_map &options,
                         const AbstractVideoInput::dimensions_t &input_dimensions,
                         const MetricStereoCamera &camera,
                         const GroundPlane &ground_plane_prior);

    ~FastStixelWorldEstimator();

    void compute();

    const GroundPlane &get_ground_plane() const;
    const stixels_t &get_stixels() const;

    /// for every bottom_v value in the image,
    /// we provide the object top_v value
    /// -1 if not defined (v is above the horizon)
    const ground_plane_corridor_t &get_ground_plane_corridor();

    int get_stixel_width() const;

protected:

    // FIXME any better way of doing this ?
    friend class StixelWorldGui; // used for debugging only
    friend class ObjectsDetectionGui; // used for debugging only
    friend class ObjectsDetectionApplication; // used for iros2012 hack
    friend class objects_detection::ObjectsDetectionLibGui; // used for debugging only
    friend class stixel_world::StixelWorldLibGui; // used for debugging only
    friend void draw_u_disparity_cost_threshold(AbstractStixelWorldEstimator *, const boost::gil::rgb8_view_t &);


    const AbstractVideoInput::dimensions_t input_dimensions;
    const MetricStereoCamera& camera;
    const StereoCameraCalibration& camera_calibration;
    GroundPlane ground_plane_prior;
    ground_plane_corridor_t ground_plane_corridor;

    boost::scoped_ptr<FastGroundPlaneEstimator> ground_plane_estimator_p;
    boost::scoped_ptr<AbstractStixelsEstimator> stixels_estimator_p;

    const float expected_object_height;
    const int minimum_object_height_in_pixels;
    bool silent_mode;

public:
    // GroundPlane is a Eigen structure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

} // end of namespace doppia

#endif // FASTSTIXELWORLDESTIMATOR_HPP
