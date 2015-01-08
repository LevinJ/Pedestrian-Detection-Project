#ifndef STIXELWORLDESTIMATOR_HPP
#define STIXELWORLDESTIMATOR_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

#include "video_input/AbstractVideoInput.hpp"
#include "AbstractStixelWorldEstimator.hpp"
#include "stereo_matching/ground_plane/GroundPlane.hpp"
#include "Stixel.hpp"

#include "video_input/preprocessing/ReverseMapper.hpp"

namespace objects_detection {
class ObjectsDetectionLibGui; // forward declaration
}

namespace stixel_world {
class StixelWorldLibGui; // forward declaration
}

namespace doppia {

// forward declarations
class StixelsEstimator;
class GroundPlaneEstimator;
class AbstractPreprocessor;
class CpuPreprocessor;

class AbstractDisparityCostVolumeEstimator;
class DisparityCostVolume;

class StixelWorldGui; // used for debugging only

/// Given two input images will estimate the ground plane and the stixels in the image
class StixelWorldEstimator: public AbstractStixelWorldEstimator
{
public:

    static boost::program_options::options_description get_args_options();

    StixelWorldEstimator(const boost::program_options::variables_map &options,
                         const AbstractVideoInput::dimensions_t &input_dimensions,
                         const MetricStereoCamera &camera,
                         const boost::shared_ptr<AbstractPreprocessor> preprocessor_p,
                         const GroundPlane &ground_plane_prior);

    ~StixelWorldEstimator();

    void compute();

    const GroundPlane &get_ground_plane() const;
    const stixels_t &get_stixels() const;

    /// for every bottom_v value in the image,
    /// we provide the object top_v value
    /// -1 if not defined (v is above the horizon)
    const ground_plane_corridor_t &get_ground_plane_corridor();

    int get_stixel_width() const;

protected:

    friend class StixelWorldGui; // used for debugging only
    friend class ObjectsDetectionGui; // used for debugging only
    friend class objects_detection::ObjectsDetectionLibGui; // used for debugging only
    friend class stixel_world::StixelWorldLibGui; // used for debugging only

    const AbstractVideoInput::dimensions_t input_dimensions;
    const MetricStereoCamera& camera;
    const StereoCameraCalibration& camera_calibration;

    bool use_stixels_for_ground_estimation;

    scoped_ptr<GroundPlaneEstimator> ground_plane_estimator_p;
    scoped_ptr<StixelsEstimator> stixels_estimator_p;

    GroundPlane ground_plane_prior;
    ground_plane_corridor_t ground_plane_corridor;

    shared_ptr<AbstractDisparityCostVolumeEstimator> cost_volume_estimator_p;
    shared_ptr<DisparityCostVolume>
    pixels_matching_cost_volume_p,
    residual_pixels_matching_cost_volume_p;

    // FIXME should be residual computation should not be directly linked to preprocessing
    boost::shared_ptr<CpuPreprocessor> preprocessor_p;
    bool should_compute_residual;
    input_image_t input_left_residual_image, input_right_residual_image;

    /// expected object height, in meters
    float expected_object_height;

    /// minimum object height, in pixels
    int minimum_object_height_in_pixels;

public:
    // GroundPlane is a Eigen structure
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

};

/// Helper method exhibed to be used inside FastStixelWorldEstimator
void compute_ground_plane_corridor(
        const GroundPlane &ground_plane,
        const std::vector<int> &disparity_given_v,
        const MetricStereoCamera&  camera,
        const float expected_object_height,
        const int minimum_object_height_in_pixels,
        AbstractStixelWorldEstimator::ground_plane_corridor_t &ground_plane_corridor);

} // end of namespace doppia

#endif // STIXELWORLDESTIMATOR_HPP
