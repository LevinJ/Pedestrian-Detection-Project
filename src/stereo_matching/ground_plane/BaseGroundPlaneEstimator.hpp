#ifndef BASEGROUNDPLANEESTIMATOR_HPP
#define BASEGROUNDPLANEESTIMATOR_HPP

#include "AbstractGroundPlaneEstimator.hpp"

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {

// forward declarations
class DisparityCostVolume;
class StereoCameraCalibration;
class GroundPlaneMovingAverage;

/// Base class of GroundPlaneEstimator and FastGroundPlaneEstimator
class BaseGroundPlaneEstimator: public AbstractGroundPlaneEstimator
{
public:

    static boost::program_options::options_description get_args_options();

    BaseGroundPlaneEstimator(
            const boost::program_options::variables_map &options,
            const StereoCameraCalibration &stereo_calibration);
    virtual ~BaseGroundPlaneEstimator();

    void set_ground_plane_prior(const GroundPlane &ground_plane_estimate);

    /// Provide the current estimate of the line separating the ground from the objects
    /// every pixels below the line is considered part of the ground
    void set_ground_area_prior(const std::vector<int> &ground_object_boundary_prior);

    // compute is not implemented
    //virtual void compute() = 0;

    /// @returns the prior ground area estimate, with respect to the left image
    const std::vector<int> &get_ground_area_prior() const;

    /// @returns the prior ground plane estimate
    const GroundPlane &get_ground_plane_prior() const;

    /// @returns the estimated ground plane
    const GroundPlane &get_ground_plane() const;

    /// @returns the line that describes the mapping from disparities to v value
    const line_t &get_ground_v_disparity_line() const;

    void get_ground_parameters_change_rate(float &ground_height, float &ground_pitch) const;

    line_t ground_plane_to_v_disparity_line(const GroundPlane &ground_plane) const;
    GroundPlane v_disparity_line_to_ground_plane(const line_t &v_disparity_line) const;

    /// this line is defined in the v-disparity space
    const line_t &get_prior_max_v_disparity_line() const;

    /// see get_prior_max_v_disparity_line
    const line_t &get_prior_min_v_disparity_line() const;

protected:

    const StereoCameraCalibration &stereo_calibration;
    float stereo_baseline, stereo_alpha, stereo_v0;

    const float rejection_threshold;

    std::vector<int> ground_object_boundary_prior;

    /// between two frames the ground parameters are expected to be at most
    /// input_ground_plane_estimate -+ (ground_height_change_rate, ground_pitch_change_rate)
    float ground_height_change_rate, ground_pitch_change_rate;

    GroundPlane input_ground_plane_estimate, estimated_ground_plane;

    line_t prior_max_v_disparity_line, prior_min_v_disparity_line;

    /// sets the values of prior_max_v_disparity_line and prior_min_v_disparity_line
    void compute_prior_v_disparity_lines();

    boost::shared_ptr<AbstractLinesDetector> lines_detector_p;
    AbstractLinesDetector::line_t v_disparity_ground_line;

    boost::shared_ptr<GroundPlaneMovingAverage> ground_plane_filter_p;

    void set_ground_plane_estimate(const GroundPlane &, const float weight);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // end of namespace doppia


#endif // BASEGROUNDPLANEESTIMATOR_HPP
