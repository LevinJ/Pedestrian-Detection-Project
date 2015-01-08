#include "BaseGroundPlaneEstimator.hpp"

#include "video_input/calibration/StereoCameraCalibration.hpp"
#include "stereo_matching/cost_volume/DisparityCostVolume.hpp"
#include "GroundPlaneMovingAverage.hpp"

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

#include <Eigen/Dense>

#include <numeric>
#include <string>
#include <limits>
#include <stdexcept>

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"
#include "helpers/xyz_indices.hpp"

namespace {

using namespace logging;

std::ostream &log_info()
{
    return log(InfoMessage, "BaseGroundPlaneEstimator");
}

std::ostream &log_warning()
{
    return log(WarningMessage, "BaseGroundPlaneEstimator");
}

std::ostream &log_error()
{

    return log(ErrorMessage, "BaseGroundPlaneEstimator");
}

std::ostream &log_debug()
{

    return log(DebugMessage, "BaseGroundPlaneEstimator");
}

} // end of private namespace

namespace doppia {



using namespace std;
using namespace boost;

program_options::options_description BaseGroundPlaneEstimator::get_args_options()
{
    program_options::options_description desc("BaseGroundPlaneEstimator options");

    desc.add_options()

            ("ground_plane_estimator.filter_estimates",
             boost::program_options::value<bool>()->default_value(false),
             "use a moving average filter over the estimates")

            ("ground_plane_estimator.rejection_threshold",
             boost::program_options::value<float>()->default_value(0.0015),
             "threshold value to accept or reject of accept a ground plane estimate. "
             "Low values will allow noisy lines to be accepted. Value of 1.5e-3 is strict, 0 is loose.")

            ;

    desc.add(GroundPlaneMovingAverage::get_args_options());

    return desc;
}


BaseGroundPlaneEstimator::BaseGroundPlaneEstimator(
        const boost::program_options::variables_map &options,
        const StereoCameraCalibration& camera_calibration_)
    :
      stereo_calibration(camera_calibration_),
      rejection_threshold(get_option_value<float>(options, "ground_plane_estimator.rejection_threshold"))
{


    // lines_detector_p == NULL

    // FIXME hardcoded values
    ground_height_change_rate = 0.2; //0.1; // +- 10 centimeters
    ground_pitch_change_rate = (M_PI/180)*20; //(M_PI/180)*10; // +- 10 degrees

    // retreive the stereo head baseline and alpha value --
    {

        const CameraCalibration &left_camera_calibration = stereo_calibration.get_left_camera_calibration();
        const CameraCalibration &right_camera_calibration = stereo_calibration.get_right_camera_calibration();

        stereo_baseline = stereo_calibration.get_baseline();

        // alpha and v0 as defined in section II of the V-disparity paper of Labayrade, Aubert and Tarel 2002.
        stereo_alpha = (
                    left_camera_calibration.get_focal_length_x() +
                    left_camera_calibration.get_focal_length_y() +
                    right_camera_calibration.get_focal_length_x() +
                    right_camera_calibration.get_focal_length_y() ) / 4.0;

        {
            log_debug() << "left camera focal length x,y == " <<
                           left_camera_calibration.get_focal_length_x() << ", " <<
                           left_camera_calibration.get_focal_length_y() << std::endl;

            log_debug() << "right camera focal length x,y == " <<
                           right_camera_calibration.get_focal_length_x() << ", " <<
                           right_camera_calibration.get_focal_length_y() << std::endl;
        }

        stereo_v0 = (
                    left_camera_calibration.get_image_center_y() +
                    right_camera_calibration.get_image_center_y() ) / 2.0;
    }


    const bool should_filter = get_option_value<bool>(options,
                                                      "ground_plane_estimator.filter_estimates");

    if(should_filter)
    {
        ground_plane_filter_p.reset(new GroundPlaneMovingAverage(options));
    }

    return;
}


BaseGroundPlaneEstimator::~BaseGroundPlaneEstimator()
{
    // nothing to do here
    return;
}


void  BaseGroundPlaneEstimator::set_ground_area_prior(const std::vector<int> &ground_object_boundary_prior_)
{
    this->ground_object_boundary_prior = ground_object_boundary_prior_;
    return;
}


void BaseGroundPlaneEstimator::get_ground_parameters_change_rate(float &ground_height, float &ground_pitch) const
{
    ground_height = ground_height_change_rate;
    ground_pitch = ground_pitch_change_rate;
    return;
}


void BaseGroundPlaneEstimator::set_ground_plane_prior(const GroundPlane &ground_plane_estimate)
{
    input_ground_plane_estimate = ground_plane_estimate;
    estimated_ground_plane = input_ground_plane_estimate;
    compute_prior_v_disparity_lines();
    return;
}


//void BaseGroundPlaneEstimator::set_ground_disparity_cost_volume(const boost::shared_ptr<DisparityCostVolume> &cost_volume_p)
//{
//    this->cost_volume_p = cost_volume_p;
//    return;
//}

void BaseGroundPlaneEstimator::compute_prior_v_disparity_lines()
{

    { // plus line -
        GroundPlane plus_ground_plane = get_ground_plane_prior();
        plus_ground_plane.offset() += ground_height_change_rate;
        const Eigen::AngleAxisf plus_pitch_change(ground_pitch_change_rate, Eigen::Vector3f::UnitX());
        plus_ground_plane.normal() = plus_pitch_change.toRotationMatrix() * plus_ground_plane.normal();

        this->prior_max_v_disparity_line =
                ground_plane_to_v_disparity_line(plus_ground_plane);

    }

    { // minus line -
        GroundPlane minus_ground_plane = get_ground_plane_prior();
        minus_ground_plane.offset() -= ground_height_change_rate;
        const Eigen::AngleAxisf minus_pitch_change(-ground_pitch_change_rate, Eigen::Vector3f::UnitX());
        minus_ground_plane.normal() = minus_pitch_change.toRotationMatrix() * minus_ground_plane.normal();

        this->prior_min_v_disparity_line =
                ground_plane_to_v_disparity_line(minus_ground_plane);
    }

    // smaller v value means high in the image, max should be higher than min
    assert(prior_max_v_disparity_line.origin()(0) < prior_min_v_disparity_line.origin()(0) );

    return;
}


const BaseGroundPlaneEstimator::line_t &BaseGroundPlaneEstimator::get_prior_max_v_disparity_line() const
{
    return prior_max_v_disparity_line;
}


const BaseGroundPlaneEstimator::line_t &BaseGroundPlaneEstimator::get_prior_min_v_disparity_line() const
{
    return prior_min_v_disparity_line;
}


BaseGroundPlaneEstimator::line_t BaseGroundPlaneEstimator::ground_plane_to_v_disparity_line(const GroundPlane &ground_plane) const
{
    const float theta = -ground_plane.get_pitch();
    const float heigth = ground_plane.get_height();

    line_t line;

    // based on equations 10 and 11 from V-disparity paper of Labayrade, Aubert and Tarel 2002.
    const float v_origin = stereo_v0 - stereo_alpha*std::tan(theta);
    const float c_r = stereo_baseline*cos(theta) / heigth;

    line.origin()(0) = v_origin;
    line.direction()(0) = 1/c_r;

    if(false)
    {
        log_debug() << "ground_plane_to_v_disparity_line ground_plane theta == "
                    << theta << " [radians] == " << (180/M_PI)*theta << " [degrees]" << std::endl;
        log_debug() << "ground_plane_to_v_disparity_line ground_plane height == " << heigth << std::endl;
        log_debug() << "ground_plane_to_v_disparity_line line direction == " << c_r << std::endl;
    }

    return line;
}


GroundPlane BaseGroundPlaneEstimator::v_disparity_line_to_ground_plane(const line_t &v_disparity_line) const
{
    // based on equations 10 and 11 from V-disparity paper of Labayrade, Aubert and Tarel 2002.
    const float v_origin = v_disparity_line.origin()(0);
    const float c_r = 1/v_disparity_line.direction()(0);
    const float v_diff = stereo_v0 - v_origin;

    const float theta = std::atan2(v_diff, stereo_alpha);
    const float height = stereo_baseline*cos(theta) / c_r;

    if(false)
    {
        log_debug() << "v_disparity_line_to_ground_plane ground_plane theta == "
                    << theta << " [radians] == " << (180/M_PI)*theta << " [degrees]" << std::endl;
        log_debug() << "v_disparity_line_to_ground_plane stereo_v0 == " << stereo_v0 << std::endl;
        log_debug() << "v_disparity_line_to_ground_plane stereo_alpha == " << stereo_alpha << std::endl;
        log_debug() << "v_disparity_line_to_ground_plane stereo_baseline == " << stereo_baseline << std::endl;
    }

    GroundPlane ground_plane;
    const float pitch = -theta;
    const float roll = 0;
    ground_plane.set_from_metric_units(pitch, roll, height);
    return ground_plane;
}


const std::vector<int> &BaseGroundPlaneEstimator::get_ground_area_prior() const
{
    return ground_object_boundary_prior;
}


const GroundPlane &BaseGroundPlaneEstimator::get_ground_plane_prior() const
{
    return input_ground_plane_estimate;
}


void BaseGroundPlaneEstimator::set_ground_plane_estimate(const GroundPlane &estimate, const float weight)
{

    if(ground_plane_filter_p)
    {
        ground_plane_filter_p->add_estimate(estimate, weight);
        estimated_ground_plane = ground_plane_filter_p->get_current_estimate();
    }
    else
    {
        estimated_ground_plane = estimate;
    }

    return;
}


const GroundPlane &BaseGroundPlaneEstimator::get_ground_plane() const
{
    return estimated_ground_plane;
}


const BaseGroundPlaneEstimator::line_t &BaseGroundPlaneEstimator::get_ground_v_disparity_line() const
{
    return v_disparity_ground_line;
}


} // end of namespace doppia
