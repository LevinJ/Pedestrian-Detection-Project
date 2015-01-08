#include "MetricStereoCamera.hpp"

#include "video_input/calibration/StereoCameraCalibration.hpp"
#include "MetricCamera.hpp"

#include <stdexcept>

namespace doppia {

MetricStereoCamera::MetricStereoCamera(const StereoCameraCalibration &calibration_)
    : calibration(calibration_)
{
    const CameraCalibration &left_calibration = calibration.get_left_camera_calibration();
    const CameraCalibration &right_calibration = calibration.get_right_camera_calibration();

    left_camera_p.reset(new MetricCamera(left_calibration));
    right_camera_p.reset(new MetricCamera(right_calibration));

    const float focal_length = (left_calibration.get_focal_length_x() +
                                right_calibration.get_focal_length_x()) * 0.5;

    baseline_times_focal_length = calibration.get_baseline() * focal_length;
    return;
}

MetricStereoCamera::~MetricStereoCamera()
{
    // nothing to do here
    return;
}

const MetricCamera& MetricStereoCamera::get_left_camera() const
{
    return *left_camera_p;
}

MetricCamera& MetricStereoCamera::get_left_camera()
{
    return *left_camera_p;
}

const MetricCamera& MetricStereoCamera::get_right_camera() const
{
    return *right_camera_p;
}

MetricCamera& MetricStereoCamera::get_right_camera()
{
    return *right_camera_p;
}

const StereoCameraCalibration &MetricStereoCamera::get_calibration() const
{
    return calibration;
}

// float MetricStereoCamera::disparity_to_depth(const float disparity) const has been inlined, see header file

} // end of namespace doppia
