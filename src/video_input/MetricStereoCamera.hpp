#ifndef METRICSTEREOCAMERA_HPP
#define METRICSTEREOCAMERA_HPP

#include <boost/scoped_ptr.hpp>
#include <stdexcept>

namespace doppia {

// forward declarations
class StereoCameraCalibration;
class MetricCamera;

/// Dummy container class for two metric cameras
class MetricStereoCamera
{
public:
    MetricStereoCamera(const StereoCameraCalibration &calibration);
    ~MetricStereoCamera();

    const MetricCamera& get_left_camera() const;
    MetricCamera& get_left_camera();

    const MetricCamera& get_right_camera() const;
    MetricCamera& get_right_camera();

    const StereoCameraCalibration &get_calibration() const;

    /// disparity in pixels, depth is returned in meters
    float disparity_to_depth(const float disparity) const;

protected:

    const StereoCameraCalibration &calibration;
    boost::scoped_ptr<MetricCamera> left_camera_p, right_camera_p;
    float baseline_times_focal_length;
};


// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

// speed critical functions need to be inlined
// to be inlined they need to be defined in the class header

inline
float MetricStereoCamera::disparity_to_depth(const float disparity) const
{
    if (disparity <= 0)
    {
        throw std::invalid_argument("MetricStereoCamera::disparity_to_depth can only transform disparities > 0");
    }

    const float depth = baseline_times_focal_length / disparity;    

    return depth;
}

} // end of namespace doppia

#endif // METRICSTEREOCAMERA_HPP
