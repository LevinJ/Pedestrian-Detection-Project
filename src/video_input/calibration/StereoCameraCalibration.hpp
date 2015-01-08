#ifndef STEREOCAMERACALIBRATION_HPP
#define STEREOCAMERACALIBRATION_HPP

#include <string>

#include <boost/scoped_ptr.hpp>

#include "CameraCalibration.hpp"

// forward declaration of protocol buffer classes
namespace doppia_protobuf
{
class RotationMatrix;
class TranslationVector;
class Pose;
class CameraCalibration;
class StereoCameraCalibration;
}

namespace doppia
{

class StereoCameraCalibration
{
public:
    StereoCameraCalibration(const std::string &calibration_file_path);
    StereoCameraCalibration(const doppia_protobuf::StereoCameraCalibration &calibration_data);

    // copy constructor
    StereoCameraCalibration(const StereoCameraCalibration &calibration);


    const CameraCalibration &get_left_camera_calibration() const;
    CameraCalibration &get_left_camera_calibration();

    const CameraCalibration &get_right_camera_calibration() const;
    CameraCalibration &get_right_camera_calibration();


    const CameraCalibration &get_camera_calibration(const int camera_index) const;
    //CameraCalibration &get_camera_calibration(const int camera_index);

    /// @returns the stereo camera baseline in meters
    float get_baseline() const;

    /// if left and rigth camera do not have the same image center,
    /// then disparities need to be computed using
    /// true_disparity = desired_disparity + disparity_offset
    int get_disparity_offset_x() const;

protected:

    void init(const doppia_protobuf::StereoCameraCalibration &calibration_data);

    boost::scoped_ptr<CameraCalibration> left_camera_calibration_p;
    boost::scoped_ptr<CameraCalibration> right_camera_calibration_p;

    float baseline;

private:
    void set_baseline();
};


} // end of namespace doppia



#endif // STEREOCAMERACALIBRATION_HPP
