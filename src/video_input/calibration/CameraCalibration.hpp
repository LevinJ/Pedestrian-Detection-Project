#ifndef CAMERACALIBRATION_HPP
#define CAMERACALIBRATION_HPP


#include "Pose.hpp"

#include <Eigen/Core>
#include <boost/scoped_ptr.hpp>

#include <boost/units/systems/si/length.hpp>
#include <boost/units/systems/si/dimensionless.hpp>
#include <boost/units/quantity.hpp>
#include <boost/units/base_unit.hpp>

#include <boost/units/systems/si/solid_angle.hpp>

#include <boost/units/static_constant.hpp>
#include <boost/units/io.hpp>

// forward declaration of protocol buffer classes
namespace doppia_protobuf
{
class CameraInternalParameters;
class RadialDistortion;
class TangentialDistortion;
class CameraCalibration;
}

// Define the pixel type ---
namespace doppia {

struct pixels_base_unit : boost::units::base_unit<pixels_base_unit, boost::units::length_dimension, 1> { };

typedef pixels_base_unit pixels_t;

//BOOST_UNITS_STATIC_CONSTANT(pixels, pixels_base_unit);
}

namespace boost {
namespace units {

template<> struct  base_unit_info<doppia::pixels_base_unit>
{
    static std::string name()               { return "pixels"; }
    static std::string symbol()             { return "px"; }
};

}
}
// end of "define the pixel type" ---

namespace doppia
{

typedef Eigen::Matrix<float, 3, 3> InternalCalibrationMatrix;
typedef Eigen::Vector3f RadialDistortionParametersVector;
typedef Eigen::Vector3f TangentialDistortionParametersVector;


class CameraCalibration
{
public:
    CameraCalibration(const doppia_protobuf::CameraCalibration &calibration_data);

    /// copy constructor
    CameraCalibration(const CameraCalibration &calibration);

    ~CameraCalibration();


    ///  Usually named K
    const InternalCalibrationMatrix &get_internal_calibration() const;
    InternalCalibrationMatrix &get_internal_calibration();

    /// focal length in pixels
    float get_focal_length_x() const;
    float get_focal_length_y() const;
    float get_image_center_x() const;
    float get_image_center_y() const;


    const RadialDistortionParametersVector &get_radial_distortion_parameters() const;
    RadialDistortionParametersVector &get_radial_distortion_parameters();

    const TangentialDistortionParametersVector &get_tangential_distortion_parameters() const;
    TangentialDistortionParametersVector &get_tangential_distortion_parameters();

    const Pose &get_pose() const;
    Pose &get_pose();

protected:
    InternalCalibrationMatrix internal_calibration;
    RadialDistortionParametersVector radial_distortion_parameters;
    TangentialDistortionParametersVector tangential_distortion_parameters;

    boost::scoped_ptr<Pose> pose_p;
};




} // end of namespace doppia


#endif // CAMERACALIBRATION_HPP
