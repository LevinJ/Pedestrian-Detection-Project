#include "CameraCalibration.hpp"

#include "calibration.pb.h"

namespace doppia
{


CameraCalibration::CameraCalibration(const doppia_protobuf::CameraCalibration &calibration_data)
{
    // --
    const doppia_protobuf::CameraInternalParameters &K = calibration_data.internal_parameters();
    internal_calibration <<
                            K.k11(), K.k12(), K.k13(),
            K.k21(), K.k22(), K.k23(),
            K.k31(), K.k32(), K.k33();

    // --
    if(calibration_data.has_radial_distortion())
    {
        const doppia_protobuf::RadialDistortion &r = calibration_data.radial_distortion();
        radial_distortion_parameters << r.k1(), r.k2(), r.k3();
    }
    else
    {
        radial_distortion_parameters = RadialDistortionParametersVector::Zero();
    }

    // --
    if(calibration_data.has_tangential_distortion())
    {
        const doppia_protobuf::TangentialDistortion &t = calibration_data.tangential_distortion();
        tangential_distortion_parameters << t.k1(), t.k2(), t.k3();
    }
    else
    {
        tangential_distortion_parameters = TangentialDistortionParametersVector::Zero();
    }

    // --
    RotationMatrix rotation(RotationMatrix::Zero());
    TranslationVector translation(TranslationVector::Zero());

    if(calibration_data.has_pose())
    {
        const doppia_protobuf::RotationMatrix &R = calibration_data.pose().rotation();
        rotation <<
                    R.r11(), R.r12(), R.r13(),
                R.r21(), R.r22(), R.r23(),
                R.r31(), R.r32(), R.r33();

        const doppia_protobuf::TranslationVector &t = calibration_data.pose().translation();
        translation << t.t1(), t.t2(), t.t3();
    }

    pose_p.reset(new Pose(rotation, translation));

    return;
}

CameraCalibration::CameraCalibration(const CameraCalibration &calibration)
{
    internal_calibration = calibration.get_internal_calibration();
    radial_distortion_parameters = calibration.get_radial_distortion_parameters();
    tangential_distortion_parameters = calibration.get_tangential_distortion_parameters();

    pose_p.reset(new Pose(calibration.get_pose()));

    return;
}

CameraCalibration::~CameraCalibration()
{
    // nothing to do here
    return;
}


const InternalCalibrationMatrix &CameraCalibration::get_internal_calibration() const
{
    return internal_calibration;
}


InternalCalibrationMatrix &CameraCalibration::get_internal_calibration()
{
    return internal_calibration;
}


float CameraCalibration::get_focal_length_x() const
{
    // focal length in pixels
    return internal_calibration(0,0);
}

float CameraCalibration::get_focal_length_y() const
{
    // focal length in pixels
    return internal_calibration(1,1);
}

float CameraCalibration::get_image_center_x() const
{
    return internal_calibration(0,2);
}

float CameraCalibration::get_image_center_y() const
{
    return internal_calibration(1,2);
}


const RadialDistortionParametersVector &CameraCalibration::get_radial_distortion_parameters() const
{

    return radial_distortion_parameters;
}


RadialDistortionParametersVector &CameraCalibration::get_radial_distortion_parameters()
{

    return radial_distortion_parameters;
}

const TangentialDistortionParametersVector &CameraCalibration::get_tangential_distortion_parameters() const
{

    return tangential_distortion_parameters;
}

TangentialDistortionParametersVector &CameraCalibration::get_tangential_distortion_parameters()
{

    return tangential_distortion_parameters;
}

const Pose &CameraCalibration::get_pose() const
{
    return *pose_p;
}

Pose &CameraCalibration::get_pose()
{
    return *pose_p;
}

} // end of namespace doppia


