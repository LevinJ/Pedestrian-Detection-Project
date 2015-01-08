#include "StereoCameraCalibration.hpp"

#include "helpers/Log.hpp"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "calibration.pb.h"

#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>
#include <cstdlib>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "StereoCameraCalibration");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "StereoCameraCalibration");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "StereoCameraCalibration");
}

} // end of anonymous namespace

namespace doppia
{

using namespace std;
using namespace google::protobuf;

StereoCameraCalibration::StereoCameraCalibration(const string &calibration_file_path)
{

    printf("Using stereo camera calibration file: %s\n", calibration_file_path.c_str());
    if(boost::filesystem::exists(calibration_file_path) == false)
    {
        throw std::runtime_error("Could not find the indicated stereo calibration file");
    }

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    boost::scoped_ptr<doppia_protobuf::StereoCameraCalibration> stereo_calibration_data_p;
    stereo_calibration_data_p.reset(new doppia_protobuf::StereoCameraCalibration());


    // parse the protocol buffer file
    fstream input_stream(calibration_file_path.c_str(), ios::in); // text is default stream format

    io::ZeroCopyInputStream *zci_stream_p = new io::IstreamInputStream(&input_stream);

    bool success = TextFormat::Parse(zci_stream_p, stereo_calibration_data_p.get());
    if (success == false)
    {
        fprintf(stderr, "Failed to parse the stereo camera calibration file '%s'\n", calibration_file_path.c_str());
        throw std::runtime_error("Failed to parse the stereo camera calibration file");
    }

    delete zci_stream_p;

    init(*stereo_calibration_data_p);
    return;
}


StereoCameraCalibration::StereoCameraCalibration(const doppia_protobuf::StereoCameraCalibration &calibration_data)
{
    init(calibration_data);
    return;
}


void StereoCameraCalibration::init(const doppia_protobuf::StereoCameraCalibration &calibration_data)
{
    if(calibration_data.IsInitialized() == false)
    {
        throw std::runtime_error("StereoCameraCalibration expects to receive "
                                 "initialized doppia_protobuf::StereoCameraCalibration data");
    }

    if (calibration_data.has_name())
    {
        log_info() << "stereo_calibration_data name: " << calibration_data.name() << std::endl;
    }

    left_camera_calibration_p.reset(new CameraCalibration(calibration_data.left_camera()));
    right_camera_calibration_p.reset(new CameraCalibration(calibration_data.right_camera()));

    set_baseline();

    return;
}


StereoCameraCalibration::StereoCameraCalibration(const StereoCameraCalibration &calibration)
{

    // copy left and right calibration
    left_camera_calibration_p.reset(new CameraCalibration(calibration.get_left_camera_calibration()));
    right_camera_calibration_p.reset(new CameraCalibration(calibration.get_right_camera_calibration()));

    set_baseline();
    return;
}


void StereoCameraCalibration::set_baseline()
{
    this->baseline = (left_camera_calibration_p->get_pose().translation -
                      right_camera_calibration_p->get_pose().translation ).norm();
    return;
}


float StereoCameraCalibration::get_baseline() const
{
    return baseline;
}


int StereoCameraCalibration::get_disparity_offset_x() const
{
    return left_camera_calibration_p->get_image_center_x() - right_camera_calibration_p->get_image_center_x();
}


const CameraCalibration &StereoCameraCalibration::get_left_camera_calibration() const
{
    return *left_camera_calibration_p;
}


CameraCalibration &StereoCameraCalibration::get_left_camera_calibration()
{
    return *left_camera_calibration_p;
}


const CameraCalibration &StereoCameraCalibration::get_right_camera_calibration() const
{
    return *right_camera_calibration_p;
}


CameraCalibration &StereoCameraCalibration::get_right_camera_calibration()
{
    return *right_camera_calibration_p;
}

const CameraCalibration &StereoCameraCalibration::get_camera_calibration(const int camera_index) const
{
    if(camera_index < 0 or camera_index > 1)
    {
        throw std::runtime_error("On a stereo camera, camera_index can only be 0 or 1");
    }

    const CameraCalibration *camera_calibration_p = NULL;
    if(camera_index == 0)
    {
        camera_calibration_p = left_camera_calibration_p.get();
    } else
    {
        camera_calibration_p = right_camera_calibration_p.get();
    }

    return *camera_calibration_p;
}




} // end of namespace doppia


