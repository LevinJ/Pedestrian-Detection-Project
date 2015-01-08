#include "AbstractVideoInput.hpp"

#include "calibration/StereoCameraCalibration.hpp"

#include "helpers/get_option_value.hpp"

#include "video_input/MetricStereoCamera.hpp"

namespace doppia
{

using namespace boost;

program_options::options_description AbstractVideoInput::get_args_options()
{

    using boost::program_options::value;

    program_options::options_description desc("AbstractVideoInput options");

    desc.add_options()

            // FIXME this is a hack, should have a proto file to describe the ground plane model
            // or at least move it into camera_calibration ?
            ("video_input.camera_height", value<float>()->default_value(1.2f),
             "specify the distance in meters to the ground plane")
            ("video_input.camera_pitch", value<float>()->default_value(0.0f),
             "define the pitch angle in radians with respect to the ground plane. This value will typically be a small negative value.")
            ("video_input.camera_roll", value<float>()->default_value(0.0f),
             "define the roll angle with respect to the ground plane")

            ;

    return desc;
}


AbstractVideoInput::AbstractVideoInput(const program_options::variables_map &options)
    : current_frame_number(0)
{

    camera_height =
            get_option_value<float>(options, "video_input.camera_height");
    camera_pitch =
            get_option_value<float>(options, "video_input.camera_pitch");
    camera_roll =
            get_option_value<float>(options, "video_input.camera_roll");

    return;
}


AbstractVideoInput::~AbstractVideoInput()
{
    // nothing to do here
    return;
}


int AbstractVideoInput::get_current_frame_number() const
{
    return this->current_frame_number;
}


const MetricStereoCamera& AbstractVideoInput::get_metric_camera() const
{
    if(metric_camera_p == false)
    {
        metric_camera_p.reset(new MetricStereoCamera(this->get_stereo_calibration()));
    }

    return *metric_camera_p;
}

} // end of namespace doppia
